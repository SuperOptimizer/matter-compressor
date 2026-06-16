/* mc_tiff.c — minimal dependency-free TIFF reader/writer (see mc_tiff.h). */
#include "mc_tiff.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

size_t mc_tiff_type_size(mc_tiff_type t){
    switch(t){ case MC_TIFF_U8: return 1; case MC_TIFF_U16: case MC_TIFF_I16: return 2;
               case MC_TIFF_U32: case MC_TIFF_F32: return 4; default: return 0; }
}

// ---- TIFF tag / type constants ----
enum { T_IMAGEWIDTH=256, T_IMAGELENGTH=257, T_BITSPERSAMPLE=258, T_COMPRESSION=259,
       T_PHOTOMETRIC=262, T_STRIPOFFSETS=273, T_SAMPLESPERPIXEL=277, T_ROWSPERSTRIP=278,
       T_STRIPBYTECOUNTS=279, T_PLANARCONFIG=284, T_EXTRASAMPLES=338, T_SAMPLEFORMAT=339 };
enum { FT_SHORT=3, FT_LONG=4 };           // IFD field types we emit/read
enum { SF_UINT=1, SF_INT=2, SF_IEEEFP=3 };

// ---- little-endian read helpers (over an mmap) ----
static uint16_t rd16(const uint8_t *p){ return (uint16_t)(p[0] | (p[1]<<8)); }
static uint32_t rd32(const uint8_t *p){ return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24); }

// Read the FIRST value of an IFD entry (SHORT or LONG), 0 if absent. `found`
// set if present. For our use every tag is either single-valued or an array
// whose elements are all equal (BitsPerSample/SampleFormat per channel), so the
// first element is representative. `b` is the file base (for out-of-line SHORT
// arrays whose value field holds an offset when count>2).
static uint32_t ifd_get(const uint8_t *b, const uint8_t *ifd, int n, size_t len,
                        uint16_t tag, int *found){
    *found=0;
    for(int i=0;i<n;++i){
        const uint8_t *e = ifd + 2 + (size_t)i*12;
        if(rd16(e)!=tag) continue;
        uint16_t ft = rd16(e+2);
        uint32_t cnt = rd32(e+4);
        *found=1;
        if(ft==FT_SHORT){
            if(cnt<=2) return rd16(e+8);          // inline
            uint32_t off = rd32(e+8);             // out-of-line array offset
            if((size_t)off+2 > len) return 0;
            return rd16(b+off);                   // first element
        }
        // FT_LONG: count>1 longs don't occur in our files; read inline first.
        return rd32(e+8);
    }
    return 0;
}

int mc_tiff_open(const char *path, mc_tiff *out){
    memset(out,0,sizeof *out);
    int fd = open(path, O_RDONLY);
    if(fd<0) return -1;
    struct stat st; if(fstat(fd,&st)!=0 || st.st_size < 8){ close(fd); return -1; }
    size_t len = (size_t)st.st_size;
    void *map = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if(map==MAP_FAILED){ close(fd); return -1; }
    const uint8_t *b = map;

    int rc = -1;
    // header: "II" 0x2A <ifd_off>. We only read little-endian classic TIFF.
    if(!(b[0]=='I' && b[1]=='I' && rd16(b+2)==42)) goto done;
    uint32_t ifd_off = rd32(b+4);
    if((size_t)ifd_off+2 > len) goto done;        // size_t add: ifd_off is u32, +2 must not wrap
    int n = rd16(b+ifd_off);
    if((size_t)ifd_off + 2 + (size_t)n*12 + 4 > len) goto done;
    const uint8_t *ifd = b + ifd_off;

    int f;
    uint32_t w   = ifd_get(b,ifd,n,len,T_IMAGEWIDTH,&f);   if(!f) goto done;
    uint32_t h   = ifd_get(b,ifd,n,len,T_IMAGELENGTH,&f);  if(!f) goto done;
    uint32_t spp = ifd_get(b,ifd,n,len,T_SAMPLESPERPIXEL,&f); if(!f) spp=1;
    uint32_t bps = ifd_get(b,ifd,n,len,T_BITSPERSAMPLE,&f); if(!f) bps=8;
    uint32_t comp= ifd_get(b,ifd,n,len,T_COMPRESSION,&f);  if(!f) comp=1;
    uint32_t sf  = ifd_get(b,ifd,n,len,T_SAMPLEFORMAT,&f); if(!f) sf=SF_UINT;
    uint32_t soff= ifd_get(b,ifd,n,len,T_STRIPOFFSETS,&f); if(!f) goto done;
    uint32_t rps = ifd_get(b,ifd,n,len,T_ROWSPERSTRIP,&f); if(!f) rps=h;
    uint32_t pc  = ifd_get(b,ifd,n,len,T_PLANARCONFIG,&f); if(!f) pc=1;

    // we only support: uncompressed, single strip (rows_per_strip >= height),
    // chunky (planar==1), 1..4 samples, supported (bps,sampleformat). Dimensions
    // must be positive and bounded — a 0-width/height image is degenerate (empty
    // strip a consumer would divide by / index past), and INT_MAX-scale dims
    // would overflow the int fields of mc_tiff. Cap at a generous 1<<20 per axis.
    if(comp!=1 || pc!=1 || spp<1 || spp>4 || rps<h) goto done;
    if(w==0 || h==0 || w>(1u<<20) || h>(1u<<20)) goto done;
    mc_tiff_type ty;
    if      (bps==8  && sf==SF_UINT)   ty=MC_TIFF_U8;
    else if (bps==16 && sf==SF_UINT)   ty=MC_TIFF_U16;
    else if (bps==16 && sf==SF_INT)    ty=MC_TIFF_I16;
    else if (bps==32 && sf==SF_UINT)   ty=MC_TIFF_U32;
    else if (bps==32 && sf==SF_IEEEFP) ty=MC_TIFF_F32;
    else goto done;

    size_t pixbytes = (size_t)w*h*spp*mc_tiff_type_size(ty);
    if((size_t)soff + pixbytes > len) goto done;   // strip must fit the file

    out->width=(int)w; out->height=(int)h; out->samples=(int)spp; out->type=ty;
    out->pixels = b + soff;
    out->pixel_bytes = pixbytes;
    out->map=map; out->map_len=len; out->fd=fd;
    return 0;

done:
    munmap(map,len); close(fd);
    return rc;
}

void mc_tiff_close(mc_tiff *t){
    if(t && t->map){ munmap(t->map, t->map_len); if(t->fd>=0) close(t->fd); }
    if(t) memset(t,0,sizeof *t);
}

// ---- little-endian write helpers ----
static void w16(uint8_t *p, uint16_t v){ p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); }
static void w32(uint8_t *p, uint32_t v){ p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); p[2]=(uint8_t)(v>>16); p[3]=(uint8_t)(v>>24); }

int mc_tiff_write(const char *path, int width, int height, int samples,
                  mc_tiff_type type, const void *pixels){
    if(width<=0||height<=0||samples<1||samples>4) return -1;
    size_t ts = mc_tiff_type_size(type); if(!ts) return -1;
    size_t pixbytes = (size_t)width*height*samples*ts;

    // LAYOUT (mmap-friendly): [8B header][pixel data @ off 8][IFD].
    // off 8 is 8-byte aligned and >= 4-byte aligned for f32/u32 mmap views.
    uint32_t pix_off = 8;
    uint32_t ifd_off = (uint32_t)(pix_off + pixbytes);
    // align IFD to 2 (TIFF requires word alignment); pixbytes for our types is
    // already a multiple of 2, but be safe.
    if(ifd_off & 1) ifd_off++;

    int bps, sf;
    switch(type){
        case MC_TIFF_U8:  bps=8;  sf=SF_UINT;   break;
        case MC_TIFF_U16: bps=16; sf=SF_UINT;   break;
        case MC_TIFF_I16: bps=16; sf=SF_INT;    break;
        case MC_TIFF_U32: bps=32; sf=SF_UINT;   break;
        case MC_TIFF_F32: bps=32; sf=SF_IEEEFP; break;
        default: return -1;
    }

    // Tags. For SamplesPerPixel>1, BitsPerSample and SampleFormat must be
    // count=samples arrays (one entry per channel) for strict readers; with >2
    // SHORTs they don't fit the inline value field, so they live in an
    // out-of-line area right after the IFD (offsets point there). Samples>1 also
    // gets ExtraSamples for channels beyond the photometric ones (RGB uses 3;
    // gray uses 1) so 4-channel images are well-formed.
    int n_extra = 0;
    if(samples>3) n_extra = samples-3;          // RGB + extras
    else if(samples==2) n_extra = 1;            // gray + extra

    int has_arr = (samples>1);                  // bps/sampleformat as arrays
    int ntag = 11 + (n_extra>0 ? 1 : 0);        // + ExtraSamples

    uint8_t hdr[8];
    hdr[0]='I'; hdr[1]='I'; w16(hdr+2,42); w32(hdr+4,ifd_off);

    size_t ifd_entries = 2 + (size_t)ntag*12 + 4;
    // out-of-line SHORT arrays placed after the IFD: bps[samples], sf[samples],
    // extrasamples[n_extra] — each only if it doesn't fit inline (count>2).
    uint32_t oo_base = ifd_off + (uint32_t)ifd_entries;
    uint32_t bps_off=0, sf_off=0, ex_off=0;
    size_t oo_len=0;
    if(has_arr && samples>2){ bps_off=oo_base+(uint32_t)oo_len; oo_len+=(size_t)samples*2; }
    if(has_arr && samples>2){ sf_off =oo_base+(uint32_t)oo_len; oo_len+=(size_t)samples*2; }
    if(n_extra>2){            ex_off =oo_base+(uint32_t)oo_len; oo_len+=(size_t)n_extra*2; }

    uint8_t *ifd = calloc(1, ifd_entries + oo_len);
    if(!ifd) return -1;
    w16(ifd, (uint16_t)ntag);
    uint8_t *e = ifd+2;
    uint8_t *oo = ifd + ifd_entries;            // out-of-line area
    size_t oo_used = 0;
    #define TAG_SHORT(tag,val) do{ w16(e,(tag)); w16(e+2,FT_SHORT); w32(e+4,1); w16(e+8,(uint16_t)(val)); w16(e+10,0); e+=12; }while(0)
    #define TAG_LONG(tag,val)  do{ w16(e,(tag)); w16(e+2,FT_LONG);  w32(e+4,1); w32(e+8,(uint32_t)(val)); e+=12; }while(0)
    // a SHORT array of `cnt`: inline if cnt<=2, else out-of-line at `oolist_off`.
    #define TAG_SHORT_ARR(tag,cnt,filler,oolist_off) do{ \
        w16(e,(tag)); w16(e+2,FT_SHORT); w32(e+4,(uint32_t)(cnt)); \
        if((cnt)<=2){ for(int _k=0;_k<(int)(cnt);++_k) w16(e+8+_k*2,(uint16_t)(filler)); \
                      for(int _k=(int)(cnt);_k<2;++_k) w16(e+8+_k*2,0); } \
        else { w32(e+8,(oolist_off)); for(int _k=0;_k<(int)(cnt);++_k){ w16(oo+oo_used,(uint16_t)(filler)); oo_used+=2; } } \
        e+=12; }while(0)
    // tags in ascending order.
    TAG_LONG (T_IMAGEWIDTH,     width);
    TAG_LONG (T_IMAGELENGTH,    height);
    TAG_SHORT_ARR(T_BITSPERSAMPLE, samples, bps, bps_off);
    TAG_SHORT(T_COMPRESSION,    1);
    TAG_SHORT(T_PHOTOMETRIC,    (samples>=3)?2:1);
    TAG_LONG (T_STRIPOFFSETS,   pix_off);
    TAG_SHORT(T_SAMPLESPERPIXEL,samples);
    TAG_LONG (T_ROWSPERSTRIP,   height);
    TAG_LONG (T_STRIPBYTECOUNTS,(uint32_t)pixbytes);
    TAG_SHORT(T_PLANARCONFIG,   1);
    if(n_extra>0) TAG_SHORT_ARR(T_EXTRASAMPLES, n_extra, 0 /*unspecified*/, ex_off);
    TAG_SHORT_ARR(T_SAMPLEFORMAT, samples, sf, sf_off);
    w32(e, 0);                                  // next IFD
    #undef TAG_SHORT
    #undef TAG_LONG
    #undef TAG_SHORT_ARR

    FILE *fp = fopen(path,"wb");
    if(!fp){ free(ifd); return -1; }
    int ok = 1;
    ok &= (fwrite(hdr,1,8,fp)==8);
    ok &= (fwrite(pixels,1,pixbytes,fp)==pixbytes);
    for(uint32_t p=pix_off+(uint32_t)pixbytes; p<ifd_off; ++p){ ok &= (fputc(0,fp)!=EOF); }
    ok &= (fwrite(ifd,1,ifd_entries+oo_len,fp)==ifd_entries+oo_len);
    fclose(fp);
    free(ifd);
    return ok ? 0 : -1;
}
