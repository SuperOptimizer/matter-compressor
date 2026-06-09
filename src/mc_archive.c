// ============================================================================
// mc_archive.c — matter-compressor archive build + decode. See mc_archive_api.h.
// Source-agnostic: pulls voxels through a callback (no zarr/S3 here). Owns the LOD
// pyramid, the sparse-tree writer (calling mc_codec), and the reader.
// ============================================================================
#include "mc_archive_api.h"
#include "mc_archive.h"
#include "mc_archive_read.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef uint8_t u8;

// ---------------------------------------------------------------- volume (one LOD)
// A contiguous dim^3 u8 buffer. LOD0 is materialized from the source callback; coarser
// LODs are box-decimated from the finer one.
typedef struct { const u8 *v; int dim; } vol_t;
static inline u8 vget(const vol_t *V, int z,int y,int x){
    if((unsigned)z>=(unsigned)V->dim||(unsigned)y>=(unsigned)V->dim||(unsigned)x>=(unsigned)V->dim) return 0;
    return V->v[((size_t)z*V->dim+y)*V->dim+x];
}
// 2x box-decimate: mean of NONZERO children; all-zero stays 0 (inherited zero-mask).
static u8 *decimate(const u8 *src,int D){ int H=D/2; u8 *o=calloc((size_t)H*H*H,1);
    for(int z=0;z<H;++z)for(int y=0;y<H;++y)for(int x=0;x<H;++x){ int s=0,c=0;
        for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
            u8 v=src[(((size_t)(2*z+dz))*D+(2*y+dy))*D+(2*x+dx)]; if(v){s+=v;c++;}}
        o[((size_t)z*H+y)*H+x]=c?(u8)((s+c/2)/c):0; } return o; }

// gather a 16^3 block (zero-padded at edges).
static int gather_blk(const vol_t *V,int cz,int cy,int cx,int bz,int by,int bx,u8 *dst){
    int z0=(cz*16+bz)*MC_BLK,y0=(cy*16+by)*MC_BLK,x0=(cx*16+bx)*MC_BLK,any=0;
    for(int z=0;z<MC_BLK;++z)for(int y=0;y<MC_BLK;++y)for(int x=0;x<MC_BLK;++x){
        u8 v=vget(V,z0+z,y0+y,x0+x); dst[(z*MC_BLK+y)*MC_BLK+x]=v; any|=v; }
    return any;
}
static int chunk_present(const vol_t *V,int cz,int cy,int cx){
    int D=V->dim,z0=cz*256,y0=cy*256,x0=cx*256;
    if(z0>=D||y0>=D||x0>=D) return 0;
    int z1=z0+256<D?z0+256:D,y1=y0+256<D?y0+256:D,x1=x0+256<D?x0+256:D;
    for(int z=z0;z<z1;++z)for(int y=y0;y<y1;++y)for(int x=x0;x<x1;++x) if(vget(V,z,y,x)) return 1;
    return 0;
}

// ---------------------------------------------------------------- archive byte buffer
typedef struct { u8 *p; size_t len, cap; } abuf;
static void a_reserve(abuf*b,size_t n){ if(b->len+n<=b->cap)return; size_t nc=b->cap?b->cap*2:1<<20; while(nc<b->len+n)nc*=2; b->p=realloc(b->p,nc); b->cap=nc; }
static size_t a_put(abuf*b,const void*s,size_t n){ a_reserve(b,n); size_t at=b->len; memcpy(b->p+at,s,n); b->len+=n; return at; }
static size_t a_zero(abuf*b,size_t n){ a_reserve(b,n); size_t at=b->len; memset(b->p+at,0,n); b->len+=n; return at; }
static void a_u32(abuf*b,size_t at,uint32_t v){ memcpy(b->p+at,&v,4); }
static void a_u64(abuf*b,size_t at,uint64_t v){ memcpy(b->p+at,&v,8); }

static int g_nchunks=0;
static _Thread_local u8 *g_cmask=0;   // prepared 256^3 air mask for the chunk being written

static void prep_chunk_mask(const vol_t *V,int cz,int cy,int cx){
    if(!g_cmask) g_cmask=malloc((size_t)MC_CHUNK*MC_CHUNK*MC_CHUNK);
    int z0=cz*256,y0=cy*256,x0=cx*256;
    for(int z=0;z<MC_CHUNK;++z)for(int y=0;y<MC_CHUNK;++y)for(int x=0;x<MC_CHUNK;++x)
        g_cmask[((size_t)z*MC_CHUNK+y)*MC_CHUNK+x] = vget(V,z0+z,y0+y,x0+x)?0:1;  // air=1
}

// write one dense 256^3 chunk: [u32 masklen][mask][512B bitmap][present lens][payloads].
static uint64_t write_chunk(abuf*b,const vol_t *V,int cz,int cy,int cx){
    prep_chunk_mask(V,cz,cy,cx);
    static _Thread_local mc_buf tmp; tmp.len=0;
    uint8_t bm[MC_BITMAP_BYTES]; memset(bm,0,sizeof bm);
    uint32_t blen[MC_GRID3]; int npresent=0;
    static _Thread_local u8 vox[MC_BLK*MC_BLK*MC_BLK], rair[MC_BLK*MC_BLK*MC_BLK];
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
        int bi=(bz*16+by)*16+bx;
        if(!gather_blk(V,cz,cy,cx,bz,by,bx,vox)) continue;
        for(int z=0;z<MC_BLK;++z)for(int y=0;y<MC_BLK;++y)for(int x=0;x<MC_BLK;++x){
            size_t ci=((size_t)(bz*MC_BLK+z)*MC_CHUNK+(by*MC_BLK+y))*MC_CHUNK+(bx*MC_BLK+x);
            rair[(z*MC_BLK+y)*MC_BLK+x]=g_cmask[ci];
        }
        uint32_t len=0; if(mc_enc_block(vox,rair,&tmp,&len)){ mc_bit_set(bm,bi); blen[bi]=len; npresent++; }
    }
    size_t at=b->len;
    static _Thread_local mc_u8 *mbuf=0; if(!mbuf) mbuf=malloc((size_t)MC_CHUNK*MC_CHUNK*MC_CHUNK/4+1024);
    uint32_t mlen=mc_enc_chunkmask(g_cmask,mbuf,(size_t)MC_CHUNK*MC_CHUNK*MC_CHUNK/4+1024);
    a_put(b,&mlen,4); a_put(b,mbuf,mlen);
    a_put(b,bm,MC_BITMAP_BYTES);
    for(int bi=0;bi<MC_GRID3;++bi) if(mc_bit_get(bm,bi)) a_put(b,&blen[bi],4);
    if(npresent) a_put(b,tmp.p,tmp.len);
    return (uint64_t)at;
}
static uint64_t write_shard(abuf*b,const vol_t *V,int bz,int by,int bx){
    uint64_t tbl[MC_GRID3]; memset(tbl,0,sizeof tbl); int any=0;
    for(int dz=0;dz<16;++dz){ int cz=bz+dz; if(cz>=g_nchunks)break;
    for(int dy=0;dy<16;++dy){ int cy=by+dy; if(cy>=g_nchunks)break;
    for(int dx=0;dx<16;++dx){ int cx=bx+dx; if(cx>=g_nchunks)break;
        if(chunk_present(V,cz,cy,cx)){ tbl[(dz*16+dy)*16+dx]=write_chunk(b,V,cz,cy,cx); any=1; } }}}
    if(!any) return 0;
    return (uint64_t)a_put(b,tbl,MC_SHARD_BYTES);
}
static uint64_t write_node(abuf*b,const vol_t *V,int level,int bz,int by,int bx){
    int span=16; for(int i=0;i<level;++i) span*=16;
    uint8_t bm[MC_BITMAP_BYTES]; memset(bm,0,sizeof bm);
    uint64_t coff[MC_GRID3]; int any=0;
    for(int dz=0;dz<16;++dz){ int cz=bz+dz*span; if(cz>=g_nchunks)break;
    for(int dy=0;dy<16;++dy){ int cy=by+dy*span; if(cy>=g_nchunks)break;
    for(int dx=0;dx<16;++dx){ int cx=bx+dx*span; if(cx>=g_nchunks)break;
        int idx=(dz*16+dy)*16+dx;
        uint64_t off = level==0 ? write_shard(b,V,cz,cy,cx) : write_node(b,V,level-1,cz,cy,cx);
        if(off){ mc_bit_set(bm,idx); coff[idx]=off; any=1; } }}}
    if(!any) return 0;
    size_t nat=a_put(b,bm,MC_BITMAP_BYTES);
    for(int idx=0;idx<MC_GRID3;++idx) if(mc_bit_get(bm,idx)) a_put(b,&coff[idx],8);
    return (uint64_t)nat;
}

uint8_t *mc_build(mc_voxel_fn src, void *ud, const mc_build_opts *opts, size_t *out_len){
    int V=opts->dim;
    if(V % MC_CHUNK_ALIGN != 0){
        fprintf(stderr,"mc_build: dim %d is not a multiple of %d (chunk-align it upstream)\n",V,MC_CHUNK_ALIGN);
        return NULL;
    }
    mc_codec_init(); mc_set_quality(opts->quality);
    // materialize LOD0 from the source callback.
    u8 *lod0=malloc((size_t)V*V*V);
    if(!lod0){ fprintf(stderr,"mc_build: OOM allocating %dx%dx%d\n",V,V,V); return NULL; }
    for(int z=0;z<V;++z)for(int y=0;y<V;++y)for(int x=0;x<V;++x)
        lod0[((size_t)z*V+y)*V+x]=src(ud,x,y,z);

    abuf b={0}; a_zero(&b,MC_META_END);    // reserve header + metadata region; data starts at 128KB
    size_t mlen=0;
    if(opts->metadata && opts->meta_len){
        mlen=opts->meta_len;
        if(mlen>MC_META_CAP){ fprintf(stderr,"mc_build: metadata %zu B > %u cap, truncating\n",mlen,(unsigned)MC_META_CAP); mlen=MC_META_CAP; }
        memcpy(b.p+MC_HDR, opts->metadata, mlen);
    }
    uint64_t roots[8]={0};
    const u8 *cur=lod0; u8 *owned=NULL; int d=V;
    for(int lod=0; lod<8 && d>=MC_BLK; ++lod){
        vol_t vv={cur,d}; g_nchunks=(d+255)/256;
        roots[lod]=write_node(&b,&vv,MC_SPARSE_LEVELS-1,0,0,0);
        if(d/2<MC_BLK){ ++lod; break; }
        u8 *next=decimate(cur,d);
        if(owned) free(owned); owned=next; cur=next; d/=2;
    }
    if(owned) free(owned);
    free(lod0);
    a_u32(&b,MCH_MAGIC,MC_MAGIC); a_u32(&b,MCH_VER,MC_VERSION);
    a_u32(&b,MCH_NX,V); a_u32(&b,MCH_NY,V); a_u32(&b,MCH_NZ,V);
    for(int l=0;l<8;++l) a_u64(&b,MCH_ROOTOFF+l*8,roots[l]);
    a_u64(&b,MCH_TOTLEN,b.len);
    a_u64(&b,MCH_METAOFF,MC_HDR); a_u64(&b,MCH_METACAP,MC_META_CAP); a_u64(&b,MCH_METALEN,mlen);
    if(out_len) *out_len=b.len;
    return b.p;
}

int mc_build_to_file(mc_voxel_fn src, void *ud, const mc_build_opts *opts, const char *outpath){
    size_t len=0; uint8_t *buf=mc_build(src,ud,opts,&len);
    if(!buf) return 1;
    FILE *of=fopen(outpath,"wb"); if(!of){ perror("fopen out"); free(buf); return 1; }
    fwrite(buf,1,len,of); fclose(of); free(buf);
    return 0;
}

// ---------------------------------------------------------------- reader
const char *mc_metadata(const uint8_t *arc, size_t *out_len){
    uint64_t off,len; memcpy(&off,arc+MCH_METAOFF,8); memcpy(&len,arc+MCH_METALEN,8);
    if(!off) off=MC_HDR;
    if(out_len) *out_len=(size_t)len;
    return (const char*)(arc+off);
}
struct mc_reader { const uint8_t *arc; size_t len; uint64_t roots[8]; u8 *cmask; uint64_t cmask_key; };
mc_reader *mc_open(const uint8_t *arc, size_t len){
    mc_codec_init();
    mc_reader *r=calloc(1,sizeof *r); r->arc=arc; r->len=len; r->cmask_key=~0ull;
    for(int l=0;l<8;++l) memcpy(&r->roots[l], arc+MCH_ROOTOFF+l*8, 8);
    return r;
}
void mc_close(mc_reader *r){ if(!r)return; free(r->cmask); free(r); }
void mc_reader_set_quality(mc_reader *r, float q){ (void)r; mc_set_quality(q); }
uint64_t mc_chunk_offset(mc_reader *r,int lod,int cz,int cy,int cx){
    if(lod<0||lod>7) return 0;
    return mc_resolve_chunk(r->arc,r->roots[lod],cz,cy,cx);
}
void mc_decode_block(mc_reader *r, uint64_t chunk_off, int bz,int by,int bx, mc_u8 *dst){
    if(!chunk_off){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    if(r->cmask_key!=chunk_off){
        if(!r->cmask) r->cmask=malloc((size_t)MC_CHUNK*MC_CHUNK*MC_CHUNK);
        uint32_t cml; const u8 *cmb=mc_chunk_mask(r->arc,chunk_off,&cml);
        if(cmb) mc_dec_chunkmask(cmb,cml,r->cmask); else memset(r->cmask,0,(size_t)MC_CHUNK*MC_CHUNK*MC_CHUNK);
        r->cmask_key=chunk_off;
    }
    static _Thread_local mc_u8 rair[MC_BLK*MC_BLK*MC_BLK];
    for(int z=0;z<MC_BLK;++z)for(int y=0;y<MC_BLK;++y)for(int x=0;x<MC_BLK;++x){
        size_t ci=((size_t)(bz*MC_BLK+z)*MC_CHUNK+(by*MC_BLK+y))*MC_CHUNK+(bx*MC_BLK+x);
        rair[(z*MC_BLK+y)*MC_BLK+x]=r->cmask[ci];
    }
    uint64_t boff; uint32_t blen;
    if(!mc_block_range(r->arc,chunk_off,bz,by,bx,&boff,&blen)){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    mc_dec_block(r->arc+boff,rair,dst);
}
