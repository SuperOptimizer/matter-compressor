#define _GNU_SOURCE
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "matter_compressor.h"
#include "mc_codec.h"
#include "mc_archive.h"
typedef uint8_t u8;
// ============================================================================
// READER (flat buffer + streaming byte-source)
// ============================================================================
// The metadata region is an ARCHITECTURAL invariant: always at MC_HDR, at most
// MC_META_CAP bytes (data starts at MC_META_END=128KB). Bound against that fixed
// layout, NOT the attacker-controlled MCH_METAOFF/TOTLEN fields.
const char *mc_metadata(const uint8_t *arc, size_t *out_len){
    if(!arc){ if(out_len)*out_len=0; return NULL; }
    uint64_t len; memcpy(&len,arc+MCH_METALEN,8);
    if(len > MC_META_CAP) len = MC_META_CAP;
    if(out_len) *out_len=(size_t)len;
    return (const char*)(arc+MC_HDR);
}

#define MC_RD_NODE_CACHE 512   // cached node tables per streaming reader (512*32KB = 16MB):
                               // a pan across a large volume touches many subtrees; a
                               // churned-out table costs a serial ranged GET to re-fetch
struct mc_reader {
    const uint8_t *arc;        // flat mode: archive buffer; streaming: NULL
    size_t len;
    uint64_t roots[8];
    int nx, ny, nz;            // LOD0 dims from the header
    float quality;             // quality the archive was built at
    uint32_t block_codec;      // MC_BLOCKCODEC_* (header MCH_BLOCKCODEC; 0=CABAC)
    u8 priors[MC_PRIORS_BYTES]; int has_priors;   // raw per-volume prior blob
    mc_read_fn read; void *read_ud;   // streaming mode
    // streaming scratch: a fetched window of the current chunk blob.
    u8 *cbuf; uint64_t cbuf_off; uint64_t cbuf_len;
    // partial-fetch mode: per-chunk header cache (bitmap + lens) + one payload.
    int partial;
    u8 hdr[MC_BLOB_HDR + MC_BITMAP_BYTES + MC_GRID3*2]; uint64_t hdr_off; int hdr_np; uint16_t hdr_fml;
    u8 *pbuf; size_t pbuf_cap;
    // streaming node-table cache: resolving a chunk needs 3 dependent 32KB
    // table reads; without a cache every resolve re-fetches them (3 GETs per
    // chunk over S3). FIFO of the last MC_RD_NODE_CACHE tables.
    uint64_t ntab_off[MC_RD_NODE_CACHE];
    u8 *ntab[MC_RD_NODE_CACHE];
    int ntab_next;
    // owned per-reader codec context (a reader's decode scratch — cbuf/pbuf/hdr
    // — is already non-reentrant, so one codec ctx per reader is the right scope).
    mc_codec_ctx *codec;
};

static void reader_hdr_load(mc_reader *r, const u8 *hdr){
    for(int l=0;l<8;++l) memcpy(&r->roots[l], hdr+MCH_ROOTOFF+l*8, 8);
    uint32_t ux,uy,uz;
    memcpy(&ux,hdr+MCH_NX,4); memcpy(&uy,hdr+MCH_NY,4); memcpy(&uz,hdr+MCH_NZ,4);
    r->nx=(int)ux; r->ny=(int)uy; r->nz=(int)uz;
    memcpy(&r->quality,hdr+MCH_QUALITY,4);
    memcpy(&r->block_codec,hdr+MCH_BLOCKCODEC,4);   // 0 = CABAC (old archives)
    r->block_codec=MC_BLOCKCODEC_CABAC;   // c3g removed: all archives are CABAC
}

mc_reader *mc_open(const uint8_t *arc, size_t len){
    // Untrusted-input gate: a valid flat archive is >= the 256-byte header with
    // the right magic. Reject shorter/garbage so no downstream path reads header
    // fields past the buffer. (Streaming uses mc_open_streaming.)
    if(!arc || len < MC_HDR) return NULL;
    uint32_t magic; memcpy(&magic,arc+MCH_MAGIC,4);
    if(magic != MC_MAGIC) return NULL;
    mc_codec_init();
    mc_reader *r=calloc(1,sizeof *r); if(!r) return NULL;
    r->arc=arc; r->len=len;
    r->codec=mc_codec_ctx_new();
    reader_hdr_load(r, arc);
    // Optional prior blob: install only if the full blob fits + magic matches.
    uint64_t poff; memcpy(&poff,arc+MCH_PRIOROFF,8);
    if(poff && poff <= len && len - poff >= (uint64_t)MC_PRIORS_BYTES){
        uint32_t pmagic; memcpy(&pmagic,arc+poff,4);
        if(pmagic==MC_PRIORS_MAGIC){
            memcpy(r->priors,arc+poff,MC_PRIORS_BYTES); r->has_priors=1;
            priors_load(arc);                       // offset validated -> safe
        }
    }
    return r;
}

// streaming: fetch exactly len bytes at off into dst (via callback).
static int sread(mc_reader *r, uint64_t off, uint32_t len, u8 *dst){
    return r->read(r->read_ud, off, len, dst);
}

mc_reader *mc_open_streaming(mc_read_fn read, void *ud, uint64_t total_len){
    mc_codec_init();
    mc_reader *r=calloc(1,sizeof *r); r->read=read; r->read_ud=ud; r->len=(size_t)total_len;
    r->codec=mc_codec_ctx_new();
    u8 hdr[MC_HDR];
    if(read(ud,0,MC_HDR,hdr)!=0){ mc_codec_ctx_free(r->codec); free(r); return NULL; }
    uint32_t magic; memcpy(&magic,hdr+MCH_MAGIC,4);
    if(magic!=MC_MAGIC){ mc_codec_ctx_free(r->codec); free(r); return NULL; }
    reader_hdr_load(r, hdr);
    // per-volume priors: flat open gets them via priors_load(arc); a streaming
    // reader must pull the blob through the callback or HF decode is wrong.
    uint64_t poff; memcpy(&poff,hdr+MCH_PRIOROFF,8);
    if(poff){
        u8 pb[MC_PRIORS_BYTES];
        uint32_t pm=0;
        if(read(ud,poff,MC_PRIORS_BYTES,pb)==0) memcpy(&pm,pb,4);
        if(pm==MC_PRIORS_MAGIC){
            memcpy(r->priors,pb,MC_PRIORS_BYTES); r->has_priors=1;
            mc_codec_set_priors((const uint16_t*)(pb+8),(const uint16_t*)(pb+8+8*32*2));
        } else {
            mc_codec_set_priors(NULL,NULL);
        }
    } else {
        mc_codec_set_priors(NULL,NULL);
    }
    return r;
}

// Raw per-volume prior arrays (plo/phi as u16[8][32]); 0 if the archive has none.
// Feed into mc_archive_set_priors to make a local mirror decode identically.
int mc_reader_priors(mc_reader *r, const uint16_t **plo, const uint16_t **phi){
    if(!r||!r->has_priors) return 0;
    if(plo)*plo=(const uint16_t*)(r->priors+8);
    if(phi)*phi=(const uint16_t*)(r->priors+8+8*32*2);
    return 1;
}

// Total byte length of the chunk blob at `chunk_off` (flat or streaming reader).
// 0 on error. Pair with mc_chunk_offset to range-copy compressed chunks verbatim.
uint64_t mc_reader_chunk_blob_len(mc_reader *r, uint64_t chunk_off){
    if(!r||!chunk_off) return 0;
    if(r->arc) return mc_chunk_blob_len(r->arc, chunk_off);
    u8 h[MC_BLOB_HDR];
    if(sread(r,chunk_off,MC_BLOB_HDR,h)!=0) return 0;
    uint16_t fml; memcpy(&fml,h+MC_BLOB_HDR-2,2);
    const uint64_t bm_off = chunk_off + MC_BLOB_HDR + fml;
    u8 bm[MC_BITMAP_BYTES];
    if(sread(r,bm_off,MC_BITMAP_BYTES,bm)!=0) return 0;
    int np=0; for(int i=0;i<MC_BITMAP_BYTES;++i) np+=__builtin_popcount(bm[i]);
    if(!np) return bm_off + MC_BITMAP_BYTES - chunk_off;
    u8 *lens=malloc((size_t)np*2);
    if(!lens||sread(r,bm_off+MC_BITMAP_BYTES,(uint32_t)((size_t)np*2),lens)!=0){ free(lens); return 0; }
    uint64_t pay=0; for(int i=0;i<np;++i){ uint16_t l; memcpy(&l,lens+(size_t)i*2,2); pay+=l; }
    free(lens);
    return bm_off + MC_BITMAP_BYTES + (uint64_t)np*2 + pay - chunk_off;
}

// Copy `len` bytes of the chunk blob at `chunk_off` into `dst` (flat or streaming
// reader). For .mca -> .mca verbatim copy: mc_reader_chunk_blob_len then this then
// mc_archive_append_chunk_compressed. NOT thread-safe on a streaming reader (single
// cbuf/codec ctx) -- caller serializes. Returns 0 on success.
int mc_reader_read_blob(mc_reader *r, uint64_t chunk_off, size_t len, uint8_t *dst){
    if(!r||!chunk_off||!len||!dst) return -1;
    if(r->arc){ memcpy(dst, r->arc + chunk_off, len); return 0; }
    // streaming: range-read in <=2^31 chunks (sread takes a uint32 len).
    size_t done=0;
    while(done<len){
        uint32_t n = (len-done > 0x40000000u) ? 0x40000000u : (uint32_t)(len-done);
        if(sread(r, chunk_off+done, n, dst+done)!=0) return -1;
        done += n;
    }
    return 0;
}

void mc_reader_dims(mc_reader *r, int *nx, int *ny, int *nz){
    if(nx)*nx=r?r->nx:0; if(ny)*ny=r?r->ny:0; if(nz)*nz=r?r->nz:0;
}
float mc_reader_quality(mc_reader *r){ return r?r->quality:0.f; }
int mc_reader_nlods(mc_reader *r){
    if(!r) return 0;
    int n=0; for(int l=0;l<8;++l) if(r->roots[l]) n=l+1;
    return n;
}

void mc_close(mc_reader *r){ if(!r)return;
    for(int i=0;i<MC_RD_NODE_CACHE;++i) free(r->ntab[i]);
    mc_codec_ctx_free(r->codec);
    free(r->cbuf); free(r->pbuf); free(r); }
// Partial-fetch mode (streaming readers only): decode a block by fetching just
// the chunk's bitmap+length table (cached per chunk, <=8.7KB) plus that block's
// own payload, instead of the whole chunk blob. Wins cold random-access latency
// over high-latency byte sources (S3); leave OFF when scanning whole chunks.
void mc_reader_set_partial_fetch(mc_reader *r, int on){ if(r){ r->partial=on; r->hdr_off=~0ull; } }
void mc_reader_set_quality(mc_reader *r, float q){ if(r&&r->codec) mc_codec_ctx_set_quality(r->codec,q); }

// streaming chunk-offset resolve: pull node tables on demand (each is
// MC_NODE_BYTES), memoized in the reader's FIFO node-table cache so repeated
// resolves (scans, neighborhoods) cost zero extra reads.
static const u8 *sfetch_node(mc_reader *r, uint64_t off){
    for(int i=0;i<MC_RD_NODE_CACHE;++i)
        if(r->ntab[i] && r->ntab_off[i]==off) return r->ntab[i];
    int slot=r->ntab_next; r->ntab_next=(slot+1)%MC_RD_NODE_CACHE;
    if(!r->ntab[slot]) r->ntab[slot]=malloc(MC_NODE_BYTES);
    if(sread(r,off,MC_NODE_BYTES,r->ntab[slot])!=0){ free(r->ntab[slot]); r->ntab[slot]=0; return NULL; }
    r->ntab_off[slot]=off;
    return r->ntab[slot];
}
static uint64_t sresolve_chunk(mc_reader *r,int lod,int cz,int cy,int cx,int *err){
    uint64_t node = r->roots[lod];
    for(int nib=MC_TREE_LEVELS-1; nib>=0; --nib){
        if(!node) return 0;                       // genuinely absent (air)
        const u8 *tbl=sfetch_node(r,node);
        if(!tbl){ if(err)*err=1; return 0; }      // FETCH FAILED -- not absent!
        int idx=(mc_nib(cz,nib)*16+mc_nib(cy,nib))*16+mc_nib(cx,nib);
        uint64_t child; memcpy(&child,tbl+(size_t)idx*8,8);
        node=child;
    }
    return node;
}

// As mc_chunk_offset, but distinguishes "resolved to absent" (ret 0, *err 0)
// from "node-table read FAILED" (ret 0, *err 1). A streaming caller that maps
// offset 0 to permanent air MUST use this: a transient network error (expired
// creds, timeout) otherwise poisons the region as ZERO forever.
uint64_t mc_chunk_offset_chk(mc_reader *r,int lod,int cz,int cy,int cx,int *err){
    if(err)*err=0;
    if(!r||lod<0||lod>7) return 0;
    if(r->arc) return mc_resolve_chunk(r->arc,r->roots[lod],cz,cy,cx,r->len);  // flat: bounded
    return sresolve_chunk(r,lod,cz,cy,cx,err);
}

uint64_t mc_chunk_offset(mc_reader *r,int lod,int cz,int cy,int cx){
    if(lod<0||lod>7) return 0;
    if(r->arc) return mc_resolve_chunk(r->arc,r->roots[lod],cz,cy,cx,r->len);
    return sresolve_chunk(r,lod,cz,cy,cx,NULL);
}

// streaming: ensure the whole chunk blob at chunk_off is cached in r->cbuf, return ptr.
static const u8 *sfetch_chunk(mc_reader *r, uint64_t chunk_off){
    if(r->cbuf && r->cbuf_off==chunk_off) return r->cbuf;
    // fetch header (q/hash/fmaplen), then bitmap + lens for total length, then
    // the full blob in one GET.
    u8 bh[MC_BLOB_HDR];
    if(sread(r,chunk_off,MC_BLOB_HDR,bh)!=0) return NULL;
    uint16_t fml; memcpy(&fml,bh+12,2);
    uint64_t bm_off = chunk_off + MC_BLOB_HDR + fml;
    u8 bm[MC_BITMAP_BYTES];
    if(sread(r,bm_off,MC_BITMAP_BYTES,bm)!=0) return NULL;
    int npresent=0; for(int i=0;i<MC_BITMAP_BYTES;++i) npresent+=__builtin_popcount(bm[i]);
    u8 *lens=malloc((size_t)npresent*2);
    if(npresent && !lens) return NULL;
    if(npresent && sread(r,bm_off+MC_BITMAP_BYTES,(uint32_t)(npresent*2),lens)!=0){ free(lens); return NULL; }
    uint64_t paybytes=0; for(int s=0;s<npresent;++s){ uint16_t l; memcpy(&l,lens+(size_t)s*2,2); paybytes+=l; }
    free(lens);
    uint64_t total = (bm_off+MC_BITMAP_BYTES+(uint64_t)npresent*2+paybytes) - chunk_off;
    { void *tmp=realloc(r->cbuf,total); if(!tmp) return NULL; r->cbuf=tmp; }
    if(sread(r,chunk_off,(uint32_t)total,r->cbuf)!=0) return NULL;
    r->cbuf_off=chunk_off; r->cbuf_len=total;
    return r->cbuf;
}

// partial-fetch path: header cache + single-payload range read.
static int spartial_decode(mc_reader *r, uint64_t chunk_off, int bi, mc_u8 *dst){
    if(r->hdr_off!=chunk_off){
        if(sread(r,chunk_off,MC_BLOB_HDR,r->hdr)!=0) return -1;
        uint16_t fml; memcpy(&fml,r->hdr+12,2);
        r->hdr_fml=fml;
        if(sread(r,chunk_off+MC_BLOB_HDR+fml,MC_BITMAP_BYTES,r->hdr+MC_BLOB_HDR)!=0) return -1;
        const u8 *bm0=r->hdr+MC_BLOB_HDR;
        int np=0; for(int i=0;i<MC_BITMAP_BYTES;++i) np+=__builtin_popcount(bm0[i]);
        if(np && sread(r,chunk_off+MC_BLOB_HDR+fml+MC_BITMAP_BYTES,(uint32_t)(np*2),r->hdr+MC_BLOB_HDR+MC_BITMAP_BYTES)!=0) return -1;
        r->hdr_np=np; r->hdr_off=chunk_off;
    }
    { float q; memcpy(&q,r->hdr,4);
      if(!(q > 0.0f) || q > 1024.0f) q = 1.0f;   // untrusted blob: reject NaN/Inf/OOR
      mc_codec_ctx_set_quality(r->codec,q); }
    const u8 *bm=r->hdr+MC_BLOB_HDR;
    if(!mc_bit_get(bm,bi)){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return 0; }
    int slot=mc_rank(bm,bi);
    const u8 *lens=bm+MC_BITMAP_BYTES;
    uint64_t cum=0; for(int s2=0;s2<slot;++s2){ uint16_t l; memcpy(&l,lens+(size_t)s2*2,2); cum+=l; }
    uint16_t mylen; memcpy(&mylen,lens+(size_t)slot*2,2);
    uint64_t pay=chunk_off+MC_BLOB_HDR+r->hdr_fml+MC_BITMAP_BYTES+(uint64_t)r->hdr_np*2+cum;
    if(r->pbuf_cap<mylen){ void *tmp=realloc(r->pbuf,mylen); if(!tmp) return -1; r->pbuf=tmp; r->pbuf_cap=mylen; }
    if(sread(r,pay,mylen,r->pbuf)!=0) return -1;
    mc_dec_block_codec(r->codec,r->block_codec,r->pbuf,mylen,dst);
    return 0;
}

void mc_decode_block(mc_reader *r, uint64_t chunk_off, int bz,int by,int bx, mc_u8 *dst){
    if(chunk_off<=MC_SLOT_ZERO){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    // Flat reader on untrusted bytes: validate the whole chunk-blob structure is
    // in-bounds BEFORE reading any blob field (mc_chunk_q/mc_block_range read
    // header/fmap/bitmap/lens at chunk_off-derived offsets). Streaming fetches
    // bounded windows already.
    if(r->arc && !mc_blob_struct_ok(r->arc, chunk_off, r->len)){
        memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return;
    }
    if(!r->arc && r->partial){
        if(spartial_decode(r,chunk_off,(bz*16+by)*16+bx,dst)==0) return;
        memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return;
    }
    // resolve the chunk-blob base pointer (flat mmap or streamed window).
    const u8 *blob_base;       // points at the chunk blob start
    uint64_t blob_origin;      // absolute archive offset of that blob start
    if(r->arc){ blob_base=r->arc; blob_origin=0; }            // absolute offsets index into arc
    else {
        const u8 *cb = sfetch_chunk(r,chunk_off);
        if(!cb){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
        blob_base = cb - chunk_off;   // so blob_base + abs_off lands inside the window
        blob_origin = 0;
    }
    (void)blob_origin;
    // Per-chunk quality is an f32 from the (untrusted) blob; NaN/Inf/OOR poisons
    // the quant-table interpolation (NaN -> uint16_t cast is UB). Clamp.
    float cq = mc_chunk_q(blob_base,chunk_off);
    if(!(cq > 0.0f) || cq > 1024.0f) cq = 1.0f;
    mc_codec_ctx_set_quality(r->codec,cq);
    uint64_t boff; uint32_t blen;
    if(!mc_block_range(blob_base,chunk_off,bz,by,bx,&boff,&blen)){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    mc_dec_block_codec(r->codec,r->block_codec,blob_base+boff,blen,dst);
}
