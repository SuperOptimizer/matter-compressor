// ============================================================================
// mc_archive.c — matter-compressor archive build + APPENDABLE writer + decode.
// See mc_archive_api.h. Source-agnostic: no zarr/S3 here.
//
// Index is a DENSE node tree (root node -> inner node -> shard), each a flat
// MC_GRID3 array of u64 offsets, updatable in place. Chunk payloads append at EOF.
//
// Three faces:
//   - mc_build / mc_build_to_file : one-shot build of a whole volume (RAM-materialized).
//   - mc_archive_* : ONE persistent, crash-safe, appendable on-disk handle that both
//                    APPENDS chunks and DECODES them (no writer/reader split). mmap +
//                    atomic append cursor + in-place dense-node index + release-published
//                    commit, modeled on volume-compressor's writer; decode reads the live
//                    mmap.
//   - mc_open / mc_open_streaming + decode : read an already-built archive from a buffer
//                    or a byte-source (streaming / S3).
// ============================================================================
#include "mc_archive_api.h"
#include "mc_archive.h"
#include "mc_archive_read.h"
#include "mc_xxhash.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdatomic.h>
#include <errno.h>
#include <math.h>
#include <stdatomic.h>

#if defined(__unix__) || defined(__APPLE__)
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
  #include <fcntl.h>
  #include <pthread.h>
  #define MC_HAVE_MMAP 1
#else
  #define MC_HAVE_MMAP 0
#endif

typedef uint8_t u8;
static void priors_load(const uint8_t *base);

// ---------------------------------------------------------------- volume (one LOD)
// A contiguous dim^3 u8 buffer. LOD0 is materialized from the source callback; coarser
// LODs are box-decimated from the finer one.
typedef struct { const u8 *v; int nz, ny, nx; } vol_t;
static inline u8 vget(const vol_t *V, int z,int y,int x){
    if((unsigned)z>=(unsigned)V->nz||(unsigned)y>=(unsigned)V->ny||(unsigned)x>=(unsigned)V->nx) return 0;
    return V->v[((size_t)z*V->ny+y)*V->nx+x];
}
// 2x box-decimate: mean of NONZERO children; all-zero stays 0 (inherited zero-mask).
// Per-axis dims; odd source dims round up (the edge voxel decimates alone).
static u8 *decimate(const u8 *src,int DZ,int DY,int DX,int *HZ,int *HY,int *HX){
    int hz=(DZ+1)/2, hy=(DY+1)/2, hx=(DX+1)/2;
    u8 *o=calloc((size_t)hz*hy*hx,1);
    for(int z=0;z<hz;++z)for(int y=0;y<hy;++y)for(int x=0;x<hx;++x){ int s=0,c=0;
        for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
            int sz=2*z+dz, sy=2*y+dy, sx=2*x+dx;
            if(sz>=DZ||sy>=DY||sx>=DX) continue;
            u8 v=src[((size_t)sz*DY+sy)*DX+sx]; if(v){s+=v;c++;}}
        o[((size_t)z*hy+y)*hx+x]=c?(u8)((s+c/2)/c):0; }
    *HZ=hz; *HY=hy; *HX=hx; return o; }

// gather a 16^3 block from a contiguous 256^3 chunk buffer (chunk is dense, no edges).
static int gather_blk256(const u8 *chunk,int bz,int by,int bx,u8 *dst){
    int z0=bz*MC_BLK,y0=by*MC_BLK,x0=bx*MC_BLK,any=0;
    for(int z=0;z<MC_BLK;++z)for(int y=0;y<MC_BLK;++y)for(int x=0;x<MC_BLK;++x){
        u8 v=chunk[((size_t)(z0+z)*MC_CHUNK+(y0+y))*MC_CHUNK+(x0+x)]; dst[(z*MC_BLK+y)*MC_BLK+x]=v; any|=v; }
    return any;
}

// ---------------------------------------------------------------- chunk-blob encoder
// Encode one dense 256^3 chunk buffer into a compressed blob in `out` (a growable
// byte sink: out_put(out, ptr, n)). Returns the blob length, or 0 if the chunk is all
// air (no blob — caller leaves the slot absent). Shared by build + writer.
// blob (v6) = [f32 q][u64 xxh64][512B block-bitmap][present-block u16 lens]
// [block payloads]; blocks are self-contained (mask in payload). q is the
// chunk's own quality; the hash covers bitmap+lens+payloads.
typedef void (*out_put_fn)(void *out, const void *s, size_t n);

static size_t encode_chunk_blob(const u8 *chunk256, out_put_fn put, void *out){
    static _Thread_local mc_buf tmp; tmp.len=0;
    uint8_t bm[MC_BITMAP_BYTES]; memset(bm,0,sizeof bm);
    uint16_t blen[MC_GRID3]; int npresent=0;
    static _Thread_local uint8_t frac[MC_GRID3];
    memset(frac,0,MC_GRID3);
    static _Thread_local u8 vox[MC_BLK*MC_BLK*MC_BLK];
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
        int bi=(bz*16+by)*16+bx;
        if(!gather_blk256(chunk256,bz,by,bx,vox)) continue;
        int cnt=0; for(int i=0;i<MC_BLK*MC_BLK*MC_BLK;++i) cnt+=vox[i]!=0;
        frac[bi]=(uint8_t)((cnt*15+2048)/4096);              // nibble 0..15
        if(cnt&&!frac[bi]) frac[bi]=1;                        // any material -> >=1
        uint32_t len=0; if(mc_enc_block(vox,&tmp,&len)){ mc_bit_set(bm,bi); blen[bi]=(uint16_t)len; npresent++; }
    }
    if(!npresent) return 0;   // all air -> no blob
    // v7 blob header: [f32 q][u64 xxh64][u16 fmaplen][fmap]
    float q=mc_get_quality();
    static _Thread_local uint16_t lens16[MC_GRID3]; int nl=0;
    for(int bi=0;bi<MC_GRID3;++bi) if(mc_bit_get(bm,bi)) lens16[nl++]=blen[bi];
    static _Thread_local uint8_t fmap[MC_GRID3/2+64];
    uint16_t fml=(uint16_t)mc_enc_fracmap(frac,fmap,sizeof fmap);
    uint64_t h=mc_xxh64(fmap,fml,0x6D636368756E6Bull);
    h^=mc_xxh64(bm,MC_BITMAP_BYTES,h);
    h^=mc_xxh64(lens16,(size_t)nl*2,h);
    h^=mc_xxh64(tmp.p,tmp.len,h);
    size_t total=MC_BLOB_HDR+fml+MC_BITMAP_BYTES+(size_t)npresent*2+tmp.len;
    put(out,&q,4); put(out,&h,8); put(out,&fml,2);
    put(out,fmap,fml);
    put(out,bm,MC_BITMAP_BYTES);
    put(out,lens16,(size_t)nl*2);
    put(out,tmp.p,tmp.len);
    return total;
}

// recompute a chunk blob's hash from its bytes (mc_verify / verify-on-decode).
uint64_t mc_chunk_compute_hash(const uint8_t *blob, uint64_t blob_len){
    uint16_t fml; memcpy(&fml,blob+12,2);
    const uint8_t *fmap=blob+MC_BLOB_HDR;
    const uint8_t *bm=fmap+fml;
    int npresent=0; for(int i=0;i<MC_BITMAP_BYTES;++i) npresent+=__builtin_popcount(bm[i]);
    const uint8_t *lens=bm+MC_BITMAP_BYTES;
    uint64_t pay=(uint64_t)MC_BLOB_HDR+fml+MC_BITMAP_BYTES+(uint64_t)npresent*2;
    uint64_t h=mc_xxh64(fmap,fml,0x6D636368756E6Bull);
    h^=mc_xxh64(bm,MC_BITMAP_BYTES,h);
    h^=mc_xxh64(lens,(size_t)npresent*2,h);
    h^=mc_xxh64(blob+pay,(size_t)(blob_len-pay),h);
    return h;
}

// ---------------------------------------------------------------- one-shot build (abuf)
typedef struct { u8 *p; size_t len, cap; } abuf;
static void a_reserve(abuf*b,size_t n){ if(b->len+n<=b->cap)return; size_t nc=b->cap?b->cap*2:1<<20; while(nc<b->len+n)nc*=2; b->p=realloc(b->p,nc); b->cap=nc; }
static size_t a_put_at(abuf*b,const void*s,size_t n){ a_reserve(b,n); size_t at=b->len; memcpy(b->p+at,s,n); b->len+=n; return at; }
static size_t a_zero(abuf*b,size_t n){ a_reserve(b,n); size_t at=b->len; memset(b->p+at,0,n); b->len+=n; return at; }
static void a_u32(abuf*b,size_t at,uint32_t v){ memcpy(b->p+at,&v,4); }
static void a_u64(abuf*b,size_t at,uint64_t v){ memcpy(b->p+at,&v,8); }
static void abuf_put(void *out, const void *s, size_t n){ a_put_at((abuf*)out,s,n); }

// gather a contiguous 256^3 chunk out of a LOD volume (zero-padded at the volume edge).
static int gather_chunk256(const vol_t *V,int cz,int cy,int cx,u8 *dst){
    int z0=cz*256,y0=cy*256,x0=cx*256,any=0;
    for(int z=0;z<MC_CHUNK;++z)for(int y=0;y<MC_CHUNK;++y)for(int x=0;x<MC_CHUNK;++x){
        u8 v=vget(V,z0+z,y0+y,x0+x); dst[((size_t)z*MC_CHUNK+y)*MC_CHUNK+x]=v; any|=v; }
    return any;
}

// dense one-shot build: write the dense node tree for one LOD volume. Allocate the
// dense tables top-down with deferred offset back-patch (we know child offsets only
// after writing children, but the tables are dense + fixed-size so we reserve then fill).
// Simpler: build bottom-up into an in-RAM dense tree, then emit. We emit chunks first,
// recording offsets in a dense shard map, then emit shards, inner nodes, root.

static uint64_t build_lod_dense(abuf*b, const vol_t *V, int ncz,int ncy,int ncx){
    // dense maps over chunk grid (sparse population). We allocate per populated node.
    // root: 16^3 inner-node offsets; inner: 16^3 shard offsets; shard: 16^3 chunk offsets.
    // chunk coord nibbles: 2 (root), 1 (inner), 0 (shard).
    // First pass: emit chunk blobs, fill shard tables in RAM keyed by (n1=hi, n0).
    // To keep it simple + correct for any nchunks up to 4096/axis, use hash-free dense
    // allocation only where present.
    // We build a 3-level RAM tree of u64 tables, emit chunk blobs to `b`, then emit the
    // tables bottom-up and return the root offset.

    // RAM node: 4096 u64 child *table-indices* during build, resolved to file offsets on emit.
    // Use dynamic arrays of tables.
    typedef struct { uint64_t slot[MC_GRID3]; } table_t;   // 32KB each
    // shards keyed by (n2,n1); inners keyed by n2; root single.
    // We accumulate present shards/inners in vectors with their grid index.
    table_t *root = calloc(1,sizeof *root);
    // inner tables: up to 4096; allocate lazily.
    table_t **inner = calloc(MC_GRID3,sizeof *inner);
    // for each inner, its shard tables lazily.
    table_t ***shard = calloc(MC_GRID3,sizeof *shard);

    static _Thread_local u8 *chunkbuf=0; if(!chunkbuf) chunkbuf=malloc((size_t)MC_CHUNK*MC_CHUNK*MC_CHUNK);

    for(int cz=0;cz<ncz;++cz)for(int cy=0;cy<ncy;++cy)for(int cx=0;cx<ncx;++cx){
        if(!gather_chunk256(V,cz,cy,cx,chunkbuf)) continue;
        size_t at=b->len;
        if(!encode_chunk_blob(chunkbuf,abuf_put,b)) continue;   // all air
        int n2=(mc_nib(cz,2)*16+mc_nib(cy,2))*16+mc_nib(cx,2);
        int n1=(mc_nib(cz,1)*16+mc_nib(cy,1))*16+mc_nib(cx,1);
        int n0=(mc_nib(cz,0)*16+mc_nib(cy,0))*16+mc_nib(cx,0);
        if(!inner[n2]) inner[n2]=calloc(1,sizeof(table_t));
        if(!shard[n2]) shard[n2]=calloc(MC_GRID3,sizeof(table_t*));
        if(!shard[n2][n1]) shard[n2][n1]=calloc(1,sizeof(table_t));
        shard[n2][n1]->slot[n0]=(uint64_t)at;   // chunk-blob offset
    }
    // emit bottom-up: shards, then inners, then root.
    int any_root=0;
    for(int n2=0;n2<MC_GRID3;++n2){
        if(!inner[n2]) continue;
        int any_inner=0;
        for(int n1=0;n1<MC_GRID3;++n1){
            if(!shard[n2] || !shard[n2][n1]) continue;
            uint64_t soff=a_put_at(b,shard[n2][n1],MC_SHARD_BYTES);
            inner[n2]->slot[n1]=soff; any_inner=1;
        }
        if(any_inner){ uint64_t ioff=a_put_at(b,inner[n2],MC_NODE_BYTES); root->slot[n2]=ioff; any_root=1; }
    }
    uint64_t root_off=0;
    if(any_root) root_off=a_put_at(b,root,MC_NODE_BYTES);

    // free RAM tree
    for(int n2=0;n2<MC_GRID3;++n2){
        if(shard[n2]){ for(int n1=0;n1<MC_GRID3;++n1) free(shard[n2][n1]); free(shard[n2]); }
        free(inner[n2]);
    }
    free(shard); free(inner); free(root);
    return root_off;
}

uint8_t *mc_build(mc_voxel_fn src, void *ud, const mc_build_opts *opts, size_t *out_len){
    int NX = opts->nx>0?opts->nx:opts->dim;
    int NY = opts->ny>0?opts->ny:opts->dim;
    int NZ = opts->nz>0?opts->nz:opts->dim;
    if(NX<=0||NY<=0||NZ<=0){ fprintf(stderr,"mc_build: bad dims %dx%dx%d\n",NX,NY,NZ); return NULL; }
    // pad each axis to the chunk boundary (zero padding is nearly free)
    int PX=(NX+MC_CHUNK_ALIGN-1)/MC_CHUNK_ALIGN*MC_CHUNK_ALIGN;
    int PY=(NY+MC_CHUNK_ALIGN-1)/MC_CHUNK_ALIGN*MC_CHUNK_ALIGN;
    int PZ=(NZ+MC_CHUNK_ALIGN-1)/MC_CHUNK_ALIGN*MC_CHUNK_ALIGN;
    mc_codec_init(); mc_set_quality(opts->quality);
    u8 *lod0=calloc((size_t)PZ*PY,(size_t)PX);
    if(!lod0){ fprintf(stderr,"mc_build: OOM allocating %dx%dx%d\n",PZ,PY,PX); return NULL; }
    for(int z=0;z<NZ;++z)for(int y=0;y<NY;++y)for(int x=0;x<NX;++x)
        lod0[((size_t)z*PY+y)*PX+x]=src(ud,x,y,z);

    abuf b={0}; a_zero(&b,MC_META_END);
    size_t mlen=0;
    if(opts->metadata && opts->meta_len){
        mlen=opts->meta_len;
        if(mlen>MC_META_CAP){ fprintf(stderr,"mc_build: metadata %zu B > %u cap, truncating\n",mlen,(unsigned)MC_META_CAP); mlen=MC_META_CAP; }
        memcpy(b.p+MC_HDR, opts->metadata, mlen);
    }
    uint64_t roots[8]={0};
    const u8 *cur=lod0; u8 *owned=NULL;
    int dz=PZ, dy=PY, dx=PX;
    for(int lod=0; lod<8 && (dz>=1&&dy>=1&&dx>=1) && (dz>=MC_CHUNK||dy>=MC_CHUNK||dx>=MC_CHUNK||lod==0); ++lod){
        vol_t vv={cur,dz,dy,dx};
        int ncz=(dz+255)/256, ncy=(dy+255)/256, ncx=(dx+255)/256;
        roots[lod]=build_lod_dense(&b,&vv,ncz,ncy,ncx);
        if(dz/2<1||dy/2<1||dx/2<1) break;
        if(dz/2<MC_CHUNK&&dy/2<MC_CHUNK&&dx/2<MC_CHUNK&&lod>=1) break;
        int hz,hy,hx;
        u8 *next=decimate(cur,dz,dy,dx,&hz,&hy,&hx);
        if(owned) free(owned); owned=next; cur=next; dz=hz; dy=hy; dx=hx;
    }
    if(owned && owned!=lod0) free(owned);
    free(lod0);
    float q=opts->quality;
    a_u32(&b,MCH_MAGIC,MC_MAGIC); a_u32(&b,MCH_VER,MC_VERSION);
    a_u32(&b,MCH_NX,(uint32_t)NX); a_u32(&b,MCH_NY,(uint32_t)NY); a_u32(&b,MCH_NZ,(uint32_t)NZ);
    for(int l=0;l<8;++l) a_u64(&b,MCH_ROOTOFF+l*8,roots[l]);
    a_u64(&b,MCH_TOTLEN,b.len);
    a_u64(&b,MCH_METAOFF,MC_HDR); a_u64(&b,MCH_METACAP,MC_META_CAP); a_u64(&b,MCH_METALEN,mlen);
    memcpy(b.p+MCH_QUALITY,&q,4);
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

// ============================================================================
// APPENDABLE WRITER — persistent mmap'd archive, modeled on volume-compressor.
// ============================================================================
#if MC_HAVE_MMAP

#define MC_RESERVE   (10ull*1024*1024*1024*1024)   // 10 TiB virtual reservation (NORESERVE)
#define MC_GROW_STEP (1ull*1024*1024*1024)          // grow the file 1 GiB at a time

struct mc_archive {
    int fd;
    u8 *base;                  // fixed mmap base (never moves)
    _Atomic uint64_t cursor;   // append EOF (bytes used)
    _Atomic uint64_t file_len; // current ftruncate'd file size
    int dim;
    float quality;
    pthread_mutex_t grow_mu;   // serializes ftruncate only; decode is lock-free
};

static int w_ensure(mc_archive *w, uint64_t need){
    uint64_t fl = atomic_load_explicit(&w->file_len, memory_order_acquire);
    if(need <= fl) return 0;
    pthread_mutex_lock(&w->grow_mu);
    fl = atomic_load_explicit(&w->file_len, memory_order_relaxed);
    if(need > fl){
        uint64_t nf = fl;
        while(nf < need) nf += MC_GROW_STEP;
        if(ftruncate(w->fd, (off_t)nf) != 0){ pthread_mutex_unlock(&w->grow_mu); return -1; }
        atomic_store_explicit(&w->file_len, nf, memory_order_release);
    }
    pthread_mutex_unlock(&w->grow_mu);
    return 0;
}
// reserve a disjoint [off, off+n) range at EOF, growing the file as needed.
static uint64_t w_alloc(mc_archive *w, uint64_t n){
    uint64_t off = atomic_fetch_add_explicit(&w->cursor, n, memory_order_relaxed);
    if(w_ensure(w, off+n)!=0) return ~0ull;
    return off;
}
static void w_write_u64(mc_archive *w, uint64_t at, uint64_t v){ memcpy(w->base+at,&v,8); }
static uint64_t w_read_u64(mc_archive *w, uint64_t at){ uint64_t v; memcpy(&v,w->base+at,8); return v; }

// growable sink wrapping a writer EOF append (used by encode_chunk_blob via a staging
// buffer; we encode to RAM first then memcpy the whole blob into one EOF range so the
// chunk payload is contiguous + committed atomically).
typedef struct { u8 *p; size_t len, cap; } stage_t;
static void stage_put(void *out, const void *s, size_t n){
    stage_t *st=(stage_t*)out;
    if(st->len+n>st->cap){ size_t nc=st->cap?st->cap*2:1<<16; while(nc<st->len+n)nc*=2; st->p=realloc(st->p,nc); st->cap=nc; }
    memcpy(st->p+st->len,s,n); st->len+=n;
}

// ensure the index path root->inner->shard exists for (lod,cz,cy,cx); return the file
// offset of the SHARD-table slot that will hold the chunk offset. Creates dense node
// tables in place as needed (allocated zeroed at EOF, parent slot published last).
static uint64_t w_ensure_shard_slot(mc_archive *w, int lod, int cz,int cy,int cx){
    uint64_t root = w_read_u64(w, MCH_ROOTOFF + (uint64_t)lod*8);
    if(!root){
        uint64_t no = w_alloc(w, MC_NODE_BYTES); if(no==~0ull) return ~0ull;
        memset(w->base+no, 0, MC_NODE_BYTES);
        // publish root in the header (single writer per (lod) path here; appends to the
        // same lod are serialized by the caller's per-writer use, but be safe: only set
        // if still empty).
        w_write_u64(w, MCH_ROOTOFF+(uint64_t)lod*8, no);
        root = no;
    }
    // walk nibble 2 (root) -> nibble 1 (inner) -> nibble 0 (shard slot).
    uint64_t node = root;
    for(int nib=MC_TREE_LEVELS-1; nib>=1; --nib){
        int idx=(mc_nib(cz,nib)*16+mc_nib(cy,nib))*16+mc_nib(cx,nib);
        uint64_t child = w_read_u64(w, node + (uint64_t)idx*8);
        if(!child){
            uint64_t no = w_alloc(w, MC_NODE_BYTES); if(no==~0ull) return ~0ull;
            memset(w->base+no, 0, MC_NODE_BYTES);
            w_write_u64(w, node + (uint64_t)idx*8, no);   // publish child slot
            child = no;
        }
        node = child;
    }
    int n0=(mc_nib(cz,0)*16+mc_nib(cy,0))*16+mc_nib(cx,0);
    return node + (uint64_t)n0*8;   // address of the shard slot for this chunk
}

// append a finished compressed blob at EOF + publish it in the shard slot (commit word).
static int w_install_blob(mc_archive *w,int lod,int cz,int cy,int cx,const u8 *blob,size_t len){
    uint64_t slot = w_ensure_shard_slot(w,lod,cz,cy,cx);
    if(slot==~0ull) return -1;
    uint64_t off = w_alloc(w, len);
    if(off==~0ull) return -1;
    memcpy(w->base+off, blob, len);
    // release fence so the payload bytes are visible before the commit word.
    atomic_thread_fence(memory_order_release);
    w_write_u64(w, slot, off);   // publish chunk offset = commit
    // keep the header's total length current so the file is valid if reopened now.
    uint64_t cur = atomic_load_explicit(&w->cursor, memory_order_relaxed);
    w_write_u64(w, MCH_TOTLEN, cur);
    return 0;
}

mc_archive *mc_archive_open_dims(const char *path, int nx, int ny, int nz, float quality);
mc_archive *mc_archive_open(const char *path, int dim, float quality){
    return mc_archive_open_dims(path,dim,dim,dim,quality);
}
mc_archive *mc_archive_open_dims(const char *path, int nx, int ny, int nz, float quality){
    if(nx<=0||ny<=0||nz<=0){ fprintf(stderr,"mc_archive_open: bad dims\n"); return NULL; }
    int dim=nx;  // legacy field below; per-axis dims live in the header
    mc_codec_init(); mc_set_quality(quality);
    int fd = open(path, O_RDWR|O_CREAT, 0644);
    if(fd<0){ perror("mc_archive_open: open"); return NULL; }
    struct stat sb; if(fstat(fd,&sb)!=0){ perror("fstat"); close(fd); return NULL; }
    int fresh = (sb.st_size==0);
    uint64_t init_len;
    if(fresh){
        init_len = MC_META_END;   // header + metadata region; data appends after.
        if(ftruncate(fd,(off_t)init_len)!=0){ perror("ftruncate"); close(fd); return NULL; }
    } else {
        init_len = (uint64_t)sb.st_size;
    }
    u8 *base = mmap(NULL, MC_RESERVE, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_NORESERVE, fd, 0);
    if(base==MAP_FAILED){ perror("mmap"); close(fd); return NULL; }

    mc_archive *w = calloc(1,sizeof *w);
    w->fd=fd; w->base=base; w->dim=dim; w->quality=quality;
    pthread_mutex_init(&w->grow_mu,NULL);
    atomic_store(&w->file_len, init_len);

    if(fresh){
        memset(base,0,MC_HDR);
        uint32_t magic=MC_MAGIC, ver=MC_VERSION;
        uint32_t ux=(uint32_t)nx, uy=(uint32_t)ny, uz=(uint32_t)nz;
        memcpy(base+MCH_MAGIC,&magic,4); memcpy(base+MCH_VER,&ver,4);
        memcpy(base+MCH_NX,&ux,4); memcpy(base+MCH_NY,&uy,4); memcpy(base+MCH_NZ,&uz,4);
        uint64_t z=0; for(int l=0;l<8;++l) memcpy(base+MCH_ROOTOFF+l*8,&z,8);
        uint64_t metaoff=MC_HDR, metacap=MC_META_CAP, totlen=MC_META_END;
        memcpy(base+MCH_METAOFF,&metaoff,8); memcpy(base+MCH_METACAP,&metacap,8);
        memcpy(base+MCH_METALEN,&z,8); memcpy(base+MCH_TOTLEN,&totlen,8);
        memcpy(base+MCH_QUALITY,&quality,4);
        atomic_store(&w->cursor, MC_META_END);
    } else {
        uint32_t magic; memcpy(&magic,base+MCH_MAGIC,4);
        uint32_t ver;   memcpy(&ver,base+MCH_VER,4);
        uint32_t ux,uy,uz; memcpy(&ux,base+MCH_NX,4); memcpy(&uy,base+MCH_NY,4); memcpy(&uz,base+MCH_NZ,4);
        if(magic!=MC_MAGIC || ver!=MC_VERSION || (int)ux!=nx || (int)uy!=ny || (int)uz!=nz){
            fprintf(stderr,"mc_archive_open: %s is not a matching mc archive (magic/ver/dims)\n",path);
            munmap(base,MC_RESERVE); close(fd); free(w); return NULL;
        }
        uint64_t totlen; memcpy(&totlen,base+MCH_TOTLEN,8);
        if(totlen < MC_META_END) totlen=MC_META_END;
        atomic_store(&w->cursor, totlen);
        priors_load(base);
    }
    return w;
}

int mc_archive_append_chunk_compressed(mc_archive *a, int lod, int cz,int cy,int cx,
                                       const uint8_t *blob, size_t len){
    if(!a||lod<0||lod>7||!blob||!len) return -1;
    return w_install_blob(a,lod,cz,cy,cx,blob,len);
}

int mc_archive_reserve_index(mc_archive *a, int lod, int cz,int cy,int cx){
    if(!a||lod<0||lod>7) return -1;
    return w_ensure_shard_slot(a,lod,cz,cy,cx)==~0ull ? -1 : 0;
}

int mc_archive_append_chunk_raw_q(mc_archive *a, int lod, int cz,int cy,int cx,
                                  const mc_u8 vox[256*256*256], float q){
    if(!a||lod<0||lod>7||!vox) return -1;
    mc_set_quality(q);
    stage_t st={0};
    size_t blen = encode_chunk_blob(vox, stage_put, &st);
    int rc = 0;
    if(blen) rc = w_install_blob(a,lod,cz,cy,cx,st.p,st.len);
    // all-air chunk (blen==0): no blob, slot stays absent (decodes to zero). rc stays 0.
    free(st.p);
    return rc;
}
int mc_archive_append_chunk_raw(mc_archive *a, int lod, int cz,int cy,int cx, const mc_u8 vox[256*256*256]){
    return mc_archive_append_chunk_raw_q(a,lod,cz,cy,cx,vox,a?a->quality:8.0f);
}

// rate control: sample-encode 256 diagonally-spread blocks at base q, scale q
// once by the empirical bytes ~ q^-GAMMA law, then encode the chunk for real.
#define MC_RC_GAMMA 0.75f
int mc_archive_append_chunk_target(mc_archive *a, int lod, int cz,int cy,int cx,
                                   const mc_u8 vox[256*256*256], float target_ratio,
                                   float *q_out){
    if(!a||lod<0||lod>7||!vox||target_ratio<=1.0f) return -1;
    float q0=a->quality;
    mc_set_quality(q0);
    static _Thread_local mc_buf samp; samp.len=0;
    static _Thread_local u8 blk[MC_BLK*MC_BLK*MC_BLK];
    size_t sample_bytes=0; int sampled=0;
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by){
        int bx=(bz+by)&15;                       // diagonal spread, 256 blocks
        if(!gather_blk256(vox,bz,by,bx,blk)) { sampled++; continue; }
        uint32_t len=0; samp.len=0;
        if(mc_enc_block(blk,&samp,&len)) sample_bytes+=len;
        sampled++;
    }
    float q=q0;
    if(sample_bytes){
        double est_total=(double)sample_bytes*(4096.0/sampled);
        double want_total=(double)(256.0*256.0*256.0)/target_ratio;
        q=(float)(q0*pow(est_total/want_total,1.0/MC_RC_GAMMA));
        if(q<0.5f)q=0.5f; if(q>24.0f)q=24.0f;
    }
    if(q_out)*q_out=q;
    return mc_archive_append_chunk_raw_q(a,lod,cz,cy,cx,vox,q);
}

int mc_archive_set_priors(struct mc_archive *a,
                          const uint16_t plo[8][32], const uint16_t phi[8][32]){
    if(!a||!plo||!phi) return -1;
    uint64_t off=w_alloc(a,MC_PRIORS_BYTES);
    if(off==~0ull) return -1;
    uint32_t magic=MC_PRIORS_MAGIC, ver=1;
    memcpy(a->base+off,&magic,4); memcpy(a->base+off+4,&ver,4);
    memcpy(a->base+off+8,plo,8*32*2);
    memcpy(a->base+off+8+8*32*2,phi,8*32*2);
    atomic_thread_fence(memory_order_release);
    w_write_u64(a,MCH_PRIOROFF,off);
    mc_codec_set_priors((const uint16_t*)plo,(const uint16_t*)phi);
    return 0;
}

// load priors from a header offset if present (open paths call this).
static void priors_load(const uint8_t *base){
    uint64_t off; memcpy(&off,base+MCH_PRIOROFF,8);
    if(!off) { mc_codec_set_priors(NULL,NULL); return; }
    uint32_t magic; memcpy(&magic,base+off,4);
    if(magic!=MC_PRIORS_MAGIC){ mc_codec_set_priors(NULL,NULL); return; }
    mc_codec_set_priors((const uint16_t*)(base+off+8),(const uint16_t*)(base+off+8+8*32*2));
}

mc_cover mc_archive_chunk_coverage(mc_archive *a, int lod, int cz,int cy,int cx){
    if(!a||lod<0||lod>7) return MC_ABSENT;
    uint64_t root = w_read_u64(a, MCH_ROOTOFF+(uint64_t)lod*8);
    return mc_resolve_chunk(a->base, root, cz,cy,cx) ? MC_PRESENT : MC_ABSENT;
}

uint64_t mc_archive_chunk_offset(mc_archive *a, int lod, int cz,int cy,int cx){
    if(!a||lod<0||lod>7) return 0;
    uint64_t root = w_read_u64(a, MCH_ROOTOFF+(uint64_t)lod*8);
    return mc_resolve_chunk(a->base, root, cz,cy,cx);
}

// Decode one 16^3 block from the live mmap. LOCK-FREE: the codec's per-call scratch is
// all _Thread_local and g_quality is constant for the archive's lifetime (set once at
// open), so concurrent decodes are safe without serialization. Blocks are fully
// self-contained (v2: per-block air mask in the payload), so a single block decode
// touches only the bitmap + its own payload. The mmap is read-only here; appends
// publish via a release fence so a resolved chunk_off always points at fully-written
// bytes.
void mc_archive_decode_block(mc_archive *a, uint64_t chunk_off, int bz,int by,int bx, mc_u8 *dst){
    if(!a||!chunk_off){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    mc_set_quality(mc_chunk_q(a->base,chunk_off));   // thread-local; per-chunk q
    uint64_t boff; uint32_t bl;
    if(!mc_block_range(a->base,chunk_off,bz,by,bx,&boff,&bl)){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    mc_dec_block(a->base+boff,bl,dst);
}

// ---- parallel whole-chunk helpers ------------------------------------------
typedef struct {
    mc_archive *a; uint64_t chunk_off; mc_u8 *out; float q;
    _Atomic uint32_t next;
} dchunk_ctx;
static void *dchunk_worker(void *p){
    dchunk_ctx *c=p;
    mc_set_quality(c->q);                       // thread-local
    mc_u8 blk[MC_BLK*MC_BLK*MC_BLK];
    for(;;){
        uint32_t bi=atomic_fetch_add_explicit(&c->next,1,memory_order_relaxed);
        if(bi>=MC_GRID3) break;
        int bz=bi>>8, by=(bi>>4)&15, bx=bi&15;
        mc_archive_decode_block(c->a,c->chunk_off,bz,by,bx,blk);
        for(int z=0;z<MC_BLK;++z)for(int y=0;y<MC_BLK;++y)
            memcpy(c->out+((size_t)(bz*16+z)*MC_CHUNK+(by*16+y))*MC_CHUNK+(size_t)bx*16,
                   blk+((size_t)z*16+y)*16,16);
    }
    return NULL;
}
static int auto_threads(int nthreads){
    if(nthreads>0) return nthreads>16?16:nthreads;
    long nc=sysconf(_SC_NPROCESSORS_ONLN);
    int nt=(int)(nc>0?nc:4); return nt>16?16:nt;
}
void mc_archive_decode_chunk(mc_archive *a, uint64_t chunk_off, mc_u8 *out, int nthreads){
    if(!a||!out) return;
    if(!chunk_off){ memset(out,0,(size_t)MC_CHUNK*MC_CHUNK*MC_CHUNK); return; }
    dchunk_ctx c={.a=a,.chunk_off=chunk_off,.out=out,.q=mc_chunk_q(a->base,chunk_off)};
    atomic_store(&c.next,0);
    int nt=auto_threads(nthreads);
    if(nt<=1){ dchunk_worker(&c); return; }
    pthread_t th[16];
    for(int t=0;t<nt;++t) pthread_create(&th[t],NULL,dchunk_worker,&c);
    for(int t=0;t<nt;++t) pthread_join(th[t],NULL);
}

// parallel encode: stripes of blocks into per-worker buffers, stitched in
// bitmap order so the blob is byte-identical to the serial path.
#define ENC_STRIPES 16
typedef struct {
    const mc_u8 *vox; float q;
    mc_buf bufs[ENC_STRIPES];
    uint16_t blen[MC_GRID3];
    uint8_t  bm[MC_BITMAP_BYTES];
    uint8_t  frac[MC_GRID3];
    pthread_mutex_t bm_mu;
    _Atomic uint32_t next;
} echunk_ctx;
static void *echunk_worker(void *p){
    echunk_ctx *c=p;
    mc_set_quality(c->q);
    static _Thread_local u8 blk[MC_BLK*MC_BLK*MC_BLK];
    for(;;){
        uint32_t s=atomic_fetch_add_explicit(&c->next,1,memory_order_relaxed);
        if(s>=ENC_STRIPES) break;
        uint32_t b0=s*(MC_GRID3/ENC_STRIPES), b1=b0+(MC_GRID3/ENC_STRIPES);
        for(uint32_t bi=b0;bi<b1;++bi){
            int bz=(int)(bi>>8), by=(int)((bi>>4)&15), bx=(int)(bi&15);
            if(!gather_blk256(c->vox,bz,by,bx,blk)) continue;
            { int cnt=0; for(int i=0;i<MC_BLK*MC_BLK*MC_BLK;++i) cnt+=blk[i]!=0;
              c->frac[bi]=(uint8_t)((cnt*15+2048)/4096);
              if(cnt&&!c->frac[bi]) c->frac[bi]=1; }
            uint32_t len=0;
            if(mc_enc_block(blk,&c->bufs[s],&len)){
                c->blen[bi]=(uint16_t)len;
                pthread_mutex_lock(&c->bm_mu);
                mc_bit_set(c->bm,bi);
                pthread_mutex_unlock(&c->bm_mu);
            }
        }
    }
    return NULL;
}
int mc_archive_append_chunk_par(mc_archive *a, int lod, int cz,int cy,int cx,
                                const mc_u8 vox[256*256*256], float q, int nthreads){
    if(!a||lod<0||lod>7||!vox) return -1;
    echunk_ctx *c=calloc(1,sizeof *c);
    c->vox=vox; c->q=q>0?q:a->quality;
    pthread_mutex_init(&c->bm_mu,NULL);
    atomic_store(&c->next,0);
    int nt=auto_threads(nthreads); if(nt>ENC_STRIPES)nt=ENC_STRIPES;
    if(nt<=1) echunk_worker(c);
    else {
        pthread_t th[16];
        for(int t=0;t<nt;++t) pthread_create(&th[t],NULL,echunk_worker,c);
        for(int t=0;t<nt;++t) pthread_join(th[t],NULL);
    }
    int npresent=0; for(int i=0;i<MC_BITMAP_BYTES;++i) npresent+=__builtin_popcount(c->bm[i]);
    int rc=0;
    if(npresent){
        // stitch: stripes hold payloads in ascending-bi order within the stripe,
        // so concatenating stripes in order matches the serial blob layout.
        stage_t st={0};
        static _Thread_local uint16_t lens16[MC_GRID3];
        int nl=0;
        for(int bi=0;bi<MC_GRID3;++bi) if(mc_bit_get(c->bm,bi)) lens16[nl++]=c->blen[bi];
        float qq=c->q; uint64_t h=0;
        static _Thread_local uint8_t fmap[MC_GRID3/2+64];
        uint16_t fml=(uint16_t)mc_enc_fracmap(c->frac,fmap,sizeof fmap);
        stage_put(&st,&qq,4); stage_put(&st,&h,8);          // hash patched below
        stage_put(&st,&fml,2); stage_put(&st,fmap,fml);
        stage_put(&st,c->bm,MC_BITMAP_BYTES);
        stage_put(&st,lens16,(size_t)nl*2);
        for(int s=0;s<ENC_STRIPES;++s) if(c->bufs[s].len) stage_put(&st,c->bufs[s].p,c->bufs[s].len);
        h=mc_chunk_compute_hash(st.p,(uint64_t)st.len);     // same bytes => same hash as serial
        memcpy(st.p+4,&h,8);
        rc=w_install_blob(a,lod,cz,cy,cx,st.p,st.len);
        free(st.p);
    }
    for(int s=0;s<ENC_STRIPES;++s) free(c->bufs[s].p);
    pthread_mutex_destroy(&c->bm_mu);
    free(c);
    return rc;
}

int mc_archive_block_present(mc_archive *a, int lod, int bz, int by, int bx){
    if(!a||lod<0||lod>7||bz<0||by<0||bx<0) return 0;
    uint64_t co=mc_archive_chunk_offset(a,lod,bz>>4,by>>4,bx>>4);
    if(!co) return 0;
    const u8 *bm=a->base+co+MC_BLOB_HDR+mc_chunk_fmaplen(a->base,co);
    return mc_bit_get(bm,((bz&15)*16+(by&15))*16+(bx&15));
}

float mc_archive_block_fraction(mc_archive *a, int lod, int bz, int by, int bx){
    if(!a||lod<0||lod>7||bz<0||by<0||bx<0) return 0.0f;
    uint64_t co=mc_archive_chunk_offset(a,lod,bz>>4,by>>4,bx>>4);
    if(!co) return 0.0f;
    static _Thread_local uint8_t fr[MC_GRID3];
    static _Thread_local uint64_t fr_key=~0ull;
    if(fr_key!=co){
        uint16_t fml=mc_chunk_fmaplen(a->base,co);
        if(!fml) return 0.0f;
        mc_dec_fracmap(a->base+co+MC_BLOB_HDR,fml,fr);
        fr_key=co;
    }
    return (float)fr[((bz&15)*16+(by&15))*16+(bx&15)]/15.0f;
}

static inline uint64_t mc_rng64(uint64_t *s){
    uint64_t x=*s; x^=x<<13; x^=x>>7; x^=x<<17; *s=x; return x;
}
int mc_archive_sample_boxes(mc_archive *a, int lod, uint64_t seed, int count,
                            long dz, long dy, long dx, float min_frac,
                            mc_box *out){
    if(!a||!out||count<=0||dz<=0||dy<=0||dx<=0) return 0;
    uint32_t unx,uny,unz;
    memcpy(&unx,a->base+MCH_NX,4); memcpy(&uny,a->base+MCH_NY,4); memcpy(&unz,a->base+MCH_NZ,4);
    long NXl=(long)unx>>lod, NYl=(long)uny>>lod, NZl=(long)unz>>lod;
    if(NZl<dz||NYl<dy||NXl<dx) return 0;
    uint64_t s=seed?seed:0x9E3779B97F4A7C15ull;
    int got=0; long attempts=0, max_attempts=(long)count*256;
    while(got<count && attempts++<max_attempts){
        long z0=(long)(mc_rng64(&s)%(uint64_t)(NZl-dz+1));
        long y0=(long)(mc_rng64(&s)%(uint64_t)(NYl-dy+1));
        long x0=(long)(mc_rng64(&s)%(uint64_t)(NXl-dx+1));
        // mean block fraction over the touched block grid
        long bz0=z0>>4, bz1=(z0+dz-1)>>4, by0=y0>>4, by1=(y0+dy-1)>>4, bx0=x0>>4, bx1=(x0+dx-1)>>4;
        double fsum=0; long nb=0;
        for(long bz=bz0;bz<=bz1;++bz)for(long by=by0;by<=by1;++by)for(long bx=bx0;bx<=bx1;++bx){
            fsum+=mc_archive_block_fraction(a,lod,(int)bz,(int)by,(int)bx); nb++;
        }
        if(nb && fsum/nb>=min_frac){ out[got].z0=z0; out[got].y0=y0; out[got].x0=x0; got++; }
    }
    return got;
}

typedef struct {
    mc_archive *a; int lod;
    const mc_box *boxes; int n;
    long dz,dy,dx;
    mc_u8 *out; size_t bstride;
    _Atomic uint32_t next;
} crops_ctx;
static void *crops_worker(void *p){
    crops_ctx *c=p;
    for(;;){
        uint32_t i=atomic_fetch_add_explicit(&c->next,1,memory_order_relaxed);
        if(i>=(uint32_t)c->n) break;
        mc_archive_read_region(c->a,c->lod,c->boxes[i].z0,c->boxes[i].y0,c->boxes[i].x0,
                               c->dz,c->dy,c->dx,
                               c->out+(size_t)i*c->bstride,
                               (size_t)c->dy*c->dx,(size_t)c->dx,1);
    }
    return NULL;
}
void mc_archive_read_regions(mc_archive *a, int lod, const mc_box *boxes, int n,
                             long dz, long dy, long dx,
                             mc_u8 *out, size_t batch_stride, int nthreads){
    if(!a||!boxes||n<=0||!out) return;
    crops_ctx c={.a=a,.lod=lod,.boxes=boxes,.n=n,.dz=dz,.dy=dy,.dx=dx,
                 .out=out,.bstride=batch_stride};
    atomic_store(&c.next,0);
    int nt=auto_threads(nthreads); if(nt>n)nt=n;
    if(nt<=1){ crops_worker(&c); return; }
    pthread_t th[16];
    for(int t=0;t<nt;++t) pthread_create(&th[t],NULL,crops_worker,&c);
    for(int t=0;t<nt;++t) pthread_join(th[t],NULL);
}

// ---- region read ------------------------------------------------------------
typedef struct {
    mc_archive *a; int lod;
    long z0,y0,x0,dz,dy,dx;
    mc_u8 *out; size_t sz,sy;
    int nbz,nby,nbx; long bz0,by0,bx0;     // touched block range
    _Atomic uint32_t next;
} region_ctx;
static void *region_worker(void *p){
    region_ctx *c=p;
    mc_u8 blk[MC_BLK*MC_BLK*MC_BLK];
    uint32_t nb=(uint32_t)(c->nbz*c->nby*c->nbx);
    for(;;){
        uint32_t w=atomic_fetch_add_explicit(&c->next,1,memory_order_relaxed);
        if(w>=nb) break;
        long bz=c->bz0+w/(c->nby*c->nbx);
        long by=c->by0+(w/c->nbx)%c->nby;
        long bx=c->bx0+w%c->nbx;
        uint64_t co=mc_archive_chunk_offset(c->a,c->lod,(int)(bz>>4),(int)(by>>4),(int)(bx>>4));
        long gz=bz*16, gy=by*16, gx=bx*16;            // block origin in voxels
        // intersection of this block with the region
        long iz0=gz>c->z0?gz:c->z0, iz1=(gz+16<c->z0+c->dz)?gz+16:c->z0+c->dz;
        long iy0=gy>c->y0?gy:c->y0, iy1=(gy+16<c->y0+c->dy)?gy+16:c->y0+c->dy;
        long ix0=gx>c->x0?gx:c->x0, ix1=(gx+16<c->x0+c->dx)?gx+16:c->x0+c->dx;
        if(iz0>=iz1||iy0>=iy1||ix0>=ix1) continue;
        int present=0;
        if(co){
            const u8 *bm=c->a->base+co+MC_BLOB_HDR+mc_chunk_fmaplen(c->a->base,co);
            present=mc_bit_get(bm,(int)(((bz&15)*16+(by&15))*16+(bx&15)));
        }
        if(present) mc_archive_decode_block(c->a,co,(int)(bz&15),(int)(by&15),(int)(bx&15),blk);
        long nrow=ix1-ix0;
        for(long z=iz0;z<iz1;++z)for(long y=iy0;y<iy1;++y){
            mc_u8 *dst=c->out+(size_t)(z-c->z0)*c->sz+(size_t)(y-c->y0)*c->sy+(size_t)(ix0-c->x0);
            if(present) memcpy(dst,blk+((size_t)(z-gz)*16+(y-gy))*16+(ix0-gx),(size_t)nrow);
            else        memset(dst,0,(size_t)nrow);
        }
    }
    return NULL;
}
void mc_archive_read_region(mc_archive *a, int lod,
                            long z0, long y0, long x0,
                            long dz, long dy, long dx,
                            mc_u8 *out, size_t sz, size_t sy, int nthreads){
    if(!a||lod<0||lod>7||!out||dz<=0||dy<=0||dx<=0) return;
    region_ctx c={.a=a,.lod=lod,.z0=z0,.y0=y0,.x0=x0,.dz=dz,.dy=dy,.dx=dx,
                  .out=out,.sz=sz,.sy=sy};
    c.bz0=z0>>4; c.by0=y0>>4; c.bx0=x0>>4;
    c.nbz=(int)(((z0+dz+15)>>4)-c.bz0);
    c.nby=(int)(((y0+dy+15)>>4)-c.by0);
    c.nbx=(int)(((x0+dx+15)>>4)-c.bx0);
    atomic_store(&c.next,0);
    int nt=auto_threads(nthreads);
    uint32_t nb=(uint32_t)(c.nbz*c.nby*c.nbx);
    if((uint32_t)nt>nb) nt=(int)nb;
    if(nt<=1){ region_worker(&c); return; }
    pthread_t th[16];
    for(int t=0;t<nt;++t) pthread_create(&th[t],NULL,region_worker,&c);
    for(int t=0;t<nt;++t) pthread_join(th[t],NULL);
}

void mc_archive_close(mc_archive *a){
    if(!a) return;
    uint64_t cur = atomic_load(&a->cursor);
    w_write_u64(a, MCH_TOTLEN, cur);
    msync(a->base, cur, MS_SYNC);
    munmap(a->base, MC_RESERVE);
    if(ftruncate(a->fd,(off_t)cur)!=0) perror("mc_archive_close: ftruncate");
    close(a->fd);
    pthread_mutex_destroy(&a->grow_mu);
    free(a);
}

#else // !MC_HAVE_MMAP — the appendable archive requires mmap/ftruncate.
mc_archive *mc_archive_open(const char *p,int d,float q){ (void)p;(void)d;(void)q;
    fprintf(stderr,"mc_archive: requires a POSIX mmap platform\n"); return NULL; }
int mc_archive_append_chunk_raw(mc_archive*a,int l,int z,int y,int x,const mc_u8*v){ (void)a;(void)l;(void)z;(void)y;(void)x;(void)v; return -1; }
int mc_archive_append_chunk_compressed(mc_archive*a,int l,int z,int y,int x,const uint8_t*b,size_t n){ (void)a;(void)l;(void)z;(void)y;(void)x;(void)b;(void)n; return -1; }
mc_cover mc_archive_chunk_coverage(mc_archive*a,int l,int z,int y,int x){ (void)a;(void)l;(void)z;(void)y;(void)x; return MC_ABSENT; }
uint64_t mc_archive_chunk_offset(mc_archive*a,int l,int z,int y,int x){ (void)a;(void)l;(void)z;(void)y;(void)x; return 0; }
void mc_archive_decode_block(mc_archive*a,uint64_t o,int z,int y,int x,mc_u8*d){ (void)a;(void)o;(void)z;(void)y;(void)x; memset(d,0,MC_BLK*MC_BLK*MC_BLK); }
void mc_archive_close(mc_archive*a){ (void)a; }
#endif

// ============================================================================
// VERIFY — walk every chunk of every LOD, recompute xxh64, compare to stored.
// ============================================================================
long mc_verify_archive(const uint8_t *arc, size_t len, int verbose){
    (void)len;
    long bad=0, total=0;
    for(int lod=0;lod<8;++lod){
        uint64_t root; memcpy(&root,arc+MCH_ROOTOFF+(uint64_t)lod*8,8);
        if(!root) continue;
        for(int n2=0;n2<MC_GRID3;++n2){
            uint64_t inner; memcpy(&inner,arc+root+(size_t)n2*8,8);
            if(!inner) continue;
            for(int n1=0;n1<MC_GRID3;++n1){
                uint64_t shard; memcpy(&shard,arc+inner+(size_t)n1*8,8);
                if(!shard) continue;
                for(int n0=0;n0<MC_GRID3;++n0){
                    uint64_t co; memcpy(&co,arc+shard+(size_t)n0*8,8);
                    if(!co) continue;
                    total++;
                    uint64_t blen=mc_chunk_blob_len(arc,co);
                    uint64_t want=mc_chunk_stored_hash(arc,co);
                    uint64_t got=mc_chunk_compute_hash(arc+co,blen);
                    if(want!=got){
                        bad++;
                        if(verbose) fprintf(stderr,"mc_verify: lod %d node %d/%d/%d chunk@%llu CORRUPT\n",
                            lod,n2,n1,n0,(unsigned long long)co);
                    }
                }
            }
        }
    }
    if(verbose) fprintf(stderr,"mc_verify: %ld chunks checked, %ld corrupt\n",total,bad);
    return bad;
}

// ============================================================================
// READER (flat buffer + streaming byte-source)
// ============================================================================
const char *mc_metadata(const uint8_t *arc, size_t *out_len){
    uint64_t off,len; memcpy(&off,arc+MCH_METAOFF,8); memcpy(&len,arc+MCH_METALEN,8);
    if(!off) off=MC_HDR;
    if(out_len) *out_len=(size_t)len;
    return (const char*)(arc+off);
}

#define MC_RD_NODE_CACHE 64    // cached node tables per streaming reader (64*32KB = 2MB)
struct mc_reader {
    const uint8_t *arc;        // flat mode: archive buffer; streaming: NULL
    size_t len;
    uint64_t roots[8];
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
};

mc_reader *mc_open(const uint8_t *arc, size_t len){
    mc_codec_init();
    mc_reader *r=calloc(1,sizeof *r); r->arc=arc; r->len=len;
    for(int l=0;l<8;++l) memcpy(&r->roots[l], arc+MCH_ROOTOFF+l*8, 8);
    priors_load(arc);
    return r;
}

// streaming: fetch exactly len bytes at off into dst (via callback).
static int sread(mc_reader *r, uint64_t off, uint32_t len, u8 *dst){
    return r->read(r->read_ud, off, len, dst);
}

mc_reader *mc_open_streaming(mc_read_fn read, void *ud, uint64_t total_len){
    mc_codec_init();
    mc_reader *r=calloc(1,sizeof *r); r->read=read; r->read_ud=ud; r->len=(size_t)total_len;
    u8 hdr[MC_HDR];
    if(read(ud,0,MC_HDR,hdr)!=0){ free(r); return NULL; }
    uint32_t magic; memcpy(&magic,hdr+MCH_MAGIC,4);
    if(magic!=MC_MAGIC){ free(r); return NULL; }
    for(int l=0;l<8;++l) memcpy(&r->roots[l], hdr+MCH_ROOTOFF+l*8, 8);
    return r;
}

void mc_close(mc_reader *r){ if(!r)return;
    for(int i=0;i<MC_RD_NODE_CACHE;++i) free(r->ntab[i]);
    free(r->cbuf); free(r->pbuf); free(r); }
// Partial-fetch mode (streaming readers only): decode a block by fetching just
// the chunk's bitmap+length table (cached per chunk, <=8.7KB) plus that block's
// own payload, instead of the whole chunk blob. Wins cold random-access latency
// over high-latency byte sources (S3); leave OFF when scanning whole chunks.
void mc_reader_set_partial_fetch(mc_reader *r, int on){ if(r){ r->partial=on; r->hdr_off=~0ull; } }
void mc_reader_set_quality(mc_reader *r, float q){ (void)r; mc_set_quality(q); }

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
static uint64_t sresolve_chunk(mc_reader *r,int lod,int cz,int cy,int cx){
    uint64_t node = r->roots[lod];
    for(int nib=MC_TREE_LEVELS-1; nib>=0; --nib){
        if(!node) return 0;
        const u8 *tbl=sfetch_node(r,node);
        if(!tbl) return 0;
        int idx=(mc_nib(cz,nib)*16+mc_nib(cy,nib))*16+mc_nib(cx,nib);
        uint64_t child; memcpy(&child,tbl+(size_t)idx*8,8);
        node=child;
    }
    return node;
}

uint64_t mc_chunk_offset(mc_reader *r,int lod,int cz,int cy,int cx){
    if(lod<0||lod>7) return 0;
    if(r->arc) return mc_resolve_chunk(r->arc,r->roots[lod],cz,cy,cx);
    return sresolve_chunk(r,lod,cz,cy,cx);
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
    if(npresent && sread(r,bm_off+MC_BITMAP_BYTES,(uint32_t)(npresent*2),lens)!=0){ free(lens); return NULL; }
    uint64_t paybytes=0; for(int s=0;s<npresent;++s){ uint16_t l; memcpy(&l,lens+(size_t)s*2,2); paybytes+=l; }
    free(lens);
    uint64_t total = (bm_off+MC_BITMAP_BYTES+(uint64_t)npresent*2+paybytes) - chunk_off;
    r->cbuf = realloc(r->cbuf, total);
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
    { float q; memcpy(&q,r->hdr,4); mc_set_quality(q); }
    const u8 *bm=r->hdr+MC_BLOB_HDR;
    if(!mc_bit_get(bm,bi)){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return 0; }
    int slot=mc_rank(bm,bi);
    const u8 *lens=bm+MC_BITMAP_BYTES;
    uint64_t cum=0; for(int s2=0;s2<slot;++s2){ uint16_t l; memcpy(&l,lens+(size_t)s2*2,2); cum+=l; }
    uint16_t mylen; memcpy(&mylen,lens+(size_t)slot*2,2);
    uint64_t pay=chunk_off+MC_BLOB_HDR+r->hdr_fml+MC_BITMAP_BYTES+(uint64_t)r->hdr_np*2+cum;
    if(r->pbuf_cap<mylen){ r->pbuf=realloc(r->pbuf,mylen); r->pbuf_cap=mylen; }
    if(sread(r,pay,mylen,r->pbuf)!=0) return -1;
    mc_dec_block(r->pbuf,mylen,dst);
    return 0;
}

void mc_decode_block(mc_reader *r, uint64_t chunk_off, int bz,int by,int bx, mc_u8 *dst){
    if(!chunk_off){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
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
    mc_set_quality(mc_chunk_q(blob_base,chunk_off));
    uint64_t boff; uint32_t blen;
    if(!mc_block_range(blob_base,chunk_off,bz,by,bx,&boff,&blen)){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    mc_dec_block(blob_base+boff,blen,dst);
}
