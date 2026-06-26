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
static void a_reserve(abuf*b,size_t n){ if(b->len+n<=b->cap)return; size_t nc=b->cap?b->cap*2:1<<20; while(nc<b->len+n)nc*=2; void*tmp=realloc(b->p,nc); if(!tmp)return; b->p=tmp; b->cap=nc; }
static size_t a_put_at(abuf*b,const void*s,size_t n){ a_reserve(b,n); size_t at=b->len; memcpy(b->p+at,s,n); b->len+=n; return at; }
static size_t a_zero(abuf*b,size_t n){ a_reserve(b,n); size_t at=b->len; memset(b->p+at,0,n); b->len+=n; return at; }
// Emit at an `align`-aligned offset (zero-padding the gap). REQUIRED for index
// node/shard tables: a statically-built archive may later be reopened and
// APPENDED to, and the appendable path reads node child slots as _Atomic uint64_t
// (ldapr on aarch64 BUS-faults on a misaligned address). a_put_at alone leaves
// tables at the arbitrary post-blob `b->len`. `align` must be a power of two.
static size_t a_put_at_aligned(abuf*b,const void*s,size_t n,size_t align){
    size_t pad=(align-(b->len&(align-1)))&(align-1);
    if(pad) a_zero(b,pad);
    return a_put_at(b,s,n);
}
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

static uint64_t build_lod_dense(mc_codec_ctx *C, abuf*b, const vol_t *V, int ncz,int ncy,int ncx){
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

    u8 *chunkbuf=malloc((size_t)MC_CHUNK*MC_CHUNK*MC_CHUNK);
    if(!chunkbuf){ free(root); free(inner); free(shard); return 0; }

    for(int cz=0;cz<ncz;++cz)for(int cy=0;cy<ncy;++cy)for(int cx=0;cx<ncx;++cx){
        if(!gather_chunk256(V,cz,cy,cx,chunkbuf)) continue;
        size_t at=b->len;
        if(!encode_chunk_blob(C,chunkbuf,abuf_put,b)) continue;   // all air
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
            uint64_t soff=a_put_at_aligned(b,shard[n2][n1],MC_SHARD_BYTES,8);
            inner[n2]->slot[n1]=soff; any_inner=1;
        }
        if(any_inner){ uint64_t ioff=a_put_at_aligned(b,inner[n2],MC_NODE_BYTES,8); root->slot[n2]=ioff; any_root=1; }
    }
    uint64_t root_off=0;
    if(any_root) root_off=a_put_at_aligned(b,root,MC_NODE_BYTES,8);

    // free RAM tree
    for(int n2=0;n2<MC_GRID3;++n2){
        if(shard[n2]){ for(int n1=0;n1<MC_GRID3;++n1) free(shard[n2][n1]); free(shard[n2]); }
        free(inner[n2]);
    }
    free(shard); free(inner); free(root);
    free(chunkbuf);
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
    mc_codec_init();
    mc_codec_ctx *C=mc_codec_ctx_new();
    if(!C){ fprintf(stderr,"mc_build: OOM allocating codec ctx\n"); return NULL; }
    mc_codec_ctx_set_quality(C,opts->quality);
    u8 *lod0=calloc((size_t)PZ*PY,(size_t)PX);
    if(!lod0){ fprintf(stderr,"mc_build: OOM allocating %dx%dx%d\n",PZ,PY,PX); mc_codec_ctx_free(C); return NULL; }
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
        roots[lod]=build_lod_dense(C,&b,&vv,ncz,ncy,ncx);
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
    mc_codec_ctx_free(C);
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

// 10 TiB virtual reservation (NORESERVE). Overridable at compile time so the
// ThreadSanitizer build can shrink it -- TSan maps its own huge shadow regions
// and a 10 TiB MAP_NORESERVE reservation collides (mmap ENOMEM). Tests that run
// under TSan pass -DMC_RESERVE=... ; production keeps the default.
#ifndef MC_RESERVE
#define MC_RESERVE   (10ull*1024*1024*1024*1024)
#endif
#define MC_GROW_STEP (1ull*1024*1024*1024)          // grow the file 1 GiB at a time

// Coverage memo: an O(1) resident-region set so frozen render reads never walk
// the node tree (mc_resolve_chunk) per absent block. Keyed by a region key that
// packs (lod,cz,cy,cx); value bit distinguishes PRESENT from ZERO(air). Open
// addressing, power-of-two, atomic slots — lock-free inserts from decode threads,
// lock-free probes from render. Slot 0 = empty. Sized generously; never resizes
// (a render archive's covered-region count is bounded by the volume's region
// count, and we cap fill at the reserve anyway).
#define MC_COV_CAP (1u<<20)            // 1M slots -> up to ~700k regions at 0.7 load
// memo value = state(2 bits) | packed region key. Region coords are 256^3-chunk
// indices (<= ~4096 even for a 77824^3 volume) so 12 bits/axis is ample; the key
// fits in 39 bits, leaving the top free for state. state 0 = empty slot.
#define MC_COV_PRESENT 1ull
#define MC_COV_ZERO    2ull
#define MC_COV_ABSENT  3ull            // VISITED-and-absent: memoized so re-probes
                                       // of a not-yet-downloaded region are O(1)
                                       // (overwritten -> PRESENT when it lands).

// region key for the coverage memo: state in the top 2 bits, then lod(3)+12/axis.
static inline uint64_t mc_covkey(int lod,int cz,int cy,int cx){
    return ((uint64_t)(lod & 7) << 36) | ((uint64_t)(cz & 0xFFF) << 24) |
           ((uint64_t)(cy & 0xFFF) << 12) | (uint64_t)(cx & 0xFFF);
}
#define MC_COV_STATE_SHIFT 62
#define MC_COV_KEYMASK ((1ull<<MC_COV_STATE_SHIFT)-1)
// probe the memo. PRESENT/ZERO/ABSENT if memoized, MC_ABSENT(== not found) only
// when the slot is empty or the probe run is exhausted (caller then tree-walks).
static int mc_cov_probe(mc_archive *a,int lod,int cz,int cy,int cx){
    if(!a->cov) return -1;
    uint64_t key = mc_covkey(lod,cz,cy,cx);
    uint32_t h=(uint32_t)((key*0x9E3779B97F4A7C15ull)>>44);
    for(int p=0;p<32;++p){
        uint32_t i=(h+(uint32_t)p)&(MC_COV_CAP-1);
        uint64_t cur=atomic_load_explicit(&a->cov[i],memory_order_acquire);
        if(cur==0) return -1;                  // empty -> not memoized
        if((cur & MC_COV_KEYMASK)==key){
            uint64_t st = cur >> MC_COV_STATE_SHIFT;
            return st==MC_COV_PRESENT?MC_PRESENT : st==MC_COV_ZERO?MC_ZERO : MC_ABSENT;
        }
    }
    return -1;                                 // probe exhausted -> tree-walk
}
// insert/update the memo with an explicit state (PRESENT/ZERO/ABSENT). A later
// PRESENT publish overwrites a prior ABSENT/ZERO for the same region in place.
static void mc_cov_put_state(mc_archive *a,int lod,int cz,int cy,int cx,uint64_t state){
    if(!a->cov) return;
    uint64_t key = mc_covkey(lod,cz,cy,cx);
    uint64_t val = key | (state<<MC_COV_STATE_SHIFT);
    uint32_t h=(uint32_t)((key*0x9E3779B97F4A7C15ull)>>44);
    for(int p=0;p<32;++p){
        uint32_t i=(h+(uint32_t)p)&(MC_COV_CAP-1);
        uint64_t cur=atomic_load_explicit(&a->cov[i],memory_order_relaxed);
        if((cur & MC_COV_KEYMASK)==key && cur!=0){   // same region: update state
            if(cur==val) return;
            atomic_store_explicit(&a->cov[i],val,memory_order_release); return;
        }
        if(cur==0){
            uint64_t exp=0;
            if(atomic_compare_exchange_strong_explicit(&a->cov[i],&exp,val,
                   memory_order_release,memory_order_relaxed)) return;
            if((exp & MC_COV_KEYMASK)==key){  // lost race to same region
                if(exp!=val) atomic_store_explicit(&a->cov[i],val,memory_order_release);
                return; }
        }
    }
}
static inline void mc_cov_put(mc_archive *a,int lod,int cz,int cy,int cx,int is_zero){
    mc_cov_put_state(a,lod,cz,cy,cx, is_zero?MC_COV_ZERO:MC_COV_PRESENT);
}

static int w_ensure(mc_archive *w, uint64_t need){
    uint64_t fl = atomic_load_explicit(&w->file_len, memory_order_acquire);
    if(need <= fl) return 0;
    pthread_mutex_lock(&w->grow_mu);
    fl = atomic_load_explicit(&w->file_len, memory_order_relaxed);
    if(need > fl){
        uint64_t nf = fl;
        while(nf < need) nf += MC_GROW_STEP;
        if(nf > w->reserve){   // past the mmap reservation: fail cleanly, not SIGBUS
            fprintf(stderr,"mc_archive: grow beyond reservation (%llu > %llu)\n",
                    (unsigned long long)nf,(unsigned long long)w->reserve);
            pthread_mutex_unlock(&w->grow_mu); return -1;
        }
        // Allocate REAL disk blocks for the grown region [fl,nf), not a sparse ftruncate. Writing into
        // a sparse mmap region defers block allocation to PAGE-FAULT time, and a transient failure there
        // (fragmentation / momentary ENOSPC on the LVM volume / truncate-vs-write race) raises an
        // UNCATCHABLE SIGBUS -- exactly the observed crash (faulted once at a 1GiB grow boundary, resumed
        // straight past it). fallocate reserves the blocks up front so the mmap store always lands on
        // backed storage; real out-of-space now surfaces as a clean -1, not a bus fault. Falls back to
        // ftruncate on filesystems without fallocate (EOPNOTSUPP/ENOSYS).
        int gr = fallocate(w->fd, 0, (off_t)fl, (off_t)(nf - fl));
        if(gr != 0 && (errno == EOPNOTSUPP || errno == ENOSYS)) gr = ftruncate(w->fd, (off_t)nf);
        if(gr != 0){ pthread_mutex_unlock(&w->grow_mu); return -1; }
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
// Like w_alloc but returns an `align`-aligned offset. REQUIRED for index nodes:
// their u64 child slots are accessed as _Atomic uint64_t (ldapr/stlr on aarch64),
// which BUS-fault on a misaligned address. w_alloc alone returns the bare cursor,
// which is at an arbitrary offset once variable-length blobs have been appended
// (only the LOD0 nodes that precede any blob happen to be aligned). A CAS loop
// reserves the aligned range atomically so it is correct lock-free too. `align`
// must be a power of two.
static uint64_t w_alloc_aligned(mc_archive *w, uint64_t n, uint64_t align){
    uint64_t cur = atomic_load_explicit(&w->cursor, memory_order_relaxed);
    for(;;){
        uint64_t off = (cur + (align-1)) & ~(align-1);
        if(atomic_compare_exchange_weak_explicit(&w->cursor, &cur, off+n,
               memory_order_relaxed, memory_order_relaxed)){
            if(w_ensure(w, off+n)!=0) return ~0ull;
            return off;
        }
        // cur reloaded by the CAS failure; retry with the fresh value.
    }
}
static void w_write_u64(mc_archive *w, uint64_t at, uint64_t v){ memcpy(w->base+at,&v,8); }
static uint64_t w_read_u64(mc_archive *w, uint64_t at){ uint64_t v; memcpy(&v,w->base+at,8); return v; }
// Atomically publish a node offset into a tree slot iff still empty (0). Index nodes
// are 32KB-aligned (w_alloc bumps by MC_NODE_BYTES) so every u64 slot is 8-aligned ->
// safe to treat as _Atomic. Returns the offset now in the slot: `want` if we won the
// CAS, else the offset another thread published first (caller abandons its node).
static uint64_t w_publish_child(mc_archive *w, uint64_t at, uint64_t want){
    _Atomic uint64_t *slot = (_Atomic uint64_t*)(void*)(w->base+at);
    uint64_t exp = 0;
    if(atomic_compare_exchange_strong_explicit(slot,&exp,want,
           memory_order_release,memory_order_acquire)) return want;
    return exp;   // lost the race: someone else's node is live; abandon ours
}

// growable sink wrapping a writer EOF append (used by encode_chunk_blob via a staging
// buffer; we encode to RAM first then memcpy the whole blob into one EOF range so the
// chunk payload is contiguous + committed atomically).
typedef struct { u8 *p; size_t len, cap; } stage_t;
static void stage_put(void *out, const void *s, size_t n){
    stage_t *st=(stage_t*)out;
    if(st->len+n>st->cap){ size_t nc=st->cap?st->cap*2:1<<16; while(nc<st->len+n)nc*=2;
        void *tmp=realloc(st->p,nc); if(!tmp) return; st->p=tmp; st->cap=nc; }
    memcpy(st->p+st->len,s,n); st->len+=n;
}

// ensure the index path root->inner->shard exists for (lod,cz,cy,cx); return the file
// offset of the SHARD-table slot that will hold the chunk offset. Creates dense node
// tables in place as needed (allocated zeroed at EOF, parent slot published last).
static uint64_t w_ensure_shard_slot(mc_archive *w, int lod, int cz,int cy,int cx){
    // Decoder/dl threads append concurrently; node creation must be race-free. We
    // allocate a zeroed node then CAS-publish it into the parent slot. If we lose the
    // CAS another thread's node is live and we abandon ours (a bounded 32KB bump-alloc
    // leak, only on genuine concurrent first-touch of the same parent). No lock.
    // acquire-load the per-LOD root slot (it is published atomically by
    // w_publish_child from concurrent appenders) — pairs with that release and
    // matches the inner-node walk below. A plain w_read_u64 here is a data race.
    uint64_t root = atomic_load_explicit(
        (_Atomic uint64_t*)(void*)(w->base + MCH_ROOTOFF + (uint64_t)lod*8),
        memory_order_acquire);
    if(!root){
        uint64_t no = w_alloc_aligned(w, MC_NODE_BYTES, 8); if(no==~0ull) return ~0ull;
        memset(w->base+no, 0, MC_NODE_BYTES);
        root = w_publish_child(w, MCH_ROOTOFF+(uint64_t)lod*8, no);
    }
    // walk nibble 2 (root) -> nibble 1 (inner) -> nibble 0 (shard slot).
    uint64_t node = root;
    for(int nib=MC_TREE_LEVELS-1; nib>=1; --nib){
        int idx=(mc_nib(cz,nib)*16+mc_nib(cy,nib))*16+mc_nib(cx,nib);
        // acquire-load to pair with w_publish_child's release: a child offset becomes
        // visible only after its node's zero-fill, so we never walk into stale bytes.
        uint64_t child = atomic_load_explicit(
            (_Atomic uint64_t*)(void*)(w->base + node + (uint64_t)idx*8),
            memory_order_acquire);
        if(!child){
            uint64_t no = w_alloc_aligned(w, MC_NODE_BYTES, 8); if(no==~0ull) return ~0ull;
            memset(w->base+no, 0, MC_NODE_BYTES);
            child = w_publish_child(w, node + (uint64_t)idx*8, no);
        }
        node = child;
    }
    int n0=(mc_nib(cz,0)*16+mc_nib(cy,0))*16+mc_nib(cx,0);
    return node + (uint64_t)n0*8;   // address of the shard slot for this chunk
}

// append a finished compressed blob at EOF + publish it in the shard slot (commit word).
// Mark a chunk's slot as VISITED-but-all-zero (air). Lets a re-run / prefetch
// tell "fetched, it was air" from "never fetched" — no blob is written.
static int w_mark_zero(mc_archive *w,int lod,int cz,int cy,int cx){
    pthread_mutex_lock(&w->write_mu);
    uint64_t slot = w_ensure_shard_slot(w,lod,cz,cy,cx);
    if(slot==~0ull){ pthread_mutex_unlock(&w->write_mu); return -1; }
    w_write_u64(w, slot, MC_SLOT_ZERO);
    mc_cov_put(w, lod, cz, cy, cx, 1 /*air*/);
    atomic_fetch_add_explicit(&w->gen, 1, memory_order_release);
    pthread_mutex_unlock(&w->write_mu);
    return 0;
}

static int w_install_blob(mc_archive *w,int lod,int cz,int cy,int cx,const u8 *blob,size_t len){
    /* serialize index/heap mutation: the lock-free node-tree create + publish had a race
     * (concurrent first-touch of a parent node -> a reader walks a half-built node).
     * The encode (the expensive part) already ran in append_chunk_ctx, so locking only the
     * cheap install costs ~nothing. */
    pthread_mutex_lock(&w->write_mu);
    uint64_t slot = w_ensure_shard_slot(w,lod,cz,cy,cx);
    if(slot==~0ull){ pthread_mutex_unlock(&w->write_mu); return -1; }
    uint64_t off = w_alloc(w, len);
    if(off==~0ull){ pthread_mutex_unlock(&w->write_mu); return -1; }
    memcpy(w->base+off, blob, len);
    // Publish the chunk offset as the commit word: an atomic RELEASE store so a
    // concurrent reader (mc_resolve_chunk) or appender (w_ensure_shard_slot)
    // that acquire-loads this slot sees the fully-written payload. (A plain
    // store here races those atomic readers.)
    atomic_store_explicit((_Atomic uint64_t*)(void*)(w->base+slot), off,
                          memory_order_release);
    mc_cov_put(w, lod, cz, cy, cx, 0 /*present*/);
    atomic_fetch_add_explicit(&w->gen, 1, memory_order_release);
    // Keep the header's total length current so the file is valid if reopened
    // now. Concurrent appenders all touch MCH_TOTLEN -> atomic relaxed store
    // (monotone-ish snapshot of the cursor; exact value not load-bearing).
    uint64_t cur = atomic_load_explicit(&w->cursor, memory_order_relaxed);
    atomic_store_explicit((_Atomic uint64_t*)(void*)(w->base+MCH_TOTLEN), cur,
                          memory_order_relaxed);
    pthread_mutex_unlock(&w->write_mu);
    return 0;
}

mc_archive *mc_archive_open_dims(const char *path, int nx, int ny, int nz, float quality);
mc_archive *mc_archive_open(const char *path, int dim, float quality){
    return mc_archive_open_dims(path,dim,dim,dim,quality);
}
// reservation sized from the volume: worst-case compressed bytes are bounded by
// ~raw size; 1.5x headroom + 1 GiB floor, capped at MC_RESERVE. A blanket 10 TiB
// map breaks sanitizer shadow memory and small-volume test runners.
static uint64_t reserve_for_dims(int nx,int ny,int nz){
    uint64_t need=0, dz=(uint64_t)nz, dy=(uint64_t)ny, dx=(uint64_t)nx;
    for(int l=0;l<8;++l){
        uint64_t pz=(dz+255)/256*256, py=(dy+255)/256*256, px=(dx+255)/256*256;
        need += pz*py*px;
        dz=(dz+1)/2; dy=(dy+1)/2; dx=(dx+1)/2;
    }
    uint64_t r = need + need/2 + (1ull<<30);
    return r > MC_RESERVE ? MC_RESERVE : r;
}

mc_archive *mc_archive_open_dims(const char *path, int nx, int ny, int nz, float quality){
    if(nx<=0||ny<=0||nz<=0){ fprintf(stderr,"mc_archive_open: bad dims\n"); return NULL; }
    int dim=nx;  // legacy field below; per-axis dims live in the header
    mc_codec_init();   // quality is per-chunk; each worker sets it on its own ctx
    int fd = open(path, O_RDWR|O_CREAT, 0644);
    if(fd<0){ perror("mc_archive_open: open"); return NULL; }
    struct stat sb; if(fstat(fd,&sb)!=0){ perror("fstat"); close(fd); return NULL; }
    int fresh = (sb.st_size==0);
    uint64_t reserve = reserve_for_dims(nx,ny,nz);
    uint64_t init_len;
    if(fresh){
        init_len = MC_META_END;   // header + metadata region; data appends after.
        if(ftruncate(fd,(off_t)init_len)!=0){ perror("ftruncate"); close(fd); return NULL; }
    } else {
        init_len = (uint64_t)sb.st_size;
    }
    u8 *base = mmap(NULL, reserve, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_NORESERVE, fd, 0);
    if(base==MAP_FAILED){ perror("mmap"); close(fd); return NULL; }

    mc_archive *w = calloc(1,sizeof *w);
    w->fd=fd; w->base=base; w->dim=dim; w->quality=quality; w->reserve=reserve;
    pthread_mutex_init(&w->grow_mu,NULL);
    pthread_mutex_init(&w->write_mu,NULL);
    atomic_store(&w->file_len, init_len);
    w->cov = calloc(MC_COV_CAP, sizeof *w->cov);   // coverage memo (0 = empty)

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
        { uint32_t bc=MC_BLOCKCODEC_CABAC; memcpy(base+MCH_BLOCKCODEC,&bc,4); }  // default
        w->block_codec=MC_BLOCKCODEC_CABAC;
        atomic_store(&w->cursor, MC_META_END);
    } else {
        uint32_t magic; memcpy(&magic,base+MCH_MAGIC,4);
        uint32_t ver;   memcpy(&ver,base+MCH_VER,4);
        uint32_t ux,uy,uz; memcpy(&ux,base+MCH_NX,4); memcpy(&uy,base+MCH_NY,4); memcpy(&uz,base+MCH_NZ,4);
        if(magic!=MC_MAGIC || ver!=MC_VERSION || (int)ux!=nx || (int)uy!=ny || (int)uz!=nz){
            fprintf(stderr,"mc_archive_open: %s is not a matching mc archive (magic/ver/dims)\n",path);
            munmap(base,reserve); close(fd);
            pthread_mutex_destroy(&w->grow_mu); pthread_mutex_destroy(&w->write_mu); free(w->cov); free(w);   // no leak on reject
            return NULL;
        }
        uint64_t totlen; memcpy(&totlen,base+MCH_TOTLEN,8);
        if(totlen < MC_META_END) totlen=MC_META_END;
        atomic_store(&w->cursor, totlen);
        // block codec from the header (old archives have 0 here = CABAC).
        memcpy(&w->block_codec,base+MCH_BLOCKCODEC,4);
        w->block_codec=MC_BLOCKCODEC_CABAC;   // c3g removed: all archives are CABAC
        priors_load(base);
    }
    return w;
}

int mc_archive_append_chunk_compressed(mc_archive *a, int lod, int cz,int cy,int cx,
                                       const uint8_t *blob, size_t len){
    if(!a||lod<0||lod>7||!blob||!len) return -1;
    return w_install_blob(a,lod,cz,cy,cx,blob,len);
}

// Select the block entropy codec for an archive being WRITTEN (call on a fresh
// archive before appending chunks). Stamps the header so the reader/decode path
// dispatches correctly. Only MC_BLOCKCODEC_CABAC is supported (c3g removed).
// Returns 0 on success, <0 if the codec is unknown.
int mc_archive_set_block_codec(mc_archive *a, uint32_t codec){
    if(!a) return -1;
    if(codec!=MC_BLOCKCODEC_CABAC) return -1;
    a->block_codec=codec;
    memcpy(a->base+MCH_BLOCKCODEC,&codec,4);
    return 0;
}
uint32_t mc_archive_block_codec(const mc_archive *a){ return a?a->block_codec:MC_BLOCKCODEC_CABAC; }

int mc_archive_reserve_index(mc_archive *a, int lod, int cz,int cy,int cx){
    if(!a||lod<0||lod>7) return -1;
    return w_ensure_shard_slot(a,lod,cz,cy,cx)==~0ull ? -1 : 0;
}

int mc_archive_append_chunk_ctx(mc_archive *a, mc_codec_ctx *C,
                                int lod, int cz,int cy,int cx,
                                const mc_u8 vox[256*256*256]){
    if(!a||!C||lod<0||lod>7||!vox) return -1;
    stage_t st={0};
    size_t blen = encode_chunk_blob_codec(C, a->block_codec, vox, stage_put, &st);
    int rc = 0;
    if(blen) rc = w_install_blob(a,lod,cz,cy,cx,st.p,st.len);
    else     rc = w_mark_zero(a,lod,cz,cy,cx);   // air, but record it as VISITED
    free(st.p);
    return rc;
}
int mc_archive_append_chunk_raw_q(mc_archive *a, int lod, int cz,int cy,int cx,
                                  const mc_u8 vox[256*256*256], float q){
    if(!a||lod<0||lod>7||!vox) return -1;
    mc_codec_ctx *C=mc_codec_ctx_new();
    if(!C) return -1;
    mc_codec_ctx_set_quality(C,q);
    int rc = mc_archive_append_chunk_ctx(a,C,lod,cz,cy,cx,vox);
    mc_codec_ctx_free(C);
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
    mc_codec_ctx *C=mc_codec_ctx_new();
    if(!C) return -1;
    mc_codec_ctx_set_quality(C,q0);
    mc_buf samp={0};
    u8 blk[MC_BLK*MC_BLK*MC_BLK];
    size_t sample_bytes=0; int sampled=0;
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by){
        int bx=(bz+by)&15;                       // diagonal spread, 256 blocks
        if(!gather_blk256(vox,bz,by,bx,blk)) { sampled++; continue; }
        uint32_t len=0; samp.len=0;
        if(mc_enc_block(C,blk,&samp,&len)) sample_bytes+=len;
        sampled++;
    }
    free(samp.p);
    mc_codec_ctx_free(C);
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
void priors_load(const uint8_t *base){
    uint64_t off; memcpy(&off,base+MCH_PRIOROFF,8);
    if(!off) { mc_codec_set_priors(NULL,NULL); return; }
    uint32_t magic; memcpy(&magic,base+off,4);
    if(magic!=MC_PRIORS_MAGIC){ mc_codec_set_priors(NULL,NULL); return; }
    mc_codec_set_priors((const uint16_t*)(base+off+8),(const uint16_t*)(base+off+8+8*32*2));
}

uint64_t mc_archive_data_len(mc_archive *a){
    return a ? atomic_load_explicit(&a->cursor, memory_order_relaxed) : 0;
}

int mc_archive_set_metadata(mc_archive *a, const void *data, size_t len){
    if(!a || (len && !data)) return -1;
    if(len > MC_META_CAP) return -1;
    if(len) memcpy(a->base + MC_HDR, data, len);
    // publish the length AFTER the bytes so a concurrent flat read never sees
    // a length covering unwritten content (same commit-word discipline as blobs).
    atomic_thread_fence(memory_order_release);
    uint64_t l = len; memcpy(a->base + MCH_METALEN, &l, 8);
    return 0;
}

const char *mc_archive_metadata(mc_archive *a, size_t *out_len){
    if(!a){ if(out_len) *out_len = 0; return NULL; }
    return mc_metadata(a->base, out_len);
}

mc_cover mc_archive_chunk_coverage(mc_archive *a, int lod, int cz,int cy,int cx){
    if(!a||lod<0||lod>7) return MC_ABSENT;
    // Fast path: the coverage memo. A hit is O(1) and never touches the node tree
    // (the per-block tree walk on the render worker was the 49ms cost). Regions
    // made resident this session are always in the memo. A memo-miss falls back to
    // the tree (covers disk-loaded archives committed in a prior session) and
    // backfills the memo so the next probe is O(1).
    int m = mc_cov_probe(a, lod, cz, cy, cx);
    if(m >= 0) return (mc_cover)m;           // memoized (incl ABSENT) -> O(1)
    uint64_t root = w_read_u64(a, MCH_ROOTOFF+(uint64_t)lod*8);
    uint64_t off = mc_resolve_chunk(a->base, root, cz,cy,cx,
                                    atomic_load_explicit(&a->file_len, memory_order_relaxed));
    if(off==0){ mc_cov_put_state(a,lod,cz,cy,cx,MC_COV_ABSENT); return MC_ABSENT; }
    int zero = (off==MC_SLOT_ZERO);
    mc_cov_put(a, lod, cz, cy, cx, zero);   // backfill for next time
    return zero ? MC_ZERO : MC_PRESENT;
}

uint64_t mc_archive_chunk_offset(mc_archive *a, int lod, int cz,int cy,int cx){
    if(!a||lod<0||lod>7) return 0;
    uint64_t root = w_read_u64(a, MCH_ROOTOFF+(uint64_t)lod*8);
    return mc_resolve_chunk(a->base, root, cz,cy,cx,
                            atomic_load_explicit(&a->file_len, memory_order_relaxed));
}

// Decode one 16^3 block from the live mmap. LOCK-FREE: the codec scratch lives in
// a caller-owned per-thread mc_codec_ctx (quality set per chunk on that ctx), so
// concurrent decodes are safe without serialization. Blocks are fully
// self-contained (v2: per-block air mask in the payload), so a single block decode
// touches only the bitmap + its own payload. The mmap is read-only here; appends
// publish via a release fence so a resolved chunk_off always points at fully-written
// bytes.
// Internal: decode using a caller-owned ctx (worker pools own one ctx and reuse
// it across the chunk's 4096 blocks; sets the per-chunk q on that ctx).
static void mc_archive_decode_block_ctx(mc_codec_ctx *C, mc_archive *a, uint64_t chunk_off, int bz,int by,int bx, mc_u8 *dst){
    if(!a||chunk_off<=MC_SLOT_ZERO){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    mc_codec_ctx_set_quality(C,mc_chunk_q(a->base,chunk_off));   // per-chunk q
    uint64_t boff; uint32_t bl;
    if(!mc_block_range(a->base,chunk_off,bz,by,bx,&boff,&bl)){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    // HARDENED: offsets derive from the on-disk length table; on a corrupt
    // archive they could point past the mapped file (SIGBUS). Bound against
    // the live append cursor. (For untrusted archives run mc_verify first —
    // the per-chunk xxh64 covers bitmap+lens+payloads.)
    uint64_t end=atomic_load_explicit(&a->cursor,memory_order_acquire);
    if(boff+bl>end){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    mc_dec_block_codec(C,a->block_codec,a->base+boff,bl,dst);
}
// Per-thread codec ctx for the hot decode_block path, freed on thread exit via a
// pthread_key destructor. (A bare _Thread_local would leak ~400KB per render
// worker thread when it joins — short-lived parallel-render thread teams churn,
// so the leak is per-call, not bounded. The key keeps the per-thread caching
// perf while plugging the leak; LSan-clean.)
static pthread_key_t g_decblk_key;
static pthread_once_t g_decblk_once = PTHREAD_ONCE_INIT;
static void decblk_dtor(void *p){ if(p) mc_codec_ctx_free((mc_codec_ctx*)p); }
static void decblk_key_init(void){ pthread_key_create(&g_decblk_key, decblk_dtor); }

void mc_archive_decode_block(mc_archive *a, uint64_t chunk_off, int bz,int by,int bx, mc_u8 *dst){
    if(!a||chunk_off<=MC_SLOT_ZERO){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    // This is the HOT render-path read (mc_cache miss -> src_archive -> here, per
    // block). A fresh ctx per call would re-run step_tab_build's 4096-powf loop
    // every block (~4% of render CPU). Keep one ctx per thread; step_tab_build
    // caches on quality, so same-q blocks skip the rebuild.
    pthread_once(&g_decblk_once, decblk_key_init);
    mc_codec_ctx *C = pthread_getspecific(g_decblk_key);
    if(!C){ C=mc_codec_ctx_new(); if(!C){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
            pthread_setspecific(g_decblk_key, C); }
    mc_archive_decode_block_ctx(C,a,chunk_off,bz,by,bx,dst);
}

// Raw compressed block payload: returns a pointer into the live mmap to the
// block's encoded bytes and its length, WITHOUT decoding. For a c3g archive
// these are exactly the bytes the GPU compute decoder consumes — the viewer
// reads them off disk and uploads them verbatim. Returns 1 on a present block,
// 0 if absent/air or out of bounds (*ptr=NULL,*len=0).
int mc_archive_block_blob(mc_archive *a, uint64_t chunk_off, int bz,int by,int bx,
                          const uint8_t **ptr, uint32_t *len){
    if(ptr) *ptr=NULL; if(len) *len=0;
    if(!a||chunk_off<=MC_SLOT_ZERO) return 0;
    uint64_t boff; uint32_t bl;
    if(!mc_block_range(a->base,chunk_off,bz,by,bx,&boff,&bl)) return 0;
    uint64_t end=atomic_load_explicit(&a->cursor,memory_order_acquire);
    if(boff+bl>end) return 0;
    if(ptr) *ptr=a->base+boff;
    if(len) *len=bl;
    return 1;
}

// ---- parallel whole-chunk helpers ------------------------------------------
typedef struct {
    mc_archive *a; uint64_t chunk_off; mc_u8 *out; float q;
    _Atomic uint32_t next;
} dchunk_ctx;
static void *dchunk_worker(void *p){
    dchunk_ctx *c=p;
    mc_codec_ctx *C=mc_codec_ctx_new();     // one ctx per worker, reused per block
    if(!C) return NULL;
    mc_codec_ctx_set_quality(C,c->q);
    mc_u8 blk[MC_BLK*MC_BLK*MC_BLK];
    for(;;){
        uint32_t bi=atomic_fetch_add_explicit(&c->next,1,memory_order_relaxed);
        if(bi>=MC_GRID3) break;
        int bz=bi>>8, by=(bi>>4)&15, bx=bi&15;
        mc_archive_decode_block_ctx(C,c->a,c->chunk_off,bz,by,bx,blk);
        for(int z=0;z<MC_BLK;++z)for(int y=0;y<MC_BLK;++y)
            memcpy(c->out+((size_t)(bz*16+z)*MC_CHUNK+(by*16+y))*MC_CHUNK+(size_t)bx*16,
                   blk+((size_t)z*16+y)*16,16);
    }
    mc_codec_ctx_free(C);
    return NULL;
}
static int auto_threads(int nthreads){
    if(nthreads>0) return nthreads>16?16:nthreads;
    long nc=sysconf(_SC_NPROCESSORS_ONLN);
    int nt=(int)(nc>0?nc:4); return nt>16?16:nt;
}
void mc_archive_decode_chunk(mc_archive *a, uint64_t chunk_off, mc_u8 *out, int nthreads){
    if(!a||!out) return;
    if(chunk_off<=MC_SLOT_ZERO){ memset(out,0,(size_t)MC_CHUNK*MC_CHUNK*MC_CHUNK); return; }
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
    _Atomic uint32_t next;
} echunk_ctx;
static void *echunk_worker(void *p){
    echunk_ctx *c=p;
    mc_codec_ctx *C=mc_codec_ctx_new();
    if(!C) return NULL;
    mc_codec_ctx_set_quality(C,c->q);
    u8 blk[MC_BLK*MC_BLK*MC_BLK];
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
            if(mc_enc_block(C,blk,&c->bufs[s],&len)){
                c->blen[bi]=(uint16_t)len;
                // No lock: each stripe spans MC_GRID3/ENC_STRIPES=256 blocks = 32
                // whole bitmap bytes, so stripes set DISJOINT bytes (8 blocks/byte).
                mc_bit_set(c->bm,bi);
            }
        }
    }
    mc_codec_ctx_free(C);
    return NULL;
}
int mc_archive_append_chunk_par(mc_archive *a, int lod, int cz,int cy,int cx,
                                const mc_u8 vox[256*256*256], float q, int nthreads){
    if(!a||lod<0||lod>7||!vox) return -1;
    echunk_ctx *c=calloc(1,sizeof *c);
    c->vox=vox; c->q=q>0?q:a->quality;
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
        uint16_t lens16[MC_GRID3];
        int nl=0;
        for(int bi=0;bi<MC_GRID3;++bi) if(mc_bit_get(c->bm,bi)) lens16[nl++]=c->blen[bi];
        float qq=c->q; uint64_t h=0;
        uint8_t fmap[MC_GRID3/2+64];
        uint32_t fml32=mc_enc_fracmap(c->frac,fmap,sizeof fmap);
        if(fml32>sizeof fmap) fml32=sizeof fmap;            // always true; gives the optimizer the bound
        uint16_t fml=(uint16_t)fml32;
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
    free(c);
    return rc;
}

int mc_archive_block_present(mc_archive *a, int lod, int bz, int by, int bx){
    if(!a||lod<0||lod>7||bz<0||by<0||bx<0) return 0;
    uint64_t co=mc_archive_chunk_offset(a,lod,bz>>4,by>>4,bx>>4);
    if(co<=MC_SLOT_ZERO) return 0;
    const u8 *bm=a->base+co+MC_BLOB_HDR+mc_chunk_fmaplen(a->base,co);
    return mc_bit_get(bm,((bz&15)*16+(by&15))*16+(bx&15));
}

float mc_archive_block_fraction(mc_archive *a, int lod, int bz, int by, int bx){
    if(!a||lod<0||lod>7||bz<0||by<0||bx<0) return 0.0f;
    uint64_t co=mc_archive_chunk_offset(a,lod,bz>>4,by>>4,bx>>4);
    if(co<=MC_SLOT_ZERO) return 0.0f;
    uint8_t fr[MC_GRID3];
    uint16_t fml=mc_chunk_fmaplen(a->base,co);
    if(!fml) return 0.0f;
    mc_dec_fracmap(a->base+co+MC_BLOB_HDR,fml,fr);
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
    munmap(a->base, a->reserve);
    if(ftruncate(a->fd,(off_t)cur)!=0) perror("mc_archive_close: ftruncate");
    close(a->fd);
    pthread_mutex_destroy(&a->grow_mu);
    pthread_mutex_destroy(&a->write_mu);
    free(a->cov);
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
    // Recommended pre-flight for UNTRUSTED archives -> safe on arbitrary bytes:
    // bound every header-derived offset to [MC_HDR,len) before dereferencing.
    if(!arc || len < MC_HDR) return -1;
    const uint64_t NODE_BYTES = (uint64_t)MC_GRID3 * 8;
    #define MC_NODE_OK(off) ((off) >= MC_HDR && (off) <= len && len - (off) >= NODE_BYTES)
    long bad=0, total=0;
    for(int lod=0;lod<8;++lod){
        uint64_t root; memcpy(&root,arc+MCH_ROOTOFF+(uint64_t)lod*8,8);
        if(!root) continue;
        if(!MC_NODE_OK(root)){ bad++; continue; }
        for(int n2=0;n2<MC_GRID3;++n2){
            uint64_t inner; memcpy(&inner,arc+root+(size_t)n2*8,8);
            if(!inner) continue;
            if(!MC_NODE_OK(inner)){ bad++; continue; }
            for(int n1=0;n1<MC_GRID3;++n1){
                uint64_t shard; memcpy(&shard,arc+inner+(size_t)n1*8,8);
                if(!shard) continue;
                if(!MC_NODE_OK(shard)){ bad++; continue; }
                for(int n0=0;n0<MC_GRID3;++n0){
                    uint64_t co; memcpy(&co,arc+shard+(size_t)n0*8,8);
                    if(co<=MC_SLOT_ZERO) continue;
                    total++;
                    uint64_t blen=mc_blob_struct_ok(arc,co,len);   // bounded: 0 if OOB/corrupt
                    if(blen==0){ bad++; continue; }
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
    #undef MC_NODE_OK
    if(verbose) fprintf(stderr,"mc_verify: %ld chunks checked, %ld corrupt\n",total,bad);
    return bad;
}
