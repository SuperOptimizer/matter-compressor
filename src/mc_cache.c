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
// mc_cache.c — sharded CLOCK/NRU decoded-block cache. See mc_cache.h.
// ============================================================================
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>
#include <unistd.h>

#if defined(__unix__) || defined(__APPLE__)
  #include <sys/mman.h>
  #include <pthread.h>
  #define MC_CACHE_MMAP 1
#else
  #define MC_CACHE_MMAP 0
#endif

#define BLK_BYTES 4096u
#define NSHARD    64          // power of two
#define EMPTY_KEY 0ull

// key: 1 (always-set guard) | lod(3) | bz(20) | by(20) | bx(20)
static inline uint64_t bkey(int lod,int bz,int by,int bx){
    return (1ull<<63) | ((uint64_t)(lod&7)<<60)
         | ((uint64_t)(bz&0xFFFFF)<<40) | ((uint64_t)(by&0xFFFFF)<<20) | (uint64_t)(bx&0xFFFFF);
}
static inline uint64_t khash(uint64_t k){
    k^=k>>33; k*=0xFF51AFD7ED558CCDull; k^=k>>33; k*=0xC4CEB9FE1A85EC53ull; k^=k>>33;
    return k;
}

// one shard: its own slice of the arena, its own map, eviction state. No lock:
// all mutation is single-owner (THAW partitions fill by shard), reads are pure.
typedef struct {
    uint64_t *map_key;        // open addressing, linear probe, backward-shift delete
    uint32_t *map_slot;
    uint32_t  map_cap;        // power of two
    uint64_t *slot_key;       // per-slot reverse key (EMPTY_KEY = free)
    uint8_t  *slot_ref;       // CLOCK ref bit / S3-FIFO 2-bit freq
    uint32_t  nslot, used, hand;
    mc_u8    *arena;          // nslot * 4KB
    uint64_t  hits, misses, evictions;
    uint32_t *slot_epoch;            // pin: slot used by the current epoch
    // S3-FIFO state: two slot-id rings + a ghost fingerprint ring w/ set.
    uint32_t *fs, fs_head, fs_tail, fs_cap;     // small queue (ring)
    uint32_t *fm, fm_head, fm_tail, fm_cap;     // main queue (ring)
    uint32_t *gfp, g_head, g_cap;               // ghost fingerprints (ring)
    int32_t  *gset; uint32_t gset_cap;          // fp -> ring idx, open addressing
    uint8_t  *slot_inmain;                      // which queue a slot lives in
} shard_t;

#define FREQ_MAX 3

#define MISSQ_CAP 65536
struct mc_cache {
    int policy;               // mc_cache_policy
    _Atomic int frozen;
    _Atomic int phase_mode;          // set by the first freeze(); until then the
                                     // cache is the plain always-thread-safe
                                     // multi-reader/multi-writer cache and pins
                                     // are inert (no behavior change for
                                     // clients that never tick)
    _Atomic uint32_t epoch;          // bumped at thaw; pins compare against it
    // Miss set: open-addressing dedup table (slot 0 == empty). A thin slice
    // re-probes the same 16^3 block from many pixels/bands; recording each unique
    // absent block ONCE (not once per pixel) keeps the per-frame fill set ~= the
    // real working set instead of ~100x larger. Lock-free insert via CAS.
    _Atomic uint64_t missq[MISSQ_CAP];
    _Atomic uint32_t miss_n;          // approx live entries (for drain early-out)
    shard_t sh[NSHARD];
    mc_cache_src_fn src; void *src_ud;
    size_t arena_bytes;
    void *arena_base;
    // reader binding (single-owner decode; no lock)
    struct mc_reader *rd;
    struct mc_archive *ar;
};

static uint32_t pow2_at_least(uint32_t v){ uint32_t p=1; while(p<v)p<<=1; return p; }
static inline int slot_pinned(mc_cache *c, shard_t *sh, uint32_t slot);
static inline int cache_frozen(mc_cache *c){ return atomic_load_explicit(&c->frozen,memory_order_acquire); }
// Dedup insert into the miss set. Lock-free: hash, linear-probe a bounded run,
// CAS an empty slot to `key`. If the key is already present (or lands in the
// probe run), do nothing — recording the same absent block from another pixel is
// free. Bounded probe (table is generously sized vs the per-frame working set);
// on a full run we just drop the record (it re-records next frame).
static void miss_record(mc_cache *c, uint64_t key){
    if(!key) return;
    uint32_t h=(uint32_t)((key*0x9E3779B97F4A7C15ull)>>40);
    for(int p=0;p<8;++p){
        uint32_t i=(h+(uint32_t)p)&(MISSQ_CAP-1);
        uint64_t cur=atomic_load_explicit(&c->missq[i],memory_order_relaxed);
        if(cur==key) return;                          // already recorded
        if(cur==0){
            uint64_t exp=0;
            if(atomic_compare_exchange_strong_explicit(&c->missq[i],&exp,key,
                   memory_order_relaxed,memory_order_relaxed)){
                atomic_fetch_add_explicit(&c->miss_n,1,memory_order_relaxed);
                return;
            }
            if(exp==key) return;                      // racer inserted same key
        }
    }
}

static void shard_free_tables(shard_t *sh);
static int  shard_init_tables(shard_t *sh, mc_u8 *arena_slice, uint32_t per);

mc_cache *mc_cache_new(size_t bytes, mc_cache_src_fn src, void *src_ud){
    mc_cache *c=calloc(1,sizeof *c);
    if(!c) return NULL;
    c->src=src; c->src_ud=src_ud;
    size_t nslot_total = bytes/BLK_BYTES; if(nslot_total<NSHARD) nslot_total=NSHARD;
    uint32_t per = (uint32_t)(nslot_total/NSHARD); if(per<1)per=1;
    c->arena_bytes = (size_t)per*NSHARD*BLK_BYTES;
#if MC_CACHE_MMAP
    c->arena_base = mmap(NULL,c->arena_bytes,PROT_READ|PROT_WRITE,
                         MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
    if(c->arena_base==MAP_FAILED){ free(c); return NULL; }
#else
    c->arena_base = malloc(c->arena_bytes);
    if(!c->arena_base){ free(c); return NULL; }
#endif
    for(int s=0;s<NSHARD;++s){
        if(!shard_init_tables(&c->sh[s], (mc_u8*)c->arena_base + (size_t)s*per*BLK_BYTES, per)){
            for(int j=0;j<s;++j) shard_free_tables(&c->sh[j]);
#if MC_CACHE_MMAP
            munmap(c->arena_base,c->arena_bytes);
#else
            free(c->arena_base);
#endif
            free(c); return NULL;
        }
    }
    atomic_store(&c->epoch,1);
    return c;
}
void mc_cache_set_policy(mc_cache *c, mc_cache_policy p){ if(c) c->policy=(int)p; }

// free a shard's lookup/eviction tables (NOT its mutex, NOT the shared arena).
static void shard_free_tables(shard_t *sh){
    free(sh->map_key); free(sh->map_slot); free(sh->slot_key); free(sh->slot_ref);
    free(sh->fs); free(sh->fm); free(sh->gfp); free(sh->gset); free(sh->slot_inmain);
    free(sh->slot_epoch);
}

// (re)allocate a shard's tables for `per` slots pointing at `arena_slice`.
// Resets the shard to empty. Mutex assumed already init'd + held during resize.
// Returns 0 on allocation failure (shard left freeable: every table is NULL or owned).
static int shard_init_tables(shard_t *sh, mc_u8 *arena_slice, uint32_t per){
    sh->nslot=per; sh->hand=0; sh->used=0;
    sh->arena=arena_slice;
    sh->map_cap=pow2_at_least(per*2);
    sh->map_key=calloc(sh->map_cap,8);
    sh->map_slot=calloc(sh->map_cap,4);
    sh->slot_key=calloc(per,8);
    sh->slot_ref=calloc(per,1);
    sh->fs_cap=per+1; sh->fm_cap=per+1;
    sh->fs=malloc(4u*sh->fs_cap); sh->fm=malloc(4u*sh->fm_cap);
    sh->fs_head=sh->fs_tail=sh->fm_head=sh->fm_tail=0;
    sh->g_cap=per; sh->gfp=calloc(sh->g_cap,4);
    sh->gset_cap=pow2_at_least(per*2); sh->gset=malloc(4u*sh->gset_cap);
    sh->slot_inmain=calloc(per,1);
    sh->slot_epoch=calloc(per,4);
    if(!sh->map_key||!sh->map_slot||!sh->slot_key||!sh->slot_ref||!sh->fs||!sh->fm||
       !sh->gfp||!sh->gset||!sh->slot_inmain||!sh->slot_epoch) return 0;
    memset(sh->gset,0xFF,4u*sh->gset_cap);
    return 1;
}

void mc_cache_free(mc_cache *c){
    if(!c) return;
    for(int s=0;s<NSHARD;++s) shard_free_tables(&c->sh[s]);
#if MC_CACHE_MMAP
    munmap(c->arena_base,c->arena_bytes);
#else
    free(c->arena_base);
#endif
    free(c);
}

// ---- runtime budget control -------------------------------------------------
// Live-resize the decoded-block cache to `new_bytes`. The cache is just a cache,
// so resizing DISCARDS resident blocks (re-decode on demand) rather than
// migrating them. Single-owner: call only between ticks (no fill in flight).
// Returns the byte budget actually installed (rounded to whole slots), or 0.
size_t mc_cache_resize(mc_cache *c, size_t new_bytes){
    if(!c) return 0;
    size_t nslot_total = new_bytes/BLK_BYTES; if(nslot_total<NSHARD) nslot_total=NSHARD;
    uint32_t per = (uint32_t)(nslot_total/NSHARD); if(per<1)per=1;
    size_t new_arena = (size_t)per*NSHARD*BLK_BYTES;
#if MC_CACHE_MMAP
    void *na = mmap(NULL,new_arena,PROT_READ|PROT_WRITE,
                    MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
    if(na==MAP_FAILED) return 0;
#else
    void *na = malloc(new_arena);
    if(!na) return 0;
#endif
    // Build the new shard tables into temporaries first; only commit (free old +
    // swap) if every shard succeeds, so an OOM leaves the old cache fully intact.
    shard_t ns[NSHARD]; memset(ns,0,sizeof ns);
    int ok=1;
    for(int s=0;s<NSHARD && ok;++s)
        ok = shard_init_tables(&ns[s], (mc_u8*)na + (size_t)s*per*BLK_BYTES, per);
    if(!ok){
        for(int s=0;s<NSHARD;++s) shard_free_tables(&ns[s]);
#if MC_CACHE_MMAP
        munmap(na,new_arena);
#else
        free(na);
#endif
        return 0;
    }
    for(int s=0;s<NSHARD;++s){ shard_free_tables(&c->sh[s]); c->sh[s]=ns[s]; }
    void *old = c->arena_base; size_t old_bytes = c->arena_bytes;
    c->arena_base = na; c->arena_bytes = new_arena;
    atomic_fetch_add(&c->epoch,1);   // invalidate outstanding pins
#if MC_CACHE_MMAP
    munmap(old,old_bytes);
#else
    free(old);
#endif
    return new_arena;
}

size_t mc_cache_capacity_bytes(const mc_cache *c){ return c ? c->arena_bytes : 0; }

size_t mc_cache_used_bytes(mc_cache *c){
    if(!c) return 0;
    size_t used=0;
    for(int s=0;s<NSHARD;++s) used += (size_t)c->sh[s].used;   // racy read ok (stat)
    return used*BLK_BYTES;
}

double mc_cache_usage_fraction(mc_cache *c){
    if(!c||c->arena_bytes==0) return 0.0;
    return (double)mc_cache_used_bytes(c)/(double)c->arena_bytes;
}

// ---- shard map ops (shard lock held) ----
static inline uint32_t map_find(shard_t *sh, uint64_t key){   // -> map index or UINT32_MAX
    uint32_t m=sh->map_cap-1, i=(uint32_t)khash(key)&m;
    for(;;){
        if(sh->map_key[i]==key) return i;
        if(sh->map_key[i]==EMPTY_KEY) return UINT32_MAX;
        i=(i+1)&m;
    }
}
static void map_insert(shard_t *sh, uint64_t key, uint32_t slot){
    uint32_t m=sh->map_cap-1, i=(uint32_t)khash(key)&m;
    while(sh->map_key[i]!=EMPTY_KEY) i=(i+1)&m;
    sh->map_key[i]=key; sh->map_slot[i]=slot;
}
static void map_delete(shard_t *sh, uint64_t key){            // backward-shift deletion
    uint32_t m=sh->map_cap-1, i=map_find(sh,key);
    if(i==UINT32_MAX) return;
    uint32_t j=i;
    for(;;){
        j=(j+1)&m;
        uint64_t kj=sh->map_key[j];
        if(kj==EMPTY_KEY) break;
        uint32_t home=(uint32_t)khash(kj)&m;
        // can kj move into the hole at i? (it must not cross its home slot)
        uint32_t dist_ij=(i-home)&m, dist_jj=(j-home)&m;
        if(dist_ij<=dist_jj){ sh->map_key[i]=kj; sh->map_slot[i]=sh->map_slot[j]; i=j; }
    }
    sh->map_key[i]=EMPTY_KEY;
}
// CLOCK sweep: find a victim slot (clearing ref bits as we pass).
static uint32_t reclaim_slot(shard_t *sh){
    if(sh->used<sh->nslot){                       // free slot exists: linear scan from hand
        for(uint32_t k=0;k<sh->nslot;++k){
            uint32_t i=(sh->hand+k)%sh->nslot;
            if(sh->slot_key[i]==EMPTY_KEY){ sh->hand=(i+1)%sh->nslot; sh->used++; return i; }
        }
    }
    for(;;){
        uint32_t i=sh->hand; sh->hand=(sh->hand+1)%sh->nslot;
        if(sh->slot_key[i]==EMPTY_KEY) return i;            // invalidated: reuse
        if(sh->slot_ref[i]){ sh->slot_ref[i]=0; continue; }
        map_delete(sh,sh->slot_key[i]);
        sh->evictions++;
        return i;
    }
}

// ---- S3-FIFO (SOSP'23) ----
static inline uint32_t s3_fp(uint64_t key){ uint32_t f=(uint32_t)(khash(key)>>17); return f?f:1; }
static inline uint32_t ring_len(uint32_t h,uint32_t t,uint32_t cap){ return (t+cap-h)%cap; }
static void ghost_put(shard_t *sh, uint32_t fp){
    uint32_t old=sh->gfp[sh->g_head];
    if(old){  // remove the overwritten fingerprint from the set
        uint32_t m=sh->gset_cap-1, i=(old*0x9E3779B1u)&m;
        for(uint32_t p=0; p<sh->gset_cap && sh->gset[i]!=-1; ++p, i=(i+1)&m)
            if(sh->gfp[sh->gset[i]]==old && (uint32_t)sh->gset[i]==sh->g_head){ sh->gset[i]=-1; break; }
    }
    sh->gfp[sh->g_head]=fp;
    uint32_t m=sh->gset_cap-1, i=(fp*0x9E3779B1u)&m;
    { uint32_t p=0; while(sh->gset[i]!=-1 && p<sh->gset_cap){ i=(i+1)&m; ++p; } }
    sh->gset[i]=(int32_t)sh->g_head;
    sh->g_head=(sh->g_head+1)%sh->g_cap;
}
static int ghost_has(shard_t *sh, uint32_t fp){
    uint32_t m=sh->gset_cap-1, i=(fp*0x9E3779B1u)&m;
    for(int probes=0;probes<64;++probes){
        int32_t v=sh->gset[i];
        if(v==-1) return 0;
        if(sh->gfp[v]==fp) return 1;
        i=(i+1)&m;
    }
    return 0;
}
// reclaim one slot under S3-FIFO rules; the freed slot's key is unmapped.
static uint32_t s3_reclaim(mc_cache *c_pin, shard_t *sh){
    uint32_t spins=0, budget=2*sh->nslot+8;
    for(;;){
        int force = ++spins>budget;        // all pinned/hot: evict regardless
        uint32_t small_len=ring_len(sh->fs_head,sh->fs_tail,sh->fs_cap);
        if(small_len >= sh->nslot/10+1){
            uint32_t s=sh->fs[sh->fs_head]; sh->fs_head=(sh->fs_head+1)%sh->fs_cap;
            if(sh->slot_key[s]==EMPTY_KEY) return s;        // invalidated: reuse
            if(!force && (sh->slot_ref[s]>0 || slot_pinned(c_pin,sh,s))){   // promote to main
                sh->slot_ref[s]=0; sh->slot_inmain[s]=1;
                sh->fm[sh->fm_tail]=s; sh->fm_tail=(sh->fm_tail+1)%sh->fm_cap;
                continue;
            }
            ghost_put(sh,s3_fp(sh->slot_key[s]));
            map_delete(sh,sh->slot_key[s]); sh->evictions++;
            return s;
        }
        if(ring_len(sh->fm_head,sh->fm_tail,sh->fm_cap)==0){   // degenerate: force small
            uint32_t s=sh->fs[sh->fs_head]; sh->fs_head=(sh->fs_head+1)%sh->fs_cap;
            if(sh->slot_key[s]==EMPTY_KEY) return s;        // invalidated: reuse
            ghost_put(sh,s3_fp(sh->slot_key[s]));
            map_delete(sh,sh->slot_key[s]); sh->evictions++;
            return s;
        }
        uint32_t s=sh->fm[sh->fm_head]; sh->fm_head=(sh->fm_head+1)%sh->fm_cap;
        if(sh->slot_key[s]==EMPTY_KEY) return s;            // invalidated: reuse
        if(!force && (sh->slot_ref[s]>0 || slot_pinned(c_pin,sh,s))){   // pinned: keep
            if(sh->slot_ref[s]>0) sh->slot_ref[s]--;
            sh->fm[sh->fm_tail]=s; sh->fm_tail=(sh->fm_tail+1)%sh->fm_cap;
            continue;
        }
        map_delete(sh,sh->slot_key[s]); sh->evictions++;
        return s;
    }
}
// allocate a slot for `key` under the active policy and enqueue it.
static uint32_t cache_alloc_slot(mc_cache *c, shard_t *sh, uint64_t key){
    uint32_t slot;
    if(c->policy==MC_CACHE_CLOCK){
        slot=reclaim_slot(sh);
        return slot;
    }
    if(sh->used<sh->nslot){
        for(uint32_t k=0;k<sh->nslot;++k){
            uint32_t i=(sh->hand+k)%sh->nslot;
            if(sh->slot_key[i]==EMPTY_KEY){ sh->hand=(i+1)%sh->nslot; sh->used++; slot=i; goto have; }
        }
    }
    slot=s3_reclaim(c,sh);
have:;
    int to_main = ghost_has(sh,s3_fp(key));
    sh->slot_inmain[slot]=(uint8_t)to_main;
    sh->slot_ref[slot]=0;
    if(to_main){ sh->fm[sh->fm_tail]=slot; sh->fm_tail=(sh->fm_tail+1)%sh->fm_cap; }
    else       { sh->fs[sh->fs_tail]=slot; sh->fs_tail=(sh->fs_tail+1)%sh->fs_cap; }
    return slot;
}
static inline void cache_touch(mc_cache *c, shard_t *sh, uint32_t slot){
    if(c->policy==MC_CACHE_CLOCK) sh->slot_ref[slot]=1;
    else if(sh->slot_ref[slot]<FREQ_MAX) sh->slot_ref[slot]++;
    sh->slot_epoch[slot]=atomic_load_explicit(&c->epoch,memory_order_relaxed);
}
static inline int slot_pinned(mc_cache *c, shard_t *sh, uint32_t slot){
    if(!atomic_load_explicit(&c->phase_mode,memory_order_relaxed)) return 0;
    return sh->slot_epoch[slot]==atomic_load_explicit(&c->epoch,memory_order_relaxed);
}


static inline shard_t *shard_of(mc_cache *c, uint64_t key){
    return &c->sh[(khash(key)>>56)&(NSHARD-1)];
}

// LOCK-FREE. FROZEN (render): bare probe; miss -> record + return NULL (caller
// falls to coarser LOD). UNFROZEN (THAW / single-owner CLI): probe; miss ->
// decode + insert. No lock: every mutation is single-owner by contract (THAW
// partitions by shard; CLI is single-threaded). The cache arena is an immutable
// snapshot during a frozen frame, so the probe needs no synchronization.
const mc_u8 *mc_cache_get(mc_cache *c, int lod, int bz, int by, int bx){
    uint64_t key=bkey(lod,bz,by,bx);
    shard_t *sh=shard_of(c,key);
    uint32_t mi=map_find(sh,key);
    if(mi!=UINT32_MAX){
        if(!cache_frozen(c)){ cache_touch(c,sh,sh->map_slot[mi]); sh->hits++; }
        return sh->arena+(size_t)sh->map_slot[mi]*BLK_BYTES;
    }
    if(cache_frozen(c)){ miss_record(c,key); return NULL; }
    sh->misses++;
    uint32_t slot=cache_alloc_slot(c,sh,key);
    sh->slot_key[slot]=key;
    map_insert(sh,key,slot);
    c->src(c->src_ud,lod,bz,by,bx,sh->arena+(size_t)slot*BLK_BYTES);   // decode in place
    cache_touch(c,sh,slot);
    return sh->arena+(size_t)slot*BLK_BYTES;
}

void mc_cache_get_copy(mc_cache *c, int lod, int bz, int by, int bx, mc_u8 *dst){
    uint64_t key=bkey(lod,bz,by,bx);
    shard_t *sh=shard_of(c,key);
    uint32_t mi=map_find(sh,key);
    if(mi!=UINT32_MAX){
        if(!cache_frozen(c)){ cache_touch(c,sh,sh->map_slot[mi]); sh->hits++; }
        memcpy(dst,sh->arena+(size_t)sh->map_slot[mi]*BLK_BYTES,BLK_BYTES);
        return;
    }
    if(cache_frozen(c)){
        miss_record(c,key);
        c->src(c->src_ud,lod,bz,by,bx,dst);   // read-through, no insert
        return;
    }
    sh->misses++;
    uint32_t slot=cache_alloc_slot(c,sh,key);
    sh->slot_key[slot]=key;
    map_insert(sh,key,slot);
    c->src(c->src_ud,lod,bz,by,bx,sh->arena+(size_t)slot*BLK_BYTES);
    memcpy(dst,sh->arena+(size_t)slot*BLK_BYTES,BLK_BYTES);
    cache_touch(c,sh,slot);
}

int mc_cache_contains(mc_cache *c, int lod, int bz, int by, int bx){
    uint64_t key=bkey(lod,bz,by,bx);
    shard_t *sh=shard_of(c,key);
    return map_find(sh,key)!=UINT32_MAX;   // lock-free probe
}

// lookup-or-decode-insert; returns 1 if a decode happened. LOCK-FREE: the caller
// guarantees single-owner access to this block's shard (THAW partitions fill work
// by shard so no two workers touch one shard). All shard mutation -- map, arena,
// eviction rings -- happens here, only during THAW, only from the shard's owner.
int cache_fill_one(mc_cache *c, int lod, int bz, int by, int bx){
    uint64_t key=bkey(lod,bz,by,bx);
    shard_t *sh=shard_of(c,key);
    uint32_t mi=map_find(sh,key);
    if(mi!=UINT32_MAX){
        cache_touch(c,sh,sh->map_slot[mi]); sh->hits++;
        return 0;
    }
    sh->misses++;
    uint32_t slot=cache_alloc_slot(c,sh,key);
    sh->slot_key[slot]=key;
    map_insert(sh,key,slot);
    c->src(c->src_ud,lod,bz,by,bx,sh->arena+(size_t)slot*BLK_BYTES);   // decode in place
    cache_touch(c,sh,slot);
    return 1;
}

// THAW batch fill, partitioned by SHARD ownership. Each block is bucketed by its
// shard; worker t owns shards { t, t+nt, t+2nt, ... }. Because a shard is touched
// by exactly one worker, every shard mutation (map insert, slot alloc, eviction,
// arena write) is single-owner and needs NO lock. This is the partitioned phase
// update: parallelism by disjoint ownership, not by shared queue. The only sync
// is the join at the end (the phase barrier). Caller must be UNFROZEN (THAW).
typedef struct {
    mc_cache *c;
    const mc_block_id *ids;     // all blocks (unsorted)
    const uint32_t *bucket;     // bucket[s] = head index into `link` for shard s (-1 end)
    const uint32_t *link;       // link[i] = next block index in the same shard (-1 end)
    int nt, t;                  // this worker owns shards s where s % nt == t
    size_t decoded;             // written by this worker only
} upd_ctx;
static void *upd_worker(void *p){
    upd_ctx *u=p;
    size_t dec=0;
    for(int s=u->t; s<NSHARD; s+=u->nt){            // this worker's disjoint shards
        for(uint32_t i=u->bucket[s]; i!=UINT32_MAX; i=u->link[i]){
            const mc_block_id *b=&u->ids[i];
            dec+=(size_t)cache_fill_one(u->c,b->lod,b->bz,b->by,b->bx);
        }
    }
    u->decoded=dec;
    return NULL;
}
size_t mc_cache_update(mc_cache *c, const mc_block_id *ids, size_t n, int nthreads){
    if(!c||!ids||!n||cache_frozen(c)) return 0;
    // Bucket block indices by shard via per-shard singly-linked lists (no sort,
    // no shared structure). bucket[s] heads shard s's chain through link[].
    uint32_t *bucket=malloc(NSHARD*sizeof *bucket);
    uint32_t *link=malloc(n*sizeof *link);
    if(!bucket||!link){ free(bucket); free(link); return 0; }
    for(int s=0;s<NSHARD;++s) bucket[s]=UINT32_MAX;
    for(size_t i=0;i<n;++i){
        uint64_t key=bkey(ids[i].lod,ids[i].bz,ids[i].by,ids[i].bx);
        int s=(int)((khash(key)>>56)&(NSHARD-1));
        link[i]=bucket[s]; bucket[s]=(uint32_t)i;
    }
    int nt=nthreads;
    if(nt<=0){ long nc=sysconf(_SC_NPROCESSORS_ONLN); nt=(int)(nc>0?nc:4); }
    if(nt>16)nt=16; if(nt>NSHARD)nt=NSHARD; if(nt<1)nt=1;
    upd_ctx u[16];
    for(int t=0;t<nt;++t) u[t]=(upd_ctx){.c=c,.ids=ids,.bucket=bucket,.link=link,.nt=nt,.t=t,.decoded=0};
    if(nt<=1){ upd_worker(&u[0]); }
    else {
        pthread_t th[16];
        for(int t=0;t<nt;++t) pthread_create(&th[t],NULL,upd_worker,&u[t]);
        for(int t=0;t<nt;++t) pthread_join(th[t],NULL);   // phase barrier (join)
    }
    size_t dec=0; for(int t=0;t<nt;++t) dec+=u[t].decoded;
    free(bucket); free(link);
    return dec;
}

void mc_cache_freeze(mc_cache *c){
    if(!c) return;
    atomic_store_explicit(&c->phase_mode,1,memory_order_relaxed);
    atomic_store_explicit(&c->frozen,1,memory_order_release);
}
void mc_cache_thaw(mc_cache *c){
    if(!c) return;
    atomic_store_explicit(&c->frozen,0,memory_order_release);
    atomic_fetch_add_explicit(&c->epoch,1,memory_order_relaxed);
}
// Record a block into the dedup miss set from outside the frozen read path
// (e.g. mc_volume_try_block marking an ABSENT block for the downloader). Same
// lock-free dedup insert; safe to call concurrently with frozen reads.
void mc_cache_miss_mark(mc_cache *c, int lod, int bz, int by, int bx){
    if(c) miss_record(c, bkey(lod,bz,by,bx));
}

// Drain the dedup miss set: emit each unique block once and clear its slot. Must
// run while no frozen reads are inserting (thaw window) — consistent with the
// game loop (thaw drains, then freeze reopens reads).
size_t mc_cache_misses_drain(mc_cache *c, mc_block_id *out, size_t cap){
    if(!c||!out) return 0;
    if(atomic_load_explicit(&c->miss_n,memory_order_relaxed)==0) return 0;
    size_t m=0;
    for(uint32_t i=0;i<MISSQ_CAP;++i){
        uint64_t k=atomic_load_explicit(&c->missq[i],memory_order_relaxed);
        if(!k) continue;
        atomic_store_explicit(&c->missq[i],0,memory_order_relaxed);   // clear slot
        if(m<cap){
            out[m].lod=(int)((k>>60)&7);
            out[m].bz=(int)((k>>40)&0xFFFFF);
            out[m].by=(int)((k>>20)&0xFFFFF);
            out[m].bx=(int)(k&0xFFFFF);
            m++;
        }
    }
    atomic_store_explicit(&c->miss_n,0,memory_order_relaxed);
    return m;
}

int mc_cache_best_lod(mc_cache *c, int finest_lod, int bz, int by, int bx){
    for(int l=finest_lod;l<8;++l){
        uint64_t key=bkey(l,bz,by,bx);
        shard_t *sh=shard_of(c,key);
        if(map_find(sh,key)!=UINT32_MAX) return l;   // lock-free probe
        bz>>=1; by>>=1; bx>>=1;
    }
    return -1;
}

// ---- async update tickets ---------------------------------------------------
// Async ticket: same shard-partitioned, lock-free fill as mc_cache_update, but on
// detached worker threads. Worker t owns shards { t, t+nth, ... } via per-shard
// linked buckets -> single-owner, no lock. Cancel sets a flag workers poll.
struct mc_cache_ticket {
    mc_cache *c;
    mc_block_id *ids;            // owned copy
    uint32_t *bucket;           // bucket[s] = head index for shard s (UINT32_MAX end)
    uint32_t *link;             // link[i]   = next block index in same shard
    _Atomic uint32_t workers_done;
    _Atomic int cancel;
    pthread_t th[16]; int nth; int t_id[16];
    int joined;
};
static void *aupd_worker(void *p){
    mc_cache_ticket *t = ((void**)p)[0];
    int me = (int)(intptr_t)((void**)p)[1];
    free(p);
    for(int s=me; s<NSHARD; s+=t->nth){
        if(atomic_load_explicit(&t->cancel,memory_order_relaxed)) break;
        for(uint32_t i=t->bucket[s]; i!=UINT32_MAX; i=t->link[i]){
            const mc_block_id *b=&t->ids[i];
            cache_fill_one(t->c,b->lod,b->bz,b->by,b->bx);
        }
    }
    atomic_fetch_add_explicit(&t->workers_done,1,memory_order_release);
    return NULL;
}
mc_cache_ticket *mc_cache_update_async(mc_cache *c, const mc_block_id *ids, size_t n, int nthreads){
    if(!c||!ids||!n||cache_frozen(c)) return NULL;
    mc_cache_ticket *t=calloc(1,sizeof *t);
    if(!t) return NULL;
    t->c=c;
    t->ids=malloc(n*sizeof *t->ids);
    t->bucket=malloc(NSHARD*sizeof *t->bucket);
    t->link=malloc(n*sizeof *t->link);
    if(!t->ids||!t->bucket||!t->link){ free(t->ids); free(t->bucket); free(t->link); free(t); return NULL; }
    memcpy(t->ids,ids,n*sizeof *t->ids);
    for(int s=0;s<NSHARD;++s) t->bucket[s]=UINT32_MAX;
    for(size_t i=0;i<n;++i){
        uint64_t key=bkey(t->ids[i].lod,t->ids[i].bz,t->ids[i].by,t->ids[i].bx);
        int s=(int)((khash(key)>>56)&(NSHARD-1));
        t->link[i]=t->bucket[s]; t->bucket[s]=(uint32_t)i;
    }
    atomic_store(&t->workers_done,0); atomic_store(&t->cancel,0);
    int nt=nthreads;
    if(nt<=0){ long nc=sysconf(_SC_NPROCESSORS_ONLN); nt=(int)(nc>0?nc:4); }
    if(nt>16)nt=16; if(nt>NSHARD)nt=NSHARD; if(nt<1)nt=1;
    t->nth=nt;
    for(int i=0;i<nt;++i){
        void **arg=malloc(2*sizeof(void*)); arg[0]=t; arg[1]=(void*)(intptr_t)i;
        pthread_create(&t->th[i],NULL,aupd_worker,arg);
    }
    return t;
}
int mc_cache_ticket_done(mc_cache_ticket *t){
    if(!t) return 1;
    return atomic_load_explicit(&t->workers_done,memory_order_acquire)>=(uint32_t)t->nth;
}
void mc_cache_ticket_cancel(mc_cache_ticket *t){
    if(t) atomic_store_explicit(&t->cancel,1,memory_order_release);
}
static void ticket_join(mc_cache_ticket *t){
    if(t->joined) return;
    for(int i=0;i<t->nth;++i) pthread_join(t->th[i],NULL);
    t->joined=1;
}
void mc_cache_ticket_wait(mc_cache_ticket *t){ if(t) ticket_join(t); }
void mc_cache_ticket_free(mc_cache_ticket *t){
    if(!t) return;
    ticket_join(t);
    free(t->ids); free(t->bucket); free(t->link); free(t);
}

// resolve: ensure resident (parallel via mc_cache_update), then fill the pointer
// table; cache_touch stamps the current epoch so these slots are pinned against
// eviction until the next thaw(). Lock-free: single-owner (THAW / CLI) contract.
size_t mc_cache_resolve(mc_cache *c, const mc_block_id *ids, size_t n,
                        const mc_u8 **ptrs, int nthreads){
    if(!c||!ids||!n||!ptrs||cache_frozen(c)) return 0;
    size_t dec=mc_cache_update(c,ids,n,nthreads);
    for(size_t i=0;i<n;++i){
        uint64_t key=bkey(ids[i].lod,ids[i].bz,ids[i].by,ids[i].bx);
        shard_t *sh=shard_of(c,key);
        uint32_t mi=map_find(sh,key);
        if(mi!=UINT32_MAX){
            uint32_t slot=sh->map_slot[mi];
            cache_touch(c,sh,slot);
            ptrs[i]=sh->arena+(size_t)slot*BLK_BYTES;
        } else ptrs[i]=NULL;   // evicted by same-batch pressure (set > capacity)
    }
    return dec;
}

void mc_cache_prefetch_chunk(mc_cache *c, int lod, int cz, int cy, int cx){
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
        int gz=cz*16+bz, gy=cy*16+by, gx=cx*16+bx;
        uint64_t key=bkey(lod,gz,gy,gx);
        if(map_find(shard_of(c,key),key)==UINT32_MAX) (void)mc_cache_get(c,lod,gz,gy,gx);
    }
}

// remove one key if present. Single-owner (UNFROZEN/THAW) — no lock.
static void shard_remove_key(shard_t *sh, uint64_t key){
    uint32_t mi=map_find(sh,key);
    if(mi==UINT32_MAX) return;
    uint32_t slot=sh->map_slot[mi];
    map_delete(sh,key);
    sh->slot_key[slot]=EMPTY_KEY;       // slot stays in its FIFO ring; the
    sh->slot_ref[slot]=0;               // reclaim path skips empty keys
    sh->evictions++;
}
void mc_cache_invalidate_chunk(mc_cache *c, int lod, int cz, int cy, int cx){
    if(cache_frozen(c)) return;
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
        uint64_t key=bkey(lod,cz*16+bz,cy*16+by,cx*16+bx);
        shard_remove_key(shard_of(c,key),key);
    }
}

void mc_cache_clear(mc_cache *c){
    for(int s=0;s<NSHARD;++s){            // single-owner: call only while no fill runs
        shard_t *sh=&c->sh[s];
        memset(sh->map_key,0,(size_t)sh->map_cap*8);
        memset(sh->slot_key,0,(size_t)sh->nslot*8);
        memset(sh->slot_ref,0,sh->nslot);
        sh->used=0; sh->hand=0;
        sh->fs_head=sh->fs_tail=sh->fm_head=sh->fm_tail=0;
        sh->g_head=0; memset(sh->gfp,0,4u*sh->g_cap);
        memset(sh->gset,0xFF,4u*sh->gset_cap);
        memset(sh->slot_inmain,0,sh->nslot);
    }
}

void mc_cache_get_stats(mc_cache *c, mc_cache_stats *out){
    memset(out,0,sizeof *out);
    for(int s=0;s<NSHARD;++s){
        shard_t *sh=&c->sh[s];
        out->hits+=sh->hits; out->misses+=sh->misses; out->evictions+=sh->evictions;
        out->slots+=sh->nslot; out->used+=sh->used;
    }
}

// ---- bindings ----
static void src_archive(void *ud, int lod, int bz,int by,int bx, mc_u8 *dst){
    struct mc_archive *a=ud;
    // The tree walk (mc_resolve_chunk) is identical for all 4096 blocks of a chunk;
    // memoize the last (lod,cz,cy,cx)->chunk_off thread-locally. Render samples are
    // chunk-coherent so this collapses the per-block walk to one walk per chunk.
    int cz=bz>>4, cy=by>>4, cx=bx>>4;
    static _Thread_local const struct mc_archive *la=NULL;
    static _Thread_local int llod=-1,lcz=-1,lcy=-1,lcx=-1; static _Thread_local uint64_t lco=0,lgen=0;
    uint64_t gen=atomic_load_explicit(&a->gen,memory_order_acquire);
    uint64_t co;
    if(la==a && lgen==gen && llod==lod && lcz==cz && lcy==cy && lcx==cx) co=lco;
    else { co=mc_archive_chunk_offset(a,lod,cz,cy,cx);   // re-resolve if archive grew
           la=a; lgen=gen; llod=lod; lcz=cz; lcy=cy; lcx=cx; lco=co; }
    mc_archive_decode_block(a,co,bz&15,by&15,bx&15,dst);
}
mc_cache *mc_cache_new_archive(size_t bytes, struct mc_archive *a){
    mc_cache *c=mc_cache_new(bytes,src_archive,a);
    if(c) c->ar=a;
    return c;
}
typedef struct { mc_cache *c; } rdwrap_t;
static void src_reader(void *ud, int lod, int bz,int by,int bx, mc_u8 *dst){
    mc_cache *c=ud;                                 // single-owner (THAW/CLI): no lock
    uint64_t co=mc_chunk_offset(c->rd,lod,bz>>4,by>>4,bx>>4);
    mc_decode_block(c->rd,co,bz&15,by&15,bx&15,dst);
}
mc_cache *mc_cache_new_reader(size_t bytes, struct mc_reader *r){
    mc_cache *c=mc_cache_new(bytes,NULL,NULL);
    if(!c) return NULL;
    c->rd=r; c->src=src_reader; c->src_ud=c;
    return c;
}

