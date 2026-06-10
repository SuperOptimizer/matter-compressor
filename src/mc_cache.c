// ============================================================================
// mc_cache.c — sharded CLOCK/NRU decoded-block cache. See mc_cache.h.
// ============================================================================
#include "mc_cache.h"
#include "mc_archive_api.h"
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

// one shard: its own slice of the arena, its own map, lock, eviction state.
typedef struct {
    pthread_mutex_t mu;
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
    _Atomic uint64_t missq[MISSQ_CAP];
    _Atomic uint32_t miss_w;
    shard_t sh[NSHARD];
    mc_cache_src_fn src; void *src_ud;
    size_t arena_bytes;
    void *arena_base;
    // reader binding (serialized decode)
    pthread_mutex_t rd_mu;
    struct mc_reader *rd;
    struct mc_archive *ar;
};

static uint32_t pow2_at_least(uint32_t v){ uint32_t p=1; while(p<v)p<<=1; return p; }
static inline int slot_pinned(mc_cache *c, shard_t *sh, uint32_t slot);
static inline int cache_frozen(mc_cache *c){ return atomic_load_explicit(&c->frozen,memory_order_acquire); }
static void miss_record(mc_cache *c, uint64_t key){
    uint32_t i=atomic_fetch_add_explicit(&c->miss_w,1,memory_order_relaxed);
    atomic_store_explicit(&c->missq[i&(MISSQ_CAP-1)],key,memory_order_relaxed);
}

mc_cache *mc_cache_new(size_t bytes, mc_cache_src_fn src, void *src_ud){
    mc_cache *c=calloc(1,sizeof *c);
    c->src=src; c->src_ud=src_ud;
    pthread_mutex_init(&c->rd_mu,NULL);
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
        shard_t *sh=&c->sh[s];
        pthread_mutex_init(&sh->mu,NULL);
        sh->nslot=per; sh->hand=0; sh->used=0;
        sh->arena=(mc_u8*)c->arena_base + (size_t)s*per*BLK_BYTES;
        sh->map_cap=pow2_at_least(per*2);
        sh->map_key=calloc(sh->map_cap,8);
        sh->map_slot=calloc(sh->map_cap,4);
        sh->slot_key=calloc(per,8);
        sh->slot_ref=calloc(per,1);
        sh->fs_cap=per+1; sh->fm_cap=per+1;
        sh->fs=malloc(4u*sh->fs_cap); sh->fm=malloc(4u*sh->fm_cap);
        sh->g_cap=per; sh->gfp=calloc(sh->g_cap,4);
        sh->gset_cap=pow2_at_least(per*2); sh->gset=malloc(4u*sh->gset_cap);
        memset(sh->gset,0xFF,4u*sh->gset_cap);
        sh->slot_inmain=calloc(per,1);
        sh->slot_epoch=calloc(per,4);
    }
    atomic_store(&c->epoch,1);
    return c;
}
void mc_cache_set_policy(mc_cache *c, mc_cache_policy p){ if(c) c->policy=(int)p; }

void mc_cache_free(mc_cache *c){
    if(!c) return;
    for(int s=0;s<NSHARD;++s){
        shard_t *sh=&c->sh[s];
        pthread_mutex_destroy(&sh->mu);
        free(sh->map_key); free(sh->map_slot); free(sh->slot_key); free(sh->slot_ref);
        free(sh->fs); free(sh->fm); free(sh->gfp); free(sh->gset); free(sh->slot_inmain);
        free(sh->slot_epoch);
    }
#if MC_CACHE_MMAP
    munmap(c->arena_base,c->arena_bytes);
#else
    free(c->arena_base);
#endif
    pthread_mutex_destroy(&c->rd_mu);
    free(c);
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
        while(sh->gset[i]!=-1){ if(sh->gfp[sh->gset[i]]==old && (uint32_t)sh->gset[i]==sh->g_head){ sh->gset[i]=-1; break; } i=(i+1)&m; }
    }
    sh->gfp[sh->g_head]=fp;
    uint32_t m=sh->gset_cap-1, i=(fp*0x9E3779B1u)&m;
    while(sh->gset[i]!=-1) i=(i+1)&m;
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

const mc_u8 *mc_cache_get(mc_cache *c, int lod, int bz, int by, int bx){
    uint64_t key=bkey(lod,bz,by,bx);
    shard_t *sh=shard_of(c,key);
    if(cache_frozen(c)){                      // immutable: bare probe, no locks
        uint32_t mi=map_find(sh,key);
        if(mi!=UINT32_MAX) return sh->arena+(size_t)sh->map_slot[mi]*BLK_BYTES;
        miss_record(c,key);
        return NULL;                          // caller: fall back to best_lod
    }
    pthread_mutex_lock(&sh->mu);
    uint32_t mi=map_find(sh,key);
    if(mi!=UINT32_MAX){
        uint32_t slot=sh->map_slot[mi];
        cache_touch(c,sh,slot); sh->hits++;
        const mc_u8 *p=sh->arena+(size_t)slot*BLK_BYTES;
        pthread_mutex_unlock(&sh->mu);
        return p;
    }
    sh->misses++;
    pthread_mutex_unlock(&sh->mu);

    static _Thread_local mc_u8 tmp[BLK_BYTES];
    c->src(c->src_ud,lod,bz,by,bx,tmp);            // decode outside the lock

    pthread_mutex_lock(&sh->mu);
    mi=map_find(sh,key);                           // racing thread may have inserted
    uint32_t slot;
    if(mi!=UINT32_MAX) slot=sh->map_slot[mi];
    else {
        slot=cache_alloc_slot(c,sh,key);
        sh->slot_key[slot]=key;
        map_insert(sh,key,slot);
        memcpy(sh->arena+(size_t)slot*BLK_BYTES,tmp,BLK_BYTES);
    }
    cache_touch(c,sh,slot);
    const mc_u8 *p=sh->arena+(size_t)slot*BLK_BYTES;
    pthread_mutex_unlock(&sh->mu);
    return p;
}

void mc_cache_get_copy(mc_cache *c, int lod, int bz, int by, int bx, mc_u8 *dst){
    uint64_t key=bkey(lod,bz,by,bx);
    shard_t *sh=shard_of(c,key);
    if(cache_frozen(c)){
        uint32_t mi=map_find(sh,key);
        if(mi!=UINT32_MAX){ memcpy(dst,sh->arena+(size_t)sh->map_slot[mi]*BLK_BYTES,BLK_BYTES); return; }
        miss_record(c,key);
        c->src(c->src_ud,lod,bz,by,bx,dst);   // read-through, no insert
        return;
    }
    pthread_mutex_lock(&sh->mu);
    uint32_t mi=map_find(sh,key);
    if(mi!=UINT32_MAX){
        uint32_t slot=sh->map_slot[mi];
        cache_touch(c,sh,slot); sh->hits++;
        memcpy(dst,sh->arena+(size_t)slot*BLK_BYTES,BLK_BYTES);
        pthread_mutex_unlock(&sh->mu);
        return;
    }
    sh->misses++;
    pthread_mutex_unlock(&sh->mu);
    c->src(c->src_ud,lod,bz,by,bx,dst);
    pthread_mutex_lock(&sh->mu);
    if(map_find(sh,key)==UINT32_MAX){
        uint32_t slot=cache_alloc_slot(c,sh,key);
        sh->slot_key[slot]=key;
        map_insert(sh,key,slot);
        memcpy(sh->arena+(size_t)slot*BLK_BYTES,dst,BLK_BYTES);
        cache_touch(c,sh,slot);
    }
    pthread_mutex_unlock(&sh->mu);
}

int mc_cache_contains(mc_cache *c, int lod, int bz, int by, int bx){
    uint64_t key=bkey(lod,bz,by,bx);
    shard_t *sh=shard_of(c,key);
    if(cache_frozen(c)) return map_find(sh,key)!=UINT32_MAX;
    pthread_mutex_lock(&sh->mu);
    int r = map_find(sh,key)!=UINT32_MAX;
    pthread_mutex_unlock(&sh->mu);
    return r;
}

// lookup-or-decode-insert; returns 1 if a decode happened.
static int cache_fill_one(mc_cache *c, int lod, int bz, int by, int bx){
    uint64_t key=bkey(lod,bz,by,bx);
    shard_t *sh=shard_of(c,key);
    pthread_mutex_lock(&sh->mu);
    uint32_t mi=map_find(sh,key);
    if(mi!=UINT32_MAX){
        cache_touch(c,sh,sh->map_slot[mi]); sh->hits++;
        pthread_mutex_unlock(&sh->mu);
        return 0;
    }
    sh->misses++;
    pthread_mutex_unlock(&sh->mu);
    static _Thread_local mc_u8 tmp[BLK_BYTES];
    c->src(c->src_ud,lod,bz,by,bx,tmp);
    pthread_mutex_lock(&sh->mu);
    if(map_find(sh,key)==UINT32_MAX){
        uint32_t slot=cache_alloc_slot(c,sh,key);
        sh->slot_key[slot]=key;
        map_insert(sh,key,slot);
        memcpy(sh->arena+(size_t)slot*BLK_BYTES,tmp,BLK_BYTES);
        cache_touch(c,sh,slot);
    }
    pthread_mutex_unlock(&sh->mu);
    return 1;
}

typedef struct {
    mc_cache *c;
    const mc_block_id *ids;     // sorted copy, grouped by chunk
    const uint32_t *group_off;  // group g = ids[group_off[g] .. group_off[g+1])
    uint32_t ngroups;
    _Atomic uint32_t next;      // work-stealing group cursor
    _Atomic size_t decoded;
} upd_ctx;
static void *upd_worker(void *p){
    upd_ctx *u=p;
    for(;;){
        uint32_t g=atomic_fetch_add_explicit(&u->next,1,memory_order_relaxed);
        if(g>=u->ngroups) break;
        size_t dec=0;
        for(uint32_t i=u->group_off[g];i<u->group_off[g+1];++i){
            const mc_block_id *b=&u->ids[i];
            dec+=(size_t)cache_fill_one(u->c,b->lod,b->bz,b->by,b->bx);
        }
        atomic_fetch_add_explicit(&u->decoded,dec,memory_order_relaxed);
    }
    return NULL;
}
static uint64_t upd_sortkey(const mc_block_id *b){    // chunk-major; low 12 bits = in-chunk
    return ((uint64_t)(b->lod&7)<<60)
         | ((uint64_t)(b->bz>>4)<<44) | ((uint64_t)(b->by>>4)<<28) | ((uint64_t)(b->bx>>4)<<12)
         | ((uint64_t)(b->bz&15)<<8)  | ((uint64_t)(b->by&15)<<4)  | (uint64_t)(b->bx&15);
}
static int upd_cmp(const void *a,const void *b){
    uint64_t ka=upd_sortkey(a), kb=upd_sortkey(b);
    return ka<kb?-1:ka>kb?1:0;
}
size_t mc_cache_update(mc_cache *c, const mc_block_id *ids, size_t n, int nthreads){
    if(!c||!ids||!n||cache_frozen(c)) return 0;
    mc_block_id *s=malloc(n*sizeof *s);
    memcpy(s,ids,n*sizeof *s);
    qsort(s,n,sizeof *s,upd_cmp);
    uint32_t *off=malloc((n+1)*sizeof *off);
    uint32_t ng=0; off[0]=0;
    for(size_t i=1;i<n;++i){
        if((upd_sortkey(&s[i])>>12)!=(upd_sortkey(&s[i-1])>>12)) off[++ng]=(uint32_t)i;
    }
    off[++ng]=(uint32_t)n;
    upd_ctx u={.c=c,.ids=s,.group_off=off,.ngroups=ng};
    atomic_store(&u.next,0); atomic_store(&u.decoded,0);
    int nt=nthreads;
    if(nt<=0){
        long nc=sysconf(_SC_NPROCESSORS_ONLN);
        nt=(int)(nc>0?nc:4);
    }
    if(nt>16)nt=16; if((uint32_t)nt>ng)nt=(int)ng;
    if(nt<=1){ upd_worker(&u); }
    else {
        pthread_t th[16];
        for(int t=0;t<nt;++t) pthread_create(&th[t],NULL,upd_worker,&u);
        for(int t=0;t<nt;++t) pthread_join(th[t],NULL);
    }
    size_t dec=atomic_load(&u.decoded);
    free(s); free(off);
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
size_t mc_cache_misses_drain(mc_cache *c, mc_block_id *out, size_t cap){
    if(!c||!out) return 0;
    uint32_t w=atomic_load_explicit(&c->miss_w,memory_order_acquire);
    uint32_t n=w>MISSQ_CAP?MISSQ_CAP:w;
    size_t m=0;
    for(uint32_t i=0;i<n&&m<cap;++i){
        uint64_t k=atomic_load_explicit(&c->missq[i&(MISSQ_CAP-1)],memory_order_relaxed);
        if(!k) continue;
        out[m].lod=(int)((k>>60)&7);
        out[m].bz=(int)((k>>40)&0xFFFFF);
        out[m].by=(int)((k>>20)&0xFFFFF);
        out[m].bx=(int)(k&0xFFFFF);
        m++;
    }
    atomic_store_explicit(&c->miss_w,0,memory_order_release);
    return m;
}

int mc_cache_best_lod(mc_cache *c, int finest_lod, int bz, int by, int bx){
    int froz=cache_frozen(c);
    for(int l=finest_lod;l<8;++l){
        uint64_t key=bkey(l,bz,by,bx);
        shard_t *sh=shard_of(c,key);
        int hit;
        if(froz) hit = map_find(sh,key)!=UINT32_MAX;
        else { pthread_mutex_lock(&sh->mu); hit = map_find(sh,key)!=UINT32_MAX; pthread_mutex_unlock(&sh->mu); }
        if(hit) return l;
        bz>>=1; by>>=1; bx>>=1;
    }
    return -1;
}

// ---- async update tickets ---------------------------------------------------
struct mc_cache_ticket {
    mc_cache *c;
    mc_block_id *ids;            // owned sorted copy
    uint32_t *group_off;
    uint32_t ngroups;
    _Atomic uint32_t next;
    _Atomic uint32_t groups_done;
    _Atomic int cancel;
    pthread_t th[16]; int nth;
    int joined;
};
static void *aupd_worker(void *p){
    mc_cache_ticket *t=p;
    for(;;){
        if(atomic_load_explicit(&t->cancel,memory_order_relaxed)){
            // mark remaining groups done so done() converges after cancel
            uint32_t g=atomic_fetch_add_explicit(&t->next,1,memory_order_relaxed);
            if(g>=t->ngroups) break;
            atomic_fetch_add_explicit(&t->groups_done,1,memory_order_relaxed);
            continue;
        }
        uint32_t g=atomic_fetch_add_explicit(&t->next,1,memory_order_relaxed);
        if(g>=t->ngroups) break;
        for(uint32_t i=t->group_off[g];i<t->group_off[g+1];++i){
            const mc_block_id *b=&t->ids[i];
            cache_fill_one(t->c,b->lod,b->bz,b->by,b->bx);
        }
        atomic_fetch_add_explicit(&t->groups_done,1,memory_order_relaxed);
    }
    return NULL;
}
mc_cache_ticket *mc_cache_update_async(mc_cache *c, const mc_block_id *ids, size_t n, int nthreads){
    if(!c||!ids||!n||cache_frozen(c)) return NULL;
    mc_cache_ticket *t=calloc(1,sizeof *t);
    t->c=c;
    t->ids=malloc(n*sizeof *t->ids);
    memcpy(t->ids,ids,n*sizeof *t->ids);
    qsort(t->ids,n,sizeof *t->ids,upd_cmp);
    t->group_off=malloc((n+1)*sizeof *t->group_off);
    uint32_t ng=0; t->group_off[0]=0;
    for(size_t i=1;i<n;++i)
        if((upd_sortkey(&t->ids[i])>>12)!=(upd_sortkey(&t->ids[i-1])>>12)) t->group_off[++ng]=(uint32_t)i;
    t->group_off[++ng]=(uint32_t)n;
    t->ngroups=ng;
    atomic_store(&t->next,0); atomic_store(&t->groups_done,0); atomic_store(&t->cancel,0);
    int nt=nthreads;
    if(nt<=0){ long nc=sysconf(_SC_NPROCESSORS_ONLN); nt=(int)(nc>0?nc:4); }
    if(nt>16)nt=16; if((uint32_t)nt>ng)nt=(int)ng;
    t->nth=nt;
    for(int i=0;i<nt;++i) pthread_create(&t->th[i],NULL,aupd_worker,t);
    return t;
}
int mc_cache_ticket_done(mc_cache_ticket *t){
    if(!t) return 1;
    return atomic_load_explicit(&t->groups_done,memory_order_acquire)>=t->ngroups;
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
    free(t->ids); free(t->group_off); free(t);
}

// resolve: ensure resident (parallel via mc_cache_update), then fill the
// pointer table under shard locks; cache_touch stamps the current epoch so
// these slots are pinned against eviction until the next thaw().
size_t mc_cache_resolve(mc_cache *c, const mc_block_id *ids, size_t n,
                        const mc_u8 **ptrs, int nthreads){
    if(!c||!ids||!n||!ptrs||cache_frozen(c)) return 0;
    size_t dec=mc_cache_update(c,ids,n,nthreads);
    for(size_t i=0;i<n;++i){
        uint64_t key=bkey(ids[i].lod,ids[i].bz,ids[i].by,ids[i].bx);
        shard_t *sh=shard_of(c,key);
        pthread_mutex_lock(&sh->mu);
        uint32_t mi=map_find(sh,key);
        if(mi!=UINT32_MAX){
            uint32_t slot=sh->map_slot[mi];
            cache_touch(c,sh,slot);
            ptrs[i]=sh->arena+(size_t)slot*BLK_BYTES;
        } else ptrs[i]=NULL;   // evicted by same-batch pressure (set > capacity)
        pthread_mutex_unlock(&sh->mu);
    }
    return dec;
}

void mc_cache_prefetch_chunk(mc_cache *c, int lod, int cz, int cy, int cx){
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
        int gz=cz*16+bz, gy=cy*16+by, gx=cx*16+bx;
        uint64_t key=bkey(lod,gz,gy,gx);
        shard_t *sh=shard_of(c,key);
        pthread_mutex_lock(&sh->mu);
        int have = map_find(sh,key)!=UINT32_MAX;
        pthread_mutex_unlock(&sh->mu);
        if(!have) (void)mc_cache_get(c,lod,gz,gy,gx);
    }
}

// remove one key if present (shard lock held by caller paths below)
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
        shard_t *sh=shard_of(c,key);
        pthread_mutex_lock(&sh->mu);
        shard_remove_key(sh,key);
        pthread_mutex_unlock(&sh->mu);
    }
}

void mc_cache_clear(mc_cache *c){
    for(int s=0;s<NSHARD;++s){
        shard_t *sh=&c->sh[s];
        pthread_mutex_lock(&sh->mu);
        memset(sh->map_key,0,(size_t)sh->map_cap*8);
        memset(sh->slot_key,0,(size_t)sh->nslot*8);
        memset(sh->slot_ref,0,sh->nslot);
        sh->used=0; sh->hand=0;
        sh->fs_head=sh->fs_tail=sh->fm_head=sh->fm_tail=0;
        sh->g_head=0; memset(sh->gfp,0,4u*sh->g_cap);
        memset(sh->gset,0xFF,4u*sh->gset_cap);
        memset(sh->slot_inmain,0,sh->nslot);
        pthread_mutex_unlock(&sh->mu);
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
    uint64_t co=mc_archive_chunk_offset(a,lod,bz>>4,by>>4,bx>>4);
    mc_archive_decode_block(a,co,bz&15,by&15,bx&15,dst);
}
mc_cache *mc_cache_new_archive(size_t bytes, struct mc_archive *a){
    mc_cache *c=mc_cache_new(bytes,src_archive,a);
    if(c) c->ar=a;
    return c;
}
typedef struct { mc_cache *c; } rdwrap_t;
static void src_reader(void *ud, int lod, int bz,int by,int bx, mc_u8 *dst){
    mc_cache *c=ud;
    pthread_mutex_lock(&c->rd_mu);
    uint64_t co=mc_chunk_offset(c->rd,lod,bz>>4,by>>4,bx>>4);
    mc_decode_block(c->rd,co,bz&15,by&15,bx&15,dst);
    pthread_mutex_unlock(&c->rd_mu);
}
mc_cache *mc_cache_new_reader(size_t bytes, struct mc_reader *r){
    mc_cache *c=mc_cache_new(bytes,NULL,NULL);
    if(!c) return NULL;
    c->rd=r; c->src=src_reader; c->src_ud=c;
    return c;
}
