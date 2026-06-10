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
    // S3-FIFO state: two slot-id rings + a ghost fingerprint ring w/ set.
    uint32_t *fs, fs_head, fs_tail, fs_cap;     // small queue (ring)
    uint32_t *fm, fm_head, fm_tail, fm_cap;     // main queue (ring)
    uint32_t *gfp, g_head, g_cap;               // ghost fingerprints (ring)
    int32_t  *gset; uint32_t gset_cap;          // fp -> ring idx, open addressing
    uint8_t  *slot_inmain;                      // which queue a slot lives in
} shard_t;

#define FREQ_MAX 3

struct mc_cache {
    int policy;               // mc_cache_policy
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
    }
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
static uint32_t s3_reclaim(shard_t *sh){
    for(;;){
        uint32_t small_len=ring_len(sh->fs_head,sh->fs_tail,sh->fs_cap);
        if(small_len >= sh->nslot/10+1){
            uint32_t s=sh->fs[sh->fs_head]; sh->fs_head=(sh->fs_head+1)%sh->fs_cap;
            if(sh->slot_key[s]==EMPTY_KEY) return s;        // invalidated: reuse
            if(sh->slot_ref[s]>0){             // promote to main
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
        if(sh->slot_ref[s]>0){                 // re-insert with decremented freq
            sh->slot_ref[s]--;
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
    slot=s3_reclaim(sh);
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
}

static inline shard_t *shard_of(mc_cache *c, uint64_t key){
    return &c->sh[(khash(key)>>56)&(NSHARD-1)];
}

const mc_u8 *mc_cache_get(mc_cache *c, int lod, int bz, int by, int bx){
    uint64_t key=bkey(lod,bz,by,bx);
    shard_t *sh=shard_of(c,key);
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
    if(!c||!ids||!n) return 0;
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
