// ============================================================================
// mc_cache.c — sharded CLOCK/NRU decoded-block cache. See mc_cache.h.
// ============================================================================
#include "mc_cache.h"
#include "mc_archive_api.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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

// one shard: its own slice of the arena, its own map, lock, clock hand.
typedef struct {
    pthread_mutex_t mu;
    uint64_t *map_key;        // open addressing, linear probe, backward-shift delete
    uint32_t *map_slot;
    uint32_t  map_cap;        // power of two
    uint64_t *slot_key;       // per-slot reverse key (EMPTY_KEY = free)
    uint8_t  *slot_ref;       // NRU reference bit
    uint32_t  nslot, used, hand;
    mc_u8    *arena;          // nslot * 4KB
    uint64_t  hits, misses, evictions;
} shard_t;

struct mc_cache {
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
    }
    return c;
}

void mc_cache_free(mc_cache *c){
    if(!c) return;
    for(int s=0;s<NSHARD;++s){
        shard_t *sh=&c->sh[s];
        pthread_mutex_destroy(&sh->mu);
        free(sh->map_key); free(sh->map_slot); free(sh->slot_key); free(sh->slot_ref);
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
        if(sh->slot_ref[i]){ sh->slot_ref[i]=0; continue; }
        map_delete(sh,sh->slot_key[i]);
        sh->evictions++;
        return i;
    }
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
        sh->slot_ref[slot]=1; sh->hits++;
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
        slot=reclaim_slot(sh);
        sh->slot_key[slot]=key;
        map_insert(sh,key,slot);
        memcpy(sh->arena+(size_t)slot*BLK_BYTES,tmp,BLK_BYTES);
    }
    sh->slot_ref[slot]=1;
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
        sh->slot_ref[slot]=1; sh->hits++;
        memcpy(dst,sh->arena+(size_t)slot*BLK_BYTES,BLK_BYTES);
        pthread_mutex_unlock(&sh->mu);
        return;
    }
    sh->misses++;
    pthread_mutex_unlock(&sh->mu);
    c->src(c->src_ud,lod,bz,by,bx,dst);
    pthread_mutex_lock(&sh->mu);
    if(map_find(sh,key)==UINT32_MAX){
        uint32_t slot=reclaim_slot(sh);
        sh->slot_key[slot]=key;
        map_insert(sh,key,slot);
        memcpy(sh->arena+(size_t)slot*BLK_BYTES,dst,BLK_BYTES);
        sh->slot_ref[slot]=1;
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

void mc_cache_clear(mc_cache *c){
    for(int s=0;s<NSHARD;++s){
        shard_t *sh=&c->sh[s];
        pthread_mutex_lock(&sh->mu);
        memset(sh->map_key,0,(size_t)sh->map_cap*8);
        memset(sh->slot_key,0,(size_t)sh->nslot*8);
        memset(sh->slot_ref,0,sh->nslot);
        sh->used=0; sh->hand=0;
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
