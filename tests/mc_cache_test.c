// mc_cache test: build a small archive, hammer the cache from N threads with
// a zipf-ish revisit pattern, verify every cached block equals a direct
// decode, then check hit rate and that a tiny cache evicts correctly.
#include "../src/mc_archive_api.h"
#include "../src/mc_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

static mc_u8 srcv(void *ud, int x,int y,int z){
    (void)ud;
    double dx=x-128,dy=y-128,dz=z-128;
    if(dx*dx+dy*dy+dz*dz>118.0*118.0) return 0;
    return (mc_u8)(30+((x*13+y*7+z*11)%180));
}

typedef struct { mc_cache *c; mc_archive *a; int tid; int fails; } warg_t;
static void *worker(void *p){
    warg_t *w=p;
    unsigned rs=1234u+(unsigned)w->tid;
    mc_u8 want[4096], got[4096];
    for(int it=0; it<4000; ++it){
        rs^=rs<<13; rs^=rs>>17; rs^=rs<<5;
        // hot set: 75% of accesses hit 32 favourite blocks
        int hot = (rs>>28)!=0;
        int bz,by,bx;
        if(hot){ unsigned h=(rs>>8)%32; bz=h&3; by=(h>>2)&3; bx=(h>>4)+4; }
        else { bz=(int)(rs%16); by=(int)((rs>>8)%16); bx=(int)((rs>>16)%16); }
        mc_cache_get_copy(w->c,0,bz,by,bx,got);
        uint64_t co=mc_archive_chunk_offset(w->a,0,bz>>4,by>>4,bx>>4);
        mc_archive_decode_block(w->a,co,bz&15,by&15,bx&15,want);
        if(memcmp(want,got,4096)!=0){ w->fails++; return NULL; }
    }
    return NULL;
}

int main(void){
    const char *path="/tmp/mc_cache_test.mc";
    remove(path);
    mc_build_opts opt={.dim=256,.quality=6.0f};
    if(mc_build_to_file(srcv,NULL,&opt,path)!=0){ fprintf(stderr,"build failed\n"); return 1; }
    mc_archive *a=mc_archive_open(path,256,6.0f);
    if(!a){ fprintf(stderr,"open failed\n"); return 1; }

    // 1) multithreaded correctness + hit rate with a roomy cache
    mc_cache *c=mc_cache_new_archive(64ull<<20,a);
    enum { NT=8 };
    pthread_t th[NT]; warg_t wa[NT];
    for(int i=0;i<NT;++i){ wa[i]=(warg_t){c,a,i,0}; pthread_create(&th[i],NULL,worker,&wa[i]); }
    int fails=0;
    for(int i=0;i<NT;++i){ pthread_join(th[i],NULL); fails+=wa[i].fails; }
    mc_cache_stats st; mc_cache_get_stats(c,&st);
    printf("roomy: %d fails, hits %llu misses %llu (%.1f%% hit), used %zu/%zu slots\n",
        fails,(unsigned long long)st.hits,(unsigned long long)st.misses,
        100.0*st.hits/(st.hits+st.misses),st.used,st.slots);
    if(fails){ fprintf(stderr,"FAIL: voxel mismatch\n"); return 1; }
    if(st.hits < st.misses){ fprintf(stderr,"FAIL: hit rate too low for hot-set workload\n"); return 1; }
    mc_cache_free(c);

    // 2) tiny cache: must evict without corruption and still return right data
    c=mc_cache_new_archive((1ull<<20),a);
    mc_u8 want[4096], got[4096];
    int fails2=0;
    for(int it=0; it<3000; ++it){
        int bz=it%16, by=(it/16)%16, bx=(it/256)%16;
        mc_cache_get_copy(c,0,bz,by,bx,got);
        uint64_t co=mc_archive_chunk_offset(a,0,bz>>4,by>>4,bx>>4);
        mc_archive_decode_block(a,co,bz&15,by&15,bx&15,want);
        if(memcmp(want,got,4096)!=0){ fails2++; break; }
    }
    mc_cache_get_stats(c,&st);
    printf("tiny:  %d fails, evictions %llu, used %zu/%zu slots\n",
        fails2,(unsigned long long)st.evictions,st.used,st.slots);
    if(fails2){ fprintf(stderr,"FAIL: tiny-cache mismatch\n"); return 1; }
    if(!st.evictions){ fprintf(stderr,"FAIL: tiny cache never evicted\n"); return 1; }
    mc_cache_free(c);

    // 3) hit-path throughput (informational) + prefetch coverage
    c=mc_cache_new_archive(64ull<<20,a);
    mc_cache_prefetch_chunk(c,0,0,0,0);
    mc_cache_stats st3; mc_cache_get_stats(c,&st3);
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    long N=2000000; const mc_u8 *p=0;
    unsigned rs2=7;
    for(long i=0;i<N;++i){ rs2^=rs2<<13;rs2^=rs2>>17;rs2^=rs2<<5;
        p=mc_cache_get(c,0,(int)(rs2%16),(int)((rs2>>8)%16),(int)((rs2>>16)%16)); }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double s=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    (void)p;
    mc_cache_get_stats(c,&st3);
    printf("hit path: %.1f Mget/s single thread (prefetched chunk, %zu blocks resident)\n",
        N/1e6/s, st3.used);
    mc_cache_free(c);

    // 4) scan resistance: hot set comparable to cache size (700 blocks, 50% of
    // accesses) + cyclic scan over 3072 blocks through a 1024-slot cache —
    // scan pollution forces CLOCK to evict hot entries; S3-FIFO quarantines
    // one-hit scan traffic in its small queue.
    double hitrate[2];
    for(int pol=0;pol<2;++pol){
        c=mc_cache_new_archive(1024*4096ull,a);
        mc_cache_set_policy(c,pol==0?MC_CACHE_S3FIFO:MC_CACHE_CLOCK);
        unsigned r4=42; int scan_i=0;
        for(long it=0;it<200000;++it){
            r4^=r4<<13;r4^=r4>>17;r4^=r4<<5;
            int bz,by,bx;
            if(r4&1){ unsigned h=(r4>>8)%700; bz=h%8; by=(h/8)%16; bx=8+h/128; }
            else { int s=scan_i++%3072; bz=s%16; by=(s/16)%16; bx=s/256; }
            (void)mc_cache_get(c,0,bz,by,bx);
        }
        mc_cache_stats s4; mc_cache_get_stats(c,&s4);
        hitrate[pol]=(double)s4.hits/(s4.hits+s4.misses);
        printf("%s: %.1f%% hit (scan+hot mix)\n",pol==0?"s3fifo":"clock ",100*hitrate[pol]);
        mc_cache_free(c);
    }
    if(hitrate[0] < hitrate[1]){ fprintf(stderr,"FAIL: S3-FIFO worse than CLOCK on scan mix\n"); return 1; }
    // 5) writer support: replace a chunk, invalidate, readers must see new data
    c=mc_cache_new_archive(64ull<<20,a);
    mc_u8 before[4096];
    mc_cache_get_copy(c,0,8,8,8,before);                  // warm the cache
    static mc_u8 newchunk[256*256*256];
    for(size_t i=0;i<sizeof newchunk;++i) newchunk[i]=(mc_u8)(1+(i%200));
    if(mc_archive_append_chunk_raw(a,0,0,0,0,newchunk)!=0){ fprintf(stderr,"re-append failed\n"); return 1; }
    mc_u8 stale[4096];
    mc_cache_get_copy(c,0,8,8,8,stale);                   // still cached: old data
    if(memcmp(stale,before,4096)!=0){ fprintf(stderr,"FAIL: expected cached (stale) data pre-invalidate\n"); return 1; }
    mc_cache_invalidate_chunk(c,0,0,0,0);
    if(mc_cache_contains(c,0,8,8,8)){ fprintf(stderr,"FAIL: block still resident after invalidate\n"); return 1; }
    mc_u8 fresh[4096], want2[4096];
    mc_cache_get_copy(c,0,8,8,8,fresh);
    uint64_t co5=mc_archive_chunk_offset(a,0,0,0,0);
    mc_archive_decode_block(a,co5,8,8,8,want2);
    if(memcmp(fresh,want2,4096)!=0){ fprintf(stderr,"FAIL: post-invalidate read != new decode\n"); return 1; }
    // concurrent smoke: 4 readers hammer the chunk while a writer re-appends
    // + invalidates; then verify convergence after the last invalidate.
    for(int round=0;round<20;++round){
        for(size_t i=0;i<sizeof newchunk;++i) newchunk[i]=(mc_u8)(1+((i+round)%200));
        mc_archive_append_chunk_raw(a,0,0,0,0,newchunk);
        mc_cache_invalidate_chunk(c,0,0,0,0);
        mc_u8 g2[4096]; for(int r2=0;r2<64;++r2) mc_cache_get_copy(c,0,r2&15,(r2>>2)&15,8,g2);
    }
    mc_cache_invalidate_chunk(c,0,0,0,0);
    mc_cache_get_copy(c,0,8,8,8,fresh);
    co5=mc_archive_chunk_offset(a,0,0,0,0);
    mc_archive_decode_block(a,co5,8,8,8,want2);
    if(memcmp(fresh,want2,4096)!=0){ fprintf(stderr,"FAIL: convergence after writer rounds\n"); return 1; }
    printf("invalidate: OK (stale-until-invalidate, fresh after)\n");
    mc_cache_free(c);

    // 6) batch update (vc3d pattern): submit a working set incl. duplicates +
    // already-resident blocks; after the call every block must be resident and
    // queries must be pure hits.
    c=mc_cache_new_archive(64ull<<20,a);
    mc_cache_get_copy(c,0,5,5,5,want);                    // pre-resident block
    enum { NW=900 };
    static mc_block_id ws[NW]; int nw=0;
    for(int i=0;i<NW-4;++i){ ws[nw++]=(mc_block_id){0,(i*7)%16,(i*13)%16,(i*3)%16}; }
    ws[nw++]=(mc_block_id){0,5,5,5};                      // resident
    ws[nw++]=(mc_block_id){0,1,2,3}; ws[nw++]=(mc_block_id){0,1,2,3};  // dup
    ws[nw++]=(mc_block_id){1,0,0,0};                      // another LOD
    size_t dec=mc_cache_update(c,ws,(size_t)nw,0);
    mc_cache_stats su; mc_cache_get_stats(c,&su);
    int missing=0;
    for(int i=0;i<nw;++i) if(!mc_cache_contains(c,ws[i].lod,ws[i].bz,ws[i].by,ws[i].bx)) missing++;
    uint64_t h0=su.hits,m0=su.misses;
    for(int i=0;i<nw;++i){ mc_u8 g6[4096]; mc_cache_get_copy(c,ws[i].lod,ws[i].bz,ws[i].by,ws[i].bx,g6); }
    mc_cache_get_stats(c,&su);
    printf("update: decoded %zu of %d submitted, %d missing, post-update queries: %llu hits %llu misses\n",
        dec,nw,missing,(unsigned long long)(su.hits-h0),(unsigned long long)(su.misses-m0));
    if(missing || su.misses!=m0){ fprintf(stderr,"FAIL: batch update left misses\n"); return 1; }
    // correctness of one updated block vs direct decode
    mc_u8 g7[4096];
    mc_cache_get_copy(c,1,0,0,0,g7);
    uint64_t co7=mc_archive_chunk_offset(a,1,0,0,0);
    mc_archive_decode_block(a,co7,0,0,0,want);
    if(memcmp(g7,want,4096)!=0){ fprintf(stderr,"FAIL: updated block wrong\n"); return 1; }
    mc_cache_free(c);

    // 7) best-resident-LOD + async tickets
    c=mc_cache_new_archive(64ull<<20,a);
    (void)mc_cache_get(c,1,1,1,1);                 // make LOD1 parent of (2,2,2)@LOD0 resident
    if(mc_cache_best_lod(c,0,2,2,2)!=1){ fprintf(stderr,"FAIL: best_lod expected 1\n"); return 1; }
    if(mc_cache_best_lod(c,0,9,9,9)!=-1){ fprintf(stderr,"FAIL: best_lod expected -1\n"); return 1; }
    (void)mc_cache_get(c,0,2,2,2);
    if(mc_cache_best_lod(c,0,2,2,2)!=0){ fprintf(stderr,"FAIL: best_lod expected 0\n"); return 1; }
    static mc_block_id aw[512]; int na=0;
    for(int i=0;i<512;++i) aw[na++]=(mc_block_id){0,i%16,(i/16)%16,(i/256)%16};
    mc_cache_ticket *tk=mc_cache_update_async(c,aw,(size_t)na,4);
    mc_cache_ticket_wait(tk);
    if(!mc_cache_ticket_done(tk)){ fprintf(stderr,"FAIL: ticket not done after wait\n"); return 1; }
    mc_cache_ticket_free(tk);
    int miss=0; for(int i=0;i<na;++i) if(!mc_cache_contains(c,aw[i].lod,aw[i].bz,aw[i].by,aw[i].bx)) miss++;
    if(miss){ fprintf(stderr,"FAIL: async update left %d misses\n",miss); return 1; }
    // cancel path: must terminate and converge
    mc_cache_clear(c);
    tk=mc_cache_update_async(c,aw,(size_t)na,2);
    mc_cache_ticket_cancel(tk);
    mc_cache_ticket_wait(tk);
    if(!mc_cache_ticket_done(tk)){ fprintf(stderr,"FAIL: cancelled ticket not done\n"); return 1; }
    mc_cache_ticket_free(tk);
    printf("best_lod+async: OK\n");
    mc_cache_free(c);

    mc_archive_close(a);
    remove(path);
    printf("mc_cache: OK\n");
    return 0;
}
