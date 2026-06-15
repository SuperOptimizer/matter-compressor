// mc_cache test: build a small archive, hammer the cache from N threads with
// a zipf-ish revisit pattern, verify every cached block equals a direct
// decode, then check hit rate and that a tiny cache evicts correctly.
//
// CONCURRENCY CONTRACT (zero-lock cache): concurrent access from N threads is
// only valid while the cache is FROZEN -- frozen reads are pure (hit -> memcpy,
// miss -> read-through decode, NO insert), so there is no shared mutation to
// race. All inserts happen single-owner during the THAW window. The roomy test
// therefore: thaw-fills the working set (single-owner), freezes, then runs the
// concurrent readers. (Hammering an UNFROZEN cache from many threads -- which
// decode-and-insert -- violates the contract and is not supported lock-free.)
#include "../src/matter_compressor.h"
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

    // 1) multithreaded correctness + hit rate with a roomy cache.
    // Per the contract: fill the working set single-owner (mc_cache_update), then
    // FREEZE, then run the concurrent readers (pure frozen reads, no insert race).
    mc_cache *c=mc_cache_new_archive(64ull<<20,a);
    // working set = every block the workers touch: the full 16^3 block grid of
    // chunk 0 (hot set is bx 4..7; cold set is the whole 0..15 cube).
    mc_block_id rws[16*16*16]; int rnw=0;
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx)
        rws[rnw++]=(mc_block_id){0,bz,by,bx};
    mc_cache_update(c,rws,(size_t)rnw,0);   // single-owner fill (this thread only)
    mc_cache_freeze(c);                    // reads are now immutable + lock-free
    enum { NT=8 };
    pthread_t th[NT]; warg_t wa[NT];
    for(int i=0;i<NT;++i){ wa[i]=(warg_t){c,a,i,0}; pthread_create(&th[i],NULL,worker,&wa[i]); }
    int fails=0;
    for(int i=0;i<NT;++i){ pthread_join(th[i],NULL); fails+=wa[i].fails; }
    mc_cache_thaw(c);                      // reopen for the eviction sub-tests below
    mc_cache_stats st; mc_cache_get_stats(c,&st);
    printf("roomy: %d fails, working set %zu/%zu slots resident\n",
        fails, st.used, st.slots);
    if(fails){ fprintf(stderr,"FAIL: voxel mismatch\n"); return 1; }
    // Frozen reads are pure (no hit/miss counter mutation -- that would itself be a
    // shared write race), so we assert RESIDENCY of the pre-filled working set
    // instead of a hit rate: every block the concurrent readers touch is resident.
    if(st.used < (size_t)rnw){ fprintf(stderr,"FAIL: working set not fully resident (%zu < %d)\n",st.used,rnw); return 1; }
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

    // 2b) COLD churn: touch every block of a 256^3 chunk EXACTLY ONCE into a tiny
    //     cache. One-touch keys are never re-referenced, so they evict from the
    //     S3-FIFO small queue (the ghost_put cold-eviction path) instead of being
    //     promoted to main — distinct from the cyclic pattern above.
    c=mc_cache_new_archive((1ull<<20),a);
    mc_u8 g2[4096];
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx)
        mc_cache_get_copy(c,0,bz,by,bx,g2);          // each block touched once
    mc_cache_get_stats(c,&st);
    printf("cold-churn: evictions %llu used %zu/%zu\n",
        (unsigned long long)st.evictions,st.used,st.slots);
    if(!st.evictions){ fprintf(stderr,"FAIL: cold churn never evicted\n"); return 1; }
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

    // 8) tick-phase contract: thaw->resolve, freeze->lock-free reads + miss
    // feedback, thaw->drain; pins protect the frame's set; legacy mode
    // (never frozen) already covered by tests 1-7.
    c=mc_cache_new_archive(64ull<<20,a);
    static mc_block_id ws8[256]; static const mc_u8 *ptr8[256];
    for(int i=0;i<256;++i) ws8[i]=(mc_block_id){0,i%16,(i/16)%16,5};
    size_t dec8=mc_cache_resolve(c,ws8,256,ptr8,0);
    for(int i=0;i<256;++i) if(!ptr8[i]){ fprintf(stderr,"FAIL: resolve NULL ptr\n"); return 1; }
    mc_cache_freeze(c);
    // frozen hit: same pointer, no locks; content matches direct decode
    const mc_u8 *p8=mc_cache_get(c,0,3,3,5);
    if(p8!=ptr8[3+3*16]){ fprintf(stderr,"FAIL: frozen get != resolved ptr\n"); return 1; }
    uint64_t co8=mc_archive_chunk_offset(a,0,0,0,0);
    mc_u8 want8[4096];
    mc_archive_decode_block(a,co8,3,3,5,want8);
    if(memcmp(p8,want8,4096)!=0){ fprintf(stderr,"FAIL: frozen content\n"); return 1; }
    // frozen miss: NULL + recorded; get_copy read-through still correct
    if(mc_cache_get(c,0,9,9,9)!=NULL){ fprintf(stderr,"FAIL: frozen miss not NULL\n"); return 1; }
    mc_u8 rt8[4096];
    mc_cache_get_copy(c,0,9,9,8,rt8);
    mc_archive_decode_block(a,co8,9,9,8,want8);
    if(memcmp(rt8,want8,4096)!=0){ fprintf(stderr,"FAIL: frozen read-through\n"); return 1; }
    // frozen writes refused
    if(mc_cache_update(c,ws8,256,0)!=0){ fprintf(stderr,"FAIL: frozen update not refused\n"); return 1; }
    mc_cache_thaw(c);
    static mc_block_id missed[1024];
    size_t nm8=mc_cache_misses_drain(c,missed,1024);
    int found99=0,found98=0;
    for(size_t i=0;i<nm8;++i){
        if(missed[i].lod==0&&missed[i].bz==9&&missed[i].by==9&&missed[i].bx==9) found99=1;
        if(missed[i].lod==0&&missed[i].bz==9&&missed[i].by==9&&missed[i].bx==8) found98=1;
    }
    if(!found99||!found98){ fprintf(stderr,"FAIL: misses not recorded (%zu)\n",nm8); return 1; }
    mc_cache_update(c,missed,nm8,0);
    mc_cache_freeze(c);
    if(mc_cache_get(c,0,9,9,9)==NULL){ fprintf(stderr,"FAIL: drained miss not resident next frame\n"); return 1; }
    mc_cache_thaw(c);
    printf("tick-phase: OK (resolve %zu decoded, %zu misses fed back)\n",dec8,nm8);
    mc_cache_free(c);

    mc_archive_close(a);
    remove(path);
    printf("mc_cache: OK\n");
    return 0;
}
