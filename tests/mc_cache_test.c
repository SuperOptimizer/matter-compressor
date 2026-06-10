// mc_cache test: build a small archive, hammer the cache from N threads with
// a zipf-ish revisit pattern, verify every cached block equals a direct
// decode, then check hit rate and that a tiny cache evicts correctly.
#include "../src/mc_archive_api.h"
#include "../src/mc_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

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
    mc_archive_close(a);
    remove(path);
    printf("mc_cache: OK\n");
    return 0;
}
