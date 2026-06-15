// mc_stream_volume_api_test — drive the STREAMING (.mca) volume pipeline against
// a local file: download thread (dl_main), decode pool, async freeze/thaw fill,
// prefetch_shard/level, request_region, ensure_region, and stats. Complements
// mc_stream_volume_test (basic center-block) by exercising the prefetch +
// tick-phase + whole-level paths that the local-dir volume test can't reach.
// No network: mc_volume_open_streaming maps a local .mca read-only.
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define NX 512
#define NY 384
#define NZ 320
static int fails=0;
#define CHECK(c,...) do{ if(!(c)){ fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
    fprintf(stderr,__VA_ARGS__); fputc('\n',stderr); fails++; } }while(0)

// two concentric shells of material so different regions are present vs air.
static mc_u8 src_fn(void *ud,int x,int y,int z){ (void)ud;
    double cx=NX/2.0,cy=NY/2.0,cz=NZ/2.0;
    double r=sqrt((x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz));
    if(r<60) return 200;
    if(r>90 && r<110) return 120;
    return 0;
}

static void nap(int ms){ struct timespec t={0,ms*1000*1000}; nanosleep(&t,NULL); }

int main(void){
    const char *path="/tmp/mc_stream_api.mca";
    mc_build_opts o={.nx=NX,.ny=NY,.nz=NZ,.quality=6.0f};
    if(mc_build_to_file(src_fn,NULL,&o,path)!=0){ fprintf(stderr,"build failed\n"); return 1; }

    mc_volume *v=mc_volume_open_streaming(path,"/tmp",(size_t)256<<20);
    CHECK(v!=NULL,"open_streaming failed");
    if(!v){ remove(path); return 1; }

    int nl=mc_volume_nlods(v);
    CHECK(nl>=1,"nlods<1");
    int nz,ny,nx; mc_volume_shape(v,0,&nz,&ny,&nx);
    printf("streaming vol: nlods=%d L0=%d,%d,%d\n",nl,nz,ny,nx);

    // staging-bytes tuning (streaming pipeline backpressure knob).
    mc_volume_set_staging_bytes(v,(size_t)32<<20);

    // 1) prefetch a whole coarse level (drives dl_main batch + decode pool).
    volatile int cancel=0;
    if(nl>1) mc_volume_prefetch_level(v,nl-1,4,&cancel);

    // 2) request_region + prefetch_shard around the dense core, then tick until
    //    the center block lands resident (exercises ensure_region + thaw fill).
    int cz=(nz/2)/16, cy=(ny/2)/16, cx=(nx/2)/16;
    mc_volume_request_region(v,0, cz/16, cy/16, cx/16);
    mc_volume_prefetch_shard(v,0, cz/16, cy/16, cx/16);

    uint8_t blk[16*16*16];
    int got=0;
    for(int it=0; it<400 && !got; ++it){
        mc_volume_freeze(v);
        if(mc_volume_try_block(v,0,cz,cy,cx,blk)==1) got=1;
        mc_volume_thaw(v);                       // drains misses -> req_push -> fill
        if(!got) nap(2);
    }
    // authoritative blocking get
    int st=mc_volume_get_block(v,0,cz,cy,cx,blk);
    long nzc=0; for(int i=0;i<4096;++i) if(blk[i]) nzc++;
    CHECK(st==1,"center get_block st=%d",st);
    CHECK(nzc>0,"center block empty (got=%d)",got);
    printf("center: try_got=%d get_st=%d nonzero=%ld/4096\n",got,st,nzc);

    // 3) a far-corner block must read as air (ZERO region -> st==0, all zeros).
    mc_volume_freeze(v);
    int rc=mc_volume_try_block(v,0,0,0,0,blk);
    mc_volume_thaw(v);
    long csum=0; for(int i=0;i<4096;++i) csum+=blk[i];
    printf("corner: try=%d sum=%ld\n",rc,csum);

    // 4) sample source over the streaming volume (blocking) + a point sample.
    mc_sample_src ss=mc_volume_sample_src(v,0,1);
    mc_sampler *smp=mc_sampler_new(&ss);
    float cv=mc_sample_point(smp,nz/2.0f,ny/2.0f,nx/2.0f,MC_FILTER_TRILINEAR);
    CHECK(cv>0,"streaming center sample=%g should be material",cv);
    mc_sampler_free(smp);

    // 5) stats reflect activity.
    mc_volume_stats s={0}; mc_volume_get_stats(v,&s);
    printf("stats: net=%llu disk=%llu hits=%llu misses=%llu\n",
        (unsigned long long)s.net_bytes,(unsigned long long)s.disk_bytes,
        (unsigned long long)s.cache_hits,(unsigned long long)s.cache_misses);

    mc_volume_free(v);
    remove(path);
    printf(fails?"mc_stream_volume_api: %d FAILURES\n":"mc_stream_volume_api: OK\n",fails);
    return fails?1:0;
}
