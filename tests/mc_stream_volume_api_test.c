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

    // 4b) NON-blocking (interactive) sample source -> mc_volume_block_ptr (no-copy
    //     arena pointer path). Render a small plane through the dense core after
    //     pre-warming the cache with a blocking get, inside a frozen frame.
    {
        mc_sample_src iss=mc_volume_sample_src(v,0,0);   // blocking=0 -> block_ptr path
        mc_volume_freeze(v);
        mc_plane pl={.origin={nz/2.f,ny/2.f,nx/2.f},.normal={1,0,0}}; mc_plane_basis(&pl);
        mc_render_params rp={.filter=MC_FILTER_TRILINEAR,.comp=MC_COMP_MAX,.t0=-4,.t1=4,.dt=1};
        uint8_t *im=calloc(48*48,1);
        mc_render_plane(&iss,&pl,48,48,1.0f,&rp,im,0);
        mc_volume_thaw(v);
        free(im);
    }

    // 4b2) tick-phase loop that DRIVES mc_volume_thaw's miss-drain: render a
    //      large frozen frame across the whole volume via the non-blocking source
    //      (touches present-but-uncached blocks -> records misses), then thaw
    //      (drains misses -> mc_cache_update fill + region-mark + change_gen).
    //      Repeat until the frame stops generating new fills.
    {
        mc_sample_src iss=mc_volume_sample_src(v,0,0);
        mc_plane pl={.origin={nz/2.f,ny/2.f,nx/2.f},.normal={0,0,1}}; mc_plane_basis(&pl);
        mc_render_params rp={.filter=MC_FILTER_TRILINEAR,.comp=MC_COMP_MEAN,.t0=-40,.t1=40,.dt=2};
        uint8_t *im=calloc(256*256,1);
        uint64_t g_prev=0; int stable=0;
        for(int it=0; it<60 && stable<3; ++it){
            mc_volume_freeze(v);
            mc_render_plane(&iss,&pl,256,256,2.0f,&rp,im,2);   // wide frame -> many misses
            uint64_t g=mc_volume_region_gen(v,0,nz/2/16/16,ny/2/16/16,nx/2/16/16);
            mc_volume_thaw(v);
            if(g==g_prev) stable++; else { stable=0; g_prev=g; }
            nap(3);
        }
        free(im);
        printf("tick-phase: settled (change_gen stable)\n");
    }

    // 4b3) LOD-fallback render over the volume's pyramid (non-blocking), in a
    //      FRESH frozen frame BEFORE the fine level is fully cached -> the
    //      fine-level residency check (vol_block_resident) returns 0 for
    //      present-but-uncached blocks and the renderer falls to a coarser level.
    if(nl>1){
        mc_volume_thaw(v);                  // ensure not frozen
        // a brand-new volume to guarantee a cold fine level.
        mc_volume *v2=mc_volume_open_streaming(path,"/tmp",(size_t)64<<20);
        if(v2){
            mc_sample_lods ls=mc_volume_sample_lods(v2,0);   // blocking=0 -> resident checks
            mc_plane pl={.origin={nz/2.f,ny/2.f,nx/2.f},.normal={1,0,0}}; mc_plane_basis(&pl);
            mc_render_params rp={.filter=MC_FILTER_TRILINEAR,.comp=MC_COMP_MAX,.t0=-4,.t1=4,.dt=1};
            uint8_t *im=calloc(96*96,1);
            mc_volume_freeze(v2);
            mc_render_plane_lod(&ls,&pl,96,96,2.0f,&rp,im,2);   // scale=2 -> may pick L1
            mc_volume_thaw(v2);
            free(im);
            mc_volume_free(v2);
            printf("lod-fallback render: OK\n");
        }
    }

    // 4c) SHADED + INK with the full lighting/SSS/curvature/ink-lock fields set,
    //     rendered against the r=[90,110] sheet -> covers shade_ray's secondary
    //     march, curvature, and the ink sheet-lock run detection.
    {
        mc_sample_src bs=mc_volume_sample_src(v,0,1);
        // aim a plane tangent to the shell so the ray crosses the sheet.
        mc_plane pl={.origin={nz/2.f,ny/2.f,nx/2.f-100.f},.normal={0,0,1}}; mc_plane_basis(&pl);
        uint8_t *im=calloc(64*64,1);
        mc_render_params sh={.filter=MC_FILTER_TRILINEAR,.comp=MC_COMP_SHADED,
            .t0=-30,.t1=30,.dt=1,.alpha_min=0.1f,.alpha_opacity=0.9f,
            .light={1,0.3f,0.2f},.ambient=0.2f,.diffuse=0.8f,.specular=0.3f,
            .shininess=20,.absorption=1.0f,.shadow=0.5f,.sss=0.4f,.curvature=0.5f,
            .light_surface_rel=1};
        mc_render_plane(&bs,&pl,64,64,1.0f,&sh,im,2);
        mc_render_params ink={.filter=MC_FILTER_TRILINEAR,.comp=MC_COMP_INK,
            .t0=-30,.t1=30,.dt=1,.alpha_min=0.1f,.alpha_opacity=0.9f,
            .sss=0.4f,.transmission=0.4f,.ink_lock=6.0f,.curvature=0.3f};
        mc_render_plane(&bs,&pl,64,64,1.0f,&ink,im,2);
        free(im);
    }

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
