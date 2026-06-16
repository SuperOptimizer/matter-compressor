// mc_segbench — microbenchmark for the classical sheet detector + post-proc.
// Times mc_seg_detect (the hot path: 6x separable gaussian blurs + per-voxel
// structure-tensor eigenvalue solve) and the topology ops over a synthetic
// chunk, reporting voxels/sec and per-op wall time. Used to baseline and
// regression-check perf work on mc_segment.c.
//
//   mc_segbench [dim] [iters]      (default dim=128, iters=5; reports best)
#include "mc_segment.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

int main(int argc, char **argv){
    int dim   = argc>1 ? atoi(argv[1]) : 128;
    int iters = argc>2 ? atoi(argv[2]) : 5;
    if(dim<8) dim=8; if(iters<1) iters=1;
    size_t n=(size_t)dim*dim*dim;

    uint8_t *vol=malloc(n), *mask=malloc(n), *mtmp=malloc(n);
    int32_t *lab=malloc(n*sizeof(int32_t));
    if(!vol||!mask||!mtmp||!lab){ fprintf(stderr,"oom\n"); return 1; }

    // a realistic-ish volume: a few wavy bright sheets + noise, so detect and
    // the post-proc ops do non-trivial work.
    for(int z=0;z<dim;++z)for(int y=0;y<dim;++y)for(int x=0;x<dim;++x){
        double w = 30.0*sin(x*0.12) + 20.0*cos(y*0.09);
        int onsheet = (fabs((z - dim/2) - w*0.1) < 1.5) || (fabs((z - dim/4) + w*0.1) < 1.5);
        int v = onsheet ? 210 : 45;
        v += ((x*131 + y*17 + z*7) % 23) - 11;        // deterministic "noise"
        vol[(size_t)(((size_t)z*dim+y)*dim+x)] = (uint8_t)(v<0?0:v>255?255:v);
    }

    mc_seg_params P = { 1.0f, 2.0f, 0.5f, 90 };

    double best_det=1e9;
    for(int it=0;it<iters;++it){
        double t0=now_s();
        if(mc_seg_detect(vol,dim,dim,dim,&P,mask)!=0){ fprintf(stderr,"detect failed\n"); return 1; }
        double dt=now_s()-t0; if(dt<best_det) best_det=dt;
    }
    // post-proc timed once on the detect output (operates in place; copy fresh).
    long det_on=0; for(size_t i=0;i<n;++i) if(mask[i]) det_on++;

    #define TIME_OP(label, expr) do{ \
        double b=1e9; for(int it=0;it<iters;++it){ memcpy(mtmp,mask,n); \
            double t0=now_s(); expr; double dt=now_s()-t0; if(dt<b)b=dt; } \
        printf("  %-16s %8.2f ms  (%6.1f Mvox/s)\n", label, b*1e3, n/b/1e6); }while(0)

    printf("mc_segbench: %dx%dx%d (%.2f Mvox), best of %d\n", dim,dim,dim, n/1e6, iters);
    printf("  %-16s %8.2f ms  (%6.1f Mvox/s)   sheet=%ld/%zu\n",
           "detect", best_det*1e3, n/best_det/1e6, det_on, n);
    TIME_OP("remove_small", mc_seg_remove_small(mtmp,dim,dim,dim,2000));
    TIME_OP("plug_holes",   mc_seg_plug_holes(mtmp,dim,dim,dim,1));
    TIME_OP("close(r1)",    mc_seg_close(mtmp,dim,dim,dim,1));
    TIME_OP("fill_cavities",mc_seg_fill_cavities(mtmp,dim,dim,dim));
    TIME_OP("label",        mc_seg_label(mtmp,dim,dim,dim,lab));

    free(vol); free(mask); free(mtmp); free(lab);
    return 0;
}
