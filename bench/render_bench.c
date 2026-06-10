#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
#define N 256
int main(void){
    uint8_t *vol=malloc((size_t)N*N*N);
    for(size_t i=0;i<(size_t)N*N*N;i++) vol[i]=(uint8_t)(i*2654435761u>>13);
    mc_sample_src src=mc_sample_src_dense(vol,N,N,N);
    enum{W=1024,H=1024,GW=256,GH=256};
    // gently curved quad: z varies with grid position
    float *grid=malloc((size_t)GW*GH*3*sizeof(float));
    for(int gy=0;gy<GH;gy++)for(int gx=0;gx<GW;gx++){
        float *p=grid+((size_t)gy*GW+gx)*3;
        p[0]=100.0f+10.0f*(float)gy/GH+6.0f*(float)gx/GW;
        p[1]=(float)gy*0.97f; p[2]=(float)gx*0.97f;
    }
    mc_quad q={.grid=grid,.gw=GW,.gh=GH};
    uint8_t *img=malloc((size_t)W*H);
    struct { const char *name; mc_render_params p; } cases[]={
        {"quad slice tri   ",{.filter=MC_FILTER_TRILINEAR,.comp=MC_COMP_NONE}},
        {"quad max x1 tri  ",{.filter=MC_FILTER_TRILINEAR,.comp=MC_COMP_MAX,.t0=0,.t1=0,.dt=1}},
        {"quad max x9 tri  ",{.filter=MC_FILTER_TRILINEAR,.comp=MC_COMP_MAX,.t0=-4,.t1=4,.dt=1}},
        {"quad max x9 nn   ",{.filter=MC_FILTER_NEAREST,.comp=MC_COMP_MAX,.t0=-4,.t1=4,.dt=1}},
    };
    // pure gen cost
    float *pts=malloc((size_t)W*H*3*sizeof(float)*2);
    double t0=now(); int reps=10;
    for(int r=0;r<reps;r++) mc_quad_gen(&q,0,0,0.25f,W,H,pts,pts+(size_t)W*H*3);
    printf("quad gen only     : %6.2f ms/frame\n",(now()-t0)/reps*1e3);
    for(int k=0;k<4;k++)for(int th=1;th<=8;th*=8){
        t0=now();
        for(int r=0;r<reps;r++) mc_render_quad(&src,&q,0,0,0.25f,W,H,&cases[k].p,img,th);
        double dt=(now()-t0)/reps;
        printf("%s %dT: %6.2f ms/frame (%.0f Mpix/s)\n",cases[k].name,th,dt*1e3,W*H/1e6/dt);
    }
    return 0;
}
