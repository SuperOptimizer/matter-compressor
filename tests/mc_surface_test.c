// mc_surface_test — round-trip the parametric-surface I/O: the 4x-f32 grid TIFF
// (x,y,z,depth), OBJ mesh export/import, and PPM. Verifies the grid (z,y,x) +
// per-point depth survive, invalid points are preserved, and the OBJ carries
// the grid back.
#include "mc_surface.h"
#include "mc_tiff.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int fails=0;
#define CHECK(x) do{ if(!(x)){ fails++; fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);} }while(0)

int main(void){
    const int gw=7, gh=5;
    mc_surface s; memset(&s,0,sizeof s);
    s.gw=gw; s.gh=gh;
    s.grid=malloc((size_t)gw*gh*3*sizeof(float));
    s.depth=malloc((size_t)gw*gh*sizeof(float));
    // a tilted plane in volume space + per-point depth; one invalid point.
    for(int y=0;y<gh;++y)for(int x=0;x<gw;++x){
        size_t i=(size_t)y*gw+x;
        if(x==2 && y==1){ s.grid[i*3]=-1; s.grid[i*3+1]=-1; s.grid[i*3+2]=-1; s.depth[i]=0; continue; }
        float vx=100.f+x*4.f, vy=200.f+y*4.f, vz=50.f+x*0.5f+y*0.25f;
        s.grid[i*3]=vz; s.grid[i*3+1]=vy; s.grid[i*3+2]=vx;   // (z,y,x)
        s.depth[i]=5.f + 0.1f*(x+y);
    }

    // ---- TIFF round-trip ----
    CHECK(mc_surface_save_tiff("/tmp/mc_surf_rt.tif",&s)==0);
    mc_surface r;
    CHECK(mc_surface_load_tiff("/tmp/mc_surf_rt.tif",&r)==0);
    CHECK(r.gw==gw && r.gh==gh);
    int bad=0, inval_ok=0;
    for(size_t i=0;i<(size_t)gw*gh;++i){
        for(int c=0;c<3;++c) if(r.grid[i*3+c]!=s.grid[i*3+c]) bad++;
        if(r.depth[i]!=s.depth[i]) bad++;
    }
    // invalid point preserved as (-1,-1,-1).
    { size_t i=(size_t)1*gw+2; if(r.grid[i*3]==-1&&r.grid[i*3+1]==-1&&r.grid[i*3+2]==-1) inval_ok=1; }
    CHECK(bad==0); CHECK(inval_ok);
    CHECK(fabsf(r.mean_depth - s.mean_depth) < 1e-3f || s.mean_depth==0);
    printf("TIFF surface round-trip: grid+depth exact (%d diffs), invalid preserved=%d\n", bad, inval_ok);

    // the quad view exposes the same grid pointer.
    mc_quad q = mc_surface_quad(&r);
    CHECK(q.gw==gw && q.gh==gh && q.grid==r.grid);

    // ---- OBJ export + re-import ----
    CHECK(mc_surface_save_obj("/tmp/mc_surf.obj",&r)==0);
    mc_surface o;
    CHECK(mc_surface_load_obj("/tmp/mc_surf.obj",&o)==0);
    CHECK(o.gw==gw && o.gh==gh);
    // OBJ skips the invalid point (no vertex), so it can't perfectly reconstruct
    // the grid cell-for-cell; instead verify the VALID points round-trip in
    // order for the rows before the gap. Count how many valid points matched.
    int vmatch=0, vtotal=0;
    for(size_t i=0;i<(size_t)gw*gh;++i){
        const float *p=&s.grid[i*3];
        if(p[0]<0) continue;       // was invalid
        vtotal++;
    }
    // re-imported grid is filled row-major from the emitted vertices; the first
    // valid vertices line up with the first valid source points.
    {   size_t si=0, oi=0;
        for(; si<(size_t)gw*gh && oi<(size_t)gw*gh; ){
            const float *sp=&s.grid[si*3];
            if(sp[0]<0){ si++; continue; }                 // source invalid -> skipped in OBJ
            const float *op=&o.grid[oi*3];
            if(fabsf(op[0]-sp[0])<1e-3f && fabsf(op[1]-sp[1])<1e-3f && fabsf(op[2]-sp[2])<1e-3f) vmatch++;
            si++; oi++;
        }
    }
    CHECK(vmatch==vtotal);
    printf("OBJ export+import: %d/%d valid vertices match\n", vmatch, vtotal);

    // ---- PPM ----
    { int w=8,h=4; uint8_t img[8*4*3]; for(int i=0;i<8*4*3;++i) img[i]=(uint8_t)(i*5);
      CHECK(mc_ppm_write("/tmp/mc_surf.ppm",w,h,3,img)==0);
      uint8_t gray[8*4]; for(int i=0;i<32;++i) gray[i]=(uint8_t)(i*7);
      CHECK(mc_ppm_write("/tmp/mc_surf_gray.ppm",w,h,1,gray)==0);
      printf("PPM write (rgb + gray) OK\n");
    }

    mc_surface_free(&s); mc_surface_free(&r); mc_surface_free(&o);
    printf(fails ? "mc_surface_test: %d FAILED\n" : "mc_surface_test: OK\n", fails);
    return fails?1:0;
}
