// mc_surface_test — round-trip the parametric-surface I/O: the 4x-f32 grid TIFF
// (x,y,z,depth), OBJ mesh export/import, and PPM. Verifies the grid (z,y,x) +
// per-point depth survive, invalid points are preserved, and the OBJ carries
// the grid back.
#include "mc_surface.h"
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

    // ---- general OBJ mesh (real-world: interleaved v/vn, f a/b/c) ----
    {
        // write a VolCart-style OBJ (interleaved v/vn, f with v/vt/vn indices).
        FILE *f=fopen("/tmp/mc_mesh.obj","wb");
        fprintf(f,"# VolCart OBJ\n");
        fprintf(f,"v 1 2 3\nvn 0 0 1\nv 4 5 6\nvn 0 1 0\nv 7 8 9\nvn 1 0 0\nv 1 1 1\nvn 0 0 1\n");
        fprintf(f,"f 1/1/1 2/2/2 3/3/3\n");          // tri
        fprintf(f,"f 1/1/1 2/2/2 3/3/3 4/4/4\n");    // quad -> fan into 2 tris
        fclose(f);
        mc_mesh m;
        CHECK(mc_mesh_load_obj("/tmp/mc_mesh.obj",&m)==0);
        CHECK(m.nv==4); CHECK(m.nt==3);              // 1 + 2 (quad fan)
        CHECK(m.vn!=NULL);
        CHECK(m.v[0]==1&&m.v[1]==2&&m.v[2]==3);
        CHECK(m.tri[0]==0&&m.tri[1]==1&&m.tri[2]==2);
        // round-trip
        CHECK(mc_mesh_save_obj("/tmp/mc_mesh_rt.obj",&m)==0);
        mc_mesh m2; CHECK(mc_mesh_load_obj("/tmp/mc_mesh_rt.obj",&m2)==0);
        CHECK(m2.nv==m.nv && m2.nt==m.nt);
        printf("OBJ mesh load (interleaved v/vn, quad fan): nv=%d nt=%d OK\n", m.nv, m.nt);
        mc_mesh_free(&m); mc_mesh_free(&m2);
    }

    // ---- VC per-pixel map (.ppm ordered map, dim=6 = xyz+normal) ----
    {
        int W=5,H=3,D=6;
        FILE *f=fopen("/tmp/mc_vc.ppm","wb");
        fprintf(f,"width: %d\nheight: %d\ndim: %d\nordered: true\ntype: double\nversion: 1\n<>\n",W,H,D);
        for(int i=0;i<W*H;++i){
            double row[6];
            if(i==4){ for(int k=0;k<6;++k) row[k]=0; }   // off-surface (0,0,0)
            else { row[0]=100+i; row[1]=200+i; row[2]=10+i; row[3]=0; row[4]=0; row[5]=1; }
            fwrite(row,sizeof(double),6,f);
        }
        fclose(f);
        mc_surface vs;
        CHECK(mc_surface_load_vcps_ppm("/tmp/mc_vc.ppm",&vs,7.0f)==0);
        CHECK(vs.gw==W && vs.gh==H);
        // point 0: xyz (100,200,10) -> grid (z,y,x)=(10,200,100), depth 7.
        CHECK(vs.grid[0]==10 && vs.grid[1]==200 && vs.grid[2]==100);
        CHECK(vs.depth[0]==7.0f);
        // point 4: off-surface -> invalid.
        CHECK(vs.grid[4*3]==-1 && vs.grid[4*3+1]==-1 && vs.grid[4*3+2]==-1);
        printf("VC ppm load (dim=6 xyz+normal -> grid+depth): %dx%d OK\n", vs.gw, vs.gh);
        mc_surface_free(&vs);
    }

    // ---- mc_grid_from_mesh: PCA plane-fit + nearest-vertex resample. Build a
    // near-planar mesh (a tilted plane sampled on a 5x5 lattice, fan-triangulated)
    // and resample it to a grid; the fit should recover plane points (exercises
    // jacobi3 eigendecomposition + the in-plane projection + nearest-vertex fill).
    {
        mc_mesh m; memset(&m,0,sizeof m);
        int N=5; m.nv=N*N;
        m.v=malloc((size_t)m.nv*3*sizeof(float));
        // plane: x,y span the lattice; z = small tilt -> dominant normal ~ +z.
        for(int j=0;j<N;++j)for(int i=0;i<N;++i){
            int vi=j*N+i;
            m.v[vi*3]  =10.f + i*2.f;            // x
            m.v[vi*3+1]=20.f + j*2.f;            // y
            m.v[vi*3+2]=5.f + 0.1f*i + 0.05f*j;  // z (gentle tilt)
        }
        // fan-triangulate each lattice quad (a realistic mesh; the fit uses
        // vertices, but this keeps the mesh well-formed for any tri consumer).
        m.nt=(N-1)*(N-1)*2; m.tri=malloc((size_t)m.nt*3*sizeof(int)); int t=0;
        for(int j=0;j<N-1;++j)for(int i=0;i<N-1;++i){
            int a=j*N+i,b=j*N+i+1,c=(j+1)*N+i,d=(j+1)*N+i+1;
            m.tri[t*3]=a;m.tri[t*3+1]=b;m.tri[t*3+2]=d;t++;
            m.tri[t*3]=a;m.tri[t*3+1]=d;m.tri[t*3+2]=c;t++;
        }
        mc_surface g;
        CHECK(mc_grid_from_mesh(&m, 8, 8, 3.0f, &g)==0);
        CHECK(g.gw==8 && g.gh==8);
        // every filled grid point should match SOME mesh vertex (nearest fill),
        // and a dense planar lattice should fill most cells.
        int filled=0;
        for(int k=0;k<g.gw*g.gh;++k){
            float vz=g.grid[k*3], vy=g.grid[k*3+1], vx=g.grid[k*3+2];
            if(vz<0&&vy<0&&vx<0) continue;        // invalid cell
            filled++;
            int match=0;
            for(int vi=0;vi<m.nv;++vi)
                if(fabsf(vx-m.v[vi*3])<1e-3f && fabsf(vy-m.v[vi*3+1])<1e-3f && fabsf(vz-m.v[vi*3+2])<1e-3f){ match=1; break; }
            CHECK(match);                          // filled point is a real vertex
            CHECK(g.depth[k]==3.0f);
        }
        CHECK(filled > g.gw*g.gh/2);               // dense plane -> mostly filled
        printf("grid_from_mesh: %d verts -> %dx%d grid, %d/%d cells filled\n",
               m.nv, g.gw, g.gh, filled, g.gw*g.gh);

        // auto-resolution path (gw/gh<=0 -> pick from vertex count).
        mc_surface g2;
        CHECK(mc_grid_from_mesh(&m, 0, 0, 2.0f, &g2)==0);
        CHECK(g2.gw>0 && g2.gh>0);
        printf("grid_from_mesh auto-res: %d verts -> %dx%d grid\n", m.nv, g2.gw, g2.gh);
        mc_surface_free(&g); mc_surface_free(&g2);

        // mc_mesh_save_obj WITH per-vertex normals (the "f a//a" face form) +
        // reload round-trip. Attach a normal per vertex first.
        m.vn=malloc((size_t)m.nv*3*sizeof(float));
        for(int vi=0;vi<m.nv;++vi){ m.vn[vi*3]=0; m.vn[vi*3+1]=0; m.vn[vi*3+2]=1; }
        CHECK(mc_mesh_save_obj("/tmp/mc_mesh_n.obj",&m)==0);
        mc_mesh rm;
        CHECK(mc_mesh_load_obj("/tmp/mc_mesh_n.obj",&rm)==0);
        CHECK(rm.nv==m.nv && rm.nt==m.nt);
        CHECK(rm.vn!=NULL);                 // normals survived the round-trip
        printf("mesh_save_obj w/normals: nv=%d nt=%d, normals round-tripped\n", rm.nv, rm.nt);
        mc_mesh_free(&rm); mc_mesh_free(&m);
    }

    mc_surface_free(&s); mc_surface_free(&r); mc_surface_free(&o);
    printf(fails ? "mc_surface_test: %d FAILED\n" : "mc_surface_test: OK\n", fails);
    return fails?1:0;
}
