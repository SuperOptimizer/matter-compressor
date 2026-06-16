// mc_trace_test — verify the geometric tracer cost terms: analytic jacobians
// vs central finite differences (the tracer's correctness rests on these), and
// a relaxation test where a perturbed flat grid is pulled back to a smooth,
// evenly-spaced sheet by Dist + Straight + SDIR.
#include "mc_trace.h"
#include "mc_solve.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int fails=0;
#define CHECK(x) do{ if(!(x)){ fails++; fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);} }while(0)

// Direct analytic-jacobian-vs-FD check, using the test hooks the .c exports
// under MC_TRACE_TEST (dist/straight/sdir eval fns + a ctx builder).
int   mc_trace__dist_eval(void*,const double*const*,double*,double*const*);
int   mc_trace__straight_eval(void*,const double*const*,double*,double*const*);
int   mc_trace__sdir_eval(void*,const double*const*,double*,double*const*);
void *mc_trace__make_ctx(int which, double unit, double w);   // 0=dist,1=straight,2=sdir
static double fd_check(mc_residual_fn fn, void *ctx, int nblk, double pts[][3]){
    const double *pp[8]; for(int k=0;k<nblk;++k) pp[k]=pts[k];
    double r0[1]; double jana[8][3]; double *jp[8];
    for(int k=0;k<nblk;++k) jp[k]=jana[k];
    fn(ctx,pp,r0,jp);
    double maxerr=0;
    for(int k=0;k<nblk;++k) for(int d=0;d<3;++d){
        double save=pts[k][d], h=1e-6, rp[1], rm[1];
        pts[k][d]=save+h; fn(ctx,pp,rp,NULL);
        pts[k][d]=save-h; fn(ctx,pp,rm,NULL);
        pts[k][d]=save;
        double fd=(rp[0]-rm[0])/(2*h);
        double e=fabs(fd-jana[k][d]); if(e>maxerr)maxerr=e;
    }
    return maxerr;
}

int main(void){
    // ===== 0. analytic jacobians vs central FD for all three terms, at a few
    // non-degenerate configs (the whole tracer rests on these). =====
    {
        double maxe=0;
        void *cd=mc_trace__make_ctx(0,10.0,1.0);
        void *cs=mc_trace__make_ctx(1,10.0,0.7);
        void *cv=mc_trace__make_ctx(2,10.0,1.0);
        double A[2][3]={{1,2,3},{9,5,7}};                 maxe=fmax(maxe,fd_check(mc_trace__dist_eval,cd,2,A));
        double A2[2][3]={{0,0,0},{2,1,0}};                maxe=fmax(maxe,fd_check(mc_trace__dist_eval,cd,2,A2)); // near (<unit)
        double S[3][3]={{0,0,0},{9,3,1},{20,1,2}};        maxe=fmax(maxe,fd_check(mc_trace__straight_eval,cs,3,S));
        double S2[3][3]={{0,0,0},{5,9,0},{10,0,0}};       maxe=fmax(maxe,fd_check(mc_trace__straight_eval,cs,3,S2)); // >30deg bend
        double V[3][3]={{0,0,0},{10,1,2},{0.5,11,-1}};    maxe=fmax(maxe,fd_check(mc_trace__sdir_eval,cv,3,V));
        double V2[3][3]={{0,0,0},{12,0,0},{0,8,3}};       maxe=fmax(maxe,fd_check(mc_trace__sdir_eval,cv,3,V2));
        CHECK(maxe < 1e-4);
        printf("0. analytic jac vs FD (dist/straight/sdir): max err %.2e\n", maxe);
        free(cd); free(cs); free(cv);
    }

    // ===== A. DistLoss optimum: two free points, target spacing 10 along x.
    // Start them 4 apart; after solve |a-b| should be 10. =====
    {
        mc_surf_grid *g=mc_surf_grid_create(2,1,10.0);
        mc_surf_cell_set(g,0,0, 0,0,0);
        mc_surf_cell_set(g,0,1, 4,0,0);
        double *x=g->p;                         // 2 blocks of 3 = the grid itself
        mc_problem *p=mc_problem_create(2,3,x);
        mc_problem_set_const(p,0,1);            // freeze a
        mc_trace_add_dist(p,g,0,1,1.0);
        // weak anchor keeping b on +x
        // (reuse dist only; b will move along the a->b line to distance 10)
        mc_solve_report rep; mc_solve(p,NULL,&rep);
        double dx=x[3]-x[0],dy=x[4]-x[1],dz=x[5]-x[2];
        double L=sqrt(dx*dx+dy*dy+dz*dz);
        CHECK(fabs(L-10.0)<1e-4);
        printf("A. dist optimum: |a-b|=%.5f (want 10) cost %.2g\n", L, rep.final_cost);
        mc_problem_free(p); mc_surf_grid_free(g);
    }

    // ===== B. StraightLoss optimum: 3 points, ends fixed at (0,0,0),(20,0,0),
    // middle starts bent at (10,8,0). Straightness should pull it onto the line
    // (y->0). Add weak dist terms so it doesn't just collapse. =====
    {
        mc_surf_grid *g=mc_surf_grid_create(3,1,10.0);
        mc_surf_cell_set(g,0,0, 0,0,0);
        mc_surf_cell_set(g,0,1, 10,8,0);
        mc_surf_cell_set(g,0,2, 20,0,0);
        double *x=g->p;
        mc_problem *p=mc_problem_create(3,3,x);
        mc_problem_set_const(p,0,1); mc_problem_set_const(p,2,1);
        mc_trace_add_straight(p,g,0,1,2,1.0);
        mc_trace_add_dist(p,g,0,1,0.3);
        mc_trace_add_dist(p,g,1,2,0.3);
        mc_solve(p,NULL,NULL);
        CHECK(fabs(x[4])<0.2);                  // middle y pulled toward 0
        CHECK(fabs(x[3]-10.0)<1.0);             // stays near x=10
        printf("B. straight optimum: mid=(%.3f,%.3f,%.3f)\n", x[3],x[4],x[5]);
        mc_problem_free(p); mc_surf_grid_free(g);
    }

    // ===== C. Relaxation of a perturbed flat 7x7 grid. Truth: a flat z=0 plane
    // on a unit-10 lattice. Perturb interior points with noise in z, fix the
    // border, then relax with Dist+Straight+SDIR. Interior z should shrink back
    // toward ~0 (smooth sheet). =====
    {
        int G=7; double U=10.0;
        mc_surf_grid *g=mc_surf_grid_create(G,G,U);
        // deterministic "noise"
        unsigned seed=12345;
        double init_rms=0; int ni=0;
        for(int r=0;r<G;++r)for(int c=0;c<G;++c){
            double z=0;
            int border = (r==0||c==0||r==G-1||c==G-1);
            if(!border){ seed=seed*1103515245u+12345u; z=(((seed>>16)&0x7fff)/32767.0-0.5)*2.0*6.0; init_rms+=z*z; ni++; }
            mc_surf_cell_set(g,r,c, c*U, r*U, z);
        }
        init_rms=sqrt(init_rms/ni);
        double *x=g->p;
        int NB=G*G;
        mc_problem *p=mc_problem_create(NB,3,x);
        #define BLK(r,c) ((r)*G+(c))
        for(int r=0;r<G;++r)for(int c=0;c<G;++c)
            if(r==0||c==0||r==G-1||c==G-1) mc_problem_set_const(p,BLK(r,c),1);
        // dist along right/down edges; straight along rows & cols; sdir per cell
        for(int r=0;r<G;++r)for(int c=0;c<G;++c){
            if(c+1<G) mc_trace_add_dist(p,g,BLK(r,c),BLK(r,c+1),1.0);
            if(r+1<G) mc_trace_add_dist(p,g,BLK(r,c),BLK(r+1,c),1.0);
            if(c>0&&c+1<G) mc_trace_add_straight(p,g,BLK(r,c-1),BLK(r,c),BLK(r,c+1),0.5);
            if(r>0&&r+1<G) mc_trace_add_straight(p,g,BLK(r-1,c),BLK(r,c),BLK(r+1,c),0.5);
            if(c+1<G&&r+1<G) mc_trace_add_sdir(p,g,BLK(r,c),BLK(r,c+1),BLK(r+1,c),1.0);
        }
        mc_solve_opts o; mc_solve_opts_default(&o); o.max_iter=200;
        mc_solve_report rep; mc_solve(p,&o,&rep);
        double final_rms=0; ni=0;
        for(int r=1;r<G-1;++r)for(int c=1;c<G-1;++c){ double z=x[BLK(r,c)*3+2]; final_rms+=z*z; ni++; }
        final_rms=sqrt(final_rms/ni);
        CHECK(final_rms < init_rms*0.5);        // perturbation at least halved
        CHECK(rep.final_cost < rep.initial_cost);
        printf("C. relax 7x7: z-rms %.3f -> %.3f (cost %.3g -> %.3g, %d iters)\n",
               init_rms, final_rms, rep.initial_cost, rep.final_cost, rep.iterations);
        #undef BLK
        mc_problem_free(p); mc_surf_grid_free(g);
    }

    // ===== D. growth: seed a 3x3 patch of a z=0 plane in a 15x15 grid, grow it
    // (geometry only). The whole grid should fill, lie exactly on z=0, and keep
    // exact unit spacing — geometry-only growth recovers a plane perfectly. =====
    {
        int G=15; double U=10.0;
        mc_surf_grid *g=mc_surf_grid_create(G,G,U);
        int c0=G/2;
        for(int dr=-1;dr<=1;++dr)for(int dc=-1;dc<=1;++dc)
            mc_surf_cell_set(g,c0+dr,c0+dc,(c0+dc)*U,(c0+dr)*U,0.0);
        mc_grow_opts o; mc_grow_opts_default(&o); o.max_gen=40; o.accept_resid=0;
        mc_grow_report rep; CHECK(mc_trace_grow(g,&o,&rep)==0);
        int nv=0; double zmax=0, sperr=0; int sp=0;
        for(int r=0;r<G;++r)for(int c=0;c<G;++c){
            if(!mc_surf_cell_valid(g,r,c)) continue; nv++;
            double *p=g->p+((size_t)r*G+c)*3;
            if(fabs(p[2])>zmax)zmax=fabs(p[2]);
            if(mc_surf_cell_valid(g,r,c+1)){ double*q=g->p+((size_t)r*G+c+1)*3;
                double d=sqrt((p[0]-q[0])*(p[0]-q[0])+(p[1]-q[1])*(p[1]-q[1])+(p[2]-q[2])*(p[2]-q[2]));
                sperr+=fabs(d-U); sp++; } }
        CHECK(nv==G*G);                 // grid fully filled
        CHECK(zmax < 1e-3);             // stayed on the plane
        CHECK(rep.rejected==0);
        CHECK(sp && sperr/sp < 1e-3);   // exact unit spacing
        printf("D. plane growth: filled %d/%d, max|z|=%.2e, mean|spacing-10|=%.2e\n",
               nv, G*G, zmax, sp?sperr/sp:0);
        mc_surf_grid_free(g);
    }

    printf(fails ? "mc_trace_test: %d FAILED\n" : "mc_trace_test: OK\n", fails);
    return fails?1:0;
}
