// mc_solve_test — verify the block-sparse LM solver against problems with known
// closed-form answers: a linear LS system, point-to-target pulls, a frozen
// block, a chain that should collapse to equal spacing, and Cauchy robustness.
#include "mc_solve.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int fails=0;
#define CHECK(x) do{ if(!(x)){ fails++; fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);} }while(0)
#define CLOSE(a,b,t) do{ double _d=fabs((a)-(b)); if(_d>(t)){ fails++; fprintf(stderr,"FAIL %s:%d |%.6g-%.6g|=%.3g > %.3g\n",__FILE__,__LINE__,(double)(a),(double)(b),_d,(double)(t)); } }while(0)

// ---- residual: block - target (3D), jac = I. Pulls a point to a target. ----
typedef struct { double t[3]; double w; } target_t;
static int res_target(void *u, const double *const *params, double *r, double *const *jac){
    target_t *T=u; const double *x=params[0];
    for(int i=0;i<3;++i) r[i]=T->w*(x[i]-T->t[i]);
    if(jac && jac[0]){ for(int i=0;i<3;++i) for(int j=0;j<3;++j) jac[0][i*3+j]=(i==j)?T->w:0.0; }
    return 0;
}

// ---- residual: ||a-b|| - d0  (1 residual over 2 blocks). Analytic jac. ----
typedef struct { double d0, w; } dist_t;
static int res_dist(void *u, const double *const *params, double *r, double *const *jac){
    dist_t *D=u; const double *a=params[0], *b=params[1];
    double d[3]={a[0]-b[0],a[1]-b[1],a[2]-b[2]};
    double L=sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
    r[0]=D->w*(L-D->d0);
    if(jac){
        double inv = L>1e-12 ? 1.0/L : 0.0;
        if(jac[0]) for(int j=0;j<3;++j) jac[0][j]= D->w*d[j]*inv;   // d/da
        if(jac[1]) for(int j=0;j<3;++j) jac[1][j]=-D->w*d[j]*inv;   // d/db
    }
    return 0;
}

int main(void){
    // ===== 1. single point pulled to a target: x -> target exactly =====
    {
        double x[3]={0,0,0};
        mc_problem *p=mc_problem_create(1,3,x);
        target_t T={{5,-3,2},1.0};
        CHECK(mc_problem_add(p,res_target,&T,3,1,(int[]){0},0)==0);
        mc_solve_report rep; CHECK(mc_solve(p,NULL,&rep)==0);
        CLOSE(x[0],5,1e-7); CLOSE(x[1],-3,1e-7); CLOSE(x[2],2,1e-7);
        CHECK(rep.final_cost < 1e-12);
        mc_problem_free(p);
        printf("1. point->target: (%.4f,%.4f,%.4f) cost %.2g\n",x[0],x[1],x[2],rep.final_cost);
    }

    // ===== 2. two targets pulling one point: lands at the weighted mean =====
    // min w1^2|x-t1|^2 + w2^2|x-t2|^2  -> x = (w1^2 t1 + w2^2 t2)/(w1^2+w2^2)
    {
        double x[3]={0,0,0};
        mc_problem *p=mc_problem_create(1,3,x);
        target_t A={{0,0,0},1.0}, Bt={{10,10,10},2.0};   // weights 1 and 2 -> 1:4
        mc_problem_add(p,res_target,&A,3,1,(int[]){0},0);
        mc_problem_add(p,res_target,&Bt,3,1,(int[]){0},0);
        mc_solve(p,NULL,NULL);
        double exp = (1.0*0 + 4.0*10)/(1.0+4.0);   // = 8
        CLOSE(x[0],exp,1e-6); CLOSE(x[1],exp,1e-6); CLOSE(x[2],exp,1e-6);
        mc_problem_free(p);
        printf("2. weighted mean: x=%.4f (expect %.4f)\n",x[0],exp);
    }

    // ===== 3. frozen block: a fixed at origin, b free, want |a-b|=10 along +x
    // b starts near (5,0,0); only the distance residual + a weak target keeping
    // b on the x-axis. Frozen a must not move. =====
    {
        double x[6]={0,0,0,  5,0,0};   // block0=a, block1=b
        mc_problem *p=mc_problem_create(2,3,x);
        mc_problem_set_const(p,0,1);            // freeze a
        dist_t D={10.0,1.0};
        mc_problem_add(p,res_dist,&D,1,2,(int[]){0,1},0);
        // weak pull keeping b near the +x axis (target far out on x)
        target_t T={{10,0,0},0.05};
        mc_problem_add(p,res_target,&T,3,1,(int[]){1},0);
        mc_solve(p,NULL,NULL);
        // a frozen
        CLOSE(x[0],0,1e-12); CLOSE(x[1],0,1e-12); CLOSE(x[2],0,1e-12);
        // |a-b| ~ 10, b on +x
        double L=sqrt(x[3]*x[3]+x[4]*x[4]+x[5]*x[5]);
        CLOSE(L,10.0,1e-3);
        CHECK(x[3]>9.0);                         // mostly along +x
        mc_problem_free(p);
        printf("3. frozen a + dist: a=(0,0,0) b=(%.3f,%.3f,%.3f) |b|=%.4f\n",x[3],x[4],x[5],L);
    }

    // ===== 4. chain of 5 points on a line, endpoints frozen at 0 and 40, inner
    // 3 free, distance residual d0=10 between consecutive -> equal 10 spacing
    // (total 40 over 4 gaps = 10 each, exactly satisfiable). Weak position
    // anchors on the inner points pin the otherwise-free slide along the line
    // (exactly what the real tracer's straightness/metric terms do). =====
    {
        double x[15];
        for(int i=0;i<5;++i){ x[i*3]=i*7.0; x[i*3+1]=(i%2)*2.0; x[i*3+2]=0; } // perturbed init
        x[0]=0;x[1]=0;x[2]=0;  x[12]=40;x[13]=0;x[14]=0;
        mc_problem *p=mc_problem_create(5,3,x);
        mc_problem_set_const(p,0,1); mc_problem_set_const(p,4,1);
        dist_t D={10.0,1.0};
        for(int i=0;i<4;++i) mc_problem_add(p,res_dist,&D,1,2,(int[]){i,i+1},0);
        target_t anc[3]={{{10,0,0},0.01},{{20,0,0},0.01},{{30,0,0},0.01}};
        for(int i=0;i<3;++i) mc_problem_add(p,res_target,&anc[i],3,1,(int[]){i+1},0);
        mc_solve_report rep; mc_solve(p,NULL,&rep);
        // each consecutive gap should be ~10 and the inner points near the anchors
        for(int i=0;i<4;++i){
            double dx=x[(i+1)*3]-x[i*3], dy=x[(i+1)*3+1]-x[i*3+1], dz=x[(i+1)*3+2]-x[i*3+2];
            double L=sqrt(dx*dx+dy*dy+dz*dz);
            CLOSE(L,10.0,1e-3);
        }
        CHECK(rep.iterations < 30);   // well-conditioned -> fast LM convergence
        mc_problem_free(p);
        printf("4. chain spacing: x=%.2f,%.2f,%.2f,%.2f,%.2f cost %.3g iters %d\n",
               x[0],x[3],x[6],x[9],x[12],rep.final_cost,rep.iterations);
    }

    // ===== 5. Cauchy robustness: a point pulled by 4 inliers at ~origin and 1
    // gross outlier far away. With squared loss the mean is dragged toward the
    // outlier; with Cauchy loss it stays near the inlier cluster. =====
    {
        target_t in[4]={{{0,0,0},1},{{1,0,0},1},{{0,1,0},1},{{1,1,0},1}};
        target_t out={{1000,1000,0},1};
        // squared-loss solve
        double xs[3]={0,0,0}; mc_problem *ps=mc_problem_create(1,3,xs);
        for(int i=0;i<4;++i) mc_problem_add(ps,res_target,&in[i],3,1,(int[]){0},0);
        mc_problem_add(ps,res_target,&out,3,1,(int[]){0},0);
        mc_solve(ps,NULL,NULL); mc_problem_free(ps);
        // robust (Cauchy scale 2) solve
        double xr[3]={0,0,0}; mc_problem *pr=mc_problem_create(1,3,xr);
        for(int i=0;i<4;++i) mc_problem_add(pr,res_target,&in[i],3,1,(int[]){0},2.0);
        mc_problem_add(pr,res_target,&out,3,1,(int[]){0},2.0);
        mc_solve(pr,NULL,NULL); mc_problem_free(pr);
        // squared loss should be dragged far toward the outlier; robust stays near 0.5
        CHECK(xs[0] > 100.0);                 // squared loss heavily pulled
        CHECK(xr[0] < 5.0);                   // robust resists the outlier
        printf("5. robustness: squared x=%.1f  cauchy x=%.3f\n", xs[0], xr[0]);
    }

    // ===== 6. analytic-jacobian check vs central finite differences. The whole
    // tracer relies on hand-derived jacobians; verify res_dist's analytic jac
    // matches FD at a few random-ish configs. =====
    {
        dist_t D={7.0,1.3};
        double cfgs[3][6]={
            {1,2,3,  4,6,8},
            {0,0,0,  3,0,0},
            {-5,2,9, 1,-4,2},
        };
        double maxerr=0;
        for(int c=0;c<3;++c){
            double *a=cfgs[c], *b=cfgs[c]+3;
            const double *pp[2]={a,b};
            double jana0[3], jana1[3], r0[1];
            double *jp[2]={jana0,jana1};
            res_dist(&D,pp,r0,jp);
            // FD each of the 6 params
            for(int blk=0;blk<2;++blk) for(int d=0;d<3;++d){
                double save=cfgs[c][blk*3+d], h=1e-6;
                double rp[1], rm[1];
                cfgs[c][blk*3+d]=save+h; res_dist(&D,pp,rp,NULL);
                cfgs[c][blk*3+d]=save-h; res_dist(&D,pp,rm,NULL);
                cfgs[c][blk*3+d]=save;
                double fd=(rp[0]-rm[0])/(2*h);
                double ana=(blk==0?jana0:jana1)[d];
                double err=fabs(fd-ana); if(err>maxerr)maxerr=err;
            }
        }
        CHECK(maxerr < 1e-5);
        printf("6. analytic jac vs FD: max err %.2e\n", maxerr);
    }

    printf(fails ? "mc_solve_test: %d FAILED\n" : "mc_solve_test: OK\n", fails);
    return fails?1:0;
}
