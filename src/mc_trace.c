/* mc_trace.c — surface grid + geometric cost terms (see mc_trace.h). */
#include "mc_trace.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---- grid container; also owns the small per-term context allocations -------
typedef struct { void **ptr; int n, cap; } ctx_arena;
struct mc_surf_grid_priv { ctx_arena arena; };

// We tuck the arena into a struct that extends mc_surf_grid's allocation.
typedef struct { mc_surf_grid g; ctx_arena arena; } grid_full;

mc_surf_grid *mc_surf_grid_create(int gw, int gh, double unit){
    if(gw<1||gh<1) return NULL;
    grid_full *f = calloc(1,sizeof *f);
    if(!f) return NULL;
    f->g.gw=gw; f->g.gh=gh; f->g.unit=(unit>0?unit:1.0);
    f->g.p = malloc((size_t)gw*gh*3*sizeof(double));
    if(!f->g.p){ free(f); return NULL; }
    double nan = NAN;
    for(size_t i=0;i<(size_t)gw*gh*3;++i) f->g.p[i]=nan;   // all invalid
    return &f->g;
}
void mc_surf_grid_free(mc_surf_grid *g){
    if(!g) return;
    grid_full *f=(grid_full*)g;
    for(int i=0;i<f->arena.n;++i) free(f->arena.ptr[i]);
    free(f->arena.ptr); free(g->p); free(f);
}
static void *arena_alloc(mc_surf_grid *g, size_t sz){
    grid_full *f=(grid_full*)g;
    void *m=malloc(sz); if(!m) return NULL;
    if(f->arena.n>=f->arena.cap){ int nc=f->arena.cap?f->arena.cap*2:16;
        void **np=realloc(f->arena.ptr,(size_t)nc*sizeof(void*)); if(!np){ free(m); return NULL; }
        f->arena.ptr=np; f->arena.cap=nc; }
    f->arena.ptr[f->arena.n++]=m;
    return m;
}

int mc_surf_cell_valid(const mc_surf_grid *g, int r, int c){
    if(r<0||c<0||r>=g->gh||c>=g->gw) return 0;
    return !isnan(g->p[((size_t)r*g->gw+c)*3]);
}
void mc_surf_cell_set(mc_surf_grid *g, int r, int c, double x, double y, double z){
    double *p=g->p+((size_t)r*g->gw+c)*3; p[0]=x; p[1]=y; p[2]=z;
}
void mc_surf_cell_invalidate(mc_surf_grid *g, int r, int c){
    g->p[((size_t)r*g->gw+c)*3]=NAN;
}

// ---- DistLoss ----------------------------------------------------------------
typedef struct { double unit, w; } dist_ctx;
static int dist_eval(void *u, const double *const *pr, double *res, double *const *jac){
    dist_ctx *C=u; const double *a=pr[0], *b=pr[1];
    double d[3]={a[0]-b[0],a[1]-b[1],a[2]-b[2]};
    double L=sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
    if(L < 1e-9){ // degenerate: pull apart toward unit, jac ~ 0
        res[0]=C->w*( -1.0 ); // unit/L huge -> clamp; treat as large positive
        res[0]=C->w*100.0;
        if(jac){ if(jac[0]) for(int i=0;i<3;++i) jac[0][i]=0; if(jac[1]) for(int i=0;i<3;++i) jac[1][i]=0; }
        return 0;
    }
    double dLda[3]={d[0]/L,d[1]/L,d[2]/L};   // d|a-b|/da ; d/db = -this
    if(L >= C->unit){
        res[0]=C->w*(L/C->unit - 1.0);
        double s=C->w/C->unit;               // dr/dL
        if(jac){ if(jac[0]) for(int i=0;i<3;++i) jac[0][i]= s*dLda[i];
                 if(jac[1]) for(int i=0;i<3;++i) jac[1][i]=-s*dLda[i]; }
    } else {
        res[0]=C->w*(C->unit/L - 1.0);
        double s=-C->w*C->unit/(L*L);        // dr/dL
        if(jac){ if(jac[0]) for(int i=0;i<3;++i) jac[0][i]= s*dLda[i];
                 if(jac[1]) for(int i=0;i<3;++i) jac[1][i]=-s*dLda[i]; }
    }
    return 0;
}
int mc_trace_add_dist(mc_problem *prob, mc_surf_grid *g, int bA, int bB, double w){
    dist_ctx *C=arena_alloc(g,sizeof *C); if(!C) return -1;
    C->unit=g->unit; C->w=(w>0?w:1.0);
    return mc_problem_add(prob,dist_eval,C,1,2,(int[]){bA,bB},0);
}

// ---- StraightLoss ------------------------------------------------------------
// r = w*(1-cos), cos = (d1.d2)/(|d1||d2|), d1=b-a, d2=c-b. Plus a sharp extra
// term when the bend exceeds 30deg (cos < cos30), matching VC3D. Analytic jac.
typedef struct { double w; } straight_ctx;
static const double COS30 = 0.86602540378443864676;
static int straight_eval(void *u, const double *const *pr, double *res, double *const *jac){
    straight_ctx *C=u; const double *a=pr[0], *b=pr[1], *c=pr[2];
    double d1[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]};
    double d2[3]={c[0]-b[0],c[1]-b[1],c[2]-b[2]};
    double n1=sqrt(d1[0]*d1[0]+d1[1]*d1[1]+d1[2]*d1[2]);
    double n2=sqrt(d2[0]*d2[0]+d2[1]*d2[1]+d2[2]*d2[2]);
    if(n1<1e-9||n2<1e-9){ res[0]=0; if(jac){ for(int k=0;k<3;++k) if(jac[k]) for(int i=0;i<3;++i) jac[k][i]=0; } return 0; }
    double dot=d1[0]*d2[0]+d1[1]*d2[1]+d1[2]*d2[2];
    double inv=1.0/(n1*n2);
    double cosv=dot*inv;
    if(cosv>1.0)cosv=1.0; if(cosv<-1.0)cosv=-1.0;
    // base residual
    double extra_k = (cosv < COS30) ? (8.0*C->w) : 0.0;   // sharp penalty weight
    double pen = COS30 - cosv;
    res[0] = C->w*(1.0-cosv) + (extra_k>0 ? extra_k*pen*pen : 0.0);

    if(jac){
        // dcos/dd1 = d2/(n1 n2) - dot*d1/(n1^3 n2) ; dcos/dd2 symmetric.
        double dcos_dd1[3], dcos_dd2[3];
        for(int i=0;i<3;++i){
            dcos_dd1[i]= d2[i]*inv - dot*d1[i]/(n1*n1*n1*n2);
            dcos_dd2[i]= d1[i]*inv - dot*d2[i]/(n1*n2*n2*n2);
        }
        // dr/dcos = -w  (+ extra: d/dcos[ k*(COS30-cos)^2 ] = -2k*(COS30-cos))
        double dr_dcos = -C->w + (extra_k>0 ? -2.0*extra_k*pen : 0.0);
        // chain to points: d1 = b-a -> dd1/da=-I, dd1/db=+I ; d2=c-b -> dd2/db=-I, dd2/dc=+I
        if(jac[0]) for(int i=0;i<3;++i) jac[0][i]= dr_dcos*(-dcos_dd1[i]);              // a
        if(jac[1]) for(int i=0;i<3;++i) jac[1][i]= dr_dcos*( dcos_dd1[i]-dcos_dd2[i]);  // b
        if(jac[2]) for(int i=0;i<3;++i) jac[2][i]= dr_dcos*( dcos_dd2[i]);              // c
    }
    return 0;
}
int mc_trace_add_straight(mc_problem *prob, mc_surf_grid *g, int bA,int bB,int bC, double w){
    straight_ctx *C=arena_alloc(g,sizeof *C); if(!C) return -1;
    C->w=(w>0?w:0.2);
    return mc_problem_add(prob,straight_eval,C,1,3,(int[]){bA,bB,bC},0);
}

// ---- SymmetricDirichlet ------------------------------------------------------
// frame: eu=(pu-p)/unit, ev=(pv-p)/unit. First fundamental form
//   G = [[a, b],[b, c]] with a=eu.eu, b=eu.ev, c=ev.ev.
// Symmetric Dirichlet energy E = tr(G) + tr(G^-1) = (a+c) + (a+c)/det,
//   det = a*c - b*b. Optimum E=4 at G=I (orthonormal unit frame == conformal
//   isometric). residual r = w*(E-4). Analytic jacobian via dE/d{a,b,c} and
//   d{a,b,c}/d points.
typedef struct { double unit, w; } sdir_ctx;
static int sdir_eval(void *u, const double *const *pr, double *res, double *const *jac){
    sdir_ctx *C=u; const double *p=pr[0], *pu=pr[1], *pv=pr[2];
    double iu=1.0/C->unit;
    double eu[3]={(pu[0]-p[0])*iu,(pu[1]-p[1])*iu,(pu[2]-p[2])*iu};
    double ev[3]={(pv[0]-p[0])*iu,(pv[1]-p[1])*iu,(pv[2]-p[2])*iu};
    double a=eu[0]*eu[0]+eu[1]*eu[1]+eu[2]*eu[2];
    double b=eu[0]*ev[0]+eu[1]*ev[1]+eu[2]*ev[2];
    double c=ev[0]*ev[0]+ev[1]*ev[1]+ev[2]*ev[2];
    double det=a*c-b*b;
    double eps=1e-9; double dets = det>eps?det:eps;     // guard near-degenerate
    double tr=a+c;
    double E = tr + tr/dets;
    res[0]=C->w*(E-4.0);
    if(jac){
        // dE/da = 1 + (dets - tr*c_or... ) ... compute via quotient rule on tr/det.
        // E = tr + tr*det^-1. dE/dX = dtr/dX + (dtr/dX)/det - tr*det^-2*ddet/dX.
        // tr=a+c: dtr/da=1, dtr/db=0, dtr/dc=1.
        // det=a*c-b^2: ddet/da=c, ddet/db=-2b, ddet/dc=a.
        double inv=1.0/dets, inv2=inv*inv;
        double dE_da = 1.0 + 1.0*inv - tr*inv2*c;
        double dE_db = 0.0 + 0.0*inv - tr*inv2*(-2.0*b);
        double dE_dc = 1.0 + 1.0*inv - tr*inv2*a;
        if(det<=eps){ /* clamped region: det frozen -> drop its derivative */
            dE_da = 1.0 + 1.0*inv;
            dE_db = 0.0;
            dE_dc = 1.0 + 1.0*inv;
        }
        // d{a,b,c}/d eu,ev :
        //  da/deu = 2 eu ; db/deu = ev ; db/dev = eu ; dc/dev = 2 ev
        // d eu/d p = -iu*I, d eu/d pu = +iu*I ; d ev/d p = -iu*I, d ev/d pv = +iu*I
        for(int i=0;i<3;++i){
            double da_deu = 2.0*eu[i];
            double db_deu = ev[i];
            double db_dev = eu[i];
            double dc_dev = 2.0*ev[i];
            double dE_deu = dE_da*da_deu + dE_db*db_deu;             // via a and b
            double dE_dev = dE_db*db_dev + dE_dc*dc_dev;             // via b and c
            // chain through unit scaling and point dependence
            double dE_dp  = (-iu)*dE_deu + (-iu)*dE_dev;
            double dE_dpu = ( iu)*dE_deu;
            double dE_dpv = ( iu)*dE_dev;
            if(jac[0]) jac[0][i]=C->w*dE_dp;
            if(jac[1]) jac[1][i]=C->w*dE_dpu;
            if(jac[2]) jac[2][i]=C->w*dE_dpv;
        }
    }
    return 0;
}
int mc_trace_add_sdir(mc_problem *prob, mc_surf_grid *g, int bP,int bU,int bV, double w){
    sdir_ctx *C=arena_alloc(g,sizeof *C); if(!C) return -1;
    C->unit=g->unit; C->w=(w>0?w:1.0);
    // Cauchy-robustify (VC3D uses CauchyLoss(1.0) on SDIR) to tolerate the
    // occasional badly-stretched cell during growth.
    return mc_problem_add(prob,sdir_eval,C,1,3,(int[]){bP,bU,bV},1.0);
}

// ---- test hooks (mc_trace_test): expose the residual evals + a ctx builder so
// the jacobians can be checked directly against finite differences. -----------
int mc_trace__dist_eval(void *u,const double*const*pr,double*r,double*const*j){ return dist_eval(u,pr,r,j); }
int mc_trace__straight_eval(void *u,const double*const*pr,double*r,double*const*j){ return straight_eval(u,pr,r,j); }
int mc_trace__sdir_eval(void *u,const double*const*pr,double*r,double*const*j){ return sdir_eval(u,pr,r,j); }
void *mc_trace__make_ctx(int which, double unit, double w){
    if(which==0){ dist_ctx *c=malloc(sizeof *c); c->unit=unit; c->w=w; return c; }
    if(which==1){ straight_ctx *c=malloc(sizeof *c); c->w=w; return c; }
    sdir_ctx *c=malloc(sizeof *c); c->unit=unit; c->w=w; return c;
}
