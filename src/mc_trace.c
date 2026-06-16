/* mc_trace.c — surface grid + geometric cost terms (see mc_trace.h). */
#include "mc_trace.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

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

// ---- growth orchestration ---------------------------------------------------
void mc_grow_opts_default(mc_grow_opts *o){
    o->w_dist=1.0; o->w_straight=0.5; o->w_sdir=1.0;
    o->radius=3; o->max_gen=100000; o->accept_resid=0.5; o->verbose=0;
    o->vol_fn=NULL; o->vol_user=NULL; o->vol_weight=1.0;
}

// single-block volume residual adapter (wraps the caller's vol_fn).
typedef struct { int (*fn)(void*,const double*,double*,double*); void *user; double w; } vol_ctx;
static int vol_eval(void *u, const double *const *pr, double *res, double *const *jac){
    vol_ctx *C=u; const double *x=pr[0];
    double r=0, g[3]={0,0,0};
    if(C->fn(C->user,x,&r,(jac&&jac[0])?g:NULL)!=0) return 1;
    res[0]=C->w*r;
    if(jac&&jac[0]){ jac[0][0]=C->w*g[0]; jac[0][1]=C->w*g[1]; jac[0][2]=C->w*g[2]; }
    return 0;
}

#define CGV(g,r,c) ((r)>=0&&(c)>=0&&(r)<(g)->gh&&(c)<(g)->gw && !isnan((g)->p[((size_t)(r)*(g)->gw+(c))*3]))
#define CP(g,r,c) ((g)->p+((size_t)(r)*(g)->gw+(c))*3)

// Initialize an invalid cell (r,c) from valid neighbors: prefer linear
// extrapolation across a valid pair (c-1,c-2 / c+1,c+2 / rows), else mirror a
// single neighbor by one grid step along the perpendicular tangent. Returns 1
// if it produced an estimate, 0 if no usable neighbor.
static int init_candidate(const mc_surf_grid *g, int r, int c, double out[3]){
    // try linear extrapolation along each of the 4 axes
    const int dirs[4][2]={{0,-1},{0,1},{-1,0},{1,0}};
    double acc[3]={0,0,0}; int n=0;
    for(int k=0;k<4;++k){
        int dr=dirs[k][0], dc=dirs[k][1];
        int r1=r+dr, c1=c+dc, r2=r+2*dr, c2=c+2*dc;
        if(CGV(g,r1,c1) && CGV(g,r2,c2)){
            const double *p1=CP(g,r1,c1), *p2=CP(g,r2,c2);
            for(int i=0;i<3;++i) acc[i]+= 2.0*p1[i]-p2[i];   // p1 + (p1-p2)
            n++;
        }
    }
    if(n){ for(int i=0;i<3;++i) out[i]=acc[i]/n; return 1; }
    // fall back: single valid neighbor + offset by `unit` along a perpendicular
    // grid tangent (estimated from any valid second-order neighbor pair), else
    // just step away from the neighbor by `unit` in a stable axis.
    for(int k=0;k<4;++k){
        int dr=dirs[k][0], dc=dirs[k][1];
        int r1=r+dr, c1=c+dc;
        if(CGV(g,r1,c1)){
            const double *p1=CP(g,r1,c1);
            // tangent along the perpendicular axis if available
            int pr=(dr==0)?1:0, pc=(dc==0)?1:0;          // perpendicular unit
            double tang[3]={0,0,0}; int have=0;
            if(CGV(g,r1+pr,c1+pc)){ const double*q=CP(g,r1+pr,c1+pc); for(int i=0;i<3;++i) tang[i]=q[i]-p1[i]; have=1; }
            else if(CGV(g,r1-pr,c1-pc)){ const double*q=CP(g,r1-pr,c1-pc); for(int i=0;i<3;++i) tang[i]=p1[i]-q[i]; have=1; }
            // outward direction = -(neighbor->cell) i.e. from neighbor toward cell
            // we don't know the 3D outward dir; reuse the in-plane step magnitude
            (void)have;
            for(int i=0;i<3;++i) out[i]=p1[i];            // start at the neighbor
            // nudge along tangent by unit so the new cell isn't coincident
            double tl=sqrt(tang[0]*tang[0]+tang[1]*tang[1]+tang[2]*tang[2]);
            if(tl>1e-9){ for(int i=0;i<3;++i) out[i]+= g->unit*tang[i]/tl; }
            else out[0]+=g->unit;
            return 1;
        }
    }
    return 0;
}

int mc_trace_grow(mc_surf_grid *g, const mc_grow_opts *opts, mc_grow_report *rep){
    if(!g) return -1;
    mc_grow_opts o; if(opts) o=*opts; else mc_grow_opts_default(&o);
    int gw=g->gw, gh=g->gh, NB=gw*gh;
    int added=0, rejected=0, gen=0;

    // scratch: which cells are currently valid (mirrors NaN test, faster)
    unsigned char *valid = malloc((size_t)NB);
    if(!valid) return -1;
    for(int i=0;i<NB;++i) valid[i] = !isnan(g->p[(size_t)i*3]);

    for(gen=0; gen<o.max_gen; ++gen){
        // collect this generation's candidates: invalid cells with >=1 valid
        // 4-neighbor. Snapshot first (don't let this gen's adds seed the same gen).
        int gen_added=0;
        // simple list of candidate (r,c)
        for(int r=0;r<gh;++r)for(int c=0;c<gw;++c){
            if(valid[r*gw+c]) continue;
            int has = (CGV(g,r,c-1)||CGV(g,r,c+1)||CGV(g,r-1,c)||CGV(g,r+1,c));
            if(!has) continue;
            // only grow cells that already had a valid neighbor at gen start:
            // mark via a separate pass would be cleaner; here we accept that a
            // cell filled this gen can seed an orthogonal neighbor same gen,
            // which just speeds fill — deterministic given row-major order.
            double init[3];
            if(!init_candidate(g,r,c,init)){ continue; }
            mc_surf_cell_set(g,r,c,init[0],init[1],init[2]);

            // build a LOCAL problem: free cells = valid cells within `radius`
            // of (r,c) plus the candidate; everything else frozen.
            mc_problem *p=mc_problem_create(NB,3,g->p);
            if(!p){ free(valid); return -1; }
            for(int b=0;b<NB;++b) mc_problem_set_const(p,b,1);   // freeze all
            int R=o.radius;
            for(int rr=r-R; rr<=r+R; ++rr) for(int cc=c-R; cc<=c+R; ++cc){
                if(rr<0||cc<0||rr>=gh||cc>=gw) continue;
                if(rr==r&&cc==c){ mc_problem_set_const(p,rr*gw+cc,0); continue; }
                if(valid[rr*gw+cc]) mc_problem_set_const(p,rr*gw+cc,0);
            }
            // add geometric terms among cells in the (radius+1) window that are
            // valid or the candidate (so residuals touching the new cell exist).
            #define INWIN(rr,cc) ((rr)>=r-R-1&&(rr)<=r+R+1&&(cc)>=c-R-1&&(cc)<=c+R+1&&(rr)>=0&&(cc)>=0&&(rr)<gh&&(cc)<gw)
            #define CELLOK(rr,cc) (((rr)==r&&(cc)==c) || (CGV(g,rr,cc)))
            for(int rr=r-R-1; rr<=r+R+1; ++rr) for(int cc=c-R-1; cc<=c+R+1; ++cc){
                if(!INWIN(rr,cc) || !CELLOK(rr,cc)) continue;
                int b0=rr*gw+cc;
                if(INWIN(rr,cc+1)&&CELLOK(rr,cc+1)) mc_trace_add_dist(p,g,b0,rr*gw+cc+1,o.w_dist);
                if(INWIN(rr+1,cc)&&CELLOK(rr+1,cc)) mc_trace_add_dist(p,g,b0,(rr+1)*gw+cc,o.w_dist);
                if(INWIN(rr,cc-1)&&CELLOK(rr,cc-1)&&INWIN(rr,cc+1)&&CELLOK(rr,cc+1))
                    mc_trace_add_straight(p,g,rr*gw+cc-1,b0,rr*gw+cc+1,o.w_straight);
                if(INWIN(rr-1,cc)&&CELLOK(rr-1,cc)&&INWIN(rr+1,cc)&&CELLOK(rr+1,cc))
                    mc_trace_add_straight(p,g,(rr-1)*gw+cc,b0,(rr+1)*gw+cc,o.w_straight);
                if(INWIN(rr,cc+1)&&CELLOK(rr,cc+1)&&INWIN(rr+1,cc)&&CELLOK(rr+1,cc))
                    mc_trace_add_sdir(p,g,b0,rr*gw+cc+1,(rr+1)*gw+cc,o.w_sdir);
            }
            #undef INWIN
            #undef CELLOK
            // optional volume term on just the new cell
            if(o.vol_fn){
                vol_ctx *vc=arena_alloc(g,sizeof *vc); if(vc){ vc->fn=o.vol_fn; vc->user=o.vol_user; vc->w=o.vol_weight;
                    mc_problem_add(p,vol_eval,vc,1,1,(int[]){r*gw+c},0); }
            }

            mc_solve_opts so; mc_solve_opts_default(&so); so.max_iter=40;
            mc_solve_report sr;
            int rc_solve = mc_solve(p,&so,&sr);
            mc_problem_free(p);

            // accept test. Reject if the solve failed or produced a non-finite
            // point, or — when a per-cell residual gate is set — if the new
            // cell's own DIST residual to its valid neighbors is too large
            // (its 3D spacing strayed too far from `unit`, i.e. the local fit
            // is poor). The volume term (phase 4) tightens this into a real
            // sheet-fit gate.
            double *np = g->p+((size_t)(r*gw+c))*3;
            int finite = isfinite(np[0]) && isfinite(np[1]) && isfinite(np[2]);
            int ok = (rc_solve==0) && finite;
            if(ok && o.accept_resid>0){
                double worst=0; int ne=0; const int d4[4][2]={{0,-1},{0,1},{-1,0},{1,0}};
                for(int k=0;k<4;++k){ int r1=r+d4[k][0], c1=c+d4[k][1];
                    if(!CGV(g,r1,c1)) continue; const double *q=CP(g,r1,c1);
                    double dx=np[0]-q[0],dy=np[1]-q[1],dz=np[2]-q[2];
                    double L=sqrt(dx*dx+dy*dy+dz*dz);
                    double e=fabs(L-g->unit)/g->unit; if(e>worst)worst=e; ne++; }
                if(ne && worst > o.accept_resid) ok=0;
            }
            if(ok){ valid[r*gw+c]=1; added++; gen_added++; }
            else { mc_surf_cell_invalidate(g,r,c); rejected++; }
        }
        if(o.verbose) fprintf(stderr,"gen %d: +%d (total %d, rej %d)\n", gen, gen_added, added, rejected);
        if(gen_added==0) break;   // fringe exhausted / grid full
    }
    free(valid);
    if(rep){ rep->generations=gen; rep->added=added; rep->rejected=rejected; }
    return 0;
}

// ---- volume term: trilinear signed-distance field sampler -------------------
// Returns the trilinearly-interpolated SDF value at (x,y,z) and the analytic
// gradient of that trilinear interpolant w.r.t. (x,y,z). Grid point is (x,y,z),
// the field is indexed (z,y,x). Out of bounds -> 0 residual, 0 gradient.
int mc_trace_sdf_vol(void *user, const double *xyz, double *resid, double *grad){
    const mc_sdf_field *F=user;
    int nx=F->nx, ny=F->ny, nz=F->nz;
    double x=xyz[0], y=xyz[1], z=xyz[2];
    if(x<0||y<0||z<0 || x>nx-1 || y>ny-1 || z>nz-1){
        *resid=0; if(grad){ grad[0]=grad[1]=grad[2]=0; } return 0;
    }
    int x0=(int)x, y0=(int)y, z0=(int)z;
    if(x0>=nx-1) x0=nx-2; if(y0>=ny-1) y0=ny-2; if(z0>=nz-1) z0=nz-2;
    if(x0<0)x0=0; if(y0<0)y0=0; if(z0<0)z0=0;
    double fx=x-x0, fy=y-y0, fz=z-z0;
    const float *S=F->sdf; size_t plane=(size_t)ny*nx;
    #define V(zz,yy,xx) ((double)S[(size_t)(zz)*plane+(size_t)(yy)*nx+(xx)])
    double c000=V(z0,y0,x0),   c001=V(z0,y0,x0+1);
    double c010=V(z0,y0+1,x0), c011=V(z0,y0+1,x0+1);
    double c100=V(z0+1,y0,x0), c101=V(z0+1,y0,x0+1);
    double c110=V(z0+1,y0+1,x0), c111=V(z0+1,y0+1,x0+1);
    #undef V
    // interpolate along x
    double c00=c000+(c001-c000)*fx, c01=c010+(c011-c010)*fx;
    double c10=c100+(c101-c100)*fx, c11=c110+(c111-c110)*fx;
    // along y
    double c0=c00+(c01-c00)*fy, c1=c10+(c11-c10)*fy;
    // along z
    double val=c0+(c1-c0)*fz;
    *resid=val;
    if(grad){
        // d/dz = c1-c0
        grad[2]=c1-c0;
        // d/dy = (c01-c00)*(1-fz) + (c11-c10)*fz
        grad[1]=(c01-c00)*(1.0-fz)+(c11-c10)*fz;
        // d/dx = differences along x propagated through y,z weights
        double d00=c001-c000, d01=c011-c010, d10=c101-c100, d11=c111-c110;
        double dx0=d00+(d01-d00)*fy, dx1=d10+(d11-d10)*fy;
        grad[0]=dx0+(dx1-dx0)*fz;
    }
    return 0;
}

// ---- export to mc_surface (.grid) -------------------------------------------
int mc_trace_to_surface(const mc_surf_grid *g, double depth, mc_surface *out){
    if(!g||!out) return -1;
    memset(out,0,sizeof *out);
    int gw=g->gw, gh=g->gh; size_t n=(size_t)gw*gh;
    out->grid=malloc(n*3*sizeof(float));
    out->depth=malloc(n*sizeof(float));
    if(!out->grid||!out->depth){ free(out->grid); free(out->depth); memset(out,0,sizeof *out); return -1; }
    out->gw=gw; out->gh=gh; out->mean_depth=(float)depth;
    for(size_t i=0;i<n;++i){
        const double *p=g->p+i*3;
        if(isnan(p[0])){ out->grid[i*3]=-1; out->grid[i*3+1]=-1; out->grid[i*3+2]=-1; out->depth[i]=0; }
        else { out->grid[i*3]=(float)p[2]; out->grid[i*3+1]=(float)p[1]; out->grid[i*3+2]=(float)p[0]; // (z,y,x)
               out->depth[i]=(float)depth; }
    }
    return 0;
}

// ---- auto-seed from an SDF --------------------------------------------------
// project a point onto the SDF zero set by a few gradient-descent steps:
// p <- p - sdf(p) * grad/|grad|^2  (Newton on the linearized SDF). Returns the
// converged point + its unit gradient (surface normal) in n[3].
static int sdf_project(const mc_sdf_field *F, double p[3], double nrm[3]){
    for(int it=0;it<40;++it){
        double s, gr[3];
        if(mc_trace_sdf_vol((void*)F,p,&s,gr)!=0) return -1;
        double gl=gr[0]*gr[0]+gr[1]*gr[1]+gr[2]*gr[2];
        if(gl<1e-12) return -1;
        for(int i=0;i<3;++i) p[i]-= s*gr[i]/gl;
        if(fabs(s)<1e-3){ double g=sqrt(gl); nrm[0]=gr[0]/g; nrm[1]=gr[1]/g; nrm[2]=gr[2]/g; return 0; }
    }
    // last gradient as normal even if not fully converged
    double s,gr[3]; if(mc_trace_sdf_vol((void*)F,p,&s,gr)!=0) return -1;
    double g=sqrt(gr[0]*gr[0]+gr[1]*gr[1]+gr[2]*gr[2]); if(g<1e-9) return -1;
    nrm[0]=gr[0]/g; nrm[1]=gr[1]/g; nrm[2]=gr[2]/g;
    return fabs(s)<1.0 ? 0 : -1;
}
int mc_trace_seed_from_sdf(mc_surf_grid *g, const mc_sdf_field *F,
                           const double start[3], int patch){
    if(!g||!F) return -1;
    if(patch<1) patch=3;
    double p[3];
    if(start && start[0]>=0 && start[1]>=0 && start[2]>=0){ p[0]=start[0]; p[1]=start[1]; p[2]=start[2]; }
    else { p[0]=F->nx/2.0; p[1]=F->ny/2.0; p[2]=F->nz/2.0; }
    double nrm[3];
    if(sdf_project(F,p,nrm)!=0) return -1;        // land on the sheet
    // build two orthonormal in-plane tangents perpendicular to nrm
    double a[3]={1,0,0}; if(fabs(nrm[0])>0.9){ a[0]=0; a[1]=1; }
    double u[3]={ a[1]*nrm[2]-a[2]*nrm[1], a[2]*nrm[0]-a[0]*nrm[2], a[0]*nrm[1]-a[1]*nrm[0] };
    double ul=sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]); for(int i=0;i<3;++i) u[i]/=ul;
    double v[3]={ nrm[1]*u[2]-nrm[2]*u[1], nrm[2]*u[0]-nrm[0]*u[2], nrm[0]*u[1]-nrm[1]*u[0] };
    // lay the patch centered at the middle grid cell, projected back onto the sheet
    int cr=g->gh/2, cc=g->gw/2, half=patch/2;
    for(int dr=-half; dr<=half; ++dr) for(int dc=-half; dc<=half; ++dc){
        int r=cr+dr, c=cc+dc; if(r<0||c<0||r>=g->gh||c>=g->gw) continue;
        double q[3]={ p[0]+g->unit*(dc*u[0]+dr*v[0]),
                      p[1]+g->unit*(dc*u[1]+dr*v[1]),
                      p[2]+g->unit*(dc*u[2]+dr*v[2]) };
        double qn[3]; sdf_project(F,q,qn);        // snap each seed cell to the sheet
        mc_surf_cell_set(g,r,c,q[0],q[1],q[2]);
    }
    return 0;
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
