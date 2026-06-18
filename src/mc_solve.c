/* mc_solve.c — block-sparse Levenberg-Marquardt least squares (see mc_solve.h). */
#include "mc_solve.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

typedef struct {
    mc_residual_fn fn;
    void  *user;
    int    rdim;
    int    nblk;
    int   *blocks;       // indices into the block array (length nblk)
    double loss_scale;   // >0 -> Cauchy with this scale; else squared loss
} residual_t;

struct mc_problem {
    int         nblocks, bdim;
    double     *x;                 // caller-owned, nblocks*bdim
    unsigned char *is_const;
    residual_t *res;
    int         nres, rescap;
    int         max_rdim, max_nblk;
};

void mc_solve_opts_default(mc_solve_opts *o){
    o->max_iter   = 50;
    o->init_lambda= 1e-3;
    o->func_tol   = 1e-6;
    o->grad_tol   = 1e-8;
    o->param_tol  = 1e-8;
    o->verbose    = 0;
}

mc_problem *mc_problem_create(int nblocks, int bdim, double *x){
    if(nblocks<1 || bdim<1 || !x) return NULL;
    mc_problem *p = calloc(1,sizeof *p);
    if(!p) return NULL;
    p->nblocks=nblocks; p->bdim=bdim; p->x=x;
    p->is_const = calloc((size_t)nblocks,1);
    p->rescap = 64;
    p->res = malloc((size_t)p->rescap*sizeof *p->res);
    if(!p->is_const || !p->res){ free(p->is_const); free(p->res); free(p); return NULL; }
    return p;
}

void mc_problem_free(mc_problem *p){
    if(!p) return;
    for(int i=0;i<p->nres;++i) free(p->res[i].blocks);
    free(p->res); free(p->is_const); free(p);
}

void mc_problem_set_const(mc_problem *p, int block, int is_const){
    if(p && block>=0 && block<p->nblocks) p->is_const[block] = is_const?1:0;
}

int mc_problem_add(mc_problem *p, mc_residual_fn fn, void *user,
                   int rdim, int nblk, const int *blocks, double loss_scale){
    if(!p||!fn||rdim<1||nblk<1||!blocks) return -1;
    for(int i=0;i<nblk;++i) if(blocks[i]<0||blocks[i]>=p->nblocks) return -1;
    if(p->nres>=p->rescap){
        int nc=p->rescap*2; residual_t *nr=realloc(p->res,(size_t)nc*sizeof *nr);
        if(!nr) return -1; p->res=nr; p->rescap=nc;
    }
    residual_t *r=&p->res[p->nres];
    r->fn=fn; r->user=user; r->rdim=rdim; r->nblk=nblk; r->loss_scale=loss_scale;
    r->blocks=malloc((size_t)nblk*sizeof(int));
    if(!r->blocks) return -1;
    memcpy(r->blocks,blocks,(size_t)nblk*sizeof(int));
    p->nres++;
    if(rdim>p->max_rdim) p->max_rdim=rdim;
    if(nblk>p->max_nblk) p->max_nblk=nblk;
    return 0;
}

// ---- dense symmetric LDL^T solve of A dx = b (A SPD-ish). 0 ok, <0 singular --
static int ldlt_solve(double *A, int n, const double *b, double *out){
    for(int j=0;j<n;++j){
        double d = A[(size_t)j*n+j];
        for(int k=0;k<j;++k) d -= A[(size_t)j*n+k]*A[(size_t)j*n+k]*A[(size_t)k*n+k];
        if(fabs(d) < 1e-300) return -1;
        A[(size_t)j*n+j]=d;
        for(int i=j+1;i<n;++i){
            double s=A[(size_t)i*n+j];
            for(int k=0;k<j;++k) s -= A[(size_t)i*n+k]*A[(size_t)j*n+k]*A[(size_t)k*n+k];
            A[(size_t)i*n+j]=s/d;
        }
    }
    for(int i=0;i<n;++i){ double s=b[i]; for(int k=0;k<i;++k) s-=A[(size_t)i*n+k]*out[k]; out[i]=s; }
    for(int i=0;i<n;++i) out[i]/=A[(size_t)i*n+i];
    for(int i=n-1;i>=0;--i){ double s=out[i]; for(int k=i+1;k<n;++k) s-=A[(size_t)k*n+i]*out[k]; out[i]=s; }
    return 0;
}

// Cauchy robust weight: rho(s)=c^2 log(1+s/c^2), rho'(s)=1/(1+s/c^2). Rows of the
// residual+jacobian get scaled by sqrt(rho'); the standard IRLS-style softening.
static double cauchy_weight(double s, double c){ double c2=c*c; return 1.0/(1.0 + s/c2); }

// ---- evaluate cost over `x`; if build, accumulate H (J^T J) and g (-J^T r) ----
// returns 0.5*sum w*r^2, or -1 (via *failed) on residual failure.
typedef struct {
    mc_problem *p; int N, B; const int *col;
    double *H, *g;                         // reduced normal equations (size N)
    double *rbuf, *jstore; double **jbuf; const double **pbuf;
} evalctx;

static double eval_cost(evalctx *e, const double *x, int build, int *failed){
    mc_problem *p=e->p; int N=e->N, B=e->B; const int *col=e->col;
    *failed=0; double cost=0;
    if(build){ memset(e->H,0,(size_t)N*N*sizeof(double)); memset(e->g,0,(size_t)N*sizeof(double)); }
    for(int ri=0; ri<p->nres; ++ri){
        residual_t *r=&p->res[ri]; int rd=r->rdim, nb=r->nblk;
        for(int k=0;k<nb;++k){
            e->pbuf[k]=x+(size_t)r->blocks[k]*B;
            e->jbuf[k] = (build && !p->is_const[r->blocks[k]]) ? e->jstore+(size_t)k*rd*B : NULL;
        }
        if(r->fn(r->user,e->pbuf,e->rbuf,build?e->jbuf:NULL)!=0){ *failed=1; return 0; }
        double w=1.0;
        if(r->loss_scale>0){ double s=0; for(int t=0;t<rd;++t) s+=e->rbuf[t]*e->rbuf[t]; w=cauchy_weight(s,r->loss_scale); }
        for(int t=0;t<rd;++t) cost += 0.5*w*e->rbuf[t]*e->rbuf[t];
        if(!build) continue;
        double sw=sqrt(w);
        // g -= Jw^T rw ; H += Jw^T Jw  (rows scaled by sw)
        for(int ka=0;ka<nb;++ka){ int ba=r->blocks[ka]; if(p->is_const[ba]) continue; int ca=col[ba];
            const double *Ja=e->jstore+(size_t)ka*rd*B;
            for(int da=0;da<B;++da){ double gv=0;
                for(int t=0;t<rd;++t) gv += (sw*Ja[(size_t)t*B+da])*(sw*e->rbuf[t]);
                e->g[ca+da] -= gv;
            }
            for(int kb=0;kb<nb;++kb){ int bb=r->blocks[kb]; if(p->is_const[bb]) continue; int cb=col[bb];
                const double *Jb=e->jstore+(size_t)kb*rd*B;
                for(int da=0;da<B;++da) for(int db=0;db<B;++db){ double acc=0;
                    for(int t=0;t<rd;++t) acc += (sw*Ja[(size_t)t*B+da])*(sw*Jb[(size_t)t*B+db]);
                    e->H[(size_t)(ca+da)*N+(cb+db)] += acc;
                }
            }
        }
    }
    return cost;
}

int mc_solve(mc_problem *p, const mc_solve_opts *opts, mc_solve_report *rep){
    if(!p) return -1;
    mc_solve_opts o; if(opts) o=*opts; else mc_solve_opts_default(&o);
    int B=p->bdim;

    int *col = malloc((size_t)p->nblocks*sizeof(int));
    if(!col) return -1;
    int nfree=0;
    for(int b=0;b<p->nblocks;++b){ if(p->is_const[b]) col[b]=-1; else { col[b]=nfree*B; nfree++; } }
    int N = nfree*B;
    if(N==0){ free(col); if(rep){ rep->initial_cost=rep->final_cost=0; rep->iterations=0; rep->success=1; } return 0; }
    // guard the dense-Hessian (N*N doubles) and jstore (max_nblk*max_rdim*B) sizes
    // against integer overflow on a pathological problem size.
    if(N < 0 || (size_t)N > (size_t)1<<20){ free(col); return -1; }
    { size_t mn=(size_t)(p->max_nblk?p->max_nblk:1), mr=(size_t)(p->max_rdim?p->max_rdim:1);
      if(mn > (SIZE_MAX/sizeof(double))/(mr?mr*(size_t)B:1)){ free(col); return -1; } }

    double *H   = malloc((size_t)N*N*sizeof(double));
    double *Hd  = malloc((size_t)N*N*sizeof(double));
    double *g   = malloc((size_t)N*sizeof(double));
    double *dx  = malloc((size_t)N*sizeof(double));
    double *xnew= malloc((size_t)p->nblocks*B*sizeof(double));
    double *rbuf = malloc((size_t)(p->max_rdim?p->max_rdim:1)*sizeof(double));
    double **jbuf = malloc((size_t)(p->max_nblk?p->max_nblk:1)*sizeof(double*));
    const double **pbuf = malloc((size_t)(p->max_nblk?p->max_nblk:1)*sizeof(double*));
    double *jstore = malloc((size_t)(p->max_nblk?p->max_nblk:1)*(p->max_rdim?p->max_rdim:1)*B*sizeof(double));
    if(!H||!Hd||!g||!dx||!xnew||!rbuf||!jbuf||!pbuf||!jstore){
        free(col);free(H);free(Hd);free(g);free(dx);free(xnew);free(rbuf);free(jbuf);free(pbuf);free(jstore);
        return -1;
    }
    evalctx e = { p, N, B, col, H, g, rbuf, jstore, jbuf, pbuf };

    int failed=0;
    double initial_cost = eval_cost(&e, p->x, 1, &failed);
    if(failed){ goto fail; }
    double prev_cost = initial_cost;
    double lambda = o.init_lambda;
    double nu = 2.0;                 // Nielsen up-factor for rejected steps
    int iter=0;

    for(iter=0; iter<o.max_iter; ++iter){
        double gmax=0; for(int i=0;i<N;++i){ double a=fabs(g[i]); if(a>gmax)gmax=a; }
        if(gmax < o.grad_tol) break;

        int accepted=0;
        for(int tryc=0; tryc<30 && !accepted; ++tryc){
            // damped: H' = H + lambda*diag(H)  (Marquardt diagonal scaling)
            memcpy(Hd,H,(size_t)N*N*sizeof(double));
            for(int i=0;i<N;++i){ double d=H[(size_t)i*N+i]; if(d<=0) d=1.0; Hd[(size_t)i*N+i]=H[(size_t)i*N+i]+lambda*d+1e-12; }
            if(ldlt_solve(Hd,N,g,dx)!=0){ lambda*=nu; nu*=2; if(lambda>1e14) goto done; continue; }

            memcpy(xnew,p->x,(size_t)p->nblocks*B*sizeof(double));
            double stepmax=0;
            for(int b=0;b<p->nblocks;++b){ if(p->is_const[b]) continue; int c=col[b];
                for(int d=0;d<B;++d){ double s=dx[c+d]; xnew[(size_t)b*B+d]+=s; double a=fabs(s); if(a>stepmax)stepmax=a; } }

            double trial = eval_cost(&e, xnew, 0, &failed);
            if(failed){ goto fail; }

            // gain ratio rho = (actual drop) / (predicted drop); predicted =
            // 0.5 * dx^T (lambda*D*dx + g), with g = -J^T r (our stored g).
            double pred=0;
            for(int b=0;b<p->nblocks;++b){ if(p->is_const[b]) continue; int c=col[b];
                for(int d=0;d<B;++d){ double Dii=H[(size_t)(c+d)*N+(c+d)]; if(Dii<=0)Dii=1.0;
                    pred += 0.5*dx[c+d]*(lambda*Dii*dx[c+d] + g[c+d]); } }
            double drop = prev_cost - trial;
            double rho = (pred>1e-300) ? drop/pred : (drop>0?1.0:-1.0);

            if(drop > 0){
                memcpy(p->x,xnew,(size_t)p->nblocks*B*sizeof(double));
                accepted=1;
                prev_cost = trial;
                // Nielsen: lambda *= max(1/3, 1-(2*rho-1)^3); reset nu.
                double f = 1.0 - (2.0*rho-1.0)*(2.0*rho-1.0)*(2.0*rho-1.0);
                if(f<1.0/3.0) f=1.0/3.0;
                lambda *= f; if(lambda<1e-12) lambda=1e-12; nu=2.0;
                (void)eval_cost(&e, p->x, 1, &failed); if(failed){ goto fail; }
                if(o.verbose) fprintf(stderr,"  it %d cost %.6g lambda %.3g rho %.3g\n", iter, prev_cost, lambda, rho);
                if(drop < o.func_tol*(prev_cost+o.func_tol)){ iter++; goto done; }
                if(stepmax < o.param_tol){ iter++; goto done; }
            } else {
                lambda *= nu; nu *= 2; if(lambda>1e14) goto done;
            }
        }
        if(!accepted) break;
    }
done:
    if(rep){ rep->initial_cost=initial_cost; rep->final_cost=prev_cost; rep->iterations=iter; rep->success=1; }
    free(col);free(H);free(Hd);free(g);free(dx);free(xnew);free(rbuf);free(jbuf);free(pbuf);free(jstore);
    return 0;
fail:
    free(col);free(H);free(Hd);free(g);free(dx);free(xnew);free(rbuf);free(jbuf);free(pbuf);free(jstore);
    return -1;
}
