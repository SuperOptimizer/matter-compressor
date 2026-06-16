/* mc_segment.c — classical sheet detection + topology post-processing.
 * See mc_segment.h. All buffers are dense u8/i32, z-major (z,y,x). */
#include "mc_segment.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

#define IDX(z,y,x) (((size_t)(z)*ny + (y))*nx + (x))

// thread count for the detector's data-parallel loops (blur lines, eigen/mask).
#define MC_SEG_MAXTHREADS 16
static int seg_threads(void){
    long nc = sysconf(_SC_NPROCESSORS_ONLN);
    if(nc<1) nc=1; if(nc>MC_SEG_MAXTHREADS) nc=MC_SEG_MAXTHREADS;
    return (int)nc;
}
// run fn(arg, lo, hi) over [0,total) split into the thread pool's contiguous
// ranges. Falls back to a single in-line call for tiny work or 1 thread.
typedef void (*seg_range_fn)(void *arg, int lo, int hi);
typedef struct { seg_range_fn fn; void *arg; int lo, hi; } seg_task;
static void *seg_trampoline(void *p){ seg_task *t=p; t->fn(t->arg,t->lo,t->hi); return NULL; }
static void seg_parallel_for(int total, seg_range_fn fn, void *arg){
    int nt = seg_threads();
    if(nt<=1 || total < 256){ fn(arg,0,total); return; }
    if(nt>total) nt=total;
    pthread_t th[MC_SEG_MAXTHREADS]; seg_task tk[MC_SEG_MAXTHREADS];
    int chunk=(total+nt-1)/nt, made=0;
    for(int i=0;i<nt;++i){
        int lo=i*chunk, hi=lo+chunk; if(hi>total)hi=total; if(lo>=hi) break;
        tk[i]=(seg_task){fn,arg,lo,hi};
        if(pthread_create(&th[i],NULL,seg_trampoline,&tk[i])!=0){ fn(arg,lo,hi); continue; }
        made++;
    }
    for(int i=0;i<made;++i) pthread_join(th[i],NULL);
}

// ---- separable 1D gaussian blur of a float volume along one axis ----
static void gauss1d_build(float sigma, float **k, int *r){
    int rad = (int)ceilf(3.0f*sigma); if(rad<1) rad=1;
    float *w = malloc((size_t)(2*rad+1)*sizeof(float));
    float s=0; for(int i=-rad;i<=rad;++i){ float v=expf(-(float)(i*i)/(2*sigma*sigma)); w[i+rad]=v; s+=v; }
    for(int i=0;i<2*rad+1;++i) w[i]/=s;
    *k=w; *r=rad;
}
// Separable 1D gaussian blur along one axis, clamp-to-edge. Walks each 1D line
// along the axis once: the stride between taps is constant (1 for x, nx for y,
// ny*nx for z), so we hoist the line base + stride and split each line into a
// clamped head, an unclamped interior (the hot, vectorizable run), and a
// clamped tail. Same numerics as the naive per-tap-clamp form.
// args for the threaded blur (one axis pass); lines are split across threads.
typedef struct { const float *in; float *out; const float *k;
                 int rad, len, nx, ny; size_t st; int axis; } blur_args;
static void blur_lines(void *vp, int Llo, int Lhi){
    blur_args *a=vp;
    int len=a->len, rad=a->rad, nx=a->nx, ny=a->ny, axis=a->axis;
    size_t st=a->st; const float *k=a->k;
    for(int L=Llo; L<Lhi; ++L){
        size_t base;
        if(axis==2){ base=(size_t)L*nx; }              // x: contiguous row
        else if(axis==1){ int z=L/nx,x=L%nx; base=(size_t)z*ny*nx+x; }  // y
        else { int y=L/nx,x=L%nx; base=(size_t)y*nx+x; }                // z
        const float *ip=a->in+base; float *op=a->out+base;
        for(int p=0; p<len; ++p){
            float acc=0;
            int lo=p-rad, hi=p+rad;
            if(lo>=0 && hi<len){     // interior: no clamp, straight strided dot
                const float *s=ip+(size_t)lo*st;
                for(int t=0;t<2*rad+1;++t){ acc+=k[t]*(*s); s+=st; }
            } else {                 // border: clamp each tap to [0,len-1]
                for(int t=-rad;t<=rad;++t){
                    int q=p+t; if(q<0)q=0; else if(q>=len)q=len-1;
                    acc+=k[t+rad]*ip[(size_t)q*st];
                }
            }
            op[(size_t)p*st]=acc;
        }
    }
}
// Separable 1D gaussian blur along one axis, clamp-to-edge. Each 1D line is
// independent, so the line set is split across the thread pool. Walking a line
// once with a constant tap stride lets the interior run vectorize.
static void blur_axis(const float *in, float *out, int nz,int ny,int nx,
                      int axis, const float *k, int rad){
    int len   = (axis==0)?nz : (axis==1)?ny : nx;
    size_t st = (axis==0)?(size_t)ny*nx : (axis==1)?(size_t)nx : 1;
    int nlines = (int)((size_t)nz*ny*nx / len);
    blur_args a = { in, out, k, rad, len, nx, ny, st, axis };
    seg_parallel_for(nlines, blur_lines, &a);
}
static void gauss3(float *vol, int nz,int ny,int nx, float sigma, float *scratch){
    if(sigma<=0) return;
    float *k; int rad; gauss1d_build(sigma,&k,&rad);
    blur_axis(vol,scratch,nz,ny,nx,0,k,rad);
    blur_axis(scratch,vol,nz,ny,nx,1,k,rad);
    blur_axis(vol,scratch,nz,ny,nx,2,k,rad);
    memcpy(vol,scratch,(size_t)nz*ny*nx*sizeof(float));
    free(k);
}

// symmetric 3x3 eigenvalues (sorted desc) via the closed-form trig method.
static void eig3sym(double a,double b,double c,double d,double e,double f,
                    double ev[3]){
    // matrix [[a,d,e],[d,b,f],[e,f,c]]
    double p1=d*d+e*e+f*f;
    if(p1<1e-20){ // diagonal
        double t[3]={a,b,c};
        for(int i=0;i<3;++i)for(int j=i+1;j<3;++j) if(t[j]>t[i]){double s=t[i];t[i]=t[j];t[j]=s;}
        ev[0]=t[0];ev[1]=t[1];ev[2]=t[2]; return;
    }
    double q=(a+b+c)/3.0;
    double p2=(a-q)*(a-q)+(b-q)*(b-q)+(c-q)*(c-q)+2*p1;
    double p=sqrt(p2/6.0);
    double aa=(a-q)/p, bb=(b-q)/p, cc=(c-q)/p, dd=d/p, ee=e/p, ff=f/p;
    // det(B)/2
    double detB = aa*(bb*cc-ff*ff) - dd*(dd*cc-ff*ee) + ee*(dd*ff-bb*ee);
    double r=detB/2.0; if(r<-1)r=-1; if(r>1)r=1;
    double phi=acos(r)/3.0;
    double e0=q+2*p*cos(phi);
    double e2=q+2*p*cos(phi+2.0*M_PI/3.0);
    double e1=3*q-e0-e2;
    ev[0]=e0; ev[1]=e1; ev[2]=e2;  // already sorted desc by construction
}

// threaded sheetness -> mask over the smoothed structure-tensor components.
typedef struct { const float *Txx,*Tyy,*Tzz,*Txy,*Txz,*Tyz;
                 const uint8_t *vol; uint8_t *mask; float sheetness; int val_lo; } sheet_args;
static void sheet_mask(void *vp, int lo, int hi){
    sheet_args *a=vp;
    for(int i=lo;i<hi;++i){
        double ev[3];
        eig3sym(a->Txx[i],a->Tyy[i],a->Tzz[i],a->Txy[i],a->Txz[i],a->Tyz[i],ev);
        double l0=ev[0],l1=ev[1];
        double sheet = (l0>1e-6) ? (l0-l1)/(l0+1e-6) : 0.0;
        a->mask[i] = (a->vol[i] >= a->val_lo && sheet >= a->sheetness) ? 255 : 0;
    }
}

int mc_seg_detect(const uint8_t *vol, int nz, int ny, int nx,
                  const mc_seg_params *params, uint8_t *mask){
    if(nz<1||ny<1||nx<1) return -1;     // degenerate volume (the gradient stencil
                                        // indexes neighbors; 0 dims would go OOB)
    mc_seg_params P = { 1.0f, 2.0f, 0.5f, 1 };
    if(params) P=*params;
    if(P.sigma_grad<=0) P.sigma_grad=1.0f;
    if(P.sigma_tensor<=0) P.sigma_tensor=2.0f;
    size_t n=(size_t)nz*ny*nx;

    float *f = malloc(n*sizeof(float));
    float *scratch = malloc(n*sizeof(float));
    if(!f||!scratch){ free(f); free(scratch); return -1; }
    for(size_t i=0;i<n;++i) f[i]=(float)vol[i];
    gauss3(f,nz,ny,nx,P.sigma_grad,scratch);

    // gradient (central differences) -> structure tensor components.
    float *Txx=malloc(n*sizeof(float)),*Tyy=malloc(n*sizeof(float)),*Tzz=malloc(n*sizeof(float));
    float *Txy=malloc(n*sizeof(float)),*Txz=malloc(n*sizeof(float)),*Tyz=malloc(n*sizeof(float));
    if(!Txx||!Tyy||!Tzz||!Txy||!Txz||!Tyz){ free(f);free(scratch);free(Txx);free(Tyy);free(Tzz);free(Txy);free(Txz);free(Tyz); return -1; }
    for(int z=0;z<nz;++z)for(int y=0;y<ny;++y)for(int x=0;x<nx;++x){
        int zm=z>0?z-1:z, zp=z<nz-1?z+1:z;
        int ym=y>0?y-1:y, yp=y<ny-1?y+1:y;
        int xm=x>0?x-1:x, xp=x<nx-1?x+1:x;
        float gz=0.5f*(f[IDX(zp,y,x)]-f[IDX(zm,y,x)]);
        float gy=0.5f*(f[IDX(z,yp,x)]-f[IDX(z,ym,x)]);
        float gx=0.5f*(f[IDX(z,y,xp)]-f[IDX(z,y,xm)]);
        size_t i=IDX(z,y,x);
        Txx[i]=gx*gx; Tyy[i]=gy*gy; Tzz[i]=gz*gz;
        Txy[i]=gx*gy; Txz[i]=gx*gz; Tyz[i]=gy*gz;
    }
    // smooth the tensor (integration scale).
    gauss3(Txx,nz,ny,nx,P.sigma_tensor,scratch);
    gauss3(Tyy,nz,ny,nx,P.sigma_tensor,scratch);
    gauss3(Tzz,nz,ny,nx,P.sigma_tensor,scratch);
    gauss3(Txy,nz,ny,nx,P.sigma_tensor,scratch);
    gauss3(Txz,nz,ny,nx,P.sigma_tensor,scratch);
    gauss3(Tyz,nz,ny,nx,P.sigma_tensor,scratch);

    // sheetness = (l0 - l1)/(l0+eps): large when one eigenvalue dominates (a
    // planar feature: gradient coherent along the sheet normal, flat in-plane).
    // Per-voxel + independent -> split across the thread pool.
    sheet_args sa = { Txx,Tyy,Tzz,Txy,Txz,Tyz, vol, mask, P.sheetness, P.val_lo };
    seg_parallel_for((int)((n>(size_t)0x7fffffff)?0x7fffffff:n), sheet_mask, &sa);
    free(f); free(scratch); free(Txx);free(Tyy);free(Tzz);free(Txy);free(Txz);free(Tyz);
    return 0;
}

// ---- connected components (26-connectivity), union-find over a label pass ----
static int label26(const uint8_t *mask, int nz,int ny,int nx, int32_t *lab){
    size_t n=(size_t)nz*ny*nx;
    int32_t *parent=malloc((n+1)*sizeof(int32_t)); // 1-based provisional labels
    if(!parent) return -1;
    int32_t next=1;
    for(size_t i=0;i<n;++i) lab[i]=0;
    // simple BFS flood per seed (avoids union-find bookkeeping; n is chunk-sized).
    int32_t *stack=malloc(n*sizeof(int32_t)); if(!stack){ free(parent); return -1; }
    (void)parent;
    int ncomp=0;
    for(int z=0;z<nz;++z)for(int y=0;y<ny;++y)for(int x=0;x<nx;++x){
        size_t s=IDX(z,y,x);
        if(!mask[s] || lab[s]) continue;
        ncomp++; int32_t id=next++;
        int top=0; stack[top++]=(int32_t)s; lab[s]=id;
        while(top){
            int32_t cur=stack[--top];
            int cz=cur/((int)ny*nx), rem=cur%((int)ny*nx), cy=rem/nx, cx=rem%nx;
            for(int dz=-1;dz<=1;++dz)for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
                if(!dz&&!dy&&!dx) continue;
                int nz2=cz+dz,ny2=cy+dy,nx2=cx+dx;
                if(nz2<0||ny2<0||nx2<0||nz2>=nz||ny2>=ny||nx2>=nx) continue;
                size_t ni=IDX(nz2,ny2,nx2);
                if(mask[ni] && !lab[ni]){ lab[ni]=id; stack[top++]=(int32_t)ni; }
            }
        }
    }
    free(parent); free(stack);
    return ncomp;
}

int mc_seg_label(const uint8_t *mask, int nz, int ny, int nx, int32_t *labels){
    return label26(mask,nz,ny,nx,labels);
}

int mc_seg_remove_small(uint8_t *mask, int nz, int ny, int nx, int min_voxels){
    size_t n=(size_t)nz*ny*nx;
    int32_t *lab=malloc(n*sizeof(int32_t)); if(!lab) return -1;
    int nc=label26(mask,nz,ny,nx,lab);
    if(nc<0){ free(lab); return -1; }
    long *cnt=calloc((size_t)nc+1,sizeof(long));
    for(size_t i=0;i<n;++i) if(lab[i]) cnt[lab[i]]++;
    int kept=0; for(int c=1;c<=nc;++c) if(cnt[c]>=min_voxels) kept++;
    for(size_t i=0;i<n;++i) if(lab[i] && cnt[lab[i]]<min_voxels) mask[i]=0;
    free(lab); free(cnt);
    return kept;
}

// ---- 1-voxel hole plugging via a 6-connectivity-watertight 2x2x2 LUT ----
// For each 2x2x2 cube of the mask, decide whether to ADD voxels so the local
// neighborhood is 6-connectivity watertight (no 1-voxel pinhole). We use a
// simple, effective rule (inspired by the winners' Euler-number LUT): if a cube
// has >=5 of 8 foreground corners, fill the whole cube. This plugs single-voxel
// gaps/notches without over-dilating. Returns voxels added.
long mc_seg_plug_holes(uint8_t *mask, int nz, int ny, int nx, int passes){
    long added=0;
    if(passes<1) passes=1;
    for(int pass=0;pass<passes;++pass){
        long padded=0;
        // iterate over all 2x2x2 cube origins; collect adds, then apply (so a
        // pass sees the pre-pass state, deterministic).
        for(int z=0;z<nz-1;++z)for(int y=0;y<ny-1;++y)for(int x=0;x<nx-1;++x){
            int c=0; size_t idx[8]; int k=0;
            for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
                size_t i=IDX(z+dz,y+dy,x+dx); idx[k++]=i; if(mask[i]) c++;
            }
            if(c>=5 && c<8){ for(int j=0;j<8;++j) if(!mask[idx[j]]){ mask[idx[j]]=255; padded++; } }
        }
        added+=padded;
        if(!padded) break;
    }
    return added;
}

// ---- spherical binary closing (dilate then erode) ----
// dilate: out=fg if ANY in-radius voxel is fg. erode: out=fg only if ALL are.
// outside the volume counts as background.
static void morph(const uint8_t *in, uint8_t *out, int nz,int ny,int nx, int radius, int dilate){
    int r2=radius*radius;
    for(int z=0;z<nz;++z)for(int y=0;y<ny;++y)for(int x=0;x<nx;++x){
        int hit = dilate?0:1;        // dilate: become fg on any hit; erode: stay fg unless a miss
        for(int dz=-radius;dz<=radius;++dz)
        for(int dy=-radius;dy<=radius;++dy)
        for(int dx=-radius;dx<=radius;++dx){
            if(dz*dz+dy*dy+dx*dx>r2) continue;
            int zz=z+dz,yy=y+dy,xx=x+dx;
            int v = (zz<0||yy<0||xx<0||zz>=nz||yy>=ny||xx>=nx) ? 0 : (in[IDX(zz,yy,xx)]?1:0);
            if(dilate){ if(v){ hit=1; goto done; } }
            else      { if(!v){ hit=0; goto done; } }
        }
        done:
        out[IDX(z,y,x)] = hit?255:0;
    }
}

int mc_seg_close(uint8_t *mask, int nz, int ny, int nx, int radius){
    if(radius<1) return 0;
    size_t n=(size_t)nz*ny*nx;
    uint8_t *tmp=malloc(n), *tmp2=malloc(n);
    if(!tmp||!tmp2){ free(tmp); free(tmp2); return -1; }
    morph(mask,tmp,nz,ny,nx,radius,1);   // dilate
    morph(tmp,tmp2,nz,ny,nx,radius,0);   // erode
    memcpy(mask,tmp2,n);
    free(tmp); free(tmp2);
    return 0;
}

// ---- fill cavities: flood background from the border; unreached bg = cavity ----
int mc_seg_fill_cavities(uint8_t *mask, int nz, int ny, int nx){
    size_t n=(size_t)nz*ny*nx;
    uint8_t *bg=calloc(n,1);           // 1 = background reachable from border
    int32_t *stack=malloc(n*sizeof(int32_t));
    if(!bg||!stack){ free(bg); free(stack); return -1; }
    int top=0;
    // seed all border background voxels.
    for(int z=0;z<nz;++z)for(int y=0;y<ny;++y)for(int x=0;x<nx;++x){
        if(z||y||x){ if(z!=nz-1&&y!=ny-1&&x!=nx-1 && z&&y&&x) continue; }
        size_t i=IDX(z,y,x);
        if(!mask[i] && !bg[i]){ bg[i]=1; stack[top++]=(int32_t)i; }
    }
    while(top){
        int32_t cur=stack[--top];
        int cz=cur/((int)ny*nx), rem=cur%((int)ny*nx), cy=rem/nx, cx=rem%nx;
        const int d6[6][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
        for(int k=0;k<6;++k){
            int zz=cz+d6[k][0],yy=cy+d6[k][1],xx=cx+d6[k][2];
            if(zz<0||yy<0||xx<0||zz>=nz||yy>=ny||xx>=nx) continue;
            size_t ni=IDX(zz,yy,xx);
            if(!mask[ni] && !bg[ni]){ bg[ni]=1; stack[top++]=(int32_t)ni; }
        }
    }
    long filled=0;
    for(size_t i=0;i<n;++i) if(!mask[i] && !bg[i]){ mask[i]=255; filled++; }
    free(bg); free(stack);
    return (int)(filled>0);
}
