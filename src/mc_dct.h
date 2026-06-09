// ============================================================================
// mc_dct.h — integer separable 3D DCT-16 (matter-compressor).
//
// Q14 fixed-point, even/odd partial-butterfly 1D core, 3 separable passes with a
// cache-blocked rotate between them. 16^3 only (the codec's fixed block size).
// Range-safe in i32 for 16^3. Lossy (integer rounding). Call mc_dct_init() once.
// ============================================================================
#ifndef MC_DCT_H
#define MC_DCT_H
#include <stdint.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MC_DCT_N    16              // block edge
#define MC_DCT_Q    14              // fixed-point fraction bits
#define MC_DCT_ALIGN 64
typedef int32_t mc_fi32;

// Q14 cosine matrix for N=16, built once.
static mc_fi32 g_mc_cm[MC_DCT_N][MC_DCT_N] __attribute__((aligned(MC_DCT_ALIGN)));
static int g_mc_cm_ready = 0;
static void mc_dct_init(void){
    if(g_mc_cm_ready) return;
    double scale=(double)((int64_t)1<<MC_DCT_Q);
    for(int k=0;k<MC_DCT_N;++k){ double ck=(k==0)?sqrt(1.0/MC_DCT_N):sqrt(2.0/MC_DCT_N);
        for(int n=0;n<MC_DCT_N;++n){ double v=ck*cos(M_PI*(2.0*n+1.0)*k/(2.0*MC_DCT_N));
            g_mc_cm[k][n]=(mc_fi32)llround(v*scale); } }
    g_mc_cm_ready=1;
}

// 1D forward DCT-II (even/odd partial butterfly). Platform-tuned form: clang/aarch64
// favour the k-parallel form, gcc/x86 the reduction form.
#if defined(__clang__) || defined(__aarch64__)
#define MC_DCT_KPARALLEL 1
#else
#define MC_DCT_KPARALLEL 0
#endif
static inline void mc_dct1d_fwd(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    const mc_fi32 rnd=(mc_fi32)1<<(MC_DCT_Q-1); const int S=MC_DCT_N, H=S/2;
    mc_fi32 s[H], d[H];
    for(int n=0;n<H;++n){ s[n]=in[n]+in[S-1-n]; d[n]=in[n]-in[S-1-n]; }
#if MC_DCT_KPARALLEL
    mc_fi32 acc[MC_DCT_N]; for(int k=0;k<S;++k) acc[k]=rnd;
    for(int n=0;n<H;++n){ mc_fi32 sn=s[n], dn=d[n];
        for(int k=0;k<S;k+=2) acc[k]+=g_mc_cm[k][n]*sn;
        for(int k=1;k<S;k+=2) acc[k]+=g_mc_cm[k][n]*dn; }
    for(int k=0;k<S;++k) out[k]=acc[k]>>MC_DCT_Q;
#else
    for(int k=0;k<S;k+=2){ mc_fi32 a=rnd; for(int n=0;n<H;++n) a+=g_mc_cm[k][n]*s[n]; out[k]=a>>MC_DCT_Q; }
    for(int k=1;k<S;k+=2){ mc_fi32 a=rnd; for(int n=0;n<H;++n) a+=g_mc_cm[k][n]*d[n]; out[k]=a>>MC_DCT_Q; }
#endif
}
static inline void mc_dct1d_inv(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    const mc_fi32 rnd=(mc_fi32)1<<(MC_DCT_Q-1); const int S=MC_DCT_N, H=S/2;
    for(int n=0;n<H;++n){ mc_fi32 evn=rnd,odd=0;
        for(int k=0;k<S;k+=2) evn+=g_mc_cm[k][n]*in[k];
        for(int k=1;k<S;k+=2) odd+=g_mc_cm[k][n]*in[k];
        out[n]=(evn+odd)>>MC_DCT_Q; out[S-1-n]=(evn-odd)>>MC_DCT_Q; }
}

// cache-blocked rotate (z,y,x)->(x,z,y): dst[(x*S+z)*S+y] = src[(z*S+y)*S+x].
#define MC_ROT_TILE 4
static inline void mc_rot(const mc_fi32 *restrict src, mc_fi32 *restrict dst){
    const int S=MC_DCT_N;
    for(int zt=0; zt<S; zt+=MC_ROT_TILE)
    for(int xt=0; xt<S; xt+=MC_ROT_TILE)
        for(int z=zt; z<zt+MC_ROT_TILE; ++z)
        for(int x=xt; x<xt+MC_ROT_TILE; ++x){
            const mc_fi32 *sp = src + ((size_t)z*S)*S + x;
            mc_fi32 *dp = dst + ((size_t)x*S+z)*S;
            for(int y=0;y<S;++y) dp[y]=sp[(size_t)y*S];
        }
}
// transform all contiguous lines (skipping all-zero lines), in place or out-of-place.
static inline void mc_lines_fwd_to(const mc_fi32 *restrict src, mc_fi32 *restrict dst){
    const int S=MC_DCT_N; mc_fi32 ol[MC_DCT_N];
    for(int off=0;off<S*S;++off){ const mc_fi32 *v=src+(size_t)off*S; mc_fi32 *o=dst+(size_t)off*S;
        int nz=0; for(int i=0;i<S;++i) if(v[i]){nz=1;break;}
        if(!nz){ for(int i=0;i<S;++i) o[i]=0; continue; }
        mc_dct1d_fwd(v,ol); for(int i=0;i<S;++i) o[i]=ol[i]; }
}
static inline void mc_lines_fwd(mc_fi32 *restrict blk){
    const int S=MC_DCT_N; mc_fi32 ol[MC_DCT_N];
    for(int off=0;off<S*S;++off){ mc_fi32 *v=blk+(size_t)off*S;
        int nz=0; for(int i=0;i<S;++i) if(v[i]){nz=1;break;} if(!nz) continue;
        mc_dct1d_fwd(v,ol); for(int i=0;i<S;++i) v[i]=ol[i]; }
}
static inline void mc_lines_inv_to(const mc_fi32 *restrict src, mc_fi32 *restrict dst){
    const int S=MC_DCT_N; mc_fi32 ol[MC_DCT_N];
    for(int off=0;off<S*S;++off){ const mc_fi32 *v=src+(size_t)off*S; mc_fi32 *o=dst+(size_t)off*S;
        int nz=0; for(int i=0;i<S;++i) if(v[i]){nz=1;break;}
        if(!nz){ for(int i=0;i<S;++i) o[i]=0; continue; }
        mc_dct1d_inv(v,ol); for(int i=0;i<S;++i) o[i]=ol[i]; }
}
static inline void mc_lines_inv(mc_fi32 *restrict blk){
    const int S=MC_DCT_N; mc_fi32 ol[MC_DCT_N];
    for(int off=0;off<S*S;++off){ mc_fi32 *v=blk+(size_t)off*S;
        int nz=0; for(int i=0;i<S;++i) if(v[i]){nz=1;break;} if(!nz) continue;
        mc_dct1d_inv(v,ol); for(int i=0;i<S;++i) v[i]=ol[i]; }
}

// 3D forward/inverse on a 16^3 block (float in/out for the codec's quant path).
// Each pass: transform contiguous lines, then rotate. 3 rotates return to (z,y,x).
static void mc_dct3_fwd(const float *restrict blk, float *restrict coef){
    const int n=MC_DCT_N*MC_DCT_N*MC_DCT_N;
    static _Thread_local mc_fi32 in[16*16*16]  __attribute__((aligned(MC_DCT_ALIGN)));
    static _Thread_local mc_fi32 a[16*16*16]   __attribute__((aligned(MC_DCT_ALIGN)));
    static _Thread_local mc_fi32 b[16*16*16]   __attribute__((aligned(MC_DCT_ALIGN)));
    for(int i=0;i<n;++i) in[i]=(mc_fi32)lrintf(blk[i]);
    mc_lines_fwd_to(in,a); mc_rot(a,b);
    mc_lines_fwd(b);       mc_rot(b,a);
    mc_lines_fwd(a);       mc_rot(a,b);
    for(int i=0;i<n;++i) coef[i]=(float)b[i];
}
static void mc_dct3_inv(const float *restrict coef, float *restrict blk){
    const int n=MC_DCT_N*MC_DCT_N*MC_DCT_N;
    static _Thread_local mc_fi32 in[16*16*16]  __attribute__((aligned(MC_DCT_ALIGN)));
    static _Thread_local mc_fi32 a[16*16*16]   __attribute__((aligned(MC_DCT_ALIGN)));
    static _Thread_local mc_fi32 b[16*16*16]   __attribute__((aligned(MC_DCT_ALIGN)));
    for(int i=0;i<n;++i) in[i]=(mc_fi32)lrintf(coef[i]);
    mc_lines_inv_to(in,a); mc_rot(a,b);
    mc_lines_inv(b);       mc_rot(b,a);
    mc_lines_inv(a);       mc_rot(a,b);
    for(int i=0;i<n;++i) blk[i]=(float)b[i];
}
#endif
