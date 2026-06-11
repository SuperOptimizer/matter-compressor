#define _GNU_SOURCE                  // pthread_setname_np (linux)
#include <pthread.h>
#include <stdatomic.h>
// ============================================================================
// matter_compressor.c — single-file implementation: codec + archive + cache.
// Unified from mc_dct.h / mc_rangecoder.h / mc_xxhash.h / mc_codec.c /
// mc_archive.h / mc_archive_read.h / mc_archive.c / mc_cache.c.
// ============================================================================
#include "matter_compressor.h"

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
// SIMD kernel selection. ARM: NEON always (Graviton, Apple M, X1 Elite).
// x86: AVX2 selected at compile time (x86-64-v3 is the project's floor, so
// build with -march=x86-64-v3 or newer); an AVX-512 line-pair variant is
// runtime-dispatched when compiled in (Zen4/5 have it; many Intel parts have
// it fused off, so never assume at compile time). SVE was evaluated and
// intentionally skipped: every hot kernel here is a fixed 8-lane i32 problem
// (half a DCT-16 line) which two NEON q-regs / one AVX2 ymm already saturate;
// scalable vectors add overhead, not lanes, at this block size.
#if defined(__ARM_NEON) || defined(__aarch64__)
  #include <arm_neon.h>
  #define MC_SIMD_NEON 1
#elif defined(MC_ENABLE_AVX512) && defined(__AVX512F__)
  #include <immintrin.h>
  #define MC_SIMD_AVX512 1   // opt-in (-DMC_ENABLE_AVX512, -march=x86-64-v4/znver4);
                             // compile-tested only — default x86 builds use the
                             // measured AVX2 path below.
#elif defined(__AVX2__)
  #include <immintrin.h>
  #define MC_SIMD_AVX2 1
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MC_DCT_N    16              // block edge
#define MC_DCT_Q    14              // fixed-point fraction bits
#define MC_DCT_ALIGN 64
typedef int32_t mc_fi32;

// Q14 cosine matrix for N=16, built once. Packed even/odd variants:
//   g_cm_eT[n][j] = cm[2j][n], g_cm_oT[n][j] = cm[2j+1][n]   (forward, k-contig)
//   g_cm_e [j][n] = cm[2j][n], g_cm_o [j][n] = cm[2j+1][n]   (inverse, n-contig)
static mc_fi32 g_mc_cm[MC_DCT_N][MC_DCT_N] __attribute__((aligned(MC_DCT_ALIGN)));
static mc_fi32 g_cm_eT[8][8] __attribute__((aligned(MC_DCT_ALIGN)));
static mc_fi32 g_cm_oT[8][8] __attribute__((aligned(MC_DCT_ALIGN)));
static mc_fi32 g_cm_e [8][8] __attribute__((aligned(MC_DCT_ALIGN)));
static mc_fi32 g_cm_o [8][8] __attribute__((aligned(MC_DCT_ALIGN)));
static mc_fi32 g_cm_eo[8][16] __attribute__((aligned(MC_DCT_ALIGN)));   // [e row | o row] for zmm
static int g_mc_cm_ready = 0;
static void mc_dct_init(void){
    if(g_mc_cm_ready) return;
    double scale=(double)((int64_t)1<<MC_DCT_Q);
    for(int k=0;k<MC_DCT_N;++k){ double ck=(k==0)?sqrt(1.0/MC_DCT_N):sqrt(2.0/MC_DCT_N);
        for(int n=0;n<MC_DCT_N;++n){ double v=ck*cos(M_PI*(2.0*n+1.0)*k/(2.0*MC_DCT_N));
            g_mc_cm[k][n]=(mc_fi32)llround(v*scale); } }
    for(int j=0;j<8;++j)for(int n=0;n<8;++n){
        g_cm_e[j][n]=g_mc_cm[2*j][n];   g_cm_eT[n][j]=g_mc_cm[2*j][n];
        g_cm_o[j][n]=g_mc_cm[2*j+1][n]; g_cm_oT[n][j]=g_mc_cm[2*j+1][n];
        g_cm_eo[j][n]=g_cm_e[j][n];     g_cm_eo[j][n+8]=g_cm_o[j][n];
    }
    g_mc_cm_ready=1;
}

// 1D forward DCT-II (even/odd partial butterfly). NEON/AVX2: packed transposed
// tables (g_cm_eT/oT) give contiguous 8-lane MACs; scalar keeps the k-parallel
// form (measured faster under autovectorization). (AVX-512 defines fwd+inv
// together above.)
#if MC_SIMD_AVX512
/* defined alongside the inverse above */
#elif MC_SIMD_NEON
static inline void mc_dct1d_fwd(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    const mc_fi32 rnd=(mc_fi32)1<<(MC_DCT_Q-1); const int S=MC_DCT_N, H=S/2;
    mc_fi32 s[8], d[8];
    for(int n=0;n<H;++n){ s[n]=in[n]+in[S-1-n]; d[n]=in[n]-in[S-1-n]; }
    int32x4_t ae0=vdupq_n_s32(rnd), ae1=vdupq_n_s32(rnd);
    int32x4_t ao0=vdupq_n_s32(rnd), ao1=vdupq_n_s32(rnd);
    for(int n=0;n<H;++n){
        int32x4_t sn=vdupq_n_s32(s[n]), dn=vdupq_n_s32(d[n]);
        ae0=vmlaq_s32(ae0,vld1q_s32(&g_cm_eT[n][0]),sn);
        ae1=vmlaq_s32(ae1,vld1q_s32(&g_cm_eT[n][4]),sn);
        ao0=vmlaq_s32(ao0,vld1q_s32(&g_cm_oT[n][0]),dn);
        ao1=vmlaq_s32(ao1,vld1q_s32(&g_cm_oT[n][4]),dn);
    }
    mc_fi32 e[8],o[8];
    vst1q_s32(e,vshrq_n_s32(ae0,MC_DCT_Q)); vst1q_s32(e+4,vshrq_n_s32(ae1,MC_DCT_Q));
    vst1q_s32(o,vshrq_n_s32(ao0,MC_DCT_Q)); vst1q_s32(o+4,vshrq_n_s32(ao1,MC_DCT_Q));
    for(int j=0;j<8;++j){ out[2*j]=e[j]; out[2*j+1]=o[j]; }
}
#elif MC_SIMD_AVX2
static inline void mc_dct1d_fwd(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    const mc_fi32 rnd=(mc_fi32)1<<(MC_DCT_Q-1); const int S=MC_DCT_N, H=S/2;
    mc_fi32 s[8], d[8];
    for(int n=0;n<H;++n){ s[n]=in[n]+in[S-1-n]; d[n]=in[n]-in[S-1-n]; }
    __m256i ae=_mm256_set1_epi32(rnd), ao=_mm256_set1_epi32(rnd);
    for(int n=0;n<H;++n){
        ae=_mm256_add_epi32(ae,_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i*)&g_cm_eT[n][0]),_mm256_set1_epi32(s[n])));
        ao=_mm256_add_epi32(ao,_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i*)&g_cm_oT[n][0]),_mm256_set1_epi32(d[n])));
    }
    mc_fi32 e[8],o[8];
    _mm256_storeu_si256((__m256i*)e,_mm256_srai_epi32(ae,MC_DCT_Q));
    _mm256_storeu_si256((__m256i*)o,_mm256_srai_epi32(ao,MC_DCT_Q));
    for(int j=0;j<8;++j){ out[2*j]=e[j]; out[2*j+1]=o[j]; }
}
#else
static inline void mc_dct1d_fwd(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    const mc_fi32 rnd=(mc_fi32)1<<(MC_DCT_Q-1); const int S=MC_DCT_N, H=S/2;
    mc_fi32 s[H], d[H];
    for(int n=0;n<H;++n){ s[n]=in[n]+in[S-1-n]; d[n]=in[n]-in[S-1-n]; }
    mc_fi32 acc[MC_DCT_N]; for(int k=0;k<S;++k) acc[k]=rnd;
    for(int n=0;n<H;++n){ mc_fi32 sn=s[n], dn=d[n];
        for(int k=0;k<S;k+=2) acc[k]+=g_mc_cm[k][n]*sn;
        for(int k=1;k<S;k+=2) acc[k]+=g_mc_cm[k][n]*dn; }
    for(int k=0;k<S;++k) out[k]=acc[k]>>MC_DCT_Q;
}
#endif
// 1D inverse, sparse-aware: skips zero coefficients (most lines have only a few
// nonzero low-frequency entries after dequant). NEON/AVX2 kernels measured
// ~1.6x over the autovectorized scalar form; scalar fallback kept.
#if MC_SIMD_NEON
static inline void mc_dct1d_inv(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    const mc_fi32 rnd=(mc_fi32)1<<(MC_DCT_Q-1);
    const mc_fi32 *ce=&g_cm_e[0][0], *co=&g_cm_o[0][0];   // one base each: the
    // unrolled loop below indexes with immediates instead of re-materializing
    // adrp+add per table row (objdump finding).
    int32x4_t ae0=vdupq_n_s32(rnd), ae1=vdupq_n_s32(rnd);
    int32x4_t ao0=vdupq_n_s32(0),  ao1=vdupq_n_s32(0);
    for(int j=0;j<8;++j){
        mc_fi32 ve=in[2*j];
        if(ve){
            int32x4_t v=vdupq_n_s32(ve);
            ae0=vmlaq_s32(ae0,vld1q_s32(ce+j*8),v);
            ae1=vmlaq_s32(ae1,vld1q_s32(ce+j*8+4),v);
        }
        mc_fi32 vo=in[2*j+1];
        if(vo){
            int32x4_t v=vdupq_n_s32(vo);
            ao0=vmlaq_s32(ao0,vld1q_s32(co+j*8),v);
            ao1=vmlaq_s32(ao1,vld1q_s32(co+j*8+4),v);
        }
    }
    int32x4_t s0=vshrq_n_s32(vaddq_s32(ae0,ao0),MC_DCT_Q);
    int32x4_t s1=vshrq_n_s32(vaddq_s32(ae1,ao1),MC_DCT_Q);
    int32x4_t d0=vshrq_n_s32(vsubq_s32(ae0,ao0),MC_DCT_Q);
    int32x4_t d1=vshrq_n_s32(vsubq_s32(ae1,ao1),MC_DCT_Q);
    vst1q_s32(out,s0); vst1q_s32(out+4,s1);
    int32x4_t r1=vrev64q_s32(d1); r1=vextq_s32(r1,r1,2);   // reverse lanes
    int32x4_t r0=vrev64q_s32(d0); r0=vextq_s32(r0,r0,2);
    vst1q_s32(out+8,r1); vst1q_s32(out+12,r0);
}
#elif MC_SIMD_AVX512
// even+odd accumulators in one zmm: lanes [ae0..ae7 | ao0..ao7].
static inline void mc_dct1d_inv(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    const mc_fi32 rnd=(mc_fi32)1<<(MC_DCT_Q-1);
    __m512i acc=_mm512_inserti64x4(_mm512_set1_epi32(rnd),_mm256_setzero_si256(),1);
    for(int j=0;j<8;++j){
        mc_fi32 ve=in[2*j], vo=in[2*j+1];
        if(!(ve|vo)) continue;
        __m512i val=_mm512_inserti64x4(_mm512_set1_epi32(ve),_mm256_set1_epi32(vo),1);
        acc=_mm512_add_epi32(acc,_mm512_mullo_epi32(_mm512_load_si512(&g_cm_eo[j][0]),val));
    }
    __m256i ae=_mm512_extracti64x4_epi64(acc,0), ao=_mm512_extracti64x4_epi64(acc,1);
    __m256i sm=_mm256_srai_epi32(_mm256_add_epi32(ae,ao),MC_DCT_Q);
    __m256i d =_mm256_srai_epi32(_mm256_sub_epi32(ae,ao),MC_DCT_Q);
    _mm256_storeu_si256((__m256i*)out,sm);
    __m256i rev=_mm256_shuffle_epi32(d,_MM_SHUFFLE(0,1,2,3));
    rev=_mm256_permute2x128_si256(rev,rev,1);
    _mm256_storeu_si256((__m256i*)(out+8),rev);
}
static inline void mc_dct1d_fwd(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    const mc_fi32 rnd=(mc_fi32)1<<(MC_DCT_Q-1); const int S=MC_DCT_N, H=S/2;
    mc_fi32 s[8], d[8];
    for(int n=0;n<H;++n){ s[n]=in[n]+in[S-1-n]; d[n]=in[n]-in[S-1-n]; }
    __m256i ae=_mm256_set1_epi32(rnd), ao=_mm256_set1_epi32(rnd);
    for(int n=0;n<H;++n){
        ae=_mm256_add_epi32(ae,_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i*)&g_cm_eT[n][0]),_mm256_set1_epi32(s[n])));
        ao=_mm256_add_epi32(ao,_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i*)&g_cm_oT[n][0]),_mm256_set1_epi32(d[n])));
    }
    mc_fi32 e[8],o[8];
    _mm256_storeu_si256((__m256i*)e,_mm256_srai_epi32(ae,MC_DCT_Q));
    _mm256_storeu_si256((__m256i*)o,_mm256_srai_epi32(ao,MC_DCT_Q));
    for(int j=0;j<8;++j){ out[2*j]=e[j]; out[2*j+1]=o[j]; }
}
#elif MC_SIMD_AVX2
static inline void mc_dct1d_inv(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    const mc_fi32 rnd=(mc_fi32)1<<(MC_DCT_Q-1);
    __m256i ae=_mm256_set1_epi32(rnd), ao=_mm256_setzero_si256();
    for(int j=0;j<8;++j){
        mc_fi32 ve=in[2*j];
        if(ve) ae=_mm256_add_epi32(ae,_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i*)&g_cm_e[j][0]),_mm256_set1_epi32(ve)));
        mc_fi32 vo=in[2*j+1];
        if(vo) ao=_mm256_add_epi32(ao,_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i*)&g_cm_o[j][0]),_mm256_set1_epi32(vo)));
    }
    __m256i s=_mm256_srai_epi32(_mm256_add_epi32(ae,ao),MC_DCT_Q);
    __m256i d=_mm256_srai_epi32(_mm256_sub_epi32(ae,ao),MC_DCT_Q);
    _mm256_storeu_si256((__m256i*)out,s);
    // reverse d into out[8..15]
    __m256i rev=_mm256_shuffle_epi32(d,_MM_SHUFFLE(0,1,2,3));
    rev=_mm256_permute2x128_si256(rev,rev,1);
    _mm256_storeu_si256((__m256i*)(out+8),rev);
}
#else
static inline void mc_dct1d_inv(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    const mc_fi32 rnd=(mc_fi32)1<<(MC_DCT_Q-1); const int S=MC_DCT_N, H=S/2;
    mc_fi32 acc_e[8], acc_o[8];
    for(int n=0;n<H;++n){ acc_e[n]=rnd; acc_o[n]=0; }
    for(int j=0;j<8;++j){
        mc_fi32 ve=in[2*j];
        if(ve) for(int n=0;n<H;++n) acc_e[n]+=g_cm_e[j][n]*ve;
        mc_fi32 vo=in[2*j+1];
        if(vo) for(int n=0;n<H;++n) acc_o[n]+=g_cm_o[j][n]*vo;
    }
    for(int n=0;n<H;++n){ out[n]=(acc_e[n]+acc_o[n])>>MC_DCT_Q; out[S-1-n]=(acc_e[n]-acc_o[n])>>MC_DCT_Q; }
}
#endif

// cache-blocked rotate (z,y,x)->(x,z,y): dst[(x*S+z)*S+y] = src[(z*S+y)*S+x].
// Per fixed z this is a 16x16 i32 transpose (src rows y, stride S -> dst rows
// x, stride S*S); done as 4x4 register tiles on NEON, scalar tiles otherwise.
#if MC_SIMD_NEON
static inline void mc_rot(const mc_fi32 *restrict src, mc_fi32 *restrict dst){
    const int S=MC_DCT_N;
    for(int z=0;z<S;++z){
        const mc_fi32 *sp=src+(size_t)z*S*S;
        mc_fi32 *dp=dst+(size_t)z*S;
        for(int y=0;y<S;y+=4)
        for(int x=0;x<S;x+=4){
            int32x4_t r0=vld1q_s32(sp+(size_t)(y+0)*S+x);
            int32x4_t r1=vld1q_s32(sp+(size_t)(y+1)*S+x);
            int32x4_t r2=vld1q_s32(sp+(size_t)(y+2)*S+x);
            int32x4_t r3=vld1q_s32(sp+(size_t)(y+3)*S+x);
            int32x4_t t0=vtrn1q_s32(r0,r1), t1=vtrn2q_s32(r0,r1);
            int32x4_t t2=vtrn1q_s32(r2,r3), t3=vtrn2q_s32(r2,r3);
            int32x4_t c0=vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(t0),vreinterpretq_s64_s32(t2)));
            int32x4_t c1=vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(t1),vreinterpretq_s64_s32(t3)));
            int32x4_t c2=vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(t0),vreinterpretq_s64_s32(t2)));
            int32x4_t c3=vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(t1),vreinterpretq_s64_s32(t3)));
            vst1q_s32(dp+(size_t)(x+0)*S*S+y, c0);
            vst1q_s32(dp+(size_t)(x+1)*S*S+y, c1);
            vst1q_s32(dp+(size_t)(x+2)*S*S+y, c2);
            vst1q_s32(dp+(size_t)(x+3)*S*S+y, c3);
        }
    }
}
#else
#define MC_ROT_TILE 8
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
#endif
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
typedef struct {
    mc_fi32 in[16*16*16] __attribute__((aligned(MC_DCT_ALIGN)));
    mc_fi32 a [16*16*16] __attribute__((aligned(MC_DCT_ALIGN)));
    mc_fi32 b [16*16*16] __attribute__((aligned(MC_DCT_ALIGN)));
} mc_dct_tls_t;
static _Thread_local mc_dct_tls_t g_dct_tls;
static void mc_dct3_fwd(const float *restrict blk, float *restrict coef){
    const int n=MC_DCT_N*MC_DCT_N*MC_DCT_N;
    mc_dct_tls_t *D=&g_dct_tls;
    mc_fi32 *in=D->in, *a=D->a, *b=D->b;
    for(int i=0;i<n;++i) in[i]=(mc_fi32)lrintf(blk[i]);
    mc_lines_fwd_to(in,a); mc_rot(a,b);
    mc_lines_fwd(b);       mc_rot(b,a);
    mc_lines_fwd(a);       mc_rot(a,b);
    for(int i=0;i<n;++i) coef[i]=(float)b[i];
}
static void mc_dct3_inv(const float *restrict coef, float *restrict blk){
    const int n=MC_DCT_N*MC_DCT_N*MC_DCT_N;
    mc_dct_tls_t *D=&g_dct_tls;
    mc_fi32 *in=D->in, *a=D->a, *b=D->b;
    for(int i=0;i<n;++i) in[i]=(mc_fi32)lrintf(coef[i]);
    mc_lines_inv_to(in,a); mc_rot(a,b);
    mc_lines_inv(b);       mc_rot(b,a);
    mc_lines_inv(a);       mc_rot(a,b);
    for(int i=0;i<n;++i) blk[i]=(float)b[i];
}
// variant taking PREPARED i32 coefficients (decoder fuses dequantization into
// the input conversion) and returning the raw i32 spatial result.
static void mc_dct3_inv_i32(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    mc_fi32 *a=g_dct_tls.a;
    mc_lines_inv_to(in,a); mc_rot(a,out);
    mc_lines_inv(out);     mc_rot(out,a);
    mc_lines_inv(a);       mc_rot(a,out);
}

#endif

// ============================================================================
// mc_rangecoder.h — CABAC-style binary range coder + DCT coefficient context coder
// (matter-compressor). Adaptive bit model + bypass bits; coefficients coded in
// ascending-frequency scan order with an EOB, per-band significance contexts, and a
// unary+Exp-Golomb magnitude coder.
// ============================================================================
#ifndef MC_RANGECODER_H
#define MC_RANGECODER_H
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

typedef uint8_t  rc_u8;  typedef int16_t rc_i16; typedef int32_t rc_i32;
typedef uint32_t rc_u32; typedef uint64_t rc_u64;

typedef struct { rc_u8 *buf; size_t cap, len; rc_u64 low; rc_u32 range; rc_u8 cache; rc_u64 cache_size; } rc_enc;
typedef struct { const rc_u8 *buf; size_t len, pos; rc_u32 code, range; } rc_dec;
typedef struct { uint16_t p0; } ctx_t;
static inline void ctx_init(ctx_t *c){ c->p0 = 1u<<11; }
static inline void ctx_init_p(ctx_t *c, uint16_t p0){ c->p0 = p0; }
#define RC_TOP (1u<<24)

// ---- trained context priors -------------------------------------------------
// Per-block contexts reset every block, so without priors every adaptive bin
// starts at p=0.5 and is no better than bypass for the first ~32 bins. These
// tables are trained on PHercParis4 2.4um (fysics-masked) at q in {1,3,6,12}
// via tools/mc_train (build with -DMC_TRAIN to retrain).
// Context classes (training bucket ids):
enum { RCC_SIG=0, RCC_MAG=1, RCC_EOB=2, RCC_MASK=3, RCC_MASKU=4, RCC_MASKA=5, RCC_FLAG=6, RCC_DC=7, RCC_NCLS=8 };
#define RCC_SLOTS 32
#ifdef MC_TRAIN
extern long mc_tr_n[RCC_NCLS][RCC_SLOTS], mc_tr_z[RCC_NCLS][RCC_SLOTS];
#define RC_TRAIN(cls,slot,bit) (mc_tr_n[cls][slot]++, mc_tr_z[cls][slot]+=((bit)==0))
#else
#define RC_TRAIN(cls,slot,bit) ((void)0)
#endif
// priors (p0 = P(bit==0) in 1/4096 units), trained at q=1 and q=12 on
// PHercParis4 2.4um (fysics-masked) via tools/mc_train; rc_prior_build()
// interpolates in log2(q) into g_pri[][] (the decoder knows q, so this costs
// no side information). 2048 = untrained slot.
#define RC_NSLOT 32
static const uint16_t RC_PLO[8][RC_NSLOT]={
  /*SIG*/ {115,110,112,144,3866,3522,3330,948,3939,3723,3483,2383,4007,3795,3575,3130,4042,3912,3805,3686,4064,4064,4028,4064,2048,2048,2048,2048,2048,2048,2048,2048},
  /*MAG*/ {1996,1234,948,743,595,539,441,421,333,358,297,305,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*EOB*/ {32,32,32,32,32,32,32,32,32,92,285,2295,4064,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*MASK*/ {4025,3064,2995,827,2107,473,576,50,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*MSKU*/ {250,1713,2961,2991,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*MSKA*/ {3936,32,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*FLAG*/ {1023,4064,4064,4064,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*DC*/ {3873,366,1820,2056,2040,2055,2041,2045,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048}
};
static const uint16_t RC_PHI[8][RC_NSLOT]={
  /*SIG*/ {456,335,535,1133,3950,3755,3590,2892,4064,4051,3990,3686,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*MAG*/ {2121,1410,1081,863,746,631,569,508,465,431,399,370,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*EOB*/ {59,32,32,32,32,63,92,315,2273,4064,4064,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*MASK*/ {4025,3064,2995,827,2107,473,576,50,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*MSKU*/ {250,1713,2961,2991,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*MSKA*/ {3936,32,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*FLAG*/ {1023,4064,4064,4064,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*DC*/ {3873,366,1820,2056,2040,2055,2041,2045,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048}
};
static _Thread_local uint16_t g_pri[8][RC_NSLOT];
static _Thread_local float g_pri_q = -1.0f;
// Per-volume prior override (format v6: a trained-prior blob stored in the
// archive replaces the baked corpus tables). Process-global, set at open
// before decode threads start; a generation counter forces per-thread rebuild.
static uint16_t g_plo_ovr[8][RC_NSLOT], g_phi_ovr[8][RC_NSLOT];
static int g_pri_ovr = 0;
static int g_pri_gen = 1;
static _Thread_local int g_pri_seen = 0;
static void rc_set_priors(const uint16_t *plo, const uint16_t *phi){
    if(plo&&phi){
        memcpy(g_plo_ovr,plo,sizeof g_plo_ovr);
        memcpy(g_phi_ovr,phi,sizeof g_phi_ovr);
        g_pri_ovr=1;
    } else g_pri_ovr=0;
    g_pri_gen++;
}
static void rc_prior_build(float q){
    if(g_pri_q==q && g_pri_seen==g_pri_gen) return;
    g_pri_seen=g_pri_gen;
    float lo=0.0f, hi=3.585f;                       // log2(1) .. log2(12)
    float w=(q<=1.0f)?0.0f:((float)(log(q)/log(2.0))-lo)/(hi-lo);
    if(w<0)w=0; if(w>1)w=1;
    const uint16_t (*tlo)[RC_NSLOT] = g_pri_ovr ? (const uint16_t(*)[RC_NSLOT])g_plo_ovr : RC_PLO;
    const uint16_t (*thi)[RC_NSLOT] = g_pri_ovr ? (const uint16_t(*)[RC_NSLOT])g_phi_ovr : RC_PHI;
    for(int c=0;c<8;++c)for(int s=0;s<RC_NSLOT;++s)
        g_pri[c][s]=(uint16_t)(tlo[c][s]+(thi[c][s]-tlo[c][s])*w+0.5f);
    g_pri_q=q;
}
#define RC_PRIOR_SIG   g_pri[0]
#define RC_PRIOR_MAG   g_pri[1]
#define RC_PRIOR_EOB   g_pri[2]
#define RC_PRIOR_MASK  g_pri[3]
#define RC_PRIOR_MASKU g_pri[4]
#define RC_PRIOR_MASKA g_pri[5]
#define RC_PRIOR_FLAG  g_pri[6]
#define RC_PRIOR_DC    g_pri[7]

static void enc_init(rc_enc *e, rc_u8 *buf, size_t cap){ e->buf=buf;e->cap=cap;e->len=0;e->low=0;e->range=0xFFFFFFFFu;e->cache=0;e->cache_size=1; }
static void enc_putbyte(rc_enc *e, rc_u8 b){ if(e->len<e->cap) e->buf[e->len++]=b; else e->len++; }
static void enc_shift_low(rc_enc *e){
    if((rc_u32)(e->low>>32)!=0 || e->low<0xFF000000ull){
        rc_u8 carry=(rc_u8)(e->low>>32);
        do{ enc_putbyte(e,(rc_u8)(e->cache+carry)); e->cache=0xFF; }while(--e->cache_size);
        e->cache=(rc_u8)(e->low>>24);
    }
    e->cache_size++; e->low=(e->low<<8)&0xFFFFFFFFull;
}
static void enc_bit(rc_enc *e, ctx_t *c, int bit){
    rc_u32 r0=(e->range>>12)*c->p0;
    if(bit==0){ e->range=r0; c->p0=(uint16_t)(c->p0+((4096-c->p0)>>4)); }
    else { e->low+=r0; e->range-=r0; c->p0=(uint16_t)(c->p0-(c->p0>>4)); }
    while(e->range<RC_TOP){ enc_shift_low(e); e->range<<=8; }
}
static void enc_bypass(rc_enc *e, int bit){ e->range>>=1; if(bit) e->low+=e->range; while(e->range<RC_TOP){ enc_shift_low(e); e->range<<=8; } }
static void enc_flush(rc_enc *e){ for(int i=0;i<5;++i) enc_shift_low(e); }

static void dec_init(rc_dec *d,const rc_u8*buf,size_t len){ d->buf=buf;d->len=len;d->pos=0;d->code=0;d->range=0xFFFFFFFFu; for(int i=0;i<5;++i){ rc_u8 b=(d->pos<d->len)?d->buf[d->pos++]:0; d->code=(d->code<<8)|b; } }
static int dec_bit(rc_dec *d, ctx_t *c){
    rc_u32 r0=(d->range>>12)*c->p0; int bit;
    if(d->code<r0){ d->range=r0;bit=0; c->p0=(uint16_t)(c->p0+((4096-c->p0)>>4)); }
    else { d->code-=r0; d->range-=r0; bit=1; c->p0=(uint16_t)(c->p0-(c->p0>>4)); }
    while(d->range<RC_TOP){ rc_u8 b=(d->pos<d->len)?d->buf[d->pos++]:0; d->code=(d->code<<8)|b; d->range<<=8; }
    return bit;
}
static int dec_bypass(rc_dec *d){ d->range>>=1; int bit=(d->code>=d->range); if(bit)d->code-=d->range; while(d->range<RC_TOP){ rc_u8 b=(d->pos<d->len)?d->buf[d->pos++]:0; d->code=(d->code<<8)|b; d->range<<=8; } return bit; }

// batched bypass: k equiprobable bits in one renorm round (bit-compatible
// with k single bypasses only when both sides batch identically — they do).
static void enc_bypass_n(rc_enc *e, rc_u32 v, int k){
    while(k>16){ enc_bypass_n(e,(v>>(k-16))&0xFFFF,16); k-=16; }
    if(!k) return;
    e->range>>=k;
    e->low+=(rc_u64)(v&((1u<<k)-1))*e->range;
    while(e->range<RC_TOP){ enc_shift_low(e); e->range<<=8; }
}
static rc_u32 dec_bypass_n(rc_dec *d, int k){
    rc_u32 v=0;
    while(k>16){ v=(v<<16)|dec_bypass_n(d,16); k-=16; }
    if(!k) return v;
    d->range>>=k;
    rc_u32 q=d->code/d->range;
    rc_u32 m=(1u<<k)-1; if(q>m)q=m;
    d->code-=q*d->range;
    while(d->range<RC_TOP){ rc_u8 b=(d->pos<d->len)?d->buf[d->pos++]:0; d->code=(d->code<<8)|b; d->range<<=8; }
    return (v<<k)|q;
}

// Exp-Golomb in bypass bits (order 0), v >= 0.
static void enc_eg(rc_enc*e,rc_u32 v){
    rc_u32 nb=0,t=v+1; while(t>1){t>>=1;nb++;}
    for(rc_u32 i=0;i<nb;++i)enc_bypass(e,1); enc_bypass(e,0);
    if(nb) enc_bypass_n(e,(v+1)&((1u<<nb)-1),(int)nb);
}
static rc_u32 dec_eg(rc_dec*d){
    rc_u32 nb=0; while(dec_bypass(d))nb++;
    if(!nb) return 0;
    return ((1u<<nb)|dec_bypass_n(d,(int)nb))-1;
}

// --- coefficient context coder (block size S) ---
// Ascending-band scan with adaptive last-sig (EOB), significance conditioned on
// (band, recent significance density), and an adaptive-unary + Exp-Golomb
// magnitude ladder. Group-skip flags and bypass Rice remainders were tried
// (HEVC-style) and measured WORSE on scroll data: the EOB already truncates the
// sparse tail, and the skewed magnitude distribution wants adaptive bins, not
// bypass remainders.
#define NB_BANDS 8       // L1-frequency band buckets
#define MAGCTX   12      // unary magnitude ladder contexts
typedef struct {
    ctx_t sig[NB_BANDS*4];     // significance: band x min(recent sig count,3)
    ctx_t mag[MAGCTX];         // unary magnitude ladder (band-conditioning the
                               // first rungs was measured ratio-neutral: per-
                               // block adaptation already learns the block's
                               // own magnitude distribution)
} atom_ctx;
static void atom_ctx_init(atom_ctx *a){
    for(int i=0;i<NB_BANDS*4;++i) ctx_init_p(&a->sig[i],RC_PRIOR_SIG[i]);
    for(int i=0;i<MAGCTX;++i)     ctx_init_p(&a->mag[i],RC_PRIOR_MAG[i]);
}
static void enc_magnitude(rc_enc*e,atom_ctx*ac,rc_u32 m){
    ctx_t*mag=ac->mag; rc_u32 v=m-1,k=0;
    while(k<(rc_u32)(MAGCTX-1)&&v>0){ RC_TRAIN(RCC_MAG,k,1); enc_bit(e,&mag[k],1); v-=1;k++; if(v==0){RC_TRAIN(RCC_MAG,k,0); enc_bit(e,&mag[k],0);return;} }
    if(v==0){ RC_TRAIN(RCC_MAG,k,0); enc_bit(e,&mag[k],0); return; }
    RC_TRAIN(RCC_MAG,k,1); enc_bit(e,&mag[k],1); rc_u32 x=v,nbits=0,tt=x+1; while(tt>1){tt>>=1;nbits++;}
    for(rc_u32 i=0;i<nbits;++i)enc_bypass(e,1); enc_bypass(e,0);
    if(nbits) enc_bypass_n(e,(x+1)&((1u<<nbits)-1),(int)nbits);
}
static rc_u32 dec_magnitude(rc_dec*d,atom_ctx*ac){
    ctx_t*mag=ac->mag; rc_u32 v=0,k=0;
    while(k<(rc_u32)(MAGCTX-1)){ if(dec_bit(d,&mag[k])){v+=1;k++;} else return v+1; }
    if(!dec_bit(d,&mag[k])) return v+1;
    rc_u32 nbits=0; while(dec_bypass(d))nbits++;
    rc_u32 x = nbits ? ((1u<<nbits)|dec_bypass_n(d,(int)nbits))-1 : 0;
    return v+x+1;
}

// per-size ascending-L1-frequency scan tables, built lazily (indexed by log2 S).
// Build is serialized: concurrent encode workers used to race the ready flag,
// double-build, and leak the losing table (caught by LeakSanitizer).
static uint16_t *g_scanS[6]; static _Atomic int g_scanS_ready[6];
static pthread_mutex_t g_scanS_mu = PTHREAD_MUTEX_INITIALIZER;
static int scanS_cmp_S;
static int scanS_cmp(const void*pa,const void*pb){
    rc_u32 a=*(const rc_u32*)pa,b=*(const rc_u32*)pb; int S=scanS_cmp_S;
    rc_u32 fa=(a/(S*S))+((a/S)%S)+(a%S), fb=(b/(S*S))+((b/S)%S)+(b%S);
    if(fa!=fb) return (int)fa-(int)fb; return (int)a-(int)b;
}
static void scanS_build(int S){
    int l=0,t=S; while(t>1){t>>=1;l++;}
    if(atomic_load_explicit(&g_scanS_ready[l],memory_order_acquire)) return;
    pthread_mutex_lock(&g_scanS_mu);
    if(!atomic_load_explicit(&g_scanS_ready[l],memory_order_relaxed)){
        int n=S*S*S; rc_u32 *ord=malloc(n*sizeof(rc_u32)); for(int i=0;i<n;++i)ord[i]=i;
        scanS_cmp_S=S; qsort(ord,n,sizeof(rc_u32),scanS_cmp);
        uint16_t *tab=malloc(n*sizeof(uint16_t));
        for(int i=0;i<n;++i)tab[i]=(uint16_t)ord[i];
        free(ord);
        g_scanS[l]=tab;
        atomic_store_explicit(&g_scanS_ready[l],1,memory_order_release);
    }
    pthread_mutex_unlock(&g_scanS_mu);
}
static inline int band_of_S(rc_u32 idx,int S){
    rc_u32 cz=idx/(S*S),cy=(idx/S)%S,cx=idx%S, freq=cz+cy+cx;
    int b=(int)(freq*NB_BANDS/(3u*S)); if(b>=NB_BANDS)b=NB_BANDS-1; return b;
}
// bit-length of eob (well-predicted: most blocks have similar sparsity), then the
// low bits in bypass. v in [0, n]; prefix k = MSB position + 1 (0 -> v==0).
#define EOB_CTX 14
typedef struct { ctx_t pfx[EOB_CTX]; } eob_ctx;
static void eob_ctx_init(eob_ctx*c){ for(int i=0;i<EOB_CTX;++i) ctx_init_p(&c->pfx[i],RC_PRIOR_EOB[i]); }
static void enc_eob(rc_enc*e,eob_ctx*c,rc_u32 v,int n){
    int kmax=0; while((1u<<kmax)<=(rc_u32)n) kmax++;          // 13 for n=4096
    int k=0; while((1u<<k)<=v) k++;                            // MSB+1 (0 for v=0)
    for(int i=0;i<k;++i){ RC_TRAIN(RCC_EOB,i,1); enc_bit(e,&c->pfx[i],1); }
    if(k<kmax){ RC_TRAIN(RCC_EOB,k,0); enc_bit(e,&c->pfx[k],0); }
    if(k>1) enc_bypass_n(e,v&((1u<<(k-1))-1),k-1);             // suffix below the MSB
}
static rc_u32 dec_eob(rc_dec*d,eob_ctx*c,int n){
    int kmax=0; while((1u<<kmax)<=(rc_u32)n) kmax++;
    int k=0; while(k<kmax && dec_bit(d,&c->pfx[k])) k++;
    if(k==0) return 0;
    if(k==1) return 1;
    return (1u<<(k-1))|dec_bypass_n(d,k-1);
}

// encode/decode quantized levels q[S^3] (raster).
static void enc_block_coefs(rc_enc*e,const rc_i16*q,int S){
    scanS_build(S); int l=0,t=S; while(t>1){t>>=1;l++;} const uint16_t*scan=g_scanS[l];
    int n=S*S*S; atom_ctx ac; atom_ctx_init(&ac);
    eob_ctx ec; eob_ctx_init(&ec);
    rc_u32 eob=0; for(rc_u32 p=n;p-->0;){ if(q[scan[p]]!=0){eob=p+1;break;} }
    enc_eob(e,&ec,eob,n);
    rc_u32 hist=0;                                  // last 16 sig decisions
    for(rc_u32 p=0;p<eob;++p){
        rc_u32 idx=scan[p]; int b=band_of_S(idx,S); rc_i16 v=q[idx];
        int dens=__builtin_popcount(hist&0xFFFFu); dens=dens<3?dens:3;
        if(p!=eob-1){                               // last sig position is nonzero by definition
            int sctx=b*4+dens;
            RC_TRAIN(RCC_SIG,sctx,v!=0); enc_bit(e,&ac.sig[sctx],v!=0);
        }
        hist=(hist<<1)|(v!=0);
        if(!v) continue;
        rc_u32 m=(rc_u32)(v<0?-v:v);
        enc_magnitude(e,&ac,m);
        enc_bypass(e,v<0?1:0);
    }
}
// decodes levels; also reports the per-axis extent of nonzero coefficients
// (ext[0..2] = max z,y,x; -1 if none) so the inverse DCT can skip empty space.
static void dec_block_coefs_ext(rc_dec*d,rc_i16*q,int S,int ext[3]){
    scanS_build(S); int l=0,t=S; while(t>1){t>>=1;l++;} const uint16_t*scan=g_scanS[l];
    int n=S*S*S; atom_ctx ac; atom_ctx_init(&ac); memset(q,0,n*sizeof(rc_i16));
    eob_ctx ec; eob_ctx_init(&ec);
    rc_u32 eob=dec_eob(d,&ec,n); if(eob>(rc_u32)n)eob=n;
    rc_u32 hist=0;
    int ez=-1,ey=-1,ex=-1;
    for(rc_u32 p=0;p<eob;++p){
        rc_u32 idx=scan[p]; int b=band_of_S(idx,S);
        int dens=__builtin_popcount(hist&0xFFFFu); dens=dens<3?dens:3;
        int sig;
        if(p==eob-1) sig=1;
        else { int sctx=b*4+dens; sig=dec_bit(d,&ac.sig[sctx]); }
        hist=(hist<<1)|sig;
        if(!sig) continue;
        rc_u32 m=dec_magnitude(d,&ac);
        int neg=dec_bypass(d);
        q[idx]=(rc_i16)(neg?-(rc_i32)m:(rc_i32)m);
        int cz=(int)(idx/(rc_u32)(S*S)), cy=(int)((idx/(rc_u32)S)%(rc_u32)S), cx=(int)(idx%(rc_u32)S);
        if(cz>ez)ez=cz; if(cy>ey)ey=cy; if(cx>ex)ex=cx;
    }
    ext[0]=ez; ext[1]=ey; ext[2]=ex;
}
static void dec_block_coefs(rc_dec*d,rc_i16*q,int S){
    int ext[3]; dec_block_coefs_ext(d,q,S,ext);
}
#endif

// mc_xxhash.h — minimal XXH64 (Yann Collet's xxHash, 64-bit variant) for
// chunk-blob integrity checksums. Public-domain-style reimplementation.
#include <stdint.h>
#include <string.h>

#define XXP1 0x9E3779B185EBCA87ULL
#define XXP2 0xC2B2AE3D27D4EB4FULL
#define XXP3 0x165667B19E3779F9ULL
#define XXP4 0x85EBCA77C2B2AE63ULL
#define XXP5 0x27D4EB2F165667C5ULL
static inline uint64_t xx_rotl(uint64_t x,int r){ return (x<<r)|(x>>(64-r)); }
static inline uint64_t xx_round(uint64_t acc,uint64_t in){ acc+=in*XXP2; acc=xx_rotl(acc,31); return acc*XXP1; }
static inline uint64_t xx_merge(uint64_t acc,uint64_t v){ acc^=xx_round(0,v); return acc*XXP1+XXP4; }
static inline uint64_t xx_read64(const uint8_t*p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint32_t xx_read32(const uint8_t*p){ uint32_t v; memcpy(&v,p,4); return v; }
static uint64_t mc_xxh64(const void *data, size_t len, uint64_t seed){
    const uint8_t *p=(const uint8_t*)data, *end=p+len;
    uint64_t h;
    if(len>=32){
        uint64_t v1=seed+XXP1+XXP2, v2=seed+XXP2, v3=seed, v4=seed-XXP1;
        const uint8_t *limit=end-32;
        do{
            v1=xx_round(v1,xx_read64(p));    p+=8;
            v2=xx_round(v2,xx_read64(p));    p+=8;
            v3=xx_round(v3,xx_read64(p));    p+=8;
            v4=xx_round(v4,xx_read64(p));    p+=8;
        }while(p<=limit);
        h=xx_rotl(v1,1)+xx_rotl(v2,7)+xx_rotl(v3,12)+xx_rotl(v4,18);
        h=xx_merge(h,v1); h=xx_merge(h,v2); h=xx_merge(h,v3); h=xx_merge(h,v4);
    } else h=seed+XXP5;
    h+=(uint64_t)len;
    while(p+8<=end){ h^=xx_round(0,xx_read64(p)); h=xx_rotl(h,27)*XXP1+XXP4; p+=8; }
    if(p+4<=end){ h^=(uint64_t)xx_read32(p)*XXP1; h=xx_rotl(h,23)*XXP2+XXP3; p+=4; }
    while(p<end){ h^=(*p)*XXP5; h=xx_rotl(h,11)*XXP1; ++p; }
    h^=h>>33; h*=XXP2; h^=h>>29; h*=XXP3; h^=h>>32;
    return h;
}

// ============================================================================
// mc_codec.c — matter-compressor block codec implementation. See mc_codec.h.
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N3 (MC_BLK*MC_BLK*MC_BLK)
#define MC_GRID3_F 4096

// Quality state is THREAD-LOCAL: format v6 stores q per chunk, so concurrent
// decodes of different-q chunks each keep their own step/prior tables (rebuilt
// only when the thread's q actually changes — once per chunk at most).
// Consolidated per-thread codec scratch. On macOS/ELF every _Thread_local
// access can compile to an opaque TLV-accessor CALL; scattered through the
// hot functions those calls forced the range-coder state to spill to the
// stack around each one (objdump: 8 blr + 102 sp-stores in mc_dec_block).
// One struct -> one TLV access at function entry -> hot loops run call-free
// with the coder state held in registers.
typedef struct {
    // decode
    mc_u8  air[N3];
    rc_i16 ql[N3];
    float  coef[N3], blk[N3];
    mc_fi32 qin[N3] __attribute__((aligned(64)));
    mc_fi32 qout[N3] __attribute__((aligned(64)));
    // encode
    float  eblk[N3], ecoef[N3];
    rc_u8  scratch[N3*4+1024];
    uint16_t cpos[N3]; mc_i32 cdel[N3];
    float  rcoef[N3], rblk[N3];
} mc_tls_t;
static _Thread_local mc_tls_t g_tls;

static _Thread_local float g_quality = 8.0f;
static int   g_max_err = 0;            // 0 = corrections off
// per-coefficient quant step table (quality * hf_weight), rebuilt when quality
// changes. powf per coefficient was 20%+ of encode AND decode time.
static _Thread_local float g_step_tab[N3];
static _Thread_local float g_rstep_tab[N3];   // 1/step: quant uses mul, not div
static _Thread_local float g_step_q = -1.0f;
static void step_tab_build(void);
void  mc_set_quality(float q){ g_quality = q; step_tab_build(); }
float mc_get_quality(void){ return g_quality; }
void  mc_set_max_error(int tau){ g_max_err = tau<0?0:tau; }
int   mc_get_max_error(void){ return g_max_err; }
void  mc_codec_init(void){ mc_dct_init(); }
void  mc_codec_set_priors(const uint16_t *plo, const uint16_t *phi){ rc_set_priors(plo,phi); }

void mc_buf_put(mc_buf *b, const void *s, size_t n){
    if(b->len+n > b->cap){ size_t nc=b->cap?b->cap*2:1<<16; while(nc<b->len+n)nc*=2; b->p=realloc(b->p,nc); b->cap=nc; }
    memcpy(b->p+b->len,s,n); b->len+=n;
}

// frozen quant: dead-zone, step = quality*(1+L1freq)^MC_HF_EXP
static inline float hf_weight(int cz,int cy,int cx){ return powf(1.0f+(float)(cz+cy+cx), MC_HF_EXP); }
static void step_tab_build(void){
    rc_prior_build(g_quality);
    if(g_step_q==g_quality) return;
    for(int cz=0;cz<MC_BLK;++cz)for(int cy=0;cy<MC_BLK;++cy)for(int cx=0;cx<MC_BLK;++cx){
        int i=(cz*MC_BLK+cy)*MC_BLK+cx;
        g_step_tab[i]=g_quality*hf_weight(cz,cy,cx);
        g_rstep_tab[i]=1.0f/g_step_tab[i];
    }
    g_step_q=g_quality;
}
static inline mc_i32 quant_one(float c, float step){
    float dz=MC_DZ_FRAC*step, a=fabsf(c); mc_i32 lv=0;
    if(a>=dz) lv=(mc_i32)((a-dz)/step+1.0f);
    return c<0?-lv:lv;
}
static inline float deq_one(mc_i32 lv, float step){
    if(!lv) return 0.0f;
    float a=(float)(lv<0?-lv:lv); float r=(a-1.0f)*step+MC_DZ_FRAC*step+0.40f*step;
    return lv<0?-r:r;
}

// block-mask surface coder: 3-neighbor (z-1,y-1,x-1) context bit coder over the
// block's 16^3 air mask (air = vox==0). Out-of-block neighbors read as 0. Codes
// into the block's single range-coder stream (shared with the coefficients).
// Two-level mask: per 4^3 subcube a class (uniform-air / uniform-material /
// mixed; ~2 adaptive bins), then per-voxel 3-neighbor context bins only inside
// mixed subcubes. Subcubes and voxels both scan in raster order so the causal
// (z-1,y-1,x-1) context always reads already-coded mask values. Cuts mask bins
// ~3-6x on boundary blocks (most blocks of a masked scroll volume).
#define MSUB 4
static void enc_blockmask(rc_enc *e, const mc_u8 *vox){
    ctx_t ctx[8]; for(int i=0;i<8;++i) ctx_init_p(&ctx[i],RC_PRIOR_MASK[i]);
    ctx_t cu[4];  for(int i=0;i<4;++i) ctx_init_p(&cu[i],RC_PRIOR_MASKU[i]);
    ctx_t ca[2];  for(int i=0;i<2;++i) ctx_init_p(&ca[i],RC_PRIOR_MASKA[i]);
    const int S=MC_BLK, G=S/MSUB;
    static _Thread_local mc_u8 air[N3];
    mc_u8 sc[4*4*4];                       // subcube class: 0=material,1=air,2=mixed
    for(int sz=0;sz<G;++sz)for(int sy=0;sy<G;++sy)for(int sx=0;sx<G;++sx){
        int si=(sz*G+sy)*G+sx;
        int nair_s=0;
        for(int z=0;z<MSUB;++z)for(int y=0;y<MSUB;++y)for(int x=0;x<MSUB;++x)
            nair_s += !vox[((sz*MSUB+z)*S+(sy*MSUB+y))*S+(sx*MSUB+x)];
        int uni = (nair_s==0 || nair_s==MSUB*MSUB*MSUB);
        int nmix=0, nairn=0, nn=0;
        if(sz){ int c=sc[si-G*G]; nn++; nmix+=c==2; nairn+=c==1; }
        if(sy){ int c=sc[si-G];   nn++; nmix+=c==2; nairn+=c==1; }
        if(sx){ int c=sc[si-1];   nn++; nmix+=c==2; nairn+=c==1; }
        int uctx=nmix<3?nmix:3, actx=nairn?1:0;
        RC_TRAIN(RCC_MASKU,uctx,uni); enc_bit(e,&cu[uctx],uni);
        if(uni){
            int isair = nair_s>0;
            RC_TRAIN(RCC_MASKA,actx,isair); enc_bit(e,&ca[actx],isair);
            for(int z=0;z<MSUB;++z)for(int y=0;y<MSUB;++y)for(int x=0;x<MSUB;++x)
                air[((sz*MSUB+z)*S+(sy*MSUB+y))*S+(sx*MSUB+x)]=(mc_u8)isair;
        } else {
            for(int z=0;z<MSUB;++z)for(int y=0;y<MSUB;++y)for(int x=0;x<MSUB;++x){
                int gz=sz*MSUB+z, gy=sy*MSUB+y, gx=sx*MSUB+x;
                int i=(gz*S+gy)*S+gx;
                int a=!vox[i];
                int nz_= gz?air[i-S*S]:0, ny_= gy?air[i-S]:0, nx_= gx?air[i-1]:0;
                int cc=(nz_<<2)|(ny_<<1)|nx_;
                RC_TRAIN(RCC_MASK,cc,a);
                enc_bit(e,&ctx[cc],a);
                air[i]=(mc_u8)a;
            }
        }
        sc[si]=(mc_u8)(uni? (nair_s>0?1:0) : 2);
    }
}
static void dec_blockmask(rc_dec *d, mc_u8 *air){
    ctx_t ctx[8]; for(int i=0;i<8;++i) ctx_init_p(&ctx[i],RC_PRIOR_MASK[i]);
    ctx_t cu[4];  for(int i=0;i<4;++i) ctx_init_p(&cu[i],RC_PRIOR_MASKU[i]);
    ctx_t ca[2];  for(int i=0;i<2;++i) ctx_init_p(&ca[i],RC_PRIOR_MASKA[i]);
    const int S=MC_BLK, G=S/MSUB;
    mc_u8 sc[4*4*4];
    for(int sz=0;sz<G;++sz)for(int sy=0;sy<G;++sy)for(int sx=0;sx<G;++sx){
        int si=(sz*G+sy)*G+sx;
        int nmix=0, nairn=0;
        if(sz){ int c=sc[si-G*G]; nmix+=c==2; nairn+=c==1; }
        if(sy){ int c=sc[si-G];   nmix+=c==2; nairn+=c==1; }
        if(sx){ int c=sc[si-1];   nmix+=c==2; nairn+=c==1; }
        int uctx=nmix<3?nmix:3, actx=nairn?1:0;
        int uni=dec_bit(d,&cu[uctx]);
        int isair_u=0;
        if(uni){
            int isair=dec_bit(d,&ca[actx]); isair_u=isair;
            for(int z=0;z<MSUB;++z)for(int y=0;y<MSUB;++y)for(int x=0;x<MSUB;++x)
                air[((sz*MSUB+z)*S+(sy*MSUB+y))*S+(sx*MSUB+x)]=(mc_u8)isair;
        } else {
            for(int z=0;z<MSUB;++z)for(int y=0;y<MSUB;++y)for(int x=0;x<MSUB;++x){
                int gz=sz*MSUB+z, gy=sy*MSUB+y, gx=sx*MSUB+x;
                int i=(gz*S+gy)*S+gx;
                int nz_= gz?air[i-S*S]:0, ny_= gy?air[i-S]:0, nx_= gx?air[i-1]:0;
                air[i]=(mc_u8)dec_bit(d,&ctx[(nz_<<2)|(ny_<<1)|nx_]);
            }
        }
        sc[si]=(mc_u8)(uni? (isair_u?1:0) : 2);
    }
}

// ---- decode-side deblocking (optional, no format change) ---------------------
// Clamped 2-tap filter across every 16-aligned block face, per axis. Gated on
// a quality-scaled flatness test (only smooth true block seams, keep edges),
// and never touches air (0) voxels.
static inline void mc_db_pair(mc_u8 *p1, mc_u8 *p0, mc_u8 *q0, mc_u8 *q1, int beta, int tc){
    int a=*p0, b=*q0;
    if(!a||!b) return;
    int d=b-a; if(d<0?-d>=beta:d>=beta) return;            // real edge: keep
    int pp=*p1?*p1:a, qq=*q1?*q1:b;
    int delta=(4*(b-a)+(pp-qq)+4)>>3;
    if(delta>tc)delta=tc; if(delta<-tc)delta=-tc;
    int na=a+delta, nb=b-delta;
    if(na<1)na=1; if(na>255)na=255; if(nb<1)nb=1; if(nb>255)nb=255;
    *p0=(mc_u8)na; *q0=(mc_u8)nb;
}
void mc_deblock(mc_u8 *v, int nz, int ny, int nx, float quality){
    int beta=(int)(3.0f*quality+6.0f);                     // flatness gate
    int tc  =(int)(0.5f*quality+1.0f);                     // max correction
    size_t sy=(size_t)nx, szp=(size_t)ny*nx;
    for(int z=0;z<nz;++z)for(int y=0;y<ny;++y)             // X faces
        for(int x=MC_BLK;x<nx;x+=MC_BLK){
            mc_u8 *p=v+(size_t)z*szp+(size_t)y*sy+x;
            mc_db_pair(p-2,p-1,p,p+1,beta,tc);
        }
    for(int z=0;z<nz;++z)for(int y=MC_BLK;y<ny;y+=MC_BLK)  // Y faces
        for(int x=0;x<nx;++x){
            mc_u8 *p=v+(size_t)z*szp+(size_t)y*sy+x;
            mc_db_pair(p-2*sy,p-sy,p,p+sy,beta,tc);
        }
    for(int z=MC_BLK;z<nz;z+=MC_BLK)for(int y=0;y<ny;++y)  // Z faces
        for(int x=0;x<nx;++x){
            mc_u8 *p=v+(size_t)z*szp+(size_t)y*sy+x;
            mc_db_pair(p-2*szp,p-szp,p,p+szp,beta,tc);
        }
}

// ---- per-chunk material-fraction map (4096 nibbles) -------------------------
// Smooth field: each nibble coded as 4 adaptive bins conditioned on the
// previous nibble bucket (0 / 1-14 / 15). ~0.1-0.3% of chunk bytes.
uint32_t mc_enc_fracmap(const uint8_t *frac, uint8_t *out, size_t cap){
    rc_enc e; enc_init(&e,out,cap);
    ctx_t cx[3][4]; for(int b=0;b<3;++b)for(int i=0;i<4;++i) ctx_init(&cx[b][i]);
    int prev=0;
    for(int i=0;i<MC_GRID3_F;++i){
        int v=frac[i]&15, pb=prev==0?0:prev==15?2:1;
        for(int b=3;b>=0;--b) enc_bit(&e,&cx[pb][b],(v>>b)&1);
        prev=v;
    }
    enc_flush(&e);
    return (uint32_t)e.len;
}
void mc_dec_fracmap(const uint8_t *in, uint32_t len, uint8_t *frac){
    rc_dec d; dec_init(&d,in,len);
    ctx_t cx[3][4]; for(int b=0;b<3;++b)for(int i=0;i<4;++i) ctx_init(&cx[b][i]);
    int prev=0;
    for(int i=0;i<MC_GRID3_F;++i){
        int v=0, pb=prev==0?0:prev==15?2:1;
        for(int b=3;b>=0;--b) v|=dec_bit(&d,&cx[pb][b])<<b;
        frac[i]=(uint8_t)v; prev=v;
    }
}

// block payload layout: ONE range-coded stream, nothing else. The stream starts
// with [mixed bit][corr bit][dc 8 bits] all context-coded with
// trained priors (the old raw dc+flags bytes were ~5% of an average payload),
// then the mask bins (if mixed), the coefficients, and the corrections. flags bit0 =
// mixed block; the stream carries the mask bins (if mixed) then the coefficients.
// One stream = one flush (~5B) instead of two streams + a 2B mask length.
int mc_enc_block(const mc_u8 *vox, mc_buf *out, uint32_t *len_out){
    int n=N3, any=0;
    mc_tls_t *T=&g_tls;                 // single TLV access for all scratch
    step_tab_build();
    float *blk=T->eblk, *coef=T->ecoef;
    long sum=0,cnt=0;
#if MC_SIMD_NEON
    {   uint32x4_t s32=vdupq_n_u32(0), c32=vdupq_n_u32(0);
        for(int i=0;i<n;i+=16){
            uint8x16_t v=vld1q_u8(vox+i);
            s32=vpadalq_u16(s32,vpaddlq_u8(v));
            uint8x16_t one=vminq_u8(v,vdupq_n_u8(1));
            c32=vpadalq_u16(c32,vpaddlq_u8(one));
        }
        sum=(long)vaddvq_u32(s32); cnt=(long)vaddvq_u32(c32);
        any=cnt>0;
    }
#else
    // branchless so gcc auto-vectorizes (the old guarded sum/cnt loop was
    // scalar and ~6% of encode on x86)
    {   int s_=0, c_=0;
        for(int i=0;i<n;++i){ s_+=vox[i]; c_+=vox[i]!=0; }
        sum=s_; cnt=c_; any=c_>0;
    }
#endif
    if(!any||!cnt){ *len_out=0; return 0; }
    int dc = (int)((sum+cnt/2)/cnt);                  // DC over material only
    int nair = n-(int)cnt;                            // air = vox==0
#if MC_SIMD_NEON
    {   int16x8_t vdc=vdupq_n_s16((int16_t)dc);
        for(int i=0;i<n;i+=16){
            uint8x16_t v=vld1q_u8(vox+i);
            uint8x16_t nz=vtstq_u8(v,v);
            int16x8_t lo=vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v)));
            int16x8_t hi=vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v)));
            lo=vsubq_s16(lo,vdc); hi=vsubq_s16(hi,vdc);
            int16x8_t mlo=vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(nz)));
            int16x8_t mhi=vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(nz)));
            // mask: 0x00FF per nonzero after widen -> turn into all-ones/all-zero
            mlo=vreinterpretq_s16_u16(vtstq_u16(vreinterpretq_u16_s16(mlo),vreinterpretq_u16_s16(mlo)));
            mhi=vreinterpretq_s16_u16(vtstq_u16(vreinterpretq_u16_s16(mhi),vreinterpretq_u16_s16(mhi)));
            lo=vandq_s16(lo,mlo); hi=vandq_s16(hi,mhi);   // air -> 0 (== dc-dc)
            vst1q_f32(blk+i   ,vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))));
            vst1q_f32(blk+i+4 ,vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))));
            vst1q_f32(blk+i+8 ,vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))));
            vst1q_f32(blk+i+12,vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))));
        }
    }
#else
    for(int i=0;i<n;++i) blk[i]=(float)((vox[i]?vox[i]:dc)-dc);
#endif
    // harmonic air-fill: relax air voxels toward the 6-neighbor mean (material
    // fixed) so the masked region carries no spurious DCT energy. Perf on real
    // masked-scroll exports showed the original raster-order Gauss-Seidel/SOR
    // over an air-voxel index list was the #1 hot spot of mc_enc_block (~31%
    // of export compute): a strictly serial scalar dependency chain. The fill
    // only shapes values UNDER the air mask — decode forces them to 0 — so its
    // exact values are free to change slightly; only encode speed and archive
    // size matter. Rewritten as: coarse 4^3 seed + RED-BLACK SOR in a dense,
    // branch-free, auto-vectorizing form (see below).
    // Measured (8x 256^3 mixed material/air chunks of a real masked scroll,
    // q=8, best-of-5 process-CPU time incl. the vectorized stats/quant loops
    // above/below): encode 0.345s -> 0.253s (-26.7%), archive size
    // 1126163 -> 1126019 bytes (-0.013%), material max-abs-diff unchanged
    // (41), air voxels still decode to exactly 0.
    if(nair>0){
        // CROSS-ISA DETERMINISM: the floats computed here feed the DCT, so
        // they must round identically on every target. The build is strict
        // IEEE (no -ffast-math — see CMakeLists); under fast-math the
        // per-target reassociation/reciprocal choices in these loops broke
        // bitstream identity (caught by CI), for zero measured speedup.
#if defined(__clang__)
#pragma clang fp reassociate(off) contract(off)
#endif
        const int S=MC_BLK;
        // (b) skip the fine SOR sweeps on nearly-pure blocks (<5% or >95% air):
        // the coarse 4^3 seed already lands within quantization noise there
        // (thin slivers / almost-all-masked blocks), so refinement is an
        // invisible cost.
        int do_fine = (nair >= n/20) && (nair <= n - n/20);
        // Coarse-to-fine init: solve the fill on the 4^3 subcube grid first
        // (each cell = mean of its material voxels, air cells relaxed), then
        // seed fine air voxels from their cell before the fine SOR sweeps.
        // Lands much closer than a flat dc start, so few sweeps converge.
        // Accumulation runs per (z,y) row with 4-wide unrolled segment sums
        // (SLP-vectorizable; air contributes 0 to the sum because blk[]==0
        // there) — no per-voxel div/mod or branches.
        {
            float cs[64]; int cm[64]; const int G=4;
            for(int c=0;c<64;++c){ cs[c]=0.0f; cm[c]=0; }
            for(int z=0;z<S;++z)for(int y=0;y<S;++y){
                const float *bp=blk+(size_t)(z*S+y)*S;
                const mc_u8 *vp=vox+(size_t)(z*S+y)*S;
                int cb=((z>>2)*G+(y>>2))*G;
                for(int sx=0;sx<G;++sx){
                    const float *b4=bp+4*sx; const mc_u8 *v4=vp+4*sx;
                    cs[cb+sx]+=b4[0]+b4[1]+b4[2]+b4[3];
                    cm[cb+sx]+=(v4[0]!=0)+(v4[1]!=0)+(v4[2]!=0)+(v4[3]!=0);
                }
            }
            for(int c=0;c<64;++c) cs[c]=cm[c]?cs[c]/(float)cm[c]:0.0f;
            for(int it=0;it<6;++it){                      // coarse relax (air cells)
                for(int cz=0;cz<G;++cz)for(int cy=0;cy<G;++cy)for(int cx=0;cx<G;++cx){
                    int c=(cz*G+cy)*G+cx; if(cm[c]) continue;
                    float a=0; int k=0;
                    if(cz){a+=cs[c-G*G];k++;} if(cz<G-1){a+=cs[c+G*G];k++;}
                    if(cy){a+=cs[c-G];k++;}   if(cy<G-1){a+=cs[c+G];k++;}
                    if(cx){a+=cs[c-1];k++;}   if(cx<G-1){a+=cs[c+1];k++;}
                    if(k) cs[c]=a/k;
                }
            }
            // seed air voxels from their cell: expand the 4 cell values of a
            // subcube row into a 16-float row pattern once per 4 rows, then a
            // dense branchless select per row (auto-vectorizes).
            float vrow[16]={0};   // always set at y==0 (init quiets -Wmaybe-uninitialized)
            for(int z=0;z<S;++z)for(int y=0;y<S;++y){
                if((y&3)==0){
                    const float *cr=cs+((z>>2)*G+(y>>2))*G;
                    for(int x=0;x<S;++x) vrow[x]=cr[x>>2];
                }
                float *bp=blk+(size_t)(z*S+y)*S;
                const mc_u8 *vp=vox+(size_t)(z*S+y)*S;
                for(int x=0;x<S;++x) bp[x]=vp[x]?bp[x]:vrow[x];
            }
        }
        // (a) RED-BLACK SOR (two-color Gauss-Seidel, omega=1.6) in a DENSE
        // vectorizable form replacing the serial scalar chain:
        //   - copy the block into an 18^3 zero-padded buffer P (pad cells are
        //     never written, so out-of-block neighbors read as 0 == dc);
        //   - fold air mask and color ((z+y+x)&1) into two per-color weight
        //     arrays (w6 = omega/6 on this color's air voxels, else 0) so a
        //     color pass is a UNIFORM branch-free stencil over the whole
        //     padded array:   P[i] += w6[i]*(nbsum[i] - cnt[i]*P[i])
        //     where cnt[i] (in-block 6-neighbor count, = the old serial
        //     code's divisor scaled into the relaxation step) is a static
        //     block-independent table, built once per thread like PM below.
        //   - within one color no voxel neighbors another, so the neighbor-sum
        //     and update loops are data-parallel and auto-vectorize (AVX/
        //     NEON); updated reds are visible to blacks (true Gauss-Seidel
        //     convergence, same omega, same sweep count as the serial code).
        if(do_fine){
            enum { PS=MC_BLK+2, PP=PS*PS, PN=PS*PS*PS };
            static _Thread_local float P[PN];              // pads stay 0
            static _Thread_local float W6[2][PN];          // pads stay 0
            // parity mask (voxel color) and in-block neighbor count are both
            // block-independent: build once per thread.
            static _Thread_local float PM[PN], CNT[PN];
            static _Thread_local int pm_init=0;
            if(!pm_init){
                for(int z=0;z<PS;++z)for(int y=0;y<PS;++y)for(int x=0;x<PS;++x){
                    int i=(z*PS+y)*PS+x;
                    PM[i]=(float)((z+y+x)&1);
                    CNT[i]=(float)((z>1)+(z<S)+(y>1)+(y<S)+(x>1)+(x<S));
                }
                pm_init=1;
            }
            // rows: copy P + build per-color weights in one vectorized pass.
            // Only real-voxel lanes are ever written, so pad/gap lanes of P
            // and W6 keep their static-zero values across blocks.
            const float O6=1.6f/6.0f;
            for(int z=0;z<S;++z)for(int y=0;y<S;++y){
                const mc_u8 *vp=vox+(size_t)(z*S+y)*S;
                const float *bp=blk+(size_t)(z*S+y)*S;
                int pb=((z+1)*PS+(y+1))*PS+1;
                const float *pm=PM+pb;
                for(int x=0;x<S;++x){
                    P[pb+x]=bp[x];
                    float w=vp[x]?0.0f:O6, a=w*pm[x];
                    W6[1][pb+x]=a; W6[0][pb+x]=w-a;
                }
            }
            // each color pass = per z-plane: (1) dense neighbor sums into a
            // small plane buffer NB (pure reads of P), (2) masked update of
            // the plane. Exact red-black Gauss-Seidel: this color's neighbors
            // are all the OTHER color, untouched within the pass, so the
            // snapshot in NB is the live value. Splitting removes the
            // read-after-write dependence that kept the fused in-place loop
            // scalar; both loops auto-vectorize (AVX/NEON). Plane blocking
            // keeps NB and the three active P planes L1-resident instead of
            // streaming a full-volume NB array through L2 every pass.
            // Pad/material lanes are killed by w6=0.
            // (c) the coarse seed does most of the work, so few fine sweeps
            // are needed: on the benchmark below 1 red-black sweep gives a
            // marginally SMALLER archive than the old 3 serial sweeps
            // (-0.013%), 2 sweeps +0.016% — the rate effect of sweep count is
            // already in the quantization noise (values are masked out at
            // decode anyway), so take the cheapest.
            int nsweep=MC_FILL_SWEEPS<1?MC_FILL_SWEEPS:1;
            for(int it=0; it<nsweep; ++it){
                for(int col=0; col<2; ++col){
                    const float *restrict w6=W6[col];
                    for(int pz=1; pz<PS-1; ++pz){
                        float NB[PP]; const int b=pz*PP;
                        for(int k=0;k<PP;++k)
                            NB[k]=P[b+k-1]+P[b+k+1]+P[b+k-PS]+P[b+k+PS]+P[b+k-PP]+P[b+k+PP];
                        for(int k=0;k<PP;++k)
                            P[b+k]+=w6[b+k]*(NB[k]-CNT[b+k]*P[b+k]);
                    }
                }
            }
            for(int z=0;z<S;++z)for(int y=0;y<S;++y)       // material rows are
                memcpy(blk+(size_t)(z*S+y)*S,              // bit-identical (w=0)
                       P+((z+1)*PS+(y+1))*PS+1, S*sizeof(float));
        }
    }
    mc_dct3_fwd(blk,coef);
    rc_i16 *ql=T->ql;
    rc_u8 *scratch=T->scratch;
    const size_t scratch_cap=sizeof T->scratch;
    // fused branchless quant+clamp (same math as quant_one up to fp rounding:
    // t = |c|/step - dzfrac + 1; for |c|>=dz, t>=1 truncates to the level,
    // for |c|<dz, t<1 so max(t,0) truncates to 0). The branchy quant_one
    // loop was scalar; this one auto-vectorizes, with a reciprocal step
    // table instead of a per-coefficient divide.
    for(int idx=0;idx<N3;++idx){
        float c=coef[idx];
        float t=fabsf(c)*g_rstep_tab[idx]+(1.0f-MC_DZ_FRAC);
        t=t>0.0f?t:0.0f; t=t<32767.0f?t:32767.0f;
        mc_i32 v=(mc_i32)t;
        ql[idx]=(rc_i16)(c<0.0f?-v:v);
    }

    // max-error corrections: locally reconstruct and list voxels with |err| > tau.
    uint16_t *cpos=T->cpos; mc_i32 *cdel=T->cdel;
    int ncorr=0;
    if(g_max_err>0){
        float *rcoef=T->rcoef, *rblk=T->rblk;
        for(int idx=0;idx<N3;++idx) rcoef[idx]=deq_one(ql[idx],g_step_tab[idx]);
        mc_dct3_inv(rcoef,rblk);
        for(int i=0;i<n;++i){
            if(!vox[i]) continue;                          // air decodes to exactly 0
            int v=(int)lrintf(rblk[i])+dc; if(v<0)v=0; if(v>255)v=255;
            int err=(int)vox[i]-v;
            int ae=err<0?-err:err;
            if(ae>g_max_err){ cpos[ncorr]=(uint16_t)i; cdel[ncorr]= err<0 ? -(ae-g_max_err) : (ae-g_max_err); ncorr++; }
        }
    }

    rc_enc e; enc_init(&e,scratch,scratch_cap);
    {   // header bins: mixed, has-corr, dc (trained priors)
        ctx_t cf[2]; for(int i=0;i<2;++i) ctx_init_p(&cf[i],RC_PRIOR_FLAG[i]);
        ctx_t cd[8]; for(int i=0;i<8;++i) ctx_init_p(&cd[i],RC_PRIOR_DC[i]);
        RC_TRAIN(RCC_FLAG,0,nair>0);  enc_bit(&e,&cf[0],nair>0);
        RC_TRAIN(RCC_FLAG,1,ncorr>0); enc_bit(&e,&cf[1],ncorr>0);
        for(int b=7;b>=0;--b){ int bit=(dc>>b)&1; RC_TRAIN(RCC_DC,7-b,bit); enc_bit(&e,&cd[7-b],bit); }
    }
    if(nair>0) enc_blockmask(&e,vox);
    enc_block_coefs(&e,ql,MC_BLK);
    if(ncorr>0){                                           // [eg count][gap, sign, eg(|d|-1)]*
        enc_eg(&e,(rc_u32)(ncorr-1));
        rc_u32 prev=0;
        for(int c=0;c<ncorr;++c){
            enc_eg(&e,(rc_u32)cpos[c]-prev); prev=cpos[c];
            mc_i32 D=cdel[c]; enc_bypass(&e,D<0); rc_u32 m=(rc_u32)(D<0?-D:D);
            enc_eg(&e,m-1);
        }
    }
    enc_flush(&e);
    uint32_t slen=(uint32_t)e.len;
    if(slen>scratch_cap){ fprintf(stderr,"mc_enc_block: scratch overflow (%u)\n",slen); abort(); }

    mc_buf_put(out,scratch,slen);
    *len_out = slen;
    return 1;
}

void mc_dec_block(const mc_u8 *p, uint32_t plen, mc_u8 *dst){
    int n=N3, dc=0, flags=0;
    mc_tls_t *T=&g_tls;                 // single TLV access for all scratch
    step_tab_build();                   // (touches its own TLVs) before hot loops
    mc_u8 *air=T->air;
    rc_i16 *ql=T->ql;
    rc_dec d; dec_init(&d,p,plen);
    {   // header bins (must mirror the encoder exactly)
        ctx_t cf[2]; for(int i=0;i<2;++i) ctx_init_p(&cf[i],RC_PRIOR_FLAG[i]);
        ctx_t cd[8]; for(int i=0;i<8;++i) ctx_init_p(&cd[i],RC_PRIOR_DC[i]);
        flags |= dec_bit(&d,&cf[0]) ? 1 : 0;
        flags |= dec_bit(&d,&cf[1]) ? 2 : 0;
        for(int b=0;b<8;++b) dc=(dc<<1)|dec_bit(&d,&cd[b]);
    }
    if(flags&1) dec_blockmask(&d,air);
    else        memset(air,0,n);
    int ext[3]; dec_block_coefs_ext(&d,ql,MC_BLK,ext);
    float *coef=T->coef, *blk=T->blk;
    int ez=ext[0],ey=ext[1],ex=ext[2];
    if(ez<0 && !(flags&1) && !(flags&2)){                   // constant block: dc fill
        memset(dst,(mc_u8)dc,n); return;
    }
    (void)ey;(void)ex;
#if MC_SIMD_NEON
    {   // fused dequant -> i32 DCT input (no float coefficient pass), then
        // integer iDCT and vectorized clamp+dc+air store.
        mc_fi32 *qin=T->qin, *qout=T->qout;
        float32x4_t bias=vdupq_n_f32(MC_DZ_FRAC-1.0f+0.40f);
        for(int i=0;i<N3;i+=4){
            int32x4_t lv=vmovl_s16(vld1_s16(ql+i));
            uint32x4_t nz=vmvnq_u32(vceqzq_s32(lv));
            uint32x4_t neg=vcltzq_s32(lv);
            float32x4_t a=vcvtq_f32_s32(vabsq_s32(lv));
            float32x4_t r=vmulq_f32(vaddq_f32(a,bias),vld1q_f32(g_step_tab+i));
            int32x4_t ri=vcvtnq_s32_f32(r);
            ri=vbslq_s32(neg,vnegq_s32(ri),ri);
            ri=vandq_s32(ri,vreinterpretq_s32_u32(nz));
            vst1q_s32(qin+i,ri);
        }
        mc_dct3_inv_i32(qin,qout);
        int32x4_t vdc=vdupq_n_s32(dc);
        for(int i=0;i<n;i+=16){
            int32x4_t a0=vaddq_s32(vld1q_s32(qout+i),vdc);
            int32x4_t a1=vaddq_s32(vld1q_s32(qout+i+4),vdc);
            int32x4_t a2=vaddq_s32(vld1q_s32(qout+i+8),vdc);
            int32x4_t a3=vaddq_s32(vld1q_s32(qout+i+12),vdc);
            uint16x8_t p0=vcombine_u16(vqmovun_s32(a0),vqmovun_s32(a1));
            uint16x8_t p1=vcombine_u16(vqmovun_s32(a2),vqmovun_s32(a3));
            uint8x16_t v8=vcombine_u8(vqmovn_u16(p0),vqmovn_u16(p1));
            uint8x16_t am=vld1q_u8(air+i);
            v8=vbicq_u8(v8,vtstq_u8(am,am));               // air -> 0
            vst1q_u8(dst+i,v8);
        }
        (void)coef;(void)blk;
    }
#else
    for(int idx=0;idx<N3;++idx) coef[idx]=deq_one(ql[idx],g_step_tab[idx]);
    mc_dct3_inv(coef,blk);
    for(int i=0;i<n;++i){
        int v = air[i] ? 0 : (int)lrintf(blk[i])+dc;  // mask-restore: air -> exactly 0
        if(v<0)v=0; if(v>255)v=255; dst[i]=(mc_u8)v;
    }
#endif
    if(flags&2){                                      // sparse max-error corrections
        // HARDENED: ncorr and positions are attacker-controlled on corrupted
        // input — clamp both so a flipped bit can never write outside dst.
        rc_u32 ncorr=dec_eg(&d)+1, pos=0;
        if(ncorr>(rc_u32)N3) ncorr=N3;
        for(rc_u32 c=0;c<ncorr;++c){
            pos+=dec_eg(&d);
            int neg=dec_bypass(&d); rc_u32 m=dec_eg(&d)+1;
            if(pos>=(rc_u32)N3) break;                // corrupt stream: stop
            int v=(int)dst[pos]+(neg?-(int)m:(int)m);
            if(v<0)v=0; if(v>255)v=255; dst[pos]=(mc_u8)v;
        }
    }
}

// ============================================================================
// mc_archive.h — matter-compressor on-disk archive format constants.
//
// An APPENDABLE, crash-safe, persistent archive: a node tree of dense 256^3 chunks,
// each a 16^3 grid of DCT blocks coded by mc_codec. 8 independent LODs (LOD0 = full
// res), each its own tree. LODs are independently fetchable AND independently
// decodable — no cross-LOD dependency (a hard design constraint).
//
// HIERARCHY (per LOD), all grids are 16x16x16 (4 bits/axis):
//   chunk coord (voxel>>8, up to 12 bits/axis) = [ region 4b | subregion 4b | shard 4b ]
//   CHUNK (256^3 voxels = 16^3 DCT blocks) is the dense leaf. Above it: a dense 16^3
//   SHARD table of chunk offsets, then 2 NODE levels. Covers 2^12 chunks/axis =
//   2^20 voxels/axis.
//
// NODE / SHARD = a DENSE flat MC_GRID3 (=4096) array of u64 child offsets, slot value
// 0 = absent child. Directly indexed by the chunk-coord nibble; no bitmap, no
// rank-packing -> every node + shard is UPDATABLE IN PLACE. This is what makes the
// archive appendable + crash-safe: chunk payloads append at EOF; the index path
// (root -> node -> node -> shard) is created/updated in place with the chunk offset
// published LAST as the commit word. The file is a fully valid, decodable archive
// after every appended chunk and PERSISTS across process runs (reopen + append).
//
// CHUNK blob (v2) = [512B block-bitmap][present-block u32 lens][block payloads];
// one range-GET fetches a whole chunk. Blocks are self-contained (per-block air
// mask lives in each block payload), so one block decode needs only the bitmap,
// the lens, and its own payload.
//
// LAYOUT: [256B header][metadata region up to 128KB][archive data from 128KB:
//          node/shard tables + chunk blobs, both appended at EOF].
// ============================================================================
#ifndef MC_ARCHIVE_H
#define MC_ARCHIVE_H
#include <stdint.h>
#include <string.h>

// ---- node / shard geometry (dense 16^3 child map of u64 offsets) ----
#define MC_GRID         16
#define MC_GRID3        4096          // 16^3 slots
#define MC_BITMAP_BYTES 512           // 4096 bits — chunk-blob block bitmap (NOT node index)
#define MC_NODE_BYTES   (MC_GRID3*8)  // dense node table: 4096 u64 child offsets (32KB)
#define MC_SHARD_BYTES  (MC_GRID3*8)  // dense shard table: 4096 u64 chunk offsets (32KB)
#define MC_TREE_LEVELS  3             // root node, 1 inner node, 1 shard (indexed by nibbles 2,1,0)

// ---- header (256B) ----
#define MC_MAGIC   0x0043434Du      // "MCC\0"
#define MCH_MAGIC   0               // u32 magic
#define MCH_VER     4               // u32 format-version field
#define MCH_NX      12              // u32 volume dims (x fastest)
#define MCH_NY      16
#define MCH_NZ      20
#define MCH_ROOTOFF 24              // u64[8] per-LOD root-node file offset (0 = empty LOD)
#define MCH_TOTLEN  88              // u64 total archive length (= append cursor / EOF)
#define MCH_METAOFF 96              // u64 metadata region start (= MC_HDR = 256)
#define MCH_METACAP 104             // u64 metadata region capacity (= MC_META_END - MC_HDR)
#define MCH_METALEN 112             // u64 metadata bytes actually written
#define MCH_QUALITY 120             // f32 quality the archive was built at (writer stamps it)
#define MCH_PRIOROFF 128            // u64 offset of an optional per-volume prior blob (0 = none)
#define MC_PRIORS_MAGIC 0x53524950u // "PRIS"
#define MC_PRIORS_BYTES (8 + 2*8*32*2)  // magic+ver, RC_PLO + RC_PHI as u16[8][32]
#define MC_HDR      256u            // header size; metadata region begins here
#define MC_META_END (128u*1024u)    // archive data begins at this offset (128KB)
#define MC_META_CAP (MC_META_END - MC_HDR)
#define MC_VERSION  7u              // format version (v7: per-chunk material-fraction map)

#define MC_CHUNK_ALIGN 256          // volume dim must be a multiple of this

// chunk-blob block-bitmap + chunk-coord nibble helpers
static inline int  mc_bit_get(const uint8_t*bm,int i){ return (bm[i>>3]>>(i&7))&1; }
static inline void mc_bit_set(uint8_t*bm,int i){ bm[i>>3]|=(uint8_t)(1u<<(i&7)); }
static inline int  mc_nib(int chunkcoord,int level){ return (chunkcoord>>(4*level))&15; }
#endif

// ============================================================================
// mc_archive_read.h — resolve a chunk by coord through the DENSE node tree, then
// locate a block within the dense chunk. Absent child = empty region (decodes to 0).
//
// The index is 3 dense levels (root node, inner node, shard), each a flat MC_GRID3
// array of u64 offsets indexed by the chunk-coord nibble (2, then 1, then 0). A slot
// value of 0 means absent. No bitmap / no rank-packing at the index level, so the
// resolve is three direct array reads.
// ============================================================================
#ifndef MC_ARCHIVE_READ_H
#define MC_ARCHIVE_READ_H
#include <stdint.h>

// number of set bits in bm[0..idx) — used for the chunk-blob block bitmap only.
static inline int mc_rank(const uint8_t*bm,int idx){
    int r=0, full=idx>>3;
    for(int i=0;i<full;++i) r+=__builtin_popcount(bm[i]);
    int rem=idx&7; if(rem) r+=__builtin_popcount(bm[full] & ((1u<<rem)-1));
    return r;
}

// resolve chunk (cz,cy,cx) -> chunk-blob offset (0 = empty/absent). Walks the dense
// node tree: nibble 2 (root node) -> nibble 1 (inner node) -> nibble 0 (shard) ->
// chunk-blob offset. Each level is a direct u64 array index.
static uint64_t mc_resolve_chunk(const uint8_t*arc, uint64_t root_off,int cz,int cy,int cx){
    uint64_t node = root_off;
    for(int nib=MC_TREE_LEVELS-1; nib>=0; --nib){
        if(!node) return 0;
        int dz=mc_nib(cz,nib), dy=mc_nib(cy,nib), dx=mc_nib(cx,nib);
        int idx=(dz*16+dy)*16+dx;
        uint64_t childoff; memcpy(&childoff, arc+node + (size_t)idx*8, 8);
        node = childoff;
    }
    return node;   // chunk-blob offset (0 if absent)
}

// chunk blob (v7): [f32 q][u64 xxh64][u16 fmaplen][fmap][512B block-bitmap]
// [present u16 lens][payloads]. q = the chunk's own quality; the hash covers
// fmap+bitmap+lens+payloads; fmap = rc-coded per-block material fractions
// (4096 nibbles, 0..15 ~= 0..100%) for rejection-free ML sampling.
#define MC_BLOB_HDR 14
// Slot sentinel: a chunk that was VISITED and decodes to all-zero (air). Real
// blob offsets are always >= MC_HDR (blobs append after the header), so 1 is a
// safe sentinel. Distinguishes "air, fetched" from "never fetched" (slot 0).
#define MC_SLOT_ZERO 1ull
static inline float mc_chunk_q(const uint8_t*arc, uint64_t chunk_off){
    float q; memcpy(&q,arc+chunk_off,4); return q;
}
static inline uint64_t mc_chunk_stored_hash(const uint8_t*arc, uint64_t chunk_off){
    uint64_t h; memcpy(&h,arc+chunk_off+4,8); return h;
}
static inline uint16_t mc_chunk_fmaplen(const uint8_t*arc, uint64_t chunk_off){
    uint16_t l; memcpy(&l,arc+chunk_off+12,2); return l;
}
// block (bz,by,bx) present? -> 1 + its payload (abs_off, len). 0 = ZERO block. Offsets
// are implicit (cumulative len of present blocks before it); ZERO blocks cost 1 bitmap bit.
static int mc_block_range(const uint8_t*arc, uint64_t chunk_off, int bz,int by,int bx,
                          uint64_t *abs_off, uint32_t *len){
    uint64_t bm_off = chunk_off + MC_BLOB_HDR + mc_chunk_fmaplen(arc,chunk_off);
    const uint8_t*bm = arc + bm_off;
    int bi=(bz*16+by)*16+bx;
    if(!mc_bit_get(bm,bi)) return 0;
    int npresent=0; for(int i=0;i<MC_BITMAP_BYTES;++i) npresent+=__builtin_popcount(bm[i]);
    const uint8_t*lens = arc + bm_off + MC_BITMAP_BYTES;
    uint64_t pay_base = bm_off + MC_BITMAP_BYTES + (uint64_t)npresent*2;
    int slot = mc_rank(bm,bi);
    uint64_t cum=0; for(int s=0;s<slot;++s){ uint16_t l; memcpy(&l,lens+(size_t)s*2,2); cum+=l; }
    uint16_t mylen; memcpy(&mylen, lens+(size_t)slot*2, 2);
    *abs_off = pay_base + cum; *len = mylen;
    return 1;
}

// total byte length of a chunk blob — used to copy a whole compressed chunk verbatim
// (mc_append_chunk_compressed) and for chunk-blob range queries. chunk_off must be valid.
static uint64_t mc_chunk_blob_len(const uint8_t*arc, uint64_t chunk_off){
    uint64_t bm_off = chunk_off + MC_BLOB_HDR + mc_chunk_fmaplen(arc,chunk_off);
    const uint8_t*bm = arc + bm_off;
    int npresent=0; for(int i=0;i<MC_BITMAP_BYTES;++i) npresent+=__builtin_popcount(bm[i]);
    const uint8_t*lens = arc + bm_off + MC_BITMAP_BYTES;
    uint64_t paybytes=0; for(int s=0;s<npresent;++s){ uint16_t l; memcpy(&l,lens+(size_t)s*2,2); paybytes+=l; }
    return bm_off + MC_BITMAP_BYTES + (uint64_t)npresent*2 + paybytes - chunk_off;
}
#endif

// ============================================================================
// mc_archive.c — matter-compressor archive build + APPENDABLE writer + decode.
// See mc_archive_api.h. Source-agnostic: no zarr/S3 here.
//
// Index is a DENSE node tree (root node -> inner node -> shard), each a flat
// MC_GRID3 array of u64 offsets, updatable in place. Chunk payloads append at EOF.
//
// Three faces:
//   - mc_build / mc_build_to_file : one-shot build of a whole volume (RAM-materialized).
//   - mc_archive_* : ONE persistent, crash-safe, appendable on-disk handle that both
//                    APPENDS chunks and DECODES them (no writer/reader split). mmap +
//                    atomic append cursor + in-place dense-node index + release-published
//                    commit, modeled on volume-compressor's writer; decode reads the live
//                    mmap.
//   - mc_open / mc_open_streaming + decode : read an already-built archive from a buffer
//                    or a byte-source (streaming / S3).
// ============================================================================
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdatomic.h>
#include <errno.h>
#include <math.h>
#include <stdatomic.h>

#if defined(__unix__) || defined(__APPLE__)
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
  #include <fcntl.h>
  #include <pthread.h>
  #define MC_HAVE_MMAP 1
#else
  #define MC_HAVE_MMAP 0
#endif

typedef uint8_t u8;
static void priors_load(const uint8_t *base);

// ---------------------------------------------------------------- volume (one LOD)
// A contiguous dim^3 u8 buffer. LOD0 is materialized from the source callback; coarser
// LODs are box-decimated from the finer one.
typedef struct { const u8 *v; int nz, ny, nx; } vol_t;
static inline u8 vget(const vol_t *V, int z,int y,int x){
    if((unsigned)z>=(unsigned)V->nz||(unsigned)y>=(unsigned)V->ny||(unsigned)x>=(unsigned)V->nx) return 0;
    return V->v[((size_t)z*V->ny+y)*V->nx+x];
}
// 2x box-decimate: mean of NONZERO children; all-zero stays 0 (inherited zero-mask).
// Per-axis dims; odd source dims round up (the edge voxel decimates alone).
static u8 *decimate(const u8 *src,int DZ,int DY,int DX,int *HZ,int *HY,int *HX){
    int hz=(DZ+1)/2, hy=(DY+1)/2, hx=(DX+1)/2;
    u8 *o=calloc((size_t)hz*hy*hx,1);
    for(int z=0;z<hz;++z)for(int y=0;y<hy;++y)for(int x=0;x<hx;++x){ int s=0,c=0;
        for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
            int sz=2*z+dz, sy=2*y+dy, sx=2*x+dx;
            if(sz>=DZ||sy>=DY||sx>=DX) continue;
            u8 v=src[((size_t)sz*DY+sy)*DX+sx]; if(v){s+=v;c++;}}
        o[((size_t)z*hy+y)*hx+x]=c?(u8)((s+c/2)/c):0; }
    *HZ=hz; *HY=hy; *HX=hx; return o; }

// gather a 16^3 block from a contiguous 256^3 chunk buffer (chunk is dense, no edges).
static int gather_blk256(const u8 *chunk,int bz,int by,int bx,u8 *dst){
    int z0=bz*MC_BLK,y0=by*MC_BLK,x0=bx*MC_BLK,any=0;
    for(int z=0;z<MC_BLK;++z)for(int y=0;y<MC_BLK;++y)for(int x=0;x<MC_BLK;++x){
        u8 v=chunk[((size_t)(z0+z)*MC_CHUNK+(y0+y))*MC_CHUNK+(x0+x)]; dst[(z*MC_BLK+y)*MC_BLK+x]=v; any|=v; }
    return any;
}

// ---------------------------------------------------------------- chunk-blob encoder
// Encode one dense 256^3 chunk buffer into a compressed blob in `out` (a growable
// byte sink: out_put(out, ptr, n)). Returns the blob length, or 0 if the chunk is all
// air (no blob — caller leaves the slot absent). Shared by build + writer.
// blob (v6) = [f32 q][u64 xxh64][512B block-bitmap][present-block u16 lens]
// [block payloads]; blocks are self-contained (mask in payload). q is the
// chunk's own quality; the hash covers bitmap+lens+payloads.
typedef void (*out_put_fn)(void *out, const void *s, size_t n);

static size_t encode_chunk_blob(const u8 *chunk256, out_put_fn put, void *out){
    static _Thread_local mc_buf tmp; tmp.len=0;
    uint8_t bm[MC_BITMAP_BYTES]; memset(bm,0,sizeof bm);
    uint16_t blen[MC_GRID3]; int npresent=0;
    static _Thread_local uint8_t frac[MC_GRID3];
    memset(frac,0,MC_GRID3);
    static _Thread_local u8 vox[MC_BLK*MC_BLK*MC_BLK];
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
        int bi=(bz*16+by)*16+bx;
        if(!gather_blk256(chunk256,bz,by,bx,vox)) continue;
        int cnt=0; for(int i=0;i<MC_BLK*MC_BLK*MC_BLK;++i) cnt+=vox[i]!=0;
        frac[bi]=(uint8_t)((cnt*15+2048)/4096);              // nibble 0..15
        if(cnt&&!frac[bi]) frac[bi]=1;                        // any material -> >=1
        uint32_t len=0; if(mc_enc_block(vox,&tmp,&len)){ mc_bit_set(bm,bi); blen[bi]=(uint16_t)len; npresent++; }
    }
    if(!npresent) return 0;   // all air -> no blob
    // v7 blob header: [f32 q][u64 xxh64][u16 fmaplen][fmap]
    float q=mc_get_quality();
    static _Thread_local uint16_t lens16[MC_GRID3]; int nl=0;
    for(int bi=0;bi<MC_GRID3;++bi) if(mc_bit_get(bm,bi)) lens16[nl++]=blen[bi];
    static _Thread_local uint8_t fmap[MC_GRID3/2+64];
    uint16_t fml=(uint16_t)mc_enc_fracmap(frac,fmap,sizeof fmap);
    uint64_t h=mc_xxh64(fmap,fml,0x6D636368756E6Bull);
    h^=mc_xxh64(bm,MC_BITMAP_BYTES,h);
    h^=mc_xxh64(lens16,(size_t)nl*2,h);
    h^=mc_xxh64(tmp.p,tmp.len,h);
    size_t total=MC_BLOB_HDR+fml+MC_BITMAP_BYTES+(size_t)npresent*2+tmp.len;
    put(out,&q,4); put(out,&h,8); put(out,&fml,2);
    put(out,fmap,fml);
    put(out,bm,MC_BITMAP_BYTES);
    put(out,lens16,(size_t)nl*2);
    put(out,tmp.p,tmp.len);
    return total;
}

// recompute a chunk blob's hash from its bytes (mc_verify / verify-on-decode).
uint64_t mc_chunk_compute_hash(const uint8_t *blob, uint64_t blob_len){
    uint16_t fml; memcpy(&fml,blob+12,2);
    const uint8_t *fmap=blob+MC_BLOB_HDR;
    const uint8_t *bm=fmap+fml;
    int npresent=0; for(int i=0;i<MC_BITMAP_BYTES;++i) npresent+=__builtin_popcount(bm[i]);
    const uint8_t *lens=bm+MC_BITMAP_BYTES;
    uint64_t pay=(uint64_t)MC_BLOB_HDR+fml+MC_BITMAP_BYTES+(uint64_t)npresent*2;
    uint64_t h=mc_xxh64(fmap,fml,0x6D636368756E6Bull);
    h^=mc_xxh64(bm,MC_BITMAP_BYTES,h);
    h^=mc_xxh64(lens,(size_t)npresent*2,h);
    h^=mc_xxh64(blob+pay,(size_t)(blob_len-pay),h);
    return h;
}

// ---------------------------------------------------------------- one-shot build (abuf)
typedef struct { u8 *p; size_t len, cap; } abuf;
static void a_reserve(abuf*b,size_t n){ if(b->len+n<=b->cap)return; size_t nc=b->cap?b->cap*2:1<<20; while(nc<b->len+n)nc*=2; b->p=realloc(b->p,nc); b->cap=nc; }
static size_t a_put_at(abuf*b,const void*s,size_t n){ a_reserve(b,n); size_t at=b->len; memcpy(b->p+at,s,n); b->len+=n; return at; }
static size_t a_zero(abuf*b,size_t n){ a_reserve(b,n); size_t at=b->len; memset(b->p+at,0,n); b->len+=n; return at; }
static void a_u32(abuf*b,size_t at,uint32_t v){ memcpy(b->p+at,&v,4); }
static void a_u64(abuf*b,size_t at,uint64_t v){ memcpy(b->p+at,&v,8); }
static void abuf_put(void *out, const void *s, size_t n){ a_put_at((abuf*)out,s,n); }

// gather a contiguous 256^3 chunk out of a LOD volume (zero-padded at the volume edge).
static int gather_chunk256(const vol_t *V,int cz,int cy,int cx,u8 *dst){
    int z0=cz*256,y0=cy*256,x0=cx*256,any=0;
    for(int z=0;z<MC_CHUNK;++z)for(int y=0;y<MC_CHUNK;++y)for(int x=0;x<MC_CHUNK;++x){
        u8 v=vget(V,z0+z,y0+y,x0+x); dst[((size_t)z*MC_CHUNK+y)*MC_CHUNK+x]=v; any|=v; }
    return any;
}

// dense one-shot build: write the dense node tree for one LOD volume. Allocate the
// dense tables top-down with deferred offset back-patch (we know child offsets only
// after writing children, but the tables are dense + fixed-size so we reserve then fill).
// Simpler: build bottom-up into an in-RAM dense tree, then emit. We emit chunks first,
// recording offsets in a dense shard map, then emit shards, inner nodes, root.

static uint64_t build_lod_dense(abuf*b, const vol_t *V, int ncz,int ncy,int ncx){
    // dense maps over chunk grid (sparse population). We allocate per populated node.
    // root: 16^3 inner-node offsets; inner: 16^3 shard offsets; shard: 16^3 chunk offsets.
    // chunk coord nibbles: 2 (root), 1 (inner), 0 (shard).
    // First pass: emit chunk blobs, fill shard tables in RAM keyed by (n1=hi, n0).
    // To keep it simple + correct for any nchunks up to 4096/axis, use hash-free dense
    // allocation only where present.
    // We build a 3-level RAM tree of u64 tables, emit chunk blobs to `b`, then emit the
    // tables bottom-up and return the root offset.

    // RAM node: 4096 u64 child *table-indices* during build, resolved to file offsets on emit.
    // Use dynamic arrays of tables.
    typedef struct { uint64_t slot[MC_GRID3]; } table_t;   // 32KB each
    // shards keyed by (n2,n1); inners keyed by n2; root single.
    // We accumulate present shards/inners in vectors with their grid index.
    table_t *root = calloc(1,sizeof *root);
    // inner tables: up to 4096; allocate lazily.
    table_t **inner = calloc(MC_GRID3,sizeof *inner);
    // for each inner, its shard tables lazily.
    table_t ***shard = calloc(MC_GRID3,sizeof *shard);

    static _Thread_local u8 *chunkbuf=0; if(!chunkbuf) chunkbuf=malloc((size_t)MC_CHUNK*MC_CHUNK*MC_CHUNK);

    for(int cz=0;cz<ncz;++cz)for(int cy=0;cy<ncy;++cy)for(int cx=0;cx<ncx;++cx){
        if(!gather_chunk256(V,cz,cy,cx,chunkbuf)) continue;
        size_t at=b->len;
        if(!encode_chunk_blob(chunkbuf,abuf_put,b)) continue;   // all air
        int n2=(mc_nib(cz,2)*16+mc_nib(cy,2))*16+mc_nib(cx,2);
        int n1=(mc_nib(cz,1)*16+mc_nib(cy,1))*16+mc_nib(cx,1);
        int n0=(mc_nib(cz,0)*16+mc_nib(cy,0))*16+mc_nib(cx,0);
        if(!inner[n2]) inner[n2]=calloc(1,sizeof(table_t));
        if(!shard[n2]) shard[n2]=calloc(MC_GRID3,sizeof(table_t*));
        if(!shard[n2][n1]) shard[n2][n1]=calloc(1,sizeof(table_t));
        shard[n2][n1]->slot[n0]=(uint64_t)at;   // chunk-blob offset
    }
    // emit bottom-up: shards, then inners, then root.
    int any_root=0;
    for(int n2=0;n2<MC_GRID3;++n2){
        if(!inner[n2]) continue;
        int any_inner=0;
        for(int n1=0;n1<MC_GRID3;++n1){
            if(!shard[n2] || !shard[n2][n1]) continue;
            uint64_t soff=a_put_at(b,shard[n2][n1],MC_SHARD_BYTES);
            inner[n2]->slot[n1]=soff; any_inner=1;
        }
        if(any_inner){ uint64_t ioff=a_put_at(b,inner[n2],MC_NODE_BYTES); root->slot[n2]=ioff; any_root=1; }
    }
    uint64_t root_off=0;
    if(any_root) root_off=a_put_at(b,root,MC_NODE_BYTES);

    // free RAM tree
    for(int n2=0;n2<MC_GRID3;++n2){
        if(shard[n2]){ for(int n1=0;n1<MC_GRID3;++n1) free(shard[n2][n1]); free(shard[n2]); }
        free(inner[n2]);
    }
    free(shard); free(inner); free(root);
    return root_off;
}

uint8_t *mc_build(mc_voxel_fn src, void *ud, const mc_build_opts *opts, size_t *out_len){
    int NX = opts->nx>0?opts->nx:opts->dim;
    int NY = opts->ny>0?opts->ny:opts->dim;
    int NZ = opts->nz>0?opts->nz:opts->dim;
    if(NX<=0||NY<=0||NZ<=0){ fprintf(stderr,"mc_build: bad dims %dx%dx%d\n",NX,NY,NZ); return NULL; }
    // pad each axis to the chunk boundary (zero padding is nearly free)
    int PX=(NX+MC_CHUNK_ALIGN-1)/MC_CHUNK_ALIGN*MC_CHUNK_ALIGN;
    int PY=(NY+MC_CHUNK_ALIGN-1)/MC_CHUNK_ALIGN*MC_CHUNK_ALIGN;
    int PZ=(NZ+MC_CHUNK_ALIGN-1)/MC_CHUNK_ALIGN*MC_CHUNK_ALIGN;
    mc_codec_init(); mc_set_quality(opts->quality);
    u8 *lod0=calloc((size_t)PZ*PY,(size_t)PX);
    if(!lod0){ fprintf(stderr,"mc_build: OOM allocating %dx%dx%d\n",PZ,PY,PX); return NULL; }
    for(int z=0;z<NZ;++z)for(int y=0;y<NY;++y)for(int x=0;x<NX;++x)
        lod0[((size_t)z*PY+y)*PX+x]=src(ud,x,y,z);

    abuf b={0}; a_zero(&b,MC_META_END);
    size_t mlen=0;
    if(opts->metadata && opts->meta_len){
        mlen=opts->meta_len;
        if(mlen>MC_META_CAP){ fprintf(stderr,"mc_build: metadata %zu B > %u cap, truncating\n",mlen,(unsigned)MC_META_CAP); mlen=MC_META_CAP; }
        memcpy(b.p+MC_HDR, opts->metadata, mlen);
    }
    uint64_t roots[8]={0};
    const u8 *cur=lod0; u8 *owned=NULL;
    int dz=PZ, dy=PY, dx=PX;
    for(int lod=0; lod<8 && (dz>=1&&dy>=1&&dx>=1) && (dz>=MC_CHUNK||dy>=MC_CHUNK||dx>=MC_CHUNK||lod==0); ++lod){
        vol_t vv={cur,dz,dy,dx};
        int ncz=(dz+255)/256, ncy=(dy+255)/256, ncx=(dx+255)/256;
        roots[lod]=build_lod_dense(&b,&vv,ncz,ncy,ncx);
        if(dz/2<1||dy/2<1||dx/2<1) break;
        if(dz/2<MC_CHUNK&&dy/2<MC_CHUNK&&dx/2<MC_CHUNK&&lod>=1) break;
        int hz,hy,hx;
        u8 *next=decimate(cur,dz,dy,dx,&hz,&hy,&hx);
        if(owned) free(owned); owned=next; cur=next; dz=hz; dy=hy; dx=hx;
    }
    if(owned && owned!=lod0) free(owned);
    free(lod0);
    float q=opts->quality;
    a_u32(&b,MCH_MAGIC,MC_MAGIC); a_u32(&b,MCH_VER,MC_VERSION);
    a_u32(&b,MCH_NX,(uint32_t)NX); a_u32(&b,MCH_NY,(uint32_t)NY); a_u32(&b,MCH_NZ,(uint32_t)NZ);
    for(int l=0;l<8;++l) a_u64(&b,MCH_ROOTOFF+l*8,roots[l]);
    a_u64(&b,MCH_TOTLEN,b.len);
    a_u64(&b,MCH_METAOFF,MC_HDR); a_u64(&b,MCH_METACAP,MC_META_CAP); a_u64(&b,MCH_METALEN,mlen);
    memcpy(b.p+MCH_QUALITY,&q,4);
    if(out_len) *out_len=b.len;
    return b.p;
}

int mc_build_to_file(mc_voxel_fn src, void *ud, const mc_build_opts *opts, const char *outpath){
    size_t len=0; uint8_t *buf=mc_build(src,ud,opts,&len);
    if(!buf) return 1;
    FILE *of=fopen(outpath,"wb"); if(!of){ perror("fopen out"); free(buf); return 1; }
    fwrite(buf,1,len,of); fclose(of); free(buf);
    return 0;
}

// ============================================================================
// APPENDABLE WRITER — persistent mmap'd archive, modeled on volume-compressor.
// ============================================================================
#if MC_HAVE_MMAP

#define MC_RESERVE   (10ull*1024*1024*1024*1024)   // 10 TiB virtual reservation (NORESERVE)
#define MC_GROW_STEP (1ull*1024*1024*1024)          // grow the file 1 GiB at a time

struct mc_archive {
    int fd;
    u8 *base;                  // fixed mmap base (never moves)
    _Atomic uint64_t cursor;   // append EOF (bytes used)
    _Atomic uint64_t file_len; // current ftruncate'd file size
    int dim;
    float quality;
    uint64_t reserve;          // mmap reservation size (dims-derived, <= MC_RESERVE)
    pthread_mutex_t grow_mu;   // serializes ftruncate only; decode is lock-free
};

static int w_ensure(mc_archive *w, uint64_t need){
    uint64_t fl = atomic_load_explicit(&w->file_len, memory_order_acquire);
    if(need <= fl) return 0;
    pthread_mutex_lock(&w->grow_mu);
    fl = atomic_load_explicit(&w->file_len, memory_order_relaxed);
    if(need > fl){
        uint64_t nf = fl;
        while(nf < need) nf += MC_GROW_STEP;
        if(nf > w->reserve){   // past the mmap reservation: fail cleanly, not SIGBUS
            fprintf(stderr,"mc_archive: grow beyond reservation (%llu > %llu)\n",
                    (unsigned long long)nf,(unsigned long long)w->reserve);
            pthread_mutex_unlock(&w->grow_mu); return -1;
        }
        if(ftruncate(w->fd, (off_t)nf) != 0){ pthread_mutex_unlock(&w->grow_mu); return -1; }
        atomic_store_explicit(&w->file_len, nf, memory_order_release);
    }
    pthread_mutex_unlock(&w->grow_mu);
    return 0;
}
// reserve a disjoint [off, off+n) range at EOF, growing the file as needed.
static uint64_t w_alloc(mc_archive *w, uint64_t n){
    uint64_t off = atomic_fetch_add_explicit(&w->cursor, n, memory_order_relaxed);
    if(w_ensure(w, off+n)!=0) return ~0ull;
    return off;
}
static void w_write_u64(mc_archive *w, uint64_t at, uint64_t v){ memcpy(w->base+at,&v,8); }
static uint64_t w_read_u64(mc_archive *w, uint64_t at){ uint64_t v; memcpy(&v,w->base+at,8); return v; }

// growable sink wrapping a writer EOF append (used by encode_chunk_blob via a staging
// buffer; we encode to RAM first then memcpy the whole blob into one EOF range so the
// chunk payload is contiguous + committed atomically).
typedef struct { u8 *p; size_t len, cap; } stage_t;
static void stage_put(void *out, const void *s, size_t n){
    stage_t *st=(stage_t*)out;
    if(st->len+n>st->cap){ size_t nc=st->cap?st->cap*2:1<<16; while(nc<st->len+n)nc*=2; st->p=realloc(st->p,nc); st->cap=nc; }
    memcpy(st->p+st->len,s,n); st->len+=n;
}

// ensure the index path root->inner->shard exists for (lod,cz,cy,cx); return the file
// offset of the SHARD-table slot that will hold the chunk offset. Creates dense node
// tables in place as needed (allocated zeroed at EOF, parent slot published last).
static uint64_t w_ensure_shard_slot(mc_archive *w, int lod, int cz,int cy,int cx){
    uint64_t root = w_read_u64(w, MCH_ROOTOFF + (uint64_t)lod*8);
    if(!root){
        uint64_t no = w_alloc(w, MC_NODE_BYTES); if(no==~0ull) return ~0ull;
        memset(w->base+no, 0, MC_NODE_BYTES);
        // publish root in the header (single writer per (lod) path here; appends to the
        // same lod are serialized by the caller's per-writer use, but be safe: only set
        // if still empty).
        w_write_u64(w, MCH_ROOTOFF+(uint64_t)lod*8, no);
        root = no;
    }
    // walk nibble 2 (root) -> nibble 1 (inner) -> nibble 0 (shard slot).
    uint64_t node = root;
    for(int nib=MC_TREE_LEVELS-1; nib>=1; --nib){
        int idx=(mc_nib(cz,nib)*16+mc_nib(cy,nib))*16+mc_nib(cx,nib);
        uint64_t child = w_read_u64(w, node + (uint64_t)idx*8);
        if(!child){
            uint64_t no = w_alloc(w, MC_NODE_BYTES); if(no==~0ull) return ~0ull;
            memset(w->base+no, 0, MC_NODE_BYTES);
            w_write_u64(w, node + (uint64_t)idx*8, no);   // publish child slot
            child = no;
        }
        node = child;
    }
    int n0=(mc_nib(cz,0)*16+mc_nib(cy,0))*16+mc_nib(cx,0);
    return node + (uint64_t)n0*8;   // address of the shard slot for this chunk
}

// append a finished compressed blob at EOF + publish it in the shard slot (commit word).
// Mark a chunk's slot as VISITED-but-all-zero (air). Lets a re-run / prefetch
// tell "fetched, it was air" from "never fetched" — no blob is written.
static int w_mark_zero(mc_archive *w,int lod,int cz,int cy,int cx){
    uint64_t slot = w_ensure_shard_slot(w,lod,cz,cy,cx);
    if(slot==~0ull) return -1;
    w_write_u64(w, slot, MC_SLOT_ZERO);
    return 0;
}

static int w_install_blob(mc_archive *w,int lod,int cz,int cy,int cx,const u8 *blob,size_t len){
    uint64_t slot = w_ensure_shard_slot(w,lod,cz,cy,cx);
    if(slot==~0ull) return -1;
    uint64_t off = w_alloc(w, len);
    if(off==~0ull) return -1;
    memcpy(w->base+off, blob, len);
    // release fence so the payload bytes are visible before the commit word.
    atomic_thread_fence(memory_order_release);
    w_write_u64(w, slot, off);   // publish chunk offset = commit
    // keep the header's total length current so the file is valid if reopened now.
    uint64_t cur = atomic_load_explicit(&w->cursor, memory_order_relaxed);
    w_write_u64(w, MCH_TOTLEN, cur);
    return 0;
}

mc_archive *mc_archive_open_dims(const char *path, int nx, int ny, int nz, float quality);
mc_archive *mc_archive_open(const char *path, int dim, float quality){
    return mc_archive_open_dims(path,dim,dim,dim,quality);
}
// reservation sized from the volume: worst-case compressed bytes are bounded by
// ~raw size; 1.5x headroom + 1 GiB floor, capped at MC_RESERVE. A blanket 10 TiB
// map breaks sanitizer shadow memory and small-volume test runners.
static uint64_t reserve_for_dims(int nx,int ny,int nz){
    uint64_t need=0, dz=(uint64_t)nz, dy=(uint64_t)ny, dx=(uint64_t)nx;
    for(int l=0;l<8;++l){
        uint64_t pz=(dz+255)/256*256, py=(dy+255)/256*256, px=(dx+255)/256*256;
        need += pz*py*px;
        dz=(dz+1)/2; dy=(dy+1)/2; dx=(dx+1)/2;
    }
    uint64_t r = need + need/2 + (1ull<<30);
    return r > MC_RESERVE ? MC_RESERVE : r;
}

mc_archive *mc_archive_open_dims(const char *path, int nx, int ny, int nz, float quality){
    if(nx<=0||ny<=0||nz<=0){ fprintf(stderr,"mc_archive_open: bad dims\n"); return NULL; }
    int dim=nx;  // legacy field below; per-axis dims live in the header
    mc_codec_init(); mc_set_quality(quality);
    int fd = open(path, O_RDWR|O_CREAT, 0644);
    if(fd<0){ perror("mc_archive_open: open"); return NULL; }
    struct stat sb; if(fstat(fd,&sb)!=0){ perror("fstat"); close(fd); return NULL; }
    int fresh = (sb.st_size==0);
    uint64_t reserve = reserve_for_dims(nx,ny,nz);
    uint64_t init_len;
    if(fresh){
        init_len = MC_META_END;   // header + metadata region; data appends after.
        if(ftruncate(fd,(off_t)init_len)!=0){ perror("ftruncate"); close(fd); return NULL; }
    } else {
        init_len = (uint64_t)sb.st_size;
    }
    u8 *base = mmap(NULL, reserve, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_NORESERVE, fd, 0);
    if(base==MAP_FAILED){ perror("mmap"); close(fd); return NULL; }

    mc_archive *w = calloc(1,sizeof *w);
    w->fd=fd; w->base=base; w->dim=dim; w->quality=quality; w->reserve=reserve;
    pthread_mutex_init(&w->grow_mu,NULL);
    atomic_store(&w->file_len, init_len);

    if(fresh){
        memset(base,0,MC_HDR);
        uint32_t magic=MC_MAGIC, ver=MC_VERSION;
        uint32_t ux=(uint32_t)nx, uy=(uint32_t)ny, uz=(uint32_t)nz;
        memcpy(base+MCH_MAGIC,&magic,4); memcpy(base+MCH_VER,&ver,4);
        memcpy(base+MCH_NX,&ux,4); memcpy(base+MCH_NY,&uy,4); memcpy(base+MCH_NZ,&uz,4);
        uint64_t z=0; for(int l=0;l<8;++l) memcpy(base+MCH_ROOTOFF+l*8,&z,8);
        uint64_t metaoff=MC_HDR, metacap=MC_META_CAP, totlen=MC_META_END;
        memcpy(base+MCH_METAOFF,&metaoff,8); memcpy(base+MCH_METACAP,&metacap,8);
        memcpy(base+MCH_METALEN,&z,8); memcpy(base+MCH_TOTLEN,&totlen,8);
        memcpy(base+MCH_QUALITY,&quality,4);
        atomic_store(&w->cursor, MC_META_END);
    } else {
        uint32_t magic; memcpy(&magic,base+MCH_MAGIC,4);
        uint32_t ver;   memcpy(&ver,base+MCH_VER,4);
        uint32_t ux,uy,uz; memcpy(&ux,base+MCH_NX,4); memcpy(&uy,base+MCH_NY,4); memcpy(&uz,base+MCH_NZ,4);
        if(magic!=MC_MAGIC || ver!=MC_VERSION || (int)ux!=nx || (int)uy!=ny || (int)uz!=nz){
            fprintf(stderr,"mc_archive_open: %s is not a matching mc archive (magic/ver/dims)\n",path);
            munmap(base,reserve); close(fd); free(w); return NULL;
        }
        uint64_t totlen; memcpy(&totlen,base+MCH_TOTLEN,8);
        if(totlen < MC_META_END) totlen=MC_META_END;
        atomic_store(&w->cursor, totlen);
        priors_load(base);
    }
    return w;
}

int mc_archive_append_chunk_compressed(mc_archive *a, int lod, int cz,int cy,int cx,
                                       const uint8_t *blob, size_t len){
    if(!a||lod<0||lod>7||!blob||!len) return -1;
    return w_install_blob(a,lod,cz,cy,cx,blob,len);
}

int mc_archive_reserve_index(mc_archive *a, int lod, int cz,int cy,int cx){
    if(!a||lod<0||lod>7) return -1;
    return w_ensure_shard_slot(a,lod,cz,cy,cx)==~0ull ? -1 : 0;
}

int mc_archive_append_chunk_raw_q(mc_archive *a, int lod, int cz,int cy,int cx,
                                  const mc_u8 vox[256*256*256], float q){
    if(!a||lod<0||lod>7||!vox) return -1;
    mc_set_quality(q);
    stage_t st={0};
    size_t blen = encode_chunk_blob(vox, stage_put, &st);
    int rc = 0;
    if(blen) rc = w_install_blob(a,lod,cz,cy,cx,st.p,st.len);
    else     rc = w_mark_zero(a,lod,cz,cy,cx);   // air, but record it as VISITED
    free(st.p);
    return rc;
}
int mc_archive_append_chunk_raw(mc_archive *a, int lod, int cz,int cy,int cx, const mc_u8 vox[256*256*256]){
    return mc_archive_append_chunk_raw_q(a,lod,cz,cy,cx,vox,a?a->quality:8.0f);
}

// rate control: sample-encode 256 diagonally-spread blocks at base q, scale q
// once by the empirical bytes ~ q^-GAMMA law, then encode the chunk for real.
#define MC_RC_GAMMA 0.75f
int mc_archive_append_chunk_target(mc_archive *a, int lod, int cz,int cy,int cx,
                                   const mc_u8 vox[256*256*256], float target_ratio,
                                   float *q_out){
    if(!a||lod<0||lod>7||!vox||target_ratio<=1.0f) return -1;
    float q0=a->quality;
    mc_set_quality(q0);
    static _Thread_local mc_buf samp; samp.len=0;
    static _Thread_local u8 blk[MC_BLK*MC_BLK*MC_BLK];
    size_t sample_bytes=0; int sampled=0;
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by){
        int bx=(bz+by)&15;                       // diagonal spread, 256 blocks
        if(!gather_blk256(vox,bz,by,bx,blk)) { sampled++; continue; }
        uint32_t len=0; samp.len=0;
        if(mc_enc_block(blk,&samp,&len)) sample_bytes+=len;
        sampled++;
    }
    float q=q0;
    if(sample_bytes){
        double est_total=(double)sample_bytes*(4096.0/sampled);
        double want_total=(double)(256.0*256.0*256.0)/target_ratio;
        q=(float)(q0*pow(est_total/want_total,1.0/MC_RC_GAMMA));
        if(q<0.5f)q=0.5f; if(q>24.0f)q=24.0f;
    }
    if(q_out)*q_out=q;
    return mc_archive_append_chunk_raw_q(a,lod,cz,cy,cx,vox,q);
}

int mc_archive_set_priors(struct mc_archive *a,
                          const uint16_t plo[8][32], const uint16_t phi[8][32]){
    if(!a||!plo||!phi) return -1;
    uint64_t off=w_alloc(a,MC_PRIORS_BYTES);
    if(off==~0ull) return -1;
    uint32_t magic=MC_PRIORS_MAGIC, ver=1;
    memcpy(a->base+off,&magic,4); memcpy(a->base+off+4,&ver,4);
    memcpy(a->base+off+8,plo,8*32*2);
    memcpy(a->base+off+8+8*32*2,phi,8*32*2);
    atomic_thread_fence(memory_order_release);
    w_write_u64(a,MCH_PRIOROFF,off);
    mc_codec_set_priors((const uint16_t*)plo,(const uint16_t*)phi);
    return 0;
}

// load priors from a header offset if present (open paths call this).
static void priors_load(const uint8_t *base){
    uint64_t off; memcpy(&off,base+MCH_PRIOROFF,8);
    if(!off) { mc_codec_set_priors(NULL,NULL); return; }
    uint32_t magic; memcpy(&magic,base+off,4);
    if(magic!=MC_PRIORS_MAGIC){ mc_codec_set_priors(NULL,NULL); return; }
    mc_codec_set_priors((const uint16_t*)(base+off+8),(const uint16_t*)(base+off+8+8*32*2));
}

uint64_t mc_archive_data_len(mc_archive *a){
    return a ? atomic_load_explicit(&a->cursor, memory_order_relaxed) : 0;
}

mc_cover mc_archive_chunk_coverage(mc_archive *a, int lod, int cz,int cy,int cx){
    if(!a||lod<0||lod>7) return MC_ABSENT;
    uint64_t root = w_read_u64(a, MCH_ROOTOFF+(uint64_t)lod*8);
    uint64_t off = mc_resolve_chunk(a->base, root, cz,cy,cx);
    if(off==0) return MC_ABSENT;
    if(off==MC_SLOT_ZERO) return MC_ZERO;
    return MC_PRESENT;
}

uint64_t mc_archive_chunk_offset(mc_archive *a, int lod, int cz,int cy,int cx){
    if(!a||lod<0||lod>7) return 0;
    uint64_t root = w_read_u64(a, MCH_ROOTOFF+(uint64_t)lod*8);
    return mc_resolve_chunk(a->base, root, cz,cy,cx);
}

// Decode one 16^3 block from the live mmap. LOCK-FREE: the codec's per-call scratch is
// all _Thread_local and g_quality is constant for the archive's lifetime (set once at
// open), so concurrent decodes are safe without serialization. Blocks are fully
// self-contained (v2: per-block air mask in the payload), so a single block decode
// touches only the bitmap + its own payload. The mmap is read-only here; appends
// publish via a release fence so a resolved chunk_off always points at fully-written
// bytes.
void mc_archive_decode_block(mc_archive *a, uint64_t chunk_off, int bz,int by,int bx, mc_u8 *dst){
    if(!a||chunk_off<=MC_SLOT_ZERO){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    mc_set_quality(mc_chunk_q(a->base,chunk_off));   // thread-local; per-chunk q
    uint64_t boff; uint32_t bl;
    if(!mc_block_range(a->base,chunk_off,bz,by,bx,&boff,&bl)){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    // HARDENED: offsets derive from the on-disk length table; on a corrupt
    // archive they could point past the mapped file (SIGBUS). Bound against
    // the live append cursor. (For untrusted archives run mc_verify first —
    // the per-chunk xxh64 covers bitmap+lens+payloads.)
    uint64_t end=atomic_load_explicit(&a->cursor,memory_order_acquire);
    if(boff+bl>end){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    mc_dec_block(a->base+boff,bl,dst);
}

// ---- parallel whole-chunk helpers ------------------------------------------
typedef struct {
    mc_archive *a; uint64_t chunk_off; mc_u8 *out; float q;
    _Atomic uint32_t next;
} dchunk_ctx;
static void *dchunk_worker(void *p){
    dchunk_ctx *c=p;
    mc_set_quality(c->q);                       // thread-local
    mc_u8 blk[MC_BLK*MC_BLK*MC_BLK];
    for(;;){
        uint32_t bi=atomic_fetch_add_explicit(&c->next,1,memory_order_relaxed);
        if(bi>=MC_GRID3) break;
        int bz=bi>>8, by=(bi>>4)&15, bx=bi&15;
        mc_archive_decode_block(c->a,c->chunk_off,bz,by,bx,blk);
        for(int z=0;z<MC_BLK;++z)for(int y=0;y<MC_BLK;++y)
            memcpy(c->out+((size_t)(bz*16+z)*MC_CHUNK+(by*16+y))*MC_CHUNK+(size_t)bx*16,
                   blk+((size_t)z*16+y)*16,16);
    }
    return NULL;
}
static int auto_threads(int nthreads){
    if(nthreads>0) return nthreads>16?16:nthreads;
    long nc=sysconf(_SC_NPROCESSORS_ONLN);
    int nt=(int)(nc>0?nc:4); return nt>16?16:nt;
}
void mc_archive_decode_chunk(mc_archive *a, uint64_t chunk_off, mc_u8 *out, int nthreads){
    if(!a||!out) return;
    if(chunk_off<=MC_SLOT_ZERO){ memset(out,0,(size_t)MC_CHUNK*MC_CHUNK*MC_CHUNK); return; }
    dchunk_ctx c={.a=a,.chunk_off=chunk_off,.out=out,.q=mc_chunk_q(a->base,chunk_off)};
    atomic_store(&c.next,0);
    int nt=auto_threads(nthreads);
    if(nt<=1){ dchunk_worker(&c); return; }
    pthread_t th[16];
    for(int t=0;t<nt;++t) pthread_create(&th[t],NULL,dchunk_worker,&c);
    for(int t=0;t<nt;++t) pthread_join(th[t],NULL);
}

// parallel encode: stripes of blocks into per-worker buffers, stitched in
// bitmap order so the blob is byte-identical to the serial path.
#define ENC_STRIPES 16
typedef struct {
    const mc_u8 *vox; float q;
    mc_buf bufs[ENC_STRIPES];
    uint16_t blen[MC_GRID3];
    uint8_t  bm[MC_BITMAP_BYTES];
    uint8_t  frac[MC_GRID3];
    pthread_mutex_t bm_mu;
    _Atomic uint32_t next;
} echunk_ctx;
static void *echunk_worker(void *p){
    echunk_ctx *c=p;
    mc_set_quality(c->q);
    static _Thread_local u8 blk[MC_BLK*MC_BLK*MC_BLK];
    for(;;){
        uint32_t s=atomic_fetch_add_explicit(&c->next,1,memory_order_relaxed);
        if(s>=ENC_STRIPES) break;
        uint32_t b0=s*(MC_GRID3/ENC_STRIPES), b1=b0+(MC_GRID3/ENC_STRIPES);
        for(uint32_t bi=b0;bi<b1;++bi){
            int bz=(int)(bi>>8), by=(int)((bi>>4)&15), bx=(int)(bi&15);
            if(!gather_blk256(c->vox,bz,by,bx,blk)) continue;
            { int cnt=0; for(int i=0;i<MC_BLK*MC_BLK*MC_BLK;++i) cnt+=blk[i]!=0;
              c->frac[bi]=(uint8_t)((cnt*15+2048)/4096);
              if(cnt&&!c->frac[bi]) c->frac[bi]=1; }
            uint32_t len=0;
            if(mc_enc_block(blk,&c->bufs[s],&len)){
                c->blen[bi]=(uint16_t)len;
                pthread_mutex_lock(&c->bm_mu);
                mc_bit_set(c->bm,bi);
                pthread_mutex_unlock(&c->bm_mu);
            }
        }
    }
    return NULL;
}
int mc_archive_append_chunk_par(mc_archive *a, int lod, int cz,int cy,int cx,
                                const mc_u8 vox[256*256*256], float q, int nthreads){
    if(!a||lod<0||lod>7||!vox) return -1;
    echunk_ctx *c=calloc(1,sizeof *c);
    c->vox=vox; c->q=q>0?q:a->quality;
    pthread_mutex_init(&c->bm_mu,NULL);
    atomic_store(&c->next,0);
    int nt=auto_threads(nthreads); if(nt>ENC_STRIPES)nt=ENC_STRIPES;
    if(nt<=1) echunk_worker(c);
    else {
        pthread_t th[16];
        for(int t=0;t<nt;++t) pthread_create(&th[t],NULL,echunk_worker,c);
        for(int t=0;t<nt;++t) pthread_join(th[t],NULL);
    }
    int npresent=0; for(int i=0;i<MC_BITMAP_BYTES;++i) npresent+=__builtin_popcount(c->bm[i]);
    int rc=0;
    if(npresent){
        // stitch: stripes hold payloads in ascending-bi order within the stripe,
        // so concatenating stripes in order matches the serial blob layout.
        stage_t st={0};
        static _Thread_local uint16_t lens16[MC_GRID3];
        int nl=0;
        for(int bi=0;bi<MC_GRID3;++bi) if(mc_bit_get(c->bm,bi)) lens16[nl++]=c->blen[bi];
        float qq=c->q; uint64_t h=0;
        static _Thread_local uint8_t fmap[MC_GRID3/2+64];
        uint16_t fml=(uint16_t)mc_enc_fracmap(c->frac,fmap,sizeof fmap);
        stage_put(&st,&qq,4); stage_put(&st,&h,8);          // hash patched below
        stage_put(&st,&fml,2); stage_put(&st,fmap,fml);
        stage_put(&st,c->bm,MC_BITMAP_BYTES);
        stage_put(&st,lens16,(size_t)nl*2);
        for(int s=0;s<ENC_STRIPES;++s) if(c->bufs[s].len) stage_put(&st,c->bufs[s].p,c->bufs[s].len);
        h=mc_chunk_compute_hash(st.p,(uint64_t)st.len);     // same bytes => same hash as serial
        memcpy(st.p+4,&h,8);
        rc=w_install_blob(a,lod,cz,cy,cx,st.p,st.len);
        free(st.p);
    }
    for(int s=0;s<ENC_STRIPES;++s) free(c->bufs[s].p);
    pthread_mutex_destroy(&c->bm_mu);
    free(c);
    return rc;
}

int mc_archive_block_present(mc_archive *a, int lod, int bz, int by, int bx){
    if(!a||lod<0||lod>7||bz<0||by<0||bx<0) return 0;
    uint64_t co=mc_archive_chunk_offset(a,lod,bz>>4,by>>4,bx>>4);
    if(co<=MC_SLOT_ZERO) return 0;
    const u8 *bm=a->base+co+MC_BLOB_HDR+mc_chunk_fmaplen(a->base,co);
    return mc_bit_get(bm,((bz&15)*16+(by&15))*16+(bx&15));
}

float mc_archive_block_fraction(mc_archive *a, int lod, int bz, int by, int bx){
    if(!a||lod<0||lod>7||bz<0||by<0||bx<0) return 0.0f;
    uint64_t co=mc_archive_chunk_offset(a,lod,bz>>4,by>>4,bx>>4);
    if(co<=MC_SLOT_ZERO) return 0.0f;
    static _Thread_local uint8_t fr[MC_GRID3];
    static _Thread_local uint64_t fr_key=~0ull;
    if(fr_key!=co){
        uint16_t fml=mc_chunk_fmaplen(a->base,co);
        if(!fml) return 0.0f;
        mc_dec_fracmap(a->base+co+MC_BLOB_HDR,fml,fr);
        fr_key=co;
    }
    return (float)fr[((bz&15)*16+(by&15))*16+(bx&15)]/15.0f;
}

static inline uint64_t mc_rng64(uint64_t *s){
    uint64_t x=*s; x^=x<<13; x^=x>>7; x^=x<<17; *s=x; return x;
}
int mc_archive_sample_boxes(mc_archive *a, int lod, uint64_t seed, int count,
                            long dz, long dy, long dx, float min_frac,
                            mc_box *out){
    if(!a||!out||count<=0||dz<=0||dy<=0||dx<=0) return 0;
    uint32_t unx,uny,unz;
    memcpy(&unx,a->base+MCH_NX,4); memcpy(&uny,a->base+MCH_NY,4); memcpy(&unz,a->base+MCH_NZ,4);
    long NXl=(long)unx>>lod, NYl=(long)uny>>lod, NZl=(long)unz>>lod;
    if(NZl<dz||NYl<dy||NXl<dx) return 0;
    uint64_t s=seed?seed:0x9E3779B97F4A7C15ull;
    int got=0; long attempts=0, max_attempts=(long)count*256;
    while(got<count && attempts++<max_attempts){
        long z0=(long)(mc_rng64(&s)%(uint64_t)(NZl-dz+1));
        long y0=(long)(mc_rng64(&s)%(uint64_t)(NYl-dy+1));
        long x0=(long)(mc_rng64(&s)%(uint64_t)(NXl-dx+1));
        // mean block fraction over the touched block grid
        long bz0=z0>>4, bz1=(z0+dz-1)>>4, by0=y0>>4, by1=(y0+dy-1)>>4, bx0=x0>>4, bx1=(x0+dx-1)>>4;
        double fsum=0; long nb=0;
        for(long bz=bz0;bz<=bz1;++bz)for(long by=by0;by<=by1;++by)for(long bx=bx0;bx<=bx1;++bx){
            fsum+=mc_archive_block_fraction(a,lod,(int)bz,(int)by,(int)bx); nb++;
        }
        if(nb && fsum/nb>=min_frac){ out[got].z0=z0; out[got].y0=y0; out[got].x0=x0; got++; }
    }
    return got;
}

typedef struct {
    mc_archive *a; int lod;
    const mc_box *boxes; int n;
    long dz,dy,dx;
    mc_u8 *out; size_t bstride;
    _Atomic uint32_t next;
} crops_ctx;
static void *crops_worker(void *p){
    crops_ctx *c=p;
    for(;;){
        uint32_t i=atomic_fetch_add_explicit(&c->next,1,memory_order_relaxed);
        if(i>=(uint32_t)c->n) break;
        mc_archive_read_region(c->a,c->lod,c->boxes[i].z0,c->boxes[i].y0,c->boxes[i].x0,
                               c->dz,c->dy,c->dx,
                               c->out+(size_t)i*c->bstride,
                               (size_t)c->dy*c->dx,(size_t)c->dx,1);
    }
    return NULL;
}
void mc_archive_read_regions(mc_archive *a, int lod, const mc_box *boxes, int n,
                             long dz, long dy, long dx,
                             mc_u8 *out, size_t batch_stride, int nthreads){
    if(!a||!boxes||n<=0||!out) return;
    crops_ctx c={.a=a,.lod=lod,.boxes=boxes,.n=n,.dz=dz,.dy=dy,.dx=dx,
                 .out=out,.bstride=batch_stride};
    atomic_store(&c.next,0);
    int nt=auto_threads(nthreads); if(nt>n)nt=n;
    if(nt<=1){ crops_worker(&c); return; }
    pthread_t th[16];
    for(int t=0;t<nt;++t) pthread_create(&th[t],NULL,crops_worker,&c);
    for(int t=0;t<nt;++t) pthread_join(th[t],NULL);
}

// ---- region read ------------------------------------------------------------
typedef struct {
    mc_archive *a; int lod;
    long z0,y0,x0,dz,dy,dx;
    mc_u8 *out; size_t sz,sy;
    int nbz,nby,nbx; long bz0,by0,bx0;     // touched block range
    _Atomic uint32_t next;
} region_ctx;
static void *region_worker(void *p){
    region_ctx *c=p;
    mc_u8 blk[MC_BLK*MC_BLK*MC_BLK];
    uint32_t nb=(uint32_t)(c->nbz*c->nby*c->nbx);
    for(;;){
        uint32_t w=atomic_fetch_add_explicit(&c->next,1,memory_order_relaxed);
        if(w>=nb) break;
        long bz=c->bz0+w/(c->nby*c->nbx);
        long by=c->by0+(w/c->nbx)%c->nby;
        long bx=c->bx0+w%c->nbx;
        uint64_t co=mc_archive_chunk_offset(c->a,c->lod,(int)(bz>>4),(int)(by>>4),(int)(bx>>4));
        long gz=bz*16, gy=by*16, gx=bx*16;            // block origin in voxels
        // intersection of this block with the region
        long iz0=gz>c->z0?gz:c->z0, iz1=(gz+16<c->z0+c->dz)?gz+16:c->z0+c->dz;
        long iy0=gy>c->y0?gy:c->y0, iy1=(gy+16<c->y0+c->dy)?gy+16:c->y0+c->dy;
        long ix0=gx>c->x0?gx:c->x0, ix1=(gx+16<c->x0+c->dx)?gx+16:c->x0+c->dx;
        if(iz0>=iz1||iy0>=iy1||ix0>=ix1) continue;
        int present=0;
        if(co){
            const u8 *bm=c->a->base+co+MC_BLOB_HDR+mc_chunk_fmaplen(c->a->base,co);
            present=mc_bit_get(bm,(int)(((bz&15)*16+(by&15))*16+(bx&15)));
        }
        if(present) mc_archive_decode_block(c->a,co,(int)(bz&15),(int)(by&15),(int)(bx&15),blk);
        long nrow=ix1-ix0;
        for(long z=iz0;z<iz1;++z)for(long y=iy0;y<iy1;++y){
            mc_u8 *dst=c->out+(size_t)(z-c->z0)*c->sz+(size_t)(y-c->y0)*c->sy+(size_t)(ix0-c->x0);
            if(present) memcpy(dst,blk+((size_t)(z-gz)*16+(y-gy))*16+(ix0-gx),(size_t)nrow);
            else        memset(dst,0,(size_t)nrow);
        }
    }
    return NULL;
}
void mc_archive_read_region(mc_archive *a, int lod,
                            long z0, long y0, long x0,
                            long dz, long dy, long dx,
                            mc_u8 *out, size_t sz, size_t sy, int nthreads){
    if(!a||lod<0||lod>7||!out||dz<=0||dy<=0||dx<=0) return;
    region_ctx c={.a=a,.lod=lod,.z0=z0,.y0=y0,.x0=x0,.dz=dz,.dy=dy,.dx=dx,
                  .out=out,.sz=sz,.sy=sy};
    c.bz0=z0>>4; c.by0=y0>>4; c.bx0=x0>>4;
    c.nbz=(int)(((z0+dz+15)>>4)-c.bz0);
    c.nby=(int)(((y0+dy+15)>>4)-c.by0);
    c.nbx=(int)(((x0+dx+15)>>4)-c.bx0);
    atomic_store(&c.next,0);
    int nt=auto_threads(nthreads);
    uint32_t nb=(uint32_t)(c.nbz*c.nby*c.nbx);
    if((uint32_t)nt>nb) nt=(int)nb;
    if(nt<=1){ region_worker(&c); return; }
    pthread_t th[16];
    for(int t=0;t<nt;++t) pthread_create(&th[t],NULL,region_worker,&c);
    for(int t=0;t<nt;++t) pthread_join(th[t],NULL);
}

void mc_archive_close(mc_archive *a){
    if(!a) return;
    uint64_t cur = atomic_load(&a->cursor);
    w_write_u64(a, MCH_TOTLEN, cur);
    msync(a->base, cur, MS_SYNC);
    munmap(a->base, a->reserve);
    if(ftruncate(a->fd,(off_t)cur)!=0) perror("mc_archive_close: ftruncate");
    close(a->fd);
    pthread_mutex_destroy(&a->grow_mu);
    free(a);
}

#else // !MC_HAVE_MMAP — the appendable archive requires mmap/ftruncate.
mc_archive *mc_archive_open(const char *p,int d,float q){ (void)p;(void)d;(void)q;
    fprintf(stderr,"mc_archive: requires a POSIX mmap platform\n"); return NULL; }
int mc_archive_append_chunk_raw(mc_archive*a,int l,int z,int y,int x,const mc_u8*v){ (void)a;(void)l;(void)z;(void)y;(void)x;(void)v; return -1; }
int mc_archive_append_chunk_compressed(mc_archive*a,int l,int z,int y,int x,const uint8_t*b,size_t n){ (void)a;(void)l;(void)z;(void)y;(void)x;(void)b;(void)n; return -1; }
mc_cover mc_archive_chunk_coverage(mc_archive*a,int l,int z,int y,int x){ (void)a;(void)l;(void)z;(void)y;(void)x; return MC_ABSENT; }
uint64_t mc_archive_chunk_offset(mc_archive*a,int l,int z,int y,int x){ (void)a;(void)l;(void)z;(void)y;(void)x; return 0; }
void mc_archive_decode_block(mc_archive*a,uint64_t o,int z,int y,int x,mc_u8*d){ (void)a;(void)o;(void)z;(void)y;(void)x; memset(d,0,MC_BLK*MC_BLK*MC_BLK); }
void mc_archive_close(mc_archive*a){ (void)a; }
#endif

// ============================================================================
// VERIFY — walk every chunk of every LOD, recompute xxh64, compare to stored.
// ============================================================================
long mc_verify_archive(const uint8_t *arc, size_t len, int verbose){
    (void)len;
    long bad=0, total=0;
    for(int lod=0;lod<8;++lod){
        uint64_t root; memcpy(&root,arc+MCH_ROOTOFF+(uint64_t)lod*8,8);
        if(!root) continue;
        for(int n2=0;n2<MC_GRID3;++n2){
            uint64_t inner; memcpy(&inner,arc+root+(size_t)n2*8,8);
            if(!inner) continue;
            for(int n1=0;n1<MC_GRID3;++n1){
                uint64_t shard; memcpy(&shard,arc+inner+(size_t)n1*8,8);
                if(!shard) continue;
                for(int n0=0;n0<MC_GRID3;++n0){
                    uint64_t co; memcpy(&co,arc+shard+(size_t)n0*8,8);
                    if(co<=MC_SLOT_ZERO) continue;
                    total++;
                    uint64_t blen=mc_chunk_blob_len(arc,co);
                    uint64_t want=mc_chunk_stored_hash(arc,co);
                    uint64_t got=mc_chunk_compute_hash(arc+co,blen);
                    if(want!=got){
                        bad++;
                        if(verbose) fprintf(stderr,"mc_verify: lod %d node %d/%d/%d chunk@%llu CORRUPT\n",
                            lod,n2,n1,n0,(unsigned long long)co);
                    }
                }
            }
        }
    }
    if(verbose) fprintf(stderr,"mc_verify: %ld chunks checked, %ld corrupt\n",total,bad);
    return bad;
}

// ============================================================================
// READER (flat buffer + streaming byte-source)
// ============================================================================
const char *mc_metadata(const uint8_t *arc, size_t *out_len){
    uint64_t off,len; memcpy(&off,arc+MCH_METAOFF,8); memcpy(&len,arc+MCH_METALEN,8);
    if(!off) off=MC_HDR;
    if(out_len) *out_len=(size_t)len;
    return (const char*)(arc+off);
}

#define MC_RD_NODE_CACHE 64    // cached node tables per streaming reader (64*32KB = 2MB)
struct mc_reader {
    const uint8_t *arc;        // flat mode: archive buffer; streaming: NULL
    size_t len;
    uint64_t roots[8];
    int nx, ny, nz;            // LOD0 dims from the header
    float quality;             // quality the archive was built at
    u8 priors[MC_PRIORS_BYTES]; int has_priors;   // raw per-volume prior blob
    mc_read_fn read; void *read_ud;   // streaming mode
    // streaming scratch: a fetched window of the current chunk blob.
    u8 *cbuf; uint64_t cbuf_off; uint64_t cbuf_len;
    // partial-fetch mode: per-chunk header cache (bitmap + lens) + one payload.
    int partial;
    u8 hdr[MC_BLOB_HDR + MC_BITMAP_BYTES + MC_GRID3*2]; uint64_t hdr_off; int hdr_np; uint16_t hdr_fml;
    u8 *pbuf; size_t pbuf_cap;
    // streaming node-table cache: resolving a chunk needs 3 dependent 32KB
    // table reads; without a cache every resolve re-fetches them (3 GETs per
    // chunk over S3). FIFO of the last MC_RD_NODE_CACHE tables.
    uint64_t ntab_off[MC_RD_NODE_CACHE];
    u8 *ntab[MC_RD_NODE_CACHE];
    int ntab_next;
};

static void reader_hdr_load(mc_reader *r, const u8 *hdr){
    for(int l=0;l<8;++l) memcpy(&r->roots[l], hdr+MCH_ROOTOFF+l*8, 8);
    uint32_t ux,uy,uz;
    memcpy(&ux,hdr+MCH_NX,4); memcpy(&uy,hdr+MCH_NY,4); memcpy(&uz,hdr+MCH_NZ,4);
    r->nx=(int)ux; r->ny=(int)uy; r->nz=(int)uz;
    memcpy(&r->quality,hdr+MCH_QUALITY,4);
}

mc_reader *mc_open(const uint8_t *arc, size_t len){
    mc_codec_init();
    mc_reader *r=calloc(1,sizeof *r); r->arc=arc; r->len=len;
    reader_hdr_load(r, arc);
    uint64_t poff; memcpy(&poff,arc+MCH_PRIOROFF,8);
    if(poff){ memcpy(r->priors,arc+poff,MC_PRIORS_BYTES); r->has_priors=1; }
    priors_load(arc);
    return r;
}

// streaming: fetch exactly len bytes at off into dst (via callback).
static int sread(mc_reader *r, uint64_t off, uint32_t len, u8 *dst){
    return r->read(r->read_ud, off, len, dst);
}

mc_reader *mc_open_streaming(mc_read_fn read, void *ud, uint64_t total_len){
    mc_codec_init();
    mc_reader *r=calloc(1,sizeof *r); r->read=read; r->read_ud=ud; r->len=(size_t)total_len;
    u8 hdr[MC_HDR];
    if(read(ud,0,MC_HDR,hdr)!=0){ free(r); return NULL; }
    uint32_t magic; memcpy(&magic,hdr+MCH_MAGIC,4);
    if(magic!=MC_MAGIC){ free(r); return NULL; }
    reader_hdr_load(r, hdr);
    // per-volume priors: flat open gets them via priors_load(arc); a streaming
    // reader must pull the blob through the callback or HF decode is wrong.
    uint64_t poff; memcpy(&poff,hdr+MCH_PRIOROFF,8);
    if(poff){
        u8 pb[MC_PRIORS_BYTES];
        uint32_t pm=0;
        if(read(ud,poff,MC_PRIORS_BYTES,pb)==0) memcpy(&pm,pb,4);
        if(pm==MC_PRIORS_MAGIC){
            memcpy(r->priors,pb,MC_PRIORS_BYTES); r->has_priors=1;
            mc_codec_set_priors((const uint16_t*)(pb+8),(const uint16_t*)(pb+8+8*32*2));
        } else {
            mc_codec_set_priors(NULL,NULL);
        }
    } else {
        mc_codec_set_priors(NULL,NULL);
    }
    return r;
}

// Raw per-volume prior arrays (plo/phi as u16[8][32]); 0 if the archive has none.
// Feed into mc_archive_set_priors to make a local mirror decode identically.
int mc_reader_priors(mc_reader *r, const uint16_t **plo, const uint16_t **phi){
    if(!r||!r->has_priors) return 0;
    if(plo)*plo=(const uint16_t*)(r->priors+8);
    if(phi)*phi=(const uint16_t*)(r->priors+8+8*32*2);
    return 1;
}

// Total byte length of the chunk blob at `chunk_off` (flat or streaming reader).
// 0 on error. Pair with mc_chunk_offset to range-copy compressed chunks verbatim.
uint64_t mc_reader_chunk_blob_len(mc_reader *r, uint64_t chunk_off){
    if(!r||!chunk_off) return 0;
    if(r->arc) return mc_chunk_blob_len(r->arc, chunk_off);
    u8 h[MC_BLOB_HDR];
    if(sread(r,chunk_off,MC_BLOB_HDR,h)!=0) return 0;
    uint16_t fml; memcpy(&fml,h+MC_BLOB_HDR-2,2);
    const uint64_t bm_off = chunk_off + MC_BLOB_HDR + fml;
    u8 bm[MC_BITMAP_BYTES];
    if(sread(r,bm_off,MC_BITMAP_BYTES,bm)!=0) return 0;
    int np=0; for(int i=0;i<MC_BITMAP_BYTES;++i) np+=__builtin_popcount(bm[i]);
    if(!np) return bm_off + MC_BITMAP_BYTES - chunk_off;
    u8 *lens=malloc((size_t)np*2);
    if(!lens||sread(r,bm_off+MC_BITMAP_BYTES,(uint32_t)((size_t)np*2),lens)!=0){ free(lens); return 0; }
    uint64_t pay=0; for(int i=0;i<np;++i){ uint16_t l; memcpy(&l,lens+(size_t)i*2,2); pay+=l; }
    free(lens);
    return bm_off + MC_BITMAP_BYTES + (uint64_t)np*2 + pay - chunk_off;
}

void mc_reader_dims(mc_reader *r, int *nx, int *ny, int *nz){
    if(nx)*nx=r?r->nx:0; if(ny)*ny=r?r->ny:0; if(nz)*nz=r?r->nz:0;
}
float mc_reader_quality(mc_reader *r){ return r?r->quality:0.f; }
int mc_reader_nlods(mc_reader *r){
    if(!r) return 0;
    int n=0; for(int l=0;l<8;++l) if(r->roots[l]) n=l+1;
    return n;
}

void mc_close(mc_reader *r){ if(!r)return;
    for(int i=0;i<MC_RD_NODE_CACHE;++i) free(r->ntab[i]);
    free(r->cbuf); free(r->pbuf); free(r); }
// Partial-fetch mode (streaming readers only): decode a block by fetching just
// the chunk's bitmap+length table (cached per chunk, <=8.7KB) plus that block's
// own payload, instead of the whole chunk blob. Wins cold random-access latency
// over high-latency byte sources (S3); leave OFF when scanning whole chunks.
void mc_reader_set_partial_fetch(mc_reader *r, int on){ if(r){ r->partial=on; r->hdr_off=~0ull; } }
void mc_reader_set_quality(mc_reader *r, float q){ (void)r; mc_set_quality(q); }

// streaming chunk-offset resolve: pull node tables on demand (each is
// MC_NODE_BYTES), memoized in the reader's FIFO node-table cache so repeated
// resolves (scans, neighborhoods) cost zero extra reads.
static const u8 *sfetch_node(mc_reader *r, uint64_t off){
    for(int i=0;i<MC_RD_NODE_CACHE;++i)
        if(r->ntab[i] && r->ntab_off[i]==off) return r->ntab[i];
    int slot=r->ntab_next; r->ntab_next=(slot+1)%MC_RD_NODE_CACHE;
    if(!r->ntab[slot]) r->ntab[slot]=malloc(MC_NODE_BYTES);
    if(sread(r,off,MC_NODE_BYTES,r->ntab[slot])!=0){ free(r->ntab[slot]); r->ntab[slot]=0; return NULL; }
    r->ntab_off[slot]=off;
    return r->ntab[slot];
}
static uint64_t sresolve_chunk(mc_reader *r,int lod,int cz,int cy,int cx){
    uint64_t node = r->roots[lod];
    for(int nib=MC_TREE_LEVELS-1; nib>=0; --nib){
        if(!node) return 0;
        const u8 *tbl=sfetch_node(r,node);
        if(!tbl) return 0;
        int idx=(mc_nib(cz,nib)*16+mc_nib(cy,nib))*16+mc_nib(cx,nib);
        uint64_t child; memcpy(&child,tbl+(size_t)idx*8,8);
        node=child;
    }
    return node;
}

uint64_t mc_chunk_offset(mc_reader *r,int lod,int cz,int cy,int cx){
    if(lod<0||lod>7) return 0;
    if(r->arc) return mc_resolve_chunk(r->arc,r->roots[lod],cz,cy,cx);
    return sresolve_chunk(r,lod,cz,cy,cx);
}

// streaming: ensure the whole chunk blob at chunk_off is cached in r->cbuf, return ptr.
static const u8 *sfetch_chunk(mc_reader *r, uint64_t chunk_off){
    if(r->cbuf && r->cbuf_off==chunk_off) return r->cbuf;
    // fetch header (q/hash/fmaplen), then bitmap + lens for total length, then
    // the full blob in one GET.
    u8 bh[MC_BLOB_HDR];
    if(sread(r,chunk_off,MC_BLOB_HDR,bh)!=0) return NULL;
    uint16_t fml; memcpy(&fml,bh+12,2);
    uint64_t bm_off = chunk_off + MC_BLOB_HDR + fml;
    u8 bm[MC_BITMAP_BYTES];
    if(sread(r,bm_off,MC_BITMAP_BYTES,bm)!=0) return NULL;
    int npresent=0; for(int i=0;i<MC_BITMAP_BYTES;++i) npresent+=__builtin_popcount(bm[i]);
    u8 *lens=malloc((size_t)npresent*2);
    if(npresent && sread(r,bm_off+MC_BITMAP_BYTES,(uint32_t)(npresent*2),lens)!=0){ free(lens); return NULL; }
    uint64_t paybytes=0; for(int s=0;s<npresent;++s){ uint16_t l; memcpy(&l,lens+(size_t)s*2,2); paybytes+=l; }
    free(lens);
    uint64_t total = (bm_off+MC_BITMAP_BYTES+(uint64_t)npresent*2+paybytes) - chunk_off;
    r->cbuf = realloc(r->cbuf, total);
    if(sread(r,chunk_off,(uint32_t)total,r->cbuf)!=0) return NULL;
    r->cbuf_off=chunk_off; r->cbuf_len=total;
    return r->cbuf;
}

// partial-fetch path: header cache + single-payload range read.
static int spartial_decode(mc_reader *r, uint64_t chunk_off, int bi, mc_u8 *dst){
    if(r->hdr_off!=chunk_off){
        if(sread(r,chunk_off,MC_BLOB_HDR,r->hdr)!=0) return -1;
        uint16_t fml; memcpy(&fml,r->hdr+12,2);
        r->hdr_fml=fml;
        if(sread(r,chunk_off+MC_BLOB_HDR+fml,MC_BITMAP_BYTES,r->hdr+MC_BLOB_HDR)!=0) return -1;
        const u8 *bm0=r->hdr+MC_BLOB_HDR;
        int np=0; for(int i=0;i<MC_BITMAP_BYTES;++i) np+=__builtin_popcount(bm0[i]);
        if(np && sread(r,chunk_off+MC_BLOB_HDR+fml+MC_BITMAP_BYTES,(uint32_t)(np*2),r->hdr+MC_BLOB_HDR+MC_BITMAP_BYTES)!=0) return -1;
        r->hdr_np=np; r->hdr_off=chunk_off;
    }
    { float q; memcpy(&q,r->hdr,4); mc_set_quality(q); }
    const u8 *bm=r->hdr+MC_BLOB_HDR;
    if(!mc_bit_get(bm,bi)){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return 0; }
    int slot=mc_rank(bm,bi);
    const u8 *lens=bm+MC_BITMAP_BYTES;
    uint64_t cum=0; for(int s2=0;s2<slot;++s2){ uint16_t l; memcpy(&l,lens+(size_t)s2*2,2); cum+=l; }
    uint16_t mylen; memcpy(&mylen,lens+(size_t)slot*2,2);
    uint64_t pay=chunk_off+MC_BLOB_HDR+r->hdr_fml+MC_BITMAP_BYTES+(uint64_t)r->hdr_np*2+cum;
    if(r->pbuf_cap<mylen){ r->pbuf=realloc(r->pbuf,mylen); r->pbuf_cap=mylen; }
    if(sread(r,pay,mylen,r->pbuf)!=0) return -1;
    mc_dec_block(r->pbuf,mylen,dst);
    return 0;
}

void mc_decode_block(mc_reader *r, uint64_t chunk_off, int bz,int by,int bx, mc_u8 *dst){
    if(chunk_off<=MC_SLOT_ZERO){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    if(!r->arc && r->partial){
        if(spartial_decode(r,chunk_off,(bz*16+by)*16+bx,dst)==0) return;
        memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return;
    }
    // resolve the chunk-blob base pointer (flat mmap or streamed window).
    const u8 *blob_base;       // points at the chunk blob start
    uint64_t blob_origin;      // absolute archive offset of that blob start
    if(r->arc){ blob_base=r->arc; blob_origin=0; }            // absolute offsets index into arc
    else {
        const u8 *cb = sfetch_chunk(r,chunk_off);
        if(!cb){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
        blob_base = cb - chunk_off;   // so blob_base + abs_off lands inside the window
        blob_origin = 0;
    }
    (void)blob_origin;
    mc_set_quality(mc_chunk_q(blob_base,chunk_off));
    uint64_t boff; uint32_t blen;
    if(!mc_block_range(blob_base,chunk_off,bz,by,bx,&boff,&blen)){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    mc_dec_block(blob_base+boff,blen,dst);
}

// ============================================================================
// mc_cache.c — sharded CLOCK/NRU decoded-block cache. See mc_cache.h.
// ============================================================================
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>
#include <unistd.h>

#if defined(__unix__) || defined(__APPLE__)
  #include <sys/mman.h>
  #include <pthread.h>
  #define MC_CACHE_MMAP 1
#else
  #define MC_CACHE_MMAP 0
#endif

#define BLK_BYTES 4096u
#define NSHARD    64          // power of two
#define EMPTY_KEY 0ull

// key: 1 (always-set guard) | lod(3) | bz(20) | by(20) | bx(20)
static inline uint64_t bkey(int lod,int bz,int by,int bx){
    return (1ull<<63) | ((uint64_t)(lod&7)<<60)
         | ((uint64_t)(bz&0xFFFFF)<<40) | ((uint64_t)(by&0xFFFFF)<<20) | (uint64_t)(bx&0xFFFFF);
}
static inline uint64_t khash(uint64_t k){
    k^=k>>33; k*=0xFF51AFD7ED558CCDull; k^=k>>33; k*=0xC4CEB9FE1A85EC53ull; k^=k>>33;
    return k;
}

// one shard: its own slice of the arena, its own map, lock, eviction state.
typedef struct {
    pthread_mutex_t mu;
    uint64_t *map_key;        // open addressing, linear probe, backward-shift delete
    uint32_t *map_slot;
    uint32_t  map_cap;        // power of two
    uint64_t *slot_key;       // per-slot reverse key (EMPTY_KEY = free)
    uint8_t  *slot_ref;       // CLOCK ref bit / S3-FIFO 2-bit freq
    uint32_t  nslot, used, hand;
    mc_u8    *arena;          // nslot * 4KB
    uint64_t  hits, misses, evictions;
    uint32_t *slot_epoch;            // pin: slot used by the current epoch
    // S3-FIFO state: two slot-id rings + a ghost fingerprint ring w/ set.
    uint32_t *fs, fs_head, fs_tail, fs_cap;     // small queue (ring)
    uint32_t *fm, fm_head, fm_tail, fm_cap;     // main queue (ring)
    uint32_t *gfp, g_head, g_cap;               // ghost fingerprints (ring)
    int32_t  *gset; uint32_t gset_cap;          // fp -> ring idx, open addressing
    uint8_t  *slot_inmain;                      // which queue a slot lives in
} shard_t;

#define FREQ_MAX 3

#define MISSQ_CAP 65536
struct mc_cache {
    int policy;               // mc_cache_policy
    _Atomic int frozen;
    _Atomic int phase_mode;          // set by the first freeze(); until then the
                                     // cache is the plain always-thread-safe
                                     // multi-reader/multi-writer cache and pins
                                     // are inert (no behavior change for
                                     // clients that never tick)
    _Atomic uint32_t epoch;          // bumped at thaw; pins compare against it
    _Atomic uint64_t missq[MISSQ_CAP];
    _Atomic uint32_t miss_w;
    shard_t sh[NSHARD];
    mc_cache_src_fn src; void *src_ud;
    size_t arena_bytes;
    void *arena_base;
    // reader binding (serialized decode)
    pthread_mutex_t rd_mu;
    struct mc_reader *rd;
    struct mc_archive *ar;
};

static uint32_t pow2_at_least(uint32_t v){ uint32_t p=1; while(p<v)p<<=1; return p; }
static inline int slot_pinned(mc_cache *c, shard_t *sh, uint32_t slot);
static inline int cache_frozen(mc_cache *c){ return atomic_load_explicit(&c->frozen,memory_order_acquire); }
static void miss_record(mc_cache *c, uint64_t key){
    uint32_t i=atomic_fetch_add_explicit(&c->miss_w,1,memory_order_relaxed);
    atomic_store_explicit(&c->missq[i&(MISSQ_CAP-1)],key,memory_order_relaxed);
}

mc_cache *mc_cache_new(size_t bytes, mc_cache_src_fn src, void *src_ud){
    mc_cache *c=calloc(1,sizeof *c);
    c->src=src; c->src_ud=src_ud;
    pthread_mutex_init(&c->rd_mu,NULL);
    size_t nslot_total = bytes/BLK_BYTES; if(nslot_total<NSHARD) nslot_total=NSHARD;
    uint32_t per = (uint32_t)(nslot_total/NSHARD); if(per<1)per=1;
    c->arena_bytes = (size_t)per*NSHARD*BLK_BYTES;
#if MC_CACHE_MMAP
    c->arena_base = mmap(NULL,c->arena_bytes,PROT_READ|PROT_WRITE,
                         MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
    if(c->arena_base==MAP_FAILED){ free(c); return NULL; }
#else
    c->arena_base = malloc(c->arena_bytes);
    if(!c->arena_base){ free(c); return NULL; }
#endif
    for(int s=0;s<NSHARD;++s){
        shard_t *sh=&c->sh[s];
        pthread_mutex_init(&sh->mu,NULL);
        sh->nslot=per; sh->hand=0; sh->used=0;
        sh->arena=(mc_u8*)c->arena_base + (size_t)s*per*BLK_BYTES;
        sh->map_cap=pow2_at_least(per*2);
        sh->map_key=calloc(sh->map_cap,8);
        sh->map_slot=calloc(sh->map_cap,4);
        sh->slot_key=calloc(per,8);
        sh->slot_ref=calloc(per,1);
        sh->fs_cap=per+1; sh->fm_cap=per+1;
        sh->fs=malloc(4u*sh->fs_cap); sh->fm=malloc(4u*sh->fm_cap);
        sh->g_cap=per; sh->gfp=calloc(sh->g_cap,4);
        sh->gset_cap=pow2_at_least(per*2); sh->gset=malloc(4u*sh->gset_cap);
        memset(sh->gset,0xFF,4u*sh->gset_cap);
        sh->slot_inmain=calloc(per,1);
        sh->slot_epoch=calloc(per,4);
    }
    atomic_store(&c->epoch,1);
    return c;
}
void mc_cache_set_policy(mc_cache *c, mc_cache_policy p){ if(c) c->policy=(int)p; }

// free a shard's lookup/eviction tables (NOT its mutex, NOT the shared arena).
static void shard_free_tables(shard_t *sh){
    free(sh->map_key); free(sh->map_slot); free(sh->slot_key); free(sh->slot_ref);
    free(sh->fs); free(sh->fm); free(sh->gfp); free(sh->gset); free(sh->slot_inmain);
    free(sh->slot_epoch);
}

// (re)allocate a shard's tables for `per` slots pointing at `arena_slice`.
// Resets the shard to empty. Mutex assumed already init'd + held during resize.
static void shard_init_tables(shard_t *sh, mc_u8 *arena_slice, uint32_t per){
    sh->nslot=per; sh->hand=0; sh->used=0;
    sh->arena=arena_slice;
    sh->map_cap=pow2_at_least(per*2);
    sh->map_key=calloc(sh->map_cap,8);
    sh->map_slot=calloc(sh->map_cap,4);
    sh->slot_key=calloc(per,8);
    sh->slot_ref=calloc(per,1);
    sh->fs_cap=per+1; sh->fm_cap=per+1;
    sh->fs=malloc(4u*sh->fs_cap); sh->fm=malloc(4u*sh->fm_cap);
    sh->fs_head=sh->fs_tail=sh->fm_head=sh->fm_tail=0;
    sh->g_cap=per; sh->gfp=calloc(sh->g_cap,4);
    sh->gset_cap=pow2_at_least(per*2); sh->gset=malloc(4u*sh->gset_cap);
    memset(sh->gset,0xFF,4u*sh->gset_cap);
    sh->slot_inmain=calloc(per,1);
    sh->slot_epoch=calloc(per,4);
}

void mc_cache_free(mc_cache *c){
    if(!c) return;
    for(int s=0;s<NSHARD;++s){
        shard_t *sh=&c->sh[s];
        pthread_mutex_destroy(&sh->mu);
        shard_free_tables(sh);
    }
#if MC_CACHE_MMAP
    munmap(c->arena_base,c->arena_bytes);
#else
    free(c->arena_base);
#endif
    pthread_mutex_destroy(&c->rd_mu);
    free(c);
}

// ---- runtime budget control -------------------------------------------------
// Live-resize the decoded-block cache to `new_bytes`. The cache is just a cache,
// so resizing DISCARDS resident blocks (re-decode on demand) rather than
// migrating them. Locks every shard for the swap. Returns the byte budget
// actually installed (rounded to whole slots over NSHARD), or 0 on failure.
size_t mc_cache_resize(mc_cache *c, size_t new_bytes){
    if(!c) return 0;
    size_t nslot_total = new_bytes/BLK_BYTES; if(nslot_total<NSHARD) nslot_total=NSHARD;
    uint32_t per = (uint32_t)(nslot_total/NSHARD); if(per<1)per=1;
    size_t new_arena = (size_t)per*NSHARD*BLK_BYTES;
#if MC_CACHE_MMAP
    void *na = mmap(NULL,new_arena,PROT_READ|PROT_WRITE,
                    MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
    if(na==MAP_FAILED) return 0;
#else
    void *na = malloc(new_arena);
    if(!na) return 0;
#endif
    for(int s=0;s<NSHARD;++s) pthread_mutex_lock(&c->sh[s].mu);
    for(int s=0;s<NSHARD;++s){
        shard_free_tables(&c->sh[s]);
        shard_init_tables(&c->sh[s], (mc_u8*)na + (size_t)s*per*BLK_BYTES, per);
    }
    void *old = c->arena_base; size_t old_bytes = c->arena_bytes;
    c->arena_base = na; c->arena_bytes = new_arena;
    atomic_fetch_add(&c->epoch,1);   // invalidate outstanding pins
    for(int s=0;s<NSHARD;++s) pthread_mutex_unlock(&c->sh[s].mu);
#if MC_CACHE_MMAP
    munmap(old,old_bytes);
#else
    free(old);
#endif
    return new_arena;
}

size_t mc_cache_capacity_bytes(const mc_cache *c){ return c ? c->arena_bytes : 0; }

size_t mc_cache_used_bytes(mc_cache *c){
    if(!c) return 0;
    size_t used=0;
    for(int s=0;s<NSHARD;++s){
        pthread_mutex_lock(&c->sh[s].mu);
        used += (size_t)c->sh[s].used;
        pthread_mutex_unlock(&c->sh[s].mu);
    }
    return used*BLK_BYTES;
}

double mc_cache_usage_fraction(mc_cache *c){
    if(!c||c->arena_bytes==0) return 0.0;
    return (double)mc_cache_used_bytes(c)/(double)c->arena_bytes;
}

// ---- shard map ops (shard lock held) ----
static inline uint32_t map_find(shard_t *sh, uint64_t key){   // -> map index or UINT32_MAX
    uint32_t m=sh->map_cap-1, i=(uint32_t)khash(key)&m;
    for(;;){
        if(sh->map_key[i]==key) return i;
        if(sh->map_key[i]==EMPTY_KEY) return UINT32_MAX;
        i=(i+1)&m;
    }
}
static void map_insert(shard_t *sh, uint64_t key, uint32_t slot){
    uint32_t m=sh->map_cap-1, i=(uint32_t)khash(key)&m;
    while(sh->map_key[i]!=EMPTY_KEY) i=(i+1)&m;
    sh->map_key[i]=key; sh->map_slot[i]=slot;
}
static void map_delete(shard_t *sh, uint64_t key){            // backward-shift deletion
    uint32_t m=sh->map_cap-1, i=map_find(sh,key);
    if(i==UINT32_MAX) return;
    uint32_t j=i;
    for(;;){
        j=(j+1)&m;
        uint64_t kj=sh->map_key[j];
        if(kj==EMPTY_KEY) break;
        uint32_t home=(uint32_t)khash(kj)&m;
        // can kj move into the hole at i? (it must not cross its home slot)
        uint32_t dist_ij=(i-home)&m, dist_jj=(j-home)&m;
        if(dist_ij<=dist_jj){ sh->map_key[i]=kj; sh->map_slot[i]=sh->map_slot[j]; i=j; }
    }
    sh->map_key[i]=EMPTY_KEY;
}
// CLOCK sweep: find a victim slot (clearing ref bits as we pass).
static uint32_t reclaim_slot(shard_t *sh){
    if(sh->used<sh->nslot){                       // free slot exists: linear scan from hand
        for(uint32_t k=0;k<sh->nslot;++k){
            uint32_t i=(sh->hand+k)%sh->nslot;
            if(sh->slot_key[i]==EMPTY_KEY){ sh->hand=(i+1)%sh->nslot; sh->used++; return i; }
        }
    }
    for(;;){
        uint32_t i=sh->hand; sh->hand=(sh->hand+1)%sh->nslot;
        if(sh->slot_key[i]==EMPTY_KEY) return i;            // invalidated: reuse
        if(sh->slot_ref[i]){ sh->slot_ref[i]=0; continue; }
        map_delete(sh,sh->slot_key[i]);
        sh->evictions++;
        return i;
    }
}

// ---- S3-FIFO (SOSP'23) ----
static inline uint32_t s3_fp(uint64_t key){ uint32_t f=(uint32_t)(khash(key)>>17); return f?f:1; }
static inline uint32_t ring_len(uint32_t h,uint32_t t,uint32_t cap){ return (t+cap-h)%cap; }
static void ghost_put(shard_t *sh, uint32_t fp){
    uint32_t old=sh->gfp[sh->g_head];
    if(old){  // remove the overwritten fingerprint from the set
        uint32_t m=sh->gset_cap-1, i=(old*0x9E3779B1u)&m;
        while(sh->gset[i]!=-1){ if(sh->gfp[sh->gset[i]]==old && (uint32_t)sh->gset[i]==sh->g_head){ sh->gset[i]=-1; break; } i=(i+1)&m; }
    }
    sh->gfp[sh->g_head]=fp;
    uint32_t m=sh->gset_cap-1, i=(fp*0x9E3779B1u)&m;
    while(sh->gset[i]!=-1) i=(i+1)&m;
    sh->gset[i]=(int32_t)sh->g_head;
    sh->g_head=(sh->g_head+1)%sh->g_cap;
}
static int ghost_has(shard_t *sh, uint32_t fp){
    uint32_t m=sh->gset_cap-1, i=(fp*0x9E3779B1u)&m;
    for(int probes=0;probes<64;++probes){
        int32_t v=sh->gset[i];
        if(v==-1) return 0;
        if(sh->gfp[v]==fp) return 1;
        i=(i+1)&m;
    }
    return 0;
}
// reclaim one slot under S3-FIFO rules; the freed slot's key is unmapped.
static uint32_t s3_reclaim(mc_cache *c_pin, shard_t *sh){
    uint32_t spins=0, budget=2*sh->nslot+8;
    for(;;){
        int force = ++spins>budget;        // all pinned/hot: evict regardless
        uint32_t small_len=ring_len(sh->fs_head,sh->fs_tail,sh->fs_cap);
        if(small_len >= sh->nslot/10+1){
            uint32_t s=sh->fs[sh->fs_head]; sh->fs_head=(sh->fs_head+1)%sh->fs_cap;
            if(sh->slot_key[s]==EMPTY_KEY) return s;        // invalidated: reuse
            if(!force && (sh->slot_ref[s]>0 || slot_pinned(c_pin,sh,s))){   // promote to main
                sh->slot_ref[s]=0; sh->slot_inmain[s]=1;
                sh->fm[sh->fm_tail]=s; sh->fm_tail=(sh->fm_tail+1)%sh->fm_cap;
                continue;
            }
            ghost_put(sh,s3_fp(sh->slot_key[s]));
            map_delete(sh,sh->slot_key[s]); sh->evictions++;
            return s;
        }
        if(ring_len(sh->fm_head,sh->fm_tail,sh->fm_cap)==0){   // degenerate: force small
            uint32_t s=sh->fs[sh->fs_head]; sh->fs_head=(sh->fs_head+1)%sh->fs_cap;
            if(sh->slot_key[s]==EMPTY_KEY) return s;        // invalidated: reuse
            ghost_put(sh,s3_fp(sh->slot_key[s]));
            map_delete(sh,sh->slot_key[s]); sh->evictions++;
            return s;
        }
        uint32_t s=sh->fm[sh->fm_head]; sh->fm_head=(sh->fm_head+1)%sh->fm_cap;
        if(sh->slot_key[s]==EMPTY_KEY) return s;            // invalidated: reuse
        if(!force && (sh->slot_ref[s]>0 || slot_pinned(c_pin,sh,s))){   // pinned: keep
            if(sh->slot_ref[s]>0) sh->slot_ref[s]--;
            sh->fm[sh->fm_tail]=s; sh->fm_tail=(sh->fm_tail+1)%sh->fm_cap;
            continue;
        }
        map_delete(sh,sh->slot_key[s]); sh->evictions++;
        return s;
    }
}
// allocate a slot for `key` under the active policy and enqueue it.
static uint32_t cache_alloc_slot(mc_cache *c, shard_t *sh, uint64_t key){
    uint32_t slot;
    if(c->policy==MC_CACHE_CLOCK){
        slot=reclaim_slot(sh);
        return slot;
    }
    if(sh->used<sh->nslot){
        for(uint32_t k=0;k<sh->nslot;++k){
            uint32_t i=(sh->hand+k)%sh->nslot;
            if(sh->slot_key[i]==EMPTY_KEY){ sh->hand=(i+1)%sh->nslot; sh->used++; slot=i; goto have; }
        }
    }
    slot=s3_reclaim(c,sh);
have:;
    int to_main = ghost_has(sh,s3_fp(key));
    sh->slot_inmain[slot]=(uint8_t)to_main;
    sh->slot_ref[slot]=0;
    if(to_main){ sh->fm[sh->fm_tail]=slot; sh->fm_tail=(sh->fm_tail+1)%sh->fm_cap; }
    else       { sh->fs[sh->fs_tail]=slot; sh->fs_tail=(sh->fs_tail+1)%sh->fs_cap; }
    return slot;
}
static inline void cache_touch(mc_cache *c, shard_t *sh, uint32_t slot){
    if(c->policy==MC_CACHE_CLOCK) sh->slot_ref[slot]=1;
    else if(sh->slot_ref[slot]<FREQ_MAX) sh->slot_ref[slot]++;
    sh->slot_epoch[slot]=atomic_load_explicit(&c->epoch,memory_order_relaxed);
}
static inline int slot_pinned(mc_cache *c, shard_t *sh, uint32_t slot){
    if(!atomic_load_explicit(&c->phase_mode,memory_order_relaxed)) return 0;
    return sh->slot_epoch[slot]==atomic_load_explicit(&c->epoch,memory_order_relaxed);
}


static inline shard_t *shard_of(mc_cache *c, uint64_t key){
    return &c->sh[(khash(key)>>56)&(NSHARD-1)];
}

const mc_u8 *mc_cache_get(mc_cache *c, int lod, int bz, int by, int bx){
    uint64_t key=bkey(lod,bz,by,bx);
    shard_t *sh=shard_of(c,key);
    if(cache_frozen(c)){                      // immutable: bare probe, no locks
        uint32_t mi=map_find(sh,key);
        if(mi!=UINT32_MAX) return sh->arena+(size_t)sh->map_slot[mi]*BLK_BYTES;
        miss_record(c,key);
        return NULL;                          // caller: fall back to best_lod
    }
    pthread_mutex_lock(&sh->mu);
    uint32_t mi=map_find(sh,key);
    if(mi!=UINT32_MAX){
        uint32_t slot=sh->map_slot[mi];
        cache_touch(c,sh,slot); sh->hits++;
        const mc_u8 *p=sh->arena+(size_t)slot*BLK_BYTES;
        pthread_mutex_unlock(&sh->mu);
        return p;
    }
    sh->misses++;
    pthread_mutex_unlock(&sh->mu);

    static _Thread_local mc_u8 tmp[BLK_BYTES];
    c->src(c->src_ud,lod,bz,by,bx,tmp);            // decode outside the lock

    pthread_mutex_lock(&sh->mu);
    mi=map_find(sh,key);                           // racing thread may have inserted
    uint32_t slot;
    if(mi!=UINT32_MAX) slot=sh->map_slot[mi];
    else {
        slot=cache_alloc_slot(c,sh,key);
        sh->slot_key[slot]=key;
        map_insert(sh,key,slot);
        memcpy(sh->arena+(size_t)slot*BLK_BYTES,tmp,BLK_BYTES);
    }
    cache_touch(c,sh,slot);
    const mc_u8 *p=sh->arena+(size_t)slot*BLK_BYTES;
    pthread_mutex_unlock(&sh->mu);
    return p;
}

void mc_cache_get_copy(mc_cache *c, int lod, int bz, int by, int bx, mc_u8 *dst){
    uint64_t key=bkey(lod,bz,by,bx);
    shard_t *sh=shard_of(c,key);
    if(cache_frozen(c)){
        uint32_t mi=map_find(sh,key);
        if(mi!=UINT32_MAX){ memcpy(dst,sh->arena+(size_t)sh->map_slot[mi]*BLK_BYTES,BLK_BYTES); return; }
        miss_record(c,key);
        c->src(c->src_ud,lod,bz,by,bx,dst);   // read-through, no insert
        return;
    }
    pthread_mutex_lock(&sh->mu);
    uint32_t mi=map_find(sh,key);
    if(mi!=UINT32_MAX){
        uint32_t slot=sh->map_slot[mi];
        cache_touch(c,sh,slot); sh->hits++;
        memcpy(dst,sh->arena+(size_t)slot*BLK_BYTES,BLK_BYTES);
        pthread_mutex_unlock(&sh->mu);
        return;
    }
    sh->misses++;
    pthread_mutex_unlock(&sh->mu);
    c->src(c->src_ud,lod,bz,by,bx,dst);
    pthread_mutex_lock(&sh->mu);
    if(map_find(sh,key)==UINT32_MAX){
        uint32_t slot=cache_alloc_slot(c,sh,key);
        sh->slot_key[slot]=key;
        map_insert(sh,key,slot);
        memcpy(sh->arena+(size_t)slot*BLK_BYTES,dst,BLK_BYTES);
        cache_touch(c,sh,slot);
    }
    pthread_mutex_unlock(&sh->mu);
}

int mc_cache_contains(mc_cache *c, int lod, int bz, int by, int bx){
    uint64_t key=bkey(lod,bz,by,bx);
    shard_t *sh=shard_of(c,key);
    if(cache_frozen(c)) return map_find(sh,key)!=UINT32_MAX;
    pthread_mutex_lock(&sh->mu);
    int r = map_find(sh,key)!=UINT32_MAX;
    pthread_mutex_unlock(&sh->mu);
    return r;
}

// lookup-or-decode-insert; returns 1 if a decode happened.
static int cache_fill_one(mc_cache *c, int lod, int bz, int by, int bx){
    uint64_t key=bkey(lod,bz,by,bx);
    shard_t *sh=shard_of(c,key);
    pthread_mutex_lock(&sh->mu);
    uint32_t mi=map_find(sh,key);
    if(mi!=UINT32_MAX){
        cache_touch(c,sh,sh->map_slot[mi]); sh->hits++;
        pthread_mutex_unlock(&sh->mu);
        return 0;
    }
    sh->misses++;
    pthread_mutex_unlock(&sh->mu);
    static _Thread_local mc_u8 tmp[BLK_BYTES];
    c->src(c->src_ud,lod,bz,by,bx,tmp);
    pthread_mutex_lock(&sh->mu);
    if(map_find(sh,key)==UINT32_MAX){
        uint32_t slot=cache_alloc_slot(c,sh,key);
        sh->slot_key[slot]=key;
        map_insert(sh,key,slot);
        memcpy(sh->arena+(size_t)slot*BLK_BYTES,tmp,BLK_BYTES);
        cache_touch(c,sh,slot);
    }
    pthread_mutex_unlock(&sh->mu);
    return 1;
}

typedef struct {
    mc_cache *c;
    const mc_block_id *ids;     // sorted copy, grouped by chunk
    const uint32_t *group_off;  // group g = ids[group_off[g] .. group_off[g+1])
    uint32_t ngroups;
    _Atomic uint32_t next;      // work-stealing group cursor
    _Atomic size_t decoded;
} upd_ctx;
static void *upd_worker(void *p){
    upd_ctx *u=p;
    for(;;){
        uint32_t g=atomic_fetch_add_explicit(&u->next,1,memory_order_relaxed);
        if(g>=u->ngroups) break;
        size_t dec=0;
        for(uint32_t i=u->group_off[g];i<u->group_off[g+1];++i){
            const mc_block_id *b=&u->ids[i];
            dec+=(size_t)cache_fill_one(u->c,b->lod,b->bz,b->by,b->bx);
        }
        atomic_fetch_add_explicit(&u->decoded,dec,memory_order_relaxed);
    }
    return NULL;
}
static uint64_t upd_sortkey(const mc_block_id *b){    // chunk-major; low 12 bits = in-chunk
    return ((uint64_t)(b->lod&7)<<60)
         | ((uint64_t)(b->bz>>4)<<44) | ((uint64_t)(b->by>>4)<<28) | ((uint64_t)(b->bx>>4)<<12)
         | ((uint64_t)(b->bz&15)<<8)  | ((uint64_t)(b->by&15)<<4)  | (uint64_t)(b->bx&15);
}
static int upd_cmp(const void *a,const void *b){
    uint64_t ka=upd_sortkey(a), kb=upd_sortkey(b);
    return ka<kb?-1:ka>kb?1:0;
}
size_t mc_cache_update(mc_cache *c, const mc_block_id *ids, size_t n, int nthreads){
    if(!c||!ids||!n||cache_frozen(c)) return 0;
    mc_block_id *s=malloc(n*sizeof *s);
    memcpy(s,ids,n*sizeof *s);
    qsort(s,n,sizeof *s,upd_cmp);
    uint32_t *off=malloc((n+1)*sizeof *off);
    uint32_t ng=0; off[0]=0;
    for(size_t i=1;i<n;++i){
        if((upd_sortkey(&s[i])>>12)!=(upd_sortkey(&s[i-1])>>12)) off[++ng]=(uint32_t)i;
    }
    off[++ng]=(uint32_t)n;
    upd_ctx u={.c=c,.ids=s,.group_off=off,.ngroups=ng};
    atomic_store(&u.next,0); atomic_store(&u.decoded,0);
    int nt=nthreads;
    if(nt<=0){
        long nc=sysconf(_SC_NPROCESSORS_ONLN);
        nt=(int)(nc>0?nc:4);
    }
    if(nt>16)nt=16; if((uint32_t)nt>ng)nt=(int)ng;
    if(nt<=1){ upd_worker(&u); }
    else {
        pthread_t th[16];
        for(int t=0;t<nt;++t) pthread_create(&th[t],NULL,upd_worker,&u);
        for(int t=0;t<nt;++t) pthread_join(th[t],NULL);
    }
    size_t dec=atomic_load(&u.decoded);
    free(s); free(off);
    return dec;
}

void mc_cache_freeze(mc_cache *c){
    if(!c) return;
    atomic_store_explicit(&c->phase_mode,1,memory_order_relaxed);
    atomic_store_explicit(&c->frozen,1,memory_order_release);
}
void mc_cache_thaw(mc_cache *c){
    if(!c) return;
    atomic_store_explicit(&c->frozen,0,memory_order_release);
    atomic_fetch_add_explicit(&c->epoch,1,memory_order_relaxed);
}
size_t mc_cache_misses_drain(mc_cache *c, mc_block_id *out, size_t cap){
    if(!c||!out) return 0;
    uint32_t w=atomic_load_explicit(&c->miss_w,memory_order_acquire);
    uint32_t n=w>MISSQ_CAP?MISSQ_CAP:w;
    size_t m=0;
    for(uint32_t i=0;i<n&&m<cap;++i){
        uint64_t k=atomic_load_explicit(&c->missq[i&(MISSQ_CAP-1)],memory_order_relaxed);
        if(!k) continue;
        out[m].lod=(int)((k>>60)&7);
        out[m].bz=(int)((k>>40)&0xFFFFF);
        out[m].by=(int)((k>>20)&0xFFFFF);
        out[m].bx=(int)(k&0xFFFFF);
        m++;
    }
    atomic_store_explicit(&c->miss_w,0,memory_order_release);
    return m;
}

int mc_cache_best_lod(mc_cache *c, int finest_lod, int bz, int by, int bx){
    int froz=cache_frozen(c);
    for(int l=finest_lod;l<8;++l){
        uint64_t key=bkey(l,bz,by,bx);
        shard_t *sh=shard_of(c,key);
        int hit;
        if(froz) hit = map_find(sh,key)!=UINT32_MAX;
        else { pthread_mutex_lock(&sh->mu); hit = map_find(sh,key)!=UINT32_MAX; pthread_mutex_unlock(&sh->mu); }
        if(hit) return l;
        bz>>=1; by>>=1; bx>>=1;
    }
    return -1;
}

// ---- async update tickets ---------------------------------------------------
struct mc_cache_ticket {
    mc_cache *c;
    mc_block_id *ids;            // owned sorted copy
    uint32_t *group_off;
    uint32_t ngroups;
    _Atomic uint32_t next;
    _Atomic uint32_t groups_done;
    _Atomic int cancel;
    pthread_t th[16]; int nth;
    int joined;
};
static void *aupd_worker(void *p){
    mc_cache_ticket *t=p;
    for(;;){
        if(atomic_load_explicit(&t->cancel,memory_order_relaxed)){
            // mark remaining groups done so done() converges after cancel
            uint32_t g=atomic_fetch_add_explicit(&t->next,1,memory_order_relaxed);
            if(g>=t->ngroups) break;
            atomic_fetch_add_explicit(&t->groups_done,1,memory_order_relaxed);
            continue;
        }
        uint32_t g=atomic_fetch_add_explicit(&t->next,1,memory_order_relaxed);
        if(g>=t->ngroups) break;
        for(uint32_t i=t->group_off[g];i<t->group_off[g+1];++i){
            const mc_block_id *b=&t->ids[i];
            cache_fill_one(t->c,b->lod,b->bz,b->by,b->bx);
        }
        atomic_fetch_add_explicit(&t->groups_done,1,memory_order_relaxed);
    }
    return NULL;
}
mc_cache_ticket *mc_cache_update_async(mc_cache *c, const mc_block_id *ids, size_t n, int nthreads){
    if(!c||!ids||!n||cache_frozen(c)) return NULL;
    mc_cache_ticket *t=calloc(1,sizeof *t);
    t->c=c;
    t->ids=malloc(n*sizeof *t->ids);
    memcpy(t->ids,ids,n*sizeof *t->ids);
    qsort(t->ids,n,sizeof *t->ids,upd_cmp);
    t->group_off=malloc((n+1)*sizeof *t->group_off);
    uint32_t ng=0; t->group_off[0]=0;
    for(size_t i=1;i<n;++i)
        if((upd_sortkey(&t->ids[i])>>12)!=(upd_sortkey(&t->ids[i-1])>>12)) t->group_off[++ng]=(uint32_t)i;
    t->group_off[++ng]=(uint32_t)n;
    t->ngroups=ng;
    atomic_store(&t->next,0); atomic_store(&t->groups_done,0); atomic_store(&t->cancel,0);
    int nt=nthreads;
    if(nt<=0){ long nc=sysconf(_SC_NPROCESSORS_ONLN); nt=(int)(nc>0?nc:4); }
    if(nt>16)nt=16; if((uint32_t)nt>ng)nt=(int)ng;
    t->nth=nt;
    for(int i=0;i<nt;++i) pthread_create(&t->th[i],NULL,aupd_worker,t);
    return t;
}
int mc_cache_ticket_done(mc_cache_ticket *t){
    if(!t) return 1;
    return atomic_load_explicit(&t->groups_done,memory_order_acquire)>=t->ngroups;
}
void mc_cache_ticket_cancel(mc_cache_ticket *t){
    if(t) atomic_store_explicit(&t->cancel,1,memory_order_release);
}
static void ticket_join(mc_cache_ticket *t){
    if(t->joined) return;
    for(int i=0;i<t->nth;++i) pthread_join(t->th[i],NULL);
    t->joined=1;
}
void mc_cache_ticket_wait(mc_cache_ticket *t){ if(t) ticket_join(t); }
void mc_cache_ticket_free(mc_cache_ticket *t){
    if(!t) return;
    ticket_join(t);
    free(t->ids); free(t->group_off); free(t);
}

// resolve: ensure resident (parallel via mc_cache_update), then fill the
// pointer table under shard locks; cache_touch stamps the current epoch so
// these slots are pinned against eviction until the next thaw().
size_t mc_cache_resolve(mc_cache *c, const mc_block_id *ids, size_t n,
                        const mc_u8 **ptrs, int nthreads){
    if(!c||!ids||!n||!ptrs||cache_frozen(c)) return 0;
    size_t dec=mc_cache_update(c,ids,n,nthreads);
    for(size_t i=0;i<n;++i){
        uint64_t key=bkey(ids[i].lod,ids[i].bz,ids[i].by,ids[i].bx);
        shard_t *sh=shard_of(c,key);
        pthread_mutex_lock(&sh->mu);
        uint32_t mi=map_find(sh,key);
        if(mi!=UINT32_MAX){
            uint32_t slot=sh->map_slot[mi];
            cache_touch(c,sh,slot);
            ptrs[i]=sh->arena+(size_t)slot*BLK_BYTES;
        } else ptrs[i]=NULL;   // evicted by same-batch pressure (set > capacity)
        pthread_mutex_unlock(&sh->mu);
    }
    return dec;
}

void mc_cache_prefetch_chunk(mc_cache *c, int lod, int cz, int cy, int cx){
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
        int gz=cz*16+bz, gy=cy*16+by, gx=cx*16+bx;
        uint64_t key=bkey(lod,gz,gy,gx);
        shard_t *sh=shard_of(c,key);
        pthread_mutex_lock(&sh->mu);
        int have = map_find(sh,key)!=UINT32_MAX;
        pthread_mutex_unlock(&sh->mu);
        if(!have) (void)mc_cache_get(c,lod,gz,gy,gx);
    }
}

// remove one key if present (shard lock held by caller paths below)
static void shard_remove_key(shard_t *sh, uint64_t key){
    uint32_t mi=map_find(sh,key);
    if(mi==UINT32_MAX) return;
    uint32_t slot=sh->map_slot[mi];
    map_delete(sh,key);
    sh->slot_key[slot]=EMPTY_KEY;       // slot stays in its FIFO ring; the
    sh->slot_ref[slot]=0;               // reclaim path skips empty keys
    sh->evictions++;
}
void mc_cache_invalidate_chunk(mc_cache *c, int lod, int cz, int cy, int cx){
    if(cache_frozen(c)) return;
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
        uint64_t key=bkey(lod,cz*16+bz,cy*16+by,cx*16+bx);
        shard_t *sh=shard_of(c,key);
        pthread_mutex_lock(&sh->mu);
        shard_remove_key(sh,key);
        pthread_mutex_unlock(&sh->mu);
    }
}

void mc_cache_clear(mc_cache *c){
    for(int s=0;s<NSHARD;++s){
        shard_t *sh=&c->sh[s];
        pthread_mutex_lock(&sh->mu);
        memset(sh->map_key,0,(size_t)sh->map_cap*8);
        memset(sh->slot_key,0,(size_t)sh->nslot*8);
        memset(sh->slot_ref,0,sh->nslot);
        sh->used=0; sh->hand=0;
        sh->fs_head=sh->fs_tail=sh->fm_head=sh->fm_tail=0;
        sh->g_head=0; memset(sh->gfp,0,4u*sh->g_cap);
        memset(sh->gset,0xFF,4u*sh->gset_cap);
        memset(sh->slot_inmain,0,sh->nslot);
        pthread_mutex_unlock(&sh->mu);
    }
}

void mc_cache_get_stats(mc_cache *c, mc_cache_stats *out){
    memset(out,0,sizeof *out);
    for(int s=0;s<NSHARD;++s){
        shard_t *sh=&c->sh[s];
        out->hits+=sh->hits; out->misses+=sh->misses; out->evictions+=sh->evictions;
        out->slots+=sh->nslot; out->used+=sh->used;
    }
}

// ---- bindings ----
static void src_archive(void *ud, int lod, int bz,int by,int bx, mc_u8 *dst){
    struct mc_archive *a=ud;
    uint64_t co=mc_archive_chunk_offset(a,lod,bz>>4,by>>4,bx>>4);
    mc_archive_decode_block(a,co,bz&15,by&15,bx&15,dst);
}
mc_cache *mc_cache_new_archive(size_t bytes, struct mc_archive *a){
    mc_cache *c=mc_cache_new(bytes,src_archive,a);
    if(c) c->ar=a;
    return c;
}
typedef struct { mc_cache *c; } rdwrap_t;
static void src_reader(void *ud, int lod, int bz,int by,int bx, mc_u8 *dst){
    mc_cache *c=ud;
    pthread_mutex_lock(&c->rd_mu);
    uint64_t co=mc_chunk_offset(c->rd,lod,bz>>4,by>>4,bx>>4);
    mc_decode_block(c->rd,co,bz&15,by&15,bx&15,dst);
    pthread_mutex_unlock(&c->rd_mu);
}
mc_cache *mc_cache_new_reader(size_t bytes, struct mc_reader *r){
    mc_cache *c=mc_cache_new(bytes,NULL,NULL);
    if(!c) return NULL;
    c->rd=r; c->src=src_reader; c->src_ud=c;
    return c;
}

// ============================================================================
// mc_sample — point sampling over blocked u8 volumes (see header)
// ============================================================================
#include <unistd.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__SSE4_1__)
#include <immintrin.h>
#endif

#define MC_S_MEMO 256   // covers an oblique 1024-px row's block working set

typedef struct {
    int bz, by, bx;             // -1 = empty
    const uint8_t *ptr;         // NULL = known-absent (sampled as 0)
    uint8_t buf[4096];
} mc_s_memo;

struct mc_sampler {
    mc_sample_src src;
    int nbz, nby, nbx;
    int lbz, lby, lbx;          // last block touched (ray-coherence cache)
    const uint8_t *lptr;
    mc_s_memo m[MC_S_MEMO];
};

static inline const uint8_t *mc_s_block(mc_sampler *s, int bz, int by, int bx) {
    if (bz == s->lbz && by == s->lby && bx == s->lbx) return s->lptr;
    unsigned h = ((unsigned)bz * 73856093u) ^ ((unsigned)by * 19349663u) ^
                 ((unsigned)bx * 83492791u);
    mc_s_memo *e = &s->m[h & (MC_S_MEMO - 1)];
    if (!(e->bz == bz && e->by == by && e->bx == bx)) {
        e->bz = bz; e->by = by; e->bx = bx;
        e->ptr = s->src.block(&s->src, bz, by, bx, e->buf);
    }
    s->lbz = bz; s->lby = by; s->lbx = bx; s->lptr = e->ptr;
    return e->ptr;
}

static inline float mc_s_voxel(mc_sampler *s, int z, int y, int x) {
    if ((unsigned)z >= (unsigned)s->src.nz ||
        (unsigned)y >= (unsigned)s->src.ny ||
        (unsigned)x >= (unsigned)s->src.nx) return 0.0f;
    if (s->src.dense)
        return (float)s->src.dense[(size_t)z * s->src.dsy +
                                   (size_t)y * s->src.dsx + (size_t)x];
    const uint8_t *b = mc_s_block(s, z >> 4, y >> 4, x >> 4);
    return b ? (float)b[((z & 15) << 8) | ((y & 15) << 4) | (x & 15)] : 0.0f;
}

static inline float mc_s_nearest(mc_sampler *s, float z, float y, float x) {
    return mc_s_voxel(s, (int)floorf(z + 0.5f), (int)floorf(y + 0.5f),
                      (int)floorf(x + 0.5f));
}

static inline float mc_s_trilinear(mc_sampler *s, float z, float y, float x) {
    float zf = floorf(z), yf = floorf(y), xf = floorf(x);
    int z0 = (int)zf, y0 = (int)yf, x0 = (int)xf;
    float dz = z - zf, dy = y - yf, dx = x - xf;
    // dense fast path: direct strided gather, only a bounds check
    if (s->src.dense &&
        (unsigned)z0 < (unsigned)(s->src.nz - 1) &&
        (unsigned)y0 < (unsigned)(s->src.ny - 1) &&
        (unsigned)x0 < (unsigned)(s->src.nx - 1)) {
        const size_t sy = s->src.dsy, sx = s->src.dsx;
        const uint8_t *p = s->src.dense + (size_t)z0 * sy + (size_t)y0 * sx + x0;
        float c00 = (float)p[0]      + ((float)p[1]        - (float)p[0])      * dx;
        float c01 = (float)p[sx]     + ((float)p[sx + 1]   - (float)p[sx])     * dx;
        float c10 = (float)p[sy]     + ((float)p[sy + 1]   - (float)p[sy])     * dx;
        float c11 = (float)p[sy + sx] + ((float)p[sy + sx + 1] - (float)p[sy + sx]) * dx;
        float c0 = c00 + (c01 - c00) * dy;
        float c1 = c10 + (c11 - c10) * dy;
        return c0 + (c1 - c0) * dz;
    }
    // blocked fast path: all 8 corners inside one block and in bounds (~82%
    // of uniformly distributed samples; far more for coherent rays)
    if (!s->src.dense &&
        (unsigned)z0 < (unsigned)(s->src.nz - 1) &&
        (unsigned)y0 < (unsigned)(s->src.ny - 1) &&
        (unsigned)x0 < (unsigned)(s->src.nx - 1) &&
        (z0 & 15) != 15 && (y0 & 15) != 15 && (x0 & 15) != 15) {
        const uint8_t *b = mc_s_block(s, z0 >> 4, y0 >> 4, x0 >> 4);
        if (!b) return 0.0f;
        const uint8_t *p = b + (((z0 & 15) << 8) | ((y0 & 15) << 4) | (x0 & 15));
        float c00 = (float)p[0]   + ((float)p[1]   - (float)p[0])   * dx;
        float c01 = (float)p[16]  + ((float)p[17]  - (float)p[16])  * dx;
        float c10 = (float)p[256] + ((float)p[257] - (float)p[256]) * dx;
        float c11 = (float)p[272] + ((float)p[273] - (float)p[272]) * dx;
        float c0 = c00 + (c01 - c00) * dy;
        float c1 = c10 + (c11 - c10) * dy;
        return c0 + (c1 - c0) * dz;
    }
    // slow path: block/bounds handled per corner (edges mix with 0)
    float c000 = mc_s_voxel(s, z0, y0, x0);
    float c001 = mc_s_voxel(s, z0, y0, x0 + 1);
    float c010 = mc_s_voxel(s, z0, y0 + 1, x0);
    float c011 = mc_s_voxel(s, z0, y0 + 1, x0 + 1);
    float c100 = mc_s_voxel(s, z0 + 1, y0, x0);
    float c101 = mc_s_voxel(s, z0 + 1, y0, x0 + 1);
    float c110 = mc_s_voxel(s, z0 + 1, y0 + 1, x0);
    float c111 = mc_s_voxel(s, z0 + 1, y0 + 1, x0 + 1);
    float c00 = c000 + (c001 - c000) * dx;
    float c01 = c010 + (c011 - c010) * dx;
    float c10 = c100 + (c101 - c100) * dx;
    float c11 = c110 + (c111 - c110) * dx;
    float c0 = c00 + (c01 - c00) * dy;
    float c1 = c10 + (c11 - c10) * dy;
    return c0 + (c1 - c0) * dz;
}

static inline float mc_s_sample(mc_sampler *s, float z, float y, float x,
                                mc_filter f) {
    if (!(z == z) || !(y == y) || !(x == x)) return 0.0f;   // NaN
    return f == MC_FILTER_NEAREST ? mc_s_nearest(s, z, y, x)
                                  : mc_s_trilinear(s, z, y, x);
}

// ---------------------------------------------------------------------------
// 4-wide trilinear (ray-step batching for the compositors)
// ---------------------------------------------------------------------------
// Sample 4 positions at once. Lanes that qualify for a fast path are
// gathered and lerped with NEON; anything else (edges, absent blocks,
// non-aarch64) falls back to the scalar sampler per lane. Uses separate
// mul+add (no fma), so every lane is bit-identical to mc_s_trilinear.
#if defined(__aarch64__)
static inline float32x4_t mc_s_lerp8x4(const uint8_t *p0, const uint8_t *p1,
                                       const uint8_t *p2, const uint8_t *p3,
                                       size_t sy, size_t sx,
                                       float32x4_t dz, float32x4_t dy,
                                       float32x4_t dx) {
    uint16x4_t g00 = vdup_n_u16(0), g01 = vdup_n_u16(0);
    uint16x4_t g10 = vdup_n_u16(0), g11 = vdup_n_u16(0);
    g00 = vld1_lane_u16((const uint16_t *)(const void *)p0, g00, 0);
    g00 = vld1_lane_u16((const uint16_t *)(const void *)p1, g00, 1);
    g00 = vld1_lane_u16((const uint16_t *)(const void *)p2, g00, 2);
    g00 = vld1_lane_u16((const uint16_t *)(const void *)p3, g00, 3);
    g01 = vld1_lane_u16((const uint16_t *)(const void *)(p0 + sx), g01, 0);
    g01 = vld1_lane_u16((const uint16_t *)(const void *)(p1 + sx), g01, 1);
    g01 = vld1_lane_u16((const uint16_t *)(const void *)(p2 + sx), g01, 2);
    g01 = vld1_lane_u16((const uint16_t *)(const void *)(p3 + sx), g01, 3);
    g10 = vld1_lane_u16((const uint16_t *)(const void *)(p0 + sy), g10, 0);
    g10 = vld1_lane_u16((const uint16_t *)(const void *)(p1 + sy), g10, 1);
    g10 = vld1_lane_u16((const uint16_t *)(const void *)(p2 + sy), g10, 2);
    g10 = vld1_lane_u16((const uint16_t *)(const void *)(p3 + sy), g10, 3);
    g11 = vld1_lane_u16((const uint16_t *)(const void *)(p0 + sy + sx), g11, 0);
    g11 = vld1_lane_u16((const uint16_t *)(const void *)(p1 + sy + sx), g11, 1);
    g11 = vld1_lane_u16((const uint16_t *)(const void *)(p2 + sy + sx), g11, 2);
    g11 = vld1_lane_u16((const uint16_t *)(const void *)(p3 + sy + sx), g11, 3);
    uint16x8_t w00 = vmovl_u8(vreinterpret_u8_u16(g00));
    uint16x8_t w01 = vmovl_u8(vreinterpret_u8_u16(g01));
    uint16x8_t w10 = vmovl_u8(vreinterpret_u8_u16(g10));
    uint16x8_t w11 = vmovl_u8(vreinterpret_u8_u16(g11));
#define MC_S_F32E(w) vcvtq_f32_u32(vmovl_u16(vuzp1_u16(vget_low_u16(w), vget_high_u16(w))))
#define MC_S_F32O(w) vcvtq_f32_u32(vmovl_u16(vuzp2_u16(vget_low_u16(w), vget_high_u16(w))))
    float32x4_t f000 = MC_S_F32E(w00), f001 = MC_S_F32O(w00);
    float32x4_t f010 = MC_S_F32E(w01), f011 = MC_S_F32O(w01);
    float32x4_t f100 = MC_S_F32E(w10), f101 = MC_S_F32O(w10);
    float32x4_t f110 = MC_S_F32E(w11), f111 = MC_S_F32O(w11);
#undef MC_S_F32E
#undef MC_S_F32O
    float32x4_t c00 = vaddq_f32(f000, vmulq_f32(vsubq_f32(f001, f000), dx));
    float32x4_t c01 = vaddq_f32(f010, vmulq_f32(vsubq_f32(f011, f010), dx));
    float32x4_t c10 = vaddq_f32(f100, vmulq_f32(vsubq_f32(f101, f100), dx));
    float32x4_t c11 = vaddq_f32(f110, vmulq_f32(vsubq_f32(f111, f110), dx));
    float32x4_t c0 = vaddq_f32(c00, vmulq_f32(vsubq_f32(c01, c00), dy));
    float32x4_t c1 = vaddq_f32(c10, vmulq_f32(vsubq_f32(c11, c10), dy));
    return vaddq_f32(c0, vmulq_f32(vsubq_f32(c1, c0), dz));
}
#elif defined(__SSE4_1__)
static inline uint16_t mc_s_ld16(const uint8_t *p) {
    uint16_t v; __builtin_memcpy(&v, p, 2); return v;
}
static inline __m128 mc_s_lerp8x4(const uint8_t *p0, const uint8_t *p1,
                                  const uint8_t *p2, const uint8_t *p3,
                                  size_t sy, size_t sx,
                                  __m128 dz, __m128 dy, __m128 dx) {
    // per corner-row: 4 samples' (c0,c1) byte pairs in u16 lanes 0..3
    __m128i z = _mm_setzero_si128();
    __m128i g00 = _mm_insert_epi16(z, mc_s_ld16(p0), 0);
    g00 = _mm_insert_epi16(g00, mc_s_ld16(p1), 1);
    g00 = _mm_insert_epi16(g00, mc_s_ld16(p2), 2);
    g00 = _mm_insert_epi16(g00, mc_s_ld16(p3), 3);
    __m128i g01 = _mm_insert_epi16(z, mc_s_ld16(p0 + sx), 0);
    g01 = _mm_insert_epi16(g01, mc_s_ld16(p1 + sx), 1);
    g01 = _mm_insert_epi16(g01, mc_s_ld16(p2 + sx), 2);
    g01 = _mm_insert_epi16(g01, mc_s_ld16(p3 + sx), 3);
    __m128i g10 = _mm_insert_epi16(z, mc_s_ld16(p0 + sy), 0);
    g10 = _mm_insert_epi16(g10, mc_s_ld16(p1 + sy), 1);
    g10 = _mm_insert_epi16(g10, mc_s_ld16(p2 + sy), 2);
    g10 = _mm_insert_epi16(g10, mc_s_ld16(p3 + sy), 3);
    __m128i g11 = _mm_insert_epi16(z, mc_s_ld16(p0 + sy + sx), 0);
    g11 = _mm_insert_epi16(g11, mc_s_ld16(p1 + sy + sx), 1);
    g11 = _mm_insert_epi16(g11, mc_s_ld16(p2 + sy + sx), 2);
    g11 = _mm_insert_epi16(g11, mc_s_ld16(p3 + sy + sx), 3);
    // split even bytes (x0 corner) / odd bytes (x1 corner) -> u32 -> f32
    const __m128i me = _mm_set_epi8(-1, -1, -1, 6, -1, -1, -1, 4,
                                    -1, -1, -1, 2, -1, -1, -1, 0);
    const __m128i mo = _mm_set_epi8(-1, -1, -1, 7, -1, -1, -1, 5,
                                    -1, -1, -1, 3, -1, -1, -1, 1);
#define MC_S_F32E(g) _mm_cvtepi32_ps(_mm_shuffle_epi8(g, me))
#define MC_S_F32O(g) _mm_cvtepi32_ps(_mm_shuffle_epi8(g, mo))
    __m128 f000 = MC_S_F32E(g00), f001 = MC_S_F32O(g00);
    __m128 f010 = MC_S_F32E(g01), f011 = MC_S_F32O(g01);
    __m128 f100 = MC_S_F32E(g10), f101 = MC_S_F32O(g10);
    __m128 f110 = MC_S_F32E(g11), f111 = MC_S_F32O(g11);
#undef MC_S_F32E
#undef MC_S_F32O
    __m128 c00 = _mm_add_ps(f000, _mm_mul_ps(_mm_sub_ps(f001, f000), dx));
    __m128 c01 = _mm_add_ps(f010, _mm_mul_ps(_mm_sub_ps(f011, f010), dx));
    __m128 c10 = _mm_add_ps(f100, _mm_mul_ps(_mm_sub_ps(f101, f100), dx));
    __m128 c11 = _mm_add_ps(f110, _mm_mul_ps(_mm_sub_ps(f111, f110), dx));
    __m128 c0 = _mm_add_ps(c00, _mm_mul_ps(_mm_sub_ps(c01, c00), dy));
    __m128 c1 = _mm_add_ps(c10, _mm_mul_ps(_mm_sub_ps(c11, c10), dy));
    return _mm_add_ps(c0, _mm_mul_ps(_mm_sub_ps(c1, c0), dz));
}

#if defined(__AVX2__) && !defined(MC_S_NO_TRI8)
// 8-wide variant for the x86-64-v3 fleet (Zen 3/4/5, 12th-gen+ Intel).
#define MC_S_HAVE_TRI8 1
static inline __m256i mc_s_g8(const uint8_t *const p[8], size_t off) {
    __m128i lo = _mm_setzero_si128(), hi = lo;
    lo = _mm_insert_epi16(lo, mc_s_ld16(p[0] + off), 0);
    lo = _mm_insert_epi16(lo, mc_s_ld16(p[1] + off), 1);
    lo = _mm_insert_epi16(lo, mc_s_ld16(p[2] + off), 2);
    lo = _mm_insert_epi16(lo, mc_s_ld16(p[3] + off), 3);
    hi = _mm_insert_epi16(hi, mc_s_ld16(p[4] + off), 0);
    hi = _mm_insert_epi16(hi, mc_s_ld16(p[5] + off), 1);
    hi = _mm_insert_epi16(hi, mc_s_ld16(p[6] + off), 2);
    hi = _mm_insert_epi16(hi, mc_s_ld16(p[7] + off), 3);
    return _mm256_set_m128i(hi, lo);
}
static inline __m256 mc_s_lerp8x8(const uint8_t *const p[8],
                                  size_t sy, size_t sx,
                                  __m256 dz, __m256 dy, __m256 dx) {
    __m256i g00 = mc_s_g8(p, 0),  g01 = mc_s_g8(p, sx);
    __m256i g10 = mc_s_g8(p, sy), g11 = mc_s_g8(p, sy + sx);
    // even byte of each u16 pair = x0 corner, odd = x1 (per 128-bit half)
    const __m256i me = _mm256_broadcastsi128_si256(
        _mm_set_epi8(-1, -1, -1, 6, -1, -1, -1, 4,
                     -1, -1, -1, 2, -1, -1, -1, 0));
    const __m256i mo = _mm256_broadcastsi128_si256(
        _mm_set_epi8(-1, -1, -1, 7, -1, -1, -1, 5,
                     -1, -1, -1, 3, -1, -1, -1, 1));
#define MC_S_F32E(g) _mm256_cvtepi32_ps(_mm256_shuffle_epi8(g, me))
#define MC_S_F32O(g) _mm256_cvtepi32_ps(_mm256_shuffle_epi8(g, mo))
    __m256 f000 = MC_S_F32E(g00), f001 = MC_S_F32O(g00);
    __m256 f010 = MC_S_F32E(g01), f011 = MC_S_F32O(g01);
    __m256 f100 = MC_S_F32E(g10), f101 = MC_S_F32O(g10);
    __m256 f110 = MC_S_F32E(g11), f111 = MC_S_F32O(g11);
#undef MC_S_F32E
#undef MC_S_F32O
    __m256 c00 = _mm256_add_ps(f000, _mm256_mul_ps(_mm256_sub_ps(f001, f000), dx));
    __m256 c01 = _mm256_add_ps(f010, _mm256_mul_ps(_mm256_sub_ps(f011, f010), dx));
    __m256 c10 = _mm256_add_ps(f100, _mm256_mul_ps(_mm256_sub_ps(f101, f100), dx));
    __m256 c11 = _mm256_add_ps(f110, _mm256_mul_ps(_mm256_sub_ps(f111, f110), dx));
    __m256 c0 = _mm256_add_ps(c00, _mm256_mul_ps(_mm256_sub_ps(c01, c00), dy));
    __m256 c1 = _mm256_add_ps(c10, _mm256_mul_ps(_mm256_sub_ps(c11, c10), dy));
    return _mm256_add_ps(c0, _mm256_mul_ps(_mm256_sub_ps(c1, c0), dz));
}
#endif  /* __AVX2__ */
#endif

static inline void mc_s_tri4(mc_sampler *s, const float *pz, const float *py,
                             const float *px, float *out) {
#if defined(__aarch64__)
    float32x4_t zv = vld1q_f32(pz), yv = vld1q_f32(py), xv = vld1q_f32(px);
    float32x4_t zf = vrndmq_f32(zv), yf = vrndmq_f32(yv), xf = vrndmq_f32(xv);
    int32x4_t zi = vcvtq_s32_f32(zf), yi = vcvtq_s32_f32(yf),
              xi = vcvtq_s32_f32(xf);
    // all-lanes in-bounds check: 0 <= c < n-1 per axis
    uint32x4_t ok = vcltq_u32(vreinterpretq_u32_s32(zi),
                              vdupq_n_u32((unsigned)(s->src.nz - 1)));
    ok = vandq_u32(ok, vcltq_u32(vreinterpretq_u32_s32(yi),
                                 vdupq_n_u32((unsigned)(s->src.ny - 1))));
    ok = vandq_u32(ok, vcltq_u32(vreinterpretq_u32_s32(xi),
                                 vdupq_n_u32((unsigned)(s->src.nx - 1))));
    if (vminvq_u32(ok) != 0) {
        float32x4_t dz = vsubq_f32(zv, zf), dy = vsubq_f32(yv, yf),
                    dx = vsubq_f32(xv, xf);
        if (s->src.dense) {
            int32_t z0[4], y0[4], x0[4];
            vst1q_s32(z0, zi); vst1q_s32(y0, yi); vst1q_s32(x0, xi);
            const size_t sy = s->src.dsy, sx = s->src.dsx;
            const uint8_t *base = s->src.dense;
            vst1q_f32(out, mc_s_lerp8x4(
                base + (size_t)z0[0] * sy + (size_t)y0[0] * sx + x0[0],
                base + (size_t)z0[1] * sy + (size_t)y0[1] * sx + x0[1],
                base + (size_t)z0[2] * sy + (size_t)y0[2] * sx + x0[2],
                base + (size_t)z0[3] * sy + (size_t)y0[3] * sx + x0[3],
                sy, sx, dz, dy, dx));
            return;
        }
        // blocked: every lane's 8 corners must sit inside one block
        uint32x4_t in15 = vmvnq_u32(vorrq_u32(vorrq_u32(
            vceqq_s32(vandq_s32(zi, vdupq_n_s32(15)), vdupq_n_s32(15)),
            vceqq_s32(vandq_s32(yi, vdupq_n_s32(15)), vdupq_n_s32(15))),
            vceqq_s32(vandq_s32(xi, vdupq_n_s32(15)), vdupq_n_s32(15))));
        if (vminvq_u32(in15) != 0) {
            int32_t z0[4], y0[4], x0[4];
            vst1q_s32(z0, zi); vst1q_s32(y0, yi); vst1q_s32(x0, xi);
            const uint8_t *b[4];
            int allb = 1;
            for (int k = 0; k < 4; k++) {
                const uint8_t *bk =
                    mc_s_block(s, z0[k] >> 4, y0[k] >> 4, x0[k] >> 4);
                if (!bk) { allb = 0; break; }
                b[k] = bk + (((z0[k] & 15) << 8) | ((y0[k] & 15) << 4) |
                             (x0[k] & 15));
            }
            if (allb) {
                vst1q_f32(out, mc_s_lerp8x4(b[0], b[1], b[2], b[3],
                                            256, 16, dz, dy, dx));
                return;
            }
        }
    }
#endif
#if defined(__SSE4_1__) && !defined(__aarch64__)
    __m128 zv = _mm_loadu_ps(pz), yv = _mm_loadu_ps(py), xv = _mm_loadu_ps(px);
    __m128 zf = _mm_floor_ps(zv), yf = _mm_floor_ps(yv), xf = _mm_floor_ps(xv);
    __m128i zi = _mm_cvttps_epi32(zf), yi = _mm_cvttps_epi32(yf),
            xi = _mm_cvttps_epi32(xf);
    // all-lanes 0 <= c < n-1 (signed compares; negatives fail the >= 0 side)
    __m128i ok = _mm_and_si128(
        _mm_cmpgt_epi32(zi, _mm_set1_epi32(-1)),
        _mm_cmpgt_epi32(_mm_set1_epi32(s->src.nz - 1), zi));
    ok = _mm_and_si128(ok, _mm_and_si128(
        _mm_cmpgt_epi32(yi, _mm_set1_epi32(-1)),
        _mm_cmpgt_epi32(_mm_set1_epi32(s->src.ny - 1), yi)));
    ok = _mm_and_si128(ok, _mm_and_si128(
        _mm_cmpgt_epi32(xi, _mm_set1_epi32(-1)),
        _mm_cmpgt_epi32(_mm_set1_epi32(s->src.nx - 1), xi)));
    if (_mm_movemask_ps(_mm_castsi128_ps(ok)) == 0xF) {
        __m128 dz = _mm_sub_ps(zv, zf), dy = _mm_sub_ps(yv, yf),
               dx = _mm_sub_ps(xv, xf);
        int32_t z0[4], y0[4], x0[4];
        _mm_storeu_si128((__m128i *)z0, zi);
        _mm_storeu_si128((__m128i *)y0, yi);
        _mm_storeu_si128((__m128i *)x0, xi);
        if (s->src.dense) {
            const size_t sy = s->src.dsy, sx = s->src.dsx;
            const uint8_t *base = s->src.dense;
            _mm_storeu_ps(out, mc_s_lerp8x4(
                base + (size_t)z0[0] * sy + (size_t)y0[0] * sx + x0[0],
                base + (size_t)z0[1] * sy + (size_t)y0[1] * sx + x0[1],
                base + (size_t)z0[2] * sy + (size_t)y0[2] * sx + x0[2],
                base + (size_t)z0[3] * sy + (size_t)y0[3] * sx + x0[3],
                sy, sx, dz, dy, dx));
            return;
        }
        int in15 = 1;
        for (int k = 0; k < 4; k++)
            in15 &= (z0[k] & 15) != 15 && (y0[k] & 15) != 15 &&
                    (x0[k] & 15) != 15;
        if (in15) {
            const uint8_t *b[4];
            int allb = 1;
            for (int k = 0; k < 4; k++) {
                const uint8_t *bk =
                    mc_s_block(s, z0[k] >> 4, y0[k] >> 4, x0[k] >> 4);
                if (!bk) { allb = 0; break; }
                b[k] = bk + (((z0[k] & 15) << 8) | ((y0[k] & 15) << 4) |
                             (x0[k] & 15));
            }
            if (allb) {
                _mm_storeu_ps(out, mc_s_lerp8x4(b[0], b[1], b[2], b[3],
                                                256, 16, dz, dy, dx));
                return;
            }
        }
    }
#endif
    out[0] = mc_s_trilinear(s, pz[0], py[0], px[0]);
    out[1] = mc_s_trilinear(s, pz[1], py[1], px[1]);
    out[2] = mc_s_trilinear(s, pz[2], py[2], px[2]);
    out[3] = mc_s_trilinear(s, pz[3], py[3], px[3]);
}

#ifdef MC_S_HAVE_TRI8
// 8 positions at once (AVX2). Same fallback discipline as mc_s_tri4.
static inline void mc_s_tri8(mc_sampler *s, const float *pz, const float *py,
                             const float *px, float *out) {
    __m256 zv = _mm256_loadu_ps(pz), yv = _mm256_loadu_ps(py),
           xv = _mm256_loadu_ps(px);
    __m256 zf = _mm256_floor_ps(zv), yf = _mm256_floor_ps(yv),
           xf = _mm256_floor_ps(xv);
    __m256i zi = _mm256_cvttps_epi32(zf), yi = _mm256_cvttps_epi32(yf),
            xi = _mm256_cvttps_epi32(xf);
    __m256i ok = _mm256_and_si256(
        _mm256_cmpgt_epi32(zi, _mm256_set1_epi32(-1)),
        _mm256_cmpgt_epi32(_mm256_set1_epi32(s->src.nz - 1), zi));
    ok = _mm256_and_si256(ok, _mm256_and_si256(
        _mm256_cmpgt_epi32(yi, _mm256_set1_epi32(-1)),
        _mm256_cmpgt_epi32(_mm256_set1_epi32(s->src.ny - 1), yi)));
    ok = _mm256_and_si256(ok, _mm256_and_si256(
        _mm256_cmpgt_epi32(xi, _mm256_set1_epi32(-1)),
        _mm256_cmpgt_epi32(_mm256_set1_epi32(s->src.nx - 1), xi)));
    if (_mm256_movemask_ps(_mm256_castsi256_ps(ok)) == 0xFF) {
        __m256 dz = _mm256_sub_ps(zv, zf), dy = _mm256_sub_ps(yv, yf),
               dx = _mm256_sub_ps(xv, xf);
        int32_t z0[8], y0[8], x0[8];
        _mm256_storeu_si256((__m256i *)z0, zi);
        _mm256_storeu_si256((__m256i *)y0, yi);
        _mm256_storeu_si256((__m256i *)x0, xi);
        const uint8_t *b[8];
        if (s->src.dense) {
            const size_t sy = s->src.dsy, sx = s->src.dsx;
            for (int k = 0; k < 8; k++)
                b[k] = s->src.dense + (size_t)z0[k] * sy +
                       (size_t)y0[k] * sx + x0[k];
            _mm256_storeu_ps(out, mc_s_lerp8x8(b, sy, sx, dz, dy, dx));
            return;
        }
        int fast = 1;
        for (int k = 0; k < 8 && fast; k++)
            fast = (z0[k] & 15) != 15 && (y0[k] & 15) != 15 &&
                   (x0[k] & 15) != 15;
        if (fast) {
            for (int k = 0; k < 8 && fast; k++) {
                const uint8_t *bk =
                    mc_s_block(s, z0[k] >> 4, y0[k] >> 4, x0[k] >> 4);
                if (!bk) { fast = 0; break; }
                b[k] = bk + (((z0[k] & 15) << 8) | ((y0[k] & 15) << 4) |
                             (x0[k] & 15));
            }
            if (fast) {
                _mm256_storeu_ps(out, mc_s_lerp8x8(b, 256, 16, dz, dy, dx));
                return;
            }
        }
    }
    mc_s_tri4(s, pz, py, px, out);
    mc_s_tri4(s, pz + 4, py + 4, px + 4, out + 4);
}
#endif


#define BLK 16
#define BLKB 4096

// ---------------------------------------------------------------------------
// sources
// ---------------------------------------------------------------------------
static const uint8_t *cache_block(const mc_sample_src *src,
                                  int bz, int by, int bx, uint8_t *tmp) {
    (void)tmp;
    // mc_cache_get decodes misses synchronously and returns an arena pointer
    // (non-owning; same stability contract as VC's BlockCache). In frozen
    // tick-phase a miss returns NULL — sampled as 0, recorded as feedback.
    return mc_cache_get((mc_cache *)src->ud, src->aux, bz, by, bx);
}

mc_sample_src mc_sample_src_cache(struct mc_cache *c, int lod,
                                  int nz, int ny, int nx) {
    mc_sample_src s = {0};
    s.ud = c; s.aux = lod; s.block = cache_block;
    s.nz = nz; s.ny = ny; s.nx = nx;
    return s;
}

static const uint8_t *dense_block(const mc_sample_src *src,
                                  int bz, int by, int bx, uint8_t *tmp) {
    const uint8_t *vox = src->ud;
    int z0 = bz * BLK, y0 = by * BLK, x0 = bx * BLK;
    if (z0 >= src->nz || y0 >= src->ny || x0 >= src->nx) return NULL;
    int dz = src->nz - z0 < BLK ? src->nz - z0 : BLK;
    int dy = src->ny - y0 < BLK ? src->ny - y0 : BLK;
    int dx = src->nx - x0 < BLK ? src->nx - x0 : BLK;
    if (dz < BLK || dy < BLK || dx < BLK) memset(tmp, 0, BLKB);
    for (int z = 0; z < dz; z++)
        for (int y = 0; y < dy; y++)
            memcpy(tmp + ((z << 8) | (y << 4)),
                   vox + ((size_t)(z0 + z) * src->ny + (y0 + y)) * src->nx + x0,
                   (size_t)dx);
    return tmp;
}

mc_sample_src mc_sample_src_dense(const uint8_t *vox, int nz, int ny, int nx) {
    mc_sample_src s = {0};
    s.ud = (void *)(uintptr_t)vox;
    s.block = dense_block;                // kept for completeness; the direct
    s.dense = vox;                        // path below short-circuits it
    s.dsy = (size_t)ny * nx; s.dsx = (size_t)nx;
    s.nz = nz; s.ny = ny; s.nx = nx;
    return s;
}

// ---------------------------------------------------------------------------
// sampler
// ---------------------------------------------------------------------------
mc_sampler *mc_sampler_new(const mc_sample_src *src) {
    if (!src || !src->block) return NULL;
    mc_sampler *s = malloc(sizeof *s);
    if (!s) return NULL;
    s->src = *src;
    s->nbz = (src->nz + BLK - 1) / BLK;
    s->nby = (src->ny + BLK - 1) / BLK;
    s->nbx = (src->nx + BLK - 1) / BLK;
    mc_sampler_reset(s);
    return s;
}

void mc_sampler_free(mc_sampler *s) { free(s); }

void mc_sampler_reset(mc_sampler *s) {
    if (!s) return;
    for (int i = 0; i < MC_S_MEMO; i++) s->m[i].bz = -1;
    s->lbz = s->lby = s->lbx = -1;
    s->lptr = NULL;
}

float mc_sample_point(mc_sampler *s, float z, float y, float x, mc_filter f) {
    return mc_s_sample(s, z, y, x, f);
}

static inline int pt_valid(const float *p) {
    if (p[0] != p[0] || p[1] != p[1] || p[2] != p[2]) return 0;   // NaN
    return p[0] >= 0.0f && p[1] >= 0.0f && p[2] >= 0.0f;
}

void mc_sample_points(mc_sampler *s, const float *zyx, size_t n,
                      mc_filter f, float *out) {
    if (f == MC_FILTER_NEAREST) {
        for (size_t i = 0; i < n; i++) {
            const float *p = zyx + i * 3;
            out[i] = pt_valid(p) ? mc_s_nearest(s, p[0], p[1], p[2]) : 0.0f;
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            const float *p = zyx + i * 3;
            out[i] = pt_valid(p) ? mc_s_trilinear(s, p[0], p[1], p[2]) : 0.0f;
        }
    }
}

void mc_sample_points_u8(mc_sampler *s, const float *zyx, size_t n,
                         mc_filter f, uint8_t *out) {
    if (f == MC_FILTER_NEAREST) {
        for (size_t i = 0; i < n; i++) {
            const float *p = zyx + i * 3;
            float v = pt_valid(p) ? mc_s_nearest(s, p[0], p[1], p[2]) : 0.0f;
            out[i] = (uint8_t)(v < 0.0f ? 0 : v > 255.0f ? 255 : (int)(v + 0.5f));
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            const float *p = zyx + i * 3;
            float v = pt_valid(p) ? mc_s_trilinear(s, p[0], p[1], p[2]) : 0.0f;
            out[i] = (uint8_t)(v < 0.0f ? 0 : v > 255.0f ? 255 : (int)(v + 0.5f));
        }
    }
}

// ============================================================================
// mc_render — surface rendering, compositing, LOD, 3D resampling
// ============================================================================
// ---------------------------------------------------------------------------
// core
// ---------------------------------------------------------------------------
static inline uint8_t to_u8(float v) {
    return (uint8_t)(v < 0.0f ? 0 : v > 255.0f ? 255 : (int)(v + 0.5f));
}

// Per-image constants hoisted out of the pixel loop.
typedef struct {
    mc_filter filter;
    mc_comp comp;
    float t0, dt;
    int nsteps;                 // iterations of the [t0, t1] dt walk
    float a_min, a_op;          // alpha params, clamped
} rcfg_t;

static rcfg_t make_cfg(const mc_render_params *p) {
    rcfg_t c;
    c.filter = p->filter;
    c.comp = p->comp;
    c.dt = p->dt > 0.0f ? p->dt : 1.0f;
    float t0 = p->t0, t1 = p->t1;
    if (t1 < t0) { float tmp = t0; t0 = t1; t1 = tmp; }
    c.t0 = t0;
    c.nsteps = 0;
    for (float t = t0; t <= t1 + 1e-4f; t += c.dt) c.nsteps++;
    c.a_min = p->alpha_min < 0.0f ? 0.0f :
              p->alpha_min > 0.99f ? 0.99f : p->alpha_min;
    c.a_op  = p->alpha_opacity <= 0.0f ? 1.0f :
              p->alpha_opacity > 1.0f ? 1.0f : p->alpha_opacity;
    return c;
}

// Composite one ray. Trilinear rays are consumed in chunks of 4 steps via
// mc_s_tri4 (NEON gather+lerp on aarch64, ~1.4x the scalar core); the
// accumulation itself stays sequential per chunk, which keeps ALPHA\'s
// front-to-back order and early-out exact. Positions are P + t*N with t
// advanced additively, as before.
static uint8_t render_pixel(mc_sampler *s, const float *P, const float *N,
                            const rcfg_t *cfg) {
    if (!pt_valid(P)) return 0;
    if (cfg->comp == MC_COMP_NONE || !N)
        return to_u8(mc_s_sample(s, P[0], P[1], P[2], cfg->filter));
    float nz = N[0], ny = N[1], nx = N[2];
    float n2 = nz * nz + ny * ny + nx * nx;
    if (n2 < 1e-12f)
        return to_u8(mc_s_sample(s, P[0], P[1], P[2], cfg->filter));
    if (n2 < 0.9998f || n2 > 1.0002f) {     // gen paths emit unit normals
        float nl = 1.0f / sqrtf(n2);
        nz *= nl; ny *= nl; nx *= nl;
    }

    const float sz_ = cfg->dt * nz, sy_ = cfg->dt * ny, sx_ = cfg->dt * nx;
    float pz = P[0] + cfg->t0 * nz, py = P[1] + cfg->t0 * ny,
          px = P[2] + cfg->t0 * nx;
    const float a_th = cfg->a_min, a_sc = cfg->a_op / (1.0f - cfg->a_min);
    float acc = 0.0f, A = 0.0f, mn = 255.0f, mx = 0.0f, sum = 0.0f;
    int it = 0, done = 0;

    if (cfg->filter == MC_FILTER_TRILINEAR) {
// NOTE: composites deliberately stay 4-wide. Measured on Zen 3 (EPYC
        // 7763): 8-wide ray chunks ran 1.6x SLOWER than two independent 4-wide
        // chunks (the 8-long insert-gather dependency chain over z-strided
        // addresses serializes); 8-wide only wins for adjacent-pixel loads
        // (the slice path below).
        for (; it + 4 <= cfg->nsteps && !done; it += 4) {
            float bz[4], by[4], bx[4], v4[4];
            for (int k = 0; k < 4; k++) {
                bz[k] = pz; by[k] = py; bx[k] = px;
                pz += sz_; py += sy_; px += sx_;
            }
            mc_s_tri4(s, bz, by, bx, v4);
            switch (cfg->comp) {
            case MC_COMP_MIN:
                for (int k = 0; k < 4; k++) if (v4[k] < mn) mn = v4[k];
                break;
            case MC_COMP_MAX:
                for (int k = 0; k < 4; k++) if (v4[k] > mx) mx = v4[k];
                if (mx >= 255.0f) done = 1;     // saturated
                break;
            case MC_COMP_MEAN:
                sum += v4[0] + v4[1] + v4[2] + v4[3];
                break;
            default:                            // ALPHA
                for (int k = 0; k < 4 && !done; k++) {
                    float a = (v4[k] * (1.0f / 255.0f) - a_th) * a_sc;
                    if (a > 0.0f) {
                        if (a > 1.0f) a = 1.0f;
                        acc += (1.0f - A) * a * v4[k];
                        A   += (1.0f - A) * a;
                        if (A >= 0.98f) done = 1;
                    }
                }
                break;
            }
        }
    }
    for (; it < cfg->nsteps && !done; it++) {
        float v = mc_s_sample(s, pz, py, px, cfg->filter);
        switch (cfg->comp) {
        case MC_COMP_MIN:  if (v < mn) mn = v; break;
        case MC_COMP_MAX:  if (v > mx) mx = v; break;
        case MC_COMP_MEAN: sum += v; break;
        default: {                              // ALPHA
            float a = (v * (1.0f / 255.0f) - a_th) * a_sc;
            if (a > 0.0f) {
                if (a > 1.0f) a = 1.0f;
                acc += (1.0f - A) * a * v;
                A   += (1.0f - A) * a;
                if (A >= 0.98f) done = 1;
            }
            break;
        }
        }
        pz += sz_; py += sy_; px += sx_;
    }
    switch (cfg->comp) {
    case MC_COMP_MIN:  return to_u8(mn);
    case MC_COMP_MAX:  return to_u8(mx);
    case MC_COMP_MEAN:
        return to_u8(cfg->nsteps ? sum / (float)cfg->nsteps : 0.0f);
    case MC_COMP_ALPHA: return to_u8(acc);
    default:           return 0;
    }
}

void mc_render_points(mc_sampler *s,
                      const float *pts, const float *normals,
                      int w, int h, const mc_render_params *p, uint8_t *out) {
    rcfg_t cfg = make_cfg(p);
    size_t n = (size_t)w * h;
    if (cfg.comp == MC_COMP_NONE || !normals) {
        // slice fast path: no per-pixel normal handling, branch on the
        // filter once
        if (cfg.filter == MC_FILTER_NEAREST) {
            for (size_t k = 0; k < n; k++) {
                const float *P = pts + k * 3;
                out[k] = pt_valid(P)
                             ? to_u8(mc_s_nearest(s, P[0], P[1], P[2])) : 0;
            }
        } else {
            // 4/8 pixels per mc_s_tri4/8 call (SIMD gather+lerp)
            size_t k = 0;
#ifdef MC_S_HAVE_TRI8
            for (; k + 8 <= n; k += 8) {
                const float *P = pts + k * 3;
                int allv = 1;
                for (int q = 0; q < 8; q++) allv &= pt_valid(P + q * 3);
                if (allv) {
                    float bz[8], by[8], bx[8], v8[8];
                    for (int q = 0; q < 8; q++) {
                        bz[q] = P[q * 3]; by[q] = P[q * 3 + 1];
                        bx[q] = P[q * 3 + 2];
                    }
                    mc_s_tri8(s, bz, by, bx, v8);
                    for (int q = 0; q < 8; q++) out[k + q] = to_u8(v8[q]);
                } else {
                    for (size_t q = k; q < k + 8; q++) {
                        const float *Q = pts + q * 3;
                        out[q] = pt_valid(Q)
                            ? to_u8(mc_s_trilinear(s, Q[0], Q[1], Q[2])) : 0;
                    }
                }
            }
#endif
            for (; k + 4 <= n; k += 4) {
                const float *P = pts + k * 3;
                if (pt_valid(P) && pt_valid(P + 3) &&
                    pt_valid(P + 6) && pt_valid(P + 9)) {
                    float bz[4] = { P[0], P[3], P[6], P[9]  };
                    float by[4] = { P[1], P[4], P[7], P[10] };
                    float bx[4] = { P[2], P[5], P[8], P[11] };
                    float v4[4];
                    mc_s_tri4(s, bz, by, bx, v4);
                    out[k]     = to_u8(v4[0]);
                    out[k + 1] = to_u8(v4[1]);
                    out[k + 2] = to_u8(v4[2]);
                    out[k + 3] = to_u8(v4[3]);
                } else {
                    for (size_t q = k; q < k + 4; q++) {
                        const float *Q = pts + q * 3;
                        out[q] = pt_valid(Q)
                            ? to_u8(mc_s_trilinear(s, Q[0], Q[1], Q[2])) : 0;
                    }
                }
            }
            for (; k < n; k++) {
                const float *P = pts + k * 3;
                out[k] = pt_valid(P)
                             ? to_u8(mc_s_trilinear(s, P[0], P[1], P[2])) : 0;
            }
        }
        return;
    }
    for (size_t k = 0; k < n; k++)
        out[k] = render_pixel(s, pts + k * 3, normals + k * 3, &cfg);
}

// ---------------------------------------------------------------------------
// parallel core: row bands, one sampler per worker
// ---------------------------------------------------------------------------
// rowgen fills one row of points (+normals) into band-local scratch; plane
// and quad renders go through this so no W*H grid is ever materialized
// (a 1024^2 trilinear frame otherwise mallocs and touches 24 MB of points).
typedef void (*rowgen_fn)(const void *ud, int row, int w,
                          float *pts, float *normals);

typedef struct {
    const mc_sample_src *src;
    const float *pts, *normals;     // dense mode (rowgen == NULL)
    rowgen_fn rowgen;               // strip mode
    const void *rg_ud;
    int w, h;
    const mc_render_params *p;
    uint8_t *out;
    int row0, row1;
} band_t;

static void *band_main(void *ud) {
    band_t *b = ud;
    mc_sampler *s = mc_sampler_new(b->src);
    if (!s) return NULL;
    if (!b->rowgen) {
        size_t off = (size_t)b->row0 * b->w;
        mc_render_points(s, b->pts + off * 3,
                         b->normals ? b->normals + off * 3 : NULL,
                         b->w, b->row1 - b->row0, b->p, b->out + off);
    } else {
        int need_n = b->p->comp != MC_COMP_NONE;
        float *row = malloc((size_t)b->w * 3 * sizeof(float) * (need_n ? 2 : 1));
        if (row) {
            float *nrm = need_n ? row + (size_t)b->w * 3 : NULL;
            for (int i = b->row0; i < b->row1; i++) {
                b->rowgen(b->rg_ud, i, b->w, row, nrm);
                mc_render_points(s, row, nrm, b->w, 1, b->p,
                                 b->out + (size_t)i * b->w);
            }
            free(row);
        }
        else memset(b->out + (size_t)b->row0 * b->w, 0,
                    (size_t)(b->row1 - b->row0) * b->w);
    }
    mc_sampler_free(s);
    return NULL;
}

static void render_bands(const mc_sample_src *src,
                         const float *pts, const float *normals,
                         rowgen_fn rowgen, const void *rg_ud,
                         int w, int h, const mc_render_params *p,
                         uint8_t *out, int nthreads) {
    if (w <= 0 || h <= 0) return;
    if (nthreads <= 0) {
        long nc = sysconf(_SC_NPROCESSORS_ONLN);
        nthreads = nc > 0 ? (int)nc : 1;
    }
    if (nthreads > 16) nthreads = 16;
    if (nthreads > h)  nthreads = h;
    pthread_t th[16];
    band_t bands[16];
    int per = (h + nthreads - 1) / nthreads;
    int nb = 0;
    for (int t = 0; t < nthreads; t++) {
        int r0 = t * per, r1 = r0 + per > h ? h : r0 + per;
        if (r0 >= r1) break;
        bands[nb] = (band_t){ src, pts, normals, rowgen, rg_ud,
                              w, h, p, out, r0, r1 };
        if (nthreads == 1) { band_main(&bands[nb]); continue; }
        if (pthread_create(&th[nb], NULL, band_main, &bands[nb]) != 0) {
            band_main(&bands[nb]);          // degrade to inline
            continue;
        }
        nb++;
    }
    for (int t = 0; t < nb; t++) pthread_join(th[t], NULL);
}

void mc_render_points_par(const mc_sample_src *src,
                          const float *pts, const float *normals,
                          int w, int h, const mc_render_params *p,
                          uint8_t *out, int nthreads) {
    render_bands(src, pts, normals, NULL, NULL, w, h, p, out, nthreads);
}

// ---------------------------------------------------------------------------
// plane surface
// ---------------------------------------------------------------------------
static inline void v3_norm(float *v) {
    float l = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (l > 1e-12f) { v[0] /= l; v[1] /= l; v[2] /= l; }
}
static inline void v3_cross(const float *a, const float *b, float *o) {
    o[0] = a[1] * b[2] - a[2] * b[1];
    o[1] = a[2] * b[0] - a[0] * b[2];
    o[2] = a[0] * b[1] - a[1] * b[0];
}

void mc_plane_basis(mc_plane *pl) {
    float *n = pl->normal;
    v3_norm(n);
    // pick the world axis least aligned with n as the seed
    float az = fabsf(n[0]), ay = fabsf(n[1]), ax = fabsf(n[2]);
    float e[3] = {0, 0, 0};
    if (az <= ay && az <= ax) e[0] = 1.0f;
    else if (ay <= ax)        e[1] = 1.0f;
    else                      e[2] = 1.0f;
    v3_cross(n, e, pl->u); v3_norm(pl->u);
    v3_cross(n, pl->u, pl->v); v3_norm(pl->v);
}

void mc_plane_gen(const mc_plane *pl, int w, int h, float scale,
                  float *pts, float *normals) {
    float cx = (float)w * 0.5f, cy = (float)h * 0.5f;
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            float du = ((float)j - cx) * scale;
            float dv = ((float)i - cy) * scale;
            float *P = pts + ((size_t)i * w + j) * 3;
            for (int k = 0; k < 3; k++)
                P[k] = pl->origin[k] + du * pl->u[k] + dv * pl->v[k];
            if (normals) {
                float *N = normals + ((size_t)i * w + j) * 3;
                N[0] = pl->normal[0]; N[1] = pl->normal[1]; N[2] = pl->normal[2];
            }
        }
}

// ---------------------------------------------------------------------------
// quad surface
// ---------------------------------------------------------------------------
static inline int qvalid(const float *p) {
    return !(p[0] == -1.0f && p[1] == -1.0f && p[2] == -1.0f) &&
           p[0] == p[0] && p[1] == p[1] && p[2] == p[2];
}

void mc_quad_gen(const mc_quad *q, float x0, float y0, float step,
                 int w, int h, float *pts, float *normals) {
    const int gw = q->gw, gh = q->gh;
    for (int i = 0; i < h; i++) {
        float *Prow = pts + (size_t)i * w * 3;
        float *Nrow = normals ? normals + (size_t)i * w * 3 : NULL;
        float gy = y0 + (float)i * step;
        // row-invalid fast fill
        if (!(gy >= 0.0f) || gy > (float)(gh - 1)) {
            for (int j = 0; j < w; j++) {
                Prow[j * 3] = Prow[j * 3 + 1] = Prow[j * 3 + 2] = -1.0f;
                if (Nrow) Nrow[j * 3] = Nrow[j * 3 + 1] = Nrow[j * 3 + 2] = 0.0f;
            }
            continue;
        }
        int y0i = (int)gy;
        if (y0i > gh - 2) y0i = gh - 2;
        if (y0i < 0) y0i = 0;               // gh == 1
        float fy = gy - (float)y0i;
        const float *r0 = q->grid + (size_t)y0i * gw * 3;
        const float *r1 = q->grid + (size_t)(y0i + (gh > 1)) * gw * 3;

        // per-cell state, reloaded only when the pixel crosses a grid cell
        int cell = -2, cell_ok = 0;
        float A[3], B[3], du[3], dv0[3], dv1[3];
        for (int j = 0; j < w; j++) {
            float *P = Prow + j * 3;
            float *N = Nrow ? Nrow + j * 3 : NULL;
            P[0] = P[1] = P[2] = -1.0f;
            if (N) N[0] = N[1] = N[2] = 0.0f;
            float gx = x0 + (float)j * step;
            if (!(gx >= 0.0f) || gx > (float)(gw - 1)) continue;
            int x0i = (int)gx;
            if (x0i > gw - 2) x0i = gw - 2;
            if (x0i < 0) x0i = 0;           // gw == 1
            if (x0i != cell) {
                cell = x0i;
                const float *p00 = r0 + (size_t)x0i * 3;
                const float *p01 = r0 + (size_t)(x0i + (gw > 1)) * 3;
                const float *p10 = r1 + (size_t)x0i * 3;
                const float *p11 = r1 + (size_t)(x0i + (gw > 1)) * 3;
                cell_ok = qvalid(p00) && qvalid(p01) &&
                          qvalid(p10) && qvalid(p11);
                if (cell_ok)
                    for (int k = 0; k < 3; k++) {
                        // y-lerped column endpoints: P = A + (B-A)*fx
                        A[k] = p00[k] + (p10[k] - p00[k]) * fy;
                        B[k] = p01[k] + (p11[k] - p01[k]) * fy;
                        // bilinear tangents (du constant per cell row)
                        du[k] = (p01[k] - p00[k]) * (1.0f - fy) +
                                (p11[k] - p10[k]) * fy;
                        dv0[k] = p10[k] - p00[k];
                        dv1[k] = p11[k] - p01[k];
                    }
            }
            if (!cell_ok) continue;
            float fx = gx - (float)x0i;
            P[0] = A[0] + (B[0] - A[0]) * fx;
            P[1] = A[1] + (B[1] - A[1]) * fx;
            P[2] = A[2] + (B[2] - A[2]) * fx;
            if (N) {
                float dv[3] = {
                    dv0[0] + (dv1[0] - dv0[0]) * fx,
                    dv0[1] + (dv1[1] - dv0[1]) * fx,
                    dv0[2] + (dv1[2] - dv0[2]) * fx,
                };
                v3_cross(du, dv, N);
                v3_norm(N);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// conveniences (row-strip rendering, no W*H grid)
// ---------------------------------------------------------------------------
typedef struct {
    mc_plane pl;                // normal pre-normalized
    float scale, cx, cy;
} plane_rg;

static void plane_rowgen(const void *ud, int row, int w,
                         float *pts, float *normals) {
    const plane_rg *g = ud;
    float dv = ((float)row - g->cy) * g->scale;
    float base[3], du[3];
    for (int k = 0; k < 3; k++) {
        base[k] = g->pl.origin[k] + dv * g->pl.v[k]
                  - g->cx * g->scale * g->pl.u[k];
        du[k] = g->scale * g->pl.u[k];
    }
    for (int j = 0; j < w; j++) {
        pts[j * 3 + 0] = base[0] + (float)j * du[0];
        pts[j * 3 + 1] = base[1] + (float)j * du[1];
        pts[j * 3 + 2] = base[2] + (float)j * du[2];
    }
    if (normals)
        for (int j = 0; j < w; j++) {
            normals[j * 3 + 0] = g->pl.normal[0];
            normals[j * 3 + 1] = g->pl.normal[1];
            normals[j * 3 + 2] = g->pl.normal[2];
        }
}

int mc_render_plane(const mc_sample_src *src, const mc_plane *pl,
                    int w, int h, float scale,
                    const mc_render_params *p, uint8_t *out, int nthreads) {
    if (!src || !pl || !p || !out || w <= 0 || h <= 0) return -1;
    plane_rg g = { *pl, scale, (float)w * 0.5f, (float)h * 0.5f };
    v3_norm(g.pl.normal);
    render_bands(src, NULL, NULL, plane_rowgen, &g, w, h, p, out, nthreads);
    return 0;
}

typedef struct {
    const mc_quad *q;
    float x0, y0, step;
} quad_rg;

static void quad_rowgen(const void *ud, int row, int w,
                        float *pts, float *normals) {
    const quad_rg *g = ud;
    mc_quad_gen(g->q, g->x0, g->y0 + (float)row * g->step, g->step,
                w, 1, pts, normals);
}

int mc_render_quad(const mc_sample_src *src, const mc_quad *q,
                   float x0, float y0, float step, int w, int h,
                   const mc_render_params *p, uint8_t *out, int nthreads) {
    if (!src || !q || !q->grid || q->gw < 1 || q->gh < 1 ||
        !p || !out || w <= 0 || h <= 0) return -1;
    quad_rg g = { q, x0, y0, step };
    render_bands(src, NULL, NULL, quad_rowgen, &g, w, h, p, out, nthreads);
    return 0;
}

// ---------------------------------------------------------------------------
// LOD-matched rendering
// ---------------------------------------------------------------------------
int mc_render_pick_lod(const mc_sample_lods *ls, float vox_per_pixel) {
    if (!ls || ls->nlods <= 1) return 0;
    int L = 0;
    float v = vox_per_pixel;
    while (v >= 2.0f && L < ls->nlods - 1) { v *= 0.5f; L++; }
    // skip levels the caller left empty
    while (L > 0 && (ls->lods[L].nz <= 0 || !ls->lods[L].block)) L--;
    return L;
}

float mc_quad_spacing(const mc_quad *q) {
    if (!q || !q->grid || q->gw < 2 || q->gh < 1) return 1.0f;
    // probe up to 32 horizontal neighbor pairs along the grid diagonal
    double sum = 0.0;
    int n = 0;
    int probes = q->gh < 32 ? q->gh : 32;
    for (int i = 0; i < probes; i++) {
        int gy = (int)(((int64_t)i * (q->gh - 1)) / (probes > 1 ? probes - 1 : 1));
        int gx = (int)(((int64_t)i * (q->gw - 2)) / (probes > 1 ? probes - 1 : 1));
        const float *a = q->grid + ((size_t)gy * q->gw + gx) * 3;
        const float *b = a + 3;
        if (!qvalid(a) || !qvalid(b)) continue;
        float dz = b[0] - a[0], dy = b[1] - a[1], dx = b[2] - a[2];
        sum += sqrtf(dz * dz + dy * dy + dx * dx);
        n++;
    }
    return n ? (float)(sum / n) : 1.0f;
}

// wrap a rowgen: remap generated LOD-0 points into LOD-L voxel space.
// c_L = c_0 * 2^-L + (0.5 * 2^-L - 0.5); border points that map a fraction
// below 0 clamp to 0 (they are inside voxel 0 of the coarse level) instead
// of tripping the <0 invalid convention.
typedef struct {
    rowgen_fn inner;
    const void *inner_ud;
    float a, b;
} lod_rg;

static void lod_rowgen(const void *ud, int row, int w,
                       float *pts, float *normals) {
    const lod_rg *g = ud;
    g->inner(g->inner_ud, row, w, pts, normals);
    for (int j = 0; j < w; j++) {
        float *P = pts + (size_t)j * 3;
        if (!pt_valid(P)) continue;
        for (int k = 0; k < 3; k++) {
            float v = P[k] * g->a + g->b;
            P[k] = v < 0.0f ? 0.0f : v;
        }
    }
    // normals are directions: unchanged under uniform scaling
}

static int render_lod(const mc_sample_lods *ls, int L,
                      rowgen_fn inner, const void *inner_ud,
                      int w, int h, const mc_render_params *p,
                      uint8_t *out, int nthreads) {
    const float inv = 1.0f / (float)(1 << L);
    lod_rg g = { inner, inner_ud, inv, 0.5f * inv - 0.5f };
    mc_render_params pl_ = *p;
    pl_.t0 = p->t0 * inv;       // same physical slab ...
    pl_.t1 = p->t1 * inv;       // ... stepped at the coarse level's pitch
    render_bands(&ls->lods[L], NULL, NULL, lod_rowgen, &g, w, h, &pl_,
                 out, nthreads);
    return 0;
}

int mc_render_plane_lod(const mc_sample_lods *ls, const mc_plane *pl,
                        int w, int h, float scale,
                        const mc_render_params *p, uint8_t *out, int nthreads) {
    if (!ls || !pl || !p || !out || w <= 0 || h <= 0) return -1;
    int L = mc_render_pick_lod(ls, scale);
    if (L == 0)
        return mc_render_plane(&ls->lods[0], pl, w, h, scale, p, out, nthreads);
    plane_rg g = { *pl, scale, (float)w * 0.5f, (float)h * 0.5f };
    v3_norm(g.pl.normal);
    return render_lod(ls, L, plane_rowgen, &g, w, h, p, out, nthreads);
}

int mc_render_quad_lod(const mc_sample_lods *ls, const mc_quad *q,
                       float x0, float y0, float step, int w, int h,
                       const mc_render_params *p, uint8_t *out, int nthreads) {
    if (!ls || !q || !q->grid || q->gw < 1 || q->gh < 1 ||
        !p || !out || w <= 0 || h <= 0) return -1;
    int L = mc_render_pick_lod(ls, step * mc_quad_spacing(q));
    if (L == 0)
        return mc_render_quad(&ls->lods[0], q, x0, y0, step, w, h, p, out,
                              nthreads);
    quad_rg g = { q, x0, y0, step };
    return render_lod(ls, L, quad_rowgen, &g, w, h, p, out, nthreads);
}

// ---------------------------------------------------------------------------
// 3D resampling (surface-aligned volumes)
// ---------------------------------------------------------------------------
typedef struct {
    const mc_sample_src *src;
    const mc_quad *q;
    float x0, y0, step;
    float t0, dt;
    int w, h, nlayers;
    mc_filter f;
    uint8_t *out;
    int row0, row1;
} qvol_band_t;

static void *qvol_band_main(void *ud) {
    qvol_band_t *b = ud;
    mc_sampler *s = mc_sampler_new(b->src);
    float *row = malloc((size_t)b->w * 6 * sizeof(float));
    const size_t layer = (size_t)b->w * b->h;
    if (!s || !row) {
        for (int k = 0; k < b->nlayers; k++)
            memset(b->out + layer * k + (size_t)b->row0 * b->w, 0,
                   (size_t)(b->row1 - b->row0) * b->w);
        free(row); mc_sampler_free(s);
        return NULL;
    }
    float *nrm = row + (size_t)b->w * 3;
    quad_rg g = { b->q, b->x0, b->y0, b->step };
    for (int i = b->row0; i < b->row1; i++) {
        quad_rowgen(&g, i, b->w, row, nrm);
        for (int j = 0; j < b->w; j++) {
            const float *P = row + (size_t)j * 3;
            const float *N = nrm + (size_t)j * 3;
            uint8_t *o = b->out + (size_t)i * b->w + j;
            if (!pt_valid(P)) {
                for (int k = 0; k < b->nlayers; k++) o[layer * k] = 0;
                continue;
            }
            float nz = N[0], ny = N[1], nx = N[2];
            float n2 = nz * nz + ny * ny + nx * nx;
            if (n2 >= 1e-12f && (n2 < 0.9998f || n2 > 1.0002f)) {
                float nl = 1.0f / sqrtf(n2);
                nz *= nl; ny *= nl; nx *= nl;
            }
            float pz = P[0] + b->t0 * nz, py = P[1] + b->t0 * ny,
                  px = P[2] + b->t0 * nx;
            const float sz_ = b->dt * nz, sy_ = b->dt * ny, sx_ = b->dt * nx;
            int k = 0;
            if (b->f == MC_FILTER_TRILINEAR) {
                for (; k + 4 <= b->nlayers; k += 4) {
                    float bz[4], by[4], bx[4], v4[4];
                    for (int t = 0; t < 4; t++) {
                        bz[t] = pz; by[t] = py; bx[t] = px;
                        pz += sz_; py += sy_; px += sx_;
                    }
                    mc_s_tri4(s, bz, by, bx, v4);
                    for (int t = 0; t < 4; t++)
                        o[layer * (k + t)] = to_u8(v4[t]);
                }
            }
            for (; k < b->nlayers; k++) {
                o[layer * k] = to_u8(mc_s_sample(s, pz, py, px, b->f));
                pz += sz_; py += sy_; px += sx_;
            }
        }
    }
    free(row);
    mc_sampler_free(s);
    return NULL;
}

int mc_sample_quad_volume(const mc_sample_src *src, const mc_quad *q,
                          float x0, float y0, float step, int w, int h,
                          float t0, float dt, int nlayers,
                          mc_filter f, uint8_t *out, int nthreads) {
    if (!src || !q || !q->grid || q->gw < 1 || q->gh < 1 ||
        !out || w <= 0 || h <= 0 || nlayers <= 0) return -1;
    if (nthreads <= 0) {
        long nc = sysconf(_SC_NPROCESSORS_ONLN);
        nthreads = nc > 0 ? (int)nc : 1;
    }
    if (nthreads > 16) nthreads = 16;
    if (nthreads > h)  nthreads = h;
    pthread_t th[16];
    qvol_band_t bands[16];
    int per = (h + nthreads - 1) / nthreads;
    int nb = 0;
    for (int t = 0; t < nthreads; t++) {
        int r0 = t * per, r1 = r0 + per > h ? h : r0 + per;
        if (r0 >= r1) break;
        bands[nb] = (qvol_band_t){ src, q, x0, y0, step, t0, dt,
                                   w, h, nlayers, f, out, r0, r1 };
        if (nthreads == 1 ||
            pthread_create(&th[nb], NULL, qvol_band_main, &bands[nb]) != 0) {
            qvol_band_main(&bands[nb]);
            continue;
        }
        nb++;
    }
    for (int t = 0; t < nb; t++) pthread_join(th[t], NULL);
    return 0;
}

int mc_sample_box(const mc_sample_src *src,
                  const float origin[3], const float du[3],
                  const float dv[3], const float dw[3],
                  int w, int h, int d,
                  mc_filter f, uint8_t *out, int nthreads) {
    if (!src || !origin || !du || !dv || !dw || !out ||
        w <= 0 || h <= 0 || d <= 0) return -1;
    // each depth slice is a plane render with the layer offset folded into
    // the origin; comp NONE so no normals are needed
    mc_render_params p = { .filter = f, .comp = MC_COMP_NONE };
    for (int k = 0; k < d; k++) {
        mc_plane pl;
        for (int c = 0; c < 3; c++) {
            // mc_plane_gen centers the image; sample with corner semantics
            pl.origin[c] = origin[c] + (float)k * dw[c] +
                           ((float)w * 0.5f) * du[c] + ((float)h * 0.5f) * dv[c];
            pl.normal[c] = 0.0f;
            pl.u[c] = du[c];
            pl.v[c] = dv[c];
        }
        if (mc_render_plane(src, &pl, w, h, 1.0f, &p,
                            out + (size_t)k * w * h, nthreads) != 0)
            return -1;
    }
    return 0;
}

// ============================================================================
// mc_zarr — zarr v3-sharded-c3d + v2-flat reader (dep: zstd)
// ============================================================================
#include <zstd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <zstd.h>

// ---------------------------------------------------------------------------
// tiny JSON scraping — enough for zarr.json / .zarray (no general parser).
// ---------------------------------------------------------------------------

// First integer array of `n` ints after the literal `"key"`. -1 on miss.
static int json_int_array(const char *j, const char *key, long out[], int n) {
    const char *p = strstr(j, key);
    if (!p) return -1;
    p = strchr(p, '[');
    if (!p) return -1;
    ++p;
    for (int i = 0; i < n; ++i) {
        char *end;
        out[i] = strtol(p, &end, 10);
        if (end == p) return -1;
        p = end;
        while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') ++p;
    }
    return 0;
}

// Is `needle` present anywhere in `j` (substring)?
static int json_has(const char *j, const char *needle) {
    return strstr(j, needle) != NULL;
}

// ---------------------------------------------------------------------------
// blosc1 decode (shuffle=0, typesize=1) — lifted from tools/mc_fetch.c.
// header: ver verlz flags typesize nbytes(4) blocksize(4) cbytes(4), LE.
// ---------------------------------------------------------------------------
static uint8_t *blosc_decode(const uint8_t *src, size_t srclen, size_t *out_len) {
    if (srclen < 16) return NULL;
    uint8_t flags = src[2];
    uint32_t nbytes, blocksize, cbytes;
    memcpy(&nbytes, src + 4, 4);
    memcpy(&blocksize, src + 8, 4);
    memcpy(&cbytes, src + 12, 4);
    if (cbytes > srclen || !nbytes) return NULL;
    if (flags & 0x1 || flags & 0x4) { fprintf(stderr, "mc_zarr: blosc shuffle unsupported\n"); return NULL; }
    uint8_t *out = malloc(nbytes);
    if (!out) return NULL;
    if (flags & 0x2) {                              // memcpyed: raw payload follows header
        if (16 + (size_t)nbytes > srclen) { free(out); return NULL; }
        memcpy(out, src + 16, nbytes);
        *out_len = nbytes;
        return out;
    }
    uint32_t nblocks = (nbytes + blocksize - 1) / blocksize;
    const uint8_t *bstarts = src + 16;
    if (16 + (size_t)nblocks * 4 > srclen) { free(out); return NULL; }
    size_t off = 0;
    for (uint32_t b = 0; b < nblocks; ++b) {
        uint32_t bs;
        memcpy(&bs, bstarts + (size_t)b * 4, 4);
        uint32_t neblock = (b == nblocks - 1) ? nbytes - b * blocksize : blocksize;
        if ((size_t)bs + 4 > srclen) { free(out); return NULL; }
        int32_t cb;
        memcpy(&cb, src + bs, 4);
        const uint8_t *payload = src + bs + 4;
        if (cb <= 0 || (size_t)bs + 4 + (size_t)cb > srclen) { free(out); return NULL; }
        if ((uint32_t)cb == neblock) {
            memcpy(out + off, payload, neblock);    // stored uncompressed
        } else {
            size_t got = ZSTD_decompress(out + off, neblock, payload, (size_t)cb);
            if (ZSTD_isError(got) || got != neblock) { free(out); return NULL; }
        }
        off += neblock;
    }
    *out_len = nbytes;
    return out;
}

// ---------------------------------------------------------------------------
// mc_zarr handle
// ---------------------------------------------------------------------------
enum { ZV3 = 3, ZV2 = 2 };

// Cached shard index footer: the (offset,len) table is immutable per shard, so
// read the 64KB footer ONCE and reuse it for all of a shard's inner chunks.
// Without this, fetching a shard's 4096 chunks one-at-a-time re-reads 64KB *4096.
#define MC_FOOTER_CACHE 64
typedef struct {
    uint64_t shard_id;     // (sz<<40)|(sy<<20)|sx, ~0 = empty slot
    uint8_t *idx;          // malloc'd footer bytes (n_inner*16)
    uint64_t lru;          // last-use tick
} footer_ent;

struct mc_zarr {
    mc_zarr_read_fn read;
    void *ud;
    int version;            // ZV3 | ZV2
    int shape[3];           // voxels (z,y,x)
    int shard_edge;         // v3: chunk_grid chunk_shape; v2: == inner_edge
    int inner_edge;         // v3: sharding inner chunk; v2: .zarray chunks
    int inner_grid[3];      // ceil(shape/inner_edge) per axis (z,y,x)
    int per;                // inner chunks per shard axis = shard_edge/inner_edge
    char codec[16];         // "c3d" | "blosc" | "raw"
    char sep;               // v2 dimension separator ('.' default, or '/')

    pthread_mutex_t fmu;    // guards the footer cache
    footer_ent fcache[MC_FOOTER_CACHE];
    uint64_t ftick;
};

// fetch a whole object by key; returns malloc'd buf or NULL (sets *len).
static uint8_t *fetch_all(mc_zarr *z, const char *key, size_t *len) {
    uint8_t *buf = NULL;
    size_t n = 0;
    if (z->read(z->ud, key, 0, 0, &buf, &n) < 0) { *len = 0; return NULL; }
    *len = n;
    return buf;
}

mc_zarr *mc_zarr_open(mc_zarr_read_fn read, void *ud) {
    if (!read) return NULL;
    size_t jl = 0;
    uint8_t *jb = NULL;
    int v3 = 1;
    if (read(ud, "zarr.json", 0, 0, &jb, &jl) < 0 || !jb || !jl) {
        free(jb);
        jb = NULL;
        jl = 0;
        v3 = 0;
        if (read(ud, ".zarray", 0, 0, &jb, &jl) < 0 || !jb || !jl) { free(jb); return NULL; }
    }
    char *j = malloc(jl + 1);
    if (!j) { free(jb); return NULL; }
    memcpy(j, jb, jl);
    j[jl] = 0;
    free(jb);

    mc_zarr *z = calloc(1, sizeof *z);
    if (!z) { free(j); return NULL; }
    pthread_mutex_init(&z->fmu, NULL);
    z->read = read;
    z->ud = ud;

    long shp[3];
    if (json_int_array(j, "\"shape\"", shp, 3) != 0) { free(j); free(z); return NULL; }
    z->shape[0] = (int)shp[0];
    z->shape[1] = (int)shp[1];
    z->shape[2] = (int)shp[2];

    if (v3) {
        z->version = ZV3;
        if (!json_has(j, "sharding_indexed")) { free(j); free(z); return NULL; }
        // chunk_grid.chunk_shape = shard edge (first int array after "chunk_grid").
        const char *cg = strstr(j, "\"chunk_grid\"");
        long shard[3];
        if (!cg || json_int_array(cg, "\"chunk_shape\"", shard, 3) != 0) { free(j); free(z); return NULL; }
        // sharding_indexed configuration.chunk_shape = inner edge (after that codec).
        const char *sh = strstr(j, "sharding_indexed");
        // chunk_shape appears in the sharding config BEFORE the codec name in the
        // emitted json; search from the codecs array start instead.
        const char *cc = strstr(j, "\"codecs\"");
        long inner[3];
        if (!cc || json_int_array(cc, "\"chunk_shape\"", inner, 3) != 0) { free(j); free(z); return NULL; }
        (void)sh;
        z->shard_edge = (int)shard[0];
        z->inner_edge = (int)inner[0];
        // inner codec: these archives are always c3d.
        if (json_has(j, "\"c3d\"")) snprintf(z->codec, sizeof z->codec, "c3d");
        else { fprintf(stderr, "mc_zarr: v3 inner codec not c3d (unsupported)\n"); free(j); free(z); return NULL; }
    } else {
        z->version = ZV2;
        long ch[3];
        if (json_int_array(j, "\"chunks\"", ch, 3) != 0) { free(j); free(z); return NULL; }
        z->inner_edge = (int)ch[0];
        z->shard_edge = (int)ch[0];           // a v2 chunk is its own shard
        // compressor: null -> raw, else blosc (the standardized scroll zarrs).
        if (json_has(j, "\"compressor\": null") || json_has(j, "\"compressor\":null"))
            snprintf(z->codec, sizeof z->codec, "raw");
        else snprintf(z->codec, sizeof z->codec, "blosc");
        // dimension_separator default '.'
        z->sep = json_has(j, "\"dimension_separator\": \"/\"") ? '/' : '.';
    }

    free(j);
    if (z->inner_edge <= 0 || z->shard_edge <= 0 || z->shard_edge % z->inner_edge) {
        free(z);
        return NULL;
    }
    z->per = z->shard_edge / z->inner_edge;
    for (int d = 0; d < 3; ++d)
        z->inner_grid[d] = (z->shape[d] + z->inner_edge - 1) / z->inner_edge;
    return z;
}

void mc_zarr_free(mc_zarr *z) {
    if (!z) return;
    for (int i = 0; i < MC_FOOTER_CACHE; ++i) free(z->fcache[i].idx);
    pthread_mutex_destroy(&z->fmu);
    free(z);
}

void mc_zarr_shape(const mc_zarr *z, int *nz, int *ny, int *nx) {
    if (nz) *nz = z->shape[0];
    if (ny) *ny = z->shape[1];
    if (nx) *nx = z->shape[2];
}
int mc_zarr_inner_edge(const mc_zarr *z) { return z->inner_edge; }
int mc_zarr_shard_edge(const mc_zarr *z) { return z->shard_edge; }
const char *mc_zarr_inner_codec(const mc_zarr *z) { return z->codec; }
void mc_zarr_inner_grid(const mc_zarr *z, int *nz, int *ny, int *nx) {
    if (nz) *nz = z->inner_grid[0];
    if (ny) *ny = z->inner_grid[1];
    if (nx) *nx = z->inner_grid[2];
}

// ---------------------------------------------------------------------------
// keys
// ---------------------------------------------------------------------------

// v3 shard key for the shard containing global inner chunk (cz,cy,cx): "c/sz/sy/sx".
// v2 chunk key for inner chunk (cz,cy,cx): "cz<sep>cy<sep>cx".
static void chunk_key(const mc_zarr *z, int cz, int cy, int cx, char out[64]) {
    if (z->version == ZV3) {
        int sz = cz / z->per, sy = cy / z->per, sx = cx / z->per;
        snprintf(out, 64, "c/%d/%d/%d", sz, sy, sx);
    } else {
        snprintf(out, 64, "%d%c%d%c%d", cz, z->sep, cy, z->sep, cx);
    }
}

// ---------------------------------------------------------------------------
// v3 shard index: n_inner entries of (offset:u64, nbytes:u64) LE at shard start.
// missing == both == 0xFFFF...F. Linear order row-major, z slowest.
// ---------------------------------------------------------------------------
static int index_entry(const uint8_t *idx, size_t n_inner, size_t lin,
                       uint64_t *off, uint64_t *nb) {
    if (lin >= n_inner) return -1;
    memcpy(off, idx + lin * 16, 8);
    memcpy(nb, idx + lin * 16 + 8, 8);
    if (*off == ~(uint64_t)0 && *nb == ~(uint64_t)0) return 1;   // missing
    return 0;
}

// shard-relative linear inner index, row-major (z slowest, x fastest).
static size_t inner_linear(const mc_zarr *z, int cz, int cy, int cx) {
    int rz = cz % z->per, ry = cy % z->per, rx = cx % z->per;
    return ((size_t)rz * z->per + ry) * z->per + rx;
}

// Get the shard's index footer (n_inner*16 bytes), cached. Reads it once per
// shard via one ranged GET, then serves all the shard's chunks from RAM.
// Returns a borrowed pointer valid until the entry is evicted; copy out the
// (off,len) you need while you still need them (callers do this immediately).
// NULL if the shard is absent (all air) or on error.
static const uint8_t *footer_get(mc_zarr *z, int cz, int cy, int cx) {
    if (z->version != ZV3) return NULL;
    uint64_t sid = ((uint64_t)(cz / z->per) << 40) |
                   ((uint64_t)(cy / z->per) << 20) | (uint64_t)(cx / z->per);
    pthread_mutex_lock(&z->fmu);
    for (int i = 0; i < MC_FOOTER_CACHE; ++i)
        if (z->fcache[i].idx && z->fcache[i].shard_id == sid) {
            z->fcache[i].lru = ++z->ftick;
            const uint8_t *p = z->fcache[i].idx;
            pthread_mutex_unlock(&z->fmu);
            return p;
        }
    pthread_mutex_unlock(&z->fmu);

    // miss: fetch the footer (outside the lock — it's one ranged GET).
    char key[64];
    chunk_key(z, cz, cy, cx, key);
    size_t n_inner = (size_t)z->per * z->per * z->per;
    size_t idx_bytes = n_inner * 16;
    uint8_t *idx = NULL;
    size_t got = 0;
    if (z->read(z->ud, key, 0, idx_bytes, &idx, &got) < 0 || !idx || got < idx_bytes) {
        free(idx);
        return NULL;
    }

    pthread_mutex_lock(&z->fmu);
    // re-check (another thread may have inserted it); if so, drop ours.
    int victim = 0;
    uint64_t oldest = ~0ull;
    for (int i = 0; i < MC_FOOTER_CACHE; ++i) {
        if (z->fcache[i].idx && z->fcache[i].shard_id == sid) {
            free(idx);
            z->fcache[i].lru = ++z->ftick;
            const uint8_t *p = z->fcache[i].idx;
            pthread_mutex_unlock(&z->fmu);
            return p;
        }
        if (!z->fcache[i].idx) { victim = i; oldest = 0; }
        else if (z->fcache[i].lru < oldest) { oldest = z->fcache[i].lru; victim = i; }
    }
    free(z->fcache[victim].idx);
    z->fcache[victim].idx = idx;
    z->fcache[victim].shard_id = sid;
    z->fcache[victim].lru = ++z->ftick;
    pthread_mutex_unlock(&z->fmu);
    return idx;
}

int mc_zarr_shard_all_air(mc_zarr *z, int cz, int cy, int cx) {
    if (z->version != ZV3) {
        // v2: "all air" == the single chunk object is absent.
        char key[64];
        chunk_key(z, cz, cy, cx, key);
        uint8_t *b = NULL;
        size_t n = 0;
        if (z->read(z->ud, key, 0, 1, &b, &n) < 0) return -1;
        free(b);
        return n == 0 ? 1 : 0;
    }
    size_t n_inner = (size_t)z->per * z->per * z->per;
    const uint8_t *idx = footer_get(z, cz, cy, cx);
    if (!idx) return 1;   // absent shard = air
    for (size_t i = 0; i < n_inner; ++i) {
        uint64_t off, nb;
        if (index_entry(idx, n_inner, i, &off, &nb) == 0) return 0;
    }
    return 1;
}

// decode a v2 chunk blob (blosc/raw) into a fresh dense buffer.
static uint8_t *v2_decode(const mc_zarr *z, uint8_t *blob, size_t blen, size_t *out_len) {
    if (strcmp(z->codec, "raw") == 0) { *out_len = blen; return blob; }   // takes ownership
    size_t dl = 0;
    uint8_t *dense = blosc_decode(blob, blen, &dl);
    free(blob);
    if (!dense) return NULL;
    *out_len = dl;
    return dense;
}

int mc_zarr_read_inner(mc_zarr *z, int cz, int cy, int cx, uint8_t **raw, size_t *len) {
    *raw = NULL;
    *len = 0;
    char key[64];
    chunk_key(z, cz, cy, cx, key);

    if (z->version == ZV2) {
        size_t blen = 0;
        uint8_t *blob = fetch_all(z, key, &blen);
        if (!blob || !blen) { free(blob); return 1; }     // absent = air
        size_t dl = 0;
        uint8_t *dense = v2_decode(z, blob, blen, &dl);
        if (!dense) return -1;
        *raw = dense;
        *len = dl;
        return 0;
    }

    // v3: get the (cached) index footer, then the one inner chunk's payload range.
    size_t n_inner = (size_t)z->per * z->per * z->per;
    const uint8_t *idx = footer_get(z, cz, cy, cx);
    if (!idx) return 1;                                      // absent shard = air
    size_t lin = inner_linear(z, cz, cy, cx);
    uint64_t off, nb;
    int st = index_entry(idx, n_inner, lin, &off, &nb);
    if (st != 0) return st < 0 ? -1 : 1;                     // missing or oob -> air
    uint8_t *payload = NULL;
    size_t plen = 0;
    if (z->read(z->ud, key, off, nb, &payload, &plen) < 0) { return -1; }
    if (!payload || plen < nb) { free(payload); return -1; }
    *raw = payload;
    *len = nb;
    return 0;   // c3d raw bytes — caller decodes.
}

int mc_zarr_read_shard(mc_zarr *z, int cz, int cy, int cx,
                       mc_zarr_chunk_fn sink, void *sink_ud) {
    if (z->version == ZV2) {
        // a v2 "shard" is one chunk.
        uint8_t *dense = NULL;
        size_t dl = 0;
        int st = mc_zarr_read_inner(z, cz, cy, cx, &dense, &dl);
        if (st < 0) return -1;
        if (st == 0) { sink(sink_ud, cz, cy, cx, dense, dl); free(dense); }
        return 0;
    }

    // v3: read the index footer ONCE, then range-GET each present inner chunk
    // individually (no whole-shard buffering). Each chunk is sunk + freed before
    // the next, so RAM stays at one chunk and disk grows per-chunk.
    char key[64];
    chunk_key(z, cz, cy, cx, key);
    size_t n_inner = (size_t)z->per * z->per * z->per;
    const uint8_t *idx = footer_get(z, cz, cy, cx);
    if (!idx) return 0;   // absent shard = all air
    int sz0 = (cz / z->per) * z->per, sy0 = (cy / z->per) * z->per, sx0 = (cx / z->per) * z->per;
    for (int iz = 0; iz < z->per; ++iz)
        for (int iy = 0; iy < z->per; ++iy)
            for (int ix = 0; ix < z->per; ++ix) {
                size_t lin = ((size_t)iz * z->per + iy) * z->per + ix;
                uint64_t off, nb;
                if (index_entry(idx, n_inner, lin, &off, &nb) != 0) continue;
                int gz = sz0 + iz, gy = sy0 + iy, gx = sx0 + ix;
                if (gz >= z->inner_grid[0] || gy >= z->inner_grid[1] || gx >= z->inner_grid[2])
                    continue;
                uint8_t *payload = NULL;
                size_t plen = 0;
                if (z->read(z->ud, key, off, nb, &payload, &plen) < 0) return -1;
                if (payload && plen >= nb)
                    sink(sink_ud, gz, gy, gx, payload, (size_t)nb);   // c3d raw bytes
                free(payload);
            }
    return 0;
}

int mc_zarr_shard_index(mc_zarr *z, int cz, int cy, int cx,
                        char key_out[64], mc_zarr_range **out, int *n) {
    *out = NULL;
    *n = 0;
    if (z->version == ZV2) {
        // a v2 "shard" is one chunk object; whole-object fetch (off/len = 0).
        chunk_key(z, cz, cy, cx, key_out);
        mc_zarr_range *r = malloc(sizeof *r);
        if (!r) return -1;
        r[0] = (mc_zarr_range){cz, cy, cx, 0, 0};
        *out = r;
        *n = 1;
        return 0;
    }
    chunk_key(z, cz, cy, cx, key_out);
    size_t n_inner = (size_t)z->per * z->per * z->per;
    const uint8_t *idx = footer_get(z, cz, cy, cx);
    if (!idx) return 0;                                     // absent shard = all air
    mc_zarr_range *arr = malloc(n_inner * sizeof *arr);
    if (!arr) return -1;
    int sz0 = (cz / z->per) * z->per, sy0 = (cy / z->per) * z->per, sx0 = (cx / z->per) * z->per;
    int cnt = 0;
    for (int iz = 0; iz < z->per; ++iz)
        for (int iy = 0; iy < z->per; ++iy)
            for (int ix = 0; ix < z->per; ++ix) {
                size_t lin = ((size_t)iz * z->per + iy) * z->per + ix;
                uint64_t off, nb;
                if (index_entry(idx, n_inner, lin, &off, &nb) != 0) continue;
                int gz = sz0 + iz, gy = sy0 + iy, gx = sx0 + ix;
                if (gz >= z->inner_grid[0] || gy >= z->inner_grid[1] || gx >= z->inner_grid[2])
                    continue;
                arr[cnt++] = (mc_zarr_range){gz, gy, gx, off, nb};
            }
    *out = arr;
    *n = cnt;
    return 0;
}

// ============================================================================
// mc_s3 — s3://-backed mc_reader glue (dep: libs3)
// ============================================================================
#include "libs3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct mc_s3 {
    s3_client *cl;
    char *url;
    mc_reader *r;
};

static int s3_read_cb(void *ud, uint64_t off, uint32_t len, uint8_t *dst){
    mc_s3 *s=ud;
    s3_response resp={0};
    if(s3_get_range(s->cl,s->url,off,len,&resp)!=S3_OK || !s3_response_ok(&resp)){
        s3_response_free(&resp);
        return -1;
    }
    int rc=-1;
    if(resp.status==206 && resp.body_len>=len){
        memcpy(dst,resp.body,len); rc=0;          // proper ranged reply
    } else if(resp.status==200 && resp.body_len>=off+len){
        memcpy(dst,resp.body+off,len); rc=0;      // server ignored Range and
    }                                             // sent the whole object
    s3_response_free(&resp);
    return rc;
}

mc_s3 *mc_s3_open(const char *url){
    if(!url) return NULL;
    mc_s3 *s=calloc(1,sizeof *s);
    s3_config cfg={0};
    s->cl=s3_client_new(&cfg);
    if(!s->cl){ free(s); return NULL; }
    s->url=strdup(url);
    s3_response head={0};
    uint64_t total=0;
    if(s3_head(s->cl,url,&head)==S3_OK && s3_response_ok(&head))
        total=head.content_length;
    s3_response_free(&head);
    if(!total){ mc_s3_close(s); return NULL; }
    s->r=mc_open_streaming(s3_read_cb,s,total);
    if(!s->r){ mc_s3_close(s); return NULL; }
    mc_reader_set_partial_fetch(s->r,1);
    return s;
}
mc_reader *mc_s3_reader(mc_s3 *s){ return s?s->r:NULL; }
void mc_s3_close(mc_s3 *s){
    if(!s) return;
    if(s->r) mc_close(s->r);
    if(s->cl) s3_client_free(s->cl);
    free(s->url);
    free(s);
}

// ============================================================================
// mc_volume — remote zarr -> local .mca (deps: mc_zarr, libs3, c3d)
// ============================================================================
#include "c3d.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#define MAXLOD 8
#define CHUNK 256
#define PER (CHUNK / BLK)   // 16 blocks per chunk axis

static void *decoder_main(void *ud);
static void *dl_main(void *ud);

// portable thread naming: macOS names the calling thread (1 arg), glibc
// takes (thread, name)
static void mc_thread_setname(const char *name) {
#if defined(__APPLE__)
    pthread_setname_np(name);
#elif defined(__linux__)
    pthread_setname_np(pthread_self(), name);
#else
    (void)name;
#endif
}
static const uint8_t *zero256(void);   // shared 32-aligned 256^3 zero buffer

// ---- timing log (MCV_LOG=1 to enable) -------------------------------------
static int g_log = -1;
static double mcv_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;   // ms
}
#define MCVLOG(...) do { \
    if (g_log < 0) g_log = getenv("MCV_LOG") ? 1 : 0; \
    if (g_log) { fprintf(stderr, "[mcv %10.1f] ", mcv_now()); \
                 fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); fflush(stderr); } \
} while (0)

// ---------------------------------------------------------------------------
// per-level source (one mc_zarr + its S3 key prefix)
// ---------------------------------------------------------------------------
typedef struct {
    mc_volume *vol;     // back-pointer (for the shared s3 client + net counter)
    char prefix[1024];  // e.g. "s3://bucket/root/0" (no trailing slash)
    mc_zarr *z;
} level_t;

struct mc_volume {
    s3_client *s3;             // NULL for a local-filesystem source
    int local;                 // 1 => root is a local dir, read via file_read
    char root[1024];           // s3://bucket/root, or /local/dir (no trailing slash)
    int nlods;
    level_t lv[MAXLOD];
    mc_archive *arc;           // ONE archive, all LODs
    mc_cache *cache;
    float quality;

    atomic_uint_fast64_t net_bytes;

    pthread_mutex_t mu;        // guards the decode queue + request stack
    pthread_cond_t cv;         // request-stack not-empty (wakes download threads)

    // Decode pipeline: download threads enqueue raw payloads here; a pool of
    // decode workers drains them (decode -> re-encode -> append). This keeps the
    // network saturated (downloaders never wait on CPU) and CPU saturated
    // (decoders run in parallel), instead of serializing download+decode.
    pthread_t decoders[32];
    int ndecoders;
    struct decode_item *dq;    // bounded ring of pending decode items
    int dq_cap, dq_head, dq_tail;
    pthread_cond_t dq_ne;      // not-empty (wake a decoder)
    pthread_cond_t dq_nf;      // not-full  (wake a blocked producer)
    int stop;

    // Interactive download-request stack (LIFO): a render miss pushes "fetch the
    // shard around region R". Download threads pop the NEWEST request (current
    // view) first; when full, the OLDEST (stalest, camera moved on) is dropped.
    uint64_t *reqstk;          // region keys
    int rs_cap, rs_n;
    pthread_t dlthreads[16];
    int ndl;

    mc_volume_ready_fn ready_cb;   // fired when a region becomes serveable
    void *ready_ud;
};

// One unit of decode work: the sub^3 cube of source chunks covering one 256^3
// region. For c3d (sub=1) nsub==1; for v2 (sub=2) up to 8. Owns the raw bytes.
typedef struct decode_item {
    int lod, rz, ry, rx;       // target 256^3 region coords
    int sub;                   // 1 (c3d) or 2 (v2)
    int nsub;                  // number of valid sub-chunks
    int oz[8], oy[8], ox[8];   // sub-chunk voxel offsets within the region
    uint8_t *raw[8];           // owned compressed bytes (freed by the decoder)
    size_t rlen[8];
} decode_item;

// ---------------------------------------------------------------------------
// s3 byte source for mc_zarr (prepends the level prefix to the object key)
// ---------------------------------------------------------------------------
static int s3_read(void *ud, const char *key, uint64_t off, uint64_t len,
                   uint8_t **out, size_t *out_len) {
    level_t *lv = ud;
    char url[1280];
    snprintf(url, sizeof url, "%s/%s", lv->prefix, key);
    s3_response resp = {0};
    s3_status st;
    if (len == 0) st = s3_get(lv->vol->s3, url, &resp);
    else          st = s3_get_range(lv->vol->s3, url, off, len, &resp);
    if (st != S3_OK) { s3_response_free(&resp); *out = NULL; *out_len = 0; return -1; }
    if (s3_response_not_found(&resp)) { s3_response_free(&resp); *out = NULL; *out_len = 0; return 0; }
    if (!s3_response_ok(&resp)) { s3_response_free(&resp); *out = NULL; *out_len = 0; return -1; }
    // honor a server that ignored Range and sent the whole object.
    const uint8_t *src = resp.body;
    size_t n = resp.body_len;
    if (len != 0 && resp.status == 200 && n >= off + len) { src += off; n = len; }
    uint8_t *buf = malloc(n ? n : 1);
    if (!buf) { s3_response_free(&resp); *out = NULL; *out_len = 0; return -1; }
    memcpy(buf, src, n);
    s3_response_free(&resp);
    atomic_fetch_add_explicit(&lv->vol->net_bytes, n, memory_order_relaxed);
    *out = buf;
    *out_len = n;
    return 0;
}

// Local-filesystem source: mirror of s3_read but reads "<prefix>/<key>" from
// disk. Same contract: return 0 with *out=NULL on a missing key (so the level
// probe / air detection behaves like a 404), <0 on real I/O error, 0 with a
// malloc'd buffer otherwise. `off`/`len` honor ranged reads (footer/index).
static int file_read(void *ud, const char *key, uint64_t off, uint64_t len,
                     uint8_t **out, size_t *out_len) {
    level_t *lv = ud;
    char path[1280];
    snprintf(path, sizeof path, "%s/%s", lv->prefix, key);
    *out = NULL; *out_len = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;                              // missing key == 404
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long fsz = ftell(f);
    if (fsz < 0) { fclose(f); return -1; }
    uint64_t start = off;
    uint64_t want  = (len == 0) ? (uint64_t)fsz : len;
    if (start > (uint64_t)fsz) { fclose(f); return 0; }      // past EOF
    if (start + want > (uint64_t)fsz) want = (uint64_t)fsz - start;
    if (fseek(f, (long)start, SEEK_SET) != 0) { fclose(f); return -1; }
    uint8_t *buf = malloc(want ? want : 1);
    if (!buf) { fclose(f); return -1; }
    size_t got = want ? fread(buf, 1, want, f) : 0;
    fclose(f);
    if (got != want) { free(buf); return -1; }
    atomic_fetch_add_explicit(&lv->vol->net_bytes, got, memory_order_relaxed);
    *out = buf; *out_len = got;
    return 0;
}

// pack a region (lod,cz,cy,cx) into a 64-bit key.
static uint64_t rkey(int lod, int cz, int cy, int cx) {
    return ((uint64_t)(lod & 7) << 60) | ((uint64_t)(cz & 0xFFFFF) << 40) |
           ((uint64_t)(cy & 0xFFFFF) << 20) | (uint64_t)(cx & 0xFFFFF);
}

// ---------------------------------------------------------------------------
// transcode one 256^3 region (cz,cy,cx) of lod into the .mca. caller ensures
// single-flight. returns 1 transcoded data, 0 air, <0 error.
// ---------------------------------------------------------------------------
// decode one source inner-chunk's raw bytes into `dst` (edge^3, edge = source
// inner_edge: 256 for c3d, 128 for v2). dst need not be 32-aligned for v2; c3d
// needs a 32-aligned 256^3 (the v3 case always passes the region buffer).
static void decode_inner(const char *codec, const uint8_t *raw, size_t rlen,
                         uint8_t *dst, int edge) {
    size_t vox = (size_t)edge * edge * edge;
    if (strcmp(codec, "c3d") == 0) {
        c3d_decoder *d = c3d_decoder_new();
        c3d_decoder_set_denoise(d, false);
        c3d_decoder_chunk_decode(d, raw, rlen, dst);   // c3d edge is always 256
        c3d_decoder_free(d);
    } else {                                            // blosc/raw: already dense u8
        if (rlen >= vox) memcpy(dst, raw, vox);
        else { memset(dst, 0, vox); memcpy(dst, raw, rlen); }
    }
}

// blit a src (edge^3) into the 256^3 region buffer at sub-offset (oz,oy,ox) voxels.
static void blit_sub(uint8_t *region, const uint8_t *src, int edge,
                     int oz, int oy, int ox) {
    for (int z = 0; z < edge; ++z)
        for (int y = 0; y < edge; ++y)
            memcpy(region + (((size_t)(oz + z) * CHUNK + (oy + y)) * CHUNK + ox),
                   src + ((size_t)z * edge + y) * edge, (size_t)edge);
}

// Decode one item (the sub^3 cube for a region) -> assemble 256^3 -> append.
// Frees the item's raw buffers. Runs on a decode-pool thread (off the download
// thread). The c3d decode + mc re-encode are the CPU cost we keep off the net.
static void decode_one(mc_volume *v, decode_item *it) {
    const char *codec = mc_zarr_inner_codec(v->lv[it->lod].z);
    const int edge = CHUNK / it->sub;
    if (it->nsub == 0) {                               // all air -> ZERO
        mc_archive_append_chunk_raw(v->arc, it->lod, it->rz, it->ry, it->rx, zero256());
        return;
    }
    uint8_t *dense = NULL;
    if (posix_memalign((void **)&dense, 64, (size_t)CHUNK * CHUNK * CHUNK)) goto done;
    double t_dec0 = mcv_now();
    if (it->sub == 1) {                                // c3d: chunk == region
        decode_inner(codec, it->raw[0], it->rlen[0], dense, CHUNK);
    } else {                                           // v2: blit the cube
        memset(dense, 0, (size_t)CHUNK * CHUNK * CHUNK);
        uint8_t *tile = malloc((size_t)edge * edge * edge);
        if (tile) {
            for (int k = 0; k < it->nsub; ++k) {
                decode_inner(codec, it->raw[k], it->rlen[k], tile, edge);
                blit_sub(dense, tile, edge, it->oz[k], it->oy[k], it->ox[k]);
            }
            free(tile);
        }
    }
    double t_enc0 = mcv_now();
    mc_archive_append_chunk_raw(v->arc, it->lod, it->rz, it->ry, it->rx, dense);
    double t_end = mcv_now();
    MCVLOG("decoded   lod%d region(%d,%d,%d) codec=%s decode=%.0fms encode=%.0fms",
           it->lod, it->rz, it->ry, it->rx, codec,
           t_enc0 - t_dec0, t_end - t_enc0);
    free(dense);
done:
    for (int k = 0; k < it->nsub; ++k) free(it->raw[k]);
}

// Decode-pool worker: drain decode items, decode off the download thread.
static void *decoder_main(void *ud) {
    mc_volume *v = ud;
    mc_thread_setname("mc-decode");        // distinguish in profilers
    for (;;) {
        pthread_mutex_lock(&v->mu);
        while (v->dq_head == v->dq_tail && !v->stop) pthread_cond_wait(&v->dq_ne, &v->mu);
        if (v->stop && v->dq_head == v->dq_tail) { pthread_mutex_unlock(&v->mu); return NULL; }
        decode_item it = v->dq[v->dq_head];
        v->dq_head = (v->dq_head + 1) % v->dq_cap;
        pthread_cond_signal(&v->dq_nf);                // a slot freed
        pthread_mutex_unlock(&v->mu);
        decode_one(v, &it);
        if (v->ready_cb) v->ready_cb(v->ready_ud);     // region became serveable
    }
}

// Producer: push a decode item, BLOCKING if the queue is full (backpressure ->
// bounded RAM; the download thread waits for decoders to catch up). Takes
// ownership of the item's raw buffers.
static void decode_push(mc_volume *v, const decode_item *it) {
    pthread_mutex_lock(&v->mu);
    int next = (v->dq_tail + 1) % v->dq_cap;
    int blocked = (next == v->dq_head);
    while (next == v->dq_head && !v->stop) pthread_cond_wait(&v->dq_nf, &v->mu);
    if (v->stop) { pthread_mutex_unlock(&v->mu);
        for (int k = 0; k < it->nsub; ++k) free(it->raw[k]); return; }
    v->dq[v->dq_tail] = *it;
    v->dq_tail = next;
    int depth = (v->dq_tail - v->dq_head + v->dq_cap) % v->dq_cap;
    pthread_cond_signal(&v->dq_ne);
    pthread_mutex_unlock(&v->mu);
    if (blocked) MCVLOG("decode_q  FULL (backpressure: decoders behind) depth=%d", depth);
}

// unpack a region key.
static void runpack(uint64_t k, int *lod, int *cz, int *cy, int *cx) {
    *lod = (int)((k >> 60) & 7);
    *cz = (int)((k >> 40) & 0xFFFFF);
    *cy = (int)((k >> 20) & 0xFFFFF);
    *cx = (int)(k & 0xFFFFF);
}

// Push an interactive download request (region key) onto the LIFO stack. Newest
// on top. If full, drop the BOTTOM (stalest). Deduped against the stack. Wakes a
// download thread. (cv doubles as the stack's not-empty signal.)
static void req_push(mc_volume *v, int lod, int cz, int cy, int cx) {
    uint64_t key = rkey(lod, cz, cy, cx);
    pthread_mutex_lock(&v->mu);
    for (int i = 0; i < v->rs_n; ++i)
        if (v->reqstk[i] == key) { pthread_mutex_unlock(&v->mu); return; }   // already queued
    if (v->rs_n == v->rs_cap) {                         // full -> drop bottom
        memmove(&v->reqstk[0], &v->reqstk[1], (size_t)(v->rs_cap - 1) * sizeof(uint64_t));
        v->rs_n--;
    }
    v->reqstk[v->rs_n++] = key;                         // push on top
    MCVLOG("req_push  lod%d region(%d,%d,%d) stack_depth=%d", lod, cz, cy, cx, v->rs_n);
    pthread_cond_signal(&v->cv);
    pthread_mutex_unlock(&v->mu);
}

// Download thread: pop the newest request, download its shard (-> decode queue).
static void *dl_main(void *ud) {
    mc_volume *v = ud;
    mc_thread_setname("mc-download");      // distinguish in profilers
    for (;;) {
        pthread_mutex_lock(&v->mu);
        while (v->rs_n == 0 && !v->stop) pthread_cond_wait(&v->cv, &v->mu);
        if (v->stop && v->rs_n == 0) { pthread_mutex_unlock(&v->mu); return NULL; }
        uint64_t key = v->reqstk[--v->rs_n];           // pop top (newest)
        pthread_mutex_unlock(&v->mu);
        int lod, cz, cy, cx;
        runpack(key, &lod, &cz, &cy, &cx);             // region coords
        MCVLOG("dl_pop    lod%d region(%d,%d,%d) -> download shard", lod, cz, cy, cx);
        const int sub = CHUNK / mc_zarr_inner_edge(v->lv[lod].z);
        mc_volume_prefetch_shard(v, lod, cz * sub, cy * sub, cx * sub);  // source coord
    }
}

// Blocking fill of one region (get_block / CLI): download its shard synchronously
// through the same decode queue, then wait for that region's coverage to resolve.
static mc_cover ensure_region(mc_volume *v, int lod, int cz, int cy, int cx) {
    mc_cover cov = mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx);
    if (cov != MC_ABSENT) return cov;
    const int sub = CHUNK / mc_zarr_inner_edge(v->lv[lod].z);
    mc_volume_prefetch_shard(v, lod, cz * sub, cy * sub, cx * sub);   // pushes to decode queue
    // wait for the decoders to drain enough that this region is covered.
    for (int spin = 0; spin < 100000; ++spin) {
        cov = mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx);
        if (cov != MC_ABSENT) return cov;
        struct timespec ts = {0, 1000000};             // 1ms
        nanosleep(&ts, NULL);
    }
    return mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx);
}

// ---------------------------------------------------------------------------
// shared 32-aligned zero region (for air)
// ---------------------------------------------------------------------------
static uint8_t *g_zero = NULL;
static void init_zero(void) {
    if (posix_memalign((void **)&g_zero, 64, (size_t)CHUNK * CHUNK * CHUNK) == 0)
        memset(g_zero, 0, (size_t)CHUNK * CHUNK * CHUNK);
}
static const uint8_t *zero256(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, init_zero);
    return g_zero;
}

// ===========================================================================
// open / discovery
// ===========================================================================

// strip a trailing '/'.
static void rstrip_slash(char *s) {
    size_t n = strlen(s);
    while (n && s[n - 1] == '/') s[--n] = 0;
}

mc_volume *mc_volume_open(const char *url, const char *cache_dir,
                          size_t cache_bytes, float quality) {
    if (!url || !cache_dir) return NULL;
    mc_volume *v = calloc(1, sizeof *v);
    if (!v) return NULL;
    v->quality = quality;
    atomic_init(&v->net_bytes, 0);
    pthread_mutex_init(&v->mu, NULL);
    pthread_cond_init(&v->cv, NULL);

    // Local vs remote: a URL with a "scheme://" is remote (s3/https), otherwise
    // `url` is a local filesystem directory read via file_read (no S3 client).
    v->local = (strstr(url, "://") == NULL);
    if (!v->local) {
        // s3 client: full credential resolution (profile/IMDS/SSO/env), else anonymous.
        s3_config cfg = {0};
        s3_credentials creds = {0};
        if (s3_credentials_load(NULL, &creds) == S3_OK) cfg.creds = creds;
        v->s3 = s3_client_new(&cfg);
        s3_credentials_free(&creds);
        if (!v->s3) { free(v); return NULL; }
    }
    mc_zarr_read_fn read_cb = v->local ? file_read : s3_read;

    snprintf(v->root, sizeof v->root, "%s", url);
    rstrip_slash(v->root);

    // discover levels: probe "<root>/<i>/zarr.json" for i=0.. until a gap.
    for (int i = 0; i < MAXLOD; ++i) {
        level_t *lv = &v->lv[i];
        lv->vol = v;
        snprintf(lv->prefix, sizeof lv->prefix, "%s/%d", v->root, i);
        mc_zarr *z = mc_zarr_open(read_cb, lv);
        if (!z) { lv->prefix[0] = 0; break; }
        lv->z = z;
        v->nlods = i + 1;
    }
    if (v->nlods == 0) { if (v->s3) s3_client_free(v->s3); free(v); return NULL; }

    // local .mca dims from LOD0 shape (padded to 256 internally by mc).
    int nz, ny, nx;
    mc_zarr_shape(v->lv[0].z, &nz, &ny, &nx);
    char path[2048];
    // archive name from the last path component of the root.
    const char *base = strrchr(v->root, '/');
    base = base ? base + 1 : v->root;
    snprintf(path, sizeof path, "%s/%s.mca", cache_dir, base);
    v->arc = mc_archive_open_dims(path, nx, ny, nz, quality);
    if (!v->arc) {
        for (int i = 0; i < v->nlods; ++i) mc_zarr_free(v->lv[i].z);
        s3_client_free(v->s3); free(v); return NULL;
    }
    v->cache = mc_cache_new_archive(cache_bytes, v->arc);
    if (!v->cache) {
        mc_archive_close(v->arc);
        for (int i = 0; i < v->nlods; ++i) mc_zarr_free(v->lv[i].z);
        s3_client_free(v->s3); free(v); return NULL;
    }

    // Pipeline: a few download threads (network-bound, pop the LIFO request
    // stack) feed a bounded decode queue drained by a decode pool (CPU-bound).
    pthread_cond_init(&v->dq_ne, NULL);
    pthread_cond_init(&v->dq_nf, NULL);
    v->dq_cap = 256;                                   // bounded decode queue
    v->dq = calloc((size_t)v->dq_cap, sizeof *v->dq);
    v->rs_cap = 512;                                   // LIFO request stack
    v->reqstk = calloc((size_t)v->rs_cap, sizeof *v->reqstk);

    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    v->ndecoders = nproc > 2 ? (nproc < 32 ? (int)nproc : 32) : 2;
    for (int i = 0; i < v->ndecoders; ++i)
        pthread_create(&v->decoders[i], NULL, decoder_main, v);
    v->ndl = 8;                                        // download threads (latency-bound)
    for (int i = 0; i < v->ndl; ++i)
        pthread_create(&v->dlthreads[i], NULL, dl_main, v);
    MCVLOG("open      %s  decoders=%d dl_threads=%d dq_cap=%d", url, v->ndecoders, v->ndl, v->dq_cap);
    return v;
}

void mc_volume_free(mc_volume *v) {
    if (!v) return;
    // stop download + decode threads.
    pthread_mutex_lock(&v->mu);
    v->stop = 1;
    pthread_cond_broadcast(&v->cv);      // wake download threads
    pthread_cond_broadcast(&v->dq_ne);   // wake decoders
    pthread_cond_broadcast(&v->dq_nf);   // wake blocked producers
    pthread_mutex_unlock(&v->mu);
    for (int i = 0; i < v->ndl; ++i) pthread_join(v->dlthreads[i], NULL);
    for (int i = 0; i < v->ndecoders; ++i) pthread_join(v->decoders[i], NULL);
    // drain any remaining decode items (free their raw buffers).
    while (v->dq_head != v->dq_tail) {
        decode_item *it = &v->dq[v->dq_head];
        for (int k = 0; k < it->nsub; ++k) free(it->raw[k]);
        v->dq_head = (v->dq_head + 1) % v->dq_cap;
    }
    pthread_cond_destroy(&v->dq_ne);
    pthread_cond_destroy(&v->dq_nf);
    if (v->cache) mc_cache_free(v->cache);
    if (v->arc) mc_archive_close(v->arc);
    for (int i = 0; i < v->nlods; ++i) if (v->lv[i].z) mc_zarr_free(v->lv[i].z);
    if (v->s3) s3_client_free(v->s3);
    free(v->dq);
    free(v->reqstk);
    pthread_mutex_destroy(&v->mu);
    pthread_cond_destroy(&v->cv);
    free(v);
}

int  mc_volume_nlods(const mc_volume *v) { return v ? v->nlods : 0; }
void mc_volume_shape(const mc_volume *v, int lod, int *nz, int *ny, int *nx) {
    mc_zarr_shape(v->lv[lod].z, nz, ny, nx);
}
void mc_volume_block_grid(const mc_volume *v, int lod, int *nz, int *ny, int *nx) {
    int sz, sy, sx;
    mc_zarr_shape(v->lv[lod].z, &sz, &sy, &sx);
    if (nz) *nz = (sz + BLK - 1) / BLK;
    if (ny) *ny = (sy + BLK - 1) / BLK;
    if (nx) *nx = (sx + BLK - 1) / BLK;
}
int mc_volume_get_level_meta(const mc_volume *v, int lod, mc_volume_level_meta *out) {
    if (!v || !out || lod < 0 || lod >= v->nlods) return -1;
    const mc_zarr *z = v->lv[lod].z;
    mc_zarr_shape(z, &out->shape[0], &out->shape[1], &out->shape[2]);
    out->inner_edge = mc_zarr_inner_edge(z);
    out->shard_edge = mc_zarr_shard_edge(z);
    const char *c = mc_zarr_inner_codec(z);
    snprintf(out->codec, sizeof out->codec, "%s", c ? c : "");
    return 0;
}

// ---------------------------------------------------------------------------
// block serving
// ---------------------------------------------------------------------------
int mc_volume_try_block(mc_volume *v, int lod, int bz, int by, int bx, uint8_t *dst) {
    if (lod < 0 || lod >= v->nlods) { memset(dst, 0, BLK * BLK * BLK); return 0; }
    int cz = bz / PER, cy = by / PER, cx = bx / PER;
    mc_cover cov = mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx);
    if (cov == MC_ABSENT) {
        req_push(v, lod, cz, cy, cx);   // LIFO download request; render falls to coarser LOD
        memset(dst, 0, BLK * BLK * BLK);
        return 0;
    }
    if (cov == MC_ZERO) { memset(dst, 0, BLK * BLK * BLK); return 1; }
    mc_cache_get_copy(v->cache, lod, bz, by, bx, dst);
    return 1;
}

int mc_volume_get_block(mc_volume *v, int lod, int bz, int by, int bx, uint8_t *dst) {
    if (lod < 0 || lod >= v->nlods) { memset(dst, 0, BLK * BLK * BLK); return -1; }
    int cz = bz / PER, cy = by / PER, cx = bx / PER;
    mc_cover cov = ensure_region(v, lod, cz, cy, cx);
    if (cov == MC_ZERO || cov == MC_ABSENT) { memset(dst, 0, BLK * BLK * BLK); return cov == MC_ZERO ? 0 : -1; }
    mc_cache_get_copy(v->cache, lod, bz, by, bx, dst);
    return 1;
}

// ---------------------------------------------------------------------------
// sampling source
// ---------------------------------------------------------------------------
static const uint8_t *vol_block(const mc_sample_src *src,
                                int bz, int by, int bx, uint8_t *tmp) {
    mc_volume *v = src->ud;
    int r = src->aux2 ? mc_volume_get_block(v, src->aux, bz, by, bx, tmp)
                      : mc_volume_try_block(v, src->aux, bz, by, bx, tmp);
    return r == 1 ? tmp : NULL;
}

mc_sample_src mc_volume_sample_src(mc_volume *v, int lod, int blocking) {
    mc_sample_src s = {0};
    s.ud = v; s.aux = lod; s.aux2 = blocking; s.block = vol_block;
    mc_volume_shape(v, lod, &s.nz, &s.ny, &s.nx);
    return s;
}

mc_sample_lods mc_volume_sample_lods(mc_volume *v, int blocking) {
    mc_sample_lods ls = {0};
    ls.nlods = v->nlods < 8 ? v->nlods : 8;
    for (int l = 0; l < ls.nlods; l++)
        ls.lods[l] = mc_volume_sample_src(v, l, blocking);
    return ls;
}

// ---------------------------------------------------------------------------
// prefetch — batch a whole shard's present inner chunks in ONE parallel
// s3_get_batch (many concurrent GETs over pooled connections), then decode +
// assemble into 256^3 regions and append. This is the throughput path: the
// parallelism lives in libs3's connection pool, so a FEW prefetch driver
// threads saturate bandwidth without a thread-per-GET explosion. RAM is bounded
// by one shard's compressed chunks (a fraction of the decoded shard).
// (cz,cy,cx) is any source inner-chunk in the target shard.
// ---------------------------------------------------------------------------
// Download a shard's present chunks (one parallel s3_get_batch) and PUSH each
// region's raw payload(s) to the decode queue — NO decode on this (download)
// thread. Decoders drain the queue in parallel, so the network stays saturated.
// Backpressure in decode_push bounds RAM. (cz,cy,cx) = source inner-chunk coord.
void mc_volume_prefetch_shard(mc_volume *v, int lod, int cz, int cy, int cx) {
    if (lod < 0 || lod >= v->nlods) return;
    level_t *lv = &v->lv[lod];
    mc_zarr *z = lv->z;
    const int edge = mc_zarr_inner_edge(z);            // 256 (c3d) or 128 (v2)
    const int sub = CHUNK / edge;                      // source chunks per region axis

    char shard_key[64];
    mc_zarr_range *ranges = NULL;
    int nr = 0;
    double t0 = mcv_now();
    if (mc_zarr_shard_index(z, cz, cy, cx, shard_key, &ranges, &nr) < 0) {
        MCVLOG("shard_idx lod%d src(%d,%d,%d) FAILED", lod, cz, cy, cx); return;
    }
    MCVLOG("shard_idx lod%d src(%d,%d,%d) -> %d present chunks (footer %.0fms)",
           lod, cz, cy, cx, nr, mcv_now() - t0);
    if (nr == 0) { free(ranges); return; }             // all air

    char shard_url[1280];
    snprintf(shard_url, sizeof shard_url, "%s/%s", lv->prefix, shard_key);
    uint64_t got = 0;
    int nbatch = 0;

    // Download the shard's chunks in batches of MC_BATCH (bounded buffering),
    // then hand each region's raw bytes to the decode pool. v2 groups the sub^3
    // cube per region; c3d is 1:1.
    enum { MC_BATCH = 48 };
    s3_range_req reqs[MC_BATCH];
    s3_response resp[MC_BATCH];
    int idx[MC_BATCH];
    for (int base = 0; base < nr; ) {
        int nq = 0;
        while (base < nr && nq < MC_BATCH) {
            mc_zarr_range *rg = &ranges[base++];
            int rz = rg->cz / sub, ry = rg->cy / sub, rx = rg->cx / sub;
            if (mc_archive_chunk_coverage(v->arc, lod, rz, ry, rx) != MC_ABSENT) continue;
            reqs[nq] = (s3_range_req){shard_url, rg->off, rg->len};
            idx[nq] = base - 1;
            ++nq;
        }
        if (nq == 0) continue;
        memset(resp, 0, sizeof resp);
        double tb = mcv_now();
        if (v->local) {
            // Local: each req is a ranged read of the shard file. No network, no
            // batching win — just pread each range into an s3_response so the
            // decode-push path below is identical to the remote case.
            FILE *lf = fopen(shard_url, "rb");
            for (int i = 0; i < nq; ++i) {
                if (!lf) { resp[i].status = 404; continue; }
                if (fseek(lf, (long)reqs[i].offset, SEEK_SET) != 0) { resp[i].status = 500; continue; }
                uint8_t *b = malloc(reqs[i].length ? reqs[i].length : 1);
                if (!b) { resp[i].status = 500; continue; }
                size_t g = reqs[i].length ? fread(b, 1, reqs[i].length, lf) : 0;
                if (g != reqs[i].length) { free(b); resp[i].status = 500; continue; }
                resp[i].status = 200; resp[i].body = b; resp[i].body_len = g;
            }
            if (lf) fclose(lf);
        } else {
            s3_get_batch(v->s3, reqs, (size_t)nq, 32, resp);   // partial ok; check each
        }
        { int ok = 0; uint64_t bytes = 0;
          for (int i = 0; i < nq; ++i) if (s3_response_ok(&resp[i])) { ok++; bytes += resp[i].body_len; }
          MCVLOG("batch#%d  lod%d nq=%d ok=%d %.2fMB in %.0fms = %.1f MB/s",
                 nbatch++, lod, nq, ok, bytes/1048576.0, mcv_now()-tb,
                 bytes/1048576.0/((mcv_now()-tb)/1000.0)); }

        if (sub == 1) {                                // c3d: one chunk == one region
            for (int i = 0; i < nq; ++i) {
                mc_zarr_range *rg = &ranges[idx[i]];
                if (s3_response_ok(&resp[i]) && rg->len && resp[i].body_len >= rg->len) {
                    decode_item it = {lod, rg->cz, rg->cy, rg->cx, 1, 1, {0},{0},{0}, {0},{0}};
                    it.raw[0] = malloc(rg->len);
                    if (it.raw[0]) { memcpy(it.raw[0], resp[i].body, rg->len); it.rlen[0] = rg->len;
                        got += rg->len; decode_push(v, &it); }
                }
                s3_response_free(&resp[i]);
            }
        } else {                                       // v2: regroup the cube per region
            // Build one decode_item per distinct region in this batch.
            for (int i = 0; i < nq; ++i) {
                if (idx[i] < 0) continue;              // already consumed into a cube
                mc_zarr_range *r0 = &ranges[idx[i]];
                int rz = r0->cz / sub, ry = r0->cy / sub, rx = r0->cx / sub;
                decode_item it = {lod, rz, ry, rx, sub, 0, {0},{0},{0}, {0},{0}};
                for (int j = i; j < nq; ++j) {
                    if (idx[j] < 0) continue;
                    mc_zarr_range *rg = &ranges[idx[j]];
                    if (rg->cz / sub != rz || rg->cy / sub != ry || rg->cx / sub != rx) continue;
                    if (s3_response_ok(&resp[j]) && resp[j].body_len >= rg->len && it.nsub < 8) {
                        size_t rlen = rg->len ? rg->len : resp[j].body_len;
                        uint8_t *buf = malloc(rlen);
                        if (buf) { memcpy(buf, resp[j].body, rlen);
                            int k = it.nsub++;
                            it.raw[k] = buf; it.rlen[k] = rlen;
                            it.oz[k] = (rg->cz % sub) * edge;
                            it.oy[k] = (rg->cy % sub) * edge;
                            it.ox[k] = (rg->cx % sub) * edge;
                            got += rlen;
                        }
                    }
                    idx[j] = -1;                       // consumed
                }
                decode_push(v, &it);                   // nsub may be 0 -> ZERO region
            }
            for (int i = 0; i < nq; ++i) s3_response_free(&resp[i]);
        }
    }
    atomic_fetch_add_explicit(&v->net_bytes, got, memory_order_relaxed);
    free(ranges);
}

void mc_volume_prefetch_level(mc_volume *v, int lod, int nthreads, volatile int *cancel) {
    (void)nthreads;   // TODO: thread team; serial walk for now.
    if (lod < 0 || lod >= v->nlods) return;
    int gz, gy, gx;
    mc_zarr_inner_grid(v->lv[lod].z, &gz, &gy, &gx);
    int per = mc_zarr_shard_edge(v->lv[lod].z) / mc_zarr_inner_edge(v->lv[lod].z);
    for (int sz = 0; sz < gz; sz += per)
        for (int sy = 0; sy < gy; sy += per)
            for (int sx = 0; sx < gx; sx += per) {
                if (cancel && *cancel) return;
                mc_volume_prefetch_shard(v, lod, sz, sy, sx);
            }
}

size_t mc_volume_set_cache_bytes(mc_volume *v, size_t bytes) {
    return (v && v->cache) ? mc_cache_resize(v->cache, bytes) : 0;
}

void mc_volume_set_ready_cb(mc_volume *v, mc_volume_ready_fn cb, void *ud) {
    v->ready_cb = cb;
    v->ready_ud = ud;
}

void mc_volume_get_stats(const mc_volume *v, mc_volume_stats *out) {
    mc_cache_stats cs = {0};
    if (v->cache) mc_cache_get_stats(v->cache, &cs);
    out->cache_hits = cs.hits;
    out->cache_misses = cs.misses;
    out->cache_used_blocks = cs.used;
    out->cache_cap_blocks = cs.slots;
    out->disk_bytes = v->arc ? mc_archive_data_len(v->arc) : 0;
    out->net_bytes = atomic_load_explicit(&v->net_bytes, memory_order_relaxed);
    out->regions_inflight = (uint64_t)v->rs_n;
}
