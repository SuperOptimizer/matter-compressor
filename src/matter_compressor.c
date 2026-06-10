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
    int32x4_t ae0=vdupq_n_s32(rnd), ae1=vdupq_n_s32(rnd);
    int32x4_t ao0=vdupq_n_s32(0),  ao1=vdupq_n_s32(0);
    for(int j=0;j<8;++j){
        mc_fi32 ve=in[2*j];
        if(ve){
            int32x4_t v=vdupq_n_s32(ve);
            ae0=vmlaq_s32(ae0,vld1q_s32(&g_cm_e[j][0]),v);
            ae1=vmlaq_s32(ae1,vld1q_s32(&g_cm_e[j][4]),v);
        }
        mc_fi32 vo=in[2*j+1];
        if(vo){
            int32x4_t v=vdupq_n_s32(vo);
            ao0=vmlaq_s32(ao0,vld1q_s32(&g_cm_o[j][0]),v);
            ao1=vmlaq_s32(ao1,vld1q_s32(&g_cm_o[j][4]),v);
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
// variant taking PREPARED i32 coefficients (decoder fuses dequantization into
// the input conversion) and returning the raw i32 spatial result.
static void mc_dct3_inv_i32(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    static _Thread_local mc_fi32 a[16*16*16]   __attribute__((aligned(MC_DCT_ALIGN)));
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
static uint16_t *g_scanS[6]; static int g_scanS_ready[6];
static int scanS_cmp_S;
static int scanS_cmp(const void*pa,const void*pb){
    rc_u32 a=*(const rc_u32*)pa,b=*(const rc_u32*)pb; int S=scanS_cmp_S;
    rc_u32 fa=(a/(S*S))+((a/S)%S)+(a%S), fb=(b/(S*S))+((b/S)%S)+(b%S);
    if(fa!=fb) return (int)fa-(int)fb; return (int)a-(int)b;
}
static void scanS_build(int S){
    int l=0,t=S; while(t>1){t>>=1;l++;}
    if(g_scanS_ready[l]) return;
    int n=S*S*S; rc_u32 *ord=malloc(n*sizeof(rc_u32)); for(int i=0;i<n;++i)ord[i]=i;
    scanS_cmp_S=S; qsort(ord,n,sizeof(rc_u32),scanS_cmp);
    g_scanS[l]=malloc(n*sizeof(uint16_t)); for(int i=0;i<n;++i)g_scanS[l][i]=(uint16_t)ord[i];
    free(ord); g_scanS_ready[l]=1;
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
static _Thread_local float g_quality = 8.0f;
static int   g_max_err = 0;            // 0 = corrections off
// per-coefficient quant step table (quality * hf_weight), rebuilt when quality
// changes. powf per coefficient was 20%+ of encode AND decode time.
static _Thread_local float g_step_tab[N3];
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
    for(int cz=0;cz<MC_BLK;++cz)for(int cy=0;cy<MC_BLK;++cy)for(int cx=0;cx<MC_BLK;++cx)
        g_step_tab[(cz*MC_BLK+cy)*MC_BLK+cx]=g_quality*hf_weight(cz,cy,cx);
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
    static _Thread_local float blk[N3], coef[N3];
    static _Thread_local mc_i32 lv[N3];
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
    for(int i=0;i<n;++i) any|=vox[i];
    for(int i=0;i<n;++i){ if(vox[i]){ sum+=vox[i]; cnt++; } }
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
    // harmonic (Jacobi) air-fill: relax air voxels to 6-neighbor mean (material
    // fixed). Air-voxel index list + ping-pong buffers: each sweep touches only
    // the air voxels and there is no per-sweep copy (material values are
    // identical in both buffers).
    if(nair>0){
        static _Thread_local float tmp[N3]; (void)tmp;
        static _Thread_local uint16_t ai[N3]; static _Thread_local uint8_t arc_[N3];
        int S=MC_BLK, na=0;
        for(int z=0;z<S;++z)for(int y=0;y<S;++y)for(int x=0;x<S;++x){
            int i=(z*S+y)*S+x; if(vox[i]) continue;
            ai[na]=(uint16_t)i;
            arc_[na]=(uint8_t)((z?1:0)|(z<S-1?2:0)|(y?4:0)|(y<S-1?8:0)|(x?16:0)|(x<S-1?32:0));
            na++;
        }
        (void)tmp;
        // Coarse-to-fine init: solve the fill on the 4^3 subcube grid first
        // (each cell = mean of its material voxels, air cells relaxed), then
        // seed fine air voxels from their cell before the fine SOR sweeps.
        // Lands much closer than a flat dc start, so 4 sweeps converge further.
        {
            float cs[64]; int cm[64]; const int G=4, GS=MSUB;
            for(int c=0;c<64;++c){ cs[c]=0; cm[c]=0; }
            for(int z=0;z<S;++z)for(int y=0;y<S;++y)for(int x=0;x<S;++x){
                int i=(z*S+y)*S+x; if(!vox[i]) continue;
                int c=((z/GS)*G+(y/GS))*G+(x/GS);
                cs[c]+=blk[i]; cm[c]++;
            }
            for(int c=0;c<64;++c) cs[c]=cm[c]?cs[c]/cm[c]:0;
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
            for(int u=0;u<na;++u){
                int i=ai[u]; int z=i/(S*S), y=(i/S)%S, x=i%S;
                blk[i]=cs[((z/GS)*4+(y/GS))*4+(x/GS)];
            }
        }
        // SOR (in-place Gauss-Seidel + over-relaxation): converges ~4x faster
        // than Jacobi per sweep, so fewer sweeps for a BETTER fill.
        const float OMEGA=1.6f;
        for(int it=0; it<MC_FILL_SWEEPS; ++it){
            for(int u=0;u<na;++u){
                int i=ai[u]; unsigned m=arc_[u];
                float a=0; int c=0;
                if(m&1){a+=blk[i-S*S];c++;} if(m&2){a+=blk[i+S*S];c++;}
                if(m&4){a+=blk[i-S];c++;}   if(m&8){a+=blk[i+S];c++;}
                if(m&16){a+=blk[i-1];c++;}  if(m&32){a+=blk[i+1];c++;}
                if(c) blk[i]+=OMEGA*(a/c-blk[i]);
            }
        }
    }
    mc_dct3_fwd(blk,coef);
    for(int idx=0;idx<N3;++idx) lv[idx]=quant_one(coef[idx],g_step_tab[idx]);
    static _Thread_local rc_i16 ql[N3];
    static _Thread_local rc_u8 scratch[N3*4+1024];
    for(int i=0;i<n;++i){ mc_i32 v=lv[i]; ql[i]=(rc_i16)(v>32767?32767:v<-32768?-32768:v); }

    // max-error corrections: locally reconstruct and list voxels with |err| > tau.
    static _Thread_local uint16_t cpos[N3]; static _Thread_local mc_i32 cdel[N3];
    int ncorr=0;
    if(g_max_err>0){
        static _Thread_local float rcoef[N3], rblk[N3];
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

    rc_enc e; enc_init(&e,scratch,sizeof scratch);
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
    if(slen>sizeof scratch){ fprintf(stderr,"mc_enc_block: scratch overflow (%u)\n",slen); abort(); }

    mc_buf_put(out,scratch,slen);
    *len_out = slen;
    return 1;
}

void mc_dec_block(const mc_u8 *p, uint32_t plen, mc_u8 *dst){
    int n=N3, dc=0, flags=0;
    static _Thread_local mc_u8 air[N3];
    static _Thread_local rc_i16 ql[N3];
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
    static _Thread_local float coef[N3], blk[N3];
    step_tab_build();
    int ez=ext[0],ey=ext[1],ex=ext[2];
    if(ez<0 && !(flags&1) && !(flags&2)){                   // constant block: dc fill
        memset(dst,(mc_u8)dc,n); return;
    }
    (void)ey;(void)ex;
#if MC_SIMD_NEON
    {   // fused dequant -> i32 DCT input (no float coefficient pass), then
        // integer iDCT and vectorized clamp+dc+air store.
        static _Thread_local mc_fi32 qin[N3] __attribute__((aligned(64)));
        static _Thread_local mc_fi32 qout[N3] __attribute__((aligned(64)));
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
        rc_u32 ncorr=dec_eg(&d)+1, pos=0;
        for(rc_u32 c=0;c<ncorr;++c){
            pos+=dec_eg(&d);
            int neg=dec_bypass(&d); rc_u32 m=dec_eg(&d)+1;
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
    // all-air chunk (blen==0): no blob, slot stays absent (decodes to zero). rc stays 0.
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

mc_cover mc_archive_chunk_coverage(mc_archive *a, int lod, int cz,int cy,int cx){
    if(!a||lod<0||lod>7) return MC_ABSENT;
    uint64_t root = w_read_u64(a, MCH_ROOTOFF+(uint64_t)lod*8);
    return mc_resolve_chunk(a->base, root, cz,cy,cx) ? MC_PRESENT : MC_ABSENT;
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
    if(!a||!chunk_off){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    mc_set_quality(mc_chunk_q(a->base,chunk_off));   // thread-local; per-chunk q
    uint64_t boff; uint32_t bl;
    if(!mc_block_range(a->base,chunk_off,bz,by,bx,&boff,&bl)){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
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
    if(!chunk_off){ memset(out,0,(size_t)MC_CHUNK*MC_CHUNK*MC_CHUNK); return; }
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
    if(!co) return 0;
    const u8 *bm=a->base+co+MC_BLOB_HDR+mc_chunk_fmaplen(a->base,co);
    return mc_bit_get(bm,((bz&15)*16+(by&15))*16+(bx&15));
}

float mc_archive_block_fraction(mc_archive *a, int lod, int bz, int by, int bx){
    if(!a||lod<0||lod>7||bz<0||by<0||bx<0) return 0.0f;
    uint64_t co=mc_archive_chunk_offset(a,lod,bz>>4,by>>4,bx>>4);
    if(!co) return 0.0f;
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
                    if(!co) continue;
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
    if(!chunk_off){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
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

void mc_cache_free(mc_cache *c){
    if(!c) return;
    for(int s=0;s<NSHARD;++s){
        shard_t *sh=&c->sh[s];
        pthread_mutex_destroy(&sh->mu);
        free(sh->map_key); free(sh->map_slot); free(sh->slot_key); free(sh->slot_ref);
        free(sh->fs); free(sh->fm); free(sh->gfp); free(sh->gset); free(sh->slot_inmain);
        free(sh->slot_epoch);
    }
#if MC_CACHE_MMAP
    munmap(c->arena_base,c->arena_bytes);
#else
    free(c->arena_base);
#endif
    pthread_mutex_destroy(&c->rd_mu);
    free(c);
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
