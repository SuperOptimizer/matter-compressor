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
