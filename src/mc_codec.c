#define _GNU_SOURCE
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "matter_compressor.h"
#include "mc_codec.h"
#include "mc_archive.h"
typedef uint8_t u8;
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

// Float DCT-16: the integer Q14 fixed-point transform was REPLACED by the f32
// transform in mc_codec_float.h. It is orthonormal with the same normalization
// as the old integer table, so the quality/step calibration carries over
// unchanged; it removes the per-pass >>14 rounding and the lrintf boundary
// casts, so reconstruction no longer accumulates fixed-point noise. The old
// integer path (g_mc_cm tables, mc_dct1d_*, mc_lines_*, mc_rot, mc_dct3_inv_i32)
// and its SIMD variants are gone; archives written by it are not supported.
#include "mc_codec_float.h"
static void mc_dct_init(void){ mc_dctf_init(); }

// 3D forward/inverse on a 16^3 block (f32 in/out for the quant path). The two
// scratch buffers live in the per-thread ctx so the hot path is allocation-free.
typedef struct {
    float a[MC_F_N3] __attribute__((aligned(MC_DCT_ALIGN)));
    float b[MC_F_N3] __attribute__((aligned(MC_DCT_ALIGN)));
} mc_dct_tls_t;
static void mc_dct3_fwd(mc_dct_tls_t *D, const float *restrict blk, float *restrict coef){
    mc_dctf3_fwd(blk,coef,D->a,D->b);
}
static void mc_dct3_inv(mc_dct_tls_t *D, const float *restrict coef, float *restrict blk){
    mc_dctf3_inv(coef,blk,D->a,D->b);
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
// PHercParis4 2.4um (fysics-masked) via tools/mc_train; rc_prior_build_into()
// interpolates in log2(q) into the ctx's pri[][] (the decoder knows q, so this
// costs no side information). 2048 = untrained slot.
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
// Per-volume prior override (format v6: a trained-prior blob stored in the
// archive replaces the baked corpus tables). Process-global config, set once at
// open before decode threads start; a generation counter forces per-ctx rebuild.
static uint16_t g_plo_ovr[8][RC_NSLOT], g_phi_ovr[8][RC_NSLOT];
static int g_pri_ovr = 0;
static int g_pri_gen = 1;
static void rc_set_priors(const uint16_t *plo, const uint16_t *phi){
    if(plo&&phi){
        memcpy(g_plo_ovr,plo,sizeof g_plo_ovr);
        memcpy(g_phi_ovr,phi,sizeof g_phi_ovr);
        g_pri_ovr=1;
    } else g_pri_ovr=0;
    g_pri_gen++;
}
// Build the per-ctx interpolated prior table from the process-global config.
static void rc_prior_build_into(uint16_t pri[8][RC_NSLOT], float *pri_q, int *pri_seen, float q){
    if(*pri_q==q && *pri_seen==g_pri_gen) return;
    *pri_seen=g_pri_gen;
    float lo=0.0f, hi=3.585f;                       // log2(1) .. log2(12)
    float w=(q<=1.0f)?0.0f:((float)(log(q)/log(2.0))-lo)/(hi-lo);
    if(w<0)w=0; if(w>1)w=1;
    const uint16_t (*tlo)[RC_NSLOT] = g_pri_ovr ? (const uint16_t(*)[RC_NSLOT])g_plo_ovr : RC_PLO;
    const uint16_t (*thi)[RC_NSLOT] = g_pri_ovr ? (const uint16_t(*)[RC_NSLOT])g_phi_ovr : RC_PHI;
    for(int c=0;c<8;++c)for(int s=0;s<RC_NSLOT;++s)
        pri[c][s]=(uint16_t)(tlo[c][s]+(thi[c][s]-tlo[c][s])*w+0.5f);
    *pri_q=q;
}
// Prior-table accessors: pri is the ctx's interpolated uint16_t[8][RC_NSLOT].
#define RC_PRIOR_SIG(pri)   ((pri)[0])
#define RC_PRIOR_MAG(pri)   ((pri)[1])
#define RC_PRIOR_EOB(pri)   ((pri)[2])
#define RC_PRIOR_MASK(pri)  ((pri)[3])
#define RC_PRIOR_MASKU(pri) ((pri)[4])
#define RC_PRIOR_MASKA(pri) ((pri)[5])
#define RC_PRIOR_FLAG(pri)  ((pri)[6])
#define RC_PRIOR_DC(pri)    ((pri)[7])

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
static void atom_ctx_init(atom_ctx *a, const uint16_t (*pri)[RC_NSLOT]){
    for(int i=0;i<NB_BANDS*4;++i) ctx_init_p(&a->sig[i],RC_PRIOR_SIG(pri)[i]);
    for(int i=0;i<MAGCTX;++i)     ctx_init_p(&a->mag[i],RC_PRIOR_MAG(pri)[i]);
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
        int n=S*S*S; rc_u32 *ord=malloc(n*sizeof(rc_u32));
        uint16_t *tab=ord?malloc(n*sizeof(uint16_t)):NULL;
        if(!ord||!tab){ free(ord); free(tab); pthread_mutex_unlock(&g_scanS_mu); return; }
        for(int i=0;i<n;++i)ord[i]=i;
        scanS_cmp_S=S; qsort(ord,n,sizeof(rc_u32),scanS_cmp);
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
static void eob_ctx_init(eob_ctx*c, const uint16_t (*pri)[RC_NSLOT]){ for(int i=0;i<EOB_CTX;++i) ctx_init_p(&c->pfx[i],RC_PRIOR_EOB(pri)[i]); }
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
static void enc_block_coefs(rc_enc*e,const rc_i16*q,int S, const uint16_t (*pri)[RC_NSLOT]){
    scanS_build(S); int l=0,t=S; while(t>1){t>>=1;l++;} const uint16_t*scan=g_scanS[l];
    int n=S*S*S; atom_ctx ac; atom_ctx_init(&ac,pri);
    eob_ctx ec; eob_ctx_init(&ec,pri);
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
static void dec_block_coefs_ext(rc_dec*d,rc_i16*q,int S,int ext[3], const uint16_t (*pri)[RC_NSLOT]){
    scanS_build(S); int l=0,t=S; while(t>1){t>>=1;l++;} const uint16_t*scan=g_scanS[l];
    int n=S*S*S; atom_ctx ac; atom_ctx_init(&ac,pri); memset(q,0,n*sizeof(rc_i16));
    eob_ctx ec; eob_ctx_init(&ec,pri);
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
uint64_t mc_xxh64(const void *data, size_t len, uint64_t seed){
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

// Per-operation codec scratch, formerly a bundle of _Thread_local globals. The
// dynamic-TLS model (PIC .so) compiled every _Thread_local access to a
// __tls_get_addr CALL (~14% of decode CPU); scattered through the hot functions
// those calls also forced the range-coder state to spill to the stack around
// each one (objdump: 8 blr + 102 sp-stores in mc_dec_block). Folding everything
// into one heap-allocated mc_codec_ctx passed by pointer turns those into plain
// member loads off one register; hot loops run call-free with the coder state
// held in registers. One ctx is owned per worker thread (decode/encode/cache
// pools each allocate their own), so concurrent operations stay race-free.
typedef struct {
    // decode
    rc_i16 ql[N3];
    float  coef[N3], blk[N3];
    // encode
    float  eblk[N3], ecoef[N3];
    rc_u8  scratch[N3*4+1024];
    uint16_t cpos[N3]; mc_i32 cdel[N3];
    float  rcoef[N3], rblk[N3];
} mc_tls_t;
// (air-fill SOR scratch removed with the per-voxel air mask.)

// chunk-level encode scratch (encode_chunk_blob / append paths). MC_GRID3 (=4096,
// the 16^3 block grid) is #defined later in the archive section; use the literal
// here since this struct precedes it.
#define MC_CTX_GRID3 4096
typedef struct {
    mc_buf   tmp;                 // concatenated block payloads (growable)
    uint8_t  frac[MC_CTX_GRID3];
    uint8_t  vox[N3];
    uint16_t lens16[MC_CTX_GRID3];
    uint8_t  fmap[MC_CTX_GRID3/2+64];
} mc_chunk_tls_t;

// Explicit per-thread codec context (replaces ~30 _Thread_local globals).
typedef struct mc_codec_ctx {
    float    quality;
    int      max_err;             // 0 = corrections off
    float    step_tab[N3];        // quality*hf_weight per coefficient
    float    rstep_tab[N3];       // 1/step: quant uses mul, not div
    float    step_q;              // cache guard for step tables
    uint16_t pri[8][RC_NSLOT];    // interpolated trained priors for this q
    float    pri_q;               // cache guard for priors
    int      pri_seen;            // priors generation seen
    mc_tls_t      scratch;        // hot enc/dec block scratch
    mc_dct_tls_t  dct;            // DCT line buffers
    mc_chunk_tls_t chunk;         // chunk-blob encode scratch
} mc_codec_ctx;

static void step_tab_build(mc_codec_ctx *C);

mc_codec_ctx *mc_codec_ctx_new(void){
    mc_codec_ctx *C = calloc(1,sizeof *C);
    if(!C) return NULL;
    C->quality = 8.0f;
    C->max_err = 0;
    C->step_q  = -1.0f;
    C->pri_q   = -1.0f;
    C->pri_seen = 0;
    step_tab_build(C);
    return C;
}
void mc_codec_ctx_free(mc_codec_ctx *C){
    if(!C) return;
    free(C->chunk.tmp.p);
    free(C);
}
void  mc_codec_ctx_set_quality(mc_codec_ctx *C, float q){
    // reject NaN/Inf/<=0 (e.g. an untrusted per-chunk q): they make step=0 ->
    // rstep=Inf and poison the quant tables (Inf*0=NaN; NaN->uint16 cast is UB).
    if(!(q > 0.0f) || q > 1024.0f) q = 8.0f;
    if(C){ C->quality=q; step_tab_build(C); }
}
float mc_codec_ctx_get_quality(mc_codec_ctx *C){ return C?C->quality:0.0f; }
void  mc_codec_ctx_set_max_error(mc_codec_ctx *C, int tau){ if(C) C->max_err = tau<0?0:tau; }
int   mc_codec_ctx_get_max_error(mc_codec_ctx *C){ return C?C->max_err:0; }

// ---- calibrated preset ladder (see header; bench/RESULTS.md for the data) --
static const struct { float q; int tau; const char *name; } g_presets[8] = {
    { 0.5f,   1, "archival"  },
    { 0.5f,   2, "master"    },
    { 1.0f,   4, "high"      },
    { 2.5f,   8, "balanced"  },
    { 6.0f,  16, "streaming" },
    {16.0f,  32, "fast"      },
    {32.0f,  64, "ultrafast" },
    {64.0f, 128, "preview"   },
};
float mc_preset_quality(mc_preset p){
    if((unsigned)p>=MC_PRESET_COUNT) p=MC_PRESET_STREAMING;
    return g_presets[p].q;
}
int mc_preset_tau(mc_preset p){
    if((unsigned)p>=MC_PRESET_COUNT) p=MC_PRESET_STREAMING;
    return g_presets[p].tau;
}
const char *mc_preset_name(mc_preset p){
    if((unsigned)p>=MC_PRESET_COUNT) return "?";
    return g_presets[p].name;
}
float mc_apply_preset(mc_codec_ctx *C, mc_preset p){
    if((unsigned)p>=MC_PRESET_COUNT) p=MC_PRESET_STREAMING;
    mc_codec_ctx_set_quality(C,g_presets[p].q);
    mc_codec_ctx_set_max_error(C,g_presets[p].tau);
    return g_presets[p].q;
}
void  mc_codec_init(void){ mc_dct_init(); }
void  mc_codec_set_priors(const uint16_t *plo, const uint16_t *phi){ rc_set_priors(plo,phi); }

void mc_buf_put(mc_buf *b, const void *s, size_t n){
    if(b->len+n > b->cap){ size_t nc=b->cap?b->cap*2:1<<16; while(nc<b->len+n)nc*=2;
        void *tmp=realloc(b->p,nc); if(!tmp) return; b->p=tmp; b->cap=nc; }
    memcpy(b->p+b->len,s,n); b->len+=n;
}

// frozen quant: dead-zone, step = quality*(1+L1freq)^MC_HF_EXP
static inline float hf_weight(int cz,int cy,int cx){ return powf(1.0f+(float)(cz+cy+cx), MC_HF_EXP); }
static void step_tab_build(mc_codec_ctx *C){
    rc_prior_build_into(C->pri,&C->pri_q,&C->pri_seen,C->quality);
    if(C->step_q==C->quality) return;
    for(int cz=0;cz<MC_BLK;++cz)for(int cy=0;cy<MC_BLK;++cy)for(int cx=0;cx<MC_BLK;++cx){
        int i=(cz*MC_BLK+cy)*MC_BLK+cx;
        C->step_tab[i]=C->quality*hf_weight(cz,cy,cx);
        C->rstep_tab[i]=1.0f/C->step_tab[i];
    }
    C->step_q=C->quality;
}
// Dead-zone quant/dequant come from mc_codec_float.h (mc_quantf_one / mc_deqf_one);
// the local copies were identical f32 math and were removed with the integer path.


// (per-voxel air mask coder removed: blocks are encoded fully-material; the
// masked input zeros are just value-0 voxels reconstructed by the transform.)


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
int mc_enc_block(mc_codec_ctx *C, const mc_u8 *vox, mc_buf *out, uint32_t *len_out){
    int n=N3, any=0;
    mc_tls_t *T=&C->scratch;
    step_tab_build(C);
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
    int dc = (int)((sum+n/2)/n);   // DC over ALL voxels: no per-voxel air concept.
    // Encode every voxel as material -- the masked input zeros are just value-0
    // voxels (blk = -dc there, reconstructing to ~0). The per-voxel air mask and
    // the harmonic air-fill were removed (air-cut is off by default; the SAM2 mask
    // zeros are preserved closely enough by the transform + max-error pass, and
    // whole all-zero blocks are still skipped via the chunk bitmap above).
#if MC_SIMD_NEON
    {   int16x8_t vdc=vdupq_n_s16((int16_t)dc);
        for(int i=0;i<n;i+=16){
            uint8x16_t v=vld1q_u8(vox+i);
            int16x8_t lo=vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v)));
            int16x8_t hi=vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v)));
            lo=vsubq_s16(lo,vdc); hi=vsubq_s16(hi,vdc);
            vst1q_f32(blk+i   ,vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))));
            vst1q_f32(blk+i+4 ,vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))));
            vst1q_f32(blk+i+8 ,vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))));
            vst1q_f32(blk+i+12,vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))));
        }
    }
#else
    for(int i=0;i<n;++i) blk[i]=(float)((int)vox[i]-dc);
#endif
    mc_dct3_fwd(&C->dct,blk,coef);
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
        float t=fabsf(c)*C->rstep_tab[idx]+(1.0f-MC_DZ_FRAC);
        t=t>0.0f?t:0.0f; t=t<32767.0f?t:32767.0f;
        mc_i32 v=(mc_i32)t;
        ql[idx]=(rc_i16)(c<0.0f?-v:v);
    }

    // max-error corrections: locally reconstruct and list voxels with |err| > tau.
    uint16_t *cpos=T->cpos; mc_i32 *cdel=T->cdel;
    int ncorr=0;
    if(C->max_err>0){
        float *rcoef=T->rcoef, *rblk=T->rblk;
        for(int idx=0;idx<N3;++idx) rcoef[idx]=mc_deqf_one(ql[idx],C->step_tab[idx]);
        mc_dct3_inv(&C->dct,rcoef,rblk);
        for(int i=0;i<n;++i){
            // correct EVERY voxel (incl. masked zeros) -- no air mask forces them to
            // 0 now, so the max-error pass is what pins them back to ~0 within tau.
            int v=(int)lrintf(rblk[i])+dc; if(v<0)v=0; if(v>255)v=255;
            int err=(int)vox[i]-v;
            int ae=err<0?-err:err;
            if(ae>C->max_err){ cpos[ncorr]=(uint16_t)i; cdel[ncorr]= err<0 ? -(ae-C->max_err) : (ae-C->max_err); ncorr++; }
        }
    }

    rc_enc e; enc_init(&e,scratch,scratch_cap);
    {   // header bins: mixed, has-corr, dc (trained priors)
        const uint16_t (*pri)[RC_NSLOT]=C->pri;
        ctx_t cf[2]; for(int i=0;i<2;++i) ctx_init_p(&cf[i],RC_PRIOR_FLAG(pri)[i]);
        ctx_t cd[8]; for(int i=0;i<8;++i) ctx_init_p(&cd[i],RC_PRIOR_DC(pri)[i]);
        RC_TRAIN(RCC_FLAG,0,0);       enc_bit(&e,&cf[0],0);   // bit0 reserved (was "mixed"; air mask removed)
        RC_TRAIN(RCC_FLAG,1,ncorr>0); enc_bit(&e,&cf[1],ncorr>0);
        for(int b=7;b>=0;--b){ int bit=(dc>>b)&1; RC_TRAIN(RCC_DC,7-b,bit); enc_bit(&e,&cd[7-b],bit); }
    }
    enc_block_coefs(&e,ql,MC_BLK,C->pri);
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

void mc_dec_block(mc_codec_ctx *C, const mc_u8 *p, uint32_t plen, mc_u8 *dst){
    int n=N3, dc=0, flags=0;
    mc_tls_t *T=&C->scratch;
    step_tab_build(C);                  // before hot loops
    rc_i16 *ql=T->ql;
    rc_dec d; dec_init(&d,p,plen);
    {   // header bins (must mirror the encoder exactly)
        const uint16_t (*pri)[RC_NSLOT]=C->pri;
        ctx_t cf[2]; for(int i=0;i<2;++i) ctx_init_p(&cf[i],RC_PRIOR_FLAG(pri)[i]);
        ctx_t cd[8]; for(int i=0;i<8;++i) ctx_init_p(&cd[i],RC_PRIOR_DC(pri)[i]);
        flags |= dec_bit(&d,&cf[0]) ? 1 : 0;
        flags |= dec_bit(&d,&cf[1]) ? 2 : 0;
        for(int b=0;b<8;++b) dc=(dc<<1)|dec_bit(&d,&cd[b]);
    }
    (void)(flags&1);   // bit0 reserved (was "mixed"; per-voxel air mask removed)
    int ext[3]; dec_block_coefs_ext(&d,ql,MC_BLK,ext,C->pri);
    float *coef=T->coef, *blk=T->blk;
    int ez=ext[0],ey=ext[1],ex=ext[2];
    if(ez<0 && !(flags&2)){                                 // constant block: dc fill
        memset(dst,(mc_u8)dc,n); return;
    }
    (void)ey;(void)ex;
    // dequant -> f32 coefficients -> f32 inverse DCT -> clamp + dc. Every voxel is
    // material now (no air mask); masked input zeros were encoded as value-0 and
    // reconstruct to ~0 (pinned by the max-error pass).
    for(int idx=0;idx<N3;++idx) coef[idx]=mc_deqf_one(ql[idx],C->step_tab[idx]);
    mc_dct3_inv(&C->dct,coef,blk);
    for(int i=0;i<n;++i){
        int v = (int)lrintf(blk[i])+dc;
        if(v<0)v=0; if(v>255)v=255; dst[i]=(mc_u8)v;
    }
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


// Per-block decode dispatch by archive block codec. Only CABAC remains (c3g
// removed); the codec arg is kept for the header field but always selects CABAC.
void mc_dec_block_codec(mc_codec_ctx *C, uint32_t codec,
                                      const mc_u8 *p, uint32_t plen, mc_u8 *dst){
    (void)codec;
    mc_dec_block(C,p,plen,dst);
}
// gather a 16^3 block from a contiguous 256^3 chunk buffer (chunk is dense, no edges).
int gather_blk256(const u8 *chunk,int bz,int by,int bx,u8 *dst){
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

size_t encode_chunk_blob_codec(mc_codec_ctx *C, uint32_t codec, const u8 *chunk256, out_put_fn put, void *out){
    (void)codec;   // c3g removed: only CABAC
    mc_buf *tmp=&C->chunk.tmp; tmp->len=0;
    uint8_t bm[MC_BITMAP_BYTES]; memset(bm,0,sizeof bm);
    uint16_t blen[MC_GRID3]; int npresent=0;
    uint8_t *frac=C->chunk.frac;
    memset(frac,0,MC_GRID3);
    u8 *vox=C->chunk.vox;
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
        int bi=(bz*16+by)*16+bx;
        if(!gather_blk256(chunk256,bz,by,bx,vox)) continue;
        int cnt=0; for(int i=0;i<MC_BLK*MC_BLK*MC_BLK;++i) cnt+=vox[i]!=0;
        frac[bi]=(uint8_t)((cnt*15+2048)/4096);              // nibble 0..15
        if(cnt&&!frac[bi]) frac[bi]=1;                        // any material -> >=1
        uint32_t len=0;
        int coded = mc_enc_block(C,vox,tmp,&len);
        if(coded){ mc_bit_set(bm,bi); blen[bi]=(uint16_t)len; npresent++; }
    }
    if(!npresent) return 0;   // all air -> no blob
    // v7 blob header: [f32 q][u64 xxh64][u16 fmaplen][fmap]
    float q=C->quality;
    uint16_t *lens16=C->chunk.lens16; int nl=0;
    for(int bi=0;bi<MC_GRID3;++bi) if(mc_bit_get(bm,bi)) lens16[nl++]=blen[bi];
    uint8_t *fmap=C->chunk.fmap;
    uint16_t fml=(uint16_t)mc_enc_fracmap(frac,fmap,sizeof C->chunk.fmap);
    uint64_t h=mc_xxh64(fmap,fml,0x6D636368756E6Bull);
    h^=mc_xxh64(bm,MC_BITMAP_BYTES,h);
    h^=mc_xxh64(lens16,(size_t)nl*2,h);
    h^=mc_xxh64(tmp->p,tmp->len,h);
    size_t total=MC_BLOB_HDR+fml+MC_BITMAP_BYTES+(size_t)npresent*2+tmp->len;
    put(out,&q,4); put(out,&h,8); put(out,&fml,2);
    put(out,fmap,fml);
    put(out,bm,MC_BITMAP_BYTES);
    put(out,lens16,(size_t)nl*2);
    put(out,tmp->p,tmp->len);
    return total;
}
// back-compat wrapper: CABAC blocks (the original behavior).
size_t encode_chunk_blob(mc_codec_ctx *C, const u8 *chunk256, out_put_fn put, void *out){
    return encode_chunk_blob_codec(C,MC_BLOCKCODEC_CABAC,chunk256,put,out);
}
