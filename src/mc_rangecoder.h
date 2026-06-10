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
