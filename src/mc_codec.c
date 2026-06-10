// ============================================================================
// mc_codec.c — matter-compressor block codec implementation. See mc_codec.h.
// ============================================================================
#include "mc_codec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mc_dct.h"          // mc_dct3_fwd / mc_dct3_inv / mc_dct_init
#include "mc_rangecoder.h"   // enc/dec block coefs + bit coders
#include "mc_klt_tab.h"      // trained 3D-LFNST secondary transform (Q14)

#define N3 (MC_BLK*MC_BLK*MC_BLK)

static float g_quality = 8.0f;
static int   g_max_err = 0;            // 0 = corrections off
// per-coefficient quant step table (quality * hf_weight), rebuilt when quality
// changes. powf per coefficient was 20%+ of encode AND decode time.
static float g_step_tab[N3];
static float g_step_q = -1.0f;
static void step_tab_build(void);
void  mc_set_quality(float q){ g_quality = q; step_tab_build(); }
float mc_get_quality(void){ return g_quality; }
void  mc_set_max_error(int tau){ g_max_err = tau<0?0:tau; }
int   mc_get_max_error(void){ return g_max_err; }
void  mc_codec_init(void){ mc_dct_init(); }

void mc_buf_put(mc_buf *b, const void *s, size_t n){
    if(b->len+n > b->cap){ size_t nc=b->cap?b->cap*2:1<<16; while(nc<b->len+n)nc*=2; b->p=realloc(b->p,nc); b->cap=nc; }
    memcpy(b->p+b->len,s,n); b->len+=n;
}

// frozen quant: dead-zone, step = quality*(1+L1freq)^MC_HF_EXP
static inline float hf_weight(int cz,int cy,int cx){ return powf(1.0f+(float)(cz+cy+cx), MC_HF_EXP); }
// ---- 3D-LFNST secondary transform (AV2 IST / VVC LFNST analog) -------------
// A trained KLT over the 4x4x4 low-band DCT corner, applied between the DCT
// and quantization; the decoder applies the transpose after dequantization.
// Corner outputs are eigenvalue-ordered; their quant steps reuse the corner's
// own band-weight profile sorted ascending (finest step on the strongest
// component), so total bit allocation matches the primary path.
// MEASURED NEUTRAL on scroll data (identical ratio/PSNR, max error -1..-5):
// the 3D DCT already decorrelates this texture; LFNST pays in video because
// directional intra-prediction residuals leave structure the DCT misses,
// which we don't have. Kept (off) with the trained table + mc_klt for
// future data regimes.
#define MC_LFNST 0
static int   g_klt_pos[64];          // corner raster positions, (L1, raster) order
static float g_klt_w[64];            // sorted corner hf weights (ascending)
static int   g_klt_ready=0;
static void klt_build(void){
    if(g_klt_ready) return;
    int n=0;
    for(int b=0;b<=9;++b)
        for(int cz=0;cz<4;++cz)for(int cy=0;cy<4;++cy)for(int cx=0;cx<4;++cx)
            if(cz+cy+cx==b) g_klt_pos[n++]=(cz*MC_BLK+cy)*MC_BLK+cx;
    for(int i=0;i<64;++i){
        int p=g_klt_pos[i];
        g_klt_w[i]=hf_weight(p>>8,(p>>4)&15,p&15);
    }
    for(int i=0;i<64;++i)for(int j=i+1;j<64;++j)
        if(g_klt_w[j]<g_klt_w[i]){ float t=g_klt_w[i]; g_klt_w[i]=g_klt_w[j]; g_klt_w[j]=t; }
    g_klt_ready=1;
}
static void klt_fwd(float *coef){
    float v[64], o[64];
    for(int i=0;i<64;++i) v[i]=coef[g_klt_pos[i]];
    for(int i=0;i<64;++i){
        float a=0;
        for(int j=0;j<64;++j) a+=(float)MC_KLT[i][j]*v[j];
        o[i]=a*(1.0f/16384.0f);
    }
    for(int i=0;i<64;++i) coef[g_klt_pos[i]]=o[i];
}
static void klt_inv(float *coef){
    float v[64], o[64];
    for(int i=0;i<64;++i) v[i]=coef[g_klt_pos[i]];
    for(int j=0;j<64;++j){
        float a=0;
        for(int i=0;i<64;++i) a+=(float)MC_KLT[i][j]*v[i];   // transpose
        o[j]=a*(1.0f/16384.0f);
    }
    for(int i=0;i<64;++i) coef[g_klt_pos[i]]=o[i];
}

static void step_tab_build(void){
    rc_prior_build(g_quality);
    if(g_step_q==g_quality) return;
    klt_build();
    for(int cz=0;cz<MC_BLK;++cz)for(int cy=0;cy<MC_BLK;++cy)for(int cx=0;cx<MC_BLK;++cx)
        g_step_tab[(cz*MC_BLK+cy)*MC_BLK+cx]=g_quality*hf_weight(cz,cy,cx);
#if MC_LFNST
    for(int i=0;i<64;++i) g_step_tab[g_klt_pos[i]]=g_quality*g_klt_w[i];
#endif
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

// ---- RDOQ: encoder-only rate-distortion level decision (MEASURED: no win) ----
// For each nonzero dead-zone level m, evaluate {0, m-1, m} against a rate model
// built from the trained priors and keep the min of D + lambda*R. Measured on
// PHercParis4 2.4um masked across lambda in {0.05..0.6}: every setting lands ON
// OR BELOW the plain dead-zone RD curve at iso-PSNR — the frozen dz=0.8 +
// band-weighted steps already encode a better allocation than flat-lambda
// thresholding. Kept (disabled) for reference / future revisit with a
// context-exact rate model.
#define MC_RDOQ 0
#define MC_AQ 0   // measured: variance-based QP reallocation lands below the
                  // dead-zone RD curve at iso-PSNR on scroll data (statistically
                  // homogeneous noise) — same lesson as RDOQ. Keep off.
#define MC_AQ_LO 60.0
#define MC_AQ_HI 900.0
#define MC_RDOQ_LAMBDA 0.20f     // J = D + lambda*step^2*bits
static float g_rd_sig0[NB_BANDS], g_rd_sig1[NB_BANDS];   // sig-flag bits by band
static float g_rd_mag[66];                                // magnitude+sign bits, m in [1,65]
static int   g_rd_ready=0;
static double rd_bits(double p){ if(p<1e-4)p=1e-4; if(p>1-1e-4)p=1-1e-4; return -log2(p); }
static void rdoq_init(void){
    if(g_rd_ready) return;
    for(int b=0;b<NB_BANDS;++b){
        double p0=0; for(int dn=0;dn<4;++dn) p0+=RC_PRIOR_SIG[b*4+dn]/4096.0;
        p0/=4.0;
        g_rd_sig0[b]=(float)rd_bits(p0); g_rd_sig1[b]=(float)rd_bits(1.0-p0);
    }
    for(int m=1;m<=65;++m){          // mirror enc_magnitude bit-for-bit
        double bits=0; int v=m-1,k=0,done=0;
        while(k<MAGCTX-1&&v>0){ bits+=rd_bits(1.0-RC_PRIOR_MAG[k]/4096.0); v-=1;k++;
            if(v==0){ bits+=rd_bits(RC_PRIOR_MAG[k]/4096.0); done=1; break; } }
        if(!done){
            if(v==0) bits+=rd_bits(RC_PRIOR_MAG[k]/4096.0);
            else { bits+=rd_bits(1.0-RC_PRIOR_MAG[k]/4096.0);
                   int nb=0,t=v+1; while(t>1){t>>=1;nb++;} bits+=2*nb+1; }
        }
        g_rd_mag[m]=(float)(bits+1.0);                     // +1: sign bypass
    }
    g_rd_ready=1;
}
// returns the RD-chosen magnitude for |coef|=a at quant step `step`, band b,
// given the dead-zone choice m (>=1).
static inline mc_i32 rdoq_level(float a, float step, int b, mc_i32 m){
    float lam=MC_RDOQ_LAMBDA*step*step;
    float bestJ=0; mc_i32 best=0;
    // candidate 0
    bestJ = a*a + lam*g_rd_sig0[b];
    // candidates m and m-1
    for(mc_i32 c=m; c>=1 && c>=m-1; --c){
        float rec=((float)c+0.2f)*step;                    // matches deq_one
        float d=a-rec;
        mc_i32 cm = c<=65?c:65;
        float J = d*d + lam*(g_rd_sig1[b]+g_rd_mag[cm]);
        if(J<bestJ){ bestJ=J; best=c; }
    }
    return best;
}

// ---- TCQ: VVC-style dependent (trellis-coded) quantization -----------------
// Two quantizers on the half-step lattice — Q0 recon = k*step, Q1 recon =
// (k-0.5)*step (k>=1) — switched by a 4-state machine driven by level parity.
// The admissible reconstruction set packs denser than a scalar quantizer
// (vector-quantization-like gain) at zero decoder cost beyond state tracking.
// Encoder runs a small Viterbi per block over the scan.
#define MC_TCQ 0   // MEASURED: lands 3-8% below the dead-zone RD curve at every
                   // lambda (0.08-0.2) — same failure mode as RDOQ: the lattice
                   // gain assumes dense high-rate coefficients; ours are sparse
                   // and sig-flag-dominated. Kept for reference.
#define MC_TCQ_LAMBDA 0.12f
static const uint8_t TCQ_NEXT[4][2]={{0,2},{2,0},{1,3},{3,1}};   // [state][lvl&1]
static inline int   tcq_useq1(int s){ return s>=2; }
static inline float tcq_recon(int lvl,int useq1,float step){
    if(!lvl) return 0.0f;
    float a=(float)(lvl<0?-lvl:lvl);
    float r=useq1? (a-0.5f)*step : a*step;
    return lvl<0?-r:r;
}
static void tcq_quantize(const float *coef, const uint16_t *scan, int n, mc_i32 *lv, float qs){
    rdoq_init();
    static _Thread_local int16_t cand_lv[16*16*16][4];
    static _Thread_local uint8_t bp[16*16*16][4];
    float J[4]={0,1e30f,1e30f,1e30f}, Jn[4];
    for(int p=0;p<n;++p){
        int idx=scan[p];
        int cz=idx>>8, cy=(idx>>4)&15, cx=idx&15;
        int b=(cz+cy+cx)*NB_BANDS/(3*MC_BLK); if(b>=NB_BANDS)b=NB_BANDS-1;
        float step=g_step_tab[idx]*qs;
        float lam=MC_TCQ_LAMBDA*step*step;
        float a=fabsf(coef[idx]);
        for(int s=0;s<4;++s) Jn[s]=1e30f;
        for(int s=0;s<4;++s){
            if(J[s]>=1e29f) continue;
            int q1=tcq_useq1(s);
            int k = q1 ? (int)(a/step+1.0f) : (int)(a/step+0.5f);
            if(k<0)k=0;
            int cands[3]={0,k,k+1};
            for(int ci=0;ci<3;++ci){
                int lvl=cands[ci]; if(ci&&lvl==0) continue;
                float rec=tcq_recon(lvl,q1,step), d=a-rec;
                float R = lvl? g_rd_sig1[b]+g_rd_mag[lvl<=65?lvl:65] : g_rd_sig0[b];
                float Jc=J[s]+d*d+lam*R;
                int ns=TCQ_NEXT[s][lvl&1];
                if(Jc<Jn[ns]){ Jn[ns]=Jc; bp[p][ns]=(uint8_t)s; cand_lv[p][ns]=(int16_t)lvl; }
            }
        }
        for(int s=0;s<4;++s) J[s]=Jn[s];
    }
    int s=0; for(int k2=1;k2<4;++k2) if(J[k2]<J[s]) s=k2;
    for(int p=n-1;p>=0;--p){
        int lvl=cand_lv[p][s];
        int idx=scan[p];
        lv[idx]= coef[idx]<0 ? -lvl : lvl;
        s=bp[p][s];
    }
}
static void tcq_dequant(const rc_i16 *ql, const uint16_t *scan, int n, float *coef, float qs){
    int s=0;
    for(int p=0;p<n;++p){
        int idx=scan[p]; int lvl=ql[idx];
        coef[idx]=tcq_recon(lvl,tcq_useq1(s),g_step_tab[idx]*qs);
        s=TCQ_NEXT[s][lvl&1];
    }
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
    ctx_t ctx[8]; for(int i=0;i<8;++i) ctx_init_ps(&ctx[i],RC_PRIOR_MASK[i],RC_SHIFT[3]);
    ctx_t cu[4];  for(int i=0;i<4;++i) ctx_init_ps(&cu[i],RC_PRIOR_MASKU[i],RC_SHIFT[4]);
    ctx_t ca[2];  for(int i=0;i<2;++i) ctx_init_ps(&ca[i],RC_PRIOR_MASKA[i],RC_SHIFT[5]);
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
    ctx_t ctx[8]; for(int i=0;i<8;++i) ctx_init_ps(&ctx[i],RC_PRIOR_MASK[i],RC_SHIFT[3]);
    ctx_t cu[4];  for(int i=0;i<4;++i) ctx_init_ps(&cu[i],RC_PRIOR_MASKU[i],RC_SHIFT[4]);
    ctx_t ca[2];  for(int i=0;i<2;++i) ctx_init_ps(&ca[i],RC_PRIOR_MASKA[i],RC_SHIFT[5]);
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

// block payload layout: ONE range-coded stream, nothing else. The stream starts
// with [mixed bit][corr bit][qpd 2 bits][dc 8 bits] all context-coded with
// trained priors (the old raw dc+flags bytes were ~5% of an average payload),
// then the mask bins (if mixed), the coefficients, and the corrections. flags bit0 =
// mixed block; the stream carries the mask bins (if mixed) then the coefficients.
// One stream = one flush (~5B) instead of two streams + a 2B mask length.
int mc_enc_block(const mc_u8 *vox, mc_buf *out, uint32_t *len_out){
    int n=N3, any=0; for(int i=0;i<n;++i) any|=vox[i];
    if(!any){ *len_out=0; return 0; }

    static _Thread_local float blk[N3], coef[N3];
    static _Thread_local mc_i32 lv[N3];
    long sum=0,cnt=0; for(int i=0;i<n;++i){ if(vox[i]){ sum+=vox[i]; cnt++; } }
    int dc = (int)((sum+cnt/2)/cnt);                  // DC over material only
    int nair = n-(int)cnt;                            // air = vox==0
    for(int i=0;i<n;++i) blk[i]=(float)((vox[i]?vox[i]:dc)-dc);
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
#if MC_LFNST
    step_tab_build(); klt_fwd(coef);
#endif
#if MC_RDOQ
    rdoq_init();
#endif
    // adaptive per-block QP: scale the quant step by local activity (variance of
    // material voxels). Flat (noise-dominated) blocks get a finer step, busy
    // blocks a coarser one; 2-bit delta signalled in flags bits 2-3.
    int qpd=0;
#if MC_AQ
    {
        double s1=0,s2=0;
        for(int i=0;i<n;++i) if(vox[i]){ double v=vox[i]; s1+=v; s2+=v*v; }
        double var = cnt? s2/cnt-(s1/cnt)*(s1/cnt) : 0;
        qpd = var<MC_AQ_LO ? 3 : var>MC_AQ_HI ? 1 : 0;     // 3 -> finer, 1 -> coarser
    }
#endif
    const float qp_scale[4]={1.0f,1.4f,1.0f,0.71f};
    float qs=qp_scale[qpd];
#if MC_TCQ
    { scanS_build(MC_BLK); tcq_quantize(coef,g_scanS[4],N3,lv,qs); }
#else
    for(int cz=0;cz<MC_BLK;++cz)for(int cy=0;cy<MC_BLK;++cy)for(int cx=0;cx<MC_BLK;++cx){
        int idx=(cz*MC_BLK+cy)*MC_BLK+cx; float step=g_step_tab[idx]*qs;
        mc_i32 m=quant_one(coef[idx],step);
#if MC_RDOQ
        if(m){
            int b=(cz+cy+cx)*NB_BANDS/(3*MC_BLK); if(b>=NB_BANDS)b=NB_BANDS-1;
            mc_i32 am=m<0?-m:m;
            mc_i32 r=rdoq_level(fabsf(coef[idx]),step,b,am);
            m = coef[idx]<0?-r:r;
        }
#endif
        lv[idx]=m;
    }
#endif
    static _Thread_local rc_i16 ql[N3];
    static _Thread_local rc_u8 scratch[N3*4+1024];
    for(int i=0;i<n;++i){ mc_i32 v=lv[i]; ql[i]=(rc_i16)(v>32767?32767:v<-32768?-32768:v); }

    // max-error corrections: locally reconstruct and list voxels with |err| > tau.
    static _Thread_local uint16_t cpos[N3]; static _Thread_local mc_i32 cdel[N3];
    int ncorr=0;
    if(g_max_err>0){
        static _Thread_local float rcoef[N3], rblk[N3];
#if MC_TCQ
        { scanS_build(MC_BLK); tcq_dequant(ql,g_scanS[4],N3,rcoef,qs); }
#else
        for(int idx=0;idx<N3;++idx) rcoef[idx]=deq_one(ql[idx],g_step_tab[idx]*qs);
#endif
#if MC_LFNST
        klt_inv(rcoef);
#endif
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
    {   // header bins: mixed, has-corr, qpd, dc (trained priors)
        ctx_t cf[4]; for(int i=0;i<4;++i) ctx_init_ps(&cf[i],RC_PRIOR_FLAG[i],RC_SHIFT[6]);
        ctx_t cd[8]; for(int i=0;i<8;++i) ctx_init_ps(&cd[i],RC_PRIOR_DC[i],RC_SHIFT[7]);
        RC_TRAIN(RCC_FLAG,0,nair>0);  enc_bit(&e,&cf[0],nair>0);
        RC_TRAIN(RCC_FLAG,1,ncorr>0); enc_bit(&e,&cf[1],ncorr>0);
        RC_TRAIN(RCC_FLAG,2,(qpd>>1)&1); enc_bit(&e,&cf[2],(qpd>>1)&1);
        RC_TRAIN(RCC_FLAG,3,qpd&1);      enc_bit(&e,&cf[3],qpd&1);
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
        ctx_t cf[4]; for(int i=0;i<4;++i) ctx_init_ps(&cf[i],RC_PRIOR_FLAG[i],RC_SHIFT[6]);
        ctx_t cd[8]; for(int i=0;i<8;++i) ctx_init_ps(&cd[i],RC_PRIOR_DC[i],RC_SHIFT[7]);
        flags |= dec_bit(&d,&cf[0]) ? 1 : 0;
        flags |= dec_bit(&d,&cf[1]) ? 2 : 0;
        int q1=dec_bit(&d,&cf[2]), q0=dec_bit(&d,&cf[3]);
        flags |= ((q1<<1)|q0)<<2;
        for(int b=0;b<8;++b) dc=(dc<<1)|dec_bit(&d,&cd[b]);
    }
    if(flags&1) dec_blockmask(&d,air);
    else        memset(air,0,n);
    int ext[3]; dec_block_coefs_ext(&d,ql,MC_BLK,ext);
    static _Thread_local float coef[N3], blk[N3];
    step_tab_build();
    const float qp_scale[4]={1.0f,1.4f,1.0f,0.71f};
    float qs=qp_scale[(flags>>2)&3];
    int ez=ext[0],ey=ext[1],ex=ext[2];
    if(ez<0 && !(flags&1) && !(flags&2)){                   // constant block: dc fill
        memset(dst,(mc_u8)dc,n); return;
    }
    (void)ey;(void)ex;
#if MC_TCQ
    { scanS_build(MC_BLK); tcq_dequant(ql,g_scanS[4],N3,coef,qs); }
#else
    for(int idx=0;idx<N3;++idx) coef[idx]=deq_one(ql[idx],g_step_tab[idx]*qs);
#endif
#if MC_LFNST
    klt_inv(coef);
#endif
    mc_dct3_inv(coef,blk);
    for(int i=0;i<n;++i){
        int v = air[i] ? 0 : (int)lrintf(blk[i])+dc;  // mask-restore: air -> exactly 0
        if(v<0)v=0; if(v>255)v=255; dst[i]=(mc_u8)v;
    }
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
