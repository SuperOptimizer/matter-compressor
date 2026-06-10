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
