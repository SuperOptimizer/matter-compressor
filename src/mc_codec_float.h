// ============================================================================
// mc_codec_float.h — FLOAT-ONLY transform + quant + max-error core for the
// matter-compressor block codec.
//
// This is the standalone float replacement for the integer Q14 path in
// mc_codec.c. It is deliberately ISOLATED (header-only, self-contained, depends
// only on libc + the codec's two tuning constants) so it can be reviewed and
// unit-tested without touching mc_codec.c / mc_archive.c while other work is in
// flight. At integration time these functions replace, one-for-one:
//
//     mc_dct3_fwd / mc_dct3_inv / mc_dct3_inv_i32   -> mc_dctf3_fwd / mc_dctf3_inv
//     quant_one  / deq_one                          -> mc_quantf_one / mc_deqf_one
//     integer "err - tau" Exp-Golomb corrections    -> mc_maxerr_build / _apply
//
// What changes vs the integer path, and WHY:
//   * The DCT runs in f32, no Q14 fixed-point. Removes the three lrintf()/(float)
//     boundary conversions per 3D transform and the per-line >>14 rounding, so
//     reconstruction stops accumulating fixed-point rounding noise (=> slightly
//     higher PSNR at matched rate). f32's 24-bit mantissa covers the 16^3 / u8
//     dynamic range with margin, so the integer "range-safe" concern is moot.
//     `-ffast-math` (FMA contraction, reassociation) is an expected build mode.
//   * Max-error corrections store the EXACT f32 residual per out-of-tolerance
//     voxel instead of an integer (err - tau) delta. Corrected voxels land
//     dead-on => strictly lower max-error and MAE.
//
// DETERMINISM CAVEAT (the one real trade — see mc_codec.c notes / mcpp review
// C1): f32 + fast-math is NOT bit-reproducible across ISAs / opt levels / FMA
// contraction. The max-error pass is therefore BEST-EFFORT: it is computed
// against THIS build's own reconstruction (mc_maxerr_build is fed the decoded
// block produced by mc_dctf3_inv here). Encode and decode within the same build
// agree exactly, so the tau bound holds for a same-build pipeline. A decoder on
// a different ISA may diverge by some delta and breach tau silently. The air
// mask stays integer (handled in mc_codec.c, outside this file) and remains
// bit-exact regardless. If a hard cross-decoder tau is ever needed, pin
// -ffp-contract=off + fixed evaluation order on the surface decode path only.
// ============================================================================
#ifndef MC_CODEC_FLOAT_H
#define MC_CODEC_FLOAT_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

// Tuning constants. Mirror src/matter_compressor.h; guarded so this header can
// compile standalone (selftest) or alongside the codec without redefinition.
#ifndef MC_BLK
#define MC_BLK         16        // DCT block edge
#endif
#ifndef MC_DZ_FRAC
#define MC_DZ_FRAC     0.80f     // dead-zone width fraction
#endif
#ifndef MC_HF_EXP
#define MC_HF_EXP      0.65f     // HF quant power-law exponent
#endif

#define MC_F_N   MC_BLK
#define MC_F_N3  (MC_BLK*MC_BLK*MC_BLK)

// ---------------------------------------------------------------------------
// Orthonormal separable DCT-II cosine matrix (f32), built once.
//   cmf[k][n] = ck * cos(pi*(2n+1)*k / (2N)),  ck = sqrt(1/N) (k=0) else sqrt(2/N)
// Orthonormal => the inverse (DCT-III) is the transpose: no separate scaling.
// Same normalization as the integer path's table, so quality/step calibration
// carries over unchanged.
// ---------------------------------------------------------------------------
static float g_mc_cmf[MC_F_N][MC_F_N];
static int   g_mc_cmf_ready = 0;

static void mc_dctf_init(void){
    if(g_mc_cmf_ready) return;
    for(int k=0;k<MC_F_N;++k){
        double ck = (k==0)? sqrt(1.0/MC_F_N) : sqrt(2.0/MC_F_N);
        for(int n=0;n<MC_F_N;++n)
            g_mc_cmf[k][n] = (float)(ck*cos(M_PI*(2.0*n+1.0)*k/(2.0*MC_F_N)));
    }
    g_mc_cmf_ready = 1;
}

// 1D forward DCT-II of one length-N line: out[k] = sum_n cmf[k][n]*in[n].
// Even/odd partial-butterfly form (halves the MACs); autovectorizes cleanly
// under -ffast-math. N=16 only.
static inline void mc_dctf1d_fwd(const float *restrict in, float *restrict out){
    const int S=MC_F_N, H=S/2;
    float s[MC_F_N/2], d[MC_F_N/2];
    for(int n=0;n<H;++n){ s[n]=in[n]+in[S-1-n]; d[n]=in[n]-in[S-1-n]; }
    for(int k=0;k<S;k+=2){                 // even rows depend on s[]
        float a=0.0f; for(int n=0;n<H;++n) a += g_mc_cmf[k][n]*s[n];
        out[k]=a;
    }
    for(int k=1;k<S;k+=2){                  // odd rows depend on d[]
        float a=0.0f; for(int n=0;n<H;++n) a += g_mc_cmf[k][n]*d[n];
        out[k]=a;
    }
}
// 1D inverse (DCT-III), sparse-aware (skips zero coefficients — post-dequant
// lines are mostly a few low-frequency nonzeros): out[n] = sum_k cmf[k][n]*in[k].
static inline void mc_dctf1d_inv(const float *restrict in, float *restrict out){
    const int S=MC_F_N, H=S/2;
    float e[MC_F_N/2], o[MC_F_N/2];
    for(int n=0;n<H;++n){ e[n]=0.0f; o[n]=0.0f; }
    for(int k=0;k<S;k+=2){ float v=in[k]; if(v) for(int n=0;n<H;++n) e[n]+=g_mc_cmf[k][n]*v; }
    for(int k=1;k<S;k+=2){ float v=in[k]; if(v) for(int n=0;n<H;++n) o[n]+=g_mc_cmf[k][n]*v; }
    for(int n=0;n<H;++n){ out[n]=e[n]+o[n]; out[S-1-n]=e[n]-o[n]; }
}

// transform every contiguous length-N line of an N*N*N buffer (last axis),
// skipping all-zero lines. Out-of-place.
static inline void mc_linesf_fwd(const float *restrict src, float *restrict dst){
    const int S=MC_F_N; float ol[MC_F_N];
    for(int off=0;off<S*S;++off){
        const float *v=src+(size_t)off*S; float *o=dst+(size_t)off*S;
        int nz=0; for(int i=0;i<S;++i) if(v[i]!=0.0f){nz=1;break;}
        if(!nz){ for(int i=0;i<S;++i) o[i]=0.0f; continue; }
        mc_dctf1d_fwd(v,ol); for(int i=0;i<S;++i) o[i]=ol[i];
    }
}
static inline void mc_linesf_inv(const float *restrict src, float *restrict dst){
    const int S=MC_F_N; float ol[MC_F_N];
    for(int off=0;off<S*S;++off){
        const float *v=src+(size_t)off*S; float *o=dst+(size_t)off*S;
        int nz=0; for(int i=0;i<S;++i) if(v[i]!=0.0f){nz=1;break;}
        if(!nz){ for(int i=0;i<S;++i) o[i]=0.0f; continue; }
        mc_dctf1d_inv(v,ol); for(int i=0;i<S;++i) o[i]=ol[i];
    }
}

// cache-blocked rotate (z,y,x) -> (x,z,y): dst[(x*S+z)*S+y] = src[(z*S+y)*S+x].
// Three rotates return a 3D pass sequence to the original axis order, same
// structure as the integer path's mc_rot.
#define MC_F_ROT_TILE 8
static inline void mc_rotf(const float *restrict src, float *restrict dst){
    const int S=MC_F_N;
    for(int zt=0; zt<S; zt+=MC_F_ROT_TILE)
    for(int xt=0; xt<S; xt+=MC_F_ROT_TILE)
        for(int z=zt; z<zt+MC_F_ROT_TILE; ++z)
        for(int x=xt; x<xt+MC_F_ROT_TILE; ++x){
            const float *sp = src + ((size_t)z*S)*S + x;
            float *dp = dst + ((size_t)x*S+z)*S;
            for(int y=0;y<S;++y) dp[y]=sp[(size_t)y*S];
        }
}

// 3D forward / inverse on a 16^3 block. Each pass: transform contiguous lines,
// then rotate; three rotates restore (z,y,x). Caller supplies 2 scratch buffers
// of MC_F_N3 floats (a,b) to keep this allocation-free on the hot path.
static inline void mc_dctf3_fwd(const float *restrict blk, float *restrict coef,
                                float *restrict a, float *restrict b){
    mc_linesf_fwd(blk,a); mc_rotf(a,b);
    mc_linesf_fwd(b,a);   mc_rotf(a,b);
    mc_linesf_fwd(b,a);   mc_rotf(a,coef);
}
static inline void mc_dctf3_inv(const float *restrict coef, float *restrict blk,
                                float *restrict a, float *restrict b){
    mc_linesf_inv(coef,a); mc_rotf(a,b);
    mc_linesf_inv(b,a);    mc_rotf(a,b);
    mc_linesf_inv(b,a);    mc_rotf(a,blk);
}

// ---------------------------------------------------------------------------
// Dead-zone quant / dequant (pure f32; identical math to mc_codec.c's
// quant_one/deq_one, just with the integer codec's boundary casts removed).
//   level  = sign(c) * floor(|c|/step + (1 - MC_DZ_FRAC))   for |c| >= dz, else 0
//   recon  = sign(l) * step * (|l| - 1 + MC_DZ_FRAC + 0.40) = sign(l)*step*(|l|+0.20)
// step = quality * (1 + L1freq)^MC_HF_EXP per coefficient (see mc_stepf_build).
// ---------------------------------------------------------------------------
static inline float mc_hf_weightf(int cz,int cy,int cx){
    return powf(1.0f+(float)(cz+cy+cx), MC_HF_EXP);
}
// Build per-coefficient step and reciprocal-step tables for a given quality.
static inline void mc_stepf_build(float quality, float step_tab[MC_F_N3],
                                  float rstep_tab[MC_F_N3]){
    for(int cz=0;cz<MC_F_N;++cz)for(int cy=0;cy<MC_F_N;++cy)for(int cx=0;cx<MC_F_N;++cx){
        int i=(cz*MC_F_N+cy)*MC_F_N+cx;
        float st = quality*mc_hf_weightf(cz,cy,cx);
        step_tab[i]  = st;
        rstep_tab[i] = 1.0f/st;
    }
}
static inline int32_t mc_quantf_one(float c, float step){
    float dz=MC_DZ_FRAC*step, a=fabsf(c); int32_t lv=0;
    if(a>=dz) lv=(int32_t)((a-dz)/step+1.0f);
    return c<0.0f ? -lv : lv;
}
static inline float mc_deqf_one(int32_t lv, float step){
    if(!lv) return 0.0f;
    float a=(float)(lv<0?-lv:lv);
    float r=(a-1.0f)*step + MC_DZ_FRAC*step + 0.40f*step;
    return lv<0 ? -r : r;
}

// ---------------------------------------------------------------------------
// Max-error corrections — EXACT f32 residuals (best-effort tau bound; see the
// determinism caveat at the top of this file).
//
// Blob layout: [u32 count][ {u32 index, f32 delta} * count ]. `decoded` is the
// reconstruction produced by THIS build's mc_dctf3_inv (clamp/dc already
// applied by the caller into a float buffer). On apply, each delta is ADDED to
// the decoded voxel, outside the transform/quant path. Sparse: only voxels with
// |orig - decoded| > tau are stored.
//
// Returns the number of corrections written. `out` must hold at least
// 4 + 8*count bytes; pass a max-sized scratch (4 + 8*MC_F_N3) to be safe.
// ---------------------------------------------------------------------------
static inline uint32_t mc_maxerr_build(const float *orig, const float *decoded,
                                       size_t n, float tau, uint8_t *out){
    uint8_t *p = out + 4;        // reserve the count slot
    uint32_t count = 0;
    for(size_t i=0;i<n;++i){
        float e = orig[i] - decoded[i];
        if(e>tau || e<-tau){
            uint32_t idx=(uint32_t)i, db; memcpy(&db,&e,4);
            p[0]=(uint8_t)idx;       p[1]=(uint8_t)(idx>>8);
            p[2]=(uint8_t)(idx>>16); p[3]=(uint8_t)(idx>>24);
            p[4]=(uint8_t)db;        p[5]=(uint8_t)(db>>8);
            p[6]=(uint8_t)(db>>16);  p[7]=(uint8_t)(db>>24);
            p+=8; ++count;
        }
    }
    out[0]=(uint8_t)count;       out[1]=(uint8_t)(count>>8);
    out[2]=(uint8_t)(count>>16); out[3]=(uint8_t)(count>>24);
    return count;
}
// Apply a correction blob in place to `decoded`. Bounds-checked against both the
// blob length and n (hardened: malformed / attacker-controlled input is clamped,
// never writes out of range). Returns bytes consumed (0 if header truncated).
static inline size_t mc_maxerr_apply(const uint8_t *blob, size_t len,
                                     float *decoded, size_t n){
    if(len<4) return 0;
    uint32_t count = (uint32_t)blob[0] | ((uint32_t)blob[1]<<8) |
                     ((uint32_t)blob[2]<<16) | ((uint32_t)blob[3]<<24);
    size_t off=4;
    for(uint32_t k=0;k<count;++k){
        if(off+8>len) break;                 // malformed -> stop
        uint32_t idx = (uint32_t)blob[off] | ((uint32_t)blob[off+1]<<8) |
                       ((uint32_t)blob[off+2]<<16) | ((uint32_t)blob[off+3]<<24);
        uint32_t db  = (uint32_t)blob[off+4] | ((uint32_t)blob[off+5]<<8) |
                       ((uint32_t)blob[off+6]<<16) | ((uint32_t)blob[off+7]<<24);
        float d; memcpy(&d,&db,4);
        if(idx<n) decoded[idx]+=d;
        off+=8;
    }
    return off;
}

#endif  // MC_CODEC_FLOAT_H

// ============================================================================
// Standalone self-test. Build:
//   cc -O2 -ffast-math -DMC_CODEC_FLOAT_SELFTEST -x c src/mc_codec_float.h -lm -o /tmp/mcf && /tmp/mcf
// Exercises: DCT fwd/inv identity round-trip; full codec round-trip (dc-remove
// -> fwd -> quant -> deq -> inv -> dc-add -> clamp) with best-effort max-error.
// ============================================================================
#ifdef MC_CODEC_FLOAT_SELFTEST
#include <stdio.h>
#include <stdlib.h>

static uint32_t mcf_rng(uint32_t *s){ uint32_t x=*s; x^=x<<13; x^=x>>17; x^=x<<5; *s=x; return x; }

int main(void){
    mc_dctf_init();
    const int n = MC_F_N3;

    // --- 1. DCT identity round-trip on random data ---------------------------
    float a[MC_F_N3], b[MC_F_N3];   // scratch
    float in[MC_F_N3], coef[MC_F_N3], rec[MC_F_N3];
    uint32_t s=0x12345678u;
    for(int i=0;i<n;++i) in[i] = (float)((int)(mcf_rng(&s)%512) - 256);
    mc_dctf3_fwd(in,coef,a,b);
    mc_dctf3_inv(coef,rec,a,b);
    double maxd=0; for(int i=0;i<n;++i){ double e=fabs((double)in[i]-rec[i]); if(e>maxd)maxd=e; }
    printf("DCT identity round-trip: max abs error = %.3e %s\n",
           maxd, maxd<1e-2?"OK":"FAIL");

    // --- 2. Full codec round-trip with quant + best-effort max-error ---------
    float quality = 6.0f; int tau = 4;
    float step[MC_F_N3], rstep[MC_F_N3];
    mc_stepf_build(quality, step, rstep);

    // synthetic u8 block: smooth ramp + texture (mimics scroll material).
    uint8_t orig[MC_F_N3];
    for(int z=0;z<MC_F_N;++z)for(int y=0;y<MC_F_N;++y)for(int x=0;x<MC_F_N;++x){
        int i=(z*MC_F_N+y)*MC_F_N+x;
        int v = 90 + 6*x + 4*y + 3*z + (int)(mcf_rng(&s)%24);
        orig[i] = (uint8_t)(v<0?0:(v>255?255:v));
    }
    // dc removal (block mean, like the codec), forward, quant.
    long sum=0; for(int i=0;i<n;++i) sum+=orig[i];
    int dc=(int)((sum + n/2)/n);
    float fin[MC_F_N3];
    for(int i=0;i<n;++i) fin[i]=(float)((int)orig[i]-dc);
    mc_dctf3_fwd(fin,coef,a,b);
    int32_t lvl[MC_F_N3];
    for(int i=0;i<n;++i) lvl[i]=mc_quantf_one(coef[i],step[i]);
    // dequant, inverse, dc add -> float reconstruction (pre-correction).
    float deq[MC_F_N3], spat[MC_F_N3], dec[MC_F_N3];
    for(int i=0;i<n;++i) deq[i]=mc_deqf_one(lvl[i],step[i]);
    mc_dctf3_inv(deq,spat,a,b);
    for(int i=0;i<n;++i){ float v=spat[i]+(float)dc; dec[i]=v<0?0:(v>255?255:v); }
    // measure pre-correction error, then build/apply best-effort corrections.
    float forig[MC_F_N3];
    for(int i=0;i<n;++i) forig[i]=(float)orig[i];
    double se=0,ae=0,mx=0;
    for(int i=0;i<n;++i){ double e=forig[i]-roundf(dec[i]); se+=e*e; ae+=fabs(e); if(fabs(e)>mx)mx=fabs(e); }
    printf("pre-correction:  PSNR=%.2f dB  MAE=%.4f  maxerr=%.0f\n",
           10*log10(255.0*255.0/(se/n)), ae/n, mx);

    uint8_t blob[4+8*MC_F_N3];
    uint32_t nc = mc_maxerr_build(forig, dec, n, (float)tau, blob);
    mc_maxerr_apply(blob, sizeof blob, dec, n);
    se=ae=mx=0;
    for(int i=0;i<n;++i){
        float vf=roundf(dec[i]); vf=vf<0?0:(vf>255?255:vf);
        double e=forig[i]-vf; se+=e*e; ae+=fabs(e); if(fabs(e)>mx)mx=fabs(e);
    }
    printf("post-correction: PSNR=%.2f dB  MAE=%.4f  maxerr=%.0f  (%u fixes, tau=%d)\n",
           10*log10(255.0*255.0/(se/n)), ae/n, mx, nc, tau);
    printf("max-error bound honored (same build): %s\n", mx<=tau?"OK":"FAIL");
    return 0;
}
#endif
