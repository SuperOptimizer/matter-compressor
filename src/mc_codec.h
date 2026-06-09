// ============================================================================
// mc_codec.h — matter-compressor block codec.
//
// Compresses a 16^3 u8 voxel block: integer separable DCT-16 + dead-zone quant +
// CABAC range coder. Mask-aware: air voxels (value 0) are harmonically air-filled
// before the DCT and force-zeroed on decode; the air boundary is coded once per
// 256^3 chunk as a coherent surface ("chunk mask").
//
// Pure transform — no I/O, no archive/volume knowledge. Operates on gathered 16^3
// blocks + a prepared air mask handed in by the caller. The only runtime parameter
// is `quality` (the quant base step; higher = more compression, lower fidelity).
// ============================================================================
#ifndef MC_CODEC_H
#define MC_CODEC_H
#include <stdint.h>
#include <stddef.h>

typedef uint8_t  mc_u8;
typedef int32_t  mc_i32;

#define MC_BLK    16     // DCT block edge
#define MC_CHUNK  256    // chunk-mask edge (16 blocks); air surface coded per 256^3 chunk

// frozen quant constants (tuned; see matter-compressor design notes)
#define MC_DZ_FRAC     0.80f   // dead-zone width fraction
#define MC_HF_EXP      0.65f   // HF quant power-law: step = quality*(1+L1freq)^MC_HF_EXP
#define MC_FILL_SWEEPS 8       // harmonic air-fill sweeps before the DCT

void  mc_set_quality(float quality);   // the one runtime parameter
float mc_get_quality(void);
void  mc_codec_init(void);             // one-time: build the DCT tables

// growable output byte buffer the codec appends block payloads to.
typedef struct { mc_u8 *p; size_t len, cap; } mc_buf;
void  mc_buf_put(mc_buf *b, const void *s, size_t n);

// encode one 16^3 block. vox = source voxels (z,y,x raster; air=0). rair = the 16^3
// reconstructed air mask (air=1), sliced from the chunk mask. Appends payload to out,
// sets *len_out. Returns 1 if coded (nonzero), 0 if all-zero (no payload).
int   mc_enc_block(const mc_u8 *vox, const mc_u8 *rair, mc_buf *out, uint32_t *len_out);
// decode one block payload into dst (16^3). rair: air -> output 0.
void  mc_dec_block(const mc_u8 *payload, const mc_u8 *rair, mc_u8 *dst);

// chunk-mask surface coder (256^3 air mask, air=1).
uint32_t mc_enc_chunkmask(const mc_u8 *mask256, mc_u8 *buf, size_t cap);
void     mc_dec_chunkmask(const mc_u8 *buf, size_t len, mc_u8 *mask256);

#endif
