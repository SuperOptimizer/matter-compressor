// ============================================================================
// mc_codec.h — matter-compressor block codec.
//
// Compresses a 16^3 u8 voxel block: integer separable DCT-16 + dead-zone quant +
// CABAC range coder. Mask-aware: air voxels (value 0) are harmonically air-filled
// before the DCT and force-zeroed on decode. The air mask is SELF-CONTAINED per
// block: mixed blocks carry their own context-coded 16^3 mask in the payload
// (all-material blocks pay 0 bytes; all-air blocks are absent entirely). This
// makes a single block independently decodable — no chunk-level mask pass.
//
// Pure transform — no I/O, no archive/volume knowledge. The only runtime
// parameter is `quality` (the quant base step; higher = more compression,
// lower fidelity).
// ============================================================================
#ifndef MC_CODEC_H
#define MC_CODEC_H
#include <stdint.h>
#include <stddef.h>

typedef uint8_t  mc_u8;
typedef int32_t  mc_i32;

#define MC_BLK    16     // DCT block edge
#define MC_CHUNK  256    // archive chunk edge (16 blocks)

// frozen quant constants (tuned; see matter-compressor design notes)
#define MC_DZ_FRAC     0.80f   // dead-zone width fraction
#define MC_HF_EXP      0.65f   // HF quant power-law: step = quality*(1+L1freq)^MC_HF_EXP
#define MC_FILL_SWEEPS 3       // SOR air-fill sweeps before the DCT (omega=1.6)

void  mc_set_quality(float quality);   // the main runtime parameter
float mc_get_quality(void);
// Optional guaranteed max-error bound (near-lossless mode): after transform
// coding, the encoder codes sparse corrections for every voxel whose error
// exceeds tau, so |orig - decoded| <= tau for all material voxels. 0 = off
// (default). Corrections are self-contained in the block payload (the decoder
// does not need to know tau). Encode-side cost: one extra inverse DCT per block.
void  mc_set_max_error(int tau);
int   mc_get_max_error(void);
void  mc_codec_init(void);             // one-time: build the DCT tables
// Override the trained context priors (q=1 / q=12 endpoint tables, u16[8][32]
// each) — used by per-volume prior blobs. NULL,NULL restores the baked tables.
// Process-global; set before encode/decode threads run.
void  mc_codec_set_priors(const uint16_t *plo, const uint16_t *phi);

// growable output byte buffer the codec appends block payloads to.
typedef struct { mc_u8 *p; size_t len, cap; } mc_buf;
void  mc_buf_put(mc_buf *b, const void *s, size_t n);

// Optional decode-side deblocking: H.264-style clamped 1D filter across the
// 16-voxel block faces of an assembled volume (any box, dims in voxels).
// Strength scales with the quality the data was coded at; air voxels (0) are
// never touched. Call AFTER assembling decoded blocks into a contiguous region.
void  mc_deblock(mc_u8 *vol, int nz, int ny, int nx, float quality);

// encode one 16^3 block (z,y,x raster; air = value 0). Appends payload to out,
// sets *len_out. Returns 1 if coded (nonzero), 0 if all-zero (no payload).
int   mc_enc_block(const mc_u8 *vox, mc_buf *out, uint32_t *len_out);
// decode one block payload of `plen` bytes into dst (16^3). Self-contained
// (mask in payload); plen comes from the chunk's block-length table.
void  mc_dec_block(const mc_u8 *payload, uint32_t plen, mc_u8 *dst);

#endif
