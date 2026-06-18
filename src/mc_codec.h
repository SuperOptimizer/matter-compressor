// mc_codec.h — cross-module surface of the block codec (DCT-16 + range coder +
// xxhash + block/chunk encode-decode). The full mc_codec_ctx and all codec
// internals stay PRIVATE to mc_codec.c; only the symbols other modules call are
// declared here. The chunk-blob encoder lives in mc_codec.c because it is the one
// place that reaches into the ctx's encode scratch.
#ifndef MC_CODEC_H
#define MC_CODEC_H

#include "matter_compressor.h"
#include <stdint.h>
#include <stddef.h>

// XXH64 (integrity hashing; shared by archive + reader + verify).
uint64_t mc_xxh64(const void *data, size_t len, uint64_t seed);

// decode one 16^3 block payload (codec dispatch; only CABAC remains).
void mc_dec_block_codec(mc_codec_ctx *C, uint32_t codec,
                        const mc_u8 *p, uint32_t plen, mc_u8 *dst);

// gather a 16^3 block out of a dense 256^3 chunk buffer; returns nonzero if any
// voxel is set.
int gather_blk256(const mc_u8 *chunk, int bz, int by, int bx, mc_u8 *dst);

#endif  // MC_CODEC_H
