// mc_sample — point sampling over blocked u8 volumes.
//
// The layer between storage (mc_cache / mc_volume / plain arrays) and
// geometry consumers (mc_render, ML feature extraction): give it a source
// of 16^3 blocks and it answers "what is the value at (z,y,x)?" for float
// coordinates, with nearest-neighbor or trilinear filtering.
//
// Coordinates are (z,y,x) voxel space of the level being sampled, matching
// the rest of matter-compressor (volume-cartographer's Vec3f is (x,y,z) —
// swap when adapting). Out-of-bounds and invalid points sample as 0.
//
// A sampler memoizes the blocks it touched (direct-mapped, small), so
// coherent access patterns (surface rendering, crop fills) cost ~one block
// fetch per 4096 voxels, not one per sample. Samplers are NOT thread-safe:
// one per thread (they are cheap; mc_render's parallel path does this).
#ifndef MC_SAMPLE_H
#define MC_SAMPLE_H
#include <stdint.h>
#include <stddef.h>

// ---------------------------------------------------------------------------
// source: anything that can produce 16^3 blocks
// ---------------------------------------------------------------------------
// block() returns a pointer to the 4096-byte block at block coords
// (bz,by,bx), in z-major (z*256 + y*16 + x) layout:
//   - a stable pointer of its own (e.g. mc_cache arena), or
//   - `tmp` after filling it (sources that must copy/decode), or
//   - NULL for "no data here" (absent / pure air) — sampled as 0.
// `tmp` is 4096 bytes of sampler-owned scratch, valid until the next
// block() call on the same sampler. block() is called outside any lock the
// sampler holds; it must tolerate out-of-range block coords (return NULL).
typedef struct mc_sample_src mc_sample_src;
struct mc_sample_src {
    void *ud;                             // binding-private
    int aux, aux2;                        // binding-private (lod, flags, ...)
    const uint8_t *(*block)(const mc_sample_src *src,
                            int bz, int by, int bx, uint8_t *tmp);
    int nz, ny, nx;                       // voxel dims of the sampled level
    // Optional direct path: when set, samplers address voxels straight off
    // this base pointer (voxel (z,y,x) at dense[z*dsy + y*dsx + x]) and
    // never call block(). mc_sample_src_dense sets it; blocked sources
    // leave it NULL.
    const uint8_t *dense;
    size_t dsy, dsx;
};

// Ready-made sources:
struct mc_cache;
// mc_cache-backed (zero-copy arena pointers; decodes misses synchronously).
// `lod` selects the level; pass that level's voxel dims.
mc_sample_src mc_sample_src_cache(struct mc_cache *c, int lod,
                                  int nz, int ny, int nx);
// Dense C-order u8 array (no copies; blocks are synthesized in `tmp`).
mc_sample_src mc_sample_src_dense(const uint8_t *vox, int nz, int ny, int nx);

// ---------------------------------------------------------------------------
// sampler
// ---------------------------------------------------------------------------
typedef struct mc_sampler mc_sampler;
typedef enum { MC_FILTER_NEAREST = 0, MC_FILTER_TRILINEAR = 1 } mc_filter;

mc_sampler *mc_sampler_new(const mc_sample_src *src);
void        mc_sampler_free(mc_sampler *s);
// Drop memoized block pointers (call when the source was invalidated, e.g.
// after mc_cache_thaw/update cycles replaced chunks).
void        mc_sampler_reset(mc_sampler *s);

// One sample at (z,y,x). Out-of-bounds / NaN -> 0.
float mc_sample_point(mc_sampler *s, float z, float y, float x, mc_filter f);

// Batch: n points, zyx[i*3+{0,1,2}] = (z,y,x). Points with any coordinate
// < 0 (volume-cartographer's invalid marker) or NaN write 0.
void mc_sample_points(mc_sampler *s, const float *zyx, size_t n,
                      mc_filter f, float *out);
void mc_sample_points_u8(mc_sampler *s, const float *zyx, size_t n,
                         mc_filter f, uint8_t *out);

#endif
