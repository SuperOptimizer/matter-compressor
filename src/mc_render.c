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
// mc_sample — point sampling over blocked u8 volumes (see header)
// ============================================================================
#include <unistd.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__SSE4_1__)
#include <immintrin.h>
#endif

#define MC_S_MEMO 256   // covers an oblique 1024-px row's block working set

// Pointer-only memo entry (16B, was 4112B). The block() source returns a STABLE
// pointer the memo caches directly: the cache/arena source (mc_volume interactive
// path) returns an arena/zero pointer valid for the frozen frame; dense bypasses
// mc_s_block entirely. The only source that synthesizes into a scratch buffer is
// the CLI blocking path -- for THAT, owns_ptr is 0 and we use a per-sampler buf[]
// (allocated only then) so cached pointers don't alias one shared scratch.
typedef struct {
    int bz, by, bx;             // -1 = empty
    const uint8_t *ptr;         // NULL = known-absent (sampled as 0)
} mc_s_memo;

struct mc_sampler {
    mc_sample_src src;
    int nbz, nby, nbx;
    int lbz, lby, lbx;          // last block touched (ray-coherence cache)
    const uint8_t *lptr;
    int rbz, rby, rbx, rres;    // last residency probe (ray-coherence cache)
    uint8_t (*scratch)[4096];   // per-entry synth buffers, only if !owns_ptr (else NULL)
    mc_s_memo m[MC_S_MEMO];
};

static inline const uint8_t *mc_s_block(mc_sampler *s, int bz, int by, int bx) {
    if (bz == s->lbz && by == s->lby && bx == s->lbx) return s->lptr;
    unsigned h = ((unsigned)bz * 73856093u) ^ ((unsigned)by * 19349663u) ^
                 ((unsigned)bx * 83492791u);
    unsigned slot = h & (MC_S_MEMO - 1);
    mc_s_memo *e = &s->m[slot];
    if (!(e->bz == bz && e->by == by && e->bx == bx)) {
        e->bz = bz; e->by = by; e->bx = bx;
        // owns_ptr sources return a stable pointer (cache arena); cache it directly.
        // Otherwise synthesize into this slot's own scratch buffer.
        uint8_t *tmp = s->scratch ? s->scratch[slot] : NULL;
        e->ptr = s->src.block(&s->src, bz, by, bx, tmp);
    }
    s->lbz = bz; s->lby = by; s->lbx = bx; s->lptr = e->ptr;
    return e->ptr;
}

static inline float mc_s_voxel(mc_sampler *s, int z, int y, int x) {
    if ((unsigned)z >= (unsigned)s->src.nz ||
        (unsigned)y >= (unsigned)s->src.ny ||
        (unsigned)x >= (unsigned)s->src.nx) return 0.0f;
    if (s->src.dense)
        return (float)s->src.dense[(size_t)z * s->src.dsy +
                                   (size_t)y * s->src.dsx + (size_t)x];
    const uint8_t *b = mc_s_block(s, z >> 4, y >> 4, x >> 4);
    return b ? (float)b[((z & 15) << 8) | ((y & 15) << 4) | (x & 15)] : 0.0f;
}

static inline float mc_s_nearest(mc_sampler *s, float z, float y, float x) {
    return mc_s_voxel(s, (int)floorf(z + 0.5f), (int)floorf(y + 0.5f),
                      (int)floorf(x + 0.5f));
}

// Interior cell: all 8 corners inside ONE block; caller verified bounds and
// no-straddle. Ints + fractions precomputed (no refloor / rebounds).
static inline float mc_s_cell_in(mc_sampler *s, int z0, int y0, int x0,
                                 float dz, float dy, float dx) {
    const uint8_t *b = mc_s_block(s, z0 >> 4, y0 >> 4, x0 >> 4);
    if (!b) return 0.0f;
    const uint8_t *p = b + (((z0 & 15) << 8) | ((y0 & 15) << 4) | (x0 & 15));
    float c00 = (float)p[0]   + ((float)p[1]   - (float)p[0])   * dx;
    float c01 = (float)p[16]  + ((float)p[17]  - (float)p[16])  * dx;
    float c10 = (float)p[256] + ((float)p[257] - (float)p[256]) * dx;
    float c11 = (float)p[272] + ((float)p[273] - (float)p[272]) * dx;
    float c0 = c00 + (c01 - c00) * dy;
    float c1 = c10 + (c11 - c10) * dy;
    return c0 + (c1 - c0) * dz;
}

// Straddling cell: crosses >=1 block face (caller verified bounds). Fetch each
// DISTINCT block once -- 2 for a face straddle (the common case), 4 edge, 8
// corner -- instead of 8 per-corner lookups.
static inline float mc_s_cell_straddle(mc_sampler *s, int z0, int y0, int x0,
                                       float dz, float dy, float dx) {
    const int sz = (z0 & 15) == 15, sy = (y0 & 15) == 15, sx = (x0 & 15) == 15;
    const int bz = z0 >> 4, by = y0 >> 4, bx = x0 >> 4;
    // Single-axis x straddle: the dominant case on scanlines (consecutive lanes
    // walk x). Two block fetches, column x=15 of B and column x=0 of B+1 --
    // no 8-pointer dedup table.
    if (sx && !sy && !sz) {
        const uint8_t *b0 = mc_s_block(s, bz, by, bx);
        const uint8_t *b1 = mc_s_block(s, bz, by, bx + 1);
        const int o00 = ((z0 & 15) << 8) | ((y0 & 15) << 4);
        const int o01 = o00 + 16, o10 = o00 + 256, o11 = o00 + 272;
        float c000 = b0 ? (float)b0[o00 | 15] : 0.0f;
        float c001 = b1 ? (float)b1[o00]      : 0.0f;
        float c010 = b0 ? (float)b0[o01 | 15] : 0.0f;
        float c011 = b1 ? (float)b1[o01]      : 0.0f;
        float c100 = b0 ? (float)b0[o10 | 15] : 0.0f;
        float c101 = b1 ? (float)b1[o10]      : 0.0f;
        float c110 = b0 ? (float)b0[o11 | 15] : 0.0f;
        float c111 = b1 ? (float)b1[o11]      : 0.0f;
        float c00 = c000 + (c001 - c000) * dx;
        float c01 = c010 + (c011 - c010) * dx;
        float c10 = c100 + (c101 - c100) * dx;
        float c11 = c110 + (c111 - c110) * dx;
        float c0 = c00 + (c01 - c00) * dy;
        float c1 = c10 + (c11 - c10) * dy;
        return c0 + (c1 - c0) * dz;
    }
    // B[a][b][c] = block holding corner (z0+a, y0+b, x0+c); reuse pointers
    // along non-straddling axes so each distinct block is fetched once.
    const uint8_t *B[2][2][2];
    B[0][0][0] = mc_s_block(s, bz, by, bx);
    B[0][0][1] = sx ? mc_s_block(s, bz, by, bx + 1) : B[0][0][0];
    B[0][1][0] = sy ? mc_s_block(s, bz, by + 1, bx) : B[0][0][0];
    B[0][1][1] = sy ? (sx ? mc_s_block(s, bz, by + 1, bx + 1) : B[0][1][0])
                    : B[0][0][1];
    if (sz) {
        B[1][0][0] = mc_s_block(s, bz + 1, by, bx);
        B[1][0][1] = sx ? mc_s_block(s, bz + 1, by, bx + 1) : B[1][0][0];
        B[1][1][0] = sy ? mc_s_block(s, bz + 1, by + 1, bx) : B[1][0][0];
        B[1][1][1] = sy ? (sx ? mc_s_block(s, bz + 1, by + 1, bx + 1) : B[1][1][0])
                        : B[1][0][1];
    } else {
        B[1][0][0] = B[0][0][0]; B[1][0][1] = B[0][0][1];
        B[1][1][0] = B[0][1][0]; B[1][1][1] = B[0][1][1];
    }
    const int oz0 = (z0 & 15) << 8, oz1 = ((z0 + 1) & 15) << 8;
    const int oy0 = (y0 & 15) << 4, oy1 = ((y0 + 1) & 15) << 4;
    const int ox0 = x0 & 15,        ox1 = (x0 + 1) & 15;
    #define MC_S_C(a, b, c, oz, oy, ox) \
        (B[a][b][c] ? (float)B[a][b][c][(oz) | (oy) | (ox)] : 0.0f)
    float c000 = MC_S_C(0, 0, 0, oz0, oy0, ox0);
    float c001 = MC_S_C(0, 0, 1, oz0, oy0, ox1);
    float c010 = MC_S_C(0, 1, 0, oz0, oy1, ox0);
    float c011 = MC_S_C(0, 1, 1, oz0, oy1, ox1);
    float c100 = MC_S_C(1, 0, 0, oz1, oy0, ox0);
    float c101 = MC_S_C(1, 0, 1, oz1, oy0, ox1);
    float c110 = MC_S_C(1, 1, 0, oz1, oy1, ox0);
    float c111 = MC_S_C(1, 1, 1, oz1, oy1, ox1);
    #undef MC_S_C
    float c00 = c000 + (c001 - c000) * dx;
    float c01 = c010 + (c011 - c010) * dx;
    float c10 = c100 + (c101 - c100) * dx;
    float c11 = c110 + (c111 - c110) * dx;
    float c0 = c00 + (c01 - c00) * dy;
    float c1 = c10 + (c11 - c10) * dy;
    return c0 + (c1 - c0) * dz;
}

// In-bounds blocked cell, ints + fracs precomputed: dispatch straddle/interior.
static inline float mc_s_cell(mc_sampler *s, int z0, int y0, int x0,
                              float dz, float dy, float dx) {
    return ((z0 & 15) == 15 || (y0 & 15) == 15 || (x0 & 15) == 15)
        ? mc_s_cell_straddle(s, z0, y0, x0, dz, dy, dx)
        : mc_s_cell_in(s, z0, y0, x0, dz, dy, dx);
}

static inline float mc_s_trilinear(mc_sampler *s, float z, float y, float x) {
    float zf = floorf(z), yf = floorf(y), xf = floorf(x);
    int z0 = (int)zf, y0 = (int)yf, x0 = (int)xf;
    float dz = z - zf, dy = y - yf, dx = x - xf;
    // dense fast path: direct strided gather, only a bounds check
    if (s->src.dense &&
        (unsigned)z0 < (unsigned)(s->src.nz - 1) &&
        (unsigned)y0 < (unsigned)(s->src.ny - 1) &&
        (unsigned)x0 < (unsigned)(s->src.nx - 1)) {
        const size_t sy = s->src.dsy, sx = s->src.dsx;
        const uint8_t *p = s->src.dense + (size_t)z0 * sy + (size_t)y0 * sx + x0;
        float c00 = (float)p[0]      + ((float)p[1]        - (float)p[0])      * dx;
        float c01 = (float)p[sx]     + ((float)p[sx + 1]   - (float)p[sx])     * dx;
        float c10 = (float)p[sy]     + ((float)p[sy + 1]   - (float)p[sy])     * dx;
        float c11 = (float)p[sy + sx] + ((float)p[sy + sx + 1] - (float)p[sy + sx]) * dx;
        float c0 = c00 + (c01 - c00) * dy;
        float c1 = c10 + (c11 - c10) * dy;
        return c0 + (c1 - c0) * dz;
    }
    // blocked in-bounds cell: interior (one block) or straddle (dedup'd blocks)
    if (!s->src.dense &&
        (unsigned)z0 < (unsigned)(s->src.nz - 1) &&
        (unsigned)y0 < (unsigned)(s->src.ny - 1) &&
        (unsigned)x0 < (unsigned)(s->src.nx - 1))
        return mc_s_cell(s, z0, y0, x0, dz, dy, dx);
    // slow path (volume edge / dense edge): block/bounds handled per corner
    float c000 = mc_s_voxel(s, z0, y0, x0);
    float c001 = mc_s_voxel(s, z0, y0, x0 + 1);
    float c010 = mc_s_voxel(s, z0, y0 + 1, x0);
    float c011 = mc_s_voxel(s, z0, y0 + 1, x0 + 1);
    float c100 = mc_s_voxel(s, z0 + 1, y0, x0);
    float c101 = mc_s_voxel(s, z0 + 1, y0, x0 + 1);
    float c110 = mc_s_voxel(s, z0 + 1, y0 + 1, x0);
    float c111 = mc_s_voxel(s, z0 + 1, y0 + 1, x0 + 1);
    float c00 = c000 + (c001 - c000) * dx;
    float c01 = c010 + (c011 - c010) * dx;
    float c10 = c100 + (c101 - c100) * dx;
    float c11 = c110 + (c111 - c110) * dx;
    float c0 = c00 + (c01 - c00) * dy;
    float c1 = c10 + (c11 - c10) * dy;
    return c0 + (c1 - c0) * dz;
}

// NaN test that survives -ffast-math (where compilers delete x!=x and isnan):
// sign-cleared bits above the +inf pattern <=> exponent all-ones, mantissa != 0.
static inline int mc_s_isnan(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    return (u & 0x7FFFFFFFu) > 0x7F800000u;
}

static inline float mc_s_sample(mc_sampler *s, float z, float y, float x,
                                mc_filter f) {
    if (mc_s_isnan(z) || mc_s_isnan(y) || mc_s_isnan(x)) return 0.0f;
    return f == MC_FILTER_NEAREST ? mc_s_nearest(s, z, y, x)
                                  : mc_s_trilinear(s, z, y, x);
}

// All-zero block for masked SIMD lanes: a straddling lane points here during
// the group gather (its lerp result is overwritten scalar after), and an
// absent-block lane's lerp over zeros IS its correct value (samples as 0).
__attribute__((unused)) static const uint8_t mc_s_zero4k[4096];   // unused in no-SIMD builds

// ---------------------------------------------------------------------------
// 4-wide trilinear (ray-step batching for the compositors)
// ---------------------------------------------------------------------------
// Sample 4 positions at once. Lanes that qualify for a fast path are
// gathered and lerped with NEON; anything else (edges, absent blocks,
// non-aarch64) falls back to the scalar sampler per lane. Uses separate
// mul+add (no fma), so every lane is bit-identical to mc_s_trilinear.
#if defined(__aarch64__)
static inline float32x4_t mc_s_lerp8x4(const uint8_t *p0, const uint8_t *p1,
                                       const uint8_t *p2, const uint8_t *p3,
                                       size_t sy, size_t sx,
                                       float32x4_t dz, float32x4_t dy,
                                       float32x4_t dx) {
    uint16x4_t g00 = vdup_n_u16(0), g01 = vdup_n_u16(0);
    uint16x4_t g10 = vdup_n_u16(0), g11 = vdup_n_u16(0);
    g00 = vld1_lane_u16((const uint16_t *)(const void *)p0, g00, 0);
    g00 = vld1_lane_u16((const uint16_t *)(const void *)p1, g00, 1);
    g00 = vld1_lane_u16((const uint16_t *)(const void *)p2, g00, 2);
    g00 = vld1_lane_u16((const uint16_t *)(const void *)p3, g00, 3);
    g01 = vld1_lane_u16((const uint16_t *)(const void *)(p0 + sx), g01, 0);
    g01 = vld1_lane_u16((const uint16_t *)(const void *)(p1 + sx), g01, 1);
    g01 = vld1_lane_u16((const uint16_t *)(const void *)(p2 + sx), g01, 2);
    g01 = vld1_lane_u16((const uint16_t *)(const void *)(p3 + sx), g01, 3);
    g10 = vld1_lane_u16((const uint16_t *)(const void *)(p0 + sy), g10, 0);
    g10 = vld1_lane_u16((const uint16_t *)(const void *)(p1 + sy), g10, 1);
    g10 = vld1_lane_u16((const uint16_t *)(const void *)(p2 + sy), g10, 2);
    g10 = vld1_lane_u16((const uint16_t *)(const void *)(p3 + sy), g10, 3);
    g11 = vld1_lane_u16((const uint16_t *)(const void *)(p0 + sy + sx), g11, 0);
    g11 = vld1_lane_u16((const uint16_t *)(const void *)(p1 + sy + sx), g11, 1);
    g11 = vld1_lane_u16((const uint16_t *)(const void *)(p2 + sy + sx), g11, 2);
    g11 = vld1_lane_u16((const uint16_t *)(const void *)(p3 + sy + sx), g11, 3);
    uint16x8_t w00 = vmovl_u8(vreinterpret_u8_u16(g00));
    uint16x8_t w01 = vmovl_u8(vreinterpret_u8_u16(g01));
    uint16x8_t w10 = vmovl_u8(vreinterpret_u8_u16(g10));
    uint16x8_t w11 = vmovl_u8(vreinterpret_u8_u16(g11));
#define MC_S_F32E(w) vcvtq_f32_u32(vmovl_u16(vuzp1_u16(vget_low_u16(w), vget_high_u16(w))))
#define MC_S_F32O(w) vcvtq_f32_u32(vmovl_u16(vuzp2_u16(vget_low_u16(w), vget_high_u16(w))))
    float32x4_t f000 = MC_S_F32E(w00), f001 = MC_S_F32O(w00);
    float32x4_t f010 = MC_S_F32E(w01), f011 = MC_S_F32O(w01);
    float32x4_t f100 = MC_S_F32E(w10), f101 = MC_S_F32O(w10);
    float32x4_t f110 = MC_S_F32E(w11), f111 = MC_S_F32O(w11);
#undef MC_S_F32E
#undef MC_S_F32O
    float32x4_t c00 = vaddq_f32(f000, vmulq_f32(vsubq_f32(f001, f000), dx));
    float32x4_t c01 = vaddq_f32(f010, vmulq_f32(vsubq_f32(f011, f010), dx));
    float32x4_t c10 = vaddq_f32(f100, vmulq_f32(vsubq_f32(f101, f100), dx));
    float32x4_t c11 = vaddq_f32(f110, vmulq_f32(vsubq_f32(f111, f110), dx));
    float32x4_t c0 = vaddq_f32(c00, vmulq_f32(vsubq_f32(c01, c00), dy));
    float32x4_t c1 = vaddq_f32(c10, vmulq_f32(vsubq_f32(c11, c10), dy));
    return vaddq_f32(c0, vmulq_f32(vsubq_f32(c1, c0), dz));
}
#elif defined(__SSE4_1__)
static inline uint16_t mc_s_ld16(const uint8_t *p) {
    uint16_t v; __builtin_memcpy(&v, p, 2); return v;
}
static inline __m128 mc_s_lerp8x4(const uint8_t *p0, const uint8_t *p1,
                                  const uint8_t *p2, const uint8_t *p3,
                                  size_t sy, size_t sx,
                                  __m128 dz, __m128 dy, __m128 dx) {
    // per corner-row: 4 samples' (c0,c1) byte pairs in u16 lanes 0..3
    __m128i z = _mm_setzero_si128();
    __m128i g00 = _mm_insert_epi16(z, mc_s_ld16(p0), 0);
    g00 = _mm_insert_epi16(g00, mc_s_ld16(p1), 1);
    g00 = _mm_insert_epi16(g00, mc_s_ld16(p2), 2);
    g00 = _mm_insert_epi16(g00, mc_s_ld16(p3), 3);
    __m128i g01 = _mm_insert_epi16(z, mc_s_ld16(p0 + sx), 0);
    g01 = _mm_insert_epi16(g01, mc_s_ld16(p1 + sx), 1);
    g01 = _mm_insert_epi16(g01, mc_s_ld16(p2 + sx), 2);
    g01 = _mm_insert_epi16(g01, mc_s_ld16(p3 + sx), 3);
    __m128i g10 = _mm_insert_epi16(z, mc_s_ld16(p0 + sy), 0);
    g10 = _mm_insert_epi16(g10, mc_s_ld16(p1 + sy), 1);
    g10 = _mm_insert_epi16(g10, mc_s_ld16(p2 + sy), 2);
    g10 = _mm_insert_epi16(g10, mc_s_ld16(p3 + sy), 3);
    __m128i g11 = _mm_insert_epi16(z, mc_s_ld16(p0 + sy + sx), 0);
    g11 = _mm_insert_epi16(g11, mc_s_ld16(p1 + sy + sx), 1);
    g11 = _mm_insert_epi16(g11, mc_s_ld16(p2 + sy + sx), 2);
    g11 = _mm_insert_epi16(g11, mc_s_ld16(p3 + sy + sx), 3);
    // split even bytes (x0 corner) / odd bytes (x1 corner) -> u32 -> f32
    const __m128i me = _mm_set_epi8(-1, -1, -1, 6, -1, -1, -1, 4,
                                    -1, -1, -1, 2, -1, -1, -1, 0);
    const __m128i mo = _mm_set_epi8(-1, -1, -1, 7, -1, -1, -1, 5,
                                    -1, -1, -1, 3, -1, -1, -1, 1);
#define MC_S_F32E(g) _mm_cvtepi32_ps(_mm_shuffle_epi8(g, me))
#define MC_S_F32O(g) _mm_cvtepi32_ps(_mm_shuffle_epi8(g, mo))
    __m128 f000 = MC_S_F32E(g00), f001 = MC_S_F32O(g00);
    __m128 f010 = MC_S_F32E(g01), f011 = MC_S_F32O(g01);
    __m128 f100 = MC_S_F32E(g10), f101 = MC_S_F32O(g10);
    __m128 f110 = MC_S_F32E(g11), f111 = MC_S_F32O(g11);
#undef MC_S_F32E
#undef MC_S_F32O
    __m128 c00 = _mm_add_ps(f000, _mm_mul_ps(_mm_sub_ps(f001, f000), dx));
    __m128 c01 = _mm_add_ps(f010, _mm_mul_ps(_mm_sub_ps(f011, f010), dx));
    __m128 c10 = _mm_add_ps(f100, _mm_mul_ps(_mm_sub_ps(f101, f100), dx));
    __m128 c11 = _mm_add_ps(f110, _mm_mul_ps(_mm_sub_ps(f111, f110), dx));
    __m128 c0 = _mm_add_ps(c00, _mm_mul_ps(_mm_sub_ps(c01, c00), dy));
    __m128 c1 = _mm_add_ps(c10, _mm_mul_ps(_mm_sub_ps(c11, c10), dy));
    return _mm_add_ps(c0, _mm_mul_ps(_mm_sub_ps(c1, c0), dz));
}

#if defined(__AVX2__) && !defined(MC_S_NO_TRI8)
// 8-wide variant for the x86-64-v3 fleet (Zen 3/4/5, 12th-gen+ Intel).
#define MC_S_HAVE_TRI8 1
static inline __m256i mc_s_g8(const uint8_t *const p[8], size_t off) {
    __m128i lo = _mm_setzero_si128(), hi = lo;
    lo = _mm_insert_epi16(lo, mc_s_ld16(p[0] + off), 0);
    lo = _mm_insert_epi16(lo, mc_s_ld16(p[1] + off), 1);
    lo = _mm_insert_epi16(lo, mc_s_ld16(p[2] + off), 2);
    lo = _mm_insert_epi16(lo, mc_s_ld16(p[3] + off), 3);
    hi = _mm_insert_epi16(hi, mc_s_ld16(p[4] + off), 0);
    hi = _mm_insert_epi16(hi, mc_s_ld16(p[5] + off), 1);
    hi = _mm_insert_epi16(hi, mc_s_ld16(p[6] + off), 2);
    hi = _mm_insert_epi16(hi, mc_s_ld16(p[7] + off), 3);
    return _mm256_set_m128i(hi, lo);
}
static inline __m256 mc_s_lerp8x8(const uint8_t *const p[8],
                                  size_t sy, size_t sx,
                                  __m256 dz, __m256 dy, __m256 dx) {
    __m256i g00 = mc_s_g8(p, 0),  g01 = mc_s_g8(p, sx);
    __m256i g10 = mc_s_g8(p, sy), g11 = mc_s_g8(p, sy + sx);
    // even byte of each u16 pair = x0 corner, odd = x1 (per 128-bit half)
    const __m256i me = _mm256_broadcastsi128_si256(
        _mm_set_epi8(-1, -1, -1, 6, -1, -1, -1, 4,
                     -1, -1, -1, 2, -1, -1, -1, 0));
    const __m256i mo = _mm256_broadcastsi128_si256(
        _mm_set_epi8(-1, -1, -1, 7, -1, -1, -1, 5,
                     -1, -1, -1, 3, -1, -1, -1, 1));
#define MC_S_F32E(g) _mm256_cvtepi32_ps(_mm256_shuffle_epi8(g, me))
#define MC_S_F32O(g) _mm256_cvtepi32_ps(_mm256_shuffle_epi8(g, mo))
    __m256 f000 = MC_S_F32E(g00), f001 = MC_S_F32O(g00);
    __m256 f010 = MC_S_F32E(g01), f011 = MC_S_F32O(g01);
    __m256 f100 = MC_S_F32E(g10), f101 = MC_S_F32O(g10);
    __m256 f110 = MC_S_F32E(g11), f111 = MC_S_F32O(g11);
#undef MC_S_F32E
#undef MC_S_F32O
    __m256 c00 = _mm256_add_ps(f000, _mm256_mul_ps(_mm256_sub_ps(f001, f000), dx));
    __m256 c01 = _mm256_add_ps(f010, _mm256_mul_ps(_mm256_sub_ps(f011, f010), dx));
    __m256 c10 = _mm256_add_ps(f100, _mm256_mul_ps(_mm256_sub_ps(f101, f100), dx));
    __m256 c11 = _mm256_add_ps(f110, _mm256_mul_ps(_mm256_sub_ps(f111, f110), dx));
    __m256 c0 = _mm256_add_ps(c00, _mm256_mul_ps(_mm256_sub_ps(c01, c00), dy));
    __m256 c1 = _mm256_add_ps(c10, _mm256_mul_ps(_mm256_sub_ps(c11, c10), dy));
    return _mm256_add_ps(c0, _mm256_mul_ps(_mm256_sub_ps(c1, c0), dz));
}
#endif  /* __AVX2__ */
#endif

static inline void mc_s_tri4(mc_sampler *s, const float *pz, const float *py,
                             const float *px, float *out) {
#if defined(__aarch64__)
    float32x4_t zv = vld1q_f32(pz), yv = vld1q_f32(py), xv = vld1q_f32(px);
    float32x4_t zf = vrndmq_f32(zv), yf = vrndmq_f32(yv), xf = vrndmq_f32(xv);
    int32x4_t zi = vcvtq_s32_f32(zf), yi = vcvtq_s32_f32(yf),
              xi = vcvtq_s32_f32(xf);
    // all-lanes in-bounds check: 0 <= c < n-1 per axis
    uint32x4_t ok = vcltq_u32(vreinterpretq_u32_s32(zi),
                              vdupq_n_u32((unsigned)(s->src.nz - 1)));
    ok = vandq_u32(ok, vcltq_u32(vreinterpretq_u32_s32(yi),
                                 vdupq_n_u32((unsigned)(s->src.ny - 1))));
    ok = vandq_u32(ok, vcltq_u32(vreinterpretq_u32_s32(xi),
                                 vdupq_n_u32((unsigned)(s->src.nx - 1))));
    if (vminvq_u32(ok) != 0) {
        float32x4_t dz = vsubq_f32(zv, zf), dy = vsubq_f32(yv, yf),
                    dx = vsubq_f32(xv, xf);
        if (s->src.dense) {
            int32_t z0[4], y0[4], x0[4];
            vst1q_s32(z0, zi); vst1q_s32(y0, yi); vst1q_s32(x0, xi);
            const size_t sy = s->src.dsy, sx = s->src.dsx;
            const uint8_t *base = s->src.dense;
            vst1q_f32(out, mc_s_lerp8x4(
                base + (size_t)z0[0] * sy + (size_t)y0[0] * sx + x0[0],
                base + (size_t)z0[1] * sy + (size_t)y0[1] * sx + x0[1],
                base + (size_t)z0[2] * sy + (size_t)y0[2] * sx + x0[2],
                base + (size_t)z0[3] * sy + (size_t)y0[3] * sx + x0[3],
                sy, sx, dz, dy, dx));
            return;
        }
        // Masked group (see mc_s_tri8): one SIMD lerp for everyone; absent ->
        // zero block (correct as-is), straddlers -> zero block + scalar fixup.
        {
            int32_t z0[4], y0[4], x0[4];
            vst1q_s32(z0, zi); vst1q_s32(y0, yi); vst1q_s32(x0, xi);
            const uint8_t *b[4];
            unsigned strad = 0;
            for (int k = 0; k < 4; k++) {
                if ((z0[k] & 15) == 15 || (y0[k] & 15) == 15 ||
                    (x0[k] & 15) == 15) {
                    strad |= 1u << k;
                    b[k] = mc_s_zero4k;
                    continue;
                }
                const uint8_t *bk =
                    mc_s_block(s, z0[k] >> 4, y0[k] >> 4, x0[k] >> 4);
                b[k] = bk ? bk + (((z0[k] & 15) << 8) | ((y0[k] & 15) << 4) |
                                  (x0[k] & 15))
                          : mc_s_zero4k;
            }
            vst1q_f32(out, mc_s_lerp8x4(b[0], b[1], b[2], b[3],
                                        256, 16, dz, dy, dx));
            if (strad) {
                float dzs[4], dys[4], dxs[4];
                vst1q_f32(dzs, dz); vst1q_f32(dys, dy); vst1q_f32(dxs, dx);
                for (int k = 0; k < 4; k++)
                    if (strad & (1u << k))
                        out[k] = mc_s_cell_straddle(s, z0[k], y0[k], x0[k],
                                                    dzs[k], dys[k], dxs[k]);
            }
        }
        return;
    }
#endif
#if defined(__SSE4_1__) && !defined(__aarch64__)
    __m128 zv = _mm_loadu_ps(pz), yv = _mm_loadu_ps(py), xv = _mm_loadu_ps(px);
    __m128 zf = _mm_floor_ps(zv), yf = _mm_floor_ps(yv), xf = _mm_floor_ps(xv);
    __m128i zi = _mm_cvttps_epi32(zf), yi = _mm_cvttps_epi32(yf),
            xi = _mm_cvttps_epi32(xf);
    // all-lanes 0 <= c < n-1 (signed compares; negatives fail the >= 0 side)
    __m128i ok = _mm_and_si128(
        _mm_cmpgt_epi32(zi, _mm_set1_epi32(-1)),
        _mm_cmpgt_epi32(_mm_set1_epi32(s->src.nz - 1), zi));
    ok = _mm_and_si128(ok, _mm_and_si128(
        _mm_cmpgt_epi32(yi, _mm_set1_epi32(-1)),
        _mm_cmpgt_epi32(_mm_set1_epi32(s->src.ny - 1), yi)));
    ok = _mm_and_si128(ok, _mm_and_si128(
        _mm_cmpgt_epi32(xi, _mm_set1_epi32(-1)),
        _mm_cmpgt_epi32(_mm_set1_epi32(s->src.nx - 1), xi)));
    if (_mm_movemask_ps(_mm_castsi128_ps(ok)) == 0xF) {
        __m128 dz = _mm_sub_ps(zv, zf), dy = _mm_sub_ps(yv, yf),
               dx = _mm_sub_ps(xv, xf);
        int32_t z0[4], y0[4], x0[4];
        _mm_storeu_si128((__m128i *)z0, zi);
        _mm_storeu_si128((__m128i *)y0, yi);
        _mm_storeu_si128((__m128i *)x0, xi);
        if (s->src.dense) {
            const size_t sy = s->src.dsy, sx = s->src.dsx;
            const uint8_t *base = s->src.dense;
            _mm_storeu_ps(out, mc_s_lerp8x4(
                base + (size_t)z0[0] * sy + (size_t)y0[0] * sx + x0[0],
                base + (size_t)z0[1] * sy + (size_t)y0[1] * sx + x0[1],
                base + (size_t)z0[2] * sy + (size_t)y0[2] * sx + x0[2],
                base + (size_t)z0[3] * sy + (size_t)y0[3] * sx + x0[3],
                sy, sx, dz, dy, dx));
            return;
        }
        // Masked group (see mc_s_tri8): one SIMD lerp for everyone; absent ->
        // zero block (correct as-is), straddlers -> zero block + scalar fixup.
        const uint8_t *b[4];
        unsigned strad = 0;
        for (int k = 0; k < 4; k++) {
            if ((z0[k] & 15) == 15 || (y0[k] & 15) == 15 || (x0[k] & 15) == 15) {
                strad |= 1u << k;
                b[k] = mc_s_zero4k;
                continue;
            }
            const uint8_t *bk =
                mc_s_block(s, z0[k] >> 4, y0[k] >> 4, x0[k] >> 4);
            b[k] = bk ? bk + (((z0[k] & 15) << 8) | ((y0[k] & 15) << 4) |
                              (x0[k] & 15))
                      : mc_s_zero4k;
        }
        _mm_storeu_ps(out, mc_s_lerp8x4(b[0], b[1], b[2], b[3],
                                        256, 16, dz, dy, dx));
        if (strad) {
            float dzs[4], dys[4], dxs[4];
            _mm_storeu_ps(dzs, dz);
            _mm_storeu_ps(dys, dy);
            _mm_storeu_ps(dxs, dx);
            for (int k = 0; k < 4; k++)
                if (strad & (1u << k))
                    out[k] = mc_s_cell_straddle(s, z0[k], y0[k], x0[k],
                                                dzs[k], dys[k], dxs[k]);
        }
        return;
    }
#endif
    out[0] = mc_s_trilinear(s, pz[0], py[0], px[0]);
    out[1] = mc_s_trilinear(s, pz[1], py[1], px[1]);
    out[2] = mc_s_trilinear(s, pz[2], py[2], px[2]);
    out[3] = mc_s_trilinear(s, pz[3], py[3], px[3]);
}

#ifdef MC_S_HAVE_TRI8
// 8 positions at once (AVX2). Same fallback discipline as mc_s_tri4.
static inline void mc_s_tri8(mc_sampler *s, const float *pz, const float *py,
                             const float *px, float *out) {
    __m256 zv = _mm256_loadu_ps(pz), yv = _mm256_loadu_ps(py),
           xv = _mm256_loadu_ps(px);
    __m256 zf = _mm256_floor_ps(zv), yf = _mm256_floor_ps(yv),
           xf = _mm256_floor_ps(xv);
    __m256i zi = _mm256_cvttps_epi32(zf), yi = _mm256_cvttps_epi32(yf),
            xi = _mm256_cvttps_epi32(xf);
    __m256i ok = _mm256_and_si256(
        _mm256_cmpgt_epi32(zi, _mm256_set1_epi32(-1)),
        _mm256_cmpgt_epi32(_mm256_set1_epi32(s->src.nz - 1), zi));
    ok = _mm256_and_si256(ok, _mm256_and_si256(
        _mm256_cmpgt_epi32(yi, _mm256_set1_epi32(-1)),
        _mm256_cmpgt_epi32(_mm256_set1_epi32(s->src.ny - 1), yi)));
    ok = _mm256_and_si256(ok, _mm256_and_si256(
        _mm256_cmpgt_epi32(xi, _mm256_set1_epi32(-1)),
        _mm256_cmpgt_epi32(_mm256_set1_epi32(s->src.nx - 1), xi)));
    if (_mm256_movemask_ps(_mm256_castsi256_ps(ok)) == 0xFF) {
        __m256 dz = _mm256_sub_ps(zv, zf), dy = _mm256_sub_ps(yv, yf),
               dx = _mm256_sub_ps(xv, xf);
        int32_t z0[8], y0[8], x0[8];
        _mm256_storeu_si256((__m256i *)z0, zi);
        _mm256_storeu_si256((__m256i *)y0, yi);
        _mm256_storeu_si256((__m256i *)x0, xi);
        const uint8_t *b[8];
        if (s->src.dense) {
            const size_t sy = s->src.dsy, sx = s->src.dsx;
            for (int k = 0; k < 8; k++)
                b[k] = s->src.dense + (size_t)z0[k] * sy +
                       (size_t)y0[k] * sx + x0[k];
            _mm256_storeu_ps(out, mc_s_lerp8x8(b, sy, sx, dz, dy, dx));
            return;
        }
        // Masked group: every lane goes through ONE SIMD gather+lerp. Interior
        // lanes use their real block pointer (absent block -> the zero block,
        // whose lerp IS the correct 0). Straddling lanes also point at the zero
        // block for the gather and are overwritten scalar after -- so a group
        // with one straddler costs one SIMD lerp + one scalar cell, not 8
        // scalar cells (the old all-or-nothing fallback).
        unsigned strad = 0;
        for (int k = 0; k < 8; k++) {
            if ((z0[k] & 15) == 15 || (y0[k] & 15) == 15 || (x0[k] & 15) == 15) {
                strad |= 1u << k;
                b[k] = mc_s_zero4k;
                continue;
            }
            const uint8_t *bk =
                mc_s_block(s, z0[k] >> 4, y0[k] >> 4, x0[k] >> 4);
            b[k] = bk ? bk + (((z0[k] & 15) << 8) | ((y0[k] & 15) << 4) |
                              (x0[k] & 15))
                      : mc_s_zero4k;
        }
        _mm256_storeu_ps(out, mc_s_lerp8x8(b, 256, 16, dz, dy, dx));
        if (strad) {
            float dzs[8], dys[8], dxs[8];
            _mm256_storeu_ps(dzs, dz);
            _mm256_storeu_ps(dys, dy);
            _mm256_storeu_ps(dxs, dx);
            for (int k = 0; k < 8; k++)
                if (strad & (1u << k))
                    out[k] = mc_s_cell_straddle(s, z0[k], y0[k], x0[k],
                                                dzs[k], dys[k], dxs[k]);
        }
        return;
    }
    mc_s_tri4(s, pz, py, px, out);
    mc_s_tri4(s, pz + 4, py + 4, px + 4, out + 4);
}
#endif


#define BLK 16
#define BLKB 4096

// ---------------------------------------------------------------------------
// sources
// ---------------------------------------------------------------------------
static const uint8_t *cache_block(const mc_sample_src *src,
                                  int bz, int by, int bx, uint8_t *tmp) {
    (void)tmp;
    // mc_cache_get decodes misses synchronously and returns an arena pointer
    // (non-owning; same stability contract as VC's BlockCache). In frozen
    // tick-phase a miss returns NULL — sampled as 0, recorded as feedback.
    return mc_cache_get((mc_cache *)src->ud, src->aux, bz, by, bx);
}

mc_sample_src mc_sample_src_cache(struct mc_cache *c, int lod,
                                  int nz, int ny, int nx) {
    mc_sample_src s = {0};
    s.ud = c; s.aux = lod; s.block = cache_block;
    s.owns_ptr = 1;                       // cache_block returns stable arena pointers
    s.nz = nz; s.ny = ny; s.nx = nx;
    return s;
}

static const uint8_t *dense_block(const mc_sample_src *src,
                                  int bz, int by, int bx, uint8_t *tmp) {
    const uint8_t *vox = src->ud;
    int z0 = bz * BLK, y0 = by * BLK, x0 = bx * BLK;
    if (z0 >= src->nz || y0 >= src->ny || x0 >= src->nx) return NULL;
    int dz = src->nz - z0 < BLK ? src->nz - z0 : BLK;
    int dy = src->ny - y0 < BLK ? src->ny - y0 : BLK;
    int dx = src->nx - x0 < BLK ? src->nx - x0 : BLK;
    if (dz < BLK || dy < BLK || dx < BLK) memset(tmp, 0, BLKB);
    for (int z = 0; z < dz; z++)
        for (int y = 0; y < dy; y++)
            memcpy(tmp + ((z << 8) | (y << 4)),
                   vox + ((size_t)(z0 + z) * src->ny + (y0 + y)) * src->nx + x0,
                   (size_t)dx);
    return tmp;
}

mc_sample_src mc_sample_src_dense(const uint8_t *vox, int nz, int ny, int nx) {
    mc_sample_src s = {0};
    s.ud = (void *)(uintptr_t)vox;
    s.block = dense_block;                // kept for completeness; the direct
    s.dense = vox;                        // path below short-circuits it
    s.dsy = (size_t)ny * nx; s.dsx = (size_t)nx;
    s.nz = nz; s.ny = ny; s.nx = nx;
    return s;
}

// ---------------------------------------------------------------------------
// sampler
// ---------------------------------------------------------------------------
mc_sampler *mc_sampler_new(const mc_sample_src *src) {
    if (!src || !src->block) return NULL;
    mc_sampler *s = malloc(sizeof *s);
    if (!s) return NULL;
    s->src = *src;
    s->nbz = (src->nz + BLK - 1) / BLK;
    s->nby = (src->ny + BLK - 1) / BLK;
    s->nbx = (src->nx + BLK - 1) / BLK;
    // Per-entry 4KB scratch only for sources that synthesize into tmp (!owns_ptr).
    // The interactive cache path (owns_ptr) caches stable arena pointers -> the
    // sampler is ~5KB instead of ~1MB (no per-frame alloc/page-fault storm).
    s->scratch = (src->owns_ptr || src->dense) ? NULL
                 : malloc((size_t)MC_S_MEMO * 4096);
    mc_sampler_reset(s);
    return s;
}

void mc_sampler_free(mc_sampler *s) { if (s) { free(s->scratch); free(s); } }

void mc_sampler_reset(mc_sampler *s) {
    if (!s) return;
    for (int i = 0; i < MC_S_MEMO; i++) s->m[i].bz = -1;
    s->lbz = s->lby = s->lbx = -1;
    s->lptr = NULL;
    s->rbz = s->rby = s->rbx = -1; s->rres = 0;
}

float mc_sample_point(mc_sampler *s, float z, float y, float x, mc_filter f) {
    return mc_s_sample(s, z, y, x, f);
}

// Is the block covering voxel (z,y,x) resident at this sampler's level?
// Mirrors mc_s_trilinear's block lookup; NULL = absent (frozen miss / air).
static inline int mc_s_block_resident(mc_sampler *s, float z, float y, float x) {
    int z0 = (int)floorf(z), y0 = (int)floorf(y), x0 = (int)floorf(x);
    if (z0 < 0 || y0 < 0 || x0 < 0 ||
        z0 >= s->src.nz || y0 >= s->src.ny || x0 >= s->src.nx) return 0;
    if (s->src.dense) return 1;                    // dense source: always resident
    int bz = z0 >> 4, by = y0 >> 4, bx = x0 >> 4;
    if (bz == s->rbz && by == s->rby && bx == s->rbx) return s->rres;  // ray-coherent
    // CHEAP probe if the source provides one: this must NOT decode (the LOD
    // fallback's whole point is to skip the fine-level decode-on-render-thread).
    int r = s->src.resident ? s->src.resident(&s->src, bz, by, bx)
                            : (mc_s_block(s, bz, by, bx) != NULL);   // may decode
    s->rbz = bz; s->rby = by; s->rbx = bx; s->rres = r;
    return r;
}

// ---------------------------------------------------------------------------
// LOD sampler: one sub-sampler per pyramid level + coarser-LOD fallback.
// ---------------------------------------------------------------------------
struct mc_lod_sampler {
    int nlods;
    mc_sampler *lv[8];          // lv[i] = sampler for level i (NULL if empty)
};

mc_lod_sampler *mc_lod_sampler_new(const mc_sample_lods *ls) {
    if (!ls || ls->nlods <= 0) return NULL;
    mc_lod_sampler *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    s->nlods = ls->nlods < 8 ? ls->nlods : 8;
    for (int i = 0; i < s->nlods; i++)
        if (ls->lods[i].block && ls->lods[i].nz > 0)
            s->lv[i] = mc_sampler_new(&ls->lods[i]);   // NULL ok: treated as empty
    return s;
}

void mc_lod_sampler_free(mc_lod_sampler *s) {
    if (!s) return;
    for (int i = 0; i < s->nlods; i++) mc_sampler_free(s->lv[i]);
    free(s);
}

void mc_lod_sampler_reset(mc_lod_sampler *s) {
    if (!s) return;
    for (int i = 0; i < s->nlods; i++) mc_sampler_reset(s->lv[i]);
}

float mc_lod_sample(mc_lod_sampler *s, int lod, int lod_fallback,
                    float z, float y, float x, mc_filter f) {
    if (!s) return 0.0f;
    if (mc_s_isnan(z) || mc_s_isnan(y) || mc_s_isnan(x)) return 0.0f;
    if (lod < 0) lod = 0;
    // L0 -> requested level: c_L = (c_0 + 0.5) * 2^-lod - 0.5
    if (lod > 0) {
        const float inv = 1.0f / (float)(1 << lod);
        z = (z + 0.5f) * inv - 0.5f;
        y = (y + 0.5f) * inv - 0.5f;
        x = (x + 0.5f) * inv - 0.5f;
    }
    for (int L = lod; L < s->nlods; L++) {
        mc_sampler *sub = s->lv[L];
        // Sample this level only if its block for the point is resident. (At the
        // requested level a resident block samples full quality; an absent one
        // either falls through to coarser or, without fallback, returns 0.)
        if (sub && mc_s_block_resident(sub, z, y, x))
            return mc_s_sample(sub, z, y, x, f);
        if (!lod_fallback) break;          // no walk: requested level only
        // descend to next coarser level: c' = (c + 0.5)*0.5 - 0.5
        z = (z + 0.5f) * 0.5f - 0.5f;
        y = (y + 0.5f) * 0.5f - 0.5f;
        x = (x + 0.5f) * 0.5f - 0.5f;
    }
    return 0.0f;   // nothing resident along the chain
}

static inline int pt_valid(const float *p) {
    if (p[0] != p[0] || p[1] != p[1] || p[2] != p[2]) return 0;   // NaN
    return p[0] >= 0.0f && p[1] >= 0.0f && p[2] >= 0.0f;
}

void mc_sample_points(mc_sampler *s, const float *zyx, size_t n,
                      mc_filter f, float *out) {
    if (f == MC_FILTER_NEAREST) {
        for (size_t i = 0; i < n; i++) {
            const float *p = zyx + i * 3;
            out[i] = pt_valid(p) ? mc_s_nearest(s, p[0], p[1], p[2]) : 0.0f;
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            const float *p = zyx + i * 3;
            out[i] = pt_valid(p) ? mc_s_trilinear(s, p[0], p[1], p[2]) : 0.0f;
        }
    }
}

void mc_sample_points_u8(mc_sampler *s, const float *zyx, size_t n,
                         mc_filter f, uint8_t *out) {
    if (f == MC_FILTER_NEAREST) {
        for (size_t i = 0; i < n; i++) {
            const float *p = zyx + i * 3;
            float v = pt_valid(p) ? mc_s_nearest(s, p[0], p[1], p[2]) : 0.0f;
            out[i] = (uint8_t)(v < 0.0f ? 0 : v > 255.0f ? 255 : (int)(v + 0.5f));
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            const float *p = zyx + i * 3;
            float v = pt_valid(p) ? mc_s_trilinear(s, p[0], p[1], p[2]) : 0.0f;
            out[i] = (uint8_t)(v < 0.0f ? 0 : v > 255.0f ? 255 : (int)(v + 0.5f));
        }
    }
}

// ============================================================================
// mc_render — surface rendering, compositing, LOD, 3D resampling
// ============================================================================
// ---------------------------------------------------------------------------
// core
// ---------------------------------------------------------------------------
static inline uint8_t to_u8(float v) {
    return (uint8_t)(v < 0.0f ? 0 : v > 255.0f ? 255 : (int)(v + 0.5f));
}

// Per-image constants hoisted out of the pixel loop.
typedef struct {
    mc_filter filter;
    mc_comp comp;
    float t0, dt;
    int nsteps;                 // iterations of the [t0, t1] dt walk
    float a_min, a_op;          // alpha params, clamped
    // MC_COMP_SHADED (defaults resolved here; see mc_render_params docs)
    float Lz, Ly, Lx;           // unit light dir, toward the light
    int headlight;              // light[] was zero: use -ray dir per pixel
    float ka, kd, ks, shin;     // ambient / diffuse / specular / exponent
    float sigma;                // extinction per unit density per voxel
    float shadow, sss;          // shadow strength, translucency weight
    float g0sq;                 // grad_g0^2 (surface-ness knee)
    int sh_steps;               // secondary-march steps toward the light
    float sh_dt;                // secondary-march step, voxels
    float curv;                 // ridge/valley shading weight
    float pct;                  // percentile rank (0,1]
    int ink;                    // MC_COMP_INK: transmission + cone SSS
    float trans;                // backlight weight [0,1]
    int lrel;                   // light[] is in the per-pixel surface frame
    int lock_steps;             // INK sheet-lock: composite this many steps from
                                // the sheet entry (0 = whole slab). The [t0,t1]
                                // slab becomes a SEARCH range; the dense run
                                // nearest t=0 is the sheet.
    float lock_thr;             // density (post alpha_min scale) that counts as
                                // sheet material during the search
    float alut[256];            // per-step opacity 1-exp(-sigma*d*dt), d=i/255
} rcfg_t;

static rcfg_t make_cfg(const mc_render_params *p) {
    rcfg_t c;
    c.filter = p->filter;
    c.comp = p->comp;
    c.dt = p->dt > 0.0f ? p->dt : 1.0f;
    float t0 = p->t0, t1 = p->t1;
    if (t1 < t0) { float tmp = t0; t0 = t1; t1 = tmp; }
    c.t0 = t0;
    c.nsteps = 0;
    for (float t = t0; t <= t1 + 1e-4f; t += c.dt) c.nsteps++;
    c.a_min = p->alpha_min < 0.0f ? 0.0f :
              p->alpha_min > 0.99f ? 0.99f : p->alpha_min;
    c.a_op  = p->alpha_opacity <= 0.0f ? 1.0f :
              p->alpha_opacity > 1.0f ? 1.0f : p->alpha_opacity;
    float ll = p->light[0] * p->light[0] + p->light[1] * p->light[1] +
               p->light[2] * p->light[2];
    c.headlight = ll < 1e-12f;
    if (c.headlight) { c.Lz = c.Ly = 0.0f; c.Lx = 1.0f; }
    else {
        float inv = 1.0f / sqrtf(ll);
        c.Lz = p->light[0] * inv; c.Ly = p->light[1] * inv;
        c.Lx = p->light[2] * inv;
    }
    c.ka   = p->ambient   > 0.0f ? p->ambient   : 0.25f;
    c.kd   = p->diffuse   > 0.0f ? p->diffuse   : 0.75f;
    c.ks   = p->specular  > 0.0f ? p->specular  : 0.20f;
    c.shin = p->shininess > 0.0f ? p->shininess : 24.0f;
    c.sigma = p->absorption > 0.0f ? p->absorption : 1.0f;
    c.shadow = p->shadow < 0.0f ? 0.0f : p->shadow > 1.0f ? 1.0f : p->shadow;
    c.sss = p->sss < 0.0f ? 0.0f : p->sss;
    float g0 = p->grad_g0 > 0.0f ? p->grad_g0 : 8.0f;
    c.g0sq = g0 * g0;
    c.sh_steps = 12;            // 24 voxels of reach at sh_dt = 2: enough to
    c.sh_dt = 2.0f;             // self-shadow a sheet, cheap enough per ray
    c.curv = p->curvature;
    c.pct = (p->percentile > 0.0f && p->percentile <= 1.0f) ? p->percentile
                                                            : 0.9f;
    c.ink = p->comp == MC_COMP_INK;
    c.trans = p->transmission > 0.0f ? (p->transmission > 1.0f ? 1.0f : p->transmission)
                                     : (c.ink ? 0.35f : 0.0f);
    if (c.ink && p->sss <= 0.0f) c.sss = 0.5f;     // ink default: SSS on
    c.lock_steps = 0;
    if (c.ink && p->ink_lock > 0.0f) {
        c.lock_steps = (int)(p->ink_lock / c.dt + 0.5f);
        if (c.lock_steps < 1) c.lock_steps = 1;
    }
    c.lock_thr = 0.35f;        // "solidly sheet" density for the search
    c.lrel = p->light_surface_rel != 0;
    for (int i = 0; i < 256; i++)
        c.alut[i] = 1.0f - expf(-c.sigma * ((float)i * (1.0f / 255.0f)) * c.dt);
    return c;
}

// Hoare quickselect: value at rank k of a[0..n-1] (a is scrambled in place).
static float rank_select(float *a, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        float p = a[(lo + hi) >> 1];
        int i = lo, j = hi;
        while (i <= j) {
            while (a[i] < p) i++;
            while (a[j] > p) j--;
            if (i <= j) { float t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
        }
        if (k <= j) hi = j; else if (k >= i) lo = i; else return a[k];
    }
    return a[k];
}

// MC_COMP_PERCENTILE: the ray's samples are collected (strided down to a
// 1024 cap for absurdly deep slabs) and the rank-`pct` value returned.
static uint8_t pct_ray(mc_sampler *s, const float *P, float nz, float ny,
                       float nx, const rcfg_t *cfg) {
    float buf[1024];
    int stride = (cfg->nsteps + 1023) / 1024;
    if (stride < 1) stride = 1;
    float sz_ = cfg->dt * nz * stride, sy_ = cfg->dt * ny * stride,
          sx_ = cfg->dt * nx * stride;
    float pz = P[0] + cfg->t0 * nz, py = P[1] + cfg->t0 * ny,
          px = P[2] + cfg->t0 * nx;
    int n = 0;
    for (int it = 0; it < cfg->nsteps; it += stride) {
        buf[n++] = mc_s_sample(s, pz, py, px, cfg->filter);
        pz += sz_; py += sy_; px += sx_;
    }
    if (!n) return 0;
    int k = (int)(cfg->pct * (float)(n - 1) + 0.5f);
    return to_u8(rank_select(buf, n, k));
}

// MC_COMP_DEPTH: first t where the value crosses alpha_min, mapped 1..255
// over the slab (0 = no hit).
static uint8_t depth_ray(mc_sampler *s, const float *P, float nz, float ny,
                         float nx, const rcfg_t *cfg) {
    const float sz_ = cfg->dt * nz, sy_ = cfg->dt * ny, sx_ = cfg->dt * nx;
    float pz = P[0] + cfg->t0 * nz, py = P[1] + cfg->t0 * ny,
          px = P[2] + cfg->t0 * nx;
    for (int it = 0; it < cfg->nsteps; it++) {
        float v = mc_s_sample(s, pz, py, px, cfg->filter);
        if (v * (1.0f / 255.0f) > cfg->a_min) {
            float frac = cfg->nsteps > 1 ? (float)it / (float)(cfg->nsteps - 1)
                                         : 0.0f;
            return (uint8_t)(1.0f + 254.0f * frac + 0.5f);
        }
        pz += sz_; py += sy_; px += sx_;
    }
    return 0;
}

// Batched point sampling: SIMD tri8/tri4 chunks + scalar tail. The shaded /
// ink march samples rays, gradient taps and cone taps through this instead of
// per-point scalar trilinear (which was 50% of the ink profile).
static inline void mc_s_batchN(mc_sampler *s, const float *bz, const float *by,
                               const float *bx, float *out, int n, mc_filter f) {
    int k = 0;
    if (f == MC_FILTER_TRILINEAR) {
#ifdef MC_S_HAVE_TRI8
        for (; k + 8 <= n; k += 8) mc_s_tri8(s, bz + k, by + k, bx + k, out + k);
#endif
#if defined(__aarch64__) || defined(__SSE4_1__)
        for (; k + 4 <= n; k += 4) mc_s_tri4(s, bz + k, by + k, bx + k, out + k);
#endif
    }
    for (; k < n; k++) out[k] = mc_s_sample(s, bz[k], by[k], bx[k], f);
}

// MC_COMP_SHADED / MC_COMP_INK: front-to-back emission-absorption along
// P + t*N with gradient-normal lighting (see the header docs). All volume
// taps run through mc_s_batchN: the slab walk in 64-step chunks, the 6
// gradient taps as one 8-wide batch, and the (INK) 3-tap cone march as one
// 36-position batch. Per-step opacity 1-exp(-sigma*d*dt) comes from a 256-
// entry LUT built per image (sigma*dt is constant); the cone march drops the
// tau early-break (bounded extra samples, batching wins more).
#define MC_SH_CHUNK 256
static uint8_t shade_ray(mc_sampler *s, const float *P, float nz, float ny,
                         float nx, const rcfg_t *cfg, const float *Lp) {
    const mc_filter f = cfg->filter;
    const float a_th = cfg->a_min, a_sc = cfg->a_op / (1.0f - cfg->a_min);
    float Lz, Ly, Lx;
    if (Lp) { Lz = Lp[0]; Ly = Lp[1]; Lx = Lp[2]; }          // surface-relative
    else if (cfg->headlight) { Lz = -nz; Ly = -ny; Lx = -nx; }
    else { Lz = cfg->Lz; Ly = cfg->Ly; Lx = cfg->Lx; }
    // view = toward the camera = -ray dir; half vector for Blinn-Phong
    float hz = Lz - nz, hy = Ly - ny, hx = Lx - nx;
    float hl = hz * hz + hy * hy + hx * hx;
    if (hl > 1e-12f) {
        hl = 1.0f / sqrtf(hl);
        hz *= hl; hy *= hl; hx *= hl;
    }
    // cone basis for the INK subsurface taps (two perpendiculars to L)
    const int ntap = cfg->ink ? 3 : 1;
    float e1z = 0, e1y = 0, e1x = 0, e2z = 0, e2y = 0, e2x = 0;
    float dirz[3], diry[3], dirx[3];
    dirz[0] = Lz; diry[0] = Ly; dirx[0] = Lx;
    if (ntap > 1) {
        float az = fabsf(Lz), ay = fabsf(Ly);
        float bz_ = 0, by_ = 0, bx_ = 0;
        if (az < 0.9f) bz_ = 1; else if (ay < 0.9f) by_ = 1; else bx_ = 1;
        e1z = Ly * bx_ - Lx * by_; e1y = Lx * bz_ - Lz * bx_; e1x = Lz * by_ - Ly * bz_;
        float il = 1.0f / sqrtf(e1z * e1z + e1y * e1y + e1x * e1x + 1e-12f);
        e1z *= il; e1y *= il; e1x *= il;
        e2z = Ly * e1x - Lx * e1y; e2y = Lx * e1z - Lz * e1x; e2x = Lz * e1y - Ly * e1z;
        const float k = 0.35f;
        for (int t = 1; t < 3; t++) {
            float dz_ = Lz + (t == 1 ? k * e1z : -k * e2z);
            float dy_ = Ly + (t == 1 ? k * e1y : -k * e2y);
            float dx_ = Lx + (t == 1 ? k * e1x : -k * e2x);
            float il2 = 1.0f / sqrtf(dz_ * dz_ + dy_ * dy_ + dx_ * dx_);
            dirz[t] = dz_ * il2; diry[t] = dy_ * il2; dirx[t] = dx_ * il2;
        }
    }
    const float sz_ = cfg->dt * nz, sy_ = cfg->dt * ny, sx_ = cfg->dt * nx;
    float acc = 0.0f, T = 1.0f;
    const int want_tau = cfg->shadow > 0.0f || cfg->sss > 0.0f;

    float vz[MC_SH_CHUNK], vy[MC_SH_CHUNK], vx[MC_SH_CHUNK], vals[MC_SH_CHUNK];
    float tb_z[64], tb_y[64], tb_x[64], tb_v[64];   // taps: 8 grad + <=36 cone

    const int total = cfg->lock_steps > 0 && cfg->nsteps > MC_SH_CHUNK
                          ? MC_SH_CHUNK : cfg->nsteps;   // lock searches chunk 0 only
    for (int base = 0; base < total && T >= 0.02f; base += MC_SH_CHUNK) {
        const int m = cfg->nsteps - base < MC_SH_CHUNK ? cfg->nsteps - base
                                                       : MC_SH_CHUNK;
        // batch-sample this chunk of the slab walk
        {
            float pz = P[0] + (cfg->t0 + (float)base * cfg->dt) * nz;
            float py = P[1] + (cfg->t0 + (float)base * cfg->dt) * ny;
            float px = P[2] + (cfg->t0 + (float)base * cfg->dt) * nx;
            for (int k = 0; k < m; k++) {
                vz[k] = pz; vy[k] = py; vx[k] = px;
                pz += sz_; py += sy_; px += sx_;
            }
            mc_s_batchN(s, vz, vy, vx, vals, m, f);
        }
        int k0 = 0, k1 = m;
        if (cfg->lock_steps > 0 && base == 0) {
            // SHEET LOCK: the slab is a SEARCH range. Surfaces wander -- the
            // sheet may sit in front of, behind, or across t=0 -- so find the
            // dense run (>=2 consecutive samples above lock_thr) whose center
            // is nearest the segmentation surface, then composite only
            // lock_steps from its entry face (the ink-bearing crust). Material
            // beyond that -- sheet interior, gaps, the neighboring wrap -- is
            // excluded instead of overlaying papyrus texture on the ink.
            const int i_surf = cfg->t0 < 0.0f ? (int)(-cfg->t0 / cfg->dt + 0.5f) : 0;
            int best_s = -1, best_d = 1 << 30;
            int run_s = -1;
            for (int k = 0; k <= m; k++) {
                float d = k < m ? (vals[k] * (1.0f / 255.0f) - cfg->a_min) *
                                      (cfg->a_op / (1.0f - cfg->a_min))
                                : 0.0f;
                if (k < m && d >= cfg->lock_thr) {
                    if (run_s < 0) run_s = k;
                } else if (run_s >= 0) {
                    if (k - run_s >= 2) {                 // a real run, not a fleck
                        const int center = (run_s + k - 1) >> 1;
                        const int dist = center > i_surf ? center - i_surf
                                                         : i_surf - center;
                        if (dist < best_d) { best_d = dist; best_s = run_s; }
                    }
                    run_s = -1;
                }
            }
            if (best_s < 0)
                return 0;                                 // no sheet in the search slab
            k0 = best_s > 0 ? best_s - 1 : 0;             // include the entry gradient
            k1 = best_s + cfg->lock_steps;
            if (k1 > m) k1 = m;
        }
        for (int k = k0; k < k1; k++) {
            float v = vals[k];
            float d = (v * (1.0f / 255.0f) - a_th) * a_sc;
            if (d <= 0.0f) continue;                // air: free skip
            if (d > 1.0f) d = 1.0f;
            float a = cfg->alut[(int)(d * 255.0f + 0.5f)];
            const float pz = vz[k], py = vy[k], px = vx[k];
            // 6 gradient taps in ONE 8-wide batch (2 dummy lanes)
            tb_z[0] = pz + 1; tb_y[0] = py;     tb_x[0] = px;
            tb_z[1] = pz - 1; tb_y[1] = py;     tb_x[1] = px;
            tb_z[2] = pz;     tb_y[2] = py + 1; tb_x[2] = px;
            tb_z[3] = pz;     tb_y[3] = py - 1; tb_x[3] = px;
            tb_z[4] = pz;     tb_y[4] = py;     tb_x[4] = px + 1;
            tb_z[5] = pz;     tb_y[5] = py;     tb_x[5] = px - 1;
            tb_z[6] = tb_z[7] = -1; tb_y[6] = tb_y[7] = -1; tb_x[6] = tb_x[7] = -1;
            mc_s_batchN(s, tb_z, tb_y, tb_x, tb_v, 8, f);
            const float vzp = tb_v[0], vzm = tb_v[1], vyp = tb_v[2],
                        vym = tb_v[3], vxp = tb_v[4], vxm = tb_v[5];
            float gz = vzp - vzm, gy = vyp - vym, gx = vxp - vxm;
            float g2 = 0.25f * (gz * gz + gy * gy + gx * gx);
            float w = g2 / (g2 + cfg->g0sq);        // surface-ness
            float diff = 0.0f, spec = 0.0f;
            if (g2 > 1e-8f) {
                float gi = 1.0f / sqrtf(gz * gz + gy * gy + gx * gx);
                float uz = gz * gi, uy = gy * gi, ux = gx * gi;
                diff = fabsf(uz * Lz + uy * Ly + ux * Lx);   // two-sided
                float ndh = fabsf(uz * hz + uy * hy + ux * hx);
                spec = powf(ndh, cfg->shin);
            }
            float lit = cfg->ka + (1.0f - w) * cfg->kd;     // interior: emissive
            float Tl = 1.0f;
            if (want_tau && w > 0.05f) {
                // cone subsurface march: all taps' positions in ONE batch
                int n = 0;
                for (int t = 0; t < ntap; t++) {
                    float qz = pz, qy = py, qx = px;
                    for (int j = 0; j < cfg->sh_steps; j++) {
                        qz += cfg->sh_dt * dirz[t];
                        qy += cfg->sh_dt * diry[t];
                        qx += cfg->sh_dt * dirx[t];
                        tb_z[n] = qz; tb_y[n] = qy; tb_x[n] = qx; n++;
                    }
                }
                mc_s_batchN(s, tb_z, tb_y, tb_x, tb_v, n, f);
                float tl_sum = 0.0f, glow_sum = 0.0f;
                for (int t = 0; t < ntap; t++) {
                    float tau = 0.0f;
                    const float *tv = tb_v + t * cfg->sh_steps;
                    for (int j = 0; j < cfg->sh_steps; j++) {
                        float sd = (tv[j] * (1.0f / 255.0f) - a_th) * a_sc;
                        if (sd > 0.0f) {
                            if (sd > 1.0f) sd = 1.0f;
                            tau += cfg->sigma * sd * cfg->sh_dt;
                        }
                    }
                    if (tau > 6.0f) tau = 6.0f;
                    tl_sum += expf(-tau);
                    glow_sum += expf(-0.3f * tau);
                }
                const float inv_n = 1.0f / (float)ntap;
                Tl = tl_sum * inv_n;
                lit += cfg->sss * w * glow_sum * inv_n;     // translucent glow
            }
            float shfac = 1.0f - cfg->shadow + cfg->shadow * Tl;
            lit += w * cfg->kd * diff * shfac;
            if (cfg->curv != 0.0f) {
                // density Laplacian, free from the gradient taps: negative at
                // ridges/crests (brighten), positive in cracks/pits (darken)
                float lap = vzp + vzm + vyp + vym + vxp + vxm - 6.0f * v;
                float cc = -lap * (1.0f / 510.0f);
                if (cc > 1.0f) cc = 1.0f; else if (cc < -1.0f) cc = -1.0f;
                lit += cfg->curv * cc * w;
                if (lit < 0.0f) lit = 0.0f;
            }
            float shade = v * lit + 255.0f * cfg->ks * spec * shfac * w;
            acc += T * a * shade;
            T *= 1.0f - a;
            if (T < 0.02f) break;
        }
    }
    // INK: backlit transmission -- the light that survives the slab. Papyrus is
    // translucent, carbon ink denser, so ink reads locally dark in transmission.
    // Gate on material actually traversed (1-T): a pure-air ray accumulates nothing
    // and must stay 0 (air -> black), not glow at trans*255.
    if (cfg->trans > 0.0f)
        acc += T * cfg->trans * 255.0f * (1.0f - T);
    return to_u8(acc);
}


// Composite one ray. Trilinear rays are consumed in chunks of 4 steps via
// mc_s_tri4 (NEON gather+lerp on aarch64, ~1.4x the scalar core); the
// accumulation itself stays sequential per chunk, which keeps ALPHA\'s
// front-to-back order and early-out exact. Positions are P + t*N with t
// advanced additively, as before.
static uint8_t render_pixel(mc_sampler *s, const float *P, const float *N,
                            const rcfg_t *cfg, const float *Tx) {
    if (!pt_valid(P)) return 0;
    if (cfg->comp == MC_COMP_NONE || !N)
        return to_u8(mc_s_sample(s, P[0], P[1], P[2], cfg->filter));
    float nz = N[0], ny = N[1], nx = N[2];
    float n2 = nz * nz + ny * ny + nx * nx;
    if (n2 < 1e-12f)
        return to_u8(mc_s_sample(s, P[0], P[1], P[2], cfg->filter));
    if (n2 < 0.9998f || n2 > 1.0002f) {     // gen paths emit unit normals
        float nl = 1.0f / sqrtf(n2);
        nz *= nl; ny *= nl; nx *= nl;
    }
    if (cfg->comp == MC_COMP_SHADED || cfg->comp == MC_COMP_INK) {
        // Surface-relative raking: build the per-pixel frame from the surface
        // normal + the screen-x tangent and express light[] in it. light[0] =
        // elevation component (along -N, toward the camera side), light[2] =
        // along screen-x (t1), light[1] = along N x t1 (t2). Falls back to the
        // volume frame when the tangent degenerates.
        float L[3];
        const float *Lp = NULL;
        if (cfg->lrel && !cfg->headlight && Tx) {
            float tz = Tx[0], ty = Tx[1], tx = Tx[2];
            const float dn = tz * nz + ty * ny + tx * nx;
            tz -= dn * nz; ty -= dn * ny; tx -= dn * nx;     // project off N
            float tl = tz * tz + ty * ty + tx * tx;
            if (tl > 1e-12f) {
                tl = 1.0f / sqrtf(tl);
                tz *= tl; ty *= tl; tx *= tl;
                // t2 = N x t1, computed in (x,y,z) then stored (z,y,x)
                const float ux = ny * tz - nz * ty;   // n_y t_z - n_z t_y
                const float uy = nz * tx - nx * tz;   // n_z t_x - n_x t_z
                const float uz = nx * ty - ny * tx;   // n_x t_y - n_y t_x
                L[0] = cfg->Lz * -nz + cfg->Ly * uz + cfg->Lx * tz;
                L[1] = cfg->Lz * -ny + cfg->Ly * uy + cfg->Lx * ty;
                L[2] = cfg->Lz * -nx + cfg->Ly * ux + cfg->Lx * tx;
                Lp = L;
            }
        }
        return shade_ray(s, P, nz, ny, nx, cfg, Lp);
    }
    if (cfg->comp == MC_COMP_PERCENTILE)
        return pct_ray(s, P, nz, ny, nx, cfg);
    if (cfg->comp == MC_COMP_DEPTH)
        return depth_ray(s, P, nz, ny, nx, cfg);

    const float sz_ = cfg->dt * nz, sy_ = cfg->dt * ny, sx_ = cfg->dt * nx;
    float pz = P[0] + cfg->t0 * nz, py = P[1] + cfg->t0 * ny,
          px = P[2] + cfg->t0 * nx;
    const float a_th = cfg->a_min, a_sc = cfg->a_op / (1.0f - cfg->a_min);
    float acc = 0.0f, A = 0.0f, mn = 255.0f, mx = 0.0f, sum = 0.0f,
          sum2 = 0.0f;
    int it = 0, done = 0;

    if (cfg->filter == MC_FILTER_TRILINEAR) {
// NOTE: composites deliberately stay 4-wide. Measured on Zen 3 (EPYC
        // 7763): 8-wide ray chunks ran 1.6x SLOWER than two independent 4-wide
        // chunks (the 8-long insert-gather dependency chain over z-strided
        // addresses serializes); 8-wide only wins for adjacent-pixel loads
        // (the slice path below).
        for (; it + 4 <= cfg->nsteps && !done; it += 4) {
            float bz[4], by[4], bx[4], v4[4];
            for (int k = 0; k < 4; k++) {
                bz[k] = pz; by[k] = py; bx[k] = px;
                pz += sz_; py += sy_; px += sx_;
            }
            mc_s_tri4(s, bz, by, bx, v4);
            switch (cfg->comp) {
            case MC_COMP_MIN:
                for (int k = 0; k < 4; k++) if (v4[k] < mn) mn = v4[k];
                break;
            case MC_COMP_MAX:
                for (int k = 0; k < 4; k++) if (v4[k] > mx) mx = v4[k];
                if (mx >= 255.0f) done = 1;     // saturated
                break;
            case MC_COMP_MEAN:
                sum += v4[0] + v4[1] + v4[2] + v4[3];
                break;
            case MC_COMP_STDDEV:
                for (int k = 0; k < 4; k++) {
                    sum += v4[k]; sum2 += v4[k] * v4[k];
                }
                break;
            default:                            // ALPHA
                for (int k = 0; k < 4 && !done; k++) {
                    float a = (v4[k] * (1.0f / 255.0f) - a_th) * a_sc;
                    if (a > 0.0f) {
                        if (a > 1.0f) a = 1.0f;
                        acc += (1.0f - A) * a * v4[k];
                        A   += (1.0f - A) * a;
                        if (A >= 0.98f) done = 1;
                    }
                }
                break;
            }
        }
    }
    for (; it < cfg->nsteps && !done; it++) {
        float v = mc_s_sample(s, pz, py, px, cfg->filter);
        switch (cfg->comp) {
        case MC_COMP_MIN:  if (v < mn) mn = v; break;
        case MC_COMP_MAX:  if (v > mx) mx = v; break;
        case MC_COMP_MEAN: sum += v; break;
        case MC_COMP_STDDEV: sum += v; sum2 += v * v; break;
        default: {                              // ALPHA
            float a = (v * (1.0f / 255.0f) - a_th) * a_sc;
            if (a > 0.0f) {
                if (a > 1.0f) a = 1.0f;
                acc += (1.0f - A) * a * v;
                A   += (1.0f - A) * a;
                if (A >= 0.98f) done = 1;
            }
            break;
        }
        }
        pz += sz_; py += sy_; px += sx_;
    }
    switch (cfg->comp) {
    case MC_COMP_MIN:  return to_u8(mn);
    case MC_COMP_MAX:  return to_u8(mx);
    case MC_COMP_MEAN:
        return to_u8(cfg->nsteps ? sum / (float)cfg->nsteps : 0.0f);
    case MC_COMP_STDDEV: {
        if (!cfg->nsteps) return 0;
        float m = sum / (float)cfg->nsteps;
        float var = sum2 / (float)cfg->nsteps - m * m;
        return to_u8(var > 0.0f ? sqrtf(var) : 0.0f);
    }
    case MC_COMP_ALPHA: return to_u8(acc);
    default:           return 0;
    }
}

void mc_render_points(mc_sampler *s,
                      const float *pts, const float *normals,
                      int w, int h, const mc_render_params *p, uint8_t *out) {
    rcfg_t cfg = make_cfg(p);
    size_t n = (size_t)w * h;
    if (cfg.comp == MC_COMP_NONE || !normals) {
        // slice fast path: no per-pixel normal handling, branch on the
        // filter once
        if (cfg.filter == MC_FILTER_NEAREST) {
            for (size_t k = 0; k < n; k++) {
                const float *P = pts + k * 3;
                out[k] = pt_valid(P)
                             ? to_u8(mc_s_nearest(s, P[0], P[1], P[2])) : 0;
            }
        } else {
            // 4/8 pixels per mc_s_tri4/8 call (SIMD gather+lerp)
            size_t k = 0;
#ifdef MC_S_HAVE_TRI8
            for (; k + 8 <= n; k += 8) {
                const float *P = pts + k * 3;
                int allv = 1;
                for (int q = 0; q < 8; q++) allv &= pt_valid(P + q * 3);
                if (allv) {
                    float bz[8], by[8], bx[8], v8[8];
                    for (int q = 0; q < 8; q++) {
                        bz[q] = P[q * 3]; by[q] = P[q * 3 + 1];
                        bx[q] = P[q * 3 + 2];
                    }
                    mc_s_tri8(s, bz, by, bx, v8);
                    for (int q = 0; q < 8; q++) out[k + q] = to_u8(v8[q]);
                } else {
                    for (size_t q = k; q < k + 8; q++) {
                        const float *Q = pts + q * 3;
                        out[q] = pt_valid(Q)
                            ? to_u8(mc_s_trilinear(s, Q[0], Q[1], Q[2])) : 0;
                    }
                }
            }
#endif
            for (; k + 4 <= n; k += 4) {
                const float *P = pts + k * 3;
                if (pt_valid(P) && pt_valid(P + 3) &&
                    pt_valid(P + 6) && pt_valid(P + 9)) {
                    float bz[4] = { P[0], P[3], P[6], P[9]  };
                    float by[4] = { P[1], P[4], P[7], P[10] };
                    float bx[4] = { P[2], P[5], P[8], P[11] };
                    float v4[4];
                    mc_s_tri4(s, bz, by, bx, v4);
                    out[k]     = to_u8(v4[0]);
                    out[k + 1] = to_u8(v4[1]);
                    out[k + 2] = to_u8(v4[2]);
                    out[k + 3] = to_u8(v4[3]);
                } else {
                    for (size_t q = k; q < k + 4; q++) {
                        const float *Q = pts + q * 3;
                        out[q] = pt_valid(Q)
                            ? to_u8(mc_s_trilinear(s, Q[0], Q[1], Q[2])) : 0;
                    }
                }
            }
            for (; k < n; k++) {
                const float *P = pts + k * 3;
                out[k] = pt_valid(P)
                             ? to_u8(mc_s_trilinear(s, P[0], P[1], P[2])) : 0;
            }
        }
        return;
    }
    const int needT = cfg.lrel && !cfg.headlight &&
                      (cfg.comp == MC_COMP_SHADED || cfg.comp == MC_COMP_INK);
    for (size_t k = 0; k < n; k++) {
        float T[3];
        const float *Tx = NULL;
        if (needT && w > 1) {
            // screen-x tangent from the neighboring pixel's position
            const size_t col = k % (size_t)w;
            const float *Pc = pts + k * 3;
            const float *Pn = pts + (col + 1 < (size_t)w ? k + 1 : k - 1) * 3;
            if (pt_valid(Pc) && pt_valid(Pn)) {
                T[0] = Pn[0] - Pc[0]; T[1] = Pn[1] - Pc[1]; T[2] = Pn[2] - Pc[2];
                if (col + 1 >= (size_t)w) { T[0] = -T[0]; T[1] = -T[1]; T[2] = -T[2]; }
                Tx = T;
            }
        }
        out[k] = render_pixel(s, pts + k * 3, normals + k * 3, &cfg, Tx);
    }
}

// ---------------------------------------------------------------------------
// parallel core: row bands, one sampler per worker
// ---------------------------------------------------------------------------
// rowgen fills one row of points (+normals) into band-local scratch; plane
// and quad renders go through this so no W*H grid is ever materialized
// (a 1024^2 trilinear frame otherwise mallocs and touches 24 MB of points).
typedef void (*rowgen_fn)(const void *ud, int row, int w,
                          float *pts, float *normals);

typedef struct {
    const mc_sample_src *src;
    const float *pts, *normals;     // dense mode (rowgen == NULL)
    rowgen_fn rowgen;               // strip mode
    const void *rg_ud;
    int w, h;
    const mc_render_params *p;
    uint8_t *out;
    int row0, row1;
    // LOD-fallback mode: when ls != NULL, sample via a per-band mc_lod_sampler
    // at level `lod` with coarser-LOD fallback instead of the single `src`.
    const mc_sample_lods *ls;
    int lod;
} band_t;

// LOD-fallback point render: coords are native level-0 voxel space; each pixel
// samples the requested level (`lod`) with coarser-LOD fallback. Slice path
// (comp==NONE) only — the interactive nav case; composites stay single-level.
//
// Fast path: the common case is "all blocks resident at `lod`" (steady state and
// air, since air decodes to a resident zero block). For a group of 8 points whose
// level-`lod` blocks are ALL resident, sample with the 8-wide SIMD kernel on the
// level-`lod` sampler — same speed as the non-fallback render. Only groups that
// touch an ABSENT block fall to the per-lane coarse-LOD walk (transient, while a
// freshly-entered level streams in).
static void render_points_lod(mc_lod_sampler *ls, int lod,
                              const float *pts, int w, int h,
                              const mc_render_params *p, uint8_t *out) {
    const mc_filter f = (mc_filter)p->filter;
    const size_t n = (size_t)w * h;
    mc_sampler *L = (lod >= 0 && lod < ls->nlods) ? ls->lv[lod] : NULL;
    const float inv = lod > 0 ? 1.0f / (float)(1 << lod) : 1.0f;
    // L0 -> level-`lod` voxel coord (half-voxel-center correct).
    #define MC_L0_TO_L(c) ((c + 0.5f) * inv - 0.5f)

    size_t k = 0;
#ifdef MC_S_HAVE_TRI8
    if (L && f == MC_FILTER_TRILINEAR) {
        float pz[8], py[8], px[8], v8[8];
        for (; k + 8 <= n; k += 8) {
            int allv = 1, allres = 1;
            for (int q = 0; q < 8; q++) {
                const float *P = pts + (k + q) * 3;
                if (!pt_valid(P)) { allv = 0; break; }
                float z = MC_L0_TO_L(P[0]), y = MC_L0_TO_L(P[1]), x = MC_L0_TO_L(P[2]);
                pz[q] = z; py[q] = y; px[q] = x;
                // resident block at L? (covers air: air is a resident zero block)
                if (!mc_s_block_resident(L, z, y, x)) { allres = 0; }
            }
            if (allv && allres) {
                mc_s_tri8(L, pz, py, px, v8);
                for (int q = 0; q < 8; q++) out[k + q] = to_u8(v8[q]);
            } else {
                for (int q = 0; q < 8; q++) {
                    const float *P = pts + (k + q) * 3;
                    out[k + q] = pt_valid(P)
                        ? to_u8(mc_lod_sample(ls, lod, 1, P[0], P[1], P[2], f)) : 0;
                }
            }
        }
    }
#endif
    for (; k < n; k++) {
        const float *P = pts + k * 3;
        out[k] = pt_valid(P)
            ? to_u8(mc_lod_sample(ls, lod, 1, P[0], P[1], P[2], f)) : 0;
    }
    #undef MC_L0_TO_L
}

static void *band_main(void *ud) {
    band_t *b = ud;
    if (b->ls) {
        mc_lod_sampler *ls = mc_lod_sampler_new(b->ls);
        if (!ls) return NULL;
        float *row = malloc((size_t)b->w * 3 * sizeof(float));
        if (row) {
            for (int i = b->row0; i < b->row1; i++) {
                b->rowgen(b->rg_ud, i, b->w, row, NULL);   // L0 coords, no normals
                render_points_lod(ls, b->lod, row, b->w, 1, b->p,
                                  b->out + (size_t)i * b->w);
            }
            free(row);
        } else memset(b->out + (size_t)b->row0 * b->w, 0,
                      (size_t)(b->row1 - b->row0) * b->w);
        mc_lod_sampler_free(ls);
        return NULL;
    }
    mc_sampler *s = mc_sampler_new(b->src);
    if (!s) return NULL;
    if (!b->rowgen) {
        size_t off = (size_t)b->row0 * b->w;
        mc_render_points(s, b->pts + off * 3,
                         b->normals ? b->normals + off * 3 : NULL,
                         b->w, b->row1 - b->row0, b->p, b->out + off);
    } else {
        int need_n = b->p->comp != MC_COMP_NONE;
        float *row = malloc((size_t)b->w * 3 * sizeof(float) * (need_n ? 2 : 1));
        if (row) {
            float *nrm = need_n ? row + (size_t)b->w * 3 : NULL;
            for (int i = b->row0; i < b->row1; i++) {
                b->rowgen(b->rg_ud, i, b->w, row, nrm);
                mc_render_points(s, row, nrm, b->w, 1, b->p,
                                 b->out + (size_t)i * b->w);
            }
            free(row);
        }
        else memset(b->out + (size_t)b->row0 * b->w, 0,
                    (size_t)(b->row1 - b->row0) * b->w);
    }
    mc_sampler_free(s);
    return NULL;
}

// ls != NULL selects LOD-fallback mode (rowgen produces level-0 coords, each
// band builds an mc_lod_sampler at `lod`). Otherwise single-source mode.
static void render_bands_ex(const mc_sample_src *src,
                            const float *pts, const float *normals,
                            rowgen_fn rowgen, const void *rg_ud,
                            const mc_sample_lods *ls, int lod,
                            int w, int h, const mc_render_params *p,
                            uint8_t *out, int nthreads) {
    if (w <= 0 || h <= 0) return;
    if (nthreads <= 0) {
        long nc = sysconf(_SC_NPROCESSORS_ONLN);
        nthreads = nc > 0 ? (int)nc : 1;
    }
    if (nthreads > 16) nthreads = 16;
    if (nthreads > h)  nthreads = h;
    pthread_t th[16];
    band_t bands[16];
    int per = (h + nthreads - 1) / nthreads;
    int nb = 0;
    for (int t = 0; t < nthreads; t++) {
        int r0 = t * per, r1 = r0 + per > h ? h : r0 + per;
        if (r0 >= r1) break;
        bands[nb] = (band_t){ src, pts, normals, rowgen, rg_ud,
                              w, h, p, out, r0, r1, ls, lod };
        if (nthreads == 1) { band_main(&bands[nb]); continue; }
        if (pthread_create(&th[nb], NULL, band_main, &bands[nb]) != 0) {
            band_main(&bands[nb]);          // degrade to inline
            continue;
        }
        nb++;
    }
    for (int t = 0; t < nb; t++) pthread_join(th[t], NULL);
}

static void render_bands(const mc_sample_src *src,
                         const float *pts, const float *normals,
                         rowgen_fn rowgen, const void *rg_ud,
                         int w, int h, const mc_render_params *p,
                         uint8_t *out, int nthreads) {
    render_bands_ex(src, pts, normals, rowgen, rg_ud, NULL, 0,
                    w, h, p, out, nthreads);
}

void mc_render_points_par(const mc_sample_src *src,
                          const float *pts, const float *normals,
                          int w, int h, const mc_render_params *p,
                          uint8_t *out, int nthreads) {
    render_bands(src, pts, normals, NULL, NULL, w, h, p, out, nthreads);
}

// dense-points renderer (one band-local row scratch fed from the caller's
// level-0 point grid) with LOD-fallback sampling at `lod`.
static void densepts_rowgen(const void *ud, int row, int w,
                            float *pts, float *normals) {
    (void)normals;
    const float *base = (const float *)ud;
    memcpy(pts, base + (size_t)row * w * 3, (size_t)w * 3 * sizeof(float));
}

void mc_render_points_par_lod(const mc_sample_lods *ls, int lod,
                              const float *ptsL0, int w, int h,
                              const mc_render_params *p,
                              uint8_t *out, int nthreads) {
    if (!ls || !ptsL0 || !out || w <= 0 || h <= 0) return;
    if (lod < 0) lod = 0;
    if (lod >= ls->nlods) lod = ls->nlods - 1;
    render_bands_ex(NULL, NULL, NULL, densepts_rowgen, ptsL0, ls, lod,
                    w, h, p, out, nthreads);
}

// ===========================================================================
// mc_colormap — window/level + colormap LUT (moved out of volume-cartographer).
// mc_render emits u8; this maps u8 -> ARGB32 for display. EVERY colormap is a
// static [256][3] grayscale->RGB table (palettes baked from the originals; tints
// generated as v*channel). mc stays dependency-free. Ids match the GUI strings.
// ===========================================================================
static const uint8_t MC_CMAP_VIRIDIS[256][3] = {
{68,1,84},{68,2,86},{69,4,87},{69,5,89},{70,7,90},{70,8,92},{70,10,93},{70,11,94},
{71,13,96},{71,14,97},{71,16,99},{71,17,100},{71,19,101},{72,20,103},{72,22,104},{72,23,105},
{72,24,106},{72,26,108},{72,27,109},{72,28,110},{72,29,111},{72,31,112},{72,32,113},{72,33,115},
{72,35,116},{72,36,117},{72,37,118},{72,38,119},{72,40,120},{72,41,121},{71,42,122},{71,44,122},
{71,45,123},{71,46,124},{71,47,125},{70,48,126},{70,50,126},{70,51,127},{70,52,128},{69,53,129},
{69,55,129},{69,56,130},{68,57,131},{68,58,131},{68,59,132},{67,61,132},{67,62,133},{66,63,133},
{66,64,134},{66,65,134},{65,66,135},{65,68,135},{64,69,136},{64,70,136},{63,71,136},{63,72,137},
{62,73,137},{62,74,137},{62,76,138},{61,77,138},{61,78,138},{60,79,138},{60,80,139},{59,81,139},
{59,82,139},{58,83,139},{58,84,140},{57,85,140},{57,86,140},{56,88,140},{56,89,140},{55,90,140},
{55,91,141},{54,92,141},{54,93,141},{53,94,141},{53,95,141},{52,96,141},{52,97,141},{51,98,141},
{51,99,141},{50,100,142},{50,101,142},{49,102,142},{49,103,142},{49,104,142},{48,105,142},{48,106,142},
{47,107,142},{47,108,142},{46,109,142},{46,110,142},{46,111,142},{45,112,142},{45,113,142},{44,113,142},
{44,114,142},{44,115,142},{43,116,142},{43,117,142},{42,118,142},{42,119,142},{42,120,142},{41,121,142},
{41,122,142},{41,123,142},{40,124,142},{40,125,142},{39,126,142},{39,127,142},{39,128,142},{38,129,142},
{38,130,142},{38,130,142},{37,131,142},{37,132,142},{37,133,142},{36,134,142},{36,135,142},{35,136,142},
{35,137,142},{35,138,141},{34,139,141},{34,140,141},{34,141,141},{33,142,141},{33,143,141},{33,144,141},
{33,145,140},{32,146,140},{32,146,140},{32,147,140},{31,148,140},{31,149,139},{31,150,139},{31,151,139},
{31,152,139},{31,153,138},{31,154,138},{30,155,138},{30,156,137},{30,157,137},{31,158,137},{31,159,136},
{31,160,136},{31,161,136},{31,161,135},{31,162,135},{32,163,134},{32,164,134},{33,165,133},{33,166,133},
{34,167,133},{34,168,132},{35,169,131},{36,170,131},{37,171,130},{37,172,130},{38,173,129},{39,173,129},
{40,174,128},{41,175,127},{42,176,127},{44,177,126},{45,178,125},{46,179,124},{47,180,124},{49,181,123},
{50,182,122},{52,182,121},{53,183,121},{55,184,120},{56,185,119},{58,186,118},{59,187,117},{61,188,116},
{63,188,115},{64,189,114},{66,190,113},{68,191,112},{70,192,111},{72,193,110},{74,193,109},{76,194,108},
{78,195,107},{80,196,106},{82,197,105},{84,197,104},{86,198,103},{88,199,101},{90,200,100},{92,200,99},
{94,201,98},{96,202,96},{99,203,95},{101,203,94},{103,204,92},{105,205,91},{108,205,90},{110,206,88},
{112,207,87},{115,208,86},{117,208,84},{119,209,83},{122,209,81},{124,210,80},{127,211,78},{129,211,77},
{132,212,75},{134,213,73},{137,213,72},{139,214,70},{142,214,69},{144,215,67},{147,215,65},{149,216,64},
{152,216,62},{155,217,60},{157,217,59},{160,218,57},{162,218,55},{165,219,54},{168,219,52},{170,220,50},
{173,220,48},{176,221,47},{178,221,45},{181,222,43},{184,222,41},{186,222,40},{189,223,38},{192,223,37},
{194,223,35},{197,224,33},{200,224,32},{202,225,31},{205,225,29},{208,225,28},{210,226,27},{213,226,26},
{216,226,25},{218,227,25},{221,227,24},{223,227,24},{226,228,24},{229,228,25},{231,228,25},{234,229,26},
{236,229,27},{239,229,28},{241,229,29},{244,230,30},{246,230,32},{248,230,33},{251,231,35},{253,231,37},
};
static const uint8_t MC_CMAP_MAGMA[256][3] = {
{0,0,4},{1,0,5},{1,1,6},{1,1,8},{2,1,9},{2,2,11},{2,2,13},{3,3,15},
{3,3,18},{4,4,20},{5,4,22},{6,5,24},{6,5,26},{7,6,28},{8,7,30},{9,7,32},
{10,8,34},{11,9,36},{12,9,38},{13,10,41},{14,11,43},{16,11,45},{17,12,47},{18,13,49},
{19,13,52},{20,14,54},{21,14,56},{22,15,59},{24,15,61},{25,16,63},{26,16,66},{28,16,68},
{29,17,71},{30,17,73},{32,17,75},{33,17,78},{34,17,80},{36,18,83},{37,18,85},{39,18,88},
{41,17,90},{42,17,92},{44,17,95},{45,17,97},{47,17,99},{49,17,101},{51,16,103},{52,16,105},
{54,16,107},{56,16,108},{57,15,110},{59,15,112},{61,15,113},{63,15,114},{64,15,116},{66,15,117},
{68,15,118},{69,16,119},{71,16,120},{73,16,120},{74,16,121},{76,17,122},{78,17,123},{79,18,123},
{81,18,124},{82,19,124},{84,19,125},{86,20,125},{87,21,126},{89,21,126},{90,22,126},{92,22,127},
{93,23,127},{95,24,127},{96,24,128},{98,25,128},{100,26,128},{101,26,128},{103,27,128},{104,28,129},
{106,28,129},{107,29,129},{109,29,129},{110,30,129},{112,31,129},{114,31,129},{115,32,129},{117,33,129},
{118,33,129},{120,34,129},{121,34,130},{123,35,130},{124,35,130},{126,36,130},{128,37,130},{129,37,129},
{131,38,129},{132,38,129},{134,39,129},{136,39,129},{137,40,129},{139,41,129},{140,41,129},{142,42,129},
{144,42,129},{145,43,129},{147,43,128},{148,44,128},{150,44,128},{152,45,128},{153,45,128},{155,46,127},
{156,46,127},{158,47,127},{160,47,127},{161,48,126},{163,48,126},{165,49,126},{166,49,125},{168,50,125},
{170,51,125},{171,51,124},{173,52,124},{174,52,123},{176,53,123},{178,53,123},{179,54,122},{181,54,122},
{183,55,121},{184,55,121},{186,56,120},{188,57,120},{189,57,119},{191,58,119},{192,58,118},{194,59,117},
{196,60,117},{197,60,116},{199,61,115},{200,62,115},{202,62,114},{204,63,113},{205,64,113},{207,64,112},
{208,65,111},{210,66,111},{211,67,110},{213,68,109},{214,69,108},{216,69,108},{217,70,107},{219,71,106},
{220,72,105},{222,73,104},{223,74,104},{224,76,103},{226,77,102},{227,78,101},{228,79,100},{229,80,100},
{231,82,99},{232,83,98},{233,84,98},{234,86,97},{235,87,96},{236,88,96},{237,90,95},{238,91,94},
{239,93,94},{240,95,94},{241,96,93},{242,98,93},{242,100,92},{243,101,92},{244,103,92},{244,105,92},
{245,107,92},{246,108,92},{246,110,92},{247,112,92},{247,114,92},{248,116,92},{248,118,92},{249,120,93},
{249,121,93},{249,123,93},{250,125,94},{250,127,94},{250,129,95},{251,131,95},{251,133,96},{251,135,97},
{252,137,97},{252,138,98},{252,140,99},{252,142,100},{252,144,101},{253,146,102},{253,148,103},{253,150,104},
{253,152,105},{253,154,106},{253,155,107},{254,157,108},{254,159,109},{254,161,110},{254,163,111},{254,165,113},
{254,167,114},{254,169,115},{254,170,116},{254,172,118},{254,174,119},{254,176,120},{254,178,122},{254,180,123},
{254,182,124},{254,183,126},{254,185,127},{254,187,129},{254,189,130},{254,191,132},{254,193,133},{254,194,135},
{254,196,136},{254,198,138},{254,200,140},{254,202,141},{254,204,143},{254,205,144},{254,207,146},{254,209,148},
{254,211,149},{254,213,151},{254,215,153},{254,216,154},{253,218,156},{253,220,158},{253,222,160},{253,224,161},
{253,226,163},{253,227,165},{253,229,167},{253,231,169},{253,233,170},{253,235,172},{252,236,174},{252,238,176},
{252,240,178},{252,242,180},{252,244,182},{252,246,184},{252,247,185},{252,249,187},{252,251,189},{252,253,191},
};
static const uint8_t MC_CMAP_FIRE[256][3] = {
{0,0,0},{2,0,0},{5,0,0},{8,0,0},{10,0,0},{12,0,0},{15,0,0},{18,0,0},
{20,0,0},{22,0,0},{25,0,0},{27,0,0},{30,0,0},{32,0,0},{35,0,0},{38,0,0},
{40,0,0},{42,0,0},{45,0,0},{48,0,0},{50,0,0},{52,0,0},{55,0,0},{57,0,0},
{60,0,0},{62,0,0},{65,0,0},{68,0,0},{70,0,0},{72,0,0},{75,0,0},{78,0,0},
{80,0,0},{82,0,0},{85,0,0},{88,0,0},{90,0,0},{92,0,0},{95,0,0},{98,0,0},
{100,0,0},{102,0,0},{105,0,0},{108,0,0},{110,0,0},{112,0,0},{115,0,0},{117,0,0},
{120,0,0},{122,0,0},{125,0,0},{128,0,0},{130,0,0},{132,0,0},{135,0,0},{138,0,0},
{140,0,0},{142,0,0},{145,0,0},{148,0,0},{150,0,0},{152,0,0},{155,0,0},{158,0,0},
{160,0,0},{162,0,0},{165,0,0},{168,0,0},{170,0,0},{172,0,0},{175,0,0},{178,0,0},
{180,0,0},{182,0,0},{185,0,0},{188,0,0},{190,0,0},{192,0,0},{195,0,0},{198,0,0},
{200,0,0},{202,0,0},{205,0,0},{208,0,0},{210,0,0},{212,0,0},{215,0,0},{218,0,0},
{220,0,0},{223,0,0},{225,0,0},{228,0,0},{230,0,0},{232,0,0},{235,0,0},{238,0,0},
{240,0,0},{243,0,0},{245,0,0},{248,0,0},{250,0,0},{252,0,0},{253,2,0},{254,4,0},
{254,6,0},{255,8,0},{255,10,0},{255,13,0},{255,15,0},{255,18,0},{255,20,0},{255,22,0},
{255,25,0},{255,28,0},{255,30,0},{255,32,0},{255,35,0},{255,38,0},{255,40,0},{255,42,0},
{255,45,0},{255,48,0},{255,50,0},{255,52,0},{255,55,0},{255,58,0},{255,60,0},{255,62,0},
{255,65,0},{255,68,0},{255,70,0},{255,72,0},{255,75,0},{255,78,0},{255,80,0},{255,82,0},
{255,85,0},{255,88,0},{255,90,0},{255,92,0},{255,95,0},{255,98,0},{255,100,0},{255,102,0},
{255,105,0},{255,108,0},{255,110,0},{255,112,0},{255,115,0},{255,118,0},{255,120,0},{255,122,0},
{255,125,0},{255,128,0},{255,130,0},{255,132,0},{255,135,0},{255,138,0},{255,140,0},{255,142,0},
{255,145,0},{255,148,0},{255,150,0},{255,152,0},{255,155,0},{255,158,0},{255,160,0},{255,162,0},
{255,165,0},{255,168,0},{255,170,0},{255,172,0},{255,175,0},{255,178,0},{255,180,0},{255,182,0},
{255,185,0},{255,188,0},{255,190,0},{255,192,0},{255,195,0},{255,198,0},{255,200,0},{255,202,0},
{255,205,0},{255,208,0},{255,210,0},{255,212,0},{255,215,0},{255,218,0},{255,220,0},{255,222,0},
{255,225,0},{255,228,0},{255,230,0},{255,232,0},{255,235,0},{255,238,0},{255,240,0},{255,242,0},
{255,245,0},{255,248,0},{255,250,0},{255,252,2},{255,253,5},{255,254,8},{255,255,11},{255,255,15},
{255,255,20},{255,255,25},{255,255,30},{255,255,35},{255,255,40},{255,255,45},{255,255,50},{255,255,55},
{255,255,60},{255,255,65},{255,255,70},{255,255,75},{255,255,80},{255,255,85},{255,255,90},{255,255,95},
{255,255,100},{255,255,105},{255,255,110},{255,255,115},{255,255,120},{255,255,125},{255,255,130},{255,255,135},
{255,255,140},{255,255,145},{255,255,150},{255,255,155},{255,255,160},{255,255,165},{255,255,170},{255,255,175},
{255,255,180},{255,255,185},{255,255,190},{255,255,195},{255,255,200},{255,255,205},{255,255,210},{255,255,215},
{255,255,220},{255,255,225},{255,255,230},{255,255,235},{255,255,240},{255,255,245},{255,255,250},{255,255,255},
};
static const uint8_t MC_CMAP_GRAY[256][3] = {
{0,0,0},{1,1,1},{2,2,2},{3,3,3},{4,4,4},{5,5,5},{6,6,6},{7,7,7},
{8,8,8},{9,9,9},{10,10,10},{11,11,11},{12,12,12},{13,13,13},{14,14,14},{15,15,15},
{16,16,16},{17,17,17},{18,18,18},{19,19,19},{20,20,20},{21,21,21},{22,22,22},{23,23,23},
{24,24,24},{25,25,25},{26,26,26},{27,27,27},{28,28,28},{29,29,29},{30,30,30},{31,31,31},
{32,32,32},{33,33,33},{34,34,34},{35,35,35},{36,36,36},{37,37,37},{38,38,38},{39,39,39},
{40,40,40},{41,41,41},{42,42,42},{43,43,43},{44,44,44},{45,45,45},{46,46,46},{47,47,47},
{48,48,48},{49,49,49},{50,50,50},{51,51,51},{52,52,52},{53,53,53},{54,54,54},{55,55,55},
{56,56,56},{57,57,57},{58,58,58},{59,59,59},{60,60,60},{61,61,61},{62,62,62},{63,63,63},
{64,64,64},{65,65,65},{66,66,66},{67,67,67},{68,68,68},{69,69,69},{70,70,70},{71,71,71},
{72,72,72},{73,73,73},{74,74,74},{75,75,75},{76,76,76},{77,77,77},{78,78,78},{79,79,79},
{80,80,80},{81,81,81},{82,82,82},{83,83,83},{84,84,84},{85,85,85},{86,86,86},{87,87,87},
{88,88,88},{89,89,89},{90,90,90},{91,91,91},{92,92,92},{93,93,93},{94,94,94},{95,95,95},
{96,96,96},{97,97,97},{98,98,98},{99,99,99},{100,100,100},{101,101,101},{102,102,102},{103,103,103},
{104,104,104},{105,105,105},{106,106,106},{107,107,107},{108,108,108},{109,109,109},{110,110,110},{111,111,111},
{112,112,112},{113,113,113},{114,114,114},{115,115,115},{116,116,116},{117,117,117},{118,118,118},{119,119,119},
{120,120,120},{121,121,121},{122,122,122},{123,123,123},{124,124,124},{125,125,125},{126,126,126},{127,127,127},
{128,128,128},{129,129,129},{130,130,130},{131,131,131},{132,132,132},{133,133,133},{134,134,134},{135,135,135},
{136,136,136},{137,137,137},{138,138,138},{139,139,139},{140,140,140},{141,141,141},{142,142,142},{143,143,143},
{144,144,144},{145,145,145},{146,146,146},{147,147,147},{148,148,148},{149,149,149},{150,150,150},{151,151,151},
{152,152,152},{153,153,153},{154,154,154},{155,155,155},{156,156,156},{157,157,157},{158,158,158},{159,159,159},
{160,160,160},{161,161,161},{162,162,162},{163,163,163},{164,164,164},{165,165,165},{166,166,166},{167,167,167},
{168,168,168},{169,169,169},{170,170,170},{171,171,171},{172,172,172},{173,173,173},{174,174,174},{175,175,175},
{176,176,176},{177,177,177},{178,178,178},{179,179,179},{180,180,180},{181,181,181},{182,182,182},{183,183,183},
{184,184,184},{185,185,185},{186,186,186},{187,187,187},{188,188,188},{189,189,189},{190,190,190},{191,191,191},
{192,192,192},{193,193,193},{194,194,194},{195,195,195},{196,196,196},{197,197,197},{198,198,198},{199,199,199},
{200,200,200},{201,201,201},{202,202,202},{203,203,203},{204,204,204},{205,205,205},{206,206,206},{207,207,207},
{208,208,208},{209,209,209},{210,210,210},{211,211,211},{212,212,212},{213,213,213},{214,214,214},{215,215,215},
{216,216,216},{217,217,217},{218,218,218},{219,219,219},{220,220,220},{221,221,221},{222,222,222},{223,223,223},
{224,224,224},{225,225,225},{226,226,226},{227,227,227},{228,228,228},{229,229,229},{230,230,230},{231,231,231},
{232,232,232},{233,233,233},{234,234,234},{235,235,235},{236,236,236},{237,237,237},{238,238,238},{239,239,239},
{240,240,240},{241,241,241},{242,242,242},{243,243,243},{244,244,244},{245,245,245},{246,246,246},{247,247,247},
{248,248,248},{249,249,249},{250,250,250},{251,251,251},{252,252,252},{253,253,253},{254,254,254},{255,255,255},
};
static const uint8_t MC_CMAP_RED[256][3] = {
{0,0,0},{1,0,0},{2,0,0},{3,0,0},{4,0,0},{5,0,0},{6,0,0},{7,0,0},
{8,0,0},{9,0,0},{10,0,0},{11,0,0},{12,0,0},{13,0,0},{14,0,0},{15,0,0},
{16,0,0},{17,0,0},{18,0,0},{19,0,0},{20,0,0},{21,0,0},{22,0,0},{23,0,0},
{24,0,0},{25,0,0},{26,0,0},{27,0,0},{28,0,0},{29,0,0},{30,0,0},{31,0,0},
{32,0,0},{33,0,0},{34,0,0},{35,0,0},{36,0,0},{37,0,0},{38,0,0},{39,0,0},
{40,0,0},{41,0,0},{42,0,0},{43,0,0},{44,0,0},{45,0,0},{46,0,0},{47,0,0},
{48,0,0},{49,0,0},{50,0,0},{51,0,0},{52,0,0},{53,0,0},{54,0,0},{55,0,0},
{56,0,0},{57,0,0},{58,0,0},{59,0,0},{60,0,0},{61,0,0},{62,0,0},{63,0,0},
{64,0,0},{65,0,0},{66,0,0},{67,0,0},{68,0,0},{69,0,0},{70,0,0},{71,0,0},
{72,0,0},{73,0,0},{74,0,0},{75,0,0},{76,0,0},{77,0,0},{78,0,0},{79,0,0},
{80,0,0},{81,0,0},{82,0,0},{83,0,0},{84,0,0},{85,0,0},{86,0,0},{87,0,0},
{88,0,0},{89,0,0},{90,0,0},{91,0,0},{92,0,0},{93,0,0},{94,0,0},{95,0,0},
{96,0,0},{97,0,0},{98,0,0},{99,0,0},{100,0,0},{101,0,0},{102,0,0},{103,0,0},
{104,0,0},{105,0,0},{106,0,0},{107,0,0},{108,0,0},{109,0,0},{110,0,0},{111,0,0},
{112,0,0},{113,0,0},{114,0,0},{115,0,0},{116,0,0},{117,0,0},{118,0,0},{119,0,0},
{120,0,0},{121,0,0},{122,0,0},{123,0,0},{124,0,0},{125,0,0},{126,0,0},{127,0,0},
{128,0,0},{129,0,0},{130,0,0},{131,0,0},{132,0,0},{133,0,0},{134,0,0},{135,0,0},
{136,0,0},{137,0,0},{138,0,0},{139,0,0},{140,0,0},{141,0,0},{142,0,0},{143,0,0},
{144,0,0},{145,0,0},{146,0,0},{147,0,0},{148,0,0},{149,0,0},{150,0,0},{151,0,0},
{152,0,0},{153,0,0},{154,0,0},{155,0,0},{156,0,0},{157,0,0},{158,0,0},{159,0,0},
{160,0,0},{161,0,0},{162,0,0},{163,0,0},{164,0,0},{165,0,0},{166,0,0},{167,0,0},
{168,0,0},{169,0,0},{170,0,0},{171,0,0},{172,0,0},{173,0,0},{174,0,0},{175,0,0},
{176,0,0},{177,0,0},{178,0,0},{179,0,0},{180,0,0},{181,0,0},{182,0,0},{183,0,0},
{184,0,0},{185,0,0},{186,0,0},{187,0,0},{188,0,0},{189,0,0},{190,0,0},{191,0,0},
{192,0,0},{193,0,0},{194,0,0},{195,0,0},{196,0,0},{197,0,0},{198,0,0},{199,0,0},
{200,0,0},{201,0,0},{202,0,0},{203,0,0},{204,0,0},{205,0,0},{206,0,0},{207,0,0},
{208,0,0},{209,0,0},{210,0,0},{211,0,0},{212,0,0},{213,0,0},{214,0,0},{215,0,0},
{216,0,0},{217,0,0},{218,0,0},{219,0,0},{220,0,0},{221,0,0},{222,0,0},{223,0,0},
{224,0,0},{225,0,0},{226,0,0},{227,0,0},{228,0,0},{229,0,0},{230,0,0},{231,0,0},
{232,0,0},{233,0,0},{234,0,0},{235,0,0},{236,0,0},{237,0,0},{238,0,0},{239,0,0},
{240,0,0},{241,0,0},{242,0,0},{243,0,0},{244,0,0},{245,0,0},{246,0,0},{247,0,0},
{248,0,0},{249,0,0},{250,0,0},{251,0,0},{252,0,0},{253,0,0},{254,0,0},{255,0,0},
};
static const uint8_t MC_CMAP_GREEN[256][3] = {
{0,0,0},{0,1,0},{0,2,0},{0,3,0},{0,4,0},{0,5,0},{0,6,0},{0,7,0},
{0,8,0},{0,9,0},{0,10,0},{0,11,0},{0,12,0},{0,13,0},{0,14,0},{0,15,0},
{0,16,0},{0,17,0},{0,18,0},{0,19,0},{0,20,0},{0,21,0},{0,22,0},{0,23,0},
{0,24,0},{0,25,0},{0,26,0},{0,27,0},{0,28,0},{0,29,0},{0,30,0},{0,31,0},
{0,32,0},{0,33,0},{0,34,0},{0,35,0},{0,36,0},{0,37,0},{0,38,0},{0,39,0},
{0,40,0},{0,41,0},{0,42,0},{0,43,0},{0,44,0},{0,45,0},{0,46,0},{0,47,0},
{0,48,0},{0,49,0},{0,50,0},{0,51,0},{0,52,0},{0,53,0},{0,54,0},{0,55,0},
{0,56,0},{0,57,0},{0,58,0},{0,59,0},{0,60,0},{0,61,0},{0,62,0},{0,63,0},
{0,64,0},{0,65,0},{0,66,0},{0,67,0},{0,68,0},{0,69,0},{0,70,0},{0,71,0},
{0,72,0},{0,73,0},{0,74,0},{0,75,0},{0,76,0},{0,77,0},{0,78,0},{0,79,0},
{0,80,0},{0,81,0},{0,82,0},{0,83,0},{0,84,0},{0,85,0},{0,86,0},{0,87,0},
{0,88,0},{0,89,0},{0,90,0},{0,91,0},{0,92,0},{0,93,0},{0,94,0},{0,95,0},
{0,96,0},{0,97,0},{0,98,0},{0,99,0},{0,100,0},{0,101,0},{0,102,0},{0,103,0},
{0,104,0},{0,105,0},{0,106,0},{0,107,0},{0,108,0},{0,109,0},{0,110,0},{0,111,0},
{0,112,0},{0,113,0},{0,114,0},{0,115,0},{0,116,0},{0,117,0},{0,118,0},{0,119,0},
{0,120,0},{0,121,0},{0,122,0},{0,123,0},{0,124,0},{0,125,0},{0,126,0},{0,127,0},
{0,128,0},{0,129,0},{0,130,0},{0,131,0},{0,132,0},{0,133,0},{0,134,0},{0,135,0},
{0,136,0},{0,137,0},{0,138,0},{0,139,0},{0,140,0},{0,141,0},{0,142,0},{0,143,0},
{0,144,0},{0,145,0},{0,146,0},{0,147,0},{0,148,0},{0,149,0},{0,150,0},{0,151,0},
{0,152,0},{0,153,0},{0,154,0},{0,155,0},{0,156,0},{0,157,0},{0,158,0},{0,159,0},
{0,160,0},{0,161,0},{0,162,0},{0,163,0},{0,164,0},{0,165,0},{0,166,0},{0,167,0},
{0,168,0},{0,169,0},{0,170,0},{0,171,0},{0,172,0},{0,173,0},{0,174,0},{0,175,0},
{0,176,0},{0,177,0},{0,178,0},{0,179,0},{0,180,0},{0,181,0},{0,182,0},{0,183,0},
{0,184,0},{0,185,0},{0,186,0},{0,187,0},{0,188,0},{0,189,0},{0,190,0},{0,191,0},
{0,192,0},{0,193,0},{0,194,0},{0,195,0},{0,196,0},{0,197,0},{0,198,0},{0,199,0},
{0,200,0},{0,201,0},{0,202,0},{0,203,0},{0,204,0},{0,205,0},{0,206,0},{0,207,0},
{0,208,0},{0,209,0},{0,210,0},{0,211,0},{0,212,0},{0,213,0},{0,214,0},{0,215,0},
{0,216,0},{0,217,0},{0,218,0},{0,219,0},{0,220,0},{0,221,0},{0,222,0},{0,223,0},
{0,224,0},{0,225,0},{0,226,0},{0,227,0},{0,228,0},{0,229,0},{0,230,0},{0,231,0},
{0,232,0},{0,233,0},{0,234,0},{0,235,0},{0,236,0},{0,237,0},{0,238,0},{0,239,0},
{0,240,0},{0,241,0},{0,242,0},{0,243,0},{0,244,0},{0,245,0},{0,246,0},{0,247,0},
{0,248,0},{0,249,0},{0,250,0},{0,251,0},{0,252,0},{0,253,0},{0,254,0},{0,255,0},
};
static const uint8_t MC_CMAP_BLUE[256][3] = {
{0,0,0},{0,0,1},{0,0,2},{0,0,3},{0,0,4},{0,0,5},{0,0,6},{0,0,7},
{0,0,8},{0,0,9},{0,0,10},{0,0,11},{0,0,12},{0,0,13},{0,0,14},{0,0,15},
{0,0,16},{0,0,17},{0,0,18},{0,0,19},{0,0,20},{0,0,21},{0,0,22},{0,0,23},
{0,0,24},{0,0,25},{0,0,26},{0,0,27},{0,0,28},{0,0,29},{0,0,30},{0,0,31},
{0,0,32},{0,0,33},{0,0,34},{0,0,35},{0,0,36},{0,0,37},{0,0,38},{0,0,39},
{0,0,40},{0,0,41},{0,0,42},{0,0,43},{0,0,44},{0,0,45},{0,0,46},{0,0,47},
{0,0,48},{0,0,49},{0,0,50},{0,0,51},{0,0,52},{0,0,53},{0,0,54},{0,0,55},
{0,0,56},{0,0,57},{0,0,58},{0,0,59},{0,0,60},{0,0,61},{0,0,62},{0,0,63},
{0,0,64},{0,0,65},{0,0,66},{0,0,67},{0,0,68},{0,0,69},{0,0,70},{0,0,71},
{0,0,72},{0,0,73},{0,0,74},{0,0,75},{0,0,76},{0,0,77},{0,0,78},{0,0,79},
{0,0,80},{0,0,81},{0,0,82},{0,0,83},{0,0,84},{0,0,85},{0,0,86},{0,0,87},
{0,0,88},{0,0,89},{0,0,90},{0,0,91},{0,0,92},{0,0,93},{0,0,94},{0,0,95},
{0,0,96},{0,0,97},{0,0,98},{0,0,99},{0,0,100},{0,0,101},{0,0,102},{0,0,103},
{0,0,104},{0,0,105},{0,0,106},{0,0,107},{0,0,108},{0,0,109},{0,0,110},{0,0,111},
{0,0,112},{0,0,113},{0,0,114},{0,0,115},{0,0,116},{0,0,117},{0,0,118},{0,0,119},
{0,0,120},{0,0,121},{0,0,122},{0,0,123},{0,0,124},{0,0,125},{0,0,126},{0,0,127},
{0,0,128},{0,0,129},{0,0,130},{0,0,131},{0,0,132},{0,0,133},{0,0,134},{0,0,135},
{0,0,136},{0,0,137},{0,0,138},{0,0,139},{0,0,140},{0,0,141},{0,0,142},{0,0,143},
{0,0,144},{0,0,145},{0,0,146},{0,0,147},{0,0,148},{0,0,149},{0,0,150},{0,0,151},
{0,0,152},{0,0,153},{0,0,154},{0,0,155},{0,0,156},{0,0,157},{0,0,158},{0,0,159},
{0,0,160},{0,0,161},{0,0,162},{0,0,163},{0,0,164},{0,0,165},{0,0,166},{0,0,167},
{0,0,168},{0,0,169},{0,0,170},{0,0,171},{0,0,172},{0,0,173},{0,0,174},{0,0,175},
{0,0,176},{0,0,177},{0,0,178},{0,0,179},{0,0,180},{0,0,181},{0,0,182},{0,0,183},
{0,0,184},{0,0,185},{0,0,186},{0,0,187},{0,0,188},{0,0,189},{0,0,190},{0,0,191},
{0,0,192},{0,0,193},{0,0,194},{0,0,195},{0,0,196},{0,0,197},{0,0,198},{0,0,199},
{0,0,200},{0,0,201},{0,0,202},{0,0,203},{0,0,204},{0,0,205},{0,0,206},{0,0,207},
{0,0,208},{0,0,209},{0,0,210},{0,0,211},{0,0,212},{0,0,213},{0,0,214},{0,0,215},
{0,0,216},{0,0,217},{0,0,218},{0,0,219},{0,0,220},{0,0,221},{0,0,222},{0,0,223},
{0,0,224},{0,0,225},{0,0,226},{0,0,227},{0,0,228},{0,0,229},{0,0,230},{0,0,231},
{0,0,232},{0,0,233},{0,0,234},{0,0,235},{0,0,236},{0,0,237},{0,0,238},{0,0,239},
{0,0,240},{0,0,241},{0,0,242},{0,0,243},{0,0,244},{0,0,245},{0,0,246},{0,0,247},
{0,0,248},{0,0,249},{0,0,250},{0,0,251},{0,0,252},{0,0,253},{0,0,254},{0,0,255},
};
static const uint8_t MC_CMAP_CYAN[256][3] = {
{0,0,0},{0,1,1},{0,2,2},{0,3,3},{0,4,4},{0,5,5},{0,6,6},{0,7,7},
{0,8,8},{0,9,9},{0,10,10},{0,11,11},{0,12,12},{0,13,13},{0,14,14},{0,15,15},
{0,16,16},{0,17,17},{0,18,18},{0,19,19},{0,20,20},{0,21,21},{0,22,22},{0,23,23},
{0,24,24},{0,25,25},{0,26,26},{0,27,27},{0,28,28},{0,29,29},{0,30,30},{0,31,31},
{0,32,32},{0,33,33},{0,34,34},{0,35,35},{0,36,36},{0,37,37},{0,38,38},{0,39,39},
{0,40,40},{0,41,41},{0,42,42},{0,43,43},{0,44,44},{0,45,45},{0,46,46},{0,47,47},
{0,48,48},{0,49,49},{0,50,50},{0,51,51},{0,52,52},{0,53,53},{0,54,54},{0,55,55},
{0,56,56},{0,57,57},{0,58,58},{0,59,59},{0,60,60},{0,61,61},{0,62,62},{0,63,63},
{0,64,64},{0,65,65},{0,66,66},{0,67,67},{0,68,68},{0,69,69},{0,70,70},{0,71,71},
{0,72,72},{0,73,73},{0,74,74},{0,75,75},{0,76,76},{0,77,77},{0,78,78},{0,79,79},
{0,80,80},{0,81,81},{0,82,82},{0,83,83},{0,84,84},{0,85,85},{0,86,86},{0,87,87},
{0,88,88},{0,89,89},{0,90,90},{0,91,91},{0,92,92},{0,93,93},{0,94,94},{0,95,95},
{0,96,96},{0,97,97},{0,98,98},{0,99,99},{0,100,100},{0,101,101},{0,102,102},{0,103,103},
{0,104,104},{0,105,105},{0,106,106},{0,107,107},{0,108,108},{0,109,109},{0,110,110},{0,111,111},
{0,112,112},{0,113,113},{0,114,114},{0,115,115},{0,116,116},{0,117,117},{0,118,118},{0,119,119},
{0,120,120},{0,121,121},{0,122,122},{0,123,123},{0,124,124},{0,125,125},{0,126,126},{0,127,127},
{0,128,128},{0,129,129},{0,130,130},{0,131,131},{0,132,132},{0,133,133},{0,134,134},{0,135,135},
{0,136,136},{0,137,137},{0,138,138},{0,139,139},{0,140,140},{0,141,141},{0,142,142},{0,143,143},
{0,144,144},{0,145,145},{0,146,146},{0,147,147},{0,148,148},{0,149,149},{0,150,150},{0,151,151},
{0,152,152},{0,153,153},{0,154,154},{0,155,155},{0,156,156},{0,157,157},{0,158,158},{0,159,159},
{0,160,160},{0,161,161},{0,162,162},{0,163,163},{0,164,164},{0,165,165},{0,166,166},{0,167,167},
{0,168,168},{0,169,169},{0,170,170},{0,171,171},{0,172,172},{0,173,173},{0,174,174},{0,175,175},
{0,176,176},{0,177,177},{0,178,178},{0,179,179},{0,180,180},{0,181,181},{0,182,182},{0,183,183},
{0,184,184},{0,185,185},{0,186,186},{0,187,187},{0,188,188},{0,189,189},{0,190,190},{0,191,191},
{0,192,192},{0,193,193},{0,194,194},{0,195,195},{0,196,196},{0,197,197},{0,198,198},{0,199,199},
{0,200,200},{0,201,201},{0,202,202},{0,203,203},{0,204,204},{0,205,205},{0,206,206},{0,207,207},
{0,208,208},{0,209,209},{0,210,210},{0,211,211},{0,212,212},{0,213,213},{0,214,214},{0,215,215},
{0,216,216},{0,217,217},{0,218,218},{0,219,219},{0,220,220},{0,221,221},{0,222,222},{0,223,223},
{0,224,224},{0,225,225},{0,226,226},{0,227,227},{0,228,228},{0,229,229},{0,230,230},{0,231,231},
{0,232,232},{0,233,233},{0,234,234},{0,235,235},{0,236,236},{0,237,237},{0,238,238},{0,239,239},
{0,240,240},{0,241,241},{0,242,242},{0,243,243},{0,244,244},{0,245,245},{0,246,246},{0,247,247},
{0,248,248},{0,249,249},{0,250,250},{0,251,251},{0,252,252},{0,253,253},{0,254,254},{0,255,255},
};
static const uint8_t MC_CMAP_MAGENTA[256][3] = {
{0,0,0},{1,0,1},{2,0,2},{3,0,3},{4,0,4},{5,0,5},{6,0,6},{7,0,7},
{8,0,8},{9,0,9},{10,0,10},{11,0,11},{12,0,12},{13,0,13},{14,0,14},{15,0,15},
{16,0,16},{17,0,17},{18,0,18},{19,0,19},{20,0,20},{21,0,21},{22,0,22},{23,0,23},
{24,0,24},{25,0,25},{26,0,26},{27,0,27},{28,0,28},{29,0,29},{30,0,30},{31,0,31},
{32,0,32},{33,0,33},{34,0,34},{35,0,35},{36,0,36},{37,0,37},{38,0,38},{39,0,39},
{40,0,40},{41,0,41},{42,0,42},{43,0,43},{44,0,44},{45,0,45},{46,0,46},{47,0,47},
{48,0,48},{49,0,49},{50,0,50},{51,0,51},{52,0,52},{53,0,53},{54,0,54},{55,0,55},
{56,0,56},{57,0,57},{58,0,58},{59,0,59},{60,0,60},{61,0,61},{62,0,62},{63,0,63},
{64,0,64},{65,0,65},{66,0,66},{67,0,67},{68,0,68},{69,0,69},{70,0,70},{71,0,71},
{72,0,72},{73,0,73},{74,0,74},{75,0,75},{76,0,76},{77,0,77},{78,0,78},{79,0,79},
{80,0,80},{81,0,81},{82,0,82},{83,0,83},{84,0,84},{85,0,85},{86,0,86},{87,0,87},
{88,0,88},{89,0,89},{90,0,90},{91,0,91},{92,0,92},{93,0,93},{94,0,94},{95,0,95},
{96,0,96},{97,0,97},{98,0,98},{99,0,99},{100,0,100},{101,0,101},{102,0,102},{103,0,103},
{104,0,104},{105,0,105},{106,0,106},{107,0,107},{108,0,108},{109,0,109},{110,0,110},{111,0,111},
{112,0,112},{113,0,113},{114,0,114},{115,0,115},{116,0,116},{117,0,117},{118,0,118},{119,0,119},
{120,0,120},{121,0,121},{122,0,122},{123,0,123},{124,0,124},{125,0,125},{126,0,126},{127,0,127},
{128,0,128},{129,0,129},{130,0,130},{131,0,131},{132,0,132},{133,0,133},{134,0,134},{135,0,135},
{136,0,136},{137,0,137},{138,0,138},{139,0,139},{140,0,140},{141,0,141},{142,0,142},{143,0,143},
{144,0,144},{145,0,145},{146,0,146},{147,0,147},{148,0,148},{149,0,149},{150,0,150},{151,0,151},
{152,0,152},{153,0,153},{154,0,154},{155,0,155},{156,0,156},{157,0,157},{158,0,158},{159,0,159},
{160,0,160},{161,0,161},{162,0,162},{163,0,163},{164,0,164},{165,0,165},{166,0,166},{167,0,167},
{168,0,168},{169,0,169},{170,0,170},{171,0,171},{172,0,172},{173,0,173},{174,0,174},{175,0,175},
{176,0,176},{177,0,177},{178,0,178},{179,0,179},{180,0,180},{181,0,181},{182,0,182},{183,0,183},
{184,0,184},{185,0,185},{186,0,186},{187,0,187},{188,0,188},{189,0,189},{190,0,190},{191,0,191},
{192,0,192},{193,0,193},{194,0,194},{195,0,195},{196,0,196},{197,0,197},{198,0,198},{199,0,199},
{200,0,200},{201,0,201},{202,0,202},{203,0,203},{204,0,204},{205,0,205},{206,0,206},{207,0,207},
{208,0,208},{209,0,209},{210,0,210},{211,0,211},{212,0,212},{213,0,213},{214,0,214},{215,0,215},
{216,0,216},{217,0,217},{218,0,218},{219,0,219},{220,0,220},{221,0,221},{222,0,222},{223,0,223},
{224,0,224},{225,0,225},{226,0,226},{227,0,227},{228,0,228},{229,0,229},{230,0,230},{231,0,231},
{232,0,232},{233,0,233},{234,0,234},{235,0,235},{236,0,236},{237,0,237},{238,0,238},{239,0,239},
{240,0,240},{241,0,241},{242,0,242},{243,0,243},{244,0,244},{245,0,245},{246,0,246},{247,0,247},
{248,0,248},{249,0,249},{250,0,250},{251,0,251},{252,0,252},{253,0,253},{254,0,254},{255,0,255},
};

// id -> the static palette table. All maps are a uniform [256][3] lookup.
static const uint8_t (*mc_cmap_table(int id))[3] {
    switch(id){
        case 1: return MC_CMAP_VIRIDIS;
        case 2: return MC_CMAP_MAGMA;
        case 3: return MC_CMAP_FIRE;
        case 4: return MC_CMAP_RED;
        case 5: return MC_CMAP_GREEN;
        case 6: return MC_CMAP_BLUE;
        case 7: return MC_CMAP_CYAN;
        case 8: return MC_CMAP_MAGENTA;
        default: return MC_CMAP_GRAY;
    }
}
int mc_colormap_id(const char *name){
    if(!name||!*name) return 0;
    if(!strcmp(name,"viridis")) return 1;
    if(!strcmp(name,"magma"))   return 2;
    if(!strcmp(name,"fire"))    return 3;
    if(!strcmp(name,"red"))     return 4;
    if(!strcmp(name,"green"))   return 5;
    if(!strcmp(name,"blue"))    return 6;
    if(!strcmp(name,"cyan"))    return 7;
    if(!strcmp(name,"magenta")) return 8;
    return 0;
}
// 256-entry ARGB32 LUT: window/level ramp then the colormap table. Matches VC3D's
// old buildWindowLevelColormapLut (lut[0]=opaque black for non-gray maps).
void mc_colormap_lut(uint32_t lut[256], float win_low, float win_high, int cmap_id){
    int lo=(int)(win_low<0?0:win_low>255?255:win_low);
    int hi=(int)(win_high<lo+1?lo+1:win_high>255?255:win_high);
    float span=(hi-lo)>1?(float)(hi-lo):1.0f;
    const uint8_t (*pal)[3]=mc_cmap_table(cmap_id);
    for(int i=0;i<256;++i){
        float g=((float)i-(float)lo)/span*255.0f;
        uint8_t v=(uint8_t)(g<0?0:g>255?255:g);
        lut[i]=0xFF000000u|((uint32_t)pal[v][0]<<16)|((uint32_t)pal[v][1]<<8)|(uint32_t)pal[v][2];
    }
    // A composite value of 0 means "no material" (air / block edge / air-bitmask),
    // independent of how it was zeroed -> always black. The colormap (incl. window/
    // level for grayscale) applies to real data, 1..255, only.
    lut[0]=0xFF000000u;
}
// Apply a 256-ARGB LUT to a w*h u8 image -> ARGB32 (out_stride in pixels).
void mc_colormap_apply(const uint8_t *vals, int w, int h, const uint32_t lut[256],
                       uint32_t *out, int out_stride){
    for(int y=0;y<h;++y){
        const uint8_t *s=vals+(size_t)y*w; uint32_t *d=out+(size_t)y*out_stride;
        for(int x=0;x<w;++x) d[x]=lut[s[x]];
    }
}


// ---------------------------------------------------------------------------
// plane surface
// ---------------------------------------------------------------------------
static inline void v3_norm(float *v) {
    float l = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (l > 1e-12f) { v[0] /= l; v[1] /= l; v[2] /= l; }
}
static inline void v3_cross(const float *a, const float *b, float *o) {
    o[0] = a[1] * b[2] - a[2] * b[1];
    o[1] = a[2] * b[0] - a[0] * b[2];
    o[2] = a[0] * b[1] - a[1] * b[0];
}

void mc_plane_basis(mc_plane *pl) {
    float *n = pl->normal;
    v3_norm(n);
    // pick the world axis least aligned with n as the seed
    float az = fabsf(n[0]), ay = fabsf(n[1]), ax = fabsf(n[2]);
    float e[3] = {0, 0, 0};
    if (az <= ay && az <= ax) e[0] = 1.0f;
    else if (ay <= ax)        e[1] = 1.0f;
    else                      e[2] = 1.0f;
    v3_cross(n, e, pl->u); v3_norm(pl->u);
    v3_cross(n, pl->u, pl->v); v3_norm(pl->v);
}

void mc_plane_gen(const mc_plane *pl, int w, int h, float scale,
                  float *pts, float *normals) {
    float cx = (float)w * 0.5f, cy = (float)h * 0.5f;
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            float du = ((float)j - cx) * scale;
            float dv = ((float)i - cy) * scale;
            float *P = pts + ((size_t)i * w + j) * 3;
            for (int k = 0; k < 3; k++)
                P[k] = pl->origin[k] + du * pl->u[k] + dv * pl->v[k];
            if (normals) {
                float *N = normals + ((size_t)i * w + j) * 3;
                N[0] = pl->normal[0]; N[1] = pl->normal[1]; N[2] = pl->normal[2];
            }
        }
}

// ---------------------------------------------------------------------------
// quad surface
// ---------------------------------------------------------------------------
static inline int qvalid(const float *p) {
    return !(p[0] == -1.0f && p[1] == -1.0f && p[2] == -1.0f) &&
           p[0] == p[0] && p[1] == p[1] && p[2] == p[2];
}

void mc_quad_gen(const mc_quad *q, float x0, float y0, float step,
                 int w, int h, float *pts, float *normals) {
    const int gw = q->gw, gh = q->gh;
    for (int i = 0; i < h; i++) {
        float *Prow = pts + (size_t)i * w * 3;
        float *Nrow = normals ? normals + (size_t)i * w * 3 : NULL;
        float gy = y0 + (float)i * step;
        // row-invalid fast fill
        if (!(gy >= 0.0f) || gy > (float)(gh - 1)) {
            for (int j = 0; j < w; j++) {
                Prow[j * 3] = Prow[j * 3 + 1] = Prow[j * 3 + 2] = -1.0f;
                if (Nrow) Nrow[j * 3] = Nrow[j * 3 + 1] = Nrow[j * 3 + 2] = 0.0f;
            }
            continue;
        }
        int y0i = (int)gy;
        if (y0i > gh - 2) y0i = gh - 2;
        if (y0i < 0) y0i = 0;               // gh == 1
        float fy = gy - (float)y0i;
        const float *r0 = q->grid + (size_t)y0i * gw * 3;
        const float *r1 = q->grid + (size_t)(y0i + (gh > 1)) * gw * 3;

        // per-cell state, reloaded only when the pixel crosses a grid cell
        int cell = -2, cell_ok = 0;
        float A[3], B[3], du[3], dv0[3], dv1[3];
        for (int j = 0; j < w; j++) {
            float *P = Prow + j * 3;
            float *N = Nrow ? Nrow + j * 3 : NULL;
            P[0] = P[1] = P[2] = -1.0f;
            if (N) N[0] = N[1] = N[2] = 0.0f;
            float gx = x0 + (float)j * step;
            if (!(gx >= 0.0f) || gx > (float)(gw - 1)) continue;
            int x0i = (int)gx;
            if (x0i > gw - 2) x0i = gw - 2;
            if (x0i < 0) x0i = 0;           // gw == 1
            if (x0i != cell) {
                cell = x0i;
                const float *p00 = r0 + (size_t)x0i * 3;
                const float *p01 = r0 + (size_t)(x0i + (gw > 1)) * 3;
                const float *p10 = r1 + (size_t)x0i * 3;
                const float *p11 = r1 + (size_t)(x0i + (gw > 1)) * 3;
                cell_ok = qvalid(p00) && qvalid(p01) &&
                          qvalid(p10) && qvalid(p11);
                if (cell_ok)
                    for (int k = 0; k < 3; k++) {
                        // y-lerped column endpoints: P = A + (B-A)*fx
                        A[k] = p00[k] + (p10[k] - p00[k]) * fy;
                        B[k] = p01[k] + (p11[k] - p01[k]) * fy;
                        // bilinear tangents (du constant per cell row)
                        du[k] = (p01[k] - p00[k]) * (1.0f - fy) +
                                (p11[k] - p10[k]) * fy;
                        dv0[k] = p10[k] - p00[k];
                        dv1[k] = p11[k] - p01[k];
                    }
            }
            if (!cell_ok) continue;
            float fx = gx - (float)x0i;
            P[0] = A[0] + (B[0] - A[0]) * fx;
            P[1] = A[1] + (B[1] - A[1]) * fx;
            P[2] = A[2] + (B[2] - A[2]) * fx;
            if (N) {
                float dv[3] = {
                    dv0[0] + (dv1[0] - dv0[0]) * fx,
                    dv0[1] + (dv1[1] - dv0[1]) * fx,
                    dv0[2] + (dv1[2] - dv0[2]) * fx,
                };
                v3_cross(du, dv, N);
                v3_norm(N);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// conveniences (row-strip rendering, no W*H grid)
// ---------------------------------------------------------------------------
typedef struct {
    mc_plane pl;                // normal pre-normalized
    float scale, cx, cy;
} plane_rg;

static void plane_rowgen(const void *ud, int row, int w,
                         float *pts, float *normals) {
    const plane_rg *g = ud;
    float dv = ((float)row - g->cy) * g->scale;
    float base[3], du[3];
    for (int k = 0; k < 3; k++) {
        base[k] = g->pl.origin[k] + dv * g->pl.v[k]
                  - g->cx * g->scale * g->pl.u[k];
        du[k] = g->scale * g->pl.u[k];
    }
    for (int j = 0; j < w; j++) {
        pts[j * 3 + 0] = base[0] + (float)j * du[0];
        pts[j * 3 + 1] = base[1] + (float)j * du[1];
        pts[j * 3 + 2] = base[2] + (float)j * du[2];
    }
    if (normals)
        for (int j = 0; j < w; j++) {
            normals[j * 3 + 0] = g->pl.normal[0];
            normals[j * 3 + 1] = g->pl.normal[1];
            normals[j * 3 + 2] = g->pl.normal[2];
        }
}

int mc_render_plane(const mc_sample_src *src, const mc_plane *pl,
                    int w, int h, float scale,
                    const mc_render_params *p, uint8_t *out, int nthreads) {
    if (!src || !pl || !p || !out || w <= 0 || h <= 0) return -1;
    plane_rg g = { *pl, scale, (float)w * 0.5f, (float)h * 0.5f };
    v3_norm(g.pl.normal);
    render_bands(src, NULL, NULL, plane_rowgen, &g, w, h, p, out, nthreads);
    return 0;
}

typedef struct {
    const mc_quad *q;
    float x0, y0, step;
} quad_rg;

static void quad_rowgen(const void *ud, int row, int w,
                        float *pts, float *normals) {
    const quad_rg *g = ud;
    mc_quad_gen(g->q, g->x0, g->y0 + (float)row * g->step, g->step,
                w, 1, pts, normals);
}

int mc_render_quad(const mc_sample_src *src, const mc_quad *q,
                   float x0, float y0, float step, int w, int h,
                   const mc_render_params *p, uint8_t *out, int nthreads) {
    if (!src || !q || !q->grid || q->gw < 1 || q->gh < 1 ||
        !p || !out || w <= 0 || h <= 0) return -1;
    quad_rg g = { q, x0, y0, step };
    render_bands(src, NULL, NULL, quad_rowgen, &g, w, h, p, out, nthreads);
    return 0;
}

// ---------------------------------------------------------------------------
// screen-space ambient occlusion over a MC_COMP_DEPTH image
// ---------------------------------------------------------------------------
// 12 fixed disk taps (two rings of 6, inner ring rotated 30deg); a pixel is
// occluded by neighbors that sit nearer the camera (smaller depth) by more
// than a small bias, with a range falloff so a distant other sheet doesn't
// read as an occluder.
// Stroke-scale local-contrast band-pass: img += gain * (img - blur(img)).
// 3-pass separable box blur of width ~sigma approximates a Gaussian; zero
// pixels (invalid/no-data) are excluded from the blur so holes don't bleed
// darkness into their surroundings.
void mc_image_dog(uint8_t *img, int w, int h, float sigma_px, float gain) {
    if (!img || w <= 0 || h <= 0 || sigma_px < 1.0f || gain == 0.0f) return;
    const size_t n = (size_t)w * h;
    // box radius for 3 passes ~= gaussian sigma: r = sigma * sqrt(12/3)/2
    int r = (int)(sigma_px + 0.5f);
    if (r < 1) r = 1;
    if (r > (w < h ? w : h) / 2) r = (w < h ? w : h) / 2;
    float *a = malloc(n * sizeof *a), *b = malloc(n * sizeof *b);
    float *wa = malloc(n * sizeof *wa), *wb = malloc(n * sizeof *wb);
    if (!a || !b || !wa || !wb) { free(a); free(b); free(wa); free(wb); return; }
    for (size_t i = 0; i < n; i++) { a[i] = (float)img[i]; wa[i] = img[i] ? 1.0f : 0.0f; }
    for (size_t i = 0; i < n; i++) a[i] *= wa[i];
    for (int pass = 0; pass < 3; pass++) {
        // horizontal box
        for (int y = 0; y < h; y++) {
            const float *src = a + (size_t)y * w, *sw = wa + (size_t)y * w;
            float *dst = b + (size_t)y * w, *dw = wb + (size_t)y * w;
            float sum = 0, wsum = 0;
            for (int x = 0; x < r && x < w; x++) { sum += src[x]; wsum += sw[x]; }
            for (int x = 0; x < w; x++) {
                if (x + r < w) { sum += src[x + r]; wsum += sw[x + r]; }
                if (x - r - 1 >= 0) { sum -= src[x - r - 1]; wsum -= sw[x - r - 1]; }
                dst[x] = sum; dw[x] = wsum;
            }
        }
        // vertical box (b -> a)
        for (int x = 0; x < w; x++) {
            float sum = 0, wsum = 0;
            for (int y = 0; y < r && y < h; y++) { sum += b[(size_t)y * w + x]; wsum += wb[(size_t)y * w + x]; }
            for (int y = 0; y < h; y++) {
                if (y + r < h) { sum += b[(size_t)(y + r) * w + x]; wsum += wb[(size_t)(y + r) * w + x]; }
                if (y - r - 1 >= 0) { sum -= b[(size_t)(y - r - 1) * w + x]; wsum -= wb[(size_t)(y - r - 1) * w + x]; }
                a[(size_t)y * w + x] = sum;
                wa[(size_t)y * w + x] = wsum;
            }
        }
        // normalize between passes so weights stay box-counts
        if (pass < 2) {
            for (size_t i = 0; i < n; i++) {
                float ww = wa[i];
                a[i] = ww > 0.0f ? a[i] / ww * (img[i] ? 1.0f : 0.0f) : 0.0f;
                wa[i] = img[i] ? 1.0f : 0.0f;
            }
        }
    }
    for (size_t i = 0; i < n; i++) {
        if (!img[i]) continue;                  // keep holes black
        float blur = wa[i] > 0.0f ? a[i] / wa[i] : (float)img[i];
        float v = (float)img[i] + gain * ((float)img[i] - blur);
        img[i] = (uint8_t)(v < 1.0f ? 1.0f : v > 255.0f ? 255.0f : v);
    }
    free(a); free(b); free(wa); free(wb);
}

void mc_image_ssao(const uint8_t *depth, int w, int h,
                   float radius_px, float strength, uint8_t *img) {
    if (!depth || !img || w <= 0 || h <= 0) return;
    const float R = radius_px > 0.0f ? radius_px : 8.0f;
    const float S = strength <= 0.0f ? 0.7f : strength > 1.0f ? 1.0f
                                                              : strength;
    static const float taps[12][2] = {
        {1.0f, 0.0f},   {0.5f, 0.866f},   {-0.5f, 0.866f},
        {-1.0f, 0.0f},  {-0.5f, -0.866f}, {0.5f, -0.866f},
        {0.433f, 0.25f},  {0.0f, 0.5f},  {-0.433f, 0.25f},
        {-0.433f, -0.25f}, {0.0f, -0.5f}, {0.433f, -0.25f},
    };
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            int dc = depth[(size_t)i * w + j];
            if (!dc) continue;
            float occ = 0.0f; int n = 0;
            for (int t = 0; t < 12; t++) {
                int jj = j + (int)(taps[t][0] * R + (taps[t][0] < 0 ? -0.5f : 0.5f));
                int ii = i + (int)(taps[t][1] * R + (taps[t][1] < 0 ? -0.5f : 0.5f));
                if (ii < 0 || ii >= h || jj < 0 || jj >= w) continue;
                int dn = depth[(size_t)ii * w + jj];
                if (!dn) continue;
                float dd = (float)dc - (float)dn - 2.0f;    // nearer by > bias
                if (dd <= 0.0f) { n++; continue; }
                float o = dd * (1.0f / 8.0f);
                if (o > 1.0f) o = 1.0f;
                if (dd > 24.0f) o *= 24.0f / dd;            // range falloff
                occ += o; n++;
            }
            if (!n) continue;
            float ao = 1.0f - S * (occ / (float)n);
            float v = (float)img[(size_t)i * w + j] * ao;
            img[(size_t)i * w + j] = to_u8(v);
        }
}

// ---------------------------------------------------------------------------
// LOD-matched rendering
// ---------------------------------------------------------------------------
int mc_render_pick_lod(const mc_sample_lods *ls, float vox_per_pixel) {
    if (!ls || ls->nlods <= 1) return 0;
    int L = 0;
    float v = vox_per_pixel;
    while (v >= 2.0f && L < ls->nlods - 1) { v *= 0.5f; L++; }
    // skip levels the caller left empty
    while (L > 0 && (ls->lods[L].nz <= 0 || !ls->lods[L].block)) L--;
    return L;
}

float mc_quad_spacing(const mc_quad *q) {
    if (!q || !q->grid || q->gw < 2 || q->gh < 1) return 1.0f;
    // probe up to 32 horizontal neighbor pairs along the grid diagonal
    double sum = 0.0;
    int n = 0;
    int probes = q->gh < 32 ? q->gh : 32;
    for (int i = 0; i < probes; i++) {
        int gy = (int)(((int64_t)i * (q->gh - 1)) / (probes > 1 ? probes - 1 : 1));
        int gx = (int)(((int64_t)i * (q->gw - 2)) / (probes > 1 ? probes - 1 : 1));
        const float *a = q->grid + ((size_t)gy * q->gw + gx) * 3;
        const float *b = a + 3;
        if (!qvalid(a) || !qvalid(b)) continue;
        float dz = b[0] - a[0], dy = b[1] - a[1], dx = b[2] - a[2];
        sum += sqrtf(dz * dz + dy * dy + dx * dx);
        n++;
    }
    return n ? (float)(sum / n) : 1.0f;
}

// wrap a rowgen: remap generated LOD-0 points into LOD-L voxel space.
// c_L = c_0 * 2^-L + (0.5 * 2^-L - 0.5); border points that map a fraction
// below 0 clamp to 0 (they are inside voxel 0 of the coarse level) instead
// of tripping the <0 invalid convention.
typedef struct {
    rowgen_fn inner;
    const void *inner_ud;
    float a, b;
} lod_rg;

static void lod_rowgen(const void *ud, int row, int w,
                       float *pts, float *normals) {
    const lod_rg *g = ud;
    g->inner(g->inner_ud, row, w, pts, normals);
    for (int j = 0; j < w; j++) {
        float *P = pts + (size_t)j * 3;
        if (!pt_valid(P)) continue;
        for (int k = 0; k < 3; k++) {
            float v = P[k] * g->a + g->b;
            P[k] = v < 0.0f ? 0.0f : v;
        }
    }
    // normals are directions: unchanged under uniform scaling
}

static int render_lod(const mc_sample_lods *ls, int L,
                      rowgen_fn inner, const void *inner_ud,
                      int w, int h, const mc_render_params *p,
                      uint8_t *out, int nthreads) {
    const float inv = 1.0f / (float)(1 << L);
    lod_rg g = { inner, inner_ud, inv, 0.5f * inv - 0.5f };
    mc_render_params pl_ = *p;
    pl_.t0 = p->t0 * inv;       // same physical slab ...
    pl_.t1 = p->t1 * inv;       // ... stepped at the coarse level's pitch
    render_bands(&ls->lods[L], NULL, NULL, lod_rowgen, &g, w, h, &pl_,
                 out, nthreads);
    return 0;
}

int mc_render_plane_lod(const mc_sample_lods *ls, const mc_plane *pl,
                        int w, int h, float scale,
                        const mc_render_params *p, uint8_t *out, int nthreads) {
    if (!ls || !pl || !p || !out || w <= 0 || h <= 0) return -1;
    int L = mc_render_pick_lod(ls, scale);
    if (L == 0)
        return mc_render_plane(&ls->lods[0], pl, w, h, scale, p, out, nthreads);
    plane_rg g = { *pl, scale, (float)w * 0.5f, (float)h * 0.5f };
    v3_norm(g.pl.normal);
    return render_lod(ls, L, plane_rowgen, &g, w, h, p, out, nthreads);
}

int mc_render_quad_lod(const mc_sample_lods *ls, const mc_quad *q,
                       float x0, float y0, float step, int w, int h,
                       const mc_render_params *p, uint8_t *out, int nthreads) {
    if (!ls || !q || !q->grid || q->gw < 1 || q->gh < 1 ||
        !p || !out || w <= 0 || h <= 0) return -1;
    int L = mc_render_pick_lod(ls, step * mc_quad_spacing(q));
    if (L == 0)
        return mc_render_quad(&ls->lods[0], q, x0, y0, step, w, h, p, out,
                              nthreads);
    quad_rg g = { q, x0, y0, step };
    return render_lod(ls, L, quad_rowgen, &g, w, h, p, out, nthreads);
}

// ---------------------------------------------------------------------------
// 3D resampling (surface-aligned volumes)
// ---------------------------------------------------------------------------
typedef struct {
    const mc_sample_src *src;
    const mc_quad *q;
    float x0, y0, step;
    float t0, dt;
    int w, h, nlayers;
    mc_filter f;
    uint8_t *out;
    int row0, row1;
} qvol_band_t;

static void *qvol_band_main(void *ud) {
    qvol_band_t *b = ud;
    mc_sampler *s = mc_sampler_new(b->src);
    float *row = malloc((size_t)b->w * 6 * sizeof(float));
    const size_t layer = (size_t)b->w * b->h;
    if (!s || !row) {
        for (int k = 0; k < b->nlayers; k++)
            memset(b->out + layer * k + (size_t)b->row0 * b->w, 0,
                   (size_t)(b->row1 - b->row0) * b->w);
        free(row); mc_sampler_free(s);
        return NULL;
    }
    float *nrm = row + (size_t)b->w * 3;
    quad_rg g = { b->q, b->x0, b->y0, b->step };
    for (int i = b->row0; i < b->row1; i++) {
        quad_rowgen(&g, i, b->w, row, nrm);
        for (int j = 0; j < b->w; j++) {
            const float *P = row + (size_t)j * 3;
            const float *N = nrm + (size_t)j * 3;
            uint8_t *o = b->out + (size_t)i * b->w + j;
            if (!pt_valid(P)) {
                for (int k = 0; k < b->nlayers; k++) o[layer * k] = 0;
                continue;
            }
            float nz = N[0], ny = N[1], nx = N[2];
            float n2 = nz * nz + ny * ny + nx * nx;
            if (n2 >= 1e-12f && (n2 < 0.9998f || n2 > 1.0002f)) {
                float nl = 1.0f / sqrtf(n2);
                nz *= nl; ny *= nl; nx *= nl;
            }
            float pz = P[0] + b->t0 * nz, py = P[1] + b->t0 * ny,
                  px = P[2] + b->t0 * nx;
            const float sz_ = b->dt * nz, sy_ = b->dt * ny, sx_ = b->dt * nx;
            int k = 0;
            if (b->f == MC_FILTER_TRILINEAR) {
                for (; k + 4 <= b->nlayers; k += 4) {
                    float bz[4], by[4], bx[4], v4[4];
                    for (int t = 0; t < 4; t++) {
                        bz[t] = pz; by[t] = py; bx[t] = px;
                        pz += sz_; py += sy_; px += sx_;
                    }
                    mc_s_tri4(s, bz, by, bx, v4);
                    for (int t = 0; t < 4; t++)
                        o[layer * (k + t)] = to_u8(v4[t]);
                }
            }
            for (; k < b->nlayers; k++) {
                o[layer * k] = to_u8(mc_s_sample(s, pz, py, px, b->f));
                pz += sz_; py += sy_; px += sx_;
            }
        }
    }
    free(row);
    mc_sampler_free(s);
    return NULL;
}

int mc_sample_quad_volume(const mc_sample_src *src, const mc_quad *q,
                          float x0, float y0, float step, int w, int h,
                          float t0, float dt, int nlayers,
                          mc_filter f, uint8_t *out, int nthreads) {
    if (!src || !q || !q->grid || q->gw < 1 || q->gh < 1 ||
        !out || w <= 0 || h <= 0 || nlayers <= 0) return -1;
    if (nthreads <= 0) {
        long nc = sysconf(_SC_NPROCESSORS_ONLN);
        nthreads = nc > 0 ? (int)nc : 1;
    }
    if (nthreads > 16) nthreads = 16;
    if (nthreads > h)  nthreads = h;
    pthread_t th[16];
    qvol_band_t bands[16];
    int per = (h + nthreads - 1) / nthreads;
    int nb = 0;
    for (int t = 0; t < nthreads; t++) {
        int r0 = t * per, r1 = r0 + per > h ? h : r0 + per;
        if (r0 >= r1) break;
        bands[nb] = (qvol_band_t){ src, q, x0, y0, step, t0, dt,
                                   w, h, nlayers, f, out, r0, r1 };
        if (nthreads == 1 ||
            pthread_create(&th[nb], NULL, qvol_band_main, &bands[nb]) != 0) {
            qvol_band_main(&bands[nb]);
            continue;
        }
        nb++;
    }
    for (int t = 0; t < nb; t++) pthread_join(th[t], NULL);
    return 0;
}

int mc_sample_box(const mc_sample_src *src,
                  const float origin[3], const float du[3],
                  const float dv[3], const float dw[3],
                  int w, int h, int d,
                  mc_filter f, uint8_t *out, int nthreads) {
    if (!src || !origin || !du || !dv || !dw || !out ||
        w <= 0 || h <= 0 || d <= 0) return -1;
    // each depth slice is a plane render with the layer offset folded into
    // the origin; comp NONE so no normals are needed
    mc_render_params p = { .filter = f, .comp = MC_COMP_NONE };
    for (int k = 0; k < d; k++) {
        mc_plane pl;
        for (int c = 0; c < 3; c++) {
            // mc_plane_gen centers the image; sample with corner semantics
            pl.origin[c] = origin[c] + (float)k * dw[c] +
                           ((float)w * 0.5f) * du[c] + ((float)h * 0.5f) * dv[c];
            pl.normal[c] = 0.0f;
            pl.u[c] = du[c];
            pl.v[c] = dv[c];
        }
        if (mc_render_plane(src, &pl, w, h, 1.0f, &p,
                            out + (size_t)k * w * h, nthreads) != 0)
            return -1;
    }
    return 0;
}

