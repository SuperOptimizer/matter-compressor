// mc_sample_internal — the sampler's guts, shared between mc_sample.c and
// mc_render.c so the per-sample hot path inlines into the render loops
// (a cross-TU call per voxel sample costs more than the sample itself).
// Not installed; consumers use mc_sample.h / mc_render.h.
#ifndef MC_SAMPLE_INTERNAL_H
#define MC_SAMPLE_INTERNAL_H
#include "mc_sample.h"
#include <math.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__SSE4_1__)
#include <immintrin.h>
#endif

#define MC_S_MEMO 256   // covers an oblique 1024-px row's block working set

typedef struct {
    int bz, by, bx;             // -1 = empty
    const uint8_t *ptr;         // NULL = known-absent (sampled as 0)
    uint8_t buf[4096];
} mc_s_memo;

struct mc_sampler {
    mc_sample_src src;
    int nbz, nby, nbx;
    int lbz, lby, lbx;          // last block touched (ray-coherence cache)
    const uint8_t *lptr;
    mc_s_memo m[MC_S_MEMO];
};

static inline const uint8_t *mc_s_block(mc_sampler *s, int bz, int by, int bx) {
    if (bz == s->lbz && by == s->lby && bx == s->lbx) return s->lptr;
    unsigned h = ((unsigned)bz * 73856093u) ^ ((unsigned)by * 19349663u) ^
                 ((unsigned)bx * 83492791u);
    mc_s_memo *e = &s->m[h & (MC_S_MEMO - 1)];
    if (!(e->bz == bz && e->by == by && e->bx == bx)) {
        e->bz = bz; e->by = by; e->bx = bx;
        e->ptr = s->src.block(&s->src, bz, by, bx, e->buf);
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
    // blocked fast path: all 8 corners inside one block and in bounds (~82%
    // of uniformly distributed samples; far more for coherent rays)
    if (!s->src.dense &&
        (unsigned)z0 < (unsigned)(s->src.nz - 1) &&
        (unsigned)y0 < (unsigned)(s->src.ny - 1) &&
        (unsigned)x0 < (unsigned)(s->src.nx - 1) &&
        (z0 & 15) != 15 && (y0 & 15) != 15 && (x0 & 15) != 15) {
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
    // slow path: block/bounds handled per corner (edges mix with 0)
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

static inline float mc_s_sample(mc_sampler *s, float z, float y, float x,
                                mc_filter f) {
    if (!(z == z) || !(y == y) || !(x == x)) return 0.0f;   // NaN
    return f == MC_FILTER_NEAREST ? mc_s_nearest(s, z, y, x)
                                  : mc_s_trilinear(s, z, y, x);
}

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
        // blocked: every lane's 8 corners must sit inside one block
        uint32x4_t in15 = vmvnq_u32(vorrq_u32(vorrq_u32(
            vceqq_s32(vandq_s32(zi, vdupq_n_s32(15)), vdupq_n_s32(15)),
            vceqq_s32(vandq_s32(yi, vdupq_n_s32(15)), vdupq_n_s32(15))),
            vceqq_s32(vandq_s32(xi, vdupq_n_s32(15)), vdupq_n_s32(15))));
        if (vminvq_u32(in15) != 0) {
            int32_t z0[4], y0[4], x0[4];
            vst1q_s32(z0, zi); vst1q_s32(y0, yi); vst1q_s32(x0, xi);
            const uint8_t *b[4];
            int allb = 1;
            for (int k = 0; k < 4; k++) {
                const uint8_t *bk =
                    mc_s_block(s, z0[k] >> 4, y0[k] >> 4, x0[k] >> 4);
                if (!bk) { allb = 0; break; }
                b[k] = bk + (((z0[k] & 15) << 8) | ((y0[k] & 15) << 4) |
                             (x0[k] & 15));
            }
            if (allb) {
                vst1q_f32(out, mc_s_lerp8x4(b[0], b[1], b[2], b[3],
                                            256, 16, dz, dy, dx));
                return;
            }
        }
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
        int in15 = 1;
        for (int k = 0; k < 4; k++)
            in15 &= (z0[k] & 15) != 15 && (y0[k] & 15) != 15 &&
                    (x0[k] & 15) != 15;
        if (in15) {
            const uint8_t *b[4];
            int allb = 1;
            for (int k = 0; k < 4; k++) {
                const uint8_t *bk =
                    mc_s_block(s, z0[k] >> 4, y0[k] >> 4, x0[k] >> 4);
                if (!bk) { allb = 0; break; }
                b[k] = bk + (((z0[k] & 15) << 8) | ((y0[k] & 15) << 4) |
                             (x0[k] & 15));
            }
            if (allb) {
                _mm_storeu_ps(out, mc_s_lerp8x4(b[0], b[1], b[2], b[3],
                                                256, 16, dz, dy, dx));
                return;
            }
        }
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
        int fast = 1;
        for (int k = 0; k < 8 && fast; k++)
            fast = (z0[k] & 15) != 15 && (y0[k] & 15) != 15 &&
                   (x0[k] & 15) != 15;
        if (fast) {
            for (int k = 0; k < 8 && fast; k++) {
                const uint8_t *bk =
                    mc_s_block(s, z0[k] >> 4, y0[k] >> 4, x0[k] >> 4);
                if (!bk) { fast = 0; break; }
                b[k] = bk + (((z0[k] & 15) << 8) | ((y0[k] & 15) << 4) |
                             (x0[k] & 15));
            }
            if (fast) {
                _mm256_storeu_ps(out, mc_s_lerp8x8(b, 256, 16, dz, dy, dx));
                return;
            }
        }
    }
    mc_s_tri4(s, pz, py, px, out);
    mc_s_tri4(s, pz + 4, py + 4, px + 4, out + 4);
}
#endif

#endif
