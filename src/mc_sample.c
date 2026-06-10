// mc_sample — point sampling over blocked u8 volumes. See mc_sample.h.
#include "mc_sample.h"
#include "matter_compressor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    s.block = dense_block;
    s.nz = nz; s.ny = ny; s.nx = nx;
    return s;
}

// ---------------------------------------------------------------------------
// sampler: direct-mapped block memo
// ---------------------------------------------------------------------------
#define MEMO 64
typedef struct {
    int bz, by, bx;             // -1 = empty
    const uint8_t *ptr;         // NULL = known-absent (sampled as 0)
    uint8_t buf[BLKB];
} memo_t;

struct mc_sampler {
    mc_sample_src src;
    int nbz, nby, nbx;
    memo_t m[MEMO];
};

mc_sampler *mc_sampler_new(const mc_sample_src *src) {
    if (!src || !src->block) return NULL;
    mc_sampler *s = malloc(sizeof *s);
    if (!s) return NULL;
    s->src = *src;
    s->nbz = (src->nz + BLK - 1) / BLK;
    s->nby = (src->ny + BLK - 1) / BLK;
    s->nbx = (src->nx + BLK - 1) / BLK;
    mc_sampler_reset(s);
    return s;
}

void mc_sampler_free(mc_sampler *s) { free(s); }

void mc_sampler_reset(mc_sampler *s) {
    if (!s) return;
    for (int i = 0; i < MEMO; i++) s->m[i].bz = -1;
}

static inline const uint8_t *sampler_block(mc_sampler *s,
                                           int bz, int by, int bx) {
    unsigned h = ((unsigned)bz * 73856093u) ^ ((unsigned)by * 19349663u) ^
                 ((unsigned)bx * 83492791u);
    memo_t *e = &s->m[h & (MEMO - 1)];
    if (e->bz == bz && e->by == by && e->bx == bx) return e->ptr;
    e->bz = bz; e->by = by; e->bx = bx;
    e->ptr = s->src.block(&s->src, bz, by, bx, e->buf);
    return e->ptr;
}

static inline float voxel(mc_sampler *s, int z, int y, int x) {
    if ((unsigned)z >= (unsigned)s->src.nz ||
        (unsigned)y >= (unsigned)s->src.ny ||
        (unsigned)x >= (unsigned)s->src.nx) return 0.0f;
    const uint8_t *b = sampler_block(s, z >> 4, y >> 4, x >> 4);
    return b ? (float)b[((z & 15) << 8) | ((y & 15) << 4) | (x & 15)] : 0.0f;
}

float mc_sample_point(mc_sampler *s, float z, float y, float x, mc_filter f) {
    if (!(z == z) || !(y == y) || !(x == x)) return 0.0f;   // NaN
    if (f == MC_FILTER_NEAREST) {
        // round to nearest voxel center (coords are voxel-center based)
        int zi = (int)floorf(z + 0.5f);
        int yi = (int)floorf(y + 0.5f);
        int xi = (int)floorf(x + 0.5f);
        return voxel(s, zi, yi, xi);
    }
    // trilinear between the 8 surrounding voxel centers; edges mix with 0
    float zf = floorf(z), yf = floorf(y), xf = floorf(x);
    int z0 = (int)zf, y0 = (int)yf, x0 = (int)xf;
    float dz = z - zf, dy = y - yf, dx = x - xf;
    float c000 = voxel(s, z0, y0, x0),     c001 = voxel(s, z0, y0, x0 + 1);
    float c010 = voxel(s, z0, y0 + 1, x0), c011 = voxel(s, z0, y0 + 1, x0 + 1);
    float c100 = voxel(s, z0 + 1, y0, x0), c101 = voxel(s, z0 + 1, y0, x0 + 1);
    float c110 = voxel(s, z0 + 1, y0 + 1, x0);
    float c111 = voxel(s, z0 + 1, y0 + 1, x0 + 1);
    float c00 = c000 + (c001 - c000) * dx;
    float c01 = c010 + (c011 - c010) * dx;
    float c10 = c100 + (c101 - c100) * dx;
    float c11 = c110 + (c111 - c110) * dx;
    float c0 = c00 + (c01 - c00) * dy;
    float c1 = c10 + (c11 - c10) * dy;
    return c0 + (c1 - c0) * dz;
}

static inline int pt_valid(const float *p) {
    if (p[0] != p[0] || p[1] != p[1] || p[2] != p[2]) return 0;   // NaN
    return p[0] >= 0.0f && p[1] >= 0.0f && p[2] >= 0.0f;
}

void mc_sample_points(mc_sampler *s, const float *zyx, size_t n,
                      mc_filter f, float *out) {
    for (size_t i = 0; i < n; i++) {
        const float *p = zyx + i * 3;
        out[i] = pt_valid(p) ? mc_sample_point(s, p[0], p[1], p[2], f) : 0.0f;
    }
}

void mc_sample_points_u8(mc_sampler *s, const float *zyx, size_t n,
                         mc_filter f, uint8_t *out) {
    for (size_t i = 0; i < n; i++) {
        const float *p = zyx + i * 3;
        float v = pt_valid(p) ? mc_sample_point(s, p[0], p[1], p[2], f) : 0.0f;
        out[i] = (uint8_t)(v < 0.0f ? 0 : v > 255.0f ? 255 : (int)(v + 0.5f));
    }
}
