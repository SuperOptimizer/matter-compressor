// mc_sample — point sampling over blocked u8 volumes. See mc_sample.h.
// Hot-path inlines live in mc_sample_internal.h (shared with mc_render.c).
#include "mc_sample_internal.h"
#include "matter_compressor.h"
#include <stdlib.h>
#include <string.h>

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
    mc_sampler_reset(s);
    return s;
}

void mc_sampler_free(mc_sampler *s) { free(s); }

void mc_sampler_reset(mc_sampler *s) {
    if (!s) return;
    for (int i = 0; i < MC_S_MEMO; i++) s->m[i].bz = -1;
    s->lbz = s->lby = s->lbx = -1;
    s->lptr = NULL;
}

float mc_sample_point(mc_sampler *s, float z, float y, float x, mc_filter f) {
    return mc_s_sample(s, z, y, x, f);
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
