// mc_render — surface rendering over mc_sample. See mc_render.h.
#include "mc_render.h"
#include "mc_sample_internal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// core
// ---------------------------------------------------------------------------
static inline int pt_valid(const float *p) {
    if (p[0] != p[0] || p[1] != p[1] || p[2] != p[2]) return 0;
    return p[0] >= 0.0f && p[1] >= 0.0f && p[2] >= 0.0f;
}

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
    return c;
}

// Composite one ray: walk pos += dvec for nsteps, reduce. The position is
// stepped incrementally (3 adds) instead of recomputing P + t*N (3 fmas);
// the additive t accumulation matches the previous behavior exactly.
#define RAY_LOOP(INIT, STEP, FINISH)                                          \
    do {                                                                      \
        float pz = P[0] + cfg->t0 * nz, py = P[1] + cfg->t0 * ny,             \
              px = P[2] + cfg->t0 * nx;                                       \
        const float sz_ = cfg->dt * nz, sy_ = cfg->dt * ny,                   \
                    sx_ = cfg->dt * nx;                                       \
        INIT;                                                                 \
        for (int it = 0; it < cfg->nsteps; it++) {                            \
            float v = mc_s_sample(s, pz, py, px, cfg->filter);                \
            STEP;                                                             \
            pz += sz_; py += sy_; px += sx_;                                  \
        }                                                                     \
        FINISH;                                                               \
    } while (0)

static uint8_t render_pixel(mc_sampler *s, const float *P, const float *N,
                            const rcfg_t *cfg) {
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

    switch (cfg->comp) {
    case MC_COMP_MIN: {
        uint8_t r;
        RAY_LOOP(float mn = 255.0f, if (v < mn) mn = v, r = to_u8(mn));
        return r;
    }
    case MC_COMP_MAX: {
        uint8_t r;
        RAY_LOOP(float mx = 0.0f, if (v > mx) mx = v, r = to_u8(mx));
        return r;
    }
    case MC_COMP_MEAN: {
        uint8_t r;
        RAY_LOOP(float sum = 0.0f, sum += v,
                 r = to_u8(cfg->nsteps ? sum / (float)cfg->nsteps : 0.0f));
        return r;
    }
    case MC_COMP_ALPHA: {
        uint8_t r;
        const float a_th = cfg->a_min, a_sc = cfg->a_op / (1.0f - cfg->a_min);
        RAY_LOOP(float acc = 0.0f; float A = 0.0f,
                 {
                     float a = (v * (1.0f / 255.0f) - a_th) * a_sc;
                     if (a > 0.0f) {
                         if (a > 1.0f) a = 1.0f;
                         acc += (1.0f - A) * a * v;
                         A   += (1.0f - A) * a;
                         if (A >= 0.98f) break;
                     }
                 },
                 r = to_u8(acc));
        return r;
    }
    default: return 0;
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
            for (size_t k = 0; k < n; k++) {
                const float *P = pts + k * 3;
                out[k] = pt_valid(P)
                             ? to_u8(mc_s_trilinear(s, P[0], P[1], P[2])) : 0;
            }
        }
        return;
    }
    for (size_t k = 0; k < n; k++)
        out[k] = render_pixel(s, pts + k * 3, normals + k * 3, &cfg);
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
} band_t;

static void *band_main(void *ud) {
    band_t *b = ud;
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

static void render_bands(const mc_sample_src *src,
                         const float *pts, const float *normals,
                         rowgen_fn rowgen, const void *rg_ud,
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
                              w, h, p, out, r0, r1 };
        if (nthreads == 1) { band_main(&bands[nb]); continue; }
        if (pthread_create(&th[nb], NULL, band_main, &bands[nb]) != 0) {
            band_main(&bands[nb]);          // degrade to inline
            continue;
        }
        nb++;
    }
    for (int t = 0; t < nb; t++) pthread_join(th[t], NULL);
}

void mc_render_points_par(const mc_sample_src *src,
                          const float *pts, const float *normals,
                          int w, int h, const mc_render_params *p,
                          uint8_t *out, int nthreads) {
    render_bands(src, pts, normals, NULL, NULL, w, h, p, out, nthreads);
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
static inline const float *qpt(const mc_quad *q, int gx, int gy) {
    return q->grid + ((size_t)gy * q->gw + gx) * 3;
}
static inline int qvalid(const float *p) {
    return !(p[0] == -1.0f && p[1] == -1.0f && p[2] == -1.0f) &&
           p[0] == p[0] && p[1] == p[1] && p[2] == p[2];
}

void mc_quad_gen(const mc_quad *q, float x0, float y0, float step,
                 int w, int h, float *pts, float *normals) {
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            float *P = pts + ((size_t)i * w + j) * 3;
            float *N = normals ? normals + ((size_t)i * w + j) * 3 : NULL;
            P[0] = P[1] = P[2] = -1.0f;
            if (N) N[0] = N[1] = N[2] = 0.0f;
            float gx = x0 + (float)j * step;
            float gy = y0 + (float)i * step;
            if (gx < 0.0f || gy < 0.0f ||
                gx > (float)(q->gw - 1) || gy > (float)(q->gh - 1)) continue;
            int x0i = (int)gx, y0i = (int)gy;
            if (x0i > q->gw - 2) x0i = q->gw - 2;
            if (y0i > q->gh - 2) y0i = q->gh - 2;
            if (x0i < 0 || y0i < 0) {       // degenerate 1-wide grids
                if (q->gw == 1) x0i = 0;
                if (q->gh == 1) y0i = 0;
            }
            float fx = gx - (float)x0i, fy = gy - (float)y0i;
            const float *p00 = qpt(q, x0i, y0i);
            const float *p01 = qpt(q, x0i + (q->gw > 1), y0i);
            const float *p10 = qpt(q, x0i, y0i + (q->gh > 1));
            const float *p11 = qpt(q, x0i + (q->gw > 1), y0i + (q->gh > 1));
            if (!qvalid(p00) || !qvalid(p01) || !qvalid(p10) || !qvalid(p11))
                continue;
            float du[3], dv[3];
            for (int k = 0; k < 3; k++) {
                float a = p00[k] + (p01[k] - p00[k]) * fx;
                float b = p10[k] + (p11[k] - p10[k]) * fx;
                P[k] = a + (b - a) * fy;
                // bilinear tangents (per grid cell, exact for the patch)
                du[k] = (p01[k] - p00[k]) * (1.0f - fy) + (p11[k] - p10[k]) * fy;
                dv[k] = (p10[k] - p00[k]) * (1.0f - fx) + (p11[k] - p01[k]) * fx;
            }
            if (N) { v3_cross(du, dv, N); v3_norm(N); }
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
