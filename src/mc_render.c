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

// Composite one ray. Trilinear rays are consumed in chunks of 4 steps via
// mc_s_tri4 (NEON gather+lerp on aarch64, ~1.4x the scalar core); the
// accumulation itself stays sequential per chunk, which keeps ALPHA\'s
// front-to-back order and early-out exact. Positions are P + t*N with t
// advanced additively, as before.
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

    const float sz_ = cfg->dt * nz, sy_ = cfg->dt * ny, sx_ = cfg->dt * nx;
    float pz = P[0] + cfg->t0 * nz, py = P[1] + cfg->t0 * ny,
          px = P[2] + cfg->t0 * nx;
    const float a_th = cfg->a_min, a_sc = cfg->a_op / (1.0f - cfg->a_min);
    float acc = 0.0f, A = 0.0f, mn = 255.0f, mx = 0.0f, sum = 0.0f;
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
                break;
            case MC_COMP_MEAN:
                sum += v4[0] + v4[1] + v4[2] + v4[3];
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
