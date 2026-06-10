// mc_render — surface rendering over mc_sample. See mc_render.h.
#include "mc_render.h"
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

static uint8_t render_pixel(mc_sampler *s, const float *P, const float *N,
                            const mc_render_params *p) {
    if (!pt_valid(P)) return 0;
    if (p->comp == MC_COMP_NONE || !N)
        return to_u8(mc_sample_point(s, P[0], P[1], P[2], p->filter));
    float nz = N[0], ny = N[1], nx = N[2];
    float nl = sqrtf(nz * nz + ny * ny + nx * nx);
    if (nl < 1e-6f)
        return to_u8(mc_sample_point(s, P[0], P[1], P[2], p->filter));
    nz /= nl; ny /= nl; nx /= nl;

    float dt = p->dt > 0.0f ? p->dt : 1.0f;
    float t0 = p->t0, t1 = p->t1;
    if (t1 < t0) { float tmp = t0; t0 = t1; t1 = tmp; }

    float acc = 0.0f, A = 0.0f, mn = 255.0f, mx = 0.0f, sum = 0.0f;
    int cnt = 0;
    const float a_min = p->alpha_min < 0.0f ? 0.0f :
                        p->alpha_min > 0.99f ? 0.99f : p->alpha_min;
    const float a_op  = p->alpha_opacity <= 0.0f ? 1.0f :
                        p->alpha_opacity > 1.0f ? 1.0f : p->alpha_opacity;

    for (float t = t0; t <= t1 + 1e-4f; t += dt) {
        float v = mc_sample_point(s, P[0] + t * nz, P[1] + t * ny,
                                  P[2] + t * nx, p->filter);
        switch (p->comp) {
        case MC_COMP_MIN:  if (v < mn) mn = v; break;
        case MC_COMP_MAX:  if (v > mx) mx = v; break;
        case MC_COMP_MEAN: sum += v; break;
        case MC_COMP_ALPHA: {
            float a = a_op * (v / 255.0f - a_min) / (1.0f - a_min);
            if (a > 0.0f) {
                if (a > 1.0f) a = 1.0f;
                acc += (1.0f - A) * a * v;
                A   += (1.0f - A) * a;
                if (A >= 0.98f) { t = t1 + 1.0f; }   // early out
            }
            break;
        }
        default: break;
        }
        cnt++;
    }
    switch (p->comp) {
    case MC_COMP_MIN:  return to_u8(mn);
    case MC_COMP_MAX:  return to_u8(mx);
    case MC_COMP_MEAN: return to_u8(cnt ? sum / (float)cnt : 0.0f);
    case MC_COMP_ALPHA: return to_u8(acc);
    default:           return 0;
    }
}

void mc_render_points(mc_sampler *s,
                      const float *pts, const float *normals,
                      int w, int h, const mc_render_params *p, uint8_t *out) {
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            size_t k = (size_t)i * w + j;
            out[k] = render_pixel(s, pts + k * 3,
                                  normals ? normals + k * 3 : NULL, p);
        }
}

// ---------------------------------------------------------------------------
// parallel core: row bands, one sampler per worker
// ---------------------------------------------------------------------------
typedef struct {
    const mc_sample_src *src;
    const float *pts, *normals;
    int w, h;
    const mc_render_params *p;
    uint8_t *out;
    int row0, row1;
} band_t;

static void *band_main(void *ud) {
    band_t *b = ud;
    mc_sampler *s = mc_sampler_new(b->src);
    if (!s) return NULL;
    size_t off = (size_t)b->row0 * b->w;
    mc_render_points(s, b->pts + off * 3,
                     b->normals ? b->normals + off * 3 : NULL,
                     b->w, b->row1 - b->row0, b->p, b->out + off);
    mc_sampler_free(s);
    return NULL;
}

void mc_render_points_par(const mc_sample_src *src,
                          const float *pts, const float *normals,
                          int w, int h, const mc_render_params *p,
                          uint8_t *out, int nthreads) {
    if (w <= 0 || h <= 0) return;
    if (nthreads <= 0) {
        long nc = sysconf(_SC_NPROCESSORS_ONLN);
        nthreads = nc > 0 ? (int)nc : 1;
    }
    if (nthreads > 16) nthreads = 16;
    if (nthreads > h)  nthreads = h;
    if (nthreads <= 1) {
        mc_sampler *s = mc_sampler_new(src);
        if (!s) { memset(out, 0, (size_t)w * h); return; }
        mc_render_points(s, pts, normals, w, h, p, out);
        mc_sampler_free(s);
        return;
    }
    pthread_t th[16];
    band_t bands[16];
    int per = (h + nthreads - 1) / nthreads;
    int nb = 0;
    for (int t = 0; t < nthreads; t++) {
        int r0 = t * per, r1 = r0 + per > h ? h : r0 + per;
        if (r0 >= r1) break;
        bands[nb] = (band_t){ src, pts, normals, w, h, p, out, r0, r1 };
        if (pthread_create(&th[nb], NULL, band_main, &bands[nb]) != 0) {
            band_main(&bands[nb]);          // degrade to inline
            continue;
        }
        nb++;
    }
    for (int t = 0; t < nb; t++) pthread_join(th[t], NULL);
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
// conveniences
// ---------------------------------------------------------------------------
int mc_render_plane(const mc_sample_src *src, const mc_plane *pl,
                    int w, int h, float scale,
                    const mc_render_params *p, uint8_t *out, int nthreads) {
    if (!src || !pl || !p || !out || w <= 0 || h <= 0) return -1;
    int need_n = p->comp != MC_COMP_NONE;
    float *pts = malloc((size_t)w * h * 3 * sizeof(float) * (need_n ? 2 : 1));
    if (!pts) return -1;
    float *nrm = need_n ? pts + (size_t)w * h * 3 : NULL;
    mc_plane_gen(pl, w, h, scale, pts, nrm);
    mc_render_points_par(src, pts, nrm, w, h, p, out, nthreads);
    free(pts);
    return 0;
}

int mc_render_quad(const mc_sample_src *src, const mc_quad *q,
                   float x0, float y0, float step, int w, int h,
                   const mc_render_params *p, uint8_t *out, int nthreads) {
    if (!src || !q || !q->grid || q->gw < 1 || q->gh < 1 ||
        !p || !out || w <= 0 || h <= 0) return -1;
    int need_n = p->comp != MC_COMP_NONE;
    float *pts = malloc((size_t)w * h * 3 * sizeof(float) * (need_n ? 2 : 1));
    if (!pts) return -1;
    float *nrm = need_n ? pts + (size_t)w * h * 3 : NULL;
    mc_quad_gen(q, x0, y0, step, w, h, pts, nrm);
    mc_render_points_par(src, pts, nrm, w, h, p, out, nthreads);
    free(pts);
    return 0;
}
