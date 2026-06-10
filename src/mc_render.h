// mc_render — surface rendering over mc_sample: volume-cartographer-style
// images of quad and plane surfaces, with compositing along surface normals.
//
// The geometry contract is deliberately generic: the core renderer takes a
// dense W*H grid of 3D points (plus optional per-point normals) and produces
// a W*H u8 image. Plane and quad surfaces are point-grid *generators* on
// top of that core, mirroring volume-cartographer's PlaneSurface and
// QuadSurface::gen() — VC hands over its control-point grid (transposed to
// (z,y,x)) and gets pixels back; anything else that can produce a point
// grid renders the same way.
//
// Compositing: for each surface point P with unit normal N, the renderer
// samples P + t*N for t in [t0, t1] step dt and reduces:
//   MC_COMP_NONE  sample at P only (t range ignored) — a slice
//   MC_COMP_MIN   minimum along the ray
//   MC_COMP_MEAN  average along the ray
//   MC_COMP_MAX   maximum along the ray (VC's default composite)
//   MC_COMP_ALPHA front-to-back alpha: each sample contributes
//                 a = alpha_opacity * max(0, v/255 - alpha_min)/(1 - alpha_min);
//                 acc += (1-A)*a*v; A += (1-A)*a; early-out at A >= 0.98.
#ifndef MC_RENDER_H
#define MC_RENDER_H
#include "mc_sample.h"

typedef enum {
    MC_COMP_NONE  = 0,
    MC_COMP_MIN   = 1,
    MC_COMP_MEAN  = 2,
    MC_COMP_MAX   = 3,
    MC_COMP_ALPHA = 4,
} mc_comp;

typedef struct {
    mc_filter filter;       // MC_FILTER_NEAREST / MC_FILTER_TRILINEAR
    mc_comp   comp;         // reduction along the normal
    float t0, t1;           // composite range along the normal, in voxels
    float dt;               // step (<= 0 -> 1.0)
    float alpha_min;        // MC_COMP_ALPHA: value threshold in [0,1)
    float alpha_opacity;    // MC_COMP_ALPHA: per-sample opacity scale (0,1]
} mc_render_params;

// ---------------------------------------------------------------------------
// core: dense point grid -> image
// ---------------------------------------------------------------------------
// pts: W*H*3 floats, (z,y,x) per pixel. normals: W*H*3 unit (z,y,x) or NULL
// (required when comp != MC_COMP_NONE). A point with any coordinate < 0 or
// NaN renders 0 (volume-cartographer's invalid marker). out: W*H bytes.
void mc_render_points(mc_sampler *s,
                      const float *pts, const float *normals,
                      int w, int h, const mc_render_params *p, uint8_t *out);

// Parallel variant: same image, row bands across `nthreads` workers
// (0 -> one per core, capped at 16). Creates one sampler per worker over
// `src` (mc_sampler is single-threaded); src->block must be thread-safe
// (mc_cache is; a dense array trivially is).
void mc_render_points_par(const mc_sample_src *src,
                          const float *pts, const float *normals,
                          int w, int h, const mc_render_params *p,
                          uint8_t *out, int nthreads);

// ---------------------------------------------------------------------------
// plane surface (volume-cartographer PlaneSurface)
// ---------------------------------------------------------------------------
// A plane through `origin` with unit `normal`; `u` and `v` are the in-plane
// pixel axes. mc_plane_basis() builds an arbitrary stable (u,v) orthonormal
// pair from `normal` when you have no preferred orientation.
typedef struct {
    float origin[3];        // (z,y,x)
    float normal[3];        // unit (z,y,x)
    float u[3], v[3];       // unit in-plane axes: image x steps u, y steps v
} mc_plane;

void mc_plane_basis(mc_plane *pl);

// Generate the W*H point grid (and constant normals, if non-NULL) for the
// image whose pixel (i,j) sits at origin + (j - w/2)*scale*u +
// (i - h/2)*scale*v. `scale` = voxels per pixel (1 = native).
void mc_plane_gen(const mc_plane *pl, int w, int h, float scale,
                  float *pts, float *normals);

// ---------------------------------------------------------------------------
// quad surface (volume-cartographer QuadSurface)
// ---------------------------------------------------------------------------
// A control grid of gw*gh 3D points (z,y,x), row-major, VC's invalid marker
// (-1,-1,-1) honored. Rendering bilinearly interpolates the control grid to
// the output resolution and derives per-pixel normals from the grid
// tangents (du x dv, normalized) — VC's gen() contract.
typedef struct {
    const float *grid;      // gw*gh*3 (z,y,x)
    int gw, gh;
} mc_quad;

// Generate a W*H point grid (+ normals, if non-NULL) sampling the control
// grid over the rect [x0, x0+w*step) x [y0, y0+h*step) in grid units
// (step = grid cells per pixel; 1 renders the grid at native density;
// VC's render scale = 1/step). Pixels mapping outside the grid or onto
// invalid control points emit invalid (-1,-1,-1) points.
void mc_quad_gen(const mc_quad *q, float x0, float y0, float step,
                 int w, int h, float *pts, float *normals);

// ---------------------------------------------------------------------------
// one-call conveniences (gen + parallel render, scratch managed internally)
// ---------------------------------------------------------------------------
int mc_render_plane(const mc_sample_src *src, const mc_plane *pl,
                    int w, int h, float scale,
                    const mc_render_params *p, uint8_t *out, int nthreads);
int mc_render_quad(const mc_sample_src *src, const mc_quad *q,
                   float x0, float y0, float step, int w, int h,
                   const mc_render_params *p, uint8_t *out, int nthreads);

// ---------------------------------------------------------------------------
// LOD-matched rendering
// ---------------------------------------------------------------------------
// Zoomed-out views shouldn't sample the finest level: at `vox_per_pixel`
// voxels per output pixel, level floor(log2(vox_per_pixel)) carries all the
// information the image can show, with 8x fewer voxels per level. Geometry
// stays in LOD-0 voxel space; the renderer picks the level, remaps
// coordinates (half-voxel-center correct: c_L = (c_0 + 0.5)/2^L - 0.5) and
// scales the composite range so the slab covers the same physical depth,
// stepped at the sampled level's voxel pitch.
typedef struct {
    mc_sample_src lods[8];      // [0] = finest; dims halve per level
    int nlods;
} mc_sample_lods;

// floor(log2(vox_per_pixel)) clamped to [0, nlods-1]; <2 vox/px -> 0.
int mc_render_pick_lod(const mc_sample_lods *ls, float vox_per_pixel);

// Mean LOD-0 voxel spacing of one rendered pixel step across the quad's
// control grid (sparse probe; multiply by your render step).
float mc_quad_spacing(const mc_quad *q);

// As mc_render_plane / mc_render_quad, but sampling the LOD matched to
// the render scale (plane: vox/px = scale; quad: step * mc_quad_spacing).
int mc_render_plane_lod(const mc_sample_lods *ls, const mc_plane *pl,
                        int w, int h, float scale,
                        const mc_render_params *p, uint8_t *out, int nthreads);
int mc_render_quad_lod(const mc_sample_lods *ls, const mc_quad *q,
                       float x0, float y0, float step, int w, int h,
                       const mc_render_params *p, uint8_t *out, int nthreads);

#endif
