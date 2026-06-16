/* mc_trace.h — classical (no-ML) surface tracer: grid + geometric cost terms.
 *
 * Ported from VC3D's GrowPatch/GrowSurface (which use Ceres). A traced surface
 * is a 2D grid of 3D points: grid cell (row,col) in (u,v) parameter space maps
 * to a 3D position in volume coords. The tracer grows this grid outward from a
 * seed, solving a local nonlinear least squares (mc_solve) over the new + nearby
 * points at each step, balancing geometric regularity terms and (later) volume-
 * data terms that lock the surface onto an actual sheet.
 *
 * This header exposes the surface grid container and the GEOMETRIC cost terms
 * (the ones that need no volume data): DistLoss, StraightLoss, SymmetricDirichlet.
 * Each is an mc_solve residual (analytic jacobians). The volume-locking terms
 * (SpaceLine / normal-field) and the growth orchestration come in later phases.
 *
 * Coordinate order: points are stored (x,y,z) per VC3D convention here (the
 * residuals are isotropic so the order is immaterial to them); the .grid export
 * later swizzles to mc's (z,y,x). Invalid cells are marked with NaN.
 */
#ifndef MC_TRACE_H
#define MC_TRACE_H

#include "mc_solve.h"

// A surface grid: gh rows x gw cols of 3D points (3 doubles each, row-major).
// p[(r*gw + c)*3 + {0,1,2}] = (x,y,z). A cell is invalid if its x is NaN.
typedef struct {
    int     gw, gh;
    double *p;          // gw*gh*3, owned
    double  unit;       // target 3D spacing between adjacent grid cells (voxels)
} mc_surf_grid;

mc_surf_grid *mc_surf_grid_create(int gw, int gh, double unit);
void          mc_surf_grid_free(mc_surf_grid *g);
int           mc_surf_cell_valid(const mc_surf_grid *g, int r, int c);
void          mc_surf_cell_set(mc_surf_grid *g, int r, int c, double x, double y, double z);
void          mc_surf_cell_invalidate(mc_surf_grid *g, int r, int c);

// ---- geometric cost terms, as mc_solve residuals -------------------------
// Each helper allocates a small context (owned by the grid `g`; all freed by
// mc_surf_grid_free) and registers the residual on `prob`. The block index for
// a grid cell is chosen by the caller (whatever block<->cell mapping it used
// when creating the problem). Default weights live in the .c.

// DistLoss: the 3D distance between two adjacent cells should equal `unit`.
//   r = w*(L/unit - 1)         if L >= unit
//   r = w*(unit/L - 1)         if L <  unit   (symmetric penalty, matches VC3D)
// 1 residual over 2 blocks.
int mc_trace_add_dist(mc_problem *prob, mc_surf_grid *g,
                      int blockA, int blockB, double weight);

// StraightLoss: three collinear cells (a-b-c) should be straight.
//   cos = dot(b-a, c-b)/(|b-a||c-b|); r = w*(1-cos) (+ sharp extra if angle>30deg)
// 1 residual over 3 blocks.
int mc_trace_add_straight(mc_problem *prob, mc_surf_grid *g,
                          int blockA, int blockB, int blockC, double weight);

// SymmetricDirichlet: the local parameterization (p at (r,c), pu at (r,c+1),
// pv at (r+1,c)) should be conformal+isometric at scale `unit`. Energy
// E = tr(G) + tr(G^-1) with G the first fundamental form of the unit-normalized
// frame; r = w*(E - 4) (optimum 4 at the identity). 1 residual over 3 blocks.
int mc_trace_add_sdir(mc_problem *prob, mc_surf_grid *g,
                      int blockP, int blockU, int blockV, double weight);

// ---- growth orchestration -------------------------------------------------
// Fringe-expansion surface growth (the role of VC3D's GrowPatch/GrowSurface).
// Starting from the already-valid cells in `g` (the seed — at minimum a small
// patch the caller has set), repeatedly: find invalid cells adjacent to valid
// ones, initialize each by extrapolating from its valid neighbors, and solve a
// LOCAL least-squares (the new cell + a radius of nearby valid cells free,
// everything else frozen) using the geometric terms — plus, when provided, a
// volume-data term that locks the surface onto a real sheet. A candidate is
// accepted if its post-solve local residual is below a threshold.

typedef struct {
    double w_dist;        // DistLoss weight            (default 1.0)
    double w_straight;    // StraightLoss weight        (default 0.5)
    double w_sdir;        // SymmetricDirichlet weight  (default 1.0)
    int    radius;        // local-solve free radius (cells) (default 3)
    int    max_gen;       // max generations            (default 100000)
    double accept_resid;  // max per-cell RMS residual to accept a candidate
                          //                            (default 0.5; <=0 = always)
    int    verbose;       // print per-generation progress to stderr

    // Optional volume-data term (phase 4): pulls each new cell toward a sheet.
    // If `vol_fn` is non-NULL it is added as a single-block residual on each
    // grown cell: vol_fn(user, xyz[3], &resid, grad_or_null[3]) writes the
    // residual value and, if grad != NULL, d(resid)/d(xyz). `vol_weight` scales
    // it. (Left NULL in phase 3 -> pure-geometry growth in free space.)
    int  (*vol_fn)(void *user, const double *xyz, double *resid, double *grad);
    void  *vol_user;
    double vol_weight;
} mc_grow_opts;

void mc_grow_opts_default(mc_grow_opts *o);

typedef struct {
    int generations;      // generations run
    int added;            // cells accepted/added
    int rejected;         // candidates rejected (residual too high / degenerate)
} mc_grow_report;

// Grow `g` in place. Returns 0 on success, <0 on a hard error.
int mc_trace_grow(mc_surf_grid *g, const mc_grow_opts *opts, mc_grow_report *rep);

// ---- volume-data term: pull cells onto a sheet's signed-distance zero set ----
// Wraps a dense float signed-distance field (e.g. mc_seg_sdt output, indexed
// (z,y,x) at sdf[(z*ny+y)*nx+x]) as a vol_fn for mc_grow_opts. The residual at
// a point is the trilinearly-sampled SDF value (0 on the sheet surface) and the
// gradient is the SDF's spatial gradient, so the solver drives cells onto the
// zero level set. Points are in grid (x,y,z) order; the sampler maps to the
// volume's (z,y,x). Out-of-bounds points yield a 0 residual + 0 gradient (no
// pull) so growth can still extrapolate past the sampled region.
//
// Usage: set opts.vol_fn = mc_trace_sdf_vol, opts.vol_user = &your mc_sdf_field,
// opts.vol_weight = e.g. 1.0.
typedef struct {
    const float *sdf;     // nz*ny*nx, (z,y,x) row-major (not owned)
    int nz, ny, nx;
} mc_sdf_field;

int mc_trace_sdf_vol(void *user, const double *xyz, double *resid, double *grad);

#endif /* MC_TRACE_H */
