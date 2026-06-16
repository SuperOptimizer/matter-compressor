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

#endif /* MC_TRACE_H */
