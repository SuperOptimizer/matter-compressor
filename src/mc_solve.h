/* mc_solve.h — a small block-sparse nonlinear least-squares solver (no deps).
 *
 * Built for the surface tracer (see the classical no-ML tracing port): the
 * problems are many 3-DOF parameter blocks (3D points) tied together by cheap
 * residuals (distance/straightness/metric/volume terms), each touching a few
 * blocks. This is the role Ceres plays in VC3D; here it is a dependency-free
 * Levenberg-Marquardt solver with caller-supplied ANALYTIC jacobians (every
 * tracer residual is closed-form, so no autodiff is needed).
 *
 * Model: minimize  sum_i || r_i(x) ||^2  over parameter blocks x_b (each a
 * contiguous run of `bdim` doubles). A residual i has `rdim` outputs and reads
 * `nblk` parameter blocks; its evaluate() fills the residual values and, when
 * asked, the per-block jacobians (row-major rdim x bdim each).
 *
 * Blocks can be held CONSTANT (frozen) — their columns are dropped from the
 * normal equations, exactly like ceres SetParameterBlockConstant. This is how
 * the tracer freezes everything outside the local optimization radius.
 *
 * The normal equations J^T J (dense, over the free blocks only) are solved by
 * LDL^T; the local tracer problems are small (tens to a few hundred free
 * blocks) so a dense factor per LM step is fine and simple.
 */
#ifndef MC_SOLVE_H
#define MC_SOLVE_H

#include <stddef.h>

// A residual's evaluation callback. `params` is an array of `nblk` pointers to
// the (read-only) current parameter blocks for this residual, in the order they
// were registered. Write `rdim` residual values into `residual`. If `jac` is
// non-NULL, jac[k] (when non-NULL itself) points to an rdim*bdim row-major
// buffer to receive d(residual)/d(block k); a NULL jac[k] means that block is
// constant and its jacobian is not needed. `user` is the per-residual context.
// Return 0 on success, non-zero to signal the residual could not be evaluated
// (treated as a hard failure of the solve).
typedef int (*mc_residual_fn)(void *user, const double *const *params,
                              double *residual, double *const *jac);

typedef struct mc_problem mc_problem;

// Create/destroy a problem over `nblocks` parameter blocks each of `bdim`
// doubles. `x` points to the contiguous parameter array (nblocks*bdim doubles),
// owned by the caller; the solver reads and writes it in place.
mc_problem *mc_problem_create(int nblocks, int bdim, double *x);
void        mc_problem_free(mc_problem *p);

// Mark a block constant (frozen) / variable. Default: all variable.
void mc_problem_set_const(mc_problem *p, int block, int is_const);

// Add a residual: `fn`/`user` evaluate it, it has `rdim` outputs, and reads the
// `nblk` blocks listed in `blocks`. An optional robust loss softens outliers:
// loss_scale > 0 applies a Cauchy loss with that scale (rho(s)=c^2*log(1+s/c^2),
// matching ceres CauchyLoss); loss_scale <= 0 means plain squared loss.
// Returns 0 on success, <0 on error (bad block index / OOM).
int mc_problem_add(mc_problem *p, mc_residual_fn fn, void *user,
                   int rdim, int nblk, const int *blocks, double loss_scale);

typedef struct {
    int    max_iter;        // LM iterations            (default 50)
    double init_lambda;     // initial damping          (default 1e-3)
    double func_tol;        // stop if cost drop < this fraction (default 1e-6)
    double grad_tol;        // stop if max |gradient| < this     (default 1e-8)
    double param_tol;       // stop if max |step| < this         (default 1e-8)
    int    verbose;         // print per-iteration cost to stderr
} mc_solve_opts;

void mc_solve_opts_default(mc_solve_opts *o);

typedef struct {
    double initial_cost;    // 0.5 * sum r^2 at start
    double final_cost;      // 0.5 * sum r^2 at end
    int    iterations;      // LM iterations taken
    int    success;         // 1 if it ran (even if it didn't move much)
} mc_solve_report;

// Solve in place (updates the caller's `x`). `opts` NULL -> defaults.
// Returns 0 on success, <0 on a hard error (residual eval failed / OOM /
// singular system that couldn't be damped).
int mc_solve(mc_problem *p, const mc_solve_opts *opts, mc_solve_report *rep);

#endif /* MC_SOLVE_H */
