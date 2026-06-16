/* mc_surface.h — parametric surface I/O for matter-compressor.
 *
 * A surface is a gw*gh control grid of points sampling a 3D volume, plus a
 * per-point DEPTH (half-thickness along the surface normal: depth d means the
 * surface spans [-d, +d] voxels each side, total thickness 2d). This is the
 * "2D image that samples a 3D volume to make a parametric surface" — VC's
 * QuadSurface idea, extended with per-point depth.
 *
 * On-disk: a 2D TIFF, gw x gh, 4x f32 per texel = (x, y, z, depth) in VOLUME
 * coordinates. (Note: file order is x,y,z; mc_quad's grid is z,y,x — load
 * swizzles.)
 *
 * mc_surface bridges to the existing renderer: mc_surface_quad() exposes the
 * grid as an mc_quad (z,y,x) for mc_render_quad / mc_sample_quad_volume; the
 * per-point depth feeds the thick-surface (multi-layer) sampling.
 */
#ifndef MC_SURFACE_H
#define MC_SURFACE_H

#include <stdint.h>
#include "matter_compressor.h"   /* mc_quad */

typedef struct {
    int    gw, gh;        // control-grid dims
    float *grid;          // gw*gh*3 (z,y,x) volume coords  (owned)
    float *depth;         // gw*gh half-thickness per point (owned), may be NULL
    float  mean_depth;    // average of valid depths (convenience)
} mc_surface;

// Load a surface from a 4x-f32 TIFF (x,y,z,depth). Returns 0 on success
// (fills *s, allocates grid+depth), <0 on error / wrong format. Free with
// mc_surface_free.
int  mc_surface_load_tiff(const char *path, mc_surface *s);

// Save a surface to a 4x-f32 TIFF (x,y,z,depth). If s->depth is NULL, writes
// the surface's mean_depth (or 0) for every point. Returns 0 on success.
int  mc_surface_save_tiff(const char *path, const mc_surface *s);

void mc_surface_free(mc_surface *s);

// Expose the grid as an mc_quad (z,y,x) for the existing quad renderer.
mc_quad mc_surface_quad(const mc_surface *s);

// ---- mesh / image interop ----
// Save the surface grid as a Wavefront OBJ: one vertex per valid grid point,
// quad faces between 4-valid neighbors (triangulated). Vertices are written in
// x y z order (OBJ convention). Invalid points (any coord < 0) are skipped and
// faces touching them are omitted. Returns 0 on success.
int  mc_surface_save_obj(const char *path, const mc_surface *s);

// Load a surface grid from an OBJ that was written on a regular gw x gh grid
// (e.g. by mc_surface_save_obj or VC). Requires the OBJ to carry the grid dims
// in a leading comment "# grid gw gh"; vertices fill the grid row-major, missing
// ones marked invalid. Returns 0 on success, <0 if no grid hint / parse error.
int  mc_surface_load_obj(const char *path, mc_surface *s);

// Write a w*h grayscale (1) or RGB (3) image as binary PPM (P6 always, gray is
// expanded to RGB). Simple, dependency-free render output. Returns 0 on success.
int  mc_ppm_write(const char *path, int w, int h, int channels, const uint8_t *pix);

#endif /* MC_SURFACE_H */
