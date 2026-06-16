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

// ---- general triangle mesh (real-world OBJ: arbitrary v/vn/vt/f) ----
// A loaded OBJ that isn't a regular grid (e.g. Vesuvius VolCart segment OBJs):
// vertices (x,y,z), optional per-vertex normals, and triangle faces (0-based
// vertex indices; polygons are fan-triangulated). Coordinates are kept in the
// file's order (x,y,z), i.e. volume coords.
typedef struct {
    int    nv, nt;        // vertex count, triangle count
    float *v;             // nv*3 (x,y,z)   (owned)
    float *vn;            // nv*3 normals or NULL  (owned)
    int   *tri;           // nt*3 vertex indices (0-based)  (owned)
} mc_mesh;

// Load any Wavefront OBJ as a triangle mesh (handles f a, a/b, a//c, a/b/c and
// negative indices; polygons fan-triangulated). Returns 0 on success.
int  mc_mesh_load_obj(const char *path, mc_mesh *m);
// Save a mesh as OBJ (v, optional vn, triangle f). Returns 0 on success.
int  mc_mesh_save_obj(const char *path, const mc_mesh *m);
void mc_mesh_free(mc_mesh *m);

// ---- VC per-pixel map (.ppm "ordered map"): the QuadSurface storage ----
// VC's segment .ppm is NOT a P6 image: a text header
//   width: W \n height: H \n dim: D \n ordered: true \n type: double \n
//   version: 1 \n <> \n  then W*H * D doubles, row-major. D=6 is xyz + normal.
// Load it into mc_surface: the xyz becomes the grid (z,y,x); depth is set to
// `default_depth` (their map stores a normal, not a thickness). A (0,0,0) map
// entry (no surface) becomes an invalid (-1,-1,-1) grid point. Returns 0 on
// success, <0 on parse error / unsupported header.
int  mc_surface_load_vcps_ppm(const char *path, mc_surface *s, float default_depth);

// ---- mesh -> grid resampling ----
// Resample an arbitrary triangle mesh (a sheet-like segment) onto a regular
// gw x gh control grid: fit the best plane through the vertices (PCA), project
// them to the plane's (u,v), then for each grid cell over the (u,v) bounding
// box take the nearest vertex's (x,y,z) (cells with no nearby vertex are
// invalid). Approximate — a folded sheet won't map perfectly to one grid — but
// good for the near-planar VolCart segment OBJs. depth is set to `default_depth`.
// gw/gh <= 0 picks a resolution from the vertex count. Returns 0 on success.
int  mc_grid_from_mesh(const mc_mesh *m, int gw, int gh, float default_depth,
                       mc_surface *s);

#endif /* MC_SURFACE_H */
