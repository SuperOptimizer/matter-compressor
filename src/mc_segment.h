/* mc_segment.h — classical surface (sheet) detection + topology post-processing
 * for the Vesuvius surface-detection task. No ML: a structure-tensor sheetness
 * detector turns a CT volume into a candidate binary surface mask, and a
 * topology post-processing pipeline (the kind that wins the metric: remove
 * specks, plug 1-voxel holes, close, fill cavities) cleans any binary mask.
 *
 * All volumes are dense u8, z-major: voxel (z,y,x) at v[(z*ny + y)*nx + x].
 * Masks are u8 0/255 (foreground = the detected sheet).
 *
 * Everything operates in memory on a (nz,ny,nx) volume the caller supplies
 * (extract a CT chunk from a .tif / mc_volume, run, write the mask out).
 */
#ifndef MC_SEGMENT_H
#define MC_SEGMENT_H

#include <stdint.h>

typedef struct {
    float sigma_grad;    // gaussian sigma for the gradient (pre-smooth)   (0->1.0)
    float sigma_tensor;  // gaussian sigma for the tensor smoothing (integration scale) (0->2.0)
    float sheetness;     // [0,1] min "planarness" to accept a voxel       (0->0.5)
    int   val_lo;        // intensity gate: ignore voxels with value < val_lo (0->material only via Otsu-ish; here a fixed floor)
} mc_seg_params;

// Structure-tensor sheet detector. For each voxel: smoothed gradient g, tensor
// T = smooth(g g^T), 3x3 eigenvalues l0>=l1>=l2. A planar sheet has one large
// eigenvalue (across the sheet) and two small (in-plane): sheetness =
// (l0 - l1) / (l0 + eps) in [0,1]. Voxels with value >= val_lo and sheetness >=
// params.sheetness become foreground (255). Writes `mask` (nz*ny*nx u8 0/255).
// `params` NULL -> defaults. Returns 0 on success.
int mc_seg_detect(const uint8_t *vol, int nz, int ny, int nx,
                  const mc_seg_params *params, uint8_t *mask);

// ---- topology post-processing (operates on any binary u8 0/255 mask) ----

// Remove connected components (26-connectivity) smaller than min_voxels. In
// place. Returns the number of components kept.
int  mc_seg_remove_small(uint8_t *mask, int nz, int ny, int nx, int min_voxels);

// Plug 1-voxel holes: for each foreground voxel, inspect its 2x2x2 cubes and add
// the voxels needed to make the local neighborhood 6-connectivity watertight
// (256-entry lookup table over the 8-voxel cube). In place; iterate `passes`
// times (1 is usually enough). Returns voxels added.
long mc_seg_plug_holes(uint8_t *mask, int nz, int ny, int nx, int passes);

// Morphological binary closing with a spherical structuring element of the
// given radius (dilate then erode) — closes gaps/cavities up to ~radius. In
// place (uses a scratch copy).
int  mc_seg_close(uint8_t *mask, int nz, int ny, int nx, int radius);

// Fill cavities: any background component not connected to the volume border
// becomes foreground (3D flood-fill of the background from the border; the
// unreached background is interior cavity). In place.
int  mc_seg_fill_cavities(uint8_t *mask, int nz, int ny, int nx);

// Label connected components (26-connectivity) into `labels` (int32, 0 =
// background, 1..n = components). Returns the component count.
int  mc_seg_label(const uint8_t *mask, int nz, int ny, int nx, int32_t *labels);

// Exact Euclidean distance transform (Felzenszwalb-Huttenlocher separable
// parabola method). For each voxel, `out` receives the Euclidean distance (in
// voxel units) to the nearest FOREGROUND voxel (mask != 0). Foreground voxels
// get 0. If the mask is all-background, every cell gets a large finite value.
// `out` is nz*ny*nx floats supplied by the caller. Returns 0 on success.
int  mc_seg_edt(const uint8_t *mask, int nz, int ny, int nx, float *out);

// Signed distance transform: negative inside the foreground region, positive
// outside, |value| = distance to the foreground boundary (the zero level set is
// the sheet surface). out = EDT(background-as-seed) - EDT(foreground-as-seed),
// giving the standard signed field the tracer's volume term pulls toward 0.
int  mc_seg_sdt(const uint8_t *mask, int nz, int ny, int nx, float *out);

#endif /* MC_SEGMENT_H */
