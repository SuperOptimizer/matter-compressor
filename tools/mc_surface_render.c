/* mc_surface_render — render a parametric surface against a volume.
 *
 * Loads a surface (4x-f32 grid TIFF: x,y,z,depth) and a volume (.mca / zarr /
 * s3://), samples the volume over the surface's quad parameterization, and
 * writes a PPM. Demonstrates the surface-I/O + quad-render chain end to end.
 *
 *   mc_surface_render <surface.tif> <volume-url> [cache_dir] [out.ppm]
 *
 * The surface grid is rendered at native density (one output pixel per grid
 * cell); --obj also dumps the surface mesh.
 */
#include "matter_compressor.h"
#include "mc_surface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv){
    if(argc < 3){
        fprintf(stderr,
          "usage: %s <surface.tif> <volume-url> [cache_dir] [out.ppm]\n", argv[0]);
        return 2;
    }
    const char *surf_path = argv[1];
    const char *vol_url   = argv[2];
    const char *cache_dir = (argc>3) ? argv[3] : "./mc_cache";
    const char *out_ppm   = (argc>4) ? argv[4] : "surface.ppm";

    mc_surface s;
    if(mc_surface_load_tiff(surf_path,&s)!=0){
        fprintf(stderr,"failed to load surface %s\n", surf_path); return 1; }
    fprintf(stderr,"surface %dx%d grid, mean depth %.2f\n", s.gw, s.gh, s.mean_depth);

    mc_volume *v = mc_volume_open(vol_url, cache_dir, (size_t)1<<30, 6.0f);
    if(!v){ fprintf(stderr,"mc_volume_open(%s) failed\n", vol_url); mc_surface_free(&s); return 1; }

    // sample source over LOD0 (blocking: ensure data is present for a one-shot
    // render). Render the quad at native grid density (step=1, 1 cell/pixel).
    mc_sample_src src = mc_volume_sample_src(v, 0, /*blocking=*/1);
    mc_quad q = mc_surface_quad(&s);
    int w=s.gw, h=s.gh;

    mc_render_params rp; memset(&rp,0,sizeof rp);
    rp.filter = MC_FILTER_TRILINEAR;
    rp.comp   = MC_COMP_MAX;          // MIP through the surface's local slab
    rp.t0 = -s.mean_depth; rp.t1 = s.mean_depth; rp.dt = 1.0f;   // use the depth band

    uint8_t *vals = malloc((size_t)w*h);
    if(!vals){ mc_volume_free(v); mc_surface_free(&s); return 1; }
    if(mc_render_quad(&src, &q, 0,0, 1.0f, w, h, &rp, vals, 0) != 0){
        fprintf(stderr,"mc_render_quad failed\n"); }

    // colormap -> ARGB -> PPM.
    uint32_t lut[256]; mc_colormap_lut(lut, 0.0f, 1.0f, 0 /*gray*/);
    uint32_t *argb = malloc((size_t)w*h*4);
    mc_colormap_apply(vals, w, h, lut, argb, w);
    uint8_t *rgb = malloc((size_t)w*h*3);
    long nz=0;
    for(int i=0;i<w*h;++i){ uint32_t p=argb[i]; rgb[i*3]=(p>>16)&255; rgb[i*3+1]=(p>>8)&255; rgb[i*3+2]=p&255; if(vals[i]) nz++; }
    if(mc_ppm_write(out_ppm, w, h, 3, rgb)==0)
        fprintf(stderr,"wrote %s (%dx%d, %ld/%d nonzero)\n", out_ppm, w, h, nz, w*h);

    free(vals); free(argb); free(rgb);
    mc_volume_free(v); mc_surface_free(&s);
    return 0;
}
