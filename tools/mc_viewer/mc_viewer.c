/* mc_viewer — integrated slice viewer for matter-compressor.
 *
 * Open a volume (local .mca / zarr path or s3://, https://), software-render an
 * axis-aligned slice with the existing LOD-matched renderer (mc_render_plane_lod),
 * colormap it to ARGB32, then hand it to the SDL_GPU frontend (mc_gpu) which
 * uploads it as a texture and draws it on a fullscreen quad (zoom/pan in the
 * vertex shader). The Nuklear UI renders in the same SDL_GPU frame. Mouse drag
 * pans, wheel zooms.
 *
 * Thin client over the public matter_compressor.h API; all the heavy lifting
 * (decode, cache, LOD, render) already lives in the core. The GPU path lives in
 * mc_gpu.h (slice pipeline + a from-scratch Nuklear SDL_GPU backend).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdatomic.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#include "nuklear.h"
#include "mc_gpu.h"
#include "mc_gpu_vol.h"

#include "matter_compressor.h"

enum { AXIS_Z = 0, AXIS_Y = 1, AXIS_X = 2 };

typedef struct {
    mc_volume       *vol;
    mc_sample_lods   lods;
    int              nx, ny, nz;     /* LOD0 voxel dims */
    int              nlods;

    /* view state */
    int   axis;                      /* AXIS_Z/Y/X */
    int   slice;                     /* index along axis, in LOD0 voxels */
    float zoom;                      /* screen px per LOD0 voxel (>1 = magnify) */
    float pan_x, pan_y;              /* slice px offset of view center */
    int   win_low, win_high;         /* window/level: display range [0,255] */
    int   cmap;                      /* colormap id */
    int   nthreads;

    /* slice render target */
    int       img_w, img_h;          /* current slice image size in px */
    uint8_t  *vals;                  /* img_w*img_h u8 (renderer output) */
    uint32_t *argb;                  /* img_w*img_h ARGB32 (colormapped) */
    uint32_t  lut[256];

    mc_gpu      *gpu;                 /* SDL_GPU frontend (slice + Nuklear) */
    mc_gpu_vol  *gvol;                /* GPU c3g decode+sample path (c3g archives) */
    int          gpu_mode;            /* 0 = CPU render+blit, 1 = GPU decode+sample */
    int          gpu_avail;           /* 1 if the volume archive is c3g (gvol usable) */

    atomic_int   dirty;              /* set by ready_cb -> repaint slice */
} viewer;

/* fired from a background transcode worker — just flag a repaint. */
static void on_region_ready(void *ud) {
    viewer *v = (viewer *)ud;
    atomic_store(&v->dirty, 1);
}

static void axis_extent(const viewer *v, int *along, int *w, int *h) {
    /* image x = u axis, image y = v axis; `along` = the slice axis extent. */
    switch (v->axis) {
    case AXIS_Z: *along = v->nz; *w = v->nx; *h = v->ny; break;  /* XY plane */
    case AXIS_Y: *along = v->ny; *w = v->nx; *h = v->nz; break;  /* XZ plane */
    default:     *along = v->nx; *w = v->ny; *h = v->nz; break;  /* YZ plane */
    }
}

/* Build the mc_plane for the current axis/slice. Coordinates are (z,y,x). */
static void build_plane(const viewer *v, mc_plane *pl) {
    memset(pl, 0, sizeof(*pl));
    switch (v->axis) {
    case AXIS_Z: /* normal +z, u=+x, v=+y */
        pl->origin[0] = (float)v->slice; pl->origin[1] = 0; pl->origin[2] = 0;
        pl->normal[0] = 1;
        pl->u[2] = 1;            /* image x -> +x */
        pl->v[1] = 1;            /* image y -> +y */
        break;
    case AXIS_Y: /* normal +y, u=+x, v=+z */
        pl->origin[1] = (float)v->slice;
        pl->normal[1] = 1;
        pl->u[2] = 1;
        pl->v[0] = 1;
        break;
    default:     /* normal +x, u=+y, v=+z */
        pl->origin[2] = (float)v->slice;
        pl->normal[2] = 1;
        pl->u[1] = 1;
        pl->v[0] = 1;
        break;
    }
    /* origin must be the image CENTER for mc_plane_gen's (j-w/2),(i-h/2). */
    int along, w, h; axis_extent(v, &along, &w, &h);
    /* center the plane on the slice's middle in the in-plane axes */
    for (int k = 0; k < 3; k++)
        pl->origin[k] += pl->u[k] * (w * 0.5f) + pl->v[k] * (h * 0.5f);
}

/* Re-render the current slice into v->vals/argb and (re)allocate as needed. */
static void render_slice(viewer *v) {
    int along, sw, sh;
    axis_extent(v, &along, &sw, &sh);
    if (v->slice < 0) v->slice = 0;
    if (v->slice >= along) v->slice = along - 1;

    /* The rendered image always covers the WHOLE slice extent (sw*sh voxels).
     * `scale` (vox/px) sets how many voxels each rendered pixel spans, so the
     * pixel count is extent/scale. When zoomed in (zoom>1) we render at native
     * resolution (scale 1, iw=sw) and let the blit magnify; when zoomed out we
     * render fewer pixels (scale>1) so we never rasterize more than the screen
     * shows. This keeps the rendered image == the full slice, never a crop. */
    float scale = (v->zoom > 1.0f) ? 1.0f : (1.0f / (v->zoom > 0 ? v->zoom : 1.0f));
    int iw = (int)ceilf(sw / scale);
    int ih = (int)ceilf(sh / scale);
    if (iw < 1) iw = 1;
    if (ih < 1) ih = 1;
    if (iw != v->img_w || ih != v->img_h || !v->vals) {
        free(v->vals); free(v->argb);
        v->img_w = iw; v->img_h = ih;
        v->vals = (uint8_t *)malloc((size_t)iw * ih);
        v->argb = (uint32_t *)malloc((size_t)iw * ih * 4);
    }
    if (!v->vals || !v->argb) return;

    mc_plane pl; build_plane(v, &pl);

    mc_render_params rp;
    memset(&rp, 0, sizeof(rp));
    rp.filter = MC_FILTER_TRILINEAR;
    rp.comp   = MC_COMP_NONE;        /* flat slice, no normal compositing */

    mc_render_plane_lod(&v->lods, &pl, iw, ih, scale, &rp, v->vals, v->nthreads);

    mc_colormap_lut(v->lut, v->win_low / 255.0f, v->win_high / 255.0f, v->cmap);
    mc_colormap_apply(v->vals, iw, ih, v->lut, v->argb, iw);

    /* Hand the colormapped slice to the GPU frontend (it owns the texture and
     * recreates it on size change). */
    if (v->gpu) mc_gpu_upload_slice(v->gpu, v->argb, iw, ih);
}

/* GPU decode path: gather the visible slab's COMPRESSED c3g block blobs from the
 * volume's local archive and upload them to the GPU vol cache. The slab is the
 * single block-row (16 voxels thick) containing the current slice along the
 * axis, over a bounded in-plane block window anchored at the volume origin
 * (LOD0). Each chunk is ensured present via a blocking get_block (the transcode
 * pipeline writes c3g into the archive), then mc_archive_block_blob reads the
 * compressed bytes with NO decode. Returns 1 if a slab was uploaded.
 *
 * Window cap keeps VRAM + the readback bounded; MCV-side cap is MCV_MAX_BLOCKS.
 * Returns the slab geometry in *gz/*gy/*gx (block dims) + *ez/*ey/*ex (voxels)
 * and the within-slab slice index in *sidx. */
#define MC_VOL_WIN 16   /* in-plane block window (16 blocks = 256 voxels) */
static int gather_slab(viewer *v, int *out_gz,int *out_gy,int *out_gx,
                       int *out_ez,int *out_ey,int *out_ex,int *out_sidx,
                       const uint8_t **blobs, uint32_t *lens, uint8_t *scratch4096){
    mc_archive *arc = mc_volume_archive(v->vol);
    if(!arc) return 0;
    // LOD0 block grid extent.
    int bgz,bgy,bgx; mc_volume_block_grid(v->vol, 0, &bgz,&bgy,&bgx);
    // the block-row containing the slice along the axis; full in-plane (capped).
    int srow = v->slice >> 4;                 // block index along the axis
    int gz,gy,gx;                             // slab block dims (z,y,x)
    if(v->axis==AXIS_Z){ gz=1; gy=bgy<MC_VOL_WIN?bgy:MC_VOL_WIN; gx=bgx<MC_VOL_WIN?bgx:MC_VOL_WIN; }
    else if(v->axis==AXIS_Y){ gy=1; gz=bgz<MC_VOL_WIN?bgz:MC_VOL_WIN; gx=bgx<MC_VOL_WIN?bgx:MC_VOL_WIN; }
    else { gx=1; gz=bgz<MC_VOL_WIN?bgz:MC_VOL_WIN; gy=bgy<MC_VOL_WIN?bgy:MC_VOL_WIN; }
    int nb = gz*gy*gx;
    if(nb<=0) return 0;

    // map slab block (sz,sy,sx) -> global LOD0 block coords. The slice-axis
    // block is fixed at srow; the in-plane axes start at 0.
    for(int sz=0; sz<gz; ++sz) for(int sy=0; sy<gy; ++sy) for(int sx=0; sx<gx; ++sx){
        int Bz, By, Bx;
        if(v->axis==AXIS_Z){ Bz=srow; By=sy; Bx=sx; }
        else if(v->axis==AXIS_Y){ By=srow; Bz=sz; Bx=sx; }
        else { Bx=srow; Bz=sz; By=sy; }
        int slabi = (sz*gy + sy)*gx + sx;
        blobs[slabi]=NULL; lens[slabi]=0;
        if(Bz>=bgz||By>=bgy||Bx>=bgx) continue;
        // chunk = 256^3 = 16^3 blocks; ensure present (blocking transcode), then
        // read the compressed block blob.
        int cz=Bz>>4, cy=By>>4, cx=Bx>>4;
        mc_volume_get_block(v->vol, 0, Bz,By,Bx, scratch4096);   // ensure chunk transcoded
        uint64_t co = mc_archive_chunk_offset(arc, 0, cz,cy,cx);
        if(!co) continue;
        const uint8_t *p=NULL; uint32_t l=0;
        if(mc_archive_block_blob(arc, co, Bz&15,By&15,Bx&15, &p, &l)){ blobs[slabi]=p; lens[slabi]=l; }
    }
    *out_gz=gz; *out_gy=gy; *out_gx=gx;
    *out_ez=gz*16; *out_ey=gy*16; *out_ex=gx*16;
    *out_sidx = v->slice & 15;          // slice index within the slab's 16-thick row
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s <url-or-path> [cache_dir]\n"
            "  url-or-path : .mca / zarr directory / s3:// / https://\n"
            "  cache_dir   : local cache for streamed volumes (default ./mc_cache)\n",
            argv[0]);
        return 2;
    }
    const char *url = argv[1];
    const char *cache_dir = (argc > 2) ? argv[2] : "./mc_cache";

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }

    viewer V;
    memset(&V, 0, sizeof(V));
    V.zoom = 1.0f;
    V.win_low = 0; V.win_high = 255;
    V.cmap = 0;
    V.nthreads = 0;   /* 0 -> renderer picks */
    atomic_store(&V.dirty, 1);

    /* mc_volume_open writes <cache_dir>/<name>.mca but does not create the
     * directory itself — make sure it exists first. */
    if (!SDL_CreateDirectory(cache_dir))
        fprintf(stderr, "warning: SDL_CreateDirectory(%s): %s\n", cache_dir, SDL_GetError());

    /* ~1 GB resident cache. quality 6.0 = the codec's default re-encode q for
     * a freshly-streamed zarr (an existing .mca decodes at its own build q). */
    V.vol = mc_volume_open(url, cache_dir, (size_t)1 << 30, 6.0f);
    if (!V.vol) {
        fprintf(stderr, "mc_volume_open(%s) failed\n", url);
        SDL_Quit();
        return 1;
    }
    V.nlods = mc_volume_nlods(V.vol);
    mc_volume_shape(V.vol, 0, &V.nz, &V.ny, &V.nx);
    V.lods = mc_volume_sample_lods(V.vol, /*blocking=*/0);
    mc_volume_set_ready_cb(V.vol, on_region_ready, &V);

    V.axis = AXIS_Z;
    V.slice = V.nz / 2;

    SDL_Window *win = SDL_CreateWindow("mc_viewer", 1280, 800, SDL_WINDOW_RESIZABLE);
    if (!win) {
        fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
        return 1;
    }
    V.gpu = mc_gpu_create(win);
    if (!V.gpu) {
        fprintf(stderr, "mc_gpu_create (SDL_GPU): %s\n", SDL_GetError());
        return 1;
    }
    struct nk_context *nk = mc_gpu_nk(V.gpu);

    // GPU c3g decode+sample path is available only when the volume's local
    // archive uses the c3g block codec (MC_BLOCK_CODEC=c3g). Then 'G' toggles
    // between the CPU render+blit and reading compressed blocks -> GPU decode.
    {
        mc_archive *arc = mc_volume_archive(V.vol);
        if (arc && mc_archive_block_codec(arc) == MC_BLOCKCODEC_C3G) {
            V.gvol = mc_gpu_vol_create(mc_gpu_device(V.gpu),
                                       SDL_GetGPUSwapchainTextureFormat(mc_gpu_device(V.gpu), win),
                                       6.0f);
            V.gpu_avail = (V.gvol != NULL);
        }
    }
    // MC_VIEWER_GPU=1 starts in GPU decode mode (headless testing / default-on).
    if (V.gpu_avail && SDL_getenv("MC_VIEWER_GPU")) V.gpu_mode = 1;

    int running = 1;
    int dragging = 0;
    static const char *cmap_names[] = {
        "gray","viridis","magma","fire","red","green","blue","cyan","magenta"
    };

    /* MC_VIEWER_FRAMES=N: render N frames then exit (headless GPU smoke test). */
    const char *frames_env = SDL_getenv("MC_VIEWER_FRAMES");
    long max_frames = frames_env ? SDL_atoi(frames_env) : 0;
    long frame = 0;

    while (running) {
        SDL_Event e;
        mc_gpu_nk_input_begin(V.gpu);
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT) running = 0;
            if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_G && V.gpu_avail) {
                V.gpu_mode = !V.gpu_mode;
                atomic_store(&V.dirty, 1);
                fprintf(stderr, "GPU decode mode: %s\n", V.gpu_mode ? "ON" : "OFF");
            }
            if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN &&
                e.button.button == SDL_BUTTON_RIGHT) dragging = 1;
            if (e.type == SDL_EVENT_MOUSE_BUTTON_UP &&
                e.button.button == SDL_BUTTON_RIGHT) dragging = 0;
            if (e.type == SDL_EVENT_MOUSE_MOTION && dragging) {
                V.pan_x += e.motion.xrel;
                V.pan_y += e.motion.yrel;
            }
            if (e.type == SDL_EVENT_MOUSE_WHEEL) {
                float f = (e.wheel.y > 0) ? 1.1f : (e.wheel.y < 0 ? 1.0f/1.1f : 1.0f);
                V.zoom *= f;
                if (V.zoom < 0.01f) V.zoom = 0.01f;
                if (V.zoom > 64.0f) V.zoom = 64.0f;
                atomic_store(&V.dirty, 1);
            }
            mc_gpu_nk_handle_event(V.gpu, &e);
        }
        mc_gpu_nk_input_end(V.gpu);

        /* --- UI panel --- */
        int along, sw, sh; axis_extent(&V, &along, &sw, &sh);
        if (nk_begin(nk, "controls", nk_rect(10, 10, 280, 360),
                     NK_WINDOW_BORDER|NK_WINDOW_MOVABLE|NK_WINDOW_TITLE|
                     NK_WINDOW_MINIMIZABLE)) {
            nk_layout_row_dynamic(nk, 24, 3);
            int prev_axis = V.axis;
            if (nk_option_label(nk, "Z", V.axis == AXIS_Z)) V.axis = AXIS_Z;
            if (nk_option_label(nk, "Y", V.axis == AXIS_Y)) V.axis = AXIS_Y;
            if (nk_option_label(nk, "X", V.axis == AXIS_X)) V.axis = AXIS_X;
            if (V.axis != prev_axis) {
                axis_extent(&V, &along, &sw, &sh);
                V.slice = along / 2;
                atomic_store(&V.dirty, 1);
            }

            nk_layout_row_dynamic(nk, 24, 1);
            int s = V.slice;
            nk_property_int(nk, "slice", 0, &s, along - 1, 1, 1);
            if (s != V.slice) { V.slice = s; atomic_store(&V.dirty, 1); }

            nk_layout_row_dynamic(nk, 24, 1);
            float z = V.zoom;
            nk_property_float(nk, "zoom", 0.01f, &z, 64.0f, 0.05f, 0.01f);
            if (z != V.zoom) { V.zoom = z; atomic_store(&V.dirty, 1); }

            int wl = V.win_low, wh = V.win_high;
            nk_property_int(nk, "win low",  0, &wl, 255, 1, 1);
            nk_property_int(nk, "win high", 0, &wh, 255, 1, 1);
            if (wl != V.win_low || wh != V.win_high) {
                V.win_low = wl; V.win_high = wh; atomic_store(&V.dirty, 1);
            }

            nk_layout_row_dynamic(nk, 24, 1);
            int newc = nk_combo(nk, cmap_names, 9, V.cmap, 24, nk_vec2(200, 200));
            if (newc != V.cmap) { V.cmap = newc; atomic_store(&V.dirty, 1); }

            nk_layout_row_dynamic(nk, 20, 1);
            char info[128];
            snprintf(info, sizeof(info), "vol %dx%dx%d  lods=%d", V.nx, V.ny, V.nz, V.nlods);
            nk_label(nk, info, NK_TEXT_LEFT);
            snprintf(info, sizeof(info), "slice img %dx%d", V.img_w, V.img_h);
            nk_label(nk, info, NK_TEXT_LEFT);
            if (V.gpu_avail) {
                snprintf(info, sizeof(info), "decode: %s  ('G' toggles)",
                         V.gpu_mode ? "GPU (c3g)" : "CPU");
                nk_label(nk, info, NK_TEXT_LEFT);
            } else {
                nk_label(nk, "decode: CPU (c3g archive: MC_BLOCK_CODEC=c3g)", NK_TEXT_LEFT);
            }
        }
        nk_end(nk);

        /* --- (re)render slice if dirty (uploads to the GPU texture) --- */
        if (atomic_exchange(&V.dirty, 0)) {
            render_slice(&V);

            /* MC_VIEWER_DUMP=path.ppm: re-render once with a BLOCKING sample
             * source (so absent regions transcode synchronously instead of
             * sampling 0), write the slice as PPM, and exit. Headless
             * verification path — no display / swapchain needed. */
            const char *dump = SDL_getenv("MC_VIEWER_DUMP");
            if (dump && V.vals && V.argb) {
                V.lods = mc_volume_sample_lods(V.vol, /*blocking=*/1);
                render_slice(&V);
                long nz = 0, mn = 256, mx = -1;
                for (int i = 0; i < V.img_w * V.img_h; i++) {
                    int g = V.vals[i];
                    if (g) nz++;
                    if (g < mn) mn = g;
                    if (g > mx) mx = g;
                }
                FILE *f = fopen(dump, "wb");
                if (f) {
                    fprintf(f, "P6\n%d %d\n255\n", V.img_w, V.img_h);
                    for (int i = 0; i < V.img_w * V.img_h; i++) {
                        uint32_t p = V.argb[i];
                        unsigned char rgb[3] = { (p>>16)&255, (p>>8)&255, p&255 };
                        fwrite(rgb, 1, 3, f);
                    }
                    fclose(f);
                }
                fprintf(stderr, "dump %s: %dx%d vals nonzero=%ld min=%ld max=%ld\n",
                        dump, V.img_w, V.img_h, nz, mn, mx);
                running = 0;
            }
        }

        /* --- draw the frame --- */
        mc_gpu_set_nearest(V.gpu, V.zoom >= 1.0f);
        if (V.gpu_mode && V.gvol) {
            /* GPU decode+sample path: gather the visible slab's COMPRESSED c3g
             * blocks, upload, and let the GPU decode + sample the slice, then
             * draw Nuklear over it in a second (LOAD) pass. */
            static const uint8_t *blobs[16*16*16];
            static uint32_t       lens[16*16*16];
            static uint8_t        scratch[4096];
            int gz,gy,gx,ez,ey,ex,sidx;
            int ok = gather_slab(&V,&gz,&gy,&gx,&ez,&ey,&ex,&sidx,blobs,lens,scratch);
            if (ok) mc_gpu_vol_upload_slab(V.gvol,gz,gy,gx,ez,ey,ex,blobs,lens);
            mc_gpu_vol_set_lut(V.gvol, V.lut);

            SDL_GPUCommandBuffer *cmd; SDL_GPUTexture *swap; Uint32 ow, oh;
            if (ok && mc_gpu_frame_begin(V.gpu, &cmd, &swap, &ow, &oh)) {
                int al2, vw, vh; axis_extent(&V, &al2, &vw, &vh);
                mc_gpu_vol_render(V.gvol, cmd, swap, V.axis, sidx,
                                  vw * V.zoom, vh * V.zoom, V.pan_x, V.pan_y,
                                  (int)ow, (int)oh);
                mc_gpu_frame_end_nuklear(V.gpu, cmd, swap, ow, oh);
            }
        } else {
            /* CPU render + GPU blit path (default). On screen the whole slice
             * spans (slice voxels * zoom) px, independent of the texture res. */
            int al2, vw, vh; axis_extent(&V, &al2, &vw, &vh);
            mc_gpu_render(V.gpu, vw * V.zoom, vh * V.zoom, V.pan_x, V.pan_y,
                          /*gain=*/1.0f, /*bias=*/0.0f);
        }

        if (max_frames && ++frame >= max_frames) {
            fprintf(stderr, "rendered %ld frames -> exit\n", frame);
            running = 0;
        }
    }

    if (V.gvol) mc_gpu_vol_destroy(V.gvol);
    mc_gpu_destroy(V.gpu);
    SDL_DestroyWindow(win);
    mc_volume_free(V.vol);
    free(V.vals); free(V.argb);
    SDL_Quit();
    return 0;
}
