/* mc_viewer — integrated slice viewer for matter-compressor.
 *
 * Milestone 1: open a volume (local .mca / zarr path or s3://, https://),
 * software-render an axis-aligned slice with the existing LOD-matched renderer
 * (mc_render_plane_lod), colormap it to ARGB32, and blit it via an SDL_Renderer
 * streaming texture. A Nuklear panel drives axis / slice index / zoom /
 * window-level / colormap. Mouse drag pans, wheel zooms.
 *
 * This is intentionally a thin client over the public matter_compressor.h API;
 * all the heavy lifting (decode, cache, LOD, render) already lives in the core.
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
#include "nk_sdl3.h"

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

    SDL_Renderer *ren;
    SDL_Texture *tex;
    int          tex_w, tex_h;

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

    /* (re)create the SDL texture to match. */
    if (v->tex && (v->tex_w != iw || v->tex_h != ih)) {
        SDL_DestroyTexture(v->tex); v->tex = NULL;
    }
    if (!v->tex) {
        v->tex = SDL_CreateTexture(v->ren, SDL_PIXELFORMAT_ARGB8888,
                                   SDL_TEXTUREACCESS_STREAMING, iw, ih);
    }
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
    SDL_Renderer *ren = SDL_CreateRenderer(win, NULL);
    if (!win || !ren) {
        fprintf(stderr, "SDL window/renderer: %s\n", SDL_GetError());
        return 1;
    }
    V.ren = ren;

    struct nk_context *nk = nk_sdl3_init(win, ren);
    {
        struct nk_font_atlas *atlas;
        nk_sdl3_font_stash_begin(&atlas);
        nk_sdl3_font_stash_end();
    }

    int running = 1;
    int dragging = 0;
    static const char *cmap_names[] = {
        "gray","viridis","magma","fire","red","green","blue","cyan","magenta"
    };

    while (running) {
        SDL_Event e;
        nk_input_begin(nk);
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT) running = 0;
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
            nk_sdl3_handle_event(&e);
        }
        nk_input_end(nk);

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
        }
        nk_end(nk);

        /* --- (re)render slice if dirty --- */
        if (atomic_exchange(&V.dirty, 0)) {
            render_slice(&V);
            if (V.tex && V.argb) {
                SDL_UpdateTexture(V.tex, NULL, V.argb, V.img_w * 4);
                V.tex_w = V.img_w; V.tex_h = V.img_h;
                /* MC_VIEWER_DUMP=path.ppm: write the first rendered slice and
                 * exit. Used for headless verification (no display needed). */
                const char *dump = SDL_getenv("MC_VIEWER_DUMP");
                if (dump) {
                    /* For deterministic headless verification, re-render once
                     * with a BLOCKING sample source so absent regions are
                     * transcoded synchronously rather than sampling as 0. */
                    V.lods = mc_volume_sample_lods(V.vol, /*blocking=*/1);
                    render_slice(&V);
                    SDL_UpdateTexture(V.tex, NULL, V.argb, V.img_w * 4);
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
        }

        /* --- draw --- */
        SDL_SetRenderDrawColor(ren, 16, 16, 20, 255);
        SDL_RenderClear(ren);
        if (V.tex) {
            int out_w, out_h; SDL_GetRenderOutputSize(ren, &out_w, &out_h);
            /* On screen the whole slice spans (slice voxels * zoom) px,
             * regardless of how many pixels we rendered the texture at. */
            int al2, vw, vh; axis_extent(&V, &al2, &vw, &vh);
            float dw = vw * V.zoom, dh = vh * V.zoom;
            SDL_FRect dst;
            dst.w = dw; dst.h = dh;
            dst.x = (out_w - dw) * 0.5f + V.pan_x;
            dst.y = (out_h - dh) * 0.5f + V.pan_y;
            SDL_SetTextureScaleMode(V.tex, V.zoom >= 1.0f ? SDL_SCALEMODE_NEAREST
                                                          : SDL_SCALEMODE_LINEAR);
            SDL_RenderTexture(ren, V.tex, NULL, &dst);
        }
        nk_sdl3_render(NK_ANTI_ALIASING_ON);
        SDL_RenderPresent(ren);
    }

    nk_sdl3_shutdown();
    if (V.tex) SDL_DestroyTexture(V.tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    mc_volume_free(V.vol);
    free(V.vals); free(V.argb);
    SDL_Quit();
    return 0;
}
