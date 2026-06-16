/* mc_sview — pure-software multi-panel volume viewer.
 *
 * A from-scratch CPU renderer (NO SDL_GPU, no shaders): SDL3 only opens the
 * window and pushes ONE CPU-rendered ARGB framebuffer to the screen via an
 * SDL_Renderer streaming texture. All pixels come from the existing software
 * renderers (mc_render_plane_lod / mc_render_quad_lod -> u8 -> mc_colormap).
 *
 * Layout (VC3D-style 2x2 of independent panels):
 *     +-----------+-----------+
 *     | surface   |  XY (Z)   |
 *     +-----------+-----------+
 *     | XZ (Y)    |  YZ (X)   |
 *     +-----------+-----------+
 * The three orthogonal panels intersect at a shared focus voxel (fz,fy,fx) and
 * draw a crosshair there. Each panel is INDEPENDENT: its own zoom (vox/px) and
 * in-plane pan. Any panel can be MAXIMIZED to fill the whole window ('1'..'4')
 * and restored to the grid ('0'). The surface panel needs a --surface .grid.
 *
 *   mc_sview <url-or-path> [cache_dir] [--surface a.grid] [--focus z y x]
 *
 * Mouse: left-drag pans the panel under the cursor (ortho panels also slide the
 * shared focus along their two in-plane axes); wheel zooms it. Keys: 1-4
 * maximize a panel, 0 restore grid, arrows + PgUp/PgDn move the focus, [/] adjust
 * window level, q/Esc quit.
 */
#include <SDL3/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdatomic.h>
#include "matter_compressor.h"
#include "mc_surface.h"

enum { PANEL_SURF=0, PANEL_XY=1, PANEL_XZ=2, PANEL_YZ=3 };

typedef struct {
    float scale;        /* vox/px (ortho panels) or grid-cells/px (surface) */
    float pan_u, pan_v; /* in-plane pan offset (voxels), ortho panels */
} panel_view;

typedef struct {
    mc_volume     *vol;
    mc_sample_lods lods;
    int            nx, ny, nz, nlods;

    int            fz, fy, fx;        /* shared focus voxel */
    panel_view     pv[4];
    int            maxed;             /* -1 = 2x2 grid, else PANEL_* fills window */
    int            win_low, win_high, cmap, nthreads;
    int            want_autowin;      /* 1 -> next XY render sets window from data */

    mc_surface     surf;
    int            surf_loaded;
    float          surf_x0, surf_y0;

    /* window framebuffer */
    int            win_w, win_h;
    uint32_t      *fb;                /* win_w*win_h ARGB */
    uint8_t       *scratch;           /* panel u8 render scratch */
    size_t         scratch_cap;
    uint32_t       lut[256];

    atomic_int     dirty;
} sview;

static void on_ready(void *ud){ atomic_store(&((sview*)ud)->dirty, 1); }

/* the screen rect (x,y,w,h) of a panel given the layout. */
static void panel_rect(const sview *s, int panel, int *x, int *y, int *w, int *h){
    if (s->maxed >= 0){ *x=0; *y=0; *w=s->win_w; *h=s->win_h; return; }
    int hw = s->win_w/2, hh = s->win_h/2;
    switch (panel){
        case PANEL_SURF: *x=0;  *y=0;  *w=hw;            *h=hh;            break;
        case PANEL_XY:   *x=hw; *y=0;  *w=s->win_w-hw;   *h=hh;            break;
        case PANEL_XZ:   *x=0;  *y=hh; *w=hw;            *h=s->win_h-hh;   break;
        default:         *x=hw; *y=hh; *w=s->win_w-hw;   *h=s->win_h-hh;   break;
    }
}

/* render one orthogonal panel into the framebuffer rect (px,py,pw,ph). */
static void draw_ortho(sview *s, int panel, int px, int py, int pw, int ph){
    panel_view *V = &s->pv[panel];
    float scale = V->scale>0 ? V->scale : 1.0f;
    int iw = pw, ih = ph;
    size_t need = (size_t)iw*ih;
    if (need > s->scratch_cap){ free(s->scratch); s->scratch = malloc(need); s->scratch_cap = need; }
    if (!s->scratch) return;

    /* plane basis + focus-on-normal; centered on (focus + pan) in-plane. */
    mc_plane pl; memset(&pl,0,sizeof pl);
    double cu, cv, cn;
    switch (panel){
        case PANEL_XY: pl.normal[0]=1; pl.u[2]=1; pl.v[1]=1; cn=s->fz; cu=s->fx; cv=s->fy; break; /* u=x v=y */
        case PANEL_XZ: pl.normal[1]=1; pl.u[2]=1; pl.v[0]=1; cn=s->fy; cu=s->fx; cv=s->fz; break; /* u=x v=z */
        default:       pl.normal[2]=1; pl.u[1]=1; pl.v[0]=1; cn=s->fx; cu=s->fy; cv=s->fz; break; /* u=y v=z */
    }
    cu += V->pan_u; cv += V->pan_v;
    for (int k=0;k<3;k++) pl.origin[k] = pl.normal[k]*cn + pl.u[k]*cu + pl.v[k]*cv;

    mc_render_params rp; memset(&rp,0,sizeof rp);
    rp.filter = MC_FILTER_TRILINEAR; rp.comp = MC_COMP_NONE;
    mc_render_plane_lod(&s->lods, &pl, iw, ih, scale, &rp, s->scratch, s->nthreads);

    /* auto-window: on request, set the display range from this panel's actual
     * value distribution (2nd..98th percentile of nonzero voxels) so bright/
     * narrow-band data shows contrast instead of a flat white. */
    if (s->want_autowin && panel==PANEL_XY){
        long hist[256]={0}, tot=0;
        for (size_t i=0;i<(size_t)iw*ih;++i){ int g=s->scratch[i]; if(g){ hist[g]++; tot++; } }
        if (tot>0){
            long lo_t=tot*2/100, hi_t=tot*98/100, acc=0; int lo=1, hi=255;
            for (int g=1; g<256; ++g){ acc+=hist[g]; if(acc>=lo_t){ lo=g; break; } }
            acc=0; for (int g=1; g<256; ++g){ acc+=hist[g]; if(acc>=hi_t){ hi=g; break; } }
            if (hi<=lo) hi=lo+1;
            s->win_low=lo; s->win_high=hi; s->want_autowin=0;
            fprintf(stderr,"auto-window -> [%d,%d]\n", lo, hi);
        }
    }
    mc_colormap_lut(s->lut, s->win_low/255.0f, s->win_high/255.0f, s->cmap);
    for (int y=0;y<ih;++y)
        mc_colormap_apply(s->scratch+(size_t)y*iw, iw, 1, s->lut,
                          s->fb + (size_t)(py+y)*s->win_w + px, iw);
    /* crosshair at the focus = panel center (pan shifts the image, focus stays
     * centered because origin is centered on focus+pan). Cross marks panel center. */
    int cx = px + pw/2, cy = py + ph/2;
    uint32_t col = 0xff20ff20;
    for (int x=px; x<px+pw; ++x) s->fb[(size_t)cy*s->win_w + x] = col;
    for (int y=py; y<py+ph; ++y) s->fb[(size_t)y*s->win_w + cx] = col;
}

/* render the surface panel (the loaded .grid) into the rect. */
static void draw_surface(sview *s, int px, int py, int pw, int ph){
    if (!s->surf_loaded) return;
    panel_view *V = &s->pv[PANEL_SURF];
    mc_quad q = mc_surface_quad(&s->surf);
    float step = V->scale>0 ? V->scale : 1.0f;     /* grid cells per px */
    int iw = pw, ih = ph;
    size_t need = (size_t)iw*ih;
    if (need > s->scratch_cap){ free(s->scratch); s->scratch = malloc(need); s->scratch_cap = need; }
    if (!s->scratch) return;
    mc_render_params rp; memset(&rp,0,sizeof rp);
    rp.filter = MC_FILTER_TRILINEAR;
    float d = s->surf.mean_depth>0 ? s->surf.mean_depth : 1.0f;
    rp.comp = MC_COMP_MAX; rp.t0 = -d; rp.t1 = d; rp.dt = 1.0f;
    mc_render_quad_lod(&s->lods, &q, s->surf_x0, s->surf_y0, step, iw, ih, &rp, s->scratch, s->nthreads);
    mc_colormap_lut(s->lut, s->win_low/255.0f, s->win_high/255.0f, s->cmap);
    for (int y=0;y<ih;++y)
        mc_colormap_apply(s->scratch+(size_t)y*iw, iw, 1, s->lut,
                          s->fb + (size_t)(py+y)*s->win_w + px, iw);
}

/* a thin separator + a tiny label tag per panel (drawn as colored corner bars). */
static void draw_borders(sview *s){
    if (s->maxed >= 0) return;
    int hw=s->win_w/2, hh=s->win_h/2;
    uint32_t b=0xff404040;
    for (int y=0;y<s->win_h;++y) s->fb[(size_t)y*s->win_w + hw] = b;
    for (int x=0;x<s->win_w;++x) s->fb[(size_t)hh*s->win_w + x] = b;
}

static void render_all(sview *s){
    if (!s->fb) return;
    memset(s->fb, 0, (size_t)s->win_w*s->win_h*4);
    if (s->maxed >= 0){
        int x,y,w,h; panel_rect(s, s->maxed, &x,&y,&w,&h);
        if (s->maxed==PANEL_SURF) draw_surface(s,x,y,w,h);
        else draw_ortho(s, s->maxed, x,y,w,h);
    } else {
        for (int p=0;p<4;++p){
            int x,y,w,h; panel_rect(s,p,&x,&y,&w,&h);
            if (w<8||h<8) continue;
            if (p==PANEL_SURF) draw_surface(s,x,y,w,h);
            else draw_ortho(s,p,x,y,w,h);
        }
        draw_borders(s);
    }
}

/* which panel is under a window pixel? */
static int panel_at(const sview *s, int mx, int my){
    if (s->maxed >= 0) return s->maxed;
    int hw=s->win_w/2, hh=s->win_h/2;
    if (mx<hw && my<hh) return PANEL_SURF;
    if (mx>=hw && my<hh) return PANEL_XY;
    if (mx<hw && my>=hh) return PANEL_XZ;
    return PANEL_YZ;
}

int main(int argc, char **argv){
    const char *url=NULL,*cache="./mc_cache",*surf_path=NULL,*dump=NULL;
    int fz=-1,fy=-1,fx=-1, dump_w=1200, dump_h=900;
    for (int i=1;i<argc;++i){
        if (!strcmp(argv[i],"--surface")&&i+1<argc) surf_path=argv[++i];
        else if (!strcmp(argv[i],"--focus")&&i+3<argc){ fz=atoi(argv[++i]); fy=atoi(argv[++i]); fx=atoi(argv[++i]); }
        else if (!strcmp(argv[i],"--dump")&&i+1<argc) dump=argv[++i];   /* headless PPM */
        else if (!url) url=argv[i];
        else cache=argv[i];
    }
    if (!url){ fprintf(stderr,"usage: %s <url-or-path> [cache] [--surface a.grid] [--focus z y x] [--dump out.ppm]\n",argv[0]); return 2; }
    setvbuf(stderr, NULL, _IONBF, 0);

    /* Headless dump mode: no SDL window; render once with a BLOCKING sample
     * source and write the framebuffer as PPM (lets the render be verified
     * without a display / streaming races). */
    if (!dump) {
        /* GNOME/Wayland (mutter) provides NO server-side decorations, so without
         * libdecor the window is borderless (no title bar / min-max-close). Force
         * the libdecor client-side path (the gtk plugin draws the buttons). Set
         * with OVERRIDE priority before SDL_Init so SDL honors it. */
        SDL_SetHintWithPriority(SDL_HINT_VIDEO_WAYLAND_ALLOW_LIBDECOR, "1", SDL_HINT_OVERRIDE);
        SDL_SetHintWithPriority(SDL_HINT_VIDEO_WAYLAND_PREFER_LIBDECOR, "1", SDL_HINT_OVERRIDE);
        if (!SDL_Init(SDL_INIT_VIDEO)){ fprintf(stderr,"SDL_Init: %s\n",SDL_GetError()); return 1; }
        fprintf(stderr,"SDL video driver: %s\n", SDL_GetCurrentVideoDriver());
    }
    SDL_CreateDirectory(cache);

    sview S; memset(&S,0,sizeof S);
    S.win_low=0; S.win_high=255; S.cmap=0; S.nthreads=0; S.maxed=-1; S.want_autowin=1;
    /* ortho panels default to 1 vox/px (native LOD0 detail at the focus); wheel
     * zooms out (coarser LOD, more context) or in. */
    for (int p=0;p<4;++p){ S.pv[p].scale=1.0f; S.pv[p].pan_u=S.pv[p].pan_v=0; }
    S.pv[PANEL_SURF].scale=1.0f;

    if (surf_path){
        if (mc_surface_load_tiff(surf_path,&S.surf)==0){ S.surf_loaded=1;
            fprintf(stderr,"surface %dx%d (mean depth %.2f)\n",S.surf.gw,S.surf.gh,S.surf.mean_depth);
        } else fprintf(stderr,"warning: failed to load surface %s\n",surf_path);
    }

    S.vol = mc_volume_open(url, cache, (size_t)1<<30, 6.0f);
    if (!S.vol){ fprintf(stderr,"mc_volume_open(%s) failed\n",url); return 1; }
    S.nlods = mc_volume_nlods(S.vol);
    mc_volume_shape(S.vol,0,&S.nz,&S.ny,&S.nx);
    S.lods = mc_volume_sample_lods(S.vol, dump?1:0);  /* dump: blocking; live: stream+repaint */
    mc_volume_set_ready_cb(S.vol, on_ready, &S);
    S.fz = (fz>=0)?fz:S.nz/2; S.fy=(fy>=0)?fy:S.ny/2; S.fx=(fx>=0)?fx:S.nx/2;
    if(S.fz<0)S.fz=0; if(S.fz>=S.nz)S.fz=S.nz-1;
    if(S.fy<0)S.fy=0; if(S.fy>=S.ny)S.fy=S.ny-1;
    if(S.fx<0)S.fx=0; if(S.fx>=S.nx)S.fx=S.nx-1;
    mc_volume_request_region(S.vol,0,S.fz/256,S.fy/256,S.fx/256);
    fprintf(stderr,"volume %dx%dx%d (z,y,x), %d LODs; focus z%d y%d x%d\n",
            S.nz,S.ny,S.nx,S.nlods,S.fz,S.fy,S.fx);

    if (dump){
        S.win_w=dump_w; S.win_h=dump_h;
        S.fb=malloc((size_t)S.win_w*S.win_h*4);
        mc_volume_freeze(S.vol);     /* lock-free, race-free reads during render */
        render_all(&S);
        mc_volume_thaw(S.vol);
        FILE *f=fopen(dump,"wb");
        if(f){ fprintf(f,"P6\n%d %d\n255\n",S.win_w,S.win_h);
            for(size_t i=0;i<(size_t)S.win_w*S.win_h;++i){ uint32_t p=S.fb[i];
                unsigned char rgb[3]={(p>>16)&255,(p>>8)&255,p&255}; fwrite(rgb,1,3,f);} fclose(f); }
        /* per-panel pixel stats so the render can be verified headless. */
        for(int pp=0;pp<4;++pp){ int x,y,w,h; panel_rect(&S,pp,&x,&y,&w,&h);
            long nz=0,sum=0; int mn=255,mx=0;
            for(int yy=y;yy<y+h;++yy)for(int xx=x;xx<x+w;++xx){ uint32_t px=S.fb[(size_t)yy*S.win_w+xx];
                int g=px&255; if(g)nz++; sum+=g; if(g<mn)mn=g; if(g>mx)mx=g; }
            const char*nm[4]={"SURF","XY","XZ","YZ"};
            fprintf(stderr,"panel %-4s: nonzero %ld/%d  min %d max %d mean %.1f\n",
                    nm[pp],nz,w*h,mn,mx,sum/(double)(w*h)); }
        fprintf(stderr,"wrote %s (%dx%d)\n",dump,S.win_w,S.win_h);
        free(S.fb); free(S.scratch);
        if(S.surf_loaded) mc_surface_free(&S.surf);
        mc_volume_free(S.vol);
        return 0;
    }

    SDL_Window *win = SDL_CreateWindow("mc_sview (software)", 1200, 900, SDL_WINDOW_RESIZABLE);
    SDL_Renderer *ren = SDL_CreateRenderer(win, NULL);
    if (!win||!ren){ fprintf(stderr,"SDL window/renderer: %s\n",SDL_GetError()); return 1; }
    SDL_SetWindowBordered(win, true);     /* libdecor draws the title bar (GNOME/Wayland) */
    SDL_Texture *tex = NULL;

    atomic_store(&S.dirty,1);
    int dragging=0, drag_panel=-1;
    int running=1;
    while (running){
        SDL_Event e;
        while (SDL_PollEvent(&e)){
            if (e.type==SDL_EVENT_QUIT) running=0;
            else if (e.type==SDL_EVENT_KEY_DOWN){
                SDL_Keycode k=e.key.key;
                int step = (e.key.mod & SDL_KMOD_SHIFT)?64:8;
                if (k==SDLK_Q||k==SDLK_ESCAPE) running=0;
                else if (k==SDLK_1) S.maxed=PANEL_SURF;
                else if (k==SDLK_2) S.maxed=PANEL_XY;
                else if (k==SDLK_3) S.maxed=PANEL_XZ;
                else if (k==SDLK_4) S.maxed=PANEL_YZ;
                else if (k==SDLK_0||k==SDLK_TAB) S.maxed=-1;
                else if (k==SDLK_PAGEUP)   S.fz+=step;
                else if (k==SDLK_PAGEDOWN) S.fz-=step;
                else if (k==SDLK_UP)       S.fy-=step;
                else if (k==SDLK_DOWN)     S.fy+=step;
                else if (k==SDLK_LEFT)     S.fx-=step;
                else if (k==SDLK_RIGHT)    S.fx+=step;
                else if (k==SDLK_LEFTBRACKET)  { S.win_high-=8; if(S.win_high<=S.win_low)S.win_high=S.win_low+1; }
                else if (k==SDLK_RIGHTBRACKET) { S.win_high+=8; if(S.win_high>255)S.win_high=255; }
                else if (k==SDLK_MINUS)        { S.win_low-=8; if(S.win_low<0)S.win_low=0; }
                else if (k==SDLK_EQUALS)       { S.win_low+=8; if(S.win_low>=S.win_high)S.win_low=S.win_high-1; }
                else if (k==SDLK_A)            { S.want_autowin=1; }   /* auto-contrast */
                if(S.fz<0)S.fz=0; if(S.fz>=S.nz)S.fz=S.nz-1;
                if(S.fy<0)S.fy=0; if(S.fy>=S.ny)S.fy=S.ny-1;
                if(S.fx<0)S.fx=0; if(S.fx>=S.nx)S.fx=S.nx-1;
                mc_volume_request_region(S.vol,0,S.fz/256,S.fy/256,S.fx/256);
                atomic_store(&S.dirty,1);
            }
            else if (e.type==SDL_EVENT_MOUSE_BUTTON_DOWN && e.button.button==SDL_BUTTON_LEFT){
                dragging=1; drag_panel=panel_at(&S,(int)e.button.x,(int)e.button.y);
            }
            else if (e.type==SDL_EVENT_MOUSE_BUTTON_UP && e.button.button==SDL_BUTTON_LEFT) dragging=0;
            else if (e.type==SDL_EVENT_MOUSE_MOTION && dragging && drag_panel>=0){
                panel_view *V=&S.pv[drag_panel];
                float sc = V->scale>0?V->scale:1.0f;
                if (drag_panel!=PANEL_SURF){           /* pan moves the in-plane view */
                    V->pan_u -= e.motion.xrel*sc;
                    V->pan_v -= e.motion.yrel*sc;
                } else {
                    S.surf_x0 -= e.motion.xrel*sc;
                    S.surf_y0 -= e.motion.yrel*sc;
                }
                atomic_store(&S.dirty,1);
            }
            else if (e.type==SDL_EVENT_MOUSE_WHEEL){
                int mx,my; SDL_GetMouseState(NULL,NULL); float fmx,fmy; SDL_GetMouseState(&fmx,&fmy); mx=(int)fmx; my=(int)fmy;
                int p=panel_at(&S,mx,my);
                float f=(e.wheel.y>0)?1.0f/1.2f:1.2f;     /* wheel up = zoom in (smaller vox/px) */
                S.pv[p].scale*=f;
                if (S.pv[p].scale<0.05f)S.pv[p].scale=0.05f;
                if (S.pv[p].scale>32.0f)S.pv[p].scale=32.0f;
                atomic_store(&S.dirty,1);
            }
            else if (e.type==SDL_EVENT_WINDOW_RESIZED) atomic_store(&S.dirty,1);
        }

        int ow,oh; SDL_GetWindowSizeInPixels(win,&ow,&oh);
        if (ow!=S.win_w || oh!=S.win_h || !S.fb){
            S.win_w=ow; S.win_h=oh;
            free(S.fb); S.fb=malloc((size_t)ow*oh*4);
            if (tex) SDL_DestroyTexture(tex);
            tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, ow, oh);
            atomic_store(&S.dirty,1);
        }
        if (atomic_exchange(&S.dirty,0) && S.fb && tex){
            /* Freeze the cache for the frame: get/get_copy then read LOCK-FREE,
             * so the render threads can't race a streaming write (that race is a
             * SIGBUS in the decoder on a torn mmap read). thaw() after lets newly
             * streamed regions land before the next frame. */
            mc_volume_freeze(S.vol);
            render_all(&S);
            mc_volume_thaw(S.vol);
            SDL_UpdateTexture(tex, NULL, S.fb, S.win_w*4);
        }
        SDL_RenderClear(ren);
        if (tex) SDL_RenderTexture(ren, tex, NULL, NULL);
        SDL_RenderPresent(ren);
        SDL_Delay(8);
    }

    if (tex) SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren); SDL_DestroyWindow(win);
    if (S.surf_loaded) mc_surface_free(&S.surf);
    mc_volume_free(S.vol);
    free(S.fb); free(S.scratch);
    SDL_Quit();
    return 0;
}
