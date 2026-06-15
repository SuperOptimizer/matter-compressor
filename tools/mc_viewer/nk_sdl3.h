/* nk_sdl3.h — minimal Nuklear backend for SDL3's SDL_Renderer.
 *
 * The upstream Nuklear demo backend (nuklear_sdl_renderer.h) targets SDL2:
 * it includes <SDL.h>, calls SDL_RenderSetClipRect, and passes byte SDL_Color
 * to SDL_RenderGeometryRaw. SDL3 renamed those (SDL_SetRenderClipRect) and
 * changed SDL_RenderGeometryRaw to take float SDL_FColor. This is a small,
 * self-contained SDL3-native rewrite of just what mc_viewer uses.
 *
 * Usage (in exactly one TU, after defining NK_IMPLEMENTATION + including
 * nuklear.h):
 *     #define NK_SDL3_IMPLEMENTATION
 *     #include "nk_sdl3.h"
 * Elsewhere, include this header without the define for the prototypes.
 */
#ifndef NK_SDL3_H_
#define NK_SDL3_H_

#include <SDL3/SDL.h>

struct nk_context *nk_sdl3_init(SDL_Window *win, SDL_Renderer *renderer);
void               nk_sdl3_font_stash_begin(struct nk_font_atlas **atlas);
void               nk_sdl3_font_stash_end(void);
bool               nk_sdl3_handle_event(SDL_Event *evt);
void               nk_sdl3_render(enum nk_anti_aliasing aa);
void               nk_sdl3_shutdown(void);

#endif /* NK_SDL3_H_ */

#ifdef NK_SDL3_IMPLEMENTATION
#undef NK_SDL3_IMPLEMENTATION

struct nk_sdl3_vertex {
    float position[2];
    float uv[2];
    nk_byte col[4];
};

static struct {
    SDL_Window               *win;
    SDL_Renderer             *renderer;
    SDL_Texture              *font_tex;
    struct nk_context         ctx;
    struct nk_font_atlas      atlas;
    struct nk_buffer          cmds;       /* draw command buffer */
    struct nk_draw_null_texture tex_null;
} nk_sdl3;

NK_API struct nk_context *
nk_sdl3_init(SDL_Window *win, SDL_Renderer *renderer)
{
    nk_sdl3.win = win;
    nk_sdl3.renderer = renderer;
    nk_init_default(&nk_sdl3.ctx, 0);
    nk_buffer_init_default(&nk_sdl3.cmds);
    return &nk_sdl3.ctx;
}

NK_API void
nk_sdl3_font_stash_begin(struct nk_font_atlas **atlas)
{
    nk_font_atlas_init_default(&nk_sdl3.atlas);
    nk_font_atlas_begin(&nk_sdl3.atlas);
    *atlas = &nk_sdl3.atlas;
}

NK_API void
nk_sdl3_font_stash_end(void)
{
    const void *image;
    int w, h;
    image = nk_font_atlas_bake(&nk_sdl3.atlas, &w, &h, NK_FONT_ATLAS_RGBA32);

    nk_sdl3.font_tex = SDL_CreateTexture(nk_sdl3.renderer, SDL_PIXELFORMAT_ABGR8888,
                                         SDL_TEXTUREACCESS_STATIC, w, h);
    SDL_UpdateTexture(nk_sdl3.font_tex, NULL, image, 4 * w);
    SDL_SetTextureBlendMode(nk_sdl3.font_tex, SDL_BLENDMODE_BLEND);

    nk_font_atlas_end(&nk_sdl3.atlas, nk_handle_ptr(nk_sdl3.font_tex), &nk_sdl3.tex_null);
    if (nk_sdl3.atlas.default_font)
        nk_style_set_font(&nk_sdl3.ctx, &nk_sdl3.atlas.default_font->handle);
}

NK_API void
nk_sdl3_render(enum nk_anti_aliasing aa)
{
    SDL_Renderer *r = nk_sdl3.renderer;
    const struct nk_draw_command *cmd;
    struct nk_buffer vbuf, ebuf;

    /* Nuklear renders in window (logical) coordinates; if the renderer has a
     * scale set, SDL applies it. We render 1:1 over the output. */
    float scale_x = 1.0f, scale_y = 1.0f;
    {
        int win_w, win_h, out_w, out_h;
        SDL_GetWindowSize(nk_sdl3.win, &win_w, &win_h);
        SDL_GetRenderOutputSize(r, &out_w, &out_h);
        if (win_w > 0) scale_x = (float)out_w / (float)win_w;
        if (win_h > 0) scale_y = (float)out_h / (float)win_h;
    }

    static const struct nk_draw_vertex_layout_element vertex_layout[] = {
        { NK_VERTEX_POSITION, NK_FORMAT_FLOAT,    NK_OFFSETOF(struct nk_sdl3_vertex, position) },
        { NK_VERTEX_TEXCOORD, NK_FORMAT_FLOAT,    NK_OFFSETOF(struct nk_sdl3_vertex, uv) },
        { NK_VERTEX_COLOR,    NK_FORMAT_R8G8B8A8, NK_OFFSETOF(struct nk_sdl3_vertex, col) },
        { NK_VERTEX_LAYOUT_END }
    };
    struct nk_convert_config config;
    memset(&config, 0, sizeof(config));
    config.vertex_layout = vertex_layout;
    config.vertex_size = sizeof(struct nk_sdl3_vertex);
    config.vertex_alignment = NK_ALIGNOF(struct nk_sdl3_vertex);
    config.tex_null = nk_sdl3.tex_null;
    config.circle_segment_count = 22;
    config.curve_segment_count = 22;
    config.arc_segment_count = 22;
    config.global_alpha = 1.0f;
    config.shape_AA = aa;
    config.line_AA = aa;

    nk_buffer_init_default(&vbuf);
    nk_buffer_init_default(&ebuf);
    nk_convert(&nk_sdl3.ctx, &nk_sdl3.cmds, &vbuf, &ebuf, &config);

    const struct nk_sdl3_vertex *verts = (const struct nk_sdl3_vertex *)nk_buffer_memory_const(&vbuf);
    const nk_draw_index *elems = (const nk_draw_index *)nk_buffer_memory_const(&ebuf);
    int nverts = (int)(vbuf.needed / sizeof(struct nk_sdl3_vertex));
    int elem_off = 0;   /* running index into the element buffer */

    /* SDL3 RenderGeometryRaw takes float SDL_FColor; Nuklear emits packed
     * R8G8B8A8 bytes. Convert into a scratch array once per frame. */
    SDL_FColor *fcol = NULL;
    if (nverts > 0) {
        fcol = (SDL_FColor *)SDL_malloc((size_t)nverts * sizeof(SDL_FColor));
        for (int i = 0; i < nverts; i++) {
            fcol[i].r = verts[i].col[0] / 255.0f;
            fcol[i].g = verts[i].col[1] / 255.0f;
            fcol[i].b = verts[i].col[2] / 255.0f;
            fcol[i].a = verts[i].col[3] / 255.0f;
        }
    }

    nk_draw_foreach(cmd, &nk_sdl3.ctx, &nk_sdl3.cmds) {
        if (!cmd->elem_count) continue;

        SDL_Rect clip;
        clip.x = (int)(cmd->clip_rect.x * scale_x);
        clip.y = (int)(cmd->clip_rect.y * scale_y);
        clip.w = (int)(cmd->clip_rect.w * scale_x);
        clip.h = (int)(cmd->clip_rect.h * scale_y);
        if (clip.x < 0) { clip.w += clip.x; clip.x = 0; }
        if (clip.y < 0) { clip.h += clip.y; clip.y = 0; }
        if (clip.w < 0) clip.w = 0;
        if (clip.h < 0) clip.h = 0;
        SDL_SetRenderClipRect(r, &clip);

        SDL_Texture *tex = (SDL_Texture *)cmd->texture.ptr;
        SDL_RenderGeometryRaw(
            r, tex,
            &verts->position[0], sizeof(struct nk_sdl3_vertex),
            fcol, sizeof(SDL_FColor),
            &verts->uv[0], sizeof(struct nk_sdl3_vertex),
            nverts,
            elems + elem_off, (int)cmd->elem_count, sizeof(nk_draw_index));
        elem_off += (int)cmd->elem_count;
    }

    SDL_free(fcol);
    SDL_SetRenderClipRect(r, NULL);
    nk_clear(&nk_sdl3.ctx);
    nk_buffer_clear(&nk_sdl3.cmds);
    nk_buffer_free(&vbuf);
    nk_buffer_free(&ebuf);
}

NK_API bool
nk_sdl3_handle_event(SDL_Event *evt)
{
    struct nk_context *ctx = &nk_sdl3.ctx;
    switch (evt->type) {
    case SDL_EVENT_MOUSE_MOTION:
        nk_input_motion(ctx, (int)evt->motion.x, (int)evt->motion.y);
        return true;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP: {
        int down = (evt->type == SDL_EVENT_MOUSE_BUTTON_DOWN);
        int x = (int)evt->button.x, y = (int)evt->button.y;
        switch (evt->button.button) {
        case SDL_BUTTON_LEFT:   nk_input_button(ctx, NK_BUTTON_LEFT,   x, y, down); break;
        case SDL_BUTTON_MIDDLE: nk_input_button(ctx, NK_BUTTON_MIDDLE, x, y, down); break;
        case SDL_BUTTON_RIGHT:  nk_input_button(ctx, NK_BUTTON_RIGHT,  x, y, down); break;
        default: break;
        }
        return true;
    }
    case SDL_EVENT_MOUSE_WHEEL:
        nk_input_scroll(ctx, nk_vec2(evt->wheel.x, evt->wheel.y));
        return true;
    default:
        return false;
    }
}

NK_API void
nk_sdl3_shutdown(void)
{
    if (nk_sdl3.font_tex) SDL_DestroyTexture(nk_sdl3.font_tex);
    nk_font_atlas_clear(&nk_sdl3.atlas);
    nk_buffer_free(&nk_sdl3.cmds);
    nk_free(&nk_sdl3.ctx);
    memset(&nk_sdl3, 0, sizeof(nk_sdl3));
}

#endif /* NK_SDL3_IMPLEMENTATION */
