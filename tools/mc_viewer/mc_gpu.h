/* mc_gpu.h — full SDL_GPU frontend for mc_viewer: slice blit + Nuklear UI.
 *
 * Owns the entire GPU frame on the SDL_GPU API (no SDL_Renderer anywhere):
 *   - a fullscreen-quad pipeline that blits the CPU-rendered ARGB slice
 *     (zoom/pan in the vertex shader), and
 *   - a from-scratch Nuklear backend (its own pos/uv/color pipeline, font-atlas
 *     texture, per-frame vertex/index buffers, per-command scissor) drawn in the
 *     same render pass on top.
 *
 * SDL_GPU is the abstraction over Vulkan/Metal/D3D12 (+ an OpenGL fallback), so
 * this is the milestone-2 GPU path on the GL -> Vulkan -> Metal -> DX12 roadmap.
 * Shaders are GLSL compiled to SPIR-V at build time and embedded as .spv.h.
 *
 * Define MC_GPU_IMPLEMENTATION in exactly one TU (after nuklear.h is included
 * with its implementation, or at least its declarations).
 */
#ifndef MC_GPU_H_
#define MC_GPU_H_

#include <SDL3/SDL.h>
#include <stdbool.h>
#include <stddef.h>

struct nk_context;

typedef struct mc_gpu mc_gpu;

/* Create device, claim `win`, build both pipelines + the Nuklear context and
 * font atlas. Returns NULL on failure (SDL_GetError). */
mc_gpu *mc_gpu_create(SDL_Window *win);
void    mc_gpu_destroy(mc_gpu *g);

/* The Nuklear context to build the UI into each frame (between frames). */
struct nk_context *mc_gpu_nk(mc_gpu *g);

/* Feed an SDL event to Nuklear. Call nk_input_begin/end around the poll loop. */
void mc_gpu_nk_input_begin(mc_gpu *g);
bool mc_gpu_nk_handle_event(mc_gpu *g, SDL_Event *e);
void mc_gpu_nk_input_end(mc_gpu *g);

/* Upload a w*h ARGB8888 (SDL ARGB32, BGRA bytes in memory) slice; (re)creates
 * the texture on size change. */
bool mc_gpu_upload_slice(mc_gpu *g, const uint32_t *argb, int w, int h);

/* Draw + present one frame: clear, slice quad, then the Nuklear draw list.
 * The slice quad is drawn `draw_w` x `draw_h` pixels, centered, then shifted by
 * (pan_x,pan_y) pixels — the caller computes draw_w/h from the slice's voxel
 * extent * zoom (so it is independent of the texture's pixel resolution).
 * gain/bias tweak the fragment output (1,0 = identity). Consumes + clears the
 * Nuklear command buffer. */
bool mc_gpu_render(mc_gpu *g, float draw_w, float draw_h, float pan_x, float pan_y,
                   float gain, float bias);

/* --- split frame, for compositing an externally-drawn slice (e.g. mc_gpu_vol)
 * with the Nuklear UI in one command buffer ---
 * begin: acquire a command buffer + swapchain texture and upload the current
 * Nuklear geometry (copy pass, before any render pass). Returns false if no
 * swapchain (minimized) — nothing to do this frame. On success *cmd/*swap/*ow/*oh
 * are set. The caller then draws the slice into *swap (its own clear render
 * pass), and finishes with mc_gpu_frame_end_nuklear. */
SDL_GPUDevice *mc_gpu_device(mc_gpu *g);
const char    *mc_gpu_driver(mc_gpu *g);     // SDL_GPU backend name ("vulkan"/"metal"/...)
bool mc_gpu_frame_begin(mc_gpu *g, SDL_GPUCommandBuffer **cmd,
                        SDL_GPUTexture **swap, Uint32 *ow, Uint32 *oh);
/* Draw the Nuklear UI into a LOAD render pass on `swap` (preserving the slice
 * already drawn there), submit `cmd`, and clear the Nuklear command list. */
void mc_gpu_frame_end_nuklear(mc_gpu *g, SDL_GPUCommandBuffer *cmd,
                              SDL_GPUTexture *swap, Uint32 ow, Uint32 oh);

void mc_gpu_output_size(const mc_gpu *g, int *w, int *h);
void mc_gpu_set_nearest(mc_gpu *g, bool nearest);

#endif /* MC_GPU_H_ */

#ifdef MC_GPU_IMPLEMENTATION
#undef MC_GPU_IMPLEMENTATION

#include "shaders/blit.vert.spv.h"
#include "shaders/blit.frag.spv.h"
#include "shaders/ui.vert.spv.h"
#include "shaders/ui.frag.spv.h"

typedef struct { float scale[2], offset[2]; } mc_gpu_vert_ubo;
typedef struct { float gain, bias, pad0, pad1; } mc_gpu_frag_ubo;

/* Nuklear vertex: screen-space pos, uv, packed RGBA8 color. */
typedef struct { float pos[2]; float uv[2]; nk_byte col[4]; } mc_ui_vertex;

#define MC_UI_VBUF_BYTES (512 * 1024)
#define MC_UI_IBUF_BYTES (128 * 1024)

struct mc_gpu {
    SDL_Window               *win;
    SDL_GPUDevice            *dev;

    /* slice */
    SDL_GPUGraphicsPipeline  *slice_pipe;
    SDL_GPUSampler           *samp_nearest, *samp_linear;
    bool                      nearest;
    SDL_GPUTexture           *slice;
    int                       sw, sh;

    /* ui */
    SDL_GPUGraphicsPipeline  *ui_pipe;
    SDL_GPUSampler           *ui_samp;
    SDL_GPUTexture           *font_tex;
    SDL_GPUBuffer            *ui_vbuf, *ui_ibuf;
    struct nk_context         nk;
    struct nk_font_atlas      atlas;
    struct nk_buffer          cmds;
    struct nk_draw_null_texture tex_null;

    /* split-frame scratch (mc_gpu_frame_begin / mc_gpu_frame_end_nuklear) */
    struct nk_buffer fr_vbuf, fr_ebuf;
    Uint32           fr_vbytes, fr_ibytes;
};

static SDL_GPUShader *mc_gpu_shader(SDL_GPUDevice *dev, SDL_GPUShaderStage stage,
                                    const uint32_t *code, size_t bytes,
                                    Uint32 nsamp, Uint32 nunif) {
    SDL_GPUShaderCreateInfo ci = {0};
    ci.code = (const Uint8 *)code; ci.code_size = bytes;
    ci.entrypoint = "main"; ci.format = SDL_GPU_SHADERFORMAT_SPIRV; ci.stage = stage;
    ci.num_samplers = nsamp; ci.num_uniform_buffers = nunif;
    return SDL_CreateGPUShader(dev, &ci);
}

mc_gpu *mc_gpu_create(SDL_Window *win) {
    mc_gpu *g = SDL_calloc(1, sizeof *g);
    if (!g) return NULL;
    g->win = win; g->nearest = true;

    // SDL_GPU picks the backend automatically (Vulkan / Metal / D3D12 per
    // platform — there is no OpenGL GPU backend). MC_GPU_DRIVER forces a
    // specific one by name (e.g. "vulkan", "metal", "direct3d12"); NULL = auto.
    const char *want = SDL_getenv("MC_GPU_DRIVER");
    g->dev = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, false, want);
    if (!g->dev) {
        if (want) fprintf(stderr, "mc_gpu: MC_GPU_DRIVER=%s unavailable; "
                                  "retrying auto-select\n", want);
        g->dev = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, false, NULL);
    }
    if (!g->dev || !SDL_ClaimWindowForGPUDevice(g->dev, win)) goto fail;
    fprintf(stderr, "mc_gpu: SDL_GPU backend = %s\n", SDL_GetGPUDeviceDriver(g->dev));

    SDL_GPUTextureFormat swfmt = SDL_GetGPUSwapchainTextureFormat(g->dev, win);

    /* ---- slice pipeline (no vertex buffer; quad from gl_VertexIndex) ---- */
    {
        SDL_GPUShader *vs = mc_gpu_shader(g->dev, SDL_GPU_SHADERSTAGE_VERTEX,
            blit_vert_spv, sizeof blit_vert_spv, 0, 1);
        SDL_GPUShader *fs = mc_gpu_shader(g->dev, SDL_GPU_SHADERSTAGE_FRAGMENT,
            blit_frag_spv, sizeof blit_frag_spv, 1, 1);
        if (!vs || !fs) { if (vs) SDL_ReleaseGPUShader(g->dev, vs);
                          if (fs) SDL_ReleaseGPUShader(g->dev, fs); goto fail; }
        SDL_GPUColorTargetDescription ct = {0};
        ct.format = swfmt;
        SDL_GPUGraphicsPipelineCreateInfo p = {0};
        p.vertex_shader = vs; p.fragment_shader = fs;
        p.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
        p.target_info.color_target_descriptions = &ct;
        p.target_info.num_color_targets = 1;
        g->slice_pipe = SDL_CreateGPUGraphicsPipeline(g->dev, &p);
        SDL_ReleaseGPUShader(g->dev, vs); SDL_ReleaseGPUShader(g->dev, fs);
        if (!g->slice_pipe) goto fail;
    }

    /* ---- ui pipeline (pos/uv/color vertex, alpha blend) ---- */
    {
        SDL_GPUShader *vs = mc_gpu_shader(g->dev, SDL_GPU_SHADERSTAGE_VERTEX,
            ui_vert_spv, sizeof ui_vert_spv, 0, 1);
        SDL_GPUShader *fs = mc_gpu_shader(g->dev, SDL_GPU_SHADERSTAGE_FRAGMENT,
            ui_frag_spv, sizeof ui_frag_spv, 1, 0);
        if (!vs || !fs) { if (vs) SDL_ReleaseGPUShader(g->dev, vs);
                          if (fs) SDL_ReleaseGPUShader(g->dev, fs); goto fail; }

        SDL_GPUVertexBufferDescription vbd = {0};
        vbd.slot = 0; vbd.pitch = sizeof(mc_ui_vertex);
        vbd.input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX;
        SDL_GPUVertexAttribute va[3] = {0};
        va[0].location = 0; va[0].buffer_slot = 0;
        va[0].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2;
        va[0].offset = offsetof(mc_ui_vertex, pos);
        va[1].location = 1; va[1].buffer_slot = 0;
        va[1].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2;
        va[1].offset = offsetof(mc_ui_vertex, uv);
        va[2].location = 2; va[2].buffer_slot = 0;
        va[2].format = SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4_NORM;
        va[2].offset = offsetof(mc_ui_vertex, col);

        SDL_GPUColorTargetDescription ct = {0};
        ct.format = swfmt;
        ct.blend_state.enable_blend = true;
        ct.blend_state.src_color_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA;
        ct.blend_state.dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
        ct.blend_state.color_blend_op = SDL_GPU_BLENDOP_ADD;
        ct.blend_state.src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
        ct.blend_state.dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
        ct.blend_state.alpha_blend_op = SDL_GPU_BLENDOP_ADD;

        SDL_GPUGraphicsPipelineCreateInfo p = {0};
        p.vertex_shader = vs; p.fragment_shader = fs;
        p.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
        p.vertex_input_state.vertex_buffer_descriptions = &vbd;
        p.vertex_input_state.num_vertex_buffers = 1;
        p.vertex_input_state.vertex_attributes = va;
        p.vertex_input_state.num_vertex_attributes = 3;
        p.target_info.color_target_descriptions = &ct;
        p.target_info.num_color_targets = 1;
        g->ui_pipe = SDL_CreateGPUGraphicsPipeline(g->dev, &p);
        SDL_ReleaseGPUShader(g->dev, vs); SDL_ReleaseGPUShader(g->dev, fs);
        if (!g->ui_pipe) goto fail;
    }

    /* samplers */
    {
        SDL_GPUSamplerCreateInfo s = {0};
        s.address_mode_u = s.address_mode_v = s.address_mode_w =
            SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
        s.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
        s.min_filter = s.mag_filter = SDL_GPU_FILTER_NEAREST;
        g->samp_nearest = SDL_CreateGPUSampler(g->dev, &s);
        g->ui_samp = SDL_CreateGPUSampler(g->dev, &s);
        s.min_filter = s.mag_filter = SDL_GPU_FILTER_LINEAR;
        g->samp_linear = SDL_CreateGPUSampler(g->dev, &s);
    }

    /* persistent UI vertex/index buffers (re-uploaded each frame) */
    {
        SDL_GPUBufferCreateInfo b = {0};
        b.usage = SDL_GPU_BUFFERUSAGE_VERTEX; b.size = MC_UI_VBUF_BYTES;
        g->ui_vbuf = SDL_CreateGPUBuffer(g->dev, &b);
        b.usage = SDL_GPU_BUFFERUSAGE_INDEX;  b.size = MC_UI_IBUF_BYTES;
        g->ui_ibuf = SDL_CreateGPUBuffer(g->dev, &b);
        if (!g->ui_vbuf || !g->ui_ibuf) goto fail;
    }

    /* Nuklear context + font atlas -> GPU texture */
    nk_init_default(&g->nk, 0);
    nk_buffer_init_default(&g->cmds);
    {
        const void *img; int aw, ah;
        nk_font_atlas_init_default(&g->atlas);
        nk_font_atlas_begin(&g->atlas);
        img = nk_font_atlas_bake(&g->atlas, &aw, &ah, NK_FONT_ATLAS_RGBA32);

        SDL_GPUTextureCreateInfo tci = {0};
        tci.type = SDL_GPU_TEXTURETYPE_2D;
        tci.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;  /* atlas is RGBA32 */
        tci.usage = SDL_GPU_TEXTUREUSAGE_SAMPLER;
        tci.width = (Uint32)aw; tci.height = (Uint32)ah;
        tci.layer_count_or_depth = 1; tci.num_levels = 1;
        g->font_tex = SDL_CreateGPUTexture(g->dev, &tci);

        Uint32 fbytes = (Uint32)aw * (Uint32)ah * 4;
        SDL_GPUTransferBufferCreateInfo bci = {0};
        bci.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD; bci.size = fbytes;
        SDL_GPUTransferBuffer *tb = SDL_CreateGPUTransferBuffer(g->dev, &bci);
        void *m = SDL_MapGPUTransferBuffer(g->dev, tb, false);
        SDL_memcpy(m, img, fbytes);
        SDL_UnmapGPUTransferBuffer(g->dev, tb);
        SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(g->dev);
        SDL_GPUCopyPass *cp = SDL_BeginGPUCopyPass(cmd);
        SDL_GPUTextureTransferInfo ti = {0};
        ti.transfer_buffer = tb; ti.pixels_per_row = (Uint32)aw; ti.rows_per_layer = (Uint32)ah;
        SDL_GPUTextureRegion tr = {0};
        tr.texture = g->font_tex; tr.w = (Uint32)aw; tr.h = (Uint32)ah; tr.d = 1;
        SDL_UploadToGPUTexture(cp, &ti, &tr, false);
        SDL_EndGPUCopyPass(cp);
        SDL_SubmitGPUCommandBuffer(cmd);
        SDL_ReleaseGPUTransferBuffer(g->dev, tb);

        nk_font_atlas_end(&g->atlas, nk_handle_ptr(g->font_tex), &g->tex_null);
        if (g->atlas.default_font)
            nk_style_set_font(&g->nk, &g->atlas.default_font->handle);
    }
    return g;

fail:
    mc_gpu_destroy(g);
    return NULL;
}

void mc_gpu_destroy(mc_gpu *g) {
    if (!g) return;
    if (g->dev) {
        if (g->slice)        SDL_ReleaseGPUTexture(g->dev, g->slice);
        if (g->font_tex)     SDL_ReleaseGPUTexture(g->dev, g->font_tex);
        if (g->ui_vbuf)      SDL_ReleaseGPUBuffer(g->dev, g->ui_vbuf);
        if (g->ui_ibuf)      SDL_ReleaseGPUBuffer(g->dev, g->ui_ibuf);
        if (g->samp_nearest) SDL_ReleaseGPUSampler(g->dev, g->samp_nearest);
        if (g->samp_linear)  SDL_ReleaseGPUSampler(g->dev, g->samp_linear);
        if (g->ui_samp)      SDL_ReleaseGPUSampler(g->dev, g->ui_samp);
        if (g->slice_pipe)   SDL_ReleaseGPUGraphicsPipeline(g->dev, g->slice_pipe);
        if (g->ui_pipe)      SDL_ReleaseGPUGraphicsPipeline(g->dev, g->ui_pipe);
        SDL_ReleaseWindowFromGPUDevice(g->dev, g->win);
        SDL_DestroyGPUDevice(g->dev);
    }
    /* nuklear teardown (init_default may not have run on an early failure;
     * nk_free is safe on a zeroed ctx). */
    nk_font_atlas_clear(&g->atlas);
    nk_buffer_free(&g->cmds);
    nk_free(&g->nk);
    SDL_free(g);
}

struct nk_context *mc_gpu_nk(mc_gpu *g) { return &g->nk; }
void mc_gpu_set_nearest(mc_gpu *g, bool n) { g->nearest = n; }
void mc_gpu_output_size(const mc_gpu *g, int *w, int *h) {
    SDL_GetWindowSizeInPixels(g->win, w, h);
}

void mc_gpu_nk_input_begin(mc_gpu *g) { nk_input_begin(&g->nk); }
void mc_gpu_nk_input_end(mc_gpu *g)   { nk_input_end(&g->nk); }

bool mc_gpu_nk_handle_event(mc_gpu *g, SDL_Event *e) {
    struct nk_context *ctx = &g->nk;
    switch (e->type) {
    case SDL_EVENT_MOUSE_MOTION:
        nk_input_motion(ctx, (int)e->motion.x, (int)e->motion.y); return true;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP: {
        int down = (e->type == SDL_EVENT_MOUSE_BUTTON_DOWN);
        int x = (int)e->button.x, y = (int)e->button.y;
        switch (e->button.button) {
        case SDL_BUTTON_LEFT:   nk_input_button(ctx, NK_BUTTON_LEFT,   x, y, down); break;
        case SDL_BUTTON_MIDDLE: nk_input_button(ctx, NK_BUTTON_MIDDLE, x, y, down); break;
        case SDL_BUTTON_RIGHT:  nk_input_button(ctx, NK_BUTTON_RIGHT,  x, y, down); break;
        default: break;
        }
        return true;
    }
    case SDL_EVENT_MOUSE_WHEEL:
        nk_input_scroll(ctx, nk_vec2(e->wheel.x, e->wheel.y)); return true;
    default: return false;
    }
}

bool mc_gpu_upload_slice(mc_gpu *g, const uint32_t *argb, int w, int h) {
    if (w <= 0 || h <= 0) return false;
    if (!g->slice || g->sw != w || g->sh != h) {
        if (g->slice) SDL_ReleaseGPUTexture(g->dev, g->slice);
        SDL_GPUTextureCreateInfo tci = {0};
        tci.type = SDL_GPU_TEXTURETYPE_2D;
        tci.format = SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM;   /* SDL ARGB32 bytes */
        tci.usage = SDL_GPU_TEXTUREUSAGE_SAMPLER;
        tci.width = (Uint32)w; tci.height = (Uint32)h;
        tci.layer_count_or_depth = 1; tci.num_levels = 1;
        g->slice = SDL_CreateGPUTexture(g->dev, &tci);
        if (!g->slice) return false;
        g->sw = w; g->sh = h;
    }
    Uint32 bytes = (Uint32)w * (Uint32)h * 4;
    SDL_GPUTransferBufferCreateInfo bci = {0};
    bci.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD; bci.size = bytes;
    SDL_GPUTransferBuffer *tb = SDL_CreateGPUTransferBuffer(g->dev, &bci);
    if (!tb) return false;
    void *m = SDL_MapGPUTransferBuffer(g->dev, tb, false);
    if (!m) { SDL_ReleaseGPUTransferBuffer(g->dev, tb); return false; }
    SDL_memcpy(m, argb, bytes);
    SDL_UnmapGPUTransferBuffer(g->dev, tb);
    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(g->dev);
    SDL_GPUCopyPass *cp = SDL_BeginGPUCopyPass(cmd);
    SDL_GPUTextureTransferInfo src = {0};
    src.transfer_buffer = tb; src.pixels_per_row = (Uint32)w; src.rows_per_layer = (Uint32)h;
    SDL_GPUTextureRegion dst = {0};
    dst.texture = g->slice; dst.w = (Uint32)w; dst.h = (Uint32)h; dst.d = 1;
    SDL_UploadToGPUTexture(cp, &src, &dst, false);
    SDL_EndGPUCopyPass(cp);
    SDL_SubmitGPUCommandBuffer(cmd);
    SDL_ReleaseGPUTransferBuffer(g->dev, tb);
    return true;
}

/* Convert the current Nuklear command list to vertices/indices and upload into
 * the persistent UI buffers via a copy pass on `cmd`. Returns index count and
 * stores byte sizes; *out_verts is the converted vertex buffer base for the
 * caller's scissor walk. The returned nk_buffer must be freed by the caller. */
static void mc_gpu_ui_convert(mc_gpu *g, SDL_GPUCommandBuffer *cmd,
                              struct nk_buffer *vbuf, struct nk_buffer *ebuf,
                              Uint32 *vbytes, Uint32 *ibytes) {
    static const struct nk_draw_vertex_layout_element layout[] = {
        { NK_VERTEX_POSITION, NK_FORMAT_FLOAT,    NK_OFFSETOF(mc_ui_vertex, pos) },
        { NK_VERTEX_TEXCOORD, NK_FORMAT_FLOAT,    NK_OFFSETOF(mc_ui_vertex, uv) },
        { NK_VERTEX_COLOR,    NK_FORMAT_R8G8B8A8, NK_OFFSETOF(mc_ui_vertex, col) },
        { NK_VERTEX_LAYOUT_END }
    };
    struct nk_convert_config cfg; SDL_memset(&cfg, 0, sizeof cfg);
    cfg.vertex_layout = layout;
    cfg.vertex_size = sizeof(mc_ui_vertex);
    cfg.vertex_alignment = NK_ALIGNOF(mc_ui_vertex);
    cfg.tex_null = g->tex_null;
    cfg.circle_segment_count = 22; cfg.curve_segment_count = 22; cfg.arc_segment_count = 22;
    cfg.global_alpha = 1.0f;
    cfg.shape_AA = NK_ANTI_ALIASING_ON; cfg.line_AA = NK_ANTI_ALIASING_ON;

    nk_buffer_init_default(vbuf);
    nk_buffer_init_default(ebuf);
    nk_convert(&g->nk, &g->cmds, vbuf, ebuf, &cfg);

    *vbytes = (Uint32)vbuf->needed;
    *ibytes = (Uint32)ebuf->needed;
    if (*vbytes == 0 || *ibytes == 0) return;
    if (*vbytes > MC_UI_VBUF_BYTES) *vbytes = MC_UI_VBUF_BYTES;
    if (*ibytes > MC_UI_IBUF_BYTES) *ibytes = MC_UI_IBUF_BYTES;

    /* one transfer buffer holding [vertices | indices] */
    Uint32 total = *vbytes + *ibytes;
    SDL_GPUTransferBufferCreateInfo bci = {0};
    bci.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD; bci.size = total;
    SDL_GPUTransferBuffer *tb = SDL_CreateGPUTransferBuffer(g->dev, &bci);
    Uint8 *m = (Uint8 *)SDL_MapGPUTransferBuffer(g->dev, tb, false);
    SDL_memcpy(m, nk_buffer_memory_const(vbuf), *vbytes);
    SDL_memcpy(m + *vbytes, nk_buffer_memory_const(ebuf), *ibytes);
    SDL_UnmapGPUTransferBuffer(g->dev, tb);

    SDL_GPUCopyPass *cp = SDL_BeginGPUCopyPass(cmd);
    SDL_GPUTransferBufferLocation s = {0}; s.transfer_buffer = tb; s.offset = 0;
    SDL_GPUBufferRegion d = {0}; d.buffer = g->ui_vbuf; d.offset = 0; d.size = *vbytes;
    SDL_UploadToGPUBuffer(cp, &s, &d, false);
    s.offset = *vbytes;
    d.buffer = g->ui_ibuf; d.offset = 0; d.size = *ibytes;
    SDL_UploadToGPUBuffer(cp, &s, &d, false);
    SDL_EndGPUCopyPass(cp);
    SDL_ReleaseGPUTransferBuffer(g->dev, tb);   /* freed after submit completes */
}

// Draw the converted Nuklear command list into an OPEN render pass `rp` (the UI
// vbuf/ibuf must already be uploaded). Used by both the CPU-blit frame and the
// GPU-vol frame (where the slice was drawn by mc_gpu_vol in a prior pass).
static void mc_gpu_draw_nuklear(mc_gpu *g, SDL_GPUCommandBuffer *cmd, SDL_GPURenderPass *rp,
                                Uint32 vbytes, Uint32 ibytes, Uint32 ow, Uint32 oh) {
    if (!(vbytes && ibytes && ow && oh)) return;
    SDL_BindGPUGraphicsPipeline(rp, g->ui_pipe);
    SDL_GPUBufferBinding vb = {0}; vb.buffer = g->ui_vbuf;
    SDL_BindGPUVertexBuffers(rp, 0, &vb, 1);
    SDL_GPUBufferBinding ib = {0}; ib.buffer = g->ui_ibuf;
    SDL_BindGPUIndexBuffer(rp, &ib, SDL_GPU_INDEXELEMENTSIZE_16BIT);

    /* ortho(0,w,0,h) with +Y down, column-major for GLSL mat4. */
    int ww, wh; SDL_GetWindowSize(g->win, &ww, &wh);
    float L = 0, R = (float)ww, T = 0, B = (float)wh;
    float proj[16] = {
        2.0f/(R-L), 0, 0, 0,
        0, 2.0f/(T-B), 0, 0,
        0, 0, -1, 0,
        (R+L)/(L-R), (T+B)/(B-T), 0, 1,
    };
    SDL_PushGPUVertexUniformData(cmd, 0, proj, sizeof proj);

    float sx = ww ? (float)ow / (float)ww : 1.0f;
    float sy = wh ? (float)oh / (float)wh : 1.0f;

    const struct nk_draw_command *c;
    Uint32 ioff = 0;
    nk_draw_foreach(c, &g->nk, &g->cmds) {
        if (!c->elem_count) continue;
        SDL_Rect sc;
        sc.x = (int)(c->clip_rect.x * sx);
        sc.y = (int)(c->clip_rect.y * sy);
        sc.w = (int)(c->clip_rect.w * sx);
        sc.h = (int)(c->clip_rect.h * sy);
        if (sc.x < 0) { sc.w += sc.x; sc.x = 0; }
        if (sc.y < 0) { sc.h += sc.y; sc.y = 0; }
        if (sc.w < 0) sc.w = 0;
        if (sc.h < 0) sc.h = 0;
        SDL_SetGPUScissor(rp, &sc);

        SDL_GPUTextureSamplerBinding tsb = {0};
        tsb.texture = c->texture.ptr ? (SDL_GPUTexture *)c->texture.ptr : g->font_tex;
        tsb.sampler = g->ui_samp;
        SDL_BindGPUFragmentSamplers(rp, 0, &tsb, 1);

        SDL_DrawGPUIndexedPrimitives(rp, c->elem_count, 1, ioff, 0, 0);
        ioff += c->elem_count;
    }
}

bool mc_gpu_render(mc_gpu *g, float draw_w, float draw_h, float pan_x, float pan_y,
                   float gain, float bias) {
    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(g->dev);
    if (!cmd) return false;

    /* Convert + upload UI geometry on this command buffer BEFORE the render
     * pass (copy passes can't be inside a render pass). */
    struct nk_buffer vbuf, ebuf;
    Uint32 vbytes = 0, ibytes = 0;
    mc_gpu_ui_convert(g, cmd, &vbuf, &ebuf, &vbytes, &ibytes);
    const mc_ui_vertex *uverts = (const mc_ui_vertex *)nk_buffer_memory_const(&vbuf);

    SDL_GPUTexture *swap = NULL; Uint32 ow = 0, oh = 0;
    if (!SDL_WaitAndAcquireGPUSwapchainTexture(cmd, g->win, &swap, &ow, &oh) || !swap) {
        SDL_SubmitGPUCommandBuffer(cmd);
        nk_buffer_free(&vbuf); nk_buffer_free(&ebuf);
        nk_clear(&g->nk); nk_buffer_clear(&g->cmds);
        return true;
    }

    SDL_GPUColorTargetInfo cti = {0};
    cti.texture = swap; cti.load_op = SDL_GPU_LOADOP_CLEAR; cti.store_op = SDL_GPU_STOREOP_STORE;
    cti.clear_color = (SDL_FColor){ 0.06f, 0.06f, 0.08f, 1.0f };
    SDL_GPURenderPass *rp = SDL_BeginGPURenderPass(cmd, &cti, 1, NULL);

    /* ---- slice quad ---- */
    if (g->slice && ow && oh) {
        SDL_BindGPUGraphicsPipeline(rp, g->slice_pipe);
        SDL_GPUTextureSamplerBinding tsb = {0};
        tsb.texture = g->slice;
        tsb.sampler = g->nearest ? g->samp_nearest : g->samp_linear;
        SDL_BindGPUFragmentSamplers(rp, 0, &tsb, 1);
        mc_gpu_vert_ubo vu;
        vu.scale[0] = draw_w / (float)ow;
        vu.scale[1] = draw_h / (float)oh;
        vu.offset[0] =  (2.0f * pan_x) / (float)ow;
        vu.offset[1] = -(2.0f * pan_y) / (float)oh;
        SDL_PushGPUVertexUniformData(cmd, 0, &vu, sizeof vu);
        mc_gpu_frag_ubo fu = { gain, bias, 0.f, 0.f };
        SDL_PushGPUFragmentUniformData(cmd, 0, &fu, sizeof fu);
        SDL_DrawGPUPrimitives(rp, 6, 1, 0, 0);
    }

    /* ---- nuklear ---- */
    mc_gpu_draw_nuklear(g, cmd, rp, vbytes, ibytes, ow, oh);
    (void)uverts;

    SDL_EndGPURenderPass(rp);
    SDL_SubmitGPUCommandBuffer(cmd);

    nk_buffer_free(&vbuf); nk_buffer_free(&ebuf);
    nk_clear(&g->nk); nk_buffer_clear(&g->cmds);
    return true;
}

SDL_GPUDevice *mc_gpu_device(mc_gpu *g) { return g->dev; }
const char    *mc_gpu_driver(mc_gpu *g) { return g->dev ? SDL_GetGPUDeviceDriver(g->dev) : "none"; }

bool mc_gpu_frame_begin(mc_gpu *g, SDL_GPUCommandBuffer **out_cmd,
                        SDL_GPUTexture **out_swap, Uint32 *ow, Uint32 *oh) {
    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(g->dev);
    if (!cmd) return false;
    // Convert + upload Nuklear geometry BEFORE any render pass (copy passes
    // can't be nested in a render pass). Stash buffers for frame_end.
    mc_gpu_ui_convert(g, cmd, &g->fr_vbuf, &g->fr_ebuf, &g->fr_vbytes, &g->fr_ibytes);
    SDL_GPUTexture *swap = NULL; Uint32 w = 0, h = 0;
    if (!SDL_WaitAndAcquireGPUSwapchainTexture(cmd, g->win, &swap, &w, &h) || !swap) {
        SDL_SubmitGPUCommandBuffer(cmd);
        nk_buffer_free(&g->fr_vbuf); nk_buffer_free(&g->fr_ebuf);
        nk_clear(&g->nk); nk_buffer_clear(&g->cmds);
        return false;
    }
    *out_cmd = cmd; *out_swap = swap; *ow = w; *oh = h;
    return true;
}

void mc_gpu_frame_end_nuklear(mc_gpu *g, SDL_GPUCommandBuffer *cmd,
                              SDL_GPUTexture *swap, Uint32 ow, Uint32 oh) {
    // LOAD (don't clear) so the slice already drawn into `swap` is preserved.
    SDL_GPUColorTargetInfo cti = {0};
    cti.texture = swap; cti.load_op = SDL_GPU_LOADOP_LOAD; cti.store_op = SDL_GPU_STOREOP_STORE;
    SDL_GPURenderPass *rp = SDL_BeginGPURenderPass(cmd, &cti, 1, NULL);
    mc_gpu_draw_nuklear(g, cmd, rp, g->fr_vbytes, g->fr_ibytes, ow, oh);
    SDL_EndGPURenderPass(rp);
    SDL_SubmitGPUCommandBuffer(cmd);
    nk_buffer_free(&g->fr_vbuf); nk_buffer_free(&g->fr_ebuf);
    nk_clear(&g->nk); nk_buffer_clear(&g->cmds);
}

#endif /* MC_GPU_IMPLEMENTATION */
