#version 450
// Nuklear UI fragment shader (SDL_GPU). Modulates the per-vertex color by the
// sampled atlas texel (font glyph alpha / white for solid fills).
//
// SDL_GPU SPIR-V graphics convention: fragment sampled texture -> set 2,
// binding 0.

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_col;
layout(location = 0) out vec4 o_color;

layout(set = 2, binding = 0) uniform sampler2D u_atlas;

void main() {
    o_color = v_col * texture(u_atlas, v_uv);
}
