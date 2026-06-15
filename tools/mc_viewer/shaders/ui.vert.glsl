#version 450
// Nuklear UI vertex shader (SDL_GPU). Transforms screen-space (pixel) vertex
// positions through an orthographic projection uniform into NDC. Passes UV +
// per-vertex color to the fragment stage.
//
// SDL_GPU SPIR-V graphics convention: vertex uniform buffer -> set 1, binding 0.

layout(location = 0) in vec2 a_pos;     // screen pixels
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_col;      // UBYTE4_NORM -> [0,1]

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_col;

layout(set = 1, binding = 0) uniform UiVertUBO {
    mat4 proj;     // ortho(0,w, 0,h) with +Y down
} u;

void main() {
    v_uv = a_uv;
    v_col = a_col;
    gl_Position = u.proj * vec4(a_pos, 0.0, 1.0);
}
