#version 450
// Vertex shader for GPU-side volume slice sampling. Same fullscreen-quad
// placement as the blit shader (zoom/pan via the vertex uniform), but it passes
// a [0,1]^2 in-plane coordinate to the fragment stage, which turns it into a 3D
// sample position inside the uploaded slab texture.
//
// SDL_GPU SPIR-V graphics convention: vertex uniform buffer -> set 1, binding 0.

layout(location = 0) out vec2 v_uv;

layout(set = 1, binding = 0) uniform VolVertUBO {
    vec2 scale;    // quad half-extent in NDC
    vec2 offset;   // center offset in NDC
} u;

const vec2 corners[6] = vec2[6](
    vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
    vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0)
);

void main() {
    vec2 c = corners[gl_VertexIndex];
    v_uv = c;
    vec2 ndc = (c * 2.0 - 1.0) * u.scale + u.offset;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
