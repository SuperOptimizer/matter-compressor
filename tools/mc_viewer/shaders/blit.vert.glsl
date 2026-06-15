#version 450
// Fullscreen-quad vertex shader for mc_viewer's SDL_GPU blit pipeline.
//
// No vertex buffer: the quad is generated from gl_VertexIndex (6 verts, two
// triangles). A vertex-stage uniform (SPIR-V set 2, binding 0 per SDL_GPU's
// resource convention) carries the on-screen placement so zoom/pan happen on
// the GPU: `scale` is the quad half-extent in NDC, `offset` shifts its center.

layout(location = 0) out vec2 v_uv;

layout(set = 1, binding = 0) uniform VertUBO {
    vec2 scale;    // half-extent of the quad in NDC (x,y)
    vec2 offset;   // center offset in NDC (x,y)
} u;

// CCW quad as two triangles, in [0,1] then mapped to [-1,1]*scale + offset.
const vec2 corners[6] = vec2[6](
    vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
    vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0)
);

void main() {
    vec2 c = corners[gl_VertexIndex];
    v_uv = c;                               // (0,0) top-left .. (1,1) bottom-right
    vec2 ndc = (c * 2.0 - 1.0) * u.scale + u.offset;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
