#version 450
// Fullscreen triangle for the volume raycaster. Emits clip-space [-1,1] coords
// and passes NDC to the fragment shader, which unprojects to build a ray.
layout(location = 0) out vec2 v_ndc;
void main() {
    // 3-vertex fullscreen triangle (covers the screen, no vertex buffer).
    vec2 p = vec2((gl_VertexIndex == 1) ? 3.0 : -1.0,
                  (gl_VertexIndex == 2) ? 3.0 : -1.0);
    v_ndc = p;
    gl_Position = vec4(p, 0.0, 1.0);
}
