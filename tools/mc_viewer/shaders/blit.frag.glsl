#version 450
// Fragment shader for mc_viewer's SDL_GPU blit pipeline.
//
// Samples the colormapped slice texture. SDL_GPU SPIR-V graphics convention:
//   fragment sampled texture -> descriptor set 2, binding 0
//   fragment uniform buffer  -> descriptor set 3, binding 0
//
// The slice is already windowed + colormapped on the CPU (mc_colormap_apply),
// so this shader just samples it. The fragment uniform is reserved for a future
// GPU-side window/level pass (gain/bias) without re-uploading the texture.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

layout(set = 2, binding = 0) uniform sampler2D u_slice;

layout(set = 3, binding = 0) uniform FragUBO {
    float gain;     // multiply (1.0 = identity)
    float bias;     // add      (0.0 = identity)
    float _pad0;
    float _pad1;
} u;

void main() {
    vec4 c = texture(u_slice, v_uv);
    o_color = vec4(clamp(c.rgb * u.gain + u.bias, 0.0, 1.0), 1.0);
}
