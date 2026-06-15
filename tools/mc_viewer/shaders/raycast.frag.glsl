#version 450
// Volume raycaster (Milestone 2): per-pixel ray through the dense decoded 3D
// texture, MIP composite (max value along the ray), mapped through the LUT.
// Emission-absorption + lighting come in later milestones; MIP first because it
// is deterministic (no transfer-function ambiguity) and easy to check vs CPU.
//
// SDL_GPU SPIR-V graphics fragment: sampled textures set 2, uniforms set 3.

layout(location = 0) in vec2 v_ndc;
layout(location = 0) out vec4 o_color;

layout(set = 2, binding = 0) uniform sampler3D u_vol;   // dense slab, r8 unorm
layout(set = 2, binding = 1) uniform sampler2D u_lut;   // 256x1 colormap

layout(set = 3, binding = 0) uniform RayUBO {
    mat4 inv_view_proj;   // clip -> world(volume) space
    vec4 cam_pos;         // ray origin (volume space), .w unused
    vec4 vol_dim;         // volume extent in voxels (x,y,z), .w = step (voxels)
    vec4 params;          // x = mode (0=MIP, 1=emission-absorption), y = gain,
                          // z = alpha_min, w = absorption
} u;

// ray vs axis-aligned box [0, vol_dim]; returns (tmin,tmax), tmin<tmax if hit.
bool intersect_box(vec3 ro, vec3 rd, vec3 bmax, out float tmin, out float tmax) {
    vec3 inv = 1.0 / rd;
    vec3 t0 = (vec3(0.0) - ro) * inv;
    vec3 t1 = (bmax - ro) * inv;
    vec3 lo = min(t0, t1), hi = max(t0, t1);
    tmin = max(max(lo.x, lo.y), max(lo.z, 0.0));
    tmax = min(min(hi.x, hi.y), hi.z);
    return tmax > tmin;
}

void main() {
    // unproject two clip points to build the world-space ray.
    vec4 pn = u.inv_view_proj * vec4(v_ndc, -1.0, 1.0);  // near plane point
    vec4 pf = u.inv_view_proj * vec4(v_ndc,  1.0, 1.0);  // far plane point
    vec3 wn = pn.xyz / pn.w, wf = pf.xyz / pf.w;
    // ray origin is the near-plane unprojected point (works for ortho and
    // perspective); cam_pos is unused for direction.
    vec3 ro = wn;
    vec3 rd = normalize(wf - wn);

    vec3 dim = u.vol_dim.xyz;
    float tmin, tmax;
    if (!intersect_box(ro, rd, dim, tmin, tmax)) {
        o_color = vec4(0.04, 0.04, 0.05, 1.0);   // miss -> background
        return;
    }

    float dt   = max(u.vol_dim.w, 0.25);
    float gain = max(u.params.y, 1.0);
    int   mode = int(u.params.x + 0.5);

    if (mode == 0) {
        // ---- MIP: max sample along the ray ----
        float maxv = 0.0;
        for (float t = tmin; t <= tmax; t += dt) {
            vec3 uvw = (ro + rd * t + 0.5) / dim;
            maxv = max(maxv, texture(u_vol, uvw).r);
        }
        maxv = clamp(maxv * gain, 0.0, 1.0);
        o_color = texture(u_lut, vec2(maxv * 255.0/256.0 + 0.5/256.0, 0.5));
        return;
    }

    // ---- emission-absorption (VTK-style direct volume rendering) ----
    // Per step: look up the transfer function (u_lut: rgb = color, a = base
    // opacity for that value). density d = clamp((v - alpha_min)/(1-alpha_min))
    // * tf.a; Beer-Lambert opacity a = 1 - exp(-absorption * d * dt). Composite
    // front-to-back with early-ray-termination. Mirrors the CPU MC_COMP_SHADED
    // emission-absorption core (lighting added in M4).
    float alpha_min  = clamp(u.params.z, 0.0, 0.99);
    float absorption = max(u.params.w, 0.0);
    vec3  Crgb = vec3(0.0);
    float A    = 0.0;
    for (float t = tmin; t <= tmax && A < 0.99; t += dt) {
        vec3 uvw = (ro + rd * t + 0.5) / dim;
        float v = clamp(texture(u_vol, uvw).r * gain, 0.0, 1.0);
        if (v <= alpha_min) continue;
        vec4 tf = texture(u_lut, vec2(v * 255.0/256.0 + 0.5/256.0, 0.5));
        float d = (v - alpha_min) / (1.0 - alpha_min) * tf.a;
        float a = 1.0 - exp(-absorption * d * dt);
        Crgb += (1.0 - A) * a * tf.rgb;
        A    += (1.0 - A) * a;
    }
    // composite over the background.
    vec3 bg = vec3(0.04, 0.04, 0.05);
    o_color = vec4(Crgb + (1.0 - A) * bg, 1.0);
}
