#version 450
// Brick-cache volume raycaster (M5). The volume is a grid of 16^3 bricks; only
// resident bricks live in a 3D-texture ATLAS (a grid of brick slots). A PAGE
// TABLE (usampler3D, one texel per brick) maps brick coord -> atlas slot+1
// (0 = not resident / air -> skipped). Per ray step: brick coord -> page-table
// lookup -> atlas slot -> sample within the slot. Empty bricks are skipped, so
// air costs nothing (empty-space skipping). This is how out-of-core volumes far
// larger than VRAM render: page in only the visible bricks.
//
// MIP and emission-absorption (with gradient lighting) modes, mirroring
// raycast.frag. Trilinear within a brick; brick boundaries fall back to nearest
// at the 1-voxel seam (acceptable — bricks are 16^3, seams are rare per ray).
//
// SDL_GPU SPIR-V graphics fragment: sampled textures set 2, uniforms set 3.

layout(location = 0) in vec2 v_ndc;
layout(location = 0) out vec4 o_color;

layout(set = 2, binding = 0) uniform sampler3D  u_atlas;   // brick atlas (r8 unorm)
layout(set = 2, binding = 1) uniform sampler2D  u_lut;     // 256x1 transfer fn
layout(set = 2, binding = 2) uniform usampler3D u_page;    // page table (r32ui), per brick

layout(set = 3, binding = 0) uniform BrickUBO {
    mat4 inv_view_proj;
    vec4 vol_dim;       // volume voxel extent (x,y,z), w = step (voxels)
    vec4 bgrid;         // brick-grid dims (x,y,z), w = atlas slots-per-axis
    vec4 params;        // mode (0=MIP,1=EA), gain, alpha_min, absorption
    vec4 light;         // dir xyz; w = lighting on
    vec4 lparams;       // ambient, diffuse, specular, shininess
    vec4 lparams2;      // grad_g0
} u;

bool intersect_box(vec3 ro, vec3 rd, vec3 bmax, out float tmin, out float tmax){
    vec3 inv=1.0/rd; vec3 t0=(vec3(0.0)-ro)*inv, t1=(bmax-ro)*inv;
    vec3 lo=min(t0,t1), hi=max(t0,t1);
    tmin=max(max(lo.x,lo.y),max(lo.z,0.0)); tmax=min(min(hi.x,hi.y),hi.z);
    return tmax>tmin;
}

// Sample the volume at voxel-space position p via the page table + atlas.
// Returns value in [0,1], or a negative sentinel if the brick is not resident.
float sample_vol(vec3 p){
    ivec3 brick = ivec3(floor(p / 16.0));
    ivec3 bg = ivec3(u.bgrid.xyz);
    if (any(lessThan(brick, ivec3(0))) || any(greaterThanEqual(brick, bg))) return -1.0;
    uint slot1 = texelFetch(u_page, brick, 0).r;     // slot+1, 0 = not resident
    if (slot1 == 0u) return -1.0;
    uint slot = slot1 - 1u;
    int aps = int(u.bgrid.w);                        // atlas slots per axis
    ivec3 sc = ivec3(int(slot) % aps, (int(slot)/aps) % aps, int(slot)/(aps*aps));
    // position within the brick [0,16), atlas texel = slot origin + local.
    vec3 local = p - vec3(brick) * 16.0;
    vec3 atexel = vec3(sc) * 16.0 + local;
    vec3 auvw = (atexel + 0.5) / (float(aps) * 16.0);
    return texture(u_atlas, auvw).r;
}

void main(){
    vec4 pn=u.inv_view_proj*vec4(v_ndc,-1.0,1.0);
    vec4 pf=u.inv_view_proj*vec4(v_ndc, 1.0,1.0);
    vec3 wn=pn.xyz/pn.w, wf=pf.xyz/pf.w;
    vec3 ro=wn, rd=normalize(wf-wn);
    vec3 dim=u.vol_dim.xyz;
    float tmin,tmax;
    if(!intersect_box(ro,rd,dim,tmin,tmax)){ o_color=vec4(0.04,0.04,0.05,1.0); return; }

    float dt=max(u.vol_dim.w,0.25);
    float gain=max(u.params.y,1.0);
    int mode=int(u.params.x+0.5);

    if(mode==0){
        float maxv=0.0;
        for(float t=tmin;t<=tmax;t+=dt){
            float s=sample_vol(ro+rd*t);
            if(s>=0.0) maxv=max(maxv,s);
        }
        maxv=clamp(maxv*gain,0.0,1.0);
        o_color=texture(u_lut,vec2(maxv*255.0/256.0+0.5/256.0,0.5));
        return;
    }

    float alpha_min=clamp(u.params.z,0.0,0.99);
    float absorption=max(u.params.w,0.0);
    bool lit=u.light.w>0.5;
    float ka=u.lparams.x,kd=u.lparams.y,ks=u.lparams.z,sh=max(u.lparams.w,1.0);
    float g0=max(u.lparams2.x,1e-3);
    vec3 L=(dot(u.light.xyz,u.light.xyz)>1e-6)?normalize(u.light.xyz):-rd;
    vec3 V=-rd, H=normalize(L+V);

    vec3 Crgb=vec3(0.0); float A=0.0;
    for(float t=tmin;t<=tmax && A<0.99;t+=dt){
        vec3 p=ro+rd*t;
        float s=sample_vol(p);
        if(s<0.0) continue;                       // brick not resident -> skip
        float v=clamp(s*gain,0.0,1.0);
        if(v<=alpha_min) continue;
        vec4 tf=texture(u_lut,vec2(v*255.0/256.0+0.5/256.0,0.5));
        vec3 rgb=tf.rgb;
        if(lit){
            float gx=sample_vol(p+vec3(1,0,0)) - sample_vol(p-vec3(1,0,0));
            float gy=sample_vol(p+vec3(0,1,0)) - sample_vol(p-vec3(0,1,0));
            float gz=sample_vol(p+vec3(0,0,1)) - sample_vol(p-vec3(0,0,1));
            // a -1 sentinel from a missing neighbor would corrupt the gradient;
            // clamp negatives to the center sample so seams stay stable.
            gx=max(gx,-1.0); gy=max(gy,-1.0); gz=max(gz,-1.0);
            vec3 g=vec3(gx,gy,gz)*(255.0*0.5);
            float gm2=dot(g,g);
            float surf=gm2/(gm2+g0*g0);
            vec3 N=(gm2>1e-8)?g*inversesqrt(gm2):vec3(0.0);
            float ndl=abs(dot(N,L));
            float spec=(gm2>1e-8)?pow(max(dot(N,H),0.0),sh):0.0;
            rgb*=mix(1.0, ka+kd*ndl+ks*spec, surf);
        }
        float d=(v-alpha_min)/(1.0-alpha_min)*tf.a;
        float a=1.0-exp(-absorption*d*dt);
        Crgb+=(1.0-A)*a*rgb; A+=(1.0-A)*a;
    }
    o_color=vec4(Crgb+(1.0-A)*vec3(0.04,0.04,0.05),1.0);
}
