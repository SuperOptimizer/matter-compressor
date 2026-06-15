#version 450
// Sample an axis-aligned slice directly from the GPU-decoded block buffer.
//
// The compute decoder (c3g_decode.comp) wrote the visible slab's blocks into a
// storage buffer, block-major: block b occupies [b*4096, b*4096+4096) u8 voxels
// in (z,y,x) raster within its 16^3. A small "block grid" uniform describes the
// slab (origin block coords + grid dims), so this shader maps a slice pixel to
// (volume z,y,x) -> (block index in slab, in-block offset) -> u8 -> colormap.
//
// Nearest sampling for now (matches the decoded-block granularity); the slab is
// always >= the displayed slice resolution. SDL_GPU SPIR-V graphics convention:
// fragment storage buffer -> set 2 (after sampled textures); we use a storage
// buffer for the voxels and a sampled texture for the colormap LUT.

layout(location = 0) in vec2 v_uv;             // [0,1] across the slice
layout(location = 0) out vec4 o_color;

// SDL_GPU SPIR-V graphics fragment set 2: sampled textures THEN storage buffers
// (sequential bindings). LUT sampler at binding 0, voxel storage buffer at
// binding 1. Fragment uniform -> set 3.
layout(set = 2, binding = 0) uniform sampler2D u_lut;                       // 256x1 colormap
layout(std430, set = 2, binding = 1) readonly buffer Vox { uint vox[]; };   // u8 packed 4/uint

layout(set = 3, binding = 0) uniform SliceUBO {
    ivec4 origin;     // slab origin in voxels (z,y,x, _)
    ivec4 gdim;       // slab block-grid dims (gz,gy,gx, _)
    ivec4 extent;     // slab voxel extent (ez,ey,ex, _)
    ivec4 axis;       // axis (0=Z,1=Y,2=X), slice index within slab (voxels), _, _
} u;

uint load_vox(int gz,int gy,int gx,int lz,int ly,int lx){
    // gdim is packed (gz,gy,gx) in (.x,.y,.z); the upload's block-major order is
    // (gz*GY + gy)*GX + gx, so use GY=gdim.y, GX=gdim.z.
    int GY=u.gdim.y, GX=u.gdim.z;
    int b = (gz*GY + gy)*GX + gx;
    int off = b*4096 + ((lz<<8)|(ly<<4)|lx);
    uint w = vox[off>>2];
    return (w >> ((uint(off)&3u)*8u)) & 0xFFu;
}

void main() {
    // map (u,v) -> in-plane voxel coords within the slab extent.
    // The render target is sampled top-down on readback but the quad rasterizes
    // bottom-up (Vulkan/SDL_GPU NDC), so flip v to keep image-y -> +voxel.
    float uu = v_uv.x, vv = v_uv.y;
    int ez=u.extent.x, ey=u.extent.y, ex=u.extent.z;
    int sidx=u.axis.y;     // slice voxel index within the slab along the axis
    int vz,vy,vx;
    if (u.axis.x == 0) {              // Z slice: u->x, v->y
        vx = int(uu * float(ex)); vy = int(vv * float(ey)); vz = sidx;
    } else if (u.axis.x == 1) {       // Y slice: u->x, v->z
        vx = int(uu * float(ex)); vz = int(vv * float(ez)); vy = sidx;
    } else {                          // X slice: u->y, v->z
        vy = int(uu * float(ey)); vz = int(vv * float(ez)); vx = sidx;
    }
    if (vz<0||vy<0||vx<0||vz>=ez||vy>=ey||vx>=ex) { o_color=vec4(0.04,0.04,0.05,1.0); return; }

    int gz=vz>>4, gy=vy>>4, gx=vx>>4;
    int lz=vz&15, ly=vy&15, lx=vx&15;
    uint val = load_vox(gz,gy,gx,lz,ly,lx);
    o_color = texture(u_lut, vec2((float(val)+0.5)/256.0, 0.5));
}
