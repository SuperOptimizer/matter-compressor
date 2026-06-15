/* mc_gpu_vol.h — GPU c3g decode + slice sampling for mc_viewer.
 *
 * The "CPU sends compressed, GPU does the rest" path: the viewer gathers the
 * visible slab's COMPRESSED c3g block blobs (read straight from the local .mca
 * via mc_archive_block_blob), uploads them to a VRAM cache, and each frame the
 * GPU:
 *   1. compute-decodes the cached compressed blocks into a decoded-block buffer
 *      (c3g_decode.comp — one workgroup per block), then
 *   2. samples that buffer at the current slice plane in a fragment shader
 *      (vol_slice.frag) with a colormap LUT, drawing the slice.
 *
 * Only step 1 ever decodes; the compressed blocks (small) stay resident in
 * VRAM, re-decoded as the view moves through the slab. Constant tables (rANS
 * freq/cdf/slot, dequant step_tab, scan order, the inverse-DCT cosine basis)
 * upload once. The CPU's only job per slab is reading compressed bytes off disk.
 *
 * Shares an existing SDL_GPUDevice (created by mc_gpu). Define
 * MC_GPU_VOL_IMPLEMENTATION in one TU.
 */
#ifndef MC_GPU_VOL_H_
#define MC_GPU_VOL_H_

#include <SDL3/SDL.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct mc_gpu_vol mc_gpu_vol;

// Create against an existing device + the swapchain color format.
mc_gpu_vol *mc_gpu_vol_create(SDL_GPUDevice *dev, SDL_GPUTextureFormat swfmt, float quality);
void        mc_gpu_vol_destroy(mc_gpu_vol *v);

// Upload one slab. `blobs[i]`/`lens[i]` are the compressed c3g payload of block
// i (NULL/0 = absent/air), in (gz,gy,gx) raster over the gz*gy*gx grid. The slab
// covers `extent` voxels (z,y,x). Packs blobs + a [off,len] table into VRAM.
// Returns false if the slab exceeds the VRAM budget.
bool mc_gpu_vol_upload_slab(mc_gpu_vol *v, int gz,int gy,int gx,
                            int ez,int ey,int ex,
                            const uint8_t *const *blobs, const uint32_t *lens);

// Set the colormap LUT (256 ARGB8888 entries) used when sampling.
void mc_gpu_vol_set_lut(mc_gpu_vol *v, const uint32_t lut[256]);

// Render the current slab's slice into `target` (the acquired swapchain texture)
// using command buffer `cmd` inside no active pass: this opens its own compute
// pass (decode) then render pass (sample+draw). axis 0=Z,1=Y,2=X; `slice` is the
// voxel index within the slab along that axis; draw_w/h + pan position the quad;
// out_w/out_h are the drawable size.
void mc_gpu_vol_render(mc_gpu_vol *v, SDL_GPUCommandBuffer *cmd, SDL_GPUTexture *target,
                       int axis, int slice, float draw_w, float draw_h,
                       float pan_x, float pan_y, int out_w, int out_h);

#endif /* MC_GPU_VOL_H_ */

#ifdef MC_GPU_VOL_IMPLEMENTATION
#undef MC_GPU_VOL_IMPLEMENTATION

#include <math.h>
#include "shaders/c3g_decode.comp.spv.h"
#include "shaders/vol.vert.spv.h"
#include "shaders/vol_slice.frag.spv.h"

#define MCV_N3      4096
#define MCV_MAXSYM  16
#define MCV_TOTAL   4096
// VRAM budget for the packed compressed slab + decoded output.
#define MCV_MAX_BLOCKS 4096          // up to a 16^3-block slab
#define MCV_PAYLOAD_CAP (8u<<20)     // 8 MB packed compressed blocks

// matches src C3G_FREQ
static const uint16_t MCV_FREQ[MCV_MAXSYM] =
    { 2200,980,360,180,110,70,48,34,24,18,14,10,8,6,5,21 };

typedef struct { int32_t origin[4], gdim[4], extent[4], axis[4]; } mcv_slice_ubo;
typedef struct { float scale[2], offset[2]; } mcv_vert_ubo;

struct mc_gpu_vol {
    SDL_GPUDevice *dev;
    float quality;

    SDL_GPUComputePipeline  *decode_pipe;
    SDL_GPUGraphicsPipeline *slice_pipe;
    SDL_GPUSampler          *lut_samp;

    // constant table buffers (uploaded once)
    SDL_GPUBuffer *b_freq, *b_cdf, *b_slot, *b_step, *b_scan, *b_dct;
    // per-slab buffers
    SDL_GPUBuffer *b_payload;    // packed compressed blocks
    SDL_GPUBuffer *b_blktab;     // [off,len] per block
    SDL_GPUBuffer *b_decoded;    // decoded u8 voxels, block-major (rw + frag-read)
    SDL_GPUTexture *lut_tex;     // 256x1 colormap

    int nblocks;                 // blocks in the current slab
    int gz,gy,gx, ez,ey,ex;      // current slab geometry
};

static SDL_GPUBuffer *mcv_storage_buf(SDL_GPUDevice *dev, uint32_t size, SDL_GPUBufferUsageFlags extra){
    SDL_GPUBufferCreateInfo bc = {0};
    bc.usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | extra;
    bc.size = size;
    return SDL_CreateGPUBuffer(dev, &bc);
}
static void mcv_upload(SDL_GPUDevice *dev, SDL_GPUBuffer *buf, const void *data, uint32_t size){
    SDL_GPUTransferBufferCreateInfo tc = {0};
    tc.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD; tc.size = size;
    SDL_GPUTransferBuffer *tb = SDL_CreateGPUTransferBuffer(dev, &tc);
    void *m = SDL_MapGPUTransferBuffer(dev, tb, false);
    SDL_memcpy(m, data, size);
    SDL_UnmapGPUTransferBuffer(dev, tb);
    SDL_GPUCommandBuffer *cb = SDL_AcquireGPUCommandBuffer(dev);
    SDL_GPUCopyPass *cp = SDL_BeginGPUCopyPass(cb);
    SDL_GPUTransferBufferLocation s = {0}; s.transfer_buffer = tb;
    SDL_GPUBufferRegion d = {0}; d.buffer = buf; d.size = size;
    SDL_UploadToGPUBuffer(cp, &s, &d, false);
    SDL_EndGPUCopyPass(cp);
    SDL_SubmitGPUCommandBuffer(cb);
    SDL_ReleaseGPUTransferBuffer(dev, tb);
}

mc_gpu_vol *mc_gpu_vol_create(SDL_GPUDevice *dev, SDL_GPUTextureFormat swfmt, float quality){
    mc_gpu_vol *v = SDL_calloc(1, sizeof *v);
    if(!v) return NULL;
    v->dev = dev; v->quality = quality;

    // --- compute decode pipeline ---
    SDL_GPUComputePipelineCreateInfo pc = {0};
    pc.code = (const Uint8*)c3g_decode_comp_spv; pc.code_size = sizeof c3g_decode_comp_spv;
    pc.entrypoint = "main"; pc.format = SDL_GPU_SHADERFORMAT_SPIRV;
    pc.num_readonly_storage_buffers = 8; pc.num_readwrite_storage_buffers = 1;
    pc.num_uniform_buffers = 1; pc.threadcount_x = 256; pc.threadcount_y = 1; pc.threadcount_z = 1;
    v->decode_pipe = SDL_CreateGPUComputePipeline(dev, &pc);
    if(!v->decode_pipe) { mc_gpu_vol_destroy(v); return NULL; }

    // --- slice graphics pipeline (fullscreen quad, samples decoded buffer + LUT) ---
    {
        SDL_GPUShaderCreateInfo vs = {0};
        vs.code=(const Uint8*)vol_vert_spv; vs.code_size=sizeof vol_vert_spv;
        vs.entrypoint="main"; vs.format=SDL_GPU_SHADERFORMAT_SPIRV;
        vs.stage=SDL_GPU_SHADERSTAGE_VERTEX; vs.num_uniform_buffers=1;
        SDL_GPUShader *vsh = SDL_CreateGPUShader(dev, &vs);
        SDL_GPUShaderCreateInfo fs = {0};
        fs.code=(const Uint8*)vol_slice_frag_spv; fs.code_size=sizeof vol_slice_frag_spv;
        fs.entrypoint="main"; fs.format=SDL_GPU_SHADERFORMAT_SPIRV;
        fs.stage=SDL_GPU_SHADERSTAGE_FRAGMENT;
        fs.num_samplers=1; fs.num_storage_buffers=1; fs.num_uniform_buffers=1;
        SDL_GPUShader *fsh = SDL_CreateGPUShader(dev, &fs);
        if(!vsh||!fsh){ if(vsh)SDL_ReleaseGPUShader(dev,vsh); if(fsh)SDL_ReleaseGPUShader(dev,fsh);
            mc_gpu_vol_destroy(v); return NULL; }
        SDL_GPUColorTargetDescription ct = {0}; ct.format = swfmt;
        SDL_GPUGraphicsPipelineCreateInfo p = {0};
        p.vertex_shader=vsh; p.fragment_shader=fsh;
        p.primitive_type=SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
        p.target_info.color_target_descriptions=&ct; p.target_info.num_color_targets=1;
        v->slice_pipe = SDL_CreateGPUGraphicsPipeline(dev, &p);
        SDL_ReleaseGPUShader(dev,vsh); SDL_ReleaseGPUShader(dev,fsh);
        if(!v->slice_pipe){ mc_gpu_vol_destroy(v); return NULL; }
    }

    // --- constant tables ---
    uint32_t freq[MCV_MAXSYM], cdf[MCV_MAXSYM+1], slot[MCV_TOTAL];
    uint32_t acc=0; for(int s=0;s<MCV_MAXSYM;++s){ freq[s]=MCV_FREQ[s]; cdf[s]=acc; acc+=MCV_FREQ[s]; } cdf[MCV_MAXSYM]=acc;
    for(int s=0;s<MCV_MAXSYM;++s) for(uint32_t f=cdf[s]; f<cdf[s+1]; ++f) slot[f]=(uint32_t)s;
    float step[MCV_N3];
    for(int cz=0;cz<16;++cz)for(int cy=0;cy<16;++cy)for(int cx=0;cx<16;++cx)
        step[(cz*16+cy)*16+cx]=quality*powf(1.0f+(float)(cz+cy+cx), MC_HF_EXP);
    // scan: ascending L1 frequency, stable (matches scanS_build for edge 16)
    uint32_t scan[MCV_N3]; for(int i=0;i<MCV_N3;++i) scan[i]=i;
    for(int i=1;i<MCV_N3;++i){ uint32_t k=scan[i]; int kl=(k/256)+((k/16)%16)+(k%16);
        int j=i-1; while(j>=0){ uint32_t s=scan[j]; int sl=(s/256)+((s/16)%16)+(s%16);
            if(sl>kl||(sl==kl&&s>k)){ scan[j+1]=s; j--; } else break; } scan[j+1]=k; }
    float cm[16*16];
    for(int k=0;k<16;++k){ double ck=(k==0)?sqrt(1.0/16):sqrt(2.0/16);
        for(int n=0;n<16;++n) cm[k*16+n]=(float)(ck*cos(M_PI*(2.0*n+1.0)*k/32.0)); }

    v->b_freq=mcv_storage_buf(dev,sizeof freq,0); mcv_upload(dev,v->b_freq,freq,sizeof freq);
    v->b_cdf =mcv_storage_buf(dev,sizeof cdf,0);  mcv_upload(dev,v->b_cdf,cdf,sizeof cdf);
    v->b_slot=mcv_storage_buf(dev,sizeof slot,0); mcv_upload(dev,v->b_slot,slot,sizeof slot);
    v->b_step=mcv_storage_buf(dev,sizeof step,0); mcv_upload(dev,v->b_step,step,sizeof step);
    v->b_scan=mcv_storage_buf(dev,sizeof scan,0); mcv_upload(dev,v->b_scan,scan,sizeof scan);
    v->b_dct =mcv_storage_buf(dev,sizeof cm,0);   mcv_upload(dev,v->b_dct,cm,sizeof cm);

    // per-slab buffers (max-sized; reused)
    v->b_payload = mcv_storage_buf(dev, MCV_PAYLOAD_CAP, 0);
    v->b_blktab  = mcv_storage_buf(dev, MCV_MAX_BLOCKS*2*4, 0);
    v->b_decoded = mcv_storage_buf(dev, MCV_MAX_BLOCKS*MCV_N3, SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE | SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ);

    // LUT texture (256x1 BGRA)
    SDL_GPUTextureCreateInfo tci = {0};
    tci.type=SDL_GPU_TEXTURETYPE_2D; tci.format=SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM;
    tci.usage=SDL_GPU_TEXTUREUSAGE_SAMPLER; tci.width=256; tci.height=1;
    tci.layer_count_or_depth=1; tci.num_levels=1;
    v->lut_tex = SDL_CreateGPUTexture(dev, &tci);
    SDL_GPUSamplerCreateInfo si = {0};
    si.address_mode_u=si.address_mode_v=si.address_mode_w=SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    si.min_filter=si.mag_filter=SDL_GPU_FILTER_NEAREST; si.mipmap_mode=SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    v->lut_samp = SDL_CreateGPUSampler(dev, &si);

    return v;
}

void mc_gpu_vol_destroy(mc_gpu_vol *v){
    if(!v) return;
    SDL_GPUDevice *d=v->dev;
    SDL_GPUBuffer *bufs[]={v->b_freq,v->b_cdf,v->b_slot,v->b_step,v->b_scan,v->b_dct,v->b_payload,v->b_blktab,v->b_decoded};
    for(int i=0;i<9;++i) if(bufs[i]) SDL_ReleaseGPUBuffer(d,bufs[i]);
    if(v->lut_tex) SDL_ReleaseGPUTexture(d,v->lut_tex);
    if(v->lut_samp) SDL_ReleaseGPUSampler(d,v->lut_samp);
    if(v->decode_pipe) SDL_ReleaseGPUComputePipeline(d,v->decode_pipe);
    if(v->slice_pipe) SDL_ReleaseGPUGraphicsPipeline(d,v->slice_pipe);
    SDL_free(v);
}

void mc_gpu_vol_set_lut(mc_gpu_vol *v, const uint32_t lut[256]){
    SDL_GPUTransferBufferCreateInfo tc = {0};
    tc.usage=SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD; tc.size=256*4;
    SDL_GPUTransferBuffer *tb=SDL_CreateGPUTransferBuffer(v->dev,&tc);
    void *m=SDL_MapGPUTransferBuffer(v->dev,tb,false); SDL_memcpy(m,lut,256*4); SDL_UnmapGPUTransferBuffer(v->dev,tb);
    SDL_GPUCommandBuffer *cb=SDL_AcquireGPUCommandBuffer(v->dev);
    SDL_GPUCopyPass *cp=SDL_BeginGPUCopyPass(cb);
    SDL_GPUTextureTransferInfo ti={0}; ti.transfer_buffer=tb; ti.pixels_per_row=256; ti.rows_per_layer=1;
    SDL_GPUTextureRegion tr={0}; tr.texture=v->lut_tex; tr.w=256; tr.h=1; tr.d=1;
    SDL_UploadToGPUTexture(cp,&ti,&tr,false); SDL_EndGPUCopyPass(cp); SDL_SubmitGPUCommandBuffer(cb);
    SDL_ReleaseGPUTransferBuffer(v->dev,tb);
}

bool mc_gpu_vol_upload_slab(mc_gpu_vol *v, int gz,int gy,int gx, int ez,int ey,int ex,
                            const uint8_t *const *blobs, const uint32_t *lens){
    int nb = gz*gy*gx;
    if(nb<=0 || nb>MCV_MAX_BLOCKS) return false;
    // pack payloads (4-byte aligned per block) + build [off,len] table.
    uint8_t *packed = SDL_malloc(MCV_PAYLOAD_CAP);
    uint32_t *tab = SDL_malloc((size_t)nb*2*4);
    uint32_t off=0;
    for(int i=0;i<nb;++i){
        uint32_t l = blobs[i] ? lens[i] : 0;
        if(off + ((l+3)&~3u) > MCV_PAYLOAD_CAP){ SDL_free(packed); SDL_free(tab); return false; }
        tab[2*i]=off; tab[2*i+1]=l;
        if(l){ SDL_memcpy(packed+off, blobs[i], l); off += (l+3)&~3u; }
    }
    if(off==0) off=4;   // never upload 0 bytes
    mcv_upload(v->dev, v->b_payload, packed, off);
    mcv_upload(v->dev, v->b_blktab, tab, (uint32_t)nb*2*4);
    SDL_free(packed); SDL_free(tab);
    v->nblocks=nb; v->gz=gz; v->gy=gy; v->gx=gx; v->ez=ez; v->ey=ey; v->ex=ex;
    return true;
}

void mc_gpu_vol_render(mc_gpu_vol *v, SDL_GPUCommandBuffer *cmd, SDL_GPUTexture *target,
                       int axis, int slice, float draw_w, float draw_h,
                       float pan_x, float pan_y, int out_w, int out_h){
    if(v->nblocks<=0) return;

    // 1) compute pass: decode the cached compressed blocks -> b_decoded.
    SDL_GPUStorageBufferReadWriteBinding rw={0}; rw.buffer=v->b_decoded;
    SDL_GPUComputePass *cp=SDL_BeginGPUComputePass(cmd,NULL,0,&rw,1);
    SDL_BindGPUComputePipeline(cp,v->decode_pipe);
    SDL_GPUBuffer *ro[8]={v->b_payload,v->b_blktab,v->b_freq,v->b_cdf,v->b_slot,v->b_step,v->b_scan,v->b_dct};
    SDL_BindGPUComputeStorageBuffers(cp,0,ro,8);
    uint32_t nb=(uint32_t)v->nblocks; SDL_PushGPUComputeUniformData(cmd,0,&nb,sizeof nb);
    SDL_DispatchGPUCompute(cp,(Uint32)v->nblocks,1,1);
    SDL_EndGPUComputePass(cp);

    // 2) render pass: sample the decoded slab at the slice plane.
    SDL_GPUColorTargetInfo cti={0};
    cti.texture=target; cti.load_op=SDL_GPU_LOADOP_CLEAR; cti.store_op=SDL_GPU_STOREOP_STORE;
    cti.clear_color=(SDL_FColor){0.06f,0.06f,0.08f,1.0f};
    SDL_GPURenderPass *rp=SDL_BeginGPURenderPass(cmd,&cti,1,NULL);
    SDL_BindGPUGraphicsPipeline(rp,v->slice_pipe);

    SDL_GPUTextureSamplerBinding tsb={0}; tsb.texture=v->lut_tex; tsb.sampler=v->lut_samp;
    SDL_BindGPUFragmentSamplers(rp,0,&tsb,1);
    SDL_BindGPUFragmentStorageBuffers(rp,0,&v->b_decoded,1);

    mcv_vert_ubo vu;
    vu.scale[0]=draw_w/(float)out_w; vu.scale[1]=draw_h/(float)out_h;
    vu.offset[0]=(2.0f*pan_x)/(float)out_w; vu.offset[1]=-(2.0f*pan_y)/(float)out_h;
    SDL_PushGPUVertexUniformData(cmd,0,&vu,sizeof vu);

    mcv_slice_ubo su; SDL_memset(&su,0,sizeof su);
    su.gdim[0]=v->gz; su.gdim[1]=v->gy; su.gdim[2]=v->gx;
    su.extent[0]=v->ez; su.extent[1]=v->ey; su.extent[2]=v->ex;
    su.axis[0]=axis; su.axis[1]=slice;
    SDL_PushGPUFragmentUniformData(cmd,0,&su,sizeof su);

    SDL_DrawGPUPrimitives(rp,6,1,0,0);
    SDL_EndGPURenderPass(rp);
}

#endif /* MC_GPU_VOL_IMPLEMENTATION */
