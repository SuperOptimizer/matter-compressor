/* mc_gpu_ray.h — GPU volume raycaster for mc_viewer (Milestones 1-2).
 *
 * Decodes compressed c3g blocks into a DENSE 3D texture (c3g_decode3d.comp),
 * then raycasts that texture from a 3D camera (raycast.vert/frag, MIP for now).
 * Reuses the proven c3g GPU decode; the only new pieces are the 3D-texture
 * output target and the per-pixel ray-march.
 *
 * Decode-into-3D-texture is validated against the CPU oracle by ray_gpu_check
 * (reads the texture back, compares to mc_c3g_dec_block) before any rendering.
 *
 * Shares an existing SDL_GPUDevice. Define MC_GPU_RAY_IMPLEMENTATION in one TU.
 */
#ifndef MC_GPU_RAY_H_
#define MC_GPU_RAY_H_

#include <SDL3/SDL.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct mc_gpu_ray mc_gpu_ray;

mc_gpu_ray *mc_gpu_ray_create(SDL_GPUDevice *dev, SDL_GPUTextureFormat swfmt, float quality);
void        mc_gpu_ray_destroy(mc_gpu_ray *r);

/* (Re)allocate the dense 3D volume texture to gx*16 x gy*16 x gz*16 voxels and
 * decode the slab's compressed c3g blocks into it (blobs[i]/lens[i] in
 * (gz,gy,gx) raster; NULL/0 = absent). Returns false on failure. */
bool mc_gpu_ray_upload(mc_gpu_ray *r, int gz,int gy,int gx,
                       const uint8_t *const *blobs, const uint32_t *lens);

void mc_gpu_ray_set_lut(mc_gpu_ray *r, const uint32_t lut[256]);

/* Raycast the decoded volume into `target` with the given inverse-view-proj
 * (clip->volume), camera position (volume space), and step (voxels). MIP. */
void mc_gpu_ray_render(mc_gpu_ray *r, SDL_GPUCommandBuffer *cmd, SDL_GPUTexture *target,
                       const float inv_view_proj[16], const float cam_pos[3],
                       float step_voxels, float gain);

/* The dense 3D texture (for readback validation). */
SDL_GPUTexture *mc_gpu_ray_volume(mc_gpu_ray *r, int *vx,int *vy,int *vz);

#endif /* MC_GPU_RAY_H_ */

#ifdef MC_GPU_RAY_IMPLEMENTATION
#undef MC_GPU_RAY_IMPLEMENTATION

#include <math.h>
#include "shaders/c3g_decode3d.comp.spv.h"
#include "shaders/raycast.vert.spv.h"
#include "shaders/raycast.frag.spv.h"

#define MCR_N3 4096
#define MCR_MAXSYM 16
#define MCR_TOTAL 4096
#define MCR_MAX_BLOCKS 4096
#define MCR_PAYLOAD_CAP (8u<<20)
static const uint16_t MCR_FREQ[MCR_MAXSYM] =
    { 2200,980,360,180,110,70,48,34,24,18,14,10,8,6,5,21 };

typedef struct { uint32_t nblocks, gz, gy, gx; } mcr_decode_ubo;
typedef struct { float inv_view_proj[16]; float cam_pos[4]; float vol_dim[4]; float params[4]; } mcr_ray_ubo;

struct mc_gpu_ray {
    SDL_GPUDevice *dev;
    float quality;
    SDL_GPUComputePipeline  *decode_pipe;
    SDL_GPUGraphicsPipeline *ray_pipe;
    SDL_GPUSampler          *vol_samp, *lut_samp;
    SDL_GPUBuffer *b_freq,*b_cdf,*b_slot,*b_step,*b_scan,*b_dct;
    SDL_GPUBuffer *b_payload,*b_blktab;
    SDL_GPUTexture *vol;          // dense 3D decoded volume (r8)
    SDL_GPUTexture *lut_tex;      // 256x1 colormap
    int vx,vy,vz;                 // current volume voxel dims
    int gz,gy,gx, nblocks;
};

static SDL_GPUBuffer *mcr_robuf(SDL_GPUDevice *d, uint32_t sz){
    SDL_GPUBufferCreateInfo bc={0}; bc.usage=SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ; bc.size=sz;
    return SDL_CreateGPUBuffer(d,&bc);
}
static void mcr_up(SDL_GPUDevice *d, SDL_GPUBuffer *buf, const void *data, uint32_t sz){
    SDL_GPUTransferBufferCreateInfo tc={0}; tc.usage=SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD; tc.size=sz;
    SDL_GPUTransferBuffer *tb=SDL_CreateGPUTransferBuffer(d,&tc);
    void *m=SDL_MapGPUTransferBuffer(d,tb,false); SDL_memcpy(m,data,sz); SDL_UnmapGPUTransferBuffer(d,tb);
    SDL_GPUCommandBuffer *cb=SDL_AcquireGPUCommandBuffer(d);
    SDL_GPUCopyPass *cp=SDL_BeginGPUCopyPass(cb);
    SDL_GPUTransferBufferLocation s={0}; s.transfer_buffer=tb;
    SDL_GPUBufferRegion dr={0}; dr.buffer=buf; dr.size=sz;
    SDL_UploadToGPUBuffer(cp,&s,&dr,false); SDL_EndGPUCopyPass(cp); SDL_SubmitGPUCommandBuffer(cb);
    SDL_ReleaseGPUTransferBuffer(d,tb);
}

mc_gpu_ray *mc_gpu_ray_create(SDL_GPUDevice *dev, SDL_GPUTextureFormat swfmt, float quality){
    mc_gpu_ray *r=SDL_calloc(1,sizeof *r);
    if(!r) return NULL;
    r->dev=dev; r->quality=quality;

    // compute decode (3D-texture output): 8 ro buffers, 1 rw storage texture.
    SDL_GPUComputePipelineCreateInfo pc={0};
    pc.code=(const Uint8*)c3g_decode3d_comp_spv; pc.code_size=sizeof c3g_decode3d_comp_spv;
    pc.entrypoint="main"; pc.format=SDL_GPU_SHADERFORMAT_SPIRV;
    pc.num_readonly_storage_buffers=8; pc.num_readwrite_storage_textures=1;
    pc.num_uniform_buffers=1; pc.threadcount_x=256; pc.threadcount_y=1; pc.threadcount_z=1;
    r->decode_pipe=SDL_CreateGPUComputePipeline(dev,&pc);
    if(!r->decode_pipe){ mc_gpu_ray_destroy(r); return NULL; }

    // raycast graphics pipeline (fullscreen tri; 2 samplers, 1 frag UBO).
    {
        SDL_GPUShaderCreateInfo vs={0};
        vs.code=(const Uint8*)raycast_vert_spv; vs.code_size=sizeof raycast_vert_spv;
        vs.entrypoint="main"; vs.format=SDL_GPU_SHADERFORMAT_SPIRV; vs.stage=SDL_GPU_SHADERSTAGE_VERTEX;
        SDL_GPUShader *vsh=SDL_CreateGPUShader(dev,&vs);
        SDL_GPUShaderCreateInfo fs={0};
        fs.code=(const Uint8*)raycast_frag_spv; fs.code_size=sizeof raycast_frag_spv;
        fs.entrypoint="main"; fs.format=SDL_GPU_SHADERFORMAT_SPIRV; fs.stage=SDL_GPU_SHADERSTAGE_FRAGMENT;
        fs.num_samplers=2; fs.num_uniform_buffers=1;
        SDL_GPUShader *fsh=SDL_CreateGPUShader(dev,&fs);
        if(!vsh||!fsh){ if(vsh)SDL_ReleaseGPUShader(dev,vsh); if(fsh)SDL_ReleaseGPUShader(dev,fsh);
            mc_gpu_ray_destroy(r); return NULL; }
        SDL_GPUColorTargetDescription ct={0}; ct.format=swfmt;
        SDL_GPUGraphicsPipelineCreateInfo p={0};
        p.vertex_shader=vsh; p.fragment_shader=fsh; p.primitive_type=SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
        p.target_info.color_target_descriptions=&ct; p.target_info.num_color_targets=1;
        r->ray_pipe=SDL_CreateGPUGraphicsPipeline(dev,&p);
        SDL_ReleaseGPUShader(dev,vsh); SDL_ReleaseGPUShader(dev,fsh);
        if(!r->ray_pipe){ mc_gpu_ray_destroy(r); return NULL; }
    }

    // constant tables (same as the c3g compute decoder)
    uint32_t freq[MCR_MAXSYM], cdf[MCR_MAXSYM+1], slot[MCR_TOTAL];
    uint32_t acc=0; for(int s=0;s<MCR_MAXSYM;++s){ freq[s]=MCR_FREQ[s]; cdf[s]=acc; acc+=MCR_FREQ[s]; } cdf[MCR_MAXSYM]=acc;
    for(int s=0;s<MCR_MAXSYM;++s) for(uint32_t f=cdf[s]; f<cdf[s+1]; ++f) slot[f]=(uint32_t)s;
    float step[MCR_N3];
    for(int cz=0;cz<16;++cz)for(int cy=0;cy<16;++cy)for(int cx=0;cx<16;++cx)
        step[(cz*16+cy)*16+cx]=quality*powf(1.0f+(float)(cz+cy+cx), MC_HF_EXP);
    uint32_t scan[MCR_N3]; for(int i=0;i<MCR_N3;++i) scan[i]=i;
    for(int i=1;i<MCR_N3;++i){ uint32_t k=scan[i]; int kl=(k/256)+((k/16)%16)+(k%16);
        int j=i-1; while(j>=0){ uint32_t s=scan[j]; int sl=(s/256)+((s/16)%16)+(s%16);
            if(sl>kl||(sl==kl&&s>k)){ scan[j+1]=s; j--; } else break; } scan[j+1]=k; }
    float cm[16*16];
    for(int k=0;k<16;++k){ double ck=(k==0)?sqrt(1.0/16):sqrt(2.0/16);
        for(int n=0;n<16;++n) cm[k*16+n]=(float)(ck*cos(M_PI*(2.0*n+1.0)*k/32.0)); }
    r->b_freq=mcr_robuf(dev,sizeof freq); mcr_up(dev,r->b_freq,freq,sizeof freq);
    r->b_cdf =mcr_robuf(dev,sizeof cdf);  mcr_up(dev,r->b_cdf,cdf,sizeof cdf);
    r->b_slot=mcr_robuf(dev,sizeof slot); mcr_up(dev,r->b_slot,slot,sizeof slot);
    r->b_step=mcr_robuf(dev,sizeof step); mcr_up(dev,r->b_step,step,sizeof step);
    r->b_scan=mcr_robuf(dev,sizeof scan); mcr_up(dev,r->b_scan,scan,sizeof scan);
    r->b_dct =mcr_robuf(dev,sizeof cm);   mcr_up(dev,r->b_dct,cm,sizeof cm);
    r->b_payload=mcr_robuf(dev,MCR_PAYLOAD_CAP);
    r->b_blktab =mcr_robuf(dev,MCR_MAX_BLOCKS*2*4);

    SDL_GPUSamplerCreateInfo si={0};
    si.address_mode_u=si.address_mode_v=si.address_mode_w=SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    si.mipmap_mode=SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    si.min_filter=si.mag_filter=SDL_GPU_FILTER_LINEAR;   // trilinear volume sampling
    r->vol_samp=SDL_CreateGPUSampler(dev,&si);
    si.min_filter=si.mag_filter=SDL_GPU_FILTER_NEAREST;
    r->lut_samp=SDL_CreateGPUSampler(dev,&si);

    SDL_GPUTextureCreateInfo lc={0};
    lc.type=SDL_GPU_TEXTURETYPE_2D; lc.format=SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM;
    lc.usage=SDL_GPU_TEXTUREUSAGE_SAMPLER; lc.width=256; lc.height=1;
    lc.layer_count_or_depth=1; lc.num_levels=1;
    r->lut_tex=SDL_CreateGPUTexture(dev,&lc);
    return r;
}

void mc_gpu_ray_destroy(mc_gpu_ray *r){
    if(!r) return;
    SDL_GPUDevice *d=r->dev;
    SDL_GPUBuffer *b[]={r->b_freq,r->b_cdf,r->b_slot,r->b_step,r->b_scan,r->b_dct,r->b_payload,r->b_blktab};
    for(int i=0;i<8;++i) if(b[i]) SDL_ReleaseGPUBuffer(d,b[i]);
    if(r->vol) SDL_ReleaseGPUTexture(d,r->vol);
    if(r->lut_tex) SDL_ReleaseGPUTexture(d,r->lut_tex);
    if(r->vol_samp) SDL_ReleaseGPUSampler(d,r->vol_samp);
    if(r->lut_samp) SDL_ReleaseGPUSampler(d,r->lut_samp);
    if(r->decode_pipe) SDL_ReleaseGPUComputePipeline(d,r->decode_pipe);
    if(r->ray_pipe) SDL_ReleaseGPUGraphicsPipeline(d,r->ray_pipe);
    SDL_free(r);
}

void mc_gpu_ray_set_lut(mc_gpu_ray *r, const uint32_t lut[256]){
    SDL_GPUTransferBufferCreateInfo tc={0}; tc.usage=SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD; tc.size=256*4;
    SDL_GPUTransferBuffer *tb=SDL_CreateGPUTransferBuffer(r->dev,&tc);
    void *m=SDL_MapGPUTransferBuffer(r->dev,tb,false); SDL_memcpy(m,lut,256*4); SDL_UnmapGPUTransferBuffer(r->dev,tb);
    SDL_GPUCommandBuffer *cb=SDL_AcquireGPUCommandBuffer(r->dev);
    SDL_GPUCopyPass *cp=SDL_BeginGPUCopyPass(cb);
    SDL_GPUTextureTransferInfo ti={0}; ti.transfer_buffer=tb; ti.pixels_per_row=256; ti.rows_per_layer=1;
    SDL_GPUTextureRegion tr={0}; tr.texture=r->lut_tex; tr.w=256; tr.h=1; tr.d=1;
    SDL_UploadToGPUTexture(cp,&ti,&tr,false); SDL_EndGPUCopyPass(cp); SDL_SubmitGPUCommandBuffer(cb);
    SDL_ReleaseGPUTransferBuffer(r->dev,tb);
}

bool mc_gpu_ray_upload(mc_gpu_ray *r, int gz,int gy,int gx,
                       const uint8_t *const *blobs, const uint32_t *lens){
    int nb=gz*gy*gx; if(nb<=0||nb>MCR_MAX_BLOCKS) return false;
    int vx=gx*16, vy=gy*16, vz=gz*16;
    // (re)create the dense 3D volume texture on size change.
    if(!r->vol || r->vx!=vx || r->vy!=vy || r->vz!=vz){
        if(r->vol) SDL_ReleaseGPUTexture(r->dev,r->vol);
        SDL_GPUTextureCreateInfo tc={0};
        tc.type=SDL_GPU_TEXTURETYPE_3D; tc.format=SDL_GPU_TEXTUREFORMAT_R8_UNORM;
        tc.usage=SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE|SDL_GPU_TEXTUREUSAGE_SAMPLER;
        tc.width=(Uint32)vx; tc.height=(Uint32)vy; tc.layer_count_or_depth=(Uint32)vz; tc.num_levels=1;
        r->vol=SDL_CreateGPUTexture(r->dev,&tc);
        if(!r->vol) return false;
        r->vx=vx; r->vy=vy; r->vz=vz;
    }
    // pack compressed blocks + [off,len] table.
    uint8_t *packed=SDL_malloc(MCR_PAYLOAD_CAP);
    uint32_t *tab=SDL_malloc((size_t)nb*2*4);
    uint32_t off=0;
    for(int i=0;i<nb;++i){
        uint32_t l=blobs[i]?lens[i]:0;
        if(off+((l+3)&~3u)>MCR_PAYLOAD_CAP){ SDL_free(packed); SDL_free(tab); return false; }
        tab[2*i]=off; tab[2*i+1]=l;
        if(l){ SDL_memcpy(packed+off,blobs[i],l); off+=(l+3)&~3u; }
    }
    if(off==0) off=4;
    mcr_up(r->dev,r->b_payload,packed,off);
    mcr_up(r->dev,r->b_blktab,tab,(uint32_t)nb*2*4);
    SDL_free(packed); SDL_free(tab);
    r->gz=gz; r->gy=gy; r->gx=gx; r->nblocks=nb;

    // compute-decode into the 3D texture.
    SDL_GPUCommandBuffer *cb=SDL_AcquireGPUCommandBuffer(r->dev);
    SDL_GPUStorageTextureReadWriteBinding rw={0}; rw.texture=r->vol;
    SDL_GPUComputePass *cp=SDL_BeginGPUComputePass(cb,&rw,1,NULL,0);
    SDL_BindGPUComputePipeline(cp,r->decode_pipe);
    SDL_GPUBuffer *ro[8]={r->b_payload,r->b_blktab,r->b_freq,r->b_cdf,r->b_slot,r->b_step,r->b_scan,r->b_dct};
    SDL_BindGPUComputeStorageBuffers(cp,0,ro,8);
    mcr_decode_ubo du={ (uint32_t)nb, (uint32_t)gz, (uint32_t)gy, (uint32_t)gx };
    SDL_PushGPUComputeUniformData(cb,0,&du,sizeof du);
    SDL_DispatchGPUCompute(cp,(Uint32)nb,1,1);
    SDL_EndGPUComputePass(cp);
    SDL_SubmitGPUCommandBuffer(cb);
    return true;
}

SDL_GPUTexture *mc_gpu_ray_volume(mc_gpu_ray *r, int *vx,int *vy,int *vz){
    if(vx)*vx=r->vx; if(vy)*vy=r->vy; if(vz)*vz=r->vz; return r->vol;
}

void mc_gpu_ray_render(mc_gpu_ray *r, SDL_GPUCommandBuffer *cmd, SDL_GPUTexture *target,
                       const float inv_view_proj[16], const float cam_pos[3],
                       float step_voxels, float gain){
    if(!r->vol) return;
    SDL_GPUColorTargetInfo cti={0};
    cti.texture=target; cti.load_op=SDL_GPU_LOADOP_CLEAR; cti.store_op=SDL_GPU_STOREOP_STORE;
    cti.clear_color=(SDL_FColor){0.04f,0.04f,0.05f,1.0f};
    SDL_GPURenderPass *rp=SDL_BeginGPURenderPass(cmd,&cti,1,NULL);
    SDL_BindGPUGraphicsPipeline(rp,r->ray_pipe);
    SDL_GPUTextureSamplerBinding tsb[2]={0};
    tsb[0].texture=r->vol;     tsb[0].sampler=r->vol_samp;
    tsb[1].texture=r->lut_tex; tsb[1].sampler=r->lut_samp;
    SDL_BindGPUFragmentSamplers(rp,0,tsb,2);
    mcr_ray_ubo u; SDL_memcpy(u.inv_view_proj,inv_view_proj,16*sizeof(float));
    u.cam_pos[0]=cam_pos[0]; u.cam_pos[1]=cam_pos[1]; u.cam_pos[2]=cam_pos[2]; u.cam_pos[3]=0;
    u.vol_dim[0]=(float)r->vx; u.vol_dim[1]=(float)r->vy; u.vol_dim[2]=(float)r->vz; u.vol_dim[3]=step_voxels;
    u.params[0]=0; u.params[1]=gain; u.params[2]=0; u.params[3]=0;
    SDL_PushGPUFragmentUniformData(cmd,0,&u,sizeof u);
    SDL_DrawGPUPrimitives(rp,3,1,0,0);
    SDL_EndGPURenderPass(rp);
}

#endif /* MC_GPU_RAY_IMPLEMENTATION */
