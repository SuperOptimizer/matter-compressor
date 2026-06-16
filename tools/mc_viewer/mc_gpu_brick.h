/* mc_gpu_brick.h — out-of-core brick cache + raycaster (M5).
 *
 * The volume is a grid of 16^3 bricks. Only resident (non-air) bricks live in a
 * 3D-texture ATLAS (a fixed grid of slots); a PAGE TABLE (3D r32ui, one texel
 * per brick) maps brick coord -> atlas slot+1 (0 = air / not resident). The
 * brick raycaster (raycast_brick.frag) walks the page table and skips empty
 * bricks (empty-space skipping), so air costs nothing and only the resident
 * working set occupies VRAM. This is the path to volumes far larger than VRAM.
 *
 * First M5 pass: sequential slot assignment for the resident bricks of a region
 * (air bricks skipped), no LRU eviction / frustum cull / LOD yet — those layer
 * on top. Reuses mc_gpu_ray's decode pipeline (mc_gpu_ray_decode_into, use_dst).
 *
 * Define MC_GPU_BRICK_IMPLEMENTATION in one TU (after mc_gpu_ray.h's impl).
 */
#ifndef MC_GPU_BRICK_H_
#define MC_GPU_BRICK_H_

#include <SDL3/SDL.h>
#include <stdbool.h>
#include <stdint.h>
#include "mc_gpu_ray.h"

typedef struct mc_gpu_brick mc_gpu_brick;

/* atlas_slots_per_axis^3 brick slots; each brick is 16^3 voxels. e.g. 8 -> 512
 * slots, atlas = 128^3 voxels (~2 MB r8). */
mc_gpu_brick *mc_gpu_brick_create(mc_gpu_ray *ray, SDL_GPUTextureFormat swfmt,
                                  int atlas_slots_per_axis);
void          mc_gpu_brick_destroy(mc_gpu_brick *b);

/* Page a region (brick-grid gz*gy*gx, origin oz,oy,ox in GLOBAL brick coords)
 * into the brick cache. blobs[i]/lens[i] = the compressed c3g block of brick i
 * in (gz,gy,gx) raster (NULL/0 = air -> not resident). Bricks already resident
 * (by global key) keep their atlas slot and are NOT re-decoded; new bricks take
 * a free slot or evict the least-recently-used one (LRU). Only newly-paged-in
 * bricks are decoded. The page table is rebuilt for this region's view.
 * Returns false on failure; *n_resident = bricks visible this region. */
bool mc_gpu_brick_page(mc_gpu_brick *b, int oz,int oy,int ox, int gz,int gy,int gx,
                       const uint8_t *const *blobs, const uint32_t *lens,
                       int *n_resident);

void mc_gpu_brick_set_lut(mc_gpu_brick *b, const uint32_t lut[256]);
void mc_gpu_brick_set_lighting(mc_gpu_brick *b, int on, const float dir[3],
                               float amb,float diff,float spec,float shin,float grad_g0);

/* Raycast the bricked volume into `target`. mode = MC_RAY_MIP | MC_RAY_EA. */
void mc_gpu_brick_render(mc_gpu_brick *b, SDL_GPUCommandBuffer *cmd, SDL_GPUTexture *target,
                         const float inv_view_proj[16], float step_voxels, float gain,
                         int mode, float alpha_min, float absorption);

/* The atlas texture + page table (for readback validation). */
SDL_GPUTexture *mc_gpu_brick_atlas(mc_gpu_brick *b, int *edge);
SDL_GPUTexture *mc_gpu_brick_pagetable(mc_gpu_brick *b, int *gx,int *gy,int *gz);

#endif /* MC_GPU_BRICK_H_ */

#ifdef MC_GPU_BRICK_IMPLEMENTATION
#undef MC_GPU_BRICK_IMPLEMENTATION

#include "shaders/raycast.vert.spv.h"        /* reuse the fullscreen-triangle VS */
#include "shaders/raycast_brick.frag.spv.h"

typedef struct {
    float inv_view_proj[16];
    float vol_dim[4];     // x,y,z, step
    float bgrid[4];       // brick grid x,y,z, atlas slots/axis
    float params[4];      // mode, gain, alpha_min, absorption
    float light[4];
    float lparams[4];
    float lparams2[4];
} mcb_ubo;

struct mc_gpu_brick {
    mc_gpu_ray *ray;        // borrowed: decode pipeline + tables + device
    SDL_GPUDevice *dev;
    SDL_GPUGraphicsPipeline *pipe;
    SDL_GPUSampler *atlas_samp, *lut_samp, *page_samp;
    SDL_GPUTexture *atlas;     // r8 brick atlas
    SDL_GPUTexture *page;      // r32ui page table
    SDL_GPUTexture *lut;       // 256x1 transfer fn
    int aps;                   // atlas slots per axis
    int gx,gy,gz;              // current region brick grid
    int n_resident;
    int light_on; float ldir[3], amb,diff,spec,shin,grad_g0;

    // LRU slot state (persistent across page() calls). Each atlas slot holds at
    // most one brick, keyed by its GLOBAL brick coord. Re-paging a region keeps
    // already-resident bricks (no re-decode) and evicts the least-recently-used
    // slot when the budget is full.
    int     max_slots;
    int64_t *slot_key;         // [max_slots] global brick key, -1 = free
    uint32_t *slot_lru;        // [max_slots] last-used tick
    uint32_t tick;             // bumped each page()
};

// pack/just a stable 63-bit key from a global brick coord (each < 2^21).
static int64_t mcb_key(int Bz,int By,int Bx){
    return ((int64_t)(uint32_t)Bz<<42) | ((int64_t)(uint32_t)By<<21) | (int64_t)(uint32_t)Bx;
}

mc_gpu_brick *mc_gpu_brick_create(mc_gpu_ray *ray, SDL_GPUTextureFormat swfmt, int aps){
    mc_gpu_brick *b=SDL_calloc(1,sizeof *b);
    if(!b) return NULL;
    b->ray=ray; b->dev=mc_gpu_ray_device(ray); b->aps=aps;

    SDL_GPUShaderCreateInfo vs={0};
    vs.code=(const Uint8*)raycast_vert_spv; vs.code_size=sizeof raycast_vert_spv;
    vs.entrypoint="main"; vs.format=SDL_GPU_SHADERFORMAT_SPIRV; vs.stage=SDL_GPU_SHADERSTAGE_VERTEX;
    SDL_GPUShader *vsh=SDL_CreateGPUShader(b->dev,&vs);
    SDL_GPUShaderCreateInfo fs={0};
    fs.code=(const Uint8*)raycast_brick_frag_spv; fs.code_size=sizeof raycast_brick_frag_spv;
    fs.entrypoint="main"; fs.format=SDL_GPU_SHADERFORMAT_SPIRV; fs.stage=SDL_GPU_SHADERSTAGE_FRAGMENT;
    fs.num_samplers=3; fs.num_uniform_buffers=1;
    SDL_GPUShader *fsh=SDL_CreateGPUShader(b->dev,&fs);
    if(!vsh||!fsh){ if(vsh)SDL_ReleaseGPUShader(b->dev,vsh); if(fsh)SDL_ReleaseGPUShader(b->dev,fsh);
        mc_gpu_brick_destroy(b); return NULL; }
    SDL_GPUColorTargetDescription ct={0}; ct.format=swfmt;
    SDL_GPUGraphicsPipelineCreateInfo p={0};
    p.vertex_shader=vsh; p.fragment_shader=fsh; p.primitive_type=SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
    p.target_info.color_target_descriptions=&ct; p.target_info.num_color_targets=1;
    b->pipe=SDL_CreateGPUGraphicsPipeline(b->dev,&p);
    SDL_ReleaseGPUShader(b->dev,vsh); SDL_ReleaseGPUShader(b->dev,fsh);
    if(!b->pipe){ mc_gpu_brick_destroy(b); return NULL; }

    // LRU slot state.
    b->max_slots=aps*aps*aps;
    b->slot_key=SDL_malloc((size_t)b->max_slots*sizeof(int64_t));
    b->slot_lru=SDL_calloc((size_t)b->max_slots,sizeof(uint32_t));
    for(int i=0;i<b->max_slots;++i) b->slot_key[i]=-1;   // all free

    // atlas: aps^3 slots of 16^3 -> (aps*16)^3 r8 texture.
    int edge=aps*16;
    SDL_GPUTextureCreateInfo ac={0};
    ac.type=SDL_GPU_TEXTURETYPE_3D; ac.format=SDL_GPU_TEXTUREFORMAT_R8_UNORM;
    ac.usage=SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE|SDL_GPU_TEXTUREUSAGE_SAMPLER;
    ac.width=(Uint32)edge; ac.height=(Uint32)edge; ac.layer_count_or_depth=(Uint32)edge; ac.num_levels=1;
    b->atlas=SDL_CreateGPUTexture(b->dev,&ac);

    SDL_GPUSamplerCreateInfo si={0};
    si.address_mode_u=si.address_mode_v=si.address_mode_w=SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    si.mipmap_mode=SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    si.min_filter=si.mag_filter=SDL_GPU_FILTER_LINEAR;  b->atlas_samp=SDL_CreateGPUSampler(b->dev,&si);
    si.min_filter=si.mag_filter=SDL_GPU_FILTER_NEAREST; b->lut_samp=SDL_CreateGPUSampler(b->dev,&si);
    b->page_samp=SDL_CreateGPUSampler(b->dev,&si);      // nearest (integer page table)

    SDL_GPUTextureCreateInfo lc={0};
    lc.type=SDL_GPU_TEXTURETYPE_2D; lc.format=SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM;
    lc.usage=SDL_GPU_TEXTUREUSAGE_SAMPLER; lc.width=256; lc.height=1; lc.layer_count_or_depth=1; lc.num_levels=1;
    b->lut=SDL_CreateGPUTexture(b->dev,&lc);
    return b;
}

void mc_gpu_brick_destroy(mc_gpu_brick *b){
    if(!b) return;
    if(b->atlas) SDL_ReleaseGPUTexture(b->dev,b->atlas);
    if(b->page)  SDL_ReleaseGPUTexture(b->dev,b->page);
    if(b->lut)   SDL_ReleaseGPUTexture(b->dev,b->lut);
    if(b->atlas_samp) SDL_ReleaseGPUSampler(b->dev,b->atlas_samp);
    if(b->lut_samp)   SDL_ReleaseGPUSampler(b->dev,b->lut_samp);
    if(b->page_samp)  SDL_ReleaseGPUSampler(b->dev,b->page_samp);
    if(b->pipe) SDL_ReleaseGPUGraphicsPipeline(b->dev,b->pipe);
    SDL_free(b->slot_key); SDL_free(b->slot_lru);
    SDL_free(b);
}

void mc_gpu_brick_set_lut(mc_gpu_brick *b, const uint32_t lut[256]){
    SDL_GPUTransferBufferCreateInfo tc={0}; tc.usage=SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD; tc.size=256*4;
    SDL_GPUTransferBuffer *tb=SDL_CreateGPUTransferBuffer(b->dev,&tc);
    void *m=SDL_MapGPUTransferBuffer(b->dev,tb,false); SDL_memcpy(m,lut,256*4); SDL_UnmapGPUTransferBuffer(b->dev,tb);
    SDL_GPUCommandBuffer *cb=SDL_AcquireGPUCommandBuffer(b->dev);
    SDL_GPUCopyPass *cp=SDL_BeginGPUCopyPass(cb);
    SDL_GPUTextureTransferInfo ti={0}; ti.transfer_buffer=tb; ti.pixels_per_row=256; ti.rows_per_layer=1;
    SDL_GPUTextureRegion tr={0}; tr.texture=b->lut; tr.w=256; tr.h=1; tr.d=1;
    SDL_UploadToGPUTexture(cp,&ti,&tr,false); SDL_EndGPUCopyPass(cp); SDL_SubmitGPUCommandBuffer(cb);
    SDL_ReleaseGPUTransferBuffer(b->dev,tb);
}
void mc_gpu_brick_set_lighting(mc_gpu_brick *b, int on, const float dir[3],
                               float amb,float diff,float spec,float shin,float grad_g0){
    b->light_on=on; if(dir){b->ldir[0]=dir[0];b->ldir[1]=dir[1];b->ldir[2]=dir[2];}
    b->amb=amb; b->diff=diff; b->spec=spec; b->shin=shin; b->grad_g0=grad_g0;
}

// slot origin (texels) from a linear slot index.
static uint32_t mcb_slot_dst(mc_gpu_brick *b, int slot){
    int sx=slot%b->aps, sy=(slot/b->aps)%b->aps, sz=slot/(b->aps*b->aps);
    return (uint32_t)sx | ((uint32_t)sy<<10) | ((uint32_t)sz<<20);
}

bool mc_gpu_brick_page(mc_gpu_brick *b, int oz,int oy,int ox, int gz,int gy,int gx,
                       const uint8_t *const *blobs, const uint32_t *lens, int *n_resident){
    int nb=gz*gy*gx; if(nb<=0) return false;
    b->tick++;

    // (re)create the page table at the region brick grid.
    if(!b->page || b->gx!=gx || b->gy!=gy || b->gz!=gz){
        if(b->page) SDL_ReleaseGPUTexture(b->dev,b->page);
        SDL_GPUTextureCreateInfo pc={0};
        pc.type=SDL_GPU_TEXTURETYPE_3D; pc.format=SDL_GPU_TEXTUREFORMAT_R32_UINT;
        pc.usage=SDL_GPU_TEXTUREUSAGE_SAMPLER;
        pc.width=(Uint32)gx; pc.height=(Uint32)gy; pc.layer_count_or_depth=(Uint32)gz; pc.num_levels=1;
        b->page=SDL_CreateGPUTexture(b->dev,&pc);
        b->gx=gx; b->gy=gy; b->gz=gz;
    }

    // residency lookup: key -> slot, for the bricks currently in the atlas.
    // (linear scan of max_slots is fine for the slot counts we use; a hash can
    // replace it if max_slots grows large.)
    uint8_t *packed=SDL_malloc(8u<<20);
    uint32_t *tab=SDL_malloc((size_t)nb*3*4);
    uint32_t *page=SDL_malloc((size_t)nb*4);
    uint32_t off=0; int dnb=0, vis=0;

    for(int bz=0;bz<gz;++bz)for(int by=0;by<gy;++by)for(int bx=0;bx<gx;++bx){
        int i=(bz*gy+by)*gx+bx;
        page[i]=0;
        uint32_t l = blobs[i] ? lens[i] : 0;
        if(!l) continue;                          // air -> not resident
        int64_t key=mcb_key(oz+bz, oy+by, ox+bx);

        // already resident? keep its slot, bump LRU, no re-decode.
        int slot=-1;
        for(int s=0;s<b->max_slots;++s) if(b->slot_key[s]==key){ slot=s; break; }
        if(slot<0){
            // find a free slot, else evict the least-recently-used one.
            int free_s=-1; for(int s=0;s<b->max_slots;++s) if(b->slot_key[s]<0){ free_s=s; break; }
            if(free_s>=0) slot=free_s;
            else { int lru=0; for(int s=1;s<b->max_slots;++s) if(b->slot_lru[s]<b->slot_lru[lru]) lru=s; slot=lru; }
            // decode this brick into `slot`.
            if(off+((l+3)&~3u) <= (8u<<20)){
                tab[3*dnb+0]=off; tab[3*dnb+1]=l; tab[3*dnb+2]=mcb_slot_dst(b,slot);
                SDL_memcpy(packed+off, blobs[i], l); off+=(l+3)&~3u; dnb++;
                b->slot_key[slot]=key;
            } else continue;                      // payload buffer full this frame
        }
        b->slot_lru[slot]=b->tick;
        page[i]=(uint32_t)slot+1;
        vis++;
    }
    b->n_resident=vis;
    if(n_resident)*n_resident=vis;

    // upload the page table for this region's view.
    {
        SDL_GPUTransferBufferCreateInfo tc={0}; tc.usage=SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD; tc.size=(uint32_t)nb*4;
        SDL_GPUTransferBuffer *tb=SDL_CreateGPUTransferBuffer(b->dev,&tc);
        void *m=SDL_MapGPUTransferBuffer(b->dev,tb,false); SDL_memcpy(m,page,(size_t)nb*4); SDL_UnmapGPUTransferBuffer(b->dev,tb);
        SDL_GPUCommandBuffer *cb=SDL_AcquireGPUCommandBuffer(b->dev);
        SDL_GPUCopyPass *cp=SDL_BeginGPUCopyPass(cb);
        SDL_GPUTextureTransferInfo ti={0}; ti.transfer_buffer=tb; ti.pixels_per_row=(Uint32)gx; ti.rows_per_layer=(Uint32)gy;
        SDL_GPUTextureRegion tr={0}; tr.texture=b->page; tr.w=(Uint32)gx; tr.h=(Uint32)gy; tr.d=(Uint32)gz;
        SDL_UploadToGPUTexture(cp,&ti,&tr,false); SDL_EndGPUCopyPass(cp); SDL_SubmitGPUCommandBuffer(cb);
        SDL_ReleaseGPUTransferBuffer(b->dev,tb);
    }
    // decode ONLY the newly-paged-in bricks into their atlas slots.
    if(dnb>0) mc_gpu_ray_decode_into(b->ray, b->atlas, packed, off?off:4, tab, dnb);

    SDL_free(packed); SDL_free(tab); SDL_free(page);
    return true;
}

void mc_gpu_brick_render(mc_gpu_brick *b, SDL_GPUCommandBuffer *cmd, SDL_GPUTexture *target,
                         const float inv_view_proj[16], float step_voxels, float gain,
                         int mode, float alpha_min, float absorption){
    if(!b->page) return;
    SDL_GPUColorTargetInfo cti={0};
    cti.texture=target; cti.load_op=SDL_GPU_LOADOP_CLEAR; cti.store_op=SDL_GPU_STOREOP_STORE;
    cti.clear_color=(SDL_FColor){0.04f,0.04f,0.05f,1.0f};
    SDL_GPURenderPass *rp=SDL_BeginGPURenderPass(cmd,&cti,1,NULL);
    SDL_BindGPUGraphicsPipeline(rp,b->pipe);
    SDL_GPUTextureSamplerBinding tsb[3]={0};
    tsb[0].texture=b->atlas; tsb[0].sampler=b->atlas_samp;
    tsb[1].texture=b->lut;   tsb[1].sampler=b->lut_samp;
    tsb[2].texture=b->page;  tsb[2].sampler=b->page_samp;
    SDL_BindGPUFragmentSamplers(rp,0,tsb,3);
    mcb_ubo u;
    for(int k=0;k<16;++k) u.inv_view_proj[k]=inv_view_proj[k];
    u.vol_dim[0]=(float)(b->gx*16); u.vol_dim[1]=(float)(b->gy*16); u.vol_dim[2]=(float)(b->gz*16); u.vol_dim[3]=step_voxels;
    u.bgrid[0]=(float)b->gx; u.bgrid[1]=(float)b->gy; u.bgrid[2]=(float)b->gz; u.bgrid[3]=(float)b->aps;
    u.params[0]=(float)mode; u.params[1]=gain; u.params[2]=alpha_min; u.params[3]=absorption;
    u.light[0]=b->ldir[0]; u.light[1]=b->ldir[1]; u.light[2]=b->ldir[2];
    u.light[3]=(mode==MC_RAY_EA && b->light_on)?1.0f:0.0f;
    u.lparams[0]=b->amb; u.lparams[1]=b->diff; u.lparams[2]=b->spec; u.lparams[3]=b->shin;
    u.lparams2[0]=b->grad_g0; u.lparams2[1]=u.lparams2[2]=u.lparams2[3]=0.0f;
    SDL_PushGPUFragmentUniformData(cmd,0,&u,sizeof u);
    SDL_DrawGPUPrimitives(rp,3,1,0,0);
    SDL_EndGPURenderPass(rp);
}

SDL_GPUTexture *mc_gpu_brick_atlas(mc_gpu_brick *b, int *edge){ if(edge)*edge=b->aps*16; return b->atlas; }
SDL_GPUTexture *mc_gpu_brick_pagetable(mc_gpu_brick *b, int *gx,int *gy,int *gz){
    if(gx)*gx=b->gx; if(gy)*gy=b->gy; if(gz)*gz=b->gz; return b->page;
}

#endif /* MC_GPU_BRICK_IMPLEMENTATION */
