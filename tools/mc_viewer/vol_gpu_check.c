/* vol_gpu_check — validate the full GPU c3g decode + slice-sample pipeline.
 *
 * Builds a synthetic 16^3-block slab, c3g-encodes each block on the CPU, feeds
 * the COMPRESSED blocks to mc_gpu_vol (upload -> GPU compute-decode -> GPU
 * slice-sample), renders a Z-slice into an offscreen target, reads it back, and
 * compares each pixel to the CPU-decoded voxel at that slice position. This is
 * the headless proof of the "CPU sends compressed, GPU decodes + samples +
 * renders" path. Exit 0 if the GPU slice matches the CPU reference.
 *
 * Headless: SDL_VIDEODRIVER=offscreen (+ Vulkan).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <SDL3/SDL.h>

#include "matter_compressor.h"
#define MC_GPU_VOL_IMPLEMENTATION
#include "mc_gpu_vol.h"

#define N3 4096
static int die(const char*m){ fprintf(stderr,"FATAL %s: %s\n",m,SDL_GetError()); return 1; }

int main(void){
    if(!SDL_Init(SDL_INIT_VIDEO)) return die("SDL_Init");
    SDL_GPUDevice *dev=SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV,true,NULL);
    if(!dev) return die("CreateGPUDevice");

    // --- slab geometry: GZ x GY x GX blocks (small, keeps the readback cheap) ---
    const int GZ=1, GY=2, GX=2;            // 16 x 32 x 32 voxels
    const int EZ=GZ*16, EY=GY*16, EX=GX*16;
    const int NB=GZ*GY*GX;

    mc_codec_ctx *C=mc_codec_ctx_new();
    mc_codec_ctx_set_quality(C,8.0f); mc_codec_ctx_set_max_error(C,0);

    // per-block: build voxels, c3g-encode, CPU-decode (reference).
    mc_buf payloads[64]; uint32_t plen[64]; int coded[64];
    static uint8_t cpu[64][N3];
    const uint8_t *blobs[64]; uint32_t lens[64];
    memset(payloads,0,sizeof payloads);
    for(int b=0;b<NB;++b){
        uint8_t vox[N3];
        for(int z=0;z<16;++z)for(int y=0;y<16;++y)for(int x=0;x<16;++x){
            int gv=(x*5 + y*3 + z*2 + b*17) & 0xFF;
            if(((x+y+z)&7)==0) gv=0;       // sprinkle air
            vox[(z*16+y)*16+x]=(uint8_t)gv;
        }
        coded[b]=mc_c3g_enc_block(C,vox,&payloads[b],&plen[b]);
        memset(cpu[b],0,N3);
        if(coded[b]) mc_c3g_dec_block(C,payloads[b].p,plen[b],cpu[b]);
        blobs[b]=coded[b]?payloads[b].p:NULL; lens[b]=coded[b]?plen[b]:0;
    }

    // --- GPU vol pipeline: upload the compressed slab ---
    SDL_GPUTextureFormat fmt=SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM;
    mc_gpu_vol *V=mc_gpu_vol_create(dev,fmt,8.0f);
    if(!V) return die("mc_gpu_vol_create");
    if(!mc_gpu_vol_upload_slab(V,GZ,GY,GX,EZ,EY,EX,blobs,lens)) return die("upload_slab");

    // identity grayscale LUT so the sampled u8 maps straight to the channel.
    uint32_t lut[256]; for(int i=0;i<256;++i){ uint32_t g=(uint32_t)i; lut[i]=0xFF000000u|(g<<16)|(g<<8)|g; }
    mc_gpu_vol_set_lut(V,lut);

    // --- offscreen render target sized to the slice (EX x EY for a Z slice) ---
    SDL_GPUTextureCreateInfo tci={0};
    tci.type=SDL_GPU_TEXTURETYPE_2D; tci.format=fmt;
    tci.usage=SDL_GPU_TEXTUREUSAGE_COLOR_TARGET|SDL_GPU_TEXTUREUSAGE_SAMPLER;
    tci.width=(Uint32)EX; tci.height=(Uint32)EY; tci.layer_count_or_depth=1; tci.num_levels=1;
    SDL_GPUTexture *rt=SDL_CreateGPUTexture(dev,&tci);

    // render a Z-slice at z=4 (within block row 0). draw the quad to fill the RT.
    int zslice=4;
    SDL_GPUCommandBuffer *cmd=SDL_AcquireGPUCommandBuffer(dev);
    mc_gpu_vol_render(V,cmd,rt,/*axis Z*/0,zslice,(float)EX,(float)EY,0,0,EX,EY);

    // download the RT
    uint32_t out_bytes=(uint32_t)EX*EY*4;
    SDL_GPUTransferBufferCreateInfo dtc={0}; dtc.usage=SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD; dtc.size=out_bytes;
    SDL_GPUTransferBuffer *dtb=SDL_CreateGPUTransferBuffer(dev,&dtc);
    SDL_GPUCopyPass *cp=SDL_BeginGPUCopyPass(cmd);
    SDL_GPUTextureRegion reg={0}; reg.texture=rt; reg.w=(Uint32)EX; reg.h=(Uint32)EY; reg.d=1;
    SDL_GPUTextureTransferInfo ti={0}; ti.transfer_buffer=dtb; ti.pixels_per_row=(Uint32)EX; ti.rows_per_layer=(Uint32)EY;
    SDL_DownloadFromGPUTexture(cp,&reg,&ti);
    SDL_EndGPUCopyPass(cp);
    SDL_GPUFence *fence=SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    SDL_WaitForGPUFences(dev,true,&fence,1); SDL_ReleaseGPUFence(dev,fence);

    uint32_t *px=SDL_MapGPUTransferBuffer(dev,dtb,false);

    // --- compare: GPU pixel (sx,sy) for a Z slice samples vox (z=zslice, y=sy, x=sx)
    //     mapped through v_uv = (x+0.5)/EX etc. The frag does int(v_uv*extent), so
    //     pixel center (sx+0.5)/EX * EX = sx. Compare to the CPU block voxel. ---
    // the shader flips v (Vulkan/SDL_GPU bottom-up raster vs top-down readback),
    // so readback row sy samples voxel vy = EY-1-sy. Mirror that here.
    long diff=0,maxd=0,checked=0;
    for(int sy=0; sy<EY; ++sy) for(int sx=0; sx<EX; ++sx){
        int vy=EY-1-sy;
        int gz2=zslice>>4, gy2=vy>>4, gx2=sx>>4;
        int lz=zslice&15, ly=vy&15, lx=sx&15;
        int bidx=(gz2*GY+gy2)*GX+gx2;
        int ref=cpu[bidx][(lz*16+ly)*16+lx];
        uint32_t p=px[sy*EX+sx];
        int got=p&0xFF;                       // B channel (gray => all equal)
        int d=ref-got; if(d<0)d=-d;
        if(d){ diff++; if(d>maxd)maxd=d; }
        checked++;
    }
    SDL_UnmapGPUTransferBuffer(dev,dtb);

    printf("Z-slice %dx%d: checked=%ld diff=%ld (maxdiff=%ld)\n",EX,EY,checked,diff,maxd);
    int ok=(maxd<=1);
    printf(ok?"vol_gpu_check: OK (GPU slice matches CPU within <=1)\n"
             :"vol_gpu_check: FAIL\n");

    for(int b=0;b<NB;++b) free(payloads[b].p);
    mc_codec_ctx_free(C);
    mc_gpu_vol_destroy(V);
    SDL_DestroyGPUDevice(dev);
    SDL_Quit();
    return ok?0:1;
}
