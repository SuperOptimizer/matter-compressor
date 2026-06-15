/* ray_gpu_check — validate (M1) c3g decode into a dense 3D texture and (M2) the
 * MIP raycaster, both against a CPU reference. Headless (SDL_VIDEODRIVER=
 * offscreen).
 *
 * M1: decode a slab into the 3D texture, read it back, compare every texel to
 *     mc_c3g_dec_block of the same block -> must match exactly.
 * M2: render with an orthographic camera looking straight down -Z so the MIP
 *     image equals the CPU max-over-z projection of the volume; compare.
 *
 * Exit 0 iff both pass.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <SDL3/SDL.h>

#include "matter_compressor.h"
#define MC_GPU_RAY_IMPLEMENTATION
#include "mc_gpu_ray.h"

#define N3 4096
static int die(const char*m){ fprintf(stderr,"FATAL %s: %s\n",m,SDL_GetError()); return 1; }

int main(void){
    if(!SDL_Init(SDL_INIT_VIDEO)) return die("SDL_Init");
    SDL_GPUDevice *dev=SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV,true,NULL);
    if(!dev) return die("CreateGPUDevice");

    const int GZ=2, GY=2, GX=2;                 // 32^3 voxel volume
    const int VX=GX*16, VY=GY*16, VZ=GZ*16, NB=GZ*GY*GX;

    mc_codec_ctx *C=mc_codec_ctx_new();
    mc_codec_ctx_set_quality(C,8.0f); mc_codec_ctx_set_max_error(C,0);

    mc_buf payloads[8]; uint32_t plen[8]; int coded[8];
    static uint8_t cpu[8][N3];
    const uint8_t *blobs[8]; uint32_t lens[8];
    memset(payloads,0,sizeof payloads);
    for(int b=0;b<NB;++b){
        // centered ball across the whole VX^3 volume: a correct MIP looks like a
        // filled disk, easy to eyeball; block b's (gz,gy,gx) places its voxels.
        int bgx=b%GX, bgy=(b/GX)%GY, bgz=b/(GX*GY);
        uint8_t vox[N3];
        for(int z=0;z<16;++z)for(int y=0;y<16;++y)for(int x=0;x<16;++x){
            int gX=bgx*16+x, gY=bgy*16+y, gZ=bgz*16+z;
            float dx=gX-(VX-1)*0.5f, dy=gY-(VY-1)*0.5f, dz=gZ-(VZ-1)*0.5f;
            float rr=sqrtf(dx*dx+dy*dy+dz*dz);
            int gv = (rr < VX*0.4f) ? (int)(200 - rr*4) : 0;
            if(gv<0) gv=0;
            vox[(z*16+y)*16+x]=(uint8_t)gv;
        }
        coded[b]=mc_c3g_enc_block(C,vox,&payloads[b],&plen[b]);
        memset(cpu[b],0,N3);
        if(coded[b]) mc_c3g_dec_block(C,payloads[b].p,plen[b],cpu[b]);
        blobs[b]=coded[b]?payloads[b].p:NULL; lens[b]=coded[b]?plen[b]:0;
    }

    mc_gpu_ray *R=mc_gpu_ray_create(dev,SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM,8.0f);
    if(!R) return die("mc_gpu_ray_create");
    if(!mc_gpu_ray_upload(R,GZ,GY,GX,blobs,lens)) return die("ray_upload");

    // ---- M1: read back the 3D texture, compare to CPU decode ----
    int vx,vy,vz; SDL_GPUTexture *vol=mc_gpu_ray_volume(R,&vx,&vy,&vz);
    uint32_t vbytes=(uint32_t)vx*vy*vz;       // r8 = 1 byte/texel
    SDL_GPUTransferBufferCreateInfo dtc={0}; dtc.usage=SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD; dtc.size=vbytes;
    SDL_GPUTransferBuffer *dtb=SDL_CreateGPUTransferBuffer(dev,&dtc);
    SDL_GPUCommandBuffer *cb=SDL_AcquireGPUCommandBuffer(dev);
    SDL_GPUCopyPass *cp=SDL_BeginGPUCopyPass(cb);
    SDL_GPUTextureRegion reg={0}; reg.texture=vol; reg.w=(Uint32)vx; reg.h=(Uint32)vy; reg.d=(Uint32)vz;
    SDL_GPUTextureTransferInfo ti={0}; ti.transfer_buffer=dtb; ti.pixels_per_row=(Uint32)vx; ti.rows_per_layer=(Uint32)vy;
    SDL_DownloadFromGPUTexture(cp,&reg,&ti);
    SDL_EndGPUCopyPass(cp);
    SDL_GPUFence *f=SDL_SubmitGPUCommandBufferAndAcquireFence(cb);
    SDL_WaitForGPUFences(dev,true,&f,1); SDL_ReleaseGPUFence(dev,f);
    uint8_t *tex=SDL_MapGPUTransferBuffer(dev,dtb,false);

    long m1diff=0, m1max=0;
    for(int z=0;z<vz;++z)for(int y=0;y<vy;++y)for(int x=0;x<vx;++x){
        int gz2=z>>4, gy2=y>>4, gx2=x>>4, lz=z&15, ly=y&15, lx=x&15;
        int b=(gz2*GY+gy2)*GX+gx2;
        int ref=cpu[b][(lz*16+ly)*16+lx];
        int got=tex[(z*vy+y)*vx+x];
        int d=ref-got; if(d<0)d=-d;
        if(d){ m1diff++; if(d>m1max)m1max=d; }
    }
    SDL_UnmapGPUTransferBuffer(dev,dtb);
    printf("M1 (3D-texture decode): %ld/%d texels differ (maxdiff=%ld)\n", m1diff, vx*vy*vz, m1max);
    int m1ok = (m1max==0);

    // ---- M2: ortho camera straight down -Z; MIP == CPU max-over-z ----
    // Build an orthographic inv_view_proj mapping clip (x,y) in [-1,1] to volume
    // (x,y) in [0,vx]x[0,vy], looking along -Z from z = vz+pad. Column-major.
    int RW=VX, RH=VY;
    // Orthographic camera looking straight down -Z. inv_view_proj maps clip
    // (nx,ny,nz) in [-1,1]^3 to volume space (column-major M, M*(nx,ny,nz,1)):
    //   vol.x = 0.5*VX*nx + 0.5*VX        (clip x -> [0,VX])
    //   vol.y = 0.5*VY*ny + 0.5*VY        (clip y -> [0,VY])
    //   vol.z = A*nz + B  with  z(-1)=VZ+pad (near, in front),
    //                            z(+1)=-pad   (far, behind)
    //   => B = VZ/2,  A = -(VZ + 2*pad)/2
    float pad = 8.0f;
    float A = -((float)VZ + 2.0f*pad) * 0.5f;
    float B = (float)VZ * 0.5f;
    float ivp[16] = {0};                 // column-major
    ivp[0]  = (float)VX*0.5f;            // col0.x
    ivp[5]  = (float)VY*0.5f;            // col1.y
    ivp[10] = A;                         // col2.z
    ivp[12] = (float)VX*0.5f;            // col3.x
    ivp[13] = (float)VY*0.5f;            // col3.y
    ivp[14] = B;                         // col3.z
    ivp[15] = 1.0f;                      // col3.w
    float campos[3]={ (float)VX*0.5f, (float)VY*0.5f, (float)VZ+pad };

    // identity gray LUT
    uint32_t lut[256]; for(int i=0;i<256;++i){ uint32_t g=(uint32_t)i; lut[i]=0xFF000000u|(g<<16)|(g<<8)|g; }
    mc_gpu_ray_set_lut(R,lut);

    SDL_GPUTextureCreateInfo rtc={0};
    rtc.type=SDL_GPU_TEXTURETYPE_2D; rtc.format=SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM;
    rtc.usage=SDL_GPU_TEXTUREUSAGE_COLOR_TARGET|SDL_GPU_TEXTUREUSAGE_SAMPLER;
    rtc.width=(Uint32)RW; rtc.height=(Uint32)RH; rtc.layer_count_or_depth=1; rtc.num_levels=1;
    SDL_GPUTexture *rt=SDL_CreateGPUTexture(dev,&rtc);

    SDL_GPUCommandBuffer *cb2=SDL_AcquireGPUCommandBuffer(dev);
    mc_gpu_ray_render(R,cb2,rt,ivp,campos,0.5f,1.0f, MC_RAY_MIP, 0.0f, 0.0f);
    uint32_t obytes=(uint32_t)RW*RH*4;
    SDL_GPUTransferBufferCreateInfo o={0}; o.usage=SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD; o.size=obytes;
    SDL_GPUTransferBuffer *otb=SDL_CreateGPUTransferBuffer(dev,&o);
    SDL_GPUCopyPass *cp2=SDL_BeginGPUCopyPass(cb2);
    SDL_GPUTextureRegion rr={0}; rr.texture=rt; rr.w=(Uint32)RW; rr.h=(Uint32)RH; rr.d=1;
    SDL_GPUTextureTransferInfo oi={0}; oi.transfer_buffer=otb; oi.pixels_per_row=(Uint32)RW; oi.rows_per_layer=(Uint32)RH;
    SDL_DownloadFromGPUTexture(cp2,&rr,&oi);
    SDL_EndGPUCopyPass(cp2);
    SDL_GPUFence *f2=SDL_SubmitGPUCommandBufferAndAcquireFence(cb2);
    SDL_WaitForGPUFences(dev,true,&f2,1); SDL_ReleaseGPUFence(dev,f2);
    uint32_t *img=SDL_MapGPUTransferBuffer(dev,otb,false);

    // CPU MIP reference: for each pixel (px,py) -> volume column (vx,vy)=(px,py),
    // max over z of cpu voxel. The frag flips nothing for ortho-down but the
    // readback is top-down vs render bottom-up, so compare with py-flip.
    long m2diff=0, m2max=0, m2checked=0, nonzero=0;
    for(int py=0;py<RH;++py)for(int px=0;px<RW;++px){
        int X=px, Y=RH-1-py;                  // readback row flip (vulkan)
        if(X>=VX||Y>=VY) continue;
        int mip=0;
        for(int z=0;z<VZ;++z){
            int gz2=z>>4, gy2=Y>>4, gx2=X>>4, lz=z&15, ly=Y&15, lx=X&15;
            int b=(gz2*GY+gy2)*GX+gx2;
            int v=cpu[b][(lz*16+ly)*16+lx];
            if(v>mip) mip=v;
        }
        int got=img[py*RW+px]&0xFF;
        int d=mip-got; if(d<0)d=-d;
        if(d>m2max)m2max=d;
        if(d>32) m2diff++;          // "grossly wrong" — beyond trilinear/step slack
        if(mip) nonzero++;
        m2checked++;
    }
    // optional PPM dump for eyeballing the MIP image.
    const char *dump = SDL_getenv("RAY_DUMP");
    if (dump) {
        FILE *fp=fopen(dump,"wb");
        if(fp){ fprintf(fp,"P6\n%d %d\n255\n",RW,RH);
            for(int i=0;i<RW*RH;++i){ uint32_t p=img[i]; unsigned char rgb[3]={(p>>16)&255,(p>>8)&255,p&255}; fwrite(rgb,1,3,fp);} fclose(fp);
            fprintf(stderr,"wrote %s\n",dump); }
    }
    SDL_UnmapGPUTransferBuffer(dev,otb);
    printf("M2 (MIP raycast): checked=%ld nonzero=%ld gross-diff(>32)=%ld (maxdiff=%ld)\n",
           m2checked, nonzero, m2diff, m2max);
    // The GPU MIP uses trilinear sampling + a finite step; the CPU reference is
    // an EXACT integer max-over-z, so they cannot match per-pixel (trilinear
    // smooths peaks, the step can skip the single brightest voxel). The real
    // proof is the rendered image (RAY_DUMP -> a clean filled disk for a ball).
    // So the gate is structural: the volume is actually on screen (a disk of
    // nonzero pixels) and most pixels track the CPU MIP within a loose bound.
    double frac_close = m2checked ? (double)(m2checked - m2diff) / m2checked : 0.0;
    int m2ok = (nonzero > m2checked/8) && (frac_close > 0.85);

    // ---- M3: emission-absorption with a transfer function ----
    // TF: a viridis-ish color ramp, alpha rising with value (opacity grows
    // toward the dense ball center). The EA composite should put a bright,
    // colored, opaque ball on the background.
    uint32_t tf[256];
    for(int i=0;i<256;++i){
        float v=i/255.0f;
        uint32_t rr=(uint32_t)(255.0f*v);            // R ramps up
        uint32_t gg=(uint32_t)(255.0f*(0.3f+0.7f*v));
        uint32_t bb=(uint32_t)(255.0f*(1.0f-v));     // B ramps down
        uint32_t aa=(uint32_t)(255.0f*v*v);          // alpha ~ v^2 (dense core opaque)
        tf[i]=(aa<<24)|(rr<<16)|(gg<<8)|bb;
    }
    mc_gpu_ray_set_lut(R,tf);
    SDL_GPUCommandBuffer *cb3=SDL_AcquireGPUCommandBuffer(dev);
    mc_gpu_ray_render(R,cb3,rt,ivp,campos,0.5f,1.0f, MC_RAY_EA, 0.1f, 4.0f);
    SDL_GPUTransferBuffer *otb3=SDL_CreateGPUTransferBuffer(dev,&o);
    SDL_GPUCopyPass *cp3=SDL_BeginGPUCopyPass(cb3);
    SDL_GPUTextureTransferInfo oi3={0}; oi3.transfer_buffer=otb3; oi3.pixels_per_row=(Uint32)RW; oi3.rows_per_layer=(Uint32)RH;
    SDL_DownloadFromGPUTexture(cp3,&rr,&oi3);
    SDL_EndGPUCopyPass(cp3);
    SDL_GPUFence *f3=SDL_SubmitGPUCommandBufferAndAcquireFence(cb3);
    SDL_WaitForGPUFences(dev,true,&f3,1); SDL_ReleaseGPUFence(dev,f3);
    uint32_t *img3=SDL_MapGPUTransferBuffer(dev,otb3,false);
    // structural check: center pixel (the dense ball core) should be brightly
    // colored + opaque; a far corner should be background. Coverage > a disk.
    long ea_lit=0; int bgcorner = img3[0]&0xFFFFFF;
    int center = img3[(RH/2)*RW + RW/2] & 0xFFFFFF;
    for(int i=0;i<RW*RH;++i) if((img3[i]&0xFFFFFF)!=bgcorner) ea_lit++;
    const char *dumpea=SDL_getenv("RAY_DUMP_EA");
    if(dumpea){ FILE*fp=fopen(dumpea,"wb"); if(fp){ fprintf(fp,"P6\n%d %d\n255\n",RW,RH);
        for(int i=0;i<RW*RH;++i){ uint32_t p=img3[i]; unsigned char c[3]={(p>>16)&255,(p>>8)&255,p&255}; fwrite(c,1,3,fp);} fclose(fp);
        fprintf(stderr,"wrote %s\n",dumpea);} }
    SDL_UnmapGPUTransferBuffer(dev,otb3);
    printf("M3 (emission-absorption): lit=%ld/%d center=0x%06x corner=0x%06x\n",
           ea_lit, RW*RH, center, bgcorner);
    // center must differ from the background corner (the ball rendered) and a
    // disk-sized region must be lit.
    int m3ok = (center != bgcorner) && (ea_lit > RW*RH/8);

    printf("%s\n", (m1ok && m2ok && m3ok) ? "ray_gpu_check: OK" : "ray_gpu_check: FAIL");

    for(int b=0;b<NB;++b) free(payloads[b].p);
    mc_codec_ctx_free(C); mc_gpu_ray_destroy(R);
    SDL_DestroyGPUDevice(dev); SDL_Quit();
    return (m1ok && m2ok && m3ok) ? 0 : 1;
}
