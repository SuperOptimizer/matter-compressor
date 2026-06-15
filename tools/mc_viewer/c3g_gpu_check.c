/* c3g_gpu_check — validate the GPU compute decoder against the CPU oracle.
 *
 * Builds c3g payloads on the CPU (mc_c3g_enc_block), decodes them two ways:
 *   - CPU: mc_c3g_dec_block  (the reference / oracle)
 *   - GPU: the c3g_decode compute shader (one workgroup per block)
 * and compares. The codec is lossy and the GPU inverse DCT is float (the CPU's
 * is integer fixed-point), so the bar is |GPU - CPU| <= 1 per voxel; the exact
 * mismatch distribution is reported. Headless: SDL_VIDEODRIVER=offscreen.
 *
 * Exit 0 if every block agrees within the tolerance, else 1.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <SDL3/SDL.h>

#include "matter_compressor.h"
#include "shaders/c3g_decode.comp.spv.h"

#define N   16
#define N3  4096
#define MAXSYM 16
#define TOTAL  4096

// --- the c3g static rANS table (must match src/matter_compressor.c C3G_FREQ) ---
static const uint16_t C3G_FREQ[MAXSYM] = {
    2200, 980, 360, 180, 110, 70, 48, 34, 24, 18, 14, 10, 8, 6, 5, 21
};

// xorshift for reproducible fixtures
static uint32_t rng = 0x1234567u;
static uint32_t rnd(void){ rng^=rng<<13; rng^=rng>>17; rng^=rng<<5; return rng; }

static int die(const char *m){ fprintf(stderr,"FATAL: %s: %s\n", m, SDL_GetError()); return 1; }

// build the scan table the codec uses (ascending L1 frequency, stable) — must
// match scanS_build for MC_BLK=16. We reconstruct it by the same key.
static int scan_cmp(const void *pa, const void *pb){
    uint32_t a=*(const uint32_t*)pa, b=*(const uint32_t*)pb;
    int az=a/256, ay=(a/16)%16, ax=a%16, bz=b/256, by=(b/16)%16, bx=b%16;
    int al=az+ay+ax, bl=bz+by+bx;
    if(al!=bl) return al-bl;
    return (int)a-(int)b;
}

int main(void){
    if(!SDL_Init(SDL_INIT_VIDEO)) return die("SDL_Init");
    SDL_GPUDevice *dev = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, true, NULL);
    if(!dev) return die("CreateGPUDevice");

    // --- build a batch of c3g payloads on the CPU ---
    mc_codec_ctx *C = mc_codec_ctx_new();
    mc_codec_ctx_set_quality(C, 8.0f);
    mc_codec_ctx_set_max_error(C, 0);

    enum { NB = 64 };
    mc_buf payloads[NB]; uint32_t plen[NB]; int coded[NB];
    uint8_t cpu[NB][N3];
    memset(payloads,0,sizeof payloads);

    for(int bi=0; bi<NB; ++bi){
        uint8_t vox[N3];
        int mode = bi & 3;
        if(mode==0){ for(int i=0;i<N3;++i) vox[i]=(uint8_t)(((rnd()%100)<40)?(rnd()&0xFF):0); }
        else if(mode==1){ for(int z=0;z<N;++z)for(int y=0;y<N;++y)for(int x=0;x<N;++x){
            int d=(z-8)*(z-8)+(y-8)*(y-8)+(x-8)*(x-8); vox[(z*N+y)*N+x]=(uint8_t)(d<50?190-d*3:0);} }
        else if(mode==2){ memset(vox, 100+(bi%80), N3); }              // constant
        else { for(int i=0;i<N3;++i) vox[i]=(uint8_t)((i<N3/2)?0:(120+(rnd()%60))); } // half-air slab
        coded[bi] = mc_c3g_enc_block(C, vox, &payloads[bi], &plen[bi]);
        memset(cpu[bi],0,N3);
        if(coded[bi]) mc_c3g_dec_block(C, payloads[bi].p, plen[bi], cpu[bi]);
    }

    // --- pack payloads contiguously + a [off,len] table (4-byte aligned) ---
    uint32_t blktab[2*NB];
    uint32_t total_bytes=0;
    for(int bi=0;bi<NB;++bi){ uint32_t l=coded[bi]?plen[bi]:0; blktab[2*bi]=total_bytes; blktab[2*bi+1]=l; total_bytes += (l+3)&~3u; }
    uint8_t *packed = calloc(total_bytes?total_bytes:4, 1);
    for(int bi=0;bi<NB;++bi) if(coded[bi]) memcpy(packed+blktab[2*bi], payloads[bi].p, plen[bi]);

    // --- the constant tables the shader needs ---
    uint32_t freq[MAXSYM], cdf[MAXSYM+1], slot[TOTAL];
    uint32_t acc=0; for(int s=0;s<MAXSYM;++s){ freq[s]=C3G_FREQ[s]; cdf[s]=acc; acc+=C3G_FREQ[s]; } cdf[MAXSYM]=acc;
    for(int s=0;s<MAXSYM;++s) for(uint32_t f=cdf[s]; f<cdf[s+1]; ++f) slot[f]=(uint32_t)s;

    // step = quality*(1+(cz+cy+cx))^MC_HF_EXP — MC_HF_EXP/MC_DZ_FRAC are public
    // in matter_compressor.h, so the GPU step_tab matches the CPU exactly.
    float step_tab[N3];
    const float QUAL=8.0f;                       // must match the ctx quality above
    for(int cz=0;cz<N;++cz)for(int cy=0;cy<N;++cy)for(int cx=0;cx<N;++cx){
        int i=(cz*N+cy)*N+cx;
        step_tab[i]=QUAL*powf(1.0f+(float)(cz+cy+cx), MC_HF_EXP);
    }

    uint32_t scan[N3]; for(int i=0;i<N3;++i) scan[i]=i; qsort(scan,N3,sizeof(uint32_t),scan_cmp);

    float cm[N*N]; // inverse basis cm[k*N+n] = ck*cos(pi*(2n+1)*k/2N)
    for(int k=0;k<N;++k){ double ck=(k==0)?sqrt(1.0/N):sqrt(2.0/N);
        for(int n=0;n<N;++n) cm[k*N+n]=(float)(ck*cos(M_PI*(2.0*n+1.0)*k/(2.0*N))); }

    // --- create GPU buffers (helper) ---
    SDL_GPUBuffer *bufs[8]; const void *src[8]; uint32_t sz[8];
    src[0]=packed;   sz[0]=total_bytes?total_bytes:4;
    src[1]=blktab;   sz[1]=sizeof blktab;
    src[2]=freq;     sz[2]=sizeof freq;
    src[3]=cdf;      sz[3]=sizeof cdf;
    src[4]=slot;     sz[4]=sizeof slot;
    src[5]=step_tab; sz[5]=sizeof step_tab;
    src[6]=scan;     sz[6]=sizeof scan;
    src[7]=cm;       sz[7]=sizeof cm;
    for(int i=0;i<8;++i){
        SDL_GPUBufferCreateInfo bc={0}; bc.usage=SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ; bc.size=sz[i];
        bufs[i]=SDL_CreateGPUBuffer(dev,&bc);
        SDL_GPUTransferBufferCreateInfo tc={0}; tc.usage=SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD; tc.size=sz[i];
        SDL_GPUTransferBuffer *tb=SDL_CreateGPUTransferBuffer(dev,&tc);
        void *m=SDL_MapGPUTransferBuffer(dev,tb,false); memcpy(m,src[i],sz[i]); SDL_UnmapGPUTransferBuffer(dev,tb);
        SDL_GPUCommandBuffer *cb=SDL_AcquireGPUCommandBuffer(dev);
        SDL_GPUCopyPass *cp=SDL_BeginGPUCopyPass(cb);
        SDL_GPUTransferBufferLocation s={0}; s.transfer_buffer=tb;
        SDL_GPUBufferRegion d={0}; d.buffer=bufs[i]; d.size=sz[i];
        SDL_UploadToGPUBuffer(cp,&s,&d,false); SDL_EndGPUCopyPass(cp); SDL_SubmitGPUCommandBuffer(cb);
        SDL_ReleaseGPUTransferBuffer(dev,tb);
    }

    // output buffer (read-write) + download transfer buffer
    uint32_t out_bytes = NB*N3;
    SDL_GPUBufferCreateInfo obc={0}; obc.usage=SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE; obc.size=out_bytes;
    SDL_GPUBuffer *obuf=SDL_CreateGPUBuffer(dev,&obc);

    // compute pipeline
    SDL_GPUComputePipelineCreateInfo pc={0};
    pc.code=(const Uint8*)c3g_decode_comp_spv; pc.code_size=sizeof c3g_decode_comp_spv;
    pc.entrypoint="main"; pc.format=SDL_GPU_SHADERFORMAT_SPIRV;
    pc.num_readonly_storage_buffers=8; pc.num_readwrite_storage_buffers=1; pc.num_uniform_buffers=1;
    pc.threadcount_x=256; pc.threadcount_y=1; pc.threadcount_z=1;
    SDL_GPUComputePipeline *pipe=SDL_CreateGPUComputePipeline(dev,&pc);
    if(!pipe) return die("CreateGPUComputePipeline");

    // dispatch: one workgroup per block. The READ-WRITE output buffer is bound
    // at BeginGPUComputePass (storage_buffer_bindings); the READ-ONLY table
    // buffers via BindGPUComputeStorageBuffers.
    SDL_GPUCommandBuffer *cb=SDL_AcquireGPUCommandBuffer(dev);
    SDL_GPUStorageBufferReadWriteBinding rw={0}; rw.buffer=obuf;
    SDL_GPUComputePass *cp=SDL_BeginGPUComputePass(cb,NULL,0,&rw,1);
    SDL_BindGPUComputePipeline(cp,pipe);
    SDL_BindGPUComputeStorageBuffers(cp,0,bufs,8);
    uint32_t nb=NB; SDL_PushGPUComputeUniformData(cb,0,&nb,sizeof nb);
    SDL_DispatchGPUCompute(cp,NB,1,1);
    SDL_EndGPUComputePass(cp);

    // download output
    SDL_GPUTransferBufferCreateInfo dtc={0}; dtc.usage=SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD; dtc.size=out_bytes;
    SDL_GPUTransferBuffer *dtb=SDL_CreateGPUTransferBuffer(dev,&dtc);
    SDL_GPUCopyPass *dcp=SDL_BeginGPUCopyPass(cb);
    SDL_GPUTransferBufferLocation dl={0}; dl.transfer_buffer=dtb;
    SDL_GPUBufferRegion dr={0}; dr.buffer=obuf; dr.size=out_bytes;
    SDL_DownloadFromGPUBuffer(dcp,&dr,&dl);
    SDL_EndGPUCopyPass(dcp);
    SDL_GPUFence *fence=SDL_SubmitGPUCommandBufferAndAcquireFence(cb);
    SDL_WaitForGPUFences(dev,true,&fence,1);
    SDL_ReleaseGPUFence(dev,fence);

    uint8_t *gpu=(uint8_t*)SDL_MapGPUTransferBuffer(dev,dtb,false);

    // --- compare ---
    long hist[4]={0,0,0,0};   // diff 0,1,2,>2
    int bad_blocks=0;
    for(int bi=0;bi<NB;++bi){
        int bb=0;
        for(int i=0;i<N3;++i){
            int d=cpu[bi][i]-gpu[bi*N3+i]; if(d<0)d=-d;
            hist[d<3?d:3]++;
            if(d>1) bb=1;
        }
        if(bb) bad_blocks++;
    }
    SDL_UnmapGPUTransferBuffer(dev,dtb);

    long tot=(long)NB*N3;
    printf("voxels=%ld  diff0=%ld (%.3f%%)  diff1=%ld  diff2=%ld  diff>2=%ld  bad_blocks=%d/%d\n",
        tot, hist[0], 100.0*hist[0]/tot, hist[1], hist[2], hist[3], bad_blocks, NB);

    int ok = (hist[2]==0 && hist[3]==0);
    printf(ok ? "c3g_gpu_check: OK (GPU matches CPU within <=1)\n"
              : "c3g_gpu_check: FAIL (GPU differs from CPU by >1)\n");

    for(int bi=0;bi<NB;++bi) free(payloads[bi].p);
    free(packed);
    mc_codec_ctx_free(C);
    SDL_DestroyGPUDevice(dev);
    SDL_Quit();
    return ok?0:1;
}
