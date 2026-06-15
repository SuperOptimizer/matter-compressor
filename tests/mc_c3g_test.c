// mc_c3g_test — the c3g (GPU-amenable) block codec must agree with the existing
// CABAC decode on the SAME source, and must round-trip deterministically.
//
// c3g reuses the production quant (it recovers mc_enc_block's exact levels/dc/
// air) and the same dequant + inverse DCT back-end, replacing only the entropy
// stage (CABAC -> static rANS). So with corrections off (max_err==0):
//     mc_c3g_dec_block(mc_c3g_enc_block(vox))  ==  mc_dec_block(mc_enc_block(vox))
// for every 16^3 block. This test asserts that exact equality and that c3g
// encode is deterministic (byte-identical across runs). The CPU decoder here is
// the oracle the GPU compute decoder is later checked against.
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N3 (16*16*16)
static int fails = 0;
#define CHECK(x) do{ if(!(x)){ fails++; \
    fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);} }while(0)

// deterministic xorshift so the fixtures are reproducible
static uint32_t rng = 0x9e3779b9u;
static uint32_t rnd(void){ rng^=rng<<13; rng^=rng>>17; rng^=rng<<5; return rng; }

static void fill_random(uint8_t *b, int density){    // density% nonzero
    for(int i=0;i<N3;++i) b[i] = ((int)(rnd()%100) < density) ? (uint8_t)(rnd()&0xFF) : 0;
}
static void fill_smooth(uint8_t *b){                 // a low-frequency gradient blob
    for(int z=0;z<16;++z)for(int y=0;y<16;++y)for(int x=0;x<16;++x){
        int d=(z-8)*(z-8)+(y-8)*(y-8)+(x-8)*(x-8);
        b[(z*16+y)*16+x]=(uint8_t)(d<48? 200 - d*3 : 0);   // sphere of material
    }
}

static int run_block(mc_codec_ctx *C, const uint8_t *vox, const char *name){
    int bad=0;
    // reference: CABAC encode -> decode
    mc_buf cb={0}; uint32_t clen=0;
    int ccoded=mc_enc_block(C,vox,&cb,&clen);
    uint8_t ref[N3]; memset(ref,0,N3);
    if(ccoded) mc_dec_block(C,cb.p,clen,ref);

    // c3g encode -> decode
    mc_buf gb={0}; uint32_t glen=0;
    int gcoded=mc_c3g_enc_block(C,vox,&gb,&glen);
    uint8_t got[N3]; memset(got,0,N3);
    if(gcoded) mc_c3g_dec_block(C,gb.p,glen,got);

    CHECK(ccoded==gcoded);
    if(ccoded){
        int diff=0, maxd=0;
        for(int i=0;i<N3;++i){ int d=ref[i]-got[i]; if(d<0)d=-d; if(d){diff++; if(d>maxd)maxd=d;} }
        if(diff){ bad=1; fails++;
            fprintf(stderr,"FAIL %s: c3g != CABAC decode (%d voxels differ, maxdiff=%d)\n",
                    name,diff,maxd); }
        // determinism: c3g encode is byte-identical on a re-run
        mc_buf gb2={0}; uint32_t glen2=0; mc_c3g_enc_block(C,vox,&gb2,&glen2);
        if(glen2!=glen || memcmp(gb.p,gb2.p,glen)){ bad=1; fails++;
            fprintf(stderr,"FAIL %s: c3g encode not deterministic\n",name); }
        free(gb2.p);
        if(!bad) printf("  %-22s OK (clen=%u glen=%u)\n",name,clen,glen);
    } else {
        printf("  %-22s OK (all-air)\n",name);
    }
    free(cb.p); free(gb.p);
    return bad;
}

int main(void){
    mc_codec_ctx *C = mc_codec_ctx_new();
    CHECK(C!=NULL);
    mc_codec_ctx_set_quality(C, 8.0f);
    mc_codec_ctx_set_max_error(C, 0);     // c3g targets corrections-off

    uint8_t blk[N3];

    // all-air
    memset(blk,0,N3); run_block(C,blk,"all-air");
    // coherent half-air slab (air RLE branch wins) + incoherent checkerboard
    // air (flat-bitmask branch wins) — exercise both air encodings.
    for(int i=0;i<N3;++i) blk[i]=(i<(N3/2))?0:160;          // bottom half air
    run_block(C,blk,"half-air-slab");
    for(int i=0;i<N3;++i) blk[i]=((i^(i>>4)^(i>>8))&1)?180:0; // scattered air
    run_block(C,blk,"checkerboard-air");
    // constant (dc only, no air)
    memset(blk,128,N3); run_block(C,blk,"constant-128");
    // smooth blob (low-frequency + air)
    fill_smooth(blk); run_block(C,blk,"smooth-sphere");
    // random densities (exercise EOB, escapes, signs, masks)
    for(int d=5; d<=100; d+=15){ fill_random(blk,d);
        char nm[32]; snprintf(nm,sizeof nm,"random-%d%%",d); run_block(C,nm[0]?blk:blk,nm); }
    // a few different quality levels
    for(float q=2.0f; q<=16.0f; q*=2.0f){
        mc_codec_ctx_set_quality(C,q);
        fill_smooth(blk);
        char nm[32]; snprintf(nm,sizeof nm,"smooth q=%.0f",q); run_block(C,blk,nm);
        fill_random(blk,40);
        snprintf(nm,sizeof nm,"random40 q=%.0f",q); run_block(C,blk,nm);
    }

    mc_codec_ctx_free(C);
    printf(fails ? "mc_c3g_test: %d FAILED\n" : "mc_c3g_test: OK\n", fails);
    return fails ? 1 : 0;
}
