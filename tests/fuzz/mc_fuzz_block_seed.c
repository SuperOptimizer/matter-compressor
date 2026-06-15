// mc_fuzz_block_seed — emit valid block-codec payloads as AFL++ seeds for
// mc_fuzz_block. Each seed is [q_byte][block payload]: a real 16^3 block encoded
// at a given quality, so the fuzzer mutates structurally-valid range-coder
// bitstreams instead of bouncing off the header bins.
//
//   usage: mc_fuzz_block_seed <corpus_dir>
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void emit(const char *dir, int qi, float q, int kind){
    mc_codec_ctx *cx = mc_codec_ctx_new();
    mc_codec_ctx_set_quality(cx, q);
    mc_u8 vox[16*16*16];
    for(int z=0;z<16;++z)for(int y=0;y<16;++y)for(int x=0;x<16;++x){
        int i=(z*16+y)*16+x;
        switch(kind){
            case 0: vox[i]=(mc_u8)(40+((i*7)%160)); break;           // textured material
            case 1: vox[i]=128; break;                               // constant (DC) block
            case 2: { double r=sqrt((x-8)*(x-8)+(y-8)*(y-8)+(z-8)*(z-8));
                      vox[i]=r>6?0:(mc_u8)(200-r*10); } break;        // mixed air/material
        }
    }
    mc_buf b={0}; uint32_t plen=0;
    mc_enc_block(cx,vox,&b,&plen);
    char p[256]; snprintf(p,sizeof p,"%s/blk_q%.0f_k%d",dir,q,kind);
    FILE*f=fopen(p,"wb");
    if(f){ fputc((int)q,f); if(plen) fwrite(b.p,1,plen,f); fclose(f); }
    free(b.p);
    mc_codec_ctx_free(cx);
    (void)qi;
}

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s <corpus_dir>\n",argv[0]); return 2; }
    float qs[3]={1,6,32};
    for(int qi=0;qi<3;++qi) for(int k=0;k<3;++k) emit(argv[1],qi,qs[qi],k);
    printf("block seeds in %s\n",argv[1]);
    return 0;
}
