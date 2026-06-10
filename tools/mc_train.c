// ============================================================================
// mc_train — train the range-coder context priors on a real volume.
//
// Compiles an instrumented private copy of the codec (MC_TRAIN), encodes the
// given volume at several qualities, and prints RC_PRIOR_* tables to paste
// into src/mc_rangecoder.h.
//
// usage: mc_train <vol.bin> <dim> [q1,q2,...]
// ============================================================================
#define MC_TRAIN 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

long mc_tr_n[8][32], mc_tr_z[8][32];   // [class][slot]: total bins, zero bins

#include "../src/matter_compressor.c"           // instrumented copy (MC_TRAIN defined)

static const char *CLS_NAME[8]={"SIG","MAG","EOB","MASK","MASKU","MASKA","FLAG","DC"};
static const int  CLS_SLOTS[8]={32,16,16,16,4,2,4,8};

int main(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s <vol.bin> <dim> [q1,q2,...]\n",argv[0]); return 1; }
    int D=atoi(argv[2]);
    const char *qlist = argc>3?argv[3]:"1,3,6,12";
    size_t n=(size_t)D*D*D;
    mc_codec_init();
    mc_buf out={0};
    static mc_u8 blk[MC_BLK*MC_BLK*MC_BLK];
    char *vols=strdup(argv[1]);
    for(char *vp=strtok(vols,",");vp;vp=strtok(NULL,",")){
        mc_u8 *vol=malloc(n);
        FILE*f=fopen(vp,"rb"); if(!f){perror(vp);return 1;}
        if(fread(vol,1,n,f)!=n){fprintf(stderr,"short read\n");return 1;} fclose(f);
        char *ql=strdup(qlist);
        for(char *tok=strtok_r(ql,",",&ql);tok;tok=strtok_r(NULL,",",&ql)){
            mc_set_quality((float)atof(tok));
            for(int bz=0;bz+MC_BLK<=D;bz+=MC_BLK)
            for(int by=0;by+MC_BLK<=D;by+=MC_BLK)
            for(int bx=0;bx+MC_BLK<=D;bx+=MC_BLK){
                for(int z=0;z<MC_BLK;++z)for(int y=0;y<MC_BLK;++y)
                    memcpy(blk+((size_t)z*MC_BLK+y)*MC_BLK,
                           vol+((size_t)(bz+z)*D+(by+y))*D+bx, MC_BLK);
                uint32_t len=0; out.len=0;
                mc_enc_block(blk,&out,&len);
            }
            fprintf(stderr,"%s q=%s done\n",vp,tok);
        }
        free(vol);
    }
    for(int c=0;c<8;++c){
        int ns=CLS_SLOTS[c];
        printf("static const uint16_t RC_PRIOR_%s[%d] ={",CLS_NAME[c],ns);
        for(int s=0;s<ns;++s){
            long N=mc_tr_n[c][s], Z=mc_tr_z[c][s];
            int p0 = N? (int)((double)Z/N*4096.0+0.5) : 2048;
            if(p0<32)p0=32; if(p0>4064)p0=4064;
            printf("%d%s",p0,s<ns-1?",":"");
        }
        printf("};\n");
    }
    for(int c=0;c<8;++c){
        fprintf(stderr,"# %s bins:",CLS_NAME[c]);
        for(int s=0;s<CLS_SLOTS[c];++s) fprintf(stderr," %ld",mc_tr_n[c][s]);
        fprintf(stderr,"\n");
    }
    return 0;
}
