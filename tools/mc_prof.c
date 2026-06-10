// mc_prof — block-codec timing on a real volume: black-box enc/dec + DCT micro.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "../src/mc_codec.c"   // private copy: reach mc_dct3_* and internals

static double now(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

int main(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s <vol.bin> <dim> [q]\n",argv[0]); return 1; }
    int D=atoi(argv[2]); float q=argc>3?(float)atof(argv[3]):6.0f;
    size_t n=(size_t)D*D*D;
    mc_u8 *vol=malloc(n);
    FILE*f=fopen(argv[1],"rb"); if(!f){perror("in");return 1;}
    if(fread(vol,1,n,f)!=n){fprintf(stderr,"short\n");return 1;} fclose(f);
    mc_codec_init(); mc_set_quality(q);

    int NB=D/MC_BLK;
    static mc_u8 vox[N3], dec[N3];
    mc_buf out={0};
    // collect payload offsets for decode pass
    typedef struct { size_t off; uint32_t len; } rec_t;
    rec_t *recs=malloc(sizeof(rec_t)*(size_t)NB*NB*NB); int nrec=0;
    long nblk=0;
    double t0=now();
    for(int bz=0;bz<NB;++bz)for(int by=0;by<NB;++by)for(int bx=0;bx<NB;++bx){
        for(int z=0;z<MC_BLK;++z)for(int y=0;y<MC_BLK;++y)
            memcpy(vox+((size_t)z*MC_BLK+y)*MC_BLK, vol+((size_t)(bz*MC_BLK+z)*D+(by*MC_BLK+y))*D+bx*MC_BLK, MC_BLK);
        uint32_t len=0; size_t off=out.len;
        if(mc_enc_block(vox,&out,&len)){ recs[nrec].off=off; recs[nrec].len=len; nrec++; nblk++; }
    }
    double t_enc=now()-t0;
    t0=now();
    for(int r=0;r<nrec;++r) mc_dec_block(out.p+recs[r].off,recs[r].len,dec);
    double t_dec=now()-t0;

    // DCT micro
    static float a[N3],b[N3];
    for(int i=0;i<N3;++i)a[i]=(float)((i*2654435761u)%255)-127;
    t0=now(); for(int r=0;r<20000;++r){ mc_dct3_fwd(a,b); } double t_f=(now()-t0)/20000;
    t0=now(); for(int r=0;r<20000;++r){ mc_dct3_inv(b,a); } double t_i=(now()-t0)/20000;

    printf("blocks %ld  payload %.1f MB\n",nblk,out.len/1e6);
    printf("encode %.2fs (%.1f MB/s)  decode %.2fs (%.1f MB/s)\n",t_enc,n/1e6/t_enc,t_dec,n/1e6/t_dec);
    printf("fwdDCT %.1f us/blk (%.0f%% of enc)   invDCT %.1f us/blk (%.0f%% of dec)\n",
        t_f*1e6, 100.0*t_f*nblk/t_enc, t_i*1e6, 100.0*t_i*nblk/t_dec);
    return 0;
}
