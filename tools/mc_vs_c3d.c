// mc_vs_c3d — head-to-head: matter-compressor vs SuperOptimizer/c3d on the
// same masked scroll volume, same metric basket, at iso-rate (c3d is driven
// at the ratio mc achieved for each quality).
//
// build: cc -O3 -march=native -w -o build/mc_vs_c3d tools/mc_vs_c3d.c \
//          build/libmatter_compressor.a /tmp/c3d/build/libc3d.a -lm -lpthread -I/tmp/c3d
#include "../src/mc_archive_api.h"
#include "c3d.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef uint8_t u8;
static double now(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

static double ssim3d(const u8 *a, const u8 *b, int D){
    const int W=8; const double C1=6.5025, C2=58.5225;
    double acc=0; long nw=0;
    for(int z=0;z+W<=D;z+=W)for(int y=0;y+W<=D;y+=W)for(int x=0;x+W<=D;x+=W){
        double sa=0,sb=0,saa=0,sbb=0,sab=0; const int n=W*W*W;
        for(int dz=0;dz<W;++dz)for(int dy=0;dy<W;++dy)for(int dx=0;dx<W;++dx){
            double va=a[((size_t)(z+dz)*D+(y+dy))*D+(x+dx)];
            double vb=b[((size_t)(z+dz)*D+(y+dy))*D+(x+dx)];
            sa+=va; sb+=vb; saa+=va*va; sbb+=vb*vb; sab+=va*vb;
        }
        double ma=sa/n, mb=sb/n;
        double va=saa/n-ma*ma, vb=sbb/n-mb*mb, cab=sab/n-ma*mb;
        acc+=((2*ma*mb+C1)*(2*cab+C2))/((ma*ma+mb*mb+C1)*(va+vb+C2)); nw++;
    }
    return nw?acc/nw:1.0;
}

static void metrics(const char*tag,const u8*orig,const u8*rec,int D,size_t bytes,
                    double enc_s,double dec_s){
    size_t n=(size_t)D*D*D;
    double se=0,ae=0; long maxe=0; size_t hist[256]={0}; size_t nm=0;
    for(size_t i=0;i<n;++i){
        int e=abs((int)orig[i]-(int)rec[i]);
        se+=(double)e*e; hist[e]++;
        if(orig[i]||rec[i]){ nm++; ae+=e; if(e>maxe)maxe=e; }
    }
    double psnr=10*log10(255.0*255.0/(se/n>0?se/n:1e-12));
    size_t acc=0; int p50=-1,p90=-1,p95=-1,p99=-1;
    for(int e=0;e<256;++e){ acc+=hist[e];
        if(p50<0&&acc>=(size_t)(0.50*n))p50=e; if(p90<0&&acc>=(size_t)(0.90*n))p90=e;
        if(p95<0&&acc>=(size_t)(0.95*n))p95=e; if(p99<0&&acc>=(size_t)(0.99*n))p99=e; }
    printf("%-4s ratio %6.1fx | PSNR %6.2f MAE %5.3f p50 %d p90 %d p95 %d p99 %d max %ld SSIM %.4f | enc %6.1f MB/s dec %6.1f MB/s\n",
        tag,(double)n/bytes,psnr,nm?ae/nm:0,p50,p90,p95,p99,maxe,ssim3d(orig,rec,D),
        n/1e6/enc_s,n/1e6/dec_s);
}

int main(int argc,char**argv){
    if(argc<4){ fprintf(stderr,"usage: %s <vol.bin> <dim:multiple of 256> <q1,q2,...>\n",argv[0]); return 1; }
    int D=atoi(argv[2]); size_t n=(size_t)D*D*D; int NC=D/256;
    u8 *vol=malloc(n);
    FILE*f=fopen(argv[1],"rb"); if(!f){perror("in");return 1;}
    if(fread(vol,1,n,f)!=n){fprintf(stderr,"short\n");return 1;} fclose(f);

    u8 *rec=malloc(n);
    u8 *chunk=malloc((size_t)256*256*256);
    u8 *cout=malloc(C3D_CHUNK_ENCODE_MAX_SIZE);

    char *ql=strdup(argv[3]);
    for(char *tok=strtok(ql,",");tok;tok=strtok(NULL,",")){
        float q=(float)atof(tok);
        // ---- matter-compressor ----
        char path[128]; snprintf(path,sizeof path,"/tmp/mcvs_%g.mc",q); remove(path);
        mc_archive *a=mc_archive_open(path,D,q);
        double t0=now();
        for(int cz=0;cz<NC;++cz)for(int cy=0;cy<NC;++cy)for(int cx=0;cx<NC;++cx){
            for(int z=0;z<256;++z)for(int y=0;y<256;++y)
                memcpy(chunk+((size_t)z*256+y)*256,
                       vol+((size_t)(cz*256+z)*D+(cy*256+y))*D+(size_t)cx*256,256);
            mc_archive_append_chunk_raw(a,0,cz,cy,cx,chunk);
        }
        double mc_enc=now()-t0;
        memset(rec,0,n);
        u8 blk[16*16*16];
        t0=now();
        for(int cz=0;cz<NC;++cz)for(int cy=0;cy<NC;++cy)for(int cx=0;cx<NC;++cx){
            uint64_t co=mc_archive_chunk_offset(a,0,cz,cy,cx); if(!co) continue;
            for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
                mc_archive_decode_block(a,co,bz,by,bx,blk);
                for(int z=0;z<16;++z)for(int y=0;y<16;++y)
                    memcpy(rec+((size_t)(cz*256+bz*16+z)*D+(cy*256+by*16+y))*D+(size_t)cx*256+bx*16,
                           blk+((size_t)z*16+y)*16,16);
            }
        }
        double mc_dec=now()-t0;
        mc_archive_close(a);
        FILE*g=fopen(path,"rb"); fseek(g,0,SEEK_END); long fsz=ftell(g)-128*1024; fclose(g); remove(path);
        printf("--- q=%g ---\n",q);
        metrics("mc",vol,rec,D,(size_t)fsz,mc_enc,mc_dec);
        double mc_ratio=(double)n/fsz;

        // ---- c3d at the same rate ----
        c3d_encoder *e=c3d_encoder_new(); c3d_decoder *d=c3d_decoder_new();
        u8 **blobs=malloc(sizeof(u8*)*NC*NC*NC); size_t *blens=malloc(sizeof(size_t)*NC*NC*NC);
        size_t total=0; int ci=0;
        t0=now();
        for(int cz=0;cz<NC;++cz)for(int cy=0;cy<NC;++cy)for(int cx=0;cx<NC;++cx){
            for(int z=0;z<256;++z)for(int y=0;y<256;++y)
                memcpy(chunk+((size_t)z*256+y)*256,
                       vol+((size_t)(cz*256+z)*D+(cy*256+y))*D+(size_t)cx*256,256);
            size_t len=c3d_encoder_chunk_encode(e,chunk,(float)mc_ratio,cout,C3D_CHUNK_ENCODE_MAX_SIZE);
            blobs[ci]=malloc(len); memcpy(blobs[ci],cout,len); blens[ci]=len; total+=len; ci++;
        }
        double c_enc=now()-t0;
        memset(rec,0,n); ci=0;
        t0=now();
        for(int cz=0;cz<NC;++cz)for(int cy=0;cy<NC;++cy)for(int cx=0;cx<NC;++cx){
            c3d_decoder_chunk_decode(d,blobs[ci],blens[ci],chunk); ci++;
            for(int z=0;z<256;++z)for(int y=0;y<256;++y)
                memcpy(rec+((size_t)(cz*256+z)*D+(cy*256+y))*D+(size_t)cx*256,
                       chunk+((size_t)z*256+y)*256,256);
        }
        double c_dec=now()-t0;
        metrics("c3d",vol,rec,D,total,c_enc,c_dec);
        for(int i=0;i<NC*NC*NC;++i) free(blobs[i]);
        free(blobs); free(blens);
        c3d_encoder_free(e); c3d_decoder_free(d);
    }
    return 0;
}
