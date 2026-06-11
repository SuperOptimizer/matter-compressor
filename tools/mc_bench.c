// ============================================================================
// mc_bench — codec benchmark harness for matter-compressor.
//
// Loads a raw u8 cube (e.g. from mc_fetch), center-crops a test cube of the
// requested size (128/256/512...), pads it to the 256-chunk grid, encodes it
// through the appendable archive, and reports the full metric basket:
//
//   ratio, PSNR (all + material), MAE, p50/p90/p95/p99, max error, SSIM,
//   encode MB/s, full-decode MB/s, cold-block decode latency.
//
// usage: mc_bench <vol.bin> <voldim> <testdim> <q1>[,q2,...] [max_err_tau]
//   e.g. mc_bench /tmp/vesuvius_512.bin 512 256 1,3,6,12 8
// Synthetic mode: mc_bench --synth <testdim> <q1>[,...]
// ============================================================================
#include "../src/matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef uint8_t u8;
static double now(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

static unsigned rng_s=12345;
static unsigned rng(void){ rng_s^=rng_s<<13; rng_s^=rng_s>>17; rng_s^=rng_s<<5; return rng_s; }

// scroll-like synthetic: wrapped sinusoidal sheets + noise + air gaps
static u8 *synth(int D){
    u8 *v=malloc((size_t)D*D*D);
    for(int z=0;z<D;++z)for(int y=0;y<D;++y)for(int x=0;x<D;++x){
        double r=sqrt((double)(x-D/2)*(x-D/2)+(double)(y-D/2)*(y-D/2));
        double sheet=sin(r*0.45+0.3*sin(z*0.05));
        int val=0;
        if(sheet>-0.25){ val=120+(int)(60*sheet)+(int)(rng()%17)-8; if(val<1)val=1; if(val>255)val=255; }
        v[((size_t)z*D+y)*D+x]=(u8)val;
    }
    return v;
}

// mean SSIM over non-overlapping 8^3 windows (standard C1/C2, dynamic range 255)
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
        acc+=((2*ma*mb+C1)*(2*cab+C2))/((ma*ma+mb*mb+C1)*(va+vb+C2));
        nw++;
    }
    return nw?acc/nw:1.0;
}

static int cmp_chunk_has(const u8 *vol,int P,int cz,int cy,int cx){
    for(int z=0;z<256;++z)for(int y=0;y<256;++y)for(int x=0;x<256;++x)
        if(vol[((size_t)(cz*256+z)*P+(cy*256+y))*P+(cx*256+x)]) return 1;
    return 0;
}

static int g_tau=0, g_deblock=0;
static void run_q(const u8 *vol,int T,int P,float q){
    size_t rawT=(size_t)T*T*T;
    int NC=P/256;
    char path[256]; snprintf(path,sizeof path,"/tmp/mc_bench_%d_%.2f.mc",T,q);
    remove(path);
    mc_archive *a=mc_archive_open(path,P,q);
    mc_codec_ctx *cx_=mc_codec_ctx_new();
    mc_codec_ctx_set_quality(cx_,q);
    mc_codec_ctx_set_max_error(cx_,g_tau);
    static _Thread_local u8 *chunk=0; if(!chunk) chunk=malloc((size_t)256*256*256);

    // ---- encode ----
    double t0=now(); size_t vox_in=0;
    for(int cz=0;cz<NC;++cz)for(int cy=0;cy<NC;++cy)for(int cx=0;cx<NC;++cx){
        for(int z=0;z<256;++z)for(int y=0;y<256;++y)
            memcpy(chunk+((size_t)z*256+y)*256,
                   vol+((size_t)(cz*256+z)*P+(cy*256+y))*P+(size_t)cx*256,256);
        mc_archive_append_chunk_ctx(a,cx_,0,cz,cy,cx,chunk);
        vox_in+=(size_t)256*256*256;
    }
    mc_codec_ctx_free(cx_);
    double enc_s=now()-t0;

    // ---- full decode + metrics ----
    u8 *rec=malloc((size_t)P*P*P); memset(rec,0,(size_t)P*P*P);
    u8 blk[16*16*16];
    t0=now();
    for(int cz=0;cz<NC;++cz)for(int cy=0;cy<NC;++cy)for(int cx=0;cx<NC;++cx){
        uint64_t co=mc_archive_chunk_offset(a,0,cz,cy,cx); if(!co) continue;
        for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
            mc_archive_decode_block(a,co,bz,by,bx,blk);
            for(int z=0;z<16;++z)for(int y=0;y<16;++y)
                memcpy(rec+((size_t)(cz*256+bz*16+z)*P+(cy*256+by*16+y))*P+(size_t)cx*256+bx*16,
                       blk+((size_t)z*16+y)*16,16);
        }
    }
    double dec_s=now()-t0;
    double db_ms=0;
    if(g_deblock){ double td=now(); mc_deblock(rec,P,P,P,q); db_ms=(now()-td)*1e3; }

    // ---- cold-block latency: alternate chunks so the mask cache always misses ----
    uint64_t cos[64]; int nco=0;
    for(int cz=0;cz<NC&&nco<64;++cz)for(int cy=0;cy<NC&&nco<64;++cy)for(int cx=0;cx<NC&&nco<64;++cx){
        uint64_t co=mc_archive_chunk_offset(a,0,cz,cy,cx); if(co) cos[nco++]=co;
    }
    double cold_ms=0; int cold_n=0;
    if(nco>=2){
        t0=now();
        for(int i=0;i<10;++i) mc_archive_decode_block(a,cos[i%nco],8,8,8,blk);
        cold_ms=(now()-t0)*1e3/10; cold_n=10;
    } else if(nco==1){
        // single chunk: append a duplicate at a spare coord to alternate against
        mc_archive_append_chunk_raw(a,1,0,0,0,chunk);
        uint64_t co2=mc_archive_chunk_offset(a,1,0,0,0);
        if(co2){ t0=now();
            for(int i=0;i<10;++i) mc_archive_decode_block(a,(i&1)?co2:cos[0],8,8,8,blk);
            cold_ms=(now()-t0)*1e3/10; cold_n=10; }
    }

    // ---- error stats over the T^3 crop ----
    double se=0,ae=0; long maxe=0; size_t hist[256]={0}; size_t nm=0;
    for(int z=0;z<T;++z)for(int y=0;y<T;++y)for(int x=0;x<T;++x){
        size_t i=((size_t)z*P+y)*P+x;
        int e=abs((int)vol[i]-(int)rec[i]);
        se+=(double)e*e; hist[e]++;
        if(vol[i]||rec[i]){ nm++; ae+=e; if(e>maxe)maxe=e; }
    }
    double mse_all=se/rawT, psnr_all=10*log10(255.0*255.0/(mse_all>0?mse_all:1e-12));
    size_t acc=0; int p50=-1,p90=-1,p95=-1,p99=-1;
    for(int e=0;e<256;++e){ acc+=hist[e];
        if(p50<0&&acc>=(size_t)(0.50*rawT))p50=e;
        if(p90<0&&acc>=(size_t)(0.90*rawT))p90=e;
        if(p95<0&&acc>=(size_t)(0.95*rawT))p95=e;
        if(p99<0&&acc>=(size_t)(0.99*rawT))p99=e; }
    // SSIM on the T^3 crop (sample stride for big volumes to keep it fast)
    double ssim;
    { // build contiguous T^3 copies
        u8 *ca=malloc(rawT), *cb=malloc(rawT);
        for(int z=0;z<T;++z)for(int y=0;y<T;++y){
            memcpy(ca+((size_t)z*T+y)*T, vol+((size_t)z*P+y)*P, T);
            memcpy(cb+((size_t)z*T+y)*T, rec+((size_t)z*P+y)*P, T);
        }
        ssim=ssim3d(ca,cb,T); free(ca); free(cb);
    }
    mc_archive_close(a);
    FILE*f=fopen(path,"rb"); fseek(f,0,SEEK_END); long fsz=ftell(f); fclose(f);
    long data=fsz-128*1024; if(data<1)data=1;
    remove(path);

    printf("T=%d q=%-5.2f ratio %6.1fx | PSNR %6.2f MAE %5.3f p50 %d p90 %d p95 %d p99 %d max %ld SSIM %.4f | enc %6.1f MB/s dec %6.1f MB/s cold %7.3f ms%s\n",
        T,q,(double)rawT/data,psnr_all,nm?ae/nm:0.0,p50,p90,p95,p99,maxe,ssim,
        vox_in/1e6/enc_s, (double)NC*NC*NC*256*256*256/1e6/dec_s, cold_ms, cold_n?"":" (n/a)");
    if(g_deblock) printf("  (deblock %.0f ms)\n", db_ms);
    free(rec);
}

int main(int argc,char**argv){
    int synth_mode = argc>=2 && !strcmp(argv[1],"--synth");
    if(argc < (synth_mode?4:5)){
        fprintf(stderr,"usage: %s <vol.bin> <voldim> <testdim> <q1>[,q2,...]\n"
                       "       %s --synth <testdim> <q1>[,q2,...]\n",argv[0],argv[0]);
        return 1;
    }
    int T, VD=0; const char *qlist; u8 *src=NULL;
    if(synth_mode){ T=atoi(argv[2]); qlist=argv[3]; if(argc>4)g_tau=atoi(argv[4]); }
    else {
        VD=atoi(argv[2]); T=atoi(argv[3]); qlist=argv[4]; if(argc>5)g_tau=atoi(argv[5]); if(argc>6)g_deblock=atoi(argv[6]);
        if(T>VD){ fprintf(stderr,"testdim > voldim\n"); return 1; }
        FILE*f=fopen(argv[1],"rb"); if(!f){ perror("open vol"); return 1; }
        src=malloc((size_t)VD*VD*VD);
        if(fread(src,1,(size_t)VD*VD*VD,f)!=(size_t)VD*VD*VD){ fprintf(stderr,"short read\n"); return 1; }
        fclose(f);
    }
    int P=((T+255)/256)*256;                     // padded archive dim
    u8 *vol=calloc((size_t)P*P*P,1);
    if(synth_mode){
        u8 *s=synth(T);
        for(int z=0;z<T;++z)for(int y=0;y<T;++y)
            memcpy(vol+((size_t)z*P+y)*P, s+((size_t)z*T+y)*T, T);
        free(s);
    } else {
        int off=(VD-T)/2;                        // center crop
        for(int z=0;z<T;++z)for(int y=0;y<T;++y)
            memcpy(vol+((size_t)z*P+y)*P, src+((size_t)(off+z)*VD+(off+y))*VD+off, T);
        free(src);
    }
    size_t zeros=0; for(int z=0;z<T;++z)for(int y=0;y<T;++y)for(int x=0;x<T;++x) zeros+=!vol[((size_t)z*P+y)*P+x];
    printf("# %s T=%d (pad %d) air %.1f%% tau %d\n", synth_mode?"synthetic":argv[1], T, P, 100.0*zeros/((size_t)T*T*T), g_tau);

    char *ql=strdup(qlist);
    for(char *tok=strtok(ql,",");tok;tok=strtok(NULL,",")) run_q(vol,T,P,(float)atof(tok));
    free(ql); free(vol);
    return 0;
}
