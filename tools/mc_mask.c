// ============================================================================
// mc_mask — fysics-style aggressive air masking for raw u8 scroll volumes.
//
// Port of the air-zero stage of SuperOptimizer/fysics (pipeline.c step (d) +
// stream.c fy_valley_depth): box-smooth a scratch copy, find the air/papyrus
// histogram valley on the smoothed samples, cut at
//     cut = phys_floor + aggr * (valley - phys_floor)
// (aggr 0 = conservative physics floor, 1 = aggressive valley cut), then zero
// every voxel whose SMOOTHED value is below the cut. Best tuned for the
// PHerc Paris 4 data; self-calibrating on anything with a bimodal histogram.
//
// usage: mc_mask <in.bin> <dim> <out.bin> [aggr=1.0] [phys_floor_u8=39]
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef uint8_t u8;

// separable box blur, radius r, float volume (fysics box scratch)
static void box_blur(float *v, float *tmp, int D, int r){
    // X
    for(int z=0;z<D;++z)for(int y=0;y<D;++y){ size_t row=((size_t)z*D+y)*D;
        double s=0; int c=0;
        for(int x=0;x<=r&&x<D;++x){ s+=v[row+x]; c++; }
        for(int x=0;x<D;++x){
            tmp[row+x]=(float)(s/c);
            int add=x+r+1, del=x-r;
            if(add<D){ s+=v[row+add]; c++; }
            if(del>=0){ s-=v[row+del]; c--; }
        }
    }
    // Y
    for(int z=0;z<D;++z)for(int x=0;x<D;++x){
        double s=0; int c=0;
        for(int y=0;y<=r&&y<D;++y){ s+=tmp[((size_t)z*D+y)*D+x]; c++; }
        for(int y=0;y<D;++y){
            v[((size_t)z*D+y)*D+x]=(float)(s/c);
            int add=y+r+1, del=y-r;
            if(add<D){ s+=tmp[((size_t)z*D+add)*D+x]; c++; }
            if(del>=0){ s-=tmp[((size_t)z*D+del)*D+x]; c--; }
        }
    }
    // Z
    for(int y=0;y<D;++y)for(int x=0;x<D;++x){
        double s=0; int c=0;
        for(int z=0;z<=r&&z<D;++z){ s+=v[((size_t)z*D+y)*D+x]; c++; }
        for(int z=0;z<D;++z){
            tmp[((size_t)z*D+y)*D+x]=(float)(s/c);
            int add=z+r+1, del=z-r;
            if(add<D){ s+=v[((size_t)add*D+y)*D+x]; c++; }
            if(del>=0){ s-=v[((size_t)del*D+y)*D+x]; c--; }
        }
    }
    memcpy(v,tmp,sizeof(float)*(size_t)D*D*D);
}

// fysics fy_valley_depth (stream.c): smooth hist, two tallest peaks, valley between.
static double valley_depth(const long hist[256], int *dark, int *light, int *valley){
    double hf[256];
    for(int i=0;i<256;++i){ double s=0; int c=0;
        for(int k=-2;k<=2;++k){ int j=i+k; if(j>=0&&j<256){ s+=hist[j]; c++; } } hf[i]=s/c; }
    hf[0]=hf[254]=hf[255]=0;
    double mx=0; for(int i=0;i<256;++i) if(hf[i]>mx) mx=hf[i];
    if(mx<=0) return -1;
    int peaks[256], np=0;
    for(int u=3;u<253;++u)
        if(hf[u]>=hf[u-1]&&hf[u]>=hf[u+1]&&hf[u]>0.05*mx){
            if(np&&u-peaks[np-1]<=10){ if(hf[u]>hf[peaks[np-1]]) peaks[np-1]=u; }
            else peaks[np++]=u;
        }
    if(np<2) return -1;
    int p1=peaks[0],p2=peaks[1];
    for(int i=0;i<np;++i){
        if(hf[peaks[i]]>hf[p1]){ p2=p1; p1=peaks[i]; }
        else if(hf[peaks[i]]>hf[p2]&&peaks[i]!=p1) p2=peaks[i];
    }
    int a=p1<p2?p1:p2, b=p1<p2?p2:p1;
    int v=a; double vmin=hf[a];
    for(int u=a;u<=b;++u) if(hf[u]<vmin){ vmin=hf[u]; v=u; }
    *dark=a; *light=b; *valley=v;
    double mn=hf[a]<hf[b]?hf[a]:hf[b];
    return 1.0-vmin/(mn+1e-12);
}

int main(int argc,char**argv){
    if(argc<4){ fprintf(stderr,"usage: %s <in.bin> <dim> <out.bin> [aggr=1.0] [phys_floor_u8=39]\n",argv[0]); return 1; }
    int D=atoi(argv[2]);
    double aggr = argc>4?atof(argv[4]):1.0; if(aggr<0)aggr=0; if(aggr>1)aggr=1;
    int pf = argc>5?atoi(argv[5]):39;
    size_t n=(size_t)D*D*D;
    u8 *raw=malloc(n);
    FILE*f=fopen(argv[1],"rb"); if(!f){perror("in");return 1;}
    if(fread(raw,1,n,f)!=n){fprintf(stderr,"short read\n");return 1;} fclose(f);

    float *sm=malloc(sizeof(float)*n), *tmp=malloc(sizeof(float)*n);
    for(size_t i=0;i<n;++i) sm[i]=raw[i]/255.0f;
    box_blur(sm,tmp,D,5);                        // fysics scratch smooth (r=5)

    long hist[256]={0};
    for(size_t i=0;i<n;++i){ int b=(int)(sm[i]*255.0f+0.5f); if(b<0)b=0; if(b>255)b=255; hist[b]++; }
    int dark=0,light=0,valley=0;
    double d=valley_depth(hist,&dark,&light,&valley);
    int cut;
    if(d>=0 && valley>pf) cut=(int)(pf+aggr*(valley-pf)+0.5);
    else cut=pf;
    printf("hist: dark %d light %d valley %d depth %.2f -> cut %d (aggr %.2f)\n",dark,light,valley,d,cut,aggr);

    float cutf=cut/255.0f; size_t zeroed=0;
    for(size_t i=0;i<n;++i) if(sm[i]<cutf){ raw[i]=0; zeroed++; }
    // also clamp: material must stay nonzero (0 is reserved for air)
    size_t bumped=0;
    for(size_t i=0;i<n;++i) if(raw[i]==0 && sm[i]>=cutf){ raw[i]=1; bumped++; }

    f=fopen(argv[3],"wb"); if(!f){perror("out");return 1;}
    fwrite(raw,1,n,f); fclose(f);
    printf("wrote %s: air %.1f%% (zeroed %zu, bumped %zu)\n",argv[3],100.0*zeroed/n,zeroed,bumped);
    free(raw); free(sm); free(tmp);
    return 0;
}
