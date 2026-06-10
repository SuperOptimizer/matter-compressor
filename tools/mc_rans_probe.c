// mc_rans_probe — decide the interleaved-rANS question with data.
//
// Part A (ratio, clock-independent): for every block of a real volume,
// compare the ACTUAL coded size of the current adaptive coefficient stage
// against the Shannon size of a static per-band symbol model (zigzag levels
// 0..63 + EG escape, the c3d-style alphabet) trained on the same corpus.
// rANS achieves ~the static entropy, so this bounds the ratio cost of
// switching the coefficient stage to static-table interleaved rANS.
//
// Part B (throughput, same-session relative): decode 20M symbols through a
// real 4-way interleaved rANS decoder (12-bit static CDF) vs 20M adaptive
// bins through the production dec_bit. Both run under the same clocks, so
// the RATIO of the two numbers is meaningful even in low-power mode.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include "../src/mc_codec.c"

static double now(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static inline uint32_t zigzag(int32_t v){ return (uint32_t)((v<<1)^(v>>31)); }

#define NSYM 65   // zigzag 0..63 + escape
static double hist_band[NB_BANDS][NSYM];

// ---- minimal 4-way interleaved rANS (12-bit), encode-reverse/decode-forward ----
#define RANS_PROB_BITS 12
#define RANS_PROB (1u<<RANS_PROB_BITS)
#define RANS_L (1u<<23)
typedef struct { uint16_t freq[NSYM], cum[NSYM+1]; uint8_t slot2sym[RANS_PROB]; } rans_tab;
static void rans_tab_build(rans_tab*t,const double*h){
    double tot=0; for(int s=0;s<NSYM;++s) tot+=h[s]+0.5;
    uint32_t acc=0;
    for(int s=0;s<NSYM;++s){ uint32_t f=(uint32_t)((h[s]+0.5)/tot*RANS_PROB); if(f<1)f=1; t->freq[s]=(uint16_t)f; acc+=f; }
    while(acc!=RANS_PROB){ // fix rounding on the most probable symbol
        int mx=0; for(int s=1;s<NSYM;++s) if(t->freq[s]>t->freq[mx])mx=s;
        if(acc>RANS_PROB){ t->freq[mx]-=(uint16_t)(acc-RANS_PROB); acc=RANS_PROB; }
        else { t->freq[mx]+=(uint16_t)(RANS_PROB-acc); acc=RANS_PROB; }
    }
    t->cum[0]=0; for(int s=0;s<NSYM;++s) t->cum[s+1]=t->cum[s]+t->freq[s];
    for(int s=0;s<NSYM;++s) for(uint32_t i=t->cum[s];i<t->cum[s+1];++i) t->slot2sym[i]=(uint8_t)s;
}
static size_t rans_encode(const rans_tab*t,const uint8_t*sym,size_t n,uint8_t*out,size_t cap){
    uint32_t st[4]={RANS_L,RANS_L,RANS_L,RANS_L};
    uint8_t *p=out+cap;                       // emit backwards
    for(size_t i=n;i-->0;){
        int lane=(int)(i&3); uint8_t s=sym[i];
        uint32_t f=t->freq[s], c=t->cum[s];
        uint32_t x=st[lane];
        uint32_t xmax=(f<<(23-RANS_PROB_BITS))<<8;
        while(x>=xmax){ *--p=(uint8_t)x; x>>=8; }
        st[lane]=((x/f)<<RANS_PROB_BITS)+(x%f)+c;
    }
    for(int l=3;l>=0;--l){ uint32_t x=st[l]; for(int b=0;b<4;++b){ *--p=(uint8_t)x; x>>=8; } }
    size_t len=(size_t)(out+cap-p);
    memmove(out,p,len);
    return len;
}
static void rans_decode(const rans_tab*t,const uint8_t*in,size_t inlen,uint8_t*sym,size_t n){
    const uint8_t*p=in;
    uint32_t st[4];
    for(int l=0;l<4;++l){
        uint32_t x=((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|p[3]; p+=4; st[l]=x; }
    for(size_t i=0;i<n;++i){
        int lane=(int)(i&3);
        uint32_t x=st[lane];
        uint32_t slot=x&(RANS_PROB-1);
        uint8_t s=t->slot2sym[slot];
        sym[i]=s;
        x=(uint32_t)t->freq[s]*(x>>RANS_PROB_BITS)+slot-t->cum[s];
        while(x<RANS_L) x=(x<<8)|*p++;
        st[lane]=x;
    }
}

int main(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s <vol.bin[,vol2]> <dim> [q]\n",argv[0]); return 1; }
    int D=atoi(argv[2]); float q=argc>3?(float)atof(argv[3]):6.0f;
    size_t n=(size_t)D*D*D;
    mc_codec_init(); mc_set_quality(q);

    // pass 1: collect per-band symbol histograms + actual adaptive coded size
    static rc_i16 ql[N3]; static float blkf[N3], coef[N3];
    static mc_u8 vox[N3];
    static rc_u8 scratch[N3*4+1024];
    double bits_adaptive=0, bits_static=0;
    long nblk=0; size_t nsym_tot=0;
    char *vols=strdup(argv[1]);
    int pass;
    for(pass=0;pass<2;++pass){
        char *vv=strdup(argv[1]);
        for(char *vp=strtok(vv,",");vp;vp=strtok(NULL,",")){
            mc_u8 *vol=malloc(n);
            FILE*f=fopen(vp,"rb"); if(!f){perror(vp);return 1;}
            if(fread(vol,1,n,f)!=n){fprintf(stderr,"short\n");return 1;} fclose(f);
            for(int bz=0;bz+MC_BLK<=D;bz+=MC_BLK)
            for(int by=0;by+MC_BLK<=D;by+=MC_BLK)
            for(int bx=0;bx+MC_BLK<=D;bx+=MC_BLK){
                int any=0;
                for(int z=0;z<MC_BLK;++z)for(int y=0;y<MC_BLK;++y)
                    memcpy(vox+((size_t)z*MC_BLK+y)*MC_BLK, vol+((size_t)(bz+z)*D+(by+y))*D+bx, MC_BLK);
                for(int i=0;i<N3;++i) any|=vox[i];
                if(!any) continue;
                // quantize exactly like the encoder (no fill needed for the probe:
                // use the encoder itself to produce ql via a private copy)
                long sum=0,cnt=0; for(int i=0;i<N3;++i){ if(vox[i]){sum+=vox[i];cnt++;} }
                int dc=(int)((sum+cnt/2)/cnt);
                for(int i=0;i<N3;++i) blkf[i]=(float)((vox[i]?vox[i]:dc)-dc);
                mc_dct3_fwd(blkf,coef);
                step_tab_build();
                for(int i=0;i<N3;++i){ mc_i32 v=quant_one(coef[i],g_step_tab[i]); ql[i]=(rc_i16)v; }
                if(pass==0){
                    for(int i=0;i<N3;++i){
                        int b=band_of_S((rc_u32)i,MC_BLK);
                        uint32_t zz=zigzag(ql[i]);
                        hist_band[b][zz<64?zz:64]+=1.0;
                    }
                } else {
                    // actual adaptive size (coef stage only)
                    rc_enc e; enc_init(&e,scratch,sizeof scratch);
                    enc_block_coefs(&e,ql,MC_BLK); enc_flush(&e);
                    bits_adaptive += 8.0*(double)e.len;
                    // static model size
                    for(int i=0;i<N3;++i){
                        int b=band_of_S((rc_u32)i,MC_BLK);
                        uint32_t zz=zigzag(ql[i]); uint32_t s=zz<64?zz:64;
                        double tot=0; for(int k=0;k<NSYM;++k) tot+=hist_band[b][k]+0.5;
                        double p=(hist_band[b][s]+0.5)/tot;
                        bits_static += -log2(p);
                        if(s==64){ uint32_t x=zz-64, nb=0,t=x+1; while(t>1){t>>=1;nb++;} bits_static += 2*nb+1; }
                        nsym_tot++;
                    }
                    nblk++;
                }
            }
            free(vol);
        }
        free(vv);
    }
    printf("q=%g  blocks %ld\n", q, nblk);
    printf("coef stage: adaptive %.2f MB, static per-band model %.2f MB  -> static is %+.1f%%\n",
        bits_adaptive/8e6, bits_static/8e6, 100.0*(bits_static-bits_adaptive)/bits_adaptive);

    // Part B: relative decode throughput, same clocks
    {
        // representative symbol stream from band-3 stats
        rans_tab T; rans_tab_build(&T,hist_band[2]);
        size_t NS=20*1000*1000;
        uint8_t *sym=malloc(NS), *sym2=malloc(NS);
        unsigned rs=99; // sample from the distribution
        for(size_t i=0;i<NS;++i){ rs^=rs<<13;rs^=rs>>17;rs^=rs<<5;
            uint32_t slot=rs&(RANS_PROB-1); sym[i]=T.slot2sym[slot]; }
        uint8_t *rbuf=malloc(NS*2+64);
        size_t rlen=rans_encode(&T,sym,NS,rbuf,NS*2+64);
        double t0=now(); rans_decode(&T,rbuf,rlen,sym2,NS); double t_rans=now()-t0;
        if(memcmp(sym,sym2,NS)!=0){ printf("rANS roundtrip MISMATCH\n"); return 1; }

        // adaptive bins through the production coder: same information content
        size_t NB2=NS*2;   // ~2 bins per symbol is the codec's typical ratio
        uint8_t *bits=malloc(NB2);
        for(size_t i=0;i<NB2;++i){ rs^=rs<<13;rs^=rs>>17;rs^=rs<<5; bits[i]=(rs>>20)&1?((rs>>21)&3?1:0):0; }
        uint8_t *bbuf=malloc(NB2);
        rc_enc e; enc_init(&e,bbuf,NB2);
        ctx_t cx[8]; for(int i=0;i<8;++i) ctx_init(&cx[i]);
        for(size_t i=0;i<NB2;++i) enc_bit(&e,&cx[i&7],bits[i]);
        enc_flush(&e);
        rc_dec d; dec_init(&d,bbuf,e.len);
        for(int i=0;i<8;++i) ctx_init(&cx[i]);
        t0=now();
        uint32_t sink=0;
        for(size_t i=0;i<NB2;++i) sink+=dec_bit(&d,&cx[i&7]);
        double t_bins=now()-t0;
        printf("rANS:    %.0f Msym/s (4-way interleaved, 12-bit static CDF)\n", NS/1e6/t_rans);
        printf("adaptive %.0f Mbin/s  (~2 bins/symbol -> %.0f Msym-equiv/s)   [sink %u]\n",
            NB2/1e6/t_bins, NB2/2e6/t_bins, sink);
        printf("relative decode speedup of rANS at the symbol level: %.1fx\n",
            (NS/t_rans)/(NB2/2.0/t_bins));
    }
    return 0;
}
