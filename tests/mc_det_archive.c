// CI helper: build a deterministic archive and print SIZE + round-trip
// quality metrics. Cross-platform CI asserts METRIC equivalence (size within
// tolerance, PSNR/max-error matching), not bit identity — encoder float
// paths are allowed to differ in last-bit rounding across compilers/ISAs as
// long as the delivered quality is identical.
#include "../src/matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static mc_u8 srcv(void *ud, int x,int y,int z){
    (void)ud;
    double dx=x-128,dy=y-128,dz=z-128;
    if(dx*dx+dy*dy+dz*dz>120.0*120.0) return 0;
    return (mc_u8)(40+((x*7+y*3+z*5)%160));
}
int main(void){
    mc_build_opts o={.dim=256,.quality=6.0f};
    size_t len=0; uint8_t *arc=mc_build(srcv,NULL,&o,&len);
    if(!arc) return 1;
    mc_reader *r=mc_open(arc,len);
    uint64_t co=mc_chunk_offset(r,0,0,0,0);
    double se=0, mae=0; long maxe=0, nvox=0;
    mc_u8 blk[4096];
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
        mc_decode_block(r,co,bz,by,bx,blk);
        for(int z=0;z<16;++z)for(int y=0;y<16;++y)for(int x=0;x<16;++x){
            int gx=bx*16+x, gy=by*16+y, gz=bz*16+z;
            int e=abs((int)blk[(z*16+y)*16+x]-(int)srcv(NULL,gx,gy,gz));
            se+=(double)e*e; mae+=e; maxe=e>maxe?e:maxe; nvox++;
        }
    }
    double psnr=10.0*log(255.0*255.0/(se/nvox))/log(10.0);
    // size in KB (tolerant), PSNR to 0.1 dB, MAE to 0.01, max error exact
    printf("size_kb=%zu psnr=%.1f mae=%.2f max=%ld\n",len/1024,psnr,mae/nvox,maxe);
    mc_close(r); free(arc);
    return 0;
}
