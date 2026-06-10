// mc_roundtrip — build an archive from a synthetic volume, decode it back, verify
// air voxels restore to 0 and material reconstructs within tolerance. No external deps.
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DIM 256   // one chunk; small + fast

// synthetic source: a smooth material ball in the center, air (0) outside — exercises
// the air-mask path + the DCT on smooth material.
static mc_u8 src_fn(void *ud, int x,int y,int z){
    (void)ud; double cx=DIM/2,cy=DIM/2,cz=DIM/2;
    double r=sqrt((x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz));
    if(r > DIM*0.4) return 0;                          // air outside the ball
    double v=128.0 + 100.0*cos(r*0.15);               // smooth interior signal
    if(v<1)v=1;
    if(v>255)v=255;
    return (mc_u8)v;                                   // material is never 0
}

int main(void){
    mc_build_opts opt={ .dim=DIM, .quality=8.0f, .metadata="{\"test\":\"roundtrip\"}", .meta_len=20 };
    size_t len=0; uint8_t *arc=mc_build(src_fn,NULL,&opt,&len);
    if(!arc){ fprintf(stderr,"build failed\n"); return 1; }
    printf("built archive: %zu bytes\n",len);

    // metadata round-trip
    size_t ml; const char*m=mc_metadata(arc,&ml);
    printf("metadata (%zu B): %.*s\n",ml,(int)ml,m);

    mc_reader *r=mc_open(arc,len); mc_reader_set_quality(r,8.0f);
    long air=0,airbad=0,mat=0,sqerr=0,maxerr=0,nblk=0;
    int nch=(DIM+255)/256;
    mc_u8 dec[16*16*16];
    for(int cz=0;cz<nch;++cz)for(int cy=0;cy<nch;++cy)for(int cx=0;cx<nch;++cx){
        uint64_t co=mc_chunk_offset(r,0,cz,cy,cx);
        for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
            mc_decode_block(r,co,bz,by,bx,dec); nblk++;
            for(int z=0;z<16;++z)for(int y=0;y<16;++y)for(int x=0;x<16;++x){
                int gx=(cx*16+bx)*16+x, gy=(cy*16+by)*16+y, gz=(cz*16+bz)*16+z;
                int s=src_fn(NULL,gx,gy,gz), d=dec[(z*16+y)*16+x];
                if(s==0){ air++; if(d!=0) airbad++; }
                else { mat++; long e=labs((long)d-s); sqerr+=e*e; if(e>maxerr)maxerr=e; }
            }
        }
    }
    double rmse=mat?sqrt((double)sqerr/mat):0, psnr=rmse>0?10*log10(255.0*255.0/(rmse*rmse)):99;
    printf("blocks=%ld material=%ld air=%ld\n",nblk,mat,air);
    printf("air->nonzero(leak)=%ld  material RMSE=%.3f maxerr=%ld PSNR=%.2f dB\n",airbad,rmse,maxerr,psnr);
    mc_close(r); free(arc);

    int ok = (airbad==0) && (rmse<6.0) && (ml==20);
    printf("%s\n", ok?"PASS ✓":"FAIL ✗");
    return ok?0:1;
}
