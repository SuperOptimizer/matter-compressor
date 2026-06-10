// format v6 test: per-axis dims (padding semantics), per-chunk q, xxh64.
#include "../src/mc_archive_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static mc_u8 srcv(void *ud, int x,int y,int z){
    (void)ud;
    return (mc_u8)(40+((x*7+y*5+z*3)%150));
}

int main(void){
    // 1) non-cubic build: 300 x 130 x 70 voxels (pads to 512 x 256 x 256)
    mc_build_opts o={.nx=300,.ny=130,.nz=70,.quality=6.0f,.metadata="v6",.meta_len=2};
    size_t alen=0; uint8_t *arc=mc_build(srcv,NULL,&o,&alen);
    if(!arc){ fprintf(stderr,"build failed\n"); return 1; }
    mc_reader *r=mc_open(arc,alen);
    // inside region decodes close to source; outside NX..pad decodes to 0
    mc_u8 blk[4096];
    uint64_t co=mc_chunk_offset(r,0,0,0,0);
    if(!co){ fprintf(stderr,"chunk(0,0,0) missing\n"); return 1; }
    mc_decode_block(r,co,0,0,0,blk);
    double mae=0; for(int z=0;z<16;++z)for(int y=0;y<16;++y)for(int x=0;x<16;++x)
        mae+=abs((int)blk[(z*16+y)*16+x]-(int)srcv(NULL,x,y,z));
    mae/=4096;
    if(mae>12){ fprintf(stderr,"FAIL: non-cubic interior MAE %.1f\n",mae); return 1; }
    // block fully beyond NX on x axis: chunk (0,0,1) covers x in [256,512); NX=300
    uint64_t co2=mc_chunk_offset(r,0,0,0,1);
    if(co2){
        mc_decode_block(r,co2,0,0,8,blk);   // bx=8 -> x in [384,400) > NX: all padding
        for(int i=0;i<4096;++i) if(blk[i]){ fprintf(stderr,"FAIL: padding not zero\n"); return 1; }
    }
    // beyond-NZ chunk should be entirely absent (cz=1 covers z>=256 > 70 pad 256)
    if(mc_chunk_offset(r,0,1,0,0)){ fprintf(stderr,"FAIL: beyond-NZ chunk present\n"); return 1; }
    printf("non-cubic: OK (interior MAE %.2f, padding zero, absent beyond-NZ)\n",mae);
    mc_close(r);

    // 2) per-chunk q + verify
    const char *path="/tmp/mc_v6.mc";
    remove(path);
    mc_archive *a=mc_archive_open_dims(path,512,256,256,6.0f);
    if(!a){ fprintf(stderr,"open_dims failed\n"); return 1; }
    static mc_u8 chunk[256*256*256];
    for(size_t i=0;i<sizeof chunk;++i) chunk[i]=(mc_u8)(30+(i%170));
    if(mc_archive_append_chunk_raw_q(a,0,0,0,0,chunk,1.5f)!=0) return 1;
    if(mc_archive_append_chunk_raw_q(a,0,0,0,1,chunk,14.0f)!=0) return 1;
    uint64_t ca=mc_archive_chunk_offset(a,0,0,0,0), cb=mc_archive_chunk_offset(a,0,0,0,1);
    mc_u8 ba[4096], bb[4096];
    mc_archive_decode_block(a,ca,3,3,3,ba);
    mc_archive_decode_block(a,cb,3,3,3,bb);
    double ea=0,eb=0;
    for(int z=0;z<16;++z)for(int y=0;y<16;++y)for(int x=0;x<16;++x){
        size_t ci=((size_t)(48+z)*256+(48+y))*256+(48+x);
        ea+=abs((int)ba[(z*16+y)*16+x]-(int)chunk[ci]);
        eb+=abs((int)bb[(z*16+y)*16+x]-(int)chunk[ci]);
    }
    ea/=4096; eb/=4096;
    printf("per-chunk q: MAE q=1.5 -> %.2f, q=14 -> %.2f\n",ea,eb);
    if(!(ea<eb)){ fprintf(stderr,"FAIL: low-q chunk not higher fidelity\n"); return 1; }
    // interleaved decode (alternating q) must stay consistent
    mc_u8 ba2[4096];
    mc_archive_decode_block(a,cb,1,1,1,bb);
    mc_archive_decode_block(a,ca,3,3,3,ba2);
    if(memcmp(ba,ba2,4096)!=0){ fprintf(stderr,"FAIL: q cross-contamination\n"); return 1; }
    mc_archive_close(a);

    // 3) integrity: clean verify, then flip one payload byte -> 1 corrupt chunk
    FILE *f=fopen(path,"rb"); fseek(f,0,SEEK_END); long flen=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t *buf=malloc((size_t)flen); fread(buf,1,(size_t)flen,f); fclose(f);
    if(mc_verify_archive(buf,(size_t)flen,0)!=0){ fprintf(stderr,"FAIL: clean archive flagged corrupt\n"); return 1; }
    buf[ca+2000]^=0x55;
    if(mc_verify_archive(buf,(size_t)flen,0)!=1){ fprintf(stderr,"FAIL: tamper not detected\n"); return 1; }
    printf("verify: OK (clean passes, tamper detected)\n");
    free(buf); free(arc); remove(path);
    printf("mc_v6: OK\n");
    return 0;
}
