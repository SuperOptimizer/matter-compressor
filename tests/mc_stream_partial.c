// partial-fetch streaming test: build a small archive, decode blocks through a
// byte-counting file callback in both streaming modes, verify identical voxels
// and that partial mode transfers far fewer bytes for sparse random access.
#include "../src/matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static mc_u8 srcv(void *ud, int x,int y,int z){
    (void)ud;
    double dx=x-128,dy=y-128,dz=z-128;
    double rr=dx*dx+dy*dy+dz*dz;
    if(rr>120.0*120.0) return 0;                       // air shell
    return (mc_u8)(40+((x*7+y*3+z*5)%160));
}

typedef struct { FILE *f; size_t bytes; long calls; } fsrc;
static int fread_at(void *ud, uint64_t off, uint32_t len, uint8_t *dst){
    fsrc *s=ud;
    if(fseek(s->f,(long)off,SEEK_SET)!=0) return -1;
    if(fread(dst,1,len,s->f)!=len) return -1;
    s->bytes+=len; s->calls++;
    return 0;
}

int main(void){
    const char *path="/tmp/mc_stream_partial.mc";
    remove(path);
    mc_build_opts opt={.dim=256,.quality=6.0f};
    if(mc_build_to_file(srcv,NULL,&opt,path)!=0){ fprintf(stderr,"build failed\n"); return 1; }
    FILE *f=fopen(path,"rb"); fseek(f,0,SEEK_END); long flen=ftell(f);

    fsrc a={f,0,0}, b={f,0,0};
    mc_reader *rf=mc_open_streaming(fread_at,&a,(uint64_t)flen); mc_reader_set_quality(rf,6.0f);
    mc_reader *rp=mc_open_streaming(fread_at,&b,(uint64_t)flen);
    mc_reader_set_partial_fetch(rp,1);

    uint64_t co=mc_chunk_offset(rf,0,0,0,0), co2=mc_chunk_offset(rp,0,0,0,0);
    if(!co||co!=co2){ fprintf(stderr,"chunk resolve mismatch\n"); return 1; }
    size_t base_a=a.bytes, base_b=b.bytes;

    mc_u8 blkf[16*16*16], blkp[16*16*16];
    // sparse random access: 5 scattered blocks
    int picks[5][3]={{8,8,8},{0,0,0},{15,15,15},{3,12,7},{12,3,9}};
    for(int i=0;i<5;++i){
        mc_decode_block(rf,co,picks[i][0],picks[i][1],picks[i][2],blkf);
        mc_decode_block(rp,co,picks[i][0],picks[i][1],picks[i][2],blkp);
        if(memcmp(blkf,blkp,sizeof blkf)!=0){ fprintf(stderr,"voxel mismatch at pick %d\n",i); return 1; }
    }
    size_t full_bytes=a.bytes-base_a, part_bytes=b.bytes-base_b;
    printf("5 scattered blocks: full-fetch %zu B, partial-fetch %zu B (%.1fx less)\n",
           full_bytes, part_bytes, (double)full_bytes/(part_bytes?part_bytes:1));
    if(part_bytes*4 > full_bytes){ fprintf(stderr,"partial fetch not significantly smaller\n"); return 1; }

    // full chunk sweep must also match
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
        mc_decode_block(rf,co,bz,by,bx,blkf);
        mc_decode_block(rp,co,bz,by,bx,blkp);
        if(memcmp(blkf,blkp,sizeof blkf)!=0){ fprintf(stderr,"sweep mismatch %d %d %d\n",bz,by,bx); return 1; }
    }

    // streaming-reader blob accessors (the sread() branch of these, distinct
    // from the flat r->arc branch covered elsewhere): chunk_blob_len + read_blob.
    uint64_t sblen = mc_reader_chunk_blob_len(rf, co);
    if(sblen==0){ fprintf(stderr,"streaming chunk_blob_len==0\n"); return 1; }
    uint8_t *sb = malloc(sblen);
    if(mc_reader_read_blob(rf, co, sblen, sb)!=0){ fprintf(stderr,"streaming read_blob failed\n"); free(sb); return 1; }
    free(sb);
    printf("streaming blob: len=%llu read OK\n",(unsigned long long)sblen);

    mc_close(rf); mc_close(rp); fclose(f); remove(path);
    printf("mc_stream_partial: OK\n");
    return 0;
}
