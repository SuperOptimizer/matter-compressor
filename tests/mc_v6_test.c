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
    free(buf);

    // 4) parallel chunk encode/decode == serial, and verify still passes
    remove(path);
    a=mc_archive_open_dims(path,512,256,256,6.0f);
    for(size_t i=0;i<sizeof chunk;++i) chunk[i]=(mc_u8)((i%37)?30+(i%170):0);
    if(mc_archive_append_chunk_raw_q(a,0,0,0,0,chunk,6.0f)!=0) return 1;       // serial
    if(mc_archive_append_chunk_par(a,0,0,0,1,chunk,6.0f,0)!=0) return 1;       // parallel
    uint64_t cs=mc_archive_chunk_offset(a,0,0,0,0), cp=mc_archive_chunk_offset(a,0,0,0,1);
    static mc_u8 outs[256*256*256], outp[256*256*256];
    mc_archive_decode_chunk(a,cs,outs,1);       // serial decode
    mc_archive_decode_chunk(a,cp,outp,0);       // parallel decode
    if(memcmp(outs,outp,sizeof outs)!=0){ fprintf(stderr,"FAIL: par chunk != serial chunk\n"); return 1; }
    mc_archive_close(a);
    FILE *f4=fopen(path,"rb"); fseek(f4,0,SEEK_END); long fl4=ftell(f4); fseek(f4,0,SEEK_SET);
    uint8_t *b4=malloc((size_t)fl4); fread(b4,1,(size_t)fl4,f4); fclose(f4);
    if(mc_verify_archive(b4,(size_t)fl4,0)!=0){ fprintf(stderr,"FAIL: verify after par append\n"); return 1; }
    free(b4);
    printf("parallel: OK (par==serial, verify clean)\n");

    // 5) fractions + sampler + region/batch reads
    remove(path);
    a=mc_archive_open_dims(path,512,256,256,6.0f);
    // chunk 0: top half material, bottom half air (z<128 material)
    for(int z=0;z<256;++z)for(int y=0;y<256;++y)for(int x=0;x<256;++x)
        chunk[((size_t)z*256+y)*256+x]=(mc_u8)(z<128?(40+((x+y+z)%150)):0);
    mc_archive_append_chunk_raw(a,0,0,0,0,chunk);
    float f_mat=mc_archive_block_fraction(a,0,2,2,2);     // z blocks 0..7 material
    float f_air=mc_archive_block_fraction(a,0,12,2,2);    // z blocks 8..15 air
    if(!(f_mat>0.9f)||!(f_air<0.05f)){ fprintf(stderr,"FAIL: fractions %f %f\n",f_mat,f_air); return 1; }
    if(!mc_archive_block_present(a,0,2,2,2)||mc_archive_block_present(a,0,12,2,2)){
        fprintf(stderr,"FAIL: occupancy\n"); return 1; }
    // sampler: deterministic + respects min_frac (volume dims 512x256x256 -> nx=512)
    mc_box b1[16], b2[16];
    int n1=mc_archive_sample_boxes(a,0,42,16,64,64,64,0.8f,b1);
    int n2=mc_archive_sample_boxes(a,0,42,16,64,64,64,0.8f,b2);
    if(n1!=n2||memcmp(b1,b2,sizeof(mc_box)*n1)!=0){ fprintf(stderr,"FAIL: sampler not deterministic\n"); return 1; }
    int frac_ok=1;
    for(int i=0;i<n1;++i) if(b1[i].z0+64>128+16) frac_ok=0;   // boxes must sit in material half
    if(!n1||!frac_ok){ fprintf(stderr,"FAIL: sampler min_frac (%d boxes)\n",n1); return 1; }
    // region read == direct blocks; batch read == per-region read
    static mc_u8 reg[64*64*64], reg2[64*64*64];
    mc_archive_read_region(a,0,10,20,30,64,64,64,reg,64*64,64,0);
    long bad5=0;
    for(int z=0;z<64;++z)for(int y=0;y<64;++y)for(int x=0;x<64;++x){
        // compare against decoded chunk content via decode_chunk once
        ;
    }
    static mc_u8 whole[256*256*256];
    uint64_t co5b=mc_archive_chunk_offset(a,0,0,0,0);
    mc_archive_decode_chunk(a,co5b,whole,0);
    for(int z=0;z<64;++z)for(int y=0;y<64;++y)for(int x=0;x<64;++x)
        if(reg[((size_t)z*64+y)*64+x]!=whole[((size_t)(10+z)*256+(20+y))*256+(30+x)]) bad5++;
    if(bad5){ fprintf(stderr,"FAIL: region read mismatch (%ld)\n",bad5); return 1; }
    mc_box bx2[4]={{0,0,0},{10,20,30},{60,100,3},{64,64,300}};   // last crosses x beyond 256 into absent chunk
    static mc_u8 batch[4][64*64*64];
    mc_archive_read_regions(a,0,bx2,4,64,64,64,&batch[0][0],sizeof batch[0],0);
    mc_archive_read_region(a,0,10,20,30,64,64,64,reg2,64*64,64,1);
    if(memcmp(batch[1],reg2,sizeof reg2)!=0){ fprintf(stderr,"FAIL: batch != single region\n"); return 1; }
    printf("ml: OK (fractions %.2f/%.2f, %d sampled boxes, region+batch reads exact)\n",f_mat,f_air,n1);
    mc_archive_close(a);

    free(arc); remove(path);
    printf("mc_v6: OK\n");
    return 0;
}
