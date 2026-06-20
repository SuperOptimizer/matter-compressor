// mc_node_align_test — regression for the index-node ALIGNMENT bug.
//
// The appendable archive's index nodes hold u64 child offsets accessed as
// _Atomic uint64_t (ldapr/stlr on aarch64). Those atomics BUS-fault on a
// misaligned address. Node tables are allocated at EOF from the SAME append
// cursor as the variable-length chunk blobs, so once any blob has been written
// the cursor sits at an arbitrary offset and the next node (e.g. a fresh LOD
// root, or a new inner/shard node) lands unaligned -> crash on the first slot
// access. (The old smooth-ball round-trip test missed this: its blob lengths
// happened to leave the cursor 8-aligned.)
//
// This test (a) appends real-material chunks across several LODs and coords so
// node tables are allocated AFTER blobs with data-dependent lengths, then
// (b) walks every materialized node in the index and asserts its file offset is
// 8-aligned -- a deterministic check that does not rely on the unaligned access
// actually faulting on the host ISA -- and (c) round-trips a chunk to prove the
// archive is still valid. Pre-fix this aborts (SIGBUS) or fails the alignment
// assert; post-fix it passes everywhere.
#include "matter_compressor.h"
#include "mc_archive.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DIM 1024   // up to 4 chunks/axis at LOD0

static mc_u8 sample(int lod,int x,int y,int z){
    // textured material (never 0 inside) so blobs have realistic, varied lengths.
    int dim=DIM>>lod;
    double cx=dim/2.0,cy=dim/2.0,cz=dim/2.0;
    double r=sqrt((x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz));
    if(r>dim*0.6) return 0;
    int v=110+(int)(70.0*cos(r*0.13))+((x^y^z)&15)+lod*3;
    if(v<1)v=1; if(v>255)v=255; return (mc_u8)v;
}
static void fill_chunk(mc_u8 *buf,int lod,int cz,int cy,int cx){
    for(int z=0;z<256;++z)for(int y=0;y<256;++y)for(int x=0;x<256;++x)
        buf[((size_t)z*256+y)*256+x]=sample(lod,cx*256+x,cy*256+y,cz*256+z);
}
// voxel source for the static mc_build phase (512^3 textured ball).
static mc_u8 static_src(void *ud,int x,int y,int z){ (void)ud; return sample(0,x,y,z); }

// recursively walk the dense node tree from a node offset, asserting 8-alignment.
static int walk_node(const uint8_t *base,uint64_t node,uint64_t flen,int level,long *nbad,long *nnodes){
    if(node==0||node>=flen) return 0;
    if(node & 7){ (*nbad)++; fprintf(stderr,"  MISALIGNED node @%llu (level %d)\n",(unsigned long long)node,level); }
    (*nnodes)++;
    if(level==0) return 0;   // shard table: children are chunk-blob offsets, not nodes
    for(int i=0;i<MC_GRID3;++i){
        uint64_t child; memcpy(&child,base+node+(size_t)i*8,8);
        if(child && child!=MC_SLOT_ZERO && level>1) walk_node(base,child,flen,level-1,nbad,nnodes);
    }
    return 0;
}

int main(void){
    const char *path="mc_node_align_test.mca";
    remove(path);
    const float Q=4.0f;   // low q -> large blobs, arbitrary lengths
    mc_u8 *chunk=malloc((size_t)256*256*256);
    mc_archive *w=mc_archive_open_dims(path,DIM,DIM,DIM,Q);
    if(!w){ fprintf(stderr,"open failed\n"); return 1; }

    // Append in an order that forces node tables to be created AFTER blobs:
    // several LOD0 chunks (new inner/shard nodes), then a fresh root for each of
    // LOD1..LOD3 (each root allocated right after the prior LOD's blobs).
    struct { int lod,cz,cy,cx; } seq[]={
        {0,0,0,0},{0,1,0,0},{0,0,1,0},{0,1,1,1},{0,2,2,2},{0,3,1,2},
        {1,0,0,0},{1,1,1,1},
        {2,0,0,0},{2,1,0,1},
        {3,0,0,0},
    };
    int n=(int)(sizeof seq/sizeof seq[0]);
    for(int i=0;i<n;++i){
        fill_chunk(chunk,seq[i].lod,seq[i].cz,seq[i].cy,seq[i].cx);
        if(mc_archive_append_chunk_raw(w,seq[i].lod,seq[i].cz,seq[i].cy,seq[i].cx,chunk)){
            fprintf(stderr,"append %d (lod%d) failed/CRASHED\n",i,seq[i].lod); return 1; }
    }
    mc_archive_close(w);

    // ---- walk the on-disk index, assert every node offset is 8-aligned ----
    FILE *f=fopen(path,"rb"); fseek(f,0,SEEK_END); long flen=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t *buf=malloc(flen);
    if(fread(buf,1,flen,f)!=(size_t)flen){ fprintf(stderr,"read file\n"); return 1; } fclose(f);
    long nbad=0,nnodes=0;
    for(int lod=0;lod<8;++lod){
        uint64_t root; memcpy(&root,buf+MCH_ROOTOFF+(uint64_t)lod*8,8);
        walk_node(buf,root,(uint64_t)flen,MC_TREE_LEVELS-1,&nbad,&nnodes);
    }
    printf("walked %ld index nodes; misaligned=%ld\n",nnodes,nbad);

    // ---- round-trip one chunk to prove the archive decodes ----
    int fail=(nbad!=0);
    mc_reader *r=mc_open(buf,flen); mc_reader_set_quality(r,Q);
    uint64_t co=mc_chunk_offset(r,0,1,1,1);
    if(!co){ fprintf(stderr,"chunk LOD0(1,1,1) missing\n"); fail=1; }
    else {
        mc_u8 dec[16*16*16]; long sq=0,mat=0,leak=0;
        for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
            mc_decode_block(r,co,bz,by,bx,dec);
            for(int z=0;z<16;++z)for(int y=0;y<16;++y)for(int x=0;x<16;++x){
                int gx=(1*16+bx)*16+x,gy=(1*16+by)*16+y,gz=(1*16+bz)*16+z;
                int s=sample(0,gx,gy,gz),d=dec[(z*16+y)*16+x];
                if(s==0 && d) leak++;
                mat++; long e=labs((long)d-s); sq+=e*e;   // air folded in (no exact-0 mask)
            }
        }
        double rmse=mat?sqrt((double)sq/mat):0;
        printf("round-trip LOD0(1,1,1): RMSE=%.2f leak=%ld (air leak informational)\n",rmse,leak);
        if(rmse>8.0) fail=1;
    }
    mc_close(r); free(buf);

    // ---- phase 2: STATIC builder (mc_build) then REOPEN + APPEND ----------------
    // The one-shot mc_build emits node/shard tables via a_put_at at the post-blob
    // buffer offset; if unaligned, a later reopen+append (appendable path, atomic
    // node-slot loads) BUS-faults. This is the mc_cache_test scenario. Build, then
    // append a chunk that forces the appendable path to walk the static tree.
    {
        const char *sp="mc_node_align_static.mca";
        remove(sp);
        mc_build_opts opt={.dim=512,.quality=Q};
        if(mc_build_to_file(static_src,NULL,&opt,sp)){ fprintf(stderr,"mc_build_to_file failed\n"); fail=1; }
        else {
            mc_archive *a=mc_archive_open(sp,512,Q);
            if(!a){ fprintf(stderr,"reopen static archive failed\n"); fail=1; }
            else {
                // append into a NEW LOD0 chunk -> walks (atomically) the static root.
                fill_chunk(chunk,0,1,1,1);
                if(mc_archive_append_chunk_raw(a,0,1,1,1,chunk)){ fprintf(stderr,"append-to-static CRASHED/failed\n"); fail=1; }
                else printf("static-build + reopen + append: OK (no unaligned atomic fault)\n");
                mc_archive_close(a);
            }
        }
        remove(sp);
    }

    free(chunk); remove(path);
    printf("%s\n", fail?"FAIL \xe2\x9c\x97":"PASS \xe2\x9c\x93");
    return fail;
}
