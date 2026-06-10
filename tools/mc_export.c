// ============================================================================
// mc_export — repack an archive (or a sub-region of one) into a fresh file
// optimized for BOTH offline use and streaming:
//
//   - chunk blobs are copied VERBATIM (no re-encode, no quality loss),
//     ordered along a Morton curve for spatial locality,
//   - the entire node-tree index is reserved up front, so it sits contiguously
//     right after the 128KB metadata region — a streaming client fetches the
//     whole index in one ranged GET; an offline client mmaps and goes,
//   - append slack is truncated (the appendable writer grows in 1GiB steps).
//
// usage: mc_export <in.mc> <out.mc> [cz0 cy0 cx0 ncz ncy ncx]
//   box is in LOD0 256^3-chunk coords; omitted = whole archive. Coarser LODs
//   are exported with the box scaled accordingly.
// ============================================================================
#include "../src/mc_archive_api.h"
#include "../src/mc_archive.h"
#include "../src/mc_archive_read.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// interleave 10 bits of z,y,x -> 30-bit Morton code
static uint32_t morton3(uint32_t z, uint32_t y, uint32_t x){
    uint32_t m=0;
    for(int b=0;b<10;++b){
        m |= ((x>>b)&1u)<<(3*b) | ((y>>b)&1u)<<(3*b+1) | ((z>>b)&1u)<<(3*b+2);
    }
    return m;
}
typedef struct { uint32_t key; uint16_t cz,cy,cx; uint8_t lod; } ent_t;
static int ent_cmp(const void*a,const void*b){
    const ent_t*p=a,*q=b;
    if(p->lod!=q->lod) return (int)p->lod-(int)q->lod;
    return p->key<q->key?-1:p->key>q->key?1:0;
}

int main(int argc,char**argv){
    if(argc!=3 && argc!=9){
        fprintf(stderr,"usage: %s <in.mc> <out.mc> [cz0 cy0 cx0 ncz ncy ncx]\n",argv[0]);
        return 1;
    }
    int box=argc==9;
    long bz0=box?atol(argv[3]):0, by0=box?atol(argv[4]):0, bx0=box?atol(argv[5]):0;
    long bnz=box?atol(argv[6]):0, bny=box?atol(argv[7]):0, bnx=box?atol(argv[8]):0;

    int fd=open(argv[1],O_RDONLY);
    if(fd<0){ perror("open in"); return 1; }
    struct stat sb; fstat(fd,&sb);
    const uint8_t *arc=mmap(NULL,(size_t)sb.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    if(arc==MAP_FAILED){ perror("mmap"); return 1; }

    uint32_t magic,ver,dim; memcpy(&magic,arc+MCH_MAGIC,4); memcpy(&ver,arc+MCH_VER,4);
    memcpy(&dim,arc+MCH_NX,4);
    if(magic!=MC_MAGIC||ver!=MC_VERSION){ fprintf(stderr,"not a current-version mc archive\n"); return 1; }
    float quality; memcpy(&quality,arc+MCH_QUALITY,4);
    uint64_t roots[8]; for(int l=0;l<8;++l) memcpy(&roots[l],arc+MCH_ROOTOFF+l*8,8);
    uint64_t metalen; memcpy(&metalen,arc+MCH_METALEN,8);

    // enumerate present chunks (filtered by the box, scaled per LOD)
    int nch=(int)((dim+255)/256);
    ent_t *ents=malloc(sizeof(ent_t)*(size_t)nch*nch*nch*2);
    size_t nent=0;
    for(int lod=0;lod<8;++lod){
        if(!roots[lod]) continue;
        int n=(nch>>lod)?(nch>>lod):1;
        long z0=box? bz0>>lod:0, y0=box? by0>>lod:0, x0=box? bx0>>lod:0;
        long z1=box? ((bz0+bnz+(1L<<lod)-1)>>lod):n, y1=box? ((by0+bny+(1L<<lod)-1)>>lod):n,
             x1=box? ((bx0+bnx+(1L<<lod)-1)>>lod):n;
        if(z1>n)z1=n; if(y1>n)y1=n; if(x1>n)x1=n;
        for(long cz=z0;cz<z1;++cz)for(long cy=y0;cy<y1;++cy)for(long cx=x0;cx<x1;++cx){
            if(!mc_resolve_chunk(arc,roots[lod],(int)cz,(int)cy,(int)cx)) continue;
            ents[nent++]=(ent_t){morton3((uint32_t)cz,(uint32_t)cy,(uint32_t)cx),
                                 (uint16_t)cz,(uint16_t)cy,(uint16_t)cx,(uint8_t)lod};
        }
    }
    qsort(ents,nent,sizeof(ent_t),ent_cmp);
    printf("exporting %zu chunks (%s)\n",nent,box?"box":"all");

    remove(argv[2]);
    mc_archive *out=mc_archive_open(argv[2],(int)dim,quality);
    if(!out){ fprintf(stderr,"open out failed\n"); return 1; }
    // carry metadata over
    if(metalen){
        // metadata region is at MC_HDR in both files; mc_archive_open mmaps out rw —
        // no public API for metadata writes yet, copy via the file
        // (the writer's mmap sees it since it maps the same file)
        FILE*of=fopen(argv[2],"rb+"); fseek(of,MC_HDR,SEEK_SET);
        fwrite(arc+MC_HDR,1,(size_t)metalen,of); fclose(of);
    }
    // pass 1: reserve every index path -> all node tables land contiguously
    for(size_t i=0;i<nent;++i)
        mc_archive_reserve_index(out,ents[i].lod,ents[i].cz,ents[i].cy,ents[i].cx);
    // pass 2: verbatim blob copy in (lod, morton) order
    for(size_t i=0;i<nent;++i){
        uint64_t off=mc_resolve_chunk(arc,roots[ents[i].lod],ents[i].cz,ents[i].cy,ents[i].cx);
        uint64_t blen=mc_chunk_blob_len(arc,off);
        if(mc_archive_append_chunk_compressed(out,ents[i].lod,ents[i].cz,ents[i].cy,ents[i].cx,
                                              arc+off,(size_t)blen)!=0){
            fprintf(stderr,"append failed at chunk %zu\n",i); return 1;
        }
    }
    // stamp metadata length in the new header
    {
        FILE*of=fopen(argv[2],"rb+"); fseek(of,MCH_METALEN,SEEK_SET);
        fwrite(&metalen,8,1,of); fclose(of);
    }
    mc_archive_close(out);
    struct stat ob; stat(argv[2],&ob);
    printf("wrote %s: %.2f MB (input %.2f MB)\n",argv[2],ob.st_size/1e6,sb.st_size/1e6);
    munmap((void*)arc,(size_t)sb.st_size); close(fd);
    return 0;
}
