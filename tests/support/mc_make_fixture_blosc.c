// mc_make_fixture_blosc — emit a zarr v2 tree whose chunks are blosc1+zstd
// frames, so mc_zarr's blosc_decode path runs offline (the raw fixture covers
// only compressor:null). Frame layout matches blosc_decode's parser: 16-byte
// header [ver,verlz,flags,typesize, nbytes:u32, blocksize:u32, cbytes:u32],
// then nblocks u32 bstarts, then per block [cbytes:i32][zstd payload].
//
//   usage: mc_make_fixture_blosc <out_dir>   (-> <out>/zarr/0/{.zarray,cz.cy.cx})
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <zstd.h>

#define CHUNK 128            // v2 inner == shard edge (mc_volume needs 128 or 256)
#define BLOCKSZ (1u<<18)     // 256KB blosc blocks

static int mkdirp(const char*p){ char t[2048]; snprintf(t,sizeof t,"%s",p);
    for(char*q=t+1;*q;++q) if(*q=='/'){*q=0;mkdir(t,0755);*q='/';} return mkdir(t,0755); }
static void wfile(const char*p,const void*d,size_t n){
    FILE*f=fopen(p,"wb"); if(!f){perror(p);exit(1);} if(n)fwrite(d,1,n,f); fclose(f); }

static uint8_t voxel(int gx,int gy,int gz,int dim){
    double c=dim/2.0,r=dim*0.32,d=(gx-c)*(gx-c)+(gy-c)*(gy-c)+(gz-c)*(gz-c);
    return d<r*r?180:0;
}

// build a blosc1 frame (zstd-compressed blocks) over `n` raw bytes.
static uint8_t *blosc_frame(const uint8_t*raw,uint32_t n,size_t*out){
    uint32_t blocksize=BLOCKSZ, nblocks=(n+blocksize-1)/blocksize;
    size_t cap=16 + (size_t)nblocks*4 + n + nblocks*(4+512) + ZSTD_compressBound(blocksize);
    uint8_t *f=malloc(cap); memset(f,0,16);
    f[0]=2; f[1]=1; f[2]=0; f[3]=1;            // version, versionlz, flags=0 (zstd blocks), typesize=1
    size_t pos=16+(size_t)nblocks*4;           // bstarts table follows header
    uint8_t *bstarts=f+16;
    for(uint32_t b=0;b<nblocks;++b){
        uint32_t neblock=(b==nblocks-1)?n-b*blocksize:blocksize;
        uint32_t bs=(uint32_t)pos; memcpy(bstarts+(size_t)b*4,&bs,4);
        size_t cb=ZSTD_compress(f+pos+4, cap-pos-4, raw+(size_t)b*blocksize, neblock, 3);
        if(ZSTD_isError(cb)){ free(f); return NULL; }
        int32_t cbi=(int32_t)cb; memcpy(f+pos,&cbi,4);
        pos+=4+cb;
    }
    uint32_t cbytes=(uint32_t)pos;
    memcpy(f+4,&n,4); memcpy(f+8,&blocksize,4); memcpy(f+12,&cbytes,4);
    *out=pos; return f;
}

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s <out_dir>\n",argv[0]); return 2; }
    const char*out=argv[1]; int dim=256;        // one 256-region, 2^3 chunks of 128
    char ldir[2048]; snprintf(ldir,sizeof ldir,"%s/zarr/0",out); mkdirp(ldir);
    char za[768];
    int n=snprintf(za,sizeof za,
        "{\"zarr_format\":2,\"shape\":[%d,%d,%d],\"chunks\":[%d,%d,%d],"
        "\"dtype\":\"|u1\",\"compressor\":{\"id\":\"blosc\",\"cname\":\"zstd\"},"
        "\"fill_value\":0,\"order\":\"C\",\"dimension_separator\":\".\"}",
        dim,dim,dim,CHUNK,CHUNK,CHUNK);
    char zp[2200]; snprintf(zp,sizeof zp,"%s/.zarray",ldir); wfile(zp,za,(size_t)n);

    int grid=(dim+CHUNK-1)/CHUNK;
    uint8_t *chunk=malloc((size_t)CHUNK*CHUNK*CHUNK);
    for(int cz=0;cz<grid;++cz)for(int cy=0;cy<grid;++cy)for(int cx=0;cx<grid;++cx){
        int any=0;
        for(int z=0;z<CHUNK;++z)for(int y=0;y<CHUNK;++y)for(int x=0;x<CHUNK;++x){
            int gx=cx*CHUNK+x,gy=cy*CHUNK+y,gz=cz*CHUNK+z;
            uint8_t v=voxel(gx,gy,gz,dim); chunk[(z*CHUNK+y)*CHUNK+x]=v; any|=v;
        }
        if(!any) continue;
        size_t fl=0; uint8_t*f=blosc_frame(chunk,(uint32_t)((size_t)CHUNK*CHUNK*CHUNK),&fl);
        if(!f){ fprintf(stderr,"blosc frame failed\n"); return 1; }
        char cp[2300]; snprintf(cp,sizeof cp,"%s/%d.%d.%d",ldir,cz,cy,cx);
        wfile(cp,f,fl); free(f);
    }
    free(chunk);
    printf("blosc v2 fixture: %s/zarr (MC_ZARR_ROOT=%s/zarr/0)\n",out,out);
    return 0;
}
