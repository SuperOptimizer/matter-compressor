// mc_make_fixture_v3 — emit an on-disk **zarr v3 / c3d-sharded** tree so the v3
// zarr read path (mc_zarr_open v3 branch, footer_get, read_inner/read_shard/
// shard_index, and mc_volume's c3d transcode) runs offline. The v2/raw fixture
// (mc_make_fixture) can't reach these branches — v3 requires the c3d inner codec.
//
// Layout per level: zarr.json (v3, sharding_indexed, inner codec c3d) + shard
// objects "c/sz/sy/sx", each = [index footer: n_inner*16 LE (off,nb)][payloads].
// Inner edge = shard edge = 256 (one c3d chunk per shard, sub=1), so the volume
// transcode's sub==1 fast path is exercised.
//
//   usage: mc_make_fixture_v3 <out_dir>   (-> <out>/zarr/<L>/{zarr.json,c/...})
#include "matter_compressor.h"
#include "c3d.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <math.h>

#define EDGE 256              // shard == inner == one c3d chunk (C3D_CHUNK_SIDE)
#define NLEV 2

static int mkdirp(const char *path){
    char t[2048]; snprintf(t,sizeof t,"%s",path);
    for(char*p=t+1;*p;++p) if(*p=='/'){ *p=0; mkdir(t,0755); *p='/'; }
    return mkdir(t,0755);
}
static void wfile(const char*path,const void*d,size_t n){
    FILE*f=fopen(path,"wb"); if(!f){perror(path);exit(1);} if(n)fwrite(d,1,n,f); fclose(f);
}

// a centered solid sphere of value 200 (scaled per level) — material vs air(0).
static uint8_t voxel(int gx,int gy,int gz,int dim){
    double c=dim/2.0,r=dim*0.32, d=(gx-c)*(gx-c)+(gy-c)*(gy-c)+(gz-c)*(gz-c);
    return d<r*r?200:0;
}

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s <out_dir>\n",argv[0]); return 2; }
    const char*out=argv[1];
    int base=512;   // 2x2x2 = 8 shards at L0
    // c3d requires C3D_ALIGN(32)-aligned raw voxel + output buffers.
    uint8_t *chunk=NULL,*enc=NULL;
    size_t cap=c3d_chunk_encode_max_size();
    if(posix_memalign((void**)&chunk,C3D_ALIGN,(size_t)EDGE*EDGE*EDGE)!=0 ||
       posix_memalign((void**)&enc,C3D_ALIGN,cap)!=0){ fprintf(stderr,"alloc\n"); return 1; }

    for(int L=0;L<NLEV;++L){
        int dim=base>>L; if(dim<EDGE) dim=EDGE;
        char ldir[2048]; snprintf(ldir,sizeof ldir,"%s/zarr/%d",out,L); mkdirp(ldir);
        // v3 zarr.json: sharding_indexed, chunk_grid.chunk_shape = shard edge,
        // sharding config codecs chunk_shape = inner edge, inner codec name c3d.
        char zj[1024];
        int n=snprintf(zj,sizeof zj,
          "{\"zarr_format\":3,\"node_type\":\"array\",\"shape\":[%d,%d,%d],"
          "\"data_type\":\"uint8\","
          "\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{\"chunk_shape\":[%d,%d,%d]}},"
          "\"codecs\":[{\"name\":\"sharding_indexed\",\"configuration\":{"
              "\"chunk_shape\":[%d,%d,%d],"
              "\"codecs\":[{\"name\":\"c3d\"}]}}]}",
          dim,dim,dim, EDGE,EDGE,EDGE, EDGE,EDGE,EDGE);
        char zp[2200]; snprintf(zp,sizeof zp,"%s/zarr.json",ldir); wfile(zp,zj,(size_t)n);

        int grid=(dim+EDGE-1)/EDGE;   // shards per axis (== inner grid, per==1)
        char cdir[2100]; snprintf(cdir,sizeof cdir,"%s/c",ldir); mkdirp(cdir);
        for(int sz=0;sz<grid;++sz)for(int sy=0;sy<grid;++sy)for(int sx=0;sx<grid;++sx){
            int any=0;
            for(int z=0;z<EDGE;++z)for(int y=0;y<EDGE;++y)for(int x=0;x<EDGE;++x){
                int gx=sx*EDGE+x,gy=sy*EDGE+y,gz=sz*EDGE+z;
                uint8_t v=(gx<dim&&gy<dim&&gz<dim)?voxel(gx<<L,gy<<L,gz<<L,base):0;
                chunk[(z*EDGE+y)*EDGE+x]=v; any|=v;
            }
            if(!any) continue;   // all-air shard: omit (absent == air)
            size_t elen=c3d_chunk_encode_at_q(chunk,8.0f,enc,cap);
            if(elen==0){ fprintf(stderr,"c3d encode failed\n"); return 1; }
            // shard object = [footer: 1 entry (off=16, nb=elen)][payload].
            char spath[2300]; snprintf(spath,sizeof spath,"%s/%d/%d/%d",cdir,sz,sy,sx);
            { char sd[2200]; snprintf(sd,sizeof sd,"%s/%d/%d",cdir,sz,sy); mkdirp(sd); }
            uint8_t footer[16]; uint64_t off=16, nb=elen;
            memcpy(footer,&off,8); memcpy(footer+8,&nb,8);
            FILE*f=fopen(spath,"wb"); if(!f){perror(spath);return 1;}
            fwrite(footer,1,16,f); fwrite(enc,1,elen,f); fclose(f);
        }
        printf("L%d dim=%d shards=%d^3 (c3d inner=%d)\n",L,dim,grid,EDGE);
    }
    free(chunk); free(enc);
    printf("v3 fixture: %s/zarr  (MC_ZARR_ROOT=%s/zarr/0)\n",out,out);
    return 0;
}
