// mc_volume_v3_test — drive the mc_volume c3d-transcode path against a local
// zarr v3 / c3d-sharded fixture (no network). This is the branch the v2/raw
// fixture can't reach: v3 footer index, mc_zarr_read_inner (c3d payload),
// c3d decode in the decode pool, and mc_volume_prefetch_shard's full v3 body
// (shard_index -> batched chunk reads -> decode_push).
//
//   usage: mc_volume_v3_test <zarr-v3-root-dir> [cache-dir]
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int fails=0;
#define CHECK(c,...) do{ if(!(c)){ fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
    fprintf(stderr,__VA_ARGS__); fputc('\n',stderr); fails++; } }while(0)

int main(int argc,char**argv){
    const char *root=argc>1?argv[1]:getenv("MC_VOLUME_V3_URL");
    const char *cache=argc>2?argv[2]:"/tmp";
    if(!root){ fprintf(stderr,"usage: %s <zarr-v3-root> [cache]\n",argv[0]); return 2; }
    { char mca[1280]; snprintf(mca,sizeof mca,"%s/zarr.mca",cache); remove(mca); }

    mc_volume *v=mc_volume_open(root,cache,(size_t)256<<20,6.0f);
    CHECK(v!=NULL,"mc_volume_open(v3) failed");
    if(!v) return 1;

    int nl=mc_volume_nlods(v);
    int nz,ny,nx; mc_volume_shape(v,0,&nz,&ny,&nx);
    printf("v3 vol: nlods=%d L0=%d,%d,%d\n",nl,nz,ny,nx);
    CHECK(nl>=1&&nz>0,"bad shape");

    // level meta should report c3d codec + inner edge 256.
    mc_volume_level_meta m={0};
    if(mc_volume_get_level_meta(v,0,&m)==0)
        printf("L0 codec=%s inner=%d shard=%d\n",m.codec,m.inner_edge,m.shard_edge);

    // prefetch_shard drives the v3 footer index + batched c3d chunk reads +
    // decode pool (sub==1 path). cz/cy/cx here are source inner-chunk coords.
    mc_volume_prefetch_shard(v,0,0,0,0);

    // center block (sphere interior) must transcode + decode to material.
    int bz=(nz/2)/16, by=(ny/2)/16, bx=(nx/2)/16;
    uint8_t blk[16*16*16];
    int st=mc_volume_get_block(v,0,bz,by,bx,blk);
    long nzc=0; for(int i=0;i<4096;++i) if(blk[i]) nzc++;
    printf("center: st=%d nonzero=%ld/4096\n",st,nzc);
    CHECK(st==1,"center get_block st=%d (v3 transcode)",st);
    CHECK(nzc>0,"center decoded empty");

    // whole coarse level prefetch (walks all shards through prefetch_shard).
    if(nl>1){ volatile int cancel=0; mc_volume_prefetch_level(v,nl-1,2,&cancel); }

    mc_volume_stats s={0}; mc_volume_get_stats(v,&s);
    printf("stats: net=%llu disk=%llu hits=%llu misses=%llu\n",
        (unsigned long long)s.net_bytes,(unsigned long long)s.disk_bytes,
        (unsigned long long)s.cache_hits,(unsigned long long)s.cache_misses);

    mc_volume_free(v);
    printf(fails?"mc_volume_v3: %d FAILURES\n":"mc_volume_v3: OK\n",fails);
    return fails?1:0;
}
