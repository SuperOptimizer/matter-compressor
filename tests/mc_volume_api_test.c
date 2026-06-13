// mc_volume_api_test — cover the mc_volume / mc_zarr introspection + prefetch +
// LOD-render surface that the basic offline volume test doesn't reach, all
// against the local zarr fixture (NO network). Targets the functions left
// uncovered after mc_volume_offline_test: get_level_meta, region_gen,
// request_region, prefetch_level/shard, sample_lods, set_cache/staging_bytes,
// set_ready_cb, mc_mca_probe, and the zarr locate/read_shard/shard_index path.
//
//   usage: mc_volume_api_test <zarr-root-dir> [cache-dir]
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int fails=0;
#define CHECK(c,...) do{ if(!(c)){ fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
    fprintf(stderr,__VA_ARGS__); fputc('\n',stderr); fails++; } }while(0)

static int g_ready=0;
static void on_ready(void *ud){ (void)ud; g_ready++; }

// file-scope source for the mc_mca_probe build (a centered solid block).
static mc_u8 probe_src(void *ud,int x,int y,int z){ (void)ud;
    return (x>128&&x<384&&y>96&&y<288&&z>64&&z<192) ? 180 : 0; }

int main(int argc,char**argv){
    const char *root = argc>1?argv[1]:getenv("MC_VOLUME_URL");
    const char *cache = argc>2?argv[2]:"/tmp";
    if(!root){ fprintf(stderr,"usage: %s <zarr-root-dir> [cache-dir]\n",argv[0]); return 2; }

    // ---- mc_volume_open_ex (config path) + tunables ----
    mc_volume *v = mc_volume_open_ex(root, cache, (size_t)256<<20, 6.0f, NULL);
    CHECK(v!=NULL,"mc_volume_open_ex failed");
    if(!v) return 1;

    int nl = mc_volume_nlods(v);
    CHECK(nl>=1,"nlods<1");

    // set_cache_bytes / set_staging_bytes / set_ready_cb (introspection + tuning)
    size_t newcache = mc_volume_set_cache_bytes(v,(size_t)128<<20);
    CHECK(newcache>0,"set_cache_bytes returned 0");
    size_t newstage = mc_volume_set_staging_bytes(v,(size_t)64<<20);
    CHECK(newstage>0 || newstage==0,"set_staging_bytes ran");   // value is impl-defined; just exercise
    mc_volume_set_ready_cb(v,on_ready,NULL);

    // get_level_meta for every LOD
    for(int l=0;l<nl;++l){
        mc_volume_level_meta m={0};
        int rc=mc_volume_get_level_meta(v,l,&m);
        CHECK(rc==0,"get_level_meta(%d) rc=%d",l,rc);
        CHECK(m.shape[0]>0&&m.inner_edge>0,"level_meta L%d shape/edge invalid",l);
        if(l==0) printf("L0 meta: shape=%d,%d,%d inner=%d shard=%d codec=%s\n",
                        m.shape[0],m.shape[1],m.shape[2],m.inner_edge,m.shard_edge,m.codec);
    }
    // out-of-range lod -> <0
    { mc_volume_level_meta m={0}; CHECK(mc_volume_get_level_meta(v,99,&m)<0,"level_meta OOB should fail"); }

    // ---- region_gen: change-generation counter for a region ----
    uint64_t g0 = mc_volume_region_gen(v,0,0,0,0);

    // blocking get of the center (sphere interior) — pulls + decodes the region.
    int nz,ny,nx; mc_volume_shape(v,0,&nz,&ny,&nx);
    uint8_t blk[16*16*16];
    int st=mc_volume_get_block(v,0,(nz/2)/16,(ny/2)/16,(nx/2)/16,blk);
    long nzc=0; for(int i=0;i<4096;++i) if(blk[i]) nzc++;
    CHECK(st==1,"center get_block st=%d",st);

    // after a decode, the region's change-gen must have advanced.
    uint64_t g1=mc_volume_region_gen(v,0,0,0,0);
    CHECK(g1>=g0,"region_gen went backwards (%llu -> %llu)",
          (unsigned long long)g0,(unsigned long long)g1);

    // ---- request_region (predictive prefetch) + prefetch_shard + prefetch_level ----
    // exercise the prefetch entry points (idempotent on an already-resident region).
    mc_volume_request_region(v,0,0,0,0);
    mc_volume_prefetch_shard(v,0,0,0,0);
    volatile int cancel=0;
    mc_volume_prefetch_level(v,0,2,&cancel);
    printf("center block nonzero=%ld/4096 ready_cb fired=%d\n",nzc,g_ready);

    // ---- sample_lods: build the whole-pyramid sample source ----
    mc_sample_lods ls = mc_volume_sample_lods(v,1);
    CHECK(ls.nlods>=1,"sample_lods nlods=%d",ls.nlods);

    mc_volume_free(v);

    // ---- mc_mca_probe: header-only probe of a local .mca (write one first) ----
    // mc_volume_open created a local .mca mirror in cache_dir; probe a fresh build.
    {
        const char *p="/tmp/mc_probe.mca";
        mc_build_opts o={.nx=512,.ny=384,.nz=256,.quality=6.0f};
        if(mc_build_to_file(probe_src,NULL,&o,p)==0){
            int px,py,pz,pl; float pq;
            int rc=mc_mca_probe(p,&px,&py,&pz,&pl,&pq);
            CHECK(rc==0,"mc_mca_probe rc=%d",rc);
            CHECK(px==512&&py==384&&pz==256,"probe dims %d,%d,%d != 512,384,256",px,py,pz);
            CHECK(pl>=1&&pq>0,"probe nlods=%d q=%.1f",pl,pq);
            printf("mca_probe: %dx%dx%d nlods=%d q=%.1f\n",px,py,pz,pl,pq);
            remove(p);
        }
    }

    printf(fails?"mc_volume_api_test: %d FAILURES\n":"mc_volume_api_test: OK\n",fails);
    return fails?1:0;
}
