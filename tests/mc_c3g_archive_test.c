// mc_c3g_archive_test — the .mca archive format carries a block-codec field
// (MCH_BLOCKCODEC). A c3g archive must:
//   1. round-trip: decode == the CABAC archive decode of the same volume
//      (c3g recovers CABAC's exact levels, so with max_err==0 they're identical),
//   2. report its codec via mc_archive_block_codec, and
//   3. an existing (CABAC) archive must still read as CABAC (default 0).
// Uses the low-level archive API (open_dims + set_block_codec + append_chunk_ctx
// + decode_block) so it drives the new header field directly. No external deps.
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DIM 256                 // one 256^3 chunk
static int fails = 0;
#define CHECK(x) do{ if(!(x)){ fails++; fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);} }while(0)

// smooth ball + air, like mc_roundtrip's source.
static void fill_vol(mc_u8 *vox){
    double c=DIM/2.0;
    for(int z=0;z<DIM;++z)for(int y=0;y<DIM;++y)for(int x=0;x<DIM;++x){
        double r=sqrt((x-c)*(x-c)+(y-c)*(y-c)+(z-c)*(z-c));
        int v=0;
        if(r<=DIM*0.4){ double f=128.0+100.0*cos(r*0.15); if(f<1)f=1; if(f>255)f=255; v=(int)f; }
        vox[((size_t)z*DIM+y)*DIM+x]=(mc_u8)v;
    }
}

// build a one-chunk archive at `path` with the given block codec; return its
// decoded 16^3-block grid flattened into `out` (DIM^3).
static int build_and_decode(const char *path, uint32_t codec, const mc_u8 *vox, mc_u8 *out){
    remove(path);
    mc_archive *a = mc_archive_open_dims(path, DIM, DIM, DIM, 8.0f);
    if(!a){ fprintf(stderr,"open_dims failed\n"); return -1; }
    CHECK(mc_archive_set_block_codec(a, codec) == 0);
    CHECK(mc_archive_block_codec(a) == codec);

    mc_codec_ctx *C = mc_codec_ctx_new();
    mc_codec_ctx_set_quality(C, 8.0f);
    mc_codec_ctx_set_max_error(C, 0);
    CHECK(mc_archive_append_chunk_ctx(a, C, 0, 0,0,0, vox) == 0);

    uint64_t co = mc_archive_chunk_offset(a, 0, 0,0,0);
    CHECK(co != 0);
    mc_u8 blk[16*16*16];
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
        mc_archive_decode_block(a, co, bz,by,bx, blk);
        for(int z=0;z<16;++z)for(int y=0;y<16;++y)for(int x=0;x<16;++x){
            int gx=bx*16+x, gy=by*16+y, gz=bz*16+z;
            out[((size_t)gz*DIM+gy)*DIM+gx] = blk[(z*16+y)*16+x];
        }
    }
    mc_codec_ctx_free(C);
    mc_archive_close(a);
    return 0;
}

int main(void){
    mc_u8 *vox = malloc((size_t)DIM*DIM*DIM);
    mc_u8 *cab = malloc((size_t)DIM*DIM*DIM);
    mc_u8 *c3g = malloc((size_t)DIM*DIM*DIM);
    CHECK(vox&&cab&&c3g);
    fill_vol(vox);

    CHECK(build_and_decode("/tmp/mc_c3g_cabac.mca", MC_BLOCKCODEC_CABAC, vox, cab) == 0);
    CHECK(build_and_decode("/tmp/mc_c3g_c3g.mca",   MC_BLOCKCODEC_C3G,   vox, c3g) == 0);

    // c3g archive decode must equal the CABAC archive decode, voxel-for-voxel.
    long diff=0, maxd=0, air=0, airbad=0;
    for(size_t i=0;i<(size_t)DIM*DIM*DIM;++i){
        int d=cab[i]-c3g[i]; if(d<0)d=-d;
        if(d){ diff++; if(d>maxd)maxd=d; }
        if(vox[i]==0){ air++; if(c3g[i]!=0) airbad++; }
    }
    CHECK(diff==0);
    CHECK(airbad==0);
    printf("c3g vs CABAC archive decode: %ld voxels differ (maxdiff=%ld), air leaks=%ld/%ld\n",
           diff, maxd, airbad, air);

    // reopening the c3g archive reports the codec from the header.
    mc_archive *re = mc_archive_open_dims("/tmp/mc_c3g_c3g.mca", DIM, DIM, DIM, 8.0f);
    CHECK(re != NULL);
    CHECK(mc_archive_block_codec(re) == MC_BLOCKCODEC_C3G);
    mc_archive_close(re);

    // and the CABAC archive reports CABAC (default).
    mc_archive *rc = mc_archive_open_dims("/tmp/mc_c3g_cabac.mca", DIM, DIM, DIM, 8.0f);
    CHECK(rc != NULL);
    CHECK(mc_archive_block_codec(rc) == MC_BLOCKCODEC_CABAC);
    mc_archive_close(rc);

    free(vox); free(cab); free(c3g);
    printf(fails ? "mc_c3g_archive_test: %d FAILED\n" : "mc_c3g_archive_test: OK\n", fails);
    return fails ? 1 : 0;
}
