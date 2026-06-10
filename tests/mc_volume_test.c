// mc_volume_test — open a remote zarr volume, fetch a block, report. Drives the
// full stream->transcode->cache path. usage: mc_volume_test <zarr-root-url> <cache-dir>
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <zarr-root-url> <cache-dir>\n", argv[0]); return 2; }
    mc_volume *v = mc_volume_open(argv[1], argv[2], (size_t)1 << 30, 8.0f);
    if (!v) { fprintf(stderr, "mc_volume_open failed\n"); return 1; }

    printf("nlods=%d\n", mc_volume_nlods(v));
    for (int l = 0; l < mc_volume_nlods(v); ++l) {
        int nz, ny, nx, gz, gy, gx;
        mc_volume_shape(v, l, &nz, &ny, &nx);
        mc_volume_block_grid(v, l, &gz, &gy, &gx);
        printf("  lod%d shape=%d,%d,%d blocks=%d,%d,%d\n", l, nz, ny, nx, gz, gy, gx);
    }

    // shard 7/3/3 of lod0 is populated; inner (112,48,48) -> block coords *16.
    int bz = 112 * 16, by = 48 * 16, bx = 48 * 16;
    uint8_t blk[16 * 16 * 16];
    printf("get_block lod0 block(%d,%d,%d)...\n", bz, by, bx);
    int st = mc_volume_get_block(v, 0, bz, by, bx, blk);
    long sum = 0, nzc = 0;
    for (int i = 0; i < 4096; ++i) { sum += blk[i]; if (blk[i]) nzc++; }
    printf("  st=%d mean=%.1f nonzero=%ld/4096\n", st, sum / 4096.0, nzc);

    mc_volume_stats s;
    mc_volume_get_stats(v, &s);
    printf("stats: net=%llu disk=%llu hits=%llu misses=%llu inflight=%llu\n",
           (unsigned long long)s.net_bytes, (unsigned long long)s.disk_bytes,
           (unsigned long long)s.cache_hits, (unsigned long long)s.cache_misses,
           (unsigned long long)s.regions_inflight);

    // second get of the same block -> cache hit, no net.
    st = mc_volume_get_block(v, 0, bz, by, bx, blk);
    mc_volume_get_stats(v, &s);
    printf("re-get st=%d net=%llu (should be unchanged)\n", st, (unsigned long long)s.net_bytes);

    mc_volume_free(v);
    return st == 1 ? 0 : 1;
}
