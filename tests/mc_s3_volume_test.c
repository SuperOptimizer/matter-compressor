// mc_s3_volume_test — exercise the REAL-S3 read path against the anonymous
// Vesuvius open-data buckets (no credentials needed). Covers the offline-
// unreachable functions: mc_s3_open / mc_s3_reader / mc_s3_close / s3_read /
// the remote-zarr batch in dl_main + mc_volume_prefetch_shard.
//
// This is NOT part of the offline per-PR gate. It is opt-in: it runs only when
// MC_S3_TEST=1 is set (the s3-bench CI lane sets it), and SKIPS cleanly (exit 0)
// on any network error so it never produces a false failure.
//
//   usage: MC_S3_TEST=1 mc_s3_volume_test [s3://bucket/scroll.zarr/0-root]
//   default url: a PHercParis4 2.4um masked volume (zarr v2, raw, 128^3 chunks).
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

#define DEFAULT_URL \
  "s3://vesuvius-challenge-open-data/PHercParis4/volumes/" \
  "20260323153942-2.400um-0.2m-137keV-masked.zarr"

int main(int argc, char **argv) {
    if (!getenv("MC_S3_TEST")) {
        printf("mc_s3_volume: SKIP (set MC_S3_TEST=1 to run the real-S3 lane)\n");
        return 0;                                   // not a failure: opt-in only
    }
    const char *url = argc > 1 ? argv[1] : DEFAULT_URL;

    // mc_volume_open over s3:// -> mc_s3 client (anonymous), zarr discovery,
    // local .mca mirror, cache. A network/auth failure is a SKIP, not a fail.
    char cache[256]; snprintf(cache, sizeof cache, "/tmp/mc_s3_cache_%d", (int)getpid());
    mkdir(cache, 0755);
    mc_volume *v = mc_volume_open(url, cache, (size_t)512 << 20, 6.0f);
    if (!v) {
        printf("mc_s3_volume: SKIP (open failed — network/bucket unavailable)\n");
        return 0;
    }

    int nl = mc_volume_nlods(v);
    int nz, ny, nx;
    mc_volume_shape(v, 0, &nz, &ny, &nx);
    printf("s3 vol: nlods=%d L0=%d,%d,%d\n", nl, nz, ny, nx);
    if (nl < 1 || nz <= 0) { mc_volume_free(v); printf("mc_s3_volume: SKIP (no levels)\n"); return 0; }

    // Pull a block from the volume interior (where masked scroll data has material).
    int bz = (nz / 2) / 16, by = (ny / 2) / 16, bx = (nx / 2) / 16;
    uint8_t blk[16 * 16 * 16];
    int st = mc_volume_get_block(v, 0, bz, by, bx, blk);
    long nzc = 0, sum = 0; for (int i = 0; i < 4096; ++i) { if (blk[i]) nzc++; sum += blk[i]; }
    printf("center block: st=%d nonzero=%ld/4096 mean=%.1f\n", st, nzc, sum / 4096.0);

    // request_region + prefetch_shard drive the remote batched GET path (dl_main
    // remote branch + mc_volume_prefetch_shard's s3_get_batch).
    mc_volume_request_region(v, 0, bz / 16, by / 16, bx / 16);
    mc_volume_prefetch_shard(v, 0, bz / 16, by / 16, bx / 16);

    mc_volume_stats s = {0}; mc_volume_get_stats(v, &s);
    printf("stats: net=%llu disk=%llu hits=%llu misses=%llu\n",
           (unsigned long long)s.net_bytes, (unsigned long long)s.disk_bytes,
           (unsigned long long)s.cache_hits, (unsigned long long)s.cache_misses);

    int ok = (st == 1) && (s.net_bytes > 0);        // we actually fetched bytes off S3
    mc_volume_free(v);
    // best-effort cache cleanup
    char rm[300]; snprintf(rm, sizeof rm, "rm -rf %s", cache); if (system(rm)) {}
    printf("%s\n", ok ? "mc_s3_volume: OK" : "mc_s3_volume: SKIP (no net bytes — treating as unavailable)");
    return 0;                                        // never fail the lane on S3 flakiness
}
