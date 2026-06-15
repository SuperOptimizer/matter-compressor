// mc_volume_offline_test — drive the full mc_volume path (zarr discovery ->
// local-dir read -> transcode -> archive -> cache -> sample) against an
// on-disk **local** zarr fixture, with NO network. Complements
// mc_volume_test.c (which probes a real remote volume).
//
//   usage: mc_volume_offline_test <zarr-root-dir> [cache-dir]
//   or set MC_VOLUME_URL=<zarr-root-dir>. The fixture is produced by
//   tests/support/mc_make_fixture (a v2 raw-u8 sphere, 3 LODs).
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] : getenv("MC_VOLUME_URL");
    const char *cache = argc > 2 ? argv[2] : "/tmp";
    if (!root) { fprintf(stderr, "usage: %s <zarr-root-dir> [cache-dir]\n", argv[0]); return 2; }

    // mc_volume_open writes a derived <cache>/<root-basename>.mca mirror. Clear a
    // stale one first (e.g. left by a prior run with different fixture dims) — the
    // mirror is rebuildable and mc_archive_open rightly refuses a dims-mismatch.
    { char mca[1280]; snprintf(mca, sizeof mca, "%s/zarr.mca", cache); remove(mca); }

    mc_volume *v = mc_volume_open(root, cache, (size_t)256 << 20, 6.0f);
    if (!v) { fprintf(stderr, "mc_volume_open(%s) failed\n", root); return 1; }

    int nl = mc_volume_nlods(v);
    printf("nlods=%d\n", nl);
    if (nl < 1) { mc_volume_free(v); return 1; }
    for (int l = 0; l < nl; ++l) {
        int nz, ny, nx, gz, gy, gx;
        mc_volume_shape(v, l, &nz, &ny, &nx);
        mc_volume_block_grid(v, l, &gz, &gy, &gx);
        printf("  lod%d shape=%d,%d,%d blocks=%d,%d,%d\n", l, nz, ny, nx, gz, gy, gx);
    }

    // Center block of LOD0 sits inside the sphere -> must fetch + decode nonzero.
    int nz, ny, nx;
    mc_volume_shape(v, 0, &nz, &ny, &nx);
    int bz = (nz / 2) / 16, by = (ny / 2) / 16, bx = (nx / 2) / 16;
    uint8_t blk[16 * 16 * 16];

    int got = 0;
    for (int it = 0; it < 400 && !got; ++it) {
        mc_volume_freeze(v);
        if (mc_volume_try_block(v, 0, bz, by, bx, blk) == 1) got = 1;
        mc_volume_thaw(v);
        if (!got) { struct timespec ts = {0, 2 * 1000 * 1000}; nanosleep(&ts, NULL); }
    }
    // Blocking fetch as the authoritative check.
    int st = mc_volume_get_block(v, 0, bz, by, bx, blk);
    long sum = 0, nzc = 0;
    for (int i = 0; i < 4096; ++i) { sum += blk[i]; if (blk[i]) nzc++; }
    printf("center try_block got=%d  get_block st=%d mean=%.1f nonzero=%ld/4096\n",
           got, st, sum / 4096.0, nzc);

    // Stats + a sample source for the renderer path.
    mc_volume_stats s; mc_volume_get_stats(v, &s);
    printf("stats: net=%llu disk=%llu hits=%llu misses=%llu\n",
           (unsigned long long)s.net_bytes, (unsigned long long)s.disk_bytes,
           (unsigned long long)s.cache_hits, (unsigned long long)s.cache_misses);

    mc_sample_src src = mc_volume_sample_src(v, 0, 1);
    (void)src;
    int net_before = (int)s.net_bytes;
    st = mc_volume_get_block(v, 0, bz, by, bx, blk);   // re-get: cache hit, no net
    mc_volume_get_stats(v, &s);
    printf("re-get st=%d net_delta=%lld (expect 0)\n", st,
           (long long)s.net_bytes - net_before);

    int ok = (st == 1) && (nzc > 1000);
    mc_volume_free(v);
    printf("%s\n", ok ? "mc_volume_offline: OK" : "mc_volume_offline: FAIL");
    return ok ? 0 : 1;
}
