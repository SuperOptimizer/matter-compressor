// Headless probe of a REMOTE .mca via the streaming volume path. Opens the
// archive, prints per-LOD shape, then runs a few freeze/thaw cycles to pull a
// center block and confirms it decodes. Mirrors exactly what VC3D's streaming
// createChunkCache does, so a green run here means the GUI path is sound.
//   usage: mc_stream_probe s3://bucket/key.mca
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <url.mca>\n", argv[0]); return 2; }
    const char *url = argv[1];
    mc_volume *v = mc_volume_open_streaming(url, "/tmp", (size_t)512 << 20);
    if (!v) { fprintf(stderr, "open_streaming FAILED for %s\n", url); return 1; }

    int nl = mc_volume_nlods(v);
    printf("nlods=%d\n", nl);
    int z0 = 0, y0 = 0, x0 = 0;
    for (int l = 0; l < nl; ++l) {
        int sz, sy, sx; mc_volume_shape(v, l, &sz, &sy, &sx);
        printf("  L%d  z=%d y=%d x=%d\n", l, sz, sy, sx);
        if (l == 0) { z0 = sz; y0 = sy; x0 = sx; }
    }

    // center 16^3 block at L0.
    int cz = (z0 / 2) / 16, cy = (y0 / 2) / 16, cx = (x0 / 2) / 16;
    uint8_t blk[16 * 16 * 16];
    int r = 0;
    for (int it = 0; it < 600; ++it) {
        mc_volume_freeze(v);
        r = mc_volume_try_block(v, 0, cz, cy, cx, blk);
        mc_volume_thaw(v);
        if (r == 1 || it % 20 == 0) printf("cycle %d: try_block(center)=%d\n", it, r);
        if (r == 1) break;
        struct timespec ts = {0, 50 * 1000 * 1000};   // 50ms tick, like the GUI
        nanosleep(&ts, NULL);
    }
    if (r == 1) {
        long sum = 0, nz = 0;
        for (int i = 0; i < 16 * 16 * 16; ++i) { sum += blk[i]; if (blk[i]) nz++; }
        printf("center block: mean=%.1f nonzero=%ld\n", sum / 4096.0, nz);
    }

    mc_volume_stats st = {0};
    mc_volume_get_stats(v, &st);
    printf("stats: used_blocks=%llu net_bytes=%llu\n",
           (unsigned long long)st.cache_used_blocks,
           (unsigned long long)st.net_bytes);

    // throughput: flood-request a NxNxN box of L0 regions around the center and
    // time until all are covered (the batched verbatim-copy path under load).
    {
        const int N = 5, half = N / 2;                  // 125 regions
        int rz0 = (z0 / 2) / 256, ry0 = (y0 / 2) / 256, rx0 = (x0 / 2) / 256;
        uint64_t bytes0 = st.net_bytes;
        struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int dz = -half; dz <= half; ++dz)
        for (int dy = -half; dy <= half; ++dy)
        for (int dx = -half; dx <= half; ++dx)
            mc_volume_request_region(v, 0, rz0 + dz, ry0 + dy, rx0 + dx);
        int done = 0;
        for (int it = 0; it < 2400 && !done; ++it) {    // up to 2 min
            mc_volume_freeze(v); mc_volume_thaw(v);     // keep the pipeline ticking
            mc_volume_get_stats(v, &st);
            done = (st.regions_inflight == 0);
            struct timespec ts = {0, 50 * 1000 * 1000};
            nanosleep(&ts, NULL);
        }
        struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
        double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        mc_volume_get_stats(v, &st);
        double mb = (st.net_bytes - bytes0) / 1048576.0;
        printf("flood: %d regions, %.1fMB in %.1fs = %.1f MB/s (inflight=%llu)\n",
               N * N * N, mb, sec, mb / sec, (unsigned long long)st.regions_inflight);
    }
    mc_volume_free(v);
    printf("DONE\n");
    return 0;
}
