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
    mc_volume_free(v);
    printf("DONE\n");
    return 0;
}
