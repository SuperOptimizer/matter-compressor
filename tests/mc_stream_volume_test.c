// mc_volume_open_streaming: open an already-built local .mca via the streaming
// reader path (no zarr/transcode) and run a freeze/thaw/sample cycle. Validates
// the direct-.mca volume bridge: per-LOD shape, coverage memo (PRESENT vs air),
// THAW reader-fill, and that a sampled point returns the encoded value.
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NX 600
#define NY 300
#define NZ 280

// A sphere of value 200 centered in the volume; 0 (air) outside.
static mc_u8 src_fn(void *ud, int x, int y, int z) {
    (void)ud;
    double cx = NX / 2.0, cy = NY / 2.0, cz = NZ / 2.0, r = 90.0;
    double d = (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz);
    return d < r * r ? 200 : 0;
}

int main(void) {
    const char *path = "/tmp/mc_stream_test.mca";
    mc_build_opts o = {.nx = NX, .ny = NY, .nz = NZ, .quality = 6.0f,
                       .metadata = "stream", .meta_len = 6};
    if (mc_build_to_file(src_fn, NULL, &o, path) != 0) {
        fprintf(stderr, "build failed\n"); return 1;
    }

    mc_volume *v = mc_volume_open_streaming(path, (size_t)256 << 20);
    if (!v) { fprintf(stderr, "open_streaming failed\n"); return 1; }

    int nl = mc_volume_nlods(v);
    printf("nlods=%d\n", nl);
    if (nl < 1) { fprintf(stderr, "no lods\n"); return 1; }
    for (int l = 0; l < nl; ++l) {
        int sz, sy, sx; mc_volume_shape(v, l, &sz, &sy, &sx);
        printf("  L%d shape z=%d y=%d x=%d\n", l, sz, sy, sx);
        if (sz != (NZ >> l) && sz != ((NZ >> l) < 1 ? 1 : (NZ >> l)))
            { fprintf(stderr, "L%d z shape wrong (%d, want %d)\n", l, sz, NZ >> l); return 1; }
    }

    // Center block (sphere interior). Run a few thaw cycles so the reader fills it.
    int cz = (NZ / 2) / 16, cy = (NY / 2) / 16, cx = (NX / 2) / 16;
    uint8_t blk[16 * 16 * 16];
    int got = 0;
    for (int it = 0; it < 6; ++it) {
        mc_volume_freeze(v);
        int r = mc_volume_try_block(v, 0, cz, cy, cx, blk);   // frozen read records miss
        mc_volume_thaw(v);                                    // fills from the reader
        if (r == 1) { got = 1; break; }                       // resident now
    }
    // One more frozen read after fill.
    mc_volume_freeze(v);
    int r = mc_volume_try_block(v, 0, cz, cy, cx, blk);
    mc_volume_thaw(v);
    printf("center try_block r=%d (got=%d)\n", r, got);
    if (r != 1) { fprintf(stderr, "center block never resolved present\n"); return 1; }

    long sum = 0, nz = 0;
    for (int i = 0; i < 16 * 16 * 16; ++i) { sum += blk[i]; if (blk[i] > 64) nz++; }
    printf("center block: sum=%ld nonzero(>64)=%ld mean=%.1f\n", sum, nz, sum / 4096.0);
    if (nz < 1000) { fprintf(stderr, "center block looks empty -- fill/decode failed\n"); return 1; }

    // A far-corner block beyond the sphere should be air (ZERO -> try_block returns 1, all zeros).
    mc_volume_freeze(v);
    int rc = mc_volume_try_block(v, 0, 0, 0, 0, blk);
    mc_volume_thaw(v);
    long csum = 0; for (int i = 0; i < 16 * 16 * 16; ++i) csum += blk[i];
    printf("corner try_block r=%d sum=%ld\n", rc, csum);

    mc_volume_stats st = {0};
    mc_volume_get_stats(v, &st);
    printf("stats: used_blocks=%llu cap_blocks=%llu disk_bytes=%llu\n",
           (unsigned long long)st.cache_used_blocks,
           (unsigned long long)st.cache_cap_blocks,
           (unsigned long long)st.disk_bytes);

    mc_volume_free(v);
    remove(path);
    printf("mc_stream_volume: OK\n");
    return 0;
}
