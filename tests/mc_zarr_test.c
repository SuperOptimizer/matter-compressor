// mc_zarr_test — drive mc_zarr against a local zarr tree via a filesystem
// byte-source. Point MC_ZARR_ROOT at a level dir (containing zarr.json or
// .zarray); the test reports geometry and reads one inner chunk.
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *g_root;

// filesystem byte source: key is relative to the level dir.
static int fs_read(void *ud, const char *key, uint64_t off, uint64_t len,
                   uint8_t **out, size_t *out_len) {
    (void)ud;
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", g_root, key);
    FILE *f = fopen(path, "rb");
    if (!f) { *out = NULL; *out_len = 0; return 0; }   // 404 == absent, not error
    fseek(f, 0, SEEK_END);
    long total = ftell(f);
    if (total < 0) { fclose(f); return -1; }
    uint64_t start = off;
    uint64_t want = (len == 0) ? (uint64_t)total - start : len;
    if (start > (uint64_t)total) { fclose(f); *out = NULL; *out_len = 0; return 0; }
    if (start + want > (uint64_t)total) want = (uint64_t)total - start;
    uint8_t *buf = malloc(want ? want : 1);
    fseek(f, (long)start, SEEK_SET);
    size_t got = fread(buf, 1, want, f);
    fclose(f);
    *out = buf;
    *out_len = got;
    return 0;
}

int main(int argc, char **argv) {
    g_root = argc > 1 ? argv[1] : getenv("MC_ZARR_ROOT");
    if (!g_root) { fprintf(stderr, "usage: %s <level-dir>\n", argv[0]); return 2; }

    mc_zarr *z = mc_zarr_open(fs_read, NULL);
    if (!z) { fprintf(stderr, "mc_zarr_open failed\n"); return 1; }

    int nz, ny, nx;
    mc_zarr_shape(z, &nz, &ny, &nx);
    int gz, gy, gx;
    mc_zarr_inner_grid(z, &gz, &gy, &gx);
    printf("codec=%s shape=%d,%d,%d inner_edge=%d shard_edge=%d inner_grid=%d,%d,%d\n",
           mc_zarr_inner_codec(z), nz, ny, nx, mc_zarr_inner_edge(z),
           mc_zarr_shard_edge(z), gz, gy, gx);

    // probe a few inner chunks for presence.
    int found = 0, air = 0, err = 0;
    for (int cz = 0; cz < gz && cz < 4; ++cz)
        for (int cy = 0; cy < gy && cy < 4; ++cy)
            for (int cx = 0; cx < gx && cx < 4; ++cx) {
                uint8_t *raw = NULL;
                size_t len = 0;
                int st = mc_zarr_read_inner(z, cz, cy, cx, &raw, &len);
                if (st < 0) { err++; }
                else if (st == 1) { air++; }
                else { found++; if (found == 1) printf("first present (%d,%d,%d): %zu raw bytes\n", cz, cy, cx, len); }
                free(raw);
            }
    printf("probe 4x4x4: found=%d air=%d err=%d\n", found, air, err);

    // shard_all_air on shard 0.
    printf("shard(0,0,0) all_air=%d\n", mc_zarr_shard_all_air(z, 0, 0, 0));

    mc_zarr_free(z);
    return err ? 1 : 0;
}
