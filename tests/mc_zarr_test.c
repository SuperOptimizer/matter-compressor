// mc_zarr_test — drive mc_zarr against a local zarr tree via a filesystem
// byte-source. Point MC_ZARR_ROOT at a level dir (containing zarr.json or
// .zarray); the test reports geometry and reads one inner chunk.
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *g_root;

// read_shard sink: count chunks that carry payload.
static void count_sink(void *ud, int cz, int cy, int cx,
                       const uint8_t *raw, size_t n) {
    (void)cz; (void)cy; (void)cx; (void)raw;
    if (n) (*(int *)ud)++;
}

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

// In-memory byte source serving a synthetic v2 `.zarray` with a chosen chunk
// edge — used to prove the inner-edge validation in mc_zarr_open.
static int g_edge;
static int mem_zarray_read(void *ud, const char *key, uint64_t off, uint64_t len,
                           uint8_t **out, size_t *out_len) {
    (void)ud; (void)off; (void)len;
    // No zarr.json -> mc_zarr_open falls back to .zarray (the v2 path).
    if (strcmp(key, ".zarray") != 0) { *out = NULL; *out_len = 0; return 0; }
    char j[256];
    int n = snprintf(j, sizeof j,
        "{\"zarr_format\":2,\"shape\":[512,512,512],\"chunks\":[%d,%d,%d],"
        "\"dtype\":\"|u1\",\"compressor\":null,\"fill_value\":0,\"order\":\"C\","
        "\"dimension_separator\":\".\"}", g_edge, g_edge, g_edge);
    *out = malloc((size_t)n);
    memcpy(*out, j, (size_t)n);
    *out_len = (size_t)n;
    return 0;
}

// REGRESSION: a too-small inner edge made sub = 256/edge exceed 2, so the
// transcode region's [8] sub-chunk arrays (decode_item.raw/oz/...) overflowed
// in mc_volume_prefetch_region (edge 64 -> sub 4 -> 64 sub-chunks). mc_zarr_open
// must now REJECT any edge that is not a divisor of the 256 region with sub<=2,
// i.e. accept only 128 and 256.
static int test_inner_edge_guard(void) {
    int bad = 0;
    struct { int edge, want_ok; } cases[] = {
        { 256, 1 }, { 128, 1 },   // valid: sub = 1, 2
        {  64, 0 },               // sub 4 -> would overflow [8] -> reject
        {  32, 0 },               // sub 8 -> reject
        {  96, 0 },               // not a divisor of 256 -> reject
        { 512, 0 },               // sub 0 (edge > region) -> reject
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        g_edge = cases[i].edge;
        mc_zarr *z = mc_zarr_open(mem_zarray_read, NULL);
        int ok = (z != NULL);
        if (z) mc_zarr_free(z);
        if (ok != cases[i].want_ok) {
            fprintf(stderr, "FAIL inner-edge guard: edge=%d opened=%d want=%d\n",
                    cases[i].edge, ok, cases[i].want_ok);
            bad++;
        }
    }
    if (!bad) printf("inner-edge guard (128/256 ok; 64/32/96/512 rejected) OK\n");
    return bad;
}

int main(int argc, char **argv) {
    if (test_inner_edge_guard()) return 1;

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

    // chunk_locate: resolve key + byte range for inner chunk (0,0,0) without fetching.
    char key[64]; uint64_t loff = 0, lnb = 0;
    int lst = mc_zarr_chunk_locate(z, 0, 0, 0, key, &loff, &lnb);
    printf("chunk_locate(0,0,0): st=%d key=%s off=%llu nb=%llu\n",
           lst, key, (unsigned long long)loff, (unsigned long long)lnb);

    // shard_index: enumerate present chunks in shard (0,0,0).
    char skey[64]; mc_zarr_range *rng = NULL; int rn = 0;
    int sst = mc_zarr_shard_index(z, 0, 0, 0, skey, &rng, &rn);
    printf("shard_index(0,0,0): st=%d key=%s nranges=%d\n", sst, skey, rn);
    free(rng);

    // read_shard: stream every present chunk of shard (0,0,0) through a sink.
    int shard_n = 0;
    mc_zarr_read_shard(z, 0, 0, 0, count_sink, &shard_n);
    printf("read_shard(0,0,0): %d chunks delivered\n", shard_n);

    mc_zarr_free(z);
    return err ? 1 : 0;
}
