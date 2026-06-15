// mc_make_fixture — emit a tiny on-disk **zarr v2 / raw-u8** tree so the
// zarr/volume/streaming read paths run fully offline (no S3, no network, no
// c3d encode dependency — v2 "raw" chunks are just uncompressed u8).
//
//   usage: mc_make_fixture <out_dir>
//   writes: <out_dir>/zarr/<L>/.zarray         (L = 0..NLEV-1, halved per level)
//           <out_dir>/zarr/<L>/<cz>.<cy>.<cx>  (raw u8 inner chunks; air omitted)
//
// Point MC_ZARR_ROOT at <out_dir>/zarr/0 (a level dir) for mc_zarr_test, and
// mc_volume_open(<out_dir>/zarr, ...) for the volume path (it probes /0,/1,..).
//
// Content: a centered solid sphere (value 200) in air (0). Chunks that are
// entirely air are not written, so the absent-key == air path is exercised too.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>

#define NLEV 3
// Inner (== shard) edge for v2. The mc_volume transcode path supports v2 inner
// edges of 256 (sub=1) or 128 (sub=2) only — its decode_item sub-chunk arrays
// are sized [8] = 2^3, so a 256^3 region holds at most 2x2x2 inner chunks.
#define CHUNK 128

static int mkdirp(const char *path) {
    char tmp[2048]; snprintf(tmp, sizeof tmp, "%s", path);
    for (char *p = tmp + 1; *p; ++p)
        if (*p == '/') { *p = 0; mkdir(tmp, 0755); *p = '/'; }
    return mkdir(tmp, 0755);
}

static void write_file(const char *path, const void *data, size_t len) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    if (len) fwrite(data, 1, len, f);
    fclose(f);
}

// sphere of radius ~0.35*dim at full res, scaled per level.
static uint8_t voxel(int gx, int gy, int gz, int dim) {
    double c = dim / 2.0, r = dim * 0.35;
    double d = (gx - c) * (gx - c) + (gy - c) * (gy - c) + (gz - c) * (gz - c);
    return d < r * r ? 200 : 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <out_dir>\n", argv[0]); return 2; }
    const char *out = argv[1];
    int base = 512;   // LOD0 cube edge: 2^3 = 8 volume regions of 256^3, each
                      // 2^3 inner chunks of edge 128 -> exercises the multi-region
                      // prefetch_level walk + per-region decode-pool batching.

    for (int L = 0; L < NLEV; ++L) {
        int dim = base >> L; if (dim < CHUNK) dim = CHUNK;
        char ldir[2048]; snprintf(ldir, sizeof ldir, "%s/zarr/%d", out, L);
        mkdirp(ldir);

        // .zarray: v2, raw (compressor null), u8, dimension_separator '.'
        char za[1024];
        int n = snprintf(za, sizeof za,
            "{\"zarr_format\":2,\"shape\":[%d,%d,%d],"
            "\"chunks\":[%d,%d,%d],\"dtype\":\"|u1\","
            "\"compressor\":null,\"fill_value\":0,"
            "\"order\":\"C\",\"dimension_separator\":\".\"}",
            dim, dim, dim, CHUNK, CHUNK, CHUNK);
        char zpath[2200]; snprintf(zpath, sizeof zpath, "%s/.zarray", ldir);
        write_file(zpath, za, (size_t)n);

        int grid = (dim + CHUNK - 1) / CHUNK;
        uint8_t *chunk = malloc((size_t)CHUNK * CHUNK * CHUNK);
        for (int cz = 0; cz < grid; ++cz)
        for (int cy = 0; cy < grid; ++cy)
        for (int cx = 0; cx < grid; ++cx) {
            int any = 0;
            for (int z = 0; z < CHUNK; ++z)
            for (int y = 0; y < CHUNK; ++y)
            for (int x = 0; x < CHUNK; ++x) {
                int gx = cx * CHUNK + x, gy = cy * CHUNK + y, gz = cz * CHUNK + z;
                uint8_t v = (gx < dim && gy < dim && gz < dim)
                          ? voxel(gx << L, gy << L, gz << L, base) : 0;
                chunk[(z * CHUNK + y) * CHUNK + x] = v;
                any |= v;
            }
            if (!any) continue;   // all-air chunk: omit (exercises absent==air)
            char cpath[2300];
            snprintf(cpath, sizeof cpath, "%s/%d.%d.%d", ldir, cz, cy, cx);
            write_file(cpath, chunk, (size_t)CHUNK * CHUNK * CHUNK);
        }
        free(chunk);
        printf("L%d dim=%d grid=%d^3\n", L, dim, grid);
    }
    printf("fixture: %s/zarr  (MC_ZARR_ROOT=%s/zarr/0)\n", out, out);
    return 0;
}
