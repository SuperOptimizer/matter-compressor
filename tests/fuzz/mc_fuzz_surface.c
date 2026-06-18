// mc_fuzz_surface — fuzz the surface text/binary loaders against arbitrary,
// untrusted input. All three parse external files (Vesuvius segment OBJs, VC
// per-pixel maps) and allocate from attacker-controlled header fields:
//   mc_surface_load_obj   — "# grid W H" hint sizes a W*H*3 float grid
//   mc_mesh_load_obj      — growable v/vn/tri arrays (realloc), fan-triangulate
//   mc_surface_load_vcps_ppm — "width/height/dim" header sizes a W*H grid, then
//                              reads W*H*dim doubles
// Contract: for ANY bytes each must reject cleanly or return a self-consistent
// structure — never an OOB read/write, NULL-deref, or unbounded alloc. Found
// bugs are fixed in src/mc_surface.c (see tests/fuzz/FINDINGS.md). On a clean
// load we walk the whole returned structure so ASan catches a bad extent.
//
// Loaders take a PATH, so we materialize each input to a temp file.
//   AFL_USE_ASAN=1 AFL_USE_UBSAN=1 afl-clang-fast \
//     tests/fuzz/mc_fuzz_surface.c src/mc_surface.c tiff/tiff.c \
//     -Isrc -Itiff -O1 -g -lm -o mc_fuzz_surface
//   afl-fuzz -i corpus -o findings -- ./mc_fuzz_surface @@
// Standalone replay (CI smoke / crash replay): -DMC_FUZZ_STANDALONE.
#include "mc_surface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

static void walk_surface(const mc_surface *s) {
    if (!s->grid || s->gw < 1 || s->gh < 1) return;
    size_t n = (size_t)s->gw * s->gh;
    volatile double sink = 0;
    for (size_t i = 0; i < n; ++i) {
        sink += s->grid[i*3] + s->grid[i*3+1] + s->grid[i*3+2];
        if (s->depth) sink += s->depth[i];
    }
    (void)sink;
}

static void exercise(const uint8_t *data, size_t size) {
    if (size == 0 || size > (1u<<22)) return;
    char path[] = "/tmp/mc_fuzz_surf_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return;
    if (write(fd, data, size) != (ssize_t)size) { close(fd); unlink(path); return; }
    close(fd);

    mc_surface s;
    if (mc_surface_load_obj(path, &s) == 0) { walk_surface(&s); mc_surface_free(&s); }
    if (mc_surface_load_vcps_ppm(path, &s, 5.0f) == 0) { walk_surface(&s); mc_surface_free(&s); }

    mc_mesh m;
    if (mc_mesh_load_obj(path, &m) == 0) {
        // every triangle index MUST be a valid vertex (else any tri consumer
        // reads m->v out of bounds). Verify, then touch all vertices/normals.
        for (int i = 0; i < m.nt; ++i)
            for (int k = 0; k < 3; ++k) {
                int vi = m.tri[i*3+k];
                if (vi < 0 || vi >= m.nv) abort();      // OOB triangle index escaped
            }
        volatile double sink = 0;
        for (int i = 0; i < m.nv; ++i) {
            sink += m.v[i*3] + m.v[i*3+1] + m.v[i*3+2];
            if (m.vn) sink += m.vn[i*3];
        }
        (void)sink;
        mc_mesh_free(&m);
    }
    unlink(path);
}

#ifdef MC_FUZZ_STANDALONE
static int replay_file(const char *path) {
    FILE *f = path ? fopen(path, "rb") : stdin;
    if (!f) { perror(path); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 0) { if (path) fclose(f); return 0; }
    uint8_t *buf = malloc((size_t)n);
    size_t got = fread(buf, 1, (size_t)n, f);
    if (path) fclose(f);
    exercise(buf, got);
    free(buf);
    return 0;
}
int main(int argc, char **argv) {
    if (argc < 2) return replay_file(NULL);
    for (int i = 1; i < argc; ++i) replay_file(argv[i]);
    printf("mc_fuzz_surface: replayed %d input(s), no crash\n", argc - 1);
    return 0;
}
#else
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    exercise(data, size);
    return 0;
}
#endif
