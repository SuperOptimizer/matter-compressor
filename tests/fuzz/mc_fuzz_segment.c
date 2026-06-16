// mc_fuzz_segment — fuzz the classical sheet detector + topology post-processing
// against adversarial volume data, dimensions, and parameters. These ops are
// pure in-memory compute over a caller-supplied (nz,ny,nx) u8 volume, but the
// internals (separable gaussian blur, central-difference gradient + structure
// tensor, closed-form eigenvalues, 26-conn flood label, 2x2x2 hole LUT,
// spherical morphology, border flood) do a lot of index arithmetic that must
// stay in bounds for ANY shape/data. Contract: no crash / OOB / UB on any input
// (enforced under ASan+UBSan).
//
// The harness derives small, bounded dims + parameters from the first bytes,
// allocates buffers sized to exactly match those dims (so we test the
// ALGORITHMS, not a deliberately-wrong alloc), fills the volume from the rest,
// then runs detect + every post-proc op + label.
//
//   AFL_USE_ASAN=1 AFL_USE_UBSAN=1 afl-clang-fast \
//     tests/fuzz/mc_fuzz_segment.c src/mc_segment.c -Isrc -O1 -g -lm \
//     -o mc_fuzz_segment
// Standalone replay (CI smoke / crash replay): -DMC_FUZZ_STANDALONE.
#include "mc_segment.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static void exercise(const uint8_t *data, size_t size) {
    if (size < 8) return;
    // header: 3 dims (each 1..24) + 4 param bytes, then volume data.
    int nz = 1 + (data[0] % 24);
    int ny = 1 + (data[1] % 24);
    int nx = 1 + (data[2] % 24);
    mc_seg_params P;
    P.sigma_grad   = (float)(data[3] % 8) * 0.5f;     // 0..3.5 (0 -> default)
    P.sigma_tensor = (float)(data[4] % 8) * 0.5f;
    P.sheetness    = (float)(data[5]) / 255.0f;       // 0..1
    P.val_lo       = data[6];                         // 0..255
    int radius     = data[7] % 4;                     // morphology radius 0..3

    size_t n = (size_t)nz * ny * nx;
    const uint8_t *vol_src = data + 8;
    size_t avail = size - 8;

    uint8_t *vol  = malloc(n);
    uint8_t *mask = malloc(n);
    int32_t *lab  = malloc(n * sizeof(int32_t));
    if (!vol || !mask || !lab) { free(vol); free(mask); free(lab); return; }
    // fill the volume from the input (repeat/pad if short).
    for (size_t i = 0; i < n; ++i) vol[i] = avail ? vol_src[i % avail] : 0;

    // detector -> mask, then the full topology post-processing chain.
    if (mc_seg_detect(vol, nz, ny, nx, &P, mask) == 0) {
        mc_seg_remove_small(mask, nz, ny, nx, (int)(data[0] * 4));   // varied min
        mc_seg_plug_holes(mask, nz, ny, nx, 1 + (data[1] % 3));
        mc_seg_close(mask, nz, ny, nx, radius);
        mc_seg_fill_cavities(mask, nz, ny, nx);
        mc_seg_label(mask, nz, ny, nx, lab);
        // touch the whole label buffer so ASan flags any OOB write inside label.
        volatile long s = 0; for (size_t i = 0; i < n; ++i) s += lab[i]; (void)s;
    }
    free(vol); free(mask); free(lab);
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
    printf("mc_fuzz_segment: replayed %d input(s), no crash\n", argc - 1);
    return 0;
}
#else
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    exercise(data, size);
    return 0;
}
#endif
