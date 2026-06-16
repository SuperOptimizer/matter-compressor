// mc_fuzz_tiff — fuzz the TIFF reader against arbitrary/corrupted file bytes.
//
// mc_tiff_open parses an UNTRUSTED file: it mmaps the bytes, walks the IFD,
// validates (compression/planar/samples/bps/sampleformat/strip-fits), and on
// success exposes out->pixels as a raw pointer INTO the mmap that callers cast
// to typed structs. The hard contract: for ANY input bytes, open either
// rejects (returns <0) or returns a view whose [pixels, pixels+pixel_bytes)
// lies fully inside the mmap — never an OOB pointer, never a crash. This is the
// executable form of that contract (enforced under ASan+UBSan): on accept we
// touch every advertised pixel byte, so any over-long/overflowed extent that
// slipped the bounds checks faults here.
//
// mc_tiff_open takes a PATH (it mmaps), so the harness materializes the fuzz
// bytes to a temp file per input. AFL++ persistent mode still drives it fast.
//
//   AFL_USE_ASAN=1 AFL_USE_UBSAN=1 afl-clang-fast \
//     tests/fuzz/mc_fuzz_tiff.c src/mc_tiff.c -Isrc -O1 -g -o mc_fuzz_tiff
//   afl-fuzz -i corpus -o findings -- ./mc_fuzz_tiff @@
//
// Standalone replay (CI smoke / crash replay): -DMC_FUZZ_STANDALONE.
#include "mc_tiff.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

static void exercise(const uint8_t *data, size_t size) {
    if (size < 8 || size > (1u<<24)) return;     // header floor; cap mmap size

    // materialize to a temp file (mc_tiff_open mmaps a path).
    char path[] = "/tmp/mc_fuzz_tiff_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return;
    if (write(fd, data, size) != (ssize_t)size) { close(fd); unlink(path); return; }
    close(fd);

    mc_tiff t;
    if (mc_tiff_open(path, &t) == 0) {
        // accepted -> the advertised extent MUST be inside the mmap. Recompute
        // it independently and assert containment, then touch every byte so ASan
        // catches any pointer that escaped the file mapping.
        size_t want = (size_t)t.width * t.height * t.samples * mc_tiff_type_size(t.type);
        const uint8_t *base = (const uint8_t *)t.map;
        const uint8_t *px   = (const uint8_t *)t.pixels;
        if (t.width <= 0 || t.height <= 0 || t.samples < 1 || t.samples > 4) abort();
        if (want != t.pixel_bytes) abort();                 // size accounting drift
        if (px < base || px + t.pixel_bytes > base + t.map_len) abort();  // OOB view

        volatile uint64_t sink = 0;
        for (size_t i = 0; i < t.pixel_bytes; ++i) sink ^= px[i];
        (void)sink;
        mc_tiff_close(&t);
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
    printf("mc_fuzz_tiff: replayed %d input(s), no crash\n", argc - 1);
    return 0;
}
#else
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    exercise(data, size);
    return 0;
}
#endif
