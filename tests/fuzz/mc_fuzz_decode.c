// mc_fuzz_decode — fuzz the HARDENED decode path against arbitrary/corrupted
// archive bytes. README promises decode is robust to corrupted payloads
// (clamped writes, bounded offsets); this is the executable form of that claim.
//
// Feeds fuzzer bytes through the read-only consumer surface:
//   mc_open -> mc_metadata / mc_reader_dims / mc_reader_nlods
//           -> mc_chunk_offset + mc_decode_block (every LOD, a sweep of blocks)
//           -> mc_verify_archive
// No crash / OOB / UAF / UB is allowed on ANY input — that is the pass bar
// (enforced by ASan+UBSan). Wrong decoded VALUES are fine: the bytes are junk.
//
// The entrypoint is the standard libFuzzer API (LLVMFuzzerTestOneInput) so the
// harness is engine-agnostic. We drive the campaign with **AFL++**, which
// compiles a libFuzzer-style harness directly (afl-clang-fast links its own
// persistent-mode driver around LLVMFuzzerTestOneInput):
//
//   # instrument + ASan/UBSan, then run AFL++ (see scripts/fuzz.sh)
//   AFL_USE_ASAN=1 AFL_USE_UBSAN=1 afl-clang-fast \
//     tests/fuzz/mc_fuzz_decode.c src/matter_compressor.c src/c3d.c \
//     tools/vendor/libs3/libs3.c -Isrc -Itools/vendor/libs3 -O1 -g \
//     -lm -lpthread -lzstd -lcurl -o mc_fuzz_decode
//   afl-fuzz -i corpus -o findings -- ./mc_fuzz_decode @@
//
// Build (no fuzz engine — CI smoke / crash replay): add -DMC_FUZZ_STANDALONE;
// the main() below replays each path argument (or stdin) once. The CI smoke job
// uses this build to replay AFL++ corpus/crash inputs under plain ASan.
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static void exercise(const uint8_t *data, size_t size) {
    if (size < 16) return;                 // too short to be anything; skip cheaply
    // mc_open may mmap/scan the header; it must reject garbage without crashing.
    mc_reader *r = mc_open(data, size);
    if (!r) return;                        // cleanly rejected — good

    // mc_metadata reads fixed header fields; its precondition is a header-sized
    // buffer. mc_open already gated len >= MC_HDR above (it returned non-NULL),
    // so this is safe here.
    size_t ml = 0;
    const char *m = mc_metadata(data, &ml);
    if (m && ml) { volatile char sink = m[0]; (void)sink; (void)m[ml - 1]; }

    int nx = 0, ny = 0, nz = 0;
    mc_reader_dims(r, &nx, &ny, &nz);
    int nlods = mc_reader_nlods(r);
    if (nlods < 0) nlods = 0;
    if (nlods > 8) nlods = 8;              // clamp the loop; header field is attacker-controlled

    mc_u8 blk[16 * 16 * 16];
    for (int lod = 0; lod < nlods; ++lod) {
        // Probe a spread of chunk coords — a corrupt node table must not walk OOB.
        for (int c = 0; c < 8; ++c) {
            int cz = (c & 1) * 3, cy = ((c >> 1) & 1) * 5, cx = ((c >> 2) & 1) * 7;
            uint64_t co = mc_chunk_offset(r, lod, cz, cy, cx);
            if (!co) continue;
            // Sweep block coords incl. the corners; bounded-offset decode must hold.
            for (int b = 0; b < 8; ++b) {
                int bz = (b & 1) * 15, by = ((b >> 1) & 1) * 15, bx = ((b >> 2) & 1) * 15;
                mc_decode_block(r, co, bz, by, bx, blk);
            }
        }
    }

    mc_close(r);
    // Integrity walk over the raw bytes (xxh64 per chunk) — its own bounds path.
    (void)mc_verify_archive(data, size, 0);
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
    printf("mc_fuzz_decode: replayed %d input(s), no crash\n", argc - 1);
    return 0;
}
#else
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    exercise(data, size);
    return 0;
}
#endif
