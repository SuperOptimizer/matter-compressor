// mc_fuzz_tiff_seed — emit valid TIFFs (every supported sample type / count) as
// the AFL++ corpus for the TIFF reader fuzzer. Real header+IFD+strip structure
// gives the mutator something to corrupt past the magic check. The known
// crash reproducers in tests/fuzz/crashes/tiff/ should also be copied into the
// corpus by the runner so regressions stay covered.
//
//   usage: mc_fuzz_tiff_seed <corpus_dir>
#include "mc_tiff.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <corpus_dir>\n", argv[0]); return 2; }
    unsigned char buf[16*16*4*4];
    memset(buf, 0xAB, sizeof buf);
    char p[1024];
    struct { const char *name; int w,h,s; mc_tiff_type t; } seeds[] = {
        {"u8_1",  8,8,1, MC_TIFF_U8},  {"u16_1", 8,8,1, MC_TIFF_U16},
        {"i16_1", 8,8,1, MC_TIFF_I16}, {"u32_1", 8,8,1, MC_TIFF_U32},
        {"f32_1", 8,8,1, MC_TIFF_F32}, {"u8_3",  4,4,3, MC_TIFF_U8},
        {"f32_4", 4,4,4, MC_TIFF_F32}, {"i16_2", 4,4,2, MC_TIFF_I16},
    };
    for (size_t i = 0; i < sizeof seeds/sizeof seeds[0]; ++i) {
        snprintf(p, sizeof p, "%s/seed_%s.tif", argv[1], seeds[i].name);
        if (mc_tiff_write(p, seeds[i].w, seeds[i].h, seeds[i].s, seeds[i].t, buf) != 0)
            fprintf(stderr, "seed %s failed\n", p);
        else printf("seed %s\n", p);
    }
    return 0;
}
