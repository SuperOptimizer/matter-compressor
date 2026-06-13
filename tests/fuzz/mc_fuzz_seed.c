// mc_fuzz_seed — emit valid .mca archives as AFL++ corpus seeds. A good seed
// corpus gives the fuzzer real header/node-table/blob structure to mutate, so
// it reaches the decode paths instead of bouncing off the magic check.
//
//   usage: mc_fuzz_seed <corpus_dir>
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_dim;
static mc_u8 ball(void *ud, int x, int y, int z) { (void)ud;
    double c = g_dim / 2.0, r = sqrt((x-c)*(x-c)+(y-c)*(y-c)+(z-c)*(z-c));
    if (r > g_dim * 0.4) return 0;
    double v = 128 + 100 * cos(r * 0.15); return (mc_u8)(v < 1 ? 1 : v > 255 ? 255 : v);
}

static void emit(const char *dir, const char *name, int dim, float q) {
    g_dim = dim;
    char path[1024]; snprintf(path, sizeof path, "%s/%s", dir, name);
    mc_build_opts o = { .dim = dim, .quality = q,
                        .metadata = "{\"seed\":1}", .meta_len = 10 };
    if (mc_build_to_file(ball, NULL, &o, path) != 0)
        fprintf(stderr, "seed %s failed\n", path);
    else
        printf("seed %s\n", path);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <corpus_dir>\n", argv[0]); return 2; }
    // A few small archives at different quality/dims = varied blob structure.
    emit(argv[1], "seed_q1.mca",  256, 1.0f);
    emit(argv[1], "seed_q6.mca",  256, 6.0f);
    emit(argv[1], "seed_q32.mca", 256, 32.0f);
    return 0;
}
