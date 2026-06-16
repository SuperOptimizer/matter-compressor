// mc_fuzz_surface_seed — emit valid surface inputs (grid OBJ, mesh OBJ with
// faces+normals, VC per-pixel-map) as the AFL++ corpus for the surface fuzzer.
// Real header/body structure lets the mutator reach the parse paths. The known
// crash reproducers in tests/fuzz/crashes/surface/ are copied in by the runner.
//
//   usage: mc_fuzz_surface_seed <corpus_dir>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <corpus_dir>\n", argv[0]); return 2; }
    char p[1024];

    snprintf(p, sizeof p, "%s/grid.obj", argv[1]);
    FILE *f = fopen(p, "wb");
    fputs("# grid 2 2\nv 1 2 3\nv 4 5 6\nv 7 8 9\nv 10 11 12\n", f);
    fclose(f);

    snprintf(p, sizeof p, "%s/mesh.obj", argv[1]);
    f = fopen(p, "wb");
    fputs("v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nvn 0 0 1\nvn 0 0 1\nf 1//1 2//2 3//3\n", f);
    fclose(f);

    snprintf(p, sizeof p, "%s/map.ppm", argv[1]);
    f = fopen(p, "wb");
    fputs("width: 2\nheight: 2\ndim: 6\ntype: double\nversion: 1\n<>\n", f);
    double d[4*6];
    for (int i = 0; i < 4*6; ++i) d[i] = (i%6<3) ? (double)(i+1) : (i%6==5 ? 1.0 : 0.0);
    fwrite(d, sizeof(double), 4*6, f);
    fclose(f);

    printf("surface seeds written to %s\n", argv[1]);
    return 0;
}
