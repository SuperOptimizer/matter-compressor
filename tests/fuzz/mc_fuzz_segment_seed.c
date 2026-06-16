// mc_fuzz_segment_seed — emit valid segment-detector inputs as the AFL++ corpus.
// Input layout (see mc_fuzz_segment.c): 8 header bytes
//   [nz%24+1][ny%24+1][nx%24+1][sigma_grad][sigma_tensor][sheetness][val_lo][radius]
// followed by nz*ny*nx volume bytes. Seeds vary dims (incl. degenerate
// 1-thick axes) and embed a mid-plane sheet so detect() does real work.
//
//   usage: mc_fuzz_segment_seed <corpus_dir>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void emit(const char *dir, int idx, int nz, int ny, int nx) {
    char path[1024]; snprintf(path, sizeof path, "%s/seed%d.bin", dir, idx);
    FILE *f = fopen(path, "wb"); if (!f) return;
    unsigned char hdr[8] = { (unsigned char)nz, (unsigned char)ny, (unsigned char)nx,
                             2, 4, 128, 80, 1 };   // sigma 1.0/2.0, sheet 0.5, lo 80, r1
    fwrite(hdr, 1, 8, f);
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                unsigned char v = (z == nz/2 || z == nz/2 - 1) ? 220 : 40;  // a sheet
                fwrite(&v, 1, 1, f);
            }
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <corpus_dir>\n", argv[0]); return 2; }
    int dims[][3] = {{8,8,8},{1,10,10},{10,1,10},{10,10,1},{2,2,2},{16,16,4},{24,24,24},{1,1,1}};
    for (int i = 0; i < (int)(sizeof dims/sizeof dims[0]); ++i)
        emit(argv[1], i, dims[i][0], dims[i][1], dims[i][2]);
    printf("segment seeds written to %s\n", argv[1]);
    return 0;
}
