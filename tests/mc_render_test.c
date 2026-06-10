// mc_render/mc_sample tests: filters, plane + quad surfaces, compositing,
// parallel==serial, cache-source parity with dense sampling.
#include "../src/mc_sample.h"
#include "../src/mc_render.h"
#include "../src/matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int fails = 0;
#define CHECK(x) do { if (!(x)) { fails++; \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); } } while (0)

#define N 256
static uint8_t *vol;
static inline uint8_t V(int z, int y, int x) {
    return (uint8_t)((z + 2 * y + 3 * x) & 0xFF);
}

int main(void) {
    vol = malloc((size_t)N * N * N);
    for (int z = 0; z < N; z++)
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++)
                vol[((size_t)z * N + y) * N + x] = V(z, y, x);

    mc_sample_src dsrc = mc_sample_src_dense(vol, N, N, N);
    mc_sampler *s = mc_sampler_new(&dsrc);
    CHECK(s != NULL);

    // ---- filters --------------------------------------------------------
    CHECK(mc_sample_point(s, 10, 20, 30, MC_FILTER_NEAREST) == (float)V(10, 20, 30));
    CHECK(mc_sample_point(s, 10.4f, 20.4f, 30.4f, MC_FILTER_NEAREST) == (float)V(10, 20, 30));
    CHECK(mc_sample_point(s, 10.6f, 20.6f, 30.6f, MC_FILTER_NEAREST) == (float)V(11, 21, 31));
    CHECK(mc_sample_point(s, 10, 20, 30, MC_FILTER_TRILINEAR) == (float)V(10, 20, 30));
    // midpoint in z: average of the two z neighbors
    float tz = mc_sample_point(s, 10.5f, 20, 30, MC_FILTER_TRILINEAR);
    CHECK(fabsf(tz - 0.5f * (V(10, 20, 30) + V(11, 20, 30))) < 1e-3f);
    // block-boundary crossing (x = 15.5 straddles two 16^3 blocks)
    float tx = mc_sample_point(s, 8, 8, 15.5f, MC_FILTER_TRILINEAR);
    CHECK(fabsf(tx - 0.5f * (V(8, 8, 15) + V(8, 8, 16))) < 1e-3f);
    // out of bounds / NaN -> 0
    CHECK(mc_sample_point(s, -5, 0, 0, MC_FILTER_NEAREST) == 0.0f);
    CHECK(mc_sample_point(s, 300, 0, 0, MC_FILTER_TRILINEAR) == 0.0f);
    CHECK(mc_sample_point(s, NAN, 0, 0, MC_FILTER_TRILINEAR) == 0.0f);
    printf("filters OK\n");

    // ---- plane slice == array slice -------------------------------------
    enum { W = 64, H = 48 };
    uint8_t img[W * H], ref[W * H];
    mc_plane pl = { .origin = {100, 24, 32}, .normal = {1, 0, 0},
                    .u = {0, 0, 1}, .v = {0, 1, 0} };
    mc_render_params prm = { .filter = MC_FILTER_NEAREST, .comp = MC_COMP_NONE };
    CHECK(mc_render_plane(&dsrc, &pl, W, H, 1.0f, &prm, img, 1) == 0);
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) {
            int y = 24 + (int)floorf((float)i - H * 0.5f + 0.5f);
            int x = 32 + (int)floorf((float)j - W * 0.5f + 0.5f);
            ref[i * W + j] = V(100, y, x);
        }
    CHECK(memcmp(img, ref, sizeof img) == 0);
    printf("plane slice OK\n");

    // ---- compositing along +z -------------------------------------------
    // V is monotonically increasing in z, so over t in [-3,3]:
    // MAX = V(z+3), MIN = V(z-3), MEAN = V(z) (arithmetic progression),
    // ...except where (z+2y+3x) wraps mod 256 — restrict to a wrap-free
    // window (z in 97..103 and value <= 255-? choose offsets small).
    mc_plane pc = { .origin = {100, 10, 12}, .normal = {1, 0, 0},
                    .u = {0, 0, 1}, .v = {0, 1, 0} };
    enum { CW = 8, CH = 8 };
    uint8_t cimg[CW * CH];
    mc_render_params pp = { .filter = MC_FILTER_NEAREST, .t0 = -3, .t1 = 3, .dt = 1 };
    int wrap_free = 1;
    for (int i = 0; i < CH && wrap_free; i++)
        for (int j = 0; j < CW; j++) {
            int y = 10 + i - CH / 2, x = 12 + j - CW / 2;
            if (97 + 2 * y + 3 * x > 252) { wrap_free = 0; break; }
        }
    CHECK(wrap_free);
    pp.comp = MC_COMP_MAX;
    CHECK(mc_render_plane(&dsrc, &pc, CW, CH, 1.0f, &pp, cimg, 1) == 0);
    for (int i = 0; i < CH; i++)
        for (int j = 0; j < CW; j++) {
            int y = 10 + (int)floorf((float)i - CH * 0.5f + 0.5f);
            int x = 12 + (int)floorf((float)j - CW * 0.5f + 0.5f);
            CHECK(cimg[i * CW + j] == V(103, y, x));
        }
    pp.comp = MC_COMP_MIN;
    CHECK(mc_render_plane(&dsrc, &pc, CW, CH, 1.0f, &pp, cimg, 1) == 0);
    {
        int y = 10 - CH / 2, x = 12 - CW / 2;
        CHECK(cimg[0] == V(97, y, x));
    }
    pp.comp = MC_COMP_MEAN;
    CHECK(mc_render_plane(&dsrc, &pc, CW, CH, 1.0f, &pp, cimg, 1) == 0);
    {
        int y = 10 - CH / 2, x = 12 - CW / 2;
        CHECK(cimg[0] == V(100, y, x));     // mean of arithmetic progression
    }
    // ALPHA: replicate the documented formula exactly
    pp.comp = MC_COMP_ALPHA;
    pp.alpha_min = 0.3f; pp.alpha_opacity = 0.5f;
    CHECK(mc_render_plane(&dsrc, &pc, CW, CH, 1.0f, &pp, cimg, 1) == 0);
    for (int i = 0; i < CH; i++)
        for (int j = 0; j < CW; j++) {
            int y = 10 + (int)floorf((float)i - CH * 0.5f + 0.5f);
            int x = 12 + (int)floorf((float)j - CW * 0.5f + 0.5f);
            float acc = 0, A = 0;
            for (int t = -3; t <= 3; t++) {
                float v = V(100 + t, y, x);
                float a = 0.5f * (v / 255.0f - 0.3f) / (1.0f - 0.3f);
                if (a > 0) {
                    if (a > 1) a = 1;
                    acc += (1 - A) * a * v;
                    A += (1 - A) * a;
                    if (A >= 0.98f) break;
                }
            }
            uint8_t e = (uint8_t)(acc < 0 ? 0 : acc > 255 ? 255 : (int)(acc + 0.5f));
            CHECK(cimg[i * CW + j] == e);
        }
    printf("compositing OK\n");

    // ---- quad surface ----------------------------------------------------
    // flat control grid at z=77: point (gy,gx) = (77, gy, gx). step=1 render
    // == the z=77 slice; normals = cross(du,dv) = (-1,0,0).
    enum { GW = 64, GH = 64 };
    float *grid = malloc((size_t)GW * GH * 3 * sizeof(float));
    for (int gy = 0; gy < GH; gy++)
        for (int gx = 0; gx < GW; gx++) {
            float *p = grid + ((size_t)gy * GW + gx) * 3;
            p[0] = 77; p[1] = (float)gy; p[2] = (float)gx;
        }
    mc_quad q = { .grid = grid, .gw = GW, .gh = GH };
    uint8_t qimg[32 * 32], qref[32 * 32];
    mc_render_params qp = { .filter = MC_FILTER_NEAREST, .comp = MC_COMP_NONE };
    CHECK(mc_render_quad(&dsrc, &q, 8, 8, 1.0f, 32, 32, &qp, qimg, 1) == 0);
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++) qref[i * 32 + j] = V(77, 8 + i, 8 + j);
    CHECK(memcmp(qimg, qref, sizeof qimg) == 0);
    // sub-cell interpolation: step 0.5 between integer grid points
    float pt[3], nm[3];
    mc_quad_gen(&q, 10.5f, 20.5f, 1.0f, 1, 1, pt, nm);
    CHECK(pt[0] == 77.0f && pt[1] == 20.5f && pt[2] == 10.5f);
    CHECK(fabsf(nm[0] - (-1.0f)) < 1e-5f && fabsf(nm[1]) < 1e-5f && fabsf(nm[2]) < 1e-5f);
    // MAX composite through the quad's own normals: N=(-1,0,0), t in [0,3]
    // walks z 77 -> 74; V increasing in z => max at t=0 (z=77). Use a
    // wrap-free window (77+2y+3x stays under 256).
    qp.comp = MC_COMP_MAX; qp.t0 = 0; qp.t1 = 3; qp.dt = 1;
    uint8_t mimg[16 * 16];
    CHECK(mc_render_quad(&dsrc, &q, 2, 2, 1.0f, 16, 16, &qp, mimg, 1) == 0);
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++)
            CHECK(mimg[i * 16 + j] == V(77, 2 + i, 2 + j));
    // invalid control point poisons its cells
    grid[((size_t)10 * GW + 10) * 3 + 0] = -1;
    grid[((size_t)10 * GW + 10) * 3 + 1] = -1;
    grid[((size_t)10 * GW + 10) * 3 + 2] = -1;
    qp.comp = MC_COMP_NONE;
    CHECK(mc_render_quad(&dsrc, &q, 8, 8, 1.0f, 32, 32, &qp, qimg, 1) == 0);
    CHECK(qimg[2 * 32 + 2] == 0);           // pixel (10,10) sits on the bad cell
    CHECK(qimg[20 * 32 + 20] == qref[20 * 32 + 20]);
    printf("quad surface OK\n");

    // ---- parallel == serial ----------------------------------------------
    enum { PW = 128, PH = 96 };
    uint8_t *pa = malloc(PW * PH), *pb = malloc(PW * PH);
    mc_plane pbig = { .origin = {128, 128, 128}, .normal = {0.577f, 0.577f, 0.577f} };
    mc_plane_basis(&pbig);
    mc_render_params pp2 = { .filter = MC_FILTER_TRILINEAR, .comp = MC_COMP_MAX,
                             .t0 = -4, .t1 = 4, .dt = 0.5f };
    CHECK(mc_render_plane(&dsrc, &pbig, PW, PH, 1.0f, &pp2, pa, 1) == 0);
    CHECK(mc_render_plane(&dsrc, &pbig, PW, PH, 1.0f, &pp2, pb, 0) == 0);
    CHECK(memcmp(pa, pb, PW * PH) == 0);
    printf("parallel==serial OK\n");

    // ---- cache source parity ---------------------------------------------
    // encode the volume into an archive, decode it back to dense, and check
    // cache-backed sampling == dense sampling of the decoded data.
    const char *path = "/tmp/mc_render_test.mca";
    remove(path);
    mc_archive *a = mc_archive_open_dims(path, N, N, N, 2.0f);
    CHECK(a != NULL);
    CHECK(mc_archive_append_chunk_par(a, 0, 0, 0, 0, vol, 2.0f, 0) == 0);
    uint8_t *dec = malloc((size_t)N * N * N);
    uint64_t co = mc_archive_chunk_offset(a, 0, 0, 0, 0);
    mc_archive_decode_chunk(a, co, dec, 0);
    mc_cache *cache = mc_cache_new_archive(64ull << 20, a);
    CHECK(cache != NULL);
    mc_sample_src csrc = mc_sample_src_cache(cache, 0, N, N, N);
    mc_sample_src d2 = mc_sample_src_dense(dec, N, N, N);
    uint8_t *ca = malloc(PW * PH), *cb = malloc(PW * PH);
    CHECK(mc_render_plane(&csrc, &pbig, PW, PH, 1.0f, &pp2, ca, 0) == 0);
    CHECK(mc_render_plane(&d2, &pbig, PW, PH, 1.0f, &pp2, cb, 0) == 0);
    CHECK(memcmp(ca, cb, PW * PH) == 0);
    printf("cache parity OK\n");
    mc_cache_free(cache);
    mc_archive_close(a);
    remove(path);

    // ---- LOD-matched rendering --------------------------------------------
    // lod1 = 2x mean-pool of vol. At scale >= 2 the _lod renderer must (a)
    // pick lod1 and (b) produce exactly what rendering lod1 directly with
    // remapped geometry produces.
    {
        enum { M = N / 2 };
        uint8_t *l1 = malloc((size_t)M * M * M);
        for (int z = 0; z < M; z++)
            for (int y = 0; y < M; y++)
                for (int x = 0; x < M; x++) {
                    int sum = 0;
                    for (int dz2 = 0; dz2 < 2; dz2++)
                        for (int dy2 = 0; dy2 < 2; dy2++)
                            for (int dx2 = 0; dx2 < 2; dx2++)
                                sum += vol[((size_t)(2 * z + dz2) * N +
                                            (2 * y + dy2)) * N + (2 * x + dx2)];
                    l1[((size_t)z * M + y) * M + x] = (uint8_t)((sum + 4) / 8);
                }
        mc_sample_lods ls = {0};
        ls.nlods = 2;
        ls.lods[0] = mc_sample_src_dense(vol, N, N, N);
        ls.lods[1] = mc_sample_src_dense(l1, M, M, M);

        CHECK(mc_render_pick_lod(&ls, 1.0f) == 0);
        CHECK(mc_render_pick_lod(&ls, 1.9f) == 0);
        CHECK(mc_render_pick_lod(&ls, 2.0f) == 1);
        CHECK(mc_render_pick_lod(&ls, 64.0f) == 1);   // clamped to nlods-1

        enum { LW = 64, LH = 64 };
        uint8_t la[LW * LH], lb[LW * LH];
        mc_plane lp = { .origin = {128, 128, 128}, .normal = {1, 0, 0},
                        .u = {0, 0, 1}, .v = {0, 1, 0} };
        mc_render_params lprm = { .filter = MC_FILTER_TRILINEAR,
                                  .comp = MC_COMP_MAX, .t0 = -4, .t1 = 4, .dt = 1 };
        CHECK(mc_render_plane_lod(&ls, &lp, LW, LH, 2.0f, &lprm, la, 1) == 0);
        // reference: render lod1 directly with remapped geometry
        // c1 = (c0 + 0.5)/2 - 0.5; scale halves; t-range halves
        mc_plane rp = lp;
        for (int k = 0; k < 3; k++)
            rp.origin[k] = (lp.origin[k] + 0.5f) * 0.5f - 0.5f;
        mc_render_params rprm = lprm;
        rprm.t0 = -2; rprm.t1 = 2;
        CHECK(mc_render_plane(&ls.lods[1], &rp, LW, LH, 1.0f, &rprm, lb, 1) == 0);
        CHECK(memcmp(la, lb, sizeof la) == 0);
        // scale 1 -> identical to the plain lod0 render
        CHECK(mc_render_plane_lod(&ls, &lp, LW, LH, 1.0f, &lprm, la, 1) == 0);
        CHECK(mc_render_plane(&ls.lods[0], &lp, LW, LH, 1.0f, &lprm, lb, 1) == 0);
        CHECK(memcmp(la, lb, sizeof la) == 0);
        // quad spacing estimate: flat unit-spaced grid -> ~1.0
        float qs = mc_quad_spacing(&q);
        CHECK(qs > 0.99f && qs < 1.01f);
        free(l1);
    }
    printf("lod rendering OK\n");

    // ---- 3D resampling ------------------------------------------------------
    {
        // restore the control point poisoned earlier
        grid[((size_t)10 * GW + 10) * 3 + 0] = 77;
        grid[((size_t)10 * GW + 10) * 3 + 1] = 10;
        grid[((size_t)10 * GW + 10) * 3 + 2] = 10;
        // flat quad at z=77, normals (-1,0,0): layer k samples z = 77 - k
        enum { QW = 8, QH = 8, QL = 8 };
        uint8_t qv[QW * QH * QL];
        CHECK(mc_sample_quad_volume(&dsrc, &q, 4, 4, 1.0f, QW, QH,
                                    0.0f, 1.0f, QL, MC_FILTER_NEAREST,
                                    qv, 1) == 0);
        for (int k = 0; k < QL; k++)
            for (int i = 0; i < QH; i++)
                for (int j = 0; j < QW; j++)
                    CHECK(qv[k * QW * QH + i * QW + j] ==
                          V(77 - k, 4 + i, 4 + j));
        // trilinear layers (exercises the 4-wide chunk path) == per-point
        // scalar sampling at the same positions
        CHECK(mc_sample_quad_volume(&dsrc, &q, 4, 4, 1.0f, QW, QH,
                                    0.5f, 1.0f, QL, MC_FILTER_TRILINEAR,
                                    qv, 1) == 0);
        for (int k = 0; k < QL; k++)
            for (int i = 0; i < QH; i += 3)
                for (int j = 0; j < QW; j += 3) {
                    float z = 77.0f - (0.5f + (float)k);
                    float ref = mc_sample_point(s, z, (float)(4 + i),
                                                (float)(4 + j),
                                                MC_FILTER_TRILINEAR);
                    uint8_t e = (uint8_t)(ref + 0.5f);
                    CHECK(qv[k * QW * QH + i * QW + j] == e);
                }
        // parallel == serial
        uint8_t qv2[QW * QH * QL];
        CHECK(mc_sample_quad_volume(&dsrc, &q, 4, 4, 1.0f, QW, QH,
                                    0.5f, 1.0f, QL, MC_FILTER_TRILINEAR,
                                    qv2, 0) == 0);
        CHECK(memcmp(qv, qv2, sizeof qv) == 0);

        // oriented box with unit axes == direct copy
        enum { BW = 8, BH = 6, BD = 4 };
        uint8_t bv[BW * BH * BD];
        float org[3] = {30, 20, 10};
        float bu[3] = {0, 0, 1}, bvx[3] = {0, 1, 0}, bw[3] = {1, 0, 0};
        CHECK(mc_sample_box(&dsrc, org, bu, bvx, bw, BW, BH, BD,
                            MC_FILTER_NEAREST, bv, 1) == 0);
        for (int k = 0; k < BD; k++)
            for (int i = 0; i < BH; i++)
                for (int j = 0; j < BW; j++)
                    CHECK(bv[k * BW * BH + i * BW + j] ==
                          V(30 + k, 20 + i, 10 + j));
        // swapped axes = transposed copy
        float su[3] = {0, 1, 0}, sv[3] = {0, 0, 1};
        CHECK(mc_sample_box(&dsrc, org, su, sv, bw, BW, BH, BD,
                            MC_FILTER_NEAREST, bv, 1) == 0);
        for (int k = 0; k < BD; k++)
            for (int i = 0; i < BH; i++)
                for (int j = 0; j < BW; j++)
                    CHECK(bv[k * BW * BH + i * BW + j] ==
                          V(30 + k, 20 + j, 10 + i));
    }
    printf("3d resampling OK\n");

    mc_sampler_free(s);
    free(grid); free(pa); free(pb); free(ca); free(cb); free(dec); free(vol);
    printf(fails ? "mc_render_test: %d FAILED\n" : "mc_render_test: OK\n", fails);
    return fails ? 1 : 0;
}
