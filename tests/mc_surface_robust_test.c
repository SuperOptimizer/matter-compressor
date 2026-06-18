// mc_surface_robust_test — deterministic regression guard for the surface
// loaders' untrusted-input hardening. mc_surface_load_obj / mc_mesh_load_obj /
// mc_surface_load_vcps_ppm parse external Vesuvius segment files and allocate
// from attacker-controlled header fields. Four bugs were found by
// tests/fuzz/mc_fuzz_surface.c (libFuzzer + ASan/UBSan) and fixed in
// src/mc_surface.c; this constructs each malformed input directly so the fixes
// are guarded without a fuzz engine.
//
// Run under ASan/UBSan with leak detection so a regression faults hard:
//   clang -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
//     tests/mc_surface_robust_test.c src/mc_surface.c tiff/tiff.c -Isrc -Itiff -lm \
//     -o t && ASAN_OPTIONS=detect_leaks=1 ./t
#include "mc_surface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(x) do{ if(!(x)){ fails++; fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);} }while(0)

static void write_file(const char *path, const void *bytes, size_t n) {
    FILE *f = fopen(path, "wb"); fwrite(bytes, 1, n, f); fclose(f);
}
static void write_text(const char *path, const char *s) { write_file(path, s, strlen(s)); }

int main(void) {
    const char *obj = "/tmp/mc_surf_robust.obj";
    const char *ppm = "/tmp/mc_surf_robust.ppm";

    // ---- bug 1: mc_mesh_load_obj leaked v/vn/tri on the no-vertex (-2) path.
    // A file with no 'v ' lines must free its buffers, not leak them. (Leak
    // detection catches a regression.) ----
    {
        write_text(obj, "# just a comment, no vertices\nfoo bar\n");
        mc_mesh m;
        int rc = mc_mesh_load_obj(obj, &m);
        CHECK(rc != 0);                       // rejected (no vertices)
        if (rc == 0) mc_mesh_free(&m);
        printf("bug1 (no-vertex mesh frees buffers): rc=%d\n", rc);
    }

    // ---- bug 2: unvalidated face indices stored OOB into tri. A face that
    // references vertices past nv (or <=0) must not produce a triangle index
    // outside [0,nv). ----
    {
        write_text(obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                        "f 1 2 999999\n"      // 999999 is way past nv=3
                        "f 1 -100 2\n"        // negative-relative out of range
                        "f 0 1 2\n"           // index 0 is invalid (1-based)
                        "f 1 2 3\n");         // the one valid triangle
        mc_mesh m;
        int rc = mc_mesh_load_obj(obj, &m);
        CHECK(rc == 0);
        if (rc == 0) {
            for (int i = 0; i < m.nt; ++i)
                for (int k = 0; k < 3; ++k)
                    CHECK(m.tri[i*3+k] >= 0 && m.tri[i*3+k] < m.nv);   // all in range
            printf("bug2 (face index validation): nv=%d nt=%d (all tri in [0,nv))\n", m.nv, m.nt);
            mc_mesh_free(&m);
        }
    }

    // ---- bug 3: dimension overflow in mc_surface_load_obj — a huge "# grid W H"
    // made W*H*3*sizeof(float) overflow into an under-alloc the fill loop wrote
    // past. Must reject absurd dims, not crash. ----
    {
        write_text(obj, "# grid 2000000000 2000000000\nv 1 2 3\n");
        mc_surface s;
        int rc = mc_surface_load_obj(obj, &s);
        CHECK(rc != 0);                       // rejected
        if (rc == 0) mc_surface_free(&s);
        printf("bug3 (obj grid dim overflow): rc=%d\n", rc);

        // a sane grid still loads.
        write_text(obj, "# grid 2 1\nv 1 2 3\nv 4 5 6\n");
        rc = mc_surface_load_obj(obj, &s);
        CHECK(rc == 0);
        if (rc == 0) { CHECK(s.gw==2 && s.gh==1); mc_surface_free(&s); }
        printf("valid obj grid 2x1: rc=%d\n", rc);
    }

    // ---- bug 4: dimension overflow in mc_surface_load_vcps_ppm — huge
    // width/height in the header. Must reject. ----
    {
        write_text(ppm, "width: 2000000000\nheight: 2000000000\ndim: 6\ntype: double\n<>\n");
        mc_surface s;
        int rc = mc_surface_load_vcps_ppm(ppm, &s, 5.0f);
        CHECK(rc != 0);                       // rejected before the giant alloc
        if (rc == 0) mc_surface_free(&s);
        printf("bug4 (vcps dim overflow): rc=%d\n", rc);

        // a valid 2x2x6 map still loads (header + 4*6 doubles). Build the body
        // in a properly-aligned double[] and append it after the text header.
        double d[4*6];
        for (int i = 0; i < 4*6; ++i) d[i] = (i%6<3) ? (double)(i+1) : 0.0;
        FILE *pf = fopen(ppm, "wb");
        fputs("width: 2\nheight: 2\ndim: 6\ntype: double\n<>\n", pf);
        fwrite(d, sizeof(double), 4*6, pf);
        fclose(pf);
        rc = mc_surface_load_vcps_ppm(ppm, &s, 5.0f);
        CHECK(rc == 0);
        if (rc == 0) { CHECK(s.gw==2 && s.gh==2); mc_surface_free(&s); }
        printf("valid vcps 2x2: rc=%d\n", rc);
    }

    unlink(obj); unlink(ppm);
    printf(fails ? "mc_surface_robust_test: %d FAILED\n" : "mc_surface_robust_test: OK\n", fails);
    return fails ? 1 : 0;
}
