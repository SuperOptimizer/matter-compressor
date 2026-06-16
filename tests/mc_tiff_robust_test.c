// mc_tiff_robust_test — deterministic regression guard for the TIFF reader's
// untrusted-input hardening. mc_tiff_open mmaps and parses attacker-controlled
// bytes and hands back a raw pointer into the mmap; for ANY input it must
// reject cleanly or return a view fully inside the file — never an OOB pointer,
// never a crash. Two bugs were found by tests/fuzz/mc_fuzz_tiff.c (libFuzzer
// + ASan/UBSan) and fixed in src/mc_tiff.c; this constructs each malformed file
// directly so the fix is guarded without a fuzz engine.
//
// Run under ASan/UBSan (-fno-sanitize-recover=all) so a regression faults hard:
//   clang -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
//     tests/mc_tiff_robust_test.c src/mc_tiff.c -Isrc -o t && ./t
#include "mc_tiff.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(x) do{ if(!(x)){ fails++; fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);} }while(0)

static void w16(uint8_t *p, uint16_t v){ p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); }
static void w32(uint8_t *p, uint32_t v){ p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); p[2]=(uint8_t)(v>>16); p[3]=(uint8_t)(v>>24); }

// write bytes to a temp file, mc_tiff_open it, return rc; on accept, touch every
// advertised pixel byte + assert the view is inside the mmap (catches OOB).
static int try_open(const uint8_t *bytes, size_t n) {
    char path[] = "/tmp/mc_tiff_robust_XXXXXX";
    int fd = mkstemp(path); if (fd < 0) return -99;
    FILE *f = fdopen(fd, "wb"); fwrite(bytes, 1, n, f); fclose(f);
    mc_tiff t;
    int rc = mc_tiff_open(path, &t);
    if (rc == 0) {
        const uint8_t *base=t.map, *px=t.pixels;
        CHECK(t.width > 0 && t.height > 0 && t.samples >= 1 && t.samples <= 4);
        CHECK(t.pixel_bytes == (size_t)t.width*t.height*t.samples*mc_tiff_type_size(t.type));
        CHECK(px >= base && px + t.pixel_bytes <= base + t.map_len);
        volatile uint64_t s=0; for(size_t i=0;i<t.pixel_bytes;++i) s^=px[i]; (void)s;
        mc_tiff_close(&t);
    }
    unlink(path);
    return rc;
}

// build a minimal single-IFD little-endian TIFF with the given width/height
// (LONG tags) into buf; ntag tags, pixel strip at offset 8. Returns total len.
// `ifd_off_override` (if nonzero) replaces the header's IFD offset (poison).
// `bps` is the BitsPerSample value to advertise (8 = the normal u8 case).
static size_t build_tiff_bps(uint8_t *buf, uint32_t w, uint32_t h, uint32_t ifd_off_override, int bps) {
    // 8B header + 64B pixel area + IFD.
    uint32_t pix_off = 8, pix_bytes = 64;     // generous strip
    uint32_t ifd_off = pix_off + pix_bytes;   // 72
    memset(buf, 0xAB, ifd_off);
    buf[0]='I'; buf[1]='I'; w16(buf+2,42);
    w32(buf+4, ifd_off_override ? ifd_off_override : ifd_off);
    uint8_t *e = buf + ifd_off;
    int ntag = 9;
    w16(e, (uint16_t)ntag); e += 2;
    #define T_LONG(tag,val)  do{ w16(e,(tag)); w16(e+2,4); w32(e+4,1); w32(e+8,(uint32_t)(val)); e+=12; }while(0)
    #define T_SHORT(tag,val) do{ w16(e,(tag)); w16(e+2,3); w32(e+4,1); w16(e+8,(uint16_t)(val)); w16(e+10,0); e+=12; }while(0)
    T_LONG (256, w);          // ImageWidth
    T_LONG (257, h);          // ImageLength
    T_SHORT(258, bps);        // BitsPerSample
    T_SHORT(259, 1);          // Compression = none
    T_SHORT(262, 1);          // Photometric
    T_LONG (273, pix_off);    // StripOffsets
    T_SHORT(277, 1);          // SamplesPerPixel
    T_LONG (278, h);          // RowsPerStrip
    T_SHORT(284, 1);          // PlanarConfig
    #undef T_LONG
    #undef T_SHORT
    w32(e, 0); e += 4;        // next IFD = 0
    return (size_t)(e - buf);
}
static size_t build_tiff(uint8_t *buf, uint32_t w, uint32_t h, uint32_t ifd_off_override) {
    return build_tiff_bps(buf, w, h, ifd_off_override, 8);
}

int main(void) {
    uint8_t buf[512];

    // ---- bug 1: 32-bit overflow in the IFD-offset bounds check. ifd_off near
    // UINT32_MAX made `ifd_off+2 > len` wrap to false, passing the check, then
    // rd16(b+ifd_off) read ~4GB past the mmap. Must reject (rc<0), not crash. ----
    {
        uint8_t poison[24]; memset(poison,0,sizeof poison);
        poison[0]='I'; poison[1]='I'; w16(poison+2,42); w32(poison+4, 0xFFFFFFFEu);
        int rc = try_open(poison, sizeof poison);
        CHECK(rc < 0);
        printf("bug1 (ifd_off=0xFFFFFFFE overflow): rc=%d %s\n", rc, rc<0?"rejected":"ACCEPTED!");
    }

    // ---- bug 2: zero width/height accepted -> degenerate empty-strip view a
    // consumer divides by / indexes past. Must reject. ----
    {
        size_t n = build_tiff(buf, 0, 8, 0);          // width = 0
        int rc = try_open(buf, n);
        CHECK(rc < 0);
        printf("bug2a (width=0): rc=%d %s\n", rc, rc<0?"rejected":"ACCEPTED!");

        n = build_tiff(buf, 8, 0, 0);                 // height = 0
        rc = try_open(buf, n);
        CHECK(rc < 0);
        printf("bug2b (height=0): rc=%d %s\n", rc, rc<0?"rejected":"ACCEPTED!");
    }

    // ---- sanity: a well-formed 8x8 u8 TIFF still opens (the fix must not over-
    // reject valid files). ----
    {
        size_t n = build_tiff(buf, 8, 8, 0);
        int rc = try_open(buf, n);
        CHECK(rc == 0);
        printf("valid 8x8 u8: rc=%d %s\n", rc, rc==0?"accepted":"REJECTED!");
    }

    // ---- a couple more adversarial shapes that must reject without crashing. ----
    {
        // absurd dims (would overflow int / blow past the file): reject.
        size_t n = build_tiff(buf, 0x7FFFFFFFu, 0x7FFFFFFFu, 0);
        int rc = try_open(buf, n);
        CHECK(rc < 0);
        printf("huge dims: rc=%d %s\n", rc, rc<0?"rejected":"ACCEPTED!");

        // truncated below the 8-byte header: reject.
        uint8_t tiny[4] = {'I','I',42,0};
        rc = try_open(tiny, sizeof tiny);
        CHECK(rc < 0);
        printf("4-byte file: rc=%d %s\n", rc, rc<0?"rejected":"ACCEPTED!");

        // unsupported sample width (bps=4): the reader only maps 8/16/32-bit
        // types and must reject anything else (the type-dispatch else-branch).
        n = build_tiff_bps(buf, 8, 8, 0, 4);
        rc = try_open(buf, n);
        CHECK(rc < 0);
        printf("unsupported bps=4: rc=%d %s\n", rc, rc<0?"rejected":"ACCEPTED!");
    }

    printf(fails ? "mc_tiff_robust_test: %d FAILED\n" : "mc_tiff_robust_test: OK\n", fails);
    return fails ? 1 : 0;
}
