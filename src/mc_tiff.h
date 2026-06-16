/* mc_tiff.h — minimal, dependency-free TIFF reader/writer for matter-compressor.
 *
 * Not a general TIFF library. It writes (and reads back) the narrow layout we
 * control: a single uncompressed, contiguous, interleaved strip whose pixel
 * data starts at an 8-byte-aligned file offset, little-endian, one IFD. That
 * layout is what makes a file MMAP-DIRECTLY-INTO-A-C-STRUCT: the reader
 * validates the geometry/type and hands back a pointer into the mmap, so e.g. a
 * width*height image of 4x f32 samples is a `const float (*)[4]` with no copy.
 *
 * It still PARSES enough standard TIFF to read such files written by anything
 * that produces a single contiguous strip in a supported sample type. Reading a
 * tiled / multi-strip / compressed / big-endian / BigTIFF file fails cleanly
 * (the caller can fall back to a copy path).
 *
 * Supported sample types (BitsPerSample x SampleFormat):
 *   u8 (8/UINT), u16 (16/UINT), i16 (16/INT), u32 (32/UINT), f32 (32/IEEEFP).
 * 1..4 samples per pixel, interleaved (chunky).
 */
#ifndef MC_TIFF_H
#define MC_TIFF_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    MC_TIFF_U8  = 0,   // 8-bit unsigned
    MC_TIFF_U16 = 1,   // 16-bit unsigned
    MC_TIFF_I16 = 2,   // 16-bit signed
    MC_TIFF_U32 = 3,   // 32-bit unsigned
    MC_TIFF_F32 = 4,   // 32-bit IEEE float
} mc_tiff_type;

// bytes per sample for a type
size_t mc_tiff_type_size(mc_tiff_type t);

// A read TIFF: an mmap of the whole file plus a validated view of its single
// contiguous interleaved strip. `pixels` points INTO the mmap at the first
// sample (row-major, chunky: pixel (y,x) sample s at
// pixels + ((y*width + x)*samples + s)*type_size). Valid until mc_tiff_close.
typedef struct {
    int          width, height, samples;
    mc_tiff_type type;
    const void  *pixels;       // pointer into the mmap (no copy)
    size_t       pixel_bytes;  // width*height*samples*type_size
    // internal
    void  *map; size_t map_len; int fd;
} mc_tiff;

// Open + mmap + validate. Returns 0 on success (fills *out), <0 on error or an
// unsupported layout (tiled/multi-strip/compressed/etc.). On failure *out is
// zeroed and nothing is left mapped.
int  mc_tiff_open(const char *path, mc_tiff *out);
void mc_tiff_close(mc_tiff *t);

// Write `pixels` (width*height*samples of `type`, interleaved, row-major) as a
// single contiguous strip with the mmap-friendly layout above. Returns 0 on
// success. The produced file round-trips through mc_tiff_open.
int  mc_tiff_write(const char *path, int width, int height, int samples,
                   mc_tiff_type type, const void *pixels);

#endif /* MC_TIFF_H */
