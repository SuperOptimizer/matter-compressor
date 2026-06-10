// mc_zarr — narrow zarr reader for the two formats VC volumes actually use:
//
//   v3 sharded (compress3d archives) — the exact subset those archives emit:
//       chunk_grid chunk_shape = SHARD (e.g. 4096^3); sharding_indexed codec's
//       chunk_shape = inner chunk (e.g. 256^3); index_location:start; "bytes"
//       index codec (little-endian); inner codec "c3d"; uint8, fill 0. No other
//       v3 variants (no blosc-in-v3, no transpose, no index_location:end).
//   v2 flat — .zarray, one object per chunk (e.g. 128^3) at key "z.y.x",
//       compressor blosc1/zstd ("blosc") or null ("raw"); uint8. A "shard" == one
//       chunk here.
//
// The caller sees a uniform inner-chunk grid and gets each chunk's bytes ready
// for the named inner codec:
//   - "c3d"  : RAW (still c3d-compressed) — caller decodes with a c3d_decoder.
//   - "blosc"/"raw": already dense u8 (mc_zarr decoded blosc / copied verbatim).
//
// Transport-agnostic: the caller supplies a byte-source callback (mc_zarr_read_fn)
// so this TU has no S3/HTTP dependency.
//
// Inner-chunk coords below are GLOBAL (level-wide), not shard-relative.
#ifndef MC_ZARR_H
#define MC_ZARR_H
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mc_zarr mc_zarr;

// Byte source. `key` is an object key relative to the level root (e.g. "c/3/0/1"
// for a shard, or "zarr.json"). [off,off+len) is the requested byte range; len 0
// means the whole object. On success store a malloc'd buffer in *out (caller-owned,
// freed with free()) and its length in *out_len, return 0. Return <0 on transport
// error, and set *out_len to 0 with *out NULL for a 404 / absent object.
typedef int (*mc_zarr_read_fn)(void *ud, const char *key,
                               uint64_t off, uint64_t len,
                               uint8_t **out, size_t *out_len);

// Open a level. `read`/`ud` is the byte source; the level's "zarr.json" is fetched
// through it. Returns NULL on parse/transport failure.
mc_zarr *mc_zarr_open(mc_zarr_read_fn read, void *ud);
void     mc_zarr_free(mc_zarr *z);

// Geometry (voxels / edges).
void mc_zarr_shape(const mc_zarr *z, int *nz, int *ny, int *nx);
int  mc_zarr_inner_edge(const mc_zarr *z);          // e.g. 256
int  mc_zarr_shard_edge(const mc_zarr *z);          // e.g. 4096
// inner codec the caller must apply to bytes from mc_zarr_read_*: "c3d" means
// RAW c3d (caller decodes); "blosc"/"raw" mean already-dense u8 (mc_zarr decoded).
const char *mc_zarr_inner_codec(const mc_zarr *z);
// inner chunks per axis across the whole level (ceil(shape/inner_edge)).
void mc_zarr_inner_grid(const mc_zarr *z, int *nz, int *ny, int *nx);

// Is the shard containing global inner chunk (cz,cy,cx) entirely air? Reads only
// the shard's index footer (one small ranged read). 1 yes, 0 no, -1 unknown/error.
int mc_zarr_shard_all_air(mc_zarr *z, int cz, int cy, int cx);

// Fetch one shard (one GET of the whole object) and hand each PRESENT inner chunk
// to `sink` as its RAW (still-compressed) bytes. (cz,cy,cx) is any global inner
// chunk in the target shard. Absent inner chunks are skipped. 0 ok, <0 error.
typedef void (*mc_zarr_chunk_fn)(void *ud, int cz, int cy, int cx,
                                 const uint8_t *raw, size_t raw_len);
int mc_zarr_read_shard(mc_zarr *z, int cz, int cy, int cx,
                       mc_zarr_chunk_fn sink, void *sink_ud);

// Fetch a single inner chunk's RAW bytes (interactive cold path). Two ranged
// reads: the shard index footer, then the chunk payload. On success store a
// malloc'd buffer in *raw (caller frees) and length in *len, return 0. Returns
// 1 if the chunk is absent (air) with *raw NULL. <0 on error.
int mc_zarr_read_inner(mc_zarr *z, int cz, int cy, int cx,
                       uint8_t **raw, size_t *len);

// One present inner chunk of a shard: global coords + its byte range in the
// shard object. (v2: off/len describe the standalone chunk object, range from 0.)
typedef struct {
    int cz, cy, cx;       // global inner-chunk coords
    uint64_t off, len;    // payload byte range within the shard object's key
} mc_zarr_range;

// Read a shard's index footer (ONE ranged read) and return the byte ranges of
// every PRESENT inner chunk, plus the shard object key (for batched GETs). The
// caller fetches the payloads however it likes (e.g. a parallel s3 batch), then
// decodes. `key` is filled with the shard object key (relative to the level).
// On success *out is a malloc'd array of *n entries (caller frees), return 0.
// All-air / absent shard -> *n = 0, *out NULL, return 0. <0 on error.
// (v2: returns the single chunk's key + {off=0,len=0-means-whole-object}.)
int mc_zarr_shard_index(mc_zarr *z, int cz, int cy, int cx,
                        char key_out[64], mc_zarr_range **out, int *n);

#ifdef __cplusplus
}
#endif

#endif
