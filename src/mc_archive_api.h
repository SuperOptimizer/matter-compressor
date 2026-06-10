// ============================================================================
// mc_archive_api.h — matter-compressor archive build + decode API.
//
// SOURCE-AGNOSTIC: the archive knows nothing about zarr or S3. The builder pulls
// voxels through a caller-supplied source callback; an exporter tool (tools/) is
// where zarr/S3 loading lives. Depends only on mc_codec.
// ============================================================================
#ifndef MC_ARCHIVE_API_H
#define MC_ARCHIVE_API_H
#include <stdint.h>
#include <stddef.h>
#include "mc_codec.h"

// Voxel source: return the u8 value at (x,y,z) of the full-res volume; out-of-range
// or absent -> 0 (air). Called during the LOD0 pass; coarser LODs are decimated from
// the source internally. `ud` is the caller's context.
typedef mc_u8 (*mc_voxel_fn)(void *ud, int x, int y, int z);

// Build options.
typedef struct {
    int   dim;            // volume edge (must be a multiple of MC_CHUNK_ALIGN=256)
    float quality;        // codec quality dial (base quant step)
    const char *metadata; // optional free-form text stored in the metadata region (NULL = none)
    size_t meta_len;      // metadata byte length
} mc_build_opts;

// Build an archive into a malloc'd buffer (caller frees via free()). Returns the
// buffer + writes its length to *out_len; NULL on error. The builder is the one
// piece that materializes the volume; for a 1024^3 it holds the full-res volume +
// the LOD pyramid, so the caller's source should be cheap to sample repeatedly.
uint8_t *mc_build(mc_voxel_fn src, void *ud, const mc_build_opts *opts, size_t *out_len);

// Convenience: build and write to a file. Returns 0 on success.
int mc_build_to_file(mc_voxel_fn src, void *ud, const mc_build_opts *opts, const char *outpath);

// ============================================================================
// mc_archive — a persistent, crash-safe, READ+WRITE on-disk archive. ONE handle both
// appends chunks and decodes them (no writer/reader split). Reopens across process
// runs and keeps appending. The file is a fully valid, decodable archive after every
// appended chunk (chunk payloads append at EOF; the dense node index is updated in
// place with the chunk offset published LAST as the commit word). Modeled on
// volume-compressor's mmap+atomic-cursor writer: a large virtual reservation whose base
// never moves, file grown by ftruncate, append cursor advanced atomically, lock-free
// concurrent appends to disjoint EOF ranges, decode reads the live mmap.
// ============================================================================
typedef struct mc_archive mc_archive;

// Per-chunk coverage (queried without decoding).
typedef enum { MC_ABSENT = 0, MC_PRESENT = 1 } mc_cover;

// Open (or create) an archive at `path` for a volume of edge `dim` (multiple of
// MC_CHUNK_ALIGN=256) at the given `quality`. If the file exists and is a valid mc
// archive, it is reopened (dim/quality must match the stored header). NULL on failure.
mc_archive *mc_archive_open(const char *path, int dim, float quality);

// Append one 256^3 chunk of raw u8 voxels at chunk coords (cz,cy,cx) in `lod`. Encodes
// via the mc codec, writes the compressed chunk blob contiguously at EOF, installs it
// in the index. Returns 0 on success. An all-air chunk is a no-op (slot stays absent,
// which decodes to zero).
int mc_archive_append_chunk_raw(mc_archive *a, int lod, int cz,int cy,int cx,
                                const mc_u8 vox[256*256*256]);

// Append an ALREADY-COMPRESSED chunk blob verbatim (no re-encode). `blob`/`len` must be
// a valid mc chunk blob. Direct .mca -> .mca fast path. Returns 0 on success.
int mc_archive_append_chunk_compressed(mc_archive *a, int lod, int cz,int cy,int cx,
                                       const uint8_t *blob, size_t len);

// Pre-create the index path (root/inner/shard tables) for a chunk WITHOUT
// writing any chunk data. Exporters call this for every chunk first, so all
// index tables are allocated contiguously right after the metadata region and
// a streaming reader can fetch the entire index with one ranged read; chunk
// blobs then follow in append order (e.g. Morton). Returns 0 on success.
int mc_archive_reserve_index(mc_archive *a, int lod, int cz,int cy,int cx);

// Coverage of a chunk without decoding.
mc_cover mc_archive_chunk_coverage(mc_archive *a, int lod, int cz,int cy,int cx);

// Resolve a chunk to its blob offset (0 = absent). Pass to mc_archive_decode_block to
// decode its 16^3 blocks (resolve once per chunk, decode 4096 blocks).
uint64_t mc_archive_chunk_offset(mc_archive *a, int lod, int cz,int cy,int cx);

// Decode one 16^3 block (bz,by,bx in [0,16)) of the chunk at `chunk_off` into `dst`
// (16^3 bytes). Missing/air -> dst zeroed. Reads the live mmap; safe vs concurrent
// appends.
void mc_archive_decode_block(mc_archive *a, uint64_t chunk_off, int bz,int by,int bx, mc_u8 *dst);

// Flush + close (msync, persist header, truncate to exact length).
void mc_archive_close(mc_archive *a);

// ---- read-only-from-bytes side (already-built archive in a buffer / mmap) ----
typedef struct mc_reader mc_reader;
mc_reader *mc_open(const uint8_t *arc, size_t len);       // in-memory / mmap'd archive
void       mc_close(mc_reader *r);
void       mc_reader_set_quality(mc_reader *r, float q);  // must match the build quality to decode
uint64_t   mc_chunk_offset(mc_reader *r, int lod, int cz,int cy,int cx);  // 0 = empty
void       mc_decode_block(mc_reader *r, uint64_t chunk_off, int bz,int by,int bx, mc_u8 *dst);

// ---- streaming read side: open an archive WITHOUT holding it whole in memory ----
// Byte-source callback: fill `dst` with exactly `len` bytes at `off`. Return 0 on
// success. The implementation typically range-fetches + caches (e.g. S3) behind this.
typedef int (*mc_read_fn)(void *ud, uint64_t off, uint32_t len, uint8_t *dst);

// Open a streaming reader: libmc pulls byte ranges through `read` on demand (header,
// node probes, chunk blobs). `total_len` is the true archive size. Decode results are
// identical to mc_open on the same bytes. The callback must outlive the handle.
mc_reader *mc_open_streaming(mc_read_fn read, void *ud, uint64_t total_len);

// Partial-fetch mode (streaming readers only): mc_decode_block fetches just the
// chunk's bitmap+length table (cached per chunk, <=8.7KB) and the block's own
// payload (typically <100B) instead of the whole chunk blob — far lower cold
// random-access cost over S3-like sources. Leave OFF for full-chunk scans
// (whole-blob fetch amortizes better).
void mc_reader_set_partial_fetch(mc_reader *r, int on);

// metadata region (pointer into arc; not owned). *out_len = bytes stored.
const char *mc_metadata(const uint8_t *arc, size_t *out_len);

#endif
