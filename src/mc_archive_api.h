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

// ---- read side ----
typedef struct mc_reader mc_reader;
mc_reader *mc_open(const uint8_t *arc, size_t len);       // quality is read from... the caller (see note)
void       mc_close(mc_reader *r);
void       mc_reader_set_quality(mc_reader *r, float q);  // must match the build quality to decode
uint64_t   mc_chunk_offset(mc_reader *r, int lod, int cz,int cy,int cx);  // 0 = empty
void       mc_decode_block(mc_reader *r, uint64_t chunk_off, int bz,int by,int bx, mc_u8 *dst);

// metadata region (pointer into arc; not owned). *out_len = bytes stored.
const char *mc_metadata(const uint8_t *arc, size_t *out_len);

#endif
