// mc_volume — remote volume as a local .mca: stream, transcode, cache, prefetch.
//
// Give it a URL (an s3:// / https:// zarr root). It discovers levels, and on
// demand streams source chunks (mc_zarr), decodes them (c3d / blosc / raw) into
// dense 256^3, and re-encodes them into ONE local .mca (mc_archive) that persists
// across runs. Decoded 16^3 blocks are served through the .mca's mc_cache.
//
// This is the whole remote/cache/prefetch layer VC3D used to carry in
// MatterCache.cpp + ChunkCache.cpp — here, self-contained and codec-aware.
//
// Async contract for the render path: mc_volume_try_block serves present blocks
// synchronously and, on an absent region, kicks ONE deduplicated background
// transcode and returns 0. The caller renders a coarser LOD meanwhile; it keeps
// no miss state. Air regions are recorded (mc ZERO coverage) so they are never
// re-fetched.
#ifndef MC_VOLUME_H
#define MC_VOLUME_H
#include <stdint.h>
#include <stddef.h>

typedef struct mc_volume mc_volume;

// Open a remote volume rooted at `url` (an NGFF multiscales zarr group; levels
// are its "0","1",... arrays). `cache_dir` holds the local <name>.mca; pass the
// resident mc_cache budget in `cache_bytes`. `quality` is the local re-encode
// quality (mc q scale). Returns NULL on failure (unreachable / unparseable).
mc_volume *mc_volume_open(const char *url, const char *cache_dir,
                          size_t cache_bytes, float quality);
void       mc_volume_free(mc_volume *v);

int  mc_volume_nlods(const mc_volume *v);
void mc_volume_shape(const mc_volume *v, int lod, int *nz, int *ny, int *nx);
// block (16^3) grid extent of a level.
void mc_volume_block_grid(const mc_volume *v, int lod, int *nz, int *ny, int *nx);

// Serve one 16^3 block (block coords) of `lod` into `dst` (4096 bytes).
//   present -> decode from mc_cache (sync), return 1.
//   absent  -> kick one deduped background region transcode, zero `dst`,
//              return 0 (caller falls back to a coarser LOD).
int mc_volume_try_block(mc_volume *v, int lod, int bz, int by, int bx, uint8_t *dst4096);

// Blocking variant (batch/CLI): transcode the enclosing region if absent, then
// decode. Returns 1 on data, 0 on air, <0 on error.
int mc_volume_get_block(mc_volume *v, int lod, int bz, int by, int bx, uint8_t *dst4096);

// Prefetch: download a whole source shard (one GET) and transcode every present
// inner chunk into the .mca. Air shards skipped via the index probe.
void mc_volume_prefetch_shard(mc_volume *v, int lod, int cz, int cy, int cx);
// Walk a level shard-by-shard with `nthreads` workers; *cancel (if non-NULL) is
// polled to abort early.
void mc_volume_prefetch_level(mc_volume *v, int lod, int nthreads, volatile int *cancel);

// Register a callback fired (from a worker thread) each time a background
// transcode completes a region — lets an interactive client schedule a repaint.
// `cb` must be cheap and thread-safe (e.g. set a flag / post to the UI loop).
typedef void (*mc_volume_ready_fn)(void *ud);
void mc_volume_set_ready_cb(mc_volume *v, mc_volume_ready_fn cb, void *ud);

typedef struct {
    uint64_t cache_hits, cache_misses;   // mc_cache residency
    uint64_t disk_bytes;                 // .mca append cursor
    uint64_t net_bytes;                  // bytes pulled from the source
    uint64_t regions_inflight;           // single-flight depth right now
} mc_volume_stats;
void mc_volume_get_stats(const mc_volume *v, mc_volume_stats *out);

#endif
