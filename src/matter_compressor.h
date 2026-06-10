// ============================================================================
// matter_compressor.h — single-header public API: codec + archive + cache.
// Unified from mc_codec.h / mc_archive_api.h / mc_cache.h.
// ============================================================================
#ifndef MATTER_COMPRESSOR_H
#define MATTER_COMPRESSOR_H

// ============================================================================
// mc_codec.h — matter-compressor block codec.
//
// Compresses a 16^3 u8 voxel block: integer separable DCT-16 + dead-zone quant +
// CABAC range coder. Mask-aware: air voxels (value 0) are harmonically air-filled
// before the DCT and force-zeroed on decode. The air mask is SELF-CONTAINED per
// block: mixed blocks carry their own context-coded 16^3 mask in the payload
// (all-material blocks pay 0 bytes; all-air blocks are absent entirely). This
// makes a single block independently decodable — no chunk-level mask pass.
//
// Pure transform — no I/O, no archive/volume knowledge. The only runtime
// parameter is `quality` (the quant base step; higher = more compression,
// lower fidelity).
// ============================================================================
#include <stdint.h>
#include <stddef.h>

typedef uint8_t  mc_u8;
typedef int32_t  mc_i32;

#define MC_BLK    16     // DCT block edge
#define MC_CHUNK  256    // archive chunk edge (16 blocks)

// frozen quant constants (tuned; see matter-compressor design notes)
#define MC_DZ_FRAC     0.80f   // dead-zone width fraction
#define MC_HF_EXP      0.65f   // HF quant power-law: step = quality*(1+L1freq)^MC_HF_EXP
#define MC_FILL_SWEEPS 3       // SOR air-fill sweeps before the DCT (omega=1.6)

void  mc_set_quality(float quality);   // the main runtime parameter
float mc_get_quality(void);
// Optional guaranteed max-error bound (near-lossless mode): after transform
// coding, the encoder codes sparse corrections for every voxel whose error
// exceeds tau, so |orig - decoded| <= tau for all material voxels. 0 = off
// (default). Corrections are self-contained in the block payload (the decoder
// does not need to know tau). Encode-side cost: one extra inverse DCT per block.
void  mc_set_max_error(int tau);
int   mc_get_max_error(void);
void  mc_codec_init(void);             // one-time: build the DCT tables
// Override the trained context priors (q=1 / q=12 endpoint tables, u16[8][32]
// each) — used by per-volume prior blobs. NULL,NULL restores the baked tables.
// Process-global; set before encode/decode threads run.
void  mc_codec_set_priors(const uint16_t *plo, const uint16_t *phi);

// growable output byte buffer the codec appends block payloads to.
typedef struct { mc_u8 *p; size_t len, cap; } mc_buf;
void  mc_buf_put(mc_buf *b, const void *s, size_t n);

// Optional decode-side deblocking: H.264-style clamped 1D filter across the
// 16-voxel block faces of an assembled volume (any box, dims in voxels).
// Strength scales with the quality the data was coded at; air voxels (0) are
// never touched. Call AFTER assembling decoded blocks into a contiguous region.
void  mc_deblock(mc_u8 *vol, int nz, int ny, int nx, float quality);

// encode one 16^3 block (z,y,x raster; air = value 0). Appends payload to out,
// sets *len_out. Returns 1 if coded (nonzero), 0 if all-zero (no payload).
int   mc_enc_block(const mc_u8 *vox, mc_buf *out, uint32_t *len_out);
// decode one block payload of `plen` bytes into dst (16^3). Self-contained
// (mask in payload); plen comes from the chunk's block-length table.
void  mc_dec_block(const mc_u8 *payload, uint32_t plen, mc_u8 *dst);

// per-chunk material-fraction map (4096 nibbles 0..15), context-coded.
uint32_t mc_enc_fracmap(const mc_u8 *frac, mc_u8 *out, size_t cap);
void     mc_dec_fracmap(const mc_u8 *in, uint32_t len, mc_u8 *frac);


// ============================================================================
// mc_archive_api.h — matter-compressor archive build + decode API.
//
// SOURCE-AGNOSTIC: the archive knows nothing about zarr or S3. The builder pulls
// voxels through a caller-supplied source callback; an exporter tool (tools/) is
// where zarr/S3 loading lives. Depends only on mc_codec.
// ============================================================================
#include <stdint.h>
#include <stddef.h>

// Voxel source: return the u8 value at (x,y,z) of the full-res volume; out-of-range
// or absent -> 0 (air). Called during the LOD0 pass; coarser LODs are decimated from
// the source internally. `ud` is the caller's context.
typedef mc_u8 (*mc_voxel_fn)(void *ud, int x, int y, int z);

// Build options. Either set cubic `dim`, or per-axis nx/ny/nz (voxels; each is
// padded up to the next 256 boundary internally — zero padding is nearly free:
// all-air blocks cost one bitmap bit and absent chunks cost nothing).
typedef struct {
    int   dim;            // cubic volume edge (used when nx/ny/nz are 0)
    int   nx, ny, nz;     // per-axis dims (0 -> use `dim`)
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
// Per-axis variant: nx/ny/nz in voxels, each padded up to the next 256
// boundary. Chunk coordinates run over the padded grid per axis.
mc_archive *mc_archive_open_dims(const char *path, int nx, int ny, int nz, float quality);

// Append one 256^3 chunk of raw u8 voxels at chunk coords (cz,cy,cx) in `lod`. Encodes
// via the mc codec, writes the compressed chunk blob contiguously at EOF, installs it
// in the index. Returns 0 on success. An all-air chunk is a no-op (slot stays absent,
// which decodes to zero).
int mc_archive_append_chunk_raw(mc_archive *a, int lod, int cz,int cy,int cx,
                                const mc_u8 vox[256*256*256]);
// Per-chunk quality variant (rate control / ROI archives): this chunk encodes
// at `q` instead of the archive default. The chunk's q is stored in its blob
// (format v6), so decode needs nothing extra.
int mc_archive_append_chunk_raw_q(mc_archive *a, int lod, int cz,int cy,int cx,
                                  const mc_u8 vox[256*256*256], float q);
// Rate-controlled variant: pick this chunk's q to hit ~target_ratio (raw bytes
// / compressed bytes) for THIS chunk. One 1/16-block sample encode at the
// archive's base q plus a single power-law correction (~6% encode overhead,
// no iteration). Heterogeneous content lands within ~10-20% of target; use
// per-volume averaging upstream if you need it exact. Chosen q is stored in
// the chunk (v6) and returned via *q_out if non-NULL.
int mc_archive_append_chunk_target(mc_archive *a, int lod, int cz,int cy,int cx,
                                   const mc_u8 vox[256*256*256], float target_ratio,
                                   float *q_out);

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

// Decode a WHOLE 256^3 chunk into `out` (z,y,x raster), using an internal
// worker pool over the chunk's 4096 independent blocks (nthreads = 0 -> one
// per core, capped 16; 1 = serial). chunk_off==0 zero-fills.
void mc_archive_decode_chunk(mc_archive *a, uint64_t chunk_off, mc_u8 *out, int nthreads);

// Parallel append: encode the chunk's blocks with an internal worker pool,
// then install the assembled blob (identical bytes to the serial path).
int mc_archive_append_chunk_par(mc_archive *a, int lod, int cz,int cy,int cx,
                                const mc_u8 vox[256*256*256], float q, int nthreads);

// Decode an arbitrary axis-aligned REGION of `lod` into a caller buffer.
// Voxel (z,y,x) of the region lands at out[z*sz + y*sy + x] (strides in
// bytes; sy=dx, sz=dx*dy gives a dense C-order array — pass torch/numpy
// strides for zero-copy tensor fills). Only the touched blocks are decoded,
// in parallel (nthreads=0 -> one per core, cap 16). Out-of-volume voxels and
// absent chunks read as 0. The region primitive for ML dataloaders (random
// crops) and viewers (slices: dz=1).
void mc_archive_read_region(mc_archive *a, int lod,
                            long z0, long y0, long x0,
                            long dz, long dy, long dx,
                            mc_u8 *out, size_t sz, size_t sy, int nthreads);

// Occupancy without decoding: is the 16^3 block at block coords (bz,by,bx)
// of `lod` present (i.e. contains any material)? Bitmap lookup only — use
// for rejection-free sampling of material-containing patches.
int mc_archive_block_present(mc_archive *a, int lod, int bz, int by, int bx);

// Material fraction of a block in [0,1] (quantized to 1/15 steps; from the
// per-chunk fraction map, decoded once per chunk per thread — no voxel
// decode). 0 for absent blocks/chunks.
float mc_archive_block_fraction(mc_archive *a, int lod, int bz, int by, int bx);

// Deterministic seeded box sampler for ML dataloaders: draw `count` boxes of
// size (dz,dy,dx) voxels at `lod`, uniformly over the volume, keeping only
// boxes whose mean block material fraction >= min_frac. Same seed -> same
// boxes, independent of thread count or machine. Returns boxes written
// (< count if the acceptance rate is too low within the attempt budget).
typedef struct { long z0, y0, x0; } mc_box;
int mc_archive_sample_boxes(mc_archive *a, int lod, uint64_t seed, int count,
                            long dz, long dy, long dx, float min_frac,
                            mc_box *out);

// Batched multi-crop read: decode n same-sized regions into one batch buffer
// (crop i at out + i*batch_stride, dense C-order dz*dy*dx each). Workers
// process whole crops in parallel — the ML dataloader primitive.
void mc_archive_read_regions(mc_archive *a, int lod, const mc_box *boxes, int n,
                             long dz, long dy, long dx,
                             mc_u8 *out, size_t batch_stride, int nthreads);

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

// Total byte length of the chunk blob at `chunk_off` (flat or streaming reader);
// 0 on error. Pair with mc_chunk_offset to range-copy compressed chunks verbatim
// into another archive via mc_archive_append_chunk_compressed.
uint64_t mc_reader_chunk_blob_len(mc_reader *r, uint64_t chunk_off);

// Raw per-volume prior arrays (plo/phi as u16[8][32]); returns 0 if the archive
// stores none. Feed into mc_archive_set_priors so a local mirror decodes identically.
int mc_reader_priors(mc_reader *r, const uint16_t **plo, const uint16_t **phi);

// Header metadata (valid for flat and streaming readers).
void  mc_reader_dims(mc_reader *r, int *nx, int *ny, int *nz);  // LOD0 voxel dims
float mc_reader_quality(mc_reader *r);                          // build quality
int   mc_reader_nlods(mc_reader *r);                            // populated LOD count

// metadata region (pointer into arc; not owned). *out_len = bytes stored.
const char *mc_metadata(const uint8_t *arc, size_t *out_len);

// Integrity: recompute a chunk blob's xxh64 (over bitmap+lens+payloads) and
// the stored value (format v6 stores it in the blob header). mc_verify walks
// every chunk of an archive buffer; returns number of corrupt chunks.
uint64_t mc_chunk_compute_hash(const uint8_t *blob, uint64_t blob_len);
long mc_verify_archive(const uint8_t *arc, size_t len, int verbose);

// Per-volume trained priors (format v6). Store a prior blob (from mc_train's
// binary output: q=1 and q=12 endpoint tables) in the archive; every open of
// the archive then decodes with these instead of the baked corpus tables.
// Set BEFORE appending (the encoder uses them too). Process-global: one prior
// set active per process at a time (matches the one-quality-dial design).
int mc_archive_set_priors(struct mc_archive *a,
                          const uint16_t plo[8][32], const uint16_t phi[8][32]);


// ============================================================================
// mc_cache.h — in-RAM decoded-block cache for interactive clients.
//
// A thin layer in front of the matter-compressor decode APIs: caches decoded
// 16^3 blocks (4 KB each, exactly one page) keyed by (lod, bz, by, bx) in an
// mmap-backed arena with CLOCK/NRU eviction. Designed for vc3d-style
// rendering clients: many reader threads, high RAM budgets (tens of GB),
// hot viewport revisits must not re-decode.
//
//   - sharded open-addressing hash (64 shards, one mutex + one clock hand
//     each) — readers on different shards never contend,
//   - arena slots are partitioned per shard, so eviction sweeps are local,
//   - get() decodes on miss via a caller-supplied source callback (bindings
//     for mc_archive and mc_reader below), with a post-decode re-check so
//     concurrent misses of the same block insert once,
//   - returned pointers are non-owning views into the arena. A slot can be
//     reused by a later eviction; a stale frame is possible, a UAF is not
//     (same contract as vc's BlockCache). Use mc_cache_get_copy when the
//     caller needs a stable snapshot.
//
// THREAD-SAFETY CONTRACT: all operations are safe from any number of threads
// (per-shard mutexes; decode runs outside the locks with a double-checked
// insert). mc_cache_get's pointer may be overwritten by a concurrent
// eviction (torn read, never UAF) — use mc_cache_get_copy where that
// matters. Writers that REPLACE a chunk in the source must call
// mc_cache_invalidate_chunk afterwards; the mc_reader binding serializes
// decode (mc_reader is single-threaded), the mc_archive binding does not.
// ============================================================================
#include <stdint.h>
#include <stddef.h>

typedef struct mc_cache mc_cache;

// Source callback: decode block (bz,by,bx) of `lod` into dst (16^3 bytes).
// Called outside any cache lock; must be thread-safe (mc decode is).
typedef void (*mc_cache_src_fn)(void *ud, int lod, int bz, int by, int bx, mc_u8 *dst);

// Eviction policy. CLOCK = classic NRU sweep. S3FIFO (default) = small/main
// FIFO queues + ghost table (SOSP'23): one-hit wonders die in the small queue,
// re-referenced ghosts go straight to main — scan-resistant, which matters for
// render loops (slice sweeps re-touch everything once per frame).
typedef enum { MC_CACHE_S3FIFO = 0, MC_CACHE_CLOCK = 1 } mc_cache_policy;

// Create a cache with ~`bytes` of block storage (rounded to slots of 4 KB).
mc_cache *mc_cache_new(size_t bytes, mc_cache_src_fn src, void *src_ud);
// Switch eviction policy (call before first use; clears nothing).
void mc_cache_set_policy(mc_cache *c, mc_cache_policy p);
void      mc_cache_free(mc_cache *c);

// Get the decoded block, from cache or by decoding (and caching) it.
// Returns a non-owning pointer to 4096 bytes. Never NULL.
const mc_u8 *mc_cache_get(mc_cache *c, int lod, int bz, int by, int bx);

// Like get, but memcpy into caller storage under the shard lock — immune to
// concurrent eviction reuse.
void mc_cache_get_copy(mc_cache *c, int lod, int bz, int by, int bx, mc_u8 *dst);

// Peek without touching the recently-used bit (residency triage).
int  mc_cache_contains(mc_cache *c, int lod, int bz, int by, int bx);

// Prefetch every block of chunk (lod, cz,cy,cx) into the cache (decoding the
// blocks that are not already resident). Renderers call this from IO/worker
// threads ahead of the viewport so the render threads only ever hit.
void mc_cache_prefetch_chunk(mc_cache *c, int lod, int cz, int cy, int cx);

// Batch update: bring an explicit working set into the cache and return when
// every listed block is resident. The vc3d pattern: compute all needed blocks
// up front from geometry, submit the list, wait, then query (all hits).
//   - already-resident blocks are touched (marked recently used) so the new
//     working set is protected; everything else decays toward eviction,
//   - misses are decoded by an internal worker pool, grouped by chunk for
//     payload locality (nthreads = 0 -> one per core, capped at 16),
//   - duplicates in the list are fine; list order does not matter,
//   - the working set should fit in the cache: if n exceeds capacity, later
//     inserts evict earlier ones from the same batch.
// Returns the number of blocks that were newly decoded.
typedef struct { int lod, bz, by, bx; } mc_block_id;
size_t mc_cache_update(mc_cache *c, const mc_block_id *ids, size_t n, int nthreads);

// Invalidate every cached block of chunk (lod, cz,cy,cx). Writers call this
// after re-appending/replacing a chunk so readers stop seeing stale blocks.
// Thread-safe against concurrent gets/inserts.
void mc_cache_invalidate_chunk(mc_cache *c, int lod, int cz, int cy, int cx);

// Finest resident LOD covering block (bz,by,bx in `finest_lod` block coords):
// probes finest_lod..7 (block coords halve per level) WITHOUT touching
// recently-used bits. Returns the lod, or -1 if nothing is resident. The
// page-table pattern: render the best resident level now, refine later.
int mc_cache_best_lod(mc_cache *c, int finest_lod, int bz, int by, int bx);

// Async batch update: like mc_cache_update but returns immediately with a
// ticket. Poll mc_cache_ticket_done, or mc_cache_ticket_wait to block.
// mc_cache_ticket_cancel makes workers stop at the next chunk-group boundary
// (camera moved -> abandon stale prefetch). Always mc_cache_ticket_free
// (it joins any remaining workers first).
typedef struct mc_cache_ticket mc_cache_ticket;
mc_cache_ticket *mc_cache_update_async(mc_cache *c, const mc_block_id *ids, size_t n, int nthreads);
int  mc_cache_ticket_done(mc_cache_ticket *t);      // 1 when all work finished/cancelled
void mc_cache_ticket_cancel(mc_cache_ticket *t);
void mc_cache_ticket_wait(mc_cache_ticket *t);
void mc_cache_ticket_free(mc_cache_ticket *t);

// ---- tick-phase contract (vc3d game-loop model) ----------------------------
// thaw():   write phase — update/resolve/invalidate/prefetch allowed; reads
//           take shard locks as usual. Increments the pin epoch.
// freeze(): read phase — the cache is immutable. get/get_copy/contains/
//           best_lod skip ALL locks (no writer can exist) and do not touch
//           eviction state. Misses do NOT insert: get returns NULL (fall back
//           to mc_cache_best_lod), get_copy decodes directly into the caller
//           buffer; both record the miss in the feedback queue.
// Write-phase calls made while frozen are refused (return error / no-op) —
// the contract is load-bearing for the lock-free reads.
void mc_cache_freeze(mc_cache *c);
void mc_cache_thaw(mc_cache *c);

// Resolve: batch update + pointer table, the render-phase fast path. Ensures
// every ids[i] is resident (decoding misses in parallel), fills ptrs[i] with
// its arena address, and epoch-pins those slots so this frame's own inserts
// cannot evict them. Pointers stay valid until the next thaw(). Thawed phase
// only. Returns blocks newly decoded.
size_t mc_cache_resolve(mc_cache *c, const mc_block_id *ids, size_t n,
                        const mc_u8 **ptrs, int nthreads);

// Drain the frozen-phase miss queue (call after thaw; feed into the next
// update/resolve batch). Duplicates possible — update() handles them.
size_t mc_cache_misses_drain(mc_cache *c, mc_block_id *out, size_t cap);

// Drop everything (e.g. source archive replaced).
void mc_cache_clear(mc_cache *c);

// Diagnostics.
typedef struct { uint64_t hits, misses, evictions; size_t slots, used; } mc_cache_stats;
void mc_cache_get_stats(mc_cache *c, mc_cache_stats *out);

// ---- ready-made source bindings ----
struct mc_archive; struct mc_reader;
// Bind to a local archive handle (lock-free concurrent decode).
mc_cache *mc_cache_new_archive(size_t bytes, struct mc_archive *a);
// Bind to a reader (flat or streaming). NOTE: mc_reader decode is not
// thread-safe across threads sharing one reader; this binding serializes
// decode with an internal mutex. For maximum parallelism, give each thread
// its own reader and use mc_cache_new with a custom callback.
mc_cache *mc_cache_new_reader(size_t bytes, struct mc_reader *r);



#endif // MATTER_COMPRESSOR_H
