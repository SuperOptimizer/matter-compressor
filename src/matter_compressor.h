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
//
// ARCHIVAL PRESET: quality 0.5 + tau 1 — every material voxel within +/-1
// greylevel (below micro-CT reconstruction noise), air bit-exact. Measured on
// real 2.4um scroll data: 2.9x ratio / 51.9 dB / SSIM 0.9996, vs 1.96x for
// true lossless (zstd-19) — there is deliberately no lossless mode; the DCT
// path is not bit-reversible and the entropy ceiling makes one pointless.
// tau 2 -> 4.0x, tau 3 (q 1) -> 5.3x, all with p99 == max == tau.
void  mc_set_max_error(int tau);
int   mc_get_max_error(void);

// Calibrated preset ladder: level L guarantees |err| <= 2^L on every material
// voxel (air is always bit-exact) with the quality that maximizes ratio under
// that bound — measured on real 2.4um scroll data (PHerc Paris 4, masked
// 512^3; 18-point calibration in bench/RESULTS.md). Ratio ~doubles per level:
//   level  tau  q     ratio  PSNR   SSIM    dec MB/s (1T)
//   0 archival   1  0.5    2.9x  51.9  0.9996   33
//   1 master     2  0.5    4.0x  48.6  0.9991   38
//   2 high       4  1      6.6x  44.3  0.9975   56
//   3 balanced   8  2.5   12.6x  39.5  0.9925  100
//   4 streaming 16  6     28.5x  35.9  0.9823  165
//   5 fast      32  16    57.5x  32.5  0.9609  241
//   6 ultrafast 64  32    78.1x  30.4  0.9341  276
//   7 preview  128  64    93.4x  28.2  0.8915  673
// No tau 256 level: 256 exceeds the u8 range, the bound could never bind.
typedef enum {
    MC_PRESET_ARCHIVAL  = 0,   // tau   1
    MC_PRESET_MASTER    = 1,   // tau   2
    MC_PRESET_HIGH      = 2,   // tau   4
    MC_PRESET_BALANCED  = 3,   // tau   8
    MC_PRESET_STREAMING = 4,   // tau  16
    MC_PRESET_FAST      = 5,   // tau  32
    MC_PRESET_ULTRAFAST = 6,   // tau  64
    MC_PRESET_PREVIEW   = 7,   // tau 128
    MC_PRESET_COUNT     = 8,
} mc_preset;
// Sets mc_set_quality + mc_set_max_error; returns the level's quality (pass
// it to the mc_archive_append_* calls, which take q explicitly).
float       mc_apply_preset(mc_preset level);
float       mc_preset_quality(mc_preset level);
int         mc_preset_tau(mc_preset level);      // == 1 << level
const char *mc_preset_name(mc_preset level);
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
// Chunk coverage: ABSENT = never fetched (slot 0, must fetch from source);
// PRESENT = has DCT data; ZERO = fetched and decodes to all-zero (air). The
// ZERO state lets a re-run / prefetch skip air chunks instead of re-fetching.
typedef enum { MC_ABSENT = 0, MC_PRESENT = 1, MC_ZERO = 2 } mc_cover;

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

// Bytes actually written so far (the append cursor / true EOF), NOT the file's
// ftruncate'd reservation. Use for an accurate on-disk-size readout while an
// archive is being filled.
uint64_t mc_archive_data_len(mc_archive *a);

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

// ---- runtime budget control ----
// Resize the decoded-block cache budget live. DISCARDS resident blocks (a cache
// loses nothing; they re-decode on demand). Returns the installed byte budget
// (rounded to whole slots over the shards), or 0 on alloc failure (unchanged).
size_t mc_cache_resize(mc_cache *c, size_t new_bytes);
size_t mc_cache_capacity_bytes(const mc_cache *c);   // installed budget
size_t mc_cache_used_bytes(mc_cache *c);             // resident decoded bytes
double mc_cache_usage_fraction(mc_cache *c);         // used/cap in [0,1]

// ---- ready-made source bindings ----
struct mc_archive; struct mc_reader;
// Bind to a local archive handle (lock-free concurrent decode).
mc_cache *mc_cache_new_archive(size_t bytes, struct mc_archive *a);
// Bind to a reader (flat or streaming). NOTE: mc_reader decode is not
// thread-safe across threads sharing one reader; this binding serializes
// decode with an internal mutex. For maximum parallelism, give each thread
// its own reader and use mc_cache_new with a custom callback.
mc_cache *mc_cache_new_reader(size_t bytes, struct mc_reader *r);




#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// source: anything that can produce 16^3 blocks
// ---------------------------------------------------------------------------
// block() returns a pointer to the 4096-byte block at block coords
// (bz,by,bx), in z-major (z*256 + y*16 + x) layout:
//   - a stable pointer of its own (e.g. mc_cache arena), or
//   - `tmp` after filling it (sources that must copy/decode), or
//   - NULL for "no data here" (absent / pure air) — sampled as 0.
// `tmp` is 4096 bytes of sampler-owned scratch, valid until the next
// block() call on the same sampler. block() is called outside any lock the
// sampler holds; it must tolerate out-of-range block coords (return NULL).
typedef struct mc_sample_src mc_sample_src;
struct mc_sample_src {
    void *ud;                             // binding-private
    int aux, aux2;                        // binding-private (lod, flags, ...)
    const uint8_t *(*block)(const mc_sample_src *src,
                            int bz, int by, int bx, uint8_t *tmp);
    int nz, ny, nx;                       // voxel dims of the sampled level
    // Optional direct path: when set, samplers address voxels straight off
    // this base pointer (voxel (z,y,x) at dense[z*dsy + y*dsx + x]) and
    // never call block(). mc_sample_src_dense sets it; blocked sources
    // leave it NULL.
    const uint8_t *dense;
    size_t dsy, dsx;
};

// Ready-made sources:
struct mc_cache;
// mc_cache-backed (zero-copy arena pointers; decodes misses synchronously).
// `lod` selects the level; pass that level's voxel dims.
mc_sample_src mc_sample_src_cache(struct mc_cache *c, int lod,
                                  int nz, int ny, int nx);
// Dense C-order u8 array (no copies; blocks are synthesized in `tmp`).
mc_sample_src mc_sample_src_dense(const uint8_t *vox, int nz, int ny, int nx);

// ---------------------------------------------------------------------------
// sampler
// ---------------------------------------------------------------------------
typedef struct mc_sampler mc_sampler;
typedef enum { MC_FILTER_NEAREST = 0, MC_FILTER_TRILINEAR = 1 } mc_filter;

mc_sampler *mc_sampler_new(const mc_sample_src *src);
void        mc_sampler_free(mc_sampler *s);
// Drop memoized block pointers (call when the source was invalidated, e.g.
// after mc_cache_thaw/update cycles replaced chunks).
void        mc_sampler_reset(mc_sampler *s);

// One sample at (z,y,x). Out-of-bounds / NaN -> 0.
float mc_sample_point(mc_sampler *s, float z, float y, float x, mc_filter f);

// Batch: n points, zyx[i*3+{0,1,2}] = (z,y,x). Points with any coordinate
// < 0 (volume-cartographer's invalid marker) or NaN write 0.
void mc_sample_points(mc_sampler *s, const float *zyx, size_t n,
                      mc_filter f, float *out);
void mc_sample_points_u8(mc_sampler *s, const float *zyx, size_t n,
                         mc_filter f, uint8_t *out);

typedef enum {
    MC_COMP_NONE   = 0,
    MC_COMP_MIN    = 1,
    MC_COMP_MEAN   = 2,
    MC_COMP_MAX    = 3,
    MC_COMP_ALPHA  = 4,
    MC_COMP_STDDEV = 5,     // stddev of the samples along the ray: texture-
                            // energy projection. Crackle/roughness reads
                            // bright regardless of absolute density.
    MC_COMP_SHADED = 6,     // emission-absorption ray march with gradient
                            // lighting (see the shaded params below)
} mc_comp;

typedef struct {
    mc_filter filter;       // MC_FILTER_NEAREST / MC_FILTER_TRILINEAR
    mc_comp   comp;         // reduction along the normal
    float t0, t1;           // composite range along the normal, in voxels
    float dt;               // step (<= 0 -> 1.0)
    float alpha_min;        // ALPHA/SHADED: value threshold in [0,1)
    float alpha_opacity;    // ALPHA/SHADED: per-sample opacity scale (0,1]

    // -- MC_COMP_SHADED only (ignored by the other modes) --------------------
    // Front-to-back emission-absorption: per step, density d = clamped
    // (v/255 - alpha_min)/(1 - alpha_min) * alpha_opacity gives extinction
    // sigma = absorption * d, opacity a = 1 - exp(-sigma * dt) (Beer-Lambert).
    // Contributing samples are lit from the local gradient normal (central
    // differences): two-sided Lambertian diffuse + Blinn-Phong specular,
    // blended by surface-ness w = |g|^2 / (|g|^2 + grad_g0^2) so smooth
    // interior emits unshaded instead of sparkling. When shadow or sss is
    // set, a coarse secondary march toward the light accumulates optical
    // depth tau: shadow scales the direct light by exp(-tau), sss adds
    // back-lit translucency exp(-0.3*tau) ("glow through thin material").
    // All zero-value fields take the defaults noted below; a zero-initialized
    // struct renders a sane headlight relief view.
    float light[3];         // direction TOWARD the light, (z,y,x) volume
                            // coords; (0,0,0) -> headlight (along -normal).
                            // Tilt it for raking light: relief pops hardest.
    float ambient;          // ambient weight             (0 -> 0.25)
    float diffuse;          // diffuse weight             (0 -> 0.75)
    float specular;         // Blinn-Phong weight         (0 -> 0.20)
    float shininess;        // specular exponent          (0 -> 24)
    float absorption;       // extinction per unit density per voxel (0 -> 1.0)
    float shadow;           // shadow strength [0,1]      (0 = no shadow rays)
    float sss;              // translucency weight        (0 = off)
    float grad_g0;          // gradient magnitude (u8/voxel) at half
                            // surface-ness               (0 -> 8)
} mc_render_params;

// ---------------------------------------------------------------------------
// core: dense point grid -> image
// ---------------------------------------------------------------------------
// pts: W*H*3 floats, (z,y,x) per pixel. normals: W*H*3 unit (z,y,x) or NULL
// (required when comp != MC_COMP_NONE). A point with any coordinate < 0 or
// NaN renders 0 (volume-cartographer's invalid marker). out: W*H bytes.
void mc_render_points(mc_sampler *s,
                      const float *pts, const float *normals,
                      int w, int h, const mc_render_params *p, uint8_t *out);

// Parallel variant: same image, row bands across `nthreads` workers
// (0 -> one per core, capped at 16). Creates one sampler per worker over
// `src` (mc_sampler is single-threaded); src->block must be thread-safe
// (mc_cache is; a dense array trivially is).
void mc_render_points_par(const mc_sample_src *src,
                          const float *pts, const float *normals,
                          int w, int h, const mc_render_params *p,
                          uint8_t *out, int nthreads);

// ---------------------------------------------------------------------------
// plane surface (volume-cartographer PlaneSurface)
// ---------------------------------------------------------------------------
// A plane through `origin` with unit `normal`; `u` and `v` are the in-plane
// pixel axes. mc_plane_basis() builds an arbitrary stable (u,v) orthonormal
// pair from `normal` when you have no preferred orientation.
typedef struct {
    float origin[3];        // (z,y,x)
    float normal[3];        // unit (z,y,x)
    float u[3], v[3];       // unit in-plane axes: image x steps u, y steps v
} mc_plane;

void mc_plane_basis(mc_plane *pl);

// Generate the W*H point grid (and constant normals, if non-NULL) for the
// image whose pixel (i,j) sits at origin + (j - w/2)*scale*u +
// (i - h/2)*scale*v. `scale` = voxels per pixel (1 = native).
void mc_plane_gen(const mc_plane *pl, int w, int h, float scale,
                  float *pts, float *normals);

// ---------------------------------------------------------------------------
// quad surface (volume-cartographer QuadSurface)
// ---------------------------------------------------------------------------
// A control grid of gw*gh 3D points (z,y,x), row-major, VC's invalid marker
// (-1,-1,-1) honored. Rendering bilinearly interpolates the control grid to
// the output resolution and derives per-pixel normals from the grid
// tangents (du x dv, normalized) — VC's gen() contract.
typedef struct {
    const float *grid;      // gw*gh*3 (z,y,x)
    int gw, gh;
} mc_quad;

// Generate a W*H point grid (+ normals, if non-NULL) sampling the control
// grid over the rect [x0, x0+w*step) x [y0, y0+h*step) in grid units
// (step = grid cells per pixel; 1 renders the grid at native density;
// VC's render scale = 1/step). Pixels mapping outside the grid or onto
// invalid control points emit invalid (-1,-1,-1) points.
void mc_quad_gen(const mc_quad *q, float x0, float y0, float step,
                 int w, int h, float *pts, float *normals);

// ---------------------------------------------------------------------------
// one-call conveniences (gen + parallel render, scratch managed internally)
// ---------------------------------------------------------------------------
int mc_render_plane(const mc_sample_src *src, const mc_plane *pl,
                    int w, int h, float scale,
                    const mc_render_params *p, uint8_t *out, int nthreads);
int mc_render_quad(const mc_sample_src *src, const mc_quad *q,
                   float x0, float y0, float step, int w, int h,
                   const mc_render_params *p, uint8_t *out, int nthreads);

// ---------------------------------------------------------------------------
// LOD-matched rendering
// ---------------------------------------------------------------------------
// Zoomed-out views shouldn't sample the finest level: at `vox_per_pixel`
// voxels per output pixel, level floor(log2(vox_per_pixel)) carries all the
// information the image can show, with 8x fewer voxels per level. Geometry
// stays in LOD-0 voxel space; the renderer picks the level, remaps
// coordinates (half-voxel-center correct: c_L = (c_0 + 0.5)/2^L - 0.5) and
// scales the composite range so the slab covers the same physical depth,
// stepped at the sampled level's voxel pitch.
typedef struct {
    mc_sample_src lods[8];      // [0] = finest; dims halve per level
    int nlods;
} mc_sample_lods;

// ---------------------------------------------------------------------------
// 3D resampling (surface-aligned volumes)
// ---------------------------------------------------------------------------
// Composite rendering's ray walk without the reduction: keep every sample.
//
// mc_sample_quad_volume samples a w*h*nlayers u8 volume over the quad's
// parameterization — pixel (i,j) of layer k samples P(i,j) + (t0 + k*dt) *
// N(i,j), i.e. the "flattened surface volume" ink-detection models consume.
// out is layer-major: out[k*w*h + i*w + j]. Invalid surface points write 0
// through all layers.
int mc_sample_quad_volume(const mc_sample_src *src, const mc_quad *q,
                          float x0, float y0, float step, int w, int h,
                          float t0, float dt, int nlayers,
                          mc_filter f, uint8_t *out, int nthreads);

// Oriented-box resample: out voxel (k,i,j) samples origin + j*du + i*dv +
// k*dw (axes in voxels; need not be unit or orthogonal). out[k*w*h + i*w + j].
// The surface-normal-aligned ML crop primitive; with unit axes and integer
// origin it degenerates to a plain copy.
int mc_sample_box(const mc_sample_src *src,
                  const float origin[3], const float du[3],
                  const float dv[3], const float dw[3],
                  int w, int h, int d,
                  mc_filter f, uint8_t *out, int nthreads);

// floor(log2(vox_per_pixel)) clamped to [0, nlods-1]; <2 vox/px -> 0.
int mc_render_pick_lod(const mc_sample_lods *ls, float vox_per_pixel);

// Mean LOD-0 voxel spacing of one rendered pixel step across the quad's
// control grid (sparse probe; multiply by your render step).
float mc_quad_spacing(const mc_quad *q);

// As mc_render_plane / mc_render_quad, but sampling the LOD matched to
// the render scale (plane: vox/px = scale; quad: step * mc_quad_spacing).
int mc_render_plane_lod(const mc_sample_lods *ls, const mc_plane *pl,
                        int w, int h, float scale,
                        const mc_render_params *p, uint8_t *out, int nthreads);
int mc_render_quad_lod(const mc_sample_lods *ls, const mc_quad *q,
                       float x0, float y0, float step, int w, int h,
                       const mc_render_params *p, uint8_t *out, int nthreads);

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
extern "C" {
#endif

// ===========================================================================
// mc_zarr — standalone zarr reader (v3-sharded-c3d + v2-flat)
// ===========================================================================
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

// Locate one inner chunk WITHOUT fetching: fills its object key + byte range
// (off/nb) from the cached shard footer, so a caller can batch many chunks'
// ranged GETs into one request. 0 found, 1 absent/air, <0 error. v2 -> off=nb=0.
int mc_zarr_chunk_locate(mc_zarr *z, int cz, int cy, int cx,
                         char key_out[64], uint64_t *off, uint64_t *nb);

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

// ===========================================================================
// mc_s3 — s3://-backed mc_reader glue (libs3)
// ===========================================================================
typedef struct mc_s3 mc_s3;

// Open an archive at `url` (s3://bucket/key or https://...). Returns NULL on
// any failure (unreachable, not an mc archive). The handle owns the HTTP
// client and the mc_reader.
mc_s3 *mc_s3_open(const char *url);
// The reader for all decode calls (mc_chunk_offset / mc_decode_block / ...).
mc_reader *mc_s3_reader(mc_s3 *s);
void mc_s3_close(mc_s3 *s);

// ===========================================================================
// mc_volume — remote volume as a local .mca: stream/transcode/cache/prefetch
// ===========================================================================
#ifdef __cplusplus
extern "C" {
#endif

typedef struct mc_volume mc_volume;

// Open a remote (s3/https) OR local (filesystem path) NGFF multiscales zarr
// rooted at `url`/path; levels are its "0","1",... arrays. `cache_dir` holds the
// local <name>.mca; `cache_bytes` is the resident mc_cache budget; `quality` is
// the local re-encode q. Returns NULL on failure. Uses default tuning (see
// mc_volume_config); call mc_volume_open_ex to tune.
mc_volume *mc_volume_open(const char *url, const char *cache_dir,
                          size_t cache_bytes, float quality);

// Streaming-pipeline tuning. Any field left 0 takes the built-in default. These
// size open-time resources (thread pools + queues), so they apply at open only.
//   decoders      : decode-pool threads. Default nproc/2 (decode is memory-
//                   bandwidth-bound; more thrashes the bus, see commit history).
//   dl_threads    : download threads (each drains a batch -> one s3_get_batch).
//                   Default 8.
//   staging_bytes : RAM budget (bytes) for the staging queue of downloaded-but-
//                   not-yet-decoded compressed chunks. Download blocks only when
//                   this is exceeded, so the network saturates ahead of the
//                   CPU-bound decode pool. Default 2 GB. Runtime-settable via
//                   mc_volume_set_staging_bytes.
//   request_stack : depth of the LIFO download-request stack (8-byte region
//                   keys). Default 65536.
typedef struct {
    int    decoders;
    int    dl_threads;
    size_t staging_bytes;
    int    request_stack;
} mc_volume_config;

// As mc_volume_open, with explicit pipeline tuning. `cfg` may be NULL (all
// defaults). Fields set to 0 also take their default.
mc_volume *mc_volume_open_ex(const char *url, const char *cache_dir,
                             size_t cache_bytes, float quality,
                             const mc_volume_config *cfg);
void       mc_volume_free(mc_volume *v);

int  mc_volume_nlods(const mc_volume *v);
void mc_volume_shape(const mc_volume *v, int lod, int *nz, int *ny, int *nx);
// block (16^3) grid extent of a level.
void mc_volume_block_grid(const mc_volume *v, int lod, int *nz, int *ny, int *nx);

// Per-level source metadata for a client that wants to describe the volume
// without re-reading the zarr (shape in voxels z,y,x; inner_edge = the source
// inner-chunk edge / render chunk size, e.g. 256 for c3d; shard_edge = the
// storage chunk edge, e.g. 4096). dtype is always u8 and fill is 0 on the .mca
// path. Returns 0 on success, <0 if lod is out of range.
typedef struct {
    int shape[3];        // voxels (z,y,x)
    int inner_edge;      // source inner-chunk edge (render chunk), e.g. 256
    int shard_edge;      // storage chunk edge, e.g. 4096
    char codec[16];      // "c3d" | "blosc" | "raw"
} mc_volume_level_meta;
int mc_volume_get_level_meta(const mc_volume *v, int lod, mc_volume_level_meta *out);

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

// Sampling source over one level (see mc_sample.h / mc_render.h).
// blocking=0: try_block semantics — absent regions sample as 0 and kick one
//             deduped background transcode (interactive render path).
// blocking=1: absent regions are transcoded synchronously first (batch path).
mc_sample_src mc_volume_sample_src(mc_volume *v, int lod, int blocking);
// All levels at once, for LOD-matched rendering (mc_render_plane_lod /
// mc_render_quad_lod pick the level from the render scale).
mc_sample_lods mc_volume_sample_lods(mc_volume *v, int blocking);

typedef struct {
    uint64_t cache_hits, cache_misses;   // mc_cache residency
    uint64_t cache_used_blocks;          // decoded 16^3 blocks resident now
    uint64_t cache_cap_blocks;           // decoded-block capacity (budget/4096)
    uint64_t disk_bytes;                 // .mca append cursor
    uint64_t net_bytes;                  // bytes pulled from the source
    uint64_t regions_inflight;           // single-flight depth right now
} mc_volume_stats;
void mc_volume_get_stats(const mc_volume *v, mc_volume_stats *out);

// Live-resize the decoded-block RAM cache (bytes). Discards resident blocks;
// they re-decode on demand. Returns the installed budget, or 0 on failure.
size_t mc_volume_set_cache_bytes(mc_volume *v, size_t bytes);

// Live-set the staging-queue RAM budget (bytes): how far downloaded-but-not-yet-
// decoded compressed chunks may run ahead of the decode pool before download
// threads block. Bigger = network saturates further ahead of CPU-bound decode.
// Default 2 GB. Returns the installed budget.
size_t mc_volume_set_staging_bytes(mc_volume *v, size_t bytes);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
}
#endif

#endif // MATTER_COMPRESSOR_H
