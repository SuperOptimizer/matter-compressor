# mc_volume / mc_zarr — design

Goal: move the remote-streaming, source-format, caching, single-flight and
prefetch logic out of VC3D and into matter-compressor, so VC3D's chunk cache
becomes a thin shim. Give mc_volume a URL; it streams, transcodes into a local
`.mca`, caches, prefetches, and serves decoded blocks.

Style: C23, matter-compressor house style (single .c/.h amalgam, snake_case,
explicit ownership with matching `*_free`, no hidden globals beyond the existing
curl/credential caches). libs3 is already vendored and used by tools.

## Layers (bottom to top)

```
  mc_zarr    standalone zarr-v3 reader: parse zarr.json, sharded reads,
             inner-chunk extraction. Transport-agnostic (byte-source callback).
  mc_volume  S3/https streaming + region single-flight + prefetch over a local
             .mca archive (mc_archive + mc_cache). Source = mc_zarr (or another
             .mca for the .mca->.mca mirror path).
  (existing) mc_archive, mc_cache, mc_codec — unchanged.
```

## mc_zarr — standalone zarr-v3 reader (rewrite from scratch)

The current `mc_fetch.c` scraper is v2-only and crude. Rewrite for the format
the c3d volumes actually use: **zarr v3, `sharding_indexed`,
`index_location:start`, inner codec `c3d` (opaque) or blosc.**

Parse one level's `zarr.json`:
- `chunk_grid.configuration.chunk_shape`  -> shard shape (e.g. 4096^3)
- `codecs[sharding_indexed].configuration.chunk_shape` -> inner chunk (256^3)
- `...codecs[]` -> inner codec name ("c3d" or "blosc"/"bytes")
- `...index_codecs`, `index_location` (always "start" for these)
- `shape`, `data_type` (uint8), `fill_value`

Byte source is a callback (so transport stays in mc_volume):
```c
typedef int (*mc_zarr_read_fn)(void *ud, const char *key,
                               uint64_t off, uint64_t len,  // len 0 = whole object
                               uint8_t **out, size_t *out_len);
```

API:
```c
typedef struct mc_zarr mc_zarr;
mc_zarr *mc_zarr_open(mc_zarr_read_fn read, void *ud, const char *level_json);
void     mc_zarr_free(mc_zarr *z);

// geometry
void mc_zarr_shape(mc_zarr*, int *nz,int *ny,int *nx);
int  mc_zarr_inner_edge(mc_zarr*);           // 256
int  mc_zarr_shard_edge(mc_zarr*);           // 4096
const char *mc_zarr_inner_codec(mc_zarr*);   // "c3d" / "blosc" / "raw"

// shard index probe: is the shard containing inner-chunk (cz,cy,cx) all-air?
// reads only the index footer (one ranged read). -1 unknown, 0 no, 1 yes.
int mc_zarr_shard_all_air(mc_zarr*, int cz,int cy,int cx);

// fetch one shard (one ranged GET of the whole object) and hand each PRESENT
// inner chunk to sink as its RAW (still-compressed) bytes + codec name. Absent
// inner chunks are skipped (air). Returns 0 ok, <0 transient error.
typedef void (*mc_zarr_chunk_fn)(void *ud, int cz,int cy,int cx,
                                 const uint8_t *raw, size_t raw_len);
int mc_zarr_read_shard(mc_zarr*, int cz,int cy,int cx, mc_zarr_chunk_fn, void *ud);

// fetch + extract a single inner chunk's raw bytes (interactive cold path).
int mc_zarr_read_inner(mc_zarr*, int cz,int cy,int cx, uint8_t **raw, size_t *len);
```

mc_zarr does NOT decode c3d/blosc — it returns raw compressed inner-chunk bytes.
Decoding is the inner-codec's job (see the c3d question below).

## mc_volume — streaming + cache + prefetch (rewrite the VC MatterCache logic)

Owns: a local `.mca` (mc_archive + mc_cache) as the transcode cache, an mc_zarr
source, the region single-flight, and the prefetch driver. This replaces
`core/src/render/MatterCache.cpp` (757 lines) and most of `ChunkCache.cpp`.

```c
typedef struct mc_volume mc_volume;

// Open a remote volume: `url` is an s3://.../zarr root or a remote .mca. Levels
// auto-discovered (NGFF multiscales or .mca header). `cache_dir` holds the local
// volume.mca; `cache_bytes` is the mc_cache resident budget. The inner-codec
// decode hook transcodes a raw source chunk to a dense 256^3 u8 (see below).
mc_volume *mc_volume_open(const char *url, const char *cache_dir,
                          size_t cache_bytes, mc_decode_fn decode, void *decode_ud);
void       mc_volume_free(mc_volume *v);

int   mc_volume_nlods(mc_volume*);
void  mc_volume_shape(mc_volume*, int lod, int *nz,int *ny,int *nx);

// Serve one 16^3 block. present -> decode from mc_cache (sync, fast). absent ->
// kick a deduped async region fetch (transcode into the .mca) and return 0
// (caller renders a coarser LOD). Returns 1 if served, 0 if absent (queued).
int mc_volume_try_block(mc_volume*, int lod, int z,int y,int x, uint8_t *dst4096);

// Blocking variant for batch/CLI: fetch the region if absent, then decode.
int mc_volume_get_block(mc_volume*, int lod, int z,int y,int x, uint8_t *dst4096);

// Prefetch a whole shard (parallel download + parallel transcode), or a level.
void mc_volume_prefetch_shard(mc_volume*, int lod, int cz,int cy,int cx);
void mc_volume_prefetch_level(mc_volume*, int lod, int nthreads, volatile int *cancel);

// stats: residency comes from mc_cache; net rate is tracked here.
typedef struct { uint64_t cache_hits, cache_misses, disk_bytes,
                 net_bytes, regions_inflight; } mc_volume_stats;
void mc_volume_get_stats(mc_volume*, mc_volume_stats*);
```

Internals (all moved from VC):
- **single-flight**: an in-flight region set + mutex/cv. Self-bounding (only
  regions being transcoded now); mc_archive coverage (PRESENT/ZERO/ABSENT) is
  the source of truth for "done". Air regions marked ZERO.
- **transcode**: on a cold region, `mc_zarr_read_inner` (or assemble from a
  shard) -> `decode` hook to dense 256^3 -> `mc_archive_append_chunk_raw`.
- **prefetch**: shard-at-a-time, `s3_get_parallel` whole-object download, the
  shard's inner chunks transcoded across a bounded thread team. Air shards
  skipped via `mc_zarr_shard_all_air`.
- **.mca -> .mca mirror**: when the source URL is a remote .mca, the source is
  a streaming mc_reader and chunks are appended verbatim (no transcode).

## The c3d codec: DECIDED — vendor c3d (decode-only) into matter-compressor

Inner codec is `c3d`. mc_volume's transcode needs a raw-c3d -> dense-256^3 step.
DECISION: **vendor `libs/c3d` (c3d.c/c3d.h, ~5.4k lines) into matter-compressor**
as `src/c3d.c` + `src/c3d.h`. mc_volume is then self-contained: URL in, decoded
blocks out, no codec callback. Only the DECODE path is used here (the encode
side already lives in matter_compressor's own DCT codec — c3d is the SOURCE
format's decoder, not our re-encode). The `mc_decode_fn` hook is dropped;
mc_volume calls `c3d_decoder_chunk_decode` directly via a per-worker decoder.

c3d decode API used: `c3d_decoder_new`/`_free` (one per worker, NOT thread-safe),
`c3d_decoder_chunk_decode(dec, in, in_len, out256)` -> 256^3 u8. The source's
inner codec name is checked against "c3d"; blosc/raw handled separately if needed.

## VC3D after the move (the shim)

- `ChunkCache` (mca mode) -> wraps one `mc_volume`. `tryGetChunk` =
  `mc_volume_try_block`; `getChunkBlocking` = `mc_volume_get_block`; stats =
  `mc_volume_get_stats` + the existing net counter. No entry table, no LRU, no
  single-flight, no listener plumbing. Legacy non-uint8 path stays as-is.
- DELETE `core/src/render/MatterCache.cpp` + `.hpp` (mc_volume replaces it).
- `ZarrChunkFetcher` / `utils/zarr.cpp` sharded-read paths: removable for the
  remote mca path (mc_zarr owns it). Local zarr write + non-mca reads stay.
- `VolumePrefetcher` -> calls `mc_volume_prefetch_level`.
- `vc_cache_prefetch` -> thin wrapper over `mc_volume_prefetch_level`.
- libs3 stays vendored in BOTH (already is); the HttpClient adapter can shrink.

## Migration order (incremental, each step builds+tests)

1. [DONE] mc_zarr (fs + S3 tested against paris4_c3d_r50.zarr; footer parse
   byte-matches python; extracted chunk byte-identical to aws range-extract).
2. [DONE] mc_volume over mc_zarr + vendored c3d. Validated on S3: 6-LOD
   discovery, get_block transcodes (mean 126, full chunk), re-get is cache-only
   (0 net), async try_block (absent->0, worker fills, poll->data), profile creds
   via s3_credentials_load(NULL). CMake: matter_compressor_volume lib + tests.
3. VC ChunkCache mca-mode -> mc_volume shim; delete MatterCache.
4. Re-point VolumePrefetcher + vc_cache_prefetch.
5. Prune now-dead VC zarr/fetch code.

### Notes for the VC port (steps 3-5)
- c3d C3D_ALIGN=32 is on the 256^3 OUTPUT buffer; mc_volume posix_memaligns it.
- mc_volume owns 4 background transcode workers; the VC shim must NOT add its own.
- block coords are 16^3 units; region = block/16. try_block(lod,bz,by,bx) maps to
  VC ChunkKey directly (level, iz, iy, ix in 16^3-block coords).
- stats: cache_hits/misses/disk_bytes from mc_cache+archive, net_bytes counted in
  the s3 read cb. VC status bar reads mc_volume_get_stats.

Risk: steps 3-5 are a big VC diff; do them only once mc_volume is proven in
isolation (step 2 bench). Keep the legacy non-uint8 ChunkCache path untouched.
