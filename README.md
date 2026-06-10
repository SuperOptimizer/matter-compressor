# matter-compressor

A small, fast lossy codec + on-disk archive for dense 3D `u8` scalar volumes
(e.g. masked micro-CT). Two layers, cleanly separated:

- **codec** (`src/mc_codec.{c,h}`) — compresses a 16³ voxel block: integer
  separable DCT-16 + dead-zone quant + a CABAC-style binary range coder with
  trained context priors. Air voxels (value 0) are handled mask-aware: SOR
  air-filled before the DCT and force-zeroed on decode. Every block payload is
  fully SELF-CONTAINED (its own two-level air mask, dc, flags — all inside one
  range-coded stream), so a single 16³ block decodes with no chunk-level state.
  Pure transform — no I/O.
- **archive** (`src/mc_archive.{c,h}` + `mc_archive_read.h`) — the on-disk format:
  a sparse multi-level node tree of dense 256³ chunks, 8 independently
  fetchable/decodable LODs. Source-agnostic: the builder pulls voxels through a
  caller callback. One layout serves both uses: streaming random access
  (partial-fetch block decode, node-table caching, clustered index after
  `mc_export`) and full offline download (a single mmap-able file).
- **cache** (`src/mc_cache.{c,h}`) — in-RAM decoded-block cache for interactive
  clients (vc3d-style renderers): 4KB blocks in an mmap arena, 64-way sharded
  hash + CLOCK/NRU eviction, multi-thread safe, chunk prefetch API. ~90M
  cache-hit gets/s/thread.

Runtime parameters: **quality** (the quant base step; higher = smaller +
lossier) and an optional **max-error bound** (`mc_set_max_error(tau)`: sparse
corrections guarantee |error| ≤ tau on every material voxel; τ≈3–4×q is nearly
free). Suggested operating points on masked scroll data: q≈1 near-lossless
(~9×), q≈6 general default (~34×), q≈12 aggressive (~51×). An optional
decode-side deblocking filter (`mc_deblock`) adds ~0.3 dB at high q for free.

See `bench/RESULTS.md` for measured ratio/quality/throughput/latency and the
experiment log (including measured-and-rejected ideas).

## File layout

```
[0,   256)    header (magic, version, dims, per-LOD roots, metadata fields)
[256, 128KB)  user metadata region — free-form text (JSON/TOML/...), zero-padded
[128KB, ...)  archive data (sparse node tree + dense chunks)
```

Chunk blob (format v3): `[512B block-bitmap][present-block u16 lens][block
payloads]` — one range-GET fetches a chunk; one bitmap+lens read locates a
block; the block payload alone decodes it.

## Build

```sh
cmake -B build -S . && cmake --build build
ctest --test-dir build          # round-trip tests
```

No external dependencies beyond libm (tools: libcurl + libzstd for `mc_fetch`).

## Tools

- `mc_fetch` — pull a sub-volume out of a Vesuvius Challenge zarr (s3:// via
  vendored libs3 with SigV4/anonymous, or plain https), blosc-zstd or raw
  chunks, writes a raw u8 cube.
- `mc_mask` — fysics-style aggressive interior air masking (box-smooth scratch
  → histogram valley → cut), for volumes where only the outside ROI is masked.
- `mc_bench` — the metric basket: ratio, PSNR, MAE, p50/p90/p95/p99, max
  error, SSIM, encode/decode MB/s, cold-block latency. Crops 128/256/512 test
  cubes from a fetched volume.
- `mc_train` — retrain the range-coder context priors on a volume; prints
  tables to paste into `mc_rangecoder.h`.
- `mc_export` — repack an archive (or a chunk box) verbatim into a fresh file:
  Morton-ordered chunks, the whole index clustered right after the metadata
  region (one ranged GET for streaming clients), no append slack.
- `mc_vs_c3d`, `mc_rans_probe`, `mc_prof` — comparison/measurement harnesses.

## API

```c
#include "mc_archive_api.h"

// build: provide a voxel source callback (x,y,z -> u8; 0 = air)
mc_u8 src(void *ud, int x,int y,int z){ ... }
mc_build_opts opt = { .dim=1024, .quality=6.0f, .metadata=json, .meta_len=n };
mc_build_to_file(src, ud, &opt, "out.mc");

// decode: random-access by LOD / chunk / block
mc_reader *r = mc_open(arc, len); mc_reader_set_quality(r, 6.0f);
uint64_t co = mc_chunk_offset(r, /*lod*/0, cz,cy,cx);
mc_u8 blk[16*16*16]; mc_decode_block(r, co, bz,by,bx, blk);
```

To compress a zarr volume (local or S3), write an exporter tool that implements
the source callback over your storage and calls `mc_build*` (or the appendable
`mc_archive_*` writer for incremental/parallel builds).
