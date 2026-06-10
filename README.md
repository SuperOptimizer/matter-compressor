# matter-compressor

A fast lossy codec + on-disk archive + in-RAM cache for dense 3D `u8` scalar
volumes — purpose-built for masked micro-CT at Vesuvius-Challenge scale
(100 TB+), consumed by interactive viewers (VC3D) and ML dataloaders.

Three layers, one header (`src/matter_compressor.h`), plain C, no
dependencies beyond libm/pthreads:

- **codec** — 16³ blocks: integer separable DCT-16, tuned dead-zone quant
  with band-weighted steps, adaptive binary range coder with trained
  (per-volume overridable) context priors. Air voxels (value 0) are
  mask-aware: SOR-filled before the transform, exactly zero after decode.
  Every block payload is fully self-contained — one block decodes alone in
  ~9 µs with zero chunk-level state.
- **archive** — appendable, crash-safe mmap file: dense node tree of 256³
  chunks, 8 independently coded LODs (a hard guarantee: no cross-LOD
  dependency), per-axis dims (padded to 256), per-chunk quality + xxh64
  integrity + material-fraction maps. The same layout serves streaming
  random access (partial fetch, node-table caching, clustered-index export)
  and offline use (mmap and go).
- **cache** — decoded-block cache for render/ML clients: 4 KB blocks in an
  mmap arena, 64-way sharded, S3-FIFO eviction (CLOCK optional), batch
  update/resolve, async tickets with cancel, and an optional tick-phase mode
  (freeze/thaw) where render-phase reads are fully lock-free.
- **sample + render** (`mc_sample.h`, `mc_render.h`) — volume-cartographer-
  style surface rendering: generic point-grid renderer with plane and quad
  surface generators, nearest/trilinear filtering, min/mean/max/alpha
  compositing along surface normals, and LOD-matched rendering: the
  _lod render variants sample the level the zoom can actually show
  (half-voxel-correct remap; ~2-3x per level on real data, sub-ms
  zoomed-out frames). Sources are pluggable (mc_cache,
  mc_volume, dense arrays). Fully in-volume 1024² quad-surface render
  (8 threads, dense source): slice 1.5 ms, 9-step trilinear composite
  8.4 ms including surface generation.

## Performance (Apple M-series reference, real 2.4 µm scroll data)

| q | ratio | PSNR | encode | decode | cold block |
|---|---:|---:|---:|---:|---:|
| 1 | 8.8× | 43.7 dB | 108 MB/s | 134 MB/s | 0.023 ms |
| 3 | 20.6× | 38.5 dB | 135 MB/s | 257 MB/s | 0.013 ms |
| 6 | 33.7× | 35.8 dB | 146 MB/s | 364 MB/s | 0.009 ms |
| 12 | 51.1× | 33.5 dB | 154 MB/s | 471 MB/s | 0.007 ms |

Single-threaded. Parallel whole-chunk helpers: **~640 MB/s encode,
~1.5–2 GB/s decode** per process, scaling with cores (blocks are
independent). PGO+ThinLTO builds add ~5–7 % decode. Cache hits:
~150 M zero-copy gets/s/thread. At iso-rate the codec leads c3d by
1.4–3.7 dB PSNR across the curve (`tools/mc_vs_c3d`). NEON and AVX2 builds
produce byte-identical archives. Full measurement history, including every
idea that was implemented and rejected: `bench/RESULTS.md`.

## Quality controls

- **quality q** — the quant step dial (see table above).
- **max-error bound** — `mc_set_max_error(tau)`: sparse corrections
  guarantee |error| ≤ τ on every material voxel (τ≈3–4×q costs <1 % ratio).
- **rate targeting** — `mc_archive_append_chunk_target(.., target_ratio,..)`
  picks q per chunk from a 1/16-block sample (+~6 % encode time; lands
  within ~5 % of target on real data).
- **deblock** — `mc_deblock(...)`: optional decode-side filter, +0.3 dB at
  high q for free.

## API tour

```c
#include "matter_compressor.h"

// ---- write (appendable, crash-safe, parallel) ----
mc_archive *a = mc_archive_open_dims("scroll.mc", nx, ny, nz, /*q*/6.0f);
mc_archive_append_chunk_par(a, /*lod*/0, cz, cy, cx, vox256, 6.0f, /*threads*/0);
mc_archive_append_chunk_target(a, 0, cz, cy, cx, vox256, /*ratio*/30.0f, NULL);

// ---- read: blocks, chunks, regions ----
uint64_t co = mc_archive_chunk_offset(a, 0, cz, cy, cx);
mc_archive_decode_block(a, co, bz, by, bx, blk16);          // 16^3, ~9 us
mc_archive_decode_chunk(a, co, out256, 0);                  // parallel, ~GB/s
mc_archive_read_region(a, 0, z0,y0,x0, dz,dy,dx,            // any box ->
                       out, sz, sy, 0);                     // strided buffer

// ---- ML dataloading ----
mc_box boxes[64];
int n = mc_archive_sample_boxes(a, 0, seed, 64, 128,128,128,
                                /*min material*/0.5f, boxes);  // deterministic
mc_archive_read_regions(a, 0, boxes, n, 128,128,128,
                        batch, crop_stride, 0);                // one batch fill
float f = mc_archive_block_fraction(a, 0, bz, by, bx);         // no decode

// ---- interactive cache (VC3D) ----
mc_cache *c = mc_cache_new_archive(/*bytes*/8ull<<30, a);
mc_cache_update(c, ids, n, 0);                 // blocking batch fill
const mc_u8 *p = mc_cache_get(c, 0, bz,by,bx); // zero-copy hit
int lod = mc_cache_best_lod(c, 0, bz,by,bx);   // render-now-refine-later

// tick-phase mode (game-loop pipelines) — optional; without freeze() the
// cache is the plain always-thread-safe multi-reader/multi-writer cache:
mc_cache_thaw(c);
size_t nm = mc_cache_misses_drain(c, missed, cap);   // last frame's feedback
mc_cache_resolve(c, ids, n, ptrs, 0);                // pin + pointer table
mc_cache_freeze(c);                                  // reads now LOCK-FREE
// render: ptrs[i] directly; stragglers via get() -> NULL -> best_lod()

// ---- 3D resampling (surface volumes, oriented ML crops) ----
mc_sample_quad_volume(&src, &q, x0,y0, 1.0f, w,h,      // w*h*nlayers u8:
                      -10.0f, 1.0f, 21, MC_FILTER_TRILINEAR, // the flattened
                      svol, 0);                        // ink-detection input
mc_sample_box(&src, origin, du,dv,dw, w,h,d,           // oriented crop
              MC_FILTER_TRILINEAR, crop, 0);

// ---- surface rendering (VC3D-style) ----
mc_sample_src src = mc_sample_src_cache(c, /*lod*/0, nz, ny, nx);
mc_plane pl = {.origin={z,y,x}, .normal={nz_,ny_,nx_}};
mc_plane_basis(&pl);
mc_render_params rp = {.filter=MC_FILTER_TRILINEAR, .comp=MC_COMP_MAX,
                       .t0=-4, .t1=4, .dt=1};
mc_render_plane(&src, &pl, 1024, 1024, 1.0f, &rp, img, 0);
mc_quad q = {.grid = vc_points_zyx, .gw=gw, .gh=gh};   // VC quad mesh
mc_render_quad(&src, &q, x0, y0, /*step*/1.0f, w, h, &rp, img, 0);

// ---- streaming (S3-style byte sources) ----
mc_reader *r = mc_open_streaming(read_cb, ud, total_len);
mc_reader_set_partial_fetch(r, 1);   // one block = header+payload bytes only

// ---- integrity / per-volume tuning ----
mc_verify_archive(arc, len, 1);                       // xxh64 every chunk
mc_archive_set_priors(a, plo, phi);                   // mc_train output
```

## File layout (format v7)

```
[0,   256)    header (magic, version, NX/NY/NZ, per-LOD roots, quality,
              priors offset, metadata fields)
[256, 128KB)  user metadata region (free-form, zero-padded)
[128KB, ...)  node tables + chunk blobs (append-at-EOF, commit-word ordering:
              the file is valid and decodable after every appended chunk)
```

Chunk blob: `[f32 q][u64 xxh64][u16 fmaplen][fraction map][512B block
bitmap][present-block u16 lens][self-contained block payloads]`. One ranged
GET fetches a chunk; bitmap+lens+one payload fetches a single block.
`tools/mc_export` repacks archives Morton-ordered with the whole index and
coarsest LODs clustered at the front (one-GET viewer startup) and supports
chunk-box sub-volume export.

## Build

```sh
cmake -B build -S . && cmake --build build && ctest --test-dir build
```

Clang strongly recommended (GCC measured −28 % encode). For fleet builds,
ThinLTO+PGO adds ~5–7 % decode — **never enable LTO without profiles** (~15 % slower) and **never add
-ffast-math** (breaks cross-ISA bitstream identity, zero speedup — hot paths
are integer):

```sh
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=cmake/homebrew-llvm.cmake -DMC_PGO_GEN=ON
cmake --build build
LLVM_PROFILE_FILE=prof/%p.profraw build/mc_bench <vol.bin> 512 512 1,3,6,12
llvm-profdata merge -output=merged.profdata prof/*.profraw
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=cmake/homebrew-llvm.cmake \
      -DMC_THINLTO=ON -DMC_PGO_USE=$PWD/merged.profdata
cmake --build build
```

SIMD: NEON on ARM (Graviton, Apple M, X1 Elite), AVX2 at the x86-64-v3
baseline, AVX-512 opt-in (`-DMC_ENABLE_AVX512` + `-march=x86-64-v4`).
The sampler/renderer ships NEON + SSE4.1 4-wide and AVX2 8-wide kernels
(8-wide for adjacent-pixel slices only — measured 1.6x slower for
z-strided ray composites on Zen 3); AVX-512 adds nothing for this fleet
(consumer Intel has it fused off, Zen 4 double-pumps 256-bit), and SVE
only exceeds NEON width on Graviton 3 — both skipped on measurement.
Vendoring via `add_subdirectory` keeps full optimization flags.

## Tools

- `mc_fetch` — pull sub-volumes from Vesuvius zarrs (`s3://` via vendored
  libs3, anonymous SigV4, or https), blosc-zstd or raw chunks → raw `u8`.
- `mc_mask` — fysics-style aggressive interior air masking (histogram-valley
  cut); the AWS 2.4 µm volumes are ROI-masked only, run this first.
- `mc_bench` — the metric basket: ratio, PSNR, MAE, p50/90/95/99, max error,
  SSIM, throughputs, cold-block latency, at 128/256/512 crops.
- `mc_export` — repack/sub-volume export (verbatim, no re-encode).
- `mc_verify` — xxh64 integrity walk of every chunk.
- `mc_train` — retrain context priors on a volume (paste tables or store
  per-volume via `mc_archive_set_priors`).
- `mc_vs_c3d`, `mc_prof` — comparison and profiling harnesses.

## Data pipeline (Vesuvius)

```sh
build/mc_fetch "s3://vesuvius-challenge-open-data/PHercParis4/volumes/<id>.zarr/0" \
    z0 y0 x0 512 /tmp/vol.bin
build/mc_mask /tmp/vol.bin 512 /tmp/masked.bin 1.0
build/mc_bench /tmp/masked.bin 512 512 1,3,6,12
```

u8 only, by design. Robustness: decode paths are hardened against corrupted
payloads (clamped writes, bounded offsets; fuzz-tested under ASan) — for
untrusted archives run `mc_verify` first.
