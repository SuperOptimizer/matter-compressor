# matter-compressor

A small, fast lossy codec + on-disk archive for dense 3D `u8` scalar volumes
(e.g. masked micro-CT). Two layers, cleanly separated:

- **codec** (`src/mc_codec.{c,h}`) — compresses a 16³ voxel block: integer
  separable DCT-16 + dead-zone quant + a CABAC-style binary range coder. Air
  voxels (value 0) are handled mask-aware: harmonically filled before the DCT and
  force-zeroed on decode; the air boundary is coded once per 256³ chunk as a
  coherent surface. Pure transform — no I/O.
- **archive** (`src/mc_archive.{c,h}` + `mc_archive_read.h`) — the on-disk format:
  a sparse multi-level node tree of dense 256³ chunks, 8 independently
  fetchable/decodable LODs. Source-agnostic: the builder pulls voxels through a
  caller callback (no zarr/S3 here — that belongs in an exporter tool).

The only runtime parameter is **quality** (the quant base step; higher = smaller +
lossier). Suggested operating points on clean dense data: q≈1 ≈ near-lossless,
q≈6 a good general default, q≈12+ for aggressive/preview.

## File layout

```
[0,   256)    header (magic, version, dims, per-LOD roots, metadata fields)
[256, 128KB)  user metadata region — free-form text (JSON/TOML/...), zero-padded
[128KB, ...)  archive data (sparse node tree + dense chunks)
```

## Build

```sh
cmake -B build -S . && cmake --build build
ctest --test-dir build          # round-trip test
```

No external dependencies beyond libm.

## API

```c
#include "mc_archive_api.h"

// build: provide a voxel source callback (x,y,z -> u8; 0 = air)
mc_u8 src(void *ud, int x,int y,int z){ ... }
mc_build_opts opt = { .dim=1024, .quality=8.0f, .metadata=json, .meta_len=n };
mc_build_to_file(src, ud, &opt, "out.mc");

// decode: random-access by LOD / chunk / block
mc_reader *r = mc_open(arc, len); mc_reader_set_quality(r, 8.0f);
uint64_t co = mc_chunk_offset(r, /*lod*/0, cz,cy,cx);
mc_u8 blk[16*16*16]; mc_decode_block(r, co, bz,by,bx, blk);
```

To compress a zarr volume (local or S3), write an exporter tool that implements
the source callback over your storage and calls `mc_build*`.
