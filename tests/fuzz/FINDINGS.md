# Fuzzing findings — matter-compressor decode path

> **STATUS: FIXED.** All sites below are hardened (see the `mc:` commit "harden
> decode path against malformed/untrusted archives"). `tests/mc_decode_robust_test`
> guards them (hard ctest gate + ASan/UBSan lane); a 240s AFL++ run from valid
> seeds now finds no crashes. This file is kept as the record of what was found.


Harness: `tests/fuzz/mc_fuzz_decode.c` (libFuzzer entrypoint, driven by AFL++ —
`scripts/fuzz.sh`). Built with ASan+UBSan. A ~45 s AFL++ run from valid-archive
seeds found crashes within ~1600 execs.

The README states: *"decode paths are hardened against corrupted payloads
(clamped writes, bounded offsets; fuzz-tested under ASan) — for untrusted
archives run `mc_verify` first."* These findings show that claim does not
currently hold: the reader-open surface performs **no length / offset
validation**, so a malformed or truncated `.mca` (or any `s3://` stream that
returns garbage) yields out-of-bounds reads. Notably `mc_verify_archive` — the
function the README recommends for untrusted archives — itself crashes on
untrusted input.

All sites are **out-of-bounds reads on attacker-controlled header fields**, with
no minimum-length or `len`-bounded offset checks.

| # | Function | Site | Trigger |
|---|----------|------|---------|
| 1 | `mc_metadata`       | `matter_compressor.c:2544` | buffer shorter than the 256 B header (down to 0 B) |
| 2 | `mc_open` / `reader_hdr_load` | `matter_compressor.c:2590` | buffer shorter than the header; reads `MCH_PRIOROFF`/fields past EOF |
| 3 | `mc_resolve_chunk`  | `matter_compressor.c:1423` | a LOD root offset in the header points past EOF; node-tree walk derefs `arc+node+idx*8` unchecked |
| 4 | `mc_chunk_fmaplen`  | `matter_compressor.c:1445` | chunk-blob offset/fmaplen field unchecked vs `len` |
| 5 | `mc_verify_archive` | `matter_compressor.c:2518` | walks the node tree from header offsets with `(void)len;` — never bounds-checks |

Root cause (single fix point): `mc_open()` and `mc_metadata()` accept `(arc,
len)` but never validate `len >= MC_HDR`, never range-check the per-LOD root
offsets / prior offset / metadata length against `len`, and the node-tree walk
(`mc_resolve_chunk`) and integrity walk (`mc_verify_archive`) dereference stored
offsets without bounding them to `[0, len)`.

## Reproduce

```sh
scripts/fuzz.sh 60            # AFL++ campaign, saves crashes under build_fuzz/findings
# or the deterministic regression test (constructs the malformed headers directly):
clang -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
  tests/mc_decode_robust_test.c src/matter_compressor.c src/c3d.c \
  tools/vendor/libs3/libs3.c -Isrc -Itools/vendor/libs3 -lm -lpthread -lzstd -lcurl \
  -o robust && ./robust
```

`tests/mc_decode_robust_test.c` is the permanent regression guard: it FAILS
under ASan today (tripwire), and will PASS once the reader validates its inputs.
It is wired into CI as a **known-failing (allowed)** case until the hardening
lands, then promoted to a hard gate.

## Suggested fix shape (for the source owner)

- `mc_open`/`mc_metadata`: return NULL / empty if `len < MC_HDR`.
- Validate every header offset (`MCH_ROOTOFF[0..7]`, `MCH_PRIOROFF`,
  `MCH_METAOFF+METALEN`) lies within `[MC_HDR, len]` before use.
- `mc_resolve_chunk`: bound each `node`/`childoff` to `node + MC_GRID3*8 <= len`.
- `mc_verify_archive`: use its `len` argument to bound the walk and each blob.

---

# Fuzzing findings — TIFF reader

> **MOVED.** The TIFF reader/writer now lives in the `tiff/` submodule
> (`tiff/tiff.c`, replacing the old `src/mc_tiff.c`). Its harness, seeds, and
> the regression guard moved with it: `tiff/fuzz_tiff.c`, `tiff/fuzz_tiff_seed.c`,
> and `tiff/tiff_robust_test.c` (ctest `tiff_robust`, run under ASan/UBSan).
> Run `scripts/fuzz.sh <secs> <out> tiff` (now builds against `tiff/tiff.c`) or
> the submodule's own `ctest`.

The findings below were against the original `mc_tiff_open` (mmap-based). The
same IFD-offset overflow class was re-checked and guarded in the rewritten
reader (`tiff_read`, which copies into an owned buffer rather than mmapping);
`tiff/tiff_robust_test.c` constructs the malformed files directly and runs under
ASan/UBSan.

| # | Bug | Trigger |
|---|-----|---------|
| 1 | 32-bit integer overflow in the IFD-offset bounds check | `ifd_off` near `UINT32_MAX`: `ifd_off+2 > len` computed in `uint32_t` wraps to a small value, **passes** the check, then the IFD read runs ~4 GB past the buffer → SEGV / OOB read |
| 2 | no positive-dimension check | `ImageWidth` or `ImageLength` = 0 was accepted; a consumer (`mc_surface_load_tiff`) then divides by / indexes past the zero-length view |

**Root cause:** mixed-width arithmetic in offset bounds checks (`uint32_t` file
offset `+` small constant, compared to `size_t len`, without promoting the LHS),
and validating sample/strip geometry but not the image dimensions themselves.

## The fix (carried into the new reader)

- Promote the offset before the add: `if((uint64_t)ifd + 2 > n) ...`; the
  IFD-extent and per-entry value reads already do 64-bit math.
- Reject zero/absurd dimensions and any strip set that can't cover
  `width*height*channels*type_size` bytes.

## Reproduce

```sh
scripts/fuzz.sh 60 build_fuzz_tiff tiff      # AFL++ campaign against tiff/tiff.c
# or the deterministic regression test in the submodule:
clang -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
  tiff/tiff_robust_test.c tiff/tiff.c -Itiff -o t && ./t
```

---

# Fuzzing findings — surface loaders (`mc_surface.c`)

> **STATUS: FIXED.** All four bugs are fixed in `src/mc_surface.c` and guarded by
> `tests/mc_surface_robust_test` (hard ctest gate `surface_robust`, ASan/UBSan +
> leak detection). A ~60 s libFuzzer run from valid seeds finds no crashes/leaks
> (coverage plateaus, only corpus-reducing events).

Harness: `tests/fuzz/mc_fuzz_surface.c` (libFuzzer; AFL++ via
`scripts/fuzz.sh surface`). Seeds: `tests/fuzz/mc_fuzz_surface_seed.c` (grid
OBJ, mesh OBJ, VC per-pixel map). Built with ASan+UBSan+leak detection. The
three loaders parse external Vesuvius segment files and allocate from
attacker-controlled header fields.

| # | Site | Bug | Trigger |
|---|------|-----|---------|
| 1 | `mc_mesh_load_obj` end | memory leak | a file with no `v ` lines returns `-2` but never freed the v/vn/tri buffers allocated up front (49 KB leaked / call) |
| 2 | `mc_mesh_load_obj` face loop | OOB triangle index | `f 1 2 999999` (or negative-relative / `0`) stored an index outside `[0,nv)` into `tri` — an OOB read for any consumer of `m->tri` |
| 3 | `mc_surface_load_obj` | dimension-product overflow | `# grid 2000000000 2000000000` → `gw*gh*3*sizeof(float)` overflows `size_t` into an under-alloc the fill loop writes past |
| 4 | `mc_surface_load_vcps_ppm` | dimension overflow | huge `width:`/`height:` header → same under-alloc class on the `W*H*3` grid |

## The fix

- bug 1: free `v`/`vn`/`tri` (and zero `*m`) on the `nv<=0` failure path.
- bug 2: validate each face index to `[1,nv]` (1-based; negative-relative
  resolved first) and drop out-of-range refs before emitting a triangle. Also
  made all three `realloc` sites checked (NULL → free + bail, was a latent
  dangling-pointer write on OOM).
- bugs 3,4: reject non-positive / absurd dims (`> 1<<20` per axis; `dim > 4096`).

## Reproduce

```sh
scripts/fuzz.sh 60 build_fuzz_surface surface   # AFL++ (seeds + crash reproducers)
# or the deterministic regression test:
clang -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
  tests/mc_surface_robust_test.c src/mc_surface.c tiff/tiff.c -Isrc -Itiff -lm \
  -o t && ASAN_OPTIONS=detect_leaks=1 ./t
```

Crash reproducers: `tests/fuzz/crashes/surface/`. `tests/mc_surface_robust_test.c`
is the permanent guard — it leaks + fails the tri-index assertion on the pre-fix
loaders (verified tripwire) and passes clean on the fix.

---

# Fuzzing findings — segment detector (`mc_segment.c`)

> **STATUS: CLEAN + hardened.** A ~2 min libFuzzer+ASan/UBSan campaign over
> adversarial dims/data/params found no crashes/leaks/UB (coverage plateaus at
> the detector's full extent; mc_segment.c is at 100% line coverage). One latent
> robustness gap the harness masked — `mc_seg_detect` had no `nz/ny/nx<1` guard
> (the harness clamps dims to 1..24) — was closed defensively anyway.

Harness: `tests/fuzz/mc_fuzz_segment.c` (libFuzzer; AFL++ via
`scripts/fuzz.sh segment`). Seeds: `tests/fuzz/mc_fuzz_segment_seed.c`. The
detector + topology post-proc are pure in-memory compute (separable gaussian
blur, central-difference gradient + structure tensor, closed-form eigenvalues,
26-conn flood label, 2x2x2 hole LUT, spherical morphology, border flood) — lots
of index arithmetic that must stay in bounds for any shape/data. The harness
derives small bounded dims + params from the header bytes, allocates buffers
sized to match, fills the volume from the rest, and runs detect + every
post-proc op + label.

## Hardening

- `mc_seg_detect`: reject `nz<1||ny<1||nx<1` up front. The gradient stencil
  indexes neighbors, so a 0-length axis would index OOB if a caller passed
  degenerate dims directly. Guarded + regression-tested (`mc_segment_test`).

The other ops (`remove_small`/`plug_holes`/`close`/`fill_cavities`/`label`)
compute `n=nz*ny*nx` and loop `i<n`, so they no-op safely on a 0-voxel volume.
