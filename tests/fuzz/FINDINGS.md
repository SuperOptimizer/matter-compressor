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

# Fuzzing findings — TIFF reader (`mc_tiff_open`)

> **STATUS: FIXED.** Both bugs are fixed in `src/mc_tiff.c` and guarded by
> `tests/mc_tiff_robust_test` (hard ctest gate `tiff_robust`, run under
> ASan/UBSan). A fresh ~45 s libFuzzer run from valid seeds finds no crashes
> (coverage plateaus, only corpus-reducing events).

Harness: `tests/fuzz/mc_fuzz_tiff.c` (libFuzzer entrypoint; also driven by
AFL++ via `scripts/fuzz.sh tiff`). Seeds: `tests/fuzz/mc_fuzz_tiff_seed.c`
(one valid TIFF per supported sample type/count). Built with ASan+UBSan.

`mc_tiff_open` parses an **untrusted file**: it mmaps the bytes, walks the IFD,
and on success hands back `out->pixels` as a raw pointer INTO the mmap that
callers cast to typed structs (`const float (*)[4]`, etc.). The contract: for
ANY bytes it must reject cleanly or return a view fully inside the mapping. A
~45 s libFuzzer run from valid seeds found two violations within ~1600 execs.

| # | Site | Bug | Trigger |
|---|------|-----|---------|
| 1 | `mc_tiff.c:67,69` | 32-bit integer overflow in the IFD-offset bounds check | `ifd_off` near `UINT32_MAX`: `ifd_off+2 > len` is computed in `uint32_t`, wraps to a small value, **passes** the check, then `rd16(b+ifd_off)` reads ~4 GB past the mmap base → SEGV / OOB read |
| 2 | `mc_tiff.c:85` | no positive-dimension check | `ImageWidth` or `ImageLength` = 0 was accepted; the reader returns a zero-length-strip view a consumer (`mc_surface_load_tiff`) divides by / indexes past |

**Root cause:** mixed-width arithmetic in offset bounds checks (`uint32_t` file
offset `+` small constant, compared to `size_t len`, without promoting the LHS),
and validating sample/strip geometry but not the image dimensions themselves.

## The fix

- Promote the offset before the add: `if((size_t)ifd_off+2 > len) ...` (and the
  same for the `ifd_off + 2 + n*12 + 4` IFD-extent check). The other offset
  checks (`(size_t)off+2`, `(size_t)soff + pixbytes`) already cast the LHS.
- Reject non-positive / absurd dimensions: `if(w==0||h==0||w>(1<<20)||h>(1<<20)) goto done;`.

## Reproduce

```sh
scripts/fuzz.sh 60 build_fuzz_tiff tiff      # AFL++ campaign (seeds + crash reproducers)
# or the deterministic regression test (constructs both malformed files directly):
clang -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
  tests/mc_tiff_robust_test.c src/mc_tiff.c -Isrc -o t && ./t
```

Crash reproducers: `tests/fuzz/crashes/tiff/{ifd_off_overflow,zero_width}.tif`.
`tests/mc_tiff_robust_test.c` is the permanent guard — it SEGVs under ASan on
the pre-fix reader (verified tripwire) and passes on the fixed one.
