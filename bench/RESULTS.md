# matter-compressor improvement campaign — results log

Test data: PHerc Paris 4, 2.4 µm 137 keV masked zarr from the AWS open-data
bucket (`s3://vesuvius-challenge-open-data`, fetched with `tools/mc_fetch`,
512³ crop at scroll center), interior air zeroed with `tools/mc_mask`
(fysics-style aggressive valley cut → 41.9 % air). Cross-checks on the 7.91 µm
Scroll1A standardized volume and synthetic sheets. Apple Silicon, 1 thread.

## Headline (512³, q=6, primary dataset)

| metric            | baseline (1b061ec) | final v3.3 | change |
|-------------------|-------------------:|-----------:|-------:|
| compression ratio | 27.6×              | 33.7×      | +22 %  |
| PSNR              | 35.66 dB           | 35.78 dB   | +0.12 dB |
| max error         | 61                 | 58 (24 with τ=24 at −0.6 % ratio) | bounded on demand |
| encode            | 79 MB/s            | 112 MB/s   | +42 %  |
| decode (warm)     | 162 MB/s           | 233 MB/s   | +44 %  |
| cold-block decode | ~71 **ms**         | 0.015 ms   | ~4700× |

At q=12: ratio 40.4→51.4× (+27 %). At q=1 (near-lossless): 8.0→8.7×.
7.91 µm cross-check at q=6: 11.7→13.3× with +0.67 dB PSNR.
Optional decode-side deblock: +0.28 dB PSNR / +0.002 SSIM at q=12, zero rate.

## What changed (in order, each round-trip tested + benched)

1. **Format v2 — self-contained blocks.** The monolithic 256³ chunk air mask
   (decoded in full — ~70 ms — before ANY single block could decode) was
   replaced by a per-block mask carried inside each block payload. Cold-block
   latency ~2000×, removed the 16 MB/thread mask cache.
2. **Payload trim.** Dropped dead reserved fields, u32→u16 length table,
   merged mask+coefficient range-coder streams (one flush instead of two),
   moved the last two raw header bytes (dc, flags) into the stream as
   context-coded bins. Combined ≈ +5 % ratio.
3. **Adaptive last-significant-position (EOB)** replacing 13 raw bypass bits.
4. **Trained context priors** (tools/mc_train, baked into mc_rangecoder.h):
   per-block contexts reset every block, so cold p=0.5 starts cost real bits.
   +12–14 % ratio, generalizes across scans. Adaptation shift 5→4: +2.8 %.
5. **Significance contexts**: band × recent-sig-density (32 contexts).
6. **Two-level air mask**: 2-bit class per 4³ subcube (uniform-air /
   uniform-material / mixed, neighbor-class contexts), per-voxel bins only in
   mixed subcubes. Decode +48 %, encode +10 %, ratio neutral.
7. **SOR air-fill** (in-place Gauss-Seidel, ω=1.6, 4 sweeps, air-voxel list)
   replacing 8 Jacobi sweeps: better fill in less time → +2 % ratio,
   +0.12 dB PSNR, +22 % encode.
8. **Precomputed quant-step table** (was: 4096 `powf` per block per side) and
   **sparse-aware inverse DCT** (skips zero coefficients, 8-wide vectorizable
   inner loops): invDCT 14.2→7.6 µs/block.
9. **Max-error corrections** (`mc_set_max_error`): encoder reconstructs each
   block and codes sparse (position, delta) corrections for |err|>τ;
   guaranteed max error, self-contained payloads. τ=24 at q=6 costs 0.6 %.
10. **Decode-side deblock** (`mc_deblock`): clamped H.264-style filter across
    16³ faces, quality-gated, never touches air. Optional, no format change.

11. **Coarse-to-fine air-fill init**: solve the fill on the 4³ subcube grid
    first, seed fine air voxels from their cell, then 3 SOR sweeps: +0.6 %
    ratio, +0.05 dB, encode +12 % vs SOR-4 alone.
12. **DC-only fast path**: a block whose coefficients all quantize to zero
    decodes as a memset of its dc (no mask, no iDCT).
13. **Partial-fetch streaming mode** (`mc_reader_set_partial_fetch`): decode a
    block from bitmap+lens (cached per chunk) + that block's payload only —
    96× fewer bytes for scattered access over S3-like sources (new test:
    tests/mc_stream_partial.c).
14. **Quality-interpolated priors**: separate prior tables trained at q=1 and
    q=12 (combined 137 keV + 78 keV corpus), interpolated in log2(q) at
    runtime — the decoder knows q, so this is side-information-free.
15. **Quantizer constant sweep** (HF_EXP × DZ_FRAC, 9 points): the frozen
    (0.65, 0.80) sit exactly on the RD optimum for the 2.4 µm data —
    validated, not just assumed.

## Negative results (tested, measured, rejected — kept in code as notes)

- **HEVC-style coefficient groups + gt1/gt2 + Rice remainders**: −2 % vs the
  adaptive-unary ladder. The EOB already truncates the sparse tail; the
  skewed magnitude distribution wants adaptive bins, not bypass remainders.
- **RDOQ** (λ sweep 0.05–0.6, with/without m−1 candidate): always on-or-below
  the dead-zone RD curve. The frozen dz=0.8 + band-weighted steps already
  encode a better allocation than flat-λ thresholding.
- **Adaptive per-block QP** (variance-based): below the RD curve at iso-PSNR.
  Scroll µCT is statistically homogeneous; spatial reallocation doesn't pay.
- **Sign data hiding**: ~1 bit/block here (no coefficient groups) + parity
  distortion — skipped.
- **Isotropic dering** (clamped 6-neighbor pull on material): −0.4 dB PSNR,
  −0.002 SSIM at q=6 — eats real fiber texture. A useful dering here would
  need directional (sheet-tangent) selection, CDEF-style.
- **Extent-aware inverse DCT** (transform only the occupied low-frequency
  box): +3 % decode at q=12 but −5–8 % at q=1/6 (strided box rotates + dense
  fallback overhead). Reverted; kept the DC-only constant-block fast path.
- **Per-chunk DC plane**: DC bytes are 0.7 % of the file; not worth
  reintroducing chunk-level decode state.
- **Band-conditioned magnitude ladder** (first 3 unary rungs × band group):
  ratio-neutral — per-block adaptation already learns the block's own
  magnitude distribution.
- **32³ blocks**: ruled out by policy — an iDCT-32 is ~8× the per-voxel work,
  decode parity with 16³ is unrealistic and the RD upside isn't massive.

## Open item

**Interleaved-rANS entropy stage** for magnitudes/signs (static per-chunk
tables) is the one remaining architectural idea — c3d demonstrates ~3×
decode throughput from 8-way rANS in plain C. It trades the adaptive bins
that won several ratio battles above, so it needs a careful hybrid design
and reliable wall-clock measurement (blocked while the bench machine is in
low-power mode).

## Reproduce

```sh
build/mc_fetch "s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260323153942-2.400um-0.2m-137keV-masked.zarr/0" \
    3328 4224 4224 512 /tmp/paris4_2.4um_512.bin
build/mc_mask  /tmp/paris4_2.4um_512.bin 512 /tmp/paris4_masked_512.bin 1.0
build/mc_bench /tmp/paris4_masked_512.bin 512 512 1,3,6,12          # main sweep
build/mc_bench /tmp/paris4_masked_512.bin 512 512 6 24              # τ=24 bound
build/mc_bench /tmp/paris4_masked_512.bin 512 512 6,12 0 1          # + deblock
cc -O2 -w -o build/mc_train tools/mc_train.c -lm                    # retrain priors
```

Raw sweep outputs for every milestone live in this directory
(baseline.txt → final_v3.3.txt).

## vs SuperOptimizer/c3d (iso-rate, masked 2.4 µm PHercParis4 512³, 1 thread)

c3d driven at the ratio mc achieved per quality (tools/mc_vs_c3d).

| rate  | mc PSNR / SSIM / p99 / max | c3d PSNR / SSIM / p99 / max | mc enc/dec MB/s | c3d enc/dec MB/s |
|-------|----------------------------|------------------------------|-----------------|-------------------|
| 8.8×  | **43.66** / .9972 / 5 / 32 | 39.99 / .9911 / 8 / 29       | 92 / 101        | 246 / 488         |
| 20.6× | **38.46** / .9905 / 10 / 44| 35.26 / .9748 / 15 / 53      | 113 / 175       | 335 / 641         |
| 33.9× | **35.83** / .9822 / 13 / 62| 33.43 / .9618 / 20 / 73      | 120 / 234       | 344 / 688         |
| 51.5× | **33.49** / .9687 / 17 / 70| 32.08 / .9467 / 23 / 102     | 125 / 288       | 349 / 703         |

78 keV full-scroll center crop at 37.5×: mc 38.92 dB / .9868 vs c3d 35.86 / .9722.

mc wins quality at iso-rate by **+1.4 to +3.7 dB PSNR** with better SSIM and
p99 everywhere; mc's random-access atom is a 16³ block (0.015 ms cold) vs
c3d's 256³ chunk (whole-chunk decode ≈ 24 ms before one voxel is usable).
c3d wins raw throughput ~3× (8-way interleaved static rANS + wavelet) — that
is the ceiling plain-C entropy-stage parallelism buys, and the motivation for
a future interleaved-rANS magnitude stage here.

## Research round 2 — actionable queue (see git history for full report)

Ranked by impact/complexity for this codec (AV2 paper arXiv:2601.02712 was the
key source; AV2 transform+entropy tools total −7.1% BD-rate all-intra):
1. 3D LFNST/IST secondary transform — IMPLEMENTED + MEASURED NEUTRAL (ratio
   and PSNR identical, max error marginally better; eigen-spread 1000:1 but
   the band-weighted quantizer already exploits that profile). Disabled;
   trained table + tools/mc_klt kept for future regimes.
2. PARA per-context adaptation-rate tuning — IMPLEMENTED (per-context shift in
   ctx_t, per-class table). Coordinate descent on the corpus found shift 4
   already optimal for every class (the earlier global 5→4 change captured the
   gain); the infrastructure stays for future corpora.
3. ATC-style split entropy — IMPLEMENTED + MEASURED, REJECTED in this form:
   adaptive LF corner + static per-band 2-lane rANS HF tail (backward-decoded
   from payload end, escapes in the rc stream reverse-ordered, 3-byte lane
   states, only for blocks with >=256 HF symbols). Round-trips clean but costs
   ~6% ratio at q=6 (31.7x vs 33.8x) with no decode win: per-block adaptation
   is strongest exactly on the HF zeros (p adapts to ~0.97+ within a block;
   a corpus-average static table can't follow), and the probe's +1.6% bound
   was for full replacement, not this split. A future attempt needs zero-RUN
   symbols in the alphabet (cuts symbol count 5-10x -> real decode win) or
   per-chunk shared rANS streams (amortized states, breaks block independence).
   Trainer (tools/mc_rans_tab.c) + tables (src/mc_rans_tab.h) kept.
4. TCQ trellis quantization (parity-driven 4–9 state lattice; distinct
   mechanism from the rejected RDOQ). Decoder ~free, encoder Viterbi.
5. HPEZ-style per-chunk parameter auto-tuning by sampling (encode-only).
6. Cache: S3-FIFO eviction beats LRU/CLOCK on scan-shaped render workloads
   (SOSP'23: 6× LRU throughput @16T, mean 14% miss cut); SIEVE explicitly NOT
   scan-resistant — avoid. Page-table residency + fallback-to-coarser-LOD
   for renderers (GigaVoxels/UE5 SVT).
7. Streaming: ≤2-GET cold reads via fixed-address/suffix-range index tiers
   (Neuroglancer sharded, Zarr v3 end-placed shard index); one-GET head
   region with root index + coarsest LODs; offset-adjacency coalescing;
   keep Morton (hash-sharding destroys locality). Optional crc32c.
Skip: MTS/MIP/ISP/FSC (need intra prediction / screen content), SZ4 (does
not exist), HPEZ interpolation core (sequential, kills random access),
INR/learned codecs (no random access, GPU-bound), SIEVE, hash-sharding.
External anchor: synchrotron studies find 3–4× lossy safe for reconstruction
quality, 6–8× for phase-retrieved data — our q=1 (~9×) and q=3 (~21×) tiers
bracket that on already-reconstructed volumes.
