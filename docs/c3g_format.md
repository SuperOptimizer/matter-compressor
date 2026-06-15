# c3g — GPU-amenable block codec (design)

A **parallel, additional** block codec for matter-compressor whose payloads are
**decodable by a single GPU thread per 16³ block**, so the viewer can push
*compressed* blocks across the PCIe bus and decode them GPU-side directly into a
small VRAM cache of decoded blocks. It does **not** replace the existing CABAC
codec (`mc_enc_block` / `mc_dec_block`) — that format, its determinism
guarantees, and every existing `.mca` are untouched.

## Why a new format

The existing entropy stage is a CABAC-style *adaptive binary range coder*:
bit-serial, with per-bit context-probability updates. That is inherently serial
and branch-divergent — the worst case for a GPU. The DCT-16 + dead-zone quant
front-end, by contrast, is already parallel-friendly.

c3g therefore **keeps the front-end and replaces only the entropy stage**:

| stage              | existing (CABAC)            | c3g                          |
|--------------------|-----------------------------|------------------------------|
| transform          | integer separable DCT-16    | **same** (`mc_dct3_*`)       |
| quant              | dead-zone, `step_tab[N3]`   | **same** (`deq_one`)         |
| coefficient entropy| adaptive binary range coder | **static rANS, per block**   |
| container          | `.mca` chunk/block index    | **same**                     |

rANS (range Asymmetric Numeral System) with a *static* frequency table is a
simple, branch-light state machine — one decode step is a multiply, a couple of
adds, a table lookup, and a byte-wise renormalization. No adaptive state, no
cross-symbol or cross-block dependency, so one GPU invocation decodes one block.

## Block payload layout (little-endian, byte-aligned)

```
offset  size  field
0       1     flags        bit0 = has air-mask, bit1 = (reserved: corrections)
1       1     dc           block DC term (0..255)
2       2     nsym         number of coded coefficient symbols (= eob in scan order)
4       2     rans_len     length of the rANS byte stream
6       4     init_state   rANS initial state x (decode reads stream backwards)
10      ...   air bitmask  present iff flags bit0; ceil(N3/8) = 512 bytes
...     ...   rANS stream  rans_len bytes
```

A *constant* block (no air, no AC coefficients) is encoded as `nsym=0`,
`rans_len=0`, `flags=0`; the decoder fills the block with `dc`.

## Symbol alphabet

Coefficients are visited in the same ascending-frequency **scan order** as the
existing codec (reused `scanS_build`), truncated at `nsym` (the EOB). Each coded
coefficient is mapped to a small symbol alphabet so the static rANS table is
tiny and GPU-cache-friendly:

- symbol `0`              : a run of zero coefficients is *not* used; instead the
  scan is dense up to EOB and zeros inside [0,eob) are coded as the literal
  zero-magnitude symbol.
- magnitude is coded as `sym = min(|level|, MAXSYM-1)`; levels `>= MAXSYM-1`
  carry an Exp-Golomb *escape* of the remainder as raw bytes after the symbol.
- sign is one bypass bit per nonzero level, packed into the rANS stream tail as
  raw bits (bypass = uniform rANS symbols, or a trailing bit buffer).

`MAXSYM = 16` (4-bit base alphabet; escapes handle the heavy tail). The static
frequency table is `freq[MAXSYM]`, normalized to `RANS_TOTAL = 4096` (12-bit),
shipped once per archive (header), not per block.

## rANS parameters (must match CPU and GPU exactly)

- `RANS_L = 1<<16` (lower bound); 8-bit renormalization (emit/consume one byte).
- `RANS_TOTAL = 1<<12` (frequency normalization).
- decode: `s = sym_from_slot(x & (TOTAL-1)); x = freq[s]*(x>>12) + (x &
  (TOTAL-1)) - cdf[s]; while (x < RANS_L) x = (x<<8) | *--ptr;`
- encode is the mirror, producing the stream back-to-front, so the decoder reads
  it front-to-back from `init_state`.

These constants are fixed in `c3g.h` and shared by the CPU reference and the GPU
shader so the two are bit-exact.

## Validation

`tests/mc_c3g_test.c`: round-trip random + structured 16³ blocks through the c3g
encoder/decoder; assert the c3g decode matches within the codec's lossy bound
the same way the CABAC path is tested, and that c3g encode→decode is itself
deterministic and bit-exact across runs. The CPU decoder is the oracle the GPU
compute shader is later checked against (same payload bytes in, same 4096 voxels
out).
