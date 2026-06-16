#!/usr/bin/env bash
# bench.sh — run the matter-compressor benchmark basket and emit JSON + a
# markdown summary for CI to log (NO regression gate — numbers are surfaced as
# artifacts / job summary; humans review trends).
#
#   usage: scripts/bench.sh [out_dir]
#   env:   MC_BENCH_VOL=/path/vol.bin  MC_BENCH_VOLDIM=512  (optional real volume;
#          otherwise synthetic). Build with -march=native for representative numbers.
#
# Output (default out_dir=build_bench):
#   micro.json     mc_microbench latency/bandwidth/throughput
#   quality.txt    mc_bench ratio/PSNR/MAE/SSIM across a quality ladder
#   summary.md     compact CI job-summary block
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/build_bench}"
CC="${CC:-clang}"
rm -rf "$OUT"; mkdir -p "$OUT"; cd "$ROOT"

INC="-Isrc -Itools/vendor/libs3"; LIBS="-lm -lpthread -lzstd -lcurl"
NATIVE="-march=native"; "$CC" $NATIVE -x c -c /dev/null -o /dev/null 2>/dev/null || NATIVE=""

$CC -O3 $NATIVE -ffp-contract=off -w $INC tools/mc_microbench.c src/matter_compressor.c src/c3d.c tools/vendor/libs3/libs3.c $LIBS -o "$OUT/microbench"
$CC -O3 $NATIVE -ffp-contract=off -w $INC tools/mc_bench.c       src/matter_compressor.c src/c3d.c tools/vendor/libs3/libs3.c $LIBS -o "$OUT/bench"
$CC -O3 $NATIVE -ffp-contract=off -w $INC tools/mc_segbench.c    src/mc_segment.c $LIBS -o "$OUT/segbench"

echo "== microbench (latency / bandwidth / throughput) =="
if [ -n "${MC_BENCH_VOL:-}" ] && [ -f "$MC_BENCH_VOL" ]; then
  "$OUT/microbench" "$MC_BENCH_VOL" "${MC_BENCH_VOLDIM:-512}" 6 | tee "$OUT/micro.json"
else
  "$OUT/microbench" --synth 256 | tee "$OUT/micro.json"
fi

echo "== quality basket across the q ladder =="
if [ -n "${MC_BENCH_VOL:-}" ] && [ -f "$MC_BENCH_VOL" ]; then
  "$OUT/bench" "$MC_BENCH_VOL" "${MC_BENCH_VOLDIM:-512}" 256 1,3,6,12 8 | tee "$OUT/quality.txt"
else
  "$OUT/bench" --synth 256 1,3,6,12 | tee "$OUT/quality.txt"
fi

echo "== sheet detector + topology post-proc =="
"$OUT/segbench" 128 5 | tee "$OUT/segment.txt"

{
  echo '### Benchmarks (log-only — no regression gate)'
  echo
  echo '**Latency / bandwidth / throughput** (`mc_microbench`):'
  echo '```json'; cat "$OUT/micro.json"; echo '```'
  echo
  echo '**Quality basket** (`mc_bench`):'
  echo '```'; cat "$OUT/quality.txt"; echo '```'
  echo
  echo '**Sheet detector + topology post-proc** (`mc_segbench`):'
  echo '```'; cat "$OUT/segment.txt"; echo '```'
} > "$OUT/summary.md"
echo "wrote $OUT/summary.md"
