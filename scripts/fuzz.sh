#!/usr/bin/env bash
# fuzz.sh — fuzz a matter-compressor surface with AFL++.
#
# Harnesses (tests/fuzz/) expose the standard libFuzzer entrypoint; AFL++
# compiles them directly (afl-clang-fast wraps its own persistent driver) under
# ASan+UBSan, so any memory/UB error is a crash AFL++ saves.
#
#   usage: scripts/fuzz.sh [seconds] [out_dir] [target]
#     target = decode (default) | block
#       decode: whole .mca through mc_open/decode_block/verify (header + tree)
#       block : a single block payload through mc_dec_block (range coder + IDCT)
#   env: AFL_DIR=/usr/local/bin  (location of afl-clang-fast / afl-fuzz)
#
# A nonzero exit means AFL++ found a crash (saved under <out>/findings/*/crashes
# and replayed under plain ASan with line/stack detail). CI runs a short bounded
# pass; locally, pass a larger budget (e.g. `scripts/fuzz.sh 3600`).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECS="${1:-120}"
OUT="${2:-$ROOT/build_fuzz}"
TARGET="${3:-decode}"
AFL_DIR="${AFL_DIR:-$(dirname "$(command -v afl-fuzz)")}"
CC="$AFL_DIR/afl-clang-fast"
[ -x "$CC" ] || { echo "afl-clang-fast not found (set AFL_DIR); is AFL++ installed?"; exit 2; }

SRC=(src/matter_compressor.c src/c3d.c tools/vendor/libs3/libs3.c)
INC=(-Isrc -Itools/vendor/libs3)
LIBS=(-lm -lpthread -lzstd -lcurl)
SEEDGEN=""
case "$TARGET" in
  decode) HARNESS=tests/fuzz/mc_fuzz_decode.c; SEEDGEN=tests/fuzz/mc_fuzz_seed.c;;
  block)  HARNESS=tests/fuzz/mc_fuzz_block.c;  SEEDGEN=tests/fuzz/mc_fuzz_block_seed.c;;
  tiff)   HARNESS=tests/fuzz/mc_fuzz_tiff.c;   SEEDGEN=tests/fuzz/mc_fuzz_tiff_seed.c
          SRC=(src/mc_tiff.c); LIBS=(-lm);;   # self-contained reader, no archive/libs3 deps
  surface) HARNESS=tests/fuzz/mc_fuzz_surface.c; SEEDGEN=tests/fuzz/mc_fuzz_surface_seed.c
          SRC=(src/mc_surface.c src/mc_tiff.c); LIBS=(-lm);;   # surface loaders + their tiff dep
  segment) HARNESS=tests/fuzz/mc_fuzz_segment.c; SEEDGEN=tests/fuzz/mc_fuzz_segment_seed.c
          SRC=(src/mc_segment.c); LIBS=(-lm);;   # self-contained detector + post-proc
  *) echo "unknown target '$TARGET' (decode|block|tiff|surface|segment)"; exit 2;;
esac

rm -rf "$OUT"; mkdir -p "$OUT/corpus" "$OUT/findings"
cd "$ROOT"

# 1. seed generator (plain clang — emits valid inputs for the chosen target).
clang -O1 -w "${INC[@]}" "$SEEDGEN" "${SRC[@]}" "${LIBS[@]}" -o "$OUT/seedgen"
"$OUT/seedgen" "$OUT/corpus"
# carry the known crash reproducers into the corpus so regressions stay covered.
if [ -d "tests/fuzz/crashes/$TARGET" ]; then
  cp tests/fuzz/crashes/"$TARGET"/* "$OUT/corpus/" 2>/dev/null || true
fi

# 2. AFL++-instrumented harness, ASan+UBSan. afl-clang-fast intercepts
#    -fsanitize=fuzzer and links its own driver (libAFLDriver.a) around the
#    libFuzzer-style LLVMFuzzerTestOneInput entrypoint.
AFL_USE_ASAN=1 AFL_USE_UBSAN=1 "$CC" -O1 -g -fsanitize=fuzzer "${INC[@]}" \
  "$HARNESS" "${SRC[@]}" "${LIBS[@]}" -o "$OUT/mc_fuzz"

# 3. standalone replay build (plain clang + ASan) for crash triage / CI smoke.
clang -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -w -DMC_FUZZ_STANDALONE \
  "${INC[@]}" "$HARNESS" "${SRC[@]}" "${LIBS[@]}" -o "$OUT/mc_fuzz_replay"

# 4. run AFL++ for the budget. -V = wall-clock seconds, then exit.
export AFL_NO_UI=1 AFL_BENCH_UNTIL_CRASH=0 AFL_SKIP_CPUFREQ=1 AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
echo "== afl-fuzz ${SECS}s =="
set +e
"$AFL_DIR/afl-fuzz" -i "$OUT/corpus" -o "$OUT/findings" -V "$SECS" -- "$OUT/mc_fuzz" @@
set -e

# 5. report + replay any crashes with full ASan detail.
crashes=$(find "$OUT/findings" -path '*/crashes/id:*' 2>/dev/null | sort)
n=$(printf '%s\n' "$crashes" | grep -c . || true)
echo "== AFL++ done: $n crash input(s) =="
if [ "$n" -gt 0 ]; then
  printf '%s\n' "$crashes" | while read -r c; do
    [ -n "$c" ] || continue
    echo "--- replay $c ---"; "$OUT/mc_fuzz_replay" "$c" || true
  done
  exit 1
fi
echo "no crashes"
