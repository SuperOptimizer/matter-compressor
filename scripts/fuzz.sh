#!/usr/bin/env bash
# fuzz.sh — fuzz the matter-compressor decode path with AFL++.
#
# The harness (tests/fuzz/mc_fuzz_decode.c) exposes the standard libFuzzer
# entrypoint; AFL++ compiles it directly (afl-clang-fast wraps its own
# persistent driver) and instruments matter_compressor.c + c3d.c. We build with
# ASan+UBSan so any memory/UB error in the decode path is a crash AFL++ saves.
#
#   usage: scripts/fuzz.sh [seconds] [out_dir]
#   env:   AFL_DIR=/usr/local/bin  (location of afl-clang-fast / afl-fuzz)
#
# A nonzero exit means AFL++ found a crash (saved under <out>/findings/*/crashes
# and replayed under plain ASan with line/stack detail). CI runs a short bounded
# pass; locally, pass a larger budget (e.g. `scripts/fuzz.sh 3600`).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECS="${1:-120}"
OUT="${2:-$ROOT/build_fuzz}"
AFL_DIR="${AFL_DIR:-$(dirname "$(command -v afl-fuzz)")}"
CC="$AFL_DIR/afl-clang-fast"
[ -x "$CC" ] || { echo "afl-clang-fast not found (set AFL_DIR); is AFL++ installed?"; exit 2; }

SRC=(src/matter_compressor.c src/c3d.c tools/vendor/libs3/libs3.c)
INC=(-Isrc -Itools/vendor/libs3)
LIBS=(-lm -lpthread -lzstd -lcurl)

rm -rf "$OUT"; mkdir -p "$OUT/corpus" "$OUT/findings"
cd "$ROOT"

# 1. seed generator (plain clang — it just writes valid .mca files).
clang -O1 -w "${INC[@]}" tests/fuzz/mc_fuzz_seed.c "${SRC[@]}" "${LIBS[@]}" -o "$OUT/seedgen"
"$OUT/seedgen" "$OUT/corpus"

# 2. AFL++-instrumented harness, ASan+UBSan. afl-clang-fast intercepts
#    -fsanitize=fuzzer and links its own driver (libAFLDriver.a) around the
#    libFuzzer-style LLVMFuzzerTestOneInput entrypoint.
AFL_USE_ASAN=1 AFL_USE_UBSAN=1 "$CC" -O1 -g -fsanitize=fuzzer "${INC[@]}" \
  tests/fuzz/mc_fuzz_decode.c "${SRC[@]}" "${LIBS[@]}" -o "$OUT/mc_fuzz_decode"

# 3. standalone replay build (plain clang + ASan) for crash triage / CI smoke.
clang -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -w -DMC_FUZZ_STANDALONE \
  "${INC[@]}" tests/fuzz/mc_fuzz_decode.c "${SRC[@]}" "${LIBS[@]}" -o "$OUT/mc_fuzz_replay"

# 4. run AFL++ for the budget. -V = wall-clock seconds, then exit.
export AFL_NO_UI=1 AFL_BENCH_UNTIL_CRASH=0 AFL_SKIP_CPUFREQ=1 AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
echo "== afl-fuzz ${SECS}s =="
set +e
"$AFL_DIR/afl-fuzz" -i "$OUT/corpus" -o "$OUT/findings" -V "$SECS" -- "$OUT/mc_fuzz_decode" @@
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
