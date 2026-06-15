#!/usr/bin/env bash
# coverage.sh — line/region/branch coverage for the matter-compressor core,
# scoped to src/matter_compressor.c ONLY. c3d.c and the vendored libs3 are
# deliberately excluded (they are third-party / separately-validated decoders).
#
# Builds every OFFLINE test with clang source-based coverage instrumentation,
# runs them (against the local zarr fixture so the zarr/volume paths count),
# merges the profiles, and emits a text summary + lcov + HTML report.
#
#   usage: scripts/coverage.sh [out_dir]
#   env:   CC=clang (required — source-based coverage is a clang feature)
#          MC_COVERAGE_MIN=NN  if set, exit nonzero when line% < NN (CI gate)
#
# Output (default out_dir=build_cov):
#   build_cov/coverage.txt   per-function + file summary (matter_compressor.c)
#   build_cov/coverage.lcov  lcov tracefile (for genhtml / Codecov)
#   build_cov/html/          browsable HTML report
#   build_cov/summary.md     one-block markdown summary (CI job summary)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/build_cov}"
CC="${CC:-clang}"
case "$("$CC" --version 2>/dev/null | head -1)" in
  *clang*) ;;
  *) echo "coverage.sh requires clang (CC=$CC is not clang)"; exit 2;;
esac

SRC=(src/matter_compressor.c src/c3d.c tools/vendor/libs3/libs3.c)
INC=(-Isrc -Itools/vendor/libs3)
LIBS=(-lm -lpthread -lzstd -lcurl)
CFLAGS=(-O1 -g -fprofile-instr-generate -fcoverage-mapping -w "${INC[@]}")

# Offline tests only. The zarr/volume tests take the fixture root as argv[1]
# (see RUN_ARGS below); mc_volume_test (the real-network probe) is excluded.
OFFLINE_TESTS=(
  mc_roundtrip mc_append_roundtrip mc_stream_partial mc_v6_test
  mc_cache_test mc_render_test mc_stream_volume_test mc_archive_concurrent_test
  mc_volume_offline_test mc_volume_api_test mc_zarr_test mc_api_test
  mc_decode_robust_test mc_stream_volume_api_test mc_volume_v3_test
)

rm -rf "$OUT"; mkdir -p "$OUT/prof" "$OUT/bin"
cd "$ROOT"

# Build the offline zarr fixture (zarr v2, raw u8 chunks — no c3d dependency).
FIXTURE="$OUT/fixture"
if [ -x "$OUT/bin/mc_make_fixture" ] || "$CC" -O1 -w "${INC[@]}" \
      tests/support/mc_make_fixture.c -lm -o "$OUT/bin/mc_make_fixture" 2>/dev/null; then
  "$OUT/bin/mc_make_fixture" "$FIXTURE" || true
fi
export MC_ZARR_ROOT="$FIXTURE/zarr/0"
export MC_VOLUME_URL="$FIXTURE/zarr"

# Build the v3 / c3d-sharded fixture (uses the c3d encoder -> link the lib).
FIXTURE_V3="$OUT/fixture_v3"
if "$CC" -O1 -w "${INC[@]}" tests/support/mc_make_fixture_v3.c "${SRC[@]}" "${LIBS[@]}" \
      -o "$OUT/bin/mc_make_fixture_v3" 2>/dev/null; then
  "$OUT/bin/mc_make_fixture_v3" "$FIXTURE_V3" || true
fi
# Build the v2 blosc+zstd fixture (covers blosc_decode).
FIXTURE_BLOSC="$OUT/fixture_blosc"
if "$CC" -O1 -w "${INC[@]}" tests/support/mc_make_fixture_blosc.c -lzstd \
      -o "$OUT/bin/mc_make_fixture_blosc" 2>/dev/null; then
  "$OUT/bin/mc_make_fixture_blosc" "$FIXTURE_BLOSC" || true
fi

built=()
for t in "${OFFLINE_TESTS[@]}"; do
  if [ -f "tests/$t.c" ] && "$CC" "${CFLAGS[@]}" "tests/$t.c" "${SRC[@]}" "${LIBS[@]}" -o "$OUT/bin/$t"; then
    built+=("$t")
  else
    echo "skip (build failed or missing): $t"
  fi
done

# Per-test argv: the zarr/volume tests take the fixture path; others take none.
run_args() {
  case "$1" in
    mc_zarr_test)          echo "$FIXTURE/zarr/0";;
    mc_volume_offline_test) echo "$FIXTURE/zarr $OUT";;
    mc_volume_api_test)     echo "$FIXTURE/zarr $OUT";;
    mc_volume_v3_test)      echo "$FIXTURE_V3/zarr $OUT";;
    *) echo "";;
  esac
}

# Run each; a nonzero exit (e.g. a test needing real network) is tolerated —
# we still keep whatever profile it produced for coverage accounting.
objs=()
for t in "${built[@]}"; do
  # shellcheck disable=SC2046
  LLVM_PROFILE_FILE="$OUT/prof/$t.profraw" "$OUT/bin/$t" $(run_args "$t") >/dev/null 2>&1 || \
    echo "note: $t exited nonzero (profile still collected)"
  objs+=(-object "$OUT/bin/$t")
done

# Extra runs of the instrumented mc_zarr_test binary against the v3 (c3d) and
# blosc fixtures (same source, different inner codec -> distinct branches).
if [ -x "$OUT/bin/mc_zarr_test" ]; then
  [ -d "$FIXTURE_V3/zarr/0" ] && LLVM_PROFILE_FILE="$OUT/prof/zarr_v3.profraw" \
      "$OUT/bin/mc_zarr_test" "$FIXTURE_V3/zarr/0" >/dev/null 2>&1 || true
  [ -d "$FIXTURE_BLOSC/zarr/0" ] && LLVM_PROFILE_FILE="$OUT/prof/zarr_blosc.profraw" \
      "$OUT/bin/mc_zarr_test" "$FIXTURE_BLOSC/zarr/0" >/dev/null 2>&1 || true
fi

llvm-profdata merge -sparse "$OUT"/prof/*.profraw -o "$OUT/merged.profdata"

# Report scoped to matter_compressor.c only.
FIRST="$OUT/bin/${built[0]}"
SCOPE="$ROOT/src/matter_compressor.c"
llvm-cov report "$FIRST" "${objs[@]:2}" -instr-profile="$OUT/merged.profdata" "$SCOPE" \
  | tee "$OUT/coverage.txt"
llvm-cov export "$FIRST" "${objs[@]:2}" -instr-profile="$OUT/merged.profdata" -format=lcov "$SCOPE" \
  > "$OUT/coverage.lcov"
llvm-cov show "$FIRST" "${objs[@]:2}" -instr-profile="$OUT/merged.profdata" \
  -format=html -output-dir="$OUT/html" "$SCOPE" >/dev/null 2>&1 || true

LINE_PCT="$(awk '/matter_compressor.c/{print $(NF-3)}' "$OUT/coverage.txt" | tr -d '%' | head -1)"
{
  echo '### Coverage — `src/matter_compressor.c`'
  echo '```'
  cat "$OUT/coverage.txt"
  echo '```'
} > "$OUT/summary.md"
echo "line coverage: ${LINE_PCT}%  (report: $OUT/coverage.txt, html: $OUT/html/index.html)"

if [ -n "${MC_COVERAGE_MIN:-}" ]; then
  awk -v min="$MC_COVERAGE_MIN" -v got="${LINE_PCT:-0}" 'BEGIN{
    if (got+0 < min+0){ printf("FAIL: line coverage %.2f%% < required %s%%\n",got,min); exit 1 }
    printf("OK: line coverage %.2f%% >= %s%%\n",got,min)
  }'
fi
