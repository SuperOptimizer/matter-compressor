cmake_minimum_required(VERSION 3.16)
project(matter_compressor C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Standalone (top-level) vs vendored (add_subdirectory in a parent project).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(MC_STANDALONE ON)
else()
  set(MC_STANDALONE OFF)
endif()

# Only set fast/native flags + tests when built standalone. When vendored, the parent
# owns the build type/flags and may build shared libs (so we force PIC below).
if(MC_STANDALONE)
  set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math -ffp-contract=fast")
  add_compile_options(-Wall -Wno-stringop-overflow)
  if(NOT DEFINED MC_NO_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()

# FLOATING-POINT: the block transform is f32 (mc_codec_float.h replaced the Q14
# integer DCT). We build with -ffast-math (FMA contraction + reassociation) so
# the f32 DCT lines autovectorize — this is the float core's intended build mode.
# TRADE-OFF: fast-math makes f32 results depend on ISA / opt level / FMA, so
# archives are NOT guaranteed byte-identical across machines, and the max-error
# (tau) correction bound is best-effort SAME-BUILD: encode and decode in one
# build agree exactly (the corrections are computed against this build's own
# reconstruction), but a decoder built differently may diverge by a small delta.
# The export pipeline encodes and decodes with the same build, so the bound holds
# in practice. (To restore strict cross-ISA reproducibility, drop -ffast-math and
# add -ffp-contract=off — the DCT then evaluates in fixed order, bit-identical.)

# Optional ThinLTO + PGO. MEASURED POLICY (bench/RESULTS.md): LTO of either
# flavor WITHOUT profiles costs ~15% decode (link-time inliner mispredicts the
# branch-dense range-coder loop); WITH profiles it gains ~5%. So: ThinLTO only
# together with MC_PGO_USE. Flow:
#   1. cmake -DMC_PGO_GEN=ON; build; run mc_bench on representative volumes
#      with LLVM_PROFILE_FILE=prof/%p.profraw
#   2. llvm-profdata merge -output=merged.profdata prof/*.profraw
#   3. cmake -DMC_THINLTO=ON -DMC_PGO_USE=/abs/path/merged.profdata; rebuild
option(MC_THINLTO "ThinLTO (pair with MC_PGO_USE; alone it is a regression)" OFF)
option(MC_PGO_GEN "instrumented build for PGO profile collection" OFF)
set(MC_PGO_USE "" CACHE FILEPATH "merged .profdata to optimize with")
if(MC_PGO_GEN)
  add_compile_options(-fprofile-instr-generate)
  add_link_options(-fprofile-instr-generate)
endif()
if(MC_PGO_USE)
  add_compile_options(-fprofile-instr-use=${MC_PGO_USE} -Wno-profile-instr-unprofiled -Wno-profile-instr-out-of-date)
endif()
if(MC_THINLTO)
  if(NOT MC_PGO_USE)
    message(WARNING "MC_THINLTO without MC_PGO_USE measured ~15% SLOWER decode — supply profiles")
  endif()
  add_compile_options(-flto=thin)
  add_link_options(-flto=thin)
endif()

# core library: codec + archive + cache + sample/render + zarr/s3/volume —
# everything in one .c/.h pair. Required deps: libcurl + libzstd (and the
# vendored libs3.c, compiled into the target unless MC_LIBS3_TARGET is set).
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
find_library(ZSTD_LIB zstd REQUIRED HINTS /opt/homebrew/lib /usr/local/lib)
find_path(ZSTD_INC zstd.h REQUIRED HINTS /opt/homebrew/include /usr/local/include)
# libs3 is vendored by default; a parent project (taberna) can supply the shared
# `s3` target via -DMC_LIBS3_TARGET=s3 so there is ONE libs3 across the build
# (avoids duplicate s3_* symbols when this lib is linked alongside another libs3 user).
set(MC_LIBS3_TARGET "" CACHE STRING "external libs3 target to link instead of the vendored tools/vendor/libs3 copy")
if(MC_LIBS3_TARGET)
  set(MC_LIBS3_SRC "")
else()
  set(MC_LIBS3_SRC tools/vendor/libs3/libs3.c)
endif()
add_library(matter_compressor STATIC
  src/mc_codec.c
  src/mc_archive.c
  src/mc_stream.c
  src/mc_cache.c
  src/mc_render.c
  src/mc_zarr.c
  src/mc_volume.c
  src/mc_surface.c
  src/mc_segment.c
  src/mc_solve.c
  src/mc_trace.c
  ${MC_LIBS3_SRC})
target_include_directories(matter_compressor PUBLIC src PRIVATE ${ZSTD_INC})
if(MC_LIBS3_TARGET)
  target_link_libraries(matter_compressor ${MC_LIBS3_TARGET})
else()
  target_include_directories(matter_compressor PRIVATE tools/vendor/libs3)
endif()
set_target_properties(matter_compressor PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(matter_compressor CURL::libcurl ${ZSTD_LIB} m Threads::Threads)

# The codec is the hot path; build it -O3 -march=native -ffast-math regardless of
# how a parent project (vendoring via add_subdirectory) sets global flags. The
# MC_STANDALONE block above already does this for standalone builds; this target-
# scoped copy guarantees it when vendored too (the f32 DCT wants fast-math +
# FMA contraction to vectorize). -DMC_NO_NATIVE=ON for portable builds.
target_compile_options(matter_compressor PRIVATE -O3 -ffast-math -ffp-contract=fast -Wno-stringop-overflow)
if(NOT MC_NO_NATIVE)
  include(CheckCCompilerFlag)
  check_c_compiler_flag(-march=native MC_HAS_MARCH_NATIVE)
  if(MC_HAS_MARCH_NATIVE)
    target_compile_options(matter_compressor PRIVATE -march=native)
  endif()
endif()

# Tests + native flags only when built standalone (vendoring consumers don't want our
# test exes added to their build).
if(MC_STANDALONE)
  enable_testing()
  add_executable(mc_roundtrip tests/mc_roundtrip.c)
  target_link_libraries(mc_roundtrip matter_compressor)
  add_test(NAME roundtrip COMMAND mc_roundtrip)

  # appendable round-trip: open -> append -> close -> reopen -> append -> decode + verify.
  add_executable(mc_append_roundtrip tests/mc_append_roundtrip.c)
  target_link_libraries(mc_append_roundtrip matter_compressor)
  add_test(NAME append_roundtrip COMMAND mc_append_roundtrip)

  # index-node 8-byte alignment regression: node tables share the append cursor
  # with variable-length blobs, so a fresh node after a blob must still be aligned
  # (its u64 child slots are accessed as _Atomic -> BUS-fault if misaligned).
  add_executable(mc_node_align_test tests/mc_node_align_test.c)
  target_link_libraries(mc_node_align_test matter_compressor m)
  add_test(NAME node_align COMMAND mc_node_align_test)

  add_executable(mc_stream_partial tests/mc_stream_partial.c)
  target_link_libraries(mc_stream_partial matter_compressor)
  add_test(NAME stream_partial COMMAND mc_stream_partial)

  add_executable(mc_v6_test tests/mc_v6_test.c)
  target_link_libraries(mc_v6_test matter_compressor)
  add_test(NAME v6 COMMAND mc_v6_test)

  add_executable(mc_cache_test tests/mc_cache_test.c)
  target_link_libraries(mc_cache_test matter_compressor)
  add_test(NAME cache COMMAND mc_cache_test)

  add_executable(mc_render_test tests/mc_render_test.c)
  target_link_libraries(mc_render_test matter_compressor)
  add_test(NAME render COMMAND mc_render_test)

  # TIFF reader/writer lives in the tiff/ submodule with its own tests
  # (cmake -B build tiff && ctest); not re-tested here.

  # parametric surface I/O: 4x-f32 grid TIFF (x,y,z,depth) + OBJ + PPM.
  add_executable(mc_surface_test tests/mc_surface_test.c)
  target_link_libraries(mc_surface_test matter_compressor)
  add_test(NAME surface COMMAND mc_surface_test)

  # classical sheet detector (structure tensor) + topology post-processing.
  add_executable(mc_segment_test tests/mc_segment_test.c)
  target_link_libraries(mc_segment_test matter_compressor)
  add_test(NAME segment COMMAND mc_segment_test)

  # TIFF reader untrusted-input hardening lives in the tiff/ submodule
  # (tiff/tiff_robust_test.c + tiff/fuzz_tiff.c).

  # surface-loader untrusted-input hardening regression (4 fuzz-found bugs: leak,
  # unchecked realloc, OOB face index, dim overflow; tests/fuzz/mc_fuzz_surface.c).
  add_executable(mc_surface_robust_test tests/mc_surface_robust_test.c)
  target_link_libraries(mc_surface_robust_test matter_compressor)
  add_test(NAME surface_robust COMMAND mc_surface_robust_test)

  # block-sparse Levenberg-Marquardt solver (foundation for the surface tracer).
  add_executable(mc_solve_test tests/mc_solve_test.c)
  target_link_libraries(mc_solve_test matter_compressor)
  add_test(NAME solve COMMAND mc_solve_test)

  # surface tracer cost terms + growth + volume term + export.
  add_executable(mc_trace_test tests/mc_trace_test.c)
  target_link_libraries(mc_trace_test matter_compressor)
  add_test(NAME trace COMMAND mc_trace_test)

  # public-API surface coverage: introspection getters, codec/quality extras,
  # colormap + image filters, plane/quad/LOD samplers + parallel renderers.
  add_executable(mc_api_test tests/mc_api_test.c)
  target_link_libraries(mc_api_test matter_compressor)
  add_test(NAME api COMMAND mc_api_test)

  # decode-path robustness: malformed/truncated archives must not crash the
  # read-only consumer surface (mc_open/metadata/chunk_offset/decode_block/
  # verify). Regression guard for the OOB + float-UB sites AFL++ found
  # (tests/fuzz/FINDINGS.md); the fix is in tests/fuzz/hardening.patch.md.
  # The reader hardening has landed, so this is a hard gate on every build (and
  # the ASan/UBSan lane runs it too, where corrupt-input bugs surface cleanly).
  add_executable(mc_decode_robust_test tests/mc_decode_robust_test.c)
  target_link_libraries(mc_decode_robust_test matter_compressor)
  add_test(NAME decode_robust COMMAND mc_decode_robust_test)
  set_tests_properties(decode_robust PROPERTIES LABELS robustness)

  # archive/reader error paths: dims-mismatch reopen rejection, reopen load,
  # mca_probe on missing/garbage, mc_open untrusted-input gate. (Caught an 8MB
  # leak on the rejected-open path under LSan.)
  add_executable(mc_archive_errpaths_test tests/mc_archive_errpaths_test.c)
  target_link_libraries(mc_archive_errpaths_test matter_compressor)
  add_test(NAME archive_errpaths COMMAND mc_archive_errpaths_test)

  # REAL-S3 lane (anonymous Vesuvius open-data): covers mc_s3_*/s3_read + the
  # remote batched download. Skips cleanly unless MC_S3_TEST=1, so it's safe in
  # the offline gate; the s3-bench CI lane sets MC_S3_TEST=1. Labelled "s3".
  add_executable(mc_s3_volume_test tests/mc_s3_volume_test.c)
  target_link_libraries(mc_s3_volume_test matter_compressor)
  add_test(NAME s3_volume COMMAND mc_s3_volume_test)
  set_tests_properties(s3_volume PROPERTIES LABELS s3)

  add_executable(mc_stream_volume_test tests/mc_stream_volume_test.c)
  target_link_libraries(mc_stream_volume_test matter_compressor)
  add_test(NAME stream_volume COMMAND mc_stream_volume_test)

  # streaming (.mca) volume pipeline against a local file: dl thread, decode pool,
  # freeze/thaw fill, prefetch_shard/level, request_region, ensure_region, stats.
  add_executable(mc_stream_volume_api_test tests/mc_stream_volume_api_test.c)
  target_link_libraries(mc_stream_volume_api_test matter_compressor)
  add_test(NAME stream_volume_api COMMAND mc_stream_volume_api_test)

  # concurrent archive append: stresses w_ensure_shard_slot node creation under N
  # threads (regression guard for the index-corruption publish race).
  add_executable(mc_archive_concurrent_test tests/mc_archive_concurrent_test.c)
  target_link_libraries(mc_archive_concurrent_test matter_compressor)
  add_test(NAME archive_concurrent COMMAND mc_archive_concurrent_test)

  # archive repacker: Morton order + clustered index for streaming + offline
  add_executable(mc_export tools/mc_export.c)
  target_include_directories(mc_export PRIVATE src tools/vendor/libs3 ${ZSTD_INC})
  target_sources(mc_export PRIVATE tools/vendor/libs3/libs3.c)
  target_link_libraries(mc_export CURL::libcurl ${ZSTD_LIB} m Threads::Threads)

  # tools: benchmark harness (no extra deps) + zarr fetcher (libcurl + libzstd).
  add_executable(mc_bench tools/mc_bench.c)
  target_link_libraries(mc_bench matter_compressor)

  # latency/bandwidth/throughput micro-metrics as JSON (CI logs these; no gate).
  # block-decode latency, chunk-decode + region-read bandwidth, cache gets/s,
  # and decode latency over a throttled mock byte-source (deterministic
  # streaming perf, no network). Complements mc_bench's quality basket.
  add_executable(mc_microbench tools/mc_microbench.c)
  target_link_libraries(mc_microbench matter_compressor)

  # fysics-style aggressive air masking preprocessor (no deps)
  add_executable(mc_mask tools/mc_mask.c)

  add_executable(mc_verify tools/mc_verify.c)
  target_link_libraries(mc_verify matter_compressor)

  # render a parametric surface (4x-f32 grid TIFF) against a volume -> PPM.
  add_executable(mc_surface_render tools/mc_surface_render.c)
  target_link_libraries(mc_surface_render matter_compressor)

  # convert surfaces (OBJ mesh / VC per-pixel map / TIFF) to/from .grid.
  add_executable(mc_grid tools/mc_grid.c)
  target_link_libraries(mc_grid matter_compressor)

  # classical sheet detector over a live volume chunk (mc_volume + mc_sample_box).
  add_executable(mc_detect tools/mc_detect.c)
  target_link_libraries(mc_detect matter_compressor)

  # microbenchmark for the sheet detector + topology post-proc (perf baseline).
  add_executable(mc_segbench tools/mc_segbench.c)
  target_link_libraries(mc_segbench matter_compressor)

  # classical surface tracer CLI: chunk -> detect -> SDT -> seed -> grow -> .grid
  add_executable(mc_trace tools/mc_trace.c)
  target_link_libraries(mc_trace matter_compressor)

  add_executable(mc_fetch tools/mc_fetch.c tools/vendor/libs3/libs3.c)
  target_include_directories(mc_fetch PRIVATE tools/vendor/libs3 ${ZSTD_INC})
  set_property(TARGET mc_fetch PROPERTY C_STANDARD 23)
  target_link_libraries(mc_fetch CURL::libcurl ${ZSTD_LIB})

  # Offline zarr fixture: a tiny on-disk zarr v2 / raw-u8 tree the zarr +
  # volume read paths run against with NO network (no S3, no c3d encode dep).
  # A CTest "fixture" target builds the tree once; the offline zarr/volume
  # tests declare FIXTURES_REQUIRED zarr_fixture so ctest sequences them.
  add_executable(mc_make_fixture tests/support/mc_make_fixture.c)
  set(MC_FIXTURE_DIR ${CMAKE_BINARY_DIR}/fixture)
  add_test(NAME zarr_fixture_setup
           COMMAND mc_make_fixture ${MC_FIXTURE_DIR})
  set_tests_properties(zarr_fixture_setup PROPERTIES FIXTURES_SETUP zarr_fixture)

  # v2 blosc+zstd fixture -> the blosc_decode path (raw fixture is compressor:null).
  add_executable(mc_make_fixture_blosc tests/support/mc_make_fixture_blosc.c)
  target_link_libraries(mc_make_fixture_blosc ${ZSTD_LIB})
  target_include_directories(mc_make_fixture_blosc PRIVATE ${ZSTD_INC})
  set(MC_FIXTURE_BLOSC_DIR ${CMAKE_BINARY_DIR}/fixture_blosc)
  add_test(NAME zarr_blosc_fixture_setup
           COMMAND mc_make_fixture_blosc ${MC_FIXTURE_BLOSC_DIR})
  set_tests_properties(zarr_blosc_fixture_setup PROPERTIES FIXTURES_SETUP zarr_blosc_fixture)
  add_executable(mc_zarr_blosc_test tests/mc_zarr_test.c)
  set_property(TARGET mc_zarr_blosc_test PROPERTY C_STANDARD 23)
  target_link_libraries(mc_zarr_blosc_test matter_compressor)
  add_test(NAME zarr_blosc COMMAND mc_zarr_blosc_test ${MC_FIXTURE_BLOSC_DIR}/zarr/0)
  set_tests_properties(zarr_blosc PROPERTIES FIXTURES_REQUIRED zarr_blosc_fixture)

  add_executable(mc_zarr_test tests/mc_zarr_test.c)
  set_property(TARGET mc_zarr_test PROPERTY C_STANDARD 23)
  target_link_libraries(mc_zarr_test matter_compressor)
  add_test(NAME zarr COMMAND mc_zarr_test ${MC_FIXTURE_DIR}/zarr/0)
  set_tests_properties(zarr PROPERTIES FIXTURES_REQUIRED zarr_fixture)

  # Offline volume path (discovery -> local read -> transcode -> cache -> sample)
  # against the fixture. The original mc_volume_test below stays a real-network
  # probe (run manually / in the opt-in S3 lane), so it is NOT registered here.
  add_executable(mc_volume_offline_test tests/mc_volume_offline_test.c)
  target_link_libraries(mc_volume_offline_test matter_compressor)
  add_test(NAME volume_offline
           COMMAND mc_volume_offline_test ${MC_FIXTURE_DIR}/zarr ${CMAKE_BINARY_DIR})
  set_tests_properties(volume_offline PROPERTIES FIXTURES_REQUIRED zarr_fixture)

  # volume/zarr introspection + prefetch + LOD-render surface against the fixture
  # (get_level_meta, region_gen, request_region, prefetch_level/shard,
  # sample_lods, set_cache/staging_bytes, set_ready_cb, mc_mca_probe).
  add_executable(mc_volume_api_test tests/mc_volume_api_test.c)
  target_link_libraries(mc_volume_api_test matter_compressor)
  add_test(NAME volume_api
           COMMAND mc_volume_api_test ${MC_FIXTURE_DIR}/zarr ${CMAKE_BINARY_DIR})
  set_tests_properties(volume_api PROPERTIES FIXTURES_REQUIRED zarr_fixture)

  add_executable(mc_volume_test tests/mc_volume_test.c)
  set_property(TARGET mc_volume_test PROPERTY C_STANDARD 23)
  target_link_libraries(mc_volume_test matter_compressor)

  add_executable(mc_stream_probe tools/mc_stream_probe.c)
  set_property(TARGET mc_stream_probe PROPERTY C_STANDARD 23)
  target_link_libraries(mc_stream_probe matter_compressor CURL::libcurl)
endif()

