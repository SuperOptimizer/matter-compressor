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
  set(CMAKE_C_FLAGS_RELEASE "-O3 -ffp-contract=off")
  add_compile_options(-Wall -Wno-stringop-overflow)
  if(NOT DEFINED MC_NO_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()

# DETERMINISM: archives are byte-identical across ISAs and compilers
# (CI-enforced). Two requirements: no -ffast-math (per-target reciprocal
# tricks in the float air-fill diverge; zero measured speedup anyway) and an
# explicit -ffp-contract=off (gcc contracts mul+add to FMA by default, clang
# does not — pin one choice). Hot paths are integer by design, so neither
# flag costs performance.

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
# in-tree c3d.c + vendored libs3.c, compiled into the target).
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
find_library(ZSTD_LIB zstd REQUIRED HINTS /opt/homebrew/lib /usr/local/lib)
find_path(ZSTD_INC zstd.h REQUIRED HINTS /opt/homebrew/include /usr/local/include)
add_library(matter_compressor STATIC
  src/matter_compressor.c
  src/c3d.c
  tools/vendor/libs3/libs3.c)
target_include_directories(matter_compressor
  PUBLIC src PRIVATE tools/vendor/libs3 ${ZSTD_INC})
set_target_properties(matter_compressor PROPERTIES POSITION_INDEPENDENT_CODE ON)
# c3d decode output feeds the .mca re-encode, so honor the archive
# determinism policy: no -ffast-math, explicit -ffp-contract=off.
set_source_files_properties(src/c3d.c PROPERTIES COMPILE_OPTIONS "-O3;-ffp-contract=off;-fno-openmp")
target_link_libraries(matter_compressor CURL::libcurl ${ZSTD_LIB} m Threads::Threads)

# The codec is the hot path; build it -O3 -march=native regardless of
# how a parent project (vendoring via add_subdirectory) sets global flags. The
# MC_STANDALONE block above already does this for standalone builds; this target-
# scoped copy guarantees it when vendored too. -DMC_NO_NATIVE=ON for portable builds.
target_compile_options(matter_compressor PRIVATE -O3 -Wno-stringop-overflow)
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

  # archive repacker: Morton order + clustered index for streaming + offline
  add_executable(mc_export tools/mc_export.c)
  target_include_directories(mc_export PRIVATE src tools/vendor/libs3 ${ZSTD_INC})
  target_sources(mc_export PRIVATE src/c3d.c tools/vendor/libs3/libs3.c)
  target_link_libraries(mc_export CURL::libcurl ${ZSTD_LIB} m Threads::Threads)

  # tools: benchmark harness (no extra deps) + zarr fetcher (libcurl + libzstd).
  add_executable(mc_bench tools/mc_bench.c)
  target_link_libraries(mc_bench matter_compressor)

  # fysics-style aggressive air masking preprocessor (no deps)
  add_executable(mc_mask tools/mc_mask.c)

  add_executable(mc_verify tools/mc_verify.c)
  target_link_libraries(mc_verify matter_compressor)

  add_executable(mc_fetch tools/mc_fetch.c tools/vendor/libs3/libs3.c)
  target_include_directories(mc_fetch PRIVATE tools/vendor/libs3 ${ZSTD_INC})
  set_property(TARGET mc_fetch PROPERTY C_STANDARD 23)
  target_link_libraries(mc_fetch CURL::libcurl ${ZSTD_LIB})

  add_executable(mc_zarr_test tests/mc_zarr_test.c)
  set_property(TARGET mc_zarr_test PROPERTY C_STANDARD 23)
  target_link_libraries(mc_zarr_test matter_compressor)

  add_executable(mc_volume_test tests/mc_volume_test.c)
  set_property(TARGET mc_volume_test PROPERTY C_STANDARD 23)
  target_link_libraries(mc_volume_test matter_compressor)
endif()
