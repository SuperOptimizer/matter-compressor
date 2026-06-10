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
  set(CMAKE_C_FLAGS_RELEASE "-O3")
  add_compile_options(-Wall -Wno-stringop-overflow)
  if(NOT DEFINED MC_NO_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()

# DETERMINISM: archives are byte-identical across ISAs (CI-enforced) because
# the build is strict-IEEE — do NOT add -ffast-math: it licenses per-target
# reassociation/reciprocal tricks in the encoder's float air-fill and breaks
# bitstream identity, while buying zero measured performance (hot paths are
# integer by design; the fill vectorizes fine under strict semantics).

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

# core library: codec + archive (deps: libm + pthread for the appendable writer)
add_library(matter_compressor STATIC
  src/matter_compressor.c)
target_include_directories(matter_compressor PUBLIC src)
set_target_properties(matter_compressor PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)
target_link_libraries(matter_compressor m Threads::Threads)

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

  # archive repacker: Morton order + clustered index for streaming + offline
  add_executable(mc_export tools/mc_export.c)
  target_include_directories(mc_export PRIVATE src)
  target_link_libraries(mc_export m Threads::Threads)

  # tools: benchmark harness (no extra deps) + zarr fetcher (libcurl + libzstd).
  add_executable(mc_bench tools/mc_bench.c)
  target_link_libraries(mc_bench matter_compressor)

  # fysics-style aggressive air masking preprocessor (no deps)
  add_executable(mc_mask tools/mc_mask.c)

  add_executable(mc_verify tools/mc_verify.c)
  target_link_libraries(mc_verify matter_compressor)

  find_package(CURL QUIET)
  find_library(ZSTD_LIB_S3 zstd HINTS /opt/homebrew/lib /usr/local/lib)
  if(CURL_FOUND)
    # optional S3/HTTP open glue (separate lib: core stays dependency-free)
    add_library(matter_compressor_s3 STATIC src/mc_s3.c tools/vendor/libs3/libs3.c)
    target_include_directories(matter_compressor_s3 PUBLIC src PRIVATE tools/vendor/libs3)
    set_property(TARGET matter_compressor_s3 PROPERTY C_STANDARD 23)
    target_link_libraries(matter_compressor_s3 matter_compressor CURL::libcurl)
  endif()

  find_library(ZSTD_LIB zstd HINTS /opt/homebrew/lib /usr/local/lib)
  find_path(ZSTD_INC zstd.h HINTS /opt/homebrew/include /usr/local/include)
  if(CURL_FOUND AND ZSTD_LIB AND ZSTD_INC)
    add_executable(mc_fetch tools/mc_fetch.c tools/vendor/libs3/libs3.c)
    target_include_directories(mc_fetch PRIVATE tools/vendor/libs3 ${ZSTD_INC})
    set_property(TARGET mc_fetch PROPERTY C_STANDARD 23)
    target_link_libraries(mc_fetch CURL::libcurl ${ZSTD_LIB})
  else()
    message(STATUS "mc_fetch skipped (need libcurl + libzstd)")
  endif()
endif()
