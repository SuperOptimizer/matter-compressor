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
  set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math")
  add_compile_options(-Wall -Wno-stringop-overflow)
  if(NOT DEFINED MC_NO_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()

# core library: codec + archive (deps: libm + pthread for the appendable writer)
add_library(matter_compressor STATIC
  src/mc_codec.c
  src/mc_archive.c)
target_include_directories(matter_compressor PUBLIC src)
set_target_properties(matter_compressor PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)
target_link_libraries(matter_compressor m Threads::Threads)

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
endif()
