cmake_minimum_required(VERSION 3.16)
project(matter_compressor C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math")
add_compile_options(-Wall -Wno-stringop-overflow)
if(NOT DEFINED MC_NO_NATIVE)
  add_compile_options(-march=native)
endif()

# core library: codec + archive (no external deps; just libm)
add_library(matter_compressor STATIC
  src/mc_codec.c
  src/mc_archive.c)
target_include_directories(matter_compressor PUBLIC src)
target_link_libraries(matter_compressor m)

# round-trip test
enable_testing()
add_executable(mc_roundtrip tests/mc_roundtrip.c)
target_link_libraries(mc_roundtrip matter_compressor)
add_test(NAME roundtrip COMMAND mc_roundtrip)
