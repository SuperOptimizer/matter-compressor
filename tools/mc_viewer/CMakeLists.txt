# mc_viewer — integrated SDL3 + Nuklear slice viewer for matter-compressor.
#
# Opt-in: enable with -DMC_BUILD_VIEWER=ON. Deliberately isolated from the core
# build so SDL3 (pulled from git via FetchContent) and the Nuklear single-header
# never touch the codec target, the tests, or CI by default.
#
# Rendering model (milestone 1): the EXISTING software slice renderer
# (mc_render_plane / mc_render_plane_lod -> u8 -> mc_colormap_apply -> ARGB32)
# produces the pixels on the CPU; an SDL_Renderer STREAMING texture is updated
# with that ARGB buffer each frame and drawn (scaled for zoom/pan) into the
# window. SDL_Renderer is itself hardware-accelerated and backend-agnostic (it
# selects Vulkan/Metal/D3D/GL automatically), so this gives a working,
# cross-platform viewer with zero shader blobs to ship.
#
# Milestone 2 will replace the blit with a hand-written SDL_GPU pipeline (and,
# eventually, GPU-side sampling) once we add precompiled shaders. SDL_GPU is the
# abstraction over Vulkan/Metal/D3D12 + an OpenGL path, so that follows the
# GL -> Vulkan -> Metal -> DX12 progression without a per-backend rewrite.

include(FetchContent)

# SDL3 pinned to a specific commit for reproducibility ("latest git" intent,
# but locked so the build is deterministic). Bump SDL3_GIT_TAG to advance.
set(SDL3_GIT_TAG "0e2e605593716059702c1a1ae8f52f4434ada18d"
    CACHE STRING "SDL3 git commit to build the viewer against")
FetchContent_Declare(
  SDL3
  GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
  GIT_TAG        ${SDL3_GIT_TAG}
  GIT_SHALLOW    FALSE
)
# Keep SDL's own extras out of our build.
set(SDL_TEST       OFF CACHE BOOL "" FORCE)
set(SDL_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(SDL_SHARED     ON  CACHE BOOL "" FORCE)
set(SDL_STATIC     OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(SDL3)

add_executable(mc_viewer
  ${CMAKE_CURRENT_SOURCE_DIR}/mc_viewer.c
  ${CMAKE_CURRENT_SOURCE_DIR}/nuklear_impl.c)

target_include_directories(mc_viewer PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor)

# The viewer is interactive glue, not the hot codec path: build it at -O2 and
# do NOT inherit -march=native / -ffp-contract=off (those exist for archive
# determinism, which the viewer does not produce).
target_compile_options(mc_viewer PRIVATE -O2)

target_link_libraries(mc_viewer PRIVATE matter_compressor SDL3::SDL3)

# Nuklear + the viewer's own immediate-mode UI pull in <math.h>; nothing else.
set_property(TARGET mc_viewer PROPERTY C_STANDARD 11)
