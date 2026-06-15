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

# Shaders: GLSL -> SPIR-V via glslangValidator. The generated .spv.h headers are
# COMMITTED, so a checkout without a shader compiler still builds. When
# glslangValidator IS present, regenerate them at build time so edits to the
# .glsl sources take effect (a custom command per shader, fed into the target).
set(MC_VIEWER_SHADERS
  blit.vert:vert blit.frag:frag ui.vert:vert ui.frag:frag
  c3g_decode.comp:comp vol.vert:vert vol_slice.frag:frag
  c3g_decode3d.comp:comp raycast.vert:vert raycast.frag:frag)
find_program(GLSLANG_VALIDATOR glslangValidator)
set(_shader_hdrs "")
if(GLSLANG_VALIDATOR)
  foreach(entry ${MC_VIEWER_SHADERS})
    string(REPLACE ":" ";" parts ${entry})
    list(GET parts 0 name)
    list(GET parts 1 stage)
    string(REPLACE "." "_" vn ${name})
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${name}.glsl)
    set(hdr ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${name}.spv.h)
    add_custom_command(
      OUTPUT ${hdr}
      COMMAND ${GLSLANG_VALIDATOR} -V -S ${stage} ${src} --vn ${vn}_spv -o ${hdr}
      DEPENDS ${src}
      COMMENT "glslang ${name}.glsl -> ${name}.spv.h")
    list(APPEND _shader_hdrs ${hdr})
  endforeach()
endif()

add_executable(mc_viewer
  ${CMAKE_CURRENT_SOURCE_DIR}/mc_viewer.c
  ${CMAKE_CURRENT_SOURCE_DIR}/nuklear_impl.c
  ${_shader_hdrs})

target_include_directories(mc_viewer PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor)

# The viewer is interactive glue, not the hot codec path: build it at -O2 and
# do NOT inherit -march=native / -ffp-contract=off (those exist for archive
# determinism, which the viewer does not produce).
target_compile_options(mc_viewer PRIVATE -O2)

target_link_libraries(mc_viewer PRIVATE matter_compressor SDL3::SDL3)

# Nuklear + the viewer's own immediate-mode UI pull in <math.h>; nothing else.
set_property(TARGET mc_viewer PROPERTY C_STANDARD 11)

# c3g_gpu_check — validates the GPU compute decoder against the CPU oracle
# (mc_c3g_dec_block). Headless (SDL_VIDEODRIVER=offscreen). Built with the
# viewer so it shares the SDL3 dependency.
add_executable(c3g_gpu_check
  ${CMAKE_CURRENT_SOURCE_DIR}/c3g_gpu_check.c
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/c3g_decode.comp.spv.h)
target_include_directories(c3g_gpu_check PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(c3g_gpu_check PRIVATE -O2)
target_link_libraries(c3g_gpu_check PRIVATE matter_compressor SDL3::SDL3 m)
set_property(TARGET c3g_gpu_check PROPERTY C_STANDARD 11)

# vol_gpu_check — validates the full GPU vol path: upload compressed c3g slab ->
# GPU compute-decode -> GPU slice-sample -> readback vs CPU reference. Headless.
add_executable(vol_gpu_check
  ${CMAKE_CURRENT_SOURCE_DIR}/vol_gpu_check.c
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/c3g_decode.comp.spv.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vol.vert.spv.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vol_slice.frag.spv.h)
target_include_directories(vol_gpu_check PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(vol_gpu_check PRIVATE -O2)
target_link_libraries(vol_gpu_check PRIVATE matter_compressor SDL3::SDL3 m)
set_property(TARGET vol_gpu_check PROPERTY C_STANDARD 11)

# ray_gpu_check — validates (M1) c3g decode into a dense 3D texture and (M2) the
# MIP raycaster, both vs a CPU reference. Headless.
add_executable(ray_gpu_check
  ${CMAKE_CURRENT_SOURCE_DIR}/ray_gpu_check.c
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/c3g_decode3d.comp.spv.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/raycast.vert.spv.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/raycast.frag.spv.h)
target_include_directories(ray_gpu_check PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(ray_gpu_check PRIVATE -O2)
target_link_libraries(ray_gpu_check PRIVATE matter_compressor SDL3::SDL3 m)
set_property(TARGET ray_gpu_check PROPERTY C_STANDARD 11)
