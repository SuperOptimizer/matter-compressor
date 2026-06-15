# mc_viewer

Integrated slice viewer for matter-compressor — SDL3 + [Nuklear](https://github.com/Immediate-Mode-UI/Nuklear).

A thin interactive client over the public `matter_compressor.h` API: it opens a
volume (local `.mca` / zarr directory, or remote `s3://` / `https://`), renders
an axis-aligned slice with the existing LOD-matched software renderer
(`mc_render_plane_lod`), colormaps it, and blits the result.

## Rendering model

The slice pixels are produced **on the CPU** by the core renderer, then drawn
through a hand-written **SDL_GPU** pipeline:

```
mc_render_plane_lod -> u8 slice -> mc_colormap_apply -> ARGB32
   -> SDL_GPU texture (transfer buffer + copy pass)
   -> fullscreen-quad pipeline (zoom/pan in the vertex shader)
   -> Nuklear UI drawn in the same render pass (its own SDL_GPU backend)
```

SDL_GPU is the abstraction over Vulkan / Metal / D3D12 (+ an OpenGL fallback),
so this is the cross-platform GPU path on the GL → Vulkan → Metal → DX12
roadmap. There is **no SDL_Renderer** anywhere — both the slice and the entire
Nuklear UI (font-atlas texture, per-frame vertex/index buffers, per-command
scissor) run on SDL_GPU. See `mc_gpu.h`.

### Shaders

GLSL sources live in `shaders/` and are compiled to SPIR-V with
`glslangValidator` and embedded as `.spv.h` byte arrays. The generated headers
are **committed**, so a checkout without a shader compiler still builds; when
`glslangValidator` is found, CMake regenerates them at build time so edits to
the `.glsl` sources take effect.

| shader | role |
|--------|------|
| `blit.vert/frag` | fullscreen quad: sample the slice texture, gain/bias |
| `ui.vert/frag`   | Nuklear: ortho-projected pos/uv/color, atlas modulate |

Next: GPU-side volume sampling (push decoded blocks into GPU textures and
sample/raycast in a compute/fragment shader) instead of CPU `mc_render_plane`.

## Build

Opt-in; off by default so the SDL3 dependency never touches the core build/CI:

```sh
cmake -S . -B build_viewer -DMC_BUILD_VIEWER=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_viewer --target mc_viewer -j
```

SDL3 is fetched from git (pinned commit, see `SDL3_GIT_TAG` in
`tools/mc_viewer/CMakeLists.txt`) and built from source via `FetchContent` — no
system SDL3 install needed. Nuklear is the vendored single header in `vendor/`.

## Run

```sh
./build_viewer/tools/mc_viewer/mc_viewer <url-or-path> [cache_dir]
```

- `url-or-path`: `.mca`, a zarr multiscales directory, `s3://…`, or `https://…`
- `cache_dir`: local cache for streamed volumes (default `./mc_cache`; created
  if absent)

### Controls

| Input | Action |
|-------|--------|
| panel `Z/Y/X` | choose slice axis |
| panel `slice` | slice index along the axis |
| panel `zoom` / mouse wheel | magnification |
| panel `win low/high` | window/level (display range) |
| panel colormap combo | gray / viridis / magma / fire / r/g/b/c/m |
| right-drag | pan |

### Headless verification

- `MC_VIEWER_DUMP=out.ppm` — render one slice (blocking sample so the data is
  present), write it as a PPM, and exit. Exercises the CPU render path; no
  swapchain needed.
- `MC_VIEWER_FRAMES=N` — render N full GPU frames (device, pipelines, swapchain
  acquire, slice quad + Nuklear draws, submit) then exit. With
  `SDL_VIDEODRIVER=offscreen` this is a headless GPU smoke test.
