# mc_viewer

Integrated slice viewer for matter-compressor — SDL3 + [Nuklear](https://github.com/Immediate-Mode-UI/Nuklear).

A thin interactive client over the public `matter_compressor.h` API: it opens a
volume (local `.mca` / zarr directory, or remote `s3://` / `https://`), renders
an axis-aligned slice with the existing LOD-matched software renderer
(`mc_render_plane_lod`), colormaps it, and blits the result.

## Rendering model (milestone 1)

The slice pixels are produced **on the CPU** by the core renderer:

```
mc_render_plane_lod  ->  u8 slice  ->  mc_colormap_apply  ->  ARGB32
        -> SDL_Renderer STREAMING texture -> scaled blit (zoom/pan)
```

`SDL_Renderer` is itself hardware-accelerated and backend-agnostic (it selects
Vulkan / Metal / D3D / GL automatically), so this gives a working,
cross-platform viewer with **zero shader blobs to ship**.

Milestone 2 will replace the blit with a hand-written `SDL_GPU` pipeline (and,
eventually, GPU-side sampling) once precompiled shaders are in place. SDL_GPU is
the abstraction over Vulkan/Metal/D3D12 + an OpenGL path, so that follows the
GL → Vulkan → Metal → DX12 progression without a per-backend rewrite.

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

`MC_VIEWER_DUMP=out.ppm` renders one slice (blocking sample so the data is
present), writes it as a PPM, and exits — used for headless verification.
