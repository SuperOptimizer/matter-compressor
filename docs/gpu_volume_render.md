# Real-time GPU volumetric rendering (design)

Goal: **interactive 3D volume rendering** of matter-compressor volumes — orbit
the camera, spin the volume, adjust the transfer function, all at 30–60 fps —
the VTK-style experience, but fed by the c3g-on-GPU decode path. This is GPU
raycasting (ray-march + composite per pixel), not the existing offline CPU
march.

This is the target the SDL_GPU + c3g work has been building toward. It is a
large feature; this doc scopes it before implementation.

## What already exists (and is reused)

- **The algorithm, on CPU.** `mc_render_params` + `MC_COMP_*` already define a
  real emission-absorption ray march (`MC_COMP_SHADED`: Beer-Lambert
  `a = 1-exp(-sigma·dt)`, front-to-back, gradient lighting), MIP (`MC_COMP_MAX`),
  percentile, first-hit depth, ink/PBR. The GPU raycaster MIRRORS these modes —
  the math and parameters are proven; the port is to a shader.
- **The GPU data path.** Compressed c3g blocks → VRAM → `c3g_decode.comp`
  (compute decode) → decoded voxels in VRAM → sampled in a fragment shader
  (`vol_slice.frag`). The raycaster swaps the slice sampler for a ray-march loop
  over the same decoded data. The hard part (compressed→GPU→decoded) is done and
  validated (`vol_gpu_check`).
- **The frame plumbing.** `mc_gpu_frame_begin` / `mc_gpu_vol_render` (compute
  pass + draw pass) / `mc_gpu_frame_end_nuklear` compose a GPU-drawn image with
  the Nuklear UI. The raycaster is a new draw pass in the same structure.

## What is new

1. **Camera.** A 3D orbit/trackball camera (azimuth/elevation/distance, or a
   quaternion), perspective or orthographic, panning. The viewer is currently
   axis-aligned slices only — this is the main new interaction.
2. **Per-pixel ray setup.** Fullscreen pass; for each pixel build a ray
   (origin, dir) in volume space from the inverse view-projection, intersect the
   volume's [0,dims] AABB → [tmin, tmax] (slab method). Skip pixels that miss.
3. **Ray-march + composite (the core shader).** From tmin to tmax in steps of
   `dt` voxels: trilinear-sample the volume, map value→(rgb,a) through a transfer
   function, composite front-to-back (`C += (1-A)·a·rgb; A += (1-A)·a`), early-
   terminate when `A > 0.99`. MIP/first-hit/percentile are variants of the same
   loop.
4. **Transfer function.** A 1D (later 2D value×gradient) RGBA LUT, editable in
   the Nuklear UI (control points → baked 256-entry texture). Replaces the
   current grayscale slice LUT.
5. **Gradient lighting (optional, second milestone).** Central-difference
   gradient as the normal; Blinn-Phong like the CPU `MC_COMP_SHADED`.

## The residency decision (the crux for real-time + scale)

A raycast touches a **3D region**, not the thin 16-voxel slab the slice path
gathers. Two coupled choices:

### Decoded storage: dense 3D texture, not the block-major buffer

The current decoded buffer is block-major (`b*4096 + (lz,ly,lx)`), so trilinear
(8 neighbors, possibly in 8 different blocks) means 8 index computations per
sample — too slow for a per-step ray march. **Decode into a dense 3D texture**
instead: free hardware trilinear, 3D cache locality, one `texture()` per sample.
`c3g_decode.comp` already produces decoded voxels; change its output target from
the flat buffer to a 3D storage image (`COMPUTE_STORAGE_WRITE`, also `SAMPLER`),
writing block b's voxels to their (x,y,z) texels. The raycaster samples that
`sampler3D`.

### Residency model: bounded brick cache, paged per camera

- **Milestone A (prove it):** one resident brick — decode a bounded region
  (e.g. 256³ or whatever fits a single 3D texture) and raycast it. No paging.
  Proves camera + ray-march + transfer function + lighting end-to-end.
- **Milestone B (scale):** a **brick cache** — the volume is bricks (e.g. 256³);
  each camera, compute the visible/front-most bricks (AABB vs frustum + a depth
  budget), page-in their compressed c3g blocks → GPU-decode into free 3D-texture
  cache slots (LRU), and the raycaster walks a small page-table indirection
  (brick coord → cache slot) per step. Empty-space skipping via per-brick
  min/max (skip air bricks — the c3g all-air blocks make this cheap).
- **Multi-resolution / LOD:** sample a coarser LOD as the ray gets far / per
  vox-per-pixel (reuse `mc_render_pick_lod`'s logic). Keeps distant volume cheap.

## Where SDL_GPU pinches (and the decision point)

Milestone A and a basic Milestone B fit SDL_GPU cleanly (3D storage textures,
compute, a page-table buffer the shader indexes). The features SDL_GPU does NOT
expose start to matter only at large out-of-core scale:

- **Sparse/virtual 3D textures** — the natural fit for volumes ≫ VRAM. SDL_GPU
  has none; we emulate with an explicit brick cache + page table (more shader
  work, but a portable and well-proven technique).
- **Bindless** — a huge array of brick textures indexed per-step. SDL_GPU's
  fixed slots push us to ONE big 3D-texture atlas (brick cache) addressed by the
  page table — which is the standard approach anyway, so not a real loss.
- **Async compute** — overlap decode (compute) with raycast (graphics) on
  separate queues. SDL_GPU serializes on one queue; a perf ceiling, not a
  correctness one.

**Recommendation:** build Milestones A and B on SDL_GPU (keeps Vulkan/Metal/
D3D12 for free). Only consider a targeted raw-Vulkan raycast subsystem if/when
out-of-core sparse residency becomes the bottleneck — and even then, keep the
decode + the rest on SDL_GPU.

## Validation strategy

Mirror the slice approach: a headless harness (`raycast_gpu_check`) renders a
known synthetic volume from a fixed camera with MIP (deterministic, no transfer-
function ambiguity) and compares against a CPU `mc_render_*`/reference march of
the same rays — within a tight tolerance (trilinear + step quantization). MIP
first (exact-ish), then emission-absorption against the CPU `MC_COMP_SHADED`.

## Milestones (each independently landable + verifiable)

1. **3D-texture decode target** — `c3g_decode.comp` variant writing a dense 3D
   texture; a `vol3d_gpu_check` confirming it equals the CPU decode. (Small,
   reuses everything.)
2. **Orbit camera + ray setup + MIP raycast** over one resident brick; headless
   MIP check vs CPU. First real 3D image.
3. **Transfer function** (1D RGBA LUT + Nuklear editor) + emission-absorption
   composite; check vs CPU `MC_COMP_SHADED`.
4. **Gradient lighting.**
5. **Brick cache + paging + empty-space skip + LOD** — the out-of-core step.

Milestones 1–4 are "real-time 3D volumetric rendering" of a brick-sized volume.
5 makes it scale to full scrolls.
