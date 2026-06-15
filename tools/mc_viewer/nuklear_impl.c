/* nuklear_impl.c — single TU compiling the Nuklear library implementation and
 * the SDL_GPU frontend implementation (slice blit + Nuklear backend). Kept in
 * its own TU so the heavy NK_IMPLEMENTATION + mc_gpu code builds once and
 * mc_viewer.c stays fast to recompile. */

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_IMPLEMENTATION
#include "nuklear.h"

#define MC_GPU_IMPLEMENTATION
#include "mc_gpu.h"
