/* nuklear_impl.c — the single translation unit that compiles the Nuklear
 * library implementation and the SDL3 backend implementation. Keeping the
 * heavy NK_IMPLEMENTATION in its own TU (built once) keeps mc_viewer.c fast to
 * recompile during iteration. */

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_IMPLEMENTATION
#include "nuklear.h"

#define NK_SDL3_IMPLEMENTATION
#include "nk_sdl3.h"
