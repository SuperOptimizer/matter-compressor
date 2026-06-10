// ============================================================================
// mc_cache.h — in-RAM decoded-block cache for interactive clients.
//
// A thin layer in front of the matter-compressor decode APIs: caches decoded
// 16^3 blocks (4 KB each, exactly one page) keyed by (lod, bz, by, bx) in an
// mmap-backed arena with CLOCK/NRU eviction. Designed for vc3d-style
// rendering clients: many reader threads, high RAM budgets (tens of GB),
// hot viewport revisits must not re-decode.
//
//   - sharded open-addressing hash (64 shards, one mutex + one clock hand
//     each) — readers on different shards never contend,
//   - arena slots are partitioned per shard, so eviction sweeps are local,
//   - get() decodes on miss via a caller-supplied source callback (bindings
//     for mc_archive and mc_reader below), with a post-decode re-check so
//     concurrent misses of the same block insert once,
//   - returned pointers are non-owning views into the arena. A slot can be
//     reused by a later eviction; a stale frame is possible, a UAF is not
//     (same contract as vc's BlockCache). Use mc_cache_get_copy when the
//     caller needs a stable snapshot.
//
// THREAD-SAFETY CONTRACT: all operations are safe from any number of threads
// (per-shard mutexes; decode runs outside the locks with a double-checked
// insert). mc_cache_get's pointer may be overwritten by a concurrent
// eviction (torn read, never UAF) — use mc_cache_get_copy where that
// matters. Writers that REPLACE a chunk in the source must call
// mc_cache_invalidate_chunk afterwards; the mc_reader binding serializes
// decode (mc_reader is single-threaded), the mc_archive binding does not.
// ============================================================================
#ifndef MC_CACHE_H
#define MC_CACHE_H
#include <stdint.h>
#include <stddef.h>
#include "mc_codec.h"

typedef struct mc_cache mc_cache;

// Source callback: decode block (bz,by,bx) of `lod` into dst (16^3 bytes).
// Called outside any cache lock; must be thread-safe (mc decode is).
typedef void (*mc_cache_src_fn)(void *ud, int lod, int bz, int by, int bx, mc_u8 *dst);

// Eviction policy. CLOCK = classic NRU sweep. S3FIFO (default) = small/main
// FIFO queues + ghost table (SOSP'23): one-hit wonders die in the small queue,
// re-referenced ghosts go straight to main — scan-resistant, which matters for
// render loops (slice sweeps re-touch everything once per frame).
typedef enum { MC_CACHE_S3FIFO = 0, MC_CACHE_CLOCK = 1 } mc_cache_policy;

// Create a cache with ~`bytes` of block storage (rounded to slots of 4 KB).
mc_cache *mc_cache_new(size_t bytes, mc_cache_src_fn src, void *src_ud);
// Switch eviction policy (call before first use; clears nothing).
void mc_cache_set_policy(mc_cache *c, mc_cache_policy p);
void      mc_cache_free(mc_cache *c);

// Get the decoded block, from cache or by decoding (and caching) it.
// Returns a non-owning pointer to 4096 bytes. Never NULL.
const mc_u8 *mc_cache_get(mc_cache *c, int lod, int bz, int by, int bx);

// Like get, but memcpy into caller storage under the shard lock — immune to
// concurrent eviction reuse.
void mc_cache_get_copy(mc_cache *c, int lod, int bz, int by, int bx, mc_u8 *dst);

// Peek without touching the recently-used bit (residency triage).
int  mc_cache_contains(mc_cache *c, int lod, int bz, int by, int bx);

// Prefetch every block of chunk (lod, cz,cy,cx) into the cache (decoding the
// blocks that are not already resident). Renderers call this from IO/worker
// threads ahead of the viewport so the render threads only ever hit.
void mc_cache_prefetch_chunk(mc_cache *c, int lod, int cz, int cy, int cx);

// Invalidate every cached block of chunk (lod, cz,cy,cx). Writers call this
// after re-appending/replacing a chunk so readers stop seeing stale blocks.
// Thread-safe against concurrent gets/inserts.
void mc_cache_invalidate_chunk(mc_cache *c, int lod, int cz, int cy, int cx);

// Drop everything (e.g. source archive replaced).
void mc_cache_clear(mc_cache *c);

// Diagnostics.
typedef struct { uint64_t hits, misses, evictions; size_t slots, used; } mc_cache_stats;
void mc_cache_get_stats(mc_cache *c, mc_cache_stats *out);

// ---- ready-made source bindings ----
struct mc_archive; struct mc_reader;
// Bind to a local archive handle (lock-free concurrent decode).
mc_cache *mc_cache_new_archive(size_t bytes, struct mc_archive *a);
// Bind to a reader (flat or streaming). NOTE: mc_reader decode is not
// thread-safe across threads sharing one reader; this binding serializes
// decode with an internal mutex. For maximum parallelism, give each thread
// its own reader and use mc_cache_new with a custom callback.
mc_cache *mc_cache_new_reader(size_t bytes, struct mc_reader *r);

#endif
