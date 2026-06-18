// mc_cache.h — cache internals shared with mc_volume (which drives the cache's
// per-shard fill machinery directly). mc_cache.c keeps its own private copies of
// the inline helpers; this header hands the same definitions to mc_volume.c.
#ifndef MC_CACHE_H
#define MC_CACHE_H

#include "matter_compressor.h"
#include <stdint.h>

#define NSHARD    64       // cache shard count (power of two)
#define MISSQ_CAP 65536    // miss-queue capacity

// block key: 1 (guard) | lod(3) | bz(20) | by(20) | bx(20)
static inline uint64_t bkey(int lod, int bz, int by, int bx) {
    return (1ull << 63) | ((uint64_t)(lod & 7) << 60)
         | ((uint64_t)(bz & 0xFFFFF) << 40) | ((uint64_t)(by & 0xFFFFF) << 20)
         | (uint64_t)(bx & 0xFFFFF);
}
static inline uint64_t khash(uint64_t k) {
    k ^= k >> 33; k *= 0xFF51AFD7ED558CCDull; k ^= k >> 33;
    k *= 0xC4CEB9FE1A85EC53ull; k ^= k >> 33;
    return k;
}

// decode+install one block into the cache (defined in mc_cache.c).
int cache_fill_one(struct mc_cache *c, int lod, int bz, int by, int bx);

#endif  // MC_CACHE_H
