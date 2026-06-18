#define _GNU_SOURCE
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "matter_compressor.h"
#include "mc_codec.h"
#include "mc_archive.h"
typedef uint8_t u8;
// ============================================================================
// mc_zarr — zarr v3-sharded-c3d + v2-flat reader (dep: zstd)
// ============================================================================
#include <zstd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <zstd.h>

// ---------------------------------------------------------------------------
// tiny JSON scraping — enough for zarr.json / .zarray (no general parser).
// ---------------------------------------------------------------------------

// First integer array of `n` ints after the literal `"key"`. -1 on miss.
static int json_int_array(const char *j, const char *key, long out[], int n) {
    const char *p = strstr(j, key);
    if (!p) return -1;
    p = strchr(p, '[');
    if (!p) return -1;
    ++p;
    for (int i = 0; i < n; ++i) {
        char *end;
        out[i] = strtol(p, &end, 10);
        if (end == p) return -1;
        p = end;
        while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') ++p;
    }
    return 0;
}

// Is `needle` present anywhere in `j` (substring)?
static int json_has(const char *j, const char *needle) {
    return strstr(j, needle) != NULL;
}

// ---------------------------------------------------------------------------
// blosc1 decode (shuffle=0, typesize=1) — lifted from tools/mc_fetch.c.
// header: ver verlz flags typesize nbytes(4) blocksize(4) cbytes(4), LE.
// ---------------------------------------------------------------------------
static uint8_t *blosc_decode(const uint8_t *src, size_t srclen, size_t *out_len) {
    if (srclen < 16) return NULL;
    uint8_t flags = src[2];
    uint32_t nbytes, blocksize, cbytes;
    memcpy(&nbytes, src + 4, 4);
    memcpy(&blocksize, src + 8, 4);
    memcpy(&cbytes, src + 12, 4);
    if (cbytes > srclen || !nbytes) return NULL;
    if (flags & 0x1 || flags & 0x4) { fprintf(stderr, "mc_zarr: blosc shuffle unsupported\n"); return NULL; }
    uint8_t *out = malloc(nbytes);
    if (!out) return NULL;
    if (flags & 0x2) {                              // memcpyed: raw payload follows header
        if (16 + (size_t)nbytes > srclen) { free(out); return NULL; }
        memcpy(out, src + 16, nbytes);
        *out_len = nbytes;
        return out;
    }
    uint32_t nblocks = (nbytes + blocksize - 1) / blocksize;
    const uint8_t *bstarts = src + 16;
    if (16 + (size_t)nblocks * 4 > srclen) { free(out); return NULL; }
    size_t off = 0;
    for (uint32_t b = 0; b < nblocks; ++b) {
        uint32_t bs;
        memcpy(&bs, bstarts + (size_t)b * 4, 4);
        uint32_t neblock = (b == nblocks - 1) ? nbytes - b * blocksize : blocksize;
        if ((size_t)bs + 4 > srclen) { free(out); return NULL; }
        int32_t cb;
        memcpy(&cb, src + bs, 4);
        const uint8_t *payload = src + bs + 4;
        if (cb <= 0 || (size_t)bs + 4 + (size_t)cb > srclen) { free(out); return NULL; }
        if ((uint32_t)cb == neblock) {
            memcpy(out + off, payload, neblock);    // stored uncompressed
        } else {
            size_t got = ZSTD_decompress(out + off, neblock, payload, (size_t)cb);
            if (ZSTD_isError(got) || got != neblock) { free(out); return NULL; }
        }
        off += neblock;
    }
    *out_len = nbytes;
    return out;
}

// ---------------------------------------------------------------------------
// mc_zarr handle
// ---------------------------------------------------------------------------
enum { ZV3 = 3, ZV2 = 2 };

// Cached shard index footer: the (offset,len) table is immutable per shard, so
// read the 64KB footer ONCE and reuse it for all of a shard's inner chunks.
// Without this, fetching a shard's 4096 chunks one-at-a-time re-reads 64KB *4096.
#define MC_FOOTER_CACHE 64
typedef struct {
    uint64_t shard_id;     // (sz<<40)|(sy<<20)|sx, ~0 = empty slot
    uint8_t *idx;          // malloc'd footer bytes (n_inner*16)
    uint64_t lru;          // last-use tick
} footer_ent;

struct mc_zarr {
    mc_zarr_read_fn read;
    void *ud;
    int version;            // ZV3 | ZV2
    int shape[3];           // voxels (z,y,x)
    int shard_edge;         // v3: chunk_grid chunk_shape; v2: == inner_edge
    int inner_edge;         // v3: sharding inner chunk; v2: .zarray chunks
    int inner_grid[3];      // ceil(shape/inner_edge) per axis (z,y,x)
    int per;                // inner chunks per shard axis = shard_edge/inner_edge
    char codec[16];         // "c3d" | "blosc" | "raw"
    char sep;               // v2 dimension separator ('.' default, or '/')

    pthread_mutex_t fmu;    // guards the footer cache
    footer_ent fcache[MC_FOOTER_CACHE];
    uint64_t ftick;
};

// fetch a whole object by key; returns malloc'd buf or NULL (sets *len).
// `err` (optional): *err=1 if the read FAILED (transient — retry, do NOT treat
// as air); *err=0 if the object is genuinely absent (clean 404 -> confirmed
// air) or present. Distinguishes z->read's -1 (error) from 0+NULL (404).
static uint8_t *fetch_all(mc_zarr *z, const char *key, size_t *len, int *err) {
    if (err) *err = 0;
    uint8_t *buf = NULL;
    size_t n = 0;
    if (z->read(z->ud, key, 0, 0, &buf, &n) < 0) {   // transient error
        free(buf); *len = 0; if (err) *err = 1; return NULL;
    }
    *len = n;                                         // 0 + NULL == clean 404 (air)
    return buf;
}

mc_zarr *mc_zarr_open(mc_zarr_read_fn read, void *ud) {
    if (!read) return NULL;
    size_t jl = 0;
    uint8_t *jb = NULL;
    int v3 = 1;
    if (read(ud, "zarr.json", 0, 0, &jb, &jl) < 0 || !jb || !jl) {
        free(jb);
        jb = NULL;
        jl = 0;
        v3 = 0;
        if (read(ud, ".zarray", 0, 0, &jb, &jl) < 0 || !jb || !jl) { free(jb); return NULL; }
    }
    char *j = malloc(jl + 1);
    if (!j) { free(jb); return NULL; }
    memcpy(j, jb, jl);
    j[jl] = 0;
    free(jb);

    mc_zarr *z = calloc(1, sizeof *z);
    if (!z) { free(j); return NULL; }
    pthread_mutex_init(&z->fmu, NULL);
    z->read = read;
    z->ud = ud;

    long shp[3];
    if (json_int_array(j, "\"shape\"", shp, 3) != 0) { free(j); free(z); return NULL; }
    z->shape[0] = (int)shp[0];
    z->shape[1] = (int)shp[1];
    z->shape[2] = (int)shp[2];

    if (v3) {
        z->version = ZV3;
        if (!json_has(j, "sharding_indexed")) { free(j); free(z); return NULL; }
        // chunk_grid.chunk_shape = shard edge (first int array after "chunk_grid").
        const char *cg = strstr(j, "\"chunk_grid\"");
        long shard[3];
        if (!cg || json_int_array(cg, "\"chunk_shape\"", shard, 3) != 0) { free(j); free(z); return NULL; }
        // sharding_indexed configuration.chunk_shape = inner edge (after that codec).
        const char *sh = strstr(j, "sharding_indexed");
        // chunk_shape appears in the sharding config BEFORE the codec name in the
        // emitted json; search from the codecs array start instead.
        const char *cc = strstr(j, "\"codecs\"");
        long inner[3];
        if (!cc || json_int_array(cc, "\"chunk_shape\"", inner, 3) != 0) { free(j); free(z); return NULL; }
        (void)sh;
        z->shard_edge = (int)shard[0];
        z->inner_edge = (int)inner[0];
        // v3-sharded zarr used the c3d inner codec, which has been removed.
        fprintf(stderr, "mc_zarr: v3-sharded zarr unsupported (c3d removed)\n");
        free(j); free(z); return NULL;
    } else {
        z->version = ZV2;
        long ch[3];
        if (json_int_array(j, "\"chunks\"", ch, 3) != 0) { free(j); free(z); return NULL; }
        z->inner_edge = (int)ch[0];
        z->shard_edge = (int)ch[0];           // a v2 chunk is its own shard
        // compressor: null -> raw, else blosc (the standardized scroll zarrs).
        if (json_has(j, "\"compressor\": null") || json_has(j, "\"compressor\":null"))
            snprintf(z->codec, sizeof z->codec, "raw");
        else snprintf(z->codec, sizeof z->codec, "blosc");
        // dimension_separator default '.'
        z->sep = json_has(j, "\"dimension_separator\": \"/\"") ? '/' : '.';
    }

    free(j);
    if (z->inner_edge <= 0 || z->shard_edge <= 0 || z->shard_edge % z->inner_edge) {
        free(z);
        return NULL;
    }
    // The transcode pipeline groups source inner-chunks into a fixed 256^3
    // region whose sub-chunk arrays (decode_item.raw/oz/oy/ox) are sized [8],
    // i.e. at most 2x2x2 = sub^3 sub-chunks. That bounds the inner edge to a
    // divisor of the 256 region giving sub = 256/edge in {1,2}: edge 128 or
    // 256. A smaller edge (e.g. 64 -> sub 4 -> 64 sub-chunks) would overflow
    // those [8] arrays (heap/stack smash) in mc_volume_prefetch_region /
    // prefetch_shard. Reject such zarrs at open rather than overflow later.
    // (256 is the MC_VOLUME region edge; see CHUNK in the volume section.)
    if (256 % z->inner_edge != 0 || (256 / z->inner_edge) > 2) {
        fprintf(stderr, "mc_zarr: inner edge %d unsupported "
                "(must be 128 or 256; the transcode region is 256^3 with <=2 "
                "sub-chunks per axis)\n", z->inner_edge);
        free(z);
        return NULL;
    }
    z->per = z->shard_edge / z->inner_edge;
    for (int d = 0; d < 3; ++d)
        z->inner_grid[d] = (z->shape[d] + z->inner_edge - 1) / z->inner_edge;
    return z;
}

void mc_zarr_free(mc_zarr *z) {
    if (!z) return;
    for (int i = 0; i < MC_FOOTER_CACHE; ++i) free(z->fcache[i].idx);
    pthread_mutex_destroy(&z->fmu);
    free(z);
}

void mc_zarr_shape(const mc_zarr *z, int *nz, int *ny, int *nx) {
    if (nz) *nz = z->shape[0];
    if (ny) *ny = z->shape[1];
    if (nx) *nx = z->shape[2];
}
int mc_zarr_inner_edge(const mc_zarr *z) { return z->inner_edge; }
int mc_zarr_shard_edge(const mc_zarr *z) { return z->shard_edge; }
const char *mc_zarr_inner_codec(const mc_zarr *z) { return z->codec; }
void mc_zarr_inner_grid(const mc_zarr *z, int *nz, int *ny, int *nx) {
    if (nz) *nz = z->inner_grid[0];
    if (ny) *ny = z->inner_grid[1];
    if (nx) *nx = z->inner_grid[2];
}

// ---------------------------------------------------------------------------
// keys
// ---------------------------------------------------------------------------

// v3 shard key for the shard containing global inner chunk (cz,cy,cx): "c/sz/sy/sx".
// v2 chunk key for inner chunk (cz,cy,cx): "cz<sep>cy<sep>cx".
static void chunk_key(const mc_zarr *z, int cz, int cy, int cx, char out[64]) {
    if (z->version == ZV3) {
        int sz = cz / z->per, sy = cy / z->per, sx = cx / z->per;
        snprintf(out, 64, "c/%d/%d/%d", sz, sy, sx);
    } else {
        snprintf(out, 64, "%d%c%d%c%d", cz, z->sep, cy, z->sep, cx);
    }
}

// ---------------------------------------------------------------------------
// v3 shard index: n_inner entries of (offset:u64, nbytes:u64) LE at shard start.
// missing == both == 0xFFFF...F. Linear order row-major, z slowest.
// ---------------------------------------------------------------------------
static int index_entry(const uint8_t *idx, size_t n_inner, size_t lin,
                       uint64_t *off, uint64_t *nb) {
    if (lin >= n_inner) return -1;
    memcpy(off, idx + lin * 16, 8);
    memcpy(nb, idx + lin * 16 + 8, 8);
    if (*off == ~(uint64_t)0 && *nb == ~(uint64_t)0) return 1;   // missing
    return 0;
}

// shard-relative linear inner index, row-major (z slowest, x fastest).
static size_t inner_linear(const mc_zarr *z, int cz, int cy, int cx) {
    int rz = cz % z->per, ry = cy % z->per, rx = cx % z->per;
    return ((size_t)rz * z->per + ry) * z->per + rx;
}

// Get the shard's index footer (n_inner*16 bytes), cached. Reads it once per
// shard via one ranged GET, then serves all the shard's chunks from RAM.
// Returns a borrowed pointer valid until the entry is evicted; copy out the
// (off,len) you need while you still need them (callers do this immediately).
// NULL if the shard is absent (all air) or on error.
// `err` (optional) distinguishes the two NULL returns: *err=1 means the footer
// READ FAILED (transient — caller must retry, NOT treat as air); *err=0 means
// the shard is genuinely absent (a clean 404 — confirmed air).
static const uint8_t *footer_get_ex(mc_zarr *z, int cz, int cy, int cx, int *err) {
    if (err) *err = 0;
    if (z->version != ZV3) return NULL;
    uint64_t sid = ((uint64_t)(cz / z->per) << 40) |
                   ((uint64_t)(cy / z->per) << 20) | (uint64_t)(cx / z->per);
    pthread_mutex_lock(&z->fmu);
    for (int i = 0; i < MC_FOOTER_CACHE; ++i)
        if (z->fcache[i].idx && z->fcache[i].shard_id == sid) {
            z->fcache[i].lru = ++z->ftick;
            const uint8_t *p = z->fcache[i].idx;
            pthread_mutex_unlock(&z->fmu);
            return p;
        }
    pthread_mutex_unlock(&z->fmu);

    // miss: fetch the footer (outside the lock — it's one ranged GET).
    char key[64];
    chunk_key(z, cz, cy, cx, key);
    size_t n_inner = (size_t)z->per * z->per * z->per;
    size_t idx_bytes = n_inner * 16;
    uint8_t *idx = NULL;
    size_t got = 0;
    int rc = z->read(z->ud, key, 0, idx_bytes, &idx, &got);
    if (rc < 0) {                       // transient read error -> retry, NOT air
        free(idx);
        if (err) *err = 1;
        return NULL;
    }
    if (!idx || got < idx_bytes) {      // clean 404 / short -> genuinely absent shard
        free(idx);
        return NULL;                    // *err stays 0
    }

    pthread_mutex_lock(&z->fmu);
    // re-check (another thread may have inserted it); if so, drop ours.
    int victim = 0;
    uint64_t oldest = ~0ull;
    for (int i = 0; i < MC_FOOTER_CACHE; ++i) {
        if (z->fcache[i].idx && z->fcache[i].shard_id == sid) {
            free(idx);
            z->fcache[i].lru = ++z->ftick;
            const uint8_t *p = z->fcache[i].idx;
            pthread_mutex_unlock(&z->fmu);
            return p;
        }
        if (!z->fcache[i].idx) { victim = i; oldest = 0; }
        else if (z->fcache[i].lru < oldest) { oldest = z->fcache[i].lru; victim = i; }
    }
    free(z->fcache[victim].idx);
    z->fcache[victim].idx = idx;
    z->fcache[victim].shard_id = sid;
    z->fcache[victim].lru = ++z->ftick;
    pthread_mutex_unlock(&z->fmu);
    return idx;
}
static const uint8_t *footer_get(mc_zarr *z, int cz, int cy, int cx) {
    return footer_get_ex(z, cz, cy, cx, NULL);
}

int mc_zarr_shard_all_air(mc_zarr *z, int cz, int cy, int cx) {
    if (z->version != ZV3) {
        // v2: "all air" == the single chunk object is absent.
        char key[64];
        chunk_key(z, cz, cy, cx, key);
        uint8_t *b = NULL;
        size_t n = 0;
        if (z->read(z->ud, key, 0, 1, &b, &n) < 0) return -1;
        free(b);
        return n == 0 ? 1 : 0;
    }
    size_t n_inner = (size_t)z->per * z->per * z->per;
    const uint8_t *idx = footer_get(z, cz, cy, cx);
    if (!idx) return 1;   // absent shard = air
    for (size_t i = 0; i < n_inner; ++i) {
        uint64_t off, nb;
        if (index_entry(idx, n_inner, i, &off, &nb) == 0) return 0;
    }
    return 1;
}

// decode a v2 chunk blob (blosc/raw) into a fresh dense buffer.
static uint8_t *v2_decode(const mc_zarr *z, uint8_t *blob, size_t blen, size_t *out_len) {
    if (strcmp(z->codec, "raw") == 0) { *out_len = blen; return blob; }   // takes ownership
    size_t dl = 0;
    uint8_t *dense = blosc_decode(blob, blen, &dl);
    free(blob);
    if (!dense) return NULL;
    *out_len = dl;
    return dense;
}

// Locate one inner chunk WITHOUT fetching: fill its object key + byte range from
// the cached shard footer. Returns 0 (found: *off/*nb set, c3d raw range), 1
// (absent/air), <0 (error). v2 chunks are whole objects -> off=0,nb=0 (full GET).
// Lets a caller batch many chunks' ranged GETs into one s3_get_batch.
// 0 found (off/nb set), 1 confirmed air (clean 404 / missing index entry),
// <0 transient READ ERROR (caller must retry, NOT mark the region air).
int mc_zarr_chunk_locate(mc_zarr *z, int cz, int cy, int cx,
                         char key_out[64], uint64_t *off, uint64_t *nb) {
    chunk_key(z, cz, cy, cx, key_out);
    if (z->version == ZV2) { *off = 0; *nb = 0; return 0; }
    size_t n_inner = (size_t)z->per * z->per * z->per;
    int err = 0;
    const uint8_t *idx = footer_get_ex(z, cz, cy, cx, &err);
    if (!idx) return err ? -1 : 1;                          // read error -> retry; else air
    size_t lin = inner_linear(z, cz, cy, cx);
    int st = index_entry(idx, n_inner, lin, off, nb);
    if (st != 0) return 1;                                   // missing entry -> confirmed air
    return 0;
}

int mc_zarr_read_inner(mc_zarr *z, int cz, int cy, int cx, uint8_t **raw, size_t *len) {
    *raw = NULL;
    *len = 0;
    char key[64];
    chunk_key(z, cz, cy, cx, key);

    if (z->version == ZV2) {
        // v2: a 404 (chunk object doesn't exist on S3) == confirmed air. A read
        // error must retry, not be recorded as air.
        size_t blen = 0; int err = 0;
        uint8_t *blob = fetch_all(z, key, &blen, &err);
        if (err) { free(blob); return -1; }               // transient -> retry
        if (!blob || !blen) { free(blob); return 1; }     // clean 404 -> confirmed air
        size_t dl = 0;
        uint8_t *dense = v2_decode(z, blob, blen, &dl);
        if (!dense) return -1;
        *raw = dense;
        *len = dl;
        return 0;
    }

    // v3: get the (cached) index footer, then the one inner chunk's payload range.
    size_t n_inner = (size_t)z->per * z->per * z->per;
    int ferr = 0;
    const uint8_t *idx = footer_get_ex(z, cz, cy, cx, &ferr);
    if (!idx) return ferr ? -1 : 1;                         // read error -> retry; else air
    size_t lin = inner_linear(z, cz, cy, cx);
    uint64_t off, nb;
    int st = index_entry(idx, n_inner, lin, &off, &nb);
    if (st != 0) return 1;                                   // index air marker -> confirmed air
    uint8_t *payload = NULL;
    size_t plen = 0;
    if (z->read(z->ud, key, off, nb, &payload, &plen) < 0) { return -1; }  // transient
    if (!payload || plen < nb) { free(payload); return -1; }               // short -> retry
    *raw = payload;
    *len = nb;
    return 0;   // c3d raw bytes — caller decodes.
}

int mc_zarr_read_shard(mc_zarr *z, int cz, int cy, int cx,
                       mc_zarr_chunk_fn sink, void *sink_ud) {
    if (z->version == ZV2) {
        // a v2 "shard" is one chunk.
        uint8_t *dense = NULL;
        size_t dl = 0;
        int st = mc_zarr_read_inner(z, cz, cy, cx, &dense, &dl);
        if (st < 0) return -1;
        if (st == 0) { sink(sink_ud, cz, cy, cx, dense, dl); free(dense); }
        return 0;
    }

    // v3: read the index footer ONCE, then range-GET each present inner chunk
    // individually (no whole-shard buffering). Each chunk is sunk + freed before
    // the next, so RAM stays at one chunk and disk grows per-chunk.
    char key[64];
    chunk_key(z, cz, cy, cx, key);
    size_t n_inner = (size_t)z->per * z->per * z->per;
    const uint8_t *idx = footer_get(z, cz, cy, cx);
    if (!idx) return 0;   // absent shard = all air
    int sz0 = (cz / z->per) * z->per, sy0 = (cy / z->per) * z->per, sx0 = (cx / z->per) * z->per;
    for (int iz = 0; iz < z->per; ++iz)
        for (int iy = 0; iy < z->per; ++iy)
            for (int ix = 0; ix < z->per; ++ix) {
                size_t lin = ((size_t)iz * z->per + iy) * z->per + ix;
                uint64_t off, nb;
                if (index_entry(idx, n_inner, lin, &off, &nb) != 0) continue;
                int gz = sz0 + iz, gy = sy0 + iy, gx = sx0 + ix;
                if (gz >= z->inner_grid[0] || gy >= z->inner_grid[1] || gx >= z->inner_grid[2])
                    continue;
                uint8_t *payload = NULL;
                size_t plen = 0;
                if (z->read(z->ud, key, off, nb, &payload, &plen) < 0) return -1;
                if (payload && plen >= nb)
                    sink(sink_ud, gz, gy, gx, payload, (size_t)nb);   // c3d raw bytes
                free(payload);
            }
    return 0;
}

int mc_zarr_shard_index(mc_zarr *z, int cz, int cy, int cx,
                        char key_out[64], mc_zarr_range **out, int *n) {
    *out = NULL;
    *n = 0;
    if (z->version == ZV2) {
        // a v2 "shard" is one chunk object; whole-object fetch (off/len = 0).
        chunk_key(z, cz, cy, cx, key_out);
        mc_zarr_range *r = malloc(sizeof *r);
        if (!r) return -1;
        r[0] = (mc_zarr_range){cz, cy, cx, 0, 0};
        *out = r;
        *n = 1;
        return 0;
    }
    chunk_key(z, cz, cy, cx, key_out);
    size_t n_inner = (size_t)z->per * z->per * z->per;
    const uint8_t *idx = footer_get(z, cz, cy, cx);
    if (!idx) return 0;                                     // absent shard = all air
    mc_zarr_range *arr = malloc(n_inner * sizeof *arr);
    if (!arr) return -1;
    int sz0 = (cz / z->per) * z->per, sy0 = (cy / z->per) * z->per, sx0 = (cx / z->per) * z->per;
    int cnt = 0;
    for (int iz = 0; iz < z->per; ++iz)
        for (int iy = 0; iy < z->per; ++iy)
            for (int ix = 0; ix < z->per; ++ix) {
                size_t lin = ((size_t)iz * z->per + iy) * z->per + ix;
                uint64_t off, nb;
                if (index_entry(idx, n_inner, lin, &off, &nb) != 0) continue;
                int gz = sz0 + iz, gy = sy0 + iy, gx = sx0 + ix;
                if (gz >= z->inner_grid[0] || gy >= z->inner_grid[1] || gx >= z->inner_grid[2])
                    continue;
                arr[cnt++] = (mc_zarr_range){gz, gy, gx, off, nb};
            }
    *out = arr;
    *n = cnt;
    return 0;
}

