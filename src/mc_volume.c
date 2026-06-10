// mc_volume — see mc_volume.h. Streaming + transcode + cache + prefetch over a
// local .mca, source = mc_zarr, decode = vendored c3d (+ mc_zarr's blosc/raw).
#include "mc_volume.h"
#include "mc_zarr.h"
#include "c3d.h"
#include "matter_compressor.h"
#include "libs3.h"   // include dir provides it (tools/vendor/libs3 or VC's libs/libs3)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#define MAXLOD 8
#define CHUNK 256
#define BLK 16
#define PER (CHUNK / BLK)   // 16 blocks per chunk axis

static void *decoder_main(void *ud);
static void *dl_main(void *ud);
static const uint8_t *zero256(void);   // shared 32-aligned 256^3 zero buffer

// ---- timing log (MCV_LOG=1 to enable) -------------------------------------
static int g_log = -1;
static double mcv_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;   // ms
}
#define MCVLOG(...) do { \
    if (g_log < 0) g_log = getenv("MCV_LOG") ? 1 : 0; \
    if (g_log) { fprintf(stderr, "[mcv %10.1f] ", mcv_now()); \
                 fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); fflush(stderr); } \
} while (0)

// ---------------------------------------------------------------------------
// per-level source (one mc_zarr + its S3 key prefix)
// ---------------------------------------------------------------------------
typedef struct {
    mc_volume *vol;     // back-pointer (for the shared s3 client + net counter)
    char prefix[1024];  // e.g. "s3://bucket/root/0" (no trailing slash)
    mc_zarr *z;
} level_t;

struct mc_volume {
    s3_client *s3;
    char root[1024];           // s3://bucket/root (no trailing slash)
    int nlods;
    level_t lv[MAXLOD];
    mc_archive *arc;           // ONE archive, all LODs
    mc_cache *cache;
    float quality;

    atomic_uint_fast64_t net_bytes;

    pthread_mutex_t mu;        // guards the decode queue + request stack
    pthread_cond_t cv;         // request-stack not-empty (wakes download threads)

    // Decode pipeline: download threads enqueue raw payloads here; a pool of
    // decode workers drains them (decode -> re-encode -> append). This keeps the
    // network saturated (downloaders never wait on CPU) and CPU saturated
    // (decoders run in parallel), instead of serializing download+decode.
    pthread_t decoders[32];
    int ndecoders;
    struct decode_item *dq;    // bounded ring of pending decode items
    int dq_cap, dq_head, dq_tail;
    pthread_cond_t dq_ne;      // not-empty (wake a decoder)
    pthread_cond_t dq_nf;      // not-full  (wake a blocked producer)
    int stop;

    // Interactive download-request stack (LIFO): a render miss pushes "fetch the
    // shard around region R". Download threads pop the NEWEST request (current
    // view) first; when full, the OLDEST (stalest, camera moved on) is dropped.
    uint64_t *reqstk;          // region keys
    int rs_cap, rs_n;
    pthread_t dlthreads[16];
    int ndl;

    mc_volume_ready_fn ready_cb;   // fired when a region becomes serveable
    void *ready_ud;
};

// One unit of decode work: the sub^3 cube of source chunks covering one 256^3
// region. For c3d (sub=1) nsub==1; for v2 (sub=2) up to 8. Owns the raw bytes.
typedef struct decode_item {
    int lod, rz, ry, rx;       // target 256^3 region coords
    int sub;                   // 1 (c3d) or 2 (v2)
    int nsub;                  // number of valid sub-chunks
    int oz[8], oy[8], ox[8];   // sub-chunk voxel offsets within the region
    uint8_t *raw[8];           // owned compressed bytes (freed by the decoder)
    size_t rlen[8];
} decode_item;

// ---------------------------------------------------------------------------
// s3 byte source for mc_zarr (prepends the level prefix to the object key)
// ---------------------------------------------------------------------------
static int s3_read(void *ud, const char *key, uint64_t off, uint64_t len,
                   uint8_t **out, size_t *out_len) {
    level_t *lv = ud;
    char url[1280];
    snprintf(url, sizeof url, "%s/%s", lv->prefix, key);
    s3_response resp = {0};
    s3_status st;
    if (len == 0) st = s3_get(lv->vol->s3, url, &resp);
    else          st = s3_get_range(lv->vol->s3, url, off, len, &resp);
    if (st != S3_OK) { s3_response_free(&resp); *out = NULL; *out_len = 0; return -1; }
    if (s3_response_not_found(&resp)) { s3_response_free(&resp); *out = NULL; *out_len = 0; return 0; }
    if (!s3_response_ok(&resp)) { s3_response_free(&resp); *out = NULL; *out_len = 0; return -1; }
    // honor a server that ignored Range and sent the whole object.
    const uint8_t *src = resp.body;
    size_t n = resp.body_len;
    if (len != 0 && resp.status == 200 && n >= off + len) { src += off; n = len; }
    uint8_t *buf = malloc(n ? n : 1);
    if (!buf) { s3_response_free(&resp); *out = NULL; *out_len = 0; return -1; }
    memcpy(buf, src, n);
    s3_response_free(&resp);
    atomic_fetch_add_explicit(&lv->vol->net_bytes, n, memory_order_relaxed);
    *out = buf;
    *out_len = n;
    return 0;
}

// pack a region (lod,cz,cy,cx) into a 64-bit key.
static uint64_t rkey(int lod, int cz, int cy, int cx) {
    return ((uint64_t)(lod & 7) << 60) | ((uint64_t)(cz & 0xFFFFF) << 40) |
           ((uint64_t)(cy & 0xFFFFF) << 20) | (uint64_t)(cx & 0xFFFFF);
}

// ---------------------------------------------------------------------------
// transcode one 256^3 region (cz,cy,cx) of lod into the .mca. caller ensures
// single-flight. returns 1 transcoded data, 0 air, <0 error.
// ---------------------------------------------------------------------------
// decode one source inner-chunk's raw bytes into `dst` (edge^3, edge = source
// inner_edge: 256 for c3d, 128 for v2). dst need not be 32-aligned for v2; c3d
// needs a 32-aligned 256^3 (the v3 case always passes the region buffer).
static void decode_inner(const char *codec, const uint8_t *raw, size_t rlen,
                         uint8_t *dst, int edge) {
    size_t vox = (size_t)edge * edge * edge;
    if (strcmp(codec, "c3d") == 0) {
        c3d_decoder *d = c3d_decoder_new();
        c3d_decoder_set_denoise(d, false);
        c3d_decoder_chunk_decode(d, raw, rlen, dst);   // c3d edge is always 256
        c3d_decoder_free(d);
    } else {                                            // blosc/raw: already dense u8
        if (rlen >= vox) memcpy(dst, raw, vox);
        else { memset(dst, 0, vox); memcpy(dst, raw, rlen); }
    }
}

// blit a src (edge^3) into the 256^3 region buffer at sub-offset (oz,oy,ox) voxels.
static void blit_sub(uint8_t *region, const uint8_t *src, int edge,
                     int oz, int oy, int ox) {
    for (int z = 0; z < edge; ++z)
        for (int y = 0; y < edge; ++y)
            memcpy(region + (((size_t)(oz + z) * CHUNK + (oy + y)) * CHUNK + ox),
                   src + ((size_t)z * edge + y) * edge, (size_t)edge);
}

// Decode one item (the sub^3 cube for a region) -> assemble 256^3 -> append.
// Frees the item's raw buffers. Runs on a decode-pool thread (off the download
// thread). The c3d decode + mc re-encode are the CPU cost we keep off the net.
static void decode_one(mc_volume *v, decode_item *it) {
    const char *codec = mc_zarr_inner_codec(v->lv[it->lod].z);
    const int edge = CHUNK / it->sub;
    if (it->nsub == 0) {                               // all air -> ZERO
        mc_archive_append_chunk_raw(v->arc, it->lod, it->rz, it->ry, it->rx, zero256());
        return;
    }
    uint8_t *dense = NULL;
    if (posix_memalign((void **)&dense, 64, (size_t)CHUNK * CHUNK * CHUNK)) goto done;
    double t_dec0 = mcv_now();
    if (it->sub == 1) {                                // c3d: chunk == region
        decode_inner(codec, it->raw[0], it->rlen[0], dense, CHUNK);
    } else {                                           // v2: blit the cube
        memset(dense, 0, (size_t)CHUNK * CHUNK * CHUNK);
        uint8_t *tile = malloc((size_t)edge * edge * edge);
        if (tile) {
            for (int k = 0; k < it->nsub; ++k) {
                decode_inner(codec, it->raw[k], it->rlen[k], tile, edge);
                blit_sub(dense, tile, edge, it->oz[k], it->oy[k], it->ox[k]);
            }
            free(tile);
        }
    }
    double t_enc0 = mcv_now();
    mc_archive_append_chunk_raw(v->arc, it->lod, it->rz, it->ry, it->rx, dense);
    double t_end = mcv_now();
    MCVLOG("decoded   lod%d region(%d,%d,%d) codec=%s decode=%.0fms encode=%.0fms",
           it->lod, it->rz, it->ry, it->rx, codec,
           t_enc0 - t_dec0, t_end - t_enc0);
    free(dense);
done:
    for (int k = 0; k < it->nsub; ++k) free(it->raw[k]);
}

// Decode-pool worker: drain decode items, decode off the download thread.
static void *decoder_main(void *ud) {
    mc_volume *v = ud;
    for (;;) {
        pthread_mutex_lock(&v->mu);
        while (v->dq_head == v->dq_tail && !v->stop) pthread_cond_wait(&v->dq_ne, &v->mu);
        if (v->stop && v->dq_head == v->dq_tail) { pthread_mutex_unlock(&v->mu); return NULL; }
        decode_item it = v->dq[v->dq_head];
        v->dq_head = (v->dq_head + 1) % v->dq_cap;
        pthread_cond_signal(&v->dq_nf);                // a slot freed
        pthread_mutex_unlock(&v->mu);
        decode_one(v, &it);
        if (v->ready_cb) v->ready_cb(v->ready_ud);     // region became serveable
    }
}

// Producer: push a decode item, BLOCKING if the queue is full (backpressure ->
// bounded RAM; the download thread waits for decoders to catch up). Takes
// ownership of the item's raw buffers.
static void decode_push(mc_volume *v, const decode_item *it) {
    pthread_mutex_lock(&v->mu);
    int next = (v->dq_tail + 1) % v->dq_cap;
    int blocked = (next == v->dq_head);
    while (next == v->dq_head && !v->stop) pthread_cond_wait(&v->dq_nf, &v->mu);
    if (v->stop) { pthread_mutex_unlock(&v->mu);
        for (int k = 0; k < it->nsub; ++k) free(it->raw[k]); return; }
    v->dq[v->dq_tail] = *it;
    v->dq_tail = next;
    int depth = (v->dq_tail - v->dq_head + v->dq_cap) % v->dq_cap;
    pthread_cond_signal(&v->dq_ne);
    pthread_mutex_unlock(&v->mu);
    if (blocked) MCVLOG("decode_q  FULL (backpressure: decoders behind) depth=%d", depth);
}

// unpack a region key.
static void runpack(uint64_t k, int *lod, int *cz, int *cy, int *cx) {
    *lod = (int)((k >> 60) & 7);
    *cz = (int)((k >> 40) & 0xFFFFF);
    *cy = (int)((k >> 20) & 0xFFFFF);
    *cx = (int)(k & 0xFFFFF);
}

// Push an interactive download request (region key) onto the LIFO stack. Newest
// on top. If full, drop the BOTTOM (stalest). Deduped against the stack. Wakes a
// download thread. (cv doubles as the stack's not-empty signal.)
static void req_push(mc_volume *v, int lod, int cz, int cy, int cx) {
    uint64_t key = rkey(lod, cz, cy, cx);
    pthread_mutex_lock(&v->mu);
    for (int i = 0; i < v->rs_n; ++i)
        if (v->reqstk[i] == key) { pthread_mutex_unlock(&v->mu); return; }   // already queued
    if (v->rs_n == v->rs_cap) {                         // full -> drop bottom
        memmove(&v->reqstk[0], &v->reqstk[1], (size_t)(v->rs_cap - 1) * sizeof(uint64_t));
        v->rs_n--;
    }
    v->reqstk[v->rs_n++] = key;                         // push on top
    MCVLOG("req_push  lod%d region(%d,%d,%d) stack_depth=%d", lod, cz, cy, cx, v->rs_n);
    pthread_cond_signal(&v->cv);
    pthread_mutex_unlock(&v->mu);
}

// Download thread: pop the newest request, download its shard (-> decode queue).
static void *dl_main(void *ud) {
    mc_volume *v = ud;
    for (;;) {
        pthread_mutex_lock(&v->mu);
        while (v->rs_n == 0 && !v->stop) pthread_cond_wait(&v->cv, &v->mu);
        if (v->stop && v->rs_n == 0) { pthread_mutex_unlock(&v->mu); return NULL; }
        uint64_t key = v->reqstk[--v->rs_n];           // pop top (newest)
        pthread_mutex_unlock(&v->mu);
        int lod, cz, cy, cx;
        runpack(key, &lod, &cz, &cy, &cx);             // region coords
        MCVLOG("dl_pop    lod%d region(%d,%d,%d) -> download shard", lod, cz, cy, cx);
        const int sub = CHUNK / mc_zarr_inner_edge(v->lv[lod].z);
        mc_volume_prefetch_shard(v, lod, cz * sub, cy * sub, cx * sub);  // source coord
    }
}

// Blocking fill of one region (get_block / CLI): download its shard synchronously
// through the same decode queue, then wait for that region's coverage to resolve.
static mc_cover ensure_region(mc_volume *v, int lod, int cz, int cy, int cx) {
    mc_cover cov = mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx);
    if (cov != MC_ABSENT) return cov;
    const int sub = CHUNK / mc_zarr_inner_edge(v->lv[lod].z);
    mc_volume_prefetch_shard(v, lod, cz * sub, cy * sub, cx * sub);   // pushes to decode queue
    // wait for the decoders to drain enough that this region is covered.
    for (int spin = 0; spin < 100000; ++spin) {
        cov = mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx);
        if (cov != MC_ABSENT) return cov;
        struct timespec ts = {0, 1000000};             // 1ms
        nanosleep(&ts, NULL);
    }
    return mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx);
}

// ---------------------------------------------------------------------------
// shared 32-aligned zero region (for air)
// ---------------------------------------------------------------------------
static uint8_t *g_zero = NULL;
static void init_zero(void) {
    if (posix_memalign((void **)&g_zero, 64, (size_t)CHUNK * CHUNK * CHUNK) == 0)
        memset(g_zero, 0, (size_t)CHUNK * CHUNK * CHUNK);
}
static const uint8_t *zero256(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, init_zero);
    return g_zero;
}

// ===========================================================================
// open / discovery
// ===========================================================================

// strip a trailing '/'.
static void rstrip_slash(char *s) {
    size_t n = strlen(s);
    while (n && s[n - 1] == '/') s[--n] = 0;
}

mc_volume *mc_volume_open(const char *url, const char *cache_dir,
                          size_t cache_bytes, float quality) {
    if (!url || !cache_dir) return NULL;
    mc_volume *v = calloc(1, sizeof *v);
    if (!v) return NULL;
    v->quality = quality;
    atomic_init(&v->net_bytes, 0);
    pthread_mutex_init(&v->mu, NULL);
    pthread_cond_init(&v->cv, NULL);

    // s3 client: full credential resolution (profile/IMDS/SSO/env), else anonymous.
    s3_config cfg = {0};
    s3_credentials creds = {0};
    if (s3_credentials_load(NULL, &creds) == S3_OK) cfg.creds = creds;
    v->s3 = s3_client_new(&cfg);
    s3_credentials_free(&creds);
    if (!v->s3) { free(v); return NULL; }

    snprintf(v->root, sizeof v->root, "%s", url);
    rstrip_slash(v->root);

    // discover levels: probe "<root>/<i>/zarr.json" for i=0.. until a gap.
    for (int i = 0; i < MAXLOD; ++i) {
        level_t *lv = &v->lv[i];
        lv->vol = v;
        snprintf(lv->prefix, sizeof lv->prefix, "%s/%d", v->root, i);
        mc_zarr *z = mc_zarr_open(s3_read, lv);
        if (!z) { lv->prefix[0] = 0; break; }
        lv->z = z;
        v->nlods = i + 1;
    }
    if (v->nlods == 0) { s3_client_free(v->s3); free(v); return NULL; }

    // local .mca dims from LOD0 shape (padded to 256 internally by mc).
    int nz, ny, nx;
    mc_zarr_shape(v->lv[0].z, &nz, &ny, &nx);
    char path[2048];
    // archive name from the last path component of the root.
    const char *base = strrchr(v->root, '/');
    base = base ? base + 1 : v->root;
    snprintf(path, sizeof path, "%s/%s.mca", cache_dir, base);
    v->arc = mc_archive_open_dims(path, nx, ny, nz, quality);
    if (!v->arc) {
        for (int i = 0; i < v->nlods; ++i) mc_zarr_free(v->lv[i].z);
        s3_client_free(v->s3); free(v); return NULL;
    }
    v->cache = mc_cache_new_archive(cache_bytes, v->arc);
    if (!v->cache) {
        mc_archive_close(v->arc);
        for (int i = 0; i < v->nlods; ++i) mc_zarr_free(v->lv[i].z);
        s3_client_free(v->s3); free(v); return NULL;
    }

    // Pipeline: a few download threads (network-bound, pop the LIFO request
    // stack) feed a bounded decode queue drained by a decode pool (CPU-bound).
    pthread_cond_init(&v->dq_ne, NULL);
    pthread_cond_init(&v->dq_nf, NULL);
    v->dq_cap = 256;                                   // bounded decode queue
    v->dq = calloc((size_t)v->dq_cap, sizeof *v->dq);
    v->rs_cap = 512;                                   // LIFO request stack
    v->reqstk = calloc((size_t)v->rs_cap, sizeof *v->reqstk);

    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    v->ndecoders = nproc > 2 ? (nproc < 32 ? (int)nproc : 32) : 2;
    for (int i = 0; i < v->ndecoders; ++i)
        pthread_create(&v->decoders[i], NULL, decoder_main, v);
    v->ndl = 8;                                        // download threads (latency-bound)
    for (int i = 0; i < v->ndl; ++i)
        pthread_create(&v->dlthreads[i], NULL, dl_main, v);
    MCVLOG("open      %s  decoders=%d dl_threads=%d dq_cap=%d", url, v->ndecoders, v->ndl, v->dq_cap);
    return v;
}

void mc_volume_free(mc_volume *v) {
    if (!v) return;
    // stop download + decode threads.
    pthread_mutex_lock(&v->mu);
    v->stop = 1;
    pthread_cond_broadcast(&v->cv);      // wake download threads
    pthread_cond_broadcast(&v->dq_ne);   // wake decoders
    pthread_cond_broadcast(&v->dq_nf);   // wake blocked producers
    pthread_mutex_unlock(&v->mu);
    for (int i = 0; i < v->ndl; ++i) pthread_join(v->dlthreads[i], NULL);
    for (int i = 0; i < v->ndecoders; ++i) pthread_join(v->decoders[i], NULL);
    // drain any remaining decode items (free their raw buffers).
    while (v->dq_head != v->dq_tail) {
        decode_item *it = &v->dq[v->dq_head];
        for (int k = 0; k < it->nsub; ++k) free(it->raw[k]);
        v->dq_head = (v->dq_head + 1) % v->dq_cap;
    }
    pthread_cond_destroy(&v->dq_ne);
    pthread_cond_destroy(&v->dq_nf);
    if (v->cache) mc_cache_free(v->cache);
    if (v->arc) mc_archive_close(v->arc);
    for (int i = 0; i < v->nlods; ++i) if (v->lv[i].z) mc_zarr_free(v->lv[i].z);
    if (v->s3) s3_client_free(v->s3);
    free(v->dq);
    free(v->reqstk);
    pthread_mutex_destroy(&v->mu);
    pthread_cond_destroy(&v->cv);
    free(v);
}

int  mc_volume_nlods(const mc_volume *v) { return v ? v->nlods : 0; }
void mc_volume_shape(const mc_volume *v, int lod, int *nz, int *ny, int *nx) {
    mc_zarr_shape(v->lv[lod].z, nz, ny, nx);
}
void mc_volume_block_grid(const mc_volume *v, int lod, int *nz, int *ny, int *nx) {
    int sz, sy, sx;
    mc_zarr_shape(v->lv[lod].z, &sz, &sy, &sx);
    if (nz) *nz = (sz + BLK - 1) / BLK;
    if (ny) *ny = (sy + BLK - 1) / BLK;
    if (nx) *nx = (sx + BLK - 1) / BLK;
}

// ---------------------------------------------------------------------------
// block serving
// ---------------------------------------------------------------------------
int mc_volume_try_block(mc_volume *v, int lod, int bz, int by, int bx, uint8_t *dst) {
    if (lod < 0 || lod >= v->nlods) { memset(dst, 0, BLK * BLK * BLK); return 0; }
    int cz = bz / PER, cy = by / PER, cx = bx / PER;
    mc_cover cov = mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx);
    if (cov == MC_ABSENT) {
        req_push(v, lod, cz, cy, cx);   // LIFO download request; render falls to coarser LOD
        memset(dst, 0, BLK * BLK * BLK);
        return 0;
    }
    if (cov == MC_ZERO) { memset(dst, 0, BLK * BLK * BLK); return 1; }
    mc_cache_get_copy(v->cache, lod, bz, by, bx, dst);
    return 1;
}

int mc_volume_get_block(mc_volume *v, int lod, int bz, int by, int bx, uint8_t *dst) {
    if (lod < 0 || lod >= v->nlods) { memset(dst, 0, BLK * BLK * BLK); return -1; }
    int cz = bz / PER, cy = by / PER, cx = bx / PER;
    mc_cover cov = ensure_region(v, lod, cz, cy, cx);
    if (cov == MC_ZERO || cov == MC_ABSENT) { memset(dst, 0, BLK * BLK * BLK); return cov == MC_ZERO ? 0 : -1; }
    mc_cache_get_copy(v->cache, lod, bz, by, bx, dst);
    return 1;
}

// ---------------------------------------------------------------------------
// sampling source
// ---------------------------------------------------------------------------
static const uint8_t *vol_block(const mc_sample_src *src,
                                int bz, int by, int bx, uint8_t *tmp) {
    mc_volume *v = src->ud;
    int r = src->aux2 ? mc_volume_get_block(v, src->aux, bz, by, bx, tmp)
                      : mc_volume_try_block(v, src->aux, bz, by, bx, tmp);
    return r == 1 ? tmp : NULL;
}

mc_sample_src mc_volume_sample_src(mc_volume *v, int lod, int blocking) {
    mc_sample_src s = {0};
    s.ud = v; s.aux = lod; s.aux2 = blocking; s.block = vol_block;
    mc_volume_shape(v, lod, &s.nz, &s.ny, &s.nx);
    return s;
}

mc_sample_lods mc_volume_sample_lods(mc_volume *v, int blocking) {
    mc_sample_lods ls = {0};
    ls.nlods = v->nlods < 8 ? v->nlods : 8;
    for (int l = 0; l < ls.nlods; l++)
        ls.lods[l] = mc_volume_sample_src(v, l, blocking);
    return ls;
}

// ---------------------------------------------------------------------------
// prefetch — batch a whole shard's present inner chunks in ONE parallel
// s3_get_batch (many concurrent GETs over pooled connections), then decode +
// assemble into 256^3 regions and append. This is the throughput path: the
// parallelism lives in libs3's connection pool, so a FEW prefetch driver
// threads saturate bandwidth without a thread-per-GET explosion. RAM is bounded
// by one shard's compressed chunks (a fraction of the decoded shard).
// (cz,cy,cx) is any source inner-chunk in the target shard.
// ---------------------------------------------------------------------------
// Download a shard's present chunks (one parallel s3_get_batch) and PUSH each
// region's raw payload(s) to the decode queue — NO decode on this (download)
// thread. Decoders drain the queue in parallel, so the network stays saturated.
// Backpressure in decode_push bounds RAM. (cz,cy,cx) = source inner-chunk coord.
void mc_volume_prefetch_shard(mc_volume *v, int lod, int cz, int cy, int cx) {
    if (lod < 0 || lod >= v->nlods) return;
    level_t *lv = &v->lv[lod];
    mc_zarr *z = lv->z;
    const int edge = mc_zarr_inner_edge(z);            // 256 (c3d) or 128 (v2)
    const int sub = CHUNK / edge;                      // source chunks per region axis

    char shard_key[64];
    mc_zarr_range *ranges = NULL;
    int nr = 0;
    double t0 = mcv_now();
    if (mc_zarr_shard_index(z, cz, cy, cx, shard_key, &ranges, &nr) < 0) {
        MCVLOG("shard_idx lod%d src(%d,%d,%d) FAILED", lod, cz, cy, cx); return;
    }
    MCVLOG("shard_idx lod%d src(%d,%d,%d) -> %d present chunks (footer %.0fms)",
           lod, cz, cy, cx, nr, mcv_now() - t0);
    if (nr == 0) { free(ranges); return; }             // all air

    char shard_url[1280];
    snprintf(shard_url, sizeof shard_url, "%s/%s", lv->prefix, shard_key);
    uint64_t got = 0;
    int nbatch = 0;

    // Download the shard's chunks in batches of MC_BATCH (bounded buffering),
    // then hand each region's raw bytes to the decode pool. v2 groups the sub^3
    // cube per region; c3d is 1:1.
    enum { MC_BATCH = 48 };
    s3_range_req reqs[MC_BATCH];
    s3_response resp[MC_BATCH];
    int idx[MC_BATCH];
    for (int base = 0; base < nr; ) {
        int nq = 0;
        while (base < nr && nq < MC_BATCH) {
            mc_zarr_range *rg = &ranges[base++];
            int rz = rg->cz / sub, ry = rg->cy / sub, rx = rg->cx / sub;
            if (mc_archive_chunk_coverage(v->arc, lod, rz, ry, rx) != MC_ABSENT) continue;
            reqs[nq] = (s3_range_req){shard_url, rg->off, rg->len};
            idx[nq] = base - 1;
            ++nq;
        }
        if (nq == 0) continue;
        memset(resp, 0, sizeof resp);
        double tb = mcv_now();
        s3_get_batch(v->s3, reqs, (size_t)nq, 32, resp);   // partial ok; check each
        { int ok = 0; uint64_t bytes = 0;
          for (int i = 0; i < nq; ++i) if (s3_response_ok(&resp[i])) { ok++; bytes += resp[i].body_len; }
          MCVLOG("batch#%d  lod%d nq=%d ok=%d %.2fMB in %.0fms = %.1f MB/s",
                 nbatch++, lod, nq, ok, bytes/1048576.0, mcv_now()-tb,
                 bytes/1048576.0/((mcv_now()-tb)/1000.0)); }

        if (sub == 1) {                                // c3d: one chunk == one region
            for (int i = 0; i < nq; ++i) {
                mc_zarr_range *rg = &ranges[idx[i]];
                if (s3_response_ok(&resp[i]) && rg->len && resp[i].body_len >= rg->len) {
                    decode_item it = {lod, rg->cz, rg->cy, rg->cx, 1, 1, {0},{0},{0}, {0},{0}};
                    it.raw[0] = malloc(rg->len);
                    if (it.raw[0]) { memcpy(it.raw[0], resp[i].body, rg->len); it.rlen[0] = rg->len;
                        got += rg->len; decode_push(v, &it); }
                }
                s3_response_free(&resp[i]);
            }
        } else {                                       // v2: regroup the cube per region
            // Build one decode_item per distinct region in this batch.
            for (int i = 0; i < nq; ++i) {
                if (idx[i] < 0) continue;              // already consumed into a cube
                mc_zarr_range *r0 = &ranges[idx[i]];
                int rz = r0->cz / sub, ry = r0->cy / sub, rx = r0->cx / sub;
                decode_item it = {lod, rz, ry, rx, sub, 0, {0},{0},{0}, {0},{0}};
                for (int j = i; j < nq; ++j) {
                    if (idx[j] < 0) continue;
                    mc_zarr_range *rg = &ranges[idx[j]];
                    if (rg->cz / sub != rz || rg->cy / sub != ry || rg->cx / sub != rx) continue;
                    if (s3_response_ok(&resp[j]) && resp[j].body_len >= rg->len && it.nsub < 8) {
                        size_t rlen = rg->len ? rg->len : resp[j].body_len;
                        uint8_t *buf = malloc(rlen);
                        if (buf) { memcpy(buf, resp[j].body, rlen);
                            int k = it.nsub++;
                            it.raw[k] = buf; it.rlen[k] = rlen;
                            it.oz[k] = (rg->cz % sub) * edge;
                            it.oy[k] = (rg->cy % sub) * edge;
                            it.ox[k] = (rg->cx % sub) * edge;
                            got += rlen;
                        }
                    }
                    idx[j] = -1;                       // consumed
                }
                decode_push(v, &it);                   // nsub may be 0 -> ZERO region
            }
            for (int i = 0; i < nq; ++i) s3_response_free(&resp[i]);
        }
    }
    atomic_fetch_add_explicit(&v->net_bytes, got, memory_order_relaxed);
    free(ranges);
}

void mc_volume_prefetch_level(mc_volume *v, int lod, int nthreads, volatile int *cancel) {
    (void)nthreads;   // TODO: thread team; serial walk for now.
    if (lod < 0 || lod >= v->nlods) return;
    int gz, gy, gx;
    mc_zarr_inner_grid(v->lv[lod].z, &gz, &gy, &gx);
    int per = mc_zarr_shard_edge(v->lv[lod].z) / mc_zarr_inner_edge(v->lv[lod].z);
    for (int sz = 0; sz < gz; sz += per)
        for (int sy = 0; sy < gy; sy += per)
            for (int sx = 0; sx < gx; sx += per) {
                if (cancel && *cancel) return;
                mc_volume_prefetch_shard(v, lod, sz, sy, sx);
            }
}

void mc_volume_set_ready_cb(mc_volume *v, mc_volume_ready_fn cb, void *ud) {
    v->ready_cb = cb;
    v->ready_ud = ud;
}

void mc_volume_get_stats(const mc_volume *v, mc_volume_stats *out) {
    mc_cache_stats cs = {0};
    if (v->cache) mc_cache_get_stats(v->cache, &cs);
    out->cache_hits = cs.hits;
    out->cache_misses = cs.misses;
    out->disk_bytes = v->arc ? mc_archive_data_len(v->arc) : 0;
    out->net_bytes = atomic_load_explicit(&v->net_bytes, memory_order_relaxed);
    out->regions_inflight = (uint64_t)v->rs_n;
}
