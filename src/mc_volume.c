// mc_volume — see mc_volume.h. Streaming + transcode + cache + prefetch over a
// local .mca, source = mc_zarr, decode = vendored c3d (+ mc_zarr's blosc/raw).
#include "mc_volume.h"
#include "mc_zarr.h"
#include "c3d.h"
#include "matter_compressor.h"
#include "../tools/vendor/libs3/libs3.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <pthread.h>

#define MAXLOD 8
#define CHUNK 256
#define BLK 16
#define PER (CHUNK / BLK)   // 16 blocks per chunk axis

static void *worker_main(void *ud);
static const uint8_t *zero256(void);   // shared 32-aligned 256^3 zero buffer

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

    // region single-flight: key = (lod<<60)|(cz<<40)|(cy<<20)|cx, self-bounding.
    pthread_mutex_t mu;
    pthread_cond_t cv;
    uint64_t *inflight;        // open-addressed set
    size_t inflight_cap, inflight_n;

    // background transcode workers for the async render path (try_block).
    pthread_t workers[8];
    int nworkers;
    uint64_t queue[256];       // pending region keys (ring)
    int qh, qt;                // head/tail
    pthread_cond_t qcv;        // signals queued work
    int stop;
};

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

// ---------------------------------------------------------------------------
// single-flight set (open addressing, grows; tiny — only in-flight regions)
// ---------------------------------------------------------------------------
static uint64_t rkey(int lod, int cz, int cy, int cx) {
    return ((uint64_t)(lod & 7) << 60) | ((uint64_t)(cz & 0xFFFFF) << 40) |
           ((uint64_t)(cy & 0xFFFFF) << 20) | (uint64_t)(cx & 0xFFFFF);
}
static uint64_t kmix(uint64_t k) {
    k ^= k >> 33; k *= 0xFF51AFD7ED558CCDULL; k ^= k >> 33;
    k *= 0xC4CEB9FE1A85EC53ULL; k ^= k >> 33; return k;
}
// must hold mu. 1 if present.
static int infl_has(mc_volume *v, uint64_t key) {
    if (!v->inflight_cap) return 0;
    size_t m = v->inflight_cap - 1, i = kmix(key) & m;
    for (size_t p = 0; p <= m; ++p, i = (i + 1) & m) {
        if (v->inflight[i] == 0) return 0;
        if (v->inflight[i] == key) return 1;
    }
    return 0;
}
static void infl_grow(mc_volume *v) {
    size_t nc = v->inflight_cap ? v->inflight_cap * 2 : 64;
    uint64_t *na = calloc(nc, sizeof *na);
    if (!na) return;
    size_t m = nc - 1;
    for (size_t j = 0; j < v->inflight_cap; ++j) {
        uint64_t k = v->inflight[j];
        if (!k) continue;
        size_t i = kmix(k) & m;
        while (na[i]) i = (i + 1) & m;
        na[i] = k;
    }
    free(v->inflight);
    v->inflight = na;
    v->inflight_cap = nc;
}
static void infl_add(mc_volume *v, uint64_t key) {
    if ((v->inflight_n + 1) * 4 >= v->inflight_cap * 3) infl_grow(v);
    size_t m = v->inflight_cap - 1, i = kmix(key) & m;
    while (v->inflight[i]) i = (i + 1) & m;
    v->inflight[i] = key;
    v->inflight_n++;
}
static void infl_del(mc_volume *v, uint64_t key) {
    size_t m = v->inflight_cap - 1, i = kmix(key) & m;
    for (size_t p = 0; p <= m; ++p, i = (i + 1) & m) {
        if (v->inflight[i] == key) { v->inflight[i] = 0; v->inflight_n--; break; }
        if (v->inflight[i] == 0) break;
    }
    // rehash the run after the hole (open-addressing delete).
    size_t j = (i + 1) & m;
    while (v->inflight[j]) {
        uint64_t k = v->inflight[j];
        v->inflight[j] = 0; v->inflight_n--;
        infl_add(v, k);
        j = (j + 1) & m;
    }
}

// ---------------------------------------------------------------------------
// transcode one 256^3 region (cz,cy,cx) of lod into the .mca. caller ensures
// single-flight. returns 1 transcoded data, 0 air, <0 error.
// ---------------------------------------------------------------------------
static int transcode_region(mc_volume *v, int lod, int cz, int cy, int cx) {
    level_t *lv = &v->lv[lod];
    uint8_t *raw = NULL;
    size_t rlen = 0;
    int st = mc_zarr_read_inner(lv->z, cz, cy, cx, &raw, &rlen);
    if (st < 0) return -1;
    if (st == 1) {                       // air -> append a zero chunk (records ZERO)
        mc_archive_append_chunk_raw(v->arc, lod, cz, cy, cx, zero256());
        return 0;
    }
    // present: decode raw -> dense 256^3 (32-aligned), then re-encode into .mca.
    uint8_t *dense = NULL;
    if (posix_memalign((void **)&dense, 64, (size_t)CHUNK * CHUNK * CHUNK)) { free(raw); return -1; }
    const char *codec = mc_zarr_inner_codec(lv->z);
    if (strcmp(codec, "c3d") == 0) {
        c3d_decoder *d = c3d_decoder_new();
        c3d_decoder_set_denoise(d, false);
        c3d_decoder_chunk_decode(d, raw, rlen, dense);
        c3d_decoder_free(d);
    } else {                              // blosc/raw: mc_zarr already gave dense u8
        if (rlen >= (size_t)CHUNK * CHUNK * CHUNK) memcpy(dense, raw, (size_t)CHUNK * CHUNK * CHUNK);
        else { memset(dense, 0, (size_t)CHUNK * CHUNK * CHUNK); memcpy(dense, raw, rlen); }
    }
    free(raw);
    int rc = mc_archive_append_chunk_raw(v->arc, lod, cz, cy, cx, dense);
    free(dense);
    return rc == 0 ? 1 : -1;
}

// ensure region present in the .mca (single-flight). returns coverage after.
static mc_cover ensure_region(mc_volume *v, int lod, int cz, int cy, int cx) {
    mc_cover cov = mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx);
    if (cov != MC_ABSENT) return cov;
    uint64_t key = rkey(lod, cz, cy, cx);
    pthread_mutex_lock(&v->mu);
    while (infl_has(v, key)) {
        pthread_cond_wait(&v->cv, &v->mu);
        cov = mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx);
        if (cov != MC_ABSENT) { pthread_mutex_unlock(&v->mu); return cov; }
    }
    // re-check under lock: another thread may have just finished.
    cov = mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx);
    if (cov != MC_ABSENT) { pthread_mutex_unlock(&v->mu); return cov; }
    infl_add(v, key);
    pthread_mutex_unlock(&v->mu);

    transcode_region(v, lod, cz, cy, cx);

    pthread_mutex_lock(&v->mu);
    infl_del(v, key);
    pthread_cond_broadcast(&v->cv);
    pthread_mutex_unlock(&v->mu);
    return mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx);
}

// unpack a region key.
static void runpack(uint64_t k, int *lod, int *cz, int *cy, int *cx) {
    *lod = (int)((k >> 60) & 7);
    *cz = (int)((k >> 40) & 0xFFFFF);
    *cy = (int)((k >> 20) & 0xFFFFF);
    *cx = (int)(k & 0xFFFFF);
}

// background worker: drain the queue, transcode each region via single-flight.
static void *worker_main(void *ud) {
    mc_volume *v = ud;
    for (;;) {
        pthread_mutex_lock(&v->mu);
        while (v->qh == v->qt && !v->stop) pthread_cond_wait(&v->qcv, &v->mu);
        if (v->stop && v->qh == v->qt) { pthread_mutex_unlock(&v->mu); return NULL; }
        uint64_t key = v->queue[v->qh];
        v->qh = (v->qh + 1) & 255;
        pthread_mutex_unlock(&v->mu);
        int lod, cz, cy, cx;
        runpack(key, &lod, &cz, &cy, &cx);
        ensure_region(v, lod, cz, cy, cx);   // does its own single-flight + dedup
    }
}

// enqueue a region for async transcode (non-blocking; drops if queue full or
// already queued/in-flight). caller holds NOTHING.
static void enqueue_region(mc_volume *v, int lod, int cz, int cy, int cx) {
    uint64_t key = rkey(lod, cz, cy, cx);
    pthread_mutex_lock(&v->mu);
    if (infl_has(v, key)) { pthread_mutex_unlock(&v->mu); return; }   // already working
    int next = (v->qt + 1) & 255;
    if (next == v->qh) { pthread_mutex_unlock(&v->mu); return; }      // queue full, drop
    // dedup against pending queue entries.
    for (int i = v->qh; i != v->qt; i = (i + 1) & 255)
        if (v->queue[i] == key) { pthread_mutex_unlock(&v->mu); return; }
    v->queue[v->qt] = key;
    v->qt = next;
    pthread_cond_signal(&v->qcv);
    pthread_mutex_unlock(&v->mu);
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

    // background transcode workers for the async render path.
    pthread_cond_init(&v->qcv, NULL);
    v->nworkers = 4;
    for (int i = 0; i < v->nworkers; ++i)
        pthread_create(&v->workers[i], NULL, worker_main, v);
    return v;
}

void mc_volume_free(mc_volume *v) {
    if (!v) return;
    // stop workers.
    pthread_mutex_lock(&v->mu);
    v->stop = 1;
    pthread_cond_broadcast(&v->qcv);
    pthread_mutex_unlock(&v->mu);
    for (int i = 0; i < v->nworkers; ++i) pthread_join(v->workers[i], NULL);
    pthread_cond_destroy(&v->qcv);
    if (v->cache) mc_cache_free(v->cache);
    if (v->arc) mc_archive_close(v->arc);
    for (int i = 0; i < v->nlods; ++i) if (v->lv[i].z) mc_zarr_free(v->lv[i].z);
    if (v->s3) s3_client_free(v->s3);
    free(v->inflight);
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
        enqueue_region(v, lod, cz, cy, cx);   // async, deduped; render falls to coarser LOD
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
// prefetch
// ---------------------------------------------------------------------------
typedef struct { mc_volume *v; int lod; } shard_ctx;
static void shard_sink(void *ud, int cz, int cy, int cx, const uint8_t *raw, size_t raw_len) {
    shard_ctx *c = ud;
    mc_volume *v = c->v;
    if (mc_archive_chunk_coverage(v->arc, c->lod, cz, cy, cx) != MC_ABSENT) return;
    uint8_t *dense = NULL;
    if (posix_memalign((void **)&dense, 64, (size_t)CHUNK * CHUNK * CHUNK)) return;
    const char *codec = mc_zarr_inner_codec(v->lv[c->lod].z);
    if (strcmp(codec, "c3d") == 0) {
        c3d_decoder *d = c3d_decoder_new();
        c3d_decoder_set_denoise(d, false);
        c3d_decoder_chunk_decode(d, raw, raw_len, dense);
        c3d_decoder_free(d);
    } else {
        memset(dense, 0, (size_t)CHUNK * CHUNK * CHUNK);
        memcpy(dense, raw, raw_len < (size_t)CHUNK*CHUNK*CHUNK ? raw_len : (size_t)CHUNK*CHUNK*CHUNK);
    }
    mc_archive_append_chunk_raw(v->arc, c->lod, cz, cy, cx, dense);
    free(dense);
}

void mc_volume_prefetch_shard(mc_volume *v, int lod, int cz, int cy, int cx) {
    if (lod < 0 || lod >= v->nlods) return;
    if (mc_zarr_shard_all_air(v->lv[lod].z, cz, cy, cx) == 1) return;
    shard_ctx ctx = {v, lod};
    mc_zarr_read_shard(v->lv[lod].z, cz, cy, cx, shard_sink, &ctx);
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

void mc_volume_get_stats(const mc_volume *v, mc_volume_stats *out) {
    mc_cache_stats cs = {0};
    if (v->cache) mc_cache_get_stats(v->cache, &cs);
    out->cache_hits = cs.hits;
    out->cache_misses = cs.misses;
    out->disk_bytes = v->arc ? mc_archive_data_len(v->arc) : 0;
    out->net_bytes = atomic_load_explicit(&v->net_bytes, memory_order_relaxed);
    out->regions_inflight = v->inflight_n;
}
