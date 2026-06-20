#define _GNU_SOURCE
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "matter_compressor.h"
#include "mc_codec.h"
#include "mc_archive.h"
#include "mc_cache.h"
#include "libs3.h"
typedef uint8_t u8;
#ifndef BLK
#define BLK 16
#endif
// coverage-memo key (mirrors mc_archive.c; volume probes the same scheme)
static inline uint64_t mc_covkey(int lod,int cz,int cy,int cx){
    return ((uint64_t)(lod & 7) << 36) | ((uint64_t)(cz & 0xFFF) << 24) |
           ((uint64_t)(cy & 0xFFF) << 12) | (uint64_t)(cx & 0xFFF);
}
// ============================================================================
// mc_s3 — s3://-backed mc_reader glue (dep: libs3)
// ============================================================================
#include "libs3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct mc_s3 {
    s3_client *cl;
    char *url;
    mc_reader *r;
    _Atomic uint64_t net_bytes;   // bytes pulled over S3 (for the volume's rate readout)
};

static int s3_read_cb(void *ud, uint64_t off, uint32_t len, uint8_t *dst){
    mc_s3 *s=ud;
    s3_response resp={0};
    if(s3_get_range(s->cl,s->url,off,len,&resp)!=S3_OK || !s3_response_ok(&resp)){
        s3_response_free(&resp);
        return -1;
    }
    int rc=-1;
    if(resp.status==206 && resp.body_len>=len){
        memcpy(dst,resp.body,len); rc=0;          // proper ranged reply
    } else if(resp.status==200 && resp.body_len>=off+len){
        memcpy(dst,resp.body+off,len); rc=0;      // server ignored Range and
    }                                             // sent the whole object
    if(rc==0) atomic_fetch_add_explicit(&s->net_bytes,len,memory_order_relaxed);
    s3_response_free(&resp);
    return rc;
}

uint64_t mc_s3_net_bytes(mc_s3 *s){
    return s ? atomic_load_explicit(&s->net_bytes,memory_order_relaxed) : 0;
}

mc_s3 *mc_s3_open(const char *url){
    if(!url) return NULL;
    mc_s3 *s=calloc(1,sizeof *s);
    // Full credential resolution (profile/IMDS/SSO/env), else anonymous -- same as
    // the zarr transcode path, so a private bucket (philodemos) authenticates.
    s3_config cfg={0};
    s3_credentials creds={0};
    if(s3_credentials_load(NULL,&creds)==S3_OK) cfg.creds=creds;
    s->cl=s3_client_new(&cfg);
    s3_credentials_free(&creds);
    if(!s->cl){ free(s); return NULL; }
    s->url=strdup(url);
    s3_response head={0};
    uint64_t total=0;
    if(s3_head(s->cl,url,&head)==S3_OK && s3_response_ok(&head))
        total=head.content_length;
    s3_response_free(&head);
    if(!total){ mc_s3_close(s); return NULL; }
    s->r=mc_open_streaming(s3_read_cb,s,total);
    if(!s->r){ mc_s3_close(s); return NULL; }
    mc_reader_set_partial_fetch(s->r,1);
    return s;
}
mc_reader *mc_s3_reader(mc_s3 *s){ return s?s->r:NULL; }
void mc_s3_close(mc_s3 *s){
    if(!s) return;
    if(s->r) mc_close(s->r);
    if(s->cl) s3_client_free(s->cl);
    free(s->url);
    free(s);
}

// ============================================================================
// mc_volume — remote zarr -> local .mca (deps: mc_zarr, libs3)
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#define MAXLOD 8
#define CHUNK 256
#define PER (CHUNK / BLK)   // 16 blocks per chunk axis

static void *decoder_main(void *ud);
static void *dl_main(void *ud);
static void fill_pool_init(mc_volume *v, int n);
static void fill_pool_stop(mc_volume *v);
static void fill_pool_start(mc_volume *v);
static void fill_pool_wait(mc_volume *v);
static void mc_volume_prefetch_region(mc_volume *v, int lod, int cz, int cy, int cx);
static int  inflight_has(mc_volume *v, uint64_t key);   // single-flight (v->mu held)
static void inflight_add(mc_volume *v, uint64_t key);
static void inflight_del(mc_volume *v, uint64_t key);
static void vol_mark_region(mc_volume *v, int lod, int cz, int cy, int cx);
#define MC_RGEN_SLOTS (1u << 16)   // per-region change-gen table (512KB)

// portable thread naming: macOS names the calling thread (1 arg), glibc
// takes (thread, name)
static void mc_thread_setname(const char *name) {
#if defined(__APPLE__)
    pthread_setname_np(name);
#elif defined(__linux__)
    pthread_setname_np(pthread_self(), name);
#else
    (void)name;
#endif
}
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
    s3_client *s3;             // NULL for a local-filesystem source
    int local;                 // 1 => root is a local dir, read via file_read
    char root[1024];           // s3://bucket/root, or /local/dir (no trailing slash)
    int nlods;
    level_t lv[MAXLOD];
    mc_archive *arc;           // ONE archive, all LODs
    mc_cache *cache;
    float quality;

    // Streaming mode: the source is an ALREADY-BUILT .mca. The download thread
    // resolves a region's offset in the source reader and copies its compressed blob
    // VERBATIM onto the LOCAL archive (v->arc) -- no decode, no re-encode. Everything
    // else is the normal local-archive path: coverage, THAW decode-from-local, cache,
    // the LIFO request stack. lv[] / decode pool are unused; one dl thread owns the
    // (non-reentrant) source reader.
    int streaming;             // 1 => source is a pre-built .mca, copied verbatim
    mc_s3 *s3mca;              // remote source reader handle (s3/https), or NULL
    struct mc_reader *rd;      // source reader: chunk offsets + verbatim blob bytes
    uint8_t *s_map;            // local-file source: whole .mca mmap'd read-only
    size_t s_map_len;
    size_t s_blob_ema;         // EMA of blob sizes -> adaptive round-A GET length
                               // (dl thread only; fixed over-read either wastes
                               // bandwidth on small blobs or two-trips big ones)
    int s_nz[MAXLOD], s_ny[MAXLOD], s_nx[MAXLOD];   // per-LOD voxel dims (n0>>lod)

    atomic_uint_fast64_t net_bytes;

    pthread_mutex_t mu;        // guards the decode queue + request stack
    pthread_cond_t cv;         // request-stack not-empty (wakes download threads)

    // Decode pipeline: download threads enqueue raw payloads here; a pool of
    // decode workers drains them (decode -> re-encode -> append). This keeps the
    // network saturated (downloaders never wait on CPU) and CPU saturated
    // (decoders run in parallel), instead of serializing download+decode.
    pthread_t decoders[32];
    int ndecoders;
    // Persistent archive->cache fill pool. thaw() hands it the frame's PRESENT
    // block set (shard-partitioned, lock-free single-owner writes) and returns;
    // the workers decode while the frame renders. freeze() waits for them (the
    // barrier) before flipping frozen, preserving no-mutation-while-frozen.
    pthread_t fillers[32];
    int nfillers;
    mc_block_id *fill_ids;           // OWNED copy of this fill's block set (NOT the
                                     // thaw thread-local: multiple caches thaw on the
                                     // same thread and would clobber a shared borrow)
    size_t fill_ids_cap;             // capacity of fill_ids
    uint32_t *fill_bucket;           // bucket[s] = head block index for shard s
    uint32_t *fill_link;             // link[i] = next block index in same shard
    size_t fill_link_cap;            // capacity of fill_link
    size_t fill_n;                   // blocks in the current fill set
    _Atomic uint64_t fill_decoded;   // blocks actually decoded this fill (summed)
    _Atomic uint64_t fill_epoch;     // bumped by thaw to start a fill
    _Atomic uint64_t fill_done;      // workers bump after finishing an epoch
    pthread_mutex_t fill_mu;
    pthread_cond_t fill_start;       // thaw signals; workers wait
    pthread_cond_t fill_fin;         // workers signal; freeze waits
    int fill_stop;
    struct decode_item *dq;    // ring of pending decode items (slot bound is high;
    int dq_cap, dq_head, dq_tail;        // the real backpressure is dq_bytes below)
    size_t dq_bytes;           // compressed bytes currently queued (staging size)
    size_t dq_byte_budget;     // block producers above this (RAM-budgeted staging)
    pthread_cond_t dq_ne;      // not-empty (wake a decoder)
    pthread_cond_t dq_nf;      // not-full  (wake a blocked producer)
    int stop;
    _Atomic int dec_active;    // items currently inside decode->re-encode->append

    // Interactive download-request stack (LIFO): a render miss / prefetch pushes
    // "fetch region R". Download threads pop the NEWEST (current view) first; when
    // full, the OLDEST (stalest, camera moved on) is dropped. DEDUPING: an open-
    // addressing membership set (rs_set) gives O(1) push-dedup instead of an O(n)
    // stack scan -- the prefetch blasts the predicted set every tick, so the stack
    // absorbs duplicates natively without the caller deduping.
    uint64_t *reqstk;          // region keys (the LIFO order)
    uint64_t *rs_set;          // membership set, power-of-two, 0 = empty slot
    int rs_set_mask;           // rs_set capacity - 1
    int rs_cap, rs_n;
    pthread_t dlthreads[16];
    int ndl;

    // Single-flight: region keys currently popped-and-being-fetched/decoded (not
    // yet appended to the .mca, so coverage is still ABSENT). req_push and the
    // download path skip keys here so a region in flight is NOT re-requested and
    // re-decoded every frame during its ~500ms fetch+decode window (the cause of
    // a measured ~15x re-decode). Flat key array (in-flight count is bounded by
    // the decode-queue depth); guarded by the same v->mu. Cleared on append/fail.
    uint64_t *inflight;
    int inflight_cap, inflight_n;

    mc_volume_ready_fn ready_cb;   // fired when a region becomes serveable
    void *ready_ud;

    // Two frozen snapshots, collated once per THAW, read during the frozen render.
    // Same tick thread writes + reads -> plain fields, no atomic, no lock.
    //  net_inflight: ACTUAL download/decode pipeline work = queued downloads +
    //    downloading + decode-queue depth. This is the user-facing "downloading N"
    //    -- stable, reflects real network/transcode work.
    //  work_pending: net_inflight + this frame's undrained misses. The render gate
    //    keys off this ("keep ticking while anything is still settling"). The miss
    //    term swings 0..thousands per frame, so it must NOT leak into the status bar.
    uint64_t net_inflight;
    uint64_t work_pending;
    uint64_t change_gen;       // bumped at THAW when the fill changed cache content
    // Per-REGION change gens (direct-map, no stored keys): slot = hash(region).
    // Writers (decode pool / dl thread / THAW) store the current render gen;
    // collisions only make an unrelated region look changed (a harmless extra
    // render, never a missed one). Lets a viewer skip the streaming re-render
    // when nothing in ITS viewport changed -- the gate was volume-global before.
    _Atomic uint64_t *rgen;
    // Per-stage breakdown of net_inflight (same THAW-collated snapshot pattern):
    // download stack / on-the-wire / decode-queue wait / active decode->encode->
    // append (+ the staging RAM the decode queue holds). Append itself is a
    // synchronous memcpy into the mmap -- there is no archive queue.
    uint64_t snap_queued, snap_downloading, snap_decq, snap_encoding, snap_staging_bytes;
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

// RAM the item occupies in the staging queue: its buffered source bytes —
// c3d compressed, or v2 blosc/zstd/raw decoded-dense, whatever was read. The
// byte budget thus naturally holds fewer of the larger (decoded) v2 items.
static size_t decode_item_bytes(const decode_item *it) {
    size_t b = 0;
    for (int k = 0; k < it->nsub; ++k) b += it->rlen[k];
    return b ? b : 1;                                   // ZERO/air items count as 1
}

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

// Local-filesystem source: mirror of s3_read but reads "<prefix>/<key>" from
// disk. Same contract: return 0 with *out=NULL on a missing key (so the level
// probe / air detection behaves like a 404), <0 on real I/O error, 0 with a
// malloc'd buffer otherwise. `off`/`len` honor ranged reads (footer/index).
static int file_read(void *ud, const char *key, uint64_t off, uint64_t len,
                     uint8_t **out, size_t *out_len) {
    level_t *lv = ud;
    char path[1280];
    snprintf(path, sizeof path, "%s/%s", lv->prefix, key);
    *out = NULL; *out_len = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;                              // missing key == 404
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long fsz = ftell(f);
    if (fsz < 0) { fclose(f); return -1; }
    uint64_t start = off;
    uint64_t want  = (len == 0) ? (uint64_t)fsz : len;
    if (start > (uint64_t)fsz) { fclose(f); return 0; }      // past EOF
    if (start + want > (uint64_t)fsz) want = (uint64_t)fsz - start;
    if (fseek(f, (long)start, SEEK_SET) != 0) { fclose(f); return -1; }
    uint8_t *buf = malloc(want ? want : 1);
    if (!buf) { fclose(f); return -1; }
    size_t got = want ? fread(buf, 1, want, f) : 0;
    fclose(f);
    if (got != want) { free(buf); return -1; }
    atomic_fetch_add_explicit(&lv->vol->net_bytes, got, memory_order_relaxed);
    *out = buf; *out_len = got;
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
// decode one source inner-chunk's raw bytes into `dst` (edge^3). With c3d
// removed, surviving inputs (v2 blosc/zstd/raw) are already dense u8 — copy them.
static void decode_inner(const char *codec,
                         const uint8_t *raw, size_t rlen, uint8_t *dst, int edge) {
    (void)codec;
    size_t vox = (size_t)edge * edge * edge;
    if (rlen >= vox) memcpy(dst, raw, vox);
    else { memset(dst, 0, vox); memcpy(dst, raw, rlen); }
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
// `dense` and `tile` are PERSISTENT per-decoder-thread scratch (each CHUNK^3 =
// 16MB), allocated once in decoder_main and reused. Per-call posix_memalign of a
// 16MB buffer across N decode threads under heavy streaming drove the kernel into
// direct page reclaim (native_queued_spin_lock_slowpath storm in profiling).
static void decode_one(mc_volume *v, decode_item *it,
                       uint8_t *dense, uint8_t *tile) {
    const char *codec = mc_zarr_inner_codec(v->lv[it->lod].z);
    const int edge = CHUNK / it->sub;
    const uint64_t key = rkey(it->lod, it->rz, it->ry, it->rx);
    // Skip if this region became resident while the item sat in the queue (a
    // duplicate that slipped past the single-flight claim) — don't redo decode.
    if (mc_archive_chunk_coverage(v->arc, it->lod, it->rz, it->ry, it->rx) != MC_ABSENT) {
        for (int k = 0; k < it->nsub; ++k) free(it->raw[k]);
        pthread_mutex_lock(&v->mu); inflight_del(v, key); pthread_mutex_unlock(&v->mu);
        return;
    }
    if (it->nsub == 0) {                               // all air -> ZERO
        if (mc_archive_append_chunk_raw(v->arc, it->lod, it->rz, it->ry, it->rx, zero256()) == 0)
            vol_mark_region(v, it->lod, it->rz, it->ry, it->rx);   // else leave absent -> refetch
        pthread_mutex_lock(&v->mu); inflight_del(v, key); pthread_mutex_unlock(&v->mu);
        return;
    }
    double t_dec0 = mcv_now();
    if (it->sub == 1) {                                // one chunk == region
        decode_inner(codec, it->raw[0], it->rlen[0], dense, CHUNK);
    } else {                                           // v2: blit the cube
        memset(dense, 0, (size_t)CHUNK * CHUNK * CHUNK);
        for (int k = 0; k < it->nsub; ++k) {
            decode_inner(codec, it->raw[k], it->rlen[k], tile, edge);
            blit_sub(dense, tile, edge, it->oz[k], it->oy[k], it->ox[k]);
        }
    }
    double t_enc0 = mcv_now();
    if (mc_archive_append_chunk_raw(v->arc, it->lod, it->rz, it->ry, it->rx, dense) == 0)
        vol_mark_region(v, it->lod, it->rz, it->ry, it->rx);       // else leave absent -> refetch
    double t_end = mcv_now();
    MCVLOG("decoded   lod%d region(%d,%d,%d) codec=%s decode=%.0fms encode=%.0fms",
           it->lod, it->rz, it->ry, it->rx, codec,
           t_enc0 - t_dec0, t_end - t_enc0);
    for (int k = 0; k < it->nsub; ++k) free(it->raw[k]);
    pthread_mutex_lock(&v->mu); inflight_del(v, key); pthread_mutex_unlock(&v->mu);  // single-flight clear
}

// Decode-pool worker: drain decode items, decode off the download thread.
static void *decoder_main(void *ud) {
    mc_volume *v = ud;
    mc_thread_setname("mc-decode");        // distinguish in profilers
    // Persistent per-thread decode scratch (16MB each), allocated once and reused
    // across every decode -- per-call 16MB allocs across the pool drove kernel page
    // reclaim under heavy streaming. tile holds the sub-cube for v2 (edge<=128).
    uint8_t *dense = NULL, *tile = NULL;
    if (posix_memalign((void **)&dense, 64, (size_t)CHUNK * CHUNK * CHUNK) ||
        posix_memalign((void **)&tile,  64, (size_t)CHUNK * CHUNK * CHUNK)) {
        free(dense); free(tile);
        return NULL;                                 // OOM at startup: this worker bows out
    }
    for (;;) {
        pthread_mutex_lock(&v->mu);
        while (v->dq_head == v->dq_tail && !v->stop) pthread_cond_wait(&v->dq_ne, &v->mu);
        if (v->stop && v->dq_head == v->dq_tail) { pthread_mutex_unlock(&v->mu); break; }
        // LIFO: decode the NEWEST queued region first (pop from the producer's end).
        // Like the download stack, interactive coords go stale as the user moves --
        // the freshest region is what's on screen now, so decode it before the
        // backlog. The ring is used as a deque: producer pushes at dq_tail, we pop
        // dq_tail-1.
        v->dq_tail = (v->dq_tail + v->dq_cap - 1) % v->dq_cap;
        decode_item it = v->dq[v->dq_tail];
        v->dq_bytes -= decode_item_bytes(&it);         // free staging budget
        pthread_cond_signal(&v->dq_nf);                // wake a blocked producer
        pthread_mutex_unlock(&v->mu);
        atomic_fetch_add_explicit(&v->dec_active, 1, memory_order_relaxed);
        decode_one(v, &it, dense, tile);
        atomic_fetch_sub_explicit(&v->dec_active, 1, memory_order_relaxed);
        if (v->ready_cb) v->ready_cb(v->ready_ud);     // region became serveable
    }
    free(dense); free(tile);
    return NULL;
}

// Producer: push a decode item. Backpressure is BYTE-budgeted (RAM-budgeted
// staging): the download thread only blocks when the queued compressed bytes
// exceed dq_byte_budget — so downloads run far ahead of the CPU-bound decode
// pool and the network stays saturated, instead of stalling on a slot count.
// (A secondary slot-full guard covers the unlikely ring-wrap.) Takes ownership
// of the item's raw buffers.
static void decode_push(mc_volume *v, const decode_item *it) {
    const size_t ib = decode_item_bytes(it);
    pthread_mutex_lock(&v->mu);
    int next = (v->dq_tail + 1) % v->dq_cap;
    int blocked = 0;
    while (!v->stop &&
           (next == v->dq_head ||                       // ring full (rare; cap is huge)
            (v->dq_bytes + ib > v->dq_byte_budget && v->dq_head != v->dq_tail))) {
        blocked = 1;
        pthread_cond_wait(&v->dq_nf, &v->mu);
        next = (v->dq_tail + 1) % v->dq_cap;
    }
    if (v->stop) { pthread_mutex_unlock(&v->mu);
        for (int k = 0; k < it->nsub; ++k) free(it->raw[k]); return; }
    v->dq[v->dq_tail] = *it;
    v->dq_tail = next;
    v->dq_bytes += ib;
    pthread_cond_signal(&v->dq_ne);
    size_t qb = v->dq_bytes;
    pthread_mutex_unlock(&v->mu);
    if (blocked) MCVLOG("decode_q  FULL (staging budget hit) queued=%.0fMB", qb / 1048576.0);
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
// Single-flight helpers. ALL require v->mu held.
static int inflight_has(mc_volume *v, uint64_t key) {
    for (int i = 0; i < v->inflight_n; ++i) if (v->inflight[i] == key) return 1;
    return 0;
}
static void inflight_add(mc_volume *v, uint64_t key) {
    if (v->inflight_n == v->inflight_cap) {            // grow
        int nc = v->inflight_cap ? v->inflight_cap * 2 : 256;
        uint64_t *p = realloc(v->inflight, (size_t)nc * sizeof *p);
        if (!p) return;                                // OOM: skip tracking (dup possible, not fatal)
        v->inflight = p; v->inflight_cap = nc;
    }
    v->inflight[v->inflight_n++] = key;
}
static void inflight_del(mc_volume *v, uint64_t key) {
    for (int i = 0; i < v->inflight_n; ++i)
        if (v->inflight[i] == key) {                  // swap-remove
            v->inflight[i] = v->inflight[--v->inflight_n];
            return;
        }
}

// Deduping-stack membership set (open addressing, linear probe). key != 0 always
// (rkey packs lod in the high nibble; region 0,0,0 at lod>0 is fine, and lod0
// 0,0,0 -> key 0 would alias empty; guard below treats key 0 specially -- but
// rkey for (0,0,0,0) is 0, which never occurs as a real interactive request).
static int rs_set_has(mc_volume *v, uint64_t key) {
    uint32_t i = (uint32_t)((key * 0x9E3779B97F4A7C15ull) >> 40) & (uint32_t)v->rs_set_mask;
    for (int p = 0; p <= v->rs_set_mask; ++p) {
        uint64_t cur = v->rs_set[i];
        if (cur == 0) return 0;
        if (cur == key) return 1;
        i = (i + 1) & (uint32_t)v->rs_set_mask;
    }
    return 0;
}
static void rs_set_add(mc_volume *v, uint64_t key) {
    uint32_t i = (uint32_t)((key * 0x9E3779B97F4A7C15ull) >> 40) & (uint32_t)v->rs_set_mask;
    for (int p = 0; p <= v->rs_set_mask; ++p) {
        if (v->rs_set[i] == 0 || v->rs_set[i] == key) { v->rs_set[i] = key; return; }
        i = (i + 1) & (uint32_t)v->rs_set_mask;
    }
}
// Rebuild the set from the stack (after a removal leaves probe-chain holes).
static void rs_set_rebuild(mc_volume *v) {
    memset(v->rs_set, 0, ((size_t)v->rs_set_mask + 1) * sizeof(uint64_t));
    for (int i = 0; i < v->rs_n; ++i) rs_set_add(v, v->reqstk[i]);
}

static void req_push(mc_volume *v, int lod, int cz, int cy, int cx) {
    uint64_t key = rkey(lod, cz, cy, cx);
    pthread_mutex_lock(&v->mu);
    if (inflight_has(v, key)) { pthread_mutex_unlock(&v->mu); return; }  // already fetching/decoding
    if (rs_set_has(v, key)) { pthread_mutex_unlock(&v->mu); return; }    // already queued (O(1))
    if (v->rs_n == v->rs_cap) {                         // full -> drop bottom (stalest)
        memmove(&v->reqstk[0], &v->reqstk[1], (size_t)(v->rs_cap - 1) * sizeof(uint64_t));
        v->rs_n--;
        rs_set_rebuild(v);                              // membership shifted; rebuild
    }
    v->reqstk[v->rs_n++] = key;                         // push on top
    rs_set_add(v, key);
    MCVLOG("req_push  lod%d region(%d,%d,%d) stack_depth=%d", lod, cz, cy, cx, v->rs_n);
    pthread_cond_signal(&v->cv);
    pthread_mutex_unlock(&v->mu);
}

enum { DL_BATCH = 64 };   // per-batch regions; streaming GETs run this deep

// Streaming fetch of ONE 256^3 region via the reader (serial; the local-file
// source path and the fallback for a header window too small to parse).
static void mc_stream_fetch_region(mc_volume *v, int lod, int rz, int ry, int rx) {
    if (mc_archive_chunk_coverage(v->arc, lod, rz, ry, rx) != MC_ABSENT) return;  // already have it
    int rerr = 0;
    uint64_t off = mc_chunk_offset_chk(v->rd, lod, rz, ry, rx, &rerr);
    if (rerr) return;                                  // transient: leave ABSENT, retry
    if (off <= MC_SLOT_ZERO) {                          // CONFIRMED air (absent OR remote
        // ZERO slot) -> record a local ZERO region. Without this, a remote MC_SLOT_ZERO
        // (==1) chunk -- the masked margin at mid LODs -- stayed ABSENT locally, so the
        // render's LOD fallback walked PAST it to a coarser PRESENT level and sampled
        // the neighbor data bled in by unmasked pyramid downsampling (junk in the void).
        if (mc_archive_append_chunk_raw(v->arc, lod, rz, ry, rx, zero256()) == 0)
            vol_mark_region(v, lod, rz, ry, rx);
        return;
    }
    uint64_t blen = mc_reader_chunk_blob_len(v->rd, off);
    if (blen == 0) return;                             // transient: leave ABSENT, retry
    uint8_t *blob = malloc((size_t)blen);
    if (!blob) return;
    if (mc_reader_read_blob(v->rd, off, (size_t)blen, blob) != 0) { free(blob); return; }
    atomic_fetch_add_explicit(&v->net_bytes, blen, memory_order_relaxed);
    if (mc_archive_append_chunk_compressed(v->arc, lod, rz, ry, rx, blob, (size_t)blen) == 0)
        vol_mark_region(v, lod, rz, ry, rx);           // else leave absent -> refetch
    free(blob);
}

// Blob total length parsed from its LEADING bytes (header + fmap + bitmap + len
// table must all be inside `buf`). 0 = window too short (caller falls back to the
// exact serial path). Mirrors mc_reader_chunk_blob_len, but over one buffer.
static uint64_t mc_blob_len_parse(const uint8_t *buf, size_t len) {
    if (len < MC_BLOB_HDR) return 0;
    uint16_t fml; memcpy(&fml, buf + MC_BLOB_HDR - 2, 2);
    uint64_t bm_off = (uint64_t)MC_BLOB_HDR + fml;
    if (len < bm_off + MC_BITMAP_BYTES) return 0;
    int np = 0;
    for (int i = 0; i < MC_BITMAP_BYTES; ++i) np += __builtin_popcount(buf[bm_off + i]);
    if (!np) return bm_off + MC_BITMAP_BYTES;
    if (len < bm_off + MC_BITMAP_BYTES + (size_t)np * 2) return 0;
    uint64_t pay = 0;
    for (int i = 0; i < np; ++i) {
        uint16_t l; memcpy(&l, buf + bm_off + MC_BITMAP_BYTES + (size_t)i * 2, 2);
        pay += l;
    }
    return bm_off + MC_BITMAP_BYTES + (uint64_t)np * 2 + pay;
}

// Streaming fetch of a BATCH of regions -- the throughput path. The serial
// per-chunk flow (resolve, header probe, length probe, blob read) is 3-4 S3
// round-trips each; one thread doing that tops out ~2MB/s. Instead: resolve all
// offsets via the reader (node tables memoized, cheap), then TWO rounds of
// s3_get_batch (32-way concurrent ranged GETs, like the zarr path): round A pulls
// each blob's leading bytes -- sized at ~2x the running average blob size, so most
// blobs complete in this single GET (exact length parsed from the leading bytes)
// without over-reading small ones; round B pulls the occasional tail.
#define MC_STREAM_HDR_MIN (64u << 10)
#define MC_STREAM_HDR_MAX (2u << 20)
static void mc_stream_fetch_batch(mc_volume *v, int m, const int *lods,
                                  const int *rz, const int *ry, const int *rx) {
    uint64_t off[DL_BATCH];
    int act[DL_BATCH], na = 0;                 // regions that still need bytes
    for (int i = 0; i < m; ++i) {
        if (mc_archive_chunk_coverage(v->arc, lods[i], rz[i], ry[i], rx[i]) != MC_ABSENT)
            continue;                                          // already resident
        int rerr = 0;
        uint64_t o = mc_chunk_offset_chk(v->rd, lods[i], rz[i], ry[i], rx[i], &rerr);
        if (rerr) continue;                                    // transient: leave ABSENT, retry
        if (o <= MC_SLOT_ZERO) {                               // CONFIRMED air (absent OR remote
            // ZERO slot) -> local ZERO region. A remote MC_SLOT_ZERO must record locally
            // as ZERO, else the masked margin stays ABSENT and the LOD fallback bleeds
            // coarser PRESENT data into the void.
            if (mc_archive_append_chunk_raw(v->arc, lods[i], rz[i], ry[i], rx[i], zero256()) == 0)
                vol_mark_region(v, lods[i], rz[i], ry[i], rx[i]);  // else leave absent -> refetch
            continue;
        }
        off[i] = o; act[na++] = i;
    }
    if (!na) return;

    if (v->local || !v->s3mca) {               // local-file source: disk is fast, go serial
        for (int k = 0; k < na; ++k)
            mc_stream_fetch_region(v, lods[act[k]], rz[act[k]], ry[act[k]], rx[act[k]]);
        return;
    }

    // Chunks are appended contiguously in the source archive, and the requested
    // regions are spatially coherent (a viewport) -- so their blobs cluster.
    // Sort by offset and COALESCE into a few LARGE sequential ranges (read through
    // small gaps), then carve each blob out of the run buffer. A handful of
    // multi-MB GETs reaches S3 large-object throughput where per-chunk ~200KB
    // GETs stall in TCP slow-start. Interior blob lengths are exact (next_off -
    // off bounds it; the parse is authoritative); each run's last blob gets an
    // EMA-sized margin, with a (rare) follow-up GET if the parse says longer.
    if (!v->s_blob_ema) v->s_blob_ema = 256u << 10;          // first batch: 256KB guess
    uint64_t margin = v->s_blob_ema * 2;
    if (margin < MC_STREAM_HDR_MIN) margin = MC_STREAM_HDR_MIN;
    if (margin > MC_STREAM_HDR_MAX) margin = MC_STREAM_HDR_MAX;

    // sort act[] by offset (insertion sort; na <= DL_BATCH).
    for (int a = 1; a < na; ++a) {
        int t = act[a]; int b = a - 1;
        while (b >= 0 && off[act[b]] > off[t]) { act[b + 1] = act[b]; --b; }
        act[b + 1] = t;
    }

    enum { RUN_GAP = 512 << 10, RUN_MAX = 64 << 20 };  // read-through gap / run size cap:
    // merge only near-adjacent blobs (raster archives put y/z neighbors MBs apart;
    // reading through those gaps wasted ~3x the useful bytes)
    s3_range_req runq[DL_BATCH]; s3_response runr[DL_BATCH];
    int rs[DL_BATCH], re[DL_BATCH], nrun = 0;          // act[] index span of each run
    for (int k = 0; k < na;) {
        int s = k, e = k;
        uint64_t end = off[act[k]] + margin;
        while (e + 1 < na &&
               off[act[e + 1]] <= end + RUN_GAP &&
               off[act[e + 1]] + margin - off[act[s]] <= RUN_MAX) {
            ++e;
            end = off[act[e]] + margin;
        }
        runq[nrun] = (s3_range_req){v->s3mca->url, off[act[s]], end - off[act[s]]};
        rs[nrun] = s; re[nrun] = e; ++nrun;
        k = e + 1;
    }

    memset(runr, 0, sizeof(s3_response) * (size_t)nrun);
    s3_get_batch(v->s3mca->cl, runq, (size_t)nrun, 16, runr);

    for (int r = 0; r < nrun; ++r) {
        if (!s3_response_ok(&runr[r]) || !runr[r].body_len) {  // transient: retry later
            s3_response_free(&runr[r]);
            continue;
        }
        atomic_fetch_add_explicit(&v->net_bytes, runr[r].body_len, memory_order_relaxed);
        for (int k = rs[r]; k <= re[r]; ++k) {
            int i = act[k];
            uint64_t rel = off[i] - runq[r].offset;
            if (rel >= runr[r].body_len) continue;             // run came back short
            size_t avail = runr[r].body_len - (size_t)rel;
            uint64_t bl = mc_blob_len_parse(runr[r].body + rel, avail);
            if (!bl) {                                         // window short of the header
                mc_stream_fetch_region(v, lods[i], rz[i], ry[i], rx[i]);
                continue;
            }
            v->s_blob_ema = (v->s_blob_ema * 7 + (size_t)bl) / 8;   // dl thread only
            if (bl <= avail) {                                 // whole blob in the run
                if (mc_archive_append_chunk_compressed(v->arc, lods[i], rz[i], ry[i], rx[i],
                                                       runr[r].body + rel, (size_t)bl) == 0)
                    vol_mark_region(v, lods[i], rz[i], ry[i], rx[i]);   // else leave absent -> refetch
            } else {                                           // run-edge tail: one follow-up GET
                s3_response tail = {0};
                if (s3_get_range(v->s3mca->cl, v->s3mca->url, off[i] + avail, bl - avail,
                                 &tail) == S3_OK &&
                    s3_response_ok(&tail) && tail.body_len >= bl - avail) {
                    atomic_fetch_add_explicit(&v->net_bytes, tail.body_len, memory_order_relaxed);
                    uint8_t *blob = malloc((size_t)bl);
                    if (blob) {
                        memcpy(blob, runr[r].body + rel, avail);
                        memcpy(blob + avail, tail.body, (size_t)(bl - avail));
                        if (mc_archive_append_chunk_compressed(v->arc, lods[i], rz[i], ry[i], rx[i],
                                                               blob, (size_t)bl) == 0)
                            vol_mark_region(v, lods[i], rz[i], ry[i], rx[i]);
                        free(blob);
                    }
                }
                s3_response_free(&tail);
            }
        }
        s3_response_free(&runr[r]);
    }
}

// Download thread: pop the newest request, download its shard (-> decode queue).
// Drain up to DL_BATCH region requests from the LIFO stack (newest first) and
// fetch their source chunks in ONE s3_get_batch — per-region precision (no
// whole-shard pull) AND high concurrency (32 GETs over the pooled connections),
// which is what saturates the link. c3d (sub=1) = one chunk per region; v2
// regions fall back to the per-chunk cube path. STREAMING: one thread drives
// batched concurrent GETs (mc_stream_fetch_batch) -- verbatim copy, no decode pool.
static void *dl_main(void *ud) {
    mc_volume *v = ud;
    mc_thread_setname("mc-download");      // distinguish in profilers
    for (;;) {
        int lods[DL_BATCH], rz[DL_BATCH], ry[DL_BATCH], rx[DL_BATCH];
        int m = 0;
        pthread_mutex_lock(&v->mu);
        while (v->rs_n == 0 && !v->stop) pthread_cond_wait(&v->cv, &v->mu);
        if (v->stop && v->rs_n == 0) { pthread_mutex_unlock(&v->mu); return NULL; }
        int popped = 0;
        // Grab only this thread's SHARE of the queue, not the whole DL_BATCH. One
        // greedy thread taking 64 regions left the other ndl-1 download threads
        // asleep on an empty stack -> a single s3_get_batch in flight -> the link
        // idles. Split the work so all ndl threads issue concurrent batches.
        int grab = (v->rs_n + v->ndl - 1) / v->ndl;     // ceil(queue / threads)
        if (grab < 4) grab = 4;                          // small floor: avoid tiny GETs
        if (grab > DL_BATCH) grab = DL_BATCH;
        while (m < grab && v->rs_n > 0) {               // grab a slice, newest first
            uint64_t key = v->reqstk[--v->rs_n];
            popped = 1;
            if (inflight_has(v, key)) continue;          // another dl thread already has it
            inflight_add(v, key);                        // single-flight: claim it
            runpack(key, &lods[m], &rz[m], &ry[m], &rx[m]);
            ++m;
        }
        if (popped) rs_set_rebuild(v);                  // popped keys leave the set
        // Work remains -> wake another download thread so the pool fans out
        // instead of this thread draining the queue alone.
        if (v->rs_n > 0) pthread_cond_signal(&v->cv);
        pthread_mutex_unlock(&v->mu);

        if (v->streaming) {
            // Verbatim .mca -> .mca copy, batched (two rounds of 32-way GETs).
            mc_stream_fetch_batch(v, m, lods, rz, ry, rx);
            pthread_mutex_lock(&v->mu);
            for (int i = 0; i < m; ++i)
                inflight_del(v, rkey(lods[i], rz[i], ry[i], rx[i]));
            pthread_mutex_unlock(&v->mu);
            if (m && v->ready_cb) v->ready_cb(v->ready_ud);
            continue;
        }

        // Locate each region's c3d source chunk via the cached footer; build one
        // batched ranged-GET. v2 (sub>1) and local both use the per-region path.
        s3_range_req reqs[DL_BATCH];
        s3_response  resp[DL_BATCH];
        char urls[DL_BATCH][1280];
        int ri[DL_BATCH], nq = 0;                       // ri[q] -> request index
        // Each of the m claimed regions is in-flight. It is cleared exactly when
        // it does NOT produce a decode_item here (decode_one clears the ones that
        // do). clr() drops the single-flight claim for region i so a later miss
        // re-requests it.
        #define DL_CLR(i) do { uint64_t _k = rkey(lods[i], rz[i], ry[i], rx[i]); \
            pthread_mutex_lock(&v->mu); inflight_del(v, _k); pthread_mutex_unlock(&v->mu); } while (0)
        for (int i = 0; i < m; ++i) {
            mc_zarr *z = v->lv[lods[i]].z;
            const int sub = CHUNK / mc_zarr_inner_edge(z);
            if (v->local || sub != 1) {                 // local / v2: direct per-region
                mc_volume_prefetch_region(v, lods[i], rz[i]*sub, ry[i]*sub, rx[i]*sub);
                DL_CLR(i);                               // prefetch_region appends synchronously
                continue;
            }
            if (mc_archive_chunk_coverage(v->arc, lods[i], rz[i], ry[i], rx[i]) != MC_ABSENT)
                { DL_CLR(i); continue; }                 // already resident
            char key[64]; uint64_t off, nb;
            int st = mc_zarr_chunk_locate(z, rz[i], ry[i], rx[i], key, &off, &nb);
            if (st < 0)                                 // transient read error: leave ABSENT
                { DL_CLR(i); continue; }                 // (retry on next miss), do NOT mark air
            if (st > 0) {                               // CONFIRMED air (footer ok + air marker)
                decode_item air = {lods[i], rz[i], ry[i], rx[i], 1, 0, {0},{0},{0}, {0},{0}};
                decode_push(v, &air);                    // decode_one clears in-flight
                continue;
            }
            snprintf(urls[nq], sizeof urls[nq], "%s/%s", v->lv[lods[i]].prefix, key);
            reqs[nq] = (s3_range_req){urls[nq], off, nb};
            ri[nq] = i;
            ++nq;
        }
        if (nq == 0) continue;
        MCVLOG("dl_batch  %d regions -> %d ranged GETs", m, nq);
        memset(resp, 0, sizeof(s3_response) * nq);
        s3_get_batch(v->s3, reqs, (size_t)nq, 32, resp);
        for (int q = 0; q < nq; ++q) {
            int i = ri[q];
            int pushed = 0;
            if (s3_response_ok(&resp[q]) && resp[q].body_len >= reqs[q].length && reqs[q].length) {
                uint8_t *raw = malloc(reqs[q].length);
                if (raw) {
                    memcpy(raw, resp[q].body, reqs[q].length);
                    atomic_fetch_add_explicit(&v->net_bytes, reqs[q].length, memory_order_relaxed);
                    decode_item it = {lods[i], rz[i], ry[i], rx[i], 1, 1, {0},{0},{0}, {0},{0}};
                    it.raw[0] = raw; it.rlen[0] = reqs[q].length;
                    decode_push(v, &it);                 // decode_one clears in-flight
                    pushed = 1;
                }
            }
            if (!pushed) DL_CLR(i);                      // download failed -> retry next miss
            s3_response_free(&resp[q]);
        }
        #undef DL_CLR
    }
}

// Blocking fill of one region (get_block / CLI): download its shard synchronously
// through the same decode queue, then wait for that region's coverage to resolve.
static mc_cover ensure_region(mc_volume *v, int lod, int cz, int cy, int cx) {
    mc_cover cov = mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx);
    if (cov != MC_ABSENT) return cov;
    if (v->streaming) {
        req_push(v, lod, cz, cy, cx);          // dl thread copies the blob verbatim
    } else {
        const int sub = CHUNK / mc_zarr_inner_edge(v->lv[lod].z);
        mc_volume_prefetch_region(v, lod, cz * sub, cy * sub, cx * sub);  // just this region
    }
    // wait for the downloaders/decoders to drain enough that this region is covered.
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

// (c3g removed) no alternate block codec to select; kept as a no-op so the
// mc_volume open path is unchanged.
static void mc_volume_apply_codec_env(mc_archive *arc) {
    (void)arc;
}

mc_volume *mc_volume_open(const char *url, const char *cache_dir,
                          size_t cache_bytes, float quality) {
    return mc_volume_open_ex(url, cache_dir, cache_bytes, quality, NULL);
}

mc_volume *mc_volume_open_ex(const char *url, const char *cache_dir,
                             size_t cache_bytes, float quality,
                             const mc_volume_config *cfg) {
    if (!url || !cache_dir) return NULL;
    mc_volume *v = calloc(1, sizeof *v);
    if (!v) return NULL;
    v->quality = quality;
    atomic_init(&v->net_bytes, 0);
    pthread_mutex_init(&v->mu, NULL);
    pthread_cond_init(&v->cv, NULL);

    // Local vs remote: a URL with a "scheme://" is remote (s3/https), otherwise
    // `url` is a local filesystem directory read via file_read (no S3 client).
    v->local = (strstr(url, "://") == NULL);
    mc_zarr_read_fn read_cb = v->local ? file_read : s3_read;

    snprintf(v->root, sizeof v->root, "%s", url);
    rstrip_slash(v->root);

    // discover levels: probe "<root>/<i>" for i=0.. until a gap (each level's
    // .zarray / zarr.json). For S3 we AUTO-DETECT access: try ANONYMOUSLY first
    // (open-data buckets like the Vesuvius public bucket reject signed requests
    // made with the wrong/expired account), and only if level 0 doesn't open do
    // we rebuild the client with resolved credentials and retry. This needs no
    // config and works for both public and private buckets.
    if (v->local) {
        for (int i = 0; i < MAXLOD; ++i) {
            level_t *lv = &v->lv[i];
            lv->vol = v;
            snprintf(lv->prefix, sizeof lv->prefix, "%s/%d", v->root, i);
            mc_zarr *z = mc_zarr_open(read_cb, lv);
            if (!z) { lv->prefix[0] = 0; break; }
            lv->z = z;
            v->nlods = i + 1;
        }
    } else {
        for (int attempt = 0; attempt < 2 && v->nlods == 0; ++attempt) {
            if (v->s3) { s3_client_free(v->s3); v->s3 = NULL; }
            s3_config cfg = {0};
            s3_credentials creds = {0};
            if (attempt == 1) {                 // signed retry: resolve creds
                if (s3_credentials_load(NULL, &creds) != S3_OK) break;  // none -> nothing to try
                cfg.creds = creds;
            }
            v->s3 = s3_client_new(&cfg);         // attempt 0 -> anonymous (no creds)
            s3_credentials_free(&creds);
            if (!v->s3) { free(v); return NULL; }
            for (int i = 0; i < MAXLOD; ++i) {
                level_t *lv = &v->lv[i];
                lv->vol = v;
                snprintf(lv->prefix, sizeof lv->prefix, "%s/%d", v->root, i);
                mc_zarr *z = mc_zarr_open(read_cb, lv);
                if (!z) { lv->prefix[0] = 0; break; }
                lv->z = z;
                v->nlods = i + 1;
            }
        }
    }
    if (v->nlods == 0) { if (v->s3) s3_client_free(v->s3); free(v); return NULL; }

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
    mc_volume_apply_codec_env(v->arc);
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
    // Staging queue: downloaded compressed chunks wait here for the (CPU-bound)
    // decode pool. Backpressure is BYTE-budgeted (default 2GB) so downloads run
    // far ahead of decode and the network stays saturated, decoupling the two.
    // The ring slot count just needs to exceed however many items fit in the
    // budget — size it from the budget assuming small (~64KB) chunks, capped.
    v->dq_byte_budget = (cfg && cfg->staging_bytes > 0) ? cfg->staging_bytes
                                                        : (2ull << 30);   // 2 GB
    if (v->dq_byte_budget < (64ull << 20)) v->dq_byte_budget = 64ull << 20;
    v->dq_bytes = 0;
    v->dq_cap = (int)(v->dq_byte_budget / (64ull << 10)) + 8;   // ~budget/64KB slots
    if (v->dq_cap < 256) v->dq_cap = 256; if (v->dq_cap > 131072) v->dq_cap = 131072;
    v->dq = calloc((size_t)v->dq_cap, sizeof *v->dq);
    // LIFO request stack: just 8-byte region keys, so it can be large — a fast
    // navigation enqueues many thousands of on-screen region misses, and we
    // don't want to drop ones still in view. 64K keys = 512KB.
    v->rs_cap = (cfg && cfg->request_stack > 0) ? cfg->request_stack : 65536;
    if (v->rs_cap < 256) v->rs_cap = 256; if (v->rs_cap > (1<<22)) v->rs_cap = (1<<22);
    v->reqstk = calloc((size_t)v->rs_cap, sizeof *v->reqstk);
    // Membership set: next pow2 >= 2*rs_cap (load factor <= 0.5 -> short probes).
    int sc = 1; while (sc < v->rs_cap * 2) sc <<= 1;
    v->rs_set_mask = sc - 1;
    v->rs_set = calloc((size_t)sc, sizeof *v->rs_set);
    v->rgen = calloc(MC_RGEN_SLOTS, sizeof *v->rgen);

    // Decoders default to nproc/2: the c3d/wavelet decode is memory-bandwidth-
    // bound (a 256^3 decode streams ~16MB), so threads past ~half the cores
    // saturate the bus and only inflate per-decode latency (measured: 1T 103ms,
    // 8T 176ms, 16T 305ms — ~same throughput past 8 but 2x the latency). Caller
    // may override via mc_volume_config; clamp to the fixed pool arrays.
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    int nd = (cfg && cfg->decoders > 0) ? cfg->decoders
                                        : (nproc > 2 ? (int)(nproc / 2) : 2);
    if (nd < 1) nd = 1; if (nd > 32) nd = 32;          // decoders[32]
    v->ndecoders = nd;
    for (int i = 0; i < v->ndecoders; ++i)
        pthread_create(&v->decoders[i], NULL, decoder_main, v);
    fill_pool_init(v, nd);                              // archive->cache fill, off the GUI thread
    int ndl = (cfg && cfg->dl_threads > 0) ? cfg->dl_threads : 8;
    if (ndl < 1) ndl = 1; if (ndl > 16) ndl = 16;      // dlthreads[16]
    v->ndl = ndl;
    for (int i = 0; i < v->ndl; ++i)
        pthread_create(&v->dlthreads[i], NULL, dl_main, v);
    MCVLOG("open      %s  decoders=%d dl_threads=%d staging=%.1fGB(slots=%d) rs_cap=%d",
           url, v->ndecoders, v->ndl, v->dq_byte_budget / 1073741824.0, v->dq_cap, v->rs_cap);
    return v;
}

// mmap a local .mca read-only: the flat reader (mc_open) then treats it as one
// big array -- chunk resolves and blob reads are plain pointer reads, the kernel
// pages in on demand (nothing is slurped into RAM).
static uint8_t *mc_map_file_ro(const char *path, size_t *out_len) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return NULL; }
    uint8_t *base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);                                   // the mapping keeps the file alive
    if (base == MAP_FAILED) return NULL;
    *out_len = (size_t)st.st_size;
    return base;
}

// Probe a built .mca's header WITHOUT opening a volume (no local archive, no
// threads): LOD0 dims, lod count, quality. url = s3://, https://, or a local
// path. Returns 0 on success.
int mc_mca_probe(const char *url, int *nx, int *ny, int *nz, int *nlods, float *quality) {
    if (!url) return -1;
    mc_s3 *s = NULL; mc_reader *r = NULL;
    uint8_t *map = NULL; size_t maplen = 0;
    if (strstr(url, "://")) {
        s = mc_s3_open(url);
        if (!s) return -1;
        r = mc_s3_reader(s);
    } else {
        map = mc_map_file_ro(url, &maplen);
        if (!map) return -1;
        r = mc_open(map, maplen);
        if (!r) { munmap(map, maplen); return -1; }
    }
    mc_reader_dims(r, nx, ny, nz);
    if (nlods) *nlods = mc_reader_nlods(r);
    if (quality) *quality = mc_reader_quality(r);
    if (s) mc_s3_close(s); else { mc_close(r); munmap(map, maplen); }
    return 0;
}

// Open an already-built remote (or local-file) .mca and stream it into a LOCAL
// .mca on demand. Same machinery as the zarr transcode path -- local archive,
// download threads, decode-from-local THAW -- but the download step COPIES the
// remote chunk's compressed blob verbatim onto the local archive (no decode, no
// re-encode): a 256^3 .mca chunk is already in the exact local format. We never
// mirror the whole remote; only the chunks the view touches get pulled+appended.
//   url        : the remote (s3://.../https://...) or local-file source .mca
//   cache_dir  : holds the local <name>.mca that fetched chunks append into
//   cache_bytes: resident RAM decoded-block budget
// Returns NULL on failure.
mc_volume *mc_volume_open_streaming(const char *url, const char *cache_dir,
                                    size_t cache_bytes) {
    if (!url || !cache_dir) return NULL;
    mc_volume *v = calloc(1, sizeof *v);
    if (!v) return NULL;
    v->streaming = 1;
    atomic_init(&v->net_bytes, 0);
    pthread_mutex_init(&v->mu, NULL);
    pthread_cond_init(&v->cv, NULL);
    snprintf(v->root, sizeof v->root, "%s", url);
    rstrip_slash(v->root);

    // Open the SOURCE reader (remote ranged GETs, or a local file mmap'd read-only
    // and read like one big array) for chunk-offset resolves + verbatim blob reads.
    v->local = (strstr(url, "://") == NULL);
    if (v->local) {
        v->s_map = mc_map_file_ro(v->root, &v->s_map_len);
        if (!v->s_map) { free(v); return NULL; }
        v->rd = mc_open(v->s_map, v->s_map_len);
        if (!v->rd) { munmap(v->s_map, v->s_map_len); free(v); return NULL; }
    } else {
        v->s3mca = mc_s3_open(url);
        if (!v->s3mca) { free(v); return NULL; }
        v->rd = mc_s3_reader(v->s3mca);
    }

    int n0x, n0y, n0z;
    mc_reader_dims(v->rd, &n0x, &n0y, &n0z);
    v->nlods = mc_reader_nlods(v->rd);
    if (v->nlods > MAXLOD) v->nlods = MAXLOD;
    if (v->nlods <= 0) {
        if (v->s3mca) mc_s3_close(v->s3mca); else { mc_close(v->rd); if (v->s_map) munmap(v->s_map, v->s_map_len); }
        free(v); return NULL;
    }
    v->quality = mc_reader_quality(v->rd);
    for (int l = 0; l < v->nlods; ++l) {
        v->s_nz[l] = n0z >> l < 1 ? 1 : n0z >> l;
        v->s_ny[l] = n0y >> l < 1 ? 1 : n0y >> l;
        v->s_nx[l] = n0x >> l < 1 ? 1 : n0x >> l;
    }

    // Local archive: fetched chunks append here verbatim. Same dims/quality as the
    // source so chunk coords + blob format line up exactly.
    char path[2048];
    const char *base = strrchr(v->root, '/');
    base = base ? base + 1 : v->root;
    snprintf(path, sizeof path, "%s/%s.local.mca", cache_dir, base);
    v->arc = mc_archive_open_dims(path, n0x, n0y, n0z, v->quality);
    if (!v->arc) {
        if (v->s3mca) mc_s3_close(v->s3mca); else { mc_close(v->rd); if (v->s_map) munmap(v->s_map, v->s_map_len); }
        free(v); return NULL;
    }
    v->cache = mc_cache_new_archive(cache_bytes, v->arc);
    if (!v->cache) {
        mc_archive_close(v->arc);
        if (v->s3mca) mc_s3_close(v->s3mca); else { mc_close(v->rd); if (v->s_map) munmap(v->s_map, v->s_map_len); }
        free(v); return NULL;
    }

    // Download threads only (no decode pool -- streaming appends compressed blobs
    // verbatim, there's nothing to decode off-thread). Reuse the LIFO request stack.
    v->rs_cap = 65536;
    v->reqstk = calloc((size_t)v->rs_cap, sizeof *v->reqstk);
    int sc = 1; while (sc < v->rs_cap * 2) sc <<= 1;
    v->rs_set_mask = sc - 1;
    v->rs_set = calloc((size_t)sc, sizeof *v->rs_set);
    v->rgen = calloc(MC_RGEN_SLOTS, sizeof *v->rgen);
    // ONE download thread: the source reader (cbuf + node-table cache + codec ctx)
    // is non-reentrant, so a single owner avoids any sharing. Chunks are large and
    // partial-fetch is efficient; this runs off the UI thread, which is the point.
    v->ndl = 1;
    for (int i = 0; i < v->ndl; ++i)
        pthread_create(&v->dlthreads[i], NULL, dl_main, v);
    // Archive->cache fill pool (decode local blobs off the GUI thread). Streaming has
    // no decode pool, so size from nproc/2.
    { long nproc = sysconf(_SC_NPROCESSORS_ONLN);
      fill_pool_init(v, nproc > 2 ? (int)(nproc / 2) : 2); }
    MCVLOG("open(stream) %s -> %s  nlods=%d dims=%dx%dx%d q=%.1f dl=%d fillers=%d",
           url, path, v->nlods, n0x, n0y, n0z, v->quality, v->ndl, v->nfillers);
    return v;
}

void mc_volume_free(mc_volume *v) {
    if (!v) return;
    if (v->streaming) {
        // Stop the download threads, then tear down cache + local archive + source
        // reader. No decode pool / zarr levels exist in streaming.
        pthread_mutex_lock(&v->mu);
        v->stop = 1;
        pthread_cond_broadcast(&v->cv);
        pthread_mutex_unlock(&v->mu);
        for (int i = 0; i < v->ndl; ++i) pthread_join(v->dlthreads[i], NULL);
        fill_pool_stop(v);                     // stop fillers before freeing the cache
        pthread_mutex_destroy(&v->fill_mu);
        pthread_cond_destroy(&v->fill_start); pthread_cond_destroy(&v->fill_fin);
        if (v->cache) mc_cache_free(v->cache);
        if (v->arc) mc_archive_close(v->arc);
        if (v->s3mca) mc_s3_close(v->s3mca);   // owns the remote reader
        else { mc_close(v->rd); if (v->s_map) munmap(v->s_map, v->s_map_len); }
        free(v->reqstk);
        free(v->rs_set);
        free((void *)v->rgen);
        free(v->inflight);
        pthread_mutex_destroy(&v->mu);
        pthread_cond_destroy(&v->cv);
        free(v);
        return;
    }
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
    fill_pool_stop(v);                         // stop fillers before freeing the cache
    pthread_mutex_destroy(&v->fill_mu);
    pthread_cond_destroy(&v->fill_start); pthread_cond_destroy(&v->fill_fin);
    if (v->cache) mc_cache_free(v->cache);
    if (v->arc) mc_archive_close(v->arc);
    for (int i = 0; i < v->nlods; ++i) if (v->lv[i].z) mc_zarr_free(v->lv[i].z);
    if (v->s3) s3_client_free(v->s3);
    free(v->dq);
    free(v->reqstk);
    free(v->rs_set);
    free((void *)v->rgen);
    free(v->inflight);
    pthread_mutex_destroy(&v->mu);
    pthread_cond_destroy(&v->cv);
    free(v);
}

int  mc_volume_nlods(const mc_volume *v) { return v ? v->nlods : 0; }
// The volume's local archive (where transcoded chunks land). Lets a client read
// compressed block blobs (mc_archive_block_blob) / the block codec for the GPU
// decode path. The transcode pipeline must have produced the chunk first
// (mc_volume_get_block / prefetch); absent chunks have no blob yet.
mc_archive *mc_volume_archive(mc_volume *v) { return v ? v->arc : NULL; }
void mc_volume_shape(const mc_volume *v, int lod, int *nz, int *ny, int *nx) {
    if (v->streaming) {
        if (lod < 0 || lod >= v->nlods) { if(nz)*nz=0; if(ny)*ny=0; if(nx)*nx=0; return; }
        if (nz) *nz = v->s_nz[lod]; if (ny) *ny = v->s_ny[lod]; if (nx) *nx = v->s_nx[lod];
        return;
    }
    mc_zarr_shape(v->lv[lod].z, nz, ny, nx);
}
void mc_volume_block_grid(const mc_volume *v, int lod, int *nz, int *ny, int *nx) {
    int sz, sy, sx;
    mc_volume_shape(v, lod, &sz, &sy, &sx);
    if (nz) *nz = (sz + BLK - 1) / BLK;
    if (ny) *ny = (sy + BLK - 1) / BLK;
    if (nx) *nx = (sx + BLK - 1) / BLK;
}
int mc_volume_get_level_meta(const mc_volume *v, int lod, mc_volume_level_meta *out) {
    if (!v || !out || lod < 0 || lod >= v->nlods) return -1;
    if (v->streaming) {
        mc_volume_shape(v, lod, &out->shape[0], &out->shape[1], &out->shape[2]);
        out->inner_edge = CHUNK; out->shard_edge = CHUNK;
        snprintf(out->codec, sizeof out->codec, "mc");
        return 0;
    }
    const mc_zarr *z = v->lv[lod].z;
    mc_zarr_shape(z, &out->shape[0], &out->shape[1], &out->shape[2]);
    out->inner_edge = mc_zarr_inner_edge(z);
    out->shard_edge = mc_zarr_shard_edge(z);
    const char *c = mc_zarr_inner_codec(z);
    snprintf(out->codec, sizeof out->codec, "%s", c ? c : "");
    return 0;
}

// ---------------------------------------------------------------------------
// block serving
// ---------------------------------------------------------------------------
// Both paths (zarr transcode AND remote-.mca streaming) serve from the LOCAL
// archive, so coverage is just the local archive's: ABSENT -> the download stack
// pulls the chunk (re-encode for zarr / verbatim copy for streaming).
static inline mc_cover vol_coverage(mc_volume *v, int lod, int cz, int cy, int cx) {
    return mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx);
}

int mc_volume_try_block(mc_volume *v, int lod, int bz, int by, int bx, uint8_t *dst) {
    if (lod < 0 || lod >= v->nlods) { memset(dst, 0, BLK * BLK * BLK); return 0; }
    int cz = bz / PER, cy = by / PER, cx = bx / PER;
    mc_cover cov = vol_coverage(v, lod, cz, cy, cx);
    if (cov == MC_ABSENT) {
        // Record the absent miss at REGION granularity (the region's corner block),
        // NOT per 16^3 block. A thin slice touches thousands of blocks across an
        // absent region but they all dedupe to one region key -- so the per-frame
        // miss set is ~tens (one per absent region) instead of ~thousands. THAW
        // drains these region keys and issues one download each. (Per-block absent
        // recording flooded the 64K miss table every frame -> the ~2ms thaw floor.)
        mc_cache_miss_mark(v->cache, lod, cz * PER, cy * PER, cx * PER);
        memset(dst, 0, BLK * BLK * BLK);
        return 0;
    }
    if (cov == MC_ZERO) { memset(dst, 0, BLK * BLK * BLK); return 1; }
    mc_cache_get_copy(v->cache, lod, bz, by, bx, dst);
    return 1;
}

// Shared 16^3 zero block (air). One static, read-only -- samplers point at it
// instead of each copying zeros into a per-entry buffer.
static const uint8_t *mc_zero16(void) {
    static uint8_t z[BLK * BLK * BLK];   // zero-initialized (BSS)
    return z;
}

// Pointer-returning block accessor (NO copy). Returns a STABLE pointer valid for
// the duration of a frozen frame: into the cache arena (RAM hit), the shared zero
// block (air), or NULL (absent or present-but-not-cached -> caller falls coarser).
// This replaces the copy-into-tmp path so the sampler memo can hold pointers, not
// 4KB buffers -- killing the per-frame 8MB sampler alloc + a per-block memcpy.
const uint8_t *mc_volume_block_ptr(mc_volume *v, int lod, int bz, int by, int bx) {
    if (lod < 0 || lod >= v->nlods) return NULL;
    int cz = bz / PER, cy = by / PER, cx = bx / PER;
    mc_cover cov = vol_coverage(v, lod, cz, cy, cx);
    if (cov == MC_ZERO) return mc_zero16();
    if (cov == MC_ABSENT) {       // build path only; streaming never returns ABSENT
        mc_cache_miss_mark(v->cache, lod, cz * PER, cy * PER, cx * PER);
        return NULL;
    }
    // present: arena pointer on a RAM hit; NULL (+ recorded miss) otherwise.
    return mc_cache_get(v->cache, lod, bz, by, bx);
}

// Predictive prefetch: request the 256^3 REGION (cz,cy,cx) be downloaded +
// transcoded if it isn't already resident/in-flight. Cheap and non-blocking: a
// coverage probe (O(1) memo) + a push onto the LIFO download stack (deduped vs
// in-flight/queued). No decode, no scratch. Present/air -> no-op. Called from the
// tick BEFORE freeze, from the geometry-predicted working set, so regions stream
// in ahead of the render instead of being discovered as misses a cycle late.
void mc_volume_request_region(mc_volume *v, int lod, int cz, int cy, int cx) {
    if (!v || lod < 0 || lod >= v->nlods) return;
    if (mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx) != MC_ABSENT) return;
    req_push(v, lod, cz, cy, cx);   // download thread fetches+appends (zarr or stream)
}

int mc_volume_get_block(mc_volume *v, int lod, int bz, int by, int bx, uint8_t *dst) {
    if (lod < 0 || lod >= v->nlods) { memset(dst, 0, BLK * BLK * BLK); return -1; }
    int cz = bz / PER, cy = by / PER, cx = bx / PER;
    mc_cover cov = ensure_region(v, lod, cz, cy, cx);   // pulls the region if ABSENT
    if (cov == MC_ZERO || cov == MC_ABSENT) { memset(dst, 0, BLK * BLK * BLK); return cov == MC_ZERO ? 0 : -1; }
    // mc_cache_get_copy below MUTATES the shard (map insert / slot alloc) on a
    // miss. The fill pool (kicked by thaw) mutates the SAME shards from worker
    // threads under the no-lock single-owner contract, so a blocking get issued
    // between thaw and the next freeze would race it. Quiesce in-flight fills
    // first — blocking get is not on the lock-free render path, so the wait is
    // acceptable. (freeze() does the same fill_pool_wait before a frozen frame.)
    fill_pool_wait(v);
    mc_cache_get_copy(v->cache, lod, bz, by, bx, dst);
    return 1;
}

// ---------------------------------------------------------------------------
// sampling source
// ---------------------------------------------------------------------------
static const uint8_t *vol_block(const mc_sample_src *src,
                                int bz, int by, int bx, uint8_t *tmp) {
    mc_volume *v = src->ud;
    if (src->aux2) {   // blocking (CLI): keep the copy-into-tmp path
        int r = mc_volume_get_block(v, src->aux, bz, by, bx, tmp);
        return r == 1 ? tmp : NULL;
    }
    // interactive: return a STABLE arena/zero pointer (no copy). tmp unused.
    (void)tmp;
    return mc_volume_block_ptr(v, src->aux, bz, by, bx);
}

// CHEAP residency for the LOD fallback: sample-able NOW without a decode? Yes iff
// the region is air (ZERO -> samples 0 trivially) or the 16^3 block is already in
// the RAM cache. A PRESENT-but-not-cached block returns 0 here so the fallback
// samples a coarser resident level instead of decoding the fine block on the
// render thread (also records the miss so THAW fills it -> next frame sharpens).
static int vol_block_resident(const mc_sample_src *src, int bz, int by, int bx) {
    mc_volume *v = src->ud;
    int lod = src->aux;
    int cz = bz / PER, cy = by / PER, cx = bx / PER;
    mc_cover cov = vol_coverage(v, lod, cz, cy, cx);
    if (cov == MC_ZERO) return 1;                        // air: trivially sample-able
    if (cov != MC_PRESENT) {                             // absent: record + fall coarser
        mc_cache_miss_mark(v->cache, lod, cz * PER, cy * PER, cx * PER);
        return 0;
    }
    if (mc_cache_contains(v->cache, lod, bz, by, bx)) return 1;   // RAM hit
    mc_cache_miss_mark(v->cache, lod, bz, by, bx);       // present but uncached -> fill
    return 0;                                            // fall coarser this frame
}

mc_sample_src mc_volume_sample_src(mc_volume *v, int lod, int blocking) {
    mc_sample_src s = {0};
    s.ud = v; s.aux = lod; s.aux2 = blocking; s.block = vol_block;
    s.resident = vol_block_resident;
    s.owns_ptr = !blocking;   // interactive vol_block returns stable arena pointers
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
// Download ONE 256^3 region's source chunk(s) and push it to the decode queue —
// the interactive path. The shard index (64KB footer) is read once and cached
// (footer_get / mc_zarr_read_inner), so this is: cached footer lookup + a single
// ranged GET per source chunk (1 for c3d, up to 8 for a v2 sub^3 cube). NO
// whole-shard download. (cz,cy,cx) = the source inner-chunk coord of the region.
static void mc_volume_prefetch_region(mc_volume *v, int lod, int cz, int cy, int cx) {
    if (lod < 0 || lod >= v->nlods) return;
    mc_zarr *z = v->lv[lod].z;
    const int edge = mc_zarr_inner_edge(z);            // 256 (c3d) or 128 (v2)
    const int sub = CHUNK / edge;                      // source chunks per region axis
    const int rz = cz / sub, ry = cy / sub, rx = cx / sub;   // 256^3 region coords
    if (mc_archive_chunk_coverage(v->arc, lod, rz, ry, rx) != MC_ABSENT) return;  // already have it

    if (sub == 1) {                                    // c3d / v2-256: one chunk == region
        uint8_t *raw = NULL; size_t rlen = 0;
        int st = mc_zarr_read_inner(z, cz, cy, cx, &raw, &rlen);
        if (st < 0) { free(raw); return; }             // transient error -> leave ABSENT, retry
        if (st > 0 || !raw || !rlen) {                 // CONFIRMED air -> ZERO region
            free(raw);
            decode_item air = {lod, rz, ry, rx, 1, 0, {0},{0},{0}, {0},{0}};
            decode_push(v, &air);
            return;
        }
        atomic_fetch_add_explicit(&v->net_bytes, rlen, memory_order_relaxed);
        decode_item it = {lod, rz, ry, rx, 1, 1, {0},{0},{0}, {0},{0}};
        it.raw[0] = raw; it.rlen[0] = rlen;
        decode_push(v, &it);
        return;
    }

    // v2 sub^3 cube: fetch each present source chunk of the region's cube. A
    // transient read error on ANY sub-chunk aborts the whole region (leave it
    // ABSENT to retry) rather than recording a partial/air result.
    int sz0 = rz * sub, sy0 = ry * sub, sx0 = rx * sub;
    decode_item it = {lod, rz, ry, rx, sub, 0, {0},{0},{0}, {0},{0}};
    for (int dz = 0; dz < sub; ++dz)
    for (int dy = 0; dy < sub; ++dy)
    for (int dx = 0; dx < sub; ++dx) {
        uint8_t *raw = NULL; size_t rlen = 0;
        int st = mc_zarr_read_inner(z, sz0 + dz, sy0 + dy, sx0 + dx, &raw, &rlen);
        if (st < 0) {                                  // transient -> abort, retry region
            for (int k = 0; k < it.nsub; ++k) free(it.raw[k]);
            free(raw);
            return;
        }
        if (st > 0 || !raw || !rlen) { free(raw); continue; }   // this sub-chunk confirmed air
        atomic_fetch_add_explicit(&v->net_bytes, rlen, memory_order_relaxed);
        int k = it.nsub++;
        it.raw[k] = raw; it.rlen[k] = rlen;
        it.oz[k] = dz * edge; it.oy[k] = dy * edge; it.ox[k] = dx * edge;
    }
    decode_push(v, &it);                               // nsub==0 -> all sub-chunks air -> ZERO
}

// Download a shard's present chunks (one parallel s3_get_batch) and PUSH each
// region's raw payload(s) to the decode queue — NO decode on this (download)
// thread. Decoders drain the queue in parallel, so the network stays saturated.
// Backpressure in decode_push bounds RAM. (cz,cy,cx) = source inner-chunk coord.
// NOTE: bulk path (CLI prefetch). The interactive render path uses
// mc_volume_prefetch_region (one region) so navigation never pulls a whole shard.
void mc_volume_prefetch_shard(mc_volume *v, int lod, int cz, int cy, int cx) {
    if (lod < 0 || lod >= v->nlods) return;
    if (v->streaming) return;   // no shard/zarr layer; THAW read-through fills on miss
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
        if (v->local) {
            // Local: each req is a ranged read of the shard file. No network, no
            // batching win — just pread each range into an s3_response so the
            // decode-push path below is identical to the remote case.
            FILE *lf = fopen(shard_url, "rb");
            for (int i = 0; i < nq; ++i) {
                if (!lf) { resp[i].status = 404; continue; }
                if (fseek(lf, (long)reqs[i].offset, SEEK_SET) != 0) { resp[i].status = 500; continue; }
                uint8_t *b = malloc(reqs[i].length ? reqs[i].length : 1);
                if (!b) { resp[i].status = 500; continue; }
                size_t g = reqs[i].length ? fread(b, 1, reqs[i].length, lf) : 0;
                if (g != reqs[i].length) { free(b); resp[i].status = 500; continue; }
                resp[i].status = 200; resp[i].body = b; resp[i].body_len = g;
            }
            if (lf) fclose(lf);
        } else {
            s3_get_batch(v->s3, reqs, (size_t)nq, 32, resp);   // partial ok; check each
        }
        { int ok = 0; uint64_t bytes = 0;
          for (int i = 0; i < nq; ++i) if (s3_response_ok(&resp[i])) { ok++; bytes += resp[i].body_len; }
          // Count ACTUAL transferred bytes per batch (the real network/disk
          // throughput), not just the kept-chunk bytes accumulated in `got`
          // below — and do it per batch so a download-rate readout updates
          // promptly instead of only when the whole shard finishes.
          atomic_fetch_add_explicit(&v->net_bytes, bytes, memory_order_relaxed);
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
    (void)got;   // net_bytes is now counted per batch above (actual transfer)
    free(ranges);
}

void mc_volume_prefetch_level(mc_volume *v, int lod, int nthreads, volatile int *cancel) {
    (void)nthreads;   // TODO: thread team; serial walk for now.
    if (lod < 0 || lod >= v->nlods) return;
    if (v->streaming) return;   // no zarr/shard layer; THAW read-through fills on miss
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

size_t mc_volume_set_cache_bytes(mc_volume *v, size_t bytes) {
    return (v && v->cache) ? mc_cache_resize(v->cache, bytes) : 0;
}

// Tick-phase bracket for the render game-loop (see mc_cache_freeze/thaw). The
// volume owns the resident mc_cache; expose its two-phase clock so a client
// (volume-cartographer's global render tick) can freeze the cache for the
// duration of a frame's lock-free reads, then thaw between frames to let newly
// transcoded regions land and to bump the pin epoch.
void mc_volume_freeze(mc_volume *v) {
    if (!v || !v->cache) return;
    // Barrier: wait for the fill pool to finish thaw's PRESENT set before freezing,
    // so no fill write races a frozen read. Then do the render-gen bookkeeping for
    // what was filled (deferred from thaw -- we only know decoded count after join).
    if (v->nfillers > 0 && v->fill_n) {
        fill_pool_wait(v);
        if (atomic_load_explicit(&v->fill_decoded, memory_order_acquire)) {
            v->change_gen++;                           // pixels can differ now
            uint64_t last_mk = ~0ull;                  // mark filled regions (dedup'd)
            for (size_t i = 0; i < v->fill_n; ++i) {
                const mc_block_id *b = &v->fill_ids[i];
                int cz = b->bz / PER, cy = b->by / PER, cx = b->bx / PER;
                uint64_t rk = rkey(b->lod, cz, cy, cx);
                if (rk != last_mk) { vol_mark_region(v, b->lod, cz, cy, cx); last_mk = rk; }
            }
        }
        v->fill_n = 0;                                 // consumed (keep fill_ids buffer)
    }
    mc_cache_freeze(v->cache);
}

// Thaw = the single batch-apply step of the render game loop. Between a frame's
// freeze and the next, the lock-free frozen reads recorded their misses (blocks
// the render wanted but weren't resident). Here, while UNFROZEN, we:
//   1. thaw the cache (clears frozen, epoch++) so the fill is permitted,
//   2. drain the recorded miss set,
//   3. keep only blocks whose 256^3 source region is MC_PRESENT in the archive
//      (ABSENT = still downloading -> skip; ZERO = air -> already served),
//   4. fill those from the archive (decode from disk), TIME-BOUNDED so a big
//      miss set (a zoom across a LOD that misses the whole viewport) can't stall
//      the render tick more than ~MC_THAW_BUDGET_MS. Leftover blocks re-record
//      next frame and fill progressively; meanwhile the render shows coarser
//      resident LODs (mc_lod_sample fallback).
// The archive->cache fill runs on the persistent fill pool (filler_main),
// OFF the calling (GUI) thread: thaw hands it the PRESENT set and returns;
// freeze() waits for the pool before flipping frozen. Decode overlaps the
// frame render. Still race-free: writes are shard-partitioned single-owner and
// the only frozen-vs-fill barrier is freeze() joining the pool. No network IO.

// One fill-pool worker: owns shards { me, me+nfillers, ... }. Waits for thaw to
// bump fill_epoch, decodes its shards' blocks (lock-free, single-owner), then
// signals completion. The shard partition guarantees no two workers touch the
// same shard, so cache_fill_one needs no lock.
typedef struct { mc_volume *v; int me; } filler_arg;
static void *filler_main(void *ud) {
    filler_arg *a = ud; mc_volume *v = a->v; int me = a->me; free(a);
    mc_thread_setname("mc-fill");          // distinguish from the GUI thread in profilers
    uint64_t seen = 0;
    for (;;) {
        pthread_mutex_lock(&v->fill_mu);
        while (!v->fill_stop &&
               atomic_load_explicit(&v->fill_epoch, memory_order_relaxed) == seen)
            pthread_cond_wait(&v->fill_start, &v->fill_mu);
        if (v->fill_stop) { pthread_mutex_unlock(&v->fill_mu); return NULL; }
        seen = atomic_load_explicit(&v->fill_epoch, memory_order_relaxed);
        pthread_mutex_unlock(&v->fill_mu);

        uint64_t dec = 0;
        if (v->fill_n && v->fill_bucket && v->fill_link && v->fill_ids) {
            for (int s = me; s < NSHARD; s += v->nfillers)
                for (uint32_t i = v->fill_bucket[s]; i != UINT32_MAX; i = v->fill_link[i]) {
                    const mc_block_id *b = &v->fill_ids[i];
                    dec += (uint64_t)cache_fill_one(v->cache, b->lod, b->bz, b->by, b->bx);
                }
        }
        atomic_fetch_add_explicit(&v->fill_decoded, dec, memory_order_relaxed);
        // Last worker to finish this epoch signals freeze().
        pthread_mutex_lock(&v->fill_mu);
        if (atomic_fetch_add_explicit(&v->fill_done, 1, memory_order_acq_rel) + 1
            == (uint64_t)v->nfillers)
            pthread_cond_signal(&v->fill_fin);
        pthread_mutex_unlock(&v->fill_mu);
    }
}
// Kick the pool on the current fill set (caller filled fill_ids/bucket/link/fill_n).
static void fill_pool_start(mc_volume *v) {
    if (v->nfillers <= 0) return;
    pthread_mutex_lock(&v->fill_mu);
    atomic_store_explicit(&v->fill_done, 0, memory_order_relaxed);
    atomic_store_explicit(&v->fill_decoded, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&v->fill_epoch, 1, memory_order_release);
    pthread_cond_broadcast(&v->fill_start);
    pthread_mutex_unlock(&v->fill_mu);
}
// Wait for the pool to finish the in-flight fill (the freeze barrier).
static void fill_pool_wait(mc_volume *v) {
    if (v->nfillers <= 0) return;
    pthread_mutex_lock(&v->fill_mu);
    while (atomic_load_explicit(&v->fill_done, memory_order_acquire) < (uint64_t)v->nfillers)
        pthread_cond_wait(&v->fill_fin, &v->fill_mu);
    pthread_mutex_unlock(&v->fill_mu);
}
// Spawn the persistent fill pool once at open. Sized like the decode pool (the
// fill is the same DCT decode, just from the local archive). Call after v->cache
// exists. On failure nfillers stays 0 and thaw falls back to the inline fill.
static void fill_pool_init(mc_volume *v, int n) {
    if (n < 1) n = 1; if (n > 32) n = 32;
    pthread_mutex_init(&v->fill_mu, NULL);
    pthread_cond_init(&v->fill_start, NULL);
    pthread_cond_init(&v->fill_fin, NULL);
    atomic_store(&v->fill_epoch, 0); atomic_store(&v->fill_done, n);   // idle: "epoch 0 done"
    v->fill_stop = 0; v->nfillers = 0;
    for (int i = 0; i < n; ++i) {
        filler_arg *a = malloc(sizeof *a);
        if (!a) break;
        a->v = v; a->me = i;
        if (pthread_create(&v->fillers[i], NULL, filler_main, a) != 0) { free(a); break; }
        v->nfillers++;
    }
}
static void fill_pool_stop(mc_volume *v) {
    if (v->nfillers <= 0) return;
    pthread_mutex_lock(&v->fill_mu);
    v->fill_stop = 1;
    pthread_cond_broadcast(&v->fill_start);
    pthread_mutex_unlock(&v->fill_mu);
    for (int i = 0; i < v->nfillers; ++i) pthread_join(v->fillers[i], NULL);
    v->nfillers = 0;
    free(v->fill_bucket); v->fill_bucket = NULL;
    free(v->fill_link);   v->fill_link = NULL;
    free(v->fill_ids);    v->fill_ids = NULL;
}

#define MC_THAW_BUDGET_MS 5.0
#define MC_THAW_CHUNK 256          // blocks per mc_cache_update slice (time-checked)
void mc_volume_thaw(mc_volume *v) {
    if (!v || !v->cache) return;
    mc_cache_thaw(v->cache);                                  // clears frozen, epoch++

    static _Thread_local mc_block_id *miss = NULL;
    static _Thread_local size_t miss_cap = 0;
    if (!miss) { miss_cap = MISSQ_CAP; miss = malloc(miss_cap * sizeof *miss); }
    if (!miss) return;

    size_t n = mc_cache_misses_drain(v->cache, miss, miss_cap);

    // Collate (thaw = the collator), BEFORE any early-out. net_inflight is the
    // real pipeline depth (status bar); work_pending adds this frame's undrained
    // misses so the render gate keeps ticking until everything's resident.
    int dqn = v->dq_cap ? (v->dq_tail - v->dq_head + v->dq_cap) % v->dq_cap : 0;   // 0 in streaming
    v->net_inflight = (uint64_t)(v->rs_n + v->inflight_n + dqn);
    v->work_pending = v->net_inflight + (uint64_t)n;
    // Per-stage split. inflight covers claim -> append: on the wire + waiting in
    // the decode queue + actively inside decode->re-encode->append on a worker.
    int act = (int)atomic_load_explicit(&v->dec_active, memory_order_relaxed);
    v->snap_queued = (uint64_t)v->rs_n;
    v->snap_decq = (uint64_t)dqn;
    v->snap_encoding = (uint64_t)act;
    int dl = v->inflight_n - dqn - act;
    v->snap_downloading = dl > 0 ? (uint64_t)dl : 0;
    v->snap_staging_bytes = v->dq_bytes;

    if (!n) return;

    // Split the drained misses by local-archive coverage:
    //  PRESENT -> keep for the cache fill below (decode from local disk -- fast).
    //  ABSENT  -> issue ONE download request per region (zarr re-encode OR remote-
    //             .mca verbatim copy, on the download thread -- never here).
    // The miss set is per-block; many blocks map to one 256^3 region, so collapse
    // consecutive same-region absent blocks (misses drain roughly in scan order).
    size_t keep = 0;
    uint64_t last_rq = ~0ull;
    for (size_t i = 0; i < n; ++i) {
        const mc_block_id *b = &miss[i];
        int cz = b->bz / PER, cy = b->by / PER, cx = b->bx / PER;
        mc_cover cov = mc_archive_chunk_coverage(v->arc, b->lod, cz, cy, cx);
        if (cov == MC_PRESENT) { miss[keep++] = *b; }
        else if (cov == MC_ABSENT) {
            uint64_t rq = rkey(b->lod, cz, cy, cx);
            if (rq != last_rq) { req_push(v, b->lod, cz, cy, cx); last_rq = rq; }
        }
    }
    // Hand the PRESENT set to the persistent fill pool and RETURN. The pool decodes
    // (shard-partitioned, lock-free) while the frame renders; freeze() waits for it
    // and does the change_gen/region-mark bookkeeping once the fill is complete.
    // `miss` is thaw's thread-local buffer: stable until the next thaw, which can't
    // run until the next freeze has drained the pool, so the workers' borrow is safe.
    if (!keep || v->nfillers <= 0) {
        // No fill pool (or nothing to fill): do it inline (small/degenerate case).
        size_t filled = keep ? mc_cache_update(v->cache, miss, keep, 1) : 0;
        if (filled) {
            v->change_gen++;
            uint64_t last_mk = ~0ull;
            for (size_t i = 0; i < keep; ++i) {
                int cz = miss[i].bz / PER, cy = miss[i].by / PER, cx = miss[i].bx / PER;
                uint64_t rk = rkey(miss[i].lod, cz, cy, cx);
                if (rk != last_mk) { vol_mark_region(v, miss[i].lod, cz, cy, cx); last_mk = rk; }
            }
        }
        return;
    }
    // Shard-bucket the PRESENT blocks (same partition mc_cache_update uses) so each
    // pool worker owns a disjoint set of shards. Buffers grow as needed, reused tick
    // to tick. On alloc failure fall back to a synchronous inline fill.
    if (!v->fill_bucket) v->fill_bucket = malloc(NSHARD * sizeof *v->fill_bucket);
    if (!v->fill_link || v->fill_link_cap < keep) {
        uint32_t *nl = realloc(v->fill_link, keep * sizeof *v->fill_link);
        if (nl) { v->fill_link = nl; v->fill_link_cap = keep; }
    }
    if (!v->fill_bucket || !v->fill_link || v->fill_link_cap < keep) {
        size_t filled = mc_cache_update(v->cache, miss, keep, 1);
        if (filled) { v->change_gen++;
            uint64_t last_mk = ~0ull;
            for (size_t i = 0; i < keep; ++i) {
                int cz = miss[i].bz / PER, cy = miss[i].by / PER, cx = miss[i].bx / PER;
                uint64_t rk = rkey(miss[i].lod, cz, cy, cx);
                if (rk != last_mk) { vol_mark_region(v, miss[i].lod, cz, cy, cx); last_mk = rk; } } }
        return;
    }
    // Copy the kept blocks into the volume's OWN buffer: `miss` is thaw's thread-local
    // scratch, and a sibling cache thawed later in the same tick would overwrite it
    // while this pool still reads it -> cross-cache block corruption (junk reads).
    if (!v->fill_ids || v->fill_ids_cap < keep) {
        mc_block_id *ni = realloc(v->fill_ids, keep * sizeof *v->fill_ids);
        if (ni) { v->fill_ids = ni; v->fill_ids_cap = keep; }
    }
    if (!v->fill_ids || v->fill_ids_cap < keep) {      // alloc failed: inline fallback
        size_t filled = mc_cache_update(v->cache, miss, keep, 1);
        if (filled) { v->change_gen++;
            uint64_t last_mk = ~0ull;
            for (size_t i = 0; i < keep; ++i) {
                int cz = miss[i].bz / PER, cy = miss[i].by / PER, cx = miss[i].bx / PER;
                uint64_t rk = rkey(miss[i].lod, cz, cy, cx);
                if (rk != last_mk) { vol_mark_region(v, miss[i].lod, cz, cy, cx); last_mk = rk; } } }
        return;
    }
    memcpy(v->fill_ids, miss, keep * sizeof *v->fill_ids);
    for (int s = 0; s < NSHARD; ++s) v->fill_bucket[s] = UINT32_MAX;
    for (size_t i = 0; i < keep; ++i) {
        uint64_t key = bkey(v->fill_ids[i].lod, v->fill_ids[i].bz, v->fill_ids[i].by, v->fill_ids[i].bx);
        int s = (int)((khash(key) >> 56) & (NSHARD - 1));
        v->fill_link[i] = v->fill_bucket[s]; v->fill_bucket[s] = (uint32_t)i;
    }
    v->fill_n = keep;
    fill_pool_start(v);                                // workers decode; freeze() waits
}

size_t mc_volume_set_staging_bytes(mc_volume *v, size_t bytes) {
    if (!v) return 0;
    if (bytes < (64ull << 20)) bytes = 64ull << 20;    // floor 64MB
    pthread_mutex_lock(&v->mu);
    v->dq_byte_budget = bytes;
    pthread_cond_broadcast(&v->dq_nf);                 // a raise may unblock producers
    size_t installed = v->dq_byte_budget;
    pthread_mutex_unlock(&v->mu);
    return installed;
}

// Monotonic render generation: changes whenever a frozen render could produce
// different pixels -- a THAW cache fill (change_gen) or a coverage publish
// (archive gen, bumped on every chunk/air append). Equal gens across two frames
// with an unchanged camera => provably identical frame; the caller may skip it.
uint64_t mc_volume_render_gen(const mc_volume *v) {
    if (!v) return 0;
    uint64_t g = 1 + v->change_gen;
    if (v->arc) g += atomic_load_explicit(&v->arc->gen, memory_order_acquire);
    return g;
}

static inline uint32_t rgen_slot(int lod, int cz, int cy, int cx) {
    uint64_t key = mc_covkey(lod, cz, cy, cx);
    return (uint32_t)((key * 0x9E3779B97F4A7C15ull) >> 48) & (MC_RGEN_SLOTS - 1);
}
// Mark a region changed at the CURRENT render gen (call AFTER the publish/fill
// bumped it). Racing writers both store >= the prior value; last-writer-wins.
static void vol_mark_region(mc_volume *v, int lod, int cz, int cy, int cx) {
    if (!v->rgen) return;
    atomic_store_explicit(&v->rgen[rgen_slot(lod, cz, cy, cx)],
                          mc_volume_render_gen(v), memory_order_release);
}
// Last change gen of ONE region (0 = never changed). A viewer takes the max
// over its predicted working set; if that's <= the gen of its last frame and
// the camera is unchanged, the frame is provably identical for THAT viewport.
uint64_t mc_volume_region_gen(const mc_volume *v, int lod, int cz, int cy, int cx) {
    if (!v || !v->rgen) return 0;
    return atomic_load_explicit(&v->rgen[rgen_slot(lod, cz, cy, cx)],
                                memory_order_acquire);
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
    out->cache_used_blocks = cs.used;
    out->cache_cap_blocks = cs.slots;
    out->disk_bytes = v->arc ? mc_archive_data_len(v->arc) : 0;
    // Streaming pulls bytes two ways: reader callbacks (node tables / serial-path
    // blobs) counted by mc_s3, plus the batched GETs counted in v->net_bytes.
    out->net_bytes = atomic_load_explicit(&v->net_bytes, memory_order_relaxed)
                   + (v->s3mca ? mc_s3_net_bytes(v->s3mca) : 0);
    // Frozen snapshots collated at the last thaw. regions_inflight = real pipeline
    // depth (status bar); work_pending = + undrained misses (render gate).
    out->regions_inflight = v->net_inflight;
    out->work_pending = v->work_pending;
    out->regions_queued = v->snap_queued;
    out->regions_downloading = v->snap_downloading;
    out->regions_decode_queued = v->snap_decq;
    out->regions_encoding = v->snap_encoding;
    out->staging_bytes = v->snap_staging_bytes;
}
