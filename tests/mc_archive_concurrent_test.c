// Concurrent archive append: N threads each append a disjoint set of chunks whose
// index paths SHARE parent tree nodes (same root, overlapping inner nodes). This
// stresses w_ensure_shard_slot's node creation. Before the CAS-publish fix, two
// threads first-touching the same parent could each create a child node and clobber
// each other's publish, orphaning a subtree -> some appended chunks read back ABSENT.
// After the fix, every appended chunk must be reachable. No false negatives: the
// air/zero distinction is avoided by appending a non-air pattern.
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

// Coords are spaced so the index path varies across ALL THREE nibble levels
// (root/inner/shard), maximizing concurrent NODE CREATION (the actual race site).
// We append via mc_archive_append_chunk_compressed (a memcpy of a 1-byte blob, NO
// DCT encode) so threads reach w_ensure_shard_slot near-simultaneously -- the encode
// cost in the raw path otherwise staggers threads past the race window.
#define NTHREAD 16
#define NCO     8
static const int COORDS[NCO] = {0, 16, 48, 80, 272, 288, 528, 800};   // mixed nibble 0/1/2
#define DIM     ((800 + 1) * 256)   // big enough to hold the largest coord
#define NCHUNK  NCO

typedef struct { mc_archive *a; int tid; int fails; } job_t;

static pthread_barrier_t g_start;
// A pre-built valid 1-block compressed blob (filled in main); appending it is a pure
// memcpy + index publish, so threads race on node creation, not on DCT encode.
static uint8_t *g_blob; static size_t g_blob_len;

// Each thread owns chunks where (linear index % NTHREAD == tid): disjoint chunks.
// A start barrier releases all threads at once so node-creation first-touches race.
static void *worker(void *ud) {
    job_t *j = ud;
    pthread_barrier_wait(&g_start);
    int idx = 0;
    for (int iz = 0; iz < NCHUNK; ++iz)
        for (int iy = 0; iy < NCHUNK; ++iy)
            for (int ix = 0; ix < NCHUNK; ++ix, ++idx) {
                if (idx % NTHREAD != j->tid) continue;
                if (mc_archive_append_chunk_compressed(j->a, 0, COORDS[iz], COORDS[iy], COORDS[ix],
                                                       g_blob, g_blob_len) != 0)
                    j->fails++;
            }
    return NULL;
}

int main(void) {
    const char *path = "/tmp/mc_archive_concurrent.mca";
    remove(path);
    // Dummy blob: coverage checks the published slot offset (present vs 0/ZERO), it
    // does not parse blob contents, so any non-empty buffer exercises the index path.
    static uint8_t blob[64]; memset(blob, 0xAB, sizeof blob);
    g_blob = blob; g_blob_len = sizeof blob;

    const int TRIALS = 400;   // repeat on fresh archives; the node-creation race is a
                              // narrow window, so many trials are needed to surface it.
    int total = NCHUNK * NCHUNK * NCHUNK, total_missing = 0, bad_trials = 0;
    for (int trial = 0; trial < TRIALS; ++trial) {
        remove(path);
        mc_archive *a = mc_archive_open_dims(path, DIM, DIM, DIM, 6.0f);
        if (!a) { fprintf(stderr, "open failed\n"); return 1; }

        pthread_barrier_init(&g_start, NULL, NTHREAD);
        pthread_t th[NTHREAD];
        job_t jobs[NTHREAD];
        for (int t = 0; t < NTHREAD; ++t) {
            jobs[t] = (job_t){.a = a, .tid = t, .fails = 0};
            if (pthread_create(&th[t], NULL, worker, &jobs[t]) != 0) {
                fprintf(stderr, "thread create failed\n"); return 1;
            }
        }
        int append_fails = 0;
        for (int t = 0; t < NTHREAD; ++t) { pthread_join(th[t], NULL); append_fails += jobs[t].fails; }
        pthread_barrier_destroy(&g_start);
        if (append_fails) { fprintf(stderr, "%d appends failed\n", append_fails); return 1; }

        // Every chunk must be reachable by WALKING THE INDEX TREE (mc_archive_chunk_offset,
        // not the coverage memo -- the memo is populated independently of the tree and
        // would mask a lost node). A node clobbered by a publish race orphans a subtree,
        // so its chunks resolve to offset 0 here.
        int missing = 0;
        for (int iz = 0; iz < NCHUNK; ++iz)
            for (int iy = 0; iy < NCHUNK; ++iy)
                for (int ix = 0; ix < NCHUNK; ++ix)
                    if (mc_archive_chunk_offset(a, 0, COORDS[iz], COORDS[iy], COORDS[ix]) == 0)
                        missing++;
        if (missing) { bad_trials++; total_missing += missing;
            if (bad_trials <= 4) fprintf(stderr, "trial %d: %d/%d chunks orphaned\n", trial, missing, total); }
        mc_archive_close(a);
    }
    remove(path);
    printf("trials=%d threads=%d chunks/trial=%d  bad_trials=%d total_orphaned=%d\n",
           TRIALS, NTHREAD, total, bad_trials, total_missing);
    if (bad_trials) { fprintf(stderr, "INDEX CORRUPTION under concurrent append\n"); return 1; }
    printf("mc_archive_concurrent: OK\n");
    return 0;
}
