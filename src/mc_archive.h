// mc_archive.h — on-disk archive FORMAT constants + read helpers.
// (Split out of matter_compressor.c; commit B rewrites this to static-sparse.)
#ifndef MC_ARCHIVE_H_WRAP
#define MC_ARCHIVE_H_WRAP
#include "matter_compressor.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

// Writer/reader handle state (shared with mc_cache, which reads a->gen). The
// on-disk dynamic-index archive; commit B replaces this with the static format.
struct mc_archive {
    int fd;
    uint8_t *base;             // fixed mmap base (never moves)
    _Atomic uint64_t cursor;   // append EOF (bytes used)
    _Atomic uint64_t file_len; // current ftruncate'd file size
    int dim;
    float quality;
    uint32_t block_codec;      // MC_BLOCKCODEC_*
    uint64_t reserve;          // mmap reservation size (dims-derived, <= MC_RESERVE)
    pthread_mutex_t grow_mu;   // serializes ftruncate only; decode is lock-free
    _Atomic uint64_t *cov;     // coverage memo slots (region key | flags), 0 = empty
    _Atomic uint64_t gen;      // bumped on every publish
};

void priors_load(const uint8_t *base);

// ============================================================================
// mc_archive.h — matter-compressor on-disk archive format constants.
//
// An APPENDABLE, crash-safe, persistent archive: a node tree of dense 256^3 chunks,
// each a 16^3 grid of DCT blocks coded by mc_codec. 8 independent LODs (LOD0 = full
// res), each its own tree. LODs are independently fetchable AND independently
// decodable — no cross-LOD dependency (a hard design constraint).
//
// HIERARCHY (per LOD), all grids are 16x16x16 (4 bits/axis):
//   chunk coord (voxel>>8, up to 12 bits/axis) = [ region 4b | subregion 4b | shard 4b ]
//   CHUNK (256^3 voxels = 16^3 DCT blocks) is the dense leaf. Above it: a dense 16^3
//   SHARD table of chunk offsets, then 2 NODE levels. Covers 2^12 chunks/axis =
//   2^20 voxels/axis.
//
// NODE / SHARD = a DENSE flat MC_GRID3 (=4096) array of u64 child offsets, slot value
// 0 = absent child. Directly indexed by the chunk-coord nibble; no bitmap, no
// rank-packing -> every node + shard is UPDATABLE IN PLACE. This is what makes the
// archive appendable + crash-safe: chunk payloads append at EOF; the index path
// (root -> node -> node -> shard) is created/updated in place with the chunk offset
// published LAST as the commit word. The file is a fully valid, decodable archive
// after every appended chunk and PERSISTS across process runs (reopen + append).
//
// CHUNK blob (v2) = [512B block-bitmap][present-block u32 lens][block payloads];
// one range-GET fetches a whole chunk. Blocks are self-contained (per-block air
// mask lives in each block payload), so one block decode needs only the bitmap,
// the lens, and its own payload.
//
// LAYOUT: [256B header][metadata region up to 128KB][archive data from 128KB:
//          node/shard tables + chunk blobs, both appended at EOF].
// ============================================================================
#ifndef MC_ARCHIVE_H
#define MC_ARCHIVE_H
#include <stdint.h>
#include <string.h>

// ---- node / shard geometry (dense 16^3 child map of u64 offsets) ----
#define MC_GRID         16
#define MC_GRID3        4096          // 16^3 slots
#define MC_BITMAP_BYTES 512           // 4096 bits — chunk-blob block bitmap (NOT node index)
#define MC_NODE_BYTES   (MC_GRID3*8)  // dense node table: 4096 u64 child offsets (32KB)
#define MC_SHARD_BYTES  (MC_GRID3*8)  // dense shard table: 4096 u64 chunk offsets (32KB)
#define MC_TREE_LEVELS  3             // root node, 1 inner node, 1 shard (indexed by nibbles 2,1,0)

// ---- header (256B) ----
#define MC_MAGIC   0x0043434Du      // "MCC\0"
#define MCH_MAGIC   0               // u32 magic
#define MCH_VER     4               // u32 format-version field
#define MCH_NX      12              // u32 volume dims (x fastest)
#define MCH_NY      16
#define MCH_NZ      20
#define MCH_ROOTOFF 24              // u64[8] per-LOD root-node file offset (0 = empty LOD)
#define MCH_TOTLEN  88              // u64 total archive length (= append cursor / EOF)
#define MCH_METAOFF 96              // u64 metadata region start (= MC_HDR = 256)
#define MCH_METACAP 104             // u64 metadata region capacity (= MC_META_END - MC_HDR)
#define MCH_METALEN 112             // u64 metadata bytes actually written
#define MCH_QUALITY 120             // f32 quality the archive was built at (writer stamps it)
#define MCH_BLOCKCODEC 124          // u32 block entropy codec: 0 = CABAC (the only codec;
                                    // c3g was removed). Old archives have 0 here = CABAC.
#define MCH_PRIOROFF 128            // u64 offset of an optional per-volume prior blob (0 = none)
// MC_BLOCKCODEC_CABAC comes from matter_compressor.h.
#define MC_PRIORS_MAGIC 0x53524950u // "PRIS"
#define MC_PRIORS_BYTES (8 + 2*8*32*2)  // magic+ver, RC_PLO + RC_PHI as u16[8][32]
#define MC_HDR      256u            // header size; metadata region begins here
#define MC_META_END (128u*1024u)    // archive data begins at this offset (128KB)
#define MC_META_CAP (MC_META_END - MC_HDR)
#define MC_VERSION  7u              // format version (v7: per-chunk material-fraction map)

#define MC_CHUNK_ALIGN 256          // volume dim must be a multiple of this

// chunk-blob block-bitmap + chunk-coord nibble helpers
static inline int  mc_bit_get(const uint8_t*bm,int i){ return (bm[i>>3]>>(i&7))&1; }
static inline void mc_bit_set(uint8_t*bm,int i){ bm[i>>3]|=(uint8_t)(1u<<(i&7)); }
static inline int  mc_nib(int chunkcoord,int level){ return (chunkcoord>>(4*level))&15; }
#endif

// ============================================================================
// mc_archive_read.h — resolve a chunk by coord through the DENSE node tree, then
// locate a block within the dense chunk. Absent child = empty region (decodes to 0).
//
// The index is 3 dense levels (root node, inner node, shard), each a flat MC_GRID3
// array of u64 offsets indexed by the chunk-coord nibble (2, then 1, then 0). A slot
// value of 0 means absent. No bitmap / no rank-packing at the index level, so the
// resolve is three direct array reads.
// ============================================================================
#ifndef MC_ARCHIVE_READ_H
#define MC_ARCHIVE_READ_H
#include <stdint.h>

// number of set bits in bm[0..idx) — used for the chunk-blob block bitmap only.
static inline int mc_rank(const uint8_t*bm,int idx){
    int r=0, full=idx>>3;
    for(int i=0;i<full;++i) r+=__builtin_popcount(bm[i]);
    int rem=idx&7; if(rem) r+=__builtin_popcount(bm[full] & ((1u<<rem)-1));
    return r;
}

// resolve chunk (cz,cy,cx) -> chunk-blob offset (0 = empty/absent). Walks the dense
// node tree: nibble 2 (root node) -> nibble 1 (inner node) -> nibble 0 (shard) ->
// chunk-blob offset. Each level is a direct u64 array index.
// Walk the dense node tree to a chunk-blob offset. `end` is the buffer length:
// a corrupt/untrusted header may point a node offset past EOF, so each level's
// MC_GRID3-u64 slot array must fit within [0,end). Out-of-range -> absent (0),
// never dereferenced. All callers pass a real bound (readers r->len, the archive
// its file_len), so there is no unbounded path.
static inline uint64_t mc_resolve_chunk(const uint8_t*arc, uint64_t root_off,
                                 int cz,int cy,int cx, uint64_t end){
    uint64_t node = root_off;
    for(int nib=MC_TREE_LEVELS-1; nib>=0; --nib){
        if(!node) return 0;
        int dz=mc_nib(cz,nib), dy=mc_nib(cy,nib), dx=mc_nib(cx,nib);
        int idx=(dz*16+dy)*16+dx;
        if(node > end || (size_t)idx*8 + 8 > end - node) return 0;   // OOB slot -> absent
        uint64_t childoff; memcpy(&childoff, arc+node + (size_t)idx*8, 8);
        node = childoff;
    }
    return node;   // chunk-blob offset (0 if absent)
}

// chunk blob (v7): [f32 q][u64 xxh64][u16 fmaplen][fmap][512B block-bitmap]
// [present u16 lens][payloads]. q = the chunk's own quality; the hash covers
// fmap+bitmap+lens+payloads; fmap = rc-coded per-block material fractions
// (4096 nibbles, 0..15 ~= 0..100%) for rejection-free ML sampling.
#define MC_BLOB_HDR 14
// Slot sentinel: a chunk that was VISITED and decodes to all-zero (air). Real
// blob offsets are always >= MC_HDR (blobs append after the header), so 1 is a
// safe sentinel. Distinguishes "air, fetched" from "never fetched" (slot 0).
#define MC_SLOT_ZERO 1ull
static inline float mc_chunk_q(const uint8_t*arc, uint64_t chunk_off){
    float q; memcpy(&q,arc+chunk_off,4); return q;
}
static inline uint64_t mc_chunk_stored_hash(const uint8_t*arc, uint64_t chunk_off){
    uint64_t h; memcpy(&h,arc+chunk_off+4,8); return h;
}
static inline uint16_t mc_chunk_fmaplen(const uint8_t*arc, uint64_t chunk_off){
    uint16_t l; memcpy(&l,arc+chunk_off+12,2); return l;
}
// block (bz,by,bx) present? -> 1 + its payload (abs_off, len). 0 = ZERO block. Offsets
// are implicit (cumulative len of present blocks before it); ZERO blocks cost 1 bitmap bit.
// Per-block payload (abs_off,len) within a chunk blob. The on-disk layout stores
// only present-block lengths (offsets implicit = prefix sum). Computing that prefix
// sum + rank per call is O(block-index): summed over a chunk's 4096 blocks it is
// O(n^2) (~16M ops/chunk) -- the render-path cost when a frame samples many blocks
// of one chunk. We cache, thread-locally, a full per-chunk index built in ONE
// O(4096) pass: bi -> abs_off (0 = absent block). Render samples are spatially
// coherent (consecutive blocks share a chunk), so the table is built once per
// chunk and every block lookup is O(1).
typedef struct { const uint8_t*arc; uint64_t chunk_off, tag; uint64_t off[MC_GRID3]; uint16_t len[MC_GRID3]; } mc_chunk_idx;
static inline const mc_chunk_idx *mc_chunk_index(const uint8_t*arc, uint64_t chunk_off){
    static _Thread_local mc_chunk_idx idx = { .arc=NULL, .chunk_off=~0ull, .tag=0 };
    // Key on (arc base, chunk_off, content hash). chunk_off alone is unsafe: it is
    // reused across archives (same tree position -> same EOF offset) and after a
    // re-append, so the stored xxh64 disambiguates content. Otherwise a stale index
    // from a different chunk is served (caught by mc_v6 par-vs-serial).
    uint64_t tag = mc_chunk_stored_hash(arc, chunk_off);
    if(idx.arc==arc && idx.chunk_off==chunk_off && idx.tag==tag) return &idx;   // hot
    uint64_t bm_off = chunk_off + MC_BLOB_HDR + mc_chunk_fmaplen(arc,chunk_off);
    const uint8_t*bm = arc + bm_off;
    int npresent=0; for(int i=0;i<MC_BITMAP_BYTES;++i) npresent+=__builtin_popcount(bm[i]);
    const uint8_t*lens = arc + bm_off + MC_BITMAP_BYTES;
    uint64_t pay = bm_off + MC_BITMAP_BYTES + (uint64_t)npresent*2;   // first payload
    int slot=0;
    for(int bi=0; bi<MC_GRID3; ++bi){
        if(mc_bit_get(bm,bi)){
            uint16_t l; memcpy(&l, lens+(size_t)slot*2, 2);
            idx.off[bi]=pay; idx.len[bi]=l; pay+=l; ++slot;
        } else { idx.off[bi]=0; idx.len[bi]=0; }
    }
    idx.arc=arc; idx.chunk_off=chunk_off; idx.tag=tag;
    return &idx;
}
static inline int mc_block_range(const uint8_t*arc, uint64_t chunk_off, int bz,int by,int bx,
                          uint64_t *abs_off, uint32_t *len){
    const mc_chunk_idx *ix = mc_chunk_index(arc, chunk_off);
    int bi=(bz*16+by)*16+bx;
    if(!ix->off[bi]) return 0;                        // absent (ZERO) block
    *abs_off = ix->off[bi]; *len = ix->len[bi];
    return 1;
}

// Validate that the chunk blob at `chunk_off` is structurally in-bounds for an
// `end`-byte buffer: header+fmap+bitmap+lens+payloads lie within [0,end).
// Returns the total blob length (>0) if so, else 0. Untrusted-path blob-field
// readers (mc_decode_block, mc_verify_archive) gate on this.
static inline uint64_t mc_blob_struct_ok(const uint8_t*arc, uint64_t chunk_off, uint64_t end){
    if(chunk_off > end || end - chunk_off < (uint64_t)MC_BLOB_HDR) return 0;
    uint16_t fml; memcpy(&fml,arc+chunk_off+12,2);
    uint64_t bm_off = chunk_off + MC_BLOB_HDR + fml;
    if(bm_off > end || end - bm_off < (uint64_t)MC_BITMAP_BYTES) return 0;
    const uint8_t*bm = arc + bm_off;
    int np=0; for(int i=0;i<MC_BITMAP_BYTES;++i) np+=__builtin_popcount(bm[i]);
    uint64_t lens_off = bm_off + MC_BITMAP_BYTES;
    if(end - lens_off < (uint64_t)np*2) return 0;
    uint64_t paybytes=0; const uint8_t*lens = arc + lens_off;
    for(int s=0;s<np;++s){ uint16_t l; memcpy(&l,lens+(size_t)s*2,2); paybytes+=l; }
    uint64_t total = lens_off + (uint64_t)np*2 + paybytes - chunk_off;
    if(chunk_off + total > end) return 0;
    return total;
}

// total byte length of a chunk blob — used to copy a whole compressed chunk verbatim
// (mc_append_chunk_compressed) and for chunk-blob range queries. chunk_off must be valid.
static inline uint64_t mc_chunk_blob_len(const uint8_t*arc, uint64_t chunk_off){
    uint64_t bm_off = chunk_off + MC_BLOB_HDR + mc_chunk_fmaplen(arc,chunk_off);
    const uint8_t*bm = arc + bm_off;
    int npresent=0; for(int i=0;i<MC_BITMAP_BYTES;++i) npresent+=__builtin_popcount(bm[i]);
    const uint8_t*lens = arc + bm_off + MC_BITMAP_BYTES;
    uint64_t paybytes=0; for(int s=0;s<npresent;++s){ uint16_t l; memcpy(&l,lens+(size_t)s*2,2); paybytes+=l; }
    return bm_off + MC_BITMAP_BYTES + (uint64_t)npresent*2 + paybytes - chunk_off;
}
#endif
#endif // MC_ARCHIVE_H_WRAP
