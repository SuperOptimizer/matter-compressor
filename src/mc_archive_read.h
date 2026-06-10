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
#include "mc_archive.h"
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
static uint64_t mc_resolve_chunk(const uint8_t*arc, uint64_t root_off,int cz,int cy,int cx){
    uint64_t node = root_off;
    for(int nib=MC_TREE_LEVELS-1; nib>=0; --nib){
        if(!node) return 0;
        int dz=mc_nib(cz,nib), dy=mc_nib(cy,nib), dx=mc_nib(cx,nib);
        int idx=(dz*16+dy)*16+dx;
        uint64_t childoff; memcpy(&childoff, arc+node + (size_t)idx*8, 8);
        node = childoff;
    }
    return node;   // chunk-blob offset (0 if absent)
}

// chunk blob (v6): [f32 q][u64 xxh64][512B block-bitmap][present u16 lens]
// [payloads]. q is the chunk's own quality (mixed-q archives / rate control);
// the hash covers everything after the hash field (q is hashed too? no — the
// hash field follows q, and covers bitmap..payload plus the q word is included
// via mc_chunk_hash below). Blocks are self-contained.
#define MC_BLOB_HDR 12
static inline float mc_chunk_q(const uint8_t*arc, uint64_t chunk_off){
    float q; memcpy(&q,arc+chunk_off,4); return q;
}
static inline uint64_t mc_chunk_stored_hash(const uint8_t*arc, uint64_t chunk_off){
    uint64_t h; memcpy(&h,arc+chunk_off+4,8); return h;
}
// block (bz,by,bx) present? -> 1 + its payload (abs_off, len). 0 = ZERO block. Offsets
// are implicit (cumulative len of present blocks before it); ZERO blocks cost 1 bitmap bit.
static int mc_block_range(const uint8_t*arc, uint64_t chunk_off, int bz,int by,int bx,
                          uint64_t *abs_off, uint32_t *len){
    uint64_t bm_off = chunk_off + MC_BLOB_HDR;
    const uint8_t*bm = arc + bm_off;
    int bi=(bz*16+by)*16+bx;
    if(!mc_bit_get(bm,bi)) return 0;
    int npresent=0; for(int i=0;i<MC_BITMAP_BYTES;++i) npresent+=__builtin_popcount(bm[i]);
    const uint8_t*lens = arc + bm_off + MC_BITMAP_BYTES;
    uint64_t pay_base = bm_off + MC_BITMAP_BYTES + (uint64_t)npresent*2;
    int slot = mc_rank(bm,bi);
    uint64_t cum=0; for(int s=0;s<slot;++s){ uint16_t l; memcpy(&l,lens+(size_t)s*2,2); cum+=l; }
    uint16_t mylen; memcpy(&mylen, lens+(size_t)slot*2, 2);
    *abs_off = pay_base + cum; *len = mylen;
    return 1;
}

// total byte length of a chunk blob — used to copy a whole compressed chunk verbatim
// (mc_append_chunk_compressed) and for chunk-blob range queries. chunk_off must be valid.
static uint64_t mc_chunk_blob_len(const uint8_t*arc, uint64_t chunk_off){
    uint64_t bm_off = chunk_off + MC_BLOB_HDR;
    const uint8_t*bm = arc + bm_off;
    int npresent=0; for(int i=0;i<MC_BITMAP_BYTES;++i) npresent+=__builtin_popcount(bm[i]);
    const uint8_t*lens = arc + bm_off + MC_BITMAP_BYTES;
    uint64_t paybytes=0; for(int s=0;s<npresent;++s){ uint16_t l; memcpy(&l,lens+(size_t)s*2,2); paybytes+=l; }
    return bm_off + MC_BITMAP_BYTES + (uint64_t)npresent*2 + paybytes - chunk_off;
}
#endif
