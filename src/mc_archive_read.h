// ============================================================================
// mc_archive_read.h — resolve a chunk by coord through the sparse node tree, then
// locate a block within the dense chunk. Absent child = empty region (decodes to 0).
// ============================================================================
#ifndef MC_ARCHIVE_READ_H
#define MC_ARCHIVE_READ_H
#include "mc_archive.h"
#include <stdint.h>

// number of set bits in bm[0..idx) — the packed slot of child idx.
static inline int mc_rank(const uint8_t*bm,int idx){
    int r=0, full=idx>>3;
    for(int i=0;i<full;++i) r+=__builtin_popcount(bm[i]);
    int rem=idx&7; if(rem) r+=__builtin_popcount(bm[full] & ((1u<<rem)-1));
    return r;
}
// resolve chunk (cz,cy,cx) -> chunk-blob offset (0 = empty/absent). Walks the 2 sparse
// levels (nibbles 2 then 1) to the shard, then direct-indexes the dense shard by nibble 0.
static uint64_t mc_resolve_chunk(const uint8_t*arc, uint64_t root_off,int cz,int cy,int cx){
    if(!root_off) return 0;
    uint64_t node = root_off;
    for(int level=MC_SPARSE_LEVELS-1; level>=0; --level){
        const uint8_t*bm = arc + node;
        int nib=level+1;
        int dz=mc_nib(cz,nib), dy=mc_nib(cy,nib), dx=mc_nib(cx,nib);
        int idx=(dz*16+dy)*16+dx;
        if(!mc_bit_get(bm,idx)) return 0;
        int slot=mc_rank(bm,idx);
        uint64_t childoff; memcpy(&childoff, arc+node+MC_BITMAP_BYTES + (size_t)slot*8, 8);
        node = childoff;
    }
    int si=((mc_nib(cz,0))*16 + mc_nib(cy,0))*16 + mc_nib(cx,0);
    uint64_t chunk; memcpy(&chunk, arc+node + (size_t)si*8, 8);
    return chunk;
}
// chunk blob: [u32 masklen][mask][512B block-bitmap][present u32 lens][payloads].
static const uint8_t* mc_chunk_mask(const uint8_t*arc, uint64_t chunk_off, uint32_t *mlen){
    uint32_t ml; memcpy(&ml, arc+chunk_off, 4); *mlen=ml;
    return ml ? arc+chunk_off+4 : 0;
}
// block (bz,by,bx) present? -> 1 + its payload (abs_off, len). 0 = ZERO block. Offsets
// are implicit (cumulative len of present blocks before it); ZERO blocks cost 1 bitmap bit.
static int mc_block_range(const uint8_t*arc, uint64_t chunk_off, int bz,int by,int bx,
                          uint64_t *abs_off, uint32_t *len){
    uint32_t ml; memcpy(&ml, arc+chunk_off, 4);
    uint64_t bm_off = chunk_off + 4 + ml;
    const uint8_t*bm = arc + bm_off;
    int bi=(bz*16+by)*16+bx;
    if(!mc_bit_get(bm,bi)) return 0;
    int npresent=0; for(int i=0;i<MC_BITMAP_BYTES;++i) npresent+=__builtin_popcount(bm[i]);
    const uint8_t*lens = arc + bm_off + MC_BITMAP_BYTES;
    uint64_t pay_base = bm_off + MC_BITMAP_BYTES + (uint64_t)npresent*4;
    int slot = mc_rank(bm,bi);
    uint64_t cum=0; for(int s=0;s<slot;++s){ uint32_t l; memcpy(&l,lens+(size_t)s*4,4); cum+=l; }
    uint32_t mylen; memcpy(&mylen, lens+(size_t)slot*4, 4);
    *abs_off = pay_base + cum; *len = mylen;
    return 1;
}
#endif
