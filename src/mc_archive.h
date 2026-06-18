// mc_archive.h — STATIC-SLOT + SPARSE-FILE on-disk archive (.mca v8).
//
// A chunk's byte position is a PURE FUNCTION of (lod, cz, cy, cx): there is no
// dynamic index, no append cursor, no node tree, no locks. The file is
// ftruncate'd ONCE to its full logical size (which may be terabytes) and mmap'd;
// untouched slots are sparse holes that cost zero disk and read back as zero.
//
//   file: [header page][user-metadata][occupancy (2-bit/chunk)][chunk slots...]
//   slot(lod,cz,cy,cx) at  chunks_off + slot_index*slot_stride
//   slot_index = lod_base[lod] (coarse-first) + (cz*ey + cy)*ex + cx
//
// within a chunk slot (page-aligned), the mcpp layout:
//   [xxh3:u64][quality:f32][flags:u32][offset_table:4097*u32][dense payloads]
//   offset_table[k] = byte offset of block k's payload from the chunk base, or
//                     MC_OFF_ABSENT (0xFFFFFFFF) for an ALL_ZERO block; entry
//                     4096 is the end-of-payload terminator (= chunk length).
//   payloads start page-aligned (MC_CHUNK_PAYLOAD_OFF) so a single block faults
//   only its own pages.
#ifndef MC_ARCHIVE_H_WRAP
#define MC_ARCHIVE_H_WRAP

#include "matter_compressor.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

// ---- format constants ------------------------------------------------------
#define MCA_MAGIC      0x3050434Du     // "MCP0"
#define MCA_VERSION    8u
#define MCA_MAXLODS    8

// header field byte offsets (within the first page)
#define MCAH_MAGIC      0    // u32
#define MCAH_VERSION    4    // u32
#define MCAH_DTYPE      8    // u8  (0 = u8; only u8 today)
#define MCAH_NUMLODS    9    // u8
#define MCAH_NX        12    // u32 dims (x fastest)
#define MCAH_NY        16    // u32
#define MCAH_NZ        20    // u32
#define MCAH_QUALITY   24    // f32 default quality
#define MCAH_BLOCKCODEC 28   // u32
#define MCAH_METAOFF   32    // u64
#define MCAH_METALEN   40    // u64
#define MCAH_METACAP   48    // u64
#define MCAH_OCCOFF    56    // u64
#define MCAH_CHUNKSOFF 64    // u64
#define MCAH_SLOTSTRIDE 72   // u64
#define MCAH_TOTCHUNKS 80    // u64

#define MCA_HDR_REGION  4096u           // header lives in the first page
#define MCA_META_REGION (128u*1024u)    // user-metadata carve-out

// within-chunk layout
#define MC_OFFTAB_LEN   4097u                       // 4096 blocks + terminator
#define MC_OFF_ABSENT   0xFFFFFFFFu                  // ALL_ZERO block sentinel
#define MC_CHUNK_HDR    (16u + MC_OFFTAB_LEN*4u)     // 8+4+4 + table = 16404
#define MC_CHUNK_PAYLOAD_OFF 20480u                  // align_up(16404, 4096)
#define MC_CHUNK_RAW    (256u*256u*256u)             // raw 256^3 u8 chunk
// fixed per-slot stride: a chunk can never exceed its raw size (lossy codec);
// payloads start page-aligned. The unused tail is a sparse hole (free).
#define MC_SLOT_STRIDE  (((MC_CHUNK_PAYLOAD_OFF + MC_CHUNK_RAW + 4095u)/4096u)*4096u)

// 3-state occupancy (2 bits/chunk); sparse-zero default = DONT_KNOW (safe).
enum { MC_OCC_DONT_KNOW = 0, MC_OCC_ALL_ZERO = 1, MC_OCC_REAL = 2 };

// ---- geometry: pure arithmetic from dims + num_lods ------------------------
static inline uint64_t mca_align(uint64_t v, uint64_t a){ return (v + a - 1) / a * a; }
static inline uint32_t mca_dim_at(uint32_t d, int lod){ uint32_t r = d >> lod; return r ? r : 1u; }
static inline uint64_t mca_chunks_axis(uint32_t d, int lod){ return (mca_dim_at(d,lod) + 255u) / 256u; }
static inline uint64_t mca_chunks_at(const uint32_t dims[3], int lod){
    return mca_chunks_axis(dims[0],lod) * mca_chunks_axis(dims[1],lod) * mca_chunks_axis(dims[2],lod);
}
static inline int mca_num_lods(uint32_t nz, uint32_t ny, uint32_t nx){
    int n = 1; for(int L=1; L<MCA_MAXLODS; ++L) if((nz>>L)||(ny>>L)||(nx>>L)) n = L+1; return n;
}
// coarse-first layout: the coarsest LOD (num_lods-1) occupies the first region.
static inline uint64_t mca_lod_base(const uint32_t dims[3], int num_lods, int lod){
    uint64_t b = 0; for(int l = num_lods-1; l > lod; --l) b += mca_chunks_at(dims,l); return b;
}
static inline uint64_t mca_slot_index(const uint32_t dims[3], int num_lods, int lod,
                                      uint64_t cz, uint64_t cy, uint64_t cx){
    uint64_t ey = mca_chunks_axis(dims[1],lod), ex = mca_chunks_axis(dims[2],lod);
    return mca_lod_base(dims,num_lods,lod) + (cz*ey + cy)*ex + cx;
}
static inline uint64_t mca_total_chunks(const uint32_t dims[3], int num_lods){
    uint64_t t = 0; for(int l=0; l<num_lods; ++l) t += mca_chunks_at(dims,l); return t;
}

// ---- per-chunk readers (operate on the chunk base pointer) -----------------
static inline uint64_t mc_chunk_xxh3(const uint8_t *c){ uint64_t h; memcpy(&h,c,8); return h; }
static inline float    mc_chunk_q(const uint8_t *c){ float q; memcpy(&q,c+8,4); return q; }
static inline uint32_t mc_chunk_offtab(const uint8_t *c, unsigned k){
    uint32_t v; memcpy(&v, c + 16 + (size_t)k*4, 4); return v;
}
static inline int mc_chunk_block_present(const uint8_t *c, int bi){
    return mc_chunk_offtab(c, (unsigned)bi) != MC_OFF_ABSENT;
}
static inline uint64_t mc_chunk_len(const uint8_t *c){ return mc_chunk_offtab(c, 4096); }
// payload span of block bi: [*off, *off+*len). returns 0 if absent.
static inline int mc_block_span(const uint8_t *c, int bi, uint32_t *off, uint32_t *len){
    uint32_t o = mc_chunk_offtab(c, (unsigned)bi);
    if(o == MC_OFF_ABSENT) return 0;
    uint32_t next = mc_chunk_offtab(c, 4096);          // terminator default
    for(unsigned j = (unsigned)bi+1; j <= 4096; ++j){
        uint32_t oj = mc_chunk_offtab(c, j);
        if(oj != MC_OFF_ABSENT){ next = oj; break; }
    }
    *off = o; *len = next - o; return 1;
}

// ---- handle ----------------------------------------------------------------
// STATIC archive: no append cursor, no grow lock, no node tree. `gen` is kept
// for the decoded-block cache's invalidation hook.
struct mc_archive {
    int fd;
    uint8_t *base;             // mmap base over the full (sparse) logical size
    uint64_t logical;          // ftruncate'd logical size = chunks_off + n*stride
    uint32_t dims[3];          // [z, y, x]
    int num_lods;
    float quality;
    uint32_t block_codec;
    uint64_t occ_off, chunks_off, slot_stride, total_chunks;
    _Atomic uint64_t gen;      // bumped on every write (cache invalidation)
};

// occupancy accessors (2 bits/chunk over the global slot space).
static inline int mca_occ_get(const struct mc_archive *a, uint64_t slot){
    const uint8_t *p = a->base + a->occ_off + (slot >> 2);
    return (*p >> ((unsigned)(slot & 3) * 2u)) & 3u;
}
static inline void mca_occ_set(struct mc_archive *a, uint64_t slot, int st){
    uint8_t *p = a->base + a->occ_off + (slot >> 2);
    unsigned sh = (unsigned)(slot & 3) * 2u;
    *p = (uint8_t)((*p & ~(3u << sh)) | ((unsigned)(st & 3) << sh));
}

void priors_load(const uint8_t *base);

#endif // MC_ARCHIVE_H_WRAP
