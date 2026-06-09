// ============================================================================
// mc_archive.h — matter-compressor on-disk archive format constants.
//
// A sparse multi-level node tree of dense 256^3 chunks, each a 16^3 grid of DCT
// blocks coded by mc_codec. 8 independent LODs (LOD0 = full res), each its own
// sparse tree. LODs are independently fetchable AND independently decodable — no
// cross-LOD dependency (a hard design constraint).
//
// HIERARCHY (per LOD), all sparse grids are 16x16x16 (4 bits/axis):
//   chunk coord (voxel>>8, up to 12 bits/axis) = [ region 4b | subregion 4b | shard 4b ]
//   CHUNK (256^3 voxels = 16^3 DCT blocks) is the dense leaf. Above it: a dense
//   16^3 SHARD table of chunk offsets, then 2 SPARSE 16^3 node levels.
//   Covers 2^12 chunks/axis = 2^20 voxels/axis.
//
// NODE = a 16^3 child map stored SPARSELY: a 512B (4096-bit) occupancy bitmap + a
// packed array of present children's u64 offsets. Absent child = empty region
// (the free "pointer to 0"). CHUNK blob = [u32 masklen][mask][512B block-bitmap]
// [present-block u32 lens][block payloads]; one range-GET fetches a whole chunk.
//
// LAYOUT: [256B header][metadata region up to 128KB][archive data from 128KB].
// ============================================================================
#ifndef MC_ARCHIVE_H
#define MC_ARCHIVE_H
#include <stdint.h>
#include <string.h>

// ---- sparse node (16^3 child map) ----
#define MC_GRID         16
#define MC_GRID3        4096        // 16^3
#define MC_BITMAP_BYTES 512         // 4096 bits
#define MC_SHARD_BYTES  (MC_GRID3*8)// dense 16^3 chunk-offset table
#define MC_SPARSE_LEVELS 2          // sparse node levels above the dense shard

// ---- header (256B) ----
#define MC_MAGIC   0x0043434Du      // "MCC\0"
#define MCH_MAGIC   0               // u32 magic
#define MCH_VER     4               // byte offset of the u32 format-version field
#define MCH_NX      12              // u32 volume dims
#define MCH_NY      16
#define MCH_NZ      20
#define MCH_ROOTOFF 24              // u64[8] per-LOD root-node file offset (0 = empty LOD)
#define MCH_TOTLEN  88              // u64 total archive length
#define MCH_METAOFF 96              // u64 metadata region start (= MC_HDR = 256)
#define MCH_METACAP 104             // u64 metadata region capacity (= MC_META_END - MC_HDR)
#define MCH_METALEN 112             // u64 metadata bytes actually written
#define MC_HDR      256u            // header size; metadata region begins here
#define MC_META_END (128u*1024u)    // archive data begins at this offset (128KB)
#define MC_META_CAP (MC_META_END - MC_HDR)
#define MC_VERSION  1u              // matter-compressor format version (starts at 1)

#define MC_CHUNK_ALIGN 256          // volume dim must be a multiple of this

// occupancy bitmap + chunk-coord nibble helpers
static inline int  mc_bit_get(const uint8_t*bm,int i){ return (bm[i>>3]>>(i&7))&1; }
static inline void mc_bit_set(uint8_t*bm,int i){ bm[i>>3]|=(uint8_t)(1u<<(i&7)); }
static inline int  mc_nib(int chunkcoord,int level){ return (chunkcoord>>(4*level))&15; }
#endif
