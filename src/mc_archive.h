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
#define MC_HDR      256u            // header size; metadata region begins here
#define MC_META_END (128u*1024u)    // archive data begins at this offset (128KB)
#define MC_META_CAP (MC_META_END - MC_HDR)
#define MC_VERSION  4u              // format version (v4: header bins in-stream, subcube mask)

#define MC_CHUNK_ALIGN 256          // volume dim must be a multiple of this

// chunk-blob block-bitmap + chunk-coord nibble helpers
static inline int  mc_bit_get(const uint8_t*bm,int i){ return (bm[i>>3]>>(i&7))&1; }
static inline void mc_bit_set(uint8_t*bm,int i){ bm[i>>3]|=(uint8_t)(1u<<(i&7)); }
static inline int  mc_nib(int chunkcoord,int level){ return (chunkcoord>>(4*level))&15; }
#endif
