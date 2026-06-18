#define _GNU_SOURCE
#include "matter_compressor.h"
#include "mc_codec.h"
#include "mc_archive.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
typedef uint8_t u8;
// ============================================================================
// mc_reader — flat-buffer / streaming reader for the STATIC .mca v8 format.
//
// Chunk addressing is pure arithmetic (chunks_off + slot_index*slot_stride); a
// sparse / never-written slot reads as zero (chunk_len < payload start). There is
// no node tree to walk -- the streaming reader fetches a chunk's window by its
// computed offset.
// ============================================================================
#define MC_PRIORS_BLOB (8u + 2u*8u*32u*2u)     // magic+ver + plo + phi = 1032
#define MCA_PRIORS_OFF_  256u
#define MCA_PRIORS_MAGIC_ 0x53524950u

struct mc_reader {
    const uint8_t *arc; size_t len;     // flat mode (NULL when streaming)
    mc_read_fn read; void *read_ud;     // streaming mode
    uint32_t dims[3];                   // [z, y, x]
    int num_lods; float quality; uint32_t block_codec;
    uint64_t occ_off, chunks_off, slot_stride, total_chunks;
    u8 priors[MC_PRIORS_BLOB]; int has_priors;
    u8 *occ_buf;                        // streaming: cached occupancy region (small)
    u8 *cbuf; uint64_t cbuf_off, cbuf_len;   // streaming chunk window
    int partial;
    mc_codec_ctx *codec;
};

// occupancy state of a slot: flat reads the mmap, streaming reads the cached copy.
// occ_off/total_chunks come from an UNTRUSTED header -> bounds-check every index.
static int occ_state(mc_reader *r, uint64_t slot){
    if(slot >= r->total_chunks) return MC_OCC_DONT_KNOW;
    uint64_t byte = slot >> 2;
    if(r->arc){
        if(r->occ_off + byte >= r->len) return MC_OCC_DONT_KNOW;   // OOB -> absent
        return (r->arc[r->occ_off + byte] >> ((unsigned)(slot&3)*2u)) & 3u;
    }
    if(!r->occ_buf) return MC_OCC_REAL;       // no cache -> don't gate (caller-managed)
    return (r->occ_buf[byte] >> ((unsigned)(slot&3)*2u)) & 3u;
}

static void hdr_load(mc_reader *r, const u8 *h){
    memcpy(&r->dims[0], h+MCAH_NZ, 4); memcpy(&r->dims[1], h+MCAH_NY, 4); memcpy(&r->dims[2], h+MCAH_NX, 4);
    r->num_lods = h[MCAH_NUMLODS]; if(r->num_lods<1) r->num_lods=1; if(r->num_lods>MCA_MAXLODS) r->num_lods=MCA_MAXLODS;
    memcpy(&r->quality, h+MCAH_QUALITY, 4);
    memcpy(&r->block_codec, h+MCAH_BLOCKCODEC, 4); r->block_codec = MC_BLOCKCODEC_CABAC;
    memcpy(&r->occ_off, h+MCAH_OCCOFF, 8); memcpy(&r->chunks_off, h+MCAH_CHUNKSOFF, 8);
    memcpy(&r->slot_stride, h+MCAH_SLOTSTRIDE, 8); memcpy(&r->total_chunks, h+MCAH_TOTCHUNKS, 8);
}

mc_reader *mc_open(const uint8_t *arc, size_t len){
    if(!arc || len < MCA_HDR_REGION) return NULL;
    uint32_t magic,ver; memcpy(&magic,arc+MCAH_MAGIC,4); memcpy(&ver,arc+MCAH_VERSION,4);
    if(magic != MCA_MAGIC || ver != MCA_VERSION) return NULL;
    mc_codec_init();
    mc_reader *r=calloc(1,sizeof *r); if(!r) return NULL;
    r->arc=arc; r->len=len; r->codec=mc_codec_ctx_new();
    hdr_load(r, arc);
    uint32_t pmagic; memcpy(&pmagic, arc+MCA_PRIORS_OFF_, 4);
    if(pmagic==MCA_PRIORS_MAGIC_){
        memcpy(r->priors, arc+MCA_PRIORS_OFF_, MC_PRIORS_BLOB); r->has_priors=1;
        priors_load(arc);
    }
    return r;
}

static int sread(mc_reader *r, uint64_t off, uint32_t len, u8 *dst){ return r->read(r->read_ud, off, len, dst); }

mc_reader *mc_open_streaming(mc_read_fn read, void *ud, uint64_t total_len){
    mc_codec_init();
    mc_reader *r=calloc(1,sizeof *r); if(!r) return NULL;
    r->read=read; r->read_ud=ud; r->len=(size_t)total_len; r->codec=mc_codec_ctx_new();
    u8 hdr[MCA_HDR_REGION];
    if(read(ud,0,MCA_HDR_REGION,hdr)!=0){ mc_codec_ctx_free(r->codec); free(r); return NULL; }
    uint32_t magic,ver; memcpy(&magic,hdr+MCAH_MAGIC,4); memcpy(&ver,hdr+MCAH_VERSION,4);
    if(magic!=MCA_MAGIC || ver!=MCA_VERSION){ mc_codec_ctx_free(r->codec); free(r); return NULL; }
    hdr_load(r, hdr);
    uint32_t pmagic; memcpy(&pmagic, hdr+MCA_PRIORS_OFF_, 4);
    if(pmagic==MCA_PRIORS_MAGIC_){
        memcpy(r->priors, hdr+MCA_PRIORS_OFF_, MC_PRIORS_BLOB); r->has_priors=1;
        mc_codec_set_priors((const uint16_t*)(r->priors+8),(const uint16_t*)(r->priors+8+8*32*2));
    } else mc_codec_set_priors(NULL,NULL);
    // cache the (small) occupancy region so chunk presence is a local lookup, not
    // a per-chunk GET. total_chunks*2 bits; e.g. ~145 KB for a full scroll.
    uint64_t occ_bytes = (r->total_chunks + 3) / 4;
    if(occ_bytes && occ_bytes < (256u<<20)){
        r->occ_buf = malloc((size_t)occ_bytes);
        if(r->occ_buf && read(ud, r->occ_off, (uint32_t)occ_bytes, r->occ_buf)!=0){
            free(r->occ_buf); r->occ_buf=NULL;
        }
    }
    return r;
}

int mc_reader_priors(mc_reader *r, const uint16_t **plo, const uint16_t **phi){
    if(!r||!r->has_priors) return 0;
    if(plo)*plo=(const uint16_t*)(r->priors+8);
    if(phi)*phi=(const uint16_t*)(r->priors+8+8*32*2);
    return 1;
}
void mc_reader_dims(mc_reader *r, int *nx, int *ny, int *nz){
    if(nx)*nx=r?(int)r->dims[2]:0; if(ny)*ny=r?(int)r->dims[1]:0; if(nz)*nz=r?(int)r->dims[0]:0;
}
float mc_reader_quality(mc_reader *r){ return r?r->quality:0.f; }
int mc_reader_nlods(mc_reader *r){ return r?r->num_lods:0; }
void mc_reader_set_partial_fetch(mc_reader *r, int on){ if(r) r->partial=on; }
void mc_reader_set_quality(mc_reader *r, float q){ if(r&&r->codec) mc_codec_ctx_set_quality(r->codec,q); }
void mc_close(mc_reader *r){ if(!r)return; mc_codec_ctx_free(r->codec); free(r->cbuf); free(r->occ_buf); free(r); }

// occupancy of a slot in flat mode (bounds-checked against the buffer length —
// occ_off comes from an untrusted header).
static int occ_flat(const u8 *arc, uint64_t occ_off, uint64_t slot, size_t len){
    uint64_t i = occ_off + (slot>>2);
    if(i >= len) return MC_OCC_DONT_KNOW;
    return (arc[i] >> ((unsigned)(slot&3)*2u)) & 3u;
}

uint64_t mc_chunk_offset(mc_reader *r, int lod, int cz,int cy,int cx){
    if(!r||lod<0||lod>=r->num_lods||cz<0||cy<0||cx<0) return 0;
    if((uint64_t)cz>=mca_chunks_axis(r->dims[0],lod)||(uint64_t)cy>=mca_chunks_axis(r->dims[1],lod)
       ||(uint64_t)cx>=mca_chunks_axis(r->dims[2],lod)) return 0;
    uint64_t slot = mca_slot_index(r->dims,r->num_lods,lod,cz,cy,cx);
    if(occ_state(r,slot)!=MC_OCC_REAL) return 0;   // air / not present
    return r->chunks_off + slot*r->slot_stride;
}
uint64_t mc_chunk_offset_chk(mc_reader *r, int lod, int cz,int cy,int cx, int *err){
    if(err)*err=0; return mc_chunk_offset(r,lod,cz,cy,cx);
}

// fetch a streaming chunk's window [chunk_off, chunk_off+chunk_len) into cbuf.
static const u8 *sfetch_chunk(mc_reader *r, uint64_t chunk_off){
    if(r->cbuf && r->cbuf_off==chunk_off) return r->cbuf;
    u8 head[MC_CHUNK_PAYLOAD_OFF];                       // header + offset table
    if(sread(r,chunk_off,MC_CHUNK_PAYLOAD_OFF,head)!=0) return NULL;
    uint64_t total = mc_chunk_len(head);
    if(total < MC_CHUNK_PAYLOAD_OFF || total > MC_SLOT_STRIDE) return NULL;
    void *tmp=realloc(r->cbuf,total); if(!tmp) return NULL; r->cbuf=tmp;
    if(sread(r,chunk_off,(uint32_t)total,r->cbuf)!=0) return NULL;
    // integrity: a streaming fetch crosses the network -> verify the stored hash
    // over the whole chunk before decoding. Mismatch = corruption in transit.
    if(mc_xxh64(r->cbuf+8,(size_t)(total-8),0x6D636168756E6Bull) != mc_chunk_xxh3(r->cbuf))
        return NULL;
    r->cbuf_off=chunk_off; r->cbuf_len=total;
    return r->cbuf;
}

void mc_decode_block(mc_reader *r, uint64_t chunk_off, int bz,int by,int bx, mc_u8 *dst){
    if(!r||!chunk_off){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    const u8 *c;
    if(r->arc){
        if(chunk_off + MC_CHUNK_PAYLOAD_OFF > r->len){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
        c = r->arc + chunk_off;
    } else {
        c = sfetch_chunk(r, chunk_off);
        if(!c){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    }
    if(mc_chunk_len(c) < MC_CHUNK_PAYLOAD_OFF){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    uint32_t off,len;
    if(!mc_block_span(c,(bz*16+by)*16+bx,&off,&len)){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    if(off+len > mc_chunk_len(c)){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    float cq = mc_chunk_q(c); if(!(cq>0.0f)||cq>1024.0f) cq=1.0f;
    mc_codec_ctx_set_quality(r->codec, cq);
    mc_dec_block_codec(r->codec, r->block_codec, c+off, len, dst);
}

// total byte length of the chunk at chunk_off (terminator entry of the table).
uint64_t mc_reader_chunk_blob_len(mc_reader *r, uint64_t chunk_off){
    if(!r||!chunk_off) return 0;
    if(r->arc){
        if(chunk_off+MC_CHUNK_PAYLOAD_OFF>r->len) return 0;
        uint64_t l=mc_chunk_len(r->arc+chunk_off); return l>=MC_CHUNK_PAYLOAD_OFF?l:0;
    }
    u8 head[MC_CHUNK_PAYLOAD_OFF];
    if(sread(r,chunk_off,MC_CHUNK_PAYLOAD_OFF,head)!=0) return 0;
    uint64_t l=mc_chunk_len(head); return (l>=MC_CHUNK_PAYLOAD_OFF&&l<=MC_SLOT_STRIDE)?l:0;
}
int mc_reader_read_blob(mc_reader *r, uint64_t chunk_off, size_t len, uint8_t *dst){
    if(!r||!chunk_off||!len||!dst) return -1;
    if(r->arc){
        if(chunk_off > r->len || len > r->len - chunk_off) return -1;   // bounds
        memcpy(dst, r->arc+chunk_off, len); return 0;
    }
    size_t done=0;
    while(done<len){
        uint32_t n=(len-done>0x40000000u)?0x40000000u:(uint32_t)(len-done);
        if(sread(r,chunk_off+done,n,dst+done)!=0) return -1;
        done+=n;
    }
    return 0;
}

// integrity hash over a chunk's stored bytes [8, len) (matches the writer).
uint64_t mc_chunk_compute_hash(const uint8_t *blob, uint64_t blob_len){
    if(blob_len<=8) return 0;
    return mc_xxh64(blob+8, (size_t)(blob_len-8), 0x6D636168756E6Bull);
}

// flat verify: walk occupancy, recompute each REAL chunk's hash.
long mc_verify_archive(const uint8_t *arc, size_t len, int verbose){
    if(!arc||len<MCA_HDR_REGION) return -1;
    uint32_t magic,ver; memcpy(&magic,arc+MCAH_MAGIC,4); memcpy(&ver,arc+MCAH_VERSION,4);
    if(magic!=MCA_MAGIC||ver!=MCA_VERSION) return -1;
    uint32_t dz,dy,dx; memcpy(&dz,arc+MCAH_NZ,4); memcpy(&dy,arc+MCAH_NY,4); memcpy(&dx,arc+MCAH_NX,4);
    uint32_t dims[3]={dz,dy,dx}; int nl=arc[MCAH_NUMLODS];
    uint64_t occ_off,chunks_off,stride; memcpy(&occ_off,arc+MCAH_OCCOFF,8);
    memcpy(&chunks_off,arc+MCAH_CHUNKSOFF,8); memcpy(&stride,arc+MCAH_SLOTSTRIDE,8);
    long bad=0, checked=0;
    for(int lod=0;lod<nl;lod++){
        uint64_t ncz=mca_chunks_axis(dims[0],lod),ncy=mca_chunks_axis(dims[1],lod),ncx=mca_chunks_axis(dims[2],lod);
        for(uint64_t cz=0;cz<ncz;cz++)for(uint64_t cy=0;cy<ncy;cy++)for(uint64_t cx=0;cx<ncx;cx++){
            uint64_t slot=mca_slot_index(dims,nl,lod,cz,cy,cx);
            if(occ_flat(arc,occ_off,slot,len)!=MC_OCC_REAL) continue;
            uint64_t co=chunks_off+slot*stride;
            if(co+MC_CHUNK_PAYLOAD_OFF>len){ bad++; continue; }
            const u8 *c=arc+co; uint64_t clen=mc_chunk_len(c);
            if(clen<MC_CHUNK_PAYLOAD_OFF||co+clen>len){ bad++; continue; }
            checked++;
            if(mc_chunk_compute_hash(c,clen)!=mc_chunk_xxh3(c)) bad++;
        }
    }
    if(verbose) fprintf(stderr,"mc_verify: %ld chunks checked, %ld bad\n",checked,bad);
    return bad;
}
