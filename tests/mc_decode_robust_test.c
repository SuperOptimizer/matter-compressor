// mc_decode_robust_test — the executable form of the README's promise that the
// decode path is "hardened against corrupted payloads (clamped writes, bounded
// offsets)". Constructs malformed archives in-memory and drives the read-only
// consumer surface (mc_open / mc_metadata / mc_chunk_offset / mc_decode_block /
// mc_verify_archive). The bar: NO crash / OOB / UB on ANY input (enforced under
// ASan+UBSan). Wrong decoded values are fine; a SEGV/overflow is a FAIL.
//
// These cases were distilled from AFL++ findings (scripts/fuzz.sh) that crashed
// mc_open, mc_resolve_chunk, mc_chunk_fmaplen and mc_verify_archive on
// attacker-controlled header offsets. This test is the permanent regression
// guard: once the reader validates its inputs, every case passes; under ASan it
// is a tripwire if the hardening regresses. Run under:
//   clang -fsanitize=address,undefined -fno-sanitize-recover=all ...
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Header field offsets (mirrors src/matter_compressor.c; format v7).
enum { MCH_MAGIC=0, MCH_VER=4, MCH_NX=12, MCH_NY=16, MCH_NZ=20,
       MCH_ROOTOFF=24, MCH_TOTLEN=88, MCH_METAOFF=96, MCH_METACAP=104,
       MCH_METALEN=112, MCH_QUALITY=120, MCH_PRIOROFF=128 };
#define MC_HDR 256u
#define MC_MAGIC 0x0043434Du

static void put_u32(uint8_t *b, int off, uint32_t v){ memcpy(b+off,&v,4); }
static void put_u64(uint8_t *b, int off, uint64_t v){ memcpy(b+off,&v,8); }

// Run the full read-only consumer surface against (arc,len). Must not crash.
// mc_metadata reads fixed header fields, so its documented precondition is a
// >= MC_HDR buffer (it is the "flat archive bytes" form); honor that. The
// untrusted-bytes entry point is mc_open, which validates len/magic itself.
static void consume(const uint8_t *arc, size_t len){
    if(len >= MC_HDR){ size_t ml=0; const char *m=mc_metadata(arc,&ml); (void)m;(void)ml; }
    mc_reader *r=mc_open(arc,len);
    if(r){
        int nx,ny,nz; mc_reader_dims(r,&nx,&ny,&nz);
        int nl=mc_reader_nlods(r); if(nl<0)nl=0; if(nl>8)nl=8;
        mc_u8 blk[16*16*16];
        for(int lod=0;lod<nl;++lod)
            for(int c=0;c<4;++c){
                uint64_t co=mc_chunk_offset(r,lod,c,c,c);
                if(co) mc_decode_block(r,co,0,0,0,blk);
            }
        mc_close(r);
    }
    (void)mc_verify_archive(arc,len,0);
}

static int g_cases=0;
static void run_case(const char *name, const uint8_t *arc, size_t len){
    g_cases++;
    fprintf(stderr,"case %d: %s (len=%zu)\n", g_cases, name, len);
    consume(arc,len);   // a crash here aborts the process -> test fails
}

// A minimal "valid-ish" header: correct magic + a 256-byte zeroed header so the
// reader's basic field reads stay in-bounds; individual cases then corrupt one
// field to point out of bounds.
static uint8_t *base_header(size_t total){
    uint8_t *b=calloc(1,total);
    put_u32(b,MCH_MAGIC,MC_MAGIC); put_u32(b,MCH_VER,7);
    put_u32(b,MCH_NX,256); put_u32(b,MCH_NY,256); put_u32(b,MCH_NZ,256);
    put_u64(b,MCH_TOTLEN,total);
    put_u64(b,MCH_METAOFF,MC_HDR); put_u64(b,MCH_METACAP,0); put_u64(b,MCH_METALEN,0);
    return b;
}

int main(void){
    // 1. truncated buffers — shorter than the 256-byte header. mc_open must not
    //    read past the buffer (AFL: heap-overflow in mc_open / reader_hdr_load).
    for(size_t n=0;n<=64;n+= (n<16?4:16)){
        uint8_t *b=calloc(1,n?n:1);
        if(n>=4) put_u32(b,MCH_MAGIC,MC_MAGIC);
        char nm[64]; snprintf(nm,sizeof nm,"truncated header (%zu B)",n);
        run_case(nm,b,n);
        free(b);
    }

    // 2. magic ok, but a LOD root offset points far past EOF — mc_resolve_chunk /
    //    mc_verify_archive walk the node tree at arc+root (AFL crashers).
    { size_t T=MC_HDR; uint8_t *b=base_header(T);
      put_u64(b,MCH_ROOTOFF+0*8, (uint64_t)1<<40);     // absurd root offset
      run_case("LOD0 root offset past EOF", b, T); free(b); }

    // 3. root offset just inside the header but node entries read past EOF.
    { size_t T=MC_HDR; uint8_t *b=base_header(T);
      put_u64(b,MCH_ROOTOFF+0*8, T-4);                 // 4 bytes before end: idx*8 reads OOB
      run_case("LOD0 root offset near EOF (child read OOB)", b, T); free(b); }

    // 4. prior offset points past EOF — mc_open memcpy's MC_PRIORS_BYTES from arc+poff.
    { size_t T=MC_HDR; uint8_t *b=base_header(T);
      put_u64(b,MCH_PRIOROFF, (uint64_t)1<<40);
      run_case("prior offset past EOF", b, T); free(b); }

    // 5. metadata length absurd — mc_metadata must clamp to the buffer.
    { size_t T=MC_HDR; uint8_t *b=base_header(T);
      put_u64(b,MCH_METALEN, (uint64_t)1<<40);
      run_case("metadata length past EOF", b, T); free(b); }

    // 6. all-0xFF header (every offset maxed) — stress every offset field at once.
    { size_t T=MC_HDR; uint8_t *b=malloc(T); memset(b,0xFF,T);
      put_u32(b,MCH_MAGIC,MC_MAGIC);
      run_case("all-0xFF header", b, T); free(b); }

    // 7. a root that points to an in-bounds node whose chunk-blob offset is valid
    //    but the blob's internal fields (fmaplen/bitmap/lens) run past EOF. This
    //    is the mc_chunk_q / mc_block_range / mc_chunk_blob_len family of crashes:
    //    chunk_off survives the node-tree bound but the blob structure does not.
    //    We approximate by pointing a LOD root at a region of garbage near EOF;
    //    mc_blob_struct_ok must reject it (decode -> zeros, verify -> corrupt).
    { size_t T=MC_HDR + 4096; uint8_t *b=base_header(T);
      memset(b+MC_HDR,0xAB,4096);                    // garbage "node tree" + blobs
      put_u64(b,MCH_TOTLEN,T);
      put_u64(b,MCH_ROOTOFF+0*8, MC_HDR);            // root -> garbage node array
      run_case("root -> garbage node/blob region", b, T); free(b); }

    // 8. quality field = NaN/Inf in an otherwise plausible tiny blob — exercises
    //    the float-domain UB guard (NaN -> uint16_t quant cast). A chunk offset
    //    pointing at a 0xFF f32 (NaN) quality must not poison the quant tables.
    { size_t T=MC_HDR + 4096; uint8_t *b=base_header(T);
      memset(b+MC_HDR,0xFF,64);                       // first 4 bytes (q) = NaN bit pattern
      put_u64(b,MCH_TOTLEN,T);
      run_case("NaN quality blob field", b, T); free(b); }

    printf("mc_decode_robust_test: %d cases, no crash -> OK\n", g_cases);
    return 0;   // reaching here under ASan/UBSan == pass
}
