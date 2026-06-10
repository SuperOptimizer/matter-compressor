// ============================================================================
// mc_fetch — pull a sub-volume out of a Vesuvius Challenge zarr (v2, u8,
// blosc-zstd chunks) into a raw u8 .bin for mc_bench.
//
// Transport is libs3 (vendored): accepts s3://bucket/key roots (SigV4-signed
// when credentials resolve, anonymous otherwise) and plain https:// roots
// (e.g. the dl.ash2txt.org mirror). Blosc1/zstd chunk decode is implemented
// here (shuffle=0, typesize=1 — what the standardized scroll zarrs use).
//
// usage: mc_fetch <zarr-root-url> <z0> <y0> <x0> <dim> <out.bin>
//   e.g. mc_fetch https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/0 \
//          7168 3840 3968 512 /tmp/vesuvius_512.bin
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <zstd.h>
#include "libs3.h"

typedef uint8_t u8;

// ---- minimal blosc1 chunk decode (shuffle=0 only) ----
// header: ver(1) verlz(1) flags(1) typesize(1) nbytes(4) blocksize(4) cbytes(4), LE.
static u8 *blosc_decode(const u8 *src, size_t srclen, size_t *out_len){
    if(srclen < 16) return NULL;
    uint8_t flags = src[2];
    uint32_t nbytes, blocksize, cbytes;
    memcpy(&nbytes, src+4, 4); memcpy(&blocksize, src+8, 4); memcpy(&cbytes, src+12, 4);
    if(cbytes > srclen || !nbytes) return NULL;
    if(flags & 0x1 || flags & 0x4){ fprintf(stderr,"blosc: shuffle unsupported\n"); return NULL; }
    u8 *out = malloc(nbytes);
    if(flags & 0x2){                       // memcpyed: raw payload follows header
        if(16+(size_t)nbytes > srclen){ free(out); return NULL; }
        memcpy(out, src+16, nbytes); *out_len = nbytes; return out;
    }
    uint32_t nblocks = (nbytes + blocksize - 1) / blocksize;
    const u8 *bstarts = src + 16;
    if(16 + (size_t)nblocks*4 > srclen){ free(out); return NULL; }
    size_t off = 0;
    for(uint32_t b = 0; b < nblocks; ++b){
        uint32_t bs; memcpy(&bs, bstarts + (size_t)b*4, 4);
        uint32_t neblock = (b == nblocks-1) ? nbytes - b*blocksize : blocksize;
        if((size_t)bs + 4 > srclen){ free(out); return NULL; }
        int32_t cb; memcpy(&cb, src + bs, 4);
        const u8 *payload = src + bs + 4;
        if(cb <= 0 || (size_t)bs + 4 + (size_t)cb > srclen){ free(out); return NULL; }
        if((uint32_t)cb == neblock){       // stored uncompressed
            memcpy(out + off, payload, neblock);
        } else {
            size_t got = ZSTD_decompress(out + off, neblock, payload, (size_t)cb);
            if(ZSTD_isError(got) || got != neblock){
                fprintf(stderr,"blosc: zstd block %u decode failed (%s)\n", b,
                        ZSTD_isError(got) ? ZSTD_getErrorName(got) : "short");
                free(out); return NULL;
            }
        }
        off += neblock;
    }
    *out_len = nbytes; return out;
}

// ---- tiny .zarray scraper: ints for "chunks":[a,b,c] ----
static int zarray_chunks(const char *json, long ch[3]){
    const char *p = strstr(json, "\"chunks\""); if(!p) return -1;
    p = strchr(p, '['); if(!p) return -1; ++p;
    for(int i = 0; i < 3; ++i){ ch[i] = strtol(p, (char**)&p, 10); while(*p==','||*p==' '||*p=='\n') ++p; }
    return (ch[0] > 0 && ch[1] > 0 && ch[2] > 0) ? 0 : -1;
}
// "compressor": null -> raw chunks; otherwise assume blosc.
static int zarray_raw(const char *json){
    const char *p = strstr(json, "\"compressor\""); if(!p) return 1;
    p += 12; while(*p==':'||*p==' '||*p=='\t'||*p=='\n') ++p;
    return strncmp(p, "null", 4) == 0;
}

int main(int argc, char **argv){
    if(argc != 7){
        fprintf(stderr,"usage: %s <zarr-root-url> <z0> <y0> <x0> <dim> <out.bin>\n", argv[0]);
        return 1;
    }
    const char *root = argv[1];
    long z0 = atol(argv[2]), y0 = atol(argv[3]), x0 = atol(argv[4]), dim = atol(argv[5]);
    const char *outpath = argv[6];

    s3_config cfg = {0};   // anonymous unless env/INI creds resolve; https passes through
    s3_client *c = s3_client_new(&cfg);
    if(!c){ fprintf(stderr,"s3 client init failed\n"); return 1; }

    char url[2048]; s3_response r = {0};
    snprintf(url, sizeof url, "%s/.zarray", root);
    if(s3_get(c, url, &r) != S3_OK || !s3_response_ok(&r)){
        fprintf(stderr,"GET %s failed: http %ld (%s)\n", url, r.status, s3_client_last_error(c));
        return 1;
    }
    long ch[3];
    if(zarray_chunks((const char*)r.body, ch) != 0){ fprintf(stderr,"bad .zarray\n"); return 1; }
    int raw_chunks = zarray_raw((const char*)r.body);
    s3_response_free(&r);
    long CH = ch[0];
    if(ch[1] != CH || ch[2] != CH || z0 % CH || y0 % CH || x0 % CH || dim % CH){
        fprintf(stderr,"need cubic chunks and chunk-aligned region (chunk=%ld)\n", CH);
        return 1;
    }

    long n = dim / CH;
    u8 *vol = calloc((size_t)dim*dim*dim, 1);
    long done = 0, total = n*n*n;
    for(long iz = 0; iz < n; ++iz) for(long iy = 0; iy < n; ++iy) for(long ix = 0; ix < n; ++ix){
        snprintf(url, sizeof url, "%s/%ld/%ld/%ld", root, z0/CH+iz, y0/CH+iy, x0/CH+ix);
        s3_response cr = {0};
        if(s3_get(c, url, &cr) != S3_OK || !s3_response_ok(&cr)){
            fprintf(stderr,"GET %s: http %ld — treating as fill (0)\n", url, cr.status);
            s3_response_free(&cr); ++done; continue;
        }
        size_t dlen = 0; u8 *chunk;
        if(raw_chunks){ dlen = cr.body_len; chunk = cr.body; cr.body = NULL; }   // take ownership
        else chunk = blosc_decode(cr.body, cr.body_len, &dlen);
        s3_response_free(&cr);
        if(!chunk || dlen != (size_t)CH*CH*CH){ fprintf(stderr,"chunk decode failed: %s\n", url); free(chunk); return 1; }
        for(long z = 0; z < CH; ++z) for(long y = 0; y < CH; ++y)
            memcpy(vol + ((size_t)(iz*CH+z)*dim + (iy*CH+y))*dim + ix*CH,
                   chunk + ((size_t)z*CH + y)*CH, CH);
        free(chunk);
        fprintf(stderr,"\r%ld/%ld", ++done, total);
    }
    fprintf(stderr,"\n");
    s3_client_free(c);

    FILE *f = fopen(outpath, "wb");
    if(!f){ perror("fopen out"); return 1; }
    fwrite(vol, 1, (size_t)dim*dim*dim, f); fclose(f);
    size_t zeros = 0; double mean = 0;
    for(size_t i = 0; i < (size_t)dim*dim*dim; ++i){ zeros += !vol[i]; mean += vol[i]; }
    printf("wrote %s: %ld^3 u8, mean %.1f, zeros %.2f%%\n", outpath, dim,
           mean/((double)dim*dim*dim), 100.0*zeros/((double)dim*dim*dim));
    free(vol);
    return 0;
}
