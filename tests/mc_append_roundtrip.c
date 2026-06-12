// mc_append_roundtrip — exercise the APPENDABLE writer: open a fresh archive, append
// some 256^3 chunks (multiple LODs), CLOSE it, REOPEN it (persistence), append more,
// close again, then decode every appended chunk with a flat reader AND a streaming
// reader and verify the values reconstruct within tolerance. Also checks coverage +
// that the file is a valid archive after each phase. No external deps.
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DIM 512   // 2 chunks/axis at LOD0 -> exercises a multi-chunk index path

// synthetic material: a smooth ball, air outside. Per (lod) the value shifts so we can
// tell LODs apart. value is never 0 inside the ball (0 is the air sentinel).
static mc_u8 sample(int lod, int x,int y,int z){
    int dim = DIM >> lod;
    double cx=dim/2.0, cy=dim/2.0, cz=dim/2.0;
    double r=sqrt((x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz));
    if(r > dim*0.42) return 0;
    double v=100.0 + 90.0*cos(r*0.2) + lod*5;
    if(v<1)v=1; if(v>255)v=255;
    return (mc_u8)v;
}

// fill a 256^3 chunk buffer for (lod,cz,cy,cx).
static void fill_chunk(mc_u8 *buf, int lod, int cz,int cy,int cx){
    for(int z=0;z<256;++z)for(int y=0;y<256;++y)for(int x=0;x<256;++x)
        buf[((size_t)z*256+y)*256+x] = sample(lod, cx*256+x, cy*256+y, cz*256+z);
}

// streaming byte-source over a FILE.
typedef struct { FILE *f; } fsrc;
static int fread_at(void *ud, uint64_t off, uint32_t len, uint8_t *dst){
    fsrc *s=(fsrc*)ud;
    if(fseek(s->f,(long)off,SEEK_SET)!=0) return -1;
    return fread(dst,1,len,s->f)==len ? 0 : -1;
}

// decode + score one chunk against the synthetic source. returns 0 if ok.
static int verify_chunk(mc_reader *r, int lod, int cz,int cy,int cx, double *rmse_out, long *leak_out){
    uint64_t co=mc_chunk_offset(r,lod,cz,cy,cx);
    if(!co){ fprintf(stderr,"  chunk lod%d (%d,%d,%d) MISSING\n",lod,cz,cy,cx); return 1; }
    mc_u8 dec[16*16*16];
    long sqerr=0, mat=0, leak=0;
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
        mc_decode_block(r,co,bz,by,bx,dec);
        for(int z=0;z<16;++z)for(int y=0;y<16;++y)for(int x=0;x<16;++x){
            int gx=(cx*16+bx)*16+x, gy=(cy*16+by)*16+y, gz=(cz*16+bz)*16+z;
            int s=sample(lod,gx,gy,gz), d=dec[(z*16+y)*16+x];
            if(s==0){ if(d!=0) leak++; }
            else { mat++; long e=labs((long)d-s); sqerr+=e*e; }
        }
    }
    *rmse_out = mat?sqrt((double)sqerr/mat):0;
    *leak_out = leak;
    return 0;
}

int main(void){
    const char *path = "mc_append_test.mca";
    remove(path);
    const float Q = 8.0f;
    mc_u8 *chunk = malloc((size_t)256*256*256);

    // ---- phase 1: create + append LOD0 chunk (0,0,0) and LOD1 chunk (0,0,0) ----
    mc_archive *w = mc_archive_open(path, DIM, Q);
    if(!w){ fprintf(stderr,"writer_open failed\n"); return 1; }
    fill_chunk(chunk,0,0,0,0); if(mc_archive_append_chunk_raw(w,0,0,0,0,chunk)){ fprintf(stderr,"append l0 failed\n"); return 1; }
    fill_chunk(chunk,1,0,0,0); if(mc_archive_append_chunk_raw(w,1,0,0,0,chunk)){ fprintf(stderr,"append l1 failed\n"); return 1; }
    const char *META = "{\"profile\":\"test\",\"quality\":8.0}";
    if(mc_archive_set_metadata(w, META, strlen(META))){ fprintf(stderr,"set_metadata failed\n"); return 1; }
    if(mc_archive_set_metadata(w, META, (size_t)1<<20)==0){ fprintf(stderr,"set_metadata accepted len > cap\n"); return 1; }
    if(mc_archive_chunk_coverage(w,0,0,0,0)!=MC_PRESENT){ fprintf(stderr,"coverage l0 wrong\n"); return 1; }
    mc_archive_close(w);
    printf("phase 1: appended LOD0(0,0,0) + LOD1(0,0,0), closed\n");

    // ---- phase 2: REOPEN (persistence) + append another LOD0 chunk (1,1,1) ----
    w = mc_archive_open(path, DIM, Q);
    if(!w){ fprintf(stderr,"reopen failed\n"); return 1; }
    if(mc_archive_chunk_coverage(w,0,0,0,0)!=MC_PRESENT){ fprintf(stderr,"reopen lost LOD0(0,0,0)\n"); return 1; }
    { size_t ml=0; const char *m=mc_archive_metadata(w,&ml);
      if(!m || ml!=strlen("{\"profile\":\"test\",\"quality\":8.0}") || memcmp(m,"{\"profile\":\"test\"",17)){
          fprintf(stderr,"reopen lost metadata (len %zu)\n",ml); return 1; }
      printf("  metadata persisted across reopen: %.*s\n",(int)ml,m); }
    fill_chunk(chunk,0,1,1,1); if(mc_archive_append_chunk_raw(w,0,1,1,1,chunk)){ fprintf(stderr,"append l0(1,1,1) failed\n"); return 1; }

    // SAME-HANDLE read-back: decode a freshly-appended chunk directly via the archive
    // (no separate reader) — proves one handle reads + writes.
    {
        uint64_t co = mc_archive_chunk_offset(w,0,1,1,1);
        if(!co){ fprintf(stderr,"same-handle resolve of just-appended chunk failed\n"); return 1; }
        mc_u8 dec[16*16*16]; long sq=0,mat=0,leak=0;
        for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
            mc_archive_decode_block(w,co,bz,by,bx,dec);
            for(int z=0;z<16;++z)for(int y=0;y<16;++y)for(int x=0;x<16;++x){
                int gx=(1*16+bx)*16+x, gy=(1*16+by)*16+y, gz=(1*16+bz)*16+z;
                int s=sample(0,gx,gy,gz), d=dec[(z*16+y)*16+x];
                if(s==0){ if(d) leak++; } else { mat++; long e=labs((long)d-s); sq+=e*e; }
            }
        }
        double rmse=mat?sqrt((double)sq/mat):0;
        printf("  self  LOD0(1,1,1) via archive handle: RMSE=%.2f leak=%ld\n",rmse,leak);
        if(leak || rmse>8.0){ fprintf(stderr,"same-handle decode bad\n"); return 1; }
    }
    mc_archive_close(w);
    printf("phase 2: reopened (persisted), appended LOD0(1,1,1), self-read OK, closed\n");

    // ---- phase 3a: flat reader verify ----
    FILE *f=fopen(path,"rb"); fseek(f,0,SEEK_END); long flen=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t *buf=malloc(flen); if(fread(buf,1,flen,f)!=(size_t)flen){ fprintf(stderr,"read file\n"); return 1; } fclose(f);
    mc_reader *r=mc_open(buf,flen); mc_reader_set_quality(r,Q);
    { size_t ml=0; const char *m=mc_metadata(buf,&ml);
      if(!m || !ml || memcmp(m,"{\"profile\":\"test\"",17)){ fprintf(stderr,"flat mc_metadata read failed\n"); return 1; }
      printf("  flat metadata: %.*s\n",(int)ml,m); }
    int fail=0; double rmse; long leak;
    struct { int lod,cz,cy,cx; } chunks[]={ {0,0,0,0},{1,0,0,0},{0,1,1,1} };
    for(int i=0;i<3;++i){
        if(verify_chunk(r,chunks[i].lod,chunks[i].cz,chunks[i].cy,chunks[i].cx,&rmse,&leak)){ fail=1; continue; }
        printf("  flat  LOD%d(%d,%d,%d): RMSE=%.2f leak=%ld\n",chunks[i].lod,chunks[i].cz,chunks[i].cy,chunks[i].cx,rmse,leak);
        if(leak!=0 || rmse>8.0) fail=1;
    }
    mc_close(r); free(buf);

    // ---- phase 3b: STREAMING reader verify (same results via byte-source) ----
    fsrc fs={ fopen(path,"rb") };
    mc_reader *sr=mc_open_streaming(fread_at,&fs,(uint64_t)flen); mc_reader_set_quality(sr,Q);
    if(!sr){ fprintf(stderr,"open_streaming failed\n"); return 1; }
    for(int i=0;i<3;++i){
        if(verify_chunk(sr,chunks[i].lod,chunks[i].cz,chunks[i].cy,chunks[i].cx,&rmse,&leak)){ fail=1; continue; }
        printf("  strm  LOD%d(%d,%d,%d): RMSE=%.2f leak=%ld\n",chunks[i].lod,chunks[i].cz,chunks[i].cy,chunks[i].cx,rmse,leak);
        if(leak!=0 || rmse>8.0) fail=1;
    }
    mc_close(sr); fclose(fs.f);

    free(chunk); remove(path);
    printf("%s\n", fail?"FAIL ✗":"PASS ✓");
    return fail;
}
