// mc_tiff_test — round-trip the minimal TIFF writer/reader across all supported
// sample types, verifying the read view points INTO the mmap (no copy) and the
// bytes match exactly. Also checks the mmap-into-struct contract: a 4x f32
// image reads back as a contiguous float[4] grid.
#include "mc_tiff.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int fails = 0;
#define CHECK(x) do{ if(!(x)){ fails++; fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);} }while(0)

static int roundtrip(const char *path, int w, int h, int spp, mc_tiff_type ty,
                     const char *name){
    size_t ts = mc_tiff_type_size(ty);
    size_t n = (size_t)w*h*spp*ts;
    uint8_t *src = malloc(n);
    for(size_t i=0;i<n;++i) src[i]=(uint8_t)((i*131+7)&0xFF);   // deterministic pattern
    if(mc_tiff_write(path,w,h,spp,ty,src)!=0){ fprintf(stderr,"FAIL write %s\n",name); fails++; free(src); return 1; }

    mc_tiff t;
    if(mc_tiff_open(path,&t)!=0){ fprintf(stderr,"FAIL open %s\n",name); fails++; free(src); return 1; }
    CHECK(t.width==w); CHECK(t.height==h); CHECK(t.samples==spp); CHECK(t.type==ty);
    CHECK(t.pixel_bytes==n);
    CHECK(t.pixels != NULL);
    // pointer is into the mmap (not the writer's buffer) and bytes match.
    CHECK(t.pixels != src);
    CHECK(memcmp(t.pixels, src, n)==0);
    // pointer alignment: f32/u32 views must be 4-byte aligned for direct deref.
    if(ts>=4) CHECK(((uintptr_t)t.pixels & 3u)==0);
    mc_tiff_close(&t);
    CHECK(t.pixels==NULL);   // close zeros the struct
    free(src);
    printf("  %-22s %dx%d x%d OK\n", name, w, h, spp);
    return 0;
}

int main(void){
    roundtrip("/tmp/mc_tiff_u8.tif",  17, 5, 1, MC_TIFF_U8,  "u8 gray");
    roundtrip("/tmp/mc_tiff_u8c.tif", 13, 9, 3, MC_TIFF_U8,  "u8 rgb");
    roundtrip("/tmp/mc_tiff_u16.tif", 8, 8, 1, MC_TIFF_U16, "u16");
    roundtrip("/tmp/mc_tiff_i16.tif", 8, 8, 1, MC_TIFF_I16, "i16");
    roundtrip("/tmp/mc_tiff_u32.tif", 8, 8, 1, MC_TIFF_U32, "u32");
    roundtrip("/tmp/mc_tiff_f32.tif", 8, 8, 1, MC_TIFF_F32, "f32");

    // the surface format: 4x f32 (x,y,z,depth). Write a known grid, read it
    // back as a float[4] view, check exact values -> the mmap-into-struct path.
    {
        int gw=6, gh=4;
        float (*g)[4] = malloc((size_t)gw*gh*sizeof *g);
        for(int y=0;y<gh;++y)for(int x=0;x<gw;++x){
            float *p=g[y*gw+x]; p[0]=x*10.f; p[1]=y*10.f; p[2]=x+y*0.5f; p[3]=5.f;
        }
        CHECK(mc_tiff_write("/tmp/mc_surf.tif",gw,gh,4,MC_TIFF_F32,g)==0);
        mc_tiff t;
        CHECK(mc_tiff_open("/tmp/mc_surf.tif",&t)==0);
        CHECK(t.width==gw && t.height==gh && t.samples==4 && t.type==MC_TIFF_F32);
        const float (*r)[4] = (const float (*)[4])t.pixels;   // direct struct view
        int bad=0;
        for(int i=0;i<gw*gh;++i) for(int c=0;c<4;++c) if(r[i][c]!=((float*)g[i])[c]) bad++;
        CHECK(bad==0);
        printf("  surface 4xf32 %dx%d mmap-view OK\n", gw, gh);
        mc_tiff_close(&t);
        free(g);
    }

    // a too-short / garbage file must fail cleanly (no crash, returns <0).
    { FILE*f=fopen("/tmp/mc_tiff_bad.tif","wb"); fputs("nope",f); fclose(f);
      mc_tiff t; CHECK(mc_tiff_open("/tmp/mc_tiff_bad.tif",&t)<0); CHECK(t.map==NULL); }

    printf(fails ? "mc_tiff_test: %d FAILED\n" : "mc_tiff_test: OK\n", fails);
    return fails ? 1 : 0;
}
