// mc_segment_test — the classical sheet detector + topology post-processing.
// Synthetic volume with a known planar sheet -> structure-tensor detect finds
// it; then unit-test each post-processing op (remove specks, plug 1-voxel hole,
// fill cavity, connected-component label).
#include "mc_segment.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NZ 40
#define NY 40
#define NX 40
#define IDX(z,y,x) (((size_t)(z)*NY+(y))*NX+(x))

static int fails=0;
#define CHECK(x) do{ if(!(x)){ fails++; fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);} }while(0)

int main(void){
    size_t n=(size_t)NZ*NY*NX;
    uint8_t *vol=calloc(n,1), *mask=malloc(n);
    CHECK(vol&&mask);

    // a bright planar sheet 2 voxels thick at z=18..19 (the "papyrus"), 40 bg.
    for(int z=0;z<NZ;++z)for(int y=0;y<NY;++y)for(int x=0;x<NX;++x)
        vol[IDX(z,y,x)] = (z==18||z==19) ? 220 : 40;

    // ---- detector ----
    mc_seg_params P = { 1.0f, 2.0f, 0.5f, 80 };   // gate out the 40-value bg
    CHECK(mc_seg_detect(vol,NZ,NY,NX,&P,mask)==0);
    long on_sheet=0, off_sheet=0, sheet_tot=0, bg_tot=0;
    for(int z=0;z<NZ;++z)for(int y=0;y<NY;++y)for(int x=0;x<NX;++x){
        int is_sheet = (z>=18 && z<=19);     // the actual sheet voxels
        if(is_sheet){ sheet_tot++; if(mask[IDX(z,y,x)]) on_sheet++; }
        else { bg_tot++; if(mask[IDX(z,y,x)]) off_sheet++; }
    }
    // the sheet is detected (most of its voxels) with little background bleed.
    printf("detect: sheet %ld/%ld on, bg %ld/%ld false\n", on_sheet, sheet_tot, off_sheet, bg_tot);
    CHECK(on_sheet > sheet_tot*3/4);
    CHECK(off_sheet < bg_tot/10);

    // ---- detector on a TILTED sheet ----
    // The axis-aligned sheet above produces a diagonal structure tensor, which
    // only exercises eig3sym's degenerate (p1~0) branch. A 45-degree sheet in
    // the x-z plane (bright where x+z is near a constant) yields off-diagonal
    // tensor terms, driving the general closed-form (acos/cos) eigenvalue path.
    // It must still read as planar and be detected.
    memset(vol,0,n);
    for(int z=0;z<NZ;++z)for(int y=0;y<NY;++y)for(int x=0;x<NX;++x){
        int onsheet = (x+z==38 || x+z==39);   // a 2-thick diagonal slab
        vol[IDX(z,y,x)] = onsheet ? 220 : 40;
    }
    CHECK(mc_seg_detect(vol,NZ,NY,NX,&P,mask)==0);
    long ton=0, ttot=0, tfalse=0;
    for(int z=0;z<NZ;++z)for(int y=0;y<NY;++y)for(int x=0;x<NX;++x){
        int onsheet = (x+z==38 || x+z==39);
        if(onsheet){ ttot++; if(mask[IDX(z,y,x)]) ton++; }
        // count false positives well away from the diagonal band.
        else if(x+z<36 || x+z>41){ if(mask[IDX(z,y,x)]) tfalse++; }
    }
    printf("detect tilted: sheet %ld/%ld on, %ld far-field false\n", ton, ttot, tfalse);
    CHECK(ton > ttot/2);          // most of the tilted sheet detected
    CHECK(tfalse < ttot/4);       // little spurious detection off the band

    // ---- remove_small: add a 3-voxel speck, ensure it's removed (min 100) ----
    memset(mask,0,n);
    for(int x=10;x<30;++x)for(int y=10;y<30;++y) mask[IDX(18,y,x)]=255;   // a 400-vox sheet patch
    mask[IDX(0,0,0)]=255; mask[IDX(0,0,1)]=255; mask[IDX(0,1,0)]=255;     // 3-vox speck
    int kept = mc_seg_remove_small(mask,NZ,NY,NX,100);
    CHECK(kept==1);                       // only the big patch survives
    CHECK(mask[IDX(0,0,0)]==0);           // speck gone
    CHECK(mask[IDX(18,15,15)]==255);      // patch kept
    printf("remove_small: kept=%d, speck removed\n", kept);

    // ---- label: two separate blobs -> 2 components ----
    memset(mask,0,n);
    mask[IDX(5,5,5)]=255; mask[IDX(5,5,6)]=255;          // blob A
    mask[IDX(30,30,30)]=255;                              // blob B
    int32_t *lab=malloc(n*sizeof(int32_t));
    int nc=mc_seg_label(mask,NZ,NY,NX,lab);
    CHECK(nc==2);
    CHECK(lab[IDX(5,5,5)]==lab[IDX(5,5,6)]);             // same component
    CHECK(lab[IDX(5,5,5)]!=lab[IDX(30,30,30)]);
    printf("label: %d components\n", nc);

    // ---- fill_cavities: a hollow box -> interior fills ----
    memset(mask,0,n);
    for(int z=10;z<=20;++z)for(int y=10;y<=20;++y)for(int x=10;x<=20;++x){
        int onface = (z==10||z==20||y==10||y==20||x==10||x==20);
        if(onface) mask[IDX(z,y,x)]=255;                 // shell only
    }
    CHECK(mask[IDX(15,15,15)]==0);                        // hollow center
    mc_seg_fill_cavities(mask,NZ,NY,NX);
    CHECK(mask[IDX(15,15,15)]==255);                      // filled
    CHECK(mask[IDX(2,2,2)]==0);                           // outside untouched
    printf("fill_cavities: interior filled, exterior untouched\n");

    // ---- plug_holes: a sheet patch with a single missing voxel -> plugged ----
    memset(mask,0,n);
    for(int z=18;z<=19;++z)for(int y=10;y<20;++y)for(int x=10;x<20;++x) mask[IDX(z,y,x)]=255;
    mask[IDX(18,14,14)]=0;                                // poke a 1-voxel hole
    long added = mc_seg_plug_holes(mask,NZ,NY,NX,2);
    CHECK(added>0);
    CHECK(mask[IDX(18,14,14)]==255);                      // hole plugged
    printf("plug_holes: added=%ld, 1-voxel hole plugged\n", added);

    // ---- close: a small notch in a solid blob fills (radius 1; closing is for
    // compact regions, not thin sheets — those use plug_holes/patching). ----
    memset(mask,0,n);
    for(int z=10;z<=16;++z)for(int y=10;y<=16;++y)for(int x=10;x<=16;++x) mask[IDX(z,y,x)]=255;
    mask[IDX(13,13,13)]=0;                                 // interior notch
    mc_seg_close(mask,NZ,NY,NX,1);
    CHECK(mask[IDX(13,13,13)]==255);                       // notch closed
    printf("close: notch filled\n");

    // ---- degenerate dims: detect must reject (not OOB on the gradient stencil). ----
    CHECK(mc_seg_detect(vol,0,NY,NX,&P,mask)==-1);
    CHECK(mc_seg_detect(vol,NZ,0,NX,&P,mask)==-1);
    CHECK(mc_seg_detect(vol,NZ,NY,0,&P,mask)==-1);
    printf("detect rejects degenerate dims\n");

    // ---- EDT: exact Euclidean distance to nearest foreground voxel. Single
    // seed at the center -> each cell's EDT equals its grid distance to it. ----
    {
        memset(mask,0,n);
        int sz=NZ/2, sy=NY/2, sx=NX/2;
        mask[IDX(sz,sy,sx)]=255;
        float *d=malloc(n*sizeof(float));
        CHECK(mc_seg_edt(mask,NZ,NY,NX,d)==0);
        CHECK(d[IDX(sz,sy,sx)]==0.0f);
        double maxerr=0;
        for(int z=0;z<NZ;++z)for(int y=0;y<NY;++y)for(int x=0;x<NX;++x){
            double exp=sqrt((double)(z-sz)*(z-sz)+(double)(y-sy)*(y-sy)+(double)(x-sx)*(x-sx));
            double e=fabs(d[IDX(z,y,x)]-exp); if(e>maxerr)maxerr=e;
        }
        CHECK(maxerr<1e-4);
        // SDT: a solid z-slab -> inside negative, outside positive.
        memset(mask,0,n);
        for(int z=15;z<=20;++z)for(int y=0;y<NY;++y)for(int x=0;x<NX;++x) mask[IDX(z,y,x)]=255;
        CHECK(mc_seg_sdt(mask,NZ,NY,NX,d)==0);
        CHECK(d[IDX(17,20,20)] < 0);     // deep inside the slab
        CHECK(d[IDX(0,20,20)]  > 0);     // far outside
        printf("edt: max err %.2e vs brute; sdt inside<0 outside>0 OK\n", maxerr);
        free(d);
    }

    free(vol); free(mask); free(lab);
    printf(fails ? "mc_segment_test: %d FAILED\n" : "mc_segment_test: OK\n", fails);
    return fails?1:0;
}
