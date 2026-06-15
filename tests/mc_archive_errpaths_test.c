// mc_archive_errpaths_test — exercise archive/reader ERROR paths that the
// happy-path tests skip: dims-mismatched reopen (mc_archive_open rejection),
// reopen of a valid archive (the non-fresh load branch), mc_mca_probe on a
// missing/garbage file, and the mc_open untrusted-input gate (short/bad-magic).
// Deterministic, no fixtures.
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails=0;
#define CHECK(c,...) do{ if(!(c)){ fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
    fprintf(stderr,__VA_ARGS__); fputc('\n',stderr); fails++; } }while(0)

static mc_u8 blk_src(void *ud,int x,int y,int z){ (void)ud;
    return (x>32&&x<224&&y>32&&y<224&&z>32&&z<224)?150:0; }

int main(void){
    const char *p="/tmp/mc_errpaths.mca";
    remove(p);

    // 1) create a real 256^3 archive, append a chunk, close.
    mc_archive *a=mc_archive_open_dims(p,256,256,256,6.0f);
    CHECK(a!=NULL,"open_dims failed"); if(!a) return 1;
    uint8_t *vox=malloc((size_t)256*256*256);
    for(int z=0;z<256;++z)for(int y=0;y<256;++y)for(int x=0;x<256;++x)
        vox[((size_t)z*256+y)*256+x]=blk_src(NULL,x,y,z);
    mc_archive_append_chunk_par(a,0,0,0,0,vox,6.0f,0);
    free(vox);
    mc_archive_close(a);

    // 2) REOPEN with matching dims -> the non-fresh load branch (reads magic/ver/
    //    dims, validates, loads cursor + priors).
    mc_archive *a2=mc_archive_open_dims(p,256,256,256,6.0f);
    CHECK(a2!=NULL,"reopen (matching dims) failed");
    if(a2){ uint64_t co=mc_archive_chunk_offset(a2,0,0,0,0); CHECK(co>0,"reopened chunk absent"); mc_archive_close(a2); }

    // 3) REOPEN with MISMATCHED dims -> must be rejected (NULL), not corrupt the file.
    mc_archive *bad=mc_archive_open_dims(p,128,128,128,6.0f);
    CHECK(bad==NULL,"dims-mismatch reopen should return NULL");
    if(bad) mc_archive_close(bad);

    // 4) the file is still valid after the rejected open (reject must not truncate).
    mc_archive *a3=mc_archive_open_dims(p,256,256,256,6.0f);
    CHECK(a3!=NULL,"archive damaged by rejected mismatch open");
    if(a3) mc_archive_close(a3);

    // 4b) per-volume priors: set them, reopen, and confirm priors_load reads the
    //     populated blob back (the non-NULL branch) + mc_reader_priors exposes it.
    {
        mc_archive *ap=mc_archive_open_dims(p,256,256,256,6.0f);
        if(ap){
            static uint16_t plo[8][32],phi[8][32];
            for(int c=0;c<8;++c)for(int s=0;s<32;++s){ plo[c][s]=900; phi[c][s]=3100; }
            CHECK(mc_archive_set_priors(ap,plo,phi)==0,"set_priors failed");
            mc_archive_close(ap);
        }
        // reopen via flat reader -> priors_load with a valid prior blob.
        size_t ml=0; FILE*pf=fopen(p,"rb"); fseek(pf,0,SEEK_END); long pl=ftell(pf); fseek(pf,0,SEEK_SET);
        uint8_t *pm=malloc(pl); if(fread(pm,1,pl,pf)!=(size_t)pl){return 1;} fclose(pf);
        mc_reader *pr=mc_open(pm,pl);
        CHECK(pr!=NULL,"reopen after set_priors failed");
        if(pr){ const uint16_t *qlo=NULL,*qhi=NULL;
            int hp=mc_reader_priors(pr,&qlo,&qhi);
            CHECK(hp==1&&qlo&&qhi,"reader_priors should report stored priors (got %d)",hp);
            mc_close(pr); }
        free(pm);
    }

    // 5) mc_mca_probe: valid file reports dims; missing/garbage file fails cleanly.
    { int nx,ny,nz,nl; float q;
      CHECK(mc_mca_probe(p,&nx,&ny,&nz,&nl,&q)==0,"mca_probe valid failed");
      CHECK(nx==256&&ny==256&&nz==256,"mca_probe dims wrong: %d,%d,%d",nx,ny,nz);
      CHECK(mc_mca_probe("/tmp/does_not_exist_xyz.mca",&nx,&ny,&nz,&nl,&q)!=0,
            "mca_probe on missing file should fail"); }

    // 6) bad-magic / too-short file: mc_archive_open_dims on a garbage existing
    //    file must reject (magic check), and mc_open must reject short buffers.
    { const char *g="/tmp/mc_garbage.mca";
      FILE*f=fopen(g,"wb"); for(int i=0;i<300;++i) fputc(0xAB,f); fclose(f);  // >=header, bad magic
      mc_archive *ga=mc_archive_open_dims(g,256,256,256,6.0f);
      CHECK(ga==NULL,"open of bad-magic file should be rejected");
      if(ga) mc_archive_close(ga);
      remove(g); }

    uint8_t shortbuf[16]={0};
    CHECK(mc_open(shortbuf,16)==NULL,"mc_open(short) should be NULL");
    CHECK(mc_open(NULL,0)==NULL,"mc_open(NULL) should be NULL");

    remove(p);
    printf(fails?"mc_archive_errpaths: %d FAILURES\n":"mc_archive_errpaths: OK\n",fails);
    return fails?1:0;
}
