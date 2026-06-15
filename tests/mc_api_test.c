// mc_api_test — exercise the public-API surface not hit by the behavioural
// tests: reader/cache introspection getters, codec/quality extras, the
// colormap + image-filter helpers, the plane/quad/LOD sampler + parallel
// renderers, and the rate-target / priors archive paths. Pure offline,
// synthetic data — built to push line coverage of matter_compressor.c toward
// the public-API ceiling and to assert each helper's contract, not just call it.
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N 256
static int fails = 0;
#define CHECK(c, ...) do{ if(!(c)){ fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
    fprintf(stderr,__VA_ARGS__); fputc('\n',stderr); fails++; } }while(0)

// smooth material ball in air — the standard synthetic source.
static mc_u8 ball(void *ud,int x,int y,int z){ (void)ud;
    double cx=N/2.0,cy=N/2.0,cz=N/2.0,r=sqrt((x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz));
    if(r>N*0.4) return 0; double v=128+100*cos(r*0.15); return (mc_u8)(v<1?1:v>255?255:v); }

static uint8_t *decode_dense(mc_reader *r,int n){
    uint8_t *vol=calloc((size_t)n*n*n,1); mc_u8 blk[4096];
    int nch=(n+255)/256;
    for(int cz=0;cz<nch;++cz)for(int cy=0;cy<nch;++cy)for(int cx=0;cx<nch;++cx){
        uint64_t co=mc_chunk_offset(r,0,cz,cy,cx);
        for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
            mc_decode_block(r,co,bz,by,bx,blk);
            for(int z=0;z<16;++z)for(int y=0;y<16;++y)for(int x=0;x<16;++x){
                int gx=(cx*16+bx)*16+x,gy=(cy*16+by)*16+y,gz=(cz*16+bz)*16+z;
                if(gx<n&&gy<n&&gz<n) vol[((size_t)gz*n+gy)*n+gx]=blk[(z*16+y)*16+x];
            }}}
    return vol;
}

int main(void){
    // ---- preset_quality ----
    float q3=mc_preset_quality(MC_PRESET_BALANCED);
    CHECK(q3>0, "preset_quality MC_PRESET_BALANCED=%g should be >0", q3);

    // ---- block codec round-trip with a MAX-ERROR bound (tau) ----
    // Exercises mc_enc_block's sparse-correction path (uncovered by the plain
    // quality-only builds) + the preset ladder + ctx getters/setters.
    {
        mc_u8 vox[16*16*16];
        for(int z=0;z<16;++z)for(int y=0;y<16;++y)for(int x=0;x<16;++x)
            vox[(z*16+y)*16+x]=(mc_u8)(40+((x*7+y*3+z*5)%160));   // all-material block
        mc_codec_ctx *cx=mc_codec_ctx_new();
        int tau=3;
        mc_codec_ctx_set_quality(cx,2.0f);
        mc_codec_ctx_set_max_error(cx,tau);
        CHECK(mc_codec_ctx_get_max_error(cx)==tau,"get_max_error != set");
        CHECK(mc_codec_ctx_get_quality(cx)==2.0f,"get_quality != set");
        mc_buf buf={0}; uint32_t plen=0;
        int coded=mc_enc_block(cx,vox,&buf,&plen);
        CHECK(coded==1&&plen>0,"enc_block coded=%d plen=%u",coded,plen);
        mc_u8 dec[16*16*16];
        mc_dec_block(cx,buf.p,plen,dec);
        int maxe=0; for(int i=0;i<4096;++i){ int e=abs((int)dec[i]-(int)vox[i]); if(e>maxe)maxe=e; }
        CHECK(maxe<=tau,"max-error bound violated: maxerr=%d > tau=%d",maxe,tau);
        // preset ladder: every level applies without crashing and returns a q.
        for(int lvl=MC_PRESET_ARCHIVAL;lvl<=MC_PRESET_PREVIEW;++lvl){
            float pq=mc_apply_preset(cx,(mc_preset)lvl);
            CHECK(pq>0,"apply_preset(%d) q=%g",lvl,pq);
        }
        free(buf.p);
        mc_codec_ctx_free(cx);
    }

    // ---- build an in-RAM archive ----
    mc_build_opts o={.dim=N,.quality=6.0f};
    size_t len=0; uint8_t *arc=mc_build(ball,NULL,&o,&len);
    CHECK(arc!=NULL,"mc_build failed");
    if(!arc) return 1;

    // ---- reader introspection ----
    mc_reader *r=mc_open(arc,len);
    int nx,ny,nz; mc_reader_dims(r,&nx,&ny,&nz);
    CHECK(nx==N&&ny==N&&nz==N,"reader_dims %d,%d,%d != %d",nx,ny,nz,N);
    CHECK(mc_reader_nlods(r)>=1,"reader_nlods<1");
    CHECK(mc_reader_quality(r)>0,"reader_quality not positive");

    uint64_t co=mc_chunk_offset(r,0,0,0,0);
    int cerr=-99; uint64_t co2=mc_chunk_offset_chk(r,0,0,0,0,&cerr);
    CHECK(co2==co&&cerr==0,"chunk_offset_chk mismatch err=%d",cerr);
    // Flat (mmap) reader: an absent/out-of-range chunk resolves to offset 0
    // ("absent") with no I/O error — err is only set on the streaming path.
    cerr=-99; uint64_t coob=mc_chunk_offset_chk(r,0,999,999,999,&cerr);
    CHECK(coob==0&&cerr==0,"flat chunk_offset_chk OOB: off=%llu err=%d (want 0,0)",
          (unsigned long long)coob,cerr);
    // An invalid lod must resolve to 0 (no crash).
    CHECK(mc_chunk_offset_chk(r,99,0,0,0,&cerr)==0,"chunk_offset_chk bad lod !=0");

    uint64_t blen=mc_reader_chunk_blob_len(r,co);
    CHECK(blen>0,"chunk_blob_len==0");
    uint8_t *blob=malloc(blen);
    int rb=mc_reader_read_blob(r,co,blen,blob);
    CHECK(rb==0,"reader_read_blob rc=%d",rb);
    free(blob);

    const uint16_t *plo=NULL,*phi=NULL;
    mc_reader_priors(r,&plo,&phi);   // may be NULL (no per-volume priors stored) — just exercise

    // ---- deblock the decoded volume (decode-side filter) ----
    uint8_t *dec=decode_dense(r,N);
    uint8_t *dvol=malloc((size_t)N*N*N); memcpy(dvol,dec,(size_t)N*N*N);
    mc_deblock(dvol,N,N,N,6.0f);
    // deblock must preserve air (0 stays 0) and not explode values.
    long air_leak=0; for(size_t i=0;i<(size_t)N*N*N;++i) if(dec[i]==0&&dvol[i]!=0) air_leak++;
    CHECK(air_leak==0,"deblock leaked %ld air voxels",air_leak);
    free(dvol);

    // ---- colormap helpers ----
    CHECK(mc_colormap_id("gray")==0,"colormap_id gray!=0");
    int vir=mc_colormap_id("viridis"); CHECK(vir>0,"colormap_id viridis<=0");
    uint32_t lut[256]; mc_colormap_lut(lut,0.0f,1.0f,vir);
    CHECK(lut[0]!=lut[255],"colormap lut endpoints identical");
    uint32_t argb[64*64]; uint8_t gimg[64*64];
    for(int i=0;i<64*64;++i) gimg[i]=(uint8_t)(i&255);
    mc_colormap_apply(gimg,64,64,lut,argb,64);
    CHECK((argb[0]>>24)!=0 || (argb[0]&0xFFFFFF)!=0 || lut[0]==argb[0],"colormap_apply produced nothing");

    // ---- image DoG band-pass ----
    uint8_t dimg[64*64]; memcpy(dimg,gimg,sizeof dimg);
    mc_image_dog(dimg,64,64,2.0f,1.0f);   // must run without OOB (ASan covers correctness)

    // ---- dense sampler: sample_points / sample_points_u8 ----
    mc_sample_src dsrc=mc_sample_src_dense(dec,N,N,N);
    mc_sampler *s=mc_sampler_new(&dsrc);
    float pts[3*4]={N/2.f,N/2.f,N/2.f,  0,0,0,  -1,0,0,  N/2.f,N/2.f,N/2.f};
    float fout[4]; uint8_t u8out[4];
    mc_sample_points(s,pts,4,MC_FILTER_TRILINEAR,fout);
    mc_sample_points_u8(s,pts,4,MC_FILTER_TRILINEAR,u8out);
    CHECK(fout[0]>0,"center sample should be material (>0), got %g",fout[0]);
    CHECK(fout[2]==0.0f,"negative-coord sample must be 0 (invalid marker), got %g",fout[2]);
    CHECK(u8out[0]==(uint8_t)(fout[0]+0.5f)||abs((int)u8out[0]-(int)fout[0])<=1,
          "sample_points_u8 disagrees with sample_points: %u vs %g",u8out[0],fout[0]);

    // ---- plane_gen: produce a point grid, render it serial + parallel ----
    mc_plane pl={.origin={N/2.f,N/2.f,N/2.f},.normal={1,0,0}}; mc_plane_basis(&pl);
    int W=64,H=64; float *gpts=malloc(sizeof(float)*3*W*H),*gnrm=malloc(sizeof(float)*3*W*H);
    mc_plane_gen(&pl,W,H,1.0f,gpts,gnrm);
    mc_render_params rp={.filter=MC_FILTER_TRILINEAR,.comp=MC_COMP_MAX,.t0=-4,.t1=4,.dt=1};
    uint8_t *im1=calloc(W*H,1),*im2=calloc(W*H,1);
    mc_render_points(s,gpts,gnrm,W,H,&rp,im1);
    mc_render_points_par(&dsrc,gpts,gnrm,W,H,&rp,im2,4);
    long diff=0; for(int i=0;i<W*H;++i) diff+=abs((int)im1[i]-(int)im2[i]);
    CHECK(diff==0,"render_points_par != serial render_points (sum|diff|=%ld)",diff);
    long nzpix=0; for(int i=0;i<W*H;++i) if(im2[i]) nzpix++;
    CHECK(nzpix>0,"plane render produced an all-zero image");

    // dense source with NON-16-multiple dims -> the partial-edge block accessor
    // (dense_block memset+partial-copy path) when a sampled block overruns the
    // volume edge. Sample across the whole volume incl. the ragged edge.
    {
        int D=100; uint8_t *odd=malloc((size_t)D*D*D);
        for(int z=0;z<D;++z)for(int y=0;y<D;++y)for(int x=0;x<D;++x)
            odd[((size_t)z*D+y)*D+x]=(uint8_t)(50+((x+y+z)%180));
        mc_sample_src os=mc_sample_src_dense(odd,D,D,D);
        mc_sampler *osmp=mc_sampler_new(&os);
        // sample near the ragged edge (block at x=96 overruns 100) both filters.
        float v_edge=mc_sample_point(osmp,D-2.0f,D-2.0f,D-2.0f,MC_FILTER_TRILINEAR);
        float v_nn  =mc_sample_point(osmp,(float)(D-1),(float)(D-1),(float)(D-1),MC_FILTER_NEAREST);
        CHECK(v_edge>=0&&v_nn>=0,"odd-dim edge sample failed");
        // a batch across the volume to exercise the block cache + partial blocks.
        float bp[3*8]; for(int i=0;i<8;++i){ bp[3*i]=i*14.0f; bp[3*i+1]=i*13.0f; bp[3*i+2]=i*12.0f; }
        float bo[8]; mc_sample_points(osmp,bp,8,MC_FILTER_TRILINEAR,bo);
        mc_sampler_free(osmp); free(odd);
    }

    // single-sample render path (t0==t1 -> one tap, no march loop) + a NON-unit
    // normal so render_pixel renormalizes it. Covers the scalar early-outs.
    {
        mc_render_params one={.filter=MC_FILTER_TRILINEAR,.comp=MC_COMP_MAX,.t0=0,.t1=0,.dt=1};
        float *n2=malloc(sizeof(float)*3*W*H);
        for(int i=0;i<W*H;++i){ n2[3*i]=2.0f; n2[3*i+1]=0; n2[3*i+2]=0; }  // |n|=2, not unit
        uint8_t *oim=calloc(W*H,1);
        mc_render_points(s,gpts,n2,W,H,&one,oim);
        long onz=0; for(int i=0;i<W*H;++i) if(oim[i]) onz++;
        CHECK(onz>0,"single-sample render all-zero");
        free(n2); free(oim);
    }

    // Composite SIMD 4-wide path: nsteps>=4 + TRILINEAR fills the vectorized
    // ray loop in render_pixel. Drive MIN/MEAN/STDDEV (the render test covers
    // these comps only at small step counts -> scalar tail). t0..t1/dt = 12 steps.
    for(int ci=0; ci<3; ++ci){
        mc_comp comps[3]={MC_COMP_MIN,MC_COMP_MEAN,MC_COMP_STDDEV};
        mc_render_params wp={.filter=MC_FILTER_TRILINEAR,.comp=comps[ci],
                             .t0=-6,.t1=6,.dt=1};
        uint8_t *wim=calloc(W*H,1);
        mc_render_points(s,gpts,gnrm,W,H,&wp,wim);   // gpts/gnrm still valid here
        free(wim);
    }

    // MC_COMP_INK: the papyrus ink-detection composite (transmission + cone SSS).
    // Shares the SHADED emission-absorption march; exercise it explicitly (the
    // render test covers the other 9 comp modes but not INK).
    float *ipts=malloc(sizeof(float)*3*W*H),*inrm=malloc(sizeof(float)*3*W*H);
    mc_plane_gen(&pl,W,H,1.0f,ipts,inrm);
    mc_render_params ink={.filter=MC_FILTER_TRILINEAR,.comp=MC_COMP_INK,
                          .t0=-8,.t1=8,.dt=1,.alpha_min=0.1f,.alpha_opacity=0.8f};
    uint8_t *iim=calloc(W*H,1);
    mc_render_points(s,ipts,inrm,W,H,&ink,iim);  // must run + not OOB (ASan)
    free(ipts);free(inrm);free(iim);
    free(gpts);free(gnrm);free(im1);free(im2);

    // ---- LOD pyramid: sample_lods + lod_sampler + lod render ----
    int M=N/2; uint8_t *l1=malloc((size_t)M*M*M);
    for(int z=0;z<M;++z)for(int y=0;y<M;++y)for(int x=0;x<M;++x){
        // box-downsample LOD0
        int s8=0; for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)
            s8+=dec[(((size_t)(2*z+dz))*N+(2*y+dy))*N+(2*x+dx)];
        l1[((size_t)z*M+y)*M+x]=(uint8_t)(s8/8);
    }
    mc_sample_lods ls={0}; ls.nlods=2;
    ls.lods[0]=mc_sample_src_dense(dec,N,N,N);
    ls.lods[1]=mc_sample_src_dense(l1,M,M,M);
    mc_lod_sampler *lsamp=mc_lod_sampler_new(&ls);
    float v0=mc_lod_sample(lsamp,0,0,N/2.f,N/2.f,N/2.f,MC_FILTER_TRILINEAR);
    float v1=mc_lod_sample(lsamp,1,1,N/2.f,N/2.f,N/2.f,MC_FILTER_TRILINEAR);
    CHECK(v0>0&&v1>0,"lod_sample center should be material: L0=%g L1=%g",v0,v1);
    mc_lod_sampler_reset(lsamp);
    mc_lod_sampler_free(lsamp);

    // quad LOD render
    int gw=4,gh=4; float qgrid[3*4*4];
    for(int j=0;j<gh;++j)for(int i=0;i<gw;++i){
        float *p=&qgrid[3*(j*gw+i)];
        p[0]=N/2.f; p[1]=N/2.f+(j-gh/2)*8; p[2]=N/2.f+(i-gw/2)*8;
    }
    mc_quad qd={.grid=qgrid,.gw=gw,.gh=gh};
    uint8_t *qim=calloc(W*H,1);
    int qr=mc_render_quad_lod(&ls,&qd,0,0,1.0f,W,H,&rp,qim,2);
    CHECK(qr==0||qr==1,"render_quad_lod rc=%d unexpected",qr);
    free(qim);

    // mc_render_points_par_lod: LOD-fallback parallel point render. Points are
    // NATIVE level-0 coords through the center plane; must paint material.
    float *lpts=malloc(sizeof(float)*3*W*H);
    for(int j=0;j<H;++j)for(int i=0;i<W;++i){ float *p=&lpts[3*(j*W+i)];
        p[0]=N/2.f; p[1]=(float)(N/2 + (j-H/2)); p[2]=(float)(N/2 + (i-W/2)); }
    uint8_t *lim=calloc(W*H,1);
    mc_render_points_par_lod(&ls,0,lpts,W,H,&rp,lim,2);
    long lnz=0; for(int i=0;i<W*H;++i) if(lim[i]) lnz++;
    CHECK(lnz>0,"render_points_par_lod produced an all-zero image");
    free(lpts); free(lim); free(l1);

    // ---- cache introspection ----
    mc_archive *a=mc_archive_open_dims("/tmp/mc_api.mca",N,N,N,6.0f);
    CHECK(a!=NULL,"archive_open_dims failed");
    if(a){
        // rate-target append (uses a 1/16 sample to pick q) + priors set.
        uint8_t *vox=malloc((size_t)N*N*N);
        for(int z=0;z<N;++z)for(int y=0;y<N;++y)for(int x=0;x<N;++x)
            vox[((size_t)z*N+y)*N+x]=ball(NULL,x,y,z);
        float qout=0;
        int tr=mc_archive_append_chunk_target(a,0,0,0,0,vox,30.0f,&qout);
        CHECK(tr==0,"append_chunk_target rc=%d",tr);
        CHECK(qout>0,"append_chunk_target qout=%g not set",qout);
        free(vox);

        // reserve_index: pre-grow the node tree for a (future) chunk slot.
        int ri=mc_archive_reserve_index(a,0,1,1,1);
        CHECK(ri==0||ri>0,"reserve_index rc=%d",ri);

        // set_priors: stamp per-volume context priors (valid in-range tables).
        static uint16_t plo[8][32], phi[8][32];
        for(int ci=0;ci<8;++ci)for(int s=0;s<32;++s){ plo[ci][s]=1<<11; phi[ci][s]=1<<11; }
        int sp=mc_archive_set_priors(a,plo,phi);
        CHECK(sp==0,"set_priors rc=%d",sp);

        mc_cache *c=mc_cache_new_archive((size_t)64<<20,a);
        CHECK(c!=NULL,"cache_new_archive failed");
        if(c){
            size_t cap=mc_cache_capacity_bytes(c);
            CHECK(cap>0,"cache capacity 0");
            // touch a block so something becomes resident
            (void)mc_cache_get(c,0,8,8,8);
            size_t used=mc_cache_used_bytes(c);
            double frac=mc_cache_usage_fraction(c);
            CHECK(frac>=0.0&&frac<=1.0,"usage_fraction out of range: %g",frac);
            CHECK((double)used<=(double)cap,"used %zu > cap %zu",used,cap);
            size_t newcap=mc_cache_resize(c,(size_t)128<<20);
            CHECK(newcap>0,"cache_resize returned 0");
            mc_cache_free(c);
        }
        mc_archive_close(a);
    }

    // ---- cache over a reader (mc_cache_new_reader) ----
    mc_cache *cr=mc_cache_new_reader((size_t)32<<20,r);
    CHECK(cr!=NULL,"cache_new_reader failed");
    if(cr){ const uint8_t *p=mc_cache_get(cr,0,8,8,8); CHECK(p!=NULL,"cache_new_reader get NULL"); mc_cache_free(cr); }

    mc_sampler_free(s);
    mc_close(r); free(arc); free(dec);
    remove("/tmp/mc_api.mca");

    printf(fails? "mc_api_test: %d FAILURES\n":"mc_api_test: OK\n", fails);
    return fails?1:0;
}
