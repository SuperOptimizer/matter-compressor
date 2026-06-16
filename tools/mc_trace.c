/* mc_trace — classical (no-ML) surface tracer over a volume chunk.
 *
 * The full pipeline in one tool: pull an axis-aligned chunk from a volume
 * (mc_volume + mc_sample_box), detect the papyrus sheet (mc_segment structure-
 * tensor), build its signed-distance field (mc_seg_sdt), auto-seed a patch on
 * the sheet, and grow a parametric surface that follows it (mc_trace_grow with
 * the SDF volume term). Writes the result as a .grid surface (4x-f32 TIFF).
 *
 *   mc_trace <volume-url> <cache_dir> --at z y x --size d h w [opts]
 * opts:
 *   --out PATH         output .grid (default traced.grid)
 *   --grid GW GH       traced grid dims in cells       (default 64 64)
 *   --step S           grid cell spacing in voxels     (default 8)
 *   --depth D          surface half-thickness          (default 5)
 *   --sheetness S      detector planarity threshold    (default 0.5)
 *   --lo N             detector intensity gate         (default 90)
 *   --vol-weight W     SDF term weight in growth        (default 4.0)
 *   --gens N           max growth generations          (default 60)
 *   --seed x y z       seed hint (chunk-local x,y,z); default = chunk center
 */
#include "matter_compressor.h"
#include "mc_segment.h"
#include "mc_surface.h"
#include "mc_trace.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv){
    const char *url=NULL,*cache=NULL,*out="traced.grid";
    int oz=0,oy=0,ox=0, d=64,h=256,w=256;
    int gw=64, gh=64, gens=60, lo=90;
    double step=8.0, depth=5.0, vol_w=4.0; float sheetness=0.5f;
    double seed[3]={-1,-1,-1};
    int npos=0;
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--at")&&i+3<argc){ oz=atoi(argv[++i]);oy=atoi(argv[++i]);ox=atoi(argv[++i]); }
        else if(!strcmp(argv[i],"--size")&&i+3<argc){ d=atoi(argv[++i]);h=atoi(argv[++i]);w=atoi(argv[++i]); }
        else if(!strcmp(argv[i],"--out")&&i+1<argc) out=argv[++i];
        else if(!strcmp(argv[i],"--grid")&&i+2<argc){ gw=atoi(argv[++i]);gh=atoi(argv[++i]); }
        else if(!strcmp(argv[i],"--step")&&i+1<argc) step=atof(argv[++i]);
        else if(!strcmp(argv[i],"--depth")&&i+1<argc) depth=atof(argv[++i]);
        else if(!strcmp(argv[i],"--sheetness")&&i+1<argc) sheetness=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"--lo")&&i+1<argc) lo=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--vol-weight")&&i+1<argc) vol_w=atof(argv[++i]);
        else if(!strcmp(argv[i],"--gens")&&i+1<argc) gens=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--seed")&&i+3<argc){ seed[0]=atof(argv[++i]);seed[1]=atof(argv[++i]);seed[2]=atof(argv[++i]); }
        else if(npos==0){ url=argv[i]; npos++; }
        else if(npos==1){ cache=argv[i]; npos++; }
    }
    if(!url||!cache){
        fprintf(stderr,"usage: %s <volume-url> <cache_dir> --at z y x --size d h w [opts]\n",argv[0]);
        return 2;
    }

    mc_volume *v=mc_volume_open(url,cache,(size_t)1<<30,6.0f);
    if(!v){ fprintf(stderr,"mc_volume_open(%s) failed\n",url); return 1; }
    mc_sample_src src=mc_volume_sample_src(v,0,1);
    size_t n=(size_t)d*h*w;
    uint8_t *chunk=malloc(n), *mask=malloc(n);
    float   *sdf=malloc(n*sizeof(float));
    if(!chunk||!mask||!sdf){ mc_volume_free(v); return 1; }
    float origin[3]={(float)oz,(float)oy,(float)ox};
    float du[3]={0,0,1}, dv[3]={0,1,0}, dw[3]={1,0,0};
    if(mc_sample_box(&src,origin,du,dv,dw,w,h,d,MC_FILTER_TRILINEAR,chunk,0)!=0){
        fprintf(stderr,"mc_sample_box failed\n"); return 1; }

    // detect the sheet + clean up, then signed-distance field.
    mc_seg_params P={1.0f,2.0f,sheetness,lo};
    if(mc_seg_detect(chunk,d,h,w,&P,mask)!=0){ fprintf(stderr,"detect failed\n"); return 1; }
    mc_seg_remove_small(mask,d,h,w,500);
    long det=0; for(size_t i=0;i<n;++i) if(mask[i]) det++;
    fprintf(stderr,"detected %ld sheet voxels (%.1f%%)\n", det, 100.0*det/n);
    if(det==0){ fprintf(stderr,"no sheet found; aborting\n"); return 1; }
    if(mc_seg_sdt(mask,d,h,w,sdf)!=0){ fprintf(stderr,"sdt failed\n"); return 1; }

    // trace: seed on the sheet, grow following the SDF.
    mc_sdf_field F={ sdf, d, h, w };       // (z,y,x) = (d,h,w)
    mc_surf_grid *g=mc_surf_grid_create(gw,gh,step);
    if(!g){ return 1; }
    if(mc_trace_seed_from_sdf(g,&F,seed,3)!=0){ fprintf(stderr,"seed failed (no on-sheet spot near hint)\n"); return 1; }
    mc_grow_opts o; mc_grow_opts_default(&o);
    o.max_gen=gens; o.accept_resid=0; o.vol_fn=mc_trace_sdf_vol; o.vol_user=&F; o.vol_weight=vol_w;
    mc_grow_report rep;
    if(mc_trace_grow(g,&o,&rep)!=0){ fprintf(stderr,"grow failed\n"); return 1; }
    fprintf(stderr,"traced: %d cells over %d generations (%d rejected)\n",
            rep.added, rep.generations, rep.rejected);

    // export. surface coords are chunk-LOCAL (x,y,z); add the chunk origin so
    // the .grid is in volume coords (chunk axes were x=du,y=dv,z=dw -> add ox,oy,oz).
    mc_surface s;
    if(mc_trace_to_surface(g,depth,&s)!=0){ fprintf(stderr,"export failed\n"); return 1; }
    for(int i=0;i<s.gw*s.gh;++i){ if(s.grid[i*3]<0) continue;
        s.grid[i*3]   += oz;    // z
        s.grid[i*3+1] += oy;    // y
        s.grid[i*3+2] += ox;    // x
    }
    if(mc_surface_save_tiff(out,&s)!=0){ fprintf(stderr,"write %s failed\n",out); return 1; }
    fprintf(stderr,"wrote %s (%dx%d grid, mean depth %.1f)\n", out, s.gw, s.gh, s.mean_depth);

    mc_surface_free(&s); mc_surf_grid_free(g);
    free(chunk); free(mask); free(sdf); mc_volume_free(v);
    return 0;
}
