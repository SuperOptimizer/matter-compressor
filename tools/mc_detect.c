/* mc_detect — classical surface (sheet) detection over a volume chunk.
 *
 * Pulls an arbitrary axis-aligned chunk from a volume (the existing 3D pipeline:
 * mc_volume + mc_sample_box — no special chunk API needed), runs the
 * structure-tensor sheet detector + topology post-processing, and reports the
 * resulting binary surface mask. A PPM of a chosen mask slice can be dumped.
 *
 *   mc_detect <volume-url> <cache_dir> --at z y x --size d h w [opts]
 * opts:
 *   --sheetness S     min planarness [0,1]            (0.5)
 *   --lo N            intensity gate (ignore < N)      (1)
 *   --min-vox N       drop components < N voxels       (2000)
 *   --no-post         skip topology post-processing
 *   --dump-slice K out.ppm   write mask slice K (in d) as PPM
 */
#include "matter_compressor.h"
#include "mc_segment.h"
#include "mc_surface.h"          /* mc_ppm_write */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv){
    const char *url=NULL,*cache=NULL,*dump=NULL;
    int oz=0,oy=0,ox=0, d=64,h=128,w=128;
    int min_vox=2000, post=1, dump_k=-1, lo=1;
    float sheetness=0.5f;
    int npos=0;
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--at")&&i+3<argc){ oz=atoi(argv[++i]); oy=atoi(argv[++i]); ox=atoi(argv[++i]); }
        else if(!strcmp(argv[i],"--size")&&i+3<argc){ d=atoi(argv[++i]); h=atoi(argv[++i]); w=atoi(argv[++i]); }
        else if(!strcmp(argv[i],"--sheetness")&&i+1<argc) sheetness=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"--lo")&&i+1<argc) lo=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--min-vox")&&i+1<argc) min_vox=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--no-post")) post=0;
        else if(!strcmp(argv[i],"--dump-slice")&&i+2<argc){ dump_k=atoi(argv[++i]); dump=argv[++i]; }
        else if(npos==0){ url=argv[i]; npos++; }
        else if(npos==1){ cache=argv[i]; npos++; }
    }
    if(!url||!cache){
        fprintf(stderr,"usage: %s <volume-url> <cache_dir> --at z y x --size d h w\n"
                       "  [--sheetness S] [--lo N] [--min-vox N] [--no-post]\n"
                       "  [--dump-slice K out.ppm]\n", argv[0]);
        return 2;
    }

    mc_volume *v = mc_volume_open(url, cache, (size_t)1<<30, 6.0f);
    if(!v){ fprintf(stderr,"mc_volume_open(%s) failed\n", url); return 1; }

    // extract the chunk via the oriented-box sampler (axis-aligned, blocking so
    // the data is present for a one-shot run).
    mc_sample_src src = mc_volume_sample_src(v, 0, /*blocking=*/1);
    size_t n=(size_t)d*h*w;
    uint8_t *chunk=malloc(n), *mask=malloc(n);
    if(!chunk||!mask){ mc_volume_free(v); return 1; }
    float origin[3]={(float)oz,(float)oy,(float)ox};
    float du[3]={0,0,1}, dv[3]={0,1,0}, dw[3]={1,0,0};   // x,y,z unit axes
    if(mc_sample_box(&src, origin, du,dv,dw, w,h,d, MC_FILTER_TRILINEAR, chunk, 0)!=0){
        fprintf(stderr,"mc_sample_box failed\n"); return 1; }
    long cnz=0; for(size_t i=0;i<n;++i) if(chunk[i]) cnz++;
    fprintf(stderr,"chunk %dx%dx%d at (z%d,y%d,x%d): %ld/%zu nonzero\n", d,h,w, oz,oy,ox, cnz, n);

    // detect.
    mc_seg_params P = { 1.0f, 2.0f, sheetness, lo };
    if(mc_seg_detect(chunk, d,h,w, &P, mask)!=0){ fprintf(stderr,"detect failed\n"); return 1; }
    long det=0; for(size_t i=0;i<n;++i) if(mask[i]) det++;
    fprintf(stderr,"detected sheet voxels: %ld (%.1f%%)\n", det, 100.0*det/n);

    // topology post-processing.
    if(post){
        int kept = mc_seg_remove_small(mask, d,h,w, min_vox);
        long plugged = mc_seg_plug_holes(mask, d,h,w, 2);
        mc_seg_close(mask, d,h,w, 1);
        mc_seg_fill_cavities(mask, d,h,w);
        long after=0; for(size_t i=0;i<n;++i) if(mask[i]) after++;
        int32_t *lab=malloc(n*sizeof(int32_t));
        int nc = lab ? mc_seg_label(mask,d,h,w,lab) : -1;
        free(lab);
        fprintf(stderr,"post: components kept=%d, holes plugged=%ld, final voxels=%ld, instances=%d\n",
                kept, plugged, after, nc);
    }

    // optional PPM of one mask slice.
    if(dump && dump_k>=0 && dump_k<d){
        uint8_t *slice=malloc((size_t)w*h);
        memcpy(slice, mask + (size_t)dump_k*w*h, (size_t)w*h);
        if(mc_ppm_write(dump, w, h, 1, slice)==0)
            fprintf(stderr,"wrote mask slice %d -> %s\n", dump_k, dump);
        free(slice);
    }

    free(chunk); free(mask); mc_volume_free(v);
    return 0;
}
