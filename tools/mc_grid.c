/* mc_grid — convert surfaces to/from the .grid format.
 *
 * The .grid format is matter-compressor's grid-native parametric surface: a
 * 2D TIFF, gw x gh, 4x f32 per texel = (x, y, z, depth) in volume coords
 * (depth = half-thickness along the normal). It is the canonical form the
 * renderer + viewer consume.
 *
 *   mc_grid <in> <out.grid>            # convert any supported input -> .grid
 *   mc_grid <in.grid> <out.obj>        # .grid -> OBJ mesh
 *   mc_grid <in.grid> <out.ppm-or-vcps># (.grid -> VC per-pixel map)  [--vcps]
 *
 * Input is detected by extension / content:
 *   .grid / .tif / .tiff  -> 4x-f32 grid TIFF
 *   .ppm                  -> VC ordered per-pixel map (xyz[+normal] doubles)
 *   .obj                  -> Wavefront mesh (resampled to a grid)
 * Options:
 *   --depth D     default half-thickness for inputs lacking one (default 4)
 *   --grid W H    target grid size for OBJ resampling (default: auto)
 */
#include "matter_compressor.h"
#include "mc_surface.h"
#include "mc_tiff.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ends_with(const char *s, const char *suf){
    size_t ls=strlen(s), lf=strlen(suf);
    return ls>=lf && strcasecmp(s+ls-lf,suf)==0;
}

// VC-PPM has a text header starting "width:"; sniff it.
static int is_vcps_ppm(const char *path){
    FILE *f=fopen(path,"rb"); if(!f) return 0;
    char b[16]={0}; size_t n=fread(b,1,8,f); fclose(f);
    return n>=6 && strncmp(b,"width:",6)==0;
}

// write a surface as a VC ordered per-pixel map (dim=3, xyz doubles).
static int save_vcps(const char *path, const mc_surface *s){
    FILE *f=fopen(path,"wb"); if(!f) return -1;
    fprintf(f,"width: %d\nheight: %d\ndim: 3\nordered: true\ntype: double\nversion: 1\n<>\n", s->gw, s->gh);
    for(size_t i=0;i<(size_t)s->gw*s->gh;++i){
        const float *p=&s->grid[i*3];       // (z,y,x)
        double row[3];
        if(p[0]<0&&p[1]<0&&p[2]<0){ row[0]=row[1]=row[2]=0; }      // invalid -> 0
        else { row[0]=p[2]; row[1]=p[1]; row[2]=p[0]; }            // x,y,z
        fwrite(row,sizeof(double),3,f);
    }
    fclose(f); return 0;
}

int main(int argc, char **argv){
    const char *in=NULL,*out=NULL; float depth=4.0f; int gw=0,gh=0; int as_vcps=0;
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--depth")&&i+1<argc) depth=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"--grid")&&i+2<argc){ gw=atoi(argv[++i]); gh=atoi(argv[++i]); }
        else if(!strcmp(argv[i],"--vcps")) as_vcps=1;
        else if(!in) in=argv[i];
        else if(!out) out=argv[i];
    }
    if(!in||!out){
        fprintf(stderr,"usage: %s <in> <out> [--depth D] [--grid W H] [--vcps]\n"
                       "  in: .grid/.tif | .ppm (VC map) | .obj (mesh)\n"
                       "  out: .grid | .obj | .ppm(--vcps)\n", argv[0]);
        return 2;
    }

    // --- load input -> mc_surface ---
    mc_surface s; memset(&s,0,sizeof s); int rc;
    if(ends_with(in,".obj")){
        mc_mesh m;
        if(mc_mesh_load_obj(in,&m)!=0){ fprintf(stderr,"load obj failed: %s\n",in); return 1; }
        fprintf(stderr,"obj: %d verts, %d tris -> resampling to grid\n", m.nv, m.nt);
        rc=mc_grid_from_mesh(&m,gw,gh,depth,&s);
        mc_mesh_free(&m);
        if(rc!=0){ fprintf(stderr,"mesh->grid failed (%d)\n",rc); return 1; }
    } else if(ends_with(in,".ppm") || is_vcps_ppm(in)){
        rc=mc_surface_load_vcps_ppm(in,&s,depth);
        if(rc!=0){ fprintf(stderr,"load VC ppm failed (%d): %s\n",rc,in); return 1; }
    } else { // .grid / .tif / .tiff
        rc=mc_surface_load_tiff(in,&s);
        if(rc!=0){ fprintf(stderr,"load grid tiff failed (%d): %s\n",rc,in); return 1; }
    }
    // count valid points.
    long valid=0; for(size_t i=0;i<(size_t)s.gw*s.gh;++i){ const float*p=&s.grid[i*3]; if(!(p[0]<0&&p[1]<0&&p[2]<0)) valid++; }
    fprintf(stderr,"surface %dx%d, %ld valid points, mean depth %.2f\n", s.gw, s.gh, valid, s.mean_depth);

    // --- save output ---
    if(ends_with(out,".obj"))            rc=mc_surface_save_obj(out,&s);
    else if(as_vcps||ends_with(out,".ppm")) rc=save_vcps(out,&s);
    else                                 rc=mc_surface_save_tiff(out,&s);   // .grid
    if(rc!=0){ fprintf(stderr,"write %s failed (%d)\n",out,rc); mc_surface_free(&s); return 1; }
    fprintf(stderr,"wrote %s\n", out);

    mc_surface_free(&s);
    return 0;
}
