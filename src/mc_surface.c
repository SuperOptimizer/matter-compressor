/* mc_surface.c — parametric surface I/O (see mc_surface.h). */
#include "mc_surface.h"
#include "mc_tiff.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void mc_surface_free(mc_surface *s){
    if(!s) return;
    free(s->grid); free(s->depth);
    memset(s,0,sizeof *s);
}

mc_quad mc_surface_quad(const mc_surface *s){
    mc_quad q = { s->grid, s->gw, s->gh };
    return q;
}

int mc_surface_load_tiff(const char *path, mc_surface *s){
    memset(s,0,sizeof *s);
    mc_tiff t;
    if(mc_tiff_open(path,&t)!=0) return -1;
    if(t.samples!=4 || t.type!=MC_TIFF_F32){ mc_tiff_close(&t); return -2; }
    int gw=t.width, gh=t.height;
    size_t n=(size_t)gw*gh;
    float *grid = malloc(n*3*sizeof(float));
    float *depth= malloc(n*sizeof(float));
    if(!grid||!depth){ free(grid); free(depth); mc_tiff_close(&t); return -3; }

    const float (*px)[4] = (const float (*)[4])t.pixels;   // (x,y,z,depth)
    double dsum=0; long dcnt=0;
    for(size_t i=0;i<n;++i){
        float x=px[i][0], y=px[i][1], z=px[i][2], d=px[i][3];
        // store as (z,y,x) for mc_quad; preserve VC's invalid marker.
        if(x<0 && y<0 && z<0){ grid[i*3]=-1; grid[i*3+1]=-1; grid[i*3+2]=-1; depth[i]=0; }
        else { grid[i*3]=z; grid[i*3+1]=y; grid[i*3+2]=x; depth[i]=d;
               if(d>0){ dsum+=d; dcnt++; } }
    }
    mc_tiff_close(&t);
    s->gw=gw; s->gh=gh; s->grid=grid; s->depth=depth;
    s->mean_depth = dcnt ? (float)(dsum/(double)dcnt) : 0.0f;
    return 0;
}

int mc_surface_save_tiff(const char *path, const mc_surface *s){
    if(!s||!s->grid||s->gw<1||s->gh<1) return -1;
    size_t n=(size_t)s->gw*s->gh;
    float (*px)[4] = malloc(n*sizeof *px);
    if(!px) return -1;
    for(size_t i=0;i<n;++i){
        float z=s->grid[i*3], y=s->grid[i*3+1], x=s->grid[i*3+2];
        if(x<0 && y<0 && z<0){ px[i][0]=-1; px[i][1]=-1; px[i][2]=-1; px[i][3]=0; }
        else { px[i][0]=x; px[i][1]=y; px[i][2]=z;
               px[i][3] = s->depth ? s->depth[i] : s->mean_depth; }
    }
    int rc = mc_tiff_write(path, s->gw, s->gh, 4, MC_TIFF_F32, px);
    free(px);
    return rc;
}

// any coord < 0 marks an invalid grid point.
static int valid_pt(const float *p){ return !(p[0]<0 || p[1]<0 || p[2]<0); }

int mc_surface_save_obj(const char *path, const mc_surface *s){
    if(!s||!s->grid||s->gw<1||s->gh<1) return -1;
    int gw=s->gw, gh=s->gh;
    FILE *f=fopen(path,"wb"); if(!f) return -1;
    fprintf(f,"# matter-compressor surface mesh\n# grid %d %d\n", gw, gh);
    // 1-based OBJ vertex index per grid cell; 0 = no vertex (invalid).
    int *vidx = calloc((size_t)gw*gh, sizeof(int));
    if(!vidx){ fclose(f); return -1; }
    int nv=0;
    for(int y=0;y<gh;++y)for(int x=0;x<gw;++x){
        const float *p=s->grid+((size_t)y*gw+x)*3;       // (z,y,x)
        if(!valid_pt(p)) continue;
        // OBJ wants x y z; our grid is z,y,x.
        fprintf(f,"v %.6f %.6f %.6f\n", p[2], p[1], p[0]);
        vidx[(size_t)y*gw+x]=++nv;
    }
    // quad faces between 4 valid corners, triangulated (two tris).
    for(int y=0;y<gh-1;++y)for(int x=0;x<gw-1;++x){
        int a=vidx[(size_t)y*gw+x], b=vidx[(size_t)y*gw+x+1];
        int c=vidx[(size_t)(y+1)*gw+x+1], d=vidx[(size_t)(y+1)*gw+x];
        if(a&&b&&c&&d){ fprintf(f,"f %d %d %d\nf %d %d %d\n", a,b,c, a,c,d); }
    }
    free(vidx);
    fclose(f);
    return 0;
}

int mc_surface_load_obj(const char *path, mc_surface *s){
    memset(s,0,sizeof *s);
    FILE *f=fopen(path,"rb"); if(!f) return -1;
    int gw=0, gh=0;
    char line[512];
    // first pass: find the grid hint + count vertices (must match gw*gh).
    long vpos=-1;
    while(fgets(line,sizeof line,f)){
        if(line[0]=='#'){ int a,b; if(sscanf(line,"# grid %d %d",&a,&b)==2){ gw=a; gh=b; } continue; }
        if(line[0]=='v'&&line[1]==' '){ if(vpos<0) vpos=ftell(f)-(long)strlen(line); }
    }
    if(gw<1||gh<1){ fclose(f); return -2; }   // need the grid hint
    size_t n=(size_t)gw*gh;
    float *grid=malloc(n*3*sizeof(float));
    float *depth=calloc(n,sizeof(float));
    if(!grid||!depth){ free(grid); free(depth); fclose(f); return -3; }
    for(size_t i=0;i<n;++i){ grid[i*3]=-1; grid[i*3+1]=-1; grid[i*3+2]=-1; }  // all invalid

    // second pass: fill the grid row-major from the v lines in order.
    rewind(f);
    size_t vi=0;
    while(fgets(line,sizeof line,f) && vi<n){
        if(line[0]=='v'&&line[1]==' '){
            float x,y,z;
            if(sscanf(line+2,"%f %f %f",&x,&y,&z)==3){
                grid[vi*3]=z; grid[vi*3+1]=y; grid[vi*3+2]=x;   // -> (z,y,x)
                vi++;
            }
        }
    }
    fclose(f);
    s->gw=gw; s->gh=gh; s->grid=grid; s->depth=depth; s->mean_depth=0;
    return 0;
}

int mc_ppm_write(const char *path, int w, int h, int channels, const uint8_t *pix){
    if(w<1||h<1||(channels!=1&&channels!=3)) return -1;
    FILE *f=fopen(path,"wb"); if(!f) return -1;
    fprintf(f,"P6\n%d %d\n255\n", w, h);
    int ok=1;
    if(channels==3) ok &= (fwrite(pix,1,(size_t)w*h*3,f)==(size_t)w*h*3);
    else for(size_t i=0;i<(size_t)w*h;++i){ unsigned char g=pix[i]; unsigned char rgb[3]={g,g,g}; ok &= (fwrite(rgb,1,3,f)==3); }
    fclose(f);
    return ok?0:-1;
}
