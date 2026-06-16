/* mc_surface.c — parametric surface I/O (see mc_surface.h). */
#include "mc_surface.h"
#include "mc_tiff.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

void mc_mesh_free(mc_mesh *m){ if(!m) return; free(m->v); free(m->vn); free(m->tri); memset(m,0,sizeof *m); }

// parse one OBJ face vertex token "a", "a/b", "a//c", or "a/b/c" -> 1-based
// vertex index (the first field). Returns 0 if no number.
static long obj_face_v(const char *tok){
    long v=0; int sign=1; const char *p=tok;
    while(*p==' '||*p=='\t') p++;
    if(*p=='-'){ sign=-1; p++; }
    if(*p<'0'||*p>'9') return 0;
    while(*p>='0'&&*p<='9'){ v=v*10+(*p-'0'); p++; }
    return sign*v;
}

int mc_mesh_load_obj(const char *path, mc_mesh *m){
    memset(m,0,sizeof *m);
    FILE *f=fopen(path,"rb"); if(!f) return -1;
    // growable arrays
    size_t vcap=1024, ncap=1024, tcap=2048;
    float *v=malloc(vcap*3*sizeof(float)); int nv=0;
    float *vn=malloc(ncap*3*sizeof(float)); int nvn=0;
    int   *tri=malloc(tcap*3*sizeof(int)); int nt=0;
    if(!v||!vn||!tri){ free(v);free(vn);free(tri);fclose(f); return -1; }
    char line[1024];
    while(fgets(line,sizeof line,f)){
        if(line[0]=='v'&&line[1]==' '){
            float a,b,c; if(sscanf(line+2,"%f %f %f",&a,&b,&c)==3){
                if((size_t)nv>=vcap){ vcap*=2; v=realloc(v,vcap*3*sizeof(float)); }
                v[nv*3]=a; v[nv*3+1]=b; v[nv*3+2]=c; nv++; }
        } else if(line[0]=='v'&&line[1]=='n'&&line[2]==' '){
            float a,b,c; if(sscanf(line+3,"%f %f %f",&a,&b,&c)==3){
                if((size_t)nvn>=ncap){ ncap*=2; vn=realloc(vn,ncap*3*sizeof(float)); }
                vn[nvn*3]=a; vn[nvn*3+1]=b; vn[nvn*3+2]=c; nvn++; }
        } else if(line[0]=='f'&&line[1]==' '){
            // collect this face's vertex indices, fan-triangulate.
            long idx[64]; int ni=0;
            char *p=line+1, *tok;
            while(ni<64 && (tok=strtok(p," \t\r\n"))){ p=NULL;
                long fv=obj_face_v(tok); if(fv==0) continue;
                if(fv<0) fv = nv + fv + 1;          // negative = relative
                idx[ni++]=fv;
            }
            for(int k=1;k+1<ni;++k){
                if((size_t)nt>=tcap){ tcap*=2; tri=realloc(tri,tcap*3*sizeof(int)); }
                tri[nt*3]  =(int)idx[0]-1;
                tri[nt*3+1]=(int)idx[k]-1;
                tri[nt*3+2]=(int)idx[k+1]-1;
                nt++;
            }
        }
    }
    fclose(f);
    // normals are usable only if there's one per vertex.
    if(nvn != nv){ free(vn); vn=NULL; }
    m->nv=nv; m->nt=nt; m->v=v; m->vn=vn; m->tri=tri;
    return (nv>0) ? 0 : -2;
}

int mc_mesh_save_obj(const char *path, const mc_mesh *m){
    if(!m||!m->v||m->nv<1) return -1;
    FILE *f=fopen(path,"wb"); if(!f) return -1;
    fprintf(f,"# matter-compressor mesh\n");
    for(int i=0;i<m->nv;++i) fprintf(f,"v %.6f %.6f %.6f\n", m->v[i*3],m->v[i*3+1],m->v[i*3+2]);
    if(m->vn) for(int i=0;i<m->nv;++i) fprintf(f,"vn %.6f %.6f %.6f\n", m->vn[i*3],m->vn[i*3+1],m->vn[i*3+2]);
    for(int i=0;i<m->nt;++i){
        int a=m->tri[i*3]+1,b=m->tri[i*3+1]+1,c=m->tri[i*3+2]+1;
        if(m->vn) fprintf(f,"f %d//%d %d//%d %d//%d\n",a,a,b,b,c,c);
        else      fprintf(f,"f %d %d %d\n",a,b,c);
    }
    fclose(f);
    return 0;
}

// VC per-pixel map: text header lines "key: value" until a "<>" line, then
// W*H*dim little-endian doubles row-major. dim>=3 = xyz (+ normal for dim 6).
int mc_surface_load_vcps_ppm(const char *path, mc_surface *s, float default_depth){
    memset(s,0,sizeof *s);
    FILE *f=fopen(path,"rb"); if(!f) return -1;
    int W=0,H=0,D=0; char line[256]; int saw_term=0; char typ[32]={0};
    while(fgets(line,sizeof line,f)){
        if(strncmp(line,"<>",2)==0){ saw_term=1; break; }
        int iv; char sv[64];
        if(sscanf(line,"width: %d",&iv)==1) W=iv;
        else if(sscanf(line,"height: %d",&iv)==1) H=iv;
        else if(sscanf(line,"dim: %d",&iv)==1) D=iv;
        else if(sscanf(line,"type: %63s",sv)==1) strncpy(typ,sv,sizeof typ-1);
    }
    if(!saw_term || W<1 || H<1 || D<3 || strcmp(typ,"double")!=0){ fclose(f); return -2; }
    size_t n=(size_t)W*H;
    double *row=malloc((size_t)D*sizeof(double));
    float *grid=malloc(n*3*sizeof(float));
    float *depth=malloc(n*sizeof(float));
    if(!row||!grid||!depth){ free(row);free(grid);free(depth);fclose(f); return -3; }
    int ok=1;
    for(size_t i=0;i<n;++i){
        if(fread(row,sizeof(double),(size_t)D,f)!=(size_t)D){ ok=0; break; }
        double x=row[0],y=row[1],z=row[2];
        if(x==0&&y==0&&z==0){ grid[i*3]=-1;grid[i*3+1]=-1;grid[i*3+2]=-1; depth[i]=0; }   // no surface here
        else { grid[i*3]=(float)z; grid[i*3+1]=(float)y; grid[i*3+2]=(float)x; depth[i]=default_depth; }
    }
    free(row); fclose(f);
    if(!ok){ free(grid); free(depth); return -4; }
    s->gw=W; s->gh=H; s->grid=grid; s->depth=depth; s->mean_depth=default_depth;
    return 0;
}

// --- tiny symmetric 3x3 eigen via Jacobi rotations (for PCA plane fit) ---
static void jacobi3(double a[3][3], double v[3][3]){
    for(int i=0;i<3;++i)for(int j=0;j<3;++j) v[i][j]=(i==j)?1.0:0.0;
    for(int sweep=0; sweep<32; ++sweep){
        // largest off-diagonal
        int p=0,q=1; double mx=fabs(a[0][1]);
        if(fabs(a[0][2])>mx){mx=fabs(a[0][2]);p=0;q=2;}
        if(fabs(a[1][2])>mx){mx=fabs(a[1][2]);p=1;q=2;}
        if(mx<1e-12) break;
        double app=a[p][p],aqq=a[q][q],apq=a[p][q];
        double phi=0.5*atan2(2*apq, aqq-app);
        double c=cos(phi), s=sin(phi);
        for(int k=0;k<3;++k){ double akp=a[k][p],akq=a[k][q]; a[k][p]=c*akp-s*akq; a[k][q]=s*akp+c*akq; }
        for(int k=0;k<3;++k){ double apk=a[p][k],aqk=a[q][k]; a[p][k]=c*apk-s*aqk; a[q][k]=s*apk+c*aqk; }
        for(int k=0;k<3;++k){ double vkp=v[k][p],vkq=v[k][q]; v[k][p]=c*vkp-s*vkq; v[k][q]=s*vkp+c*vkq; }
    }
}

int mc_grid_from_mesh(const mc_mesh *m, int gw, int gh, float default_depth, mc_surface *s){
    memset(s,0,sizeof *s);
    if(!m||!m->v||m->nv<3) return -1;
    int nv=m->nv;
    // centroid
    double cx=0,cy=0,cz=0;
    for(int i=0;i<nv;++i){ cx+=m->v[i*3]; cy+=m->v[i*3+1]; cz+=m->v[i*3+2]; }
    cx/=nv; cy/=nv; cz/=nv;
    // covariance
    double cov[3][3]={{0,0,0},{0,0,0},{0,0,0}};
    for(int i=0;i<nv;++i){
        double dx=m->v[i*3]-cx, dy=m->v[i*3+1]-cy, dz=m->v[i*3+2]-cz;
        cov[0][0]+=dx*dx; cov[0][1]+=dx*dy; cov[0][2]+=dx*dz;
        cov[1][1]+=dy*dy; cov[1][2]+=dy*dz; cov[2][2]+=dz*dz;
    }
    cov[1][0]=cov[0][1]; cov[2][0]=cov[0][2]; cov[2][1]=cov[1][2];
    double ev[3][3]; jacobi3(cov, ev);
    double eval[3]={cov[0][0],cov[1][1],cov[2][2]};
    // two largest eigenvalues -> in-plane axes u,v; smallest -> normal.
    int i0=0,i1=1,i2=2;
    if(eval[i0]<eval[i1]){int t=i0;i0=i1;i1=t;}
    if(eval[i1]<eval[i2]){int t=i1;i1=i2;i2=t;}
    if(eval[i0]<eval[i1]){int t=i0;i0=i1;i1=t;}
    double U[3]={ev[0][i0],ev[1][i0],ev[2][i0]};
    double V[3]={ev[0][i1],ev[1][i1],ev[2][i1]};

    // project verts to (u,v); track bounds.
    float *uu=malloc((size_t)nv*sizeof(float)), *vv=malloc((size_t)nv*sizeof(float));
    if(!uu||!vv){ free(uu); free(vv); return -1; }
    double umin=1e30,umax=-1e30,vmin=1e30,vmax=-1e30;
    for(int i=0;i<nv;++i){
        double dx=m->v[i*3]-cx, dy=m->v[i*3+1]-cy, dz=m->v[i*3+2]-cz;
        double u=dx*U[0]+dy*U[1]+dz*U[2], v=dx*V[0]+dy*V[1]+dz*V[2];
        uu[i]=(float)u; vv[i]=(float)v;
        if(u<umin)umin=u; if(u>umax)umax=u; if(v<vmin)vmin=v; if(v>vmax)vmax=v;
    }
    if(gw<=0||gh<=0){ int side=(int)(sqrt((double)nv)*1.5); if(side<8)side=8; gw=gh=side; }
    double du=(umax-umin)/(gw-1>0?gw-1:1), dv=(vmax-vmin)/(gh-1>0?gh-1:1);
    if(du<=0)du=1; if(dv<=0)dv=1;
    double cell = (du<dv?du:dv);

    size_t n=(size_t)gw*gh;
    float *grid=malloc(n*3*sizeof(float)), *depth=malloc(n*sizeof(float));
    if(!grid||!depth){ free(uu);free(vv);free(grid);free(depth); return -1; }
    for(size_t i=0;i<n;++i){ grid[i*3]=-1;grid[i*3+1]=-1;grid[i*3+2]=-1; depth[i]=0; }
    // nearest-vertex fill: for each grid cell (gu,gv) find the closest projected
    // vertex within ~1.5 cells; copy its xyz. O(gw*gh*nv) — fine for segments.
    for(int gy=0;gy<gh;++gy)for(int gx=0;gx<gw;++gx){
        double tu=umin+gx*du, tv=vmin+gy*dv;
        int best=-1; double bd=cell*cell*2.5;
        for(int i=0;i<nv;++i){ double ddx=uu[i]-tu, ddy=vv[i]-tv, d2=ddx*ddx+ddy*ddy; if(d2<bd){bd=d2;best=i;} }
        if(best>=0){ size_t gi=(size_t)gy*gw+gx;
            grid[gi*3]=m->v[best*3+2]; grid[gi*3+1]=m->v[best*3+1]; grid[gi*3+2]=m->v[best*3]; // (z,y,x)
            depth[gi]=default_depth; }
    }
    free(uu); free(vv);
    s->gw=gw; s->gh=gh; s->grid=grid; s->depth=depth; s->mean_depth=default_depth;
    return 0;
}
