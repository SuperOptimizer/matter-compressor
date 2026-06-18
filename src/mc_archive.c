// mc_archive.c — STATIC-SLOT + SPARSE-FILE archive (.mca v8). See mc_archive.h.
//
// A chunk lives at a COMPUTED slot (chunks_off + slot_index*slot_stride); the
// file is ftruncate'd once to its full logical size (sparse) and mmap'd. Writing
// a chunk is a lock-free memcpy into its slot + a 2-bit occupancy update. There
// is no append cursor, no node tree, no grow lock. Reading a never-written slot
// returns zeros (sparse hole) and the occupancy map says ALL_ZERO/DONT_KNOW.
#define _GNU_SOURCE
#include "matter_compressor.h"
#include "mc_codec.h"
#include "mc_archive.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>
#include <stdatomic.h>

#define MC_BLK3 (MC_BLK*MC_BLK*MC_BLK)            // 16^3 = 4096
#define MCA_PRIORS_OFF  256u                       // priors blob inside header page
#define MCA_PRIORS_MAGIC 0x53524950u               // "PRIS"

static int auto_threads(int n){
    if(n>0) return n;
    long c = sysconf(_SC_NPROCESSORS_ONLN);
    if(c<1) c=4; if(c>16) c=16; return (int)c;
}

// ============================================================================
// open / close
// ============================================================================
mc_archive *mc_archive_open_dims(const char *path, int nx, int ny, int nz, float quality){
    if(nx<=0||ny<=0||nz<=0){ fprintf(stderr,"mc_archive_open: bad dims\n"); return NULL; }
    mc_codec_init();
    int fd = open(path, O_RDWR|O_CREAT, 0644);
    if(fd<0){ perror("mc_archive_open: open"); return NULL; }
    struct stat sb; if(fstat(fd,&sb)!=0){ perror("fstat"); close(fd); return NULL; }
    int fresh = (sb.st_size==0);

    mc_archive *a = calloc(1,sizeof *a);
    if(!a){ close(fd); return NULL; }
    a->fd = fd;
    a->dims[0]=(uint32_t)nz; a->dims[1]=(uint32_t)ny; a->dims[2]=(uint32_t)nx;
    a->num_lods = mca_num_lods((uint32_t)nz,(uint32_t)ny,(uint32_t)nx);
    a->quality = quality;
    a->block_codec = MC_BLOCKCODEC_CABAC;
    a->slot_stride = MC_SLOT_STRIDE;
    a->total_chunks = mca_total_chunks(a->dims, a->num_lods);
    uint64_t occ_off    = MCA_HDR_REGION + MCA_META_REGION;
    uint64_t occ_bytes  = mca_align((a->total_chunks + 3) / 4, 4096);
    a->occ_off    = occ_off;
    a->chunks_off = occ_off + occ_bytes;
    a->logical    = a->chunks_off + a->total_chunks * a->slot_stride;

    if(!fresh){
        uint8_t hdr[96];
        if(pread(fd,hdr,sizeof hdr,0)!=(ssize_t)sizeof hdr){ free(a); close(fd); return NULL; }
        uint32_t magic,ver; memcpy(&magic,hdr+MCAH_MAGIC,4); memcpy(&ver,hdr+MCAH_VERSION,4);
        uint32_t ux,uy,uz; memcpy(&ux,hdr+MCAH_NX,4); memcpy(&uy,hdr+MCAH_NY,4); memcpy(&uz,hdr+MCAH_NZ,4);
        if(magic!=MCA_MAGIC || ver!=MCA_VERSION || (int)ux!=nx||(int)uy!=ny||(int)uz!=nz){
            fprintf(stderr,"mc_archive_open: %s is not a matching v%u archive\n",path,MCA_VERSION);
            free(a); close(fd); return NULL;
        }
    }
    if(ftruncate(fd,(off_t)a->logical)!=0){ perror("mc_archive_open: ftruncate"); free(a); close(fd); return NULL; }
    a->base = mmap(NULL, a->logical, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if(a->base==MAP_FAILED){ perror("mmap"); free(a); close(fd); return NULL; }

    if(fresh){
        memset(a->base, 0, MCA_HDR_REGION);
        uint32_t magic=MCA_MAGIC, ver=MCA_VERSION, bc=a->block_codec;
        uint32_t ux=(uint32_t)nx,uy=(uint32_t)ny,uz=(uint32_t)nz;
        a->base[MCAH_DTYPE]=0; a->base[MCAH_NUMLODS]=(uint8_t)a->num_lods;
        memcpy(a->base+MCAH_MAGIC,&magic,4); memcpy(a->base+MCAH_VERSION,&ver,4);
        memcpy(a->base+MCAH_NX,&ux,4); memcpy(a->base+MCAH_NY,&uy,4); memcpy(a->base+MCAH_NZ,&uz,4);
        memcpy(a->base+MCAH_QUALITY,&quality,4); memcpy(a->base+MCAH_BLOCKCODEC,&bc,4);
        uint64_t mo=MCA_HDR_REGION, ml=0, mc=MCA_META_REGION;
        memcpy(a->base+MCAH_METAOFF,&mo,8); memcpy(a->base+MCAH_METALEN,&ml,8); memcpy(a->base+MCAH_METACAP,&mc,8);
        memcpy(a->base+MCAH_OCCOFF,&a->occ_off,8); memcpy(a->base+MCAH_CHUNKSOFF,&a->chunks_off,8);
        memcpy(a->base+MCAH_SLOTSTRIDE,&a->slot_stride,8); memcpy(a->base+MCAH_TOTCHUNKS,&a->total_chunks,8);
    } else {
        priors_load(a->base);
    }
    return a;
}

mc_archive *mc_archive_open(const char *path, int dim, float quality){
    return mc_archive_open_dims(path,dim,dim,dim,quality);
}

void mc_archive_close(mc_archive *a){
    if(!a) return;
    msync(a->base, a->logical, MS_SYNC);
    munmap(a->base, a->logical);
    close(a->fd);
    free(a);
}

int      mc_archive_set_block_codec(mc_archive *a, uint32_t codec){
    if(!a||codec!=MC_BLOCKCODEC_CABAC) return -1;
    a->block_codec=codec; memcpy(a->base+MCAH_BLOCKCODEC,&codec,4); return 0;
}
uint32_t mc_archive_block_codec(const mc_archive *a){ return a?a->block_codec:MC_BLOCKCODEC_CABAC; }
uint64_t mc_archive_data_len(mc_archive *a){ return a?a->logical:0; }
int mc_archive_reserve_index(mc_archive *a,int lod,int cz,int cy,int cx){
    (void)a;(void)lod;(void)cz;(void)cy;(void)cx; return 0;   // static: nothing to reserve
}

// ============================================================================
// metadata + priors
// ============================================================================
int mc_archive_set_metadata(mc_archive *a, const void *data, size_t len){
    if(!a || (len && !data)) return -1;
    if(len > MCA_META_REGION) return -1;
    if(len) memcpy(a->base + MCA_HDR_REGION, data, len);
    atomic_thread_fence(memory_order_release);
    uint64_t l=len; memcpy(a->base + MCAH_METALEN, &l, 8);
    return 0;
}
const char *mc_metadata(const uint8_t *arc, size_t *out_len){
    uint64_t off,len; memcpy(&off,arc+MCAH_METAOFF,8); memcpy(&len,arc+MCAH_METALEN,8);
    if(out_len) *out_len=(size_t)len;
    return len ? (const char*)(arc+off) : NULL;
}
const char *mc_archive_metadata(mc_archive *a, size_t *out_len){
    if(!a){ if(out_len)*out_len=0; return NULL; }
    return mc_metadata(a->base, out_len);
}

int mc_archive_set_priors(struct mc_archive *a, const uint16_t plo[8][32], const uint16_t phi[8][32]){
    if(!a||!plo||!phi) return -1;
    uint8_t *p = a->base + MCA_PRIORS_OFF;
    uint32_t magic=MCA_PRIORS_MAGIC, ver=1;
    memcpy(p,&magic,4); memcpy(p+4,&ver,4);
    memcpy(p+8,plo,8*32*2); memcpy(p+8+8*32*2,phi,8*32*2);
    mc_codec_set_priors((const uint16_t*)plo,(const uint16_t*)phi);
    return 0;
}
void priors_load(const uint8_t *base){
    uint32_t magic; memcpy(&magic, base+MCA_PRIORS_OFF, 4);
    if(magic!=MCA_PRIORS_MAGIC){ mc_codec_set_priors(NULL,NULL); return; }
    mc_codec_set_priors((const uint16_t*)(base+MCA_PRIORS_OFF+8),
                        (const uint16_t*)(base+MCA_PRIORS_OFF+8+8*32*2));
}

// ============================================================================
// addressing + occupancy
// ============================================================================
static inline int mca_coords_ok(const mc_archive *a,int lod,uint64_t cz,uint64_t cy,uint64_t cx){
    if(lod<0||lod>=a->num_lods) return 0;
    return cz<mca_chunks_axis(a->dims[0],lod) && cy<mca_chunks_axis(a->dims[1],lod)
        && cx<mca_chunks_axis(a->dims[2],lod);
}
uint64_t mc_archive_chunk_offset(mc_archive *a,int lod,int cz,int cy,int cx){
    if(!a||cz<0||cy<0||cx<0||!mca_coords_ok(a,lod,cz,cy,cx)) return 0;
    uint64_t slot = mca_slot_index(a->dims,a->num_lods,lod,cz,cy,cx);
    return a->chunks_off + slot * a->slot_stride;
}
mc_cover mc_archive_chunk_coverage(mc_archive *a,int lod,int cz,int cy,int cx){
    if(!a||cz<0||cy<0||cx<0||!mca_coords_ok(a,lod,cz,cy,cx)) return MC_ABSENT;
    uint64_t slot = mca_slot_index(a->dims,a->num_lods,lod,cz,cy,cx);
    int st = mca_occ_get(a, slot);
    return st==MC_OCC_REAL ? MC_PRESENT : st==MC_OCC_ALL_ZERO ? MC_ZERO : MC_ABSENT;
}

// ============================================================================
// write path — encode a 256^3 chunk straight into its static slot
// ============================================================================
int mc_archive_append_chunk_ctx(mc_archive *a, mc_codec_ctx *C,
                                int lod, int cz,int cy,int cx,
                                const mc_u8 vox[256*256*256]){
    if(!a||!C||!vox||!mca_coords_ok(a,lod,cz,cy,cx)) return -1;
    uint64_t slot = mca_slot_index(a->dims,a->num_lods,lod,cz,cy,cx);

    uint32_t tab[MC_OFFTAB_LEN];
    for(unsigned i=0;i<MC_BLK3;i++) tab[i]=MC_OFF_ABSENT;
    mc_buf pay = {0,0,0};
    mc_u8 blk[MC_BLK3];
    for(int bz=0;bz<16;bz++)for(int by=0;by<16;by++)for(int bx=0;bx<16;bx++){
        int bi=(bz*16+by)*16+bx;
        if(!gather_blk256(vox,bz,by,bx,blk)) continue;     // all-air block -> ABSENT
        uint64_t base_in = pay.len; uint32_t len=0;
        if(!mc_enc_block(C, blk, &pay, &len)) continue;    // encoder declined (air)
        tab[bi] = (uint32_t)(MC_CHUNK_PAYLOAD_OFF + base_in);
    }
    tab[MC_BLK3] = (uint32_t)(MC_CHUNK_PAYLOAD_OFF + pay.len);   // terminator = chunk len

    if(pay.len==0){                                   // whole chunk is air
        free(pay.p);
        mca_occ_set(a, slot, MC_OCC_ALL_ZERO);
        atomic_fetch_add_explicit(&a->gen,1,memory_order_release);
        return 0;
    }
    uint64_t chunk_len = MC_CHUNK_PAYLOAD_OFF + pay.len;
    if(chunk_len > a->slot_stride){ fprintf(stderr,"mc_archive: chunk exceeds slot stride\n"); free(pay.p); return -1; }

    uint8_t *c = a->base + a->chunks_off + slot*a->slot_stride;
    float q = mc_codec_ctx_get_quality(C); uint32_t flags=0;
    memcpy(c+8,&q,4); memcpy(c+12,&flags,4);
    memcpy(c+16, tab, MC_OFFTAB_LEN*4);
    memcpy(c+MC_CHUNK_PAYLOAD_OFF, pay.p, pay.len);
    free(pay.p);
    uint64_t h = mc_xxh64(c+8, (size_t)(chunk_len-8), 0x6D636168756E6Bull);
    memcpy(c, &h, 8);
    mca_occ_set(a, slot, MC_OCC_REAL);
    atomic_fetch_add_explicit(&a->gen,1,memory_order_release);
    return 0;
}

int mc_archive_append_chunk_raw_q(mc_archive *a,int lod,int cz,int cy,int cx,
                                  const mc_u8 vox[256*256*256], float q){
    if(!a) return -1;
    mc_codec_ctx *C=mc_codec_ctx_new(); if(!C) return -1;
    mc_codec_ctx_set_quality(C,q);
    int rc=mc_archive_append_chunk_ctx(a,C,lod,cz,cy,cx,vox);
    mc_codec_ctx_free(C); return rc;
}
int mc_archive_append_chunk_raw(mc_archive *a,int lod,int cz,int cy,int cx, const mc_u8 vox[256*256*256]){
    return mc_archive_append_chunk_raw_q(a,lod,cz,cy,cx,vox,a?a->quality:8.0f);
}
// parallel single-chunk encode: with the lock-free static writer this is just the
// serial path (no external callers need the intra-chunk parallelism).
int mc_archive_append_chunk_par(mc_archive *a,int lod,int cz,int cy,int cx,
                                const mc_u8 vox[256*256*256], float q, int nthreads){
    (void)nthreads; return mc_archive_append_chunk_raw_q(a,lod,cz,cy,cx,vox,q>0?q:(a?a->quality:8.0f));
}

#define MC_RC_GAMMA 0.75f
int mc_archive_append_chunk_target(mc_archive *a,int lod,int cz,int cy,int cx,
                                   const mc_u8 vox[256*256*256], float target_ratio, float *q_out){
    if(!a||!vox||target_ratio<=1.0f) return -1;
    float q0=a->quality;
    mc_codec_ctx *C=mc_codec_ctx_new(); if(!C) return -1;
    mc_codec_ctx_set_quality(C,q0);
    mc_buf samp={0,0,0}; mc_u8 blk[MC_BLK3]; long sampled=0; size_t sample_bytes=0;
    for(int d=0; d<16; ++d){
        if(!gather_blk256(vox,d,d,d,blk)){ sampled++; continue; }
        uint32_t len=0; if(mc_enc_block(C,blk,&samp,&len)) sample_bytes+=len;
        sampled++;
    }
    free(samp.p); mc_codec_ctx_free(C);
    float q=q0;
    if(sample_bytes && sampled){
        double est_total=(double)sample_bytes*(4096.0/(double)sampled);
        double want_total=(double)MC_CHUNK_RAW/target_ratio;
        q=(float)(q0*pow(est_total/want_total,1.0/MC_RC_GAMMA));
        if(q<0.5f)q=0.5f; if(q>24.0f)q=24.0f;
    }
    if(q_out)*q_out=q;
    return mc_archive_append_chunk_raw_q(a,lod,cz,cy,cx,vox,q);
}

// copy a pre-encoded chunk blob (new format) verbatim into its slot.
int mc_archive_append_chunk_compressed(mc_archive *a,int lod,int cz,int cy,int cx,
                                       const uint8_t *blob, size_t len){
    if(!a||!blob||!len||!mca_coords_ok(a,lod,cz,cy,cx)) return -1;
    if(len > a->slot_stride) return -1;
    uint64_t slot = mca_slot_index(a->dims,a->num_lods,lod,cz,cy,cx);
    memcpy(a->base + a->chunks_off + slot*a->slot_stride, blob, len);
    mca_occ_set(a, slot, MC_OCC_REAL);
    atomic_fetch_add_explicit(&a->gen,1,memory_order_release);
    return 0;
}

// ============================================================================
// read path
// ============================================================================
int mc_archive_block_blob(mc_archive *a, uint64_t chunk_off, int bz,int by,int bx,
                          const uint8_t **ptr, uint32_t *len){
    if(!a||!chunk_off) return 0;
    const uint8_t *c = a->base + chunk_off;
    if(mc_chunk_len(c) < MC_CHUNK_PAYLOAD_OFF) return 0;     // sparse / empty slot
    uint32_t off,l;
    if(!mc_block_span(c,(bz*16+by)*16+bx,&off,&l)) return 0;
    if(chunk_off+off+l > a->logical) return 0;
    if(ptr) *ptr = c+off; if(len) *len=l; return 1;
}

static void decode_block_ctx(mc_codec_ctx *C, mc_archive *a, uint64_t chunk_off,
                             int bz,int by,int bx, mc_u8 *dst){
    if(!a||!chunk_off){ memset(dst,0,MC_BLK3); return; }
    const uint8_t *c = a->base + chunk_off;
    if(mc_chunk_len(c) < MC_CHUNK_PAYLOAD_OFF){ memset(dst,0,MC_BLK3); return; }  // sparse/empty
    uint32_t off,len;
    if(!mc_block_span(c,(bz*16+by)*16+bx,&off,&len)){ memset(dst,0,MC_BLK3); return; }
    if(chunk_off+off+len > a->logical){ memset(dst,0,MC_BLK3); return; }
    mc_codec_ctx_set_quality(C, mc_chunk_q(c));
    mc_dec_block_codec(C, a->block_codec, c+off, len, dst);
}

static pthread_key_t g_decblk_key;
static pthread_once_t g_decblk_once = PTHREAD_ONCE_INIT;
static void decblk_dtor(void *p){ if(p) mc_codec_ctx_free((mc_codec_ctx*)p); }
static void decblk_key_init(void){ pthread_key_create(&g_decblk_key, decblk_dtor); }
void mc_archive_decode_block(mc_archive *a, uint64_t chunk_off, int bz,int by,int bx, mc_u8 *dst){
    pthread_once(&g_decblk_once, decblk_key_init);
    mc_codec_ctx *C = pthread_getspecific(g_decblk_key);
    if(!C){ C=mc_codec_ctx_new(); if(!C){ memset(dst,0,MC_BLK3); return; } pthread_setspecific(g_decblk_key,C); }
    decode_block_ctx(C,a,chunk_off,bz,by,bx,dst);
}

void mc_archive_decode_chunk(mc_archive *a, uint64_t chunk_off, mc_u8 *out, int nthreads){
    (void)nthreads;
    mc_codec_ctx *C=mc_codec_ctx_new(); if(!C){ memset(out,0,MC_CHUNK_RAW); return; }
    mc_u8 blk[MC_BLK3];
    for(int bz=0;bz<16;bz++)for(int by=0;by<16;by++)for(int bx=0;bx<16;bx++){
        decode_block_ctx(C,a,chunk_off,bz,by,bx,blk);
        for(int z=0;z<16;z++)for(int y=0;y<16;y++)
            memcpy(out+(((size_t)(bz*16+z)*256+(by*16+y))*256)+bx*16, blk+((z*16+y)*16), 16);
    }
    mc_codec_ctx_free(C);
}

int mc_archive_block_present(mc_archive *a,int lod,int bz,int by,int bx){
    if(!a) return 0;
    if(mc_archive_chunk_coverage(a,lod,bz>>4,by>>4,bx>>4)!=MC_PRESENT) return 0;
    uint64_t co=mc_archive_chunk_offset(a,lod,bz>>4,by>>4,bx>>4);
    if(!co) return 0;
    return mc_chunk_block_present(a->base+co, ((bz&15)*16+(by&15))*16+(bx&15));
}
// material fraction is no longer stored (mcpp drops the fracmap): report 1.0 for
// a present block, 0.0 otherwise.
float mc_archive_block_fraction(mc_archive *a,int lod,int bz,int by,int bx){
    return mc_archive_block_present(a,lod,bz,by,bx) ? 1.0f : 0.0f;
}

// ---- region read (decode the blocks overlapping a box) ---------------------
typedef struct {
    mc_archive *a; int lod; long z0,y0,x0,dz,dy,dx; mc_u8 *out; size_t sz,sy;
    int nbz,nby,nbx; long bz0,by0,bx0; _Atomic uint32_t next;
} region_ctx;
static void *region_worker(void *p){
    region_ctx *c=p; mc_u8 blk[MC_BLK3];
    uint32_t nb=(uint32_t)(c->nbz*c->nby*c->nbx);
    for(;;){
        uint32_t w=atomic_fetch_add_explicit(&c->next,1,memory_order_relaxed);
        if(w>=nb) break;
        long bz=c->bz0+w/(c->nby*c->nbx), by=c->by0+(w/c->nbx)%c->nby, bx=c->bx0+w%c->nbx;
        long gz=bz*16, gy=by*16, gx=bx*16;
        long iz0=gz>c->z0?gz:c->z0, iz1=(gz+16<c->z0+c->dz)?gz+16:c->z0+c->dz;
        long iy0=gy>c->y0?gy:c->y0, iy1=(gy+16<c->y0+c->dy)?gy+16:c->y0+c->dy;
        long ix0=gx>c->x0?gx:c->x0, ix1=(gx+16<c->x0+c->dx)?gx+16:c->x0+c->dx;
        if(iz0>=iz1||iy0>=iy1||ix0>=ix1) continue;
        int present=0;
        if(mc_archive_chunk_coverage(c->a,c->lod,(int)(bz>>4),(int)(by>>4),(int)(bx>>4))==MC_PRESENT){
            uint64_t co=mc_archive_chunk_offset(c->a,c->lod,(int)(bz>>4),(int)(by>>4),(int)(bx>>4));
            if(co && mc_chunk_block_present(c->a->base+co,(int)(((bz&15)*16+(by&15))*16+(bx&15)))){
                mc_archive_decode_block(c->a,co,(int)(bz&15),(int)(by&15),(int)(bx&15),blk);
                present=1;
            }
        }
        long nrow=ix1-ix0;
        for(long z=iz0;z<iz1;++z)for(long y=iy0;y<iy1;++y){
            mc_u8 *dst=c->out+(size_t)(z-c->z0)*c->sz+(size_t)(y-c->y0)*c->sy+(size_t)(ix0-c->x0);
            if(present) memcpy(dst,blk+((size_t)(z-gz)*16+(y-gy))*16+(ix0-gx),(size_t)nrow);
            else        memset(dst,0,(size_t)nrow);
        }
    }
    return NULL;
}
void mc_archive_read_region(mc_archive *a,int lod, long z0,long y0,long x0, long dz,long dy,long dx,
                            mc_u8 *out, size_t sz, size_t sy, int nthreads){
    if(!a||lod<0||lod>=a->num_lods||!out||dz<=0||dy<=0||dx<=0) return;
    region_ctx c={.a=a,.lod=lod,.z0=z0,.y0=y0,.x0=x0,.dz=dz,.dy=dy,.dx=dx,.out=out,.sz=sz,.sy=sy};
    c.bz0=z0>>4; c.by0=y0>>4; c.bx0=x0>>4;
    c.nbz=(int)(((z0+dz+15)>>4)-c.bz0); c.nby=(int)(((y0+dy+15)>>4)-c.by0); c.nbx=(int)(((x0+dx+15)>>4)-c.bx0);
    atomic_store(&c.next,0);
    int nt=auto_threads(nthreads); uint32_t nb=(uint32_t)(c.nbz*c.nby*c.nbx);
    if((uint32_t)nt>nb) nt=(int)nb;
    if(nt<=1){ region_worker(&c); return; }
    pthread_t th[16]; for(int t=0;t<nt;++t) pthread_create(&th[t],NULL,region_worker,&c);
    for(int t=0;t<nt;++t) pthread_join(th[t],NULL);
}

// ---- batched region crops --------------------------------------------------
typedef struct { mc_archive *a; int lod; const mc_box *boxes; int n; long dz,dy,dx;
                 mc_u8 *out; size_t bstride; _Atomic uint32_t next; } crops_ctx;
static void *crops_worker(void *p){
    crops_ctx *c=p;
    for(;;){
        uint32_t i=atomic_fetch_add_explicit(&c->next,1,memory_order_relaxed);
        if((int)i>=c->n) break;
        const mc_box *b=&c->boxes[i];
        mc_archive_read_region(c->a,c->lod,b->z0,b->y0,b->x0,c->dz,c->dy,c->dx,
                               c->out+(size_t)i*c->bstride,(size_t)c->dy*c->dx,(size_t)c->dx,1);
    }
    return NULL;
}
void mc_archive_read_regions(mc_archive *a,int lod, const mc_box *boxes,int n,
                             long dz,long dy,long dx, mc_u8 *out, size_t batch_stride, int nthreads){
    if(!a||!boxes||n<=0||!out) return;
    crops_ctx c={.a=a,.lod=lod,.boxes=boxes,.n=n,.dz=dz,.dy=dy,.dx=dx,.out=out,.bstride=batch_stride};
    atomic_store(&c.next,0);
    int nt=auto_threads(nthreads); if(nt>n)nt=n;
    if(nt<=1){ crops_worker(&c); return; }
    pthread_t th[16]; for(int t=0;t<nt;++t) pthread_create(&th[t],NULL,crops_worker,&c);
    for(int t=0;t<nt;++t) pthread_join(th[t],NULL);
}

// pick `count` random present chunk-aligned boxes (min_frac>0 -> must be present).
int mc_archive_sample_boxes(mc_archive *a,int lod,uint64_t seed,int count,
                            long dz,long dy,long dx,float min_frac, mc_box *out){
    if(!a||lod<0||lod>=a->num_lods||count<=0||!out) return 0;
    (void)dz;(void)dy;(void)dx;
    uint64_t ncz=mca_chunks_axis(a->dims[0],lod), ncy=mca_chunks_axis(a->dims[1],lod), ncx=mca_chunks_axis(a->dims[2],lod);
    uint64_t total=ncz*ncy*ncx; if(!total) return 0;
    uint64_t s=seed?seed:0x9E3779B97F4A7C15ull; int got=0;
    for(uint64_t tries=0; tries<total*4 && got<count; ++tries){
        s^=s>>12; s^=s<<25; s^=s>>27; uint64_t r=(s*0x2545F4914F6CDD1Dull)%total;
        uint64_t cz=r/(ncy*ncx), rem=r%(ncy*ncx), cy=rem/ncx, cx=rem%ncx;
        mc_cover cov=mc_archive_chunk_coverage(a,lod,(int)cz,(int)cy,(int)cx);
        if(min_frac>0.0f && cov!=MC_PRESENT) continue;
        out[got].z0=(long)cz*256; out[got].y0=(long)cy*256; out[got].x0=(long)cx*256; got++;
    }
    return got;
}

