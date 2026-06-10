// CI helper: build a deterministic archive and print its xxh64-style digest.
// Used to assert cross-ISA bitstream identity (NEON vs AVX2 builds must
// produce byte-identical archives).
#include "../src/matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>

static mc_u8 srcv(void *ud, int x,int y,int z){
    (void)ud;
    double dx=x-128,dy=y-128,dz=z-128;
    if(dx*dx+dy*dy+dz*dz>120.0*120.0) return 0;
    return (mc_u8)(40+((x*7+y*3+z*5)%160));
}
int main(void){
    mc_build_opts o={.dim=256,.quality=6.0f};
    size_t len=0; uint8_t *arc=mc_build(srcv,NULL,&o,&len);
    if(!arc) return 1;
    // FNV-1a over the whole archive (simple, dependency-free digest)
    uint64_t h=0xcbf29ce484222325ull;
    for(size_t i=0;i<len;++i){ h^=arc[i]; h*=0x100000001b3ull; }
    printf("%zu %016llx\n",len,(unsigned long long)h);
    free(arc);
    return 0;
}
