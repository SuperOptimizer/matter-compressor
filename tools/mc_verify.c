// mc_verify — integrity-check every chunk of an archive (xxh64, format v6).
// usage: mc_verify <archive.mc>
#include "../src/matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
int main(int argc,char**argv){
    if(argc!=2){ fprintf(stderr,"usage: %s <archive.mc>\n",argv[0]); return 2; }
    int fd=open(argv[1],O_RDONLY); if(fd<0){ perror("open"); return 2; }
    struct stat sb; fstat(fd,&sb);
    const uint8_t *arc=mmap(NULL,(size_t)sb.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    if(arc==MAP_FAILED){ perror("mmap"); return 2; }
    long bad=mc_verify_archive(arc,(size_t)sb.st_size,1);
    munmap((void*)arc,(size_t)sb.st_size); close(fd);
    return bad?1:0;
}
