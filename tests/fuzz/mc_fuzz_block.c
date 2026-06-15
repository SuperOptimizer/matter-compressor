// mc_fuzz_block — fuzz the block codec's decode primitive (mc_dec_block) on
// arbitrary payload bytes. This is the hottest, most-exposed routine: every
// archive/streamed block funnels its self-contained payload through here, so it
// must be robust to corrupt payloads (the range coder + block-mask + coefficient
// decoders all read attacker-controlled bits). The bar: no crash / OOB / UB on
// ANY (payload, len) under ASan+UBSan. Wrong decoded voxels are fine.
//
// Entrypoint is the libFuzzer API (LLVMFuzzerTestOneInput); driven by AFL++
// (scripts/fuzz.sh builds it with afl-clang-fast -fsanitize=fuzzer). With
// -DMC_FUZZ_STANDALONE it replays files/stdin once (CI smoke + crash triage).
#include "matter_compressor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static void exercise(const uint8_t *data, size_t size){
    if(size < 1) return;
    // First byte selects the codec quality so the fuzzer explores different
    // quant tables / step ladders; the rest is the block payload.
    mc_codec_ctx *ctx = mc_codec_ctx_new();
    if(!ctx) return;
    float q = 0.5f + (float)(data[0] % 64);          // 0.5 .. 64.5, the real q range
    mc_codec_ctx_set_quality(ctx, q);
    // also exercise the max-error decode-agnostic path (tau is encode-side, but
    // setting it must not change decode bounds).
    mc_codec_ctx_set_max_error(ctx, data[0] & 7);

    const uint8_t *payload = data + 1;
    uint32_t plen = (uint32_t)(size - 1);
    mc_u8 dst[16*16*16];
    mc_dec_block(ctx, payload, plen, dst);            // must not crash on garbage

    mc_codec_ctx_free(ctx);
}

#ifdef MC_FUZZ_STANDALONE
static int replay(const char *path){
    FILE *f = path ? fopen(path,"rb") : stdin;
    if(!f){ perror(path); return 1; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n<0){ if(path)fclose(f); return 1; }
    uint8_t *buf=malloc((size_t)n?n:1);
    size_t got=fread(buf,1,(size_t)n,f);
    if(path)fclose(f);
    exercise(buf,got); free(buf);
    return 0;
}
int main(int argc,char**argv){
    if(argc<2) return replay(NULL);
    for(int i=1;i<argc;++i) replay(argv[i]);
    printf("mc_fuzz_block: replayed %d input(s), no crash\n", argc-1);
    return 0;
}
#else
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size){ exercise(data,size); return 0; }
#endif
