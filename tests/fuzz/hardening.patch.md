# Decode-path hardening — proposed patch

The fix for the 5 OOB crash sites in `tests/fuzz/FINDINGS.md`. I drafted and
verified these edits against `src/matter_compressor.c`, but an external agent is
editing the same file concurrently and reverted them. Recording the exact patch
here so it isn't lost — apply when the file is quiescent. Each hunk is
self-contained and was confirmed to fix its crash under ASan+UBSan.

## 1. Bounded node-tree walk (`mc_resolve_chunk` → `mc_resolve_chunk_b`)

Add an `end` bound; reject any node offset whose `MC_GRID3`-slot array doesn't
fit in `[0,end)`. Update the 4 callers: archive callers pass
`atomic_load(&a->file_len)`; reader callers (`mc_chunk_offset`,
`mc_chunk_offset_chk`) pass `r->len`.

```c
static uint64_t mc_resolve_chunk_b(const uint8_t*arc, uint64_t root_off,
                                   int cz,int cy,int cx, uint64_t end){
    uint64_t node = root_off;
    for(int nib=MC_TREE_LEVELS-1; nib>=0; --nib){
        if(!node) return 0;
        int dz=mc_nib(cz,nib), dy=mc_nib(cy,nib), dx=mc_nib(cx,nib);
        int idx=(dz*16+dy)*16+dx;
        if(node > end || (size_t)idx*8 + 8 > end - node) return 0;   // OOB -> absent
        uint64_t childoff; memcpy(&childoff, arc+node + (size_t)idx*8, 8);
        node = childoff;
    }
    return node;
}
```

## 2. `mc_open` — untrusted-input gate

```c
mc_reader *mc_open(const uint8_t *arc, size_t len){
    if(!arc || len < MC_HDR) return NULL;                 // too small for a header
    uint32_t magic; memcpy(&magic,arc+MCH_MAGIC,4);
    if(magic != MC_MAGIC) return NULL;
    mc_codec_init();
    mc_reader *r=calloc(1,sizeof *r); if(!r) return NULL;
    r->arc=arc; r->len=len; r->codec=mc_codec_ctx_new();
    reader_hdr_load(r, arc);
    uint64_t poff; memcpy(&poff,arc+MCH_PRIOROFF,8);
    if(poff && poff <= len && len - poff >= (uint64_t)MC_PRIORS_BYTES){
        uint32_t pmagic; memcpy(&pmagic,arc+poff,4);
        if(pmagic==MC_PRIORS_MAGIC){
            memcpy(r->priors,arc+poff,MC_PRIORS_BYTES); r->has_priors=1;
            priors_load(arc);                              // offset validated -> safe
        }
    }
    return r;
}
```

## 3. `mc_metadata` — clamp to the archive's declared length

Reads fixed header fields, so its precondition is a `>= MC_HDR` buffer (the
"flat archive bytes" form). Defensive clamp against a corrupt off/len:

```c
const char *mc_metadata(const uint8_t *arc, size_t *out_len){
    if(!arc){ if(out_len)*out_len=0; return NULL; }
    uint64_t off,len,tot;
    memcpy(&off,arc+MCH_METAOFF,8); memcpy(&len,arc+MCH_METALEN,8);
    memcpy(&tot,arc+MCH_TOTLEN,8);
    if(tot < MC_HDR) tot = MC_HDR;
    if(!off) off=MC_HDR;
    if(off > tot){ if(out_len)*out_len=0; return (const char*)(arc+MC_HDR); }
    if(off + len > tot) len = tot - off;
    if(out_len) *out_len=(size_t)len;
    return (const char*)(arc+off);
}
```

## 4. `mc_verify_archive` — use its `len` argument (drop `(void)len;`)

Bound `root`/`inner`/`shard` (each a `MC_GRID3*8`-byte node array) and each
chunk blob `[co, co+blen)` to `[MC_HDR, len)`. Out-of-range -> counted corrupt,
never dereferenced. Return `-1` if `len < MC_HDR`. (Full hunk in FINDINGS.md
history / ask me to re-emit.)

## Verify

```sh
clang -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
  tests/mc_decode_robust_test.c src/matter_compressor.c src/c3d.c \
  tools/vendor/libs3/libs3.c -Isrc -Itools/vendor/libs3 -lm -lpthread -lzstd -lcurl \
  -o robust && ./robust          # -> "mc_decode_robust_test: N cases, no crash -> OK"
scripts/fuzz.sh 120              # AFL++ should no longer find crashes from valid seeds
```
