// mc_s3.h — optional convenience: open a matter-compressor archive directly
// over S3/HTTP (vendored libs3: s3:// with SigV4/anonymous, or https://).
// Wires up the streaming reader with ranged GETs; node-table caching is
// built into the reader, and partial-fetch mode is enabled by default so a
// cold block costs bitmap+lens+payload bytes only.
// Separate translation unit (tools/vendor/libs3 + libcurl) so the core
// library keeps zero dependencies.
#ifndef MC_S3_H
#define MC_S3_H
#include "matter_compressor.h"

typedef struct mc_s3 mc_s3;

// Open an archive at `url` (s3://bucket/key or https://...). Returns NULL on
// any failure (unreachable, not an mc archive). The handle owns the HTTP
// client and the mc_reader.
mc_s3 *mc_s3_open(const char *url);
// The reader for all decode calls (mc_chunk_offset / mc_decode_block / ...).
mc_reader *mc_s3_reader(mc_s3 *s);
void mc_s3_close(mc_s3 *s);

#endif
