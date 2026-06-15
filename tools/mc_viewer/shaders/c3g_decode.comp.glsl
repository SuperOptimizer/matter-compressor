#version 450
// c3g GPU compute decoder — ONE WORKGROUP decodes ONE 16^3 block from its
// compressed c3g payload, writing 4096 u8 voxels. GPU mirror of the CPU oracle
// mc_c3g_dec_block.
//
// Model: the rANS + sign/escape pass is inherently serial, so invocation 0 of
// the workgroup decodes the coefficient levels into SHARED memory; then all 256
// invocations cooperate on dequant + the separable inverse DCT-16 + store. The
// block's working set (levels, coefficients, two scratch buffers, air mask)
// lives in shared memory, NOT per-thread registers, so it fits.
//
// Tables constant across blocks (rANS freq/cdf/slot, dequant step_tab, scan
// order, inverse DCT basis) are uploaded once as read-only storage buffers, so
// the shader does no powf/cos and stays close to the CPU math.
//
// SDL_GPU SPIR-V compute resource sets: read-only storage buffers set 0,
// read-write storage buffers set 1, uniforms set 2.

layout(local_size_x = 256) in;

const int  N    = 16;
const int  N3   = 4096;
const uint TBITS = 12u;
const uint TOTAL = 4096u;
const uint RANS_L = 65536u;
const int  MAXSYM = 16;
const float MC_DZ_FRAC = 0.80;     // matches src/matter_compressor.h MC_DZ_FRAC

layout(std430, set = 0, binding = 0) readonly buffer Payloads { uint payload[]; };
layout(std430, set = 0, binding = 1) readonly buffer BlockTab { uint blkoff[]; };   // 2*nblocks
layout(std430, set = 0, binding = 2) readonly buffer RansFreq { uint freq[]; };
layout(std430, set = 0, binding = 3) readonly buffer RansCdf  { uint cdf[]; };
layout(std430, set = 0, binding = 4) readonly buffer RansSlot { uint slot[]; };
layout(std430, set = 0, binding = 5) readonly buffer StepTab  { float step_tab[]; };
layout(std430, set = 0, binding = 6) readonly buffer ScanTab  { uint scan[]; };
layout(std430, set = 0, binding = 7) readonly buffer DctBasis { float cm[]; };      // [N*N], cm[k*N+n]

layout(std430, set = 1, binding = 0) buffer OutVox { uint outv[]; };                // u8 voxels, 4/uint

layout(set = 2, binding = 0) uniform Params { uint nblocks; } P;

// shared per-block working set. Two float buffers ping-pong through the three
// iDCT passes (NOT three: 3*16KB + air would exceed the 48KB shared limit and
// silently alias). s_air is exactly N3/32 uints (512 B). Total ~33KB.
shared float s_buf0[N3];          // coef in, then pass-y out
shared float s_buf1[N3];          // pass-x out, pass-z out (final)
shared uint  s_air[N3/32];        // packed air bits (1/voxel) = 128 uints
shared uint  s_dc;
shared uint  s_has_air;

uint pbyte(uint boff) { uint w = payload[boff >> 2]; return (w >> ((boff & 3u) * 8u)) & 0xFFu; }

void main() {
    uint b   = gl_WorkGroupID.x;
    uint lid = gl_LocalInvocationID.x;
    if (b >= P.nblocks) return;

    uint base  = blkoff[2u*b + 0u];
    uint len   = blkoff[2u*b + 1u];
    uint obase = b * uint(N3);

    // zero shared coef + air (cooperative)
    for (uint i = lid; i < uint(N3); i += gl_WorkGroupSize.x) s_buf0[i] = 0.0;
    for (uint i = lid; i < uint(N3/32); i += gl_WorkGroupSize.x) s_air[i] = 0u;
    barrier();

    // ---- invocation 0: serial header + rANS + sign/escape + dequant -> s_coef
    //      and air-mask expansion -> s_air ----
    if (lid == 0u) {
        if (len < 12u) {
            s_has_air = 0u; s_dc = 0u;
        } else {
            uint flags = pbyte(base + 0u);
            uint has_air = flags & 1u;
            uint air_rle = (flags >> 1u) & 1u;
            uint dc = pbyte(base + 1u);
            uint nsym = pbyte(base+2u) | (pbyte(base+3u) << 8u);
            uint rans_len = pbyte(base+4u) | (pbyte(base+5u) << 8u);
            uint x = pbyte(base+6u) | (pbyte(base+7u)<<8u) | (pbyte(base+8u)<<16u) | (pbyte(base+9u)<<24u);
            uint air_len = pbyte(base+10u) | (pbyte(base+11u) << 8u);
            if (nsym > uint(N3)) nsym = uint(N3);
            s_dc = dc; s_has_air = has_air;

            // offsets are ABSOLUTE byte positions into the packed payload buffer
            // (base-relative -> add base), mirroring the header reads.
            uint off = base + 12u;
            uint air_off = off; if (has_air != 0u) off += air_len;
            uint rans_off = off; off += rans_len;
            uint sign_off = off;

            // rANS decode magnitudes into a temporary stored in s_buf1 (scratch;
            // the iDCT overwrites it later). dequant targets s_buf0.
            uint rpos = 0u;
            uint nz = 0u;
            for (uint i = 0u; i < nsym; ++i) {
                uint sl = x & (TOTAL - 1u);
                uint s = slot[sl];
                x = freq[s] * (x >> TBITS) + sl - cdf[s];
                while (x < RANS_L && rpos < rans_len) { x = (x << 8u) | pbyte(rans_off + rpos); rpos++; }
                s_buf1[i] = float(s);
                if (s != 0u) nz++;
            }
            uint sign_bytes = (nz + 7u) >> 3u;
            uint esc_off = sign_off + sign_bytes;

            uint signbit = 0u, esc_cur = 0u;
            for (uint i = 0u; i < nsym; ++i) {
                int s = int(s_buf1[i]);
                if (s == 0) continue;
                int a = s;
                if (s == MAXSYM - 1) {
                    uint r = 0u, shamt = 0u, bb;
                    do { bb = pbyte(esc_off + esc_cur); esc_cur++; r |= (bb & 0x7Fu) << shamt; shamt += 7u; } while ((bb & 0x80u) != 0u);
                    a = (MAXSYM - 1) + int(r);
                }
                uint sbb = pbyte(sign_off + (signbit >> 3u));
                int neg = int((sbb >> (signbit & 7u)) & 1u); signbit++;
                int lv = (neg != 0) ? -a : a;
                // dequant immediately into s_coef at the scan target.
                uint vi = scan[i];
                float av = float(a);
                float st = step_tab[vi];
                float rr = (av - 1.0) * st + MC_DZ_FRAC * st + 0.40 * st;
                s_buf0[vi] = (lv < 0) ? -rr : rr;
            }

            // air mask -> packed bits in s_air
            if (has_air != 0u) {
                if (air_rle != 0u) {
                    uint o = 0u, p = 0u; uint cur = 0u;   // 0 = material
                    while (o < uint(N3) && p < air_len) {
                        uint r = 0u, shamt = 0u, bb;
                        do { bb = pbyte(air_off + p); p++; r |= (bb & 0x7Fu) << shamt; shamt += 7u; } while ((bb & 0x80u) != 0u && p < air_len);
                        uint endi = o + r; if (endi > uint(N3)) endi = uint(N3);
                        if (cur != 0u) { for (uint i = o; i < endi; ++i) { s_air[i >> 5u] |= (1u << (i & 31u)); } }
                        o = endi; cur = 1u - cur;
                    }
                } else {
                    for (uint i = 0u; i < uint(N3); ++i) {
                        uint by = pbyte(air_off + (i >> 3u));
                        if (((by >> (i & 7u)) & 1u) != 0u) s_air[i >> 5u] |= (1u << (i & 31u));
                    }
                }
            }
        }
    }
    barrier();

    // ---- cooperative separable inverse DCT-16 (float), ping-pong buf0<->buf1 ----
    // pass along x: coef (s_buf0) -> s_buf1
    for (uint line = lid; line < uint(N*N); line += gl_WorkGroupSize.x) {
        uint boff = line * uint(N);
        for (int n = 0; n < N; ++n) {
            float acc = 0.0;
            for (int k = 0; k < N; ++k) acc += cm[k*N + n] * s_buf0[boff + uint(k)];
            s_buf1[boff + uint(n)] = acc;
        }
    }
    barrier();
    // pass along y: s_buf1 -> s_buf0
    for (uint line = lid; line < uint(N*N); line += gl_WorkGroupSize.x) {
        uint z = line / uint(N), x2 = line % uint(N);
        for (int n = 0; n < N; ++n) {
            float acc = 0.0;
            for (int k = 0; k < N; ++k) acc += cm[k*N + n] * s_buf1[(z*uint(N) + uint(k))*uint(N) + x2];
            s_buf0[(z*uint(N) + uint(n))*uint(N) + x2] = acc;
        }
    }
    barrier();
    // pass along z: s_buf0 -> s_buf1 (final)
    for (uint line = lid; line < uint(N*N); line += gl_WorkGroupSize.x) {
        uint y = line / uint(N), x2 = line % uint(N);
        for (int n = 0; n < N; ++n) {
            float acc = 0.0;
            for (int k = 0; k < N; ++k) acc += cm[k*N + n] * s_buf0[(uint(k)*uint(N) + y)*uint(N) + x2];
            s_buf1[(uint(n)*uint(N) + y)*uint(N) + x2] = acc;
        }
    }
    barrier();

    // ---- clamp + dc + air, cooperative store ----
    // Each block owns a disjoint, uint-aligned 1024-uint output region (obase =
    // b*4096 is a multiple of 4). Assign one OUTPUT UINT per thread iteration so
    // each thread packs all 4 of its voxels and writes the whole uint -> no
    // intra-uint races, no atomics.
    uint dc = s_dc; uint has_air = s_has_air;
    uint nwords = uint(N3) / 4u;                  // 1024
    for (uint w = lid; w < nwords; w += gl_WorkGroupSize.x) {
        uint packed = 0u;
        for (uint j = 0u; j < 4u; ++j) {
            uint i = w*4u + j;
            bool isair = (has_air != 0u) && (((s_air[i >> 5u] >> (i & 31u)) & 1u) != 0u);
            int v = isair ? 0 : int(floor(s_buf1[i] + 0.5)) + int(dc);
            if (v < 0) v = 0; if (v > 255) v = 255;
            packed |= (uint(v) & 0xFFu) << (j * 8u);
        }
        outv[(obase >> 2) + w] = packed;
    }
}
