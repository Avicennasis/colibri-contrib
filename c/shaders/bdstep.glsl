// bdstep.glsl: 32 weights of a GGUF block row (100 + ggml type T, the file's layout) dotted
// with x, one decoder per type, for the GEMVs that take a row over several lanes
// (chain_gemv_bd.comp, a pipeline per type; chain_moe.comp, a GGUF's routed experts). The
// types: 8 Q8_0, 12 Q4_K, 13 Q5_K, 14 Q6_K, 16 IQ2_XXS, 20 IQ4_NL, 21 IQ3_S, 22 IQ2_S,
// 23 IQ4_XS (bs_has). Q4_K and Q5_K read 16-byte steps (their blocks, 144 and 176 bytes, keep
// a row's 16-byte alignment); the others words, two for one where a block of 34, 210, 18,
// 110, 82 or 66 bytes leaves a field at an even byte (bs_ldu). A step q is weights
// 32q..32q+31 of the row at byte rb. The sums are the CPU's float decoders' in another
// order: per run of 16 weights a scale times sum(w * x), minus a min times sum(x) where the
// type has one.
// The including shader defines, before #include:
//   BS_W1(i)  uint word i of the matrix          BS_W4(i)  its uvec4 i (16-byte aligned)
//   BS_X(i)   the vec4 of x at elements 4i..4i+3  BS_G(i)   word i of blockdec_table.h's table
// and keeps `shared float bs_kvt[16]` (kvalues_iq4nl), filled by bs_kv_fill before a barrier.
#ifndef BDSTEP_GLSL
#define BDSTEP_GLSL
#include "blockdec_table.h"

bool bs_has(int t) { return t == 8 || t == 12 || t == 13 || t == 14 || t == 16 || t == 20 || t == 21 || t == 22 || t == 23; }
void bs_kv_fill(int tid) { if (tid < 16) bs_kvt[tid] = float(int(BS_G(BDT_IQ4NL + uint(tid)))); }

// four bytes from an even byte offset: one word when aligned, two otherwise; the second
// word is read only when the four bytes run into it, so a read never leaves the row
uint bs_ldu(uint b) {
    uint a = b >> 2u;
    return (b & 2u) == 0u ? BS_W1(a) : (BS_W1(a) >> 16) | (BS_W1(a + 1u) << 16);
}
float bs_f16at(uint b) { return unpackHalf2x16(BS_W1(b >> 2u) >> ((b & 2u) * 8u)).x; }
uint bs_u8at(uint b) { return (BS_W1(b >> 2u) >> ((b & 3u) * 8u)) & 0xffu; }
float bs_nib(uint w, int k) { return float(bitfieldExtract(w, k * 4, 4)); }
float bs_sbyte(uint w, int k) { return float(bitfieldExtract(int(w), k * 8, 8)); }
vec4 bs_nib4(uint w, int sh) {   // the nibbles at bit sh of each of w's bytes
    return vec4(bitfieldExtract(w, sh, 4), bitfieldExtract(w, sh + 8, 4), bitfieldExtract(w, sh + 16, 4), bitfieldExtract(w, sh + 24, 4));
}
vec4 bs_sb4(uint w) { return vec4(bitfieldExtract(int(w), 0, 8), bitfieldExtract(int(w), 8, 8), bitfieldExtract(int(w), 16, 8), bitfieldExtract(int(w), 24, 8)); }
vec4 bs_kv4(uint w, int sh) {    // kvalues of the nibbles at bit sh of each byte
    return vec4(bs_kvt[bitfieldExtract(w, sh, 4)], bs_kvt[bitfieldExtract(w, sh + 8, 4)], bs_kvt[bitfieldExtract(w, sh + 16, 4)], bs_kvt[bitfieldExtract(w, sh + 24, 4)]);
}
// a grid row's bytes (unsigned) with the signs of bits sg[0..3] (for bytes 0..3 of the row)
vec4 bs_sgn4(uint g, uint sg) {
    vec4 v = vec4(g & 255u, (g >> 8) & 255u, (g >> 16) & 255u, g >> 24);
    return v * (vec4(1.0) - 2.0 * vec4(sg & 1u, (sg >> 1) & 1u, (sg >> 2) & 1u, (sg >> 3) & 1u));
}
float bs_sum4(vec4 v) { return v.x + v.y + v.z + v.w; }

// Q4_K / Q5_K: byte j of the block's 12 scale bytes (words 1..3), and the 6-bit scale and
// min of group g from them (in registers: no array indexed at run time)
uint bs_kb(uvec4 hd, uint j) { uint w = j < 4u ? hd.y : j < 8u ? hd.z : hd.w; return (w >> (8u * (j & 3u))) & 255u; }
uvec2 bs_k4sm(uvec4 hd, uint g) {
    if (g < 4u) return uvec2(bs_kb(hd, g) & 63u, bs_kb(hd, g + 4u) & 63u);
    return uvec2((bs_kb(hd, g + 4u) & 15u) | ((bs_kb(hd, g - 4u) >> 6) << 4), (bs_kb(hd, g + 4u) >> 4) | ((bs_kb(hd, g) >> 6) << 4));
}

// sum over step q's 32 weights (of the row at byte rb, ggml type t) of w * x
float bs_step(int t, uint rb, uint q) {
    if (t == 12 || t == 13) {   // 8 steps a block: chunk c (64 weights), half h of its 32 bytes
        uint b = q >> 3, c = (q >> 1) & 3u, h = q & 1u;
        uint base = rb + b * (t == 12 ? 144u : 176u);
        uvec4 hd = BS_W4(base >> 4);
        vec2 dm = unpackHalf2x16(hd.x);
        uvec2 lo = bs_k4sm(hd, 2u * c), hi = bs_k4sm(hd, 2u * c + 1u);
        uvec4 qs = BS_W4((base + (t == 12 ? 16u : 48u) + 32u * c + 16u * h) >> 4);
        uvec4 qh = t == 13 ? BS_W4((base + 16u + 16u * h) >> 4) : uvec4(0u);
        uint e = b * 64u + c * 16u + h * 4u;   // the low run's first vec4 of x (weight 256b + 64c + 16h)
        float al = 0.0, ah = 0.0, xl = 0.0, xh = 0.0;
        [[unroll]] for (int k = 0; k < 4; k++) {
            uint wd = qs[k];
            vec4 vl = bs_nib4(wd, 0), vh = bs_nib4(wd, 4);
            if (t == 13) {
                uint hb = qh[k] >> (2u * c);
                vl += 16.0 * vec4(hb & 1u, (hb >> 8) & 1u, (hb >> 16) & 1u, (hb >> 24) & 1u);
                vh += 16.0 * vec4((hb >> 1) & 1u, (hb >> 9) & 1u, (hb >> 17) & 1u, (hb >> 25) & 1u);
            }
            vec4 x0 = BS_X(e + uint(k)), x1 = BS_X(e + 8u + uint(k));
            al += dot(vl, x0); ah += dot(vh, x1); xl += bs_sum4(x0); xh += bs_sum4(x1);
        }
        return dm.x * (float(lo.x) * al + float(hi.x) * ah) - dm.y * (float(lo.y) * xl + float(hi.y) * xh);
    }
    if (t == 14) {   // Q6_K: 8 steps a block: half n (128 weights), bytes l0..l0+7 of its 32-runs
        uint b = q >> 3, n = (q >> 2) & 1u, l0 = (q & 3u) * 8u;
        uint base = rb + b * 210u;
        uint qa0 = bs_ldu(base + 64u * n + l0), qa1 = bs_ldu(base + 64u * n + l0 + 4u);
        uint qb0 = bs_ldu(base + 64u * n + 32u + l0), qb1 = bs_ldu(base + 64u * n + 32u + l0 + 4u);
        uint qh0 = bs_ldu(base + 128u + 32u * n + l0), qh1 = bs_ldu(base + 128u + 32u * n + l0 + 4u);
        uint s0 = bs_ldu(base + 192u + 8u * n), s1 = bs_ldu(base + 196u + 8u * n);
        uint g = l0 >> 4;   // the scales of runs m = 0..3: bytes 2m + g of s0|s1
        float sc0 = bs_sbyte(s0, int(g)), sc1 = bs_sbyte(s0, int(2u + g)), sc2 = bs_sbyte(s1, int(g)), sc3 = bs_sbyte(s1, int(2u + g));
        uint e = b * 64u + n * 32u + l0 / 4u;   // weight 256b + 128n + l0
        float a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
        [[unroll]] for (int k = 0; k < 2; k++) {
            uint la = k == 0 ? qa0 : qa1, lb = k == 0 ? qb0 : qb1, hh = k == 0 ? qh0 : qh1;
            vec4 h0 = vec4(bitfieldExtract(hh, 0, 2), bitfieldExtract(hh, 8, 2), bitfieldExtract(hh, 16, 2), bitfieldExtract(hh, 24, 2));
            vec4 h1 = vec4(bitfieldExtract(hh, 2, 2), bitfieldExtract(hh, 10, 2), bitfieldExtract(hh, 18, 2), bitfieldExtract(hh, 26, 2));
            vec4 h2 = vec4(bitfieldExtract(hh, 4, 2), bitfieldExtract(hh, 12, 2), bitfieldExtract(hh, 20, 2), bitfieldExtract(hh, 28, 2));
            vec4 h3 = vec4(bitfieldExtract(hh, 6, 2), bitfieldExtract(hh, 14, 2), bitfieldExtract(hh, 22, 2), bitfieldExtract(hh, 30, 2));
            a0 += dot(bs_nib4(la, 0) + 16.0 * h0 - 32.0, BS_X(e + uint(k)));
            a1 += dot(bs_nib4(lb, 0) + 16.0 * h1 - 32.0, BS_X(e + 8u + uint(k)));
            a2 += dot(bs_nib4(la, 4) + 16.0 * h2 - 32.0, BS_X(e + 16u + uint(k)));
            a3 += dot(bs_nib4(lb, 4) + 16.0 * h3 - 32.0, BS_X(e + 24u + uint(k)));
        }
        return bs_f16at(base + 208u) * (sc0 * a0 + sc1 * a1 + sc2 * a2 + sc3 * a3);
    }
    if (t == 8) {   // Q8_0: the step is the block
        uint base = rb + q * 34u;
        float a = 0.0;
        [[unroll]] for (int k = 0; k < 8; k++) a += dot(bs_sb4(bs_ldu(base + 2u + 4u * uint(k))), BS_X(q * 8u + uint(k)));
        return bs_f16at(base) * a;
    }
    if (t == 20) {   // IQ4_NL: the step is the block, low nibbles weights 0..15, high 16..31
        uint base = rb + q * 18u;
        float a = 0.0;
        [[unroll]] for (int k = 0; k < 4; k++) {
            uint wd = bs_ldu(base + 2u + 4u * uint(k));
            a += dot(bs_kv4(wd, 0), BS_X(q * 8u + uint(k))) + dot(bs_kv4(wd, 4), BS_X(q * 8u + 4u + uint(k)));
        }
        return bs_f16at(base) * a;
    }
    if (t == 23) {   // IQ4_XS: 8 steps a block, one 6-bit scale each
        uint b = q >> 3, ib = q & 7u, base = rb + b * 136u;
        uint w0 = BS_W1(base >> 2), w2 = BS_W1((base >> 2) + 1u);
        uint ls = ((w2 >> (8u * (ib >> 1) + 4u * (ib & 1u))) & 15u) | (((w0 >> (16u + 2u * ib)) & 3u) << 4);
        float a = 0.0;
        uint qa = (base + 8u + 16u * ib) >> 2;
        [[unroll]] for (int k = 0; k < 4; k++) {
            uint wd = BS_W1(qa + uint(k));
            a += dot(bs_kv4(wd, 0), BS_X(q * 8u + uint(k))) + dot(bs_kv4(wd, 4), BS_X(q * 8u + 4u + uint(k)));
        }
        return unpackHalf2x16(w0).x * float(int(ls) - 32) * a;
    }
    if (t == 21) {   // IQ3_S: 8 steps a block; 8 chunks of 4 weights, a 9-bit grid index each
        uint b = q >> 3, ib = q & 7u, base = rb + b * 110u;
        uint i0 = bs_ldu(base + 2u + 8u * ib), i1 = bs_ldu(base + 6u + 8u * ib);
        uint hb = bs_u8at(base + 66u + ib), sg = bs_ldu(base + 74u + 4u * ib);
        float a = 0.0;
        [[unroll]] for (int l = 0; l < 8; l++) {
            uint idx = ((l < 4 ? i0 : i1) >> (8u * uint(l & 3))) & 255u;
            idx |= ((hb >> uint(l)) & 1u) << 8;
            a += dot(bs_sgn4(BS_G(BDT_IQ3S + idx), sg >> (4u * uint(l))), BS_X(q * 8u + uint(l)));
        }
        uint sc = (bs_u8at(base + 106u + (ib >> 1)) >> (4u * (ib & 1u))) & 15u;
        return bs_f16at(base) * float(1u + 2u * sc) * a;
    }
    if (t == 22) {   // IQ2_S: 8 steps a block; 4 chunks of 8 weights, a 10-bit grid index each
        uint b = q >> 3, ib = q & 7u, base = rb + b * 82u;
        uint ix = bs_ldu(base + 2u + 4u * ib), sg = bs_ldu(base + 34u + 4u * ib);
        uint hb = bs_u8at(base + 66u + ib), scb = bs_u8at(base + 74u + ib);
        float d = bs_f16at(base), a0 = 0.0, a1 = 0.0;
        [[unroll]] for (int l = 0; l < 4; l++) {
            uint idx = ((ix >> (8u * uint(l))) & 255u) | (((hb >> (2u * uint(l))) & 3u) << 8);
            uint s8 = sg >> (8u * uint(l));
            float t = dot(bs_sgn4(BS_G(BDT_IQ2S + 2u * idx), s8), BS_X(q * 8u + 2u * uint(l))) +
                      dot(bs_sgn4(BS_G(BDT_IQ2S + 2u * idx + 1u), s8 >> 4), BS_X(q * 8u + 2u * uint(l) + 1u));
            if (l < 2) a0 += t; else a1 += t;
        }
        return d * 0.25 * ((0.5 + float(scb & 15u)) * a0 + (0.5 + float(scb >> 4)) * a1);
    }
    if (t == 16) {   // IQ2_XXS: 8 steps a block; 4 grid indices, their signs and a 4-bit scale
        uint b = q >> 3, ib = q & 7u, base = rb + b * 66u;
        uint i0 = bs_ldu(base + 2u + 8u * ib), i1 = bs_ldu(base + 6u + 8u * ib);
        float a = 0.0;
        [[unroll]] for (int l = 0; l < 4; l++) {
            uint idx = (i0 >> (8u * uint(l))) & 255u, s8 = BS_G(BDT_KSIGNS + ((i1 >> (7u * uint(l))) & 127u));
            a += dot(bs_sgn4(BS_G(BDT_IQ2XXS + 2u * idx), s8), BS_X(q * 8u + 2u * uint(l))) +
                 dot(bs_sgn4(BS_G(BDT_IQ2XXS + 2u * idx + 1u), s8 >> 4), BS_X(q * 8u + 2u * uint(l) + 1u));
        }
        return bs_f16at(base) * (0.5 + float(i1 >> 28)) * 0.25 * a;
    }
    return 0.0;
}

#endif
