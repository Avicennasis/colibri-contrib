/* ggml_quants.h -- the GGUF weight formats, decoded natively.
 *
 * A GGUF file stores each tensor as rows of fixed-size blocks in one of the
 * ggml types (Q4_K, IQ2_XXS, Q8_0, MXFP4 ...). This header lets colibri use
 * those bytes as they are, with no second quantization: every type has a
 * row decoder that reproduces the reference dequantization exactly, and a
 * matrix-vector product built on it.
 *
 * Exactness contract: gq_dequant_row() returns, bit for bit, the float32
 * values the reference (gguf-py, gguf.quants.dequantize) computes. The
 * arithmetic is written in the same order as the reference and the file is
 * compiled without floating-point contraction, so a*b - c is never fused
 * into an FMA with a different rounding. tests/test_ggml_quants.py holds
 * this for every type on random blocks.
 *
 * Block layouts follow the GGUF specification (ggml-common.h in the ggml
 * project). Codebook tables live in ggml_grids.h (generated).
 */
#ifndef GGML_QUANTS_H
#define GGML_QUANTS_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ggml_grids.h"

#if defined(__AVX2__) || defined(__F16C__)
#include <immintrin.h>
#endif

#if defined(__clang__)
#define GQ_EXACT_BEGIN _Pragma("clang fp contract(off)")
#define GQ_EXACT_FN
#elif defined(__GNUC__)
#define GQ_EXACT_BEGIN
#define GQ_EXACT_FN __attribute__((optimize("fp-contract=off")))
#else
#define GQ_EXACT_BEGIN
#define GQ_EXACT_FN
#endif

/* ggml type ids, as stored in GGUF tensor infos. */
enum {
    GQ_F32 = 0, GQ_F16 = 1, GQ_Q4_0 = 2, GQ_Q4_1 = 3, GQ_Q5_0 = 6, GQ_Q5_1 = 7,
    GQ_Q8_0 = 8, GQ_Q8_1 = 9, GQ_Q2_K = 10, GQ_Q3_K = 11, GQ_Q4_K = 12,
    GQ_Q5_K = 13, GQ_Q6_K = 14, GQ_Q8_K = 15, GQ_IQ2_XXS = 16, GQ_IQ2_XS = 17,
    GQ_IQ3_XXS = 18, GQ_IQ1_S = 19, GQ_IQ4_NL = 20, GQ_IQ3_S = 21, GQ_IQ2_S = 22,
    GQ_IQ4_XS = 23, GQ_I8 = 24, GQ_I16 = 25, GQ_I32 = 26, GQ_I64 = 27,
    GQ_F64 = 28, GQ_IQ1_M = 29, GQ_BF16 = 30, GQ_TQ1_0 = 34, GQ_TQ2_0 = 35,
    GQ_MXFP4 = 39, GQ_TYPE_MAX = 40
};

typedef struct { const char *name; int blck; int bytes; int supported; } gq_type_info;

static const gq_type_info GQ_TYPES[GQ_TYPE_MAX] = {
    [GQ_F32]     = { "F32",     1,   4, 1 },
    [GQ_F16]     = { "F16",     1,   2, 1 },
    [GQ_Q4_0]    = { "Q4_0",    32,  18, 1 },
    [GQ_Q4_1]    = { "Q4_1",    32,  20, 1 },
    [GQ_Q5_0]    = { "Q5_0",    32,  22, 1 },
    [GQ_Q5_1]    = { "Q5_1",    32,  24, 1 },
    [GQ_Q8_0]    = { "Q8_0",    32,  34, 1 },
    [GQ_Q8_1]    = { "Q8_1",    32,  36, 0 },
    [GQ_Q2_K]    = { "Q2_K",    256, 84, 1 },
    [GQ_Q3_K]    = { "Q3_K",    256, 110, 1 },
    [GQ_Q4_K]    = { "Q4_K",    256, 144, 1 },
    [GQ_Q5_K]    = { "Q5_K",    256, 176, 1 },
    [GQ_Q6_K]    = { "Q6_K",    256, 210, 1 },
    [GQ_Q8_K]    = { "Q8_K",    256, 292, 0 },
    [GQ_IQ2_XXS] = { "IQ2_XXS", 256, 66, 1 },
    [GQ_IQ2_XS]  = { "IQ2_XS",  256, 74, 1 },
    [GQ_IQ3_XXS] = { "IQ3_XXS", 256, 98, 1 },
    [GQ_IQ1_S]   = { "IQ1_S",   256, 50, 1 },
    [GQ_IQ4_NL]  = { "IQ4_NL",  32,  18, 1 },
    [GQ_IQ3_S]   = { "IQ3_S",   256, 110, 1 },
    [GQ_IQ2_S]   = { "IQ2_S",   256, 82, 1 },
    [GQ_IQ4_XS]  = { "IQ4_XS",  256, 136, 1 },
    [GQ_I8]      = { "I8",      1,   1, 0 },
    [GQ_I16]     = { "I16",     1,   2, 0 },
    [GQ_I32]     = { "I32",     1,   4, 0 },
    [GQ_I64]     = { "I64",     1,   8, 0 },
    [GQ_F64]     = { "F64",     1,   8, 0 },
    [GQ_IQ1_M]   = { "IQ1_M",   256, 56, 1 },
    [GQ_BF16]    = { "BF16",    1,   2, 1 },
    [GQ_TQ1_0]   = { "TQ1_0",   256, 54, 1 },
    [GQ_TQ2_0]   = { "TQ2_0",   256, 66, 1 },
    [GQ_MXFP4]   = { "MXFP4",   32,  17, 1 },
};

static inline int gq_type_ok(int t) { return t >= 0 && t < GQ_TYPE_MAX && GQ_TYPES[t].blck > 0 && GQ_TYPES[t].supported; }
static inline const char *gq_type_name(int t) { return (t >= 0 && t < GQ_TYPE_MAX && GQ_TYPES[t].name) ? GQ_TYPES[t].name : "?"; }
/* Bytes of one row of n elements, or -1 when n is not a whole number of blocks. */
static inline int64_t gq_row_bytes(int t, int64_t n) {
    if (t < 0 || t >= GQ_TYPE_MAX || GQ_TYPES[t].blck <= 0) return -1;
    if (n % GQ_TYPES[t].blck) return -1;
    return n / GQ_TYPES[t].blck * GQ_TYPES[t].bytes;
}

/* ---- scalar conversions (exact) ---- */
static inline float gq_f16(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, mant = h & 0x3ff, f;
    if (exp == 0) {
        if (mant == 0) f = sign;
        else {
            int e = -1;
            do { e++; mant <<= 1; } while (!(mant & 0x400));
            mant &= 0x3ff;
            f = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 31) f = sign | 0x7f800000u | (mant << 13);
    else f = sign | ((exp + 112) << 23) | (mant << 13);
    float r; memcpy(&r, &f, 4); return r;
#endif
}
static inline float gq_bf16(uint16_t h) { uint32_t u = (uint32_t)h << 16; float r; memcpy(&r, &u, 4); return r; }
static inline uint16_t gq_rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static inline uint32_t gq_rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
/* ggml_e8m0_to_fp32_half: the MXFP4 block exponent, already halved because
 * the value table holds e2m1 values doubled. */
static inline float gq_e8m0_half(uint8_t x) {
    uint32_t bits = x < 2 ? (0x00200000u << x) : ((uint32_t)(x - 1) << 23);
    float r; memcpy(&r, &bits, 4); return r;
}

/* ---- per-type block decoders: one block in, blck floats out ---- */

GQ_EXACT_FN static void gq_blk_q4_0(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
    for (int j = 0; j < 16; j++) {
        y[j]      = d * (float)((int)(qs[j] & 0x0F) - 8);
        y[j + 16] = d * (float)((int)(qs[j] >> 4) - 8);
    }
}
GQ_EXACT_FN static void gq_blk_q4_1(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)), m = gq_f16(gq_rd16(b + 2)); const uint8_t *qs = b + 4;
    for (int j = 0; j < 16; j++) {
        y[j]      = d * (float)(qs[j] & 0x0F) + m;
        y[j + 16] = d * (float)(qs[j] >> 4) + m;
    }
}
GQ_EXACT_FN static void gq_blk_q5_0(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const uint32_t qh = gq_rd32(b + 2); const uint8_t *qs = b + 6;
    for (int j = 0; j < 16; j++) {
        int lo = (qs[j] & 0x0F) | (int)(((qh >> j) & 1) << 4);
        int hi = (qs[j] >> 4) | (int)(((qh >> (j + 16)) & 1) << 4);
        y[j] = d * (float)(lo - 16); y[j + 16] = d * (float)(hi - 16);
    }
}
GQ_EXACT_FN static void gq_blk_q5_1(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)), m = gq_f16(gq_rd16(b + 2));
    const uint32_t qh = gq_rd32(b + 4); const uint8_t *qs = b + 8;
    for (int j = 0; j < 16; j++) {
        int lo = (qs[j] & 0x0F) | (int)(((qh >> j) & 1) << 4);
        int hi = (qs[j] >> 4) | (int)(((qh >> (j + 16)) & 1) << 4);
        y[j] = d * (float)lo + m; y[j + 16] = d * (float)hi + m;
    }
}
GQ_EXACT_FN static void gq_blk_q8_0(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const int8_t *qs = (const int8_t *)(b + 2);
    for (int j = 0; j < 32; j++) y[j] = (float)qs[j] * d;
}
GQ_EXACT_FN static void gq_blk_q2_k(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const uint8_t *sc = b, *qs = b + 16;
    const float d = gq_f16(gq_rd16(b + 80)), dmin = gq_f16(gq_rd16(b + 82));
    for (int i = 0; i < 256; i++) {
        int c = i >> 7, s = (i >> 5) & 3, j = i & 31, g = i >> 4;
        int q = (qs[c * 32 + j] >> (2 * s)) & 3;
        float dl = d * (float)(sc[g] & 0xF), ml = dmin * (float)(sc[g] >> 4);
        y[i] = dl * (float)q - ml;
    }
}
GQ_EXACT_FN static void gq_blk_q3_k(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const uint8_t *hm = b, *qs = b + 32, *s = b + 96;
    const float d = gq_f16(gq_rd16(b + 108));
    float dl[16];
    for (int k = 0; k < 16; k++) {
        int l = k < 8 ? (s[k] & 0x0F) : (s[k - 8] >> 4);
        int h = (s[8 + (k & 3)] >> (2 * (k >> 2))) & 3;
        dl[k] = d * (float)((l | (h << 4)) - 32);
    }
    for (int i = 0; i < 256; i++) {
        int c = i >> 7, sh = (i >> 5) & 3, j = i & 31;
        int ql = (qs[c * 32 + j] >> (2 * sh)) & 3;
        int hb = (hm[i & 31] >> (i >> 5)) & 1;
        int q = ql - ((hb ^ 1) << 2);
        y[i] = dl[i >> 4] * (float)q;
    }
}
static inline void gq_scale_min_k4(const uint8_t *s, int g, int *sc, int *mn) {
    if (g < 4) { *sc = s[g] & 0x3F; *mn = s[g + 4] & 0x3F; }
    else {
        *sc = (s[g + 4] & 0x0F) | ((s[g - 4] >> 2) & 0x30);
        *mn = (s[g + 4] >> 4)   | ((s[g] >> 2) & 0x30);
    }
}
GQ_EXACT_FN static void gq_blk_q4_k(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)), dmin = gq_f16(gq_rd16(b + 2));
    const uint8_t *sc = b + 4, *qs = b + 16;
    for (int g = 0; g < 8; g++) {
        int s, m; gq_scale_min_k4(sc, g, &s, &m);
        const float d1 = d * (float)s, m1 = dmin * (float)m;
        const uint8_t *q = qs + (g >> 1) * 32; int sh = (g & 1) * 4;
        for (int j = 0; j < 32; j++) y[g * 32 + j] = d1 * (float)((q[j] >> sh) & 0x0F) - m1;
    }
}
GQ_EXACT_FN static void gq_blk_q5_k(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)), dmin = gq_f16(gq_rd16(b + 2));
    const uint8_t *sc = b + 4, *qh = b + 16, *qs = b + 48;
    for (int g = 0; g < 8; g++) {
        int s, m; gq_scale_min_k4(sc, g, &s, &m);
        const float d1 = d * (float)s, m1 = dmin * (float)m;
        const uint8_t *q = qs + (g >> 1) * 32; int sh = (g & 1) * 4;
        for (int j = 0; j < 32; j++) {
            int v = ((q[j] >> sh) & 0x0F) | (((qh[j] >> g) & 1) << 4);
            y[g * 32 + j] = d1 * (float)v - m1;
        }
    }
}
GQ_EXACT_FN static void gq_blk_q6_k(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const uint8_t *ql = b, *qh = b + 128; const int8_t *sc = (const int8_t *)(b + 192);
    const float d = gq_f16(gq_rd16(b + 208));
    for (int i = 0; i < 256; i++) {
        int c = i >> 7, r = i & 127;                 /* chunk of 128 */
        int nib = r >> 6, k = r & 63;                /* ql: low nibbles 0..63, high 64..127 */
        int lo = (ql[c * 64 + k] >> (4 * nib)) & 0x0F;
        int s = r >> 5, j = r & 31;                  /* qh: 2 bits, 4 shifts of 32 */
        int hi = (qh[c * 32 + j] >> (2 * s)) & 3;
        int q = (lo | (hi << 4)) - 32;
        y[i] = (d * (float)sc[i >> 4]) * (float)q;
    }
}
GQ_EXACT_FN static void gq_blk_iq2_xxs(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
    for (int ib = 0; ib < 8; ib++) {
        uint32_t a0 = gq_rd32(qs + 8 * ib), a1 = gq_rd32(qs + 8 * ib + 4);
        const float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            const uint8_t *grid = GGML_IQ2XXS_GRID + 8 * ((a0 >> (8 * l)) & 0xFF);
            uint8_t signs = GGML_KSIGNS_IQ2XS[(a1 >> (7 * l)) & 127];
            for (int j = 0; j < 8; j++) {
                float v = db * (float)grid[j];
                y[ib * 32 + l * 8 + j] = (signs >> j) & 1 ? -v : v;
            }
        }
    }
}
GQ_EXACT_FN static void gq_blk_iq2_xs(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *sc = b + 66;
    for (int k = 0; k < 32; k++) {                   /* 32 groups of 8 */
        uint16_t q = gq_rd16(qs + 2 * k);
        int si = k >> 1;                              /* one 4-bit scale per 16 */
        int s = (sc[si >> 1] >> (4 * (si & 1))) & 0x0F;
        const float db = d * (0.5f + (float)s) * 0.25f;
        const uint8_t *grid = GGML_IQ2XS_GRID + 8 * (q & 511);
        uint8_t signs = GGML_KSIGNS_IQ2XS[q >> 9];
        for (int j = 0; j < 8; j++) {
            float v = db * (float)grid[j];
            y[k * 8 + j] = (signs >> j) & 1 ? -v : v;
        }
    }
}
GQ_EXACT_FN static void gq_blk_iq2_s(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b));
    const uint8_t *qs = b + 2, *sg = b + 34, *qh = b + 66, *sc = b + 74;
    for (int k = 0; k < 32; k++) {
        int si = k >> 1;
        int s = (sc[si >> 1] >> (4 * (si & 1))) & 0x0F;
        const float db = d * (0.5f + (float)s) * 0.25f;
        int idx = qs[k] | (((qh[k >> 2] >> (2 * (k & 3))) & 3) << 8);
        const uint8_t *grid = GGML_IQ2S_GRID + 8 * idx;
        for (int j = 0; j < 8; j++) {
            float v = db * (float)grid[j];
            y[k * 8 + j] = (sg[k] >> j) & 1 ? -v : v;
        }
    }
}
GQ_EXACT_FN static void gq_blk_iq3_xxs(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *ss = b + 66;
    for (int ib = 0; ib < 8; ib++) {
        uint32_t a = gq_rd32(ss + 4 * ib);
        const float db = d * (0.5f + (float)(a >> 28)) * 0.5f;
        for (int l = 0; l < 4; l++) {
            uint8_t signs = GGML_KSIGNS_IQ2XS[(a >> (7 * l)) & 127];
            const uint8_t *g1 = GGML_IQ3XXS_GRID + 4 * qs[8 * ib + 2 * l];
            const uint8_t *g2 = GGML_IQ3XXS_GRID + 4 * qs[8 * ib + 2 * l + 1];
            for (int j = 0; j < 4; j++) {
                float v1 = db * (float)g1[j], v2 = db * (float)g2[j];
                y[ib * 32 + l * 8 + j]     = (signs >> j) & 1 ? -v1 : v1;
                y[ib * 32 + l * 8 + j + 4] = (signs >> (j + 4)) & 1 ? -v2 : v2;
            }
        }
    }
}
GQ_EXACT_FN static void gq_blk_iq3_s(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b));
    const uint8_t *qs = b + 2, *qh = b + 66, *sg = b + 74, *sc = b + 106;
    for (int k = 0; k < 64; k++) {                   /* 64 groups of 4 */
        int ib = k >> 3;                              /* 32-element block */
        int s = (sc[ib >> 1] >> (4 * (ib & 1))) & 0x0F;
        const float db = d * (float)(1 + 2 * s);
        int idx = qs[k] | (((qh[k >> 3] >> (k & 7)) & 1) << 8);
        const uint8_t *grid = GGML_IQ3S_GRID + 4 * idx;
        for (int j = 0; j < 4; j++) {
            int e = k * 4 + j;
            float v = db * (float)grid[j];
            y[e] = (sg[e >> 3] >> (e & 7)) & 1 ? -v : v;
        }
    }
}
#define GQ_IQ1_DELTA 0.125f
GQ_EXACT_FN static void gq_blk_iq1_s(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *qhp = b + 34;
    for (int ib = 0; ib < 8; ib++) {
        uint16_t qh = gq_rd16(qhp + 2 * ib);
        const float dl = d * (float)(2 * ((qh >> 12) & 7) + 1);
        const float delta = (qh & 0x8000) ? -GQ_IQ1_DELTA : GQ_IQ1_DELTA;
        for (int l = 0; l < 4; l++) {
            int idx = qs[4 * ib + l] | (((qh >> (3 * l)) & 7) << 8);
            const int8_t *grid = GGML_IQ1S_GRID + 8 * idx;
            for (int j = 0; j < 8; j++) y[ib * 32 + l * 8 + j] = dl * ((float)grid[j] + delta);
        }
    }
}
GQ_EXACT_FN static void gq_blk_iq1_m(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const uint8_t *qs = b, *qh = b + 32, *scp = b + 48;
    uint16_t sc[4]; for (int i = 0; i < 4; i++) sc[i] = gq_rd16(scp + 2 * i);
    uint16_t dh = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
    const float d = gq_f16(dh);
    for (int k = 0; k < 32; k++) {                   /* 32 groups of 8 */
        int s16 = k >> 1;                             /* 3-bit scale per 16 */
        int s = (sc[s16 >> 2] >> (3 * (s16 & 3))) & 7;
        const float dl = d * (float)(2 * s + 1);
        int h = (qh[k >> 1] >> (4 * (k & 1))) & 0x0F;
        int idx = qs[k] | ((h & 7) << 8);
        const float delta = (h & 8) ? -GQ_IQ1_DELTA : GQ_IQ1_DELTA;
        const int8_t *grid = GGML_IQ1S_GRID + 8 * idx;
        for (int j = 0; j < 8; j++) y[k * 8 + j] = dl * ((float)grid[j] + delta);
    }
}
GQ_EXACT_FN static void gq_blk_iq4_nl(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
    for (int j = 0; j < 16; j++) {
        y[j]      = d * (float)GGML_KVALUES_IQ4NL[qs[j] & 0x0F];
        y[j + 16] = d * (float)GGML_KVALUES_IQ4NL[qs[j] >> 4];
    }
}
GQ_EXACT_FN static void gq_blk_iq4_xs(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const uint16_t sh = gq_rd16(b + 2);
    const uint8_t *sl = b + 4, *qs = b + 8;
    for (int ib = 0; ib < 8; ib++) {
        int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((sh >> (2 * ib)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        const uint8_t *q = qs + 16 * ib;
        for (int j = 0; j < 16; j++) {
            y[ib * 32 + j]      = dl * (float)GGML_KVALUES_IQ4NL[q[j] & 0x0F];
            y[ib * 32 + j + 16] = dl * (float)GGML_KVALUES_IQ4NL[q[j] >> 4];
        }
    }
}
GQ_EXACT_FN static void gq_blk_mxfp4(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_e8m0_half(b[0]); const uint8_t *qs = b + 1;
    for (int j = 0; j < 16; j++) {
        y[j]      = d * (float)GGML_KVALUES_MXFP4[qs[j] & 0x0F];
        y[j + 16] = d * (float)GGML_KVALUES_MXFP4[qs[j] >> 4];
    }
}
GQ_EXACT_FN static void gq_blk_tq1_0(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    static const uint8_t pow3[5] = { 1, 3, 9, 27, 81 };
    const uint8_t *qs = b, *qh = b + 48; const float d = gq_f16(gq_rd16(b + 52));
    int o = 0;
    for (int n = 0; n < 5; n++) for (int j = 0; j < 32; j++) {
        uint8_t v = (uint8_t)(qs[j] * pow3[n]); y[o++] = d * (float)((int)(((uint16_t)v * 3) >> 8) - 1);
    }
    for (int n = 0; n < 5; n++) for (int j = 0; j < 16; j++) {
        uint8_t v = (uint8_t)(qs[32 + j] * pow3[n]); y[o++] = d * (float)((int)(((uint16_t)v * 3) >> 8) - 1);
    }
    for (int n = 0; n < 4; n++) for (int j = 0; j < 4; j++) {
        uint8_t v = (uint8_t)(qh[j] * pow3[n]); y[o++] = d * (float)((int)(((uint16_t)v * 3) >> 8) - 1);
    }
}
GQ_EXACT_FN static void gq_blk_tq2_0(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const uint8_t *qs = b; const float d = gq_f16(gq_rd16(b + 64));
    for (int i = 0; i < 256; i++) {
        int c = i >> 7, s = (i >> 5) & 3, j = i & 31;
        y[i] = d * (float)((int)((qs[c * 32 + j] >> (2 * s)) & 3) - 1);
    }
}

typedef void (*gq_blk_fn)(const uint8_t *, float *);
static inline gq_blk_fn gq_blk_decoder(int t) {
    switch (t) {
    case GQ_Q4_0: return gq_blk_q4_0;     case GQ_Q4_1: return gq_blk_q4_1;
    case GQ_Q5_0: return gq_blk_q5_0;     case GQ_Q5_1: return gq_blk_q5_1;
    case GQ_Q8_0: return gq_blk_q8_0;     case GQ_Q2_K: return gq_blk_q2_k;
    case GQ_Q3_K: return gq_blk_q3_k;     case GQ_Q4_K: return gq_blk_q4_k;
    case GQ_Q5_K: return gq_blk_q5_k;     case GQ_Q6_K: return gq_blk_q6_k;
    case GQ_IQ2_XXS: return gq_blk_iq2_xxs; case GQ_IQ2_XS: return gq_blk_iq2_xs;
    case GQ_IQ2_S: return gq_blk_iq2_s;   case GQ_IQ3_XXS: return gq_blk_iq3_xxs;
    case GQ_IQ3_S: return gq_blk_iq3_s;   case GQ_IQ1_S: return gq_blk_iq1_s;
    case GQ_IQ1_M: return gq_blk_iq1_m;   case GQ_IQ4_NL: return gq_blk_iq4_nl;
    case GQ_IQ4_XS: return gq_blk_iq4_xs; case GQ_MXFP4: return gq_blk_mxfp4;
    case GQ_TQ1_0: return gq_blk_tq1_0;   case GQ_TQ2_0: return gq_blk_tq2_0;
    default: return NULL;
    }
}

/* Decode n elements (a whole number of blocks) of type t into dst.
 * Returns 0, or -1 for an unsupported type or a partial block. */
static int gq_dequant_row(int t, const uint8_t *src, float *dst, int64_t n) {
    if (t == GQ_F32) { memcpy(dst, src, (size_t)n * 4); return 0; }
    if (t == GQ_F16) { for (int64_t i = 0; i < n; i++) dst[i] = gq_f16(gq_rd16(src + 2 * i)); return 0; }
    if (t == GQ_BF16) { for (int64_t i = 0; i < n; i++) dst[i] = gq_bf16(gq_rd16(src + 2 * i)); return 0; }
    gq_blk_fn f = gq_blk_decoder(t);
    if (!f || n % GQ_TYPES[t].blck) return -1;
    const int bl = GQ_TYPES[t].blck, by = GQ_TYPES[t].bytes;
    for (int64_t i = 0; i < n / bl; i++) f(src + i * by, dst + i * bl);
    return 0;
}


/* ---- matrix x vector on the decoded rows ------------------------------
 * y[s*O + o] = sum_i W[o, i] * x[s*I + i], W stored as O rows of type t.
 * Reference path: each row is decoded to float once and dotted with every
 * activation row. Rows are split across OpenMP threads; each thread owns a
 * row buffer. Deterministic: the per-row sum order is fixed. */
static inline float gq_dot_f32(const float *a, const float *b, int64_t n) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    s0 = _mm256_add_ps(s0, s1);
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h)); h = _mm_add_ss(h, _mm_movehdup_ps(h));
    float r = _mm_cvtss_f32(h);
    for (; i < n; i++) r += a[i] * b[i];
    return r;
#else
    float r = 0.f; for (int64_t i = 0; i < n; i++) r += a[i] * b[i]; return r;
#endif
}

/* Returns 0, or -1 for an unsupported type / shape (nothing written). */
static int gq_matmul(float *y, const float *x, const uint8_t *W, int t, int S, int I, int O) {
    const int64_t rb = gq_row_bytes(t, I);
    if (rb < 0 || (t != GQ_F32 && t != GQ_F16 && t != GQ_BF16 && !gq_blk_decoder(t))) return -1;
    #pragma omp parallel
    {
        float *row = (float *)malloc((size_t)I * sizeof(float));
        #pragma omp for schedule(static)
        for (int o = 0; o < O; o++) {
            if (!row) continue;
            gq_dequant_row(t, W + (int64_t)o * rb, row, I);
            for (int s = 0; s < S; s++) y[(int64_t)s * O + o] = gq_dot_f32(row, x + (int64_t)s * I, I);
        }
        free(row);
    }
    return 0;
}

#endif /* GGML_QUANTS_H */
