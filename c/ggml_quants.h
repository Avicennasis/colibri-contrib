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
#if defined(__ARM_NEON)
#include <arm_neon.h>
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
    GQ_MXFP4 = 39, GQ_NVFP4 = 40, GQ_Q1_0 = 41, GQ_Q2_0 = 42, GQ_TYPE_MAX = 43
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
    [GQ_NVFP4]   = { "NVFP4",   64,  36, 1 },
    [GQ_Q1_0]    = { "Q1_0",    128, 18, 1 },
    [GQ_Q2_0]    = { "Q2_0",    64,  18, 1 },
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

/* NVFP4 block scale: unsigned E4M3 (bias 7), halved like the MXFP4 exponent
 * because the value table holds e2m1 values doubled; 0 and 0x7F decode to 0. */
static inline float gq_ue4m3_half(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.f;
    int e = (x >> 3) & 0xF, m = x & 7;
    return e == 0 ? ldexpf((float)m, -10) : ldexpf((float)(8 + m), e - 11);
}
GQ_EXACT_FN static void gq_blk_nvfp4(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const uint8_t *qs = b + 4;
    for (int sb = 0; sb < 4; sb++) {
        const float d = gq_ue4m3_half(b[sb]);
        for (int j = 0; j < 8; j++) {
            y[16 * sb + j]     = d * (float)GGML_KVALUES_MXFP4[qs[8 * sb + j] & 0x0F];
            y[16 * sb + 8 + j] = d * (float)GGML_KVALUES_MXFP4[qs[8 * sb + j] >> 4];
        }
    }
}
GQ_EXACT_FN static void gq_blk_q1_0(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)), nd = -d; const uint8_t *qs = b + 2;
    for (int j = 0; j < 128; j++) y[j] = (qs[j >> 3] >> (j & 7)) & 1 ? d : nd;
}
GQ_EXACT_FN static void gq_blk_q2_0(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
    for (int j = 0; j < 64; j++) y[j] = (float)((int)((qs[j >> 2] >> ((j & 3) * 2)) & 3) - 1) * d;
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
    case GQ_NVFP4: return gq_blk_nvfp4;   case GQ_Q1_0: return gq_blk_q1_0;
    case GQ_Q2_0: return gq_blk_q2_0;
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
static int gq_matmul_f32(float *y, const float *x, const uint8_t *W, int t, int S, int I, int O) {
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


/* ---- integer form: every type as int8 values + a scale and a min per 16 ----
 * gq_unpack_row(t, src, n, w8, sw, mw) writes, for element i of group g = i/16,
 *     w8[i] (int8)   sw[g] (float)   mw[g] (float)
 * such that  sw[g] * w8[i] - mw[g]  is, bit for bit, what gq_dequant_row gives.
 * The products are formed with the same operations as the decoders above, so
 * the identity is exact, and it is what the integer dot product below uses:
 * one kernel for every type, the type only decides how the row is unpacked.
 * IQ1 folds its +-1/8 delta into the integer (8*grid +- 1, scale /8). */
#define GQ_G 16
static inline void gq_set_g(float *sw, float *mw, int g0, int ng, float s, float m) {
    for (int g = g0; g < g0 + ng; g++) { sw[g] = s; mw[g] = m; }
}
/* sign byte -> 8 int8 of +1 / -1 (bit j set = element j negative). A generated
 * constant (ggml_grids.h): it was once filled lazily inside the parallel
 * unpack, and a thread could see "ready" before the stores that built it --
 * wrong signs on some rows, depending on timing. Found by comparing the
 * per-layer state of the AVX2, scalar and NEON builds. */
#define GQ_SIGN8(v) (GGML_SIGN8 + 8 * (v))
static inline void gq_sign8_init(void) {}
#if defined(__AVX2__)
/* four 8-byte groups (grid rows g0..g3, sign bytes s0..s3) -> 32 signed int8 */
static inline __m256i gq_grid4(const uint8_t *g0, const uint8_t *g1, const uint8_t *g2, const uint8_t *g3,
                               int s0, int s1, int s2, int s3) {
    uint64_t a, b, c, d, e, f, h, i;
    memcpy(&a, g0, 8); memcpy(&b, g1, 8); memcpy(&c, g2, 8); memcpy(&d, g3, 8);
    memcpy(&e, GQ_SIGN8(s0), 8); memcpy(&f, GQ_SIGN8(s1), 8); memcpy(&h, GQ_SIGN8(s2), 8); memcpy(&i, GQ_SIGN8(s3), 8);
    return _mm256_sign_epi8(_mm256_set_epi64x((long long)d, (long long)c, (long long)b, (long long)a),
                            _mm256_set_epi64x((long long)i, (long long)h, (long long)f, (long long)e));
}
/* 32 bytes -> low nibbles and high nibbles as two 32-byte vectors */
static inline void gq_nib32(const uint8_t *p, __m256i *lo, __m256i *hi) {
    const __m256i m = _mm256_set1_epi8(0x0F), v = _mm256_loadu_si256((const __m256i *)p);
    *lo = _mm256_and_si256(v, m); *hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m);
}
/* 16 bytes -> 32 values: low nibbles then high nibbles, looked up in tab */
static inline __m256i gq_nib16_lut(const uint8_t *p, __m128i tab) {
    const __m128i m = _mm_set1_epi8(0x0F), v = _mm_loadu_si128((const __m128i *)p);
    __m128i lo = _mm_shuffle_epi8(tab, _mm_and_si128(v, m)), hi = _mm_shuffle_epi8(tab, _mm_and_si128(_mm_srli_epi16(v, 4), m));
    return _mm256_set_m128i(hi, lo);
}
#endif
GQ_EXACT_FN static void gq_ublk(int t, const uint8_t *b, int8_t *w, float *sw, float *mw) {
    GQ_EXACT_BEGIN
#if defined(__AVX2__)
    /* vector unpack for the common types: same integers and scales as the
     * scalar cases below (tests/test_ggml_quants.py checks both builds) */
    switch (t) {
    case GQ_Q4_K: case GQ_Q5_K: {
        const float d = gq_f16(gq_rd16(b)), dmin = gq_f16(gq_rd16(b + 2));
        const uint8_t *sc = b + 4; const uint8_t *qs = t == GQ_Q4_K ? b + 16 : b + 48;
        for (int g = 0; g < 8; g++) { int s, m; gq_scale_min_k4(sc, g, &s, &m); gq_set_g(sw, mw, 2 * g, 2, d * (float)s, dmin * (float)m); }
        __m256i qh = t == GQ_Q5_K ? _mm256_loadu_si256((const __m256i *)(b + 16)) : _mm256_setzero_si256();
        const __m256i sixteen = _mm256_set1_epi8(16);
        for (int c = 0; c < 4; c++) {
            __m256i lo, hi; gq_nib32(qs + 32 * c, &lo, &hi);
            if (t == GQ_Q5_K) {
                const __m256i b0 = _mm256_set1_epi8((char)(1 << (2 * c))), b1 = _mm256_set1_epi8((char)(1 << (2 * c + 1)));
                lo = _mm256_or_si256(lo, _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, b0), b0), sixteen));
                hi = _mm256_or_si256(hi, _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, b1), b1), sixteen));
            }
            _mm256_storeu_si256((__m256i *)(w + 64 * c), lo); _mm256_storeu_si256((__m256i *)(w + 64 * c + 32), hi);
        }
        return; }
    case GQ_Q6_K: {
        const uint8_t *ql = b, *qhp = b + 128; const int8_t *sc = (const int8_t *)(b + 192);
        const float d = gq_f16(gq_rd16(b + 208));
        for (int g = 0; g < 16; g++) { sw[g] = d * (float)sc[g]; mw[g] = 0.f; }
        const __m256i m3 = _mm256_set1_epi8(3), m32 = _mm256_set1_epi8(32);
        for (int c = 0; c < 2; c++) {
            __m256i l0, h0, l1, h1; gq_nib32(ql + 64 * c, &l0, &h0); gq_nib32(ql + 64 * c + 32, &l1, &h1);
            __m256i q = _mm256_loadu_si256((const __m256i *)(qhp + 32 * c));
            __m256i q0 = _mm256_and_si256(q, m3), q1 = _mm256_and_si256(_mm256_srli_epi16(q, 2), m3);
            __m256i q2 = _mm256_and_si256(_mm256_srli_epi16(q, 4), m3), q3 = _mm256_and_si256(_mm256_srli_epi16(q, 6), m3);
            #define GQ6(L, H) _mm256_sub_epi8(_mm256_or_si256(L, _mm256_slli_epi16(H, 4)), m32)
            _mm256_storeu_si256((__m256i *)(w + 128 * c),      GQ6(l0, q0));
            _mm256_storeu_si256((__m256i *)(w + 128 * c + 32), GQ6(l1, q1));
            _mm256_storeu_si256((__m256i *)(w + 128 * c + 64), GQ6(h0, q2));
            _mm256_storeu_si256((__m256i *)(w + 128 * c + 96), GQ6(h1, q3));
            #undef GQ6
        }
        return; }
    case GQ_Q4_0: {
        const float d = gq_f16(gq_rd16(b));
        const __m128i m = _mm_set1_epi8(0x0F), v = _mm_loadu_si128((const __m128i *)(b + 2)), e8 = _mm_set1_epi8(8);
        _mm_storeu_si128((__m128i *)w, _mm_sub_epi8(_mm_and_si128(v, m), e8));
        _mm_storeu_si128((__m128i *)(w + 16), _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(v, 4), m), e8));
        gq_set_g(sw, mw, 0, 2, d, 0.f); return; }
    case GQ_IQ4_NL: case GQ_MXFP4: {
        const int nl = t == GQ_IQ4_NL;
        const float d = nl ? gq_f16(gq_rd16(b)) : gq_e8m0_half(b[0]);
        const __m128i tab = _mm_loadu_si128((const __m128i *)(nl ? GGML_KVALUES_IQ4NL : GGML_KVALUES_MXFP4));
        _mm256_storeu_si256((__m256i *)w, gq_nib16_lut(b + (nl ? 2 : 1), tab));
        gq_set_g(sw, mw, 0, 2, d, 0.f); return; }
    case GQ_IQ4_XS: {
        const float d = gq_f16(gq_rd16(b)); const uint16_t sh = gq_rd16(b + 2); const uint8_t *sl = b + 4, *qs = b + 8;
        const __m128i tab = _mm_loadu_si128((const __m128i *)GGML_KVALUES_IQ4NL);
        for (int ib = 0; ib < 8; ib++) {
            int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((sh >> (2 * ib)) & 3) << 4);
            gq_set_g(sw, mw, 2 * ib, 2, d * (float)(ls - 32), 0.f);
            _mm256_storeu_si256((__m256i *)(w + 32 * ib), gq_nib16_lut(qs + 16 * ib, tab));
        }
        return; }
    case GQ_IQ2_XXS: {
        gq_sign8_init(); const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
        for (int ib = 0; ib < 8; ib++) {
            uint32_t a0 = gq_rd32(qs + 8 * ib), a1 = gq_rd32(qs + 8 * ib + 4);
            gq_set_g(sw, mw, 2 * ib, 2, d * (0.5f + (float)(a1 >> 28)) * 0.25f, 0.f);
            const uint8_t *G = GGML_IQ2XXS_GRID;
            _mm256_storeu_si256((__m256i *)(w + 32 * ib), gq_grid4(G + 8 * (a0 & 0xFF), G + 8 * ((a0 >> 8) & 0xFF), G + 8 * ((a0 >> 16) & 0xFF), G + 8 * (a0 >> 24),
                GGML_KSIGNS_IQ2XS[a1 & 127], GGML_KSIGNS_IQ2XS[(a1 >> 7) & 127], GGML_KSIGNS_IQ2XS[(a1 >> 14) & 127], GGML_KSIGNS_IQ2XS[(a1 >> 21) & 127]));
        }
        return; }
    case GQ_IQ2_XS: {
        gq_sign8_init(); const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *sc = b + 66;
        for (int ib = 0; ib < 8; ib++) {
            for (int h = 0; h < 2; h++) { int si = 2 * ib + h; int s = (sc[si >> 1] >> (4 * (si & 1))) & 0x0F; sw[si] = d * (0.5f + (float)s) * 0.25f; mw[si] = 0.f; }
            uint16_t q0 = gq_rd16(qs + 8 * ib), q1 = gq_rd16(qs + 8 * ib + 2), q2 = gq_rd16(qs + 8 * ib + 4), q3 = gq_rd16(qs + 8 * ib + 6);
            const uint8_t *G = GGML_IQ2XS_GRID;
            _mm256_storeu_si256((__m256i *)(w + 32 * ib), gq_grid4(G + 8 * (q0 & 511), G + 8 * (q1 & 511), G + 8 * (q2 & 511), G + 8 * (q3 & 511),
                GGML_KSIGNS_IQ2XS[q0 >> 9], GGML_KSIGNS_IQ2XS[q1 >> 9], GGML_KSIGNS_IQ2XS[q2 >> 9], GGML_KSIGNS_IQ2XS[q3 >> 9]));
        }
        return; }
    case GQ_IQ2_S: {
        gq_sign8_init(); const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *sg = b + 34, *qh = b + 66, *sc = b + 74;
        for (int ib = 0; ib < 8; ib++) {
            for (int h = 0; h < 2; h++) { int si = 2 * ib + h; int s = (sc[si >> 1] >> (4 * (si & 1))) & 0x0F; sw[si] = d * (0.5f + (float)s) * 0.25f; mw[si] = 0.f; }
            int k = 4 * ib; const uint8_t *G = GGML_IQ2S_GRID;
            #define IX(kk) (qs[kk] | (((qh[(kk) >> 2] >> (2 * ((kk) & 3))) & 3) << 8))
            _mm256_storeu_si256((__m256i *)(w + 32 * ib), gq_grid4(G + 8 * IX(k), G + 8 * IX(k + 1), G + 8 * IX(k + 2), G + 8 * IX(k + 3),
                sg[k], sg[k + 1], sg[k + 2], sg[k + 3]));
            #undef IX
        }
        return; }
    case GQ_IQ3_XXS: {
        gq_sign8_init(); const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *ss = b + 66;
        for (int ib = 0; ib < 8; ib++) {
            uint32_t a = gq_rd32(ss + 4 * ib);
            gq_set_g(sw, mw, 2 * ib, 2, d * (0.5f + (float)(a >> 28)) * 0.5f, 0.f);
            uint8_t gb[32]; const uint8_t *q = qs + 8 * ib;
            for (int j = 0; j < 8; j++) memcpy(gb + 4 * j, GGML_IQ3XXS_GRID + 4 * q[j], 4);
            _mm256_storeu_si256((__m256i *)(w + 32 * ib), gq_grid4(gb, gb + 8, gb + 16, gb + 24,
                GGML_KSIGNS_IQ2XS[a & 127], GGML_KSIGNS_IQ2XS[(a >> 7) & 127], GGML_KSIGNS_IQ2XS[(a >> 14) & 127], GGML_KSIGNS_IQ2XS[(a >> 21) & 127]));
        }
        return; }
    case GQ_IQ3_S: {
        gq_sign8_init(); const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *qh = b + 66, *sg = b + 74, *sc = b + 106;
        for (int ib = 0; ib < 8; ib++) {
            int s = (sc[ib >> 1] >> (4 * (ib & 1))) & 0x0F; gq_set_g(sw, mw, 2 * ib, 2, d * (float)(1 + 2 * s), 0.f);
            uint8_t gb[32];
            for (int j = 0; j < 8; j++) { int k = 8 * ib + j; int idx = qs[k] | (((qh[k >> 3] >> (k & 7)) & 1) << 8); memcpy(gb + 4 * j, GGML_IQ3S_GRID + 4 * idx, 4); }
            _mm256_storeu_si256((__m256i *)(w + 32 * ib), gq_grid4(gb, gb + 8, gb + 16, gb + 24, sg[4 * ib], sg[4 * ib + 1], sg[4 * ib + 2], sg[4 * ib + 3]));
        }
        return; }
    case GQ_IQ1_S: {
        const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *qhp = b + 34;
        for (int ib = 0; ib < 8; ib++) {
            uint16_t qh = gq_rd16(qhp + 2 * ib);
            gq_set_g(sw, mw, 2 * ib, 2, d * (float)(2 * ((qh >> 12) & 7) + 1) * 0.125f, 0.f);
            uint64_t g[4];
            for (int l = 0; l < 4; l++) memcpy(&g[l], GGML_IQ1S_GRID + 8 * (qs[4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)), 8);
            __m256i v = _mm256_set_epi64x((long long)g[3], (long long)g[2], (long long)g[1], (long long)g[0]);
            v = _mm256_add_epi8(_mm256_add_epi8(v, v), _mm256_add_epi8(v, v));      /* 4x */
            v = _mm256_add_epi8(v, v);                                               /* 8x */
            v = _mm256_add_epi8(v, _mm256_set1_epi8((qh & 0x8000) ? -1 : 1));
            _mm256_storeu_si256((__m256i *)(w + 32 * ib), v);
        }
        return; }
    case GQ_IQ1_M: {
        const uint8_t *qs = b, *qh = b + 32, *scp = b + 48;
        uint16_t sc[4]; for (int i = 0; i < 4; i++) sc[i] = gq_rd16(scp + 2 * i);
        const float d = gq_f16((uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000)));
        for (int ib = 0; ib < 8; ib++) {
            for (int h = 0; h < 2; h++) { int s16 = 2 * ib + h; int s = (sc[s16 >> 2] >> (3 * (s16 & 3))) & 7; sw[s16] = d * (float)(2 * s + 1) * 0.125f; mw[s16] = 0.f; }
            uint64_t g[4]; int8_t ds[4];
            for (int l = 0; l < 4; l++) {
                int k = 4 * ib + l, hh = (qh[k >> 1] >> (4 * (k & 1))) & 0x0F;
                memcpy(&g[l], GGML_IQ1S_GRID + 8 * (qs[k] | ((hh & 7) << 8)), 8); ds[l] = (hh & 8) ? -1 : 1;
            }
            __m256i v = _mm256_set_epi64x((long long)g[3], (long long)g[2], (long long)g[1], (long long)g[0]);
            v = _mm256_add_epi8(v, v); v = _mm256_add_epi8(v, v); v = _mm256_add_epi8(v, v);   /* 8 * grid, byte-wise (a 16-bit shift would carry 0xFF into the neighbour) */
            __m256i dl = _mm256_set_epi64x((long long)(0x0101010101010101ull * (uint8_t)ds[3]), (long long)(0x0101010101010101ull * (uint8_t)ds[2]),
                                           (long long)(0x0101010101010101ull * (uint8_t)ds[1]), (long long)(0x0101010101010101ull * (uint8_t)ds[0]));
            _mm256_storeu_si256((__m256i *)(w + 32 * ib), _mm256_add_epi8(v, dl));
        }
        return; }
    case GQ_Q2_K: {
        const uint8_t *sc = b, *qs = b + 16; const float d = gq_f16(gq_rd16(b + 80)), dmin = gq_f16(gq_rd16(b + 82));
        for (int g = 0; g < 16; g++) { sw[g] = d * (float)(sc[g] & 0xF); mw[g] = dmin * (float)(sc[g] >> 4); }
        const __m256i m3 = _mm256_set1_epi8(3);
        for (int c = 0; c < 2; c++) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(qs + 32 * c));
            for (int sh = 0; sh < 4; sh++)
                _mm256_storeu_si256((__m256i *)(w + 128 * c + 32 * sh), _mm256_and_si256(_mm256_srli_epi16(q, 2 * sh), m3));
        }
        return; }
    case GQ_Q3_K: {
        const uint8_t *hm = b, *qs = b + 32, *s = b + 96; const float d = gq_f16(gq_rd16(b + 108));
        for (int k = 0; k < 16; k++) {
            int l = k < 8 ? (s[k] & 0x0F) : (s[k - 8] >> 4), h = (s[8 + (k & 3)] >> (2 * (k >> 2))) & 3;
            sw[k] = d * (float)((l | (h << 4)) - 32); mw[k] = 0.f;
        }
        const __m256i m3 = _mm256_set1_epi8(3), four = _mm256_set1_epi8(4), hv = _mm256_loadu_si256((const __m256i *)hm);
        for (int c = 0; c < 2; c++) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(qs + 32 * c));
            for (int sh = 0; sh < 4; sh++) {
                int row = 4 * c + sh;                 /* element i = 32*row + j uses bit row of hm[j] */
                __m256i bit = _mm256_set1_epi8((char)(1 << row));
                __m256i set = _mm256_cmpeq_epi8(_mm256_and_si256(hv, bit), bit);            /* 0xFF where the bit is set */
                __m256i sub = _mm256_andnot_si256(set, four);                               /* 4 where it is clear */
                _mm256_storeu_si256((__m256i *)(w + 128 * c + 32 * sh), _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(q, 2 * sh), m3), sub));
            }
        }
        return; }
    case GQ_TQ2_0: {
        const float d = gq_f16(gq_rd16(b + 64)); const __m256i m3 = _mm256_set1_epi8(3), one = _mm256_set1_epi8(1);
        gq_set_g(sw, mw, 0, 16, d, 0.f);
        for (int c = 0; c < 2; c++) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(b + 32 * c));
            for (int sh = 0; sh < 4; sh++)
                _mm256_storeu_si256((__m256i *)(w + 128 * c + 32 * sh), _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(q, 2 * sh), m3), one));
        }
        return; }
    default: break;
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    /* the same vector unpack on NEON (AArch64): same integers and scales */
    switch (t) {
    case GQ_Q4_K: case GQ_Q5_K: {
        const float d = gq_f16(gq_rd16(b)), dmin = gq_f16(gq_rd16(b + 2));
        const uint8_t *sc = b + 4; const uint8_t *qs = t == GQ_Q4_K ? b + 16 : b + 48;
        for (int g = 0; g < 8; g++) { int s, m; gq_scale_min_k4(sc, g, &s, &m); gq_set_g(sw, mw, 2 * g, 2, d * (float)s, dmin * (float)m); }
        const uint8x16_t m4 = vdupq_n_u8(0x0F), sixteen = vdupq_n_u8(16);
        uint8x16_t qh0 = vdupq_n_u8(0), qh1 = vdupq_n_u8(0);
        if (t == GQ_Q5_K) { qh0 = vld1q_u8(b + 16); qh1 = vld1q_u8(b + 32); }
        for (int c = 0; c < 4; c++) {
            for (int h = 0; h < 2; h++) {
                uint8x16_t v = vld1q_u8(qs + 32 * c + 16 * h);
                uint8x16_t lo = vandq_u8(v, m4), hi = vshrq_n_u8(v, 4);
                if (t == GQ_Q5_K) {
                    uint8x16_t qh = h ? qh1 : qh0;
                    uint8x16_t b0 = vdupq_n_u8((uint8_t)(1 << (2 * c))), b1 = vdupq_n_u8((uint8_t)(1 << (2 * c + 1)));
                    lo = vorrq_u8(lo, vandq_u8(vtstq_u8(qh, b0), sixteen));
                    hi = vorrq_u8(hi, vandq_u8(vtstq_u8(qh, b1), sixteen));
                }
                vst1q_s8(w + 64 * c + 16 * h, vreinterpretq_s8_u8(lo));
                vst1q_s8(w + 64 * c + 32 + 16 * h, vreinterpretq_s8_u8(hi));
            }
        }
        return; }
    case GQ_Q6_K: {
        const uint8_t *ql = b, *qhp = b + 128; const int8_t *sc = (const int8_t *)(b + 192);
        const float d = gq_f16(gq_rd16(b + 208));
        for (int g = 0; g < 16; g++) { sw[g] = d * (float)sc[g]; mw[g] = 0.f; }
        const uint8x16_t m4 = vdupq_n_u8(0x0F), m3 = vdupq_n_u8(3); const int8x16_t m32 = vdupq_n_s8(32);
        for (int c = 0; c < 2; c++) for (int h = 0; h < 2; h++) {
            uint8x16_t a = vld1q_u8(ql + 64 * c + 16 * h), bq = vld1q_u8(ql + 64 * c + 32 + 16 * h), q = vld1q_u8(qhp + 32 * c + 16 * h);
            #define GQ6N(L, H) vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(L, vshlq_n_u8(H, 4))), m32)
            vst1q_s8(w + 128 * c + 16 * h,      GQ6N(vandq_u8(a, m4),  vandq_u8(q, m3)));
            vst1q_s8(w + 128 * c + 32 + 16 * h, GQ6N(vandq_u8(bq, m4), vandq_u8(vshrq_n_u8(q, 2), m3)));
            vst1q_s8(w + 128 * c + 64 + 16 * h, GQ6N(vshrq_n_u8(a, 4), vandq_u8(vshrq_n_u8(q, 4), m3)));
            vst1q_s8(w + 128 * c + 96 + 16 * h, GQ6N(vshrq_n_u8(bq, 4), vshrq_n_u8(q, 6)));
            #undef GQ6N
        }
        return; }
    case GQ_Q4_0: {
        const float d = gq_f16(gq_rd16(b)); uint8x16_t v = vld1q_u8(b + 2); const int8x16_t e8 = vdupq_n_s8(8);
        vst1q_s8(w, vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(15))), e8));
        vst1q_s8(w + 16, vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), e8));
        gq_set_g(sw, mw, 0, 2, d, 0.f); return; }
    case GQ_IQ4_NL: case GQ_MXFP4: {
        const int nl = t == GQ_IQ4_NL;
        const float d = nl ? gq_f16(gq_rd16(b)) : gq_e8m0_half(b[0]);
        const int8x16_t tab = vld1q_s8(nl ? GGML_KVALUES_IQ4NL : GGML_KVALUES_MXFP4);
        uint8x16_t v = vld1q_u8(b + (nl ? 2 : 1));
        vst1q_s8(w, vqtbl1q_s8(tab, vandq_u8(v, vdupq_n_u8(15)))); vst1q_s8(w + 16, vqtbl1q_s8(tab, vshrq_n_u8(v, 4)));
        gq_set_g(sw, mw, 0, 2, d, 0.f); return; }
    case GQ_IQ4_XS: {
        const float d = gq_f16(gq_rd16(b)); const uint16_t sh = gq_rd16(b + 2); const uint8_t *sl = b + 4, *qs = b + 8;
        const int8x16_t tab = vld1q_s8(GGML_KVALUES_IQ4NL);
        for (int ib = 0; ib < 8; ib++) {
            int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((sh >> (2 * ib)) & 3) << 4);
            gq_set_g(sw, mw, 2 * ib, 2, d * (float)(ls - 32), 0.f);
            uint8x16_t v = vld1q_u8(qs + 16 * ib);
            vst1q_s8(w + 32 * ib, vqtbl1q_s8(tab, vandq_u8(v, vdupq_n_u8(15)))); vst1q_s8(w + 32 * ib + 16, vqtbl1q_s8(tab, vshrq_n_u8(v, 4)));
        }
        return; }
    case GQ_IQ2_XXS: {
        gq_sign8_init(); const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
        for (int ib = 0; ib < 8; ib++) {
            uint32_t a0 = gq_rd32(qs + 8 * ib), a1 = gq_rd32(qs + 8 * ib + 4);
            gq_set_g(sw, mw, 2 * ib, 2, d * (0.5f + (float)(a1 >> 28)) * 0.25f, 0.f);
            for (int l = 0; l < 4; l++) {
                int8x8_t g = vreinterpret_s8_u8(vld1_u8(GGML_IQ2XXS_GRID + 8 * ((a0 >> (8 * l)) & 0xFF)));
                vst1_s8(w + 32 * ib + 8 * l, vmul_s8(g, vld1_s8(GQ_SIGN8(GGML_KSIGNS_IQ2XS[(a1 >> (7 * l)) & 127]))));
            }
        }
        return; }
    case GQ_IQ2_XS: {
        gq_sign8_init(); const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *sc = b + 66;
        for (int k = 0; k < 32; k++) {
            uint16_t q = gq_rd16(qs + 2 * k); int si = k >> 1;
            if (!(k & 1)) { int s = (sc[si >> 1] >> (4 * (si & 1))) & 0x0F; sw[si] = d * (0.5f + (float)s) * 0.25f; mw[si] = 0.f; }
            int8x8_t g = vreinterpret_s8_u8(vld1_u8(GGML_IQ2XS_GRID + 8 * (q & 511)));
            vst1_s8(w + 8 * k, vmul_s8(g, vld1_s8(GQ_SIGN8(GGML_KSIGNS_IQ2XS[q >> 9]))));
        }
        return; }
    case GQ_IQ2_S: {
        gq_sign8_init(); const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *sg = b + 34, *qh = b + 66, *sc = b + 74;
        for (int k = 0; k < 32; k++) {
            int si = k >> 1;
            if (!(k & 1)) { int s = (sc[si >> 1] >> (4 * (si & 1))) & 0x0F; sw[si] = d * (0.5f + (float)s) * 0.25f; mw[si] = 0.f; }
            int idx = qs[k] | (((qh[k >> 2] >> (2 * (k & 3))) & 3) << 8);
            vst1_s8(w + 8 * k, vmul_s8(vreinterpret_s8_u8(vld1_u8(GGML_IQ2S_GRID + 8 * idx)), vld1_s8(GQ_SIGN8(sg[k]))));
        }
        return; }
    case GQ_IQ3_XXS: {
        gq_sign8_init(); const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *ss = b + 66;
        for (int ib = 0; ib < 8; ib++) {
            uint32_t a = gq_rd32(ss + 4 * ib);
            gq_set_g(sw, mw, 2 * ib, 2, d * (0.5f + (float)(a >> 28)) * 0.5f, 0.f);
            for (int l = 0; l < 4; l++) {
                uint8_t gb[8]; memcpy(gb, GGML_IQ3XXS_GRID + 4 * qs[8 * ib + 2 * l], 4); memcpy(gb + 4, GGML_IQ3XXS_GRID + 4 * qs[8 * ib + 2 * l + 1], 4);
                vst1_s8(w + 32 * ib + 8 * l, vmul_s8(vreinterpret_s8_u8(vld1_u8(gb)), vld1_s8(GQ_SIGN8(GGML_KSIGNS_IQ2XS[(a >> (7 * l)) & 127]))));
            }
        }
        return; }
    case GQ_IQ3_S: {
        gq_sign8_init(); const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *qh = b + 66, *sg = b + 74, *sc = b + 106;
        for (int ib = 0; ib < 8; ib++) { int s = (sc[ib >> 1] >> (4 * (ib & 1))) & 0x0F; gq_set_g(sw, mw, 2 * ib, 2, d * (float)(1 + 2 * s), 0.f); }
        for (int g8 = 0; g8 < 32; g8++) {
            uint8_t gb[8];
            for (int h = 0; h < 2; h++) { int k = 2 * g8 + h; int idx = qs[k] | (((qh[k >> 3] >> (k & 7)) & 1) << 8); memcpy(gb + 4 * h, GGML_IQ3S_GRID + 4 * idx, 4); }
            vst1_s8(w + 8 * g8, vmul_s8(vreinterpret_s8_u8(vld1_u8(gb)), vld1_s8(GQ_SIGN8(sg[g8]))));
        }
        return; }
    case GQ_Q2_K: {
        const uint8_t *sc = b, *qs = b + 16; const float d = gq_f16(gq_rd16(b + 80)), dmin = gq_f16(gq_rd16(b + 82));
        for (int g = 0; g < 16; g++) { sw[g] = d * (float)(sc[g] & 0xF); mw[g] = dmin * (float)(sc[g] >> 4); }
        const uint8x16_t m3 = vdupq_n_u8(3);
        for (int c = 0; c < 2; c++) for (int h = 0; h < 2; h++) {
            uint8x16_t q = vld1q_u8(qs + 32 * c + 16 * h);
            vst1q_s8(w + 128 * c + 16 * h,      vreinterpretq_s8_u8(vandq_u8(q, m3)));
            vst1q_s8(w + 128 * c + 32 + 16 * h, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q, 2), m3)));
            vst1q_s8(w + 128 * c + 64 + 16 * h, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q, 4), m3)));
            vst1q_s8(w + 128 * c + 96 + 16 * h, vreinterpretq_s8_u8(vshrq_n_u8(q, 6)));
        }
        return; }
    default: break;
    }
#endif
    switch (t) {
    case GQ_Q4_0: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
        for (int j = 0; j < 16; j++) { w[j] = (int8_t)((qs[j] & 15) - 8); w[j + 16] = (int8_t)((qs[j] >> 4) - 8); }
        gq_set_g(sw, mw, 0, 2, d, 0.f); break; }
    case GQ_Q4_1: { const float d = gq_f16(gq_rd16(b)), m = gq_f16(gq_rd16(b + 2)); const uint8_t *qs = b + 4;
        for (int j = 0; j < 16; j++) { w[j] = (int8_t)(qs[j] & 15); w[j + 16] = (int8_t)(qs[j] >> 4); }
        gq_set_g(sw, mw, 0, 2, d, -m); break; }
    case GQ_Q5_0: { const float d = gq_f16(gq_rd16(b)); const uint32_t qh = gq_rd32(b + 2); const uint8_t *qs = b + 6;
        for (int j = 0; j < 16; j++) {
            w[j] = (int8_t)(((qs[j] & 15) | (int)(((qh >> j) & 1) << 4)) - 16);
            w[j + 16] = (int8_t)(((qs[j] >> 4) | (int)(((qh >> (j + 16)) & 1) << 4)) - 16);
        }
        gq_set_g(sw, mw, 0, 2, d, 0.f); break; }
    case GQ_Q5_1: { const float d = gq_f16(gq_rd16(b)), m = gq_f16(gq_rd16(b + 2));
        const uint32_t qh = gq_rd32(b + 4); const uint8_t *qs = b + 8;
        for (int j = 0; j < 16; j++) {
            w[j] = (int8_t)((qs[j] & 15) | (int)(((qh >> j) & 1) << 4));
            w[j + 16] = (int8_t)((qs[j] >> 4) | (int)(((qh >> (j + 16)) & 1) << 4));
        }
        gq_set_g(sw, mw, 0, 2, d, -m); break; }
    case GQ_Q8_0: { const float d = gq_f16(gq_rd16(b)); memcpy(w, b + 2, 32); gq_set_g(sw, mw, 0, 2, d, 0.f); break; }
    case GQ_Q2_K: { const uint8_t *sc = b, *qs = b + 16;
        const float d = gq_f16(gq_rd16(b + 80)), dmin = gq_f16(gq_rd16(b + 82));
        for (int i = 0; i < 256; i++) { int c = i >> 7, sh = (i >> 5) & 3, j = i & 31; w[i] = (int8_t)((qs[c * 32 + j] >> (2 * sh)) & 3); }
        for (int g = 0; g < 16; g++) { sw[g] = d * (float)(sc[g] & 0xF); mw[g] = dmin * (float)(sc[g] >> 4); }
        break; }
    case GQ_Q3_K: { const uint8_t *hm = b, *qs = b + 32, *s = b + 96; const float d = gq_f16(gq_rd16(b + 108));
        for (int k = 0; k < 16; k++) {
            int l = k < 8 ? (s[k] & 0x0F) : (s[k - 8] >> 4);
            int h = (s[8 + (k & 3)] >> (2 * (k >> 2))) & 3;
            sw[k] = d * (float)((l | (h << 4)) - 32); mw[k] = 0.f;
        }
        for (int i = 0; i < 256; i++) {
            int c = i >> 7, sh = (i >> 5) & 3, j = i & 31;
            int ql = (qs[c * 32 + j] >> (2 * sh)) & 3, hb = (hm[i & 31] >> (i >> 5)) & 1;
            w[i] = (int8_t)(ql - ((hb ^ 1) << 2));
        }
        break; }
    case GQ_Q4_K: case GQ_Q5_K: { const float d = gq_f16(gq_rd16(b)), dmin = gq_f16(gq_rd16(b + 2));
        const uint8_t *sc = b + 4, *qh = b + 16, *qs = t == GQ_Q4_K ? b + 16 : b + 48;
        for (int g = 0; g < 8; g++) {
            int s, m; gq_scale_min_k4(sc, g, &s, &m);
            gq_set_g(sw, mw, 2 * g, 2, d * (float)s, dmin * (float)m);
            const uint8_t *q = qs + (g >> 1) * 32; int sh = (g & 1) * 4;
            for (int j = 0; j < 32; j++) {
                int v = (q[j] >> sh) & 0x0F;
                if (t == GQ_Q5_K) v |= ((qh[j] >> g) & 1) << 4;
                w[g * 32 + j] = (int8_t)v;
            }
        }
        break; }
    case GQ_Q6_K: { const uint8_t *ql = b, *qh = b + 128; const int8_t *sc = (const int8_t *)(b + 192);
        const float d = gq_f16(gq_rd16(b + 208));
        for (int i = 0; i < 256; i++) {
            int c = i >> 7, r = i & 127, nib = r >> 6, k = r & 63, sh = r >> 5, j = r & 31;
            int lo = (ql[c * 64 + k] >> (4 * nib)) & 0x0F, hi = (qh[c * 32 + j] >> (2 * sh)) & 3;
            w[i] = (int8_t)((lo | (hi << 4)) - 32);
        }
        for (int g = 0; g < 16; g++) { sw[g] = d * (float)sc[g]; mw[g] = 0.f; }
        break; }
    case GQ_IQ2_XXS: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
        for (int ib = 0; ib < 8; ib++) {
            uint32_t a0 = gq_rd32(qs + 8 * ib), a1 = gq_rd32(qs + 8 * ib + 4);
            gq_set_g(sw, mw, 2 * ib, 2, d * (0.5f + (float)(a1 >> 28)) * 0.25f, 0.f);
            for (int l = 0; l < 4; l++) {
                const uint8_t *grid = GGML_IQ2XXS_GRID + 8 * ((a0 >> (8 * l)) & 0xFF);
                uint8_t signs = GGML_KSIGNS_IQ2XS[(a1 >> (7 * l)) & 127];
                for (int j = 0; j < 8; j++) w[ib * 32 + l * 8 + j] = (int8_t)((signs >> j) & 1 ? -grid[j] : grid[j]);
            }
        }
        break; }
    case GQ_IQ2_XS: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *sc = b + 66;
        for (int k = 0; k < 32; k++) {
            uint16_t q = gq_rd16(qs + 2 * k); int si = k >> 1;
            if (!(k & 1)) { int s = (sc[si >> 1] >> (4 * (si & 1))) & 0x0F; sw[si] = d * (0.5f + (float)s) * 0.25f; mw[si] = 0.f; }
            const uint8_t *grid = GGML_IQ2XS_GRID + 8 * (q & 511); uint8_t signs = GGML_KSIGNS_IQ2XS[q >> 9];
            for (int j = 0; j < 8; j++) w[k * 8 + j] = (int8_t)((signs >> j) & 1 ? -grid[j] : grid[j]);
        }
        break; }
    case GQ_IQ2_S: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *sg = b + 34, *qh = b + 66, *sc = b + 74;
        for (int k = 0; k < 32; k++) {
            int si = k >> 1;
            if (!(k & 1)) { int s = (sc[si >> 1] >> (4 * (si & 1))) & 0x0F; sw[si] = d * (0.5f + (float)s) * 0.25f; mw[si] = 0.f; }
            int idx = qs[k] | (((qh[k >> 2] >> (2 * (k & 3))) & 3) << 8);
            const uint8_t *grid = GGML_IQ2S_GRID + 8 * idx;
            for (int j = 0; j < 8; j++) w[k * 8 + j] = (int8_t)((sg[k] >> j) & 1 ? -grid[j] : grid[j]);
        }
        break; }
    case GQ_IQ3_XXS: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *ss = b + 66;
        for (int ib = 0; ib < 8; ib++) {
            uint32_t a = gq_rd32(ss + 4 * ib);
            gq_set_g(sw, mw, 2 * ib, 2, d * (0.5f + (float)(a >> 28)) * 0.5f, 0.f);
            for (int l = 0; l < 4; l++) {
                uint8_t signs = GGML_KSIGNS_IQ2XS[(a >> (7 * l)) & 127];
                const uint8_t *g1 = GGML_IQ3XXS_GRID + 4 * qs[8 * ib + 2 * l], *g2 = GGML_IQ3XXS_GRID + 4 * qs[8 * ib + 2 * l + 1];
                for (int j = 0; j < 4; j++) {
                    w[ib * 32 + l * 8 + j]     = (int8_t)((signs >> j) & 1 ? -g1[j] : g1[j]);
                    w[ib * 32 + l * 8 + j + 4] = (int8_t)((signs >> (j + 4)) & 1 ? -g2[j] : g2[j]);
                }
            }
        }
        break; }
    case GQ_IQ3_S: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *qh = b + 66, *sg = b + 74, *sc = b + 106;
        for (int ib = 0; ib < 8; ib++) { int s = (sc[ib >> 1] >> (4 * (ib & 1))) & 0x0F; gq_set_g(sw, mw, 2 * ib, 2, d * (float)(1 + 2 * s), 0.f); }
        for (int k = 0; k < 64; k++) {
            int idx = qs[k] | (((qh[k >> 3] >> (k & 7)) & 1) << 8);
            const uint8_t *grid = GGML_IQ3S_GRID + 4 * idx;
            for (int j = 0; j < 4; j++) { int e = k * 4 + j; w[e] = (int8_t)((sg[e >> 3] >> (e & 7)) & 1 ? -grid[j] : grid[j]); }
        }
        break; }
    case GQ_IQ1_S: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *qhp = b + 34;
        for (int ib = 0; ib < 8; ib++) {
            uint16_t qh = gq_rd16(qhp + 2 * ib);
            const float dl = d * (float)(2 * ((qh >> 12) & 7) + 1);
            int dsign = (qh & 0x8000) ? -1 : 1;
            gq_set_g(sw, mw, 2 * ib, 2, dl * 0.125f, 0.f);
            for (int l = 0; l < 4; l++) {
                const int8_t *grid = GGML_IQ1S_GRID + 8 * (qs[4 * ib + l] | (((qh >> (3 * l)) & 7) << 8));
                for (int j = 0; j < 8; j++) w[ib * 32 + l * 8 + j] = (int8_t)(8 * grid[j] + dsign);
            }
        }
        break; }
    case GQ_IQ1_M: { const uint8_t *qs = b, *qh = b + 32, *scp = b + 48;
        uint16_t sc[4]; for (int i = 0; i < 4; i++) sc[i] = gq_rd16(scp + 2 * i);
        const float d = gq_f16((uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000)));
        for (int k = 0; k < 32; k++) {
            int s16 = k >> 1;
            if (!(k & 1)) { int s = (sc[s16 >> 2] >> (3 * (s16 & 3))) & 7; sw[s16] = d * (float)(2 * s + 1) * 0.125f; mw[s16] = 0.f; }
            int h = (qh[k >> 1] >> (4 * (k & 1))) & 0x0F;
            const int8_t *grid = GGML_IQ1S_GRID + 8 * (qs[k] | ((h & 7) << 8));
            int dsign = (h & 8) ? -1 : 1;
            for (int j = 0; j < 8; j++) w[k * 8 + j] = (int8_t)(8 * grid[j] + dsign);
        }
        break; }
    case GQ_IQ4_NL: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
        for (int j = 0; j < 16; j++) { w[j] = GGML_KVALUES_IQ4NL[qs[j] & 15]; w[j + 16] = GGML_KVALUES_IQ4NL[qs[j] >> 4]; }
        gq_set_g(sw, mw, 0, 2, d, 0.f); break; }
    case GQ_IQ4_XS: { const float d = gq_f16(gq_rd16(b)); const uint16_t sh = gq_rd16(b + 2); const uint8_t *sl = b + 4, *qs = b + 8;
        for (int ib = 0; ib < 8; ib++) {
            int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((sh >> (2 * ib)) & 3) << 4);
            gq_set_g(sw, mw, 2 * ib, 2, d * (float)(ls - 32), 0.f);
            const uint8_t *q = qs + 16 * ib;
            for (int j = 0; j < 16; j++) { w[ib * 32 + j] = GGML_KVALUES_IQ4NL[q[j] & 15]; w[ib * 32 + j + 16] = GGML_KVALUES_IQ4NL[q[j] >> 4]; }
        }
        break; }
    case GQ_MXFP4: { const float d = gq_e8m0_half(b[0]); const uint8_t *qs = b + 1;
        for (int j = 0; j < 16; j++) { w[j] = GGML_KVALUES_MXFP4[qs[j] & 15]; w[j + 16] = GGML_KVALUES_MXFP4[qs[j] >> 4]; }
        gq_set_g(sw, mw, 0, 2, d, 0.f); break; }
    case GQ_TQ1_0: { static const uint8_t pow3[5] = { 1, 3, 9, 27, 81 };
        const uint8_t *qs = b, *qh = b + 48; const float d = gq_f16(gq_rd16(b + 52)); int o = 0;
        for (int n = 0; n < 5; n++) for (int j = 0; j < 32; j++) { uint8_t v = (uint8_t)(qs[j] * pow3[n]); w[o++] = (int8_t)((int)(((uint16_t)v * 3) >> 8) - 1); }
        for (int n = 0; n < 5; n++) for (int j = 0; j < 16; j++) { uint8_t v = (uint8_t)(qs[32 + j] * pow3[n]); w[o++] = (int8_t)((int)(((uint16_t)v * 3) >> 8) - 1); }
        for (int n = 0; n < 4; n++) for (int j = 0; j < 4; j++) { uint8_t v = (uint8_t)(qh[j] * pow3[n]); w[o++] = (int8_t)((int)(((uint16_t)v * 3) >> 8) - 1); }
        gq_set_g(sw, mw, 0, 16, d, 0.f); break; }
    case GQ_TQ2_0: { const uint8_t *qs = b; const float d = gq_f16(gq_rd16(b + 64));
        for (int i = 0; i < 256; i++) { int c = i >> 7, s = (i >> 5) & 3, j = i & 31; w[i] = (int8_t)((int)((qs[c * 32 + j] >> (2 * s)) & 3) - 1); }
        gq_set_g(sw, mw, 0, 16, d, 0.f); break; }
    case GQ_NVFP4: { const uint8_t *qs = b + 4;
        for (int sb = 0; sb < 4; sb++) {
            sw[sb] = gq_ue4m3_half(b[sb]); mw[sb] = 0.f;
            for (int j = 0; j < 8; j++) { w[16 * sb + j] = GGML_KVALUES_MXFP4[qs[8 * sb + j] & 0x0F]; w[16 * sb + 8 + j] = GGML_KVALUES_MXFP4[qs[8 * sb + j] >> 4]; }
        }
        break; }
    case GQ_Q1_0: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
        for (int j = 0; j < 128; j++) w[j] = (qs[j >> 3] >> (j & 7)) & 1 ? 1 : -1;
        gq_set_g(sw, mw, 0, 8, d, 0.f); break; }
    case GQ_Q2_0: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
        for (int j = 0; j < 64; j++) w[j] = (int8_t)((int)((qs[j >> 2] >> ((j & 3) * 2)) & 3) - 1);
        gq_set_g(sw, mw, 0, 4, d, 0.f); break; }
    }
}
/* 1 if type t has an integer form (every block type does; F32/F16/BF16 do not). */
static inline int gq_has_int(int t) { return gq_blk_decoder(t) != NULL; }
/* Unpack n elements (whole blocks) of row src. 0, or -1 for a float/unknown type. */
static int gq_unpack_row(int t, const uint8_t *src, int64_t n, int8_t *w8, float *sw, float *mw) {
    if (!gq_has_int(t) || n % GQ_TYPES[t].blck) return -1;
    const int bl = GQ_TYPES[t].blck, by = GQ_TYPES[t].bytes;
    for (int64_t i = 0; i < n / bl; i++) gq_ublk(t, src + i * by, w8 + i * bl, sw + i * (bl / GQ_G), mw + i * (bl / GQ_G));
    return 0;
}


/* ---- activations in int8, and the integer dot product ----------------
 * An activation row is quantized once per call, in blocks of 32: int8 codes,
 * one float scale per 32 and the code sum per 16 (the min term of the K
 * types). A weight row in integer form (gq_unpack_row) is then dotted as
 *     sum_g dx[g/2] * (sw[g] * sum_{i in g} w8[i]*xq[i] - mw[g] * xs[g])
 * with exact integer inner sums. The only approximation against the float
 * reference is the activation's own rounding to int8, the same one colibri's
 * other integer paths make. GQ_EXACT=1 selects the float reference path. */
static void gq_quant_act(const float *x, int64_t n, int8_t *xq, float *dx, int32_t *xs) {
    for (int64_t b = 0; b < n / 32; b++) {
        const float *v = x + b * 32; float amax = 0.f;
        for (int i = 0; i < 32; i++) { float a = fabsf(v[i]); if (a > amax) amax = a; }
        const float d = amax / 127.f, id = amax > 0.f ? 127.f / amax : 0.f;
        dx[b] = d;
        int32_t s0 = 0, s1 = 0;
        for (int i = 0; i < 32; i++) {
            int q = (int)lrintf(v[i] * id); if (q > 127) q = 127; if (q < -127) q = -127;
            xq[b * 32 + i] = (int8_t)q; if (i < 16) s0 += q; else s1 += q;
        }
        xs[2 * b] = s0; xs[2 * b + 1] = s1;
    }
}
static inline int gq_has_min(int t) { return t == GQ_Q4_1 || t == GQ_Q5_1 || t == GQ_Q2_K || t == GQ_Q4_K || t == GQ_Q5_K; }

static float gq_dot_i8(const int8_t *w8, const float *sw, const float *mw, int has_min,
                       const int8_t *xq, const float *dx, const int32_t *xs, int64_t n) {
    float acc = 0.f;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 vacc = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi16(1);
    for (int64_t b = 0; b < n / 32; b++) {
        __m256i w = _mm256_loadu_si256((const __m256i *)(w8 + b * 32));
        __m256i x = _mm256_loadu_si256((const __m256i *)(xq + b * 32));
        __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w)), ones);
        /* int32 lanes 0..3 are group 2b, 4..7 group 2b+1 */
        const float d = dx[b];
        __m256 sc = _mm256_set_m128(_mm_set1_ps(sw[2 * b + 1] * d), _mm_set1_ps(sw[2 * b] * d));
        vacc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), sc, vacc);
    }
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(vacc), _mm256_extractf128_ps(vacc, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h)); h = _mm_add_ss(h, _mm_movehdup_ps(h));
    acc = _mm_cvtss_f32(h);
#elif defined(__ARM_NEON) && defined(__aarch64__)
    float32x4_t vacc = vdupq_n_f32(0.f);
    for (int64_t b = 0; b < n / 32; b++) {
        int8x16_t w0 = vld1q_s8(w8 + b * 32), w1 = vld1q_s8(w8 + b * 32 + 16);
        int8x16_t x0 = vld1q_s8(xq + b * 32), x1 = vld1q_s8(xq + b * 32 + 16);
#if defined(__ARM_FEATURE_DOTPROD)
        int32x4_t p0 = vdotq_s32(vdupq_n_s32(0), w0, x0), p1 = vdotq_s32(vdupq_n_s32(0), w1, x1);
#else
        int32x4_t p0 = vpaddlq_s16(vmlal_s8(vmull_s8(vget_low_s8(w0), vget_low_s8(x0)), vget_high_s8(w0), vget_high_s8(x0)));
        int32x4_t p1 = vpaddlq_s16(vmlal_s8(vmull_s8(vget_low_s8(w1), vget_low_s8(x1)), vget_high_s8(w1), vget_high_s8(x1)));
#endif
        const float d = dx[b];
        vacc = vmlaq_n_f32(vacc, vcvtq_f32_s32(p0), sw[2 * b] * d);
        vacc = vmlaq_n_f32(vacc, vcvtq_f32_s32(p1), sw[2 * b + 1] * d);
    }
    acc = vaddvq_f32(vacc);
#else
    for (int64_t b = 0; b < n / 32; b++) {
        int32_t s0 = 0, s1 = 0;
        for (int i = 0; i < 16; i++) s0 += (int32_t)w8[b * 32 + i] * xq[b * 32 + i];
        for (int i = 16; i < 32; i++) s1 += (int32_t)w8[b * 32 + i] * xq[b * 32 + i];
        acc += dx[b] * (sw[2 * b] * (float)s0 + sw[2 * b + 1] * (float)s1);
    }
#endif
    if (has_min) {
        float m = 0.f;
        for (int64_t b = 0; b < n / 32; b++) m += dx[b] * (mw[2 * b] * (float)xs[2 * b] + mw[2 * b + 1] * (float)xs[2 * b + 1]);
        acc -= m;
    }
    return acc;
}

static int gq_exact_mode(void) { static int v = -1; if (v < 0) { const char *e = getenv("GQ_EXACT"); v = e && *e && *e != '0'; } return v; }

/* Q8_0 needs no unpack: the block already is 32 int8 codes and one scale. */
static inline float gq_dot_q8_0(const uint8_t *row, const int8_t *xq, const float *dx, int64_t n) {
    float acc = 0.f;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 vacc = _mm256_setzero_ps(); const __m256i ones = _mm256_set1_epi16(1);
    for (int64_t b = 0; b < n / 32; b++) {
        const uint8_t *blk = row + b * 34;
        __m256i w = _mm256_loadu_si256((const __m256i *)(blk + 2)), x = _mm256_loadu_si256((const __m256i *)(xq + b * 32));
        __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w)), ones);
        vacc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(gq_f16(gq_rd16(blk)) * dx[b]), vacc);
    }
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(vacc), _mm256_extractf128_ps(vacc, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h)); h = _mm_add_ss(h, _mm_movehdup_ps(h));
    acc = _mm_cvtss_f32(h);
#elif defined(__ARM_NEON) && defined(__aarch64__)
    float32x4_t vacc = vdupq_n_f32(0.f);
    for (int64_t b = 0; b < n / 32; b++) {
        const uint8_t *blk = row + b * 34;
        int8x16_t w0 = vld1q_s8((const int8_t *)(blk + 2)), w1 = vld1q_s8((const int8_t *)(blk + 18));
        int8x16_t x0 = vld1q_s8(xq + b * 32), x1 = vld1q_s8(xq + b * 32 + 16);
#if defined(__ARM_FEATURE_DOTPROD)
        int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, x0), w1, x1);
#else
        int32x4_t p = vaddq_s32(vpaddlq_s16(vmlal_s8(vmull_s8(vget_low_s8(w0), vget_low_s8(x0)), vget_high_s8(w0), vget_high_s8(x0))),
                                vpaddlq_s16(vmlal_s8(vmull_s8(vget_low_s8(w1), vget_low_s8(x1)), vget_high_s8(w1), vget_high_s8(x1))));
#endif
        vacc = vmlaq_n_f32(vacc, vcvtq_f32_s32(p), gq_f16(gq_rd16(blk)) * dx[b]);
    }
    acc = vaddvq_f32(vacc);
#else
    for (int64_t b = 0; b < n / 32; b++) {
        const uint8_t *blk = row + b * 34; const int8_t *w = (const int8_t *)(blk + 2); int32_t si = 0;
        for (int i = 0; i < 32; i++) si += (int32_t)w[i] * xq[b * 32 + i];
        acc += gq_f16(gq_rd16(blk)) * dx[b] * (float)si;
    }
#endif
    return acc;
}

/* An activation block quantized once and shared by every matrix that reads it
 * (gate and up read the same x). */
typedef struct { int8_t *q; float *d; int32_t *s; int S, I, cap; } gq_act;
static int gq_act_set(gq_act *a, const float *x, int S, int I) {
    if ((int64_t)S * I > a->cap) {
        free(a->q); free(a->d); free(a->s);
        a->cap = S * I; a->q = (int8_t *)malloc((size_t)a->cap); a->d = (float *)malloc((size_t)(a->cap / 32 + 1) * sizeof(float));
        a->s = (int32_t *)malloc((size_t)(a->cap / 16 + 1) * sizeof(int32_t));
        if (!a->q || !a->d || !a->s) { a->cap = 0; return -1; }
    }
    a->S = S; a->I = I;
    for (int s = 0; s < S; s++) gq_quant_act(x + (int64_t)s * I, I, a->q + (int64_t)s * I, a->d + (int64_t)s * (I / 32), a->s + (int64_t)s * (I / 16));
    return 0;
}
static void gq_act_free(gq_act *a) { free(a->q); free(a->d); free(a->s); memset(a, 0, sizeof *a); }

/* per-thread unpack buffers, grown on demand and kept */
static __thread int8_t *gq_tw8; static __thread float *gq_tsw, *gq_tmw; static __thread int gq_tcap;
static inline int gq_tbuf(int I) {
    if (I > gq_tcap) {
        free(gq_tw8); free(gq_tsw); free(gq_tmw);
        gq_tw8 = (int8_t *)malloc((size_t)I); gq_tsw = (float *)malloc((size_t)(I / 16 + 1) * sizeof(float)); gq_tmw = (float *)malloc((size_t)(I / 16 + 1) * sizeof(float));
        gq_tcap = (gq_tw8 && gq_tsw && gq_tmw) ? I : 0;
    }
    return gq_tcap >= I;
}
/* one output row of W against every activation row */
static inline void gq_row_all(float *y, int ystride, const gq_act *a, const uint8_t *row, int t, int I) {
    if (t == GQ_Q8_0) {
        for (int s = 0; s < a->S; s++) y[(int64_t)s * ystride] = gq_dot_q8_0(row, a->q + (int64_t)s * I, a->d + (int64_t)s * (I / 32), I);
        return;
    }
    if (!gq_tbuf(I)) return;
    gq_unpack_row(t, row, I, gq_tw8, gq_tsw, gq_tmw);
    const int hm = gq_has_min(t);
    for (int s = 0; s < a->S; s++)
        y[(int64_t)s * ystride] = gq_dot_i8(gq_tw8, gq_tsw, gq_tmw, hm, a->q + (int64_t)s * I, a->d + (int64_t)s * (I / 32), a->s + (int64_t)s * (I / 16), I);
}
static inline int gq_int_ok(int t, int I) { return !gq_exact_mode() && gq_has_int(t) && I % 32 == 0; }

/* y = W x on an already quantized activation block (types with an integer form). */
static void gq_matmul_act(float *y, const gq_act *a, const uint8_t *W, int t, int O) {
    const int I = a->I; const int64_t rb = gq_row_bytes(t, I);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) gq_row_all(y + o, O, a, W + (int64_t)o * rb, t, I);
}
/* Two matrices on the same activations in one parallel region (gate and up). */
static void gq_matmul2_act(float *y1, const uint8_t *W1, int t1, float *y2, const uint8_t *W2, int t2, const gq_act *a, int O) {
    const int I = a->I; const int64_t rb1 = gq_row_bytes(t1, I), rb2 = gq_row_bytes(t2, I);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < 2 * O; r++) {
        if (r < O) gq_row_all(y1 + r, O, a, W1 + (int64_t)r * rb1, t1, I);
        else       gq_row_all(y2 + (r - O), O, a, W2 + (int64_t)(r - O) * rb2, t2, I);
    }
}

/* y[s*O + o] = W[o,:] . x[s,:]; integer path for every block type, the float
 * reference for F32/F16/BF16, odd widths, or GQ_EXACT=1. 0 or -1. */
static int gq_matmul(float *y, const float *x, const uint8_t *W, int t, int S, int I, int O) {
    if (!gq_int_ok(t, I)) return gq_matmul_f32(y, x, W, t, S, I, O);
    if (gq_row_bytes(t, I) < 0) return -1;
    gq_act a; memset(&a, 0, sizeof a);
    if (gq_act_set(&a, x, S, I)) { gq_act_free(&a); return gq_matmul_f32(y, x, W, t, S, I, O); }
    gq_matmul_act(y, &a, W, t, O);
    gq_act_free(&a);
    return 0;
}

#endif /* GGML_QUANTS_H */
