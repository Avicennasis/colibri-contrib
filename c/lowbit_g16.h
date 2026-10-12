/* lowbit_g16.h — colibri's int2-g16 and int3-g16 routed-expert formats and their kernels.
 *
 *   codes   a row of I weights (I % 128 == 0) is I/128 blocks. Each block holds the low
 *           two bits of its 128 codes as int2_g32.h does (32 bytes: byte k = elements k,
 *           k+32, k+64, k+96 in bits 0-1, 2-3, 4-5, 6-7) and, for 3 bits, then the high
 *           bits (16 bytes: byte k = elements k, k+16, k+32, ..., k+112 in bits 0..7, so
 *           a group of 32 is bits 2h and 2h+1 of the same 16 bytes: one variable shift
 *           per half of a register finds them). A block is 32 bytes at 2 bits, 48 at 3.
 *   values  code -> level, the four-level (-10 -3 3 10) or eight-level (-26 -16 -9 -3 3 9
 *           16 26) Lloyd-Max grid of a bell-shaped weight in integers: fitted per layer and
 *           matrix on Qwen3.8's experts the levels came out the same, so the grid is fixed
 *           and one pshufb decodes it.
 *   scales  one 4-bit scale per 16 inputs (two to a byte: group 2k low, 2k+1 high, so a row
 *           has I/32 bytes) under one f32 per row: w = d[row] * q * level. At the same size
 *           as one byte per 32, half the groups: on 24 Qwen3.8 experts the weights' error
 *           is 5% lower at 2 bits and 9% lower at 3.
 *   size    2.25 and 3.25 bits per weight, plus 32 bits per row.
 *   x       the activation contract of int2_g32.h (x2_quant_act): int8 per 32, one f32
 *           scale each.
 *
 * Per group of 32 the two 4-bit scales go into the integer sum itself: |level| * sign(x)
 * byte pairs (maddubs, <= 2*26*127) multiplied by a vector holding the first scale for the
 * group's first 16 elements and the second for the rest (madd, <= 15 * 2 * 6604), so the
 * group costs one f32 fma, d * dx, as int2_g32.h's does. The scalar reference defines the
 * numerics (lane l = elements 4l..4l+3 times its half's scale, one fmaf per group with
 * d*dx[g], the lanes summed as GQ_HSUM8); the AVX2 kernel reproduces it bit for bit for
 * every S. Header-only, all static, pure compute. */
#ifndef COLI_LOWBIT_G16_H
#define COLI_LOWBIT_G16_H

#include <string.h>
#include "int2_g32.h"   /* x2_quant_act, x2_hsum8_scalar, X2_MAX_ROWS */

static const int8_t XL_LEVELS2[4] = { -10, -3, 3, 10 };
static const int8_t XL_LEVELS3[8] = { -26, -16, -9, -3, 3, 9, 16, 26 };

static inline int    xl_layout_ok(int I)             { return I > 0 && I % 128 == 0; }
static inline size_t xl_block_bytes(int bits)        { return bits == 3 ? 48 : 32; }
static inline size_t xl_code_bytes(int bits, int I)  { return (size_t)I / 128 * xl_block_bytes(bits); }
static inline size_t xl_scale_bytes(int I)           { return (size_t)I / 32; }

/* code of element i of a row (tests, packing checks) */
static inline int xl_code(const uint8_t *row, int bits, int i) {
    const uint8_t *blk = row + (size_t)(i / 128) * xl_block_bytes(bits);
    int k = i % 128, h = k / 32, j = k % 32;
    int c = (blk[j] >> (2 * h)) & 3;
    if (bits == 3) c |= ((blk[32 + k % 16] >> (k / 16)) & 1) << 2;
    return c;
}
static inline int xl_level(int bits, int code) { return bits == 3 ? XL_LEVELS3[code] : XL_LEVELS2[code]; }
static inline int xl_scale(const uint8_t *q, int g16) { return (q[g16 / 2] >> (4 * (g16 & 1))) & 15; }

/* y = sum_i w[i] * x[i] for one weight row and one activation row (S = 1) */
static inline float xl_dot_ref(const uint8_t *codes, const uint8_t *q, float d, int bits,
                               const int8_t *xq, const float *dx, int I) {
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int g = 0; g < I / 32; g++) {
        const int8_t *xb = xq + (size_t)g * 32;
        const float sc = d * dx[g];
        for (int l = 0; l < 8; l++) {
            int32_t s = 0;
            for (int j = 4 * l; j < 4 * l + 4; j++) s += xl_level(bits, xl_code(codes, bits, g * 32 + j)) * xb[j];
            s *= xl_scale(q, 2 * g + (l >= 4));
            acc[l] = fmaf((float)s, sc, acc[l]);
        }
    }
    return x2_hsum8_scalar(acc);
}

#ifdef X2_HAVE_AVX2
#define XL_HAVE_AVX2 1
static inline __attribute__((always_inline)) void xl_dot_body(float *y, int ystride, const uint8_t *codes, const uint8_t *q,
                                                             float d, const int8_t *xq, const float *dx, int I,
                                                             const int S, const int bits) {
    const __m256i lut = bits == 3 ? _mm256_setr_epi8(-26, -16, -9, -3, 3, 9, 16, 26, 0, 0, 0, 0, 0, 0, 0, 0,
                                                     -26, -16, -9, -3, 3, 9, 16, 26, 0, 0, 0, 0, 0, 0, 0, 0)
                                  : _mm256_setr_epi8(-10, -3, 3, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                     -10, -3, 3, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m256i m3 = _mm256_set1_epi8(3), one = _mm256_set1_epi8(1);
    const int ng = I / 32;
    const size_t bb_bytes = xl_block_bytes(bits);
    __m256 acc[X2_MAX_ROWS];
    for (int s = 0; s < S; s++) acc[s] = _mm256_setzero_ps();
    for (int b = 0; b < I / 128; b++) {
        const uint8_t *blk = codes + (size_t)b * bb_bytes;
        const __m256i bb = _mm256_loadu_si256((const __m256i *)blk);
        /* the high plane in both halves: group h's first 16 elements are bit 2h, the rest 2h+1 */
        const __m256i hp = bits == 3 ? _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(blk + 32)))
                                     : _mm256_setzero_si256();
        for (int h = 0; h < 4; h++) {
            const int g = 4 * b + h;
            __m256i c = _mm256_and_si256(h == 0 ? bb : _mm256_srli_epi16(bb, 2 * h), m3);
            if (bits == 3) {
                const __m256i cnt = _mm256_setr_epi32(2 * h, 2 * h, 2 * h, 2 * h, 2 * h + 1, 2 * h + 1, 2 * h + 1, 2 * h + 1);
                c = _mm256_or_si256(c, _mm256_slli_epi16(_mm256_and_si256(_mm256_srlv_epi32(hp, cnt), one), 2));
            }
            const __m256i w = _mm256_shuffle_epi8(lut, c), aw = _mm256_sign_epi8(w, w);
            const int qb = q[g];
            const __m256i qv = _mm256_set_m128i(_mm_set1_epi16((short)(qb >> 4)), _mm_set1_epi16((short)(qb & 15)));
            for (int s = 0; s < S; s++) {
                const __m256i x = _mm256_loadu_si256((const __m256i *)(xq + (size_t)s * I + (size_t)g * 32));
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(x, w)), qv);
                acc[s] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * dx[(size_t)s * ng + g]), acc[s]);
            }
        }
    }
    for (int s = 0; s < S; s++) y[(size_t)s * ystride] = x2_hsum8(acc[s]);
}
#endif

/* S (1..X2_MAX_ROWS) activation rows of one weight row; xq/dx row-major [S][I], [S][I/32] */
static inline void xl_dot(float *y, int ystride, const uint8_t *codes, const uint8_t *q, float d, int bits,
                          const int8_t *xq, const float *dx, int I, int S) {
#ifdef XL_HAVE_AVX2
#define XL_CASE(n) case n: if (bits == 3) xl_dot_body(y, ystride, codes, q, d, xq, dx, I, n, 3); \
                           else xl_dot_body(y, ystride, codes, q, d, xq, dx, I, n, 2); break;
    switch (S) { XL_CASE(1) XL_CASE(2) XL_CASE(3) default: XL_CASE(4) }
#undef XL_CASE
#else
    for (int s = 0; s < S; s++)
        y[(size_t)s * ystride] = xl_dot_ref(codes, q, d, bits, xq + (size_t)s * I, dx + (size_t)s * (I / 32), I);
#endif
}

/* y[s*O + o] = W[o,:] . x[s,:] for an int2/int3-g16 matrix: codes [O][xl_code_bytes],
 * scales [O][I/32], d [O]. As x2_matmul: activations quantized once, rows four at a time.
 * 0, or -1 when out of memory. */
static inline int xl_matmul(float *y, const float *x, const uint8_t *codes, const uint8_t *q, const float *d,
                            int bits, int S, int I, int O) {
    const size_t cb = xl_code_bytes(bits, I), qb = xl_scale_bytes(I); const int ng = I / 32;
    int8_t *xq = (int8_t *)malloc((size_t)S * I); float *dx = (float *)malloc((size_t)S * ng * sizeof(float));
    if (!xq || !dx) { free(xq); free(dx); return -1; }
    for (int s = 0; s < S; s++) x2_quant_act(x + (size_t)s * I, I, xq + (size_t)s * I, dx + (size_t)s * ng);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++)
        for (int s0 = 0; s0 < S; s0 += X2_MAX_ROWS) {
            const int n = S - s0 < X2_MAX_ROWS ? S - s0 : X2_MAX_ROWS;
            xl_dot(y + (size_t)s0 * O + o, O, codes + (size_t)o * cb, q + (size_t)o * qb, d[o], bits,
                   xq + (size_t)s0 * I, dx + (size_t)s0 * ng, I, n);
        }
    free(xq); free(dx);
    return 0;
}

#endif /* COLI_LOWBIT_G16_H */
