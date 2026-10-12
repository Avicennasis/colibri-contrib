/* vq2_g32.h — colibri's vq2 routed-expert format: two-stage vector codes, and its kernels.
 *
 *   codes   a row of I weights (I % 32 == 0) is I/8 vectors of 8 weights; each vector is
 *           two bytes, an index into codebook A and one into codebook B (I/4 bytes a row,
 *           2 bits a weight), in column order.
 *   values  the vector is A[a] + B[b]: two 256 x 8 int8 codebooks of the model (the
 *           sidecar's index carries them), scaled so that in every coordinate the
 *           largest |A| plus the largest |B| is at most 127: any sum is an int8 and the
 *           integer path of int2_g32.h / lowbit_g16.h reads it as it reads their levels.
 *           Fitted by k-means to the experts' weight vectors (over the RMS of their group
 *           of 32), then rounded.
 *   scales  one 4-bit scale per 32 inputs (two to a byte: group 2k low, 2k+1 high, I/64
 *           bytes a row) under one f32 per row: w = d[row] * q * (A[a] + B[b]), the
 *           codebooks' own scale folded into d.
 *   size    2.125 bits per weight, plus 32 bits per row.
 *   x       the activation contract of int2_g32.h (x2_quant_act): int8 per 32, one f32
 *           scale each.
 *
 * Measured on 23 Qwen3.8 experts against the rows they receive (held-out tokens, GPTQ
 * under the same H): output error 0.371 against 0.398 for int2-g16 and 0.435 for ISTA's
 * Q2_0, at 2.125 bits a weight against 2.27 and 2.25; on 6 of them the converter's beam
 * of 4 and int8 codebooks cost 0.004 over a beam of 16 in floats.
 *
 * The scalar reference defines the numerics (lane l = elements 4l..4l+3 of a group, one
 * fmaf per group with d*dx[g] on the lane's integer sum times q, the lanes summed as
 * x2_hsum8); the AVX2 kernel reproduces it bit for bit for every number of rows.
 * Header-only, all static, pure compute. */
#ifndef COLI_VQ2_G32_H
#define COLI_VQ2_G32_H

#include <string.h>
#include "int2_g32.h"   /* x2_quant_act, x2_hsum8_scalar, X2_MAX_ROWS */

#define VQ2_BOOK 256
typedef struct { int8_t a[VQ2_BOOK][8], b[VQ2_BOOK][8]; } Vq2Books;

static inline int    vq2_layout_ok(int I)    { return I > 0 && I % 32 == 0; }
static inline size_t vq2_code_bytes(int I)   { return (size_t)I / 4; }
static inline size_t vq2_scale_bytes(int I)  { return (size_t)I / 64 + (I % 64 ? 1 : 0); }
static inline int    vq2_scale(const uint8_t *q, int g) { return (q[g / 2] >> (4 * (g & 1))) & 15; }
/* weight value (before d and q) of element i of a row */
static inline int vq2_value(const Vq2Books *bk, const uint8_t *row, int i) {
    const uint8_t *v = row + (size_t)(i / 8) * 2;
    return bk->a[v[0]][i % 8] + bk->b[v[1]][i % 8];
}

/* y = sum_i w[i] * x[i] for one weight row and one activation row (S = 1) */
static inline float vq2_dot_ref(const Vq2Books *bk, const uint8_t *codes, const uint8_t *q, float d,
                                const int8_t *xq, const float *dx, int I) {
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int g = 0; g < I / 32; g++) {
        const int8_t *xb = xq + (size_t)g * 32;
        const float sc = d * dx[g];
        const int qs = vq2_scale(q, g);
        for (int l = 0; l < 8; l++) {
            int32_t s = 0;
            for (int j = 4 * l; j < 4 * l + 4; j++) s += vq2_value(bk, codes, g * 32 + j) * xb[j];
            acc[l] = fmaf((float)(s * qs), sc, acc[l]);
        }
    }
    return x2_hsum8_scalar(acc);
}

#ifdef X2_HAVE_AVX2
#define VQ2_HAVE_AVX2 1
static inline __attribute__((always_inline)) void vq2_dot_body(float *y, int ystride, const Vq2Books *bk,
                                                              const uint8_t *codes, const uint8_t *q, float d,
                                                              const int8_t *xq, const float *dx, int I, const int S) {
    const int ng = I / 32;
    __m256 acc[X2_MAX_ROWS];
    for (int s = 0; s < S; s++) acc[s] = _mm256_setzero_ps();
    for (int g = 0; g < ng; g++) {
        const uint8_t *v = codes + (size_t)g * 8;   /* four vectors: a0 b0 a1 b1 a2 b2 a3 b3 */
        int64_t a0, a1, a2, a3, b0, b1, b2, b3;
        memcpy(&a0, bk->a[v[0]], 8); memcpy(&b0, bk->b[v[1]], 8);
        memcpy(&a1, bk->a[v[2]], 8); memcpy(&b1, bk->b[v[3]], 8);
        memcpy(&a2, bk->a[v[4]], 8); memcpy(&b2, bk->b[v[5]], 8);
        memcpy(&a3, bk->a[v[6]], 8); memcpy(&b3, bk->b[v[7]], 8);
        const __m256i w = _mm256_add_epi8(_mm256_set_epi64x(a3, a2, a1, a0), _mm256_set_epi64x(b3, b2, b1, b0));
        const __m256i aw = _mm256_sign_epi8(w, w), qv = _mm256_set1_epi16((short)vq2_scale(q, g));
        for (int s = 0; s < S; s++) {
            const __m256i x = _mm256_loadu_si256((const __m256i *)(xq + (size_t)s * I + (size_t)g * 32));
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(x, w)), qv);
            acc[s] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * dx[(size_t)s * ng + g]), acc[s]);
        }
    }
    for (int s = 0; s < S; s++) y[(size_t)s * ystride] = x2_hsum8(acc[s]);
}
#endif

/* S (1..X2_MAX_ROWS) activation rows of one weight row; xq/dx row-major [S][I], [S][I/32] */
static inline void vq2_dot(float *y, int ystride, const Vq2Books *bk, const uint8_t *codes, const uint8_t *q, float d,
                           const int8_t *xq, const float *dx, int I, int S) {
#ifdef VQ2_HAVE_AVX2
    switch (S) {
    case 1: vq2_dot_body(y, ystride, bk, codes, q, d, xq, dx, I, 1); break;
    case 2: vq2_dot_body(y, ystride, bk, codes, q, d, xq, dx, I, 2); break;
    case 3: vq2_dot_body(y, ystride, bk, codes, q, d, xq, dx, I, 3); break;
    default: vq2_dot_body(y, ystride, bk, codes, q, d, xq, dx, I, 4); break;
    }
#else
    for (int s = 0; s < S; s++)
        y[(size_t)s * ystride] = vq2_dot_ref(bk, codes, q, d, xq + (size_t)s * I, dx + (size_t)s * (I / 32), I);
#endif
}

/* y[s*O + o] = W[o,:] . x[s,:] for a vq2 matrix: codes [O][I/4], scales [O][vq2_scale_bytes],
 * d [O]. Activations quantized once, rows four at a time. 0, or -1 when out of memory. */
static inline int vq2_matmul(float *y, const float *x, const Vq2Books *bk, const uint8_t *codes, const uint8_t *q,
                             const float *d, int S, int I, int O) {
    const size_t cb = vq2_code_bytes(I), qb = vq2_scale_bytes(I); const int ng = I / 32;
    int8_t *xq = (int8_t *)malloc((size_t)S * I); float *dx = (float *)malloc((size_t)S * ng * sizeof(float));
    if (!xq || !dx) { free(xq); free(dx); return -1; }
    for (int s = 0; s < S; s++) x2_quant_act(x + (size_t)s * I, I, xq + (size_t)s * I, dx + (size_t)s * ng);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++)
        for (int s0 = 0; s0 < S; s0 += X2_MAX_ROWS) {
            const int n = S - s0 < X2_MAX_ROWS ? S - s0 : X2_MAX_ROWS;
            vq2_dot(y + (size_t)s0 * O + o, O, bk, codes + (size_t)o * cb, q + (size_t)o * qb, d[o],
                    xq + (size_t)s0 * I, dx + (size_t)s0 * ng, I, n);
        }
    free(xq); free(dx);
    return 0;
}

#endif /* COLI_VQ2_G32_H */
