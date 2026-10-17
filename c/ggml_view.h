/* ggml_view.h -- a GGUF tensor behind a ColiTensorView (tensor.h, COLI_TENSOR_GGML).
 *
 * The view holds the tensor's blocks as the file has them (ggml_quants.h types: Q8_0,
 * Q4_K, IQ2_XS, MXFP4, ...); block_columns carries the ggml type. An engine that reads a
 * GGUF in place (st_gguf.h) hands such a view to its own matmul entries, which come here. */
#ifndef GGML_VIEW_H
#define GGML_VIEW_H

#include <stdint.h>
#include "tensor.h"
#include "ggml_quants.h"

static inline int coli_ggml_view_type(const ColiTensorView *w) { return (int)w->block_columns; }

/* y[s][rows] = W x[s] for S rows of f32 activations: int8 activations per 32 and the
 * integer kernels, or the float reference under GQ_EXACT=1. 0, or -1. */
static inline int coli_ggml_matmul(float *y, const ColiTensorView *w, const float *x, int S) {
    if (!y || !w || !x || w->format != COLI_TENSOR_GGML || !w->data || S < 1 ||
        w->rows < 1 || w->columns < 1 || w->rows > INT32_MAX || w->columns > INT32_MAX) return -1;
    return gq_matmul(y, x, (const uint8_t *)w->data, coli_ggml_view_type(w), S, (int)w->columns, (int)w->rows);
}

#endif /* GGML_VIEW_H */
