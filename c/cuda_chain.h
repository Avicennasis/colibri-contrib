/* cuda_chain.h -- a layer's dense chain on a CUDA device, issued on one stream.
 *
 * The CUDA twin of vk_chain.h. With the dense trunk in VRAM one coli_cuda_matmul at a
 * time, every matrix is a host round trip (the input up, the kernel, the output down,
 * a synchronization): on Qwen3.6-35B about seventy per decode token, the attention
 * core, the norms, the router and the shared expert still on the CPU between them.
 * Here an engine issues a whole layer -- norms, projections, RoPE, attention over a
 * KV cache that lives on the device, the Gated DeltaNet recurrence with its state on
 * the device, gates, the router logits, the shared expert, the residual add -- on one
 * stream, and the residual stream stays on the device from one layer to the next.
 * Only what the CPU needs crosses: the rows the CPU's routed experts read and the
 * router logits, then the routed sum coming back.
 *
 * The pieces, shaped as vk_chain.h's so an engine wires both chains the same way:
 *   - buffers (CcBuf): CC_DEV the device's own (state, scratch); CC_UP written by the
 *     host and read by the device; CC_DOWN written by the device and read by the host.
 *     UP and DOWN are page-locked host memory mapped into the device's address space:
 *     the host writes or reads them in place (cc_ptr), the device reads or writes them
 *     over the bus, as the Vulkan chain's host-visible buffers.
 *   - recording: cc_begin opens a frame; every op goes to the device's stream in order
 *     (a stream is the barrier Vulkan records by hand); cc_submit(wait) ends it with an
 *     event, waited for or not. Frames run in order on the one stream.
 *   - ops: cc_matmul over the resident tensors the engines already upload
 *     (coli_cuda_tensor_upload: the backend's GEMV / GEMM kernel on the chain's
 *     stream), and the chain's kernels, each the arithmetic of the shader of the same
 *     name in shaders/chain_*.comp: cc_norm, cc_rope, cc_attn, cc_dnconv, cc_dnrec,
 *     cc_ew. Offsets and strides are in floats; the parameter structs are field for
 *     field the Vkc ones.
 *
 * The backend exports the ops as one table (coli_cuda_chain_ops): one symbol for the
 * Windows loader, versioned by its size, so a DLL that predates the chain simply has
 * none and the engine stays on the per-matrix path. The cc_* calls below go through
 * the table.
 *
 * Threading: the engine thread only. The expert tier's groups run on the backend's own
 * stream of the device and overlap the chain's frames. A CUDA error inside a frame
 * marks the device lost (cc_lost() says so, every later call returns 0): the engine
 * then rebuilds its state on the CPU, as with the Vulkan chain.
 *
 * tests/test_cuda_chain.cu checks every op against a CPU reference on a real device;
 * tests/qwen36_fake_cuda.h carries a host-side table of the same ops for the engine
 * tests. */
#ifndef COLI_CUDA_CHAIN_H
#define COLI_CUDA_CHAIN_H
#include <stddef.h>
#include <string.h>
#include "backend_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CcBuf CcBuf;
#define CC_DEV  0
#define CC_UP   1
#define CC_DOWN 2

/* chain_norm.comp */
typedef struct { int nseg, D, per_row, x_off, x_row, x_seg, y_off, y_row, y_seg, w_off, w_mod, flags; float eps, post; } CcNorm;
#define CC_NORM_ADD1 1
#define CC_NORM_NOW  2
#define CC_NORM_L2   4
/* chain_rope.comp */
typedef struct { int nseg, per_row, x_off, x_row, x_seg, half_, cs_off, cs_row; } CcRope;
/* chain_attn.comp (k_off/v_off: where the layer's cache starts in kc/vc; the cache
 * head-major, K[(kvh*cap + t)*hd + d]; sel_row > 0: a selection list per row) */
typedef struct { int S, H, KVH, hd, pos_base, cap, q_off, q_row, q_seg, g_off, g_row, g_seg, has_gate,
                 o_off, o_row, sel_off, sel_row; float scale; int k_off, v_off; } CcAttn;
/* chain_dnconv.comp */
typedef struct { int S, CD, CK, in_off, in_row, out_off, out_row, snap_row, order, w_off, ring_off, snap_off; } CcDnConv;
/* chain_dnrec.comp (KD a template parameter: 4, 8, 16, 32, 64, 128 or 256; VD <= 128) */
typedef struct { int S, VH, KH, VD, Ktot, cv_off, cv_row, b_off, b_row, a_off, a_row, z_off, z_row,
                 y_off, y_row, snap_row, flags; float eps, qscale; int st_off, snap_off, prm_off; } CcDnRec;
/* chain_ew.comp */
#define CC_EW_ADD      0
#define CC_EW_COMBINE  1
#define CC_EW_SWIGLU   2
#define CC_EW_HC_LOW   3
#define CC_EW_HC_MIX   4
#define CC_EW_HC_INJ   5
#define CC_EW_HC_APPLY 6
#define CC_EW_SCALE    7
#define CC_EW_GATE_ADD 8
typedef struct { int op, n, D, C, flags, e_row, y_off, a_off, b_off, c_off, e_off; float fc; } CcEw;
/* chain_qsa.comp (mode 0: nb block keys from b0; mode 1: S rows' selections) */
typedef struct { int mode, ID, R, b0, half_, S, pos_base, budget, IQ, q_off, q_row, nbmax, sel_row; float eps;
                 int src_off, w_off, pk_off, nb; } CcQsa;
/* chain_ple.comp (mode 0: the gate over S*C (row, stream) pairs; mode 1: the convolution) */
typedef struct { int mode, S, C, H, CK, NG, keys_off, hyp_off, val_off, snap_row, snap_off; float eps;
                 int prm_off, conv_off, ring_off; } CcPle;
/* chain_route.comp (vk_chain.h's VkcRoute, field for field): every MoE engine's routing
 * for S rows of E logits (l_off + s*l_row): kind 0 softmax, 1 sigmoid, 2 sqrt(softplus)
 * of logit / temp; the bias (b_off) added for the pick only with has_bias 1, to the
 * logit with 2; with n_group > 1 the topk_group best groups' experts only; the K best
 * picks, the lower expert on a tie, in rank order into ids (i_off + s*i_row, -1 for a
 * slot no finite pick fills), their scores into w (w_off + s*w_row), divided by their
 * sum plus norm_eps with norm 1 (when that is above 0) or norm 2 (from K > 1 only),
 * times scale. ids holds ints. */
#define CC_ROUTE_EMAX 1024
#define CC_ROUTE_KMAX 32
typedef struct {
    int S, E, K, kind;
    int l_off, l_row, b_off, has_bias;
    int i_off, i_row, w_off, w_row;
    int n_group, topk_group, norm;
    float temp, scale, norm_eps;
} CcRoute;
/* A routed expert as a device holds it (the CUDA tier's resident tensors: their weights
 * and scales, coli_cuda_tensor_dev); g NULL: not on the device (the host computes it). */
typedef struct { const void *g, *u, *d; const float *gs, *us, *ds; } CcExpert;
/* A layer's routed experts on the device for S rows (cc_moe): x (x_off + s*x_row, D
 * floats) routed to ids and w as cc_route writes them (i_off + s*i_row, w_off + s*w_row,
 * K each); expert e is entry t_off + e of the table, its matrices of format fmt (1 int8
 * rows, 2 int4 rows, 4 int4 with a scale per gs inputs, 8 fp8-e4m3 with a scale per
 * 128 x 128 block: the tier's formats), D and I multiples of 32 up to CC_MOE_DIM_MAX. hid holds S*K rows of
 * I floats (silu(gate) * up, act 0, the only one) and then S*K rows of D (down's
 * output). parts 1: the experts the table holds; parts 2: y (y_off + s*y_row) = the
 * host's row (hy at h_off + s*D, h_off >= 0; else 0) plus each slot the table holds,
 * w * down's output, in rank order (the CPU tier's order: its misses, then the device's
 * results); 3 both. A slot whose id is -1 or whose entry is not on the device adds
 * nothing. */
#define CC_MOE_DIM_MAX 12032   /* D and I: an input row and the fp8 table in 48 KiB of shared memory */
typedef struct {
    int S, K, D, I, fmt, gs, act, parts;
    int t_off, x_off, x_row, i_off, i_row, w_off, w_row, y_off, y_row, h_off;
} CcMoe;
/* one copy over n regions (a KV row per head, say); offsets and counts in floats */
typedef struct { size_t dst, src, n; } CcRegion;
/* counters, for the engines' [chain] lines */
typedef struct {
    unsigned long long frames, waits, ops, matmuls, bytes_up, bytes_down;
    double wait_ms;      /* host time blocked in event waits */
    size_t dev_bytes;    /* live chain buffers on the device */
} CcStats;

/* The table the backend exports. `size` is the backend's sizeof: a field past it is
 * not there (an older DLL), and the engine must not call it. */
typedef struct ColiCudaChainOps {
    size_t size;
    /* after coli_cuda_init, for a configured device; 1 = the chain is up on it. Each
     * device has a context of its own; cc_device makes one current (returns the
     * previous), cc_device_now says which. shutdown closes every context. */
    int  (*init)(int device);
    int  (*ready)(void);
    int  (*lost)(void);
    void (*shutdown)(void);
    int  (*device)(int d);
    int  (*device_now)(void);
    CcBuf *(*buf)(size_t bytes, int kind);                 /* zero-filled; NULL when out of memory */
    void   (*free)(CcBuf *b);                              /* waits for the frames that may read it */
    int    (*reserve)(CcBuf **b, size_t bytes, int kind);  /* at least `bytes`; growing drops the contents */
    void  *(*ptr)(const CcBuf *b);                         /* host mapping (CC_UP, CC_DOWN) */
    size_t (*bytes)(const CcBuf *b);
    int  (*begin)(void);
    int  (*submit)(int wait);
    int  (*finish)(void);                                  /* wait for every submitted frame */
    int  (*copy)(CcBuf *dst, size_t doff, CcBuf *src, size_t soff, size_t n);
    int  (*zero)(CcBuf *dst, size_t off, size_t n);
    int  (*copy_regions)(CcBuf *dst, CcBuf *src, const CcRegion *r, int n);
    int  (*write)(CcBuf *dst, size_t off, const void *src, size_t bytes);   /* into the open frame */
    int  (*read)(CcBuf *src, size_t off, void *dst, size_t bytes);          /* synchronous */
    /* y[S][O] = x[S][I] @ W^T for a resident tensor (its fmt, I, O); x and y at float offsets */
    int  (*matmul)(ColiCudaTensor *t, CcBuf *x, size_t xo, CcBuf *y, size_t yo, int S);
    int  (*norm)(CcBuf *x, CcBuf *w, CcBuf *y, const CcNorm *p);
    int  (*rope)(CcBuf *x, CcBuf *cs, const CcRope *p);
    int  (*attn)(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *sel, const CcAttn *p);
    int  (*dnconv)(CcBuf *in, CcBuf *w, CcBuf *ring, CcBuf *out, CcBuf *snap, const CcDnConv *p);
    int  (*dnrec)(int KD, CcBuf *cv, CcBuf *ab, CcBuf *z, CcBuf *st, CcBuf *prm, CcBuf *y, CcBuf *snap, const CcDnRec *p);
    int  (*ew)(CcBuf *y, CcBuf *a, CcBuf *b, CcBuf *c, CcBuf *e, const CcEw *p);
    void (*stats)(CcStats *st);
    /* Qwen3.8's two: the QSA indexer's block keys and selections, the PLE gate and convolution */
    int  (*qsa)(CcBuf *src, CcBuf *w, CcBuf *pk, CcBuf *cs, CcBuf *sc, CcBuf *sel, const CcQsa *p);
    int  (*ple)(CcBuf *keys, CcBuf *hyp, CcBuf *val, CcBuf *prm, CcBuf *gated, CcBuf *normv, CcBuf *conv, CcBuf *ring, const CcPle *p);
    /* The routed experts in the frame (optional: a table that ends before them has none).
     * tab: CcExpert entries (written with write, offsets in floats as ever: an entry is
     * sizeof(CcExpert) / 4 of them). signal: once the frame's earlier ops are done, word
     * `word` of a DOWN buffer holds v (the host polls it); wait_host: the frame's later
     * ops wait until word `word` of an UP buffer is at least v (the host writes it). A
     * hybrid step: route, signal, the device's experts, wait_host, the sum with the
     * host's rows; the host computes its experts between the two. */
    int  (*route)(CcBuf *logits, CcBuf *bias, CcBuf *ids, CcBuf *w, const CcRoute *p);
    int  (*moe)(CcBuf *tab, CcBuf *x, CcBuf *ids, CcBuf *w, CcBuf *hid, CcBuf *y, CcBuf *hy, const CcMoe *p);
    int  (*signal)(CcBuf *flag, size_t word, unsigned v);
    int  (*wait_host)(CcBuf *flag, size_t word, unsigned v);
    /* a resident tensor's device copy, for a CcExpert: its weights, scales and the scales'
     * group size (fmt 4); 0 for none */
    int  (*tensor_dev)(const ColiCudaTensor *t, const void **w, const float **sc, int *gs);
} ColiCudaChainOps;
/* the table's size without the optional ops: a backend at least this big has the chain */
#define CC_OPS_BASE_SIZE (offsetof(ColiCudaChainOps, route))

/* The backend's table; NULL from a backend without the chain. Optional in the DLL
 * loader, like every entry point added after the first release. */
COLI_CUDA_DLLEXPORT const ColiCudaChainOps *coli_cuda_chain_ops(void);

/* ---- the engine's calls, through the table ----------------------------------- */
#ifndef COLI_CUDA_CHAIN_NO_WRAPPERS
static inline const ColiCudaChainOps *cc_ops(void) {
    static const ColiCudaChainOps *t; static int asked;
    if (!asked) { asked = 1; t = coli_cuda_chain_ops(); if (t && t->size < CC_OPS_BASE_SIZE) t = NULL; }
    return t;
}
static inline int cc_available(void) { return cc_ops() != NULL; }
static inline int  cc_init(int device) { const ColiCudaChainOps *t = cc_ops(); return t ? t->init(device) : 0; }
static inline int  cc_ready(void) { const ColiCudaChainOps *t = cc_ops(); return t ? t->ready() : 0; }
static inline int  cc_lost(void) { const ColiCudaChainOps *t = cc_ops(); return t ? t->lost() : 1; }
static inline void cc_shutdown(void) { const ColiCudaChainOps *t = cc_ops(); if (t) t->shutdown(); }
static inline int  cc_device(int d) { const ColiCudaChainOps *t = cc_ops(); return t ? t->device(d) : -1; }
static inline int  cc_device_now(void) { const ColiCudaChainOps *t = cc_ops(); return t ? t->device_now() : -1; }
static inline CcBuf *cc_buf(size_t bytes, int kind) { const ColiCudaChainOps *t = cc_ops(); return t ? t->buf(bytes, kind) : NULL; }
static inline void   cc_free(CcBuf *b) { const ColiCudaChainOps *t = cc_ops(); if (t && b) t->free(b); }
static inline int    cc_reserve(CcBuf **b, size_t bytes, int kind) { const ColiCudaChainOps *t = cc_ops(); return t ? t->reserve(b, bytes, kind) : 0; }
static inline void  *cc_ptr(const CcBuf *b) { const ColiCudaChainOps *t = cc_ops(); return t && b ? t->ptr(b) : NULL; }
static inline size_t cc_bytes(const CcBuf *b) { const ColiCudaChainOps *t = cc_ops(); return t && b ? t->bytes(b) : 0; }
static inline int cc_begin(void) { const ColiCudaChainOps *t = cc_ops(); return t ? t->begin() : 0; }
static inline int cc_submit(int wait) { const ColiCudaChainOps *t = cc_ops(); return t ? t->submit(wait) : 0; }
static inline int cc_finish(void) { const ColiCudaChainOps *t = cc_ops(); return t ? t->finish() : 0; }
static inline int cc_copy(CcBuf *dst, size_t doff, CcBuf *src, size_t soff, size_t n) { const ColiCudaChainOps *t = cc_ops(); return t ? t->copy(dst, doff, src, soff, n) : 0; }
static inline int cc_zero(CcBuf *dst, size_t off, size_t n) { const ColiCudaChainOps *t = cc_ops(); return t ? t->zero(dst, off, n) : 0; }
static inline int cc_copy_regions(CcBuf *dst, CcBuf *src, const CcRegion *r, int n) { const ColiCudaChainOps *t = cc_ops(); return t ? t->copy_regions(dst, src, r, n) : 0; }
static inline int cc_write(CcBuf *dst, size_t off, const void *src, size_t bytes) { const ColiCudaChainOps *t = cc_ops(); return t ? t->write(dst, off, src, bytes) : 0; }
static inline int cc_read(CcBuf *src, size_t off, void *dst, size_t bytes) { const ColiCudaChainOps *t = cc_ops(); return t ? t->read(src, off, dst, bytes) : 0; }
static inline int cc_matmul(ColiCudaTensor *w, CcBuf *x, size_t xo, CcBuf *y, size_t yo, int S) { const ColiCudaChainOps *t = cc_ops(); return t ? t->matmul(w, x, xo, y, yo, S) : 0; }
static inline int cc_norm(CcBuf *x, CcBuf *w, CcBuf *y, const CcNorm *p) { const ColiCudaChainOps *t = cc_ops(); return t ? t->norm(x, w, y, p) : 0; }
static inline int cc_rope(CcBuf *x, CcBuf *cs, const CcRope *p) { const ColiCudaChainOps *t = cc_ops(); return t ? t->rope(x, cs, p) : 0; }
static inline int cc_attn(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *sel, const CcAttn *p) { const ColiCudaChainOps *t = cc_ops(); return t ? t->attn(q, kc, vc, o, gate, sel, p) : 0; }
static inline int cc_dnconv(CcBuf *in, CcBuf *w, CcBuf *ring, CcBuf *out, CcBuf *snap, const CcDnConv *p) { const ColiCudaChainOps *t = cc_ops(); return t ? t->dnconv(in, w, ring, out, snap, p) : 0; }
static inline int cc_dnrec(int KD, CcBuf *cv, CcBuf *ab, CcBuf *z, CcBuf *st, CcBuf *prm, CcBuf *y, CcBuf *snap, const CcDnRec *p) { const ColiCudaChainOps *t = cc_ops(); return t ? t->dnrec(KD, cv, ab, z, st, prm, y, snap, p) : 0; }
static inline int cc_ew(CcBuf *y, CcBuf *a, CcBuf *b, CcBuf *c, CcBuf *e, const CcEw *p) { const ColiCudaChainOps *t = cc_ops(); return t ? t->ew(y, a, b, c, e, p) : 0; }
static inline void cc_stats(CcStats *st) { const ColiCudaChainOps *t = cc_ops(); if (t) t->stats(st); else memset(st, 0, sizeof *st); }
static inline int cc_qsa(CcBuf *src, CcBuf *w, CcBuf *pk, CcBuf *cs, CcBuf *sc, CcBuf *sel, const CcQsa *p) { const ColiCudaChainOps *t = cc_ops(); return t ? t->qsa(src, w, pk, cs, sc, sel, p) : 0; }
static inline int cc_ple(CcBuf *keys, CcBuf *hyp, CcBuf *val, CcBuf *prm, CcBuf *gated, CcBuf *normv, CcBuf *conv, CcBuf *ring, const CcPle *p) { const ColiCudaChainOps *t = cc_ops(); return t ? t->ple(keys, hyp, val, prm, gated, normv, conv, ring, p) : 0; }
/* the routed experts in the frame: a backend whose table has them */
static inline int cc_moe_ready(void) { const ColiCudaChainOps *t = cc_ops(); return t && t->size >= offsetof(ColiCudaChainOps, tensor_dev) + sizeof t->tensor_dev; }
static inline int cc_route(CcBuf *l, CcBuf *b, CcBuf *ids, CcBuf *w, const CcRoute *p) { return cc_moe_ready() ? cc_ops()->route(l, b, ids, w, p) : 0; }
static inline int cc_moe(CcBuf *tab, CcBuf *x, CcBuf *ids, CcBuf *w, CcBuf *hid, CcBuf *y, CcBuf *hy, const CcMoe *p) { return cc_moe_ready() ? cc_ops()->moe(tab, x, ids, w, hid, y, hy, p) : 0; }
static inline int cc_signal(CcBuf *flag, size_t word, unsigned v) { return cc_moe_ready() ? cc_ops()->signal(flag, word, v) : 0; }
static inline int cc_wait_host(CcBuf *flag, size_t word, unsigned v) { return cc_moe_ready() ? cc_ops()->wait_host(flag, word, v) : 0; }
static inline int cc_tensor_dev(const ColiCudaTensor *w, const void **p, const float **sc, int *gs) { return cc_moe_ready() ? cc_ops()->tensor_dev(w, p, sc, gs) : 0; }
#endif

#ifdef __cplusplus
}
#endif
#endif
