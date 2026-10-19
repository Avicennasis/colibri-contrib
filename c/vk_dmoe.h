/* vk_dmoe.h -- a layer's routed experts inside an engine's dense chain, with no host step
 * (vk_chain.h: vkc_route, vkc_moe; vk_tier.h: vkt_dev_table, vkt_dev_count). Included by
 * the engines' *_chain.h in a COLI_VULKAN build.
 *
 * A chain normally ends its frame at every MoE layer: the router's logits go down, the
 * host routes and joins the tier's device batch with the CPU's experts, the routed sum
 * comes back up. For a step of up to VKC_DMOE_ROWS rows (decode, a short verify) and a
 * layer whose experts the tier holds all of on the chain's device, the frame instead
 * routes (vkc_route) and computes the experts (vkc_moe) itself and goes on to the next
 * layer: one frame for the whole forward when every layer qualifies, which the tier's
 * warm start makes the usual case on a device that holds the model. An engine that sets
 * big gets a prompt chunk's rows the same way, each layer whose experts are block formats
 * (a GGUF's) through vkc_moe_rows' grouped GEMMs; a layer whose formats have none
 * takes the host's step in that chunk, as every layer did before.
 *
 * The engine keeps what is its own: its router's parameters (VkcRoute), its expert's
 * activation (VkcMoe), the conditions under which its host routing has state the device
 * does not keep (cache routing, a momentum PILOT reads, ROUTE_TRACE), its K/V rows, and
 * its books after the frame, from the ids that come down with it (vkc_dmoe_ids):
 * vkt_dev_count for the tier, and whatever its router counts (the history, HITS).
 *
 *   VkcDmoe d = {0};
 *   if (vkc_dmoe_prepare(&d, lo, L, E, K, D, I, rows)) ...      once per step
 *   vkc_dmoe_upload(&d)                                          in every frame the step begins
 *   if (vkc_dmoe_on(&d, i)) vkc_dmoe_layer(&d, i, n, lg, bias, h2, &route, &moe)
 *                                                                then d.y holds the routed sum
 *   after the frame: vkc_dmoe_ids(&d, i) for each such layer
 *
 * The table holds device addresses of the tier's experts. A layer the host steps through
 * (it holds only part of its experts) runs the tier, which may evict and free experts and
 * upload others; a frame that read the table from before would read freed memory (on a
 * software device, the process's own: a crash, wrong tokens, a stalled frame). So every
 * frame the step begins, the first one and each after a host layer, calls
 * vkc_dmoe_upload, which reads the table again when the tier's residents changed
 * (vkt_epoch), and vkc_dmoe_on turns a layer down while its table is not the current one.
 *
 * COLI_VK_CHAIN_MOE=0 turns it off for every engine. */
#ifndef COLI_VK_DMOE_H
#define COLI_VK_DMOE_H
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <math.h>
#include "vk_chain.h"
#include "vk_tier.h"
#include "backend_vulkan.h"
#include "ggml_quants.h"

#define VKC_DMOE_ROWS 8

/* Hybrid steps (COLI_VK_HYBRID, on where the device has timeline semaphores): in a step of
 * up to VKC_DMOE_ROWS rows, the experts of the tier's host pool (system RAM the device
 * would read over PCIe, about 10 GB/s on an RTX PRO 4000 at PCIe 4.0 x8) are computed by
 * the CPU in place, at its memory's speed, while the device computes the ones it holds:
 *   frame A  ... the router, the route, the input rows copied down; signals the device's
 *            timeline (vkc_hybrid_submit)
 *   frame B  the device's experts (a table where the host pool's are zero)
 *   frame C  waits for the host's timeline, adds the CPU's sum, goes on
 * A worker thread takes the steps in order: it waits for A, computes the host pool's
 * pairs as the CPU computes a GGUF's matmuls (ggml_quants.h: int8 activations, integer
 * dots), writes their weighted sum and signals. A prompt chunk keeps the full table. */
#define VKC_HYB_Q 256
struct VkcDmoe_;
typedef struct {
    struct VkcDmoe_ *d; int i, n; uint64_t g;
    int act; float limit;
} VkcHybJob;
typedef struct {
    pthread_t th; int th_on, quit, err;
    pthread_mutex_t mx; pthread_cond_t cv, cv_idle;
    VkcHybJob q[VKC_HYB_Q]; int qh, qt, busy;
    unsigned long long steps, pairs;   /* steps done, host pool pairs computed */
    double cpu_ms;                     /* the worker's compute time */
    float *gu, *hh, *dn; int cap;      /* its scratch: gate and up, the hidden rows, down, for cap pairs */
} VkcHyb;
/* one worker for every engine's steps on the device: the host's timeline rises in its order */
static VkcHyb g_vkc_hyb;
static uint64_t g_vkc_hyb_seq;
static double vkc_hyb_now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

typedef struct VkcDmoe_ {
    int lo, L, E, K;               /* the chain's layers lo..lo+L-1, E experts each, top K */
    uint8_t *on;                   /* per layer: the tier holds all of its experts here */
    uint32_t *tabh;                /* their entries, as the tier gave them at `epoch` */
    unsigned epoch; int read, dirty, off;
    VkcBuf *tab, *rid, *rwt, *hid, *y;   /* the table, ids [L][cap][K] (down), weights, hidden rows, the routed sum */
    unsigned long long steps;      /* layer steps that ran so */
    /* big: the engine takes a step of more than VKC_DMOE_ROWS rows too (a prompt chunk),
     * its layers through vkc_moe_rows when their experts are block formats; cap the rows
     * the buffers hold a layer (VKC_DMOE_ROWS, or the step's) */
    int big, cap;
    int big_step;                  /* this step's rows are more than VKC_DMOE_ROWS */
    uint8_t *rows_off;             /* per layer, in a big step: its formats have no grouped GEMM (the host's step) */
    VkcMoeRows rows;
    unsigned long long big_steps;  /* layer steps of more than VKC_DMOE_ROWS rows */
    /* hybrid steps: the table where the host pool's experts are zero (tabhh, tabhyb on the
     * device), their rows for the CPU (hx [L][E]), the input rows copied down (hxx), the
     * CPU's sum (cy), its worker, and the timeline's last value */
    int hybrid;
    uint32_t *tabhh; VktHostExpert *hx; VkcBuf *tabhyb, *hxx, *cy;
    int *hx_n;                     /* per layer: its experts the CPU computes (0: the full table's step) */
    int D, I;
    unsigned long long hyb_steps;
} VkcDmoe;

static int vkc_dmoe_wanted(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_CHAIN_MOE"); on = !(e && *e == '0'); }
    return on;
}
static int vkc_hybrid_wanted(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_HYBRID"); on = !(e && *e == '0'); }
    return on && vkc_hybrid_ready();
}
/* ---- the hybrid worker ---- */
/* the worker's team: COLI_VK_HYBRID_THREADS, else OpenMP's */
static int vkc_hyb_threads(void) {
    static int n = -1;
    if (n < 0) { const char *e = getenv("COLI_VK_HYBRID_THREADS"); n = e && atoi(e) > 0 ? atoi(e) : 0; }
#ifdef _OPENMP
    return n > 0 ? n : omp_get_max_threads();
#else
    return 1;
#endif
}
/* a row of type t (I inputs) against act a's one row: the CPU's integer form, or with
 * GQ_EXACT=1 its float reference (gq_dequant_row), as its own matmuls would take it */
static inline void vkc_hyb_row(float *y, const gq_act *a, const float *xf, const uint8_t *row, int t, int I, float *tmp) {
    if (!gq_exact_mode()) { gq_row_all(y, 1, a, row, t, I); return; }
    double acc = 0; gq_dequant_row(t, row, tmp, I);
    for (int i = 0; i < I; i++) acc += (double)tmp[i] * xf[i];
    *y = (float)acc;
}
/* layer i's host pool pairs of the step's n rows, their weighted sum into the CPU's rows */
static void vkc_hyb_compute(VkcDmoe *d, const VkcHybJob *j) {
    int E = d->E, K = d->K, D = d->D, I = d->I, n = j->n;
    const int *ids = (const int *)vkc_ptr(d->rid) + (size_t)j->i * d->cap * K;
    const float *wt = (const float *)vkc_ptr(d->rwt), *x = (const float *)vkc_ptr(d->hxx);
    float *y = (float *)vkc_ptr(d->cy);
    memset(y, 0, (size_t)n * D * sizeof(float));
    int ps[VKC_DMOE_ROWS * VKC_ROUTE_KMAX], pk[VKC_DMOE_ROWS * VKC_ROUTE_KMAX], np = 0;
    const VktHostExpert *hx = d->hx + (size_t)j->i * E;
    for (int s = 0; s < n; s++)
        for (int k = 0; k < K; k++) { int e = ids[s * K + k]; if (e >= 0 && e < E && hx[e].g) { ps[np] = s; pk[np] = k; np++; } }
    if (!np) return;
    VkcHyb *w = &g_vkc_hyb;
    w->pairs += (unsigned long long)np;
    if (w->cap < np) {
        free(w->gu); free(w->hh); free(w->dn);
        w->gu = malloc((size_t)np * 2 * I * sizeof(float)); w->hh = malloc((size_t)np * I * sizeof(float)); w->dn = malloc((size_t)np * D * sizeof(float));
        w->cap = w->gu && w->hh && w->dn ? np : 0;
        if (!w->cap) { w->err = 1; return; }
    }
    float *gu = w->gu, *hh = w->hh, *dn = w->dn;
    gq_act ax[VKC_DMOE_ROWS], ah[VKC_DMOE_ROWS * VKC_ROUTE_KMAX];
    memset(ax, 0, sizeof ax); memset(ah, 0, sizeof(gq_act) * (size_t)np);
    for (int s = 0; s < n; s++) if (gq_act_set(&ax[s], x + (size_t)s * D, 1, D)) w->err = 1;
    /* gate and up of every pair, the activation, then down: one parallel region, the
     * rows handed out in chunks (the cores of a hybrid CPU do not run at one speed) */
    int nt = vkc_hyb_threads();
    #pragma omp parallel num_threads(nt)
    {
        float *tmp = gq_exact_mode() ? malloc((size_t)(D > I ? D : I) * sizeof(float)) : NULL;
        #pragma omp for schedule(dynamic, 32)
        for (int r = 0; r < np * 2 * I; r++) {
            int p = r / (2 * I), o = r % (2 * I); const VktHostExpert *h = &hx[ids[ps[p] * K + pk[p]]];
            const uint8_t *row = o < I ? h->g + (size_t)o * h->rgu : h->u + (size_t)(o - I) * h->rgu;
            vkc_hyb_row(gu + (size_t)p * 2 * I + o, &ax[ps[p]], x + (size_t)ps[p] * D, row, h->tg, D, tmp);
        }
        #pragma omp for schedule(static)
        for (int p = 0; p < np; p++) {
            const float *g = gu + (size_t)p * 2 * I, *u = g + I; float *h = hh + (size_t)p * I;
            for (int o = 0; o < I; o++) {
                float gv = g[o], uv = u[o];
                if (j->limit > 0.f) { gv = gv < j->limit ? gv : j->limit; uv = uv < -j->limit ? -j->limit : uv > j->limit ? j->limit : uv; }
                h[o] = gv / (1.f + expf(-gv)) * uv;
            }
            if (gq_act_set(&ah[p], h, 1, I)) w->err = 1;
        }
        #pragma omp for schedule(dynamic, 64)
        for (int r = 0; r < np * D; r++) {
            int p = r / D, o = r % D; const VktHostExpert *h = &hx[ids[ps[p] * K + pk[p]]];
            vkc_hyb_row(dn + (size_t)p * D + o, &ah[p], hh + (size_t)p * I, h->d + (size_t)o * h->rd, h->td, I, tmp);
        }
        free(tmp);
    }
    for (int p = 0; p < np; p++) {   /* in rank order within a row */
        float wv = wt[ps[p] * K + pk[p]]; float *ys = y + (size_t)ps[p] * D; const float *o = dn + (size_t)p * D;
        for (int c = 0; c < D; c++) ys[c] += wv * o[c];
    }
    for (int s = 0; s < n; s++) gq_act_free(&ax[s]);
    for (int p = 0; p < np; p++) gq_act_free(&ah[p]);
}
static void *vkc_hyb_main(void *arg) {
    (void)arg;
    VkcHyb *w = &g_vkc_hyb;
    for (;;) {
        pthread_mutex_lock(&w->mx);
        while (w->qh == w->qt && !w->quit) { w->busy = 0; pthread_cond_broadcast(&w->cv_idle); pthread_cond_wait(&w->cv, &w->mx); }
        if (w->qh == w->qt && w->quit) { w->busy = 0; pthread_cond_broadcast(&w->cv_idle); pthread_mutex_unlock(&w->mx); break; }
        VkcHybJob j = w->q[w->qh % VKC_HYB_Q]; w->busy = 1;
        pthread_mutex_unlock(&w->mx);
        if (!vkc_hybrid_host_wait(j.g)) w->err = 1;
        else {
            double t0 = vkc_hyb_now_ms();
            vkc_hyb_compute(j.d, &j);
            w->cpu_ms += vkc_hyb_now_ms() - t0;
        }
        vkc_hybrid_host_signal(j.g);   /* always: the device's next frame waits for it */
        pthread_mutex_lock(&w->mx);
        w->qh++; w->steps++;
        pthread_mutex_unlock(&w->mx);
    }
    return NULL;
}
static int vkc_hyb_start(void) {
    VkcHyb *w = &g_vkc_hyb;
    if (w->th_on) return 1;
    pthread_mutex_init(&w->mx, NULL); pthread_cond_init(&w->cv, NULL); pthread_cond_init(&w->cv_idle, NULL);
    if (pthread_create(&w->th, NULL, vkc_hyb_main, NULL)) return 0;
    w->th_on = 1;
    return 1;
}
/* every step posted so far done (the table is about to change, or the end) */
static void vkc_hyb_drain(void) {
    VkcHyb *w = &g_vkc_hyb;
    if (!w->th_on) return;
    pthread_mutex_lock(&w->mx);
    while (w->qh != w->qt || w->busy) pthread_cond_wait(&w->cv_idle, &w->mx);
    pthread_mutex_unlock(&w->mx);
}
static int vkc_hyb_post(const VkcHybJob *j) {
    VkcHyb *w = &g_vkc_hyb;
    pthread_mutex_lock(&w->mx);
    while (w->qt - w->qh >= VKC_HYB_Q) { pthread_mutex_unlock(&w->mx); vkc_hyb_drain(); pthread_mutex_lock(&w->mx); }
    w->q[w->qt % VKC_HYB_Q] = *j; w->qt++; w->busy = 1;
    pthread_cond_signal(&w->cv);
    pthread_mutex_unlock(&w->mx);
    return 1;
}

/* The tier's table for layers lo..lo+L-1, read again when its residents changed since. */
static int vkc_dmoe_read(VkcDmoe *d) {
    if (d->read && d->epoch == vkt_epoch()) return 1;
    int L = d->L, E = d->E;
    if (!d->on) d->on = calloc((size_t)L, 1);
    if (!d->tabh) d->tabh = malloc((size_t)L * E * COLI_VK_XB_ENTRY * sizeof(uint32_t));
    if (!d->on || !d->tabh) { d->off = 1; return 0; }
    vkc_hyb_drain();   /* the worker reads the host pool's rows of the table it had */
    for (int i = 0; i < L; i++) {
        int n = vkt_dev_table(d->lo + i, d->tabh + (size_t)i * E * COLI_VK_XB_ENTRY);
        if (n < 0) { d->off = 1; return 0; }
        d->on[i] = n == E;
    }
    if (d->hybrid) {
        if (!d->tabhh) d->tabhh = malloc((size_t)L * E * COLI_VK_XB_ENTRY * sizeof(uint32_t));
        if (!d->hx) d->hx = calloc((size_t)L * E, sizeof *d->hx);
        if (!d->hx_n) d->hx_n = calloc((size_t)L, sizeof *d->hx_n);
        if (!d->tabhh || !d->hx || !d->hx_n) d->hybrid = 0;
        for (int i = 0; i < L && d->hybrid; i++) {
            if (vkt_dev_table_hybrid(d->lo + i, d->tabhh + (size_t)i * E * COLI_VK_XB_ENTRY, d->hx + (size_t)i * E) < 0) { d->hybrid = 0; break; }
            d->hx_n[i] = 0;
            for (int e = 0; e < E; e++) d->hx_n[i] += d->hx[(size_t)i * E + e].g != NULL;
        }
    }
    d->epoch = vkt_epoch(); d->read = 1; d->dirty = 1;
    return 1;
}
/* layer i's gate/up and down formats, from its first entry (a layer's experts share them) */
/* layer i's format word w (12 gate/up, 15 down): the tier's form for the layer, the same in
 * every resident entry, read from the first resident one (0 when none is) */
static int vkc_dmoe_fmt(const VkcDmoe *d, int i, int w) {
    const uint32_t *t = d->tabh + (size_t)i * d->E * COLI_VK_XB_ENTRY;
    for (int e = 0; e < d->E; e++, t += COLI_VK_XB_ENTRY) if (t[18]) return (int)t[w];
    return 0;
}
/* For a step of `rows` rows on the primary device: whether any of layers lo..lo+L-1 takes
 * it; reads the tier's table again when its residents changed. 0 for good when the
 * device cannot read the tier's experts by address. */
static int vkc_dmoe_prepare(VkcDmoe *d, int lo, int L, int E, int K, int D, int I, int rows) {
    int big = rows > VKC_DMOE_ROWS;
    if (!vkc_dmoe_wanted() || d->off || (big && !d->big) || L < 1 || vkc_device_now() != 0 || !vkt_ready() ||
        !vkc_moe_ready() || !vkc_route_ready() || K < 1 || K > VKC_ROUTE_KMAX || E < K || E > VKC_ROUTE_EMAX) return 0;
    if (d->on && (d->lo != lo || d->L != L || d->E != E || d->K != K)) { d->off = 1; return 0; }
    d->lo = lo; d->L = L; d->E = E; d->K = K;
    if (!vkc_dmoe_read(d)) return 0;
    int any = 0;
    for (int i = 0; i < L; i++) any |= d->on[i];
    if (!any) return 0;
    /* a big step: the layers whose formats have the grouped GEMMs run here, the others
     * (a GGUF's IQ1_M layers, say) take the host's step */
    if (big && (D % 64 || I % 64)) return 0;
    if (big) {
        if (!d->rows_off && !(d->rows_off = calloc((size_t)L, 1))) return 0;
        int some = 0;
        for (int i = 0; i < L; i++) {
            d->rows_off[i] = d->on[i] && !vkc_moe_rows_ready(vkc_dmoe_fmt(d, i, 12), vkc_dmoe_fmt(d, i, 15));
            some |= d->on[i] && !d->rows_off[i];
        }
        if (!some) return 0;
    }
    d->big_step = big;
    d->D = D; d->I = I;
    size_t r = VKC_DMOE_ROWS, rc = big ? (size_t)rows : r;
    /* the hybrid steps' table and rows (the weights where the worker reads them) */
    if (!d->tab && !d->hybrid && vkc_hybrid_wanted()) {
        d->hybrid = 1;
        if (!vkc_hyb_start() || !(d->tabhyb = vkc_buf((size_t)L * E * COLI_VK_XB_ENTRY * sizeof(uint32_t), VKC_DEV)) ||
            !vkc_reserve(&d->hxx, r * (size_t)D * sizeof(float), VKC_DOWN) || !vkc_reserve(&d->cy, r * (size_t)D * sizeof(float), VKC_UP))
            d->hybrid = 0;
        else { d->read = 0; if (!vkc_dmoe_read(d)) return 0; }
    }
    if ((!d->tab && !(d->tab = vkc_buf((size_t)L * E * COLI_VK_XB_ENTRY * sizeof(uint32_t), VKC_DEV))) ||
        !vkc_reserve(&d->rid, (size_t)L * rc * K * sizeof(int), VKC_DOWN) || !vkc_reserve(&d->rwt, rc * K * sizeof(float), d->hybrid ? VKC_DOWN : VKC_DEV) ||
        !vkc_reserve(&d->hid, r * K * ((size_t)I + D) * sizeof(float), VKC_DEV) || !vkc_reserve(&d->y, rc * (size_t)D * sizeof(float), VKC_DEV)) {
        d->off = 1;
        return 0;
    }
    d->cap = (int)rc;
    return 1;
}
/* The table into the device, in a frame just begun: read again first when the tier's
 * residents changed since (a host layer before it ran the tier), written when it changed.
 * 0 only when the write fails; a table that cannot be read turns the layers down. */
static int vkc_dmoe_upload(VkcDmoe *d) {
    if (d->off || !d->tab) return 1;
    if (!vkc_dmoe_read(d)) return 1;   /* d->off: every layer takes the host's step from here */
    if (!d->dirty) return 1;
    if (!vkc_write(d->tab, 0, d->tabh, (size_t)d->L * d->E * COLI_VK_XB_ENTRY * sizeof(uint32_t))) return 0;
    if (d->hybrid && !vkc_write(d->tabhyb, 0, d->tabhh, (size_t)d->L * d->E * COLI_VK_XB_ENTRY * sizeof(uint32_t))) return 0;
    d->dirty = 0;
    return 1;
}
/* Whether layer `layer` runs its experts in the frame: all of them on the device, in the
 * table the frame has (the tier's residents unchanged since it was read). */
static int vkc_dmoe_on(const VkcDmoe *d, int layer) {
    return d->on && !d->off && d->epoch == vkt_epoch() && layer >= d->lo && layer < d->lo + d->L && d->on[layer - d->lo] &&
           !(d->big_step && d->rows_off && d->rows_off[layer - d->lo]);
}
/* Layer `layer` for n rows: route the logits lg (n rows of E from 0, bias as rt's
 * has_bias says) and run the experts on x (n rows of D from 0) into d->y. rt gives the
 * router's kind, bias, groups, norm, temp, scale and eps; mp the activation (act, limit,
 * a, b), D and I; the rest is filled here. */
static int vkc_dmoe_layer(VkcDmoe *d, int layer, int n, VkcBuf *lg, VkcBuf *bias, VkcBuf *x,
                          const VkcRoute *rt, const VkcMoe *mp) {
    int i = layer - d->lo, io = i * d->cap * d->K;
    VkcRoute r = *rt;
    r.S = n; r.E = d->E; r.K = d->K; r.l_off = 0; r.l_row = d->E; r.i_off = io; r.i_row = d->K; r.w_off = 0; r.w_row = d->K;
    VkcMoe m = *mp;
    m.S = n; m.K = d->K; m.t_off = i * d->E; m.x_off = 0; m.x_row = mp->D; m.i_off = io; m.i_row = d->K;
    m.w_off = 0; m.w_row = d->K; m.y_off = 0; m.y_row = mp->D;
    if (!m.gfmt && d->tabh) m.gfmt = vkc_dmoe_fmt(d, i, 12);   /* the layer's gate/up format */
    if (!m.dfmt && d->tabh) m.dfmt = vkc_dmoe_fmt(d, i, 15);   /* and down's (the integer form wants both) */
    if (n > VKC_DMOE_ROWS) {   /* a prompt chunk: the grouped GEMMs (vkc_dmoe_prepare checked the layer's formats) */
        m.E = d->E;
        /* With a host pool the GEMMs read some experts over PCIe, and the floats' were faster
         * there: Qwen3.8 Next IQ2_XS on an RTX PRO 4000 (19 GiB of experts in the host pool), a
         * 507-token prompt, 115-116 tok/s against 93-94 on the matrix units; without the pool the
         * two were even (104-114 against 109-120). */
        m.floats = vkt_host_pool_on();
        if (!vkc_route(lg, bias, d->rid, d->rwt, &r) || !vkc_moe_rows(x, d->rid, d->rwt, d->tab, &d->rows, d->y, &m)) return 0;
        d->steps++; d->big_steps++;
        return 1;
    }
    if (d->hybrid && d->hx_n && d->hx_n[i] > 0 && m.act == 0) {
        /* frame A: the route and the input rows down, then the device's timeline; frame B:
         * the device's experts; frame C, after the host's: the CPU's sum added */
        uint64_t g = ++g_vkc_hyb_seq;
        if (!vkc_route(lg, bias, d->rid, d->rwt, &r) || !vkc_copy(d->hxx, 0, x, 0, (size_t)n * d->D)) return 0;
        VkcHybJob j = {d, i, n, g, m.act, m.limit};
        vkc_hyb_post(&j);
        if (!vkc_hybrid_submit(g) || !vkc_begin() || !vkc_moe(x, d->rid, d->rwt, d->tabhyb, d->hid, d->y, &m) || !vkc_submit(0)) return 0;
        vkc_hybrid_wait_next(g);
        VkcEw add = {VKC_EW_ADD, n * d->D, d->D, 1, 0, 0, 0, 0, 0, 0, 0, 1.f};
        if (!vkc_begin() || !vkc_ew(d->y, d->y, d->cy, NULL, NULL, &add)) return 0;
        d->steps++; d->hyb_steps++;
        return 1;
    }
    if (!vkc_route(lg, bias, d->rid, d->rwt, &r) || !vkc_moe(x, d->rid, d->rwt, d->tab, d->hid, d->y, &m)) return 0;
    d->steps++;
    return 1;
}
/* Layer `layer`'s ids (n rows of K, rank order, -1 for an empty slot) once its frame is done. */
static const int *vkc_dmoe_ids(const VkcDmoe *d, int layer) {
    return (const int *)vkc_ptr(d->rid) + (size_t)(layer - d->lo) * d->cap * d->K;
}
static void vkc_dmoe_free(VkcDmoe *d) {
    vkc_hyb_drain();   /* the worker reads its buffers */
    vkc_free(d->tab); vkc_free(d->rid); vkc_free(d->rwt); vkc_free(d->hid); vkc_free(d->y);
    vkc_free(d->tabhyb); vkc_free(d->hxx); vkc_free(d->cy);
    vkc_moe_rows_free(&d->rows);
    free(d->on); free(d->tabh); free(d->rows_off); free(d->tabhh); free(d->hx); free(d->hx_n);
    memset(d, 0, sizeof *d);
}
/* the hybrid steps' line: steps, pairs the CPU computed, its time */
static void vkc_dmoe_hyb_report(const char *engine, const VkcDmoe *d) {
    const VkcHyb *w = &g_vkc_hyb;
    if (!d->hyb_steps) return;
    fprintf(stderr, "[VK] %s chain: %llu hybrid layer steps, %llu host pool pairs on the CPU in %.1f ms (%.1f us a step)%s\n",
            engine, d->hyb_steps, w->pairs, w->cpu_ms, w->steps ? w->cpu_ms * 1e3 / (double)w->steps : 0.0,
            w->err ? ", errors seen" : "");
}
#endif
