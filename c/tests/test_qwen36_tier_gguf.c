/* The CUDA tier with a GGUF's experts (qt_init_gguf), on the fake backend: the file's rows go
 * up as the backend's block formats, gate/up and down of each layer's own types, and a swap
 * only ever pairs layers of the same types. A swap into another layer's types would free and
 * allocate device memory under a dense chain's hybrid layer, which deadlocked it (the device
 * waiting for the host's word, the host's next launch behind the free). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../compat.h"
#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}
static void drain(int want) {
    for (int i = 0; i < 1000 && fake_uploads < want; i++) {
        struct timespec ts = {0, 2000000};
        nanosleep(&ts, NULL);
    }
}

int main(void) {
    enum { NL = 3, NE = 4, D = 256, IH = 256, TOPK = 2 };
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    /* layers 0 and 1: IQ2_XXS gate/up, IQ2_S down; layer 2 a bigger down (IQ3_S), as published files mix them */
    const int tg[NL] = {16, 16, 16}, td[NL] = {22, 22, 21};
    const size_t gb[NL] = {(size_t)IH * 66, (size_t)IH * 66, (size_t)IH * 66}, db[NL] = {(size_t)D * 82, (size_t)D * 82, (size_t)D * 110};

    fake_block_elems = 0;
    check(!qt_init_gguf(NL, NE, D, IH, NE, TOPK, tg, td, gb, db), "a backend that takes no block format refuses a GGUF's experts");
    fake_block_elems = 256;
    if (!qt_init_gguf(NL, NE, D, IH, NE, TOPK, tg, td, gb, db)) { printf("  FAIL: the tier refused a GGUF's experts\n"); return 1; }
    check(G.wfmt == QT_GGUF, "a GGUF's tier holds its block formats");
    size_t biggest = 2 * dev_alloc_footprint((gb[2] + 3 * (size_t)IH) / 4 * 4) + dev_alloc_footprint((db[2] + 3 * (size_t)D) / 4 * 4);
    check(G.exp_bytes == biggest, "an expert is charged as the biggest layer's");
    int f, fd;
    check(qt_chain_format(&f, NULL) && f == QT_GGUF, "the chain sees a GGUF's formats");
    check(qt_chain_layer_format(0, &f, &fd) && f == 116 && fd == 122, "layer 0: IQ2_XXS gate/up, IQ2_S down");
    check(qt_chain_layer_format(2, &f, &fd) && f == 116 && fd == 121, "layer 2: IQ3_S down");
    check(swap_ok(0, 1) && swap_ok(1, 0) && !swap_ok(0, 2) && !swap_ok(2, 1), "swaps pair only same-typed layers");

    /* an expert of layer 2 goes up as the file's rows, its own types */
    unsigned char *g = malloc(gb[2]), *u = malloc(gb[2]), *d = malloc(db[2]);
    memset(g, 0x31, gb[2]); memset(u, 0x32, gb[2]); memset(d, 0x33, db[2]);
    int layers[NL * NE], eids[NL * NE];
    int planned = qt_plan_fill(layers, eids, NL * NE);
    check(planned == NL * NE, "every expert fits the fake device");
    int k2 = -1;
    for (int i = 0; i < planned; i++) if (layers[i] == 2) { k2 = i; break; }
    check(k2 >= 0, "an expert of layer 2 is planned");
    if (k2 >= 0) {
        int before = fake_uploads;
        qt_note_planned(layers[k2], eids[k2], g, u, d, NULL, NULL, NULL);
        drain(before + 3);
        check(fake_uploads == before + 3, "three matrices went up");
        check(last_fmt == 121, "down went up as IQ3_S (fmt 121)");
        check(captured_len == 0 || captured[0] != 0x00, "the bytes arrived");
    }

    /* the re-plan: candidates in layer 0, residents in layers 1 and 2, layer 2's the coldest -- the
     * victims still come from layer 1 only */
    pthread_mutex_lock(&G.mx);
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) {
        QSlot *s = qs(l, e);
        s->queued = 0; s->planned = 0;
        s->g4 = g; s->u4 = u; s->d4 = d;
        res_set(s, l > 0);
        if (l > 0 && !s->tg) { coli_cuda_tensor_upload(&s->tg, g, NULL, 116, D, IH, 0); coli_cuda_tensor_upload(&s->tu, u, NULL, 116, D, IH, 0);
                               coli_cuda_tensor_upload(&s->td, d, NULL, 100 + td[l], IH, D, 0); }
    }
    pthread_mutex_unlock(&G.mx);
    uint32_t *counts = calloc(NL * NE, sizeof *counts);
    for (int e = 0; e < NE; e++) { counts[e] = 100 + e; counts[NE + e] = 5; }   /* layer 0's experts hot, layer 2's residents the coldest */
    int np = qt_replan(-1, counts, 8);
    check(np == NE, "each of layer 0's experts found a same-typed victim");
    qt_fill_wait();
    int vl[16], ve[16], nv = qt_evicted_take(vl, ve, 16), only1 = nv > 0;
    for (int i = 0; i < nv; i++) only1 &= vl[i] == 1;
    check(only1, "the victims are layer 1's (IQ2_S down), never layer 2's (IQ3_S)");

    qt_shutdown();
    free(g); free(u); free(d); free(counts);
    printf(fails ? "qwen36 tier gguf: %d FAIL\n" : "qwen36 tier gguf: ok\n", fails);
    return fails != 0;
}
