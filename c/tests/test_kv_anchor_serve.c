/* test_kv_anchor_serve — the ENGINE side of #50403: the anchor ring as a serve
 * slot actually owns it.
 *
 * test_kv_anchor.c pins the decision rule against synthetic planes. This one
 * pins the seam between that rule and colibri.c's real KV: the per-ServeCtx
 * ring, the plane view of Lc/Rc/Ic (which is not uniform — Lc/Rc carry the
 * extra MTP row that Ic does not, and Ic has holes for the layers that reuse a
 * neighbour's DSA index), the env policy that keeps the feature off by default,
 * and the capture/restore round trip through a slot whose KV a later
 * conversation has overwritten. That last one IS the feature: the whole point
 * is that a slot can get a previous conversation's rows back byte-for-byte.
 *
 * Built like test_kvb_notice: include the engine, rename its main, drive the
 * real functions against a hand-built Model so no snapshot on disk is needed.
 */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

/* Same portable env seam the other engine tests use: MinGW's CRT has no
 * setenv, and a test that silently fails to set the variable under test would
 * report the DEFAULT behaviour as if it were the configured one. */
static void env_set(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
static void env_unset(const char *name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; } } while (0)

enum { NL = 4, MAXT = 32 };

/* A GLM-shaped Model with DSA on every other layer — the same shape kv_alloc
 * builds for the real engine, small enough to compare row by row. */
static void model_init_kv(Model *m, ServeCtx *sc, int dsa) {
    memset(m, 0, sizeof(*m));
    m->c.n_layers = NL;
    m->c.kv_lora  = 6;
    m->c.qk_rope  = 4;
    m->c.index_hd = 3;
    m->has_dsa    = dsa;
    m->has_mtp    = 1;
    for (int i = 0; i < NL; i++) m->c.idx_type[i] = (i % 2 == 0);   /* holes on odd layers */
    memset(sc, 0, sizeof(*sc));
    sc->kv.kv_start = calloc(NL + 1, sizeof(int));
    sc->kv.kv_start[NL] = -1;
    kv_bind(m, &sc->kv);
    kv_alloc(m, MAXT);
    sc->hist = calloc(MAXT, sizeof(int));
}
static void model_free_kv(Model *m, ServeCtx *sc) {
    KVState *k = &sc->kv;
    for (int i = 0; i < NL + 1; i++) { free(k->Lc[i]); free(k->Rc[i]); }
    if (k->Ic) { for (int i = 0; i < NL; i++) free(k->Ic[i]); free(k->Ic); }
    free(k->Lc); free(k->Rc); free(k->kv_start); free(sc->hist);
    kv_anchor_ring_free(&sc->anch);
    (void)m;
}

/* Paint every live KV row with a value derived from (plane, layer, position,
 * lane) and `seed`, so any row that comes back from the wrong place is visible. */
static void paint(Model *m, ServeCtx *sc, int len, float seed) {
    Cfg *c = &m->c;
    for (int l = 0; l <= c->n_layers; l++) {
        for (int p = 0; p < len; p++) {
            for (int i = 0; i < c->kv_lora; i++)
                coli_kv_row(sc->kv.Lc[l], p, c->kv_lora)[i] = seed + l * 1000 + p * 10 + i;
            for (int i = 0; i < c->qk_rope; i++)
                coli_kv_row(sc->kv.Rc[l], p, c->qk_rope)[i] = -(seed + l * 1000 + p * 10 + i);
        }
    }
    if (!sc->kv.Ic) return;
    for (int l = 0; l < c->n_layers; l++) {
        if (!sc->kv.Ic[l]) continue;
        for (int p = 0; p < len; p++)
            for (int i = 0; i < c->index_hd; i++)
                coli_kv_row(sc->kv.Ic[l], p, c->index_hd)[i] = seed * 2 + l * 1000 + p * 10 + i;
    }
}
/* Bytes of every live row in [from,to), so two states can be compared exactly. */
static size_t snapshot(Model *m, ServeCtx *sc, int from, int to, float *out) {
    Cfg *c = &m->c; size_t n = 0;
    for (int l = 0; l <= c->n_layers; l++) {
        for (int p = from; p < to; p++)
            for (int i = 0; i < c->kv_lora; i++)
                out[n++] = coli_kv_row(sc->kv.Lc[l], p, c->kv_lora)[i];
        for (int p = from; p < to; p++)
            for (int i = 0; i < c->qk_rope; i++)
                out[n++] = coli_kv_row(sc->kv.Rc[l], p, c->qk_rope)[i];
    }
    if (sc->kv.Ic) for (int l = 0; l < c->n_layers; l++) {
        if (!sc->kv.Ic[l]) continue;
        for (int p = from; p < to; p++)
            for (int i = 0; i < c->index_hd; i++)
                out[n++] = coli_kv_row(sc->kv.Ic[l], p, c->index_hd)[i];
    }
    return n;
}

int main(void) {
    Model m; ServeCtx sc;
    kv_anchor_plane pl[3];
    static float before[NL * 4 * MAXT * 16], after[NL * 4 * MAXT * 16];

    /* ---- the plane view of this engine's KV ---------------------------- */
    model_init_kv(&m, &sc, 1);
    int npl = kvanchor_planes(&m, &sc.kv, pl);
    CHECK(npl == 3, "a DSA model must expose three planes, got %d", npl);
    CHECK(pl[0].n == NL + 1 && pl[0].width == m.c.kv_lora,
          "Lc plane must carry the extra MTP row at kv_lora width (n=%d w=%d)",
          pl[0].n, pl[0].width);
    CHECK(pl[1].n == NL + 1 && pl[1].width == m.c.qk_rope,
          "Rc plane must carry the extra MTP row at qk_rope width");
    /* Ic is per-attention-layer ONLY. Handing it n_layers+1 would read one
     * pointer past the array kv_alloc sized — the shape of the over-read this
     * ticket found next door in the cross-slot adopt loop. */
    CHECK(pl[2].n == NL && pl[2].width == m.c.index_hd,
          "Ic plane must be n_layers wide (got %d), never n_layers+1", pl[2].n);
    CHECK(pl[2].plane[1] == NULL, "a layer without its own index must stay a hole");
    model_free_kv(&m, &sc);

    model_init_kv(&m, &sc, 0);
    CHECK(kvanchor_planes(&m, &sc.kv, pl) == 2,
          "a model without DSA must expose two planes, not a NULL third");
    model_free_kv(&m, &sc);

    /* ---- the policy: OFF unless asked ---------------------------------- */
    g_kvanchor = -1; env_unset("COLI_KV_ANCHOR"); env_unset("COLI_KV_ANCHOR_MB");
    CHECK(kvanchor_slots() == 0, "anchors must be off unless COLI_KV_ANCHOR is set");
    g_kvanchor = -1; env_set("COLI_KV_ANCHOR", "3"); env_set("COLI_KV_ANCHOR_MB", "7");
    CHECK(kvanchor_slots() == 3, "COLI_KV_ANCHOR must arm the ring");
    CHECK(g_kvanchor_budget == 7u * 1024u * 1024u, "COLI_KV_ANCHOR_MB must set the cap");
    g_kvanchor = -1; env_set("COLI_KV_ANCHOR", "-4");
    CHECK(kvanchor_slots() == 0, "a negative slot count must disarm, not underflow");
    g_kvanchor = -1; env_set("COLI_KV_ANCHOR", "9999");
    CHECK(kvanchor_slots() == 64, "the slot count must be clamped");

    /* The tool-call opener comes from the tokenizer, but an explicit override
     * must not touch it at all — the engine resolves this before any Tok is
     * usable in some paths. */
    g_kvanchor_tok = -2; env_set("COLI_KV_ANCHOR_TOK", "4242");
    kvanchor_arm_tok(NULL);
    CHECK(g_kvanchor_tok == 4242, "COLI_KV_ANCHOR_TOK must win without consulting the tokenizer");
    g_kvanchor_tok = -2; env_set("COLI_KV_ANCHOR_TOK", "-1");
    kvanchor_arm_tok(NULL);
    CHECK(g_kvanchor_tok == -1, "COLI_KV_ANCHOR_TOK=-1 must disable the tool-call capture");
    env_unset("COLI_KV_ANCHOR_TOK");

    /* ---- capture, clobber, restore: the feature itself ------------------ */
    g_kvanchor = -1; env_set("COLI_KV_ANCHOR", "4"); env_unset("COLI_KV_ANCHOR_MB");
    model_init_kv(&m, &sc, 1);
    int slots = kvanchor_slots();      /* sets g_kvanchor_budget; read it after */
    CHECK(kv_anchor_ring_init(&sc.anch, slots, g_kvanchor_budget) == 1,
          "ring init failed");

    /* Conversation A: eight tokens, its own KV. */
    const int convA[] = {10, 11, 12, 13, 14, 15, 16, 17};
    memcpy(sc.hist, convA, sizeof(convA)); sc.len = 8;
    paint(&m, &sc, 8, 1.0f);
    size_t n_tail = snapshot(&m, &sc, 1, 8, before);    /* the rows a restore must reinstate */
    kvanchor_capture(&m, &sc, 8, "prompt_end");
    CHECK(sc.anch.stores == 1, "capture must land in the ring");

    /* Conversation B lands on the same slot and overwrites A's rows — exactly
     * what KV_SLOTS=1 does to two interleaved agent sessions. */
    const int convB[] = {10, 90, 91, 92};
    memcpy(sc.hist, convB, sizeof(convB)); sc.len = 4;
    paint(&m, &sc, 8, 500.0f);
    CHECK(snapshot(&m, &sc, 1, 8, after) == n_tail, "snapshot size must be stable");
    CHECK(memcmp(before, after, n_tail * sizeof(float)) != 0,
          "conversation B must actually have clobbered A's rows");
    static float p0_b[NL * 4 * MAXT * 16];
    size_t n_p0 = snapshot(&m, &sc, 0, 1, p0_b);   /* B's row 0, the one already agreed */

    /* A comes back with one more token. The live state agrees for 1 position;
     * the ring remembers 8. */
    const int convA2[] = {10, 11, 12, 13, 14, 15, 16, 17, 18};
    int have = 0;
    while (have < sc.len && have < 9 && sc.hist[have] == convA2[have]) have++;
    CHECK(have == 1, "the live strict prefix must be 1, got %d", have);
    int ai = kv_anchor_match(&sc.anch, convA2, 9, have);
    CHECK(ai >= 0, "the ring must offer A's remembered state");
    npl = kvanchor_planes(&m, &sc.kv, pl);
    int alen = kv_anchor_restore(&sc.anch, ai, pl, npl, have);
    CHECK(alen == 8, "restore must reinstate all 8 positions, got %d", alen);
    memcpy(sc.hist + have, kv_anchor_ids(&sc.anch, ai) + have,
           (size_t)(alen - have) * sizeof(int));
    sc.len = alen;

    /* THE claim of this feature, stated as bytes: what the slot holds now is
     * what the engine had built for A, not an approximation of it. */
    CHECK(snapshot(&m, &sc, 1, 8, after) == n_tail, "snapshot size must be stable");
    CHECK(memcmp(before, after, n_tail * sizeof(float)) == 0,
          "restored rows must be byte-identical to the rows originally built");
    CHECK(memcmp(sc.hist, convA2, 8 * sizeof(int)) == 0,
          "restore must also reinstate the token ids those rows belong to");

    /* Position 0 was already agreed and must NOT have been rewritten: the
     * restore copies [have,len) exactly like the cross-slot adopt it mirrors. */
    {
        static float p0now[NL * 4 * MAXT * 16];
        CHECK(snapshot(&m, &sc, 0, 1, p0now) == n_p0, "snapshot size must be stable");
        CHECK(memcmp(p0_b, p0now, n_p0 * sizeof(float)) == 0,
              "restore must leave rows below `from` byte-for-byte alone");
    }

    /* The mid-history insertion, at engine level: A's remembered state must NOT
     * be offered to a prompt that splices tokens in after position 2. */
    const int spliced[] = {10, 11, 77, 78, 12, 13, 14, 15, 16, 17, 19};
    CHECK(kv_anchor_match(&sc.anch, spliced, 11, 0) == -1,
          "an anchor must not be offered past a mid-history insertion");

    /* A disarmed ring on a real slot must leave the KV untouched. */
    kv_anchor_ring_free(&sc.anch);
    CHECK(sc.anch.n == 0, "ring must be disarmed after free");
    kvanchor_capture(&m, &sc, 8, "prompt_end");        /* must be a no-op, not a crash */
    CHECK(sc.anch.stores == 0, "a disarmed ring must not capture");
    CHECK(kv_anchor_match(&sc.anch, convA2, 9, 0) == -1, "a disarmed ring must not match");

    model_free_kv(&m, &sc);
    env_unset("COLI_KV_ANCHOR");

    if (failures) { fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    puts("kv_anchor_serve: ok");
    return 0;
}
