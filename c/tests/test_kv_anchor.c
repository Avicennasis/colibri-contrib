/* test_kv_anchor — the multi-state reuse decision, exhaustively.
 *
 * Same stakes as test_kv_prefix: a wrong reuse length does not crash, it
 * answers the user from an attention state that belongs to a different
 * conversation, and the reply looks plausible. kv_anchor widens the candidate
 * SET (every remembered state, not just the live one) without weakening the
 * RULE (strict prefix), so every rejection rule gets its own case here — and
 * so does the boundary the whole feature is often misread as crossing: an
 * anchor must NEVER resume past a mid-history insertion.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../kv_anchor.h"

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; } } while (0)

/* Two synthetic KV planes: 2 "layers" of width 3, 1 "layer" of width 2, with a
 * NULL hole standing in for the layers GLM's DSA index does not own. */
enum { NL = 3, W0 = 3, W1 = 2, POS = 12 };
static float p0buf[NL][POS * W0], p1buf[NL][POS * W1];
static float *p0[NL], *p1[NL];

static void planes_reset(void) {
    for (int l = 0; l < NL; l++) { p0[l] = p0buf[l]; p1[l] = p1buf[l]; }
    p1[1] = NULL;                              /* the hole */
    memset(p0buf, 0, sizeof(p0buf));
    memset(p1buf, 0, sizeof(p1buf));
}
static void planes_fill(float seed) {
    for (int l = 0; l < NL; l++) {
        for (int i = 0; i < POS * W0; i++) p0buf[l][i] = seed + l * 100.0f + i;
        if (p1[l]) for (int i = 0; i < POS * W1; i++) p1buf[l][i] = -seed - l * 100.0f - i;
    }
}
static int planes_equal_range(const float *snap0, const float *snap1, int from, int to) {
    for (int l = 0; l < NL; l++) {
        for (int i = from * W0; i < to * W0; i++)
            if (p0buf[l][i] != snap0[l * POS * W0 + i]) return 0;
        if (!p1[l]) continue;
        for (int i = from * W1; i < to * W1; i++)
            if (p1buf[l][i] != snap1[l * POS * W1 + i]) return 0;
    }
    return 1;
}
static void planes_snapshot(float *snap0, float *snap1) {
    memcpy(snap0, p0buf, sizeof(p0buf));
    memcpy(snap1, p1buf, sizeof(p1buf));
}
static void mkplanes(kv_anchor_plane *pl) {
    pl[0] = (kv_anchor_plane){p0, NL, W0};
    pl[1] = (kv_anchor_plane){p1, NL, W1};
}

int main(void) {
    kv_anchor_ring r;
    kv_anchor_plane pl[2];
    static float snap0[NL * POS * W0], snap1[NL * POS * W1];

    /* A disarmed ring is inert on every entry point — this is the DEFAULT
     * build state (COLI_KV_ANCHOR unset), so "inert" has to mean it. */
    memset(&r, 0, sizeof(r));
    planes_reset(); mkplanes(pl);
    const int base[] = {1, 2, 3, 4, 5};
    CHECK(kv_anchor_ring_init(&r, 0, 0) == 0, "n=0 must leave the ring disarmed");
    CHECK(r.n == 0, "disarmed ring must report n=0");
    CHECK(kv_anchor_store(&r, base, 3, pl, 2) == 0, "disarmed ring stores nothing");
    CHECK(kv_anchor_match(&r, base, 5, 0) == -1, "disarmed ring matches nothing");
    kv_anchor_ring_free(&r);

    /* ---- the reuse decision ------------------------------------------- */
    CHECK(kv_anchor_ring_init(&r, 4, 0) == 1, "ring init failed");
    planes_reset(); mkplanes(pl); planes_fill(1.0f);
    CHECK(kv_anchor_store(&r, base, 3, pl, 2) == 1, "store failed");

    const int extends[]  = {1, 2, 3, 9, 9};        /* base is a strict prefix */
    const int diverges[] = {1, 2, 7, 4, 5};        /* diverges inside base    */
    const int equal[]    = {1, 2, 3};              /* nothing left to prefill */
    const int shorter[]  = {1, 2};

    CHECK(kv_anchor_match(&r, extends, 5, 0) >= 0, "an extending prompt must match");
    CHECK(kv_anchor_match(&r, diverges, 5, 0) == -1, "a diverging prompt must not match");
    CHECK(kv_anchor_match(&r, equal, 3, 0) == -1,
          "an equal-length prompt must not match: no token would be left to prefill");
    CHECK(kv_anchor_match(&r, shorter, 2, 0) == -1,
          "a shorter prompt must not match: this cannot rewind");
    CHECK(kv_anchor_match(&r, extends, 5, 3) == -1,
          "an anchor no longer than the live state must not be installed");
    CHECK(kv_anchor_match(&r, extends, 5, 2) >= 0,
          "an anchor longer than the live state must be installed");

    /* THE BOUNDARY THIS FEATURE IS MISREAD AS CROSSING. `edit` inserts two
     * tokens after position 2 — the tool-result splice #50403 is about. The
     * tail (3,4,5) is present, at shifted positions, and an anchor MUST NOT
     * offer it: those rows never attended to the inserted tokens, and their
     * RoPE'd keys belong to different absolute positions. Reuse resumes only
     * where the ids agree from position 0. */
    const int edit[] = {1, 2, 8, 8, 3, 4, 5};
    CHECK(kv_anchor_match(&r, edit, 7, 0) == -1,
          "an anchor must not resume past a mid-history insertion");
    /* ...and the untouched HEAD of the same edit still matches when it is long
     * enough to be worth installing, which is all an anchor ever promises. */
    const int head_only[] = {1, 2, 3, 8, 8, 4, 5};
    CHECK(kv_anchor_match(&r, head_only, 7, 0) >= 0,
          "the pre-edit head is still a strict prefix and must match");

    /* ---- restore is a byte-exact reinstatement ------------------------- */
    planes_snapshot(snap0, snap1);                 /* what the anchor holds */
    planes_fill(50.0f);                            /* a different conversation ran */
    CHECK(!planes_equal_range(snap0, snap1, 0, 3), "the overwrite must have changed rows");
    int idx = kv_anchor_match(&r, extends, 5, 1);
    CHECK(idx >= 0, "match before restore");
    CHECK(kv_anchor_restore(&r, idx, pl, 2, 1) == 3, "restore must report the anchor length");
    CHECK(planes_equal_range(snap0, snap1, 1, 3),
          "restored rows must be byte-identical to the captured ones");
    /* Position 0 was NOT restored: the caller said it already held it. Rows
     * outside [from,len) must be left exactly as the caller had them. */
    CHECK(!planes_equal_range(snap0, snap1, 0, 1),
          "restore must not touch rows below `from`");
    CHECK(!planes_equal_range(snap0, snap1, 3, POS),
          "restore must not touch rows past the anchor length");

    /* ---- geometry validation ------------------------------------------- */
    kv_anchor_plane bad[2];
    bad[0] = (kv_anchor_plane){p0, NL, W0 + 1};    /* wrong width */
    bad[1] = (kv_anchor_plane){p1, NL, W1};
    CHECK(kv_anchor_restore(&r, idx, bad, 2, 1) == 0,
          "a width mismatch must refuse the restore, not write garbage rows");
    bad[0] = (kv_anchor_plane){p0, NL - 1, W0};    /* wrong layer count */
    bad[1] = (kv_anchor_plane){p1, NL, W1};
    CHECK(kv_anchor_restore(&r, idx, bad, 2, 1) == 0,
          "a layer-count mismatch must refuse the restore");
    CHECK(kv_anchor_restore(&r, idx, pl, 2, 3) == 0,
          "from == len leaves nothing to restore");
    CHECK(kv_anchor_restore(&r, idx, pl, 2, 99) == 0, "from past len is refused");
    CHECK(kv_anchor_restore(&r, 99, pl, 2, 0) == 0, "an out-of-range index is refused");

    /* The NULL hole must stay a hole: a plane the engine does not own must
     * neither be captured nor written. Widths recorded for a NULL layer are 0,
     * which is what makes the validation above see it as a hole and not a
     * mismatch. */
    CHECK(p1[1] == NULL, "test fixture lost its hole");

    /* ---- ring behaviour ------------------------------------------------ */
    /* Re-storing a state the ring already holds must NOT consume a slot: an
     * agent resends the same prefix every turn, and duplicating it would evict
     * the branches the ring exists to keep. */
    size_t before = r.bytes;
    uint64_t stores = r.stores;
    CHECK(kv_anchor_store(&r, base, 3, pl, 2) == 1, "re-store failed");
    CHECK(r.bytes == before, "re-storing a held state must not grow the ring");
    CHECK(r.stores == stores, "re-storing a held state must not count as a store");

    /* LRU: fill the ring, touch the oldest, then overflow it — the touched one
     * must survive and the untouched one must go. */
    kv_anchor_ring_free(&r);
    CHECK(kv_anchor_ring_init(&r, 2, 0) == 1, "ring init failed");
    const int a1[] = {1}, a2[] = {1, 2}, a3[] = {1, 2, 3};
    const int probe1[] = {1, 7}, probe3[] = {1, 2, 3, 7};
    CHECK(kv_anchor_store(&r, a1, 1, pl, 2) == 1, "store a1");
    CHECK(kv_anchor_store(&r, a2, 2, pl, 2) == 1, "store a2");
    CHECK(kv_anchor_restore(&r, kv_anchor_match(&r, probe1, 2, 0), pl, 2, 0) == 1,
          "touching a1 must succeed");                      /* a1 becomes MRU */
    CHECK(kv_anchor_store(&r, a3, 3, pl, 2) == 1, "store a3 evicts the LRU");
    CHECK(kv_anchor_match(&r, probe1, 2, 0) >= 0, "the touched anchor must survive");
    const int probe2[] = {1, 2, 7};
    int m2 = kv_anchor_match(&r, probe2, 3, 0);
    CHECK(m2 < 0 || kv_anchor_len(&r, m2) != 2, "the untouched anchor must be evicted");
    CHECK(kv_anchor_match(&r, probe3, 4, 0) >= 0, "the newest anchor must be present");
    CHECK(r.evictions >= 1, "an eviction must be counted");

    /* ---- budget --------------------------------------------------------- */
    kv_anchor_ring_free(&r);
    CHECK(kv_anchor_ring_init(&r, 4, 16) == 1, "ring init failed");
    CHECK(kv_anchor_store(&r, base, 3, pl, 2) == 0,
          "an anchor bigger than the whole cap must be refused, not truncated");
    CHECK(r.bytes == 0, "a refused store must charge nothing");
    kv_anchor_ring_free(&r);

    /* A budget that fits exactly one anchor evicts to make room rather than
     * overshooting it. */
    CHECK(kv_anchor_ring_init(&r, 4, 0) == 1, "ring init failed");
    CHECK(kv_anchor_store(&r, a1, 1, pl, 2) == 1, "sizing store");
    size_t one = r.bytes;
    kv_anchor_ring_free(&r);
    CHECK(kv_anchor_ring_init(&r, 4, one) == 1, "ring init failed");
    const int b1[] = {5}, b2[] = {6};   /* DIFFERENT ids: a same-id store is a
                                         * refresh, which would never exercise
                                         * the budget path at all */
    CHECK(kv_anchor_store(&r, b1, 1, pl, 2) == 1, "first fits");
    CHECK(r.bytes == one, "the first anchor must charge exactly one anchor");
    CHECK(kv_anchor_store(&r, b2, 1, pl, 2) == 1, "second must evict to fit");
    CHECK(r.bytes <= one, "the ring must never exceed its budget, got %zu > %zu",
          r.bytes, one);
    const int pb1[] = {5, 9};
    CHECK(kv_anchor_match(&r, pb1, 2, 0) == -1,
          "the anchor evicted for budget must really be gone");

    /* ---- defensive ------------------------------------------------------ */
    kv_anchor_ring r2; memset(&r2, 0, sizeof(r2));
    CHECK(kv_anchor_match(NULL, base, 5, 0) == -1, "NULL ring matches nothing");
    CHECK(kv_anchor_store(&r2, base, 3, pl, 2) == 0, "uninitialised ring stores nothing");
    CHECK(kv_anchor_restore(&r2, 0, pl, 2, 0) == 0, "uninitialised ring restores nothing");
    CHECK(kv_anchor_ids(&r2, 0) == NULL, "uninitialised ring has no ids");
    CHECK(kv_anchor_len(&r2, 0) == 0, "uninitialised ring has no length");
    kv_anchor_ring_free(&r2);                      /* must not crash */
    kv_anchor_ring_clear(&r2);

    kv_anchor_ring_free(&r);
    CHECK(r.slot == NULL && r.n == 0, "free must clear the ring");

    if (failures) { fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    puts("kv_anchor: ok");
    return 0;
}
