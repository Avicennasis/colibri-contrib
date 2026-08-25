/* kv_anchor.h — let a serve slot remember MORE THAN ONE reusable attention state.
 *
 * WHAT kv_prefix.h ALREADY DOES, AND WHERE IT STOPS. colibri.c pins a
 * conversation to a KV slot and, on every prompt, walks the longest common
 * prefix of the new token ids against the ids the slot currently holds
 * (colibri.c mux_submit). Positions past the divergence are simply abandoned —
 * this KV is position-addressed, so truncation is free. Measured on v1.7.0
 * (#50401, 6-turn bench): context grew 25 -> 181 tokens while prefill work
 * stayed flat at 11-25 tokens. For an APPEND-ONLY conversation that is already
 * optimal and kv_anchor buys exactly nothing; the rescope note on #50403 says
 * so in as many words.
 *
 * The gap is that a slot remembers ONE state: the last one. The moment a
 * different prompt lands on the slot, the old state is overwritten, and when
 * the first conversation comes back it re-prefills from the divergence point —
 * even though the engine held precisely the right rows a turn ago. That is one
 * mechanism with three faces in an agent workload:
 *
 *   - two conversations interleaved on one slot. KV_SLOTS defaults to 1 and
 *     openai_server.py spawns the engine that way, so cross-slot adoption
 *     (COLI_KV_SHARE, which needs nctx>1) cannot help there at all;
 *   - a branch: the agent rewrites history mid-conversation, then a later turn
 *     continues the ORIGINAL branch (a retry, a discarded tool call);
 *   - a tool loop whose reply the client re-renders, so the live state diverges
 *     inside the reply while an earlier prompt-end is still an exact prefix.
 *
 * WHAT THIS IS NOT. An anchor cannot resume past an INSERTION. If a turn
 * splices a tool result in at position k, every later token genuinely attends
 * to it: its rows are a different function of the history, and the RoPE'd key
 * rows sit at the wrong absolute positions besides. Reusing them would answer
 * from a state the conversation never had. So the matching rule here is the
 * same strict prefix kv_prefix.h uses — the candidate SET grows, the RULE does
 * not weaken — and reuse stays bit-exact by construction. Measured
 * consequence, honestly: the post-insert turn does not improve. The turn that
 * returns to a remembered branch does.
 *
 * COST. An anchor is a copy of the KV rows it covers, so it is priced like KV:
 * (n_layers+1)*(kv_lora+qk_rope)*4 + dsa_layers*index_hd*4 bytes per position
 * (~217 KB/token for GLM-5.2). That is why the ring is opt-in and budgeted:
 * COLI_KV_ANCHOR=<slots> arms it, COLI_KV_ANCHOR_MB caps the total.
 *
 * INVARIANT: an anchor's rows are exactly the KV the engine held at positions
 * 0..len-1 when ids[0..len-1] had been fed, in position order. Everything else
 * follows. A restore therefore only ever installs rows the engine could have
 * computed itself from the same ids.
 */
#ifndef KV_ANCHOR_H
#define KV_ANCHOR_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* One family of position-addressed KV buffers: `n` per-layer planes, each laid
 * out as [positions x width] floats — the layout coli_kv_row() indexes. GLM
 * passes three: Lc (compressed kv), Rc (decoupled RoPE keys) and Ic (the DSA
 * index keys). A NULL entry inside plane[] is legal and skipped: Ic exists only
 * for the layers that own an index, and only when the model has DSA at all. */
typedef struct { float *const *plane; int n; int width; } kv_anchor_plane;

typedef struct {
    int      *ids;      /* token ids at positions 0..len-1 */
    int       len;      /* positions this anchor covers */
    float   **rows;     /* one buffer per (plane, layer); len*width floats each */
    int      *width;    /* that buffer's row width, for restore-time validation */
    int       nrows;    /* == sum of plane counts at capture time */
    uint64_t  used;     /* LRU clock stamp */
    size_t    bytes;    /* what this anchor charges against the ring budget */
} kv_anchor;

typedef struct {
    kv_anchor *slot;
    int        n;       /* 0 = ring disarmed; every entry point is then inert */
    uint64_t   clock;
    size_t     bytes;   /* currently held */
    size_t     budget;  /* 0 = unlimited */
    uint64_t   hits, stores, evictions;
} kv_anchor_ring;

static inline void kv_anchor_drop(kv_anchor_ring *r, kv_anchor *a) {
    if (a->rows) { for (int i = 0; i < a->nrows; i++) free(a->rows[i]); free(a->rows); }
    free(a->width);
    free(a->ids);
    if (r) { r->bytes = (r->bytes >= a->bytes) ? r->bytes - a->bytes : 0; }
    memset(a, 0, sizeof(*a));
}

/* Arm a ring with `n` slots and a byte budget (0 = unlimited). n<=0 leaves the
 * ring disarmed, which every other entry point treats as "do nothing" —
 * allocation failure must never be the reason a turn fails. */
static inline int kv_anchor_ring_init(kv_anchor_ring *r, int n, size_t budget) {
    if (!r) return 0;
    memset(r, 0, sizeof(*r));
    if (n <= 0) return 0;
    r->slot = (kv_anchor *)calloc((size_t)n, sizeof(kv_anchor));
    if (!r->slot) return 0;
    r->n = n;
    r->budget = budget;
    return 1;
}

/* Forget every anchor. Pair with whatever invalidates the slot's KV geometry
 * (kv_alloc): the rows describe positions in buffers that no longer exist. */
static inline void kv_anchor_ring_clear(kv_anchor_ring *r) {
    if (!r || !r->slot) return;
    for (int i = 0; i < r->n; i++) kv_anchor_drop(r, &r->slot[i]);
    r->bytes = 0;
}

static inline void kv_anchor_ring_free(kv_anchor_ring *r) {
    if (!r) return;
    kv_anchor_ring_clear(r);
    free(r->slot);
    memset(r, 0, sizeof(*r));
}

static inline int kv_anchor_len(const kv_anchor_ring *r, int idx) {
    if (!r || !r->slot || idx < 0 || idx >= r->n) return 0;
    return r->slot[idx].len;
}

static inline const int *kv_anchor_ids(const kv_anchor_ring *r, int idx) {
    if (!r || !r->slot || idx < 0 || idx >= r->n) return NULL;
    return r->slot[idx].ids;
}

/* The longest anchor whose ids are a strict prefix of ids[0..n-1] and which
 * covers MORE than `have` positions; -1 when none does.
 *
 * Two rejection rules, both inherited from kv_prefix_reuse and both load-
 * bearing. `a->len < n` keeps at least one token to prefill, so the caller
 * still has a final hidden state to sample from — and it is what makes an
 * anchor a resume point rather than a rewind. `a->len > have` means an anchor
 * is only ever installed when it strictly beats what the slot already holds:
 * restoring a SHORTER state would throw away live rows for nothing. */
static inline int kv_anchor_match(const kv_anchor_ring *r, const int *ids,
                                  int n, int have) {
    if (!r || !r->slot || !ids || n <= 0) return -1;
    int best = -1, best_len = have;
    for (int i = 0; i < r->n; i++) {
        const kv_anchor *a = &r->slot[i];
        if (!a->ids || a->len <= best_len || a->len >= n) continue;
        if (memcmp(a->ids, ids, (size_t)a->len * sizeof(int)) != 0) continue;
        best = i; best_len = a->len;
    }
    return best;
}

/* Copy anchor `idx`'s rows for positions [from, len) into the live planes.
 *
 * `from` is what the slot already holds and has verified identical, so those
 * rows are skipped — the same shape as colibri.c's cross-slot adopt, which
 * copies only [sc->len, blen). Returns the number of positions now covered, or
 * 0 if the planes do not describe the geometry this anchor was captured from
 * (in which case nothing was written and the caller keeps its own state).
 *
 * Validation is not paranoia: a restore into mismatched geometry does not
 * crash, it answers the user out of rows that mean something else. */
static inline int kv_anchor_restore(kv_anchor_ring *r, int idx,
                                    const kv_anchor_plane *pl, int npl,
                                    int from) {
    if (!r || !r->slot || idx < 0 || idx >= r->n || !pl || npl <= 0) return 0;
    kv_anchor *a = &r->slot[idx];
    if (!a->ids || !a->rows || from < 0 || from >= a->len) return 0;
    int want = 0;
    for (int p = 0; p < npl; p++) want += pl[p].n;
    if (want != a->nrows) return 0;
    for (int p = 0, k = 0; p < npl; p++)
        for (int l = 0; l < pl[p].n; l++, k++)
            if (a->width[k] != (pl[p].plane[l] ? pl[p].width : 0)) return 0;
    for (int p = 0, k = 0; p < npl; p++) {
        int w = pl[p].width;
        for (int l = 0; l < pl[p].n; l++, k++) {
            float *dst = pl[p].plane[l];
            if (!dst || !a->rows[k]) continue;
            memcpy(dst + (size_t)from * (size_t)w,
                   a->rows[k] + (size_t)from * (size_t)w,
                   (size_t)(a->len - from) * (size_t)w * sizeof(float));
        }
    }
    a->used = ++r->clock;
    r->hits++;
    return a->len;
}

/* Remember the state that ids[0..len-1] built.
 *
 * Re-capturing a state the ring already holds only refreshes its LRU stamp:
 * a tool loop resends the same prefix every turn, and duplicating it would
 * evict the very branches the ring exists to keep. Returns 1 if the ring now
 * holds this state. Any allocation failure leaves the ring smaller but
 * consistent and returns 0 — capture is an optimisation, never a hard error. */
static inline int kv_anchor_store(kv_anchor_ring *r, const int *ids, int len,
                                  const kv_anchor_plane *pl, int npl) {
    if (!r || !r->slot || !ids || len <= 0 || !pl || npl <= 0) return 0;
    int nrows = 0;
    size_t need = (size_t)len * sizeof(int);
    for (int p = 0; p < npl; p++) {
        nrows += pl[p].n;
        for (int l = 0; l < pl[p].n; l++)
            if (pl[p].plane[l]) need += (size_t)len * (size_t)pl[p].width * sizeof(float);
    }
    if (nrows <= 0) return 0;
    if (r->budget && need > r->budget) return 0;      /* one anchor over the whole cap */

    for (int i = 0; i < r->n; i++) {
        kv_anchor *a = &r->slot[i];
        if (a->ids && a->len == len &&
            !memcmp(a->ids, ids, (size_t)len * sizeof(int))) {
            a->used = ++r->clock;
            return 1;
        }
    }
    /* Victim: an empty slot, else the least recently used one. */
    int victim = -1;
    for (int i = 0; i < r->n && victim < 0; i++) if (!r->slot[i].ids) victim = i;
    if (victim < 0) {
        victim = 0;
        for (int i = 1; i < r->n; i++)
            if (r->slot[i].used < r->slot[victim].used) victim = i;
        r->evictions++;
    }
    kv_anchor_drop(r, &r->slot[victim]);
    /* Make room under the budget by evicting other LRU anchors. */
    while (r->budget && r->bytes + need > r->budget) {
        int lru = -1;
        for (int i = 0; i < r->n; i++) {
            if (i == victim || !r->slot[i].ids) continue;
            if (lru < 0 || r->slot[i].used < r->slot[lru].used) lru = i;
        }
        if (lru < 0) return 0;                        /* nothing left to give */
        kv_anchor_drop(r, &r->slot[lru]);
        r->evictions++;
    }

    kv_anchor *a = &r->slot[victim];
    a->ids   = (int *)malloc((size_t)len * sizeof(int));
    a->rows  = (float **)calloc((size_t)nrows, sizeof(float *));
    a->width = (int *)calloc((size_t)nrows, sizeof(int));
    if (!a->ids || !a->rows || !a->width) { kv_anchor_drop(r, a); return 0; }
    a->nrows = nrows;
    memcpy(a->ids, ids, (size_t)len * sizeof(int));
    for (int p = 0, k = 0; p < npl; p++) {
        int w = pl[p].width;
        for (int l = 0; l < pl[p].n; l++, k++) {
            if (!pl[p].plane[l]) { a->width[k] = 0; continue; }
            size_t nb = (size_t)len * (size_t)w * sizeof(float);
            a->rows[k] = (float *)malloc(nb);
            if (!a->rows[k]) { kv_anchor_drop(r, a); return 0; }
            memcpy(a->rows[k], pl[p].plane[l], nb);
            a->width[k] = w;
        }
    }
    a->len   = len;
    a->bytes = need;
    a->used  = ++r->clock;
    r->bytes += need;
    r->stores++;
    return 1;
}

#endif /* KV_ANCHOR_H */
