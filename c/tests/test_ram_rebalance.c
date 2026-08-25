/* test_ram_rebalance.c -- #50402: the serve request-boundary rebalance must be
 * shrink-only, floored at dense-resident + KV + pinned, hysteretic past the same
 * 2% + 300 MB band rss_guard uses, inert when RSS_GUARD_GB is explicit, and
 * default-off on unified memory.
 *
 * ram_rebalance_core() takes avail/rss as PARAMETERS precisely so this test can
 * fake a memory-pressure storm (k3_cap_for_ram purity precedent). Section B is
 * the mutation half: real RSS is pumped past the tightened budget with
 * fabricated ecache slabs, and the EXISTING rss_guard must actually free them.
 */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

/* Deterministic harness: budget 24 GB resolved from a 40 GB-quiet boot, two
 * sparse layers + the MTP row of ecache (the guard walks l<=n_layers), and a
 * floor driven only by resident_bytes (cfg widths are zero -> kv pool 0). */
static void rr_reset(Model *m){
    memset(m,0,sizeof *m);
    m->c.n_layers=2;
    m->ecache=calloc(3,sizeof(ESlot*));
    m->ecn=calloc(3,sizeof(int));
    m->ecap=6; m->n_emit=1000;
    for(int l=0;l<3;l++){ m->ecache[l]=calloc(2,sizeof(ESlot)); m->ecn[l]=2; }
    unsetenv("RSS_GUARD_GB");
    g_ram_rebalance=1; g_ram_ceded_gb=0;
    g_ram_budget_gb=24.0; g_mem_avail_boot=40.0; g_rssg_last=0;
}

static void rr_free(Model *m){
    for(int l=0;l<3;l++){ for(int z=0;z<2;z++) free(m->ecache[l][z].slab); free(m->ecache[l]); }
    free(m->ecache); free(m->ecn);
}

/* ext = GB the OUTSIDE world took since boot; boot/rss are arranged so the
 * arithmetic lands exactly on it regardless of this process's real footprint. */
static void rr_pressure(Model *m,double rss_now,double ext_gb){
    double avail_now=g_mem_avail_boot-rss_now-ext_gb;
    ram_rebalance_core(m,4096,avail_now,rss_now);
}

static void section_a_arithmetic(void){
    Model m; rr_reset(&m);
    printf("A. arithmetic (fake sensors, no eviction: floor keeps the guard's\n"
           "   lim*1.02+0.3 threshold above this process's real RSS)\n");

    /* 1. quiet machine: nothing taken -> budget untouched */
    m.resident_bytes=0; rr_pressure(&m,10.0,0.0);
    CHECK(g_ram_budget_gb==24.0);

    /* 2. the motivating storm: 8 GB taken by others -> 24 -> 16 */
    rr_pressure(&m,10.0,8.0);
    CHECK(fabs(g_ram_budget_gb-16.0)<1e-9);
    CHECK(fabs(g_ram_ceded_gb-8.0)<1e-9);

    /* 3. idempotent under the SAME pressure (no double-count: what we freed
     *    returned to MemAvailable and left the external count) */
    rr_pressure(&m,10.0,8.0);
    CHECK(fabs(g_ram_budget_gb-16.0)<1e-9);

    /* 4. our own frees are invisible: rss -F, avail +F -> still no step */
    rr_pressure(&m,8.0,8.0);
    CHECK(fabs(g_ram_budget_gb-16.0)<1e-9);

    /* 5. pressure GREW by 2 -> exactly one 2 GB step more */
    rr_pressure(&m,10.0,10.0);
    CHECK(fabs(g_ram_budget_gb-14.0)<1e-9);
    CHECK(fabs(g_ram_ceded_gb-10.0)<1e-9);

    /* 6. GROW PATH ABSENT: the world gives 10 GB back -> budget stays put */
    rr_pressure(&m,10.0,0.0);
    CHECK(fabs(g_ram_budget_gb-14.0)<1e-9);
    CHECK(fabs(g_ram_ceded_gb-10.0)<1e-9);

    /* 7. floor: dense-resident + KV (pinned lives in resident_bytes) */
    rr_reset(&m); m.resident_bytes=8e9;
    rr_pressure(&m,10.0,29.0);            /* raw target 24-29 = -5 -> clamped at 8 */
    CHECK(fabs(g_ram_budget_gb-8.0)<1e-9);
    CHECK(fabs(g_ram_ceded_gb-16.0)<1e-9);

    /* 8. hysteresis: 0.4 GB is inside the 2%+300 MB band (0.78 GB at 24) ->
     *    no step AND no ceded bookkeeping, so a slow leak still adds up and
     *    fires once it crosses the band (1 GB does) */
    rr_reset(&m); rr_pressure(&m,10.0,0.4);
    CHECK(g_ram_budget_gb==24.0 && g_ram_ceded_gb==0.0);
    rr_pressure(&m,10.0,1.0);
    CHECK(fabs(g_ram_budget_gb-23.0)<1e-9);

    /* 9. explicit RSS_GUARD_GB stays authoritative (explicit wins, #379 rule) */
    rr_reset(&m); setenv("RSS_GUARD_GB","4",1);
    rr_pressure(&m,10.0,20.0);
    CHECK(g_ram_budget_gb==24.0 && g_ram_ceded_gb==0.0);
    unsetenv("RSS_GUARD_GB");

    /* 10. the unified-memory default shape: flag off -> total no-op */
    rr_reset(&m); g_ram_rebalance=0;
    rr_pressure(&m,10.0,20.0);
    CHECK(g_ram_budget_gb==24.0 && g_ram_ceded_gb==0.0);

    rr_free(&m);
    printf("A. done (%d fails)\n", fails);
}

/* B. MUTATION: pump real RSS past a tightened budget; the existing rss_guard
 * must evict the fabricated LRU slabs in place and drop ecap, exactly as it
 * does in production. ~0.45 GB touched + ~0.24 GB of untouched (virtual)
 * slabs, skipped politely when the box is too tight to probe it. */
static void section_b_mutation(void){
    Model m; rr_reset(&m);
    double mem0=mem_available_gb();
    printf("B. mutation (real RSS pumped; MemAvailable %.1f GB)\n", mem0);
    if(mem0<2.0){ printf("B. SKIP: MemAvailable %.1f GB < 2 -- probe needs ~0.7 GB\n", mem0); rr_free(&m); return; }

    size_t pump=450u<<20; char *p=malloc(pump);
    if(!p){ printf("B. SKIP: pump alloc failed\n"); rr_free(&m); return; }
    for(size_t i=0;i<pump;i+=4096) p[i]=(char)i;         /* touch -> real RSS */

    /* 2 x 120 MB slabs on layer 0 (untouched: virtual, guard reads slab_cap) */
    for(int z=0;z<2;z++){
        ESlot *s=&m.ecache[0][z];
        s->eid=z; s->used=(uint64_t)(z+1);
        s->slab=malloc(120u<<20); s->slab_cap=120LL<<20;
    }
    double rss=rss_gb(), avail=mem_available_gb();
    g_mem_avail_boot=avail+rss+23.99;                    /* ext = 23.99 GB */
    m.resident_bytes=10e6;                               /* floor = 0.01 GB -> lim valid */
    int ecap0=m.ecap;
    rr_pressure(&m,rss,23.99);                           /* 24 -> 0.01: guard must fire */

    CHECK(fabs(g_ram_budget_gb-0.01)<1e-6);              /* budget at the floor */
    CHECK(fabs(g_ram_ceded_gb-23.99)<1e-6);
    CHECK(m.ecache[0][0].eid==-1 && m.ecache[0][0].slab==NULL);   /* evicted in place */
    CHECK(m.ecache[0][0].used==0);
    CHECK(m.ecache[0][1].eid==-1 && m.ecache[0][1].slab==NULL);
    CHECK(m.ecap<ecap0 && m.ecap>=2);                    /* cap fell, floor at 2 */

    /* single step per boundary: same pressure again changes nothing */
    rss=rss_gb(); avail=mem_available_gb(); g_mem_avail_boot=avail+rss+23.99;
    rr_pressure(&m,rss,23.99);
    CHECK(fabs(g_ram_budget_gb-0.01)<1e-6);
    CHECK(m.ecap>=2);

    /* and the world recovering grows nothing back */
    rss=rss_gb(); g_mem_avail_boot=rss+40.0;
    rr_pressure(&m,rss,0.0);
    CHECK(fabs(g_ram_budget_gb-0.01)<1e-6);

    free(p); rr_free(&m);
    printf("B. done\n");
}

int main(void){
    section_a_arithmetic();
    section_b_mutation();
    if(fails){ printf("test_ram_rebalance: %d FAIL\n", fails); return 1; }
    printf("test_ram_rebalance: all checks passed\n");
    return 0;
}
