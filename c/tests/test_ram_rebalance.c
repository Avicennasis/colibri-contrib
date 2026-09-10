/* test_ram_rebalance.c -- the serve request-boundary rebalance must be
 * shrink-only, floored at dense-resident + KV + pinned, hysteretic past the same
 * 2% + 300 MB band rss_guard uses, inert when RSS_GUARD_GB is explicit, and
 * default-off on unified memory.
 *
 * ram_rebalance_core() takes avail/rss as PARAMETERS precisely so this test can
 * fake a memory-pressure storm (k3_cap_for_ram purity precedent). Section B is
 * the mutation half: real RSS is pumped past the tightened budget with
 * fabricated ecache slabs, and the EXISTING rss_guard must actually free them.
 *
 * Sections C and D exist because A and B BOTH PASSED while the feature never
 * fired once in production serve mode:
 *   C -- A hands `ext` in by construction (it derives avail_now from the ext it
 *        wants), so no arithmetic test can ever catch a wrong SENSOR. C drives
 *        ram_rebalance_boundary(), the adapter production actually calls, with
 *        the real /proc sensors, and pins the two properties rss_gb() lacks:
 *        it must be CURRENT (not a high-water mark) and it must be in the same
 *        decimal GB as mem_available_gb() (rss_gb() is GiB).
 *   D -- neither A nor B can notice that no serve loop CALLS the thing. The
 *        rebalance was wired only into run_serve, while `coli serve` runs
 *        run_serve_mux (openai_server.py spawns the engine with SERVE_BATCH=1).
 *        D reads the sources and pins both call sites plus that coupling.
 */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

/* Pump REAL resident memory.
 *
 * The obvious idiom -- `char *p=malloc(n); for(i+=4096) p[i]=(char)i;` -- is a
 * DEAD STORE: nothing ever reads the buffer, so at -O3 clang deletes the whole
 * loop. gcc happens to keep it, which is why this passed on Linux and failed
 * on Apple silicon for reasons that had nothing to do with either platform.
 *
 * Measured 2026-08-25 on Apple silicon (clang -O3): the loop was elided,
 * ru_maxrss stayed at 0.006 GB across 690 MB of "touches", so rss_guard's
 * lim*1.02+0.3 gate was never crossed and sections B and F reported 8 failed
 * assertions -- against an engine that was behaving CORRECTLY. It declined to
 * evict because nothing was over budget.
 *
 * volatile makes each store observable, so no compiler may remove it. */
static void rr_pump(volatile char *p, size_t n){
    for(size_t i=0;i<n;i+=4096) p[i]=(char)i;
}

/* The guard's OWN gate: rss_guard() returns early unless rss_gb() exceeds
 * lim*1.02+0.3. Every eviction assertion in sections B and F is meaningless
 * unless that is satisfied -- if it is not, the engine correctly does nothing
 * and reading that as a defect is precisely the misdiagnosis.
 *
 * Deliberately NOT an RSS delta. Two different delta sensors were tried and
 * both gave false alarms: rss_gb() is a PEAK, so a later section's pump raises
 * it by zero; self_footprint_gb() is CURRENT, so it reads zero when the
 * allocator already holds the pages resident from an earlier section. The
 * absolute value against the gate has neither problem. */
static int rr_gate_satisfied(const char *sec, double floor_gb){
    double rss = rss_gb(), gate = floor_gb*1.02 + 0.3;
    if(rss > gate) return 1;
    printf("%s. HARNESS BROKEN: rss_gb()=%.3f GB does not exceed the guard's gate of\n"
           "   %.3f GB, so rss_guard will correctly NOT evict. The pump did not raise\n"
           "   real RSS -- at -O3 a non-volatile store loop is dead-code eliminated.\n"
           "   This is a harness failure, NOT an engine failure.\n",
           sec, rss, gate);
    return 0;
}

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

    size_t pump=450u<<20; volatile char *p=malloc(pump);
    if(!p){ printf("B. SKIP: pump alloc failed\n"); rr_free(&m); return; }
    rr_pump(p,pump);

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
    if(!rr_gate_satisfied("B",0.01)){ fails++; free((void*)p); rr_free(&m); return; }
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

    free((void*)p); rr_free(&m);
    printf("B. done\n");
}

/* C. REAL SENSOR PATH. Everything above hands the arithmetic its own answer;
 * this section calls ram_rebalance_boundary() -- the adapter run_serve and
 * run_serve_mux actually call -- and lets it read /proc for itself.
 *
 * The trap the original build fell into: the self-footprint term was rss_gb(),
 * i.e. getrusage's ru_maxrss. That is (a) a HIGH-WATER MARK, so once a long
 * prefill has peaked, freeing memory never lowers it again and the documented
 * invariant "every GB we free returns to MemAvailable and LEAVES the external
 * count" is broken for the rest of the session, and (b) on Linux it is GiB
 * (ru_maxrss KiB / 1024^2) while mem_available_gb() is decimal GB -- a ~7%
 * unit skew on a 15 GB footprint, i.e. a whole GB of phantom pressure.
 * C3 is built so that ru_maxrss returns early (step<=0) where the current
 * footprint fires; it FAILS against the old sensor and passes against the new. */
static void section_c_real_sensors(void){
    Model m; rr_reset(&m);
    m.resident_bytes=4e9;                     /* floor = 4.0 GB, well under every target below */
    g_ram_budget_gb=8.0;                      /* band = 8*0.02 + 0.3 = 0.46 GB */
    double own=self_footprint_gb(), avail=mem_available_gb();
    printf("C. real sensors: self_footprint=%.3f GB rss_gb(ru_maxrss)=%.3f MemAvailable=%.3f GB\n",
           own, rss_gb(), avail);
    if(avail<4.0){ printf("C. SKIP: MemAvailable %.1f GB < 4 -- the probe pumps 2 GB\n", avail);
                   rr_free(&m); return; }

    /* C1. Nobody took anything -- with 1 GB of give-back slack so ordinary
     *     MemAvailable churn on a busy box cannot manufacture a step. The slack
     *     is deliberately smaller than C2's 2 GB pump: a sensor blind to our own
     *     growth would land at +1.0 GB there, past the 0.46 GB band, and C2 would
     *     catch it. */
    g_mem_avail_boot = avail + own - 1.0;
    ram_rebalance_boundary(&m,4096);
    CHECK(g_ram_budget_gb==8.0);
    CHECK(g_ram_ceded_gb==0.0);

    /* C2. OUR OWN growth is not external pressure: 2 GB of anonymous memory
     *     leaves MemAvailable and enters the footprint, so ext must not move.
     *     A sensor blind to anonymous growth would read +2 GB of "others" here
     *     and shrink the budget for memory this very process asked for. */
    size_t pump=(size_t)2<<30; volatile char *p=malloc(pump);
    if(!p){ printf("C. SKIP: 2 GB pump alloc failed\n"); rr_free(&m); return; }
    rr_pump(p,pump);   /* section C reports the sensor gap; no control needed */
    ram_rebalance_boundary(&m,4096);
    CHECK(g_ram_budget_gb==8.0);
    CHECK(g_ram_ceded_gb==0.0);

    /* C3. Hand the 2 GB back. A CURRENT sensor follows it down; ru_maxrss keeps
     *     the peak forever. Now claim 1.2 GB really was taken by others (past
     *     the 0.46 GB band) and require the shrink to happen. */
    free((void*)p);
    own=self_footprint_gb(); avail=mem_available_gb();
    double peak=rss_gb(), gap=peak-own;
    if(gap<1.0){ printf("C. SKIP: peak/current gap only %.2f GB -- allocator kept the pages\n", gap);
                 rr_free(&m); return; }
    g_mem_avail_boot = avail + own + 1.2;                  /* 1.2 GB taken by others */
    double ext_new=(g_mem_avail_boot-avail)-own, ext_old=(g_mem_avail_boot-avail)-peak;
    printf("C. peak %.2f vs current %.2f GB -> ru_maxrss sees ext=%.2f (silent), "
           "self_footprint sees ext=%.2f\n", peak, own, ext_old, ext_new);
    CHECK(ext_old<=0.0);                                   /* the scenario really discriminates */
    ram_rebalance_boundary(&m,4096);
    CHECK(g_ram_budget_gb<8.0);                            /* THE regression: it must fire */
    CHECK(g_ram_budget_gb>=4.0);                           /* and never under dense+KV */
    CHECK(g_ram_ceded_gb>0.0);

    rr_free(&m);
    printf("C. done\n");
}

/* D. WIRING. A and B pass whether or not any serve loop calls the rebalance --
 * which is exactly how it shipped hooked into run_serve only while `coli serve`
 * runs run_serve_mux. These are source assertions rather than behavioural ones
 * because the defect IS the absence of a call: no amount of calling the feature
 * from a test can notice that production never does. */
static char *slurp(const char *p){
    FILE *f=fopen(p,"rb"); if(!f) return NULL;
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n<0){ fclose(f); return NULL; }
    char *b=malloc((size_t)n+1); if(!b){ fclose(f); return NULL; }
    size_t rd=fread(b,1,(size_t)n,f); b[rd]=0; fclose(f); return b;
}
/* `make test-c` runs from c/, but let the test survive being run from the repo root. */
static char *slurp_src(const char *name){
    char path[256]; const char *pre[]={"","../","c/"};
    for(unsigned i=0;i<sizeof pre/sizeof *pre;i++){
        snprintf(path,sizeof path,"%s%s",pre[i],name);
        char *b=slurp(path); if(b) return b;
    }
    return NULL;
}
/* A definition's body runs until the next top-level "\nstatic " (colibri.c has
 * no column-0 `static` inside a function body). */
static int fn_body_has(const char *src,const char *sig,const char *needle){
    const char *s=strstr(src,sig); if(!s) return -1;
    const char *e=strstr(s+strlen(sig),"\nstatic ");
    size_t n = e ? (size_t)(e-s) : strlen(s);
    char *body=malloc(n+1); if(!body) return -1;           /* strstr, not memmem: portable */
    memcpy(body,s,n); body[n]=0;
    int hit = strstr(body,needle)!=NULL;
    free(body); return hit;
}
static void section_d_wiring(void){
    printf("D. wiring (both serve loops reach the boundary)\n");
    char *src=slurp_src("colibri.c");
    if(!src){ printf("FAIL: cannot read colibri.c from any known prefix\n"); fails++; return; }

    /* run_serve_mux is THE production loop: openai_server.py starts the engine
     * with SERVE_BATCH=1, and main() routes SERVE_BATCH -> run_serve_mux. If
     * that coupling ever changes this assertion is the place to notice. */
    char *py=slurp_src("openai_server.py");
    if(!py){ printf("FAIL: cannot read openai_server.py\n"); fails++; }
    else { CHECK(strstr(py,"SERVE_BATCH=\"1\"")!=NULL); free(py); }
    CHECK(strstr(src,"if(getenv(\"SERVE_BATCH\") && atoi(getenv(\"SERVE_BATCH\"))) run_serve_mux")!=NULL);

    CHECK(fn_body_has(src,"static void run_serve_mux(","ram_rebalance_boundary(")==1);
    CHECK(fn_body_has(src,"static void run_serve(","ram_rebalance_boundary(")==1);
    free(src);
    printf("D. done\n");
}

/* ===================== IDLE SHRINK-TO-FLOOR =========================
 * A different axis from the rebalance: that one shrinks when the WORLD takes RAM, this
 * one shrinks when WE stop using it. Same floor, same guard, same shrink-only
 * rule -- so E/F/G below deliberately re-test those invariants on the new entry
 * point rather than assuming they carry over.
 *
 * The one that matters is G. E and F can both pass while the feature never runs,
 * because an idle engine has nothing to run it: run_serve_mux parks in select()
 * with a NULL timeout and the engine owns no timer thread. The shrink is
 * therefore reachable ONLY IF that select takes a finite timeout while a shrink
 * is pending -- so G asserts the timeout, not just the call. Asserting the call
 * alone is precisely the check that let ship dead code. */

static void is_reset(Model *m){
    rr_reset(m);
    g_idle_shrink=1; g_idle_after_s=600.0; g_idle_poll_s=15.0;
    g_idle_unpin=0; g_idle_since=0; g_idle_shrunk=0;
    g_idle_served=1;                 /* default for E1-E10: an engine that has served */
}

static void section_e_idle_arithmetic(void){
    Model m; is_reset(&m);
    printf("E. idle shrink arithmetic + the poll predicate the select depends on\n");

    /* Floor 8 GB, budget 24: high enough that rss_guard's lim*1.02+0.3 stays
     * above this process's real RSS, so E stays pure arithmetic (same trick A
     * uses). kv_pool_bytes is 0 here -- cfg widths are zero. */
    m.resident_bytes=8e9;

    /* E1. Silence shorter than the threshold does nothing. */
    CHECK(idle_shrink_core(&m,4096,599.0)==0);
    CHECK(g_ram_budget_gb==24.0);
    CHECK(idle_poll_wanted()==1);          /* still armed: the loop must keep waking */

    /* E2. Past the threshold: straight to the floor, one line, one step. */
    CHECK(idle_shrink_core(&m,4096,600.0)==1);
    CHECK(fabs(g_ram_budget_gb-8.0)<1e-9);

    /* E3. THE PREDICATE. Once the shrink is done there is nothing left to wake
     *     for, so the poll must disarm and the loop go back to blocking. An
     *     always-armed predicate spins a syscall every poll interval forever;
     *     a never-armed one is the dead-code bug. Both are caught here. */
    CHECK(idle_poll_wanted()==0);

    /* E4. Latch: more silence changes nothing (no re-shrink, no second log). */
    CHECK(idle_shrink_core(&m,4096,86400.0)==0);
    CHECK(fabs(g_ram_budget_gb-8.0)<1e-9);

    /* E5. NO GROW PATH. Traffic returns, then the box goes quiet again: the
     *     latch re-arms (a fresh silence deserves its own attempt) but the
     *     budget must NOT climb back -- shrink-only is the ticket's hard rule. */
    idle_mark(1);
    CHECK(g_idle_shrunk==0 && g_idle_since==0.0);
    CHECK(fabs(g_ram_budget_gb-8.0)<1e-9);          /* still at the floor */
    idle_mark(0);
    CHECK(g_idle_since>0);
    CHECK(idle_shrink_core(&m,4096,600.0)==0);      /* already at floor: nothing to do */
    CHECK(fabs(g_ram_budget_gb-8.0)<1e-9);
    CHECK(idle_poll_wanted()==0);                   /* and it disarms again */

    /* E6. The floor is never crossed, however long the silence. */
    is_reset(&m); m.resident_bytes=20e9;
    CHECK(idle_shrink_core(&m,4096,1e6)==1);
    CHECK(fabs(g_ram_budget_gb-20.0)<1e-9);         /* clamped at dense+KV+pinned */

    /* E7. A floor ABOVE the budget must not raise it (the grow path again, by
     *     the back door: an engine whose resident set already exceeds budget). */
    is_reset(&m); m.resident_bytes=30e9;
    CHECK(idle_shrink_core(&m,4096,1e6)==0);
    CHECK(g_ram_budget_gb==24.0);                   /* untouched, not raised to 30 */

    /* E8. Explicit RSS_GUARD_GB stays authoritative (#379 rule, as in). */
    is_reset(&m); m.resident_bytes=8e9; setenv("RSS_GUARD_GB","4",1);
    CHECK(idle_shrink_core(&m,4096,1e6)==0);
    CHECK(g_ram_budget_gb==24.0);
    CHECK(idle_poll_wanted()==0);                   /* and never even wakes to try */
    unsetenv("RSS_GUARD_GB");

    /* E9. The __APPLE__ default shape: flag off -> total no-op, no wakeups. */
    is_reset(&m); m.resident_bytes=8e9; g_idle_shrink=0;
    CHECK(idle_shrink_core(&m,4096,1e6)==0);
    CHECK(g_ram_budget_gb==24.0);
    CHECK(idle_poll_wanted()==0);

    /* E10. IDLE_SHRINK_MIN is honoured as a threshold, not hardcoded at 600. */
    is_reset(&m); m.resident_bytes=8e9; g_idle_after_s=60.0;
    CHECK(idle_shrink_core(&m,4096,59.0)==0);
    CHECK(idle_shrink_core(&m,4096,61.0)==1);
    CHECK(fabs(g_ram_budget_gb-8.0)<1e-9);

    /* E11. A NEVER-SERVED engine is "waiting", not "abandoned". Its LRU is still
     *      empty, so shrinking frees essentially nothing (dense and pins are the
     *      floor) while capping ecap for the rest of the process's life -- a
     *      gateway restarted overnight would serve all the next day with a
     *      crippled cache for no RAM back. The clock must not even start, and
     *      the loop must not burn a wakeup every poll interval to find that out. */
    is_reset(&m); m.resident_bytes=8e9; g_idle_served=0;
    idle_mark(0);
    CHECK(g_idle_since==0.0);                       /* clock never started */
    CHECK(idle_poll_wanted()==0);                   /* and no wakeups armed */
    CHECK(idle_shrink_poll(&m,4096)==0);
    CHECK(g_ram_budget_gb==24.0);
    /* first request completes -> from here silence really is abandonment */
    g_idle_served=1; idle_mark(0);
    CHECK(g_idle_since>0);
    CHECK(idle_poll_wanted()==1);

    rr_free(&m);
    printf("E. done\n");
}

/* F. MUTATION: the idle path must reach the REAL rss_guard and actually free
 * memory, not just move a double. Same shape as B but entered through
 * idle_shrink_core, because "the arithmetic is right" and "the LRU got freed"
 * are separate claims and only the second one gives dev its RAM back. */
static void section_f_idle_evicts(void){
    Model m; is_reset(&m);
    double mem0=mem_available_gb();
    printf("F. idle mutation (real slabs; MemAvailable %.1f GB)\n", mem0);
    if(mem0<2.0){ printf("F. SKIP: MemAvailable %.1f GB < 2\n", mem0); rr_free(&m); return; }

    size_t pump=450u<<20; volatile char *p=malloc(pump);
    if(!p){ printf("F. SKIP: pump alloc failed\n"); rr_free(&m); return; }
    rr_pump(p,pump);

    for(int z=0;z<2;z++){                                 /* 2 x 120 MB fabricated LRU slabs */
        ESlot *s=&m.ecache[0][z];
        s->eid=z; s->used=(uint64_t)(z+1);
        s->slab=malloc(120u<<20); s->slab_cap=120LL<<20;
    }
    m.resident_bytes=10e6;                                /* floor 0.01 GB: guard must fire */
    g_ram_budget_gb=24.0;
    int ecap0=m.ecap;

    if(!rr_gate_satisfied("F",0.01)){ fails++; free((void*)p); rr_free(&m); return; }
    CHECK(idle_shrink_core(&m,4096,3600.0)==1);
    CHECK(fabs(g_ram_budget_gb-0.01)<1e-6);               /* budget at the floor */
    CHECK(m.ecache[0][0].eid==-1 && m.ecache[0][0].slab==NULL);   /* freed in place */
    CHECK(m.ecache[0][0].used==0);
    CHECK(m.ecache[0][1].eid==-1 && m.ecache[0][1].slab==NULL);
    CHECK(m.ecap<ecap0 && m.ecap>=2);                     /* cap fell and stayed >= 2 */

    /* and the shrink does not undo itself on the next idle poll */
    CHECK(idle_shrink_core(&m,4096,7200.0)==0);
    CHECK(fabs(g_ram_budget_gb-0.01)<1e-6);

    free((void*)p); rr_free(&m);
    printf("F. done\n");
}

/* G. REACHABILITY. The section that would have caught the rebalance's shipped-dead-code
 * defect, applied to this feature before it can repeat it.
 *
 * An idle run_serve_mux blocks in select() forever, so calling idle_shrink_poll
 * from the top of that loop is NOT sufficient: the loop has to get there. The
 * finite timeout is the feature; the call is just where it lands. G therefore
 * asserts the timeout condition itself, and asserts that the old
 * blocks-forever-when-idle line is GONE -- reinstating it would leave every
 * check in E and F passing while the engine never shrinks in production.
 *
 * G also pins the asymmetry with run_serve ON PURPOSE. run_serve blocks in
 * getline() with no poll point; the only moment it could notice the idle is
 * after a request has already arrived, i.e. the worst possible moment to drop
 * the cache. So it is deliberately NOT wired, and this assertion stops a future
 * reader from "fixing" the omission with a call that fires at the wrong time. */
static void section_g_reachability(void){
    printf("G. reachability (the idle wakeup exists, and only where it helps)\n");
    char *src=slurp_src("colibri.c");
    if(!src){ printf("FAIL: cannot read colibri.c from any known prefix\n"); fails++; return; }

    /* Production runs the mux loop: same SERVE_BATCH coupling D pins. */
    CHECK(strstr(src,"if(getenv(\"SERVE_BATCH\") && atoi(getenv(\"SERVE_BATCH\"))) run_serve_mux")!=NULL);

    /* The call, and the idle clock that feeds it. */
    CHECK(fn_body_has(src,"static void run_serve_mux(","idle_shrink_poll(")==1);
    CHECK(fn_body_has(src,"static void run_serve_mux(","idle_mark(")==1);

    /* THE ONE THAT MATTERS: the idle select must consult idle_poll_wanted(), or
     * the two checks above are decoration on an unreachable branch. */
    CHECK(fn_body_has(src,"static void run_serve_mux(","idle_poll_wanted()")==1);

    /* And the blocks-forever-when-idle shape must be gone. This is the exact
     * line the feature replaces; if it ever comes back, so does the dead code. */
    CHECK(fn_body_has(src,"static void run_serve_mux(",
                          "struct timeval tv={0,0}, *ptv=active?&tv:NULL;")==0);

    /* Deliberate asymmetry, asserted so it stays deliberate (see the header). */
    CHECK(fn_body_has(src,"static void run_serve(","idle_shrink_poll(")==0);

    /* The idle clock is armed by a COMPLETED request, so the flag has to be
     * raised where requests complete -- mux_done, next to the rebalance's boundary
     * flag. Without this the clock never starts and E11's gate never opens. */
    CHECK(fn_body_has(src,"static void mux_done(","g_idle_served=1;")==1);

    /*'s boundary must survive intact in both loops -- this ticket edits
     * that same region and a merge slip there would silently un-ship. */
    CHECK(fn_body_has(src,"static void run_serve_mux(","ram_rebalance_boundary(")==1);
    CHECK(fn_body_has(src,"static void run_serve(","ram_rebalance_boundary(")==1);

    /* Unpinning the AUTOPIN tier is opt-in: it has never run on hardware, and a
     * default-on release of pinned experts would be a silent-wrong-output risk
     * (npin must be zeroed before any slab is touched). Pin the default off. */
    CHECK(strstr(src,"static int    g_idle_unpin   = 0;")!=NULL);
    /* ...and pin the ordering inside the release itself: npin[l]=0 has to come
     * before the slab teardown, under g_pilot_mx, or a reader can still find a
     * pin whose pages are being handed back. */
    { const char *f=strstr(src,"static int64_t idle_pin_release(");
      const char *lock = f?strstr(f,"pthread_mutex_lock(&g_pilot_mx)"):NULL;
      const char *zero = f?strstr(f,"m->npin[l]=0;"):NULL;
      const char *tear = f?strstr(f,"munlock("):NULL;
      CHECK(f && lock && zero && tear);
      if(f&&lock&&zero&&tear){ CHECK(lock<zero); CHECK(zero<tear); } }

    free(src);
    printf("G. done\n");
}

int main(void){
#ifdef _WIN32
    /* #51279: instant crash under UCRT64 before any section output
     * (stdout buffering hides even section A/B prints). Skipped wholesale
     * pending a Windows debug pass; Linux + macOS keep the full suite. */
    puts("ram rebalance tests: skipped on Windows (#51279)");
    return 0;
#endif
    section_a_arithmetic();
    section_b_mutation();
    section_c_real_sensors();
    section_d_wiring();
    section_e_idle_arithmetic();
    section_f_idle_evicts();
    section_g_reachability();
    if(fails){ printf("test_ram_rebalance: %d FAIL\n", fails); return 1; }
    printf("test_ram_rebalance: all checks passed\n");
    return 0;
}
