/* Throughput of gq_matmul per type on an expert-shaped matrix (O x I). */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "../ggml_quants.h"
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
int main(int argc, char **argv){
    int O = argc > 1 ? atoi(argv[1]) : 512, I = argc > 2 ? atoi(argv[2]) : 2048, reps = 200;
    int types[] = { GQ_Q4_K, GQ_Q5_K, GQ_Q6_K, GQ_Q8_0, GQ_Q2_K, GQ_Q3_K, GQ_IQ2_XXS, GQ_IQ2_S, GQ_IQ3_S, GQ_IQ3_XXS, GQ_IQ4_XS, GQ_IQ1_M, GQ_MXFP4 };
    float *x = malloc((size_t)I*4), *y = malloc((size_t)O*4);
    for (int i = 0; i < I; i++) x[i] = (float)((i*37)%17) * 0.01f;
    printf("%d threads, matrix %d x %d, S=1\n", omp_get_max_threads(), O, I);
    for (size_t k = 0; k < sizeof types/sizeof *types; k++) {
        int t = types[k]; int64_t rb = gq_row_bytes(t, I);
        uint8_t *W = malloc((size_t)rb*O); for (int64_t i = 0; i < rb*O; i++) W[i] = (uint8_t)(i*131u + 7u);
        for (int64_t o = 0; o < O; o++) for (int64_t b = 0; b < rb; b += GQ_TYPES[t].bytes) { if (t != GQ_MXFP4 && t != GQ_IQ1_M) { W[o*rb+b] = 0x00; W[o*rb+b+1] = 0x30; } }
        gq_matmul(y, x, W, t, 1, I, O);
        double t0 = now(); for (int r = 0; r < reps; r++) gq_matmul(y, x, W, t, 1, I, O); double dt = (now()-t0)/reps;
        printf("%-8s %7.3f ms  %6.2f GB/s weights  %6.2f Gelem/s\n", gq_type_name(t), dt*1e3, rb*O/dt/1e9, (double)O*I/dt/1e9);
        free(W);
    }
    return 0;
}
