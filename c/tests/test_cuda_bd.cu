/* A GGUF's block formats on the CUDA device (cuda_bd.cuh): every ggml block type the
 * backend takes, random blocks with sane scale fields, against the CPU's decoders
 * (ggml_quants.h through tests/cuda_bd_ref.c):
 *   - the integer form's types against the CPU's integer form (the same codes, scales and
 *     integer sums, in double), the float form's against the CPU's dequantized rows;
 *   - a verify's 2, 3 and 4 rows, sharing every weight load, each with the bits of the
 *     same row alone (a decode step's).
 *   - the backend's path: coli_cuda_tensor_upload of the file's rows (padded to words on the
 *     device where a row is not), the dense chain's matmul, the same bits as the kernel's.
 * BENCH=1 also times the GEMV at a few shapes, 1 to 4 rows. Build from c/:
 *   gcc -O2 -c tests/cuda_bd_ref.c -o tests/cuda_bd_ref.o
 *   nvcc -O2 -std=c++17 -arch=native tests/test_cuda_bd.cu tests/cuda_bd_ref.o -o tests/test_cuda_bd */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include "../backend_cuda.cu"

extern "C" {
int ref_unpack(int t, const uint8_t *row, int n, int8_t *w8, float *sw, float *mw);
void ref_quant(const float *x, int n, int8_t *xq, float *dx, int32_t *xs);
int ref_dequant(int t, const uint8_t *row, float *out, int n);
long ref_row_bytes(int t, int n);
int ref_block_bytes(int t);
int ref_block_elems(int t);
const char *ref_name(int t);
}

static int fails;
static unsigned long long g_rng = 0x9e3779b97f4a7c15ull;
static unsigned rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (unsigned)(g_rng >> 11); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* sane scale fields in a random block (test_vk_chain.c's bd_fix, with the float-form types) */
static void bd_fix(int t, uint8_t *b) {
#define F16AT(o) do { b[o] = 0x00; b[(o) + 1] = (uint8_t)(0x24 + rnd() % 3); } while (0)
    switch (t) {
    case 12: case 13: F16AT(0); F16AT(2); break;   /* Q4_K, Q5_K */
    case 3: case 7: F16AT(0); F16AT(2); break;     /* Q4_1, Q5_1 */
    case 10: F16AT(80); F16AT(82); break;          /* Q2_K */
    case 11: F16AT(108); break;                    /* Q3_K */
    case 14: F16AT(208); break;                    /* Q6_K */
    case 34: F16AT(52); break;                     /* TQ1_0 */
    case 35: F16AT(64); break;                     /* TQ2_0 */
    case 39: b[0] = (uint8_t)(122 + rnd() % 4); break;   /* MXFP4: e8m0 */
    case 40: break;                                /* NVFP4: ue4m3 scales, any byte finite */
    case 29: {   /* IQ1_M: d is the scales' four top nibbles: 0x24n0, n of 0..2 */
        int n1 = (int)(rnd() % 3);
        b[49] = (uint8_t)(b[49] & 0x0F); b[51] = (uint8_t)((b[51] & 0x0F) | (n1 << 4));
        b[53] = (uint8_t)((b[53] & 0x0F) | (4 << 4)); b[55] = (uint8_t)((b[55] & 0x0F) | (2 << 4));
        break; }
    default: F16AT(0); break;
    }
#undef F16AT
}

static bd_u32 *g_tab;
static size_t g_smem;

/* type t at I inputs, O rows: the rows against the CPU, a verify's rows against the rows alone */
static void check(int t, int I, int O, int bench) {
    long rb = ref_row_bytes(t, I);
    int bb = ref_block_bytes(t), be = ref_block_elems(t);
    if (rb <= 0 || I % be) return;
    int rw = (int)((rb + 3) / 4), fmt = 100 + t;
    uint8_t *w = (uint8_t *)calloc((size_t)O, (size_t)rw * 4);
    for (int o = 0; o < O; o++) {
        uint8_t *r = w + (size_t)o * rw * 4;
        for (long i = 0; i < rb; i++) r[i] = (uint8_t)rnd();
        for (long k = 0; k + bb <= rb; k += bb) bd_fix(t, r + k);
    }
    const int SM = 4;
    float *x = (float *)malloc((size_t)SM * I * 4);
    for (int i = 0; i < SM * I; i++) x[i] = ((int)(rnd() % 20001) - 10000) * 1e-4f;
    bd_u32 *wd; float *xd, *yd;
    cudaMalloc(&wd, (size_t)O * rw * 4); cudaMalloc(&xd, (size_t)SM * I * 4); cudaMalloc(&yd, (size_t)SM * O * 4);
    cudaMemcpy(wd, w, (size_t)O * rw * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(xd, x, (size_t)SM * I * 4, cudaMemcpyHostToDevice);
    float *one = (float *)malloc((size_t)SM * O * 4), *got = (float *)malloc((size_t)SM * O * 4);
    int ok = 1;
    for (int s = 0; s < SM; s++) ok &= bd_gemv_launch(yd + (size_t)s * O, xd + (size_t)s * I, wd, g_tab, fmt, I, O, rw, 1, g_smem, 0, 0);
    ok &= cudaDeviceSynchronize() == cudaSuccess;
    cudaMemcpy(one, yd, (size_t)SM * O * 4, cudaMemcpyDeviceToHost);
    if (!ok) { printf("FAIL %s [%d x %d]: the launch failed (%s)\n", ref_name(t), O, I, cudaGetErrorString(cudaGetLastError())); fails++; goto out; }
    {
        /* against the CPU: the integer form's sums from its own codes, or the dequantized row */
        int idot = bdi_has(t);
        int8_t *w8 = (int8_t *)malloc(I), *xq = (int8_t *)malloc(I);
        float *sw = (float *)malloc((size_t)I / 16 * 4), *mw = (float *)malloc((size_t)I / 16 * 4), *dx = (float *)malloc((size_t)I / 32 * 4);
        float *deq = (float *)malloc((size_t)I * 4);
        int32_t *xs = (int32_t *)malloc((size_t)I / 16 * 4);
        double worst = 0;
        for (int s = 0; s < SM && ok; s++) {
            ref_quant(x + (size_t)s * I, I, xq, dx, xs);
            for (int o = 0; o < O; o++) {
                const uint8_t *r = w + (size_t)o * rw * 4;
                double ref = 0, mag = 0;
                if (idot) {
                    if (ref_unpack(t, r, I, w8, sw, mw)) { ok = 0; break; }
                    for (int b = 0; b < I / 32; b++) {
                        double s0 = 0, s1 = 0, a0 = 0, a1 = 0;
                        for (int i = 0; i < 16; i++) { s0 += w8[32 * b + i] * xq[32 * b + i]; a0 += fabs((double)w8[32 * b + i] * xq[32 * b + i]); }
                        for (int i = 16; i < 32; i++) { s1 += w8[32 * b + i] * xq[32 * b + i]; a1 += fabs((double)w8[32 * b + i] * xq[32 * b + i]); }
                        ref += (double)dx[b] * ((double)sw[2 * b] * s0 + (double)sw[2 * b + 1] * s1 - (double)mw[2 * b] * xs[2 * b] - (double)mw[2 * b + 1] * xs[2 * b + 1]);
                        mag += fabs((double)dx[b]) * (fabs((double)sw[2 * b]) * a0 + fabs((double)sw[2 * b + 1]) * a1 +
                                                      fabs((double)mw[2 * b] * xs[2 * b]) + fabs((double)mw[2 * b + 1] * xs[2 * b + 1]));
                    }
                } else {
                    if (ref_dequant(t, r, deq, I)) { ok = 0; break; }
                    for (int i = 0; i < I; i++) { ref += (double)deq[i] * x[(size_t)s * I + i]; mag += fabs((double)deq[i] * x[(size_t)s * I + i]); }
                }
                double e = fabs(one[(size_t)s * O + o] - ref) / (mag + 1e-6);
                if (!(e <= worst)) worst = e;   /* NaN too */
            }
        }
        if (!ok) { printf("FAIL %s: the CPU's decoder refused the row\n", ref_name(t)); fails++; }
        else if (!(worst <= 2e-5)) { printf("FAIL %s [%d x %d]: %s off by %.3g of the terms' magnitude\n", ref_name(t), O, I, idot ? "integer form" : "float form", worst); fails++; }
        else printf("%-8s [%6d x %5d] %s: within %.2g of the CPU's\n", ref_name(t), O, I, idot ? "integer form" : "float form  ", worst);
        free(w8); free(xq); free(sw); free(mw); free(dx); free(deq); free(xs);
    }
    /* the backend's upload (the file's rows, unpadded) and the chain's matmul: the kernel's bits */
    if (ok) {
        uint8_t *packed = (uint8_t *)malloc((size_t)O * rb);
        for (int o = 0; o < O; o++) memcpy(packed + (size_t)o * rb, w + (size_t)o * rw * 4, (size_t)rb);
        ColiCudaTensor *ct = NULL;
        int okb = coli_cuda_tensor_upload(&ct, packed, NULL, fmt, I, O, 0) && coli_cuda_block_elems(fmt) == be;
        cudaMemset(yd, 0, (size_t)SM * O * 4);
        for (int s = 0; s < SM && okb; s++) okb = coli_cuda_tensor_gemm_async(ct, yd + (size_t)s * O, xd + (size_t)s * I, 1, NULL);
        okb = okb && cudaDeviceSynchronize() == cudaSuccess;
        cudaMemcpy(got, yd, (size_t)SM * O * 4, cudaMemcpyDeviceToHost);
        if (!okb || memcmp(got, one, (size_t)SM * O * 4)) { printf("FAIL %s [%d x %d]: the backend's upload and matmul differ from the kernel's\n", ref_name(t), O, I); fails++; }
        if (ct) coli_cuda_tensor_free(ct);
        free(packed);
    }
    /* a verify's rows together: the rows alone's bits */
    for (int S = 2; S <= SM && ok; S++) {
        cudaMemset(yd, 0, (size_t)SM * O * 4);
        int okl = bd_gemv_launch(yd, xd, wd, g_tab, fmt, I, O, rw, S, g_smem, 0, 0) && cudaDeviceSynchronize() == cudaSuccess;
        cudaMemcpy(got, yd, (size_t)S * O * 4, cudaMemcpyDeviceToHost);
        if (!okl || memcmp(got, one, (size_t)S * O * 4)) { printf("FAIL %s [%d x %d]: %d rows differ from the rows alone\n", ref_name(t), O, I, S); fails++; }
    }
    if (bench) {
        printf("  %-8s [%6d x %5d] %6.1f MB:", ref_name(t), O, I, (double)O * rw * 4 / 1e6);
        for (int S = 1; S <= 4; S++) {
            double best = 1e9;
            for (int k = 0; k < 5; k++) {
                cudaDeviceSynchronize(); double t0 = now();
                for (int j = 0; j < 20; j++) bd_gemv_launch(yd, xd, wd, g_tab, fmt, I, O, rw, S, g_smem, 0, 0);
                cudaDeviceSynchronize(); double dt = (now() - t0) / 20;
                if (dt < best) best = dt;
            }
            printf(" %d: %.1f us (%.0f GB/s)", S, best * 1e6, (double)O * rw * 4 / best / 1e9);
        }
        printf("\n");
    }
out:
    cudaFree(wd); cudaFree(xd); cudaFree(yd);
    free(w); free(x); free(one); free(got);
}

int main(void) {
    int devs[1] = {0};
    if (!coli_cuda_init(devs, 1)) { printf("SKIP: no CUDA device\n"); return 0; }
    int bench = getenv("BENCH") && *getenv("BENCH") == '1';
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    g_smem = optin > 48 * 1024 ? (size_t)optin : 48 * 1024;
    uint32_t *tab = (uint32_t *)malloc((size_t)BDT_WORDS * 4);
    bdtab_fill(tab, NULL);
    cudaMalloc(&g_tab, (size_t)BDT_WORDS * 4);
    cudaMemcpy(g_tab, tab, (size_t)BDT_WORDS * 4, cudaMemcpyHostToDevice);
    free(tab);
    const int types[] = {8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 42, 2, 3, 6, 7, 34, 35, 39, 40, 41};
    const int shapes[][2] = {{2048, 300}, {4096, 70}, {768, 1000}, {12288, 40}, {2816, 33}};
    for (size_t i = 0; i < sizeof types / sizeof *types; i++)
        for (size_t k = 0; k < sizeof shapes / sizeof *shapes; k++) check(types[i], shapes[k][0], shapes[k][1], 0);
    if (bench) {
        printf("BENCH (rows of x: 1 2 3 4)\n");
        const int bt[] = {12, 14, 8, 17, 22, 16, 21, 23, 11, 10};
        const int bs[][2] = {{2048, 248320}, {2048, 8192}, {8192, 2048}, {2048, 512}};
        for (size_t i = 0; i < sizeof bt / sizeof *bt; i++)
            for (size_t k = 0; k < sizeof bs / sizeof *bs; k++) check(bt[i], bs[k][0], bs[k][1], 1);
    }
    cudaFree(g_tab);
    printf(fails ? "FAILED: %d\n" : "OK\n", fails);
    return fails != 0;
}
