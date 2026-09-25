/* Drives ggml_quants.h for tests/test_ggml_quants.py: reads
 *   int32 type, int64 n_elements, then the raw row bytes
 * from stdin and writes n_elements float32 to stdout (gq_dequant_row). */
#include <stdio.h>
#include <stdlib.h>
#include "../ggml_quants.h"

int main(void) {
    int32_t t; int64_t n;
    if (fread(&t, 4, 1, stdin) != 1 || fread(&n, 8, 1, stdin) != 1) return 2;
    int64_t rb = gq_row_bytes(t, n);
    if (rb < 0) { fprintf(stderr, "bad type/len %d %lld\n", t, (long long)n); return 3; }
    uint8_t *src = malloc((size_t)rb); float *dst = malloc((size_t)n * 4);
    if (!src || !dst || fread(src, 1, (size_t)rb, stdin) != (size_t)rb) return 4;
    if (gq_dequant_row(t, src, dst, n)) { fprintf(stderr, "unsupported %s\n", gq_type_name(t)); return 5; }
    fwrite(dst, 4, (size_t)n, stdout);
    free(src); free(dst);
    return 0;
}
