// Golden vectors for the lapid-quant i-quants (IQ2_XXS, IQ2_S, IQ3_XXS), generated once
// from ggml's reference implementation. Not part of the cargo build; see gen.sh.
//
// Writes into the output directory:
//   input.bin    u32 rows | u32 cols | f32 x[rows*cols] | f32 imatrix[cols]
//   grids.bin    iq2xxs_grid (u64 x256) | iq2s_grid (u64 x1024) | iq3xxs_grid (u32 x256) | ksigns (u8 x128)
//   <fmt>.bin    quantize(x, imatrix) | dequantize(that) as f32
//                [| quantize(x, NULL) | dequantize(that)]   (iq2_s, iq3_xxs: ggml default weighting)
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// stubs for the few ggml.c symbols ggml-quants.c references
void ggml_abort(const char * file, int line, const char * fmt, ...) { fprintf(stderr, "abort %s:%d %s\n", file, line, fmt); abort(); }
size_t ggml_row_size(enum ggml_type t, int64_t n) { (void)t; (void)n; abort(); }
const char * ggml_type_name(enum ggml_type t) { (void)t; return "?"; }
size_t ggml_type_size(enum ggml_type t) { (void)t; abort(); }

void iq2xs_init_impl(enum ggml_type type);
void iq3xs_init_impl(int grid_size);

static uint64_t s = 0x9E3779B97F4A7C15ull;
static float uni(void) {  // [-1, 1)
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return (float)(s >> 40) / 16777216.0f * 2.0f - 1.0f;
}

enum { ROWS = 8, COLS = 512 };

static FILE * out(const char * dir, const char * name) {
    char p[4096];
    snprintf(p, sizeof p, "%s/%s", dir, name);
    FILE * f = fopen(p, "wb");
    if (!f) { perror(p); exit(1); }
    return f;
}

typedef size_t (*qfn)(const float *, void *, int64_t, int64_t, const float *);
typedef void (*dfn)(const void *, float *, int64_t);

static void emit(FILE * f, qfn q, dfn d, size_t bb, const float * x, const float * w) {
    size_t n = (size_t)ROWS * COLS / 256 * bb;
    uint8_t * buf = calloc(n, 1);
    float * y = malloc(sizeof(float) * ROWS * COLS);
    q(x, buf, ROWS, COLS, w);
    d(buf, y, (int64_t)ROWS * COLS);
    fwrite(buf, 1, n, f);
    fwrite(y, sizeof(float), ROWS * COLS, f);
    free(buf); free(y);
}

int main(int argc, char ** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <out dir>\n", argv[0]); return 1; }
    const char * dir = argv[1];
    static float x[ROWS * COLS], w[COLS];
    for (int r = 0; r < ROWS; ++r) {
        float scale = 0.002f + 0.03f * (float)r;
        for (int c = 0; c < COLS; ++c) {
            float g = uni() + uni() + uni() + uni();  // roughly gaussian
            x[r * COLS + c] = g * scale;
        }
    }
    for (int c = 0; c < 256; ++c) x[0 * COLS + 256 + c] = 0.0f;           // zero block
    for (int c = 0; c < 256; ++c) x[1 * COLS + c] *= 1e-9f;               // tiny block
    x[2 * COLS + 17] = 3.0f;                                              // outlier
    for (int c = 0; c < 32; ++c) x[3 * COLS + 64 + c] = 0.0f;             // zero 32-group
    for (int c = 0; c < 256; ++c) x[4 * COLS + 256 + c] = 0.05f;          // constant block
    for (int c = 0; c < 256; ++c) x[5 * COLS + c] = (c % 3 == 0) ? -0.02f * (float)(c % 7) : 0.01f * (float)(c % 5);
    for (int c = 0; c < COLS; ++c) w[c] = 0.05f + (uni() + 1.0f) * (uni() + 1.0f) * 2.0f;
    w[5] = 40.0f;

    FILE * f = out(dir, "input.bin");
    uint32_t hdr[2] = {ROWS, COLS};
    fwrite(hdr, 4, 2, f);
    fwrite(x, 4, ROWS * COLS, f);
    fwrite(w, 4, COLS, f);
    fclose(f);

    f = out(dir, "grids.bin");
    fwrite(iq2xxs_grid, 8, 256, f);
    fwrite(iq2s_grid, 8, 1024, f);
    fwrite(iq3xxs_grid, 4, 256, f);
    fwrite(ksigns_iq2xs, 1, 128, f);
    fclose(f);

    iq2xs_init_impl(GGML_TYPE_IQ2_XXS);
    iq2xs_init_impl(GGML_TYPE_IQ2_S);
    iq3xs_init_impl(256);

    f = out(dir, "iq2_xxs.bin");
    emit(f, quantize_iq2_xxs, (dfn)dequantize_row_iq2_xxs, sizeof(block_iq2_xxs), x, w);
    fclose(f);
    f = out(dir, "iq2_s.bin");
    emit(f, quantize_iq2_s, (dfn)dequantize_row_iq2_s, sizeof(block_iq2_s), x, w);
    emit(f, quantize_iq2_s, (dfn)dequantize_row_iq2_s, sizeof(block_iq2_s), x, NULL);
    fclose(f);
    f = out(dir, "iq3_xxs.bin");
    emit(f, quantize_iq3_xxs, (dfn)dequantize_row_iq3_xxs, sizeof(block_iq3_xxs), x, w);
    emit(f, quantize_iq3_xxs, (dfn)dequantize_row_iq3_xxs, sizeof(block_iq3_xxs), x, NULL);
    fclose(f);
    return 0;
}
