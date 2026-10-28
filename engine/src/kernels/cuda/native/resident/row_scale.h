/*
 * Launch entry of row_scale for the Gemma 4 Humanizer resident step: one kernel that
 * multiplies m rows of n f32 values by a constant. Used once per step for the tied
 * embedding row times sqrt(hidden). Launch-only: nothing here allocates, copies or
 * synchronizes; a fault in queued work surfaces at the next synchronize. Per-kernel
 * header of the layout in kernels/cuda/resident/mod.rs; abi_resident.h stays closed.
 */
#ifndef INFERENCE_CUDA_RESIDENT_ROW_SCALE_H
#define INFERENCE_CUDA_RESIDENT_ROW_SCALE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_row_scale_args {
    void *out;        /* f32 [m][n], written */
    const void *x;    /* f32 [m][n], read */
    float scale;      /* finite multiplier */
    int m;            /* rows, 1..65536 */
    int n;            /* values per row, 1..1<<22 */
} inference_cuda_row_scale_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_row_scale_args) == 32 && offsetof(inference_cuda_row_scale_args, out) == 0 && offsetof(inference_cuda_row_scale_args, x) == 8 && offsetof(inference_cuda_row_scale_args, scale) == 16 && offsetof(inference_cuda_row_scale_args, m) == 20 && offsetof(inference_cuda_row_scale_args, n) == 24, "layout");
#else
_Static_assert(sizeof(inference_cuda_row_scale_args) == 32 && offsetof(inference_cuda_row_scale_args, out) == 0 && offsetof(inference_cuda_row_scale_args, x) == 8 && offsetof(inference_cuda_row_scale_args, scale) == 16 && offsetof(inference_cuda_row_scale_args, m) == 20 && offsetof(inference_cuda_row_scale_args, n) == 24, "layout");
#endif

/* One launch on stream (must not be NULL: no default-stream fallback). out and x hold
 * m * n floats each; out is written and x read. Returns INFERENCE_CUDA_INVALID_ARGUMENT
 * before any CUDA call for a bad argument, else the launch status. */
int inference_cuda_row_scale_launch(const inference_cuda_row_scale_args *args, void *stream,
                                    int *out_stage);

/* Shared F32 working rows, independent of weight format. Mode 0 normalizes each
 * row to maxabs <= 1024 with factor 2^s (0 <= s <= 118); mode 1 restores that scale.
 * Nonfinite input rows are zeroed in mode 0. Mode 1 zeroes invalid-factor rows or
 * nonfinite products. Either failure sets flag |= 1; no operation clears it.
 * Caller zeroes flag once and retains all owners through stream completion, then
 * checks flag before publishing. Power-of-two scaling is not universal bit identity:
 * tiny mixed-magnitude values can underflow; later FP16 operands remain approximate. */
typedef struct inference_cuda_row_pow2_args {
    const float *x;
    size_t x_bytes;
    float *y;
    size_t y_bytes;
    float *scales;
    size_t scales_bytes;
    unsigned *flag;
    size_t flag_bytes;
    size_t rows;
    size_t columns;
    int mode;
    int reserved;
} inference_cuda_row_pow2_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_row_pow2_args) == 88 &&
              offsetof(inference_cuda_row_pow2_args, mode) == 80 &&
              offsetof(inference_cuda_row_pow2_args, reserved) == 84, "row pow2 layout");
#endif

/* Launches one CTA per row, sets flag |= 1 for invalid data and never clears it.
 * All four addressed regions must be sized, aligned and disjoint; no synchronization. */
int inference_cuda_row_pow2_launch(const inference_cuda_row_pow2_args *args, void *stream,
                                  int *out_stage);

/* RN-even BF16 publication, widened to F32 working storage, never through F16.
 * Per-element nonfinite results (including finite-F32 overflow) become zero and
 * set flag |= 1; earlier bits are never cleared. Caller initializes flag once.
 * Stream must be explicit/non-NULL; x/y/flag regions must be sized, four-byte
 * aligned and pairwise disjoint. Retain all owners and check flag after completion. */
typedef struct inference_cuda_bf16_publish_args {
    const float *x;
    size_t x_bytes;
    float *y;
    size_t y_bytes;
    unsigned *flag;
    size_t flag_bytes;
    size_t rows;
    size_t columns;
} inference_cuda_bf16_publish_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_bf16_publish_args) == 64 &&
              offsetof(inference_cuda_bf16_publish_args, flag) == 32 &&
              offsetof(inference_cuda_bf16_publish_args, columns) == 56, "BF16 publication layout");
#else
_Static_assert(sizeof(inference_cuda_bf16_publish_args) == 64 &&
               offsetof(inference_cuda_bf16_publish_args, flag) == 32 &&
               offsetof(inference_cuda_bf16_publish_args, columns) == 56, "BF16 publication layout");
#endif

int inference_cuda_bf16_publish_rows(const inference_cuda_bf16_publish_args *args,
                                   void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_ROW_SCALE_H */
