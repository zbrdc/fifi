/* Shared F32 row operations. Launch-only on an explicit non-NULL stream.
 * All pointers are four-byte aligned and sized for the active contiguous extents.
 * Output and flag share no bytes with each other or any input; read-only inputs
 * may overlap. Caller initializes flag once and retains owners until completion.
 * Numeric failures write zero and OR mask1; invalid gather IDs write a zero row
 * and OR PREFILL_STATUS_BOUND_VIOLATION. Neither operation clears earlier bits.
 * No BF16 conversion occurs here: use the shared row_scale publication entry.
 */
#ifndef INFERENCE_CUDA_PREFILL_ROW_OPS_H
#define INFERENCE_CUDA_PREFILL_ROW_OPS_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_row_unary_args {
    const float *x;
    size_t x_bytes;
    float *out;
    size_t out_bytes;
    unsigned *flag;
    size_t flag_bytes;
    size_t rows;
    size_t columns;
} inference_cuda_row_unary_args;

typedef struct inference_cuda_row_mul_args {
    const float *x;
    size_t x_bytes;
    const float *rhs;
    size_t rhs_bytes;
    float *out;
    size_t out_bytes;
    unsigned *flag;
    size_t flag_bytes;
    size_t rows;
    size_t columns;
    size_t rhs_columns;  /* exactly1 (per-row broadcast) or columns */
} inference_cuda_row_mul_args;

typedef struct inference_cuda_row_gather_args {
    const float *x;
    size_t x_bytes;
    const int32_t *indices;
    size_t indices_bytes;
    float *out;
    size_t out_bytes;
    unsigned *flag;
    size_t flag_bytes;
    size_t rows;
    size_t columns;
    size_t source_rows;
} inference_cuda_row_gather_args;

typedef struct inference_cuda_row_rank_sum_args {
    const float *x;  /* [rows][rank][columns], original selected-rank order */
    size_t x_bytes;
    float *out;     /* [rows][columns] */
    size_t out_bytes;
    unsigned *flag;
    size_t flag_bytes;
    size_t rows;
    size_t columns;
    size_t rank;    /* 1..32; ONE F32 sum, no intermediate BF16 rounding */
} inference_cuda_row_rank_sum_args;

int inference_cuda_row_sigmoid_launch(const inference_cuda_row_unary_args *args,
                                      void *stream, int *out_stage);
int inference_cuda_row_mul_launch(const inference_cuda_row_mul_args *args,
                                  void *stream, int *out_stage);
int inference_cuda_row_gather_launch(const inference_cuda_row_gather_args *args,
                                     void *stream, int *out_stage);
int inference_cuda_row_rank_sum_launch(const inference_cuda_row_rank_sum_args *args,
                                       void *stream, int *out_stage);

#ifdef __cplusplus
}
#define ROW_OPS_ASSERT static_assert
#else
#define ROW_OPS_ASSERT _Static_assert
#endif
ROW_OPS_ASSERT(sizeof(inference_cuda_row_unary_args) == 64 &&
               offsetof(inference_cuda_row_unary_args, flag) == 32 &&
               offsetof(inference_cuda_row_unary_args, columns) == 56, "row unary layout");
ROW_OPS_ASSERT(sizeof(inference_cuda_row_mul_args) == 88 &&
               offsetof(inference_cuda_row_mul_args, flag) == 48 &&
               offsetof(inference_cuda_row_mul_args, rhs_columns) == 80, "row multiply layout");
ROW_OPS_ASSERT(sizeof(inference_cuda_row_gather_args) == 88 &&
               offsetof(inference_cuda_row_gather_args, indices) == 16 &&
               offsetof(inference_cuda_row_gather_args, source_rows) == 80, "row gather layout");
ROW_OPS_ASSERT(sizeof(inference_cuda_row_rank_sum_args) == 72 &&
               offsetof(inference_cuda_row_rank_sum_args, rank) == 64, "row rank sum layout");
#undef ROW_OPS_ASSERT
#endif
