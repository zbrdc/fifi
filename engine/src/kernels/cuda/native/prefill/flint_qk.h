/* Private D128 Q/K preparation. F32 inputs/outputs; BF16 publication is separate.
 * Explicit stream, launch-only. Caller retains all owners and checks sticky flag8.
 * Rows1..4096, Q32/K4; separate128-value gammas. Read-only regions may overlap;
 * outputs/flag must be disjoint from all other active regions.
 */
#ifndef INFERENCE_CUDA_PREFILL_FLINT_QK_H
#define INFERENCE_CUDA_PREFILL_FLINT_QK_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_flint_qk_args {
    const float *q;
    size_t q_bytes;
    const float *k;
    size_t k_bytes;
    const float *gamma_q;
    size_t gamma_q_bytes;
    const float *gamma_k;
    size_t gamma_k_bytes;
    float *out_q;
    size_t out_q_bytes;
    float *out_k;
    size_t out_k_bytes;
    unsigned *flag;
    size_t flag_bytes;
    size_t rows;
    uint32_t position;
    uint32_t reserved;
} inference_cuda_flint_qk_args;

int inference_cuda_flint_qk_norm(const inference_cuda_flint_qk_args *args,
                                 void *stream, int *out_stage);
int inference_cuda_flint_qk_rope(const inference_cuda_flint_qk_args *args,
                                 void *stream, int *out_stage);

#ifdef __cplusplus
}
#define QK_ASSERT static_assert
#else
#define QK_ASSERT _Static_assert
#endif
QK_ASSERT(sizeof(inference_cuda_flint_qk_args) == 128 &&
          offsetof(inference_cuda_flint_qk_args, gamma_q) == 32 &&
          offsetof(inference_cuda_flint_qk_args, out_q) == 64 &&
          offsetof(inference_cuda_flint_qk_args, flag) == 96 &&
          offsetof(inference_cuda_flint_qk_args, rows) == 112 &&
          offsetof(inference_cuda_flint_qk_args, position) == 120 &&
          offsetof(inference_cuda_flint_qk_args, reserved) == 124, "Q/K layout");
#undef QK_ASSERT
#endif
