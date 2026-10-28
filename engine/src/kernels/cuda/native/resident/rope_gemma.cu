/* Entry of rope_gemma on an explicit stream. Glue only, like the projector adapter. */
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/rope_gemma.h"

namespace {

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }
bool apart(const void *a, size_t ab, const void *b, size_t bb) {
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? ab <= y - x : bb <= x - y;
}

__global__ void rope_gemma_kernel(
    float *q, float *k, const int *pos, const float *factors,
    int rows_q, int rows_k, int n_q_head, int n_kv_head, int head_dim,
    int global_mode) {
    const int pair = threadIdx.x;      /* 0..head_dim/2-1 */
    const int qh = blockIdx.x;         /* 0..n_q_head-1 */
    const int row = blockIdx.y;        /* 0..max(rows_q,rows_k)-1 */
    const float posf = (float)pos[0] + (float)row;
    float freq = 0.0f;
    if (!global_mode) {
        freq = (float)pow(1.0e4, -(2.0 * (double)pair) / (double)head_dim);
    } else if (pair < 64) {
        freq = (float)pow(1.0e6, -(2.0 * (double)pair) / (double)head_dim);
    }
    if (factors != NULL && freq != 0.0f) {
        const float factor = factors[pair];
        freq = (factor >= 1.0e29f) ? 0.0f : freq / factor;
    }
    const float angle = posf * freq;
    const float c = cosf(angle);
    const float s = sinf(angle);
    const int half = head_dim / 2;
    if (row < rows_q) {
        float *qr = q + ((size_t)row * n_q_head + qh) * head_dim;
        const float a = qr[pair];
        const float b = qr[pair + half];
        qr[pair] = a * c - b * s;
        qr[pair + half] = a * s + b * c;
    }
    if (row < rows_k && qh < n_kv_head) {
        float *kr = k + ((size_t)row * n_kv_head + qh) * head_dim;
        const float a = kr[pair];
        const float b = kr[pair + half];
        kr[pair] = a * c - b * s;
        kr[pair + half] = a * s + b * c;
    }
}

}  // namespace

extern "C" int inference_cuda_rope_gemma_launch(const inference_cuda_rope_gemma_args *args,
                                                void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->q == NULL || args->k == NULL || args->pos == NULL)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->rows_q < 1 || args->rows_q > 16 || args->rows_k < 1 || args->rows_k > 16)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->n_q_head != 16) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->n_kv_head != 1 && args->n_kv_head != 8)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->head_dim != 256 && args->head_dim != 512)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->global_mode != 0 && args->global_mode != 1)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        if ((args->global_mode && args->head_dim != 512) ||
            (!args->global_mode && args->head_dim != 256))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (!aligned(args->q, 4) || !aligned(args->k, 4) || !aligned(args->pos, 4) ||
            (args->factors != NULL && !aligned(args->factors, 4)))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const size_t q_bytes =
            (size_t)args->rows_q * args->n_q_head * args->head_dim * sizeof(float);
        const size_t k_bytes =
            (size_t)args->rows_k * args->n_kv_head * args->head_dim * sizeof(float);
        const size_t f_bytes = (size_t)(args->head_dim / 2) * sizeof(float);
        if (!apart(args->q, q_bytes, args->k, k_bytes) ||
            !apart(args->q, q_bytes, args->pos, 4) ||
            !apart(args->k, k_bytes, args->pos, 4) ||
            (args->factors != NULL &&
             (!apart(args->q, q_bytes, args->factors, f_bytes) ||
              !apart(args->k, k_bytes, args->factors, f_bytes))))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const dim3 grid((unsigned)args->n_q_head,
                        (unsigned)(args->rows_q > args->rows_k ? args->rows_q : args->rows_k));
        Failure failure;
        rope_gemma_kernel<<<grid, args->head_dim / 2, 0, (cudaStream_t)stream>>>(
            (float *)args->q, (float *)args->k, (const int *)args->pos,
            (const float *)args->factors, args->rows_q, args->rows_k, args->n_q_head,
            args->n_kv_head, args->head_dim, args->global_mode);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
