/* Entry of scatter_cols on an explicit stream (contract: prefill/scatter_cols.h). */
#include "prefill/scatter_cols.h"

#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"

namespace {

constexpr int kThreads = 128;  // float4 columns per block
constexpr int kMaxRows = 32768;
constexpr int kMaxWidth = 65536;

bool aligned16(const void *p) { return ((uintptr_t)p & 15) == 0; }

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return (uintptr_t)a + a_bytes <= (uintptr_t)b || (uintptr_t)b + b_bytes <= (uintptr_t)a;
}

bool valid_shape(const inference_cuda_scatter_cols_args *a) {
    const bool rows = a->rows >= 1 && a->rows <= kMaxRows;
    const bool width = a->n >= 4 && a->n <= kMaxWidth && a->n % 4 == 0;
    const bool cols = a->col_off >= 0 && a->col_off % 4 == 0;
    const bool ld = a->ld % 4 == 0 && (long long)a->ld >= (long long)a->col_off + a->n;
    return rows && width && cols && ld;
}

/* Valid after valid_shape, which bounds every factor, so the 64-bit products cannot wrap. */
size_t src_bytes(const inference_cuda_scatter_cols_args *a) {
    return (size_t)a->rows * (size_t)a->n * 4;
}

size_t dst_bytes(const inference_cuda_scatter_cols_args *a) {
    return ((size_t)(a->rows - 1) * (size_t)a->ld + (size_t)a->col_off + (size_t)a->n) * 4;
}

bool valid_scatter(const inference_cuda_scatter_cols_args *a) {
    if (a == NULL || a->src == NULL || a->dst == NULL || !aligned16(a->src) ||
        !aligned16(a->dst) || !valid_shape(a)) {
        return false;
    }
    return apart(a->dst, dst_bytes(a), a->src, src_bytes(a));
}

// Block (x, r): thread i copies float4 column c4 = x * kThreads + i of row r.
__global__ void scatter_cols_k(const float *__restrict__ src, float *__restrict__ dst, int n4,
                               int n, int ld, int col_off) {
    const int c4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (c4 >= n4) return;
    const float4 v = reinterpret_cast<const float4 *>(src + (int64_t)blockIdx.y * n)[c4];
    reinterpret_cast<float4 *>(dst + (int64_t)blockIdx.y * ld + col_off)[c4] = v;
}

}  // namespace

extern "C" int inference_cuda_scatter_cols_launch(const inference_cuda_scatter_cols_args *args,
                                                  void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (stream == NULL || !valid_scatter(args)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const int n4 = args->n / 4;
        const dim3 grid((n4 + kThreads - 1) / kThreads, args->rows);
        scatter_cols_k<<<grid, kThreads, 0, (cudaStream_t)stream>>>(
            (const float *)args->src, (float *)args->dst, n4, args->n, args->ld, args->col_off);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
