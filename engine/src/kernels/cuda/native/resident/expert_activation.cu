/*
 * Launch entry for the chunk expert stream's device-side activation:
 * relu-squared, the family-specific elementwise op between the grouped up and
 * down projections (the row guard/undo is the shared resident::row_scale).
 * Glue only: C linkage, C types only, try/catch wrapped, like row_scale.cu.
 * Launch-only: nothing allocates, copies or synchronizes; a fault in queued
 * work surfaces at the next synchronize.
 */
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include "../abi.h"
#include "../device.h"
#include "expert_activation.h"

namespace {

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }
bool apart(const void *a, size_t ab, const void *b, size_t bb) {
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? ab <= y - x : bb <= x - y;
}

constexpr unsigned kBlock = 128;

__global__ void relu_squared_kernel(float *out, const float *x, long long total) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        const float r = fmaxf(x[i], 0.0f);
        out[i] = r * r;
    }
}

}  // namespace

extern "C" int inference_cuda_relu_squared_launch(const inference_cuda_element_args *args,
                                                  void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->out == NULL || args->x == NULL) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->values < 1 || args->values > (1LL << 32)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (!aligned(args->out, sizeof(float)) || !aligned(args->x, sizeof(float))) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (!apart(args->out, args->values * 4, args->x, args->values * 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const long long blocks = (args->values + kBlock - 1) / kBlock;
        relu_squared_kernel<<<(unsigned)blocks, kBlock, 0, (cudaStream_t)stream>>>(
            (float *)args->out, (const float *)args->x, args->values);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
