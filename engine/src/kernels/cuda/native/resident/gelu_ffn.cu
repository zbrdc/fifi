/* Entry of gelu_ffn on an explicit stream. Glue only, like the projector adapter. */
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/gelu_ffn.h"

namespace {

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }
bool apart(const void *a, size_t ab, const void *b, size_t bb) {
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? ab <= y - x : bb <= x - y;
}

__global__ void gelu_ffn_kernel(float *out, const float *gate, const float *up, long long total) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float g = gate[i];
    const float cube = g * g * g;
    const float inner = 0.7978845608028654f * (g + 0.044715f * cube);
    out[i] = 0.5f * g * (1.0f + tanhf(inner)) * up[i];
}

}  // namespace

extern "C" int inference_cuda_gelu_ffn_launch(const inference_cuda_gelu_ffn_args *args,
                                              void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->out == NULL || args->gate == NULL || args->up == NULL)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->m < 1 || args->m > 65535 || args->n < 32 || args->n > 65536 ||
            args->n % 32 != 0)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (!aligned(args->out, 4) || !aligned(args->gate, 4) || !aligned(args->up, 4))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const long long total = (long long)args->m * args->n;
        const size_t bytes = (size_t)total * sizeof(float);
        if (!apart(args->out, bytes, args->gate, bytes) ||
            !apart(args->out, bytes, args->up, bytes))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const int block = 256;
        const unsigned grid = (unsigned)((total + block - 1) / block);
        Failure failure;
        gelu_ffn_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
            (float *)args->out, (const float *)args->gate, (const float *)args->up, total);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
