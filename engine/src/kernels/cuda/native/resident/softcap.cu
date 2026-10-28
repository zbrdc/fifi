/* Entry of softcap on an explicit stream. Glue only, like the projector adapter. */
#include <cuda_runtime.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/softcap.h"

namespace {

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }
bool apart(const void *a, size_t ab, const void *b, size_t bb) {
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? ab <= y - x : bb <= x - y;
}

__global__ void softcap_kernel(float *out, const float *x, float cap, long long total) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    out[i] = cap * tanhf(x[i] / cap);
}

}  // namespace

extern "C" int inference_cuda_softcap_launch(const inference_cuda_softcap_args *args,
                                             void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->out == NULL || args->x == NULL) return INFERENCE_CUDA_INVALID_ARGUMENT;
        // NaN fails the first comparison and infinity fails the second.
        if (!(args->cap > 0.0f) || !isfinite(args->cap))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->n < 32 || args->n > 4194304 || args->n % 32 != 0)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (!aligned(args->out, 4) || !aligned(args->x, 4))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const long long total = (long long)args->n;
        const size_t bytes = (size_t)total * sizeof(float);
        if (!apart(args->out, bytes, args->x, bytes)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const int block = 256;
        const unsigned grid = (unsigned)((total + block - 1) / block);
        Failure failure;
        softcap_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
            (float *)args->out, (const float *)args->x, args->cap, total);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
