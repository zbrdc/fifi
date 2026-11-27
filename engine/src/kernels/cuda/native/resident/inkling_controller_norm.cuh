// One controller role's weighted norm (`controller_norm_one` of inkling_controller.cu), shared
// with the fused decode controller launch (FIFI_INKLING_CUDA_DECODE_GLUE): the block's tree
// reduction of the squares and the order of every rounded operation are the host's bits.
#ifndef INFERENCE_CUDA_RESIDENT_INKLING_CONTROLLER_NORM_CUH
#define INFERENCE_CUDA_RESIDENT_INKLING_CONTROLLER_NORM_CUH

#include <cuda_runtime.h>
#include <math.h>

#include "resident/inkling_stream_norm.cuh"

// `out = rmsnorm((input - tanh(forget) * input) + tanh(gain) * message) * gamma`, the host's
// controller step, every operation rounded; a block of kNormThreads threads. `health` takes a
// quiet NaN when a value was not finite, and is left alone otherwise.
__device__ __forceinline__ void controller_norm_one(const float *input, const float *message,
                                                     const float *gamma, const float *gain,
                                                     const float *forget, float *out,
                                                     float *health, int width, float eps) {
    const int lane = threadIdx.x;

    __shared__ float sums[kNormThreads];
    float squares = 0.0f;
    bool invalid = false;
    for (int i = lane; i < width; i += kNormThreads) {
        const float kept = __fsub_rn(input[i], __fmul_rn(tanhf(forget[i]), input[i]));
        const float x = __fadd_rn(kept, __fmul_rn(tanhf(gain[i]), message[i]));
        invalid |= !isfinite(gain[i]) || !isfinite(forget[i]) || !isfinite(x);
        out[i] = x;
        squares += x * x;
    }
    sums[lane] = squares;
    __syncthreads();
    for (int stride = kNormThreads / 2; stride != 0; stride /= 2) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        __syncthreads();
    }
    const float variance = sums[0] / width + eps;
    const float scale = 1.0f / sqrtf(variance);
    invalid |= !isfinite(variance) || !isfinite(scale);
    for (int i = lane; i < width; i += kNormThreads) {
        out[i] = __fmul_rn(__fmul_rn(out[i], scale), gamma[i]);
        invalid |= !isfinite(out[i]);
    }
    const int failed = __syncthreads_or(invalid);
    if (lane == 0) health[0] = failed ? __int_as_float(0x7fc00000) : 0.0f;
}

#endif /* INFERENCE_CUDA_RESIDENT_INKLING_CONTROLLER_NORM_CUH */
