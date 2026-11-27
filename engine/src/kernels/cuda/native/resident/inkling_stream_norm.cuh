// The block RMS norm of the depth stream's gate halves (`norm_one` of inkling_stream.cu), shared
// with the fused decode controller launch (FIFI_INKLING_CUDA_DECODE_GLUE). Every addition keeps
// the host's element order: 256 thread-strided partials, then thread 0 adds them in lane order.
// A change of an order here is a change of the gate bits of every visit.
#ifndef INFERENCE_CUDA_RESIDENT_INKLING_STREAM_NORM_CUH
#define INFERENCE_CUDA_RESIDENT_INKLING_STREAM_NORM_CUH

#include <cuda_runtime.h>

// The lanes whose partials `block_sum` adds; a block of the norm has exactly this many threads.
constexpr int kNormThreads = 256;

// Sum of the 256 per-thread partials: thread 0 adds them in lane order.
__device__ __forceinline__ float block_sum(float value, float *scratch) {
    const int lane = threadIdx.x;
    scratch[lane] = value;
    __syncthreads();
    if (lane == 0) {
        float total = 0.0f;
        for (int i = 0; i < kNormThreads; ++i) total = __fadd_rn(total, scratch[i]);
        scratch[0] = total;
    }
    __syncthreads();
    const float total = scratch[0];
    __syncthreads();
    return total;
}

// Rust rms_norm_with_scale: scale = (dot(x, x) / len + eps).sqrt().recip().
__device__ __forceinline__ float rms_scale(const float *row, int width, float eps, float *scratch,
                                           int &invalid) {
    const int lane = threadIdx.x;
    float squares = 0.0f;
    for (int i = lane; i < width; i += kNormThreads) {
        const float value = row[i];
        invalid |= !isfinite(value);
        squares = __fadd_rn(squares, __fmul_rn(value, value));
    }
    const float total = block_sum(squares, scratch);
    __shared__ float shared;
    if (lane == 0) {
        const float mean = __fdiv_rn(total, static_cast<float>(width));
        const float root = __fsqrt_rn(__fadd_rn(mean, eps));
        const float scale = __frcp_rn(root);
        invalid |= !isfinite(total) || !isfinite(root) || !isfinite(scale);
        shared = scale;
    }
    __syncthreads();
    const float scale = shared;
    __syncthreads();
    return scale;
}

#endif /* INFERENCE_CUDA_RESIDENT_INKLING_STREAM_NORM_CUH */
