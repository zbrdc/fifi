// Row reductions of the resident Inkling kernels in one order, for prompt rows and decode rows
// alike. A block of `kLaneThreads` lanes reduces a row: lane k adds elements k, k + kLaneThreads,
// ... in element order from +0.0, each warp folds its lanes by halving shuffles, and lane 0 adds
// the warp sums in warp order. A kernel that reduces a row this way gives the bits of any other
// that does, whatever else it fuses, so a prompt row and a decode row of the same inputs agree.
// Every arithmetic intrinsic is _rn, so no build flag can contract a multiply and an add.
#ifndef INFERENCE_CUDA_RESIDENT_LANE_SUM_CUH
#define INFERENCE_CUDA_RESIDENT_LANE_SUM_CUH

#include <math.h>

constexpr int kLaneThreads = 256;
constexpr int kLaneWarps = kLaneThreads / 32;
// Loads a lane issues ahead of its dependent adds: one block cannot hide a memory latency behind
// its threads. A lane still adds the same elements in the same order.
constexpr int kLaneLoads = 9;

// The sum of a warp's values by halving shuffles; lane 0 of the warp holds it.
static __device__ __forceinline__ float lane_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

// The block's sum of one partial per lane, on every lane. `scratch` holds kLaneWarps floats.
// Every lane of the block calls it; it synchronizes the block.
static __device__ __forceinline__ float lane_block_sum(float value, float *scratch) {
    const int lane = threadIdx.x;
    value = lane_warp_sum(value);
    if ((lane & 31) == 0) scratch[lane >> 5] = value;
    __syncthreads();
    float total = scratch[0];
#pragma unroll
    for (int w = 1; w < kLaneWarps; ++w) total = __fadd_rn(total, scratch[w]);
    __syncthreads();
    return total;
}

// The squares of `x` summed in the lane order, on every lane; a lane that loads a value that is
// not finite sets `invalid`.
static __device__ __forceinline__ float lane_squares(const float *x, int width, float *scratch,
                                                     int &invalid) {
    const int lane = threadIdx.x;
    float squares = 0.0f;
    int i = lane;
    for (; i + kLaneThreads * (kLaneLoads - 1) < width; i += kLaneThreads * kLaneLoads) {
        float value[kLaneLoads];
#pragma unroll
        for (int u = 0; u < kLaneLoads; ++u) value[u] = x[i + kLaneThreads * u];
#pragma unroll
        for (int u = 0; u < kLaneLoads; ++u) {
            invalid |= !isfinite(value[u]);
            squares = __fadd_rn(squares, __fmul_rn(value[u], value[u]));
        }
    }
    for (; i < width; i += kLaneThreads) {
        const float value = x[i];
        invalid |= !isfinite(value);
        squares = __fadd_rn(squares, __fmul_rn(value, value));
    }
    return lane_block_sum(squares, scratch);
}

// The host's `rms_norm_with_scale` scale, `(dot(x, x) / len + eps).sqrt().recip()`, with the dot
// in the lane order, on every lane. A scale that is not finite sets `invalid`.
static __device__ __forceinline__ float lane_rms_scale(const float *x, int width, float eps,
                                                       float *scratch, int &invalid) {
    const float total = lane_squares(x, width, scratch, invalid);
    const float root = __fsqrt_rn(__fadd_rn(__fdiv_rn(total, static_cast<float>(width)), eps));
    const float scale = __frcp_rn(root);
    invalid |= !isfinite(total) || !isfinite(root) || !isfinite(scale);
    return scale;
}

#endif  // INFERENCE_CUDA_RESIDENT_LANE_SUM_CUH
