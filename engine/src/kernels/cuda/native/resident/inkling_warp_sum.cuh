// The 32-lane xor butterfly reduction of dense_gemv.cu (`warp_sum`), shared with the fused
// decode controller launch (FIFI_INKLING_CUDA_DECODE_GLUE), whose down reduction must keep the
// bits of `decode_controller_down_exact`.
#ifndef INFERENCE_CUDA_RESIDENT_INKLING_WARP_SUM_CUH
#define INFERENCE_CUDA_RESIDENT_INKLING_WARP_SUM_CUH

#include <cuda_runtime.h>

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

#endif /* INFERENCE_CUDA_RESIDENT_INKLING_WARP_SUM_CUH */
