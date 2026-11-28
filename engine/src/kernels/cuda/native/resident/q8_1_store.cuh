/* The block_q8_1 writer shared by the fused producers (add_rmsnorm_q8, swiglu_q8, block_ops):
 * one warp quantizes one 32-value block with the vendored quantize_q8_1 rule, in IEEE
 * arithmetic: d = amax / 127, q = roundf(x / d) (0 when amax is 0), s the warp sum of the
 * inputs. Lane `threadIdx.x % 32` holds element `lane` of the block; a value past the live
 * width is passed as 0 so the padding blocks come out as zeros. The header includes
 * quant_blocks.h, which defines block_q8_1, so the include order is free. */
#ifndef INFERENCE_CUDA_RESIDENT_Q8_1_STORE_CUH
#define INFERENCE_CUDA_RESIDENT_Q8_1_STORE_CUH

#ifndef QK8_1
#include "quant_blocks.h"
#endif

__device__ __forceinline__ void store_q8_1_block(block_q8_1 *y, float xi) {
  const int lane = threadIdx.x % 32;
  float amax = fabsf(xi);
  float sum = xi;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    sum += __shfl_xor_sync(0xffffffffu, sum, o);
  }
  const float d = amax / 127.0f;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
  y->qs[lane] = q;
  if (lane == 0) y->ds = make_half2(d, sum);
}

/* Scale-consistent, rounded-sum block_q8_1 writer: rounds d to fp16 before quantizing q against
 * 1 / __half2float(d_half), and stores d * sum(q_i) in ds.y (llama.cpp#31247). */
__device__ __forceinline__ void store_q8_1_block_rounded(block_q8_1 *y, float xi) {
  const int lane = threadIdx.x % 32;
  float amax = fabsf(xi);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  }
  const __half d_half = __float2half(amax / 127.0f);
  const float d = __half2float(d_half);
  const int q_raw = d == 0.0f ? 0 : __float2int_rn(xi / d);
  const int q_clamped = q_raw < -127 ? -127 : (q_raw > 127 ? 127 : q_raw);
  y->qs[lane] = (int8_t)q_clamped;
  int qsum = q_clamped;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    qsum += __shfl_xor_sync(0xffffffffu, qsum, o);
  }
  if (lane == 0) y->ds = __halves2half2(d_half, __float2half(d * (float)qsum));
}

template <bool Rounded>
__device__ __forceinline__ void store_q8_1_block_mode(block_q8_1 *y, float xi) {
  if constexpr (Rounded) {
    store_q8_1_block_rounded(y, xi);
  } else {
    store_q8_1_block(y, xi);
  }
}

#endif /* INFERENCE_CUDA_RESIDENT_Q8_1_STORE_CUH */
