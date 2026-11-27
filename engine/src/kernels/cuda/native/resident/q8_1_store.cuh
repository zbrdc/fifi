/* The block_q8_1 writer shared by the fused producers (add_rmsnorm_q8, swiglu_q8, block_ops):
 * one warp quantizes one 32-value block with the vendored quantize_q8_1 rule, in IEEE
 * arithmetic: d = amax / 127, q = roundf(x / d) (0 when amax is 0), s the warp sum of the
 * inputs. Lane `threadIdx.x % 32` holds element `lane` of the block; a value past the live
 * width is passed as 0 so the padding blocks come out as zeros. The header includes
 * quant_blocks.h, which defines block_q8_1, so the include order is free. */
#ifndef INFERENCE_CUDA_RESIDENT_Q8_1_STORE_CUH
#define INFERENCE_CUDA_RESIDENT_Q8_1_STORE_CUH

#include "quant_blocks.h"

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

#endif /* INFERENCE_CUDA_RESIDENT_Q8_1_STORE_CUH */
