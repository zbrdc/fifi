/*
 * MATVEC-PAR S3a K2: the swiglu_q8 kernel and its launch entry. One kernel batches the SwiGLU of
 * the expert slots and the q8_1 quantization of its result: block s handles slot s (grid =
 * slots), reads gate at columns 0..n_ff and up at n_ff..2*n_ff of the slot's row of gu (the A2
 * stride layout), computes silu(gate) * up with swiglu_pack_k's fast form (x / (1 + __expf(-x)),
 * kept by decision; the E2 domain of the swiglu_pack doc), and quantizes per 32 values with the
 * vendored quantize_q8_1 block rule in IEEE arithmetic (this TU is in the warnings-into-errors
 * group WITHOUT -use_fast_math, unlike the vendored quantize.cu object), one warp per block over
 * the padded width whose tail is written as zeros. Glue only: every exported function has C
 * linkage, takes and returns C types only, and wraps its body in try/catch, like the projector
 * adapter.
 */
#include <cuda_runtime.h>

#include <cmath>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/swiglu_q8.h"

// The vendored headers keep static host helpers their own objects reference; in this glue TU they
// are unreferenced, which -Wall reports although nothing here is wrong (the quantize_q8_1 glue
// includes the same header the same way).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "quantize.cuh"
#pragma GCC diagnostic pop

namespace {

constexpr int kThreads = 256;              // swiglu_pack_k's block size (ops.cu)
constexpr int kMaxWidth = 32768;
constexpr int64_t kMatrixRowPadding = 512; // common.cuh, the MMVQ activation padding
constexpr int64_t kQk8_1 = 32;             // QK8_1

/* silu with the fast __expf of swiglu_pack_k (ops.cu), kept by decision: the E2 numerics
 * statement of the swiglu_pack doc is the domain, and the tests state the tolerance. */
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

/* The batched producer: one block per slot, one warp per 32-value block of the padded width. */
__global__ void swiglu_q8_k(const float *gu, int n_ff, int64_t slot_stride, block_q8_1 *y) {
  const float *g = gu + (int64_t)blockIdx.x * slot_stride;
  const int64_t padded =
      ((int64_t)n_ff + kMatrixRowPadding - 1) / kMatrixRowPadding * kMatrixRowPadding;
  // Slot s writes its own padded row of padded / 32 blocks: without the offset every slot wrote
  // the first row (the first 4050 run: slot 0 changed with slot 3, replays differed).
  block_q8_1 *row = y + (int64_t)blockIdx.x * (padded / kQk8_1);
  const int nwarps = (int)blockDim.x / 32;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (int64_t b = warp; b < padded / kQk8_1; b += nwarps) {
    const int i = (int)(b * kQk8_1) + lane;
    const float xi = i < n_ff ? silu(g[i]) * g[n_ff + i] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
      sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    row[b].qs[lane] = q;
    if (lane == 0) row[b].ds = make_half2(d, sum);
  }
}

bool aligned(const void *p, size_t alignment) { return (uintptr_t)p % alignment == 0; }

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return (uintptr_t)a + a_bytes <= (uintptr_t)b || (uintptr_t)b + b_bytes <= (uintptr_t)a;
}

}  // namespace

extern "C" int inference_cuda_swiglu_q8_launch(const inference_cuda_swiglu_q8_args *args,
                                               void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || args->gu == NULL || args->q8 == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int n_ff = args->n_ff;
        const int slots = args->slots;
        const int64_t stride = args->y_slot_stride;
        if (n_ff < 32 || n_ff > kMaxWidth || n_ff % 32 != 0 || slots < 1 || slots > 8 ||
            stride < 2 * (int64_t)n_ff || !aligned(args->gu, 4) || !aligned(args->q8, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        // The kernel reads the A2 footprint ((slots - 1) * stride + 2 * n_ff floats) and writes
        // the padded block rows; both counts must not wrap.
        const int64_t gu_floats = (int64_t)(slots - 1) * stride + 2 * (int64_t)n_ff;
        const int64_t padded =
            ((int64_t)n_ff + kMatrixRowPadding - 1) / kMatrixRowPadding * kMatrixRowPadding;
        const int64_t q8_bytes = padded / kQk8_1 * (int64_t)sizeof(block_q8_1) * slots;
        if (gu_floats > INT64_MAX / (int64_t)sizeof(float) ||
            !apart(args->q8, (size_t)q8_bytes, args->gu,
                   (size_t)(gu_floats * (int64_t)sizeof(float)))) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        swiglu_q8_k<<<slots, kThreads, 0, (cudaStream_t)stream>>>(
            (const float *)args->gu, n_ff, stride, (block_q8_1 *)args->q8);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
