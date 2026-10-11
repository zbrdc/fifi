/*
 * MATVEC-PAR S3a K1: the add_rmsnorm_q8 kernel and its launch entry. The kernel restates the
 * math of the vendored add_rmsnorm_pack_k (xavier-qwen ops.cu, the user's own MIT code) for the
 * f32 norm row — the same strided sum of squares, the same block reduction order, the same
 * rsqrtf and the same expression row[i] * rstd * w[i], so norm_out is byte-identical to what
 * add_rmsnorm_pack_k writes on the same input — and quantizes that normalized row to
 * block_q8_1 with the vendored quantize_q8_1 block rule in IEEE arithmetic (this TU is in the
 * warnings-into-errors group WITHOUT -use_fast_math, unlike the vendored quantize.cu object).
 * Glue only: every exported function has C linkage, takes and returns C types only, and wraps
 * its body in try/catch, like the projector adapter.
 */
#include <cuda_runtime.h>

#include <cmath>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/add_rmsnorm_q8.h"

// The vendored headers keep static host helpers their own objects reference; in this glue TU they
// are unreferenced, which -Wall reports although nothing here is wrong (the quantize_q8_1 glue
// includes the same header the same way).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "quantize.cuh"
#pragma GCC diagnostic pop

namespace {

constexpr int kThreads = 256;  // add_rmsnorm_pack_k's block size (ops.cu)
constexpr int kMaxWidth = 32768;
constexpr int64_t kMatrixRowPadding = 512;  // common.cuh, the MMVQ activation padding
constexpr int64_t kQk8_1 = 32;              // QK8_1

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

/* Sum of v over the block, returned to every thread: the order of kern::block_sum (ops.cu),
 * restated here so the f32 sum of squares is bit-identical to add_rmsnorm_pack_k's. */
__device__ __forceinline__ float block_sum(float v) {
  __shared__ float red[32];
  v = warp_sum(v);
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  __syncthreads();  // earlier callers may still be reading red
  if (lane == 0) red[warp] = v;
  __syncthreads();
  v = lane < (int)blockDim.x / 32 ? red[lane] : 0.f;
  return warp_sum(v);
}

/* The fused producer: h += delta in place (when given), the f32 normalized row into norm_out
 * (byte-identical to add_rmsnorm_pack_k's norm_out), and the block_q8_1 quantization of the same
 * values into q8, one warp per 32-value block over the padded width (the tail past n is zeros:
 * d = 0, q = 0, s = 0). One block of 256 threads, m = 1. */
__global__ void add_rmsnorm_q8_k(float *h, const float *delta, const float *w, int n, float eps,
                                 float *norm_out, block_q8_1 *y) {
  float ss = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = h[i];
    if (delta) {
      v += delta[i];
      h[i] = v;
    }
    ss += v * v;
  }
  float rstd = rsqrtf(block_sum(ss) / n + eps);
  const int64_t padded =
      ((int64_t)n + kMatrixRowPadding - 1) / kMatrixRowPadding * kMatrixRowPadding;
  const int nwarps = (int)blockDim.x / 32;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (int64_t b = warp; b < padded / kQk8_1; b += nwarps) {
    const int i = (int)(b * kQk8_1) + lane;
    const float xi = i < n ? h[i] * rstd * w[i] : 0.0f;
    if (i < n) norm_out[i] = xi;
    float amax = fabsf(xi);
    float sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
      sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    y[b].qs[lane] = q;
    if (lane == 0) y[b].ds = make_half2(d, sum);
  }
}

bool aligned(const void *p, size_t alignment) { return (uintptr_t)p % alignment == 0; }

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return (uintptr_t)a + a_bytes <= (uintptr_t)b || (uintptr_t)b + b_bytes <= (uintptr_t)a;
}

struct Extent {
    const void *p;
    size_t bytes;
};

}  // namespace

extern "C" int inference_cuda_add_rmsnorm_q8_launch(
    const inference_cuda_add_rmsnorm_q8_args *args, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int n = args->n;
        if (args->h == NULL || args->w == NULL || args->norm_out == NULL || args->q8 == NULL ||
            n < 256 || n > kMaxWidth || n % 256 != 0 || !std::isfinite(args->eps) ||
            args->eps <= 0.0f || !aligned(args->h, 4) ||
            (args->delta != NULL && !aligned(args->delta, 4)) || !aligned(args->w, 4) ||
            !aligned(args->norm_out, 16) || !aligned(args->q8, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        // The kernel writes h, norm_out and q8 while it reads delta and w; the guard needs the
        // byte counts (pad(n, 512)/32 blocks of 36 bytes) and must not wrap.
        const int64_t padded = ((int64_t)n + kMatrixRowPadding - 1) / kMatrixRowPadding * kMatrixRowPadding;
        const int64_t row = (int64_t)n * (int64_t)sizeof(float);
        const int64_t q8_bytes = padded / kQk8_1 * (int64_t)sizeof(block_q8_1);
        const Extent written[3] = {{args->h, (size_t)row}, {args->norm_out, (size_t)row},
                                   {args->q8, (size_t)q8_bytes}};
        const Extent read[2] = {{args->delta, (size_t)row}, {args->w, (size_t)row}};
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                if (!apart(written[i].p, written[i].bytes, written[j].p, written[j].bytes)) {
                    return INFERENCE_CUDA_INVALID_ARGUMENT;
                }
            }
            for (int j = 0; j < 2; ++j) {
                if (read[j].p != NULL &&
                    !apart(written[i].p, written[i].bytes, read[j].p, read[j].bytes)) {
                    return INFERENCE_CUDA_INVALID_ARGUMENT;
                }
            }
        }
        add_rmsnorm_q8_k<<<1, kThreads, 0, (cudaStream_t)stream>>>(
            (float *)args->h, (const float *)args->delta, (const float *)args->w, n, args->eps,
            (float *)args->norm_out, (block_q8_1 *)args->q8);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
