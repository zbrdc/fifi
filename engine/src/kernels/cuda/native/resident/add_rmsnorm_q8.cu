/*
 * The add_rmsnorm_q8 kernel and its launch entry. The kernel restates the math of the vendored
 * add_rmsnorm_pack_k (xavier-qwen ops.cu, MIT) for the f32 norm row — the same strided sum of
 * squares, the same block reduction order, the same rsqrtf and the same expression
 * row[i] * rstd * w[i], so norm_out is byte-identical to what add_rmsnorm_pack_k writes on the
 * same input — and quantizes that normalized row to
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

#include "quant_blocks.h"
#include "resident/q8_1_store.cuh"

namespace {

constexpr int kThreads = 256;  // add_rmsnorm_pack_k's block size (ops.cu)
// The opt-in block: a quarter of the strided rounds per row, another float order.
constexpr int kWideThreads = 1024;
constexpr int kMaxWidth = 32768;
constexpr int kMaxRows = 262144;  // one block per row; far past any head or batch loop
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
 * d = 0, q = 0, s = 0). One block of 256 threads per row: the grid's x dimension is the row
 * count, every block strides h (and delta), norm_out and q8 by one row, and the rows share the
 * weight — a whole per-head or per-batch norm loop collapses into one launch. With 1024
 * threads the same block code runs a quarter of the strided rounds per row: the q8 blocks are
 * independent, but the sum of squares adds in another float order, so the last bits differ. */
__global__ void add_rmsnorm_q8_k(float *base_h, const float *base_delta, const float *w, int n,
                                 float eps, float *base_norm_out, block_q8_1 *base_y, int64_t h_stride) {
  const int64_t padded =
      ((int64_t)n + kMatrixRowPadding - 1) / kMatrixRowPadding * kMatrixRowPadding;
  const int64_t r = blockIdx.x;
  float *h = base_h + r * h_stride;
  const float *delta = base_delta != NULL ? base_delta + r * n : NULL;
  float *norm_out = base_norm_out + r * n;
  block_q8_1 *y = base_y + r * (padded / kQk8_1);
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
  const int nwarps = (int)blockDim.x / 32;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (int64_t b = warp; b < padded / kQk8_1; b += nwarps) {
    const int i = (int)(b * kQk8_1) + lane;
    const float xi = i < n ? h[i] * rstd * w[i] : 0.0f;
    if (i < n) norm_out[i] = xi;
    store_q8_1_block(&y[b], xi);
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
        const int64_t rows = args->rows;
        if (args->h == NULL || args->w == NULL || args->norm_out == NULL || args->q8 == NULL ||
            n < 32 || n > kMaxWidth || n % 32 != 0 || !std::isfinite(args->eps) ||
            args->eps <= 0.0f || rows < 1 || rows > kMaxRows || !aligned(args->h, 4) ||
            (args->delta != NULL && !aligned(args->delta, 4)) || !aligned(args->w, 4) ||
            !aligned(args->norm_out, 16) || !aligned(args->q8, 4) || args->h_stride < 0 ||
            (args->h_stride != 0 && args->h_stride < n) ||
            (args->threads != 0 && args->threads != kThreads && args->threads != kWideThreads)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        // The kernel writes h, norm_out and q8 (row-strided over `rows` blocks) while it reads
        // delta (row-strided) and w (one shared row); the guard needs the byte counts
        // (pad(n, 512)/32 blocks of 36 bytes) and must not wrap.
        const int64_t padded = ((int64_t)n + kMatrixRowPadding - 1) / kMatrixRowPadding * kMatrixRowPadding;
        const int64_t row = (int64_t)n * (int64_t)sizeof(float);
        const int64_t q8_bytes = padded / kQk8_1 * (int64_t)sizeof(block_q8_1);
        const int64_t h_stride = args->h_stride == 0 ? n : args->h_stride;
        const Extent written[3] = {{args->h, (size_t)((rows - 1) * h_stride * (int64_t)sizeof(float) + row)},
                                   {args->norm_out, (size_t)(row * rows)},
                                   {args->q8, (size_t)(q8_bytes * rows)}};
        const Extent read[2] = {{args->delta, (size_t)(row * rows)}, {args->w, (size_t)row}};
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
        add_rmsnorm_q8_k<<<dim3((unsigned)rows), dim3((unsigned)(args->threads ? args->threads : kThreads)), 0, (cudaStream_t)stream>>>(
            (float *)args->h, (const float *)args->delta, (const float *)args->w, n, args->eps,
            (float *)args->norm_out, (block_q8_1 *)args->q8, h_stride);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
