// Engine kernels for Qwen3.8 (qwen35 arch) on sm_72. Activations are f32 in
// [token][feature] rows; matmul inputs go through the fp16 buffer xf: fragment
// order (what mma884 reads) for m <= kMaxTokens, row-major [m][k] (what the
// gemm path reads) for larger m. Every entry takes a stream; nullptr is the default stream.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "repack.h"

namespace kern {

constexpr int kMaxTokens = 16;  // tokens per forward (mma884 limit)
constexpr int kMaxK = 32768;    // matmul input width (fragment-order xf capacity)

// uint4s of a fragment-order xf buffer; the last one must stay zero (padding
// tokens read it). A row-major buffer needs m * k / 8.
size_t xf_vecs();

// y[t * ldy + r] = (x W^T)[t][r] for tiled weights w of the given ggml type.
// m > kMaxTokens runs the tensor-core gemm (xf row-major); results differ from
// the decode kernels only by fp32 summation order. Returns cudaErrorInvalidValue
// for an untiled type or a bad shape, else the memset, attribute or launch status.
cudaError_t matmul(GgmlType type, int m, const uint4* w, const uint4* xf, float* y, int ldy,
                   int n_rows, int k, cudaStream_t stream = nullptr);

// xf = x for x [m][k]; returns the launch status.
cudaError_t pack_f32(const float* x, int m, int k, uint4* xf, cudaStream_t stream = nullptr);

// Picks one expert of a stack on the device: the expert is ids[slot], valid in [0, count).
struct ExpertSelect {
  const int* ids;
  int slot;
  int count;
};

// y[t * ldy + r] is the product with expert ids[slot] of `stack`, the tiled weight of `count`
// experts of n_rows x k each, back to back. Same layout and results as matmul on that expert's
// tiles, but the id is read by the kernel, so a captured graph needs no host-known id. m <= 8
// and split_k == 1 only (decode shapes); an id outside [0, count) writes NaN to y and reads no
// weight. Returns cudaErrorInvalidValue without a launch for a bad shape, a null ids, an
// untiled type, or a shape that matmul would split along k.
cudaError_t matmul_expert(GgmlType type, int m, const uint4* stack, ExpertSelect sel,
                          const uint4* xf, float* y, int ldy, int n_rows, int k,
                          cudaStream_t stream = nullptr);

}  // namespace kern
