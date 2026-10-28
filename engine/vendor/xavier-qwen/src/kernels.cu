#include "kernels.h"

#include "frag_layout.cuh"
#include "gemm.cuh"
#include "mma884.cuh"

namespace kern {
namespace {

__global__ void pack_f32_k(const float* x, int k, bool rowmajor, uint4* xf) {
  const float* row = x + (size_t)blockIdx.x * k;
  write_xf(xf, k, blockIdx.x, rowmajor, [&](int i) { return row[i]; });
}

// Wide-batch matmul: xf holds fp16 [m][k] (see write_xf). Tile shapes were
// picked by bench/gemm_bench: the CTA count grows with the column tiles, so a
// 32-token tile saves ~35% per matmul at 17..32 tokens (short prompts and
// prefill tails) over the 64-token one.
template <class D>
cudaError_t gemm_launch(int m, const uint4* w, const uint4* xf, float* y, int ldy, int n_rows,
                        int k, cudaStream_t stream) {
  const __half* x = reinterpret_cast<const __half*>(xf);
  if (m <= 32)
    return gemm::launch<D, gemm::Tile<128, 32, 4, 2, 1>>(w, x, y, ldy, n_rows, k, m, stream);
  if (m <= 64)
    return gemm::launch<D, gemm::Tile<128, 64, 4, 2, 1>>(w, x, y, ldy, n_rows, k, m, stream);
  return gemm::launch<D, gemm::Tile<128, 128, 2, 4, 1>>(w, x, y, ldy, n_rows, k, m, stream);
}

}  // namespace

static_assert(mma884::kXfEntries == 2 * (kMaxK / mma884::kTileK) * 8 * 32,
              "xf capacity must match kMaxK");

size_t xf_vecs() { return mma884::kXfEntries + 1; }

cudaError_t matmul(GgmlType type, int m, const uint4* w, const uint4* xf, float* y, int ldy,
                   int n_rows, int k, cudaStream_t stream) {
  bool wide = m > kMaxTokens;
  switch (type) {
    case GgmlType::Q8_0:
      if (wide) return gemm_launch<gemm::DQ8>(m, w, xf, y, ldy, n_rows, k, stream);
      return mma884::launch_packed<mma884::Q8_0>(m, w, xf, y, ldy, n_rows, k, stream);
    case GgmlType::Q4_K:
      if (wide) return gemm_launch<gemm::DQ4K>(m, w, xf, y, ldy, n_rows, k, stream);
      return mma884::launch_packed<mma884::Q4_K>(m, w, xf, y, ldy, n_rows, k, stream);
    case GgmlType::Q6_K:
      if (wide) return gemm_launch<gemm::DQ6K>(m, w, xf, y, ldy, n_rows, k, stream);
      return mma884::launch_packed<mma884::Q6_K>(m, w, xf, y, ldy, n_rows, k, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

cudaError_t pack_f32(const float* x, int m, int k, uint4* xf, cudaStream_t stream) {
  pack_f32_k<<<m, 256, 0, stream>>>(x, k, m > kMaxTokens, xf);
  return cudaGetLastError();
}

cudaError_t matmul_expert(GgmlType type, int m, const uint4* stack, ExpertSelect sel,
                          const uint4* xf, float* y, int ldy, int n_rows, int k,
                          cudaStream_t stream) {
  if (!sel.ids || sel.count <= 0 || sel.slot < 0 || m < 1 || m > 8 || n_rows <= 0 ||
      n_rows % 8 != 0 || k <= 0 || k % mma884::kTileK != 0 || k > kMaxK ||
      mma884::split_k(n_rows / 8, k / mma884::kTileK) != 1)
    return cudaErrorInvalidValue;
  switch (type) {
    case GgmlType::Q8_0:
      return mma884::launch_expert<mma884::Q8_0>(m, stack, sel.ids, sel.slot, sel.count, xf, y,
                                                 ldy, n_rows, k, stream);
    case GgmlType::Q4_K:
      return mma884::launch_expert<mma884::Q4_K>(m, stack, sel.ids, sel.slot, sel.count, xf, y,
                                                 ldy, n_rows, k, stream);
    case GgmlType::Q6_K:
      return mma884::launch_expert<mma884::Q6_K>(m, stack, sel.ids, sel.slot, sel.count, xf, y,
                                                 ldy, n_rows, k, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace kern
