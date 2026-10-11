// MIT License, Copyright (c) 2026 Dan Campos (see ../LICENSE).
// The matmul input layout (xf) shared by kernels.cu and ops.cu. write_xf and pack_half2x4 moved
// here unchanged from kernels.cu by RD-4V so that there is one copy of the fragment-order write.
// The anonymous namespace keeps the device code in each translation unit, as it was in kernels.cu.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>

#include "mma884.cuh"

namespace kern {
namespace {

// Four fp16 pairs as one uint4. Built from 32-bit words: a __half2[4] local
// array is only 4-byte aligned, so reading it as a uint4 is a misaligned
// 16-byte local access (undefined behaviour).
__device__ inline uint4 pack_half2x4(__half2 a, __half2 b, __half2 c, __half2 d) {
  unsigned w[4];
  memcpy(&w[0], &a, sizeof(unsigned));
  memcpy(&w[1], &b, sizeof(unsigned));
  memcpy(&w[2], &c, sizeof(unsigned));
  memcpy(&w[3], &d, sizeof(unsigned));
  return make_uint4(w[0], w[1], w[2], w[3]);
}

// Writes token tok's k values f(0..k-1) into the matmul input xf. Up to
// kMaxTokens tokens use fragment order (the layout mma884::pack_x produces):
// one uint4 per (tile, wi, quadpair) holding k base..base+3 and base+32..base+35.
// Larger batches (rowmajor) use fp16 [token][k], which the gemm path reads.
template <class F>
__device__ void write_xf(uint4* xf, int k, int tok, bool rowmajor, F f) {
  int nb = k / mma884::kTileK, t = tok / 8, r = tok % 8;
  for (int e = threadIdx.x; e < k / 8; e += blockDim.x) {
    if (rowmajor) {
      xf[(size_t)tok * (k / 8) + e] =
          pack_half2x4(__floats2half2_rn(f(8 * e), f(8 * e + 1)),
                       __floats2half2_rn(f(8 * e + 2), f(8 * e + 3)),
                       __floats2half2_rn(f(8 * e + 4), f(8 * e + 5)),
                       __floats2half2_rn(f(8 * e + 6), f(8 * e + 7)));
      continue;
    }
    int ib = e / 32, q = (e / 8) % 4, wi = e % 8;
    int base = ib * mma884::kTileK + 64 * q + 4 * wi;
    int lane = (r & 3) + 4 * q + 16 * (r >> 2);
    xf[((size_t)(t * nb + ib) * 8 + wi) * 32 + lane] =
        pack_half2x4(__floats2half2_rn(f(base), f(base + 1)),
                     __floats2half2_rn(f(base + 2), f(base + 3)),
                     __floats2half2_rn(f(base + 32), f(base + 33)),
                     __floats2half2_rn(f(base + 34), f(base + 35)));
  }
}

}  // namespace
}  // namespace kern
