/*
 * The Q5_0 decode matmul (MEL-C1b). One warp per 8-row group; the arithmetic is the vendored one
 * (mma884.cuh, unmodified): exact integer dequantization through the 0x6400 exponent trick with one
 * fp16 multiply by d, and fp32 accumulation in m8n8k4. Only the loop is new, because the tile is
 * 8 x 128 (MEL-C1a) and the vendored loop reads one 8 x 256 tile per step.
 *
 * A logical tile is 256 k = two weight tiles A = 2L and B = 2L + 1 of the group. Quadpair q takes x
 * chunk 64q..64q+63 of the logical tile: the two consecutive blocks 2 (q & 1) and 2 (q & 1) + 1 of
 * weight tile A (q < 2) or B (q >= 2). When the row has an odd number of weight tiles (K = 896 has 7)
 * the last logical tile has only A: the lanes of quadpairs 2 and 3 load nothing, feed a zero A
 * fragment and read the shared zero x entry, so the xf content beyond k is never part of a sum.
 * Every exported function has C linkage, takes and returns C types only, and wraps its body in
 * try/catch, like matmul_expert.cu.
 */
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "kernels.h"
#include "mma884.cuh"
#include "resident/matmul_q5_0.h"

namespace q5_0 {
namespace {

const int kTileCols = 128;       // k values of one weight tile
const int kTileVecs = 44;        // uint4 of one 8 x 128 tile (704 bytes)
const int kMaxDenseTokens = 16;  // two token groups
const int kMaxExpertTokens = 8;

// The raw bytes of one lane for one logical tile: the d, qh and qs of its two blocks.
struct Raw {
  uint32_t d;   // fp16 d of block jb (low half) and block jb + 1 (high half)
  uint2 qh;     // qh of block jb (.x) and block jb + 1 (.y)
  uint4 qs[2];  // the 16 qs bytes of block jb and of block jb + 1
};

// b0 is the tile block index 4 * row + jb (MEL-C1a: b = 4 * row + j); jb is even, so every load
// below is aligned to its width: d at tile byte 2 b0, qh at 64 + 4 b0, qs at 192 + 16 b0.
__device__ inline Raw load_raw(const uint4* tile, int b0) {
  const unsigned char* bytes = reinterpret_cast<const unsigned char*>(tile);
  Raw r;
  r.d = mma884::ld_stream(reinterpret_cast<const uint32_t*>(bytes) + (b0 >> 1));
  r.qh = __ldcs(reinterpret_cast<const uint2*>(bytes + 64) + (b0 >> 1));
  r.qs[0] = mma884::ld_stream(tile + 12 + b0);
  r.qs[1] = mma884::ld_stream(tile + 13 + b0);
  return r;
}

// Loads the lane's bytes of weight tile `tile` of the group, or zeros when the row has no such tile.
__device__ inline Raw load_tile(const uint4* tiles, int tile, int nt, int b0) {
  if (tile >= nt) {
    Raw zero = {};
    return zero;
  }
  return load_raw(tiles + (size_t)tile * kTileVecs, b0);
}

// Bit i of t (i < 4) to bit 8 i. The partial products t << (7 j) never collide in the mask: bit i
// of t lands on bit i + 7 j, which is a multiple of 8 only for j == i.
__device__ inline uint32_t spread4(uint32_t t) { return (t * 0x00204081u) & 0x01010101u; }

// fp16 A fragments of word wi: lo is k 64q + 4wi.., hi is k 64q + 32 + 4wi.. of the lane's row. A
// block holds elements 0..15 in the low nibbles and 16..31 in the high nibbles of qs[0..15]; bit e
// of qh is the fifth bit of element e. Each byte is q5 in 0..31, and bytes_to_half4 with bias
// 1040 = 1024 + 16 gives fp16((q5 - 16) * d) with one rounding.
__device__ inline void dq(const Raw& r, __half2 s_lo, __half2 s_hi, int wi, uint2& lo, uint2& hi) {
  const __half2 bias = __float2half2_rn(1040.f);
  const int shift = 4 * (wi >> 2);
  const uint32_t qa = (mma884::component(r.qs[0], wi & 3) >> shift) & 0x0F0F0F0Fu;
  const uint32_t qb = (mma884::component(r.qs[1], wi & 3) >> shift) & 0x0F0F0F0Fu;
  const uint32_t ha = spread4((r.qh.x >> (4 * wi)) & 0xFu) << 4;
  const uint32_t hb = spread4((r.qh.y >> (4 * wi)) & 0xFu) << 4;
  lo = mma884::bytes_to_half4(qa | ha, bias, s_lo);
  hi = mma884::bytes_to_half4(qb | hb, bias, s_hi);
}

// Dense (EXPERT false) or expert (EXPERT true: ids, slot and count are read). TG token groups of 8.
// No split-K: the sum order is fixed, which the expert form requires.
template <int TG, bool EXPERT>
__global__ void __launch_bounds__(mma884::kWarps * 32)
    matmul_q5_0_k(const uint4* __restrict__ w, const int* __restrict__ ids, int slot, int count,
                  const uint4* __restrict__ xf, float* __restrict__ y, int ldy, int n_rows, int k,
                  int m) {
  using namespace mma884;
  const int lane = threadIdx.x & 31;
  const int g = blockIdx.x * kWarps + threadIdx.x / 32;
  if (g >= n_rows / 8) return;
  const int nt = k / kTileCols;
  const int nl = (nt + 1) / 2;
  size_t base = 0;
  if (EXPERT) {
    const int id = __ldg(ids + slot);
    if ((unsigned)id >= (unsigned)count) {
      float bad_acc[TG][8];
#pragma unroll
      for (int t = 0; t < TG; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) bad_acc[t][i] = __int_as_float(0x7fc00000);
      }
      store<TG, false>(bad_acc, y, g, ldy, m, lane);
      return;
    }
    base = (size_t)id * (size_t)(n_rows / 8) * (size_t)nt * kTileVecs;
  }
  const int q = quadpair(lane);
  const int tsel = q >> 1;
  const int b0 = 4 * frag_row(lane) + 2 * (q & 1);
  const uint4* tiles = w + base + (size_t)g * nt * kTileVecs;

  float acc[TG][8] = {};
  Raw next = load_tile(tiles, tsel, nt, b0);
  for (int L = 0; L < nl; ++L) {
    Raw cur = next;
    if (L + 1 < nl) next = load_tile(tiles, 2 * (L + 1) + tsel, nt, b0);
    const int mx = (2 * L + tsel < nt) ? m : 0;  // a tile-less lane reads the zero x entry
    const __half2 s_lo = __half2half2(__ushort_as_half((unsigned short)(cur.d & 0xFFFF)));
    const __half2 s_hi = __half2half2(__ushort_as_half((unsigned short)(cur.d >> 16)));
#pragma unroll
    for (int wi = 0; wi < 8; ++wi) {
      uint2 lo, hi;
      dq(cur, s_lo, s_hi, wi, lo, hi);
#pragma unroll
      for (int t = 0; t < TG; ++t) {
        const uint4 xv = load_x(xf, t, nl, L, wi, lane, mx);
        mma(acc[t], lo, make_uint2(xv.x, xv.y));
        mma(acc[t], hi, make_uint2(xv.z, xv.w));
      }
    }
  }
  store<TG, false>(acc, y, g, ldy, m, lane);
}

bool aligned(const void* p, size_t alignment) { return (uintptr_t)p % alignment == 0; }

bool apart(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
  return (uintptr_t)a + a_bytes <= (uintptr_t)b || (uintptr_t)b + b_bytes <= (uintptr_t)a;
}

bool shape_ok(int m, int n_rows, int k, bool expert) {
  const int max_m = expert ? kMaxExpertTokens : kMaxDenseTokens;
  return m >= 1 && m <= max_m && n_rows > 0 && n_rows % 8 == 0 && k > 0 && k % kTileCols == 0 &&
         k <= kern::kMaxK;
}

// Bytes of the tiles of one expert: n_rows / 8 groups of k / 128 tiles of 704 bytes.
size_t expert_bytes(int n_rows, int k) {
  return (size_t)(n_rows / 8) * (size_t)(k / kTileCols) * kTileVecs * sizeof(uint4);
}

// The arguments both entries share, checked the same way: pointers, alignment, ldy.
bool operands_ok(const void* weight, const void* xf, const float* y, const void* stream, int ldy,
                 int n_rows) {
  return weight != NULL && xf != NULL && y != NULL && stream != NULL && ldy >= n_rows &&
         aligned(weight, sizeof(uint4)) && aligned(xf, sizeof(uint4)) && aligned(y, sizeof(float));
}

size_t y_bytes_of(int m, int ldy, int n_rows) {
  return (((size_t)(m - 1) * (size_t)ldy) + (size_t)n_rows) * sizeof(float);
}

dim3 grid_of(int n_rows) { return dim3((n_rows / 8 + mma884::kWarps - 1) / mma884::kWarps); }

}  // namespace

cudaError_t matmul(int m, const uint4* weight, const uint4* xf, float* y, int ldy, int n_rows,
                   int k, cudaStream_t stream) {
  const dim3 block(mma884::kWarps * 32);
  if (m <= 8) {
    matmul_q5_0_k<1, false><<<grid_of(n_rows), block, 0, stream>>>(weight, NULL, 0, 0, xf, y, ldy,
                                                                   n_rows, k, m);
  } else {
    matmul_q5_0_k<2, false><<<grid_of(n_rows), block, 0, stream>>>(weight, NULL, 0, 0, xf, y, ldy,
                                                                   n_rows, k, m);
  }
  return cudaGetLastError();
}

namespace {

cudaError_t matmul_expert(int m, const uint4* stack, const int* ids, int slot, int count,
                          const uint4* xf, float* y, int ldy, int n_rows, int k,
                          cudaStream_t stream) {
  const dim3 block(mma884::kWarps * 32);
  matmul_q5_0_k<1, true><<<grid_of(n_rows), block, 0, stream>>>(stack, ids, slot, count, xf, y, ldy,
                                                                n_rows, k, m);
  return cudaGetLastError();
}

}  // namespace
}  // namespace q5_0

extern "C" int inference_cuda_q5_0_shape_ok(int m, int n_rows, int k, int expert) {
  try {
    return q5_0::shape_ok(m, n_rows, k, expert != 0) ? INFERENCE_CUDA_OK
                                                      : INFERENCE_CUDA_INVALID_ARGUMENT;
  } catch (...) {
    return INFERENCE_CUDA_NATIVE_EXCEPTION;
  }
}

extern "C" int inference_cuda_matmul_q5_0_launch(int m, const void* weight, const void* xf_dev,
                                                 float* y_dev, int ldy, int n_rows, int k,
                                                 void* stream, int* out_stage) {
  try {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!q5_0::shape_ok(m, n_rows, k, false) ||
        !q5_0::operands_ok(weight, xf_dev, y_dev, stream, ldy, n_rows)) {
      return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    // y is written while the weight and xf are read.
    const size_t y_bytes = q5_0::y_bytes_of(m, ldy, n_rows);
    const size_t xf_bytes = kern::xf_vecs() * sizeof(uint4);
    if (!q5_0::apart(y_dev, y_bytes, xf_dev, xf_bytes) ||
        !q5_0::apart(y_dev, y_bytes, weight, q5_0::expert_bytes(n_rows, k))) {
      return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    Failure failure;
    failure.record(q5_0::matmul(m, (const uint4*)weight, (const uint4*)xf_dev, y_dev, ldy, n_rows, k,
                                (cudaStream_t)stream),
                   INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
  } catch (...) {
    return INFERENCE_CUDA_NATIVE_EXCEPTION;
  }
}

extern "C" int inference_cuda_matmul_expert_q5_0_launch(int m, const void* stack, const int* ids_dev,
                                                        int slot, int count, const void* xf_dev,
                                                        float* y_dev, int ldy, int n_rows, int k,
                                                        void* stream, int* out_stage) {
  try {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (ids_dev == NULL || !q5_0::aligned(ids_dev, sizeof(int)) || slot < 0 || count < 1 ||
        !q5_0::shape_ok(m, n_rows, k, true) ||
        !q5_0::operands_ok(stack, xf_dev, y_dev, stream, ldy, n_rows)) {
      return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    // count * expert_bytes must not wrap before the overlap proof uses it.
    const size_t one = q5_0::expert_bytes(n_rows, k);
    if (one == 0 || (size_t)count > SIZE_MAX / one) {
      return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const size_t y_bytes = q5_0::y_bytes_of(m, ldy, n_rows);
    const size_t xf_bytes = kern::xf_vecs() * sizeof(uint4);
    const size_t ids_bytes = ((size_t)slot + 1) * sizeof(int);
    if (!q5_0::apart(y_dev, y_bytes, xf_dev, xf_bytes) ||
        !q5_0::apart(y_dev, y_bytes, stack, one * (size_t)count) ||
        !q5_0::apart(y_dev, y_bytes, ids_dev, ids_bytes)) {
      return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    Failure failure;
    failure.record(q5_0::matmul_expert(m, (const uint4*)stack, ids_dev, slot, count,
                                       (const uint4*)xf_dev, y_dev, ldy, n_rows, k,
                                       (cudaStream_t)stream),
                   INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
  } catch (...) {
    return INFERENCE_CUDA_NATIVE_EXCEPTION;
  }
}
