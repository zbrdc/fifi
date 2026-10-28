// Shared pieces of the register-direct Volta tensor-core matmuls.
//
// mma.m8n8k4 runs as four independent 8x8x4 products, one per quadpair
// (lanes 4q..4q+3 and 16+4q..16+4q+3). Weight kernels for every quant type
// work on 256-k tiles of 8 rows: quadpair q takes k chunk 64q..64q+63 of the
// tile, and each lane's A fragment is four consecutive k of one row. All of
// them read x in the same fragment order, so its producer writes it once.
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

namespace mma884 {
constexpr int kTileK = 256;
// Fragment-ordered x capacity: 2 token groups at K <= 32768. The entry at
// index kXfEntries is kept zero for padding-token lanes.
constexpr int kXfEntries = 2 * (32768 / kTileK) * 8 * 32;

__device__ __host__ inline int quadpair(int lane) { return (lane >> 2) & 3; }
// Fragment row of A, and fragment column of B, held by this lane.
__device__ __host__ inline int frag_row(int lane) { return (lane & 3) + 4 * (lane >> 4); }

__device__ inline void mma(float (&c)[8], uint2 a, uint2 b) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]),
        "+f"(c[7])
      : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
}

// x in B-fragment order: entry ((t * nb + ib) * 8 + wi) * 32 + lane holds the
// lane's x for word wi of tile ib — k 64q + 4wi in .x/.y, k 64q + 32 + 4wi in
// .z/.w — so every fragment load is one coalesced 512-byte warp access.
// Tokens >= m are zero. In the engine the producer of x (rmsnorm) would write
// this layout.
static __global__ void pack_x(const __half* __restrict__ x, uint4* __restrict__ xf, int k, int m,
                       int n_entries) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_entries) return;
  int lane = i & 31, wi = (i >> 5) & 7, rest = i >> 8;
  int nb = k / kTileK, ib = rest % nb, t = rest / nb;
  int tok = frag_row(lane) + 8 * t;
  uint4 v = make_uint4(0, 0, 0, 0);
  if (tok < m) {
    const __half* p = x + (size_t)tok * k + ib * kTileK + 64 * quadpair(lane) + 4 * wi;
    uint2 lo = __ldg(reinterpret_cast<const uint2*>(p));
    uint2 hi = __ldg(reinterpret_cast<const uint2*>(p + 32));
    v = make_uint4(lo.x, lo.y, hi.x, hi.y);
  }
  xf[i] = v;
}

// Lane's x fragment for word wi of tile ib, token group t. Lanes whose B
// column is a padding token all read the shared zero entry (one broadcast
// sector), so at M=1 only 8 of 32 lanes pull real x traffic. Selecting the
// address instead of predicating the load keeps the caller branch-free.
__device__ inline uint4 load_x(const uint4* xf, int t, int nb, int ib, int wi, int lane, int m) {
  bool ok = frag_row(lane) + 8 * t < m;
  const uint4* p = ok ? xf + (((size_t)t * nb + ib) * 8 + wi) * 32 + lane : xf + kXfEntries;
  return __ldg(p);
}

// Weight bytes are read once, so keep them out of L1 where the x fragments
// every warp rereads need to stay.
__device__ inline uint4 ld_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ inline uint32_t ld_stream(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

__device__ inline uint32_t component(const uint4& v, int i) {
  switch (i) {
    case 0: return v.x;
    case 1: return v.y;
    case 2: return v.z;
    default: return v.w;
  }
}

// Adds the four quadpairs' partial sums for (row, token) and writes rows
// 8*g..8*g+7 of y (token stride ldy); atomically when K is split across blocks.
// Atomic order follows block scheduling, so split-K results can differ between
// runs in the last fp32 bits; greedy output changes only at exact near-ties.
template <int TG, bool SPLIT>
__device__ inline void store(const float (&acc)[TG][8], float* y, int g, int ldy, int m,
                             int lane) {
#pragma unroll
  for (int t = 0; t < TG; ++t) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float v = acc[t][i];
      v += __shfl_xor_sync(0xffffffffu, v, 4);
      v += __shfl_xor_sync(0xffffffffu, v, 8);
      int row = (lane & 1) + (i & 2) + 4 * (lane >> 4);
      int tok = (i & 4) + (lane & 2) + (i & 1) + 8 * t;
      if ((lane & 12) != 0 || tok >= m) continue;
      float* out = y + (size_t)tok * ldy + g * 8 + row;
      if (SPLIT)
        atomicAdd(out, v);
      else
        *out = v;
    }
  }
}

// K split so short-N matrices still fill every SM. 4096 warps measured best
// (ffn_down M=8: 63 -> 76 GB/s vs 2048).
inline int split_k(int warps, int nb) {
  int split = 1;
  while (warps * split < 4096 && nb / (split * 2) >= 8) split *= 2;
  return split;
}

// Packs x (m tokens x k, row-major fp16) on `stream` into a process-wide fragment-order buffer
// whose last entry stays zero, or returns nullptr when setup fails. Bench helper, not capture-safe.
inline const uint4* prepare_x(const __half* x, int k, int m, cudaStream_t stream = nullptr) {
  static uint4* xf = nullptr;
  if (!xf) {
    if (cudaMalloc(&xf, (kXfEntries + 1) * sizeof(uint4)) != cudaSuccess) {
      xf = nullptr;
      return nullptr;
    }
    if (cudaMemsetAsync(xf + kXfEntries, 0, sizeof(uint4), stream) != cudaSuccess) {
      cudaFree(xf);
      xf = nullptr;
      return nullptr;
    }
  }
  int entries = (m <= 8 ? 1 : 2) * (k / kTileK) * 8 * 32;
  pack_x<<<(entries + 255) / 256, 256, 0, stream>>>(x, xf, k, m, entries);
  return xf;
}

// Four bytes of b -> four fp16 (byte + 1024 - bias) * s.
// Placing a byte under exponent byte 0x64 gives the half 1024 + byte, so subtracting
// 1024 + bias is exact and s is the only rounding step. Fusing the two into
// an FMA would round 1024 * s first and lose the low bits.
__device__ inline uint2 bytes_to_half4(uint32_t b, __half2 bias, __half2 s) {
  uint32_t lo = __byte_perm(b, 0x64646464u, 0x4140);
  uint32_t hi = __byte_perm(b, 0x64646464u, 0x4342);
  __half2 a = __hmul2(__hsub2(*reinterpret_cast<__half2*>(&lo), bias), s);
  __half2 c = __hmul2(__hsub2(*reinterpret_cast<__half2*>(&hi), bias), s);
  return make_uint2(*reinterpret_cast<uint32_t*>(&a), *reinterpret_cast<uint32_t*>(&c));
}

// y[m x n_rows] = x[m x k] * W^T for one weight format F, which supplies:
//   kTileVecs                 uint4s per 8-row x 256-k tile
//   Raw load(tile, lane)      the lane's raw bytes of one tile
//   Scales scales(raw, lane)  per-tile dequantization constants
//   void dq(raw, sc, wi, lo, hi)  fp16 A fragments for word wi: k 64q + 4wi
//                             (lo) and k 64q + 32 + 4wi (hi) of the lane's row
// Each warp owns one 8-row group and double-buffers its tiles in registers.
constexpr int kWarps = 4;

template <class F, int TG, bool SPLIT>
__global__ void __launch_bounds__(kWarps * 32)
    matmul(const uint4* __restrict__ w, const uint4* __restrict__ xf, float* __restrict__ y,
           int ldy, int n_rows, int k, int m) {
  int lane = threadIdx.x & 31;
  int g = blockIdx.x * kWarps + threadIdx.x / 32;
  if (g >= n_rows / 8) return;
  int nb = k / kTileK;
  int ib0 = SPLIT ? nb * blockIdx.y / gridDim.y : 0;
  int ib1 = SPLIT ? nb * (blockIdx.y + 1) / gridDim.y : nb;

  float acc[TG][8] = {};
  const uint4* tiles = w + (size_t)g * nb * F::kTileVecs;
  typename F::Raw next = F::load(tiles + (size_t)ib0 * F::kTileVecs, lane);
  for (int ib = ib0; ib < ib1; ++ib) {
    typename F::Raw cur = next;
    if (ib + 1 < ib1) next = F::load(tiles + (size_t)(ib + 1) * F::kTileVecs, lane);
    typename F::Scales sc = F::scales(cur, lane);
#pragma unroll
    for (int wi = 0; wi < 8; ++wi) {
      uint2 lo, hi;
      F::dq(cur, sc, wi, lo, hi);
#pragma unroll
      for (int t = 0; t < TG; ++t) {
        uint4 xv = load_x(xf, t, nb, ib, wi, lane, m);
        mma(acc[t], lo, make_uint2(xv.x, xv.y));
        mma(acc[t], hi, make_uint2(xv.z, xv.w));
      }
    }
  }
  store<TG, SPLIT>(acc, y, g, ldy, m, lane);
}

// y[t * ldy + r] = (x W^T)[t][r] for packed tiles w and fragment-order xf of
// m <= 16 tokens; on `stream` (nullptr: default). Returns the memset or launch status.
template <class F>
cudaError_t launch_packed(int m, const uint4* w, const uint4* xf, float* y, int ldy,
                          int n_rows, int k, cudaStream_t stream = nullptr) {
  int warps = n_rows / 8;
  int split = split_k(warps, k / kTileK);
  if (split > 1) {
    cudaError_t err = cudaMemset2DAsync(y, sizeof(float) * ldy, 0, sizeof(float) * n_rows, m, stream);
    if (err != cudaSuccess) return err;
  }
  dim3 grid((warps + kWarps - 1) / kWarps, split), block(kWarps * 32);
  bool one = m <= 8;
  if (one && split == 1)
    matmul<F, 1, false><<<grid, block, 0, stream>>>(w, xf, y, ldy, n_rows, k, m);
  else if (one)
    matmul<F, 1, true><<<grid, block, 0, stream>>>(w, xf, y, ldy, n_rows, k, m);
  else if (split == 1)
    matmul<F, 2, false><<<grid, block, 0, stream>>>(w, xf, y, ldy, n_rows, k, m);
  else
    matmul<F, 2, true><<<grid, block, 0, stream>>>(w, xf, y, ldy, n_rows, k, m);
  return cudaGetLastError();
}

// Bench entry: packs row-major x, then y is [m][n_rows].
template <class F>
cudaError_t launch(int m, const uint4* w, const __half* x, float* y, int n_rows, int k,
                   cudaStream_t stream = nullptr) {
  const uint4* xf = prepare_x(x, k, m, stream);
  if (!xf) return cudaErrorMemoryAllocation;
  return launch_packed<F>(m, w, xf, y, n_rows, n_rows, k, stream);
}

// Q4_K tile (1152 B, same as its 8 blocks): 8 row heads (fp16 d and dmin,
// then the 12 packed scale/min bytes), then the lane's 32 qs bytes of chunk q
// as two 512-byte warp loads. qs byte j holds k 64q + j in its low nibble and
// k 64q + 32 + j in its high nibble.
struct Q4_K {
  static constexpr int kTileVecs = 72;
  struct Raw {
    uint4 head;
    uint4 qs[2];
  };
  struct Scales {
    __half2 s_lo, m_lo, s_hi, m_hi;  // d * scale and -dmin * min per sub-block
  };

  __device__ static Raw load(const uint4* tile, int lane) {
    Raw r;
    r.head = ld_stream(tile + frag_row(lane));
    r.qs[0] = ld_stream(tile + 8 + lane);
    r.qs[1] = ld_stream(tile + 40 + lane);
    return r;
  }
  // ggml's get_scale_min_k4 on the scale words y, z, w of a row head.
  __device__ static void scale_min(int j, const uint4& h, int& sc, int& mn) {
    int sh = 8 * (j & 3);
    if (j < 4) {
      sc = (h.y >> sh) & 63;
      mn = (h.z >> sh) & 63;
    } else {
      sc = ((h.w >> sh) & 0xF) | (((h.y >> (sh + 6)) & 3) << 4);
      mn = ((h.w >> (sh + 4)) & 0xF) | (((h.z >> (sh + 6)) & 3) << 4);
    }
  }
  __device__ static Scales scales(const Raw& r, int lane) {
    float d = __half2float(__ushort_as_half((unsigned short)(r.head.x & 0xFFFF)));
    float dmin = __half2float(__ushort_as_half((unsigned short)(r.head.x >> 16)));
    int q = quadpair(lane), sc_lo, mn_lo, sc_hi, mn_hi;
    scale_min(2 * q, r.head, sc_lo, mn_lo);
    scale_min(2 * q + 1, r.head, sc_hi, mn_hi);
    return {__float2half2_rn(d * sc_lo), __float2half2_rn(-dmin * mn_lo),
            __float2half2_rn(d * sc_hi), __float2half2_rn(-dmin * mn_hi)};
  }
  // Nibbles at bits 0 and 16 of v -> scale * n - min. 1024 + n is exact, so
  // the FMA rounds only once.
  __device__ static uint32_t nib2(uint32_t v, __half2 scale, __half2 neg_min) {
    const __half2 k1024 = __half2half2(__ushort_as_half(0x6400));
    uint32_t biased = (v & 0x000F000Fu) | 0x64006400u;
    __half2 h = __hfma2(__hsub2(*reinterpret_cast<__half2*>(&biased), k1024), scale, neg_min);
    return *reinterpret_cast<uint32_t*>(&h);
  }
  __device__ static void dq(const Raw& r, const Scales& s, int wi, uint2& lo, uint2& hi) {
    uint32_t p = __byte_perm(component(r.qs[wi / 4], wi % 4), 0, 0x3120);  // b0 b2 b1 b3
    lo = make_uint2(nib2(p, s.s_lo, s.m_lo), nib2(p >> 8, s.s_lo, s.m_lo));
    hi = make_uint2(nib2(p >> 4, s.s_hi, s.m_hi), nib2(p >> 12, s.s_hi, s.m_hi));
  }
};

// Q8_0 tile (2176 B, same as its 64 blocks): 128-byte table of fp16 d per row
// per 32-k block, then qs as four 512-byte warp loads, each lane getting 16
// consecutive k of its row (lo chunk k 64q.., then hi chunk k 64q+32..).
struct Q8_0 {
  static constexpr int kTileVecs = 136;
  struct Raw {
    uint32_t d;  // fp16 d of the lane's lo and hi blocks
    uint4 qs[4];
  };
  struct Scales {
    __half2 lo, hi;
  };

  __device__ static Raw load(const uint4* tile, int lane) {
    Raw r;
    r.d = ld_stream(reinterpret_cast<const uint32_t*>(tile) + frag_row(lane) * 4 + quadpair(lane));
#pragma unroll
    for (int j = 0; j < 4; ++j) r.qs[j] = ld_stream(tile + 8 + 32 * j + lane);
    return r;
  }
  __device__ static Scales scales(const Raw& r, int) {
    return {__half2half2(__ushort_as_half((unsigned short)(r.d & 0xFFFF))),
            __half2half2(__ushort_as_half((unsigned short)(r.d >> 16)))};
  }
  // Flipping the sign bit turns int8 q into q + 128.
  __device__ static void dq(const Raw& r, const Scales& s, int wi, uint2& lo, uint2& hi) {
    const __half2 bias = __float2half2_rn(1152.f);
    lo = bytes_to_half4(component(r.qs[wi / 4], wi % 4) ^ 0x80808080u, bias, s.lo);
    hi = bytes_to_half4(component(r.qs[2 + wi / 4], wi % 4) ^ 0x80808080u, bias, s.hi);
  }
};

// Q6_K tile (1792 B; its 8 blocks are 1680): three 512-byte warp loads per
// lane: ql nibbles for words 0-3 and 4-7 (byte j = lo value j | hi value j << 4),
// and the top two bits of all 64 values (byte j of uint32 p: bits 0-1 lo / 2-3
// hi of word 2p, bits 4-5 lo / 6-7 hi of word 2p + 1). Then a 256-byte warp
// load of the lane's four fp16 sub-block scales d * sc (k 64q + 16i, i < 4),
// precomputed at repack: one load instead of the separate d and int8 scale
// loads, which cost 25% of the bandwidth.
struct Q6_K {
  static constexpr int kTileVecs = 112;
  struct Raw {
    uint2 sc;  // fp16 d * scale of the lane's four 16-k sub-blocks
    uint4 ql[2];
    uint4 qh;
  };
  struct Scales {
    __half2 s[4];  // lo words 0-3, lo 4-7, hi 0-3, hi 4-7
  };

  __device__ static Raw load(const uint4* tile, int lane) {
    Raw r;
    r.ql[0] = __ldcs(tile + lane);
    r.ql[1] = __ldcs(tile + 32 + lane);
    r.qh = __ldcs(tile + 64 + lane);
    r.sc = __ldcs(reinterpret_cast<const uint2*>(tile + 96) + lane);
    return r;
  }
  __device__ static Scales scales(const Raw& r, int) {
    Scales s;
    uint32_t v[4] = {__byte_perm(r.sc.x, 0, 0x1010), __byte_perm(r.sc.x, 0, 0x3232),
                     __byte_perm(r.sc.y, 0, 0x1010), __byte_perm(r.sc.y, 0, 0x3232)};
#pragma unroll
    for (int i = 0; i < 4; ++i) s.s[i] = *reinterpret_cast<__half2*>(&v[i]);
    return s;
  }
  // 6-bit q is stored unsigned; the value is (q - 32) * d * scale.
  __device__ static void dq(const Raw& r, const Scales& s, int wi, uint2& lo, uint2& hi) {
    uint32_t l = component(r.ql[wi / 4], wi % 4);
    uint32_t h = component(r.qh, wi / 2);
    uint32_t h_lo = (wi % 2 == 0 ? h << 4 : h) & 0x30303030u;
    uint32_t h_hi = (wi % 2 == 0 ? h << 2 : h >> 2) & 0x30303030u;
    const __half2 bias = __float2half2_rn(1056.f);
    lo = bytes_to_half4((l & 0x0F0F0F0Fu) | h_lo, bias, s.s[wi / 4]);
    hi = bytes_to_half4(((l >> 4) & 0x0F0F0F0Fu) | h_hi, bias, s.s[2 + wi / 4]);
  }
};
}  // namespace mma884

// Device-indexed expert matmul. A resident decode step routes on the device, so
// the expert id is not known to the host when a graph is captured; this kernel
// reads it from ids[slot] and offsets the weight base itself. `stack` holds the
// tiles of `count` experts back to back, each n_rows / 8 * k / kTileK tiles.
namespace mma884 {
// The main loop below repeats the one in matmul (TG = 1, no split) on purpose, with
// one change: the base of the weight tiles. Sharing a __device__ body would edit
// matmul, whose SASS and cited source lines must stay as they are.
// m <= 8 and one K range per warp only (split_k == 1; the caller checks).
// An id outside [0, count) reads no weight: the accumulators are NaN, so the
// store writes NaN into y and the failure reaches the logits. That case leaves
// by an early return, ahead of the loop. Written as the else branch of the loop
// instead, the kernel ran 6 percent slower than matmul on the same tiles (3090,
// Q4_K 512x2048); with the early return it is not slower.
template <class F>
__global__ void __launch_bounds__(kWarps * 32)
    matmul_expert_k(const uint4* __restrict__ stack, const int* __restrict__ ids, int slot,
                    int count, const uint4* __restrict__ xf, float* __restrict__ y, int ldy,
                    int n_rows, int k, int m) {
  int lane = threadIdx.x & 31;
  int g = blockIdx.x * kWarps + threadIdx.x / 32;
  if (g >= n_rows / 8) return;
  int nb = k / kTileK;
  int id = __ldg(ids + slot);
  if ((unsigned)id >= (unsigned)count) {
    float bad_acc[1][8];
#pragma unroll
    for (int i = 0; i < 8; ++i) bad_acc[0][i] = __int_as_float(0x7fc00000);
    store<1, false>(bad_acc, y, g, ldy, m, lane);
    return;
  }
  float acc[1][8] = {};
  size_t expert_vecs = (size_t)(n_rows / 8) * nb * F::kTileVecs;
  const uint4* tiles = stack + (size_t)id * expert_vecs + (size_t)g * nb * F::kTileVecs;
  typename F::Raw next = F::load(tiles, lane);
  for (int ib = 0; ib < nb; ++ib) {
    typename F::Raw cur = next;
    if (ib + 1 < nb) next = F::load(tiles + (size_t)(ib + 1) * F::kTileVecs, lane);
    typename F::Scales sc = F::scales(cur, lane);
#pragma unroll
    for (int wi = 0; wi < 8; ++wi) {
      uint2 lo, hi;
      F::dq(cur, sc, wi, lo, hi);
      uint4 xv = load_x(xf, 0, nb, ib, wi, lane, m);
      mma(acc[0], lo, make_uint2(xv.x, xv.y));
      mma(acc[0], hi, make_uint2(xv.z, xv.w));
    }
  }
  store<1, false>(acc, y, g, ldy, m, lane);
}

// y[t * ldy + r] for m <= 8 tokens against expert ids[slot] of `stack`; on `stream`.
// Returns the launch status. The caller has validated the shape.
template <class F>
cudaError_t launch_expert(int m, const uint4* stack, const int* ids, int slot, int count,
                          const uint4* xf, float* y, int ldy, int n_rows, int k,
                          cudaStream_t stream) {
  dim3 grid((n_rows / 8 + kWarps - 1) / kWarps), block(kWarps * 32);
  matmul_expert_k<F><<<grid, block, 0, stream>>>(stack, ids, slot, count, xf, y, ldy, n_rows, k, m);
  return cudaGetLastError();
}
}  // namespace mma884
