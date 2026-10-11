// Large-M (prefill) tensor-core GEMM for the repacked Q4_K / Q8_0 / Q6_K tiles.
//
// y[t * ldy + r] = sum_k W[r][k] * x[t * k_dim + k]   (fp32 out, fp16 x)
//
// Weights stay in the decode path's tiled layout (8 rows x 256 k per tile, see
// mma884.cuh). Activation layout: plain row-major fp16 x[m][k] (the engine's norm
// kernels write it directly). No fragment packing: the CTA stages x through
// shared memory.
//
// One CTA computes a BM-row x BN-token tile. The k loop runs in 64-k steps (one
// 64-k "chunk" q of a tile, the unit a quadpair owns in the decode kernels):
//   1. each thread loads the raw bytes of 32 weights of one row (one 16 B vec
//      for Q4_K/Q6_K, two for Q8_0) and dequantizes them to fp16 in registers
//      with the SAME arithmetic as the decode kernels, then stores them to a
//      shared A tile [BM][64];
//   2. x [BN][64] fp16 is copied to a shared B tile;
//   3. warps run mma.m8n8k4 on the shared tiles; fp32 accumulators.
// Global loads of step s+1 are issued before the mma work of step s and the
// shared tiles are double-buffered (one __syncthreads per step).
//
// mma.m8n8k4 gives each quadpair its own A and B fragments; here the four
// quadpairs of a warp compute the four 8x8 quadrants of one 16x16 output tile
// (the decode kernels instead give them different k slices of one 8x8 tile).
// Quadpair q = (lane >> 2) & 3 takes weight rows 2*fr + (q & 1) and tokens
// 2*fr + (q >> 1) of the 16-wide block, fr = frag_row(lane). Interleaving the
// rows this way makes the 8 lanes of a 128-bit shared load hit 8 consecutive
// smem rows, which is bank-conflict free with the odd row stride (9 uint4).
#pragma once

#include <cuda_fp16.h>

#include "mma884.cuh"

namespace gemm {
constexpr int kBK = 64;            // k per pipeline step
constexpr int kStride = kBK + 8;   // halves per smem row; 9 uint4 (odd) avoids bank conflicts

// Non-volatile so the scheduler may interleave it with dequantization.
__device__ inline void mma(float (&c)[8], uint2 a, uint2 b) {
  asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]),
        "+f"(c[7])
      : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
}

// Weight-format adapters. A thread owns (rt, h): row rt of the 8-row tile and
// half h of 64-k chunk q. load() fetches its raw bytes; dq() produces 32 fp16
// (4 uint4) forming two 16-k runs that land at smem columns col(0, h) and
// col(1, h) of the chunk.
struct DQ4K {
  using F = mma884::Q4_K;
  struct Raw {
    uint4 head, qs;
  };
  __device__ static int lane_of(int rt, int q) { return 16 * (rt >> 2) + 4 * q + (rt & 3); }
  __device__ static int col(int run, int h) { return 32 * run + 16 * h; }
  __device__ static Raw load(const uint4* tile, int rt, int q, int h) {
    Raw r;
    r.head = __ldg(tile + rt);
    r.qs = mma884::ld_stream(tile + 8 + 32 * h + lane_of(rt, q));
    return r;
  }
  __device__ static void dq(const Raw& r, int q, int, uint4 (&out)[4]) {
    float d = __half2float(__ushort_as_half((unsigned short)(r.head.x & 0xFFFF)));
    float dmin = __half2float(__ushort_as_half((unsigned short)(r.head.x >> 16)));
    int sc_lo, mn_lo, sc_hi, mn_hi;
    F::scale_min(2 * q, r.head, sc_lo, mn_lo);
    F::scale_min(2 * q + 1, r.head, sc_hi, mn_hi);
    __half2 s_lo = __float2half2_rn(d * sc_lo), m_lo = __float2half2_rn(-dmin * mn_lo);
    __half2 s_hi = __float2half2_rn(d * sc_hi), m_hi = __float2half2_rn(-dmin * mn_hi);
    uint2 lo[4], hi[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      uint32_t p = __byte_perm(mma884::component(r.qs, w), 0, 0x3120);
      lo[w] = make_uint2(F::nib2(p, s_lo, m_lo), F::nib2(p >> 8, s_lo, m_lo));
      hi[w] = make_uint2(F::nib2(p >> 4, s_hi, m_hi), F::nib2(p >> 12, s_hi, m_hi));
    }
    out[0] = make_uint4(lo[0].x, lo[0].y, lo[1].x, lo[1].y);
    out[1] = make_uint4(lo[2].x, lo[2].y, lo[3].x, lo[3].y);
    out[2] = make_uint4(hi[0].x, hi[0].y, hi[1].x, hi[1].y);
    out[3] = make_uint4(hi[2].x, hi[2].y, hi[3].x, hi[3].y);
  }
};

struct DQ8 {
  using F = mma884::Q8_0;
  struct Raw {
    uint4 v[2];
    unsigned short d;
  };
  __device__ static int lane_of(int rt, int q) { return 16 * (rt >> 2) + 4 * q + (rt & 3); }
  __device__ static int col(int run, int h) { return 32 * h + 16 * run; }
  __device__ static Raw load(const uint4* tile, int rt, int q, int h) {
    Raw r;
    int lane = lane_of(rt, q);
    r.v[0] = mma884::ld_stream(tile + 8 + 32 * (2 * h) + lane);
    r.v[1] = mma884::ld_stream(tile + 8 + 32 * (2 * h + 1) + lane);
    r.d = __ldg(reinterpret_cast<const unsigned short*>(tile) + rt * 8 + 2 * q + h);
    return r;
  }
  __device__ static void dq(const Raw& r, int, int, uint4 (&out)[4]) {
    const __half2 bias = __float2half2_rn(1152.f);
    __half2 s = __half2half2(__ushort_as_half(r.d));
    uint2 v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      v[i] = mma884::bytes_to_half4(mma884::component(r.v[i / 4], i % 4) ^ 0x80808080u, bias, s);
#pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = make_uint4(v[2 * i].x, v[2 * i].y, v[2 * i + 1].x, v[2 * i + 1].y);
  }
};

struct DQ6K {
  using F = mma884::Q6_K;
  struct Raw {
    uint4 ql;
    uint2 qh;
    unsigned short s_lo, s_hi;
  };
  __device__ static int lane_of(int rt, int q) { return 16 * (rt >> 2) + 4 * q + (rt & 3); }
  __device__ static int col(int run, int h) { return 32 * run + 16 * h; }
  __device__ static Raw load(const uint4* tile, int rt, int q, int h) {
    Raw r;
    int lane = lane_of(rt, q);
    r.ql = __ldcs(tile + 32 * h + lane);
    r.qh = __ldcs(reinterpret_cast<const uint2*>(tile + 64) + 2 * lane + h);
    const unsigned short* sc = reinterpret_cast<const unsigned short*>(tile + 96) + 4 * lane;
    r.s_lo = __ldcs(sc + h);
    r.s_hi = __ldcs(sc + 2 + h);
    return r;
  }
  __device__ static void dq(const Raw& r, int, int, uint4 (&out)[4]) {
    const __half2 bias = __float2half2_rn(1056.f);
    __half2 s_lo = __half2half2(__ushort_as_half(r.s_lo));
    __half2 s_hi = __half2half2(__ushort_as_half(r.s_hi));
    uint2 lo[4], hi[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      uint32_t l = mma884::component(r.ql, w);
      uint32_t hh = w < 2 ? r.qh.x : r.qh.y;
      uint32_t h_lo = (w % 2 == 0 ? hh << 4 : hh) & 0x30303030u;
      uint32_t h_hi = (w % 2 == 0 ? hh << 2 : hh >> 2) & 0x30303030u;
      lo[w] = mma884::bytes_to_half4((l & 0x0F0F0F0Fu) | h_lo, bias, s_lo);
      hi[w] = mma884::bytes_to_half4(((l >> 4) & 0x0F0F0F0Fu) | h_hi, bias, s_hi);
    }
    out[0] = make_uint4(lo[0].x, lo[0].y, lo[1].x, lo[1].y);
    out[1] = make_uint4(lo[2].x, lo[2].y, lo[3].x, lo[3].y);
    out[2] = make_uint4(hi[0].x, hi[0].y, hi[1].x, hi[1].y);
    out[3] = make_uint4(hi[2].x, hi[2].y, hi[3].x, hi[3].y);
  }
};

// CTA tile: BM weight rows x BN tokens, WM x WN warps, each warp (BM/WM) x (BN/WN).
template <int BM_, int BN_, int WM_, int WN_, int MINB_ = 1>
struct Tile {
  static constexpr int BM = BM_, BN = BN_, WM = WM_, WN = WN_, kMinBlocks = MINB_;
  static constexpr int kThreads = 32 * WM * WN;
  static constexpr int WTM = BM / (WM * 16), WTN = BN / (WN * 16);  // 16x16 tiles per warp
  static constexpr int kNA = (2 * BM + kThreads - 1) / kThreads;    // A items (row, half) per thread
  static constexpr int kNB = (BN * 8 + kThreads - 1) / kThreads;    // B uint4 per thread
  static constexpr size_t kSmem = (size_t)2 * (BM + BN) * kStride * sizeof(__half);
  static_assert(BM % (WM * 16) == 0 && BN % (WN * 16) == 0, "warp tiles are multiples of 16");
};

template <class D, class T>
struct Kernel {
  using Raw = typename D::Raw;

  struct Fp16 {
    uint4 v[4];  // 32 dequantized weights
  };

  // Everything a thread needs to stage its A and B items.
  struct Stage {
    Raw ra[T::kNA];
    uint4 rb[T::kNB];
  };

  __device__ static void load_a(const uint4* w, Stage& st, int step, int row0, int groups,
                                int nb) {
    int ib = step >> 2, q = step & 3;
#pragma unroll
    for (int j = 0; j < T::kNA; ++j) {
      int i = threadIdx.x + j * T::kThreads;
      if (i >= 2 * T::BM) continue;
      int r = i % T::BM, h = i / T::BM;
      int g = min(row0 / 8 + r / 8, groups - 1);
      st.ra[j] = D::load(w + ((size_t)g * nb + ib) * D::F::kTileVecs, r & 7, q, h);
    }
  }

  __device__ static void load_b(const __half* x, Stage& st, int step, int tok0, int k, int m) {
#pragma unroll
    for (int j = 0; j < T::kNB; ++j) {
      int i = threadIdx.x + j * T::kThreads;
      int t = tok0 + i / 8;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (i < T::BN * 8 && t < m)
        v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)t * k + step * kBK) + i % 8);
      st.rb[j] = v;
    }
  }

  // Dequantizes the staged weights; kept apart from the shared-memory stores
  // so the ALU work can overlap the previous step's mma instructions.
  __device__ static void dequant(const Stage& st, Fp16 (&out)[T::kNA], int q) {
#pragma unroll
    for (int j = 0; j < T::kNA; ++j) {
      int i = threadIdx.x + j * T::kThreads;
      if (i >= 2 * T::BM) continue;
      D::dq(st.ra[j], q, i / T::BM, out[j].v);
    }
  }

  __device__ static void store_a(const Fp16 (&dq)[T::kNA], __half* as) {
#pragma unroll
    for (int j = 0; j < T::kNA; ++j) {
      int i = threadIdx.x + j * T::kThreads;
      if (i >= 2 * T::BM) continue;
      int r = i % T::BM, h = i / T::BM;
      uint4* row = reinterpret_cast<uint4*>(as + r * kStride);
      int c0 = D::col(0, h) / 8, c1 = D::col(1, h) / 8;
      row[c0] = dq[j].v[0];
      row[c0 + 1] = dq[j].v[1];
      row[c1] = dq[j].v[2];
      row[c1 + 1] = dq[j].v[3];
    }
  }

  // x is copied raw, so it is stored before the compute that hides the loads.
  __device__ static void store_b(const Stage& st, __half* bs) {
#pragma unroll
    for (int j = 0; j < T::kNB; ++j) {
      int i = threadIdx.x + j * T::kThreads;
      if (i < T::BN * 8) reinterpret_cast<uint4*>(bs + (i / 8) * kStride)[i % 8] = st.rb[j];
    }
  }

  // 64 k of mma on one shared buffer pair.
  __device__ static void compute(float (&acc)[T::WTM][T::WTN][8], const __half* as,
                                 const __half* bs, int a_off, int b_off) {
#pragma unroll
    for (int ks = 0; ks < kBK / 8; ++ks) {
      uint4 fa[T::WTM], fb[T::WTN];
#pragma unroll
      for (int a = 0; a < T::WTM; ++a)
        fa[a] = *reinterpret_cast<const uint4*>(as + a_off + a * 16 * kStride + ks * 8);
#pragma unroll
      for (int b = 0; b < T::WTN; ++b)
        fb[b] = *reinterpret_cast<const uint4*>(bs + b_off + b * 16 * kStride + ks * 8);
#pragma unroll
      for (int a = 0; a < T::WTM; ++a) {
#pragma unroll
        for (int b = 0; b < T::WTN; ++b) {
          mma(acc[a][b], make_uint2(fa[a].x, fa[a].y), make_uint2(fb[b].x, fb[b].y));
          mma(acc[a][b], make_uint2(fa[a].z, fa[a].w), make_uint2(fb[b].z, fb[b].w));
        }
      }
    }
  }

  __device__ static void epilogue(const float (&acc)[T::WTM][T::WTN][8], float* y, int ldy,
                                  int row_base, int tok_base, int n_rows, int m) {
    int lane = threadIdx.x & 31, q = mma884::quadpair(lane);
#pragma unroll
    for (int a = 0; a < T::WTM; ++a) {
#pragma unroll
      for (int b = 0; b < T::WTN; ++b) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          int rr = (lane & 1) + (i & 2) + 4 * (lane >> 4);
          int cc = (i & 4) + (lane & 2) + (i & 1);
          int row = row_base + 16 * a + 2 * rr + (q & 1);
          int tok = tok_base + 16 * b + 2 * cc + (q >> 1);
          if (row < n_rows && tok < m) y[(size_t)tok * ldy + row] = acc[a][b][i];
        }
      }
    }
  }

  __device__ static void run(const uint4* w, const __half* x, float* y, int ldy, int n_rows,
                             int k, int m) {
    extern __shared__ __align__(16) __half smem[];
    constexpr int kTileA = T::BM * kStride, kTileB = T::BN * kStride;
    __half* as = smem;
    __half* bs = smem + 2 * kTileA;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int wm = warp / T::WN, wn = warp % T::WN;
    int q = mma884::quadpair(lane), fr = mma884::frag_row(lane);
    int row0 = blockIdx.x * T::BM, tok0 = blockIdx.y * T::BN;
    int nb = k / mma884::kTileK, steps = k / kBK, groups = n_rows / 8;
    int a_off = (wm * T::WTM * 16 + 2 * fr + (q & 1)) * kStride;
    int b_off = (wn * T::WTN * 16 + 2 * fr + (q >> 1)) * kStride;

    float acc[T::WTM][T::WTN][8] = {};
    Stage st;
    Fp16 dq[T::kNA];
    load_a(w, st, 0, row0, groups, nb);
    load_b(x, st, 0, tok0, k, m);
    dequant(st, dq, 0);
    store_a(dq, as);
    store_b(st, bs);
    __syncthreads();
    // Staging runs two steps ahead of the mma: at step s the registers hold
    // the raw bytes of s + 1 (loaded during step s - 1, so their latency is
    // hidden); they are dequantized before the mma of s to overlap it, and the
    // loads of s + 2 reuse the registers.
    if (steps > 1) {
      load_a(w, st, 1, row0, groups, nb);
      load_b(x, st, 1, tok0, k, m);
    }
    for (int s = 0; s < steps; ++s) {
      int cur = s & 1;
      if (s + 1 < steps) dequant(st, dq, (s + 1) & 3);
      __half* b_next = bs + (cur ^ 1) * kTileB;
      if (s + 1 < steps) store_b(st, b_next);
      if (s + 2 < steps) {
        load_a(w, st, s + 2, row0, groups, nb);
        load_b(x, st, s + 2, tok0, k, m);
      }
      compute(acc, as + cur * kTileA, bs + cur * kTileB, a_off, b_off);
      if (s + 1 < steps) store_a(dq, as + (cur ^ 1) * kTileA);
      __syncthreads();
    }
    epilogue(acc, y, ldy, row0 + wm * T::WTM * 16, tok0 + wn * T::WTN * 16, n_rows, m);
  }
};

template <class D, class T>
__global__ void __launch_bounds__(T::kThreads, T::kMinBlocks)
    gemm_kernel(const uint4* __restrict__ w, const __half* __restrict__ x, float* __restrict__ y,
                int ldy, int n_rows, int k, int m) {
  Kernel<D, T>::run(w, x, y, ldy, n_rows, k, m);
}

// y[t * ldy + r] for t < m, r < n_rows (n_rows % 8 == 0, k % 256 == 0,
// ldy >= n_rows); x is row-major fp16 [m][k]. Runs on `stream` (nullptr: default). Returns
// cudaErrorInvalidValue for a bad shape, else the attribute or launch status.
//
// Dispatch rule (measured on Xavier, see bench/gemm_bench.cu): use
// mma884::launch_packed for m <= 16 (padding waste makes this kernel 0.3-0.8x
// of it), Tile<128, 32, 4, 2, 1> for 17..32 tokens, Tile<128, 64, 4, 2, 1> for
// 33..64 and Tile<128, 128, 2, 4, 1> above.
template <class D, class T>
cudaError_t launch(const uint4* w, const __half* x, float* y, int ldy, int n_rows, int k,
                   int m, cudaStream_t stream = nullptr) {
  if (m <= 0 || n_rows % 8 != 0 || k % mma884::kTileK != 0 || ldy < n_rows)
    return cudaErrorInvalidValue;
  // Per template instance; not thread-safe and only valid for the current
  // device (the engine drives a single GPU from a single host thread).
  static bool configured = false;
  if (!configured) {
    cudaError_t err = cudaFuncSetAttribute(
        gemm_kernel<D, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)T::kSmem);
    if (err != cudaSuccess) return err;
    configured = true;
  }
  dim3 grid((n_rows + T::BM - 1) / T::BM, (m + T::BN - 1) / T::BN);
  gemm_kernel<D, T><<<grid, T::kThreads, T::kSmem, stream>>>(w, x, y, ldy, n_rows, k, m);
  return cudaGetLastError();
}
}  // namespace gemm
