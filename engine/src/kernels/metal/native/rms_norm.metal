// The RMS norm of a step with its optional residual add, f32 out, for a row x and its weight w:
//   v = x + delta (when delta is given, and x takes the sum),
//   out = v * rsqrt(sum(v * v) / n + eps) * w.
//
// One threadgroup of 256 threads a row, the CUDA adapter's block: the grid's y dimension the
// row. A thread takes every 256th float4 of its row, two quads in flight at a time: n is a
// multiple of 32 and the four ranges start on a 16-byte grid, so every load and every store is
// whole. A thread sums its squares in f32, the SIMD butterfly adds the 32 lanes of its group,
// and the eight group sums meet in threadgroup memory behind one barrier. The division and the
// reciprocal square root are the precise ones: correctly rounded on this device (M4 Max,
// 2026-11-27, the bridge's safe math mode; plain rsqrt reaches 0.61 ulp), so the f32 row
// differs from an f64 reference by the order of the sum of squares only. The device's
// arithmetic makes a subnormal an operand or a result a zero of its sign (row_add.metal has the
// measurement): a subnormal square adds nothing, a subnormal sum is written as a zero.
//
// A thread writes the sums of its own quads in the first pass and reads them back in the
// second. One SIMD group a row was measured first: 15.1 us a dispatch in a frame at n 2304
// against the row add's 2.3 us, one dependent load chain a round.
//
// The layout of RmsNormParams is the layout of the Rust parameter of resident/rms_norm.rs.

#include <metal_stdlib>
using namespace metal;

struct RmsNormParams {
  uint n;
  float eps;
};

constant constexpr uint kThreads = 256;
constant constexpr uint kGroups = kThreads / 32u;

// The sum of the 32 lanes' values, in every lane.
static inline float lane_sum(float v) {
  for (ushort mask = ushort(16); mask != 0; mask >>= 1u) v += simd_shuffle_xor(v, mask);
  return v;
}

// The sum of the squares of a row of `quads` float4, as thread `t` of the threadgroup's 256
// adds it; with the add, each sum lands back in its quad first.
template <bool with_delta>
static inline float sum_of_squares(device float4 *xq, device const float4 *dq, uint t,
                                   uint quads) {
  float4 acc0 = 0.0f, acc1 = 0.0f;
  uint q = t;
  for (; q + kThreads < quads; q += 2u * kThreads) {
    float4 a = xq[q];
    float4 b = xq[q + kThreads];
    if (with_delta) {
      a += dq[q];
      b += dq[q + kThreads];
      xq[q] = a;
      xq[q + kThreads] = b;
    }
    acc0 += a * a;
    acc1 += b * b;
  }
  if (q < quads) {
    float4 a = xq[q];
    if (with_delta) {
      a += dq[q];
      xq[q] = a;
    }
    acc0 += a * a;
  }
  const float4 acc = acc0 + acc1;
  return lane_sum(acc.x + (acc.y + (acc.z + acc.w)));
}

template <bool with_delta>
static inline void rms_norm_row(device float *x, device const float *delta, device const float *w,
                                device float *out, threadgroup float *partial,
                                constant RmsNormParams &p, uint t) {
  const uint quads = p.n / 4u;
  device float4 *xq = reinterpret_cast<device float4 *>(x);
  device const float4 *dq = reinterpret_cast<device const float4 *>(delta);
  float s = sum_of_squares<with_delta>(xq, dq, t, quads);
  if (t % 32u == 0u) partial[t / 32u] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  s = ((partial[0] + partial[1]) + (partial[2] + partial[3])) +
      ((partial[4] + partial[5]) + (partial[6] + partial[7]));
  const float rstd = precise::rsqrt(precise::divide(s, float(p.n)) + p.eps);
  device const float4 *wq = reinterpret_cast<device const float4 *>(w);
  device float4 *oq = reinterpret_cast<device float4 *>(out);
  for (uint q = t; q < quads; q += kThreads) {
    oq[q] = (xq[q] * rstd) * wq[q];
  }
}

kernel void rms_norm(device float *x [[buffer(0)]],
                     device const float *w [[buffer(1)]],
                     device float *out [[buffer(2)]],
                     constant RmsNormParams &p [[buffer(3)]],
                     uint2 pos [[thread_position_in_grid]],
                     uint t [[thread_index_in_threadgroup]]) {
  threadgroup float partial[kGroups];
  const size_t row = size_t(pos.y) * p.n;
  rms_norm_row<false>(x + row, x, w, out + row, partial, p, t);
}

kernel void rms_norm_add(device float *x [[buffer(0)]],
                         device const float *delta [[buffer(1)]],
                         device const float *w [[buffer(2)]],
                         device float *out [[buffer(3)]],
                         constant RmsNormParams &p [[buffer(4)]],
                         uint2 pos [[thread_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]]) {
  threadgroup float partial[kGroups];
  const size_t row = size_t(pos.y) * p.n;
  rms_norm_row<true>(x + row, delta + row, w, out + row, partial, p, t);
}
