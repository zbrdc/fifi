// Dense half-precision matvec, the decode batch: y[t][r] is the sum over the columns c of
// w[r][c] * x[t][c], every product one rounded multiply and add.
//
// One SIMD group (32 lanes) reads one weight row; a threadgroup of eight groups covers eight
// rows of one activation row, so the grid is (rows / 8, m). A lane adds its terms in column
// order: eight a step on the vector path, which needs the weight and activation ranges on a
// 16-byte grid, a column count that is a multiple of eight and an activation stride that is a
// multiple of four, then the columns of a last partial step one a lane; one a step otherwise.
// The group then reduces by a fixed butterfly, so a result does not depend on how many rows
// the call carries. A weight that is not finite sets the flag and leaves its result unwritten.
//
// The layout of GemvParams is the layout of the Rust GemvParams of resident/dense_gemv.rs.

#include <metal_stdlib>
using namespace metal;

struct GemvParams {
  uint m;
  uint rows;
  uint cols;
  uint ld_x;
  uint ld_y;
  uint vector;
};

constant constexpr uint kLanes = 32;
constant constexpr uint kGroups = 8;

// bfloat16 to float is exact: sixteen zero bits appended.
static inline float widen(ushort code) { return as_type<float>(uint(code) << 16); }

// binary16 to float is exact.
static inline float widen(half code) { return float(code); }

// Eight codes of one 16-byte load, in column order.
static inline void widen8(ushort /* tag */, uint4 packed, thread float *out) {
  const uint codes[4] = {packed.x, packed.y, packed.z, packed.w};
  for (uint i = 0; i < 4; ++i) {
    out[2 * i] = as_type<float>(codes[i] << 16);
    out[2 * i + 1] = as_type<float>(codes[i] & 0xffff0000u);
  }
}

static inline void widen8(half /* tag */, uint4 packed, thread float *out) {
  const float4 lo = float4(as_type<half4>(uint2(packed.x, packed.y)));
  const float4 hi = float4(as_type<half4>(uint2(packed.z, packed.w)));
  out[0] = lo.x;
  out[1] = lo.y;
  out[2] = lo.z;
  out[3] = lo.w;
  out[4] = hi.x;
  out[5] = hi.y;
  out[6] = hi.z;
  out[7] = hi.w;
}

template <typename Code>
static void dense_gemv_body(device const Code *w, device const float *x, device float *y,
                            device atomic_uint *nonfinite, constant GemvParams &p, uint lane,
                            uint group, uint2 gid) {
  const uint row = gid.x * kGroups + group;
  if (row >= p.rows) return;
  const uint t = gid.y;
  device const Code *wrow = w + size_t(row) * p.cols;
  device const float *xrow = x + size_t(t) * p.ld_x;
  float acc = 0.0f;
  bool bad = false;
  uint wide = 0u;
  if (p.vector != 0u) {
    // Whole steps of kLanes * 8 columns; the rest goes the scalar way below.
    wide = p.cols / (kLanes * 8u) * (kLanes * 8u);
    device const uint4 *wv = reinterpret_cast<device const uint4 *>(wrow);
    for (uint base = 0; base < wide; base += kLanes * 8u) {
      const uint c = base + lane * 8u;
      float values[8];
      widen8(Code(0), wv[c / 8u], values);
      const float4 x0 = *reinterpret_cast<device const float4 *>(xrow + c);
      const float4 x1 = *reinterpret_cast<device const float4 *>(xrow + c + 4u);
      const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
      for (uint i = 0; i < 8; ++i) {
        bad |= !isfinite(values[i]);
        acc = fma(values[i], xv[i], acc);
      }
    }
  }
  for (uint c = wide + lane; c < p.cols; c += kLanes) {
    const float value = widen(wrow[c]);
    bad |= !isfinite(value);
    acc = fma(value, xrow[c], acc);
  }
  // One bad weight refuses the whole row, whichever lane read it.
  bad = simd_any(bad);
  for (uint mask = kLanes / 2u; mask > 0u; mask >>= 1u) {
    acc += simd_shuffle_xor(acc, ushort(mask));
  }
  if (lane == 0u) {
    if (bad) {
      atomic_fetch_or_explicit(nonfinite, 1u, memory_order_relaxed);
    } else {
      y[size_t(t) * p.ld_y + row] = acc;
    }
  }
}

kernel void dense_gemv_f16(device const half *w [[buffer(0)]],
                           device const float *x [[buffer(1)]],
                           device float *y [[buffer(2)]],
                           device atomic_uint *nonfinite [[buffer(3)]],
                           constant GemvParams &p [[buffer(4)]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint group [[simdgroup_index_in_threadgroup]],
                           uint2 gid [[threadgroup_position_in_grid]]) {
  dense_gemv_body(w, x, y, nonfinite, p, lane, group, gid);
}

kernel void dense_gemv_bf16(device const ushort *w [[buffer(0)]],
                            device const float *x [[buffer(1)]],
                            device float *y [[buffer(2)]],
                            device atomic_uint *nonfinite [[buffer(3)]],
                            constant GemvParams &p [[buffer(4)]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint group [[simdgroup_index_in_threadgroup]],
                            uint2 gid [[threadgroup_position_in_grid]]) {
  dense_gemv_body(w, x, y, nonfinite, p, lane, group, gid);
}
