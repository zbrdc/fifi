// The element-wise residual sum of a step: out[i] = x[i] + y[i] in f32, for i below n. One
// thread a float. The device's arithmetic does not keep a subnormal: an operand or a result
// below the least normal is a zero of its sign (M4 Max, 2026-11-27, under every math option the
// SDK names); the CUDA adapter's add keeps them.
//
// row_add_fused is the publish contract of the CUDA row add: out holds the bf16 round trip of
// each sum, round to nearest even, and a sum whose round trip is not finite is written 0 with
// bit 0 of the flag set. The round trip and its finiteness test are bit arithmetic: they do not
// depend on the compile options a shader gets.
//
// The layout of RowAddParams is the layout of the Rust parameter of resident/row_add.rs.

#include <metal_stdlib>
using namespace metal;

struct RowAddParams {
  uint n;
};

// f32 to bf16 and back, round to nearest even. A NaN stays what it is: the rounding bias would
// carry a NaN of a low mantissa into a signed zero.
static inline float bf16_roundtrip(float v) {
  const uint bits = as_type<uint>(v);
  if ((bits & 0x7fffffffu) > 0x7f800000u) return v;
  const uint bias = 0x7fffu + ((bits >> 16u) & 1u);
  return as_type<float>(((bits + bias) >> 16u) << 16u);
}

// Whether a float is an infinity or a NaN, on its bits.
static inline bool nonfinite(float v) {
  return (as_type<uint>(v) & 0x7f800000u) == 0x7f800000u;
}

kernel void row_add(device float *out [[buffer(0)]],
                    device const float *x [[buffer(1)]],
                    device const float *y [[buffer(2)]],
                    constant RowAddParams &p [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  out[i] = x[i] + y[i];
}

kernel void row_add_fused(device float *out [[buffer(0)]],
                          device const float *x [[buffer(1)]],
                          device const float *y [[buffer(2)]],
                          device atomic_uint *flag [[buffer(3)]],
                          constant RowAddParams &p [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  const float published = bf16_roundtrip(x[i] + y[i]);
  if (nonfinite(published)) {
    out[i] = 0.0f;
    atomic_fetch_or_explicit(flag, 1u, memory_order_relaxed);
    return;
  }
  out[i] = published;
}
