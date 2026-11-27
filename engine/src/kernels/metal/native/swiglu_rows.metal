// The SwiGLU of a step's feed-forward block, f32 out:
//   act[r][c] = silu(gu[r][c]) * gu[r][up_off + c],  silu(g) = g / (1 + exp(-g)),
// for r below rows and c below n_ff.
//
// One thread four columns, a float4 of the gate and one of the up values. n_ff, up_off, ld_in
// and ld_out are multiples of four and both ranges start on a 16-byte grid, so every load and
// every store is whole. The silu takes the precise exp and division, whatever the compile
// options a shader gets: the result is within a few f32 roundings of the IEEE silu, inside the
// CUDA adapter's bound of 2^-18. The device's arithmetic makes every subnormal a zero of its
// sign, an operand or a result (row_add.metal has the measurement): an error below 2^-126 times
// the other operand, and below 2^-126 on the result.
//
// The layout of SwigluParams is the layout of the Rust parameter of resident/swiglu_rows.rs.

#include <metal_stdlib>
using namespace metal;

struct SwigluParams {
  uint rows;
  uint n_ff;
  uint up_off;
  uint ld_in;
  uint ld_out;
};

static inline float silu(float gate) {
  return precise::divide(gate, 1.0f + precise::exp(-gate));
}

kernel void swiglu_rows(device const float *gu [[buffer(0)]],
                        device float *act [[buffer(1)]],
                        constant SwigluParams &p [[buffer(2)]],
                        uint2 pos [[thread_position_in_grid]]) {
  const uint quads = p.n_ff / 4u;
  if (pos.x >= quads) return;
  const size_t in = size_t(pos.y) * p.ld_in + 4u * pos.x;
  const float4 gate = *reinterpret_cast<device const float4 *>(gu + in);
  const float4 up = *reinterpret_cast<device const float4 *>(gu + in + p.up_off);
  float4 out;
  out.x = silu(gate.x) * up.x;
  out.y = silu(gate.y) * up.y;
  out.z = silu(gate.z) * up.z;
  out.w = silu(gate.w) * up.w;
  *reinterpret_cast<device float4 *>(act + size_t(pos.y) * p.ld_out + 4u * pos.x) = out;
}
