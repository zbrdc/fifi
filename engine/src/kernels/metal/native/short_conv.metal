// The short convolution of a step, the contract of the CUDA adapter's resident short conv
// (kernels/cuda/native/resident/short_conv.h), which is the arithmetic of family/step/conv.rs's
// Variant::Residual. With taps T and width W, for every channel c, in this order and every
// operation rounded to nearest even:
//   acc    = 0 + kernel[c * T + T - 1] * input[c] + kernel[c * T + T - 2] * history[0 * W + c] + ..
//          + kernel[c * T + 0] * history[(T - 2) * W + c]
//   out[c] = (acc + input[c]) * norm
// norm is 1 / sqrt(T + 1), the sum, the square root and the reciprocal each rounded in f32. The
// Rust caller computes it and passes it in the parameters. The history holds T - 1 rows of W
// floats, row 0 the newest past input; each launch advances it in place to the raw input, the
// old row 0, and so on, dropping the oldest row.
//
// One thread a channel. A thread owns channel c of every history row: it reads each old sample
// before it moves the raw sample one row newer into its place, so the shift needs no barrier.
// `out` may be `input` itself: the input is read before the output is written. A thread that
// saw a non-finite input, tap, history value, output or state sum stores a quiet NaN of no
// payload into the health word; a launch that saw none leaves the word alone.
// short_conv_state also takes the current row of a depth stream: state[c] += out[c], one rounded
// add, the host's add_current of the convolved row.
//
// A product and the add that takes it are two roundings: contraction into an fma is off. The
// device's arithmetic makes a subnormal an operand or a result a zero of its sign
// (row_add.metal has the measurement); the CUDA adapter keeps them.
//
// The layout of ShortConvParams is the layout of the Rust parameter of resident/short_conv.rs.

#pragma clang fp contract(off)

#include <metal_stdlib>
using namespace metal;

struct ShortConvParams {
  uint width;
  uint taps;
  float norm;
};

// The quiet NaN of the health word.
constant constexpr uint kNotFinite = 0x7fc00000u;

// Channel c of the convolution, and whether a value it used or made is not finite.
static inline float convolve(const device float *input, const device float *weights,
                             device float *history, device float *out, uint c, uint width,
                             uint taps, float norm, thread bool &invalid) {
  const float x = input[c];
  const device float *k = weights + size_t(c) * taps;
  invalid = invalid || !isfinite(x) || !isfinite(k[taps - 1u]);
  float acc = 0.0f + k[taps - 1u] * x;
  float newer = x;
  for (uint lag = 1u; lag < taps; ++lag) {
    device float &sample = history[size_t(lag - 1u) * width + c];
    const float h = sample;
    sample = newer;
    newer = h;
    const float w = k[taps - 1u - lag];
    invalid = invalid || !isfinite(h) || !isfinite(w);
    acc = acc + w * h;
  }
  const float value = (acc + x) * norm;
  out[c] = value;
  invalid = invalid || !isfinite(value);
  return value;
}

kernel void short_conv(const device float *input [[buffer(0)]],
                       const device float *weights [[buffer(1)]],
                       device float *history [[buffer(2)]],
                       device float *out [[buffer(3)]],
                       device atomic_uint *health [[buffer(4)]],
                       constant ShortConvParams &p [[buffer(5)]],
                       uint c [[thread_position_in_grid]]) {
  if (c >= p.width) return;
  bool invalid = false;
  convolve(input, weights, history, out, c, p.width, p.taps, p.norm, invalid);
  if (invalid) atomic_store_explicit(health, kNotFinite, memory_order_relaxed);
}

kernel void short_conv_state(const device float *input [[buffer(0)]],
                             const device float *weights [[buffer(1)]],
                             device float *history [[buffer(2)]],
                             device float *out [[buffer(3)]],
                             device float *state [[buffer(4)]],
                             device atomic_uint *health [[buffer(5)]],
                             constant ShortConvParams &p [[buffer(6)]],
                             uint c [[thread_position_in_grid]]) {
  if (c >= p.width) return;
  bool invalid = false;
  const float value = convolve(input, weights, history, out, c, p.width, p.taps, p.norm, invalid);
  const float sum = state[c] + value;
  state[c] = sum;
  invalid = invalid || !isfinite(sum);
  if (invalid) atomic_store_explicit(health, kNotFinite, memory_order_relaxed);
}
