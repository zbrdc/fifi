// The soft cap of a head's logits, in place: x[i] = cap * tanh(x[i] / cap) for i below n, the
// rule of the CUDA adapter's in-place cap. One thread a float. A NaN or an infinity is left as
// it is: a caller that validates the raw projection reads them back. A zero is left as it is
// too: it is a fixed point of the cap, and the library's tanh gives +0 for -0 on this device
// (M4 Max, 2026-11-27), where the sign of a zero logit stays on CUDA.
//
// The shader computes tanh itself. The library's is (exp(2t) - 1)/(exp(2t) + 1) with a fast
// exp: on this device 1.5e-4 of relative error below 1e-3, 1.9e-5 below 1e-2, and NaN from
// 44.4 on, where the CUDA adapter's frozen bound is 2e-5. cap_tanh is in three parts, measured
// on 22,001 arguments from -45 to 45 (2026-11-27): at most 1.3e-7 of relative error over all.
//
// The layout of SoftcapParams is the layout of the Rust parameter of resident/softcap.rs.

#include <metal_stdlib>
using namespace metal;

struct SoftcapParams {
  uint n;
  float cap;
};

// Whether a float is an infinity or a NaN, on its bits.
static inline bool nonfinite(float v) {
  return (as_type<uint>(v) & 0x7f800000u) == 0x7f800000u;
}

// The odd Taylor series of tanh through t^13, in u = t*t: at 0.5 the rest is 4.03e-8, less below.
static inline float tanh_series(float t) {
  const float u = t * t;
  float p = 0.00359212804f;
  p = -0.00886323553f + u * p;
  p = 0.0218694885f + u * p;
  p = -0.0539682540f + u * p;
  p = 0.133333333f + u * p;
  p = -0.333333333f + u * p;
  return t * (1.0f + u * p);
}

// tanh for the cap: the series below 0.5, the exp form from 0.5 to 10 (at most 1.3e-7), and
// from 10 on 1 with the sign of t, the f32 result to the last bit (tanh(10) is 1 less 4.1e-9).
static inline float cap_tanh(float t) {
  const float a = fabs(t);
  if (a >= 10.0f) return copysign(1.0f, t);
  if (a < 0.5f) return tanh_series(t);
  const float e = exp(2.0f * t);
  return (e - 1.0f) / (e + 1.0f);
}

kernel void softcap_in_place(device float *x [[buffer(0)]],
                             constant SoftcapParams &p [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  const float v = x[i];
  if (nonfinite(v) || v == 0.0f) return;
  x[i] = p.cap * cap_tanh(v / p.cap);
}
