// gdn_prep -- ONE kernel for the whole DeltaNet pre-recurrence chain of prefill/delta.rs::chunk:
//   causal conv (ring + 3 taps + last tap) -> silu -> split q|k|v -> L2 norm of q and k -> gates g and beta -> new ring.
// Replaces ~35 MLX kernels and ~1 GB of F32 round trips per 1024-row layer chunk (see REPORT.md).
// Written for fifi, following the llama.cpp and mlx-lm gated-delta decode kernels.
//
// Rounding mirrors the unfused op sequence: every product and every sum is rounded to F32 separately
// (fp contract off), sigmoid / logaddexp / exp use the same Metal functions as MLX's unary / binary kernels;
// only the sum of squares over the 128 elements of a head is summed in a different order (simd_sum).
//
// Inputs (names fixed, the signature is generated):
//   u_in   [rows, LDU] bf16   qkv projection (columns OFFU .. OFFU + 8191), row stride LDU
//   a_in   [rows, LDA] bf16   `a` projection (32 columns at OFFA);  b_in [rows, LDB] bf16 `b` projection (OFFB)
//   qk_eps, sqrt_kd (f32 scalars: 128 * 1e-6 and sqrt(128)).
//   ring   [8192, 3]  f32     previous 3 conv inputs per channel, oldest first
//   taps   [8192, 3]  f32     past taps (oldest first);   last [8192] f32 the current-row tap
//   neg_exp_a_log [32] f32;   dt_bias [32] f32;   rows (int);  the geometry ints LDU OFFU LDA OFFA LDB OFFB are template args
// Outputs: q [rows,16,128] f32 (normalised and scaled by 1/sqrt(128)), k [rows,16,128] f32, v [rows,32,128] f32,
//          g [rows,32] f32 (decay factor), beta [rows,32] f32, ring_out [8192,3] f32.
// Launch: threadgroup (128,1,1) = 4 simdgroups = 4 consecutive heads of 128 channels; grid (16*128, ceil(rows/RB), 1) threads.
// Each simdgroup walks RB rows of its head with the 3-row conv window in registers.
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int hd = int(threadgroup_position_in_grid.x) * 4 + int(sg);          // 0..63: q 0-15, k 16-31, v 32-63
const int t0 = int(threadgroup_position_in_grid.y) * RB;
const int t1 = min(t0 + RB, int(rows));
const int c = hd * 128 + int(lane) * 4;
float lastv[4], tp0[4], tp1[4], tp2[4], w0[4], w1[4], w2[4];
for (int j = 0; j < 4; ++j) {
  const int ch = c + j;
  lastv[j] = last[ch];
  tp0[j] = taps[ch * 3 + 0]; tp1[j] = taps[ch * 3 + 1]; tp2[j] = taps[ch * 3 + 2];
  // seq[i] = ring[ch][i] for i < 3, u[i - 3][ch] afterwards
  const int i0 = t0, i1 = t0 + 1, i2 = t0 + 2;
  w0[j] = i0 < 3 ? ring[ch * 3 + i0] : float(u_in[(i0 - 3) * LDU + OFFU + ch]);
  w1[j] = i1 < 3 ? ring[ch * 3 + i1] : float(u_in[(i1 - 3) * LDU + OFFU + ch]);
  w2[j] = i2 < 3 ? ring[ch * 3 + i2] : float(u_in[(i2 - 3) * LDU + OFFU + ch]);
}
const int vh = hd - 32;
for (int t = t0; t < t1; ++t) {
  float y[4];
  float xs[4];
  for (int j = 0; j < 4; ++j) {
    const float x = float(u_in[t * LDU + OFFU + c + j]);
    xs[j] = x;
    float acc = lastv[j] * x;
    acc = acc + tp0[j] * w0[j];
    acc = acc + tp1[j] * w1[j];
    acc = acc + tp2[j] * w2[j];
    y[j] = acc * mlx_sigmoid_f(acc);
    w0[j] = w1[j]; w1[j] = w2[j]; w2[j] = x;
  }
  if (hd < 32) {
    float ss = 0.0f;
    for (int j = 0; j < 4; ++j) ss += y[j] * y[j];
    ss = simd_sum(ss);
    ss = ss + qk_eps;
    float den = metal::precise::sqrt(ss);
    if (hd < 16) {
      den = den * sqrt_kd;
      device float* o = q_out + ((long)t * 16 + hd) * 128 + lane * 4;
      for (int j = 0; j < 4; ++j) o[j] = y[j] / den;
    } else {
      device float* o = k_out + ((long)t * 16 + (hd - 16)) * 128 + lane * 4;
      for (int j = 0; j < 4; ++j) o[j] = y[j] / den;
    }
  } else {
    device float* o = v_out + ((long)t * 32 + vh) * 128 + lane * 4;
    for (int j = 0; j < 4; ++j) o[j] = y[j];
    if (lane == 0) {
      const float pre = float(a_in[t * LDA + OFFA + vh]) + dt_bias[vh];
      const float soft = mlx_logaddexp_f(pre, 0.0f);
      const float dec = neg_exp_a_log[vh] * soft;
      g_out[t * 32 + vh] = metal::precise::exp(dec);
      beta_out[t * 32 + vh] = mlx_sigmoid_f(float(b_in[t * LDB + OFFB + vh]));
    }
  }
}
// the block that holds the last row writes the new ring (seq[rows .. rows + 2])
if (t0 <= int(rows) - 1 && int(rows) - 1 < t0 + RB) {
  for (int j = 0; j < 4; ++j) {
    const int ch = c + j;
    for (int i = 0; i < 3; ++i) {
      const int s = int(rows) + i;
      ring_out[ch * 3 + i] = s < 3 ? ring[ch * 3 + s] : float(u_in[(s - 3) * LDU + OFFU + ch]);
    }
  }
}
