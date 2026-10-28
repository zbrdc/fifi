// gdn_post -- gated RMS norm and output gate of prefill/delta.rs::chunk in one kernel:
//   o = bf16( rms_norm(core) * silu(z) ),  rms_norm F32 with weight, silu in the activation dtype (bf16), product in F32.
// One simdgroup per (row, value head): 4 contiguous elements per lane, the same lane layout and accumulation order as MLX's
// rms_single_row kernel, so the sum of squares is bit-identical.
// Inputs: core [rows*32, 128] f32 (gated-delta y), z_in [rows, LDZ] bf16 (columns OFFZ ..), norm_w [128] f32, rows (int).
// Output: o [rows, 4096] bf16.   Launch: threadgroup (128,1,1), grid (ceil(rows*32/4) * 128, 1, 1).
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int idx = int(threadgroup_position_in_grid.x) * 4 + int(sg);
if (idx >= int(rows) * 32) return;
const int t = idx / 32, vh = idx % 32;
float x[4];
float acc = 0.0f;
for (int j = 0; j < 4; ++j) {
  x[j] = core[(long)idx * 128 + lane * 4 + j];
  acc += x[j] * x[j];
}
acc = simd_sum(acc);
const float normalizer = metal::precise::rsqrt(acc / 128.0f + eps);
for (int j = 0; j < 4; ++j) {
  const int d = int(lane) * 4 + j;
  const float normed = norm_w[d] * (x[j] * normalizer);
  const bfloat16_t zb = z_in[t * LDZ + OFFZ + vh * 128 + d];
  const bfloat16_t sgv = mlx_sigmoid_bf(zb);
  const bfloat16_t gate = zb * sgv;
  o_out[(long)t * 4096 + vh * 128 + d] = bfloat16_t(normed * float(gate));
}
