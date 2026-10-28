// moe_combine -- the MoE combine of prefill/moe.rs::chunk_device in one kernel:
//   out[t] = bf16( sum_k gates[t][k] * eo[inverse[t*8 + k]]  +  sigmoid(raw[t]) * so[t] )
// i.e. the un-sort gather (take), the F32 gate product, the sum over the 8 choices in choice order, the shared-expert
// sigmoid factor product, the final add and the cast to the activation dtype.  No intermediate [T,8,H] F32 tensor.
// Rounding mirrors the unfused graph: bf16 -> f32 exact, each product rounded to F32, sequential sum from 0 in choice order.
// Inputs: eo [P, H] bf16 (expert outputs in SORTED pair order), inverse [P] u32, gates [T, 8] f32,
//         so [T, H] bf16 (shared-expert down projection), raw [T, 1] f32 (shared scalar logit), rows (int).
// Output: out [T, H] bf16.   Launch: threadgroup (128,1,1), grid (H/4 * 1, rows, 1) threads: thread = 4 consecutive channels.
const int t = int(thread_position_in_grid.y);
const int c = int(thread_position_in_grid.x) * 4;
if (t >= int(rows) || c >= HID) return;
float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
for (int k = 0; k < 8; ++k) {
  const float gt = gates[t * 8 + k];
  const long src = (long)inverse[t * 8 + k] * HID + c;
  for (int j = 0; j < 4; ++j) acc[j] = acc[j] + float(eo[src + j]) * gt;
}
const float fac = mlx_sigmoid_f(raw[t]);
for (int j = 0; j < 4; ++j) {
  const float sh = fac * float(so[(long)t * HID + c + j]);
  out[(long)t * HID + c + j] = bfloat16_t(acc[j] + sh);
}
