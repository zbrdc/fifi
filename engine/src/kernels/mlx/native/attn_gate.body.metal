// attn_gate -- tail of prefill/attn.rs::chunk in one kernel: the flash-attention output [1, QH, rows, D] (query dtype)
// is transposed to [rows, QH, D], multiplied by sigmoid(gate) (gate = the second half of every head row of the q projection,
// bf16, sigmoid rounded per op like MLX) and cast to the activation dtype:  o[t, h*D + d] = bf16( att[h][t][d] * sigmoid(gate[t][h][d]) ).
// Inputs: att [QH, rows, D] (TA), qp [rows, LDQ] bf16 (head h gate at column h*2D + D + d), rows (int).  Output o [rows, QH*D] bf16.
// Launch: threadgroup (64,1,1), grid (QH * D / 4, rows, 1) threads: thread = 4 consecutive d of one head.
const int t = int(thread_position_in_grid.y);
const int col = int(thread_position_in_grid.x) * 4;          // 0 .. QH*D - 4
const int h = col / D, d = col % D;
if (t >= int(rows)) return;
for (int j = 0; j < 4; ++j) {
  const float a = float(att[((long)h * rows + t) * D + d + j]);
  const bfloat16_t gb = qp[(long)t * LDQ + h * 2 * D + D + d + j];
  const float sg = float(mlx_sigmoid_bf(gb));
  o_out[(long)t * (QHN * D) + col + j] = bfloat16_t(a * sg);
}
