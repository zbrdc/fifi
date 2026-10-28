// act_mul -- SwiGLU activation of the experts in one kernel:  h = bf16( bf16(g * sigmoid(g)) * u )  with MLX's per-op bf16 rounding:
// replaces sigmoid, multiply, multiply (three kernels, three bf16 round trips).
// Inputs: g_in, u_in [n_rows, N] bf16 with row strides LDG / LDU (so the two halves of a fused gate|up output can be passed
// without a copy), n (int rows).  Output h [n_rows, N] bf16.  Launch: grid (N/4, n_rows, 1), threadgroup (64,1,1).
const int r = int(thread_position_in_grid.y);
const int c = int(thread_position_in_grid.x) * 4;
if (r >= int(n_rows) || c >= NCOL) return;
for (int j = 0; j < 4; ++j) {
  const bfloat16_t g = g_in[(long)r * LDG + OFFG + c + j];
  const bfloat16_t u = u_in[(long)r * LDU + OFFU + c + j];
  const bfloat16_t act = g * mlx_sigmoid_bf(g);
  h_out[(long)r * NCOL + c + j] = act * u;
}
