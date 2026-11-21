const uint index = thread_position_in_grid.x;
if (index >= uint(n_items)) return;
const uint base = (index / WIDTH) * (PICKS * WIDTH) + index % WIDTH;
float acc = 0.0f;
for (uint slot = 0; slot < PICKS; ++slot) {
    acc += float(bfloat16_t(projected[base + slot * WIDTH]));
}
result[index] = float(bfloat16_t(acc));
