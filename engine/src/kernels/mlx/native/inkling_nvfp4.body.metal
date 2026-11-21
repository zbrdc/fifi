// One threadgroup per row. Stash BF16-derived values so the row maximum and
// pairwise block error both reuse the same activations.
const uint row = thread_position_in_grid.y;
const uint lane = thread_position_in_grid.x;
const uint base = row * WIDTH;
threadgroup float values[WIDTH];
threadgroup float simd_maxima[4];
float local_max = 0.0f;
for (uint col = lane; col < WIDTH; col += 128) {
    const uint i = base + col;
    const float value = GATED
        ? inkling_capped_value(projected[2 * i], projected[2 * i + 1], route_weights[row])
        : float(bfloat16_t(projected[i]));
    values[col] = value;
    if (metal::isfinite(value)) local_max = metal::max(local_max, metal::abs(value));
}
const float maximum = simd_max(local_max);
if (thread_index_in_simdgroup == 0) simd_maxima[simdgroup_index_in_threadgroup] = maximum;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float row_max = metal::max(metal::max(simd_maxima[0], simd_maxima[1]),
                                metal::max(simd_maxima[2], simd_maxima[3]));
if (row_max == 0.0f) {
    for (uint col = lane; col < WIDTH; col += 128) result[base + col] = values[col];
    return;
}
const float encoder = metal::min(1536.0f / row_max, 0x1.fffffep127f);
const float decoder = 1.0f / encoder;
const float outer = row_max * (1.0f / 1536.0f);
for (uint block = lane; block < WIDTH / 16; block += 128) {
    const uint col = block * 16;
    threadgroup const float* group = values + col;
    float block_max = 0.0f;
    for (uint j = 0; j < 16; ++j) {
        if (metal::isfinite(group[j])) block_max = metal::max(block_max, metal::abs(group[j]));
    }
    const float six = inkling_e4m3_magnitude(block_max * (encoder * (1.0f / 6.0f)));
    const float four = inkling_e4m3_magnitude(block_max * (encoder * 0.25f));
    const float inverse_six = inkling_nvfp4_inverse(six, decoder);
    const float inverse_four = inkling_nvfp4_inverse(four, decoder);
    const bool use_four = inkling_nvfp4_error(group, four, inverse_four, outer)
        < inkling_nvfp4_error(group, six, inverse_six, outer);
    const float scale = use_four ? four : six;
    const float inverse = use_four ? inverse_four : inverse_six;
    for (uint j = 0; j < 16; ++j) {
        const float x = group[j];
        result[base + col + j] = metal::isfinite(x)
            ? inkling_nvfp4_value(x, scale, inverse, outer)
            : x;
    }
}
