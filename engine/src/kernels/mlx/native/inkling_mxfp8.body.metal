// Width is a multiple of the 32-lane SIMD group.
const uint index = thread_position_in_grid.x;
if (index >= uint(n_items)) return;
const float value = GATED
    ? inkling_capped_value(projected[2 * index], projected[2 * index + 1],
                           route_weights[index / WIDTH])
    : float(bfloat16_t(projected[index]));
const bool finite = metal::isfinite(value);
const float amax = simd_max(finite ? metal::abs(value) : 0.0f);
const float raw = amax * (1.0f / 448.0f);
const uint exponent = raw <= 0x1.0p-127f ? 0u
    : metal::min(((as_type<uint>(raw) + 0x007fffffu) & 0x7f800000u) >> 23, 254u);
const float scale = as_type<float>(exponent == 0 ? 0x00400000u : exponent << 23);
const float inverse = as_type<float>(exponent == 254 ? 0x00400000u : (254u - exponent) << 23);
result[index] = finite
    ? metal::copysign(inkling_e4m3_magnitude(metal::abs(value * inverse)) * scale, value)
    : value;
