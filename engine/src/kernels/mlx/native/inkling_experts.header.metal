#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
#pragma clang fp contract(off)

inline float inkling_capped_value(float raw_gate, float raw_up, float weight) {
    const float gate = float(bfloat16_t(raw_gate));
    const float up = float(bfloat16_t(raw_up));
    const float capped_gate = 4.0f * metal::tanh(gate * 0.25f);
    const float capped_up = 25.0f * metal::tanh(up / 25.0f);
    const float sigmoid = 1.0f / (1.0f + metal::exp(-gate));
    return float(bfloat16_t(((capped_gate * sigmoid) * capped_up) * weight));
}

inline float inkling_e4m3_magnitude(float value) {
    const float finite = metal::min(value, 448.0f);
    const int exponent = int((as_type<uint>(finite) >> 23) & 255u) - 127;
    const int shift = metal::max(-9, exponent - 3);
    const float step = as_type<float>(uint(shift + 127) << 23);
    const float inverse_step = as_type<float>(uint(127 - shift) << 23);
    return metal::rint(finite * inverse_step) * step;
}

inline float inkling_e2m1(float value) {
    const float x = metal::abs(value);
    // At a midpoint the even E2M1 code wins.
    float magnitude;
    if (x <= 1.25f) {
        magnitude = x <= 0.25f ? 0.0f : (x < 0.75f ? 0.5f : 1.0f);
    } else if (x < 3.5f) {
        magnitude = x < 1.75f ? 1.5f : (x <= 2.5f ? 2.0f : 3.0f);
    } else {
        magnitude = x <= 5.0f ? 4.0f : 6.0f;
    }
    return metal::copysign(magnitude, value);
}

inline float inkling_nvfp4_inverse(float scale, float decoder) {
    return metal::min(1.0f / (scale * decoder), 0x1.fffffep127f);
}

inline float inkling_nvfp4_value(float value, float scale, float inverse, float outer) {
    return (inkling_e2m1(value * inverse) * scale) * outer;
}

inline float inkling_nvfp4_error(
    threadgroup const float* values, float scale, float inverse, float outer) {
    float error[16];
    for (uint i = 0; i < 16; ++i) {
        const float x = values[i];
        error[i] = metal::isfinite(x)
            ? metal::abs(inkling_nvfp4_value(x, scale, inverse, outer) - x)
            : 0.0f;
    }
    for (uint stride = 1; stride < 16; stride *= 2) {
        for (uint i = 0; i < 16; i += stride * 2) {
            error[i] = error[i] + error[i + stride];
        }
    }
    return error[0];
}
