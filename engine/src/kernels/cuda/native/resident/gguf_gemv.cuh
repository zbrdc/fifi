// Original packed GGUF weights with F32 activations. The block tables below
// use the same byte layout as weights/gguf.rs and weights/gguf_dequant.rs.
#pragma once

#include "resident/iq_tables.cuh"
#include "resident/k4_scales.cuh"

#define FIFI_GGUF_GEMV_TYPES(X)  \
    X(2, 32, 18)                \
    X(6, 32, 22)                \
    X(8, 32, 34)                \
    X(10, 256, 84)              \
    X(11, 256, 110)             \
    X(12, 256, 144)             \
    X(13, 256, 176)             \
    X(14, 256, 210)             \
    X(16, 256, 66)              \
    X(17, 256, 74)              \
    X(18, 256, 98)              \
    X(19, 256, 50)              \
    X(20, 32, 18)               \
    X(21, 256, 110)             \
    X(22, 256, 82)              \
    X(23, 256, 136)             \
    X(29, 256, 56)              \
    X(39, 32, 17)               \
    X(42, 64, 18)

namespace fifi_gguf {

constexpr int kDtypeBase = 256;

inline size_t row_bytes(int dtype, int columns) {
    if (dtype < kDtypeBase) return 0;
    switch (dtype - kDtypeBase) {
#define ROW_BYTES(id, values, bytes) case id: return columns % values == 0 ? (columns / values) * bytes : 0;
        FIFI_GGUF_GEMV_TYPES(ROW_BYTES)
#undef ROW_BYTES
    default:
        return 0;
    }
}

__device__ __forceinline__ unsigned read16(const uint8_t *p) {
    return p[0] | (unsigned(p[1]) << 8);
}

__device__ __forceinline__ unsigned read32(const uint8_t *p) {
    return read16(p) | (read16(p + 2) << 16);
}

__device__ __forceinline__ float half_value(const uint8_t *p) {
    return __half2float(__ushort_as_half(read16(p)));
}

__device__ __forceinline__ unsigned nibble(unsigned value, int upper) {
    return (value >> (4 * upper)) & 15u;
}

// kvalues_iq4nl[index] from four words of immediates: a select and a byte extract instead of a
// load from the table in device memory, which every decoded weight would wait on. The words are
// the table's bytes, so the values are the table's.
__device__ __forceinline__ int iq4nl_value(unsigned index) {
    const unsigned word = index < 8 ? (index < 4 ? 0xbfad9881u : 0xf6eaddcfu) : (index < 12 ? 0x26190d01u : 0x71594535u);
    return int(int8_t(word >> (8 * (index & 3u))));
}

template <typename T>
__device__ __forceinline__ float grid_value(T grid, int lane, float scale, unsigned signs) {
    const float value = scale * float((grid >> (8 * lane)) & 255u);
    return (signs & (1u << lane)) ? -value : value;
}

__device__ __forceinline__ float iq1_value(unsigned index, int lane, float scale, bool negative_delta) {
    // The four low nibbles describe lanes 0–3; the high nibbles describe lanes 4–7.
    const int code = int(nibble(iq1s_grid_gpu[index] >> (8 * (lane % 4)), lane / 4));
    return scale * (float(code - 1) + (negative_delta ? -0.125f : 0.125f));
}

struct Iq2Group {
    uint64_t grid;
    float scale;
    unsigned signs;
};

// Eight adjacent values share the codebook entry, scale, and sign mask.
template <int Type>
__device__ __forceinline__ Iq2Group iq2_group(const uint8_t *b, int row, int segment, float d) {
    if constexpr (Type == 16) {
        const unsigned a = read32(b + 6 + 8 * row);
        const float scale = d * (0.5f + float(a >> 28)) * 0.25f;
        return {iq2xxs_grid[b[2 + 8 * row + segment]], scale,
                ksigns_iq2xs[(a >> (7 * segment)) & 127]};
    } else if constexpr (Type == 17) {
        const unsigned code = read16(b + 2 + 2 * (4 * row + segment));
        const float scale = d * (0.5f + float(nibble(b[66 + row], segment / 2))) * 0.25f;
        return {iq2xs_grid[code & 511], scale, ksigns_iq2xs[code >> 9]};
    } else {
        static_assert(Type == 22, "not an IQ2 grid type");
        const unsigned hi = (unsigned(b[66 + row]) << (8 - 2 * segment)) & 768;
        const unsigned code = b[2 + 4 * row + segment] | hi;
        const float scale = d * (0.5f + float(nibble(b[74 + row], segment / 2))) * 0.25f;
        return {iq2s_grid[code], scale, b[34 + 4 * row + segment]};
    }
}

template <int Type>
__device__ __forceinline__ float decode(const uint8_t *b, int i) {
    if constexpr (Type == 8) {
        return half_value(b) * float(int8_t(b[2 + i]));
    } else if constexpr (Type == 2 || Type == 6) {
        const int start = Type == 2 ? 2 : 6;
        const int upper = Type == 2 ? 0 : ((b[2 + i / 8] >> (i % 8)) & 1) * 16;
        return half_value(b) * float(int(nibble(b[start + i % 16], i / 16)) + upper - (Type == 2 ? 8 : 16));
    } else if constexpr (Type == 42) {
        return half_value(b) * float(int((b[2 + i / 4] >> (2 * (i % 4))) & 3) - 1);
    } else if constexpr (Type == 39) {
        const unsigned e = b[0];
        const float scale = __uint_as_float(e < 2 ? (0x00200000u << e) : (e - 1) << 23);
        return scale * float(kvalues_fp4[nibble(b[1 + i % 16], i / 16)]);
    } else if constexpr (Type == 10) {
        const unsigned sc = b[i / 16];
        const unsigned q = (b[16 + 32 * (i / 128) + i % 32] >> (2 * ((i % 128) / 32))) & 3;
        return __fsub_rn((half_value(b + 80) * float(sc & 15)) * float(q),
                         half_value(b + 82) * float(sc >> 4));
    } else if constexpr (Type == 11) {
        const int group = i / 16;
        const int sc = int(nibble(b[96 + group % 8], group / 8) | (((b[104 + group % 4] >> (2 * (group / 4))) & 3) << 4)) - 32;
        const int pair = (i % 128) / 32;
        const int lo = (b[32 + 32 * (i / 128) + i % 32] >> (2 * pair)) & 3;
        const int correction = b[i % 32] & (1u << (i / 32)) ? 0 : 4;
        return (half_value(b + 108) * float(sc)) * float(lo - correction);
    } else if constexpr (Type == 12 || Type == 13) {
        const int g = i / 32;
        unsigned scale, minimum;
        scale_min_k4(b + 4, g, scale, minimum);
        const unsigned high = Type == 13 ? ((b[16 + i % 32] >> g) & 1) << 4 : 0;
        const int start = Type == 12 ? 16 : 48;
        const unsigned code = nibble(b[start + 32 * (i / 64) + i % 32], g % 2) | high;
        return __fsub_rn((half_value(b) * float(scale)) * float(code), half_value(b + 2) * float(minimum));
    } else if constexpr (Type == 14) {
        const int half = i / 128, part = (i % 128) / 32, lane = i % 32;
        const unsigned low = nibble(b[64 * half + 32 * (part % 2) + lane], part / 2);
        const unsigned hi = (b[128 + 32 * half + lane] >> (2 * part)) & 3;
        return (half_value(b + 208) * float(int8_t(b[192 + i / 16]))) * float(int(low | (hi << 4)) - 32);
    } else if constexpr (Type == 19 || Type == 29) {
        const int row = i / 32, segment = (i % 32) / 8, lane = i % 8;
        if constexpr (Type == 19) {
            const unsigned high = read16(b + 34 + 2 * row);
            const unsigned index = b[2 + 4 * row + segment] | (((high >> (3 * segment)) & 7) << 8);
            return iq1_value(index, lane, half_value(b) * float(1 + 2 * ((high >> 12) & 7)), high & 0x8000);
        } else {
            const unsigned w0 = read16(b + 48), w1 = read16(b + 50);
            const unsigned w2 = read16(b + 52), w3 = read16(b + 54);
            const unsigned base = (w0 >> 12) | ((w1 >> 8) & 0xf0u) | ((w2 >> 4) & 0xf00u) | (w3 & 0xf000u);
            const unsigned packed = read16(b + 48 + 2 * (row / 2));
            const unsigned factor = (packed >> (6 * (row % 2) + 3 * (segment / 2))) & 7;
            const unsigned high = nibble(b[32 + 2 * row + segment / 2], segment % 2);
            const unsigned index = b[4 * row + segment] | ((high & 7) << 8);
            return iq1_value(index, lane, __half2float(__ushort_as_half(base)) * float(1 + 2 * factor), high & 8);
        }
    } else if constexpr (Type == 20 || Type == 23) {
        const int row = i / 32;
        const int lane = i % 32;
        if constexpr (Type == 20) {
            return half_value(b) * float(iq4nl_value(nibble(b[2 + lane % 16], lane / 16)));
        } else {
            const unsigned lower = nibble(b[4 + row / 2], row % 2);
            const unsigned higher = (read16(b + 2) >> (2 * row)) & 3;
            const float scale = half_value(b) * float(int(lower | (higher << 4)) - 32);
            return scale * float(iq4nl_value(nibble(b[8 + 16 * row + lane % 16], lane / 16)));
        }
    } else {
        const int row = i / 32, segment = (i % 32) / 8, lane = i % 8;
        const float d = half_value(b);
        if constexpr (Type == 16 || Type == 17 || Type == 22) {
            const auto group = iq2_group<Type>(b, row, segment, d);
            return grid_value(group.grid, lane, group.scale, group.signs);
        } else if constexpr (Type == 18) {
            const unsigned a = read32(b + 66 + 4 * row);
            const float scale = d * (0.5f + float(a >> 28)) * 0.5f;
            return grid_value(iq3xxs_grid[b[2 + i / 4]], lane % 4, scale,
                              ksigns_iq2xs[(a >> (7 * segment)) & 127] >> (4 * (lane / 4)));
        } else {
            static_assert(Type == 21, "unsupported GGUF type");
            const unsigned hi = (unsigned(b[66 + row]) << (8 - (i % 32) / 4)) & 256;
            const float scale = d * float(1 + 2 * nibble(b[106 + row / 2], row % 2));
            return grid_value(iq3s_grid[b[2 + i / 4] | hi], lane % 4, scale,
                              b[74 + 4 * row + segment] >> (4 * (lane / 4)));
        }
    }
}

}  // namespace fifi_gguf
