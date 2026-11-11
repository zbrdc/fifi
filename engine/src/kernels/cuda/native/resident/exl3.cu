/* In-house arithmetic decoder of the admitted EXL3 format; no lookup tables. */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <atomic>
#include <cstdlib>
#include <stddef.h>
#include <stdint.h>
#include "../abi.h"
#include "../device.h"
#include "exl3.h"

namespace {
bool geometry(size_t k, size_t n, int bits, int book) {
    if (k < 128 || n < 128 || k > (1u << 22) || n > (1u << 22) ||
        k % 128 || n % 128 || k * n > (1ULL << 32) || book < 0 || book > 2) return false;
    return (bits >= 2 && bits <= 16 && bits % 2 == 0) ||
        (book == 2 && (bits == 3 || bits == 5 || bits == 7));
}
bool region(const void *p, size_t bytes, size_t need, size_t alignment) {
    return p && bytes >= need && uintptr_t(p) % alignment == 0 &&
        need - 1 <= UINTPTR_MAX - uintptr_t(p);
}
bool apart(const void *a, size_t ab, const void *b, size_t bb) {
    const uintptr_t x = uintptr_t(a), y = uintptr_t(b);
    return x <= y ? ab <= y - x : bb <= x - y;
}

__device__ __forceinline__ half decode_state(unsigned state, int book) {
    unsigned x;
    if (book == 2) {
        x = state * 0x83dcd12du;
        const unsigned sum = (x & 255u) + ((x >> 8) & 255u) +
            ((x >> 16) & 255u) + (x >> 24);
        // The offset+byte sum is an exactly representable F16 integer.
        const half value = __hadd(__ushort_as_half(0x6400), __uint2half_rn(sum));
        return __hfma(value, __ushort_as_half(0x1eee), __ushort_as_half(0xc931));
    }
    x = book == 0 ? state * 89226354u + 64248484u : state * 0xcbac1fedu;
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    return __hadd(__ushort_as_half(static_cast<unsigned short>(x)),
                  __ushort_as_half(static_cast<unsigned short>(x >> 16)));
}

// Book 2 (MUL1) through one packed byte dot product: the opt-in DP4A option of the id-indexed
// launch (`inference_cuda_exl3_grouped_indexed_option`); every other kernel keeps decode_state.
// decode_state sums the four bytes of x and adds the F16 1024.0 (0x6400). The sum is at most
// 4 * 255 = 1020, so 1024 + sum lies in [1024, 2044], where F16 values are the integers: the
// result is exact and its bits are 0x6400 + sum. __dp4a with unsigned operands multiplies each
// byte of x, as an unsigned value, by 1 and adds 0x6400, giving those bits in one instruction;
// the top 16 bits of the result are 0. The scale and bias FMA is decode_state's.
__device__ __forceinline__ half decode_book2_dp4a(unsigned state) {
    const unsigned x = state * 0x83dcd12du;
    const unsigned bits = __dp4a(x, 0x01010101u, 0x6400u);
    const half value = __ushort_as_half(static_cast<unsigned short>(bits));
    return __hfma(value, __ushort_as_half(0x1eee), __ushort_as_half(0xc931));
}

__device__ __forceinline__ unsigned word32(const uint16_t *tile, unsigned index) {
    return unsigned(tile[2 * index]) | (unsigned(tile[2 * index + 1]) << 16);
}

__device__ __forceinline__ half decode_element(const uint16_t *tile, unsigned row,
                                               unsigned col, int bits, int book) {
    const unsigned p = 32 * (col % 8) + 8 * ((row % 8) / 2) + row % 2 +
        2 * (row / 8) + 4 * (col / 8);
    const unsigned words = unsigned(bits) * 4;
    const unsigned start = (((p + 1) * unsigned(bits) / 2) + words * 32 - 16) % (words * 32);
    const unsigned index = start / 32;
    const unsigned long long pair = (static_cast<unsigned long long>(word32(tile, index)) << 32) |
        word32(tile, (index + 1) % words);
    return decode_state(unsigned((pair >> (48 - start % 32)) & 65535u), book);
}

__global__ void prepare(inference_cuda_exl3_prep_args a) {
    const size_t tile = blockIdx.x;
    const size_t row = (tile / (a.n / 16)) * 16 + threadIdx.y;
    const size_t col = (tile % (a.n / 16)) * 16 + threadIdx.x;
    const uint16_t *packed = a.trellis + tile * size_t(a.bits_twice) * 8;
    a.out[row * a.n + col] = __half_as_ushort(
        decode_element(packed, threadIdx.y, threadIdx.x, a.bits_twice, a.book));
}

__device__ float finite_or_flag(float value, uint32_t *flag) {
    if (isfinite(value)) return value;
    atomicOr(flag, 1u);
    return 0.0f;
}
__device__ float rotate128(float value, float *values) {
    const unsigned lane = threadIdx.x;
    values[lane] = value;
    __syncthreads();
    for (unsigned step = 1; step < 128; step *= 2) {
        const float own = values[lane], other = values[lane ^ step];
        __syncthreads();
        values[lane] = (lane & step) ? other - own : own + other;
        __syncthreads();
    }
    return values[lane] * rsqrtf(128.0f);
}

// rotate128 for a 256-thread block: two independent 128-element groups advance in lockstep
// through the SAME butterfly network and add order per group, so each group's result is
// bit-identical to rotate128's; the barriers are CTA-wide by necessity and harmless to a group
// whose row is absent (its threads exchange a staged zero nobody reads).
__device__ float rotate128_pair(float value, float *values) {
    const unsigned lane = threadIdx.x & 127, slot = (threadIdx.x >> 7) * 128 + lane;
    values[slot] = value;
    __syncthreads();
    for (unsigned step = 1; step < 128; step *= 2) {
        const float own = values[slot], other = values[slot ^ step];
        __syncthreads();
        values[slot] = (lane & step) ? other - own : own + other;
        __syncthreads();
    }
    return values[slot] * rsqrtf(128.0f);
}
__global__ void input_rotate(inference_cuda_exl3_linear_args a) {
    __shared__ float values[128];
    const size_t index = size_t(blockIdx.x) * 128 + threadIdx.x;
    const float value = finite_or_flag(a.x[index] * a.suh[index % a.k], a.flag);
    a.xh[index] = finite_or_flag(rotate128(value, values), a.flag);
}
__global__ void project(inference_cuda_exl3_linear_args a) {
    __shared__ float x[16][16], w[16][16];
    const size_t row = size_t(blockIdx.y) * 16 + threadIdx.y;
    const size_t col = size_t(blockIdx.x) * 16 + threadIdx.x;
    float sum = 0.0f;
    for (size_t base = 0; base < a.k; base += 16) {
        x[threadIdx.y][threadIdx.x] = row < a.rows ? a.xh[row * a.k + base + threadIdx.x] : 0.0f;
        const size_t tile = (base / 16) * (a.n / 16) + blockIdx.x;
        const uint16_t *packed = a.trellis + tile * size_t(a.bits_twice) * 8;
        w[threadIdx.y][threadIdx.x] = __half2float(
            decode_element(packed, threadIdx.y, threadIdx.x, a.bits_twice, a.book));
        __syncthreads();
        #pragma unroll
        for (unsigned i = 0; i < 16; ++i) sum = fmaf(x[threadIdx.y][i], w[i][threadIdx.x], sum);
        __syncthreads();
    }
    if (row < a.rows) a.z[row * a.n + col] = finite_or_flag(sum, a.flag);
}
__host__ __device__ constexpr unsigned packed_row(unsigned p) {
    return (p & 1u) + 8 * ((p >> 1) & 1u) + 2 * ((p >> 3) & 3u);
}
__host__ __device__ constexpr unsigned state_start(unsigned p, unsigned bits) {
    const unsigned period = bits * 128;
    const unsigned start = (p + 1) * bits / 2 + period - 16;
    return start >= period ? start - period : start;
}
constexpr bool packed_layout_matches_format() {
    for (unsigned bits = 2; bits <= 16; ++bits) {
        for (unsigned p = 0; p < 256; ++p) {
            const unsigned row = packed_row(p), col = (p >> 5) + 8 * ((p >> 2) & 1u);
            const unsigned mapped = 32 * (col % 8) + 8 * ((row % 8) / 2) +
                row % 2 + 2 * (row / 8) + 4 * (col / 8);
            if (mapped != p || state_start(p, bits) !=
                ((p + 1) * bits / 2 + bits * 128 - 16) % (bits * 128)) return false;
        }
    }
    return true;
}
static_assert(packed_layout_matches_format(), "EXL3 packed-state coordinates and wrap");

// F16 bits of one trellis state for the tensor-core projection. Book 2 is decode_state with
// the byte sum done by one DP4A: 1024 + sum (sum <= 1020) is the F16 integer whose bit
// pattern is 0x6400 + sum, so the DP4A accumulator carries the offset and no integer to
// half conversion is needed. Books 0 and 1 are decode_state unchanged.
template <int Book>
__device__ __forceinline__ unsigned mma_state_bits(unsigned state) {
    if constexpr (Book == 2) {
        const unsigned value = __dp4a(state * 0x83dcd12du, 0x01010101u, 0x6400u);
        return __half_as_ushort(__hfma(__ushort_as_half(static_cast<unsigned short>(value)),
                                       __ushort_as_half(0x1eee), __ushort_as_half(0xc931)));
    } else {
        return __half_as_ushort(decode_state(state, Book));
    }
}

// Packed state p is in MMA fragment order. Thread t of a midrows CTA decodes p=2t and 2t+1:
// adjacent windows of one 64-bit span of the tile (they slide by at most 8 bits), which land
// in adjacent K rows of one column of a column-major weight tile. Everything here depends on
// t and the bit rate only, so it is computed once per thread, not once per K tile.
struct MidrowsStates {
    unsigned lead_word, tail_word;  // the span is words lead_word:tail_word of the tile
    unsigned shift0, shift1;        // right shifts of the span that expose each 16-bit state
    unsigned slot;                  // 32-bit slot of the column-major tile holding both halves
};
__host__ __device__ constexpr MidrowsStates midrows_states(unsigned thread, unsigned bits) {
    const unsigned words = bits * 4, period = bits * 128, p = 2 * thread;
    const unsigned start0 = state_start(p, bits), start1 = state_start(p + 1, bits);
    MidrowsStates s{};
    s.lead_word = start0 / 32;
    s.tail_word = s.lead_word + 1 == words ? 0 : s.lead_word + 1;
    unsigned offset1 = start1 + period - s.lead_word * 32;  // start1 relative to the span
    if (offset1 >= period) offset1 -= period;
    s.shift0 = 48 - start0 % 32;
    s.shift1 = 48 - offset1;
    s.slot = ((p >> 5) + 8 * ((p >> 2) & 1u)) * 8 + packed_row(p) / 2;
    return s;
}
// decode_element's start formula, written out independently of state_start for the proof below.
constexpr unsigned legacy_state_start(unsigned p, unsigned bits) {
    const unsigned words = bits * 4;
    return (((p + 1) * bits / 2) + words * 32 - 16) % (words * 32);
}
// For every bit rate and every midrows thread: the span words and shifts expose exactly the two
// 16-bit states decode_element would read (checked on a pseudo-random tile), both windows fit
// inside the 64-bit span (the second starts at most 39 bits in), the tail word wraps like
// (index + 1) % words, and the shared slots are a bijection onto the column-major tile cells
// that decode_element's (row, col) -> p layout assigns to p and p + 1.
constexpr bool midrows_states_match_format() {
    for (unsigned bits = 2; bits <= 16; ++bits) {
        const unsigned words = bits * 4;
        unsigned tile[64] = {};
        unsigned seed = bits * 2654435761u;
        for (unsigned w = 0; w < words; ++w) {
            seed = seed * 1664525u + 1013904223u;
            tile[w] = seed;
        }
        bool taken[128] = {};
        for (unsigned thread = 0; thread < 128; ++thread) {
            const MidrowsStates s = midrows_states(thread, bits);
            const unsigned p = 2 * thread;
            if (s.lead_word >= words || s.tail_word != (s.lead_word + 1) % words) return false;
            if (s.shift0 < 17 || s.shift0 > 48 || s.shift1 < 9 || s.shift1 > 48) return false;
            if (s.slot >= 128 || taken[s.slot]) return false;
            taken[s.slot] = true;
            const unsigned row = packed_row(p), col = (p >> 5) + 8 * ((p >> 2) & 1u);
            if (s.slot != (col * 16 + row) / 2) return false;
            const unsigned long long span =
                (static_cast<unsigned long long>(tile[s.lead_word]) << 32) | tile[s.tail_word];
            for (unsigned k = 0; k < 2; ++k) {
                const unsigned r = row + k;
                if (32 * (col % 8) + 8 * ((r % 8) / 2) + r % 2 + 2 * (r / 8) + 4 * (col / 8) != p + k)
                    return false;
                const unsigned start = legacy_state_start(p + k, bits), index = start / 32;
                const unsigned long long pair =
                    (static_cast<unsigned long long>(tile[index]) << 32) | tile[(index + 1) % words];
                const unsigned want = unsigned((pair >> (48 - start % 32)) & 65535u);
                const unsigned got = unsigned((span >> (k == 0 ? s.shift0 : s.shift1)) & 65535u);
                if (got != want) return false;
            }
        }
    }
    return true;
}
static_assert(midrows_states_match_format(), "EXL3 midrows two-state span, shifts and slots");

// The two decoded weights of `s` as one word, state p in the low half. `halves` are the four
// 16-bit trellis halves at lead_word and tail_word. A decoded weight is always finite; the
// guard keeps the previous kernel's flag-and-zero behavior for a nonfinite one.
template <int Book>
__device__ __forceinline__ unsigned midrows_weight_pair(const MidrowsStates &s,
                                                        const unsigned (&halves)[4], uint32_t *flag) {
    const unsigned long long span =
        (static_cast<unsigned long long>(halves[0] | (halves[1] << 16)) << 32) |
        (halves[2] | (halves[3] << 16));
    unsigned low = mma_state_bits<Book>(unsigned(span >> s.shift0) & 65535u);
    unsigned high = mma_state_bits<Book>(unsigned(span >> s.shift1) & 65535u);
    const bool low_bad = (low & 0x7c00u) == 0x7c00u, high_bad = (high & 0x7c00u) == 0x7c00u;
    if (low_bad || high_bad) {
        atomicOr(flag, 1u);
        if (low_bad) low = 0;
        if (high_bad) high = 0;
    }
    return low | (high << 16);
}

// Rows 9..144: one CTA is 64 rows x 16 columns, one warp 16 rows, K in tiles of 16.
//  * Each warp stages only its own 16 input rows; a warp with no live rows skips input
//    staging and the MMA. Rows past the end are never loaded, so their operands stay zero.
//  * The next K tile's global values are fetched before the barriers, hiding their latency.
// The MMA shape, K order and operand values match the previous body, but B is now loaded
// col_major (it was row_major). Valid rows were measured bit-identical to the previous body on
// sm_86 only; the default aarch64 sm_72 build is compile-checked and has never run. The device
// tests compare against FP64 references, not bit for bit.
//
// W warps make a CTA of 16 * W rows (the default 4 is the 64-row CTA of project_midrows). A wider
// CTA decodes each weight tile once for more rows, which is what the job-table launch uses for
// tall jobs. Only the first 128 threads decode (a tile has 256 weights, two per thread); the other
// warps stage inputs and run the MMA.
template <int Book, int W = 4>
__device__ __forceinline__ void midrows_projection(const inference_cuda_exl3_linear_args &a,
                                                   half2 *inputs2, unsigned *weights32,
                                                   float *result) {
    namespace wmma = nvcuda::wmma;
    const half *inputs = reinterpret_cast<const half *>(inputs2);
    const half *weights = reinterpret_cast<const half *>(weights32);
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned steps = unsigned(a.k / 16);
    constexpr unsigned block_rows = 16 * W;
    const size_t first_row = size_t(blockIdx.y) * block_rows + warp * 16;
    const bool live = first_row < a.rows;
    const bool decoder = W == 4 || threadIdx.x < 128;
    const MidrowsStates states =
        midrows_states(decoder ? threadIdx.x : 0u, unsigned(a.bits_twice));

    const size_t tile_halves = size_t(a.bits_twice) * 8, step_halves = size_t(a.n / 16) * tile_halves;
    const uint16_t *tile = a.trellis + size_t(blockIdx.x) * tile_halves;
    const uint16_t *lead = tile + 2 * size_t(states.lead_word);
    const uint16_t *tail = tile + 2 * size_t(states.tail_word);
    // This lane stages rows 4 apart. One running pointer and one stride keep the register
    // count low (see project_midrows); a row past the end is not loaded. A lane with no such
    // row still forms source and steps row past the end of the input: those pointers are never
    // dereferenced (formally undefined behavior). Forming none cost either a 49th register (room
    // for 9 CTAs per SM, not 10) or about 7% more static K-loop instructions, so they stay.
    const size_t lane_row = first_row + (lane >> 3);
    const float *source = a.xh + lane_row * a.k + 2 * (lane & 7);
    const unsigned row_stride = unsigned(4 * a.k);
    const bool valid0 = live && lane_row < a.rows, valid1 = live && lane_row + 4 < a.rows,
               valid2 = live && lane_row + 8 < a.rows, valid3 = live && lane_row + 12 < a.rows;

    float x[8] = {};
    unsigned halves[4] = {};
    auto fetch = [&]() {
        const float *row = source;   // may point past the input; only dereferenced under its valid guard
        if (valid0) { x[0] = row[0]; x[1] = row[1]; }
        row += row_stride;
        if (valid1) { x[2] = row[0]; x[3] = row[1]; }
        row += row_stride;
        if (valid2) { x[4] = row[0]; x[5] = row[1]; }
        row += row_stride;
        if (valid3) { x[6] = row[0]; x[7] = row[1]; }
        if (decoder) {
            halves[0] = lead[0]; halves[1] = lead[1]; halves[2] = tail[0]; halves[3] = tail[1];
        }
    };
    auto stage = [&]() {
        if (live) {
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j)
                inputs2[warp * 128 + 32 * j + lane] = __floats2half2_rn(x[2 * j], x[2 * j + 1]);
        }
        if (decoder) weights32[states.slot] = midrows_weight_pair<Book>(states, halves, a.flag);
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> sum;
    wmma::fill_fragment(sum, 0.0f);
    fetch();
    #pragma unroll 1
    for (unsigned step = 0; step < steps; ++step) {
        stage();
        if (step + 1 < steps) {
            source += 16;
            lead += step_halves;
            tail += step_halves;
            fetch();
        }
        __syncthreads();
        if (live) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> xf;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> wf;
            wmma::load_matrix_sync(xf, inputs + warp * 256, 16);
            wmma::load_matrix_sync(wf, weights, 16);
            wmma::mma_sync(sum, xf, wf, sum);
        }
        __syncthreads();
    }
    if (live) wmma::store_matrix_sync(result + warp * 256, sum, 16, wmma::mem_row_major);
    __syncthreads();
    for (unsigned i = threadIdx.x; i < 256 * W; i += 32 * W) {
        const size_t row = size_t(blockIdx.y) * block_rows + i / 16;
        if (row < a.rows) {
            a.z[row * a.n + size_t(blockIdx.x) * 16 + i % 16] = finite_or_flag(result[i], a.flag);
        }
    }
}
// Opt-in hi/lo variant of midrows_projection (FIFI_EXL3_TALL_HILO, 64 rows x 16 columns per CTA, W = 4): the
// F32 activation x is staged as two F16 tiles, hi = F16(x) and lo = F16((x - hi) * 2048), and each K step
// multiplies both against the SAME decoded weight tile (one trellis decode per tile, as before) into two F32
// accumulators; the output is sum_hi + sum_lo / 2048. The operand error falls from 2^-11 |x| to about
// 2^-22 |x|, with the range of F16 unchanged for hi and kept for lo by the scale. The default kernel above
// is untouched; this body is a copy with the staging and the accumulators changed.
template <int Book>
__device__ __forceinline__ void midrows_projection_hilo(const inference_cuda_exl3_linear_args &a,
                                                        half2 *inputs2, half2 *inputs2_lo,
                                                        unsigned *weights32, float *result) {
    constexpr int W = 4;
    namespace wmma = nvcuda::wmma;
    const half *inputs = reinterpret_cast<const half *>(inputs2);
    const half *inputs_lo = reinterpret_cast<const half *>(inputs2_lo);
    const half *weights = reinterpret_cast<const half *>(weights32);
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned steps = unsigned(a.k / 16);
    constexpr unsigned block_rows = 16 * W;
    const size_t first_row = size_t(blockIdx.y) * block_rows + warp * 16;
    const bool live = first_row < a.rows;
    const bool decoder = W == 4 || threadIdx.x < 128;
    const MidrowsStates states =
        midrows_states(decoder ? threadIdx.x : 0u, unsigned(a.bits_twice));

    const size_t tile_halves = size_t(a.bits_twice) * 8, step_halves = size_t(a.n / 16) * tile_halves;
    const uint16_t *tile = a.trellis + size_t(blockIdx.x) * tile_halves;
    const uint16_t *lead = tile + 2 * size_t(states.lead_word);
    const uint16_t *tail = tile + 2 * size_t(states.tail_word);
    // This lane stages rows 4 apart. One running pointer and one stride keep the register
    // count low (see project_midrows); a row past the end is not loaded. A lane with no such
    // row still forms source and steps row past the end of the input: those pointers are never
    // dereferenced (formally undefined behavior). Forming none cost either a 49th register (room
    // for 9 CTAs per SM, not 10) or about 7% more static K-loop instructions, so they stay.
    const size_t lane_row = first_row + (lane >> 3);
    const float *source = a.xh + lane_row * a.k + 2 * (lane & 7);
    const unsigned row_stride = unsigned(4 * a.k);
    const bool valid0 = live && lane_row < a.rows, valid1 = live && lane_row + 4 < a.rows,
               valid2 = live && lane_row + 8 < a.rows, valid3 = live && lane_row + 12 < a.rows;

    float x[8] = {};
    unsigned halves[4] = {};
    auto fetch = [&]() {
        const float *row = source;   // may point past the input; only dereferenced under its valid guard
        if (valid0) { x[0] = row[0]; x[1] = row[1]; }
        row += row_stride;
        if (valid1) { x[2] = row[0]; x[3] = row[1]; }
        row += row_stride;
        if (valid2) { x[4] = row[0]; x[5] = row[1]; }
        row += row_stride;
        if (valid3) { x[6] = row[0]; x[7] = row[1]; }
        if (decoder) {
            halves[0] = lead[0]; halves[1] = lead[1]; halves[2] = tail[0]; halves[3] = tail[1];
        }
    };
    auto stage = [&]() {
        if (live) {
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                // x = hi + lo / 2048: hi is the F16 cast, the difference is exact in F32 (hi and x
                // are within a factor of two), the scale is a power of two, so lo keeps its own 11
                // bits whatever the size of x (no underflow of a small residual).
                const half2 hi = __floats2half2_rn(x[2 * j], x[2 * j + 1]);
                const float2 back = __half22float2(hi);
                inputs2[warp * 128 + 32 * j + lane] = hi;
                inputs2_lo[warp * 128 + 32 * j + lane] =
                    __floats2half2_rn((x[2 * j] - back.x) * 2048.0f, (x[2 * j + 1] - back.y) * 2048.0f);
            }
        }
        if (decoder) weights32[states.slot] = midrows_weight_pair<Book>(states, halves, a.flag);
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> sum, sum_lo;
    wmma::fill_fragment(sum, 0.0f);
    wmma::fill_fragment(sum_lo, 0.0f);
    fetch();
    #pragma unroll 1
    for (unsigned step = 0; step < steps; ++step) {
        stage();
        if (step + 1 < steps) {
            source += 16;
            lead += step_halves;
            tail += step_halves;
            fetch();
        }
        __syncthreads();
        if (live) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> xf;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> wf;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> xl;
            wmma::load_matrix_sync(xf, inputs + warp * 256, 16);
            wmma::load_matrix_sync(xl, inputs_lo + warp * 256, 16);
            wmma::load_matrix_sync(wf, weights, 16);
            wmma::mma_sync(sum, xf, wf, sum);
            wmma::mma_sync(sum_lo, xl, wf, sum_lo);
        }
        __syncthreads();
    }
    if (live) {
        // The two accumulators are separate so the small lo terms are not lost to the alignment of a large
        // running sum; they meet here, in F32, with the exact power-of-two scale.
        #pragma unroll
        for (int i = 0; i < sum.num_elements; ++i) sum.x[i] += sum_lo.x[i] * (1.0f / 2048.0f);
        wmma::store_matrix_sync(result + warp * 256, sum, 16, wmma::mem_row_major);
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < 256 * W; i += 32 * W) {
        const size_t row = size_t(blockIdx.y) * block_rows + i / 16;
        if (row < a.rows) {
            a.z[row * a.n + size_t(blockIdx.x) * 16 + i % 16] = finite_or_flag(result[i], a.flag);
        }
    }
}
// The launcher fixes 128 threads per CTA. Registers are not bounded here: ptxas (nvcc 12.6, sm_86
// and sm_89) allocates 47 without spills in the harness build and 48 in the production build, so
// 10 CTAs fit an SM and the occupancy is unchanged from the previous body. That is a measurement,
// not a guarantee: a (128, 10) bound was tried and made ptxas recompute the hoisted
// midrows_states inside the K loop, so the plain bound stays and the allocation is rechecked
// whenever this kernel changes.
__global__ void __launch_bounds__(128) project_midrows(inference_cuda_exl3_linear_args a) {
    __shared__ __align__(32) half2 inputs2[512];
    __shared__ __align__(32) unsigned weights32[128];
    __shared__ __align__(32) float result[1024];
    // The launcher admits books 0..2 only.
    switch (a.book) {
        case 0: midrows_projection<0>(a, inputs2, weights32, result); break;
        case 1: midrows_projection<1>(a, inputs2, weights32, result); break;
        default: midrows_projection<2>(a, inputs2, weights32, result); break;
    }
}
// Resources of this kernel (nvcc 12.6, sm_86, production build, measured with cuobjdump --dump-resource-usage): 63 registers, no spills,
// 8,704 bytes of static shared memory, so 8 CTAs of 128 threads fit an SM where project_midrows fits 10 (48 registers). Recheck whenever
// this kernel changes (project_midrows records the same warning about a tighter launch bound).
__global__ void __launch_bounds__(128) project_midrows_hilo(inference_cuda_exl3_linear_args a) {
    __shared__ __align__(32) half2 inputs2[512];
    __shared__ __align__(32) half2 inputs2_lo[512];
    __shared__ __align__(32) unsigned weights32[128];
    __shared__ __align__(32) float result[1024];
    switch (a.book) {
        case 0: midrows_projection_hilo<0>(a, inputs2, inputs2_lo, weights32, result); break;
        case 1: midrows_projection_hilo<1>(a, inputs2, inputs2_lo, weights32, result); break;
        default: midrows_projection_hilo<2>(a, inputs2, inputs2_lo, weights32, result); break;
    }
}
__global__ void project_small(inference_cuda_exl3_linear_args a, uint32_t split_k) {
    __shared__ float partial[8][16];
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned words = unsigned(a.bits_twice) * 4;
    const size_t begin = (a.k / 16) * blockIdx.z / split_k;
    const size_t end = (a.k / 16) * (blockIdx.z + 1) / split_k;
    float sums[8] = {};
    for (size_t tile = begin + warp; tile < end; tile += 8) {
        const uint16_t *packed = a.trellis +
            (tile * (a.n / 16) + blockIdx.x) * size_t(a.bits_twice) * 8;
        const unsigned low = lane < words ? word32(packed, lane) : 0;
        const unsigned high = lane + 32 < words ? word32(packed, lane + 32) : 0;
        const float x = a.xh[size_t(blockIdx.y) * a.k + tile * 16 + packed_row(lane)];
        #pragma unroll
        for (unsigned col = 0; col < 8; ++col) {
            const unsigned start = state_start(lane + 32 * col, unsigned(a.bits_twice));
            const unsigned index = start / 32, next = index + 1 == words ? 0 : index + 1;
            const unsigned first_low = __shfl_sync(0xffffffffu, low, index & 31u);
            const unsigned first_high = __shfl_sync(0xffffffffu, high, index & 31u);
            const unsigned next_low = __shfl_sync(0xffffffffu, low, next & 31u);
            const unsigned next_high = __shfl_sync(0xffffffffu, high, next & 31u);
            const unsigned first = index < 32 ? first_low : first_high;
            const unsigned second = next < 32 ? next_low : next_high;
            const unsigned long long pair = (static_cast<unsigned long long>(first) << 32) | second;
            const unsigned state = unsigned((pair >> (48 - start % 32)) & 65535u);
            sums[col] = fmaf(x, __half2float(decode_state(state, a.book)), sums[col]);
        }
    }
    // Bit2 selects the output half-tile; the other lane bits span K.
    #pragma unroll
    for (unsigned col = 0; col < 8; ++col) {
        #pragma unroll
        for (unsigned bit = 0; bit < 5; ++bit) {
            if (bit != 2) sums[col] += __shfl_xor_sync(0xffffffffu, sums[col], 1u << bit);
        }
        if ((lane & ~4u) == 0) partial[warp][col + (lane & 4u ? 8 : 0)] = sums[col];
    }
    __syncthreads();
    if (threadIdx.x < 16) {
        float sum = 0.0f;
        #pragma unroll
        for (unsigned w = 0; w < 8; ++w) sum += partial[w][threadIdx.x];
        float *target = blockIdx.z + 1 == split_k ? a.out :
            a.z + size_t(blockIdx.z) * a.rows * a.n;
        target[size_t(blockIdx.y) * a.n + size_t(blockIdx.x) * 16 + threadIdx.x] =
            finite_or_flag(sum, a.flag);
    }
}
__global__ void output_rotate(inference_cuda_exl3_linear_args a, uint32_t split_k) {
    __shared__ float values[128];
    const size_t index = size_t(blockIdx.x) * 128 + threadIdx.x;
    const size_t col = index % a.n;
    // All partials are complete; fixed bank order keeps replay deterministic.
    float z = a.z[index];
    if (a.rows <= 8) {
        for (uint32_t split = 1; split < split_k; ++split) {
            const float partial = split + 1 == split_k ? a.out[index] :
                a.z[size_t(split) * a.rows * a.n + index];
            z = finite_or_flag(z + partial, a.flag);
        }
    }
    const float rotated = rotate128(z, values);
    const float value = rotated * a.svh[col] + (a.bias ? a.bias[col] : 0.0f);
    // Fused publish: the roundtrip the separate publish kernel applies,
    // at the same point in the value's life, before the finite check.
    const float stored = a.fused_publish ?
        __bfloat162float(__float2bfloat16_rn(value)) : value;
    a.out[index] = finite_or_flag(stored, a.flag);
}

bool linear_views(const inference_cuda_exl3_linear_args &a, uint32_t split_k) {
    if (a.fused_publish > 1) return false;
    const size_t packed = a.k * a.n / 256 * size_t(a.bits_twice) * 16;
    const size_t input = a.rows * a.k * 4, output = a.rows * a.n * 4;
    const size_t z = output * (a.rows <= 8 ? split_k - 1 : 1);
    struct View { const void *p; size_t bytes, need, alignment; bool write; };
    const View views[] = {
        {a.trellis,a.trellis_bytes,packed,2,false}, {a.suh,a.suh_bytes,a.k*4,4,false},
        {a.svh,a.svh_bytes,a.n*4,4,false}, {a.bias,a.bias_bytes,a.bias?a.n*4:0,4,false},
        {a.x,a.x_bytes,input,4,false}, {a.xh,a.xh_bytes,input,4,true},
        {a.z,a.z_bytes,z,4,true}, {a.out,a.out_bytes,output,4,true},
        {a.flag,a.flag_bytes,4,4,true},
    };
    if (!a.bias && a.bias_bytes) return false;
    for (size_t i=0; i<9; ++i) {
        if (!views[i].need) continue;
        if (!region(views[i].p,views[i].bytes,views[i].need,views[i].alignment)) return false;
        for (size_t j=0; j<i; ++j) {
            if (views[j].need && (views[i].write || views[j].write) &&
                !apart(views[i].p,views[i].need,views[j].p,views[j].need)) return false;
        }
    }
    return true;
}
} // namespace

extern "C" int inference_cuda_exl3_prep(const inference_cuda_exl3_prep_args *a,
                                       void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!a || !stream || !geometry(a->k, a->n, a->bits_twice, a->book))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const size_t packed = a->k * a->n / 256 * size_t(a->bits_twice) * 16;
        const size_t dense = a->k * a->n * 2;
        if (!region(a->trellis, a->trellis_bytes, packed, 2) ||
            !region(a->out, a->out_bytes, dense, 2) || !apart(a->trellis, packed, a->out, dense))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        prepare<<<unsigned(a->k * a->n / 256), dim3(16, 16), 0, (cudaStream_t)stream>>>(*a);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

enum class LinearKind : uint32_t { Input, Small, Midrows, Large, Output };
struct LinearEmission {
    LinearKind kind;
    uint32_t gx, gy, gz, bx, by, bz;
};
static_assert(sizeof(LinearEmission) == 28);

static bool linear_geometry(size_t rows, size_t k, size_t n, int bits, int book,
                            uint32_t split_k, bool midrows) {
    return (split_k == 2 || split_k == 4 || split_k == 8) && geometry(k, n, bits, book) &&
        rows >= 1 && rows <= 65536 && (!midrows || (rows >= 9 && rows <= 144)) &&
        rows * k <= (1ULL << 32) && rows * n <= (1ULL << 32);
}

template<class Emit>
static inline __attribute__((always_inline)) bool emit_linear(
    size_t rows, size_t k, size_t n, uint32_t split_k, bool midrows, Emit &&emit) {
    const LinearEmission input{LinearKind::Input, uint32_t(rows * k / 128), 1, 1, 128, 1, 1};
    const LinearEmission projection = midrows ?
        LinearEmission{LinearKind::Midrows, uint32_t(n / 16), uint32_t((rows + 63) / 64), 1, 128, 1, 1} :
        rows <= 8 ? LinearEmission{LinearKind::Small, uint32_t(n / 16), uint32_t(rows), split_k, 256, 1, 1} :
        LinearEmission{LinearKind::Large, uint32_t(n / 16), uint32_t((rows + 15) / 16), 1, 16, 16, 1};
    const LinearEmission output{LinearKind::Output, uint32_t(rows * n / 128), 1, 1, 128, 1, 1};
    return emit(input) && emit(projection) && emit(output);
}

template<bool Observe = false>
static inline __attribute__((always_inline)) int linear_launch(
    const inference_cuda_exl3_linear_args *a, uint32_t split_k,
    bool midrows, void *stream, int *out_stage, uint64_t *launches = nullptr, bool hilo = false) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!a || !stream ||
            !linear_geometry(a->rows, a->k, a->n, a->bits_twice, a->book, split_k, midrows) ||
            !linear_views(*a, split_k)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        emit_linear(a->rows, a->k, a->n, split_k, midrows, [&](const LinearEmission &e) {
            const dim3 grid(e.gx, e.gy, e.gz), block(e.bx, e.by, e.bz);
            switch (e.kind) {
                case LinearKind::Input: input_rotate<<<grid,block,0,(cudaStream_t)stream>>>(*a); break;
                case LinearKind::Small: project_small<<<grid,block,0,(cudaStream_t)stream>>>(*a, split_k); break;
                case LinearKind::Midrows:
                    if (hilo) project_midrows_hilo<<<grid,block,0,(cudaStream_t)stream>>>(*a);
                    else project_midrows<<<grid,block,0,(cudaStream_t)stream>>>(*a);
                    break;
                case LinearKind::Large: project<<<grid,block,0,(cudaStream_t)stream>>>(*a); break;
                case LinearKind::Output: output_rotate<<<grid,block,0,(cudaStream_t)stream>>>(*a, split_k); break;
            }
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
            if constexpr (Observe) {
                if (failure.err == cudaSuccess) ++*launches;
            }
            return failure.err == cudaSuccess;
        });
        report_stage(out_stage,failure.stage);
        return int(failure.err);
    } catch (...) { return INFERENCE_CUDA_NATIVE_EXCEPTION; }
}

extern "C" int inference_cuda_exl3_linear_midrows(const inference_cuda_exl3_linear_args *a,
                                                 void *stream, int *out_stage) {
    return linear_launch<>(a, 2, true, stream, out_stage);
}

extern "C" int inference_cuda_exl3_linear_midrows_hilo(const inference_cuda_exl3_linear_args *a,
                                                      void *stream, int *out_stage) {
    return linear_launch<>(a, 2, true, stream, out_stage, nullptr, true);
}

extern "C" int inference_cuda_exl3_linear_split(const inference_cuda_exl3_linear_args *a,
                                               uint32_t split_k, void *stream, int *out_stage) {
    return linear_launch<>(a, split_k, false, stream, out_stage);
}

extern "C" int inference_cuda_exl3_linear(const inference_cuda_exl3_linear_args *a,
                                         void *stream, int *out_stage) {
    return inference_cuda_exl3_linear_split(a, 2, stream, out_stage);
}

// Only cfg(test) Rust references these sections; production links must discard them.
extern "C" __attribute__((section(".text.inference_cuda_exl3_observer"), visibility("hidden")))
int inference_cuda_exl3_linear_observed(const inference_cuda_exl3_linear_args *a,
    uint32_t split_k, uint32_t midrows, void *stream, int *out_stage, uint64_t *launches) {
    if (!launches || midrows > 1) return INFERENCE_CUDA_INVALID_ARGUMENT;
    *launches = 0;
    return linear_launch<true>(a, split_k, midrows != 0, stream, out_stage, launches);
}

struct LinearObserveArgs {
    size_t rows, k, n;
    uint32_t split_k, midrows, fail_at;
};
struct LinearObservation {
    LinearEmission emissions[3];
    uint32_t count, complete;
};
static_assert(sizeof(LinearObserveArgs) == 40 && sizeof(LinearObservation) == 92);
extern "C" __attribute__((section(".text.inference_cuda_exl3_observer"), visibility("hidden")))
int inference_cuda_exl3_observe_linear(const LinearObserveArgs *a, LinearObservation *out) {
    if (!a || !out || a->midrows > 1 || a->fail_at > 3 ||
        !linear_geometry(a->rows, a->k, a->n, 6, 0, a->split_k, a->midrows != 0))
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    *out = {};
    out->complete = emit_linear(a->rows, a->k, a->n, a->split_k, a->midrows != 0,
        [&](const LinearEmission &e) {
            out->emissions[out->count++] = e;
            return out->count != a->fail_at;
        });
    return INFERENCE_CUDA_OK;
}

namespace {
constexpr size_t grouped_capacity = 48;
// Rows of one group. The rows kernel is instantiated for this bound, so group_args and the
// launch share it.
constexpr uint32_t grouped_max_rows = 8;
struct GroupEntry {
    const uint16_t *trellis;
    const float *suh, *svh, *bias, *x;
    float *xh, *z, *out;
    int bits_twice, book;
    uint32_t first, rows;
};
struct GroupArgs {
    GroupEntry entries[grouped_capacity];
    uint8_t row_to_group[grouped_capacity];
    uint32_t *flag;
    size_t k, n;
    uint32_t pairs, groups;
    /* Shared like k/n/flag: one arm read for the whole group set. */
    uint32_t fused_publish;
    /* K slices per group; 2 is the byte-pinned default. 4/8 trade the pinned fold
     * order for more blocks in flight (Q36 fill-limited decode) and are selected by
     * the split entry, which refuses rows > 1 and cols != 1 first. */
    uint32_t split_k;
};
// By-value launch parameters need neither a device descriptor buffer nor an upload.
static_assert(sizeof(GroupEntry) == 80 && offsetof(GroupEntry, first) == 72);
static_assert(sizeof(GroupArgs) == 3928 && sizeof(GroupArgs) <= 4096 &&
    offsetof(GroupArgs, row_to_group) == 3840 && offsetof(GroupArgs, flag) == 3888 &&
    offsetof(GroupArgs, k) == 3896 && offsetof(GroupArgs, pairs) == 3912 &&
    offsetof(GroupArgs, fused_publish) == 3920);

// Keep address-taken metadata in shared launch parameters, not per-thread local copies.
__global__ void grouped_input_rotate(const __grid_constant__ GroupArgs p) {
    __shared__ float values[128];
    const GroupEntry &a = p.entries[p.row_to_group[blockIdx.y]];
    const size_t row = size_t(blockIdx.y) - a.first;
    const size_t index = row * p.k + size_t(blockIdx.x) * 128 + threadIdx.x;
    const float value = finite_or_flag(a.x[index] * a.suh[index % p.k], p.flag);
    a.xh[index] = finite_or_flag(rotate128(value, values), p.flag);
}

// Shuffle-gather grouped projection kernel: the unchanged rollback path for
// any non-enable FIFI_FLINT_CUDA_GROUPED_GATHER value (the default since
// 2026-10-28 is the table gather in grouped_project_small_gather).
__global__ void grouped_project_small(const __grid_constant__ GroupArgs p) {
    __shared__ float partial[8][16];
    const GroupEntry &a = p.entries[p.row_to_group[blockIdx.y]];
    const size_t row = size_t(blockIdx.y) - a.first;
    constexpr uint32_t split_k = 2;
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned words = unsigned(a.bits_twice) * 4;
    const size_t begin = (p.k / 16) * blockIdx.z / split_k;
    const size_t end = (p.k / 16) * (blockIdx.z + 1) / split_k;
    float sums[8] = {};
    for (size_t tile = begin + warp; tile < end; tile += 8) {
        const uint16_t *packed = a.trellis +
            (tile * (p.n / 16) + blockIdx.x) * size_t(a.bits_twice) * 8;
        const unsigned low = lane < words ? word32(packed, lane) : 0;
        const unsigned high = lane + 32 < words ? word32(packed, lane + 32) : 0;
        const float x = a.xh[row * p.k + tile * 16 + packed_row(lane)];
        #pragma unroll
        for (unsigned col = 0; col < 8; ++col) {
            const unsigned start = state_start(lane + 32 * col, unsigned(a.bits_twice));
            const unsigned index = start / 32, next = index + 1 == words ? 0 : index + 1;
            const unsigned first_low = __shfl_sync(0xffffffffu, low, index & 31u);
            const unsigned first_high = __shfl_sync(0xffffffffu, high, index & 31u);
            const unsigned next_low = __shfl_sync(0xffffffffu, low, next & 31u);
            const unsigned next_high = __shfl_sync(0xffffffffu, high, next & 31u);
            const unsigned first = index < 32 ? first_low : first_high;
            const unsigned second = next < 32 ? next_low : next_high;
            const unsigned long long pair = (static_cast<unsigned long long>(first) << 32) | second;
            const unsigned state = unsigned((pair >> (48 - start % 32)) & 65535u);
            sums[col] = fmaf(x, __half2float(decode_state(state, a.book)), sums[col]);
        }
    }
    #pragma unroll
    for (unsigned col = 0; col < 8; ++col) {
        #pragma unroll
        for (unsigned bit = 0; bit < 5; ++bit) {
            if (bit != 2) sums[col] += __shfl_xor_sync(0xffffffffu, sums[col], 1u << bit);
        }
        if ((lane & ~4u) == 0) partial[warp][col + (lane & 4u ? 8 : 0)] = sums[col];
    }
    __syncthreads();
    if (threadIdx.x < 16) {
        float sum = 0.0f;
        #pragma unroll
        for (unsigned w = 0; w < 8; ++w) sum += partial[w][threadIdx.x];
        float *target = blockIdx.z + 1 == split_k ? a.out :
            a.z + size_t(blockIdx.z) * a.rows * p.n;
        target[row * p.n + size_t(blockIdx.x) * 16 + threadIdx.x] =
            finite_or_flag(sum, p.flag);
    }
}


// Table-gather projection (FIFI_FLINT_CUDA_GROUPED_GATHER, default ON since
// 2026-10-28; any non-enable value rolls back to the shuffle kernel): gathers
// trellis words through a per-warp doubled shared table instead of four
// shuffles per column. wtab[warp][i] and [i+words] hold lane i's
// word (plus its i+32 twin for bits_twice > 8), so successor reads never wrap; the
// column geometry is tile-invariant (even rates: index advances by bits/2 per column,
// deliberately unwrapped against the duplicate table, and the window shift is constant;
// odd rates keep the generic per-column path) and the 16-bit state window is a funnel
// shift — bit-identical to the old (pair >> (48 - r)) & 0xffff for every r in [0,31].
// Same decode, fmaf and reduction order as the shuffle gather it replaces; the sweep
// pins it bitwise against the default kernel and the legacy single-call path.
template <int Book>
__device__ __forceinline__ void grouped_projection(
    const GroupArgs &p, const GroupEntry &a, unsigned *wtab_w, float (&partial)[8][16]) {
    const size_t row = size_t(blockIdx.y) - a.first;
    constexpr uint32_t split_k = 2;
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned bits = unsigned(a.bits_twice), words = bits * 4;
    const size_t begin = (p.k / 16) * blockIdx.z / split_k;
    const size_t end = (p.k / 16) * (blockIdx.z + 1) / split_k;
    float sums[8] = {};
    const unsigned start0 = state_start(lane, bits);
    const unsigned r = start0 % 32, index0 = start0 / 32;
    const bool even = (bits & 1u) == 0;
    for (size_t tile = begin + warp; tile < end; tile += 8) {
        const uint16_t *packed = a.trellis +
            (tile * (p.n / 16) + blockIdx.x) * size_t(bits) * 8;
        if (lane < words) {
            const unsigned w = word32(packed, lane);
            wtab_w[lane] = w;
            wtab_w[lane + words] = w;
        }
        if (lane + 32u < words) {
            const unsigned w = word32(packed, lane + 32u);
            wtab_w[lane + 32u] = w;
            wtab_w[lane + 32u + words] = w;
        }
        __syncwarp();
        const float x = a.xh[row * p.k + tile * 16 + packed_row(lane)];
        #pragma unroll
        for (unsigned col = 0; col < 8; ++col) {
            unsigned index, shift;
            if (even) {
                index = index0 + col * (bits >> 1);
                shift = r;
            } else {
                const unsigned start = state_start(lane + 32 * col, bits);
                index = start / 32;
                shift = start % 32;
            }
            const unsigned state =
                __funnelshift_l(wtab_w[index + 1], wtab_w[index], shift) >> 16;
            sums[col] = fmaf(x, __half2float(decode_state(state, Book)), sums[col]);
        }
        __syncwarp();
    }
    #pragma unroll
    for (unsigned col = 0; col < 8; ++col) {
        #pragma unroll
        for (unsigned bit = 0; bit < 5; ++bit) {
            if (bit != 2) sums[col] += __shfl_xor_sync(0xffffffffu, sums[col], 1u << bit);
        }
        if ((lane & ~4u) == 0) partial[warp][col + (lane & 4u ? 8 : 0)] = sums[col];
    }
    __syncthreads();
    if (threadIdx.x < 16) {
        float sum = 0.0f;
        #pragma unroll
        for (unsigned w = 0; w < 8; ++w) sum += partial[w][threadIdx.x];
        float *target = blockIdx.z + 1 == split_k ? a.out :
            a.z + size_t(blockIdx.z) * a.rows * p.n;
        target[row * p.n + size_t(blockIdx.x) * 16 + threadIdx.x] =
            finite_or_flag(sum, p.flag);
    }
}

__global__ void __launch_bounds__(256, 4) grouped_project_small_gather(
    const __grid_constant__ GroupArgs p) {
    __shared__ float partial[8][16];
    __shared__ unsigned wtab[8][128];
    const GroupEntry &a = p.entries[p.row_to_group[blockIdx.y]];
    // The launcher admits books 0..2 only (project_midrows switch pattern).
    switch (a.book) {
        case 0: grouped_projection<0>(p, a, wtab[threadIdx.x / 32], partial); break;
        case 1: grouped_projection<1>(p, a, wtab[threadIdx.x / 32], partial); break;
        default: grouped_projection<2>(p, a, wtab[threadIdx.x / 32], partial); break;
    }
}

// F32 value of one trellis state of book 0 or 1, bit-identical to
// __half2float(decode_state(state, book)): the same multiply-add, mask and xor, and the same half
// add (low half + high half, one round to nearest). The mask and xor are one LOP3 with the
// constants in registers, and the half add reads both halves of the word through operand
// swizzles, which drops the separate shift that extracts the high half.
struct DecodeConst { unsigned keep, flip; };
__device__ __forceinline__ DecodeConst decode_const() {
    DecodeConst c;
    asm("mov.b32 %0, 0x8fff8fff;" : "=r"(c.keep));
    asm("mov.b32 %0, 0x3b603b60;" : "=r"(c.flip));
    return c;
}
template <int Book>
__device__ __forceinline__ float decode_float(unsigned state, DecodeConst c) {
    static_assert(Book == 0 || Book == 1, "books 0 and 1 share this form");
    unsigned x = Book == 0 ? state * 89226354u + 64248484u : state * 0xcbac1fedu;
    asm("lop3.b32 %0, %1, %2, %3, 0x6a;" : "=r"(x) : "r"(x), "r"(c.keep), "r"(c.flip));
    const __half2 h = *reinterpret_cast<const __half2 *>(&x);
    return __low2float(__hadd2(h, __lowhigh2highlow(h)));
}

__device__ __forceinline__ unsigned load_word(const uint16_t *tile, unsigned index, bool aligned4) {
    return aligned4 ? *reinterpret_cast<const unsigned *>(tile + 2 * index) : word32(tile, index);
}

// Instruction counts quoted for these kernels are SASS instructions of the innermost tile loop per
// weight column and k step (nvcc 12.6, sm_86, cuobjdump -sass on a probe kernel that calls one
// body at a fixed book and rate; the loop with the most FFMA).
//
// One-tile body of the wide projection (Cols == 1): the same sums in the same order and the same
// reductions as grouped_projection_wide, with the loop overhead taken out. Loop-invariant values
// are computed once, the trellis and input pointers advance by constant strides, the 4-byte
// aligned and the unaligned trellis are two loop copies chosen once per block, and the neighbour
// word of the rate-8 pair table is a second load instead of a shuffle, so the loop carries no
// warp-convergent shuffle. For book 1 at rate 8 the loop is 104 instructions per tile and lane
// (115 before): the decode of the eight weights is 64 of them, the rest is loop and address work.
// Dp4a (the DP4A option of the id-indexed launch) decodes book 2 with decode_book2_dp4a; the
// other books ignore it.
template <int Book, int Bits, bool Aligned, class Args, bool Dp4a = false, bool Pipe = false>
__device__ __forceinline__ void grouped_projection_one_tile(
    const Args &p, const GroupEntry &a, unsigned (&wtab)[128], float (&partial)[8][16],
    uint32_t split_k) {
    const size_t row = size_t(blockIdx.y) - a.first;
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned bits = Bits ? unsigned(Bits) : unsigned(a.bits_twice), words = bits * 4;
    const unsigned k_tiles = unsigned(p.k / 16);
    const unsigned begin = k_tiles * blockIdx.z / split_k;
    const unsigned end = k_tiles * (blockIdx.z + 1) / split_k;
    const unsigned first = begin + warp;
    const unsigned iterations = end > first ? (end - first + 7u) / 8u : 0u;
    const size_t columns = p.n / 16;
    const size_t tile_halves = size_t(bits) * 8;
    const uint16_t *packed = a.trellis + (size_t(first) * columns + blockIdx.x) * tile_halves;
    const size_t tile_step = 8 * columns * tile_halves;
    const float *input = a.xh + row * p.k + size_t(first) * 16 + packed_row(lane);
    const DecodeConst consts = decode_const();
    const unsigned start0 = state_start(lane, bits);
    const unsigned r = start0 % 32, index0 = start0 / 32;
    const bool even = (bits & 1u) == 0;
    const unsigned next_lane = (lane + 1u) & 31u;
    float sums[8] = {};
    auto word_at = [&](const uint16_t *base, unsigned w) -> unsigned {
        if constexpr (Aligned) return *reinterpret_cast<const unsigned *>(base + 2 * w);
        else return word32(base, w);
    };
    // Pipe carries the next iteration's words and input in registers, issued before the
    // barriers of the current one, so the load latency overlaps the decode instead of stalling
    // at each __syncwarp. The words staged and the order of every fma are those of the default
    // loop, so the bytes written are too.
    unsigned low_cur = 0, next_cur = 0, high_cur = 0;
    float x_cur = 0.0f;
    if (iterations) {
        x_cur = *input;
        if constexpr (Bits == 8) {
            low_cur = word_at(packed, lane);
            next_cur = word_at(packed, next_lane);
        } else {
            if (lane < words) low_cur = word_at(packed, lane);
            if (lane + 32u < words) high_cur = word_at(packed, lane + 32u);
        }
    }
    for (unsigned it = 0; it < iterations; ++it) {
        unsigned low_nxt = 0, next_nxt = 0, high_nxt = 0;
        float x_nxt = 0.0f;
        if constexpr (Pipe) {
            if (it + 1 < iterations) {
                const uint16_t *ahead = packed + tile_step;
                x_nxt = input[128];
                if constexpr (Bits == 8) {
                    low_nxt = word_at(ahead, lane);
                    next_nxt = word_at(ahead, next_lane);
                } else {
                    if (lane < words) low_nxt = word_at(ahead, lane);
                    if (lane + 32u < words) high_nxt = word_at(ahead, lane + 32u);
                }
            }
        } else {
            if constexpr (Bits == 8) {
                low_cur = word_at(packed, lane);
                next_cur = word_at(packed, next_lane);
            } else {
                if (lane < words) low_cur = word_at(packed, lane);
                if (lane + 32u < words) high_cur = word_at(packed, lane + 32u);
            }
            x_cur = *input;
        }
        if constexpr (Bits == 8) {
            // 32 words per tile: entry i of the doubled table holds words i and i + 1, so one
            // 64-bit read gives the two words of a state window.
            const uint2 staged = make_uint2(low_cur, next_cur);
            uint2 *pair = reinterpret_cast<uint2 *>(&wtab[0]);
            pair[lane] = staged;
            pair[lane + 32] = staged;
        } else {
            if (lane < words) {
                wtab[lane] = low_cur;
                wtab[lane + words] = low_cur;
            }
            if (lane + 32u < words) {
                wtab[lane + 32u] = high_cur;
                wtab[lane + 32u + words] = high_cur;
            }
        }
        __syncwarp();
        const float x = x_cur;
        #pragma unroll
        for (unsigned col = 0; col < 8; ++col) {
            unsigned index, shift;
            if (even) {
                index = index0 + col * (bits >> 1);
                shift = r;
            } else {
                const unsigned start = state_start(lane + 32 * col, bits);
                index = start / 32;
                shift = start % 32;
            }
            unsigned state;
            if constexpr (Bits == 8) {
                const uint2 words2 = reinterpret_cast<const uint2 *>(&wtab[0])[index];
                state = __funnelshift_l(words2.y, words2.x, shift) >> 16;
            } else {
                state = __funnelshift_l(wtab[index + 1], wtab[index], shift) >> 16;
            }
            if constexpr (Book == 2 && Dp4a)
                sums[col] = fmaf(x, __half2float(decode_book2_dp4a(state)), sums[col]);
            else if constexpr (Book == 2)
                sums[col] = fmaf(x, __half2float(decode_state(state, Book)), sums[col]);
            else
                sums[col] = fmaf(x, decode_float<Book>(state, consts), sums[col]);
        }
        __syncwarp();
        low_cur = low_nxt;
        next_cur = next_nxt;
        high_cur = high_nxt;
        x_cur = x_nxt;
        packed += tile_step;
        input += 128;
    }
    #pragma unroll
    for (unsigned col = 0; col < 8; ++col) {
        #pragma unroll
        for (unsigned bit = 0; bit < 5; ++bit) {
            if (bit != 2) sums[col] += __shfl_xor_sync(0xffffffffu, sums[col], 1u << bit);
        }
        if ((lane & ~4u) == 0) partial[warp][col + (lane & 4u ? 8 : 0)] = sums[col];
    }
    __syncthreads();
    if (threadIdx.x < 16) {
        float sum = 0.0f;
        #pragma unroll
        for (unsigned w = 0; w < 8; ++w) sum += partial[w][threadIdx.x];
        float *target = blockIdx.z + 1 == split_k ? a.out :
            a.z + size_t(blockIdx.z) * a.rows * p.n;
        target[row * p.n + size_t(blockIdx.x) * 16 + threadIdx.x] = finite_or_flag(sum, p.flag);
    }
}

// Wide variant of the table-gather projection (same arithmetic, same order). One block owns
// Cols adjacent 16-column tiles instead of one. A warp walks the same k tiles as before
// (begin + warp + 8 j), and for each k tile it reads the Cols neighbouring trellis tiles, which
// are contiguous in memory, with all their loads issued before the first is consumed; the input
// value is read once for all of them. The sum of every column still starts from zero and adds
// the same products in the same order, and the cross-lane and cross-warp reductions are the
// ones of grouped_projection, so every written byte is bit-identical.
template <int Book, int Cols, int Bits, class Args>
__device__ __forceinline__ void grouped_projection_wide(
    const Args &p, const GroupEntry &a, unsigned (&wtab_w)[Cols][128],
    float (&partial)[8][16 * Cols]) {
    const size_t row = size_t(blockIdx.y) - a.first;
    constexpr uint32_t split_k = 2;
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    // A nonzero Bits is the bit rate of the matrix, known at compile time: the packed layout
    // constants (words per tile, the even-rate index step, the odd-rate branch) fold away.
    // Bits == 0 reads the rate from the entry, as the gather kernel does.
    const unsigned bits = Bits ? unsigned(Bits) : unsigned(a.bits_twice), words = bits * 4;
    const size_t begin = (p.k / 16) * blockIdx.z / split_k;
    const size_t end = (p.k / 16) * (blockIdx.z + 1) / split_k;
    const size_t first_tile = size_t(blockIdx.x) * Cols;
    float sums[Cols][8] = {};
    const bool aligned4 = (reinterpret_cast<uintptr_t>(a.trellis) & 3u) == 0;
    const DecodeConst consts = decode_const();
    const unsigned start0 = state_start(lane, bits);
    const unsigned r = start0 % 32, index0 = start0 / 32;
    const bool even = (bits & 1u) == 0;
    for (size_t tile = begin + warp; tile < end; tile += 8) {
        unsigned low[Cols], high[Cols];
        #pragma unroll
        for (int c = 0; c < Cols; ++c) {
            const uint16_t *packed = a.trellis +
                (tile * (p.n / 16) + first_tile + c) * size_t(bits) * 8;
            // A tile starts at a multiple of 16 bytes from the matrix, so a 4-byte aligned
            // matrix reads each word as one little-endian load, the value word32 assembles.
            low[c] = lane < words ? load_word(packed, lane, aligned4) : 0;
            high[c] = lane + 32u < words ? load_word(packed, lane + 32u, aligned4) : 0;
        }
        const float x = a.xh[row * p.k + tile * 16 + packed_row(lane)];
        #pragma unroll
        for (int c = 0; c < Cols; ++c) {
            if constexpr (Bits == 8) {
                // 32 words per tile: entry i of the doubled table holds words i and i + 1, so one
                // 64-bit read gives the two words of a state window.
                uint2 *pair = reinterpret_cast<uint2 *>(&wtab_w[c][0]);
                const unsigned next = __shfl_sync(0xffffffffu, low[c], (lane + 1) & 31u);
                pair[lane] = make_uint2(low[c], next);
                pair[lane + 32] = make_uint2(low[c], next);
            } else {
                if (lane < words) {
                    wtab_w[c][lane] = low[c];
                    wtab_w[c][lane + words] = low[c];
                }
                if (lane + 32u < words) {
                    wtab_w[c][lane + 32u] = high[c];
                    wtab_w[c][lane + 32u + words] = high[c];
                }
            }
        }
        __syncwarp();
        #pragma unroll
        for (int c = 0; c < Cols; ++c) {
            #pragma unroll
            for (unsigned col = 0; col < 8; ++col) {
                unsigned index, shift;
                if (even) {
                    index = index0 + col * (bits >> 1);
                    shift = r;
                } else {
                    const unsigned start = state_start(lane + 32 * col, bits);
                    index = start / 32;
                    shift = start % 32;
                }
                unsigned state;
                if constexpr (Bits == 8) {
                    const uint2 words2 = reinterpret_cast<const uint2 *>(&wtab_w[c][0])[index];
                    state = __funnelshift_l(words2.y, words2.x, shift) >> 16;
                } else {
                    state = __funnelshift_l(wtab_w[c][index + 1], wtab_w[c][index], shift) >> 16;
                }
                if constexpr (Book == 2)
                    sums[c][col] = fmaf(x, __half2float(decode_state(state, Book)), sums[c][col]);
                else
                    sums[c][col] = fmaf(x, decode_float<Book>(state, consts), sums[c][col]);
            }
        }
        __syncwarp();
    }
    #pragma unroll
    for (int c = 0; c < Cols; ++c) {
        #pragma unroll
        for (unsigned col = 0; col < 8; ++col) {
            #pragma unroll
            for (unsigned bit = 0; bit < 5; ++bit) {
                if (bit != 2) sums[c][col] += __shfl_xor_sync(0xffffffffu, sums[c][col], 1u << bit);
            }
            if ((lane & ~4u) == 0) partial[warp][16 * c + col + (lane & 4u ? 8 : 0)] = sums[c][col];
        }
    }
    __syncthreads();
    if (threadIdx.x < 16 * Cols) {
        float sum = 0.0f;
        #pragma unroll
        for (unsigned w = 0; w < 8; ++w) sum += partial[w][threadIdx.x];
        float *target = blockIdx.z + 1 == split_k ? a.out :
            a.z + size_t(blockIdx.z) * a.rows * p.n;
        target[row * p.n + first_tile * 16 + threadIdx.x] = finite_or_flag(sum, p.flag);
    }
}

// The one-tile body with the loop copy that matches the trellis alignment.
template <int Book, int Bits, class Args, bool Dp4a = false, bool Pipe = false>
__device__ __forceinline__ void one_tile_by_alignment(const Args &p, const GroupEntry &a,
                                                      unsigned (&wtab)[128],
                                                      float (&partial)[8][16],
                                                      uint32_t split_k) {
    if ((reinterpret_cast<uintptr_t>(a.trellis) & 3u) == 0)
        grouped_projection_one_tile<Book, Bits, true, Args, Dp4a, Pipe>(p, a, wtab, partial, split_k);
    else
        grouped_projection_one_tile<Book, Bits, false, Args, Dp4a, Pipe>(p, a, wtab, partial, split_k);
}

// 8 is 4.0 bpw (every projection of the Qwen3.6 pack except the head), 12 is 6.0 bpw (the
// head); any other rate runs the generic body.
template <int Book, int Cols, bool LeanAnyRate, class Args, bool Dp4a = false, bool Pipe = false>
__device__ __forceinline__ void wide_by_rate(const Args &p, const GroupEntry &a,
                                             unsigned (&wtab)[Cols][128],
                                             float (&partial)[8][16 * Cols]) {
    // The DP4A option exists for the one-tile body of the id-indexed launch only.
    static_assert(!Dp4a || (Cols == 1 && LeanAnyRate), "the DP4A decode runs one tile per block");
    if constexpr (Cols == 1) {
        switch (a.bits_twice) {
            case 8: one_tile_by_alignment<Book, 8, Args, Dp4a, Pipe>(p, a, wtab[0], partial, p.split_k); break;
            case 12: one_tile_by_alignment<Book, 12, Args, Dp4a, Pipe>(p, a, wtab[0], partial, p.split_k); break;
            default:
                // The dense kernel keeps the generic body for the rates the model does not use:
                // the lean body at an unknown rate spills there; the indexed kernel does not.
                if constexpr (LeanAnyRate)
                    one_tile_by_alignment<Book, 0, Args, Dp4a, Pipe>(p, a, wtab[0], partial, p.split_k);
                else
                    grouped_projection_wide<Book, 1, 0>(p, a, wtab, partial);
                break;
        }
    } else {
        switch (a.bits_twice) {
            case 8: grouped_projection_wide<Book, Cols, 8>(p, a, wtab, partial); break;
            case 12: grouped_projection_wide<Book, Cols, 12>(p, a, wtab, partial); break;
            default: grouped_projection_wide<Book, Cols, 0>(p, a, wtab, partial); break;
        }
    }
}

// Resident blocks per SM the one-tile wide kernels are compiled for. Without a bound the previous
// body needed 72 registers, which capped an SM at three 256-thread blocks; five blocks fit in 48
// registers and keep more weight loads in flight per SM. The one-tile body below fits in 48
// registers without local memory in both kernels (the previous body spilled 56 bytes in the
// indexed one). The wider variants keep the compiler's choice: their tile loop already holds more
// state per thread. The arithmetic and the order of every sum are unchanged, so the bytes written
// are too.
template <int Cols>
constexpr int wide_min_blocks = Cols == 1 ? 5 : 1;

template <int Cols>
__global__ void __launch_bounds__(256, wide_min_blocks<Cols>) grouped_project_small_wide(
    const __grid_constant__ GroupArgs p) {
    __shared__ float partial[8][16 * Cols];
    // 8-byte aligned: the rate-8 table is read and written as uint2 pairs.
    __shared__ __align__(8) unsigned wtab[8][Cols][128];
    const GroupEntry &a = p.entries[p.row_to_group[blockIdx.y]];
    switch (a.book) {
        case 0: wide_by_rate<0, Cols, false>(p, a, wtab[threadIdx.x / 32], partial); break;
        case 1: wide_by_rate<1, Cols, false>(p, a, wtab[threadIdx.x / 32], partial); break;
        default: wide_by_rate<2, Cols, false>(p, a, wtab[threadIdx.x / 32], partial); break;
    }
}

// Book 2 (MUL1) at rate 10 (5.0 bpw) or rate 6 (3.0 bpw), the rates of the Qwen3.8 Flash-Next dense
// projections, with the bit rate known at compile time: the one-tile body that wide_by_rate runs
// for rates 8 and 12, so the packed-layout constants (words per tile, the index step between
// columns, the doubled-table writes) fold away instead of being computed from the entry on every
// tile. The arithmetic, the order of every sum and the reductions are those of the generic wide
// body, so the bytes written are identical. It is a kernel of its own, launched only for a launch
// whose every group is book 2 at one of these rates, so the kernels the other models run are
// unchanged.
template <bool Dp4a>
__global__ void __launch_bounds__(256, wide_min_blocks<1>) grouped_project_small_wide_r56(
    const __grid_constant__ GroupArgs p) {
    __shared__ float partial[8][16];
    __shared__ unsigned wtab[8][128];
    const GroupEntry &a = p.entries[p.row_to_group[blockIdx.y]];
    if (a.bits_twice == 10)
        one_tile_by_alignment<2, 10, GroupArgs, Dp4a>(p, a, wtab[threadIdx.x / 32], partial, 2);
    else
        one_tile_by_alignment<2, 6, GroupArgs, Dp4a>(p, a, wtab[threadIdx.x / 32], partial, 2);
}

#include "exl3_decode_fused.h"

// Whether every group of a launch is book 2 at rate 10 or 6: the launches grouped_project_small_wide_r56 runs.
static bool wide_odd_rates(const GroupArgs &p) {
    for (uint32_t g = 0; g < p.groups; ++g) {
        const GroupEntry &e = p.entries[g];
        if (e.book != 2 || (e.bits_twice != 10 && e.bits_twice != 6)) return false;
    }
    return p.groups > 0;
}

// Rows kernel: one block owns one 16-column tile for all rows of a group (R is a compile-time
// bound, rows <= R), so each trellis tile is loaded, put in the table and decoded once and every
// decoded weight multiplies the R input values of the group. The wide kernel does the same work
// once per row block. The sum of every row starts from zero and adds the products of the same
// tiles in the same order, and the cross-lane and cross-warp reductions are the ones of
// grouped_projection_wide, so each row's bytes equal those of a launch of that row alone.
template <int Book, int Bits, int R, class Args>
__device__ __forceinline__ void grouped_projection_rows(
    const Args &p, const GroupEntry &a, unsigned (&wtab_w)[128], float (&partial)[8][R][16]) {
    constexpr uint32_t split_k = 2;
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned bits = Bits ? unsigned(Bits) : unsigned(a.bits_twice), words = bits * 4;
    const size_t begin = (p.k / 16) * blockIdx.z / split_k;
    const size_t end = (p.k / 16) * (blockIdx.z + 1) / split_k;
    const size_t first_tile = size_t(blockIdx.x);
    float sums[R][8] = {};
    const bool aligned4 = (reinterpret_cast<uintptr_t>(a.trellis) & 3u) == 0;
    const DecodeConst consts = decode_const();
    const unsigned start0 = state_start(lane, bits);
    const unsigned r = start0 % 32, index0 = start0 / 32;
    const bool even = (bits & 1u) == 0;
    for (size_t tile = begin + warp; tile < end; tile += 8) {
        const uint16_t *packed = a.trellis + (tile * (p.n / 16) + first_tile) * size_t(bits) * 8;
        const unsigned low = lane < words ? load_word(packed, lane, aligned4) : 0;
        const unsigned high = lane + 32u < words ? load_word(packed, lane + 32u, aligned4) : 0;
        float x[R];
        #pragma unroll
        for (int row = 0; row < R; ++row)
            x[row] = unsigned(row) < a.rows ?
                a.xh[size_t(row) * p.k + tile * 16 + packed_row(lane)] : 0.0f;
        if constexpr (Bits == 8) {
            uint2 *pair = reinterpret_cast<uint2 *>(&wtab_w[0]);
            const unsigned next = __shfl_sync(0xffffffffu, low, (lane + 1) & 31u);
            pair[lane] = make_uint2(low, next);
            pair[lane + 32] = make_uint2(low, next);
        } else {
            if (lane < words) {
                wtab_w[lane] = low;
                wtab_w[lane + words] = low;
            }
            if (lane + 32u < words) {
                wtab_w[lane + 32u] = high;
                wtab_w[lane + 32u + words] = high;
            }
        }
        __syncwarp();
        #pragma unroll
        for (unsigned col = 0; col < 8; ++col) {
            unsigned index, shift;
            if (even) {
                index = index0 + col * (bits >> 1);
                shift = r;
            } else {
                const unsigned start = state_start(lane + 32 * col, bits);
                index = start / 32;
                shift = start % 32;
            }
            unsigned state;
            if constexpr (Bits == 8) {
                const uint2 words2 = reinterpret_cast<const uint2 *>(&wtab_w[0])[index];
                state = __funnelshift_l(words2.y, words2.x, shift) >> 16;
            } else {
                state = __funnelshift_l(wtab_w[index + 1], wtab_w[index], shift) >> 16;
            }
            float weight;
            if constexpr (Book == 2)
                weight = __half2float(decode_state(state, Book));
            else
                weight = decode_float<Book>(state, consts);
            #pragma unroll
            for (int row = 0; row < R; ++row)
                sums[row][col] = fmaf(x[row], weight, sums[row][col]);
        }
        __syncwarp();
    }
    #pragma unroll
    for (int row = 0; row < R; ++row) {
        #pragma unroll
        for (unsigned col = 0; col < 8; ++col) {
            #pragma unroll
            for (unsigned bit = 0; bit < 5; ++bit) {
                if (bit != 2)
                    sums[row][col] += __shfl_xor_sync(0xffffffffu, sums[row][col], 1u << bit);
            }
            if ((lane & ~4u) == 0) partial[warp][row][col + (lane & 4u ? 8 : 0)] = sums[row][col];
        }
    }
    __syncthreads();
    const unsigned row = threadIdx.x / 16, col = threadIdx.x % 16;
    if (row < R && row < a.rows) {
        float sum = 0.0f;
        #pragma unroll
        for (unsigned w = 0; w < 8; ++w) sum += partial[w][row][col];
        float *target = blockIdx.z + 1 == split_k ? a.out :
            a.z + size_t(blockIdx.z) * a.rows * p.n;
        target[size_t(row) * p.n + first_tile * 16 + col] = finite_or_flag(sum, p.flag);
    }
}

template <int Book, int R, class Args>
__device__ __forceinline__ void rows_by_rate(const Args &p, const GroupEntry &a,
                                             unsigned (&wtab)[128], float (&partial)[8][R][16]) {
    switch (a.bits_twice) {
        case 8: grouped_projection_rows<Book, 8, R>(p, a, wtab, partial); break;
        case 12: grouped_projection_rows<Book, 12, R>(p, a, wtab, partial); break;
        default: grouped_projection_rows<Book, 0, R>(p, a, wtab, partial); break;
    }
}

// Resident blocks per SM the rows kernel is sized for: the per-row sums grow with R.
template <int R>
constexpr int rows_min_blocks = R <= 2 ? 4 : 2;

// Grid (n / 16, groups, 2): block y is a group, not a row.
template <int R>
__global__ void __launch_bounds__(256, rows_min_blocks<R>) grouped_project_rows(
    const __grid_constant__ GroupArgs p) {
    __shared__ float partial[8][R][16];
    __shared__ __align__(8) unsigned wtab[8][128];
    const GroupEntry &a = p.entries[blockIdx.y];
    switch (a.book) {
        case 0: rows_by_rate<0, R>(p, a, wtab[threadIdx.x / 32], partial); break;
        case 1: rows_by_rate<1, R>(p, a, wtab[threadIdx.x / 32], partial); break;
        default: rows_by_rate<2, R>(p, a, wtab[threadIdx.x / 32], partial); break;
    }
}

__global__ void grouped_output_rotate(const __grid_constant__ GroupArgs p) {
    __shared__ float values[128];
    const GroupEntry &a = p.entries[p.row_to_group[blockIdx.y]];
    const size_t row = size_t(blockIdx.y) - a.first;
    const size_t index = row * p.n + size_t(blockIdx.x) * 128 + threadIdx.x;
    const size_t col = index % p.n;
    float z = a.z[index];
    // Each group retains the original small-row split order, regardless of total P.
    // split_k == 2 degenerates to the historical single add (z + out), byte-identical;
    // 4/8 fold the extra k partials in the same left-to-right order as output_rotate.
    for (uint32_t split = 1; split < p.split_k; ++split) {
        const float partial = split + 1 == p.split_k ? a.out[index] :
            a.z[size_t(split) * a.rows * p.n + index];
        z = finite_or_flag(z + partial, p.flag);
    }
    const float rotated = rotate128(z, values);
    const float value = rotated * a.svh[col] + (a.bias ? a.bias[col] : 0.0f);
    // Fused publish: same roundtrip point as bf16_publish_rows.
    const float stored = p.fused_publish ?
        __bfloat162float(__float2bfloat16_rn(value)) : value;
    a.out[index] = finite_or_flag(stored, p.flag);
}

static bool follows(const void *p, const void *base, size_t bytes) {
    return uintptr_t(p) >= uintptr_t(base) && uintptr_t(p) - uintptr_t(base) == bytes;
}

static bool group_args(const inference_cuda_exl3_linear_args *calls, size_t count, uint32_t split,
                       GroupArgs &p) {
    if (!calls || count < 1 || count > grouped_capacity) return false;
    if (split != 2 && split != 4 && split != 8) return false;
    p = {};
    p.k = calls[0].k;
    p.n = calls[0].n;
    p.flag = calls[0].flag;
    p.groups = uint32_t(count);
    p.fused_publish = calls[0].fused_publish;
    p.split_k = split;
    size_t first = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto &a = calls[i];
        if (!linear_geometry(a.rows, a.k, a.n, a.bits_twice, a.book, split, false) ||
            a.rows > grouped_max_rows || a.k != p.k || a.n != p.n || a.flag != p.flag ||
            a.fused_publish != p.fused_publish ||
            (split != 2 && a.rows != 1) ||
            a.rows > grouped_capacity - first || !linear_views(a, split) ||
            !follows(a.xh, calls[0].xh, first * p.k * 4) ||
            !follows(a.z, calls[0].z, first * (split - 1) * p.n * 4)) return false;
        p.entries[i] = {a.trellis, a.suh, a.svh, a.bias, a.x, a.xh, a.z, a.out,
            a.bits_twice, a.book, uint32_t(first), uint32_t(a.rows)};
        for (size_t row = 0; row < a.rows; ++row) p.row_to_group[first + row] = uint8_t(i);
        first += a.rows;
    }
    p.pairs = uint32_t(first);
    struct View { const void *p; size_t bytes; };
    // Unions are used for alias checks only; each live per-group extent was checked above.
    // The out extents may be scattered: the banked decode writes disjoint rows of one
    // activation buffer per routed expert, so out is checked per call, not as one span.
    const View writes[] = {{calls[0].xh, first * p.k * 4},
        {calls[0].z, first * (split - 1) * p.n * 4}, {p.flag, 4}};
    for (size_t i = 0; i < 3; ++i) {
        if (!region(writes[i].p, writes[i].bytes, writes[i].bytes, 4)) return false;
        for (size_t j = 0; j < i; ++j)
            if (!apart(writes[i].p, writes[i].bytes, writes[j].p, writes[j].bytes)) return false;
    }
    for (size_t i = 0; i < count; ++i) {
        const auto &a = calls[i];
        const size_t out_bytes = size_t(a.rows) * p.n * 4;
        if (!region(a.out, out_bytes, out_bytes, 4)) return false;
        for (size_t j = 0; j < 3; ++j)
            if (!apart(a.out, out_bytes, writes[j].p, writes[j].bytes)) return false;
        for (size_t j = 0; j < i; ++j)
            if (!apart(a.out, out_bytes, calls[j].out,
                       size_t(calls[j].rows) * p.n * 4)) return false;
        const View reads[] = {{a.trellis, a.k * a.n / 256 * size_t(a.bits_twice) * 16},
            {a.suh, a.k * 4}, {a.svh, a.n * 4}, {a.bias, a.bias ? a.n * 4 : 0},
            {a.x, a.rows * a.k * 4}};
        for (const auto &read : reads)
            if (read.bytes) {
                for (const auto &write : writes)
                    if (!apart(read.p, read.bytes, write.p, write.bytes)) return false;
                for (size_t j = 0; j < count; ++j)
                    if (!apart(read.p, read.bytes, calls[j].out,
                               size_t(calls[j].rows) * p.n * 4)) return false;
            }
    }
    return true;
}


// ---- Id-indexed grouped launch (the routed experts of one decode token) ----
// Each pair names a member of an expert bank by a device-side id: the launch plan (pairs, grids,
// pointers of the working spans) is fixed when the launch is queued, and the weight pointers are
// bank base + id * stride when the kernels run, so the id array may be written by an earlier
// kernel of the same stream and the launch can be captured in a graph. Every pair is one row.
// The arithmetic is the wide projection's, hence the grouped gather kernel's.
constexpr size_t indexed_capacity = 16;
// The 20-pair option takes Flash-Next's 20 gate and up pairs in one launch.
constexpr size_t indexed_option_capacity = 20;
struct IndexInfo {
    const int32_t *ids;
    uint32_t slot, members;
    size_t trellis_stride;  // bytes between the trellises of consecutive members
    size_t scale_stride;    // floats between the suh (and svh) of consecutive members
    const uint32_t *offsets;  // optional byte offset of each member, replaces the strides
};
constexpr uint32_t absent_member = 0xffffffffu;
template <size_t Capacity>
struct IndexedArgsOf {
    static constexpr size_t capacity = Capacity;
    GroupEntry entries[Capacity];  // member 0 of the bank, and the pair's spans
    IndexInfo info[Capacity];
    uint32_t *flag;
    size_t k, n;
    uint32_t pairs, groups, fused_publish;
    // The id-indexed launch is byte-pinned to the split-2 fold; the grouped launch reads its
    // runtime split_k field. Both spell `p.split_k` the same way in the shared bodies.
    static constexpr uint32_t split_k = 2;
};
using IndexedArgs = IndexedArgsOf<indexed_capacity>;
using IndexedOptionArgs = IndexedArgsOf<indexed_option_capacity>;
static_assert(sizeof(IndexInfo) == 40 && sizeof(IndexedOptionArgs) <= 4096,
              "indexed launch parameters");

// The pair of this block with its weights resolved; false for an id outside the bank, which also
// raises the sticky flag (once per pair, from the first block of the grid).
template <class P>
__device__ __forceinline__ bool resolve_indexed(const P &p, GroupEntry &a) {
    a = p.entries[blockIdx.y];
    const IndexInfo &info = p.info[blockIdx.y];
    const int id = info.ids[info.slot];
    if (id < 0 || uint32_t(id) >= info.members) {
        if (blockIdx.x == 0 && blockIdx.z == 0 && threadIdx.x == 0) atomicOr(p.flag, 1u);
        return false;
    }
    if (info.offsets) {
        const uint32_t offset = info.offsets[id];
        if (offset == absent_member) {
            if (blockIdx.x == 0 && blockIdx.z == 0 && threadIdx.x == 0) atomicOr(p.flag, 1u);
            return false;
        }
        a.trellis = reinterpret_cast<const uint16_t *>(
            reinterpret_cast<const char *>(a.trellis) + offset);
        a.suh += offset / 4;
        a.svh += offset / 4;
        return true;
    }
    a.trellis = reinterpret_cast<const uint16_t *>(
        reinterpret_cast<const char *>(a.trellis) + size_t(id) * info.trellis_stride);
    a.suh += size_t(id) * info.scale_stride;
    a.svh += size_t(id) * info.scale_stride;
    return true;
}

// The kernel bodies of the id-indexed launch, for the 16-pair parameter block and the 20-pair
// one. Each kernel declares its own shared arrays and passes them in: a __shared__
// variable declared in a body that two kernels instantiate is one variable for both, and ptxas
// then reserves it for neither kernel.
template <class P>
__device__ __forceinline__ void indexed_input_rotate_body(const P &p, float (&values)[128]) {
    GroupEntry a;
    if (!resolve_indexed(p, a)) return;
    const size_t index = size_t(blockIdx.x) * 128 + threadIdx.x;
    const float value = finite_or_flag(a.x[index] * a.suh[index % p.k], p.flag);
    a.xh[index] = finite_or_flag(rotate128(value, values), p.flag);
}

template <int Cols, bool Dp4a, bool Pipe, class P>
__device__ __forceinline__ void indexed_project_body(const P &p, float (&partial)[8][16 * Cols],
                                                     unsigned (&wtab)[8][Cols][128]) {
    GroupEntry a;
    if (!resolve_indexed(p, a)) return;
    switch (a.book) {
        case 0: wide_by_rate<0, Cols, true, P, Dp4a, Pipe>(p, a, wtab[threadIdx.x / 32], partial); break;
        case 1: wide_by_rate<1, Cols, true, P, Dp4a, Pipe>(p, a, wtab[threadIdx.x / 32], partial); break;
        default: wide_by_rate<2, Cols, true, P, Dp4a, Pipe>(p, a, wtab[threadIdx.x / 32], partial); break;
    }
}

template <class P>
__device__ __forceinline__ void indexed_output_rotate_body(const P &p, float (&values)[128]) {
    GroupEntry a;
    if (!resolve_indexed(p, a)) return;
    const size_t index = size_t(blockIdx.x) * 128 + threadIdx.x;
    const size_t col = index % p.n;
    float z = a.z[index];
    z = finite_or_flag(z + a.out[index], p.flag);
    const float rotated = rotate128(z, values);
    const float value = rotated * a.svh[col] + (a.bias ? a.bias[col] : 0.0f);
    const float stored = p.fused_publish ?
        __bfloat162float(__float2bfloat16_rn(value)) : value;
    a.out[index] = finite_or_flag(stored, p.flag);
}

__global__ void indexed_input_rotate(const __grid_constant__ IndexedArgs p) {
    __shared__ float values[128];
    indexed_input_rotate_body(p, values);
}

template <int Cols>
__global__ void __launch_bounds__(256, wide_min_blocks<Cols>) indexed_project_wide(
    const __grid_constant__ IndexedArgs p) {
    __shared__ float partial[8][16 * Cols];
    __shared__ __align__(8) unsigned wtab[8][Cols][128];
    indexed_project_body<Cols, false, false>(p, partial, wtab);
}

__global__ void indexed_output_rotate(const __grid_constant__ IndexedArgs p) {
    __shared__ float values[128];
    indexed_output_rotate_body(p, values);
}

// The kernels of the opt-in launch options (`inference_cuda_exl3_grouped_indexed_option`): the
// same bodies on parameter block P (16 or 20 pairs), one tile per block, with the DP4A decode
// when Dp4a is set. The launch above stays the default and the fallback.
template <class P>
__global__ void option_input_rotate(const __grid_constant__ P p) {
    __shared__ float values[128];
    indexed_input_rotate_body(p, values);
}

template <class P, bool Dp4a, bool Pipe = false>
__global__ void __launch_bounds__(256, wide_min_blocks<1>) option_project(
    const __grid_constant__ P p) {
    __shared__ float partial[8][16];
    __shared__ __align__(8) unsigned wtab[8][1][128];
    indexed_project_body<1, Dp4a, Pipe>(p, partial, wtab);
}

// ---- Pointer-table batched launch (`inference_cuda_exl3_indexed_table`) ----
// The parameter block caps one launch at 20 pairs; a routed prefill chunk wants hundreds.
// The table launch moves the pair records to device memory: the host fills one scratch table
// with a single copy and each stage is ONE kernel over the whole batch. The records are what
// the option launch packs per pair, plus k and n per record because a batch may mix shapes
// (gate/up k=hidden n=inter, down k=inter n=hidden). The grid keeps the option launch's axes -
// x is the column unit, y the pair - so the shared bodies read their coordinates unchanged;
// the kernel guards the units past a narrow record's shape. Bodies, arithmetic and bytes are
// the option launch's. These three records are the C ABI of the header's table entries, so
// they sit at global scope beside the parameter blocks the other C entries take.
} // namespace
struct IndexedTablePair {
    GroupEntry entry;  // as the option launch packs it: first = the record grid row, rows = 1
    IndexInfo info;
    uint32_t k, n;
};
struct TableLaunch {
    const IndexedTablePair *pairs;
    uint32_t *flag;
    uint32_t fused_publish;
};
static_assert(sizeof(IndexedTablePair) == 128 && sizeof(IndexedTablePair) % 8 == 0
    && sizeof(TableLaunch) <= 32,
    "table launch parameters: the Rust pack buffer sizes one record at 128 bytes");

namespace {

// The per-record view the shared bodies take where the option launch passes its parameter
// block: shape, flag and the split-2 fold constant are the only fields the indexed bodies read.
struct PairView {
    uint32_t k, n;
    uint32_t fused_publish;
    uint32_t *flag;
    static constexpr uint32_t split_k = 2;
};

// The resolve of the table launch: record y's entry with its bank id applied. A bad id raises
// the flag from the block of unit 0 only, as the option launch does from x = 0, z = 0.
template <bool Unit0Only>
__device__ __forceinline__ bool resolve_table(const TableLaunch &l, PairView &p, GroupEntry &a) {
    const IndexedTablePair &tp = l.pairs[blockIdx.y];
    p = {tp.k, tp.n, l.fused_publish, l.flag};
    a = tp.entry;
    const IndexInfo &info = tp.info;
    const int id = info.ids[info.slot];
    const bool first_block = blockIdx.x == 0 && blockIdx.z == 0 && threadIdx.x == 0;
    if (id < 0 || uint32_t(id) >= info.members) {
        if (Unit0Only && first_block) atomicOr(l.flag, 1u);
        return false;
    }
    if (info.offsets) {
        const uint32_t offset = info.offsets[id];
        if (offset == absent_member) {
            if (Unit0Only && first_block) atomicOr(l.flag, 1u);
            return false;
        }
        a.trellis = reinterpret_cast<const uint16_t *>(
            reinterpret_cast<const char *>(a.trellis) + offset);
        a.suh += offset / 4;
        a.svh += offset / 4;
        return true;
    }
    a.trellis = reinterpret_cast<const uint16_t *>(
        reinterpret_cast<const char *>(a.trellis) + size_t(id) * info.trellis_stride);
    a.suh += size_t(id) * info.scale_stride;
    a.svh += size_t(id) * info.scale_stride;
    return true;
}

__global__ void table_input_rotate(const __grid_constant__ TableLaunch l) {
    __shared__ float values[128];
    PairView p;
    GroupEntry a;
    if (!resolve_table<true>(l, p, a)) return;
    const size_t index = size_t(blockIdx.x) * 128 + threadIdx.x;
    if (index >= size_t(p.k)) return;  // k is a multiple of 128, so a whole block retires here;
                                       // no thread diverges across rotate128's barriers
    const float value = finite_or_flag(a.x[index] * a.suh[index % p.k], p.flag);
    a.xh[index] = finite_or_flag(rotate128(value, values), p.flag);
}

template <bool Dp4a, bool Pipe>
__global__ void __launch_bounds__(256, wide_min_blocks<1>) table_project(
    const __grid_constant__ TableLaunch l) {
    __shared__ float partial[8][16];
    __shared__ __align__(8) unsigned wtab[8][1][128];
    PairView p;
    GroupEntry a;
    if (!resolve_table<false>(l, p, a)) return;
    if (size_t(blockIdx.x) * 16 >= p.n) return;
    switch (a.book) {
        case 0: wide_by_rate<0, 1, true, PairView, Dp4a, Pipe>(p, a, wtab[threadIdx.x / 32], partial); break;
        case 1: wide_by_rate<1, 1, true, PairView, Dp4a, Pipe>(p, a, wtab[threadIdx.x / 32], partial); break;
        default: wide_by_rate<2, 1, true, PairView, Dp4a, Pipe>(p, a, wtab[threadIdx.x / 32], partial); break;
    }
}

__global__ void table_output_rotate(const __grid_constant__ TableLaunch l) {
    __shared__ float values[128];
    PairView p;
    GroupEntry a;
    if (!resolve_table<true>(l, p, a)) return;
    const size_t index = size_t(blockIdx.x) * 128 + threadIdx.x;
    if (index >= size_t(p.n)) return;
    const size_t col = index % p.n;
    float z = a.z[index];
    z = finite_or_flag(z + a.out[index], p.flag);
    const float rotated = rotate128(z, values);
    const float value = rotated * a.svh[col] + (a.bias ? a.bias[col] : 0.0f);
    const float stored = p.fused_publish ?
        __bfloat162float(__float2bfloat16_rn(value)) : value;
    a.out[index] = finite_or_flag(stored, p.flag);
}

template <class P>
__global__ void option_output_rotate(const __grid_constant__ P p) {
    __shared__ float values[128];
    indexed_output_rotate_body(p, values);
}

// Probe of the DP4A decode for its tests (`inference_cuda_exl3_book2_probe`). For each word x:
// sums[2 i] is decode_state's byte sum of x (four masks and adds), sums[2 i + 1] the DP4A result
// __dp4a(x, 0x01010101, 0x6400). For each of the 65536 states s: decoded[2 s] holds the bits of
// decode_state(s, 2), decoded[2 s + 1] those of decode_book2_dp4a(s).
__global__ void book2_probe_sums(const uint32_t *words, size_t count, uint32_t *sums) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned x = words[i];
    sums[2 * i] = (x & 255u) + ((x >> 8) & 255u) + ((x >> 16) & 255u) + (x >> 24);
    sums[2 * i + 1] = __dp4a(x, 0x01010101u, 0x6400u);
}

__global__ void book2_probe_states(uint16_t *decoded) {
    const unsigned state = blockIdx.x * blockDim.x + threadIdx.x;
    if (state > 0xffffu) return;
    decoded[2 * state] = __half_as_ushort(decode_state(state, 2));
    decoded[2 * state + 1] = __half_as_ushort(decode_book2_dp4a(state));
}


template<class Emit>
static inline __attribute__((always_inline)) bool emit_grouped(
    size_t pairs, size_t k, size_t n, uint32_t split, Emit &&emit) {
    return emit(LinearEmission{LinearKind::Input, uint32_t(k / 128), uint32_t(pairs), 1, 128, 1, 1}) &&
        emit(LinearEmission{LinearKind::Small, uint32_t(n / 16), uint32_t(pairs), split, 256, 1, 1}) &&
        emit(LinearEmission{LinearKind::Output, uint32_t(n / 128), uint32_t(pairs), 1, 128, 1, 1});
}


// The projection kernel the wide launch (Cols == 1) runs for a launch whose largest group has
// `max_rows` rows: 1 is grouped_project_small_wide<1> (every group has one row), 2, 4 and
// grouped_max_rows are grouped_project_rows<R>. One function decides the launch and, through
// inference_cuda_exl3_rows_kernel, the launch counters of the test builds.
static inline int rows_kernel_for(uint32_t max_rows) {
    if (max_rows <= 1) return 1;
    if (max_rows <= 2) return 2;
    if (max_rows <= 4) return 4;
    return int(grouped_max_rows);
}

// Test builds only (see the note above inference_cuda_exl3_linear_observed): which projection
// kernel a wide launch runs for a largest group of `max_rows` rows, 0 for a row count the launch
// refuses.
extern "C" __attribute__((section(".text.inference_cuda_exl3_observer"), visibility("hidden")))
int inference_cuda_exl3_rows_kernel(uint32_t max_rows) {
    return max_rows >= 1 && max_rows <= grouped_max_rows ? rows_kernel_for(max_rows) : 0;
}

// Odd: 0, or a launch with one tile per block may run grouped_project_small_wide_r56 when its groups
// allow it (1), with the DP4A decode (2); only the entry the Flash-Next context opts into sets it.
template<bool Observe = false, bool Gather = false, int Cols = 0, int Odd = 0, bool FinIn = false>
static inline __attribute__((always_inline)) int grouped_launch(
    const inference_cuda_exl3_linear_args *calls, size_t count,
    void *stream, int *out_stage, uint64_t *launches = nullptr,
    unsigned *variant = nullptr) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        GroupArgs args;
        if (!stream || !group_args(calls, count, 2, args)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if constexpr (FinIn) {
            // The fused-input kernel stages the row half in shared memory: one row per
            // group, the Flash-Next rates only, a half of whole 256-value rounds.
            if (args.k % 512u) return INFERENCE_CUDA_INVALID_ARGUMENT;
            for (uint32_t g = 0; g < args.groups; ++g) {
                const GroupEntry &e = args.entries[g];
                if (e.rows != 1 || e.book != 2 ||
                    (e.bits_twice != 10 && e.bits_twice != 6))
                    return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
        }
        Failure failure;
        // Whether the projection kernel the launch enqueued was grouped_project_small_wide_r56.
        bool used_odd = false;
        emit_grouped(args.pairs, args.k, args.n, 2, [&](const LinearEmission &e) {
            const dim3 grid(e.gx, e.gy, e.gz), block(e.bx, e.by, e.bz);
            if (e.kind == LinearKind::Input) {
                if constexpr (FinIn)
                    return true;  // rotated inside the projection kernel's shared half
                grouped_input_rotate<<<grid,block,0,(cudaStream_t)stream>>>(args);
            }
            else if (e.kind == LinearKind::Small) {
                if constexpr (FinIn) {
                    const size_t shared = size_t(args.k) / 2 * 4;
                    grouped_project_small_wide_r56_fin<Odd == 2>
                        <<<grid,block,shared,(cudaStream_t)stream>>>(args);
                } else if constexpr (Cols > 0) {
                    const dim3 wide(grid.x / Cols, grid.y, grid.z);
                    if constexpr (Cols == 1) {
                        uint32_t max_rows = 1;
                        for (uint32_t g = 0; g < args.groups; ++g)
                            max_rows = args.entries[g].rows > max_rows ? args.entries[g].rows : max_rows;
                        const dim3 by_group(grid.x, args.groups, grid.z);
                        static_assert(grouped_max_rows == 8, "rows kernel bounds: 2, 4 and 8");
                        const int which = rows_kernel_for(max_rows);
                        if constexpr (Odd) {
                            used_odd = which == 1 && wide_odd_rates(args);
                            if (used_odd)
                                grouped_project_small_wide_r56<Odd == 2><<<wide,block,0,(cudaStream_t)stream>>>(args);
                        }
                        if (used_odd) {
                            // enqueued above
                        } else if (which == 1) {
                            grouped_project_small_wide<1><<<wide,block,0,(cudaStream_t)stream>>>(args);
                        } else if (which == 2) {
                            grouped_project_rows<2><<<by_group,block,0,(cudaStream_t)stream>>>(args);
                        } else if (which == 4) {
                            grouped_project_rows<4><<<by_group,block,0,(cudaStream_t)stream>>>(args);
                        } else {
                            grouped_project_rows<grouped_max_rows><<<by_group,block,0,(cudaStream_t)stream>>>(args);
                        }
                    } else {
                        grouped_project_small_wide<Cols><<<wide,block,0,(cudaStream_t)stream>>>(args);
                    }
                } else if constexpr (Gather)
                    grouped_project_small_gather<<<grid,block,0,(cudaStream_t)stream>>>(args);
                else
                    grouped_project_small<<<grid,block,0,(cudaStream_t)stream>>>(args);
            }
            else grouped_output_rotate<<<grid,block,0,(cudaStream_t)stream>>>(args);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
            if constexpr (Observe) {
                if (failure.err == cudaSuccess) {
                    ++*launches;
                    // Record which projection kernel actually enqueued: the
                    // observer must observe the launch, not echo the request.
                    // 3 and 4 are grouped_project_small_wide_r56 without and with DP4A.
                    if (e.kind == LinearKind::Small && variant)
                        *variant = used_odd ? (Odd == 2 ? 4u : 3u) : (Gather ? 2u : 1u);
                }
            }
            return failure.err == cudaSuccess;
        });
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) { return INFERENCE_CUDA_NATIVE_EXCEPTION; }
}
// Checks an id-indexed group of `count` calls (at most P::capacity, every n a multiple of
// 16 * cols) and fills the parameter block; false refuses it.
template <class P>
static bool indexed_args(const inference_cuda_exl3_linear_args *calls,
    const inference_cuda_exl3_index_args *index, size_t count, size_t cols, P &args) {
    GroupArgs checked;
    if (!index || count < 1 || count > P::capacity || !group_args(calls, count, 2, checked) ||
        checked.pairs != count || checked.n % (16 * cols)) return false;
    args = {};
    args.flag = checked.flag;
    args.k = checked.k;
    args.n = checked.n;
    args.pairs = checked.pairs;
    args.groups = checked.groups;
    args.fused_publish = checked.fused_publish;
    for (size_t i = 0; i < count; ++i) {
        const auto &x = index[i];
        if (!x.ids || x.ids_bytes < (size_t(x.slot) + 1) * 4 || uintptr_t(x.ids) % 4 ||
            !x.members || calls[i].rows != 1 ||
            (x.offsets ? uintptr_t(x.offsets) % 4 != 0
                       : (!x.trellis_stride || !x.scale_stride)))
            return false;
        args.entries[i] = checked.entries[i];
        args.info[i] = {x.ids, x.slot, x.members, x.trellis_stride, x.scale_stride, x.offsets};
    }
    return true;
}

template <int Cols>
static int indexed_launch(const inference_cuda_exl3_linear_args *calls,
    const inference_cuda_exl3_index_args *index, size_t count, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        IndexedArgs args;
        if (!stream || !indexed_args(calls, index, count, Cols, args))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        emit_grouped(args.pairs, args.k, args.n, 2, [&](const LinearEmission &e) {
            const dim3 grid(e.gx, e.gy, e.gz), block(e.bx, e.by, e.bz);
            if (e.kind == LinearKind::Input)
                indexed_input_rotate<<<grid,block,0,(cudaStream_t)stream>>>(args);
            else if (e.kind == LinearKind::Small) {
                const dim3 wide(grid.x / Cols, grid.y, grid.z);
                indexed_project_wide<Cols><<<wide,block,0,(cudaStream_t)stream>>>(args);
            } else indexed_output_rotate<<<grid,block,0,(cudaStream_t)stream>>>(args);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
            return failure.err == cudaSuccess;
        });
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) { return INFERENCE_CUDA_NATIVE_EXCEPTION; }
}

// Packs the pointer-table records host-side and validates them for
// inference_cuda_exl3_indexed_table: per record the option launch's geometry, view and id-index
// checks, but shape may differ per record (k and n travel with the record) and the spans are
// per record, so the option launch's shared-span contiguity gives way to cross-record checks:
// every record's writes apart from every record's reads and writes, one flat flag and one
// fused-publish arm for the batch.
static bool table_pack(const inference_cuda_exl3_linear_args *calls,
    const inference_cuda_exl3_index_args *index, size_t count, uint32_t split,
    IndexedTablePair *out) {
    if (!calls || !index || count < 1) return false;
    struct View { const void *p; size_t bytes; };
    for (size_t i = 0; i < count; ++i) {
        const auto &a = calls[i];
        if (!linear_geometry(a.rows, a.k, a.n, a.bits_twice, a.book, split, false) ||
            a.rows != 1 || !linear_views(a, split)) return false;
        const auto &x = index[i];
        if (!x.ids || x.ids_bytes < (size_t(x.slot) + 1) * 4 || uintptr_t(x.ids) % 4 ||
            !x.members ||
            (x.offsets ? uintptr_t(x.offsets) % 4 != 0
                       : (!x.trellis_stride || !x.scale_stride)))
            return false;
        if (a.flag != calls[0].flag || a.fused_publish != calls[0].fused_publish) return false;
        out[i].entry = {a.trellis, a.suh, a.svh, a.bias, a.x, a.xh, a.z, a.out,
            a.bits_twice, a.book, uint32_t(i), 1u};
        out[i].info = {x.ids, x.slot, x.members, x.trellis_stride, x.scale_stride, x.offsets};
        out[i].k = uint32_t(a.k);
        out[i].n = uint32_t(a.n);
    }
    // Cross-record isolation: one record's writes never overlap another record's reads or
    // writes. Reads may overlap reads (records of one bank share its trellis).
    const auto writes_of = [&](size_t i, View (&w)[4]) {
        const auto &a = calls[i];
        w[0] = {a.xh, a.k * 4};
        w[1] = {a.z, a.n * 4};  // split-2 folds one partial plane
        w[2] = {a.out, a.n * 4};
        w[3] = {a.flag, 4};
    };
    const auto reads_of = [&](size_t i, View (&r)[7]) {
        const auto &a = calls[i];
        r[0] = {a.trellis, a.k * a.n / 256 * size_t(a.bits_twice) * 16};
        r[1] = {a.suh, a.k * 4};
        r[2] = {a.svh, a.n * 4};
        r[3] = {a.bias, a.bias ? a.n * 4 : 0};
        r[4] = {a.x, a.k * 4};
        r[5] = {index[i].ids, size_t(index[i].slot + 1) * 4};
        r[6] = {index[i].offsets, index[i].offsets ? size_t(index[i].members) * 4 : 0};
    };
    View wi[4], wj[4], ri[7], rj[7];
    for (size_t i = 0; i < count; ++i) {
        writes_of(i, wi);
        reads_of(i, ri);
        for (size_t j = 0; j < count; ++j) {
            if (i == j) continue;
            writes_of(j, wj);
            reads_of(j, rj);
            for (const auto &w : wi) {
                if (!w.bytes) continue;
                for (const auto &o : wj)
                    // The batch shares one completion flag: the only alias two records may
                    // write (the pack already refused records with different flag pointers).
                    // Any other same-pointer pair stays refused, like the option launch.
                    if (o.bytes
                        && !(w.p == o.p && w.bytes == 4 && o.bytes == 4
                             && w.p == (const void *)calls[0].flag)
                        && !apart(w.p, w.bytes, o.p, o.bytes))
                        return false;
                for (const auto &r : rj)
                    if (r.bytes && !apart(w.p, w.bytes, r.p, r.bytes)) return false;
            }
            for (const auto &r : ri)
                if (r.bytes) {
                    for (const auto &o : wj)
                        if (o.bytes && !apart(r.p, r.bytes, o.p, o.bytes)) return false;
                }
        }
    }
    return true;
}

// The id-indexed launch of an option on parameter block P, one tile per block.
template <class P, bool Dp4a, bool Pipe = false>
static int option_launch(const inference_cuda_exl3_linear_args *calls,
    const inference_cuda_exl3_index_args *index, size_t count, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        P args;
        if (!stream || !indexed_args(calls, index, count, 1, args))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        emit_grouped(args.pairs, args.k, args.n, 2, [&](const LinearEmission &e) {
            const dim3 grid(e.gx, e.gy, e.gz), block(e.bx, e.by, e.bz);
            if (e.kind == LinearKind::Input)
                option_input_rotate<P><<<grid,block,0,(cudaStream_t)stream>>>(args);
            else if (e.kind == LinearKind::Small)
                option_project<P, Dp4a, Pipe><<<grid,block,0,(cudaStream_t)stream>>>(args);
            else option_output_rotate<P><<<grid,block,0,(cudaStream_t)stream>>>(args);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
            return failure.err == cudaSuccess;
        });
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) { return INFERENCE_CUDA_NATIVE_EXCEPTION; }
}
} // namespace

extern "C" int inference_cuda_exl3_grouped(const inference_cuda_exl3_linear_args *calls,
    size_t count, void *stream, int *out_stage) {
    return grouped_launch<>(calls, count, stream, out_stage);
}

// The grouped launch with an explicit k split of 4 or 8 (2 is refused: callers stay on
// inference_cuda_exl3_grouped). Slicing k further puts more blocks in flight, which the
// fill-limited one-row decode needs; the extra partials fold left-to-right in
// grouped_output_rotate, a different but still fixed order, so the bytes differ from the
// split-2 launch in the last ulp. One-row calls only, the plain one-tile variants only.
extern "C" int inference_cuda_exl3_grouped_split(const inference_cuda_exl3_linear_args *calls,
    size_t count, uint32_t split_k, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        GroupArgs args;
        if (!stream || split_k == 2 || !group_args(calls, count, split_k, args))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        // group_args refuses rows > 1 and non-plain mixes under split != 2; the one-tile
        // body and the output fold carry the split from GroupArgs.
        const dim3 input(uint32_t(args.k / 128), args.pairs, 1), iblock(128, 1, 1);
        grouped_input_rotate<<<input, iblock, 0, (cudaStream_t)stream>>>(args);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.err == cudaSuccess) {
            const dim3 wide(uint32_t(args.n / 16), args.pairs, split_k), block(256, 1, 1);
            grouped_project_small_wide<1><<<wide, block, 0, (cudaStream_t)stream>>>(args);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        if (failure.err == cudaSuccess) {
            const dim3 output(uint32_t(args.n / 128), args.pairs, 1), oblock(128, 1, 1);
            grouped_output_rotate<<<output, oblock, 0, (cudaStream_t)stream>>>(args);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) { return INFERENCE_CUDA_NATIVE_EXCEPTION; }
}


// Same contract as inference_cuda_exl3_grouped with the shared-table gather variant.
extern "C" int inference_cuda_exl3_grouped_gathered(const inference_cuda_exl3_linear_args *calls,
    size_t count, void *stream, int *out_stage) {
    return grouped_launch<false, true>(calls, count, stream, out_stage);
}

// Same contract as inference_cuda_exl3_grouped_gathered with the wide projection: each block
// covers `cols` (1, 2, 4 or 8) adjacent 16-column tiles, so n / 16 must be a multiple of `cols`.
// Bit-identical sums.
extern "C" int inference_cuda_exl3_grouped_wide(const inference_cuda_exl3_linear_args *calls,
    size_t count, void *stream, int *out_stage, unsigned cols) {
    if (!calls || count < 1 || calls[0].n % (16 * size_t(cols ? cols : 1))) return INFERENCE_CUDA_INVALID_ARGUMENT;
    switch (cols) {
        case 1: return grouped_launch<false, true, 1>(calls, count, stream, out_stage);
        case 2: return grouped_launch<false, true, 2>(calls, count, stream, out_stage);
        case 4: return grouped_launch<false, true, 4>(calls, count, stream, out_stage);
        case 8: return grouped_launch<false, true, 8>(calls, count, stream, out_stage);
        default: return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
}

// inference_cuda_exl3_grouped_wide with one tile per block that may take the rate 10 and rate 6 body
// of book 2 (grouped_project_small_wide_r56). Same validation, grids and bytes as the entry with
// cols 1; a launch whose groups are not all book 2 at those rates runs the same kernel as that entry.
extern "C" int inference_cuda_exl3_grouped_wide_odd(const inference_cuda_exl3_linear_args *calls,
    size_t count, void *stream, int *out_stage, unsigned dp4a) {
    if (!calls || count < 1 || calls[0].n % 16) return INFERENCE_CUDA_INVALID_ARGUMENT;
    return dp4a ? grouped_launch<false, true, 1, 2>(calls, count, stream, out_stage)
                : grouped_launch<false, true, 1, 1>(calls, count, stream, out_stage);
}

// inference_cuda_exl3_grouped_wide_odd with the input rotation fused into the projection kernel
// (exl3_decode_fused.h): two launches per group set instead of three, no xh round trip. Requires
// every group to be one row of book 2 at rate 10 or 6 and k a multiple of 512; every byte written
// equals the odd entry's.
extern "C" int inference_cuda_exl3_grouped_wide_fin(const inference_cuda_exl3_linear_args *calls,
    size_t count, void *stream, int *out_stage, unsigned dp4a) {
    if (!calls || count < 1 || calls[0].n % 16) return INFERENCE_CUDA_INVALID_ARGUMENT;
    return dp4a ? grouped_launch<false, true, 1, 2, true>(calls, count, stream, out_stage)
                : grouped_launch<false, true, 1, 1, true>(calls, count, stream, out_stage);
}


// The routed experts of one token: every call is one row of one bank member chosen on the device
// (see IndexedArgs). Plan, grids and spans are fixed at queue time; `cols` is the wide tile count
// (1, 2, 4 or 8) and n must be a multiple of 16 * cols.
extern "C" int inference_cuda_exl3_grouped_indexed(const inference_cuda_exl3_linear_args *calls,
    const inference_cuda_exl3_index_args *index, size_t count, void *stream, int *out_stage,
    unsigned cols) {
    switch (cols) {
        case 1: return indexed_launch<1>(calls, index, count, stream, out_stage);
        case 2: return indexed_launch<2>(calls, index, count, stream, out_stage);
        case 4: return indexed_launch<4>(calls, index, count, stream, out_stage);
        case 8: return indexed_launch<8>(calls, index, count, stream, out_stage);
        default: return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
}

// The id-indexed launch with opt-in options, one tile per block. `options` bit 0 decodes book 2
// with DP4A; bit 1 takes up to 20 calls in one launch (else up to 16); bit 2 pipelines the tile
// loop (the next tile's loads issued before the current one's barriers; same bytes). Options 0
// is the default launch (inference_cuda_exl3_grouped_indexed) and is refused here.
extern "C" int inference_cuda_exl3_grouped_indexed_option(
    const inference_cuda_exl3_linear_args *calls, const inference_cuda_exl3_index_args *index,
    size_t count, void *stream, int *out_stage, unsigned options) {
    switch (options) {
        case 1: return option_launch<IndexedArgs, true>(calls, index, count, stream, out_stage);
        case 2:
            return option_launch<IndexedOptionArgs, false>(calls, index, count, stream, out_stage);
        case 3:
            return option_launch<IndexedOptionArgs, true>(calls, index, count, stream, out_stage);
        case 4: return option_launch<IndexedArgs, false, true>(calls, index, count, stream, out_stage);
        case 5: return option_launch<IndexedArgs, true, true>(calls, index, count, stream, out_stage);
        case 6:
            return option_launch<IndexedOptionArgs, false, true>(calls, index, count, stream, out_stage);
        case 7:
            return option_launch<IndexedOptionArgs, true, true>(calls, index, count, stream, out_stage);
        default: return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
}

// The pointer-table batched launch of the id-indexed kernels: one kernel per stage over every
// record of the batch (`pairs_count` device records, filled and validated host-side). The
// grids take the batch's widest shape; narrow records idle their tail blocks (per-thread for
// the rotates, per-tile for the project). `k_units` is ceil(widest k / 128), `n_tiles` the
// widest n / 16 and `n_units` ceil(widest n / 128). Options bit 0 is the DP4A decode, bit 2
// the pipelined tile loop; the 20-pair and pong bits do not apply (no parameter block).
extern "C" int inference_cuda_exl3_indexed_table(
    const IndexedTablePair *pairs, size_t pairs_count, uint32_t k_units, uint32_t n_tiles,
    uint32_t n_units, uint32_t *flag, uint32_t fused_publish, void *stream, int *out_stage,
    unsigned options) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!stream || !pairs || !flag || pairs_count < 1 || pairs_count > 65535 ||
            !k_units || !n_tiles || !n_units || options & ~5u)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const TableLaunch l{pairs, flag, fused_publish};
        Failure failure;
        table_input_rotate<<<dim3(k_units, unsigned(pairs_count)), 128, 0,
            (cudaStream_t)stream>>>(l);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.err == cudaSuccess) {
            const dim3 grid(n_tiles, unsigned(pairs_count), 2);
            if (options & 1u)
                if (options & 4u) table_project<true, true><<<grid, 256, 0, (cudaStream_t)stream>>>(l);
                else table_project<true, false><<<grid, 256, 0, (cudaStream_t)stream>>>(l);
            else if (options & 4u) table_project<false, true><<<grid, 256, 0, (cudaStream_t)stream>>>(l);
            else table_project<false, false><<<grid, 256, 0, (cudaStream_t)stream>>>(l);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        if (failure.err == cudaSuccess) {
            table_output_rotate<<<dim3(n_units, unsigned(pairs_count)), 128, 0,
                (cudaStream_t)stream>>>(l);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        if (failure.err != cudaSuccess) return INFERENCE_CUDA_NATIVE_EXCEPTION;
        return INFERENCE_CUDA_OK;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

// Packs `pairs_count` records into the caller's HOST buffer `out` for
// inference_cuda_exl3_indexed_table: the option launch's per-call checks per record, shapes may
// differ, spans per record, cross-record isolation enforced. Returns 0 on success; the caller
// uploads the buffer and computes the widest-shape grids from its own records.
extern "C" int inference_cuda_exl3_indexed_table_pack(
    const inference_cuda_exl3_linear_args *calls, const inference_cuda_exl3_index_args *index,
    size_t pairs_count, size_t out_bytes, IndexedTablePair *out) {
    try {
        // The Rust side sizes its buffer from its own constant; this bound makes a grown
        // record refuse here instead of writing past it.
        if (!out || pairs_count > out_bytes / sizeof(IndexedTablePair))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (!table_pack(calls, index, pairs_count, 2, out))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        return INFERENCE_CUDA_OK;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

// The DP4A decode probe (book2_probe_sums, book2_probe_states): `sums` holds 2 * count words,
// `decoded` 2 * 65536 halves.
extern "C" int inference_cuda_exl3_book2_probe(const uint32_t *words, size_t count,
    uint32_t *sums, uint16_t *decoded, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!stream || !words || !count || !sums || !decoded)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        book2_probe_sums<<<unsigned((count + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
            words, count, sums);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.err == cudaSuccess) {
            book2_probe_states<<<256, 256, 0, (cudaStream_t)stream>>>(decoded);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) { return INFERENCE_CUDA_NATIVE_EXCEPTION; }
}

extern "C" __attribute__((section(".text.inference_cuda_exl3_observer"), visibility("hidden")))
int inference_cuda_exl3_grouped_observed(const inference_cuda_exl3_linear_args *calls,
    size_t count, void *stream, int *out_stage, uint64_t *launches,
    unsigned *variant) {
    if (!launches || !variant) return INFERENCE_CUDA_INVALID_ARGUMENT;
    *launches = 0;
    *variant = 0;
    return grouped_launch<true>(calls, count, stream, out_stage, launches, variant);
}

extern "C" __attribute__((section(".text.inference_cuda_exl3_observer"), visibility("hidden")))
int inference_cuda_exl3_grouped_observed_gathered(const inference_cuda_exl3_linear_args *calls,
    size_t count, void *stream, int *out_stage, uint64_t *launches,
    unsigned *variant) {
    if (!launches || !variant) return INFERENCE_CUDA_INVALID_ARGUMENT;
    *launches = 0;
    *variant = 0;
    return grouped_launch<true, true>(calls, count, stream, out_stage, launches, variant);
}

// Test builds only: inference_cuda_exl3_grouped_wide_odd that reports the projection kernel it
// enqueued in `variant` (3 and 4 the rate 10 and 6 body without and with DP4A, 2 any other).
extern "C" __attribute__((section(".text.inference_cuda_exl3_observer"), visibility("hidden")))
int inference_cuda_exl3_grouped_wide_odd_observed(const inference_cuda_exl3_linear_args *calls,
    size_t count, void *stream, int *out_stage, unsigned dp4a, uint64_t *launches,
    unsigned *variant) {
    if (!launches || !variant || !calls || count < 1 || calls[0].n % 16)
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    *launches = 0;
    *variant = 0;
    return dp4a ? grouped_launch<true, true, 1, 2>(calls, count, stream, out_stage, launches, variant)
                : grouped_launch<true, true, 1, 1>(calls, count, stream, out_stage, launches, variant);
}

extern "C" __attribute__((section(".text.inference_cuda_exl3_observer"), visibility("hidden")))
int inference_cuda_exl3_observe_grouped(const LinearObserveArgs *a, LinearObservation *out) {
    if (!a || !out || a->rows < 1 || a->rows > grouped_capacity ||
        a->split_k != 2 || a->midrows != 0 || a->fail_at > 3 ||
        !geometry(a->k, a->n, 6, 0)) return INFERENCE_CUDA_INVALID_ARGUMENT;
    *out = {};
    out->complete = emit_grouped(a->rows, a->k, a->n, 2, [&](const LinearEmission &e) {
        out->emissions[out->count++] = e;
        return out->count != a->fail_at;
    });
    return INFERENCE_CUDA_OK;
}

// ---- the job-table projection (contract: exl3.h, inference_cuda_exl3_jobs) -------------------
namespace {
constexpr uint32_t jobs_max = 4096;
constexpr uint32_t jobs_row_cap = 256;

// A job the table may not run: outside the bank, outside the rows of the buffers, or taller than
// the grid was sized for. Every thread of a block evaluates the same job, so a refusal ends the
// whole block before any barrier.
__device__ __forceinline__ bool job_in_bounds(const inference_cuda_exl3_jobs_args &p,
                                              const inference_cuda_exl3_job &job) {
    return job.member >= 0 && size_t(job.member) < p.members && job.row0 >= 0 &&
        job.rows >= 1 && uint32_t(job.rows) <= p.max_rows &&
        size_t(job.row0) + size_t(job.rows) <= p.total_rows;
}
__device__ bool job_runs(const inference_cuda_exl3_jobs_args &p,
                         const inference_cuda_exl3_job &job) {
    const bool inside = job_in_bounds(p, job);
    if (!inside && blockIdx.x == 0 && threadIdx.x == 0) {
        atomicOr(p.flag, 1u);
    }
    return inside;
}

// Grid (ceil(max_rows * K / 128), jobs_count): block y rotates the input rows of job y.
__global__ void jobs_input_rotate(const __grid_constant__ inference_cuda_exl3_jobs_args p) {
    __shared__ float values[128];
    const inference_cuda_exl3_job job = p.jobs[blockIdx.y];
    if (!job_runs(p, job)) return;
    const size_t local = size_t(blockIdx.x) * 128 + threadIdx.x;
    if (size_t(blockIdx.x) * 128 >= size_t(job.rows) * p.k) return;
    const size_t index = size_t(job.row0) * p.k + local;
    const size_t row = size_t(job.row0) + local / p.k;
    const size_t source = p.in_rows ? size_t(p.in_rows[row]) : row;
    const float *suh = inference_cuda_exl3_member_suh(p, job.member);
    // A list entry past the rows of x drops the row (zero input) and flags the call.
    const bool inside = source < p.x_rows;
    if (!inside) atomicOr(p.flag, 1u);
    const float read = inside ? p.x[source * p.k + local % p.k] : 0.0f;
    const float value = finite_or_flag(read * suh[local % p.k], p.flag);
    // The rows kernel rounds its operands to F16 anyway; rounding here gives it the same bits at
    // half the bytes to re-read per column group.
    p.xh[index] = __half_as_ushort(__float2half_rn(finite_or_flag(rotate128(value, values), p.flag)));
}

#include "exl3_rotate_wide.h"
#include "exl3_rotate_pair.h"

// The rows of one job, as the rows kernel (midrows_projection) takes them, with the activations
// already F16 (two per word) and NT column tiles per CTA: a CTA of 16 * W rows by 16 * NT columns.
// The input rows are staged once per K step for all NT tiles, where a 16-column CTA reads every
// input row once per tile, and one barrier pair serves NT matrix multiplies. Each output element
// accumulates the same 16-wide K steps in the same order as midrows_projection, so the result is
// bit-identical to it.
struct JobsTile {
    const uint16_t *trellis;
    const unsigned *xh;  // the job's rows, k halves each
    float *z;
    uint32_t *flag;
    size_t rows, k, n;
    int bits_twice;
    // The fused output-rotation epilogue only: the output scale, the scatter target and
    // the job's first row in the table (row0 plus the CTA's row block). Null/0 when unfused.
    const float *svh;
    const uint32_t *out_rows;
    float *out;
    size_t out_total_rows, out_ld, out_col, table_row0;
};

template <int Book, int W, int NT>
__device__ __forceinline__ void jobs_projection(const JobsTile &a, unsigned *inputs32,
                                                unsigned *weights32, float *result) {
    namespace wmma = nvcuda::wmma;
    const half *inputs = reinterpret_cast<const half *>(inputs32);
    const half *weights = reinterpret_cast<const half *>(weights32);
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned steps = unsigned(a.k / 16);
    constexpr unsigned block_rows = 16 * W, wide = 16 * NT;
    const size_t first_row = size_t(blockIdx.y) * block_rows + warp * 16;
    const bool live = first_row < a.rows;
    const bool decoder = threadIdx.x < 128;
    const MidrowsStates states =
        midrows_states(decoder ? threadIdx.x : 0u, unsigned(a.bits_twice));

    const size_t tile_halves = size_t(a.bits_twice) * 8, step_halves = size_t(a.n / 16) * tile_halves;
    const uint16_t *tile = a.trellis + size_t(blockIdx.x) * NT * tile_halves;
    const uint16_t *lead = tile + 2 * size_t(states.lead_word);
    const uint16_t *tail = tile + 2 * size_t(states.tail_word);
    // This lane stages rows 4 apart, one F16 pair (a word) each; the pointers of a lane without
    // such a row are formed but never dereferenced (see midrows_projection).
    const size_t lane_row = first_row + (lane >> 3);
    const unsigned *source = a.xh + lane_row * (a.k / 2) + (lane & 7);
    const unsigned row_stride = unsigned(4 * (a.k / 2));
    const bool valid0 = live && lane_row < a.rows, valid1 = live && lane_row + 4 < a.rows,
               valid2 = live && lane_row + 8 < a.rows, valid3 = live && lane_row + 12 < a.rows;

    unsigned x[4] = {};
    unsigned halves[NT][4] = {};
    auto fetch = [&]() {
        const unsigned *row = source;
        if (valid0) x[0] = row[0];
        row += row_stride;
        if (valid1) x[1] = row[0];
        row += row_stride;
        if (valid2) x[2] = row[0];
        row += row_stride;
        if (valid3) x[3] = row[0];
        if (decoder) {
            #pragma unroll
            for (unsigned j = 0; j < NT; ++j) {
                halves[j][0] = lead[j * tile_halves]; halves[j][1] = lead[j * tile_halves + 1];
                halves[j][2] = tail[j * tile_halves]; halves[j][3] = tail[j * tile_halves + 1];
            }
        }
    };
    auto stage = [&]() {
        if (live) {
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j) inputs32[warp * 128 + 32 * j + lane] = x[j];
        }
        if (decoder) {
            #pragma unroll
            for (unsigned j = 0; j < NT; ++j)
                weights32[j * 128 + states.slot] = midrows_weight_pair<Book>(states, halves[j], a.flag);
        }
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> sum[NT];
    #pragma unroll
    for (unsigned j = 0; j < NT; ++j) wmma::fill_fragment(sum[j], 0.0f);
    fetch();
    #pragma unroll 1
    for (unsigned step = 0; step < steps; ++step) {
        stage();
        if (step + 1 < steps) {
            source += 8;
            lead += step_halves;
            tail += step_halves;
            fetch();
        }
        __syncthreads();
        if (live) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> xf;
            wmma::load_matrix_sync(xf, inputs + warp * 256, 16);
            #pragma unroll
            for (unsigned j = 0; j < NT; ++j) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> wf;
                wmma::load_matrix_sync(wf, weights + j * 256, 16);
                wmma::mma_sync(sum[j], xf, wf, sum[j]);
            }
        }
        __syncthreads();
    }
    if (live) {
        #pragma unroll
        for (unsigned j = 0; j < NT; ++j)
            wmma::store_matrix_sync(result + warp * 16 * wide + j * 16, sum[j], wide,
                                    wmma::mem_row_major);
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < block_rows * wide; i += 32 * W) {
        const size_t row = size_t(blockIdx.y) * block_rows + i / wide;
        if (row < a.rows) {
            a.z[row * a.n + size_t(blockIdx.x) * wide + i % wide] = finite_or_flag(result[i], a.flag);
        }
    }
}

// The widest column group of a CTA of W warps whose result tile fits 32 KiB of shared memory;
// N is a multiple of 128, so it always divides the 16-column tile count. ptxas (nvcc 12.6, sm_86)
// allocates 168, 102 and 68 registers and 38,912, 38,912 and 41,984 bytes of static shared memory
// for W = 4, 8 and 16, with no spills: two CTAs per SM at W = 4 and 8 (shared memory) and one at
// W = 16 (registers), recheck whenever the kernel changes.
template <int W>
constexpr int jobs_tiles = W == 4 ? 8 : (W == 8 ? 4 : 2);

// Grid (N / (16 * jobs_tiles<W>), ceil(max_rows / (16 * W)), jobs_count): the rows kernel on the
// job's rows and member.
template <int W>
__global__ void __launch_bounds__(32 * W) jobs_project(
    const __grid_constant__ inference_cuda_exl3_jobs_args p) {
    constexpr int NT = jobs_tiles<W>;
    __shared__ __align__(32) unsigned inputs32[128 * W];
    __shared__ __align__(32) unsigned weights32[128 * NT];
    __shared__ __align__(32) float result[256 * W * NT];
    const inference_cuda_exl3_job job = p.jobs[blockIdx.z];
    if (!job_runs(p, job)) return;
    if (size_t(blockIdx.y) * (16 * W) >= size_t(job.rows)) return;
    JobsTile a{};
    a.trellis = inference_cuda_exl3_member_trellis(p, job.member);
    a.xh = reinterpret_cast<const unsigned *>(p.xh) + size_t(job.row0) * (p.k / 2);
    a.z = p.z + size_t(job.row0) * p.n;
    a.flag = p.flag;
    a.rows = size_t(job.rows);
    a.k = p.k;
    a.n = p.n;
    a.bits_twice = p.bits_twice;
    switch (p.book) {
        case 0: jobs_projection<0, W, NT>(a, inputs32, weights32, result); break;
        case 1: jobs_projection<1, W, NT>(a, inputs32, weights32, result); break;
        default: jobs_projection<2, W, NT>(a, inputs32, weights32, result); break;
    }
}

// ---- the register-decode job-table kernel ---------------------------------------------------
// One CTA of 8 warps covers 8 column tiles (128 columns) and 64 rows of one job. A warp decodes
// the trellis tile of its column into the B fragments of mma.m16n8k16 itself (the packed states are
// in fragment order: lane (g, t) holds columns g and g + 8 at rows 2t, 2t + 1, 2t + 8, 2t + 9, the
// four states of the threads 16 g + 4 t + 0..3 of the shared-memory decoder), so the tile no
// longer goes through shared memory and no barrier separates decode from math. The F16 input rows
// of 32 values of K arrive through a three-stage cp.async ring with the raw trellis tiles (one
// barrier per stage) and are read with ldmatrix. The mma instruction, its operands and the ascending order of the K16 steps per
// output element are those of the wmma kernel, so the results are bit-identical to it.
constexpr int jobs_reg_rows = 64;           // rows of one CTA
constexpr int jobs_reg_stride = 80;         // bytes of one staged row: 32 halves and 16 of padding
constexpr int jobs_reg_stages = 3;          // depth of the cp.async ring
constexpr int jobs_reg_a_bytes = jobs_reg_rows * jobs_reg_stride;
constexpr int jobs_reg_b_bytes = 8 * 2 * 16 * 16;  // 8 tiles x 2 K16 steps x 16 bytes x bits_twice
constexpr int jobs_reg_stage_bytes = jobs_reg_a_bytes + jobs_reg_b_bytes;

// The fused output-rotation epilogue stages its 32-row slab and the pair butterfly's values
// onto the cp.async ring, which is dead once the last stage is consumed; both must fit.
constexpr int jobs_fused_tile_bytes = 32 * 128 * int(sizeof(float));
static_assert(jobs_reg_stages * jobs_reg_stage_bytes >= jobs_fused_tile_bytes + 1024,
              "fused epilogue no longer fits the dead ring");

// ldmatrix needs sm_75 and mma.m16n8k16 and cp.async need sm_80: an older device pass (the sm_72
// build of aarch64) gets the kernel as a trap that the launch below never reaches.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void *shared_address) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(shared_address));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(address)
                 : "memory");
}
__device__ __forceinline__ void mma_16816(float (&c)[4], const unsigned (&a)[4], unsigned b0,
                                          unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(void *shared_address, const void *global_address) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(shared_address));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(address), "l"(global_address)
                 : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int Pending>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(Pending) : "memory");
}

// The ring holds jobs_reg_stages stages of 32 values of K: the 64 F16 input rows of the job and the
// raw trellis words of the 8 tiles of the CTA for the two K16 steps. Stage s is copied by cp.async
// while stage s - 1 is multiplied; one barrier per stage.
template <int Book, bool FusedOut = false>
__device__ __forceinline__ void jobs_reg_projection(const JobsTile &a,
                                                    unsigned char (*ring)[jobs_reg_stage_bytes]) {
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned group = lane >> 2, quad = lane & 3;
    const size_t rows = a.rows;
    const unsigned stages = unsigned(a.k / 32);
    const unsigned bits_twice = unsigned(a.bits_twice);
    const unsigned tile_bytes = 16 * bits_twice;
    const size_t tile_halves = size_t(a.bits_twice) * 8, step_halves = size_t(a.n / 16) * tile_halves;
    const uint16_t *columns = a.trellis + size_t(blockIdx.x) * 8 * tile_halves;

    MidrowsStates states[4];
    #pragma unroll
    for (unsigned q = 0; q < 4; ++q)
        states[q] = midrows_states(16 * group + 4 * quad + q, bits_twice);

    // Copy roles: this thread moves 16 bytes of the input rows (row thread / 4, chunk thread % 4)
    // and, below 16 * bits_twice, 16 bytes of the raw tiles.
    const unsigned a_row = threadIdx.x / 4, a_chunk = threadIdx.x % 4;
    const bool a_live = a_row < rows;
    const uint4 *a_source = reinterpret_cast<const uint4 *>(a.xh) + size_t(a_row) * (a.k / 8) + a_chunk;
    const bool b_live = threadIdx.x < 16 * bits_twice;
    const unsigned b_tile = threadIdx.x / (2 * bits_twice);
    const unsigned b_step = (threadIdx.x % (2 * bits_twice)) / bits_twice;
    const unsigned b_chunk = threadIdx.x % bits_twice;
    const uint16_t *b_source = columns + size_t(b_tile) * tile_halves + size_t(b_step) * step_halves +
        size_t(b_chunk) * 8;
    const unsigned b_target = (b_tile * 2 + b_step) * tile_bytes + b_chunk * 16;

    // Input rows past the job stay zero in every slot.
    if (!a_live) {
        #pragma unroll
        for (unsigned slot = 0; slot < jobs_reg_stages; ++slot)
            *reinterpret_cast<uint4 *>(&ring[slot][a_row * jobs_reg_stride + a_chunk * 16]) =
                make_uint4(0, 0, 0, 0);
    }
    auto issue = [&](unsigned stage) {
        unsigned char *slot = ring[stage % jobs_reg_stages];
        if (a_live) cp_async16(slot + a_row * jobs_reg_stride + a_chunk * 16, a_source + stage * 4);
        if (b_live)
            cp_async16(slot + jobs_reg_a_bytes + b_target, b_source + size_t(2 * stage) * step_halves);
        cp_async_commit();
    };
    #pragma unroll
    for (unsigned stage = 0; stage < jobs_reg_stages - 1; ++stage) {
        if (stage < stages) issue(stage); else cp_async_commit();
    }

    float acc[4][2][4] = {};
    for (unsigned stage = 0; stage < stages; ++stage) {
        cp_async_wait<jobs_reg_stages - 2>();
        __syncthreads();
        if (stage + jobs_reg_stages - 1 < stages) issue(stage + jobs_reg_stages - 1);
        else cp_async_commit();
        const unsigned char *slot = ring[stage % jobs_reg_stages];
        #pragma unroll
        for (unsigned half_step = 0; half_step < 2; ++half_step) {
            const uint16_t *raw = reinterpret_cast<const uint16_t *>(
                slot + jobs_reg_a_bytes + (warp * 2 + half_step) * tile_bytes);
            unsigned b[4];
            #pragma unroll
            for (unsigned q = 0; q < 4; ++q) {
                const uint16_t *lead = raw + 2 * size_t(states[q].lead_word);
                const uint16_t *tail = raw + 2 * size_t(states[q].tail_word);
                const unsigned halves[4] = {lead[0], lead[1], tail[0], tail[1]};
                b[q] = midrows_weight_pair<Book>(states[q], halves, a.flag);
            }
            #pragma unroll
            for (unsigned fragment = 0; fragment < 4; ++fragment) {
                if (size_t(fragment) * 16 >= rows) break;
                unsigned fa[4];
                const unsigned matrix = lane >> 3;
                const unsigned row = fragment * 16 + (lane & 7) + 8 * (matrix & 1);
                const unsigned column = half_step * 16 + 8 * (matrix >> 1);
                ldmatrix_x4(fa, slot + row * jobs_reg_stride + column * 2);
                mma_16816(acc[fragment][0], fa, b[0], b[1]);
                mma_16816(acc[fragment][1], fa, b[2], b[3]);
            }
        }
    }

    const size_t column0 = (size_t(blockIdx.x) * 8 + warp) * 16;
    if constexpr (!FusedOut) {
        #pragma unroll
        for (unsigned fragment = 0; fragment < 4; ++fragment) {
            #pragma unroll
            for (unsigned half_column = 0; half_column < 2; ++half_column) {
                #pragma unroll
                for (unsigned part = 0; part < 4; ++part) {
                    const size_t row = size_t(fragment) * 16 + group + 8 * (part >> 1);
                    if (row >= rows) continue;
                    a.z[row * a.n + column0 + half_column * 8 + 2 * quad + (part & 1)] =
                        finite_or_flag(acc[fragment][half_column][part], a.flag);
                }
            }
        }
    } else {
        // The output rotation of jobs_output_rotate fused into the epilogue: a CTA holds every
        // 128-column group of its rows complete (n is a multiple of 128 and the CTA covers 128
        // consecutive columns), so the butterfly needs no exchange beyond this block. The staged
        // values are the same finite_or_flag(acc) the unfused store writes to z, the butterfly is
        // the same network, and the scale, flag and scatter are the same ops in the same order, so
        // the published bits are identical; only z itself stays unwritten.
        //
        // The ring is dead only AFTER this barrier: the loop's last barrier sits at the top of
        // its final iteration, so without this one a warp leaving the loop could overwrite the
        // slot a straggler still reads (every stages % 3 except 0 lands that slot inside the
        // tile). The staging below aliases the ring, so the ordering lives here, not in layout.
        __syncthreads();
        float *const tile = reinterpret_cast<float *>(&ring[0][0]);
        float *const values = reinterpret_cast<float *>(
            &ring[0][0] + jobs_fused_tile_bytes);
        const unsigned rows_u = unsigned(a.rows);
        const size_t col0 = size_t(blockIdx.x) * 128;
        for (unsigned base = 0; base < rows_u; base += 32) {
            #pragma unroll
            for (unsigned fragment = 0; fragment < 4; ++fragment) {
                #pragma unroll
                for (unsigned half_column = 0; half_column < 2; ++half_column) {
                    #pragma unroll
                    for (unsigned part = 0; part < 4; ++part) {
                        const unsigned row = fragment * 16 + group + 8 * (part >> 1);
                        if (row >= rows_u || row < base || row >= base + 32) continue;
                        const unsigned col = warp * 16 + half_column * 8 + 2 * quad + (part & 1);
                        tile[size_t(row - base) * 128 + col] =
                            finite_or_flag(acc[fragment][half_column][part], a.flag);
                    }
                }
            }
            __syncthreads();
            const unsigned top = base + 32 < rows_u ? base + 32 : rows_u;
            for (unsigned row = base; row < top; row += 2) {
                const bool live = row + (threadIdx.x >> 7) < top;
                const float staged = live
                    ? tile[size_t(row - base + (threadIdx.x >> 7)) * 128 + (threadIdx.x & 127)]
                    : 0.0f;
                const float rotated = rotate128_pair(staged, values);
                if (live) {
                    const unsigned col = unsigned(col0) + (threadIdx.x & 127);
                    const size_t table_row = a.table_row0 + row + (threadIdx.x >> 7);
                    const size_t target = a.out_rows ? size_t(a.out_rows[table_row]) : table_row;
                    const float value = finite_or_flag(rotated * a.svh[col], a.flag);
                    if (target >= a.out_total_rows) atomicOr(a.flag, 1u);
                    else a.out[target * a.out_ld + a.out_col + col] = value;
                }
            }
            __syncthreads();  // the next slab overwrites the tile
        }
    }
}

#endif

__global__ void __launch_bounds__(256) jobs_project_reg(
    const __grid_constant__ inference_cuda_exl3_jobs_args p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    __trap();
#else
    __shared__ __align__(16) unsigned char ring[jobs_reg_stages][jobs_reg_stage_bytes];
    const inference_cuda_exl3_job job = p.jobs[blockIdx.z];
    if (!job_runs(p, job)) return;
    // Block y covers rows 64 y .. 64 y + 63 of the job; the weights are decoded once per block.
    const size_t first = size_t(blockIdx.y) * jobs_reg_rows;
    if (first >= size_t(job.rows)) return;
    JobsTile a{};
    a.trellis = inference_cuda_exl3_member_trellis(p, job.member);
    a.xh = reinterpret_cast<const unsigned *>(p.xh) + (size_t(job.row0) + first) * (p.k / 2);
    a.z = p.z + (size_t(job.row0) + first) * p.n;
    a.flag = p.flag;
    a.rows = min(size_t(jobs_reg_rows), size_t(job.rows) - first);
    a.k = p.k;
    a.n = p.n;
    a.bits_twice = p.bits_twice;
    switch (p.book) {
        case 0: jobs_reg_projection<0>(a, ring); break;
        case 1: jobs_reg_projection<1>(a, ring); break;
        default: jobs_reg_projection<2>(a, ring); break;
    }
#endif
}

// The register-decode kernel with the output rotation fused into its epilogue (see the
// FusedOut tail of jobs_reg_projection); only the launch differs from jobs_project_reg.
__global__ void __launch_bounds__(256) jobs_project_reg_fused(
    const __grid_constant__ inference_cuda_exl3_jobs_args p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    __trap();
#else
    __shared__ __align__(16) unsigned char ring[jobs_reg_stages][jobs_reg_stage_bytes];
    const inference_cuda_exl3_job job = p.jobs[blockIdx.z];
    if (!job_runs(p, job)) return;
    const size_t first = size_t(blockIdx.y) * jobs_reg_rows;
    if (first >= size_t(job.rows)) return;
    JobsTile a{};
    a.trellis = inference_cuda_exl3_member_trellis(p, job.member);
    a.xh = reinterpret_cast<const unsigned *>(p.xh) + (size_t(job.row0) + first) * (p.k / 2);
    a.z = p.z + (size_t(job.row0) + first) * p.n;
    a.flag = p.flag;
    a.rows = min(size_t(jobs_reg_rows), size_t(job.rows) - first);
    a.k = p.k;
    a.n = p.n;
    a.bits_twice = p.bits_twice;
    a.svh = inference_cuda_exl3_member_svh(p, job.member);
    a.out_rows = p.out_rows;
    a.out = p.out;
    a.out_total_rows = p.out_total_rows;
    a.out_ld = p.out_ld;
    a.out_col = p.out_col;
    a.table_row0 = size_t(job.row0) + first;
    switch (p.book) {
        case 0: jobs_reg_projection<0, true>(a, ring); break;
        case 1: jobs_reg_projection<1, true>(a, ring); break;
        default: jobs_reg_projection<2, true>(a, ring); break;
    }
#endif
}

// Grid (ceil(max_rows * N / 128), jobs_count): the output rotation, written at the row stride and
// column offset of the call.
__global__ void jobs_output_rotate(const __grid_constant__ inference_cuda_exl3_jobs_args p) {
    __shared__ float values[128];
    const inference_cuda_exl3_job job = p.jobs[blockIdx.y];
    if (!job_runs(p, job)) return;
    if (size_t(blockIdx.x) * 128 >= size_t(job.rows) * p.n) return;
    const size_t local = size_t(blockIdx.x) * 128 + threadIdx.x;
    const size_t row = local / p.n, col = local % p.n;
    const float *svh = inference_cuda_exl3_member_svh(p, job.member);
    const float rotated = rotate128(p.z[size_t(job.row0) * p.n + local], values);
    const size_t table_row = size_t(job.row0) + row;
    const size_t target = p.out_rows ? size_t(p.out_rows[table_row]) : table_row;
    const float value = finite_or_flag(rotated * svh[col], p.flag);
    if (target >= p.out_total_rows) {
        atomicOr(p.flag, 1u);
        return;
    }
    p.out[target * p.out_ld + p.out_col + col] = value;
}

// The geometry, the extents and the overlap rules of exl3.h.
bool jobs_views(const inference_cuda_exl3_jobs_args &a) {
    if (!geometry(a.k, a.n, a.bits_twice, a.book) || a.jobs_count < 1 || a.jobs_count > jobs_max ||
        a.max_rows < 1 || a.max_rows > jobs_row_cap || a.members < 1 || a.members > (1u << 24) ||
        a.total_rows < 1 || a.total_rows > 65536 || a.total_rows * a.k > (1ULL << 32) ||
        a.total_rows * a.n > (1ULL << 32) || a.out_ld < a.out_col + a.n ||
        a.out_ld > (1ULL << 24) || a.out_col > (1ULL << 24)) return false;
    // Without a row list the rows of x (of out) are the rows of the table; with one they may be
    // any count up to the limit.
    if (a.x_rows < 1 || a.x_rows > 65536 || a.out_total_rows < 1 || a.out_total_rows > 65536 ||
        a.x_rows * a.k > (1ULL << 32) || (!a.in_rows && a.x_rows != a.total_rows) ||
        (!a.out_rows && a.out_total_rows != a.total_rows)) return false;
    const size_t packed = a.k * a.n / 256 * size_t(a.bits_twice) * 8;  // u16 elements
    if (a.members > 1 && (a.trellis_stride < packed || a.scale_stride < a.k + a.n)) return false;
    if (a.members > 1 && (a.trellis_stride > (1ULL << 36) || a.scale_stride > (1ULL << 36)))
        return false;
    const size_t last = a.members - 1;
    const size_t trellis = (last * (a.members > 1 ? a.trellis_stride : 0) + packed) * 2;
    const size_t suh = (last * (a.members > 1 ? a.scale_stride : 0) + a.k) * 4;
    const size_t svh = (last * (a.members > 1 ? a.scale_stride : 0) + a.n) * 4;
    const size_t input = a.x_rows * a.k * 4, rotated = a.total_rows * a.k * 2,
                 projected = a.total_rows * a.n * 4;
    const size_t output = ((a.out_total_rows - 1) * a.out_ld + a.out_col + a.n) * 4;
    struct View { const void *p; size_t bytes, need, alignment; bool write; };
    View views[11] = {
        {a.trellis,a.trellis_bytes,trellis,2,false}, {a.suh,a.suh_bytes,suh,4,false},
        {a.svh,a.svh_bytes,svh,4,false}, {a.jobs,a.jobs_bytes,size_t(a.jobs_count)*16,4,false},
        {a.x,a.x_bytes,input,4,false}, {a.xh,a.xh_bytes,rotated,4,true},
        {a.z,a.z_bytes,projected,4,true}, {a.out,a.out_bytes,output,4,true},
        {a.flag,a.flag_bytes,4,4,true},
    };
    size_t count = 9;
    if (a.in_rows) views[count++] = {a.in_rows, a.in_rows_bytes, a.total_rows * 4, 4, false};
    if (a.out_rows) views[count++] = {a.out_rows, a.out_rows_bytes, a.total_rows * 4, 4, false};
    for (size_t i=0; i<count; ++i) {
        if (!region(views[i].p,views[i].bytes,views[i].need,views[i].alignment)) return false;
        for (size_t j=0; j<i; ++j) {
            if ((views[i].write || views[j].write) &&
                !apart(views[i].p,views[i].need,views[j].p,views[j].need)) return false;
        }
    }
    return true;
}

// What the current device offers the register-decode kernel, read once per device: the compute
// capability major and the ptxVersion of jobs_project_reg, the virtual arch the loaded code was
// compiled for (the __CUDA_ARCH__ its #if saw). The device's capability alone does not say, and
// neither does binaryVersion (the SASS arch): a compute_72-only build JIT-compiled on an sm_86
// device reports binaryVersion 86 but holds the trap stub, which ptxVersion 72 shows (the
// reasoning of attn_prefill.cu's arm_part). Both are 0 when they cannot be read.
struct DeviceSupport {
    int capability_major;
    int ptx_version;
};
DeviceSupport device_support() {
    constexpr int slots = 64;
    // Entry d is read after state[d] holds 2; a failed read is not cached, so the next call
    // reads again. The packed pair is written before the release store of the state.
    static std::atomic<int> state[slots];
    static std::atomic<int> packed[slots];  // capability_major * 1000 + ptx_version
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= slots) return {0, 0};
    if (state[device].load(std::memory_order_acquire) == 2) {
        const int value = packed[device].load(std::memory_order_relaxed);
        return {value / 1000, value % 1000};
    }
    int major = 0;
    cudaFuncAttributes attributes{};
    const bool read =
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
        cudaFuncGetAttributes(&attributes, reinterpret_cast<const void *>(jobs_project_reg)) ==
            cudaSuccess;
    if (!read) {
        cudaGetLastError();  // a failed query leaves a sticky error that the next launch would see
        return {0, 0};
    }
    packed[device].store(major * 1000 + attributes.ptxVersion, std::memory_order_relaxed);
    state[device].store(2, std::memory_order_release);
    return {major, attributes.ptxVersion};
}

// Whether the rollback variable FIFI_EXL3_JOBS_REG leaves the kernel on (anything but exactly 0),
// read once.
bool jobs_registers_enabled() {
    static const bool enabled = [] {
        const char *text = std::getenv("FIFI_EXL3_JOBS_REG");
        return !(text && text[0] == '0' && text[1] == '\0');
    }();
    return enabled;
}

// Whether FIFI_EXL3_JOBS_FUSED_OUT turns the fused output-rotation epilogue on (exactly "1";
// unset or any other value keeps the separate kernel), read once. Default off.
bool jobs_fused_out_enabled() {
    static const bool enabled = [] {
        const char *text = std::getenv("FIFI_EXL3_JOBS_FUSED_OUT");
        return text && text[0] == '1' && text[1] == '\0';
    }();
    return enabled;
}

// Whether a call takes the register-decode kernel: the switch on, a device of compute capability 8
// or more running code compiled for sm_80 or more (the kernel needs mma.m16n8k16 and cp.async),
// 16-byte aligned input rows, trellis and member stride (cp.async copies 16 bytes), and K a
// multiple of the 32 values of a stage. Everything else runs the shared-memory kernels.
bool jobs_use_registers(const inference_cuda_exl3_jobs_args &a, DeviceSupport support,
                        bool enabled) {
    const bool aligned = uintptr_t(a.xh) % 16 == 0 && uintptr_t(a.trellis) % 16 == 0 &&
        (a.members == 1 || (a.trellis_stride * 2) % 16 == 0);
    return enabled && support.capability_major >= 8 && support.ptx_version >= 80 && aligned &&
        a.k % 32 == 0;
}
} // namespace

// What a job-table call is asked to do besides the default: the fused output rotation and the
// warp-wide input rotation, each already resolved from the environment or an explicit request.
struct JobsModes {
    bool fused_out;
    bool wide_input;
    // The rows of xh are already rotated (inference_cuda_exl3_jobs_rotated): skip the rotation.
    bool rotated = false;
};

static int exl3_jobs_launch(const inference_cuda_exl3_jobs_args *a, void *stream,
                            int *out_stage, JobsModes modes) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!a || !stream || !jobs_views(*a)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        const cudaStream_t s = (cudaStream_t)stream;
        const unsigned jobs = a->jobs_count;
        if (!modes.rotated) {
            if (jobs_wide_input_usable(*a, modes.wide_input))
                launch_wide_input<wide_exact>(*a, s);
            else
                jobs_input_rotate<<<dim3(unsigned(a->max_rows * a->k / 128), jobs), 128, 0, s>>>(*a);
        }
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        // The fused epilogue exists only on the register-decode kernel; a call that falls back
        // to the shared-memory kernels keeps the separate output rotation.
        bool fused = false;
        if (failure.err == cudaSuccess) {
            // One CTA covers 64, 128 or 256 rows: the narrowest that holds the tallest job in one
            // row block, so a table of short jobs does not pay for wide CTAs.
            const unsigned tiles = unsigned(a->n / 16);
            if (jobs_use_registers(*a, device_support(), jobs_registers_enabled())) {
                fused = modes.fused_out;
                if (fused)
                    jobs_project_reg_fused<<<dim3(tiles / 8, unsigned((a->max_rows + 63) / 64), jobs), 256, 0, s>>>(*a);
                else
                    jobs_project_reg<<<dim3(tiles / 8, unsigned((a->max_rows + 63) / 64), jobs), 256, 0, s>>>(*a);
            } else if (a->max_rows <= 64) {
                jobs_project<4><<<dim3(tiles / jobs_tiles<4>, 1, jobs), 128, 0, s>>>(*a);
            } else if (a->max_rows <= 128) {
                jobs_project<8><<<dim3(tiles / jobs_tiles<8>, 1, jobs), 256, 0, s>>>(*a);
            } else {
                jobs_project<16><<<dim3(tiles / jobs_tiles<16>, 1, jobs), 512, 0, s>>>(*a);
            }
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        if (failure.err == cudaSuccess && !fused) {
            jobs_output_rotate<<<dim3(unsigned(a->max_rows * a->n / 128), jobs), 128, 0, s>>>(*a);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) { return INFERENCE_CUDA_NATIVE_EXCEPTION; }
}

extern "C" int inference_cuda_exl3_jobs(const inference_cuda_exl3_jobs_args *a,
                                       void *stream, int *out_stage) {
    return exl3_jobs_launch(a, stream, out_stage,
                            {jobs_fused_out_enabled(), jobs_wide_input_enabled()});
}

// The same launches with an explicit fused-output request, for the device tests and the GPU
// qualification (an env flip would race the other tests of the process).
extern "C" int inference_cuda_exl3_jobs_fused_out(const inference_cuda_exl3_jobs_args *a,
                                                 void *stream, int *out_stage, int fused_out) {
    return exl3_jobs_launch(a, stream, out_stage, {fused_out != 0, jobs_wide_input_enabled()});
}

// The same launches with both requests explicit, for the device tests and the GPU qualification.
extern "C" int inference_cuda_exl3_jobs_modes(const inference_cuda_exl3_jobs_args *a, void *stream,
                                              int *out_stage, int fused_out, int wide_input) {
    return exl3_jobs_launch(a, stream, out_stage, {fused_out != 0, wide_input != 0});
}

// The projection and output rotation of a call whose xh already holds the rotated rows (written
// by inference_cuda_exl3_jobs_rotate_pair); `fused_out` is the request of the fused output
// rotation, negative for the environment's.
extern "C" int inference_cuda_exl3_jobs_rotated(const inference_cuda_exl3_jobs_args *a, void *stream,
                                                int *out_stage, int fused_out) {
    return exl3_jobs_launch(a, stream, out_stage,
                            {fused_out < 0 ? jobs_fused_out_enabled() : fused_out != 0, false, true});
}

// The input rotation of two calls that read the same rows, in one pass: call a into its xh by its
// scales, call b into its xh by its scales (exl3_rotate_pair.h). Refused unless the calls
// are consistent and both meet the alignment rules of the wide kernel.
extern "C" int inference_cuda_exl3_jobs_rotate_pair(const inference_cuda_exl3_jobs_args *a,
                                                    const inference_cuda_exl3_jobs_args *b,
                                                    void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!a || !b || !stream || !jobs_views(*a) || !jobs_views(*b) || !jobs_pair_usable(*a, *b))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        launch_pair_input<pair_exact>(*a, *b, (cudaStream_t)stream);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) { return INFERENCE_CUDA_NATIVE_EXCEPTION; }
}

// Whether inference_cuda_exl3_jobs_rotate_pair would accept the two calls (1) or not (0).
extern "C" int inference_cuda_exl3_jobs_pair_usable(const inference_cuda_exl3_jobs_args *a,
                                                    const inference_cuda_exl3_jobs_args *b) {
    if (!a || !b) return INFERENCE_CUDA_INVALID_ARGUMENT;
    return jobs_pair_usable(*a, *b) ? 1 : 0;
}

// Whether the warp-wide input rotation would run for a call with the request `enabled` (the
// alignment rules of jobs_wide_input_usable); -1 for a null block.
extern "C" int inference_cuda_exl3_jobs_wide_input_kernel(const inference_cuda_exl3_jobs_args *a,
                                                          int enabled) {
    if (!a) return INFERENCE_CUDA_INVALID_ARGUMENT;
    return jobs_wide_input_usable(*a, enabled != 0) ? 1 : 0;
}

// The reading of the FIFI_EXL3_JOBS_WARP_ROTATE text: 1 on, 0 off (also for a null text), -1 for a
// value that is neither (which the launch treats as off).
extern "C" int inference_cuda_exl3_jobs_wide_input_setting(const char *text) {
    return jobs_wide_input_setting(text);
}

#ifdef INFERENCE_CUDA_EXL3_ROTATE_MUTANTS
// The input rotation of a job-table call by a deliberately wrong variant (WideMutant, 1 to 7),
// alone: the projection and the output rotation do not run. For the mutation tests only.
extern "C" int inference_cuda_exl3_jobs_rotate_mutant(const inference_cuda_exl3_jobs_args *a,
                                                      void *stream, int *out_stage, int mutant) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!a || !stream || !jobs_views(*a)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const cudaStream_t s = (cudaStream_t)stream;
        switch (mutant) {
            case wide_reversed_butterfly: launch_wide_input<wide_reversed_butterfly>(*a, s); break;
            case wide_drop_suh: launch_wide_input<wide_drop_suh>(*a, s); break;
            case wide_contract: launch_wide_input<wide_contract>(*a, s); break;
            case wide_gather_row: launch_wide_input<wide_gather_row>(*a, s); break;
            case wide_tail: launch_wide_input<wide_tail>(*a, s); break;
            case wide_drop_product_flag: launch_wide_input<wide_drop_product_flag>(*a, s); break;
            case wide_drop_scale_flag: launch_wide_input<wide_drop_scale_flag>(*a, s); break;
            default: return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) { return INFERENCE_CUDA_NATIVE_EXCEPTION; }
}

// The pair rotation by a deliberately wrong variant (PairMutant, 1 to 3), for the mutation tests.
extern "C" int inference_cuda_exl3_jobs_rotate_pair_mutant(const inference_cuda_exl3_jobs_args *a,
                                                           const inference_cuda_exl3_jobs_args *b,
                                                           void *stream, int *out_stage,
                                                           int mutant) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!a || !b || !stream || !jobs_views(*a) || !jobs_views(*b) || !jobs_pair_usable(*a, *b))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const cudaStream_t s = (cudaStream_t)stream;
        switch (mutant) {
            case pair_swapped_outputs: launch_pair_input<pair_swapped_outputs>(*a, *b, s); break;
            case pair_first_scales_twice: launch_pair_input<pair_first_scales_twice>(*a, *b, s); break;
            case pair_drop_b_flag: launch_pair_input<pair_drop_b_flag>(*a, *b, s); break;
            default: return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) { return INFERENCE_CUDA_NATIVE_EXCEPTION; }
}
#endif

extern "C" int inference_cuda_exl3_jobs_kernel(const inference_cuda_exl3_jobs_args *a,
                                               int capability_major, int ptx_version,
                                               int enabled) {
    if (!a) return INFERENCE_CUDA_INVALID_ARGUMENT;
    if (capability_major < 0)
        return jobs_use_registers(*a, device_support(), jobs_registers_enabled()) ? 1 : 0;
    return jobs_use_registers(*a, {capability_major, ptx_version}, enabled != 0) ? 1 : 0;
}
