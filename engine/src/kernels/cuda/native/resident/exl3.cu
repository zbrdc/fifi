/* In-house arithmetic decoder of the admitted EXL3 format; no lookup tables. */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
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
template <int Book>
__device__ __forceinline__ void midrows_projection(const inference_cuda_exl3_linear_args &a,
                                                   half2 *inputs2, unsigned *weights32,
                                                   float *result) {
    namespace wmma = nvcuda::wmma;
    const half *inputs = reinterpret_cast<const half *>(inputs2);
    const half *weights = reinterpret_cast<const half *>(weights32);
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned steps = unsigned(a.k / 16);
    const size_t first_row = size_t(blockIdx.y) * 64 + warp * 16;
    const bool live = first_row < a.rows;
    const MidrowsStates states = midrows_states(threadIdx.x, unsigned(a.bits_twice));

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
        halves[0] = lead[0]; halves[1] = lead[1]; halves[2] = tail[0]; halves[3] = tail[1];
    };
    auto stage = [&]() {
        if (live) {
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j)
                inputs2[warp * 128 + 32 * j + lane] = __floats2half2_rn(x[2 * j], x[2 * j + 1]);
        }
        weights32[states.slot] = midrows_weight_pair<Book>(states, halves, a.flag);
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
    for (unsigned i = threadIdx.x; i < 1024; i += 128) {
        const size_t row = size_t(blockIdx.y) * 64 + i / 16;
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
    a.out[index] = finite_or_flag(value, a.flag);
}

bool linear_views(const inference_cuda_exl3_linear_args &a, uint32_t split_k) {
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
    bool midrows, void *stream, int *out_stage, uint64_t *launches = nullptr) {
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
                case LinearKind::Midrows: project_midrows<<<grid,block,0,(cudaStream_t)stream>>>(*a); break;
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
};
// By-value launch parameters need neither a device descriptor buffer nor an upload.
static_assert(sizeof(GroupEntry) == 80 && offsetof(GroupEntry, first) == 72);
static_assert(sizeof(GroupArgs) == 3920 && sizeof(GroupArgs) <= 4096 &&
    offsetof(GroupArgs, row_to_group) == 3840 && offsetof(GroupArgs, flag) == 3888 &&
    offsetof(GroupArgs, k) == 3896 && offsetof(GroupArgs, pairs) == 3912);

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

__global__ void grouped_output_rotate(const __grid_constant__ GroupArgs p) {
    __shared__ float values[128];
    const GroupEntry &a = p.entries[p.row_to_group[blockIdx.y]];
    const size_t row = size_t(blockIdx.y) - a.first;
    const size_t index = row * p.n + size_t(blockIdx.x) * 128 + threadIdx.x;
    const size_t col = index % p.n;
    float z = a.z[index];
    // Each group retains the original small-row split order, regardless of total P.
    z = finite_or_flag(z + a.out[index], p.flag);
    const float rotated = rotate128(z, values);
    const float value = rotated * a.svh[col] + (a.bias ? a.bias[col] : 0.0f);
    a.out[index] = finite_or_flag(value, p.flag);
}

static bool follows(const void *p, const void *base, size_t bytes) {
    return uintptr_t(p) >= uintptr_t(base) && uintptr_t(p) - uintptr_t(base) == bytes;
}

static bool group_args(const inference_cuda_exl3_linear_args *calls, size_t count, GroupArgs &p) {
    if (!calls || count < 1 || count > grouped_capacity) return false;
    p = {};
    p.k = calls[0].k;
    p.n = calls[0].n;
    p.flag = calls[0].flag;
    p.groups = uint32_t(count);
    size_t first = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto &a = calls[i];
        if (!linear_geometry(a.rows, a.k, a.n, a.bits_twice, a.book, 2, false) ||
            a.rows > 8 || a.k != p.k || a.n != p.n || a.flag != p.flag ||
            a.rows > grouped_capacity - first || !linear_views(a, 2) ||
            !follows(a.xh, calls[0].xh, first * p.k * 4) ||
            !follows(a.z, calls[0].z, first * p.n * 4) ||
            !follows(a.out, calls[0].out, first * p.n * 4)) return false;
        p.entries[i] = {a.trellis, a.suh, a.svh, a.bias, a.x, a.xh, a.z, a.out,
            a.bits_twice, a.book, uint32_t(first), uint32_t(a.rows)};
        for (size_t row = 0; row < a.rows; ++row) p.row_to_group[first + row] = uint8_t(i);
        first += a.rows;
    }
    p.pairs = uint32_t(first);
    struct View { const void *p; size_t bytes; };
    // Unions are used for alias checks only; each live per-group extent was checked above.
    const View writes[] = {{calls[0].xh, first * p.k * 4},
        {calls[0].z, first * p.n * 4}, {calls[0].out, first * p.n * 4}, {p.flag, 4}};
    for (size_t i = 0; i < 4; ++i) {
        if (!region(writes[i].p, writes[i].bytes, writes[i].bytes, 4)) return false;
        for (size_t j = 0; j < i; ++j)
            if (!apart(writes[i].p, writes[i].bytes, writes[j].p, writes[j].bytes)) return false;
    }
    for (size_t i = 0; i < count; ++i) {
        const auto &a = calls[i];
        const View reads[] = {{a.trellis, a.k * a.n / 256 * size_t(a.bits_twice) * 16},
            {a.suh, a.k * 4}, {a.svh, a.n * 4}, {a.bias, a.bias ? a.n * 4 : 0},
            {a.x, a.rows * a.k * 4}};
        for (const auto &read : reads)
            if (read.bytes)
                for (const auto &write : writes)
                    if (!apart(read.p, read.bytes, write.p, write.bytes)) return false;
    }
    return true;
}

template<class Emit>
static inline __attribute__((always_inline)) bool emit_grouped(
    size_t pairs, size_t k, size_t n, Emit &&emit) {
    return emit(LinearEmission{LinearKind::Input, uint32_t(k / 128), uint32_t(pairs), 1, 128, 1, 1}) &&
        emit(LinearEmission{LinearKind::Small, uint32_t(n / 16), uint32_t(pairs), 2, 256, 1, 1}) &&
        emit(LinearEmission{LinearKind::Output, uint32_t(n / 128), uint32_t(pairs), 1, 128, 1, 1});
}


template<bool Observe = false, bool Gather = false>
static inline __attribute__((always_inline)) int grouped_launch(
    const inference_cuda_exl3_linear_args *calls, size_t count,
    void *stream, int *out_stage, uint64_t *launches = nullptr,
    unsigned *variant = nullptr) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        GroupArgs args;
        if (!stream || !group_args(calls, count, args)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        emit_grouped(args.pairs, args.k, args.n, [&](const LinearEmission &e) {
            const dim3 grid(e.gx, e.gy, e.gz), block(e.bx, e.by, e.bz);
            if (e.kind == LinearKind::Input)
                grouped_input_rotate<<<grid,block,0,(cudaStream_t)stream>>>(args);
            else if (e.kind == LinearKind::Small) {
                if constexpr (Gather)
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
                    if (e.kind == LinearKind::Small && variant)
                        *variant = Gather ? 2u : 1u;
                }
            }
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


// Same contract as inference_cuda_exl3_grouped with the shared-table gather variant.
extern "C" int inference_cuda_exl3_grouped_gathered(const inference_cuda_exl3_linear_args *calls,
    size_t count, void *stream, int *out_stage) {
    return grouped_launch<false, true>(calls, count, stream, out_stage);
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

extern "C" __attribute__((section(".text.inference_cuda_exl3_observer"), visibility("hidden")))
int inference_cuda_exl3_observe_grouped(const LinearObserveArgs *a, LinearObservation *out) {
    if (!a || !out || a->rows < 1 || a->rows > grouped_capacity ||
        a->split_k != 2 || a->midrows != 0 || a->fail_at > 3 ||
        !geometry(a->k, a->n, 6, 0)) return INFERENCE_CUDA_INVALID_ARGUMENT;
    *out = {};
    out->complete = emit_grouped(a->rows, a->k, a->n, [&](const LinearEmission &e) {
        out->emissions[out->count++] = e;
        return out->count != a->fail_at;
    });
    return INFERENCE_CUDA_OK;
}
