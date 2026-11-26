#include "../expert_math.cuh"
// Dense matvec for floating-point or packed GGUF weights (see resident/dense_gemv.h).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/dense_gemv.h"
#include "resident/gguf_gemv.cuh"

namespace {

constexpr int kMaxM = 32768;
constexpr int kMaxRows = 1 << 20;
constexpr int kMaxCols = 12288;
constexpr int kMaxLd = 1 << 20;
constexpr int kWarps = 8;
constexpr int kDtypeF16 = 0;
constexpr int kDtypeBf16 = 1;
constexpr int kDtypeGgufNvfp4 = 2;
constexpr int kDtypeF32 = 3;

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }

struct Span {
    const void *p;
    size_t bytes;
};

bool apart(const Span &a, const Span &b) {
    const uintptr_t x = (uintptr_t)a.p, y = (uintptr_t)b.p;
    return x <= y ? a.bytes <= y - x : b.bytes <= x - y;
}

__device__ __forceinline__ float widen(float v) { return v; }
__device__ __forceinline__ float widen(__half v) { return __half2float(v); }
__device__ __forceinline__ float widen(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// grid (ceil(rows / kWarps), m); dynamic shared memory holds one row of x.
template <typename W, bool kVector>
__global__ void dense_gemv_k(const W *__restrict__ w, const float *__restrict__ x,
                             float *__restrict__ y, int rows, int cols, int ld_x, int ld_y) {
    extern __shared__ float staged[];
    const int t = blockIdx.y;
    const float *in = x + (size_t)t * ld_x;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) staged[c] = in[c];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (r >= rows) return;
    const W *row = w + (size_t)r * cols;
    float acc = 0.0f;
    if (kVector) {
        constexpr int width = sizeof(uint4) / sizeof(W);
        const uint4 *wide = reinterpret_cast<const uint4 *>(row);
        for (int v = lane; v < cols / width; v += 32) {
            const uint4 raw = wide[v];
            const W *e = reinterpret_cast<const W *>(&raw);
#pragma unroll
            for (int j = 0; j < width; ++j) acc = fmaf(widen(e[j]), staged[v * width + j], acc);
        }
    } else {
        for (int c = lane; c < cols; c += 32) acc = fmaf(widen(row[c]), staged[c], acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t)t * ld_y + r] = acc;
}

// Each lane reads one packed byte and consumes both nibbles of its GGUF block.
__device__ float gguf_scale(unsigned code) {
    if (code == 0x7f) return 0.0f;
    const unsigned exp = (code >> 3) & 15u, frac = code & 7u;
    return exp == 0 ? float(frac) / 512.0f
                    : __int_as_float(((exp + 120u) << 23) | (frac << 20));
}

__device__ float gguf_fp4(unsigned code) {
    const unsigned m = code & 7u;
    const float v = m < 2 ? float(m) * 0.5f :
        __int_as_float(((126u + (m >> 1)) << 23) | ((m & 1u) << 22));
    return code > 8 ? -v : v;
}

struct SharedWeights {
    __device__ size_t offset(unsigned) const { return 0; }
};

struct SelectedWeights {
    size_t stride;
    uint32_t ids[INFERENCE_CUDA_GGUF_MOE_MAX_PICKS];

    __device__ size_t offset(unsigned slot) const { return stride * ids[slot]; }
};

struct DeviceSelectedWeights {
    size_t stride;
    const int *ids;
    unsigned experts;

    __device__ size_t offset(unsigned slot) const {
        const unsigned id = unsigned(ids[slot]);
        // Faulty router rows publish -1. Bind them to a safe matrix; the sum reports NaN.
        return id < experts ? stride * id : 0;
    }
};

struct SeparateWeights {
    const void *matrices[INFERENCE_CUDA_GGUF_MOE_MAX_PICKS];
};

template <typename Index>
__device__ __forceinline__ const uint8_t *matrix_base(const void *base, Index index, unsigned slot) {
    return static_cast<const uint8_t *>(base) + index.offset(slot);
}

__device__ __forceinline__ const uint8_t *matrix_base(const void *, SeparateWeights index, unsigned slot) {
    return static_cast<const uint8_t *>(index.matrices[slot]);
}

template <typename Index>
__global__ void gguf_nvfp4_gemv(inference_cuda_dense_gemv_args a, Index index) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (row >= a.rows) return;
    const float *x = static_cast<const float *>(a.x) + size_t(blockIdx.y) * a.ld_x;
    const int blocks = a.cols / 64;
    const auto *w = matrix_base(a.w, index, blockIdx.y) + size_t(row) * blocks * 36;
    const int group = lane / 8;
    float sum = 0;
    for (int block = 0; block < blocks; ++block) {
        const auto *b = w + block * 36;
        const float scale = gguf_scale(b[group]);
        const unsigned pair = b[4 + lane];
        const int col = block * 64 + group * 16 + lane % 8;
        sum = fmaf(gguf_fp4(pair & 15u) * scale, x[col], sum);
        sum = fmaf(gguf_fp4(pair >> 4) * scale, x[col + 8], sum);
    }
    sum = warp_sum(sum);
    if (lane == 0) static_cast<float *>(a.y)[size_t(blockIdx.y) * a.ld_y + row] = sum;
}

// Bytes of weight row one warp stages at a time.
constexpr int kStageBytes = 2048;

// Copies the `bytes` bytes at `src` into a warp's shared `stage`, which starts on a 16-byte
// boundary and holds bytes + 16, and returns the copy. The copy keeps the source's offset within
// 16 bytes, so the aligned body moves in 16-byte words and only the head and tail (under 16 bytes
// each) move bytewise; no read leaves the span. Every lane of the warp calls it; the copy is
// visible to the whole warp.
__device__ __forceinline__ const uint8_t *stage_span(const uint8_t *src, int bytes, uint8_t *stage,
                                                     int lane) {
    const int skew = int(uintptr_t(src) & 15);
    uint8_t *dst = stage + skew;
    const int head = min(bytes, (16 - skew) & 15);
    const int words = (bytes - head) / 16;
    const int tail = head + words * 16;
    const auto *in = reinterpret_cast<const uint4 *>(src + head);
    auto *out = reinterpret_cast<uint4 *>(dst + head);
    for (int i = lane; i < words; i += 32) out[i] = __ldg(in + i);
    if (lane < head) dst[lane] = src[lane];
    if (tail + lane < bytes) dst[tail + lane] = src[tail + lane];
    __syncwarp();
    return dst;
}

// One warp per weight row. The warp stages its row a chunk of whole blocks at a time with wide
// loads, so each warp has the chunk's bytes in flight at once, then each lane decodes its columns
// from shared memory. Lane l adds the products of columns l, l + 32, ... of every block in block
// order and the warp sums the lanes with the XOR butterfly: other kernels reproduce these bits.
template <int Type, int Block, int Bytes, typename Index>
__global__ void gguf_packed_gemv(inference_cuda_dense_gemv_args a, Index index) {
    constexpr int kChunk = kStageBytes / Bytes;
    // Each warp's stage starts on a 16-byte boundary, so the copy's offset within 16 bytes is
    // the source's: the wide stores stay aligned and every decoder load keeps its alignment.
    constexpr int kPitch = (kChunk * Bytes + 16 + 15) / 16 * 16;
    __shared__ __align__(16) uint8_t stage[kWarps][kPitch];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarps + warp;
    if (row >= a.rows) return;
    const auto *x = static_cast<const float *>(a.x) + size_t(blockIdx.y) * a.ld_x;
    const int blocks = a.cols / Block;
    const auto *w = matrix_base(a.w, index, blockIdx.y) + size_t(row) * blocks * Bytes;
    // A lane's columns of one block, and the blocks it decodes before it adds their products:
    // the decodes and the loads of a batch do not wait on the sum, so they are issued together,
    // and the adds then run in block and column order, as one block at a time would.
    constexpr int kCols = Block / 32;
    constexpr int kBatch = kCols >= 8 ? 1 : 8 / kCols;
    float sum = 0.0f;
    for (int first = 0; first < blocks; first += kChunk) {
        const int count = min(kChunk, blocks - first);
        const uint8_t *staged = stage_span(w + size_t(first) * Bytes, count * Bytes, stage[warp], lane);
        int block = 0;
        for (; block + kBatch <= count; block += kBatch) {
            float value[kBatch][kCols], input[kBatch][kCols];
#pragma unroll
            for (int b = 0; b < kBatch; ++b) {
                const auto *weights = staged + (block + b) * Bytes;
                const float *in = x + (first + block + b) * Block;
#pragma unroll
                for (int c = 0; c < kCols; ++c) {
                    value[b][c] = fifi_gguf::decode<Type>(weights, lane + 32 * c);
                    input[b][c] = __ldg(in + lane + 32 * c);
                }
            }
#pragma unroll
            for (int b = 0; b < kBatch; ++b) {
#pragma unroll
                for (int c = 0; c < kCols; ++c) sum = fmaf(value[b][c], input[b][c], sum);
            }
        }
        for (; block < count; ++block) {
            const auto *weights = staged + block * Bytes;
            const float *in = x + (first + block) * Block;
#pragma unroll
            for (int col = lane; col < Block; col += 32) {
                sum = fmaf(fifi_gguf::decode<Type>(weights, col), __ldg(in + col), sum);
            }
        }
        __syncwarp();
    }
    sum = warp_sum(sum);
    if (lane == 0) static_cast<float *>(a.y)[size_t(blockIdx.y) * a.ld_y + row] = sum;
}

// One shared matrix per call. Each warp owns the same weight row and lane
// columns as gguf_packed_gemv, but reuses each decoded weight in Tile separate
// F32 accumulators. Do not transpose the lane mapping or split K: the exact
// packed-GEMV lane FMA sequence and XOR butterfly are the reference.
template <int Type, int Block, int Bytes, int Tile>
__global__ void gguf_exact_expert_tile(inference_cuda_dense_gemv_args a) {
    extern __shared__ float staged[];
    const int first = blockIdx.y * Tile;
    const auto *x = static_cast<const float *>(a.x);
#pragma unroll
    for (int t = 0; t < Tile; ++t) {
        // The tail stages a valid row in unused slots; no invalid reads or stores.
        const auto *in = x + size_t(first + (first + t < a.m ? t : 0)) * a.ld_x;
        for (int col = threadIdx.x; col < a.cols; col += blockDim.x) {
            staged[t * a.cols + col] = in[col];
        }
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (row >= a.rows) return;
    const int blocks = a.cols / Block;
    const auto *w = static_cast<const uint8_t *>(a.w) + size_t(row) * blocks * Bytes;
    float sums[Tile] = {};
    for (int block = 0; block < blocks; ++block) {
        const auto *weights = w + block * Bytes;
#pragma unroll
        for (int col = lane; col < Block; col += 32) {
            const float value = fifi_gguf::decode<Type>(weights, col);
#pragma unroll
            for (int t = 0; t < Tile; ++t) {
                sums[t] = fmaf(value, staged[t * a.cols + block * Block + col], sums[t]);
            }
        }
    }
    auto *y = static_cast<float *>(a.y);
#pragma unroll
    for (int t = 0; t < Tile; ++t) {
        const float sum = warp_sum(sums[t]);
        if (lane == 0 && first + t < a.m) y[size_t(first + t) * a.ld_y + row] = sum;
    }
}

constexpr int kInklingGateWidth = 2304;

// Called only after the ordinary GEMV operand and shape checks.
constexpr int exact_expert_tile(int type, int m, int rows, int cols) {
    if (type == 14) {
        // Inkling attention projections at the measured narrow and wide tile shapes.
        return rows == 3072 && cols == 2304 && (m == 8 || m == 63 || m == 64) ? 2 : 0;
    }
    // Q2_0, IQ4_NL and Q8_0 tile four rows when the activations are narrow enough.
    const bool four_rows = type == 42 || type == 20 || type == 8;
    if (m < 2 || m > 512 || rows > kInklingGateWidth || cols > kInklingGateWidth ||
        (type != 16 && !four_rows)) return 0;
    // At most 18 KiB of activation staging per CTA.
    return four_rows && cols <= kInklingGateWidth / 2 && m >= 4 ? 4 : 2;
}

static_assert(exact_expert_tile(16, 512, 2304, 2304) == 2 &&
              exact_expert_tile(42, 5, 2304, 1152) == 4 &&
              exact_expert_tile(42, 3, 2304, 1152) == 2 &&
              exact_expert_tile(42, 5, 2304, 1216) == 2 &&
              exact_expert_tile(20, 5, 2304, 1152) == 4 &&
              exact_expert_tile(20, 5, 2304, 2304) == 2 &&
              exact_expert_tile(8, 2, 2304, 2304) == 2 &&
              exact_expert_tile(8, 31, 2304, 1152) == 4);
static_assert(exact_expert_tile(16, 1, 2304, 2304) == 0 &&
              exact_expert_tile(16, 513, 2304, 2304) == 0 &&
              exact_expert_tile(16, 2, 2305, 2304) == 0 &&
              exact_expert_tile(16, 2, 2304, 2560) == 0 &&
              exact_expert_tile(13, 2, 2304, 2304) == 0 &&
              exact_expert_tile(20, 1, 2304, 2304) == 0);
static_assert(exact_expert_tile(14, 8, 3072, 2304) == 2 &&
              exact_expert_tile(14, 63, 3072, 2304) == 2 &&
              exact_expert_tile(14, 64, 3072, 2304) == 2 &&
              exact_expert_tile(14, 1, 3072, 2304) == 0 &&
              exact_expert_tile(14, 8, 3072, 2560) == 0 &&
              exact_expert_tile(14, 64, 2304, 2304) == 0);

// One lane owns each eight-value IQ2 group. Transpose each activation block
// once per CTA so the eight inner FMA steps read shared memory without bank
// conflicts. Weight values stay F32; only the dot-product reduction order changes.
template <int Type, int Bytes>
__global__ void inkling_iq2_gate_gemv(const float *x, const uint8_t *gate, float *y,
                                      DeviceSelectedWeights index) {
    constexpr int cols = kInklingGateWidth, rows = kInklingGateWidth, blocks = cols / 256;
    __shared__ float staged[cols];
    const int tid = threadIdx.x;
    for (int col = tid; col < cols; col += kWarps * 32) {
        staged[(col / 256) * 256 + (col % 8) * 32 + (col % 256) / 8] = x[col];
    }
    __syncthreads();

    const int lane = tid & 31;
    const int row = blockIdx.x * kWarps + (tid >> 5);
    const auto *w = gate + index.offset(blockIdx.y) + size_t(row) * blocks * Bytes;
    float sum = 0;
    for (int block = 0; block < blocks; ++block) {
        const auto *b = w + block * Bytes;
        const auto group = fifi_gguf::iq2_group<Type>(b, lane / 4, lane % 4, fifi_gguf::half_value(b));
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            sum = fmaf(fifi_gguf::grid_value(group.grid, j, group.scale, group.signs),
                       staged[block * 256 + j * 32 + lane], sum);
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) y[size_t(blockIdx.y) * rows + row] = sum;
}

bool launch_inkling_iq2_gate(const inference_cuda_gguf_moe_args &a, DeviceSelectedWeights gate,
                             cudaStream_t stream, Failure &failure) {
    if (a.hidden != kInklingGateWidth || 2 * a.middle != kInklingGateWidth || a.count != 8) return false;
    const dim3 grid(kInklingGateWidth / kWarps, a.count);
    switch (a.gate_dtype - fifi_gguf::kDtypeBase) {
#define LAUNCH_GATE(id, bytes) \
    case id: \
        inkling_iq2_gate_gemv<id, bytes><<<grid, kWarps * 32, 0, stream>>>( \
            a.x, static_cast<const uint8_t *>(a.gate), a.scratch, gate); \
        break;
        LAUNCH_GATE(16, 66)
        LAUNCH_GATE(17, 74)
        LAUNCH_GATE(22, 82)
#undef LAUNCH_GATE
    default: return false;
    }
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    return true;
}

#include "resident/inkling_decode_weights.cuh"
#include "resident/inkling_decode_control_pair.cuh"
#include "resident/inkling_decode_attention_group.cuh"
#include "resident/inkling_gate_dot.cuh"
#include "resident/inkling_q2_down.cuh"

size_t weight_row_bytes(int dtype, int cols) {
    switch (dtype) {
    case kDtypeF16:
    case kDtypeBf16:
        return size_t(cols) * 2;
    case kDtypeF32:
        return size_t(cols) * sizeof(float);
    case kDtypeGgufNvfp4:
        return cols % 64 == 0 ? size_t(cols / 64) * 36 : 0;
    default:
        return fifi_gguf::row_bytes(dtype, cols);
    }
}

bool valid_operands(const inference_cuda_dense_gemv_args &a, void *stream) {
    if (!stream || !a.w || !a.x || !a.y ||
        a.m < 1 || a.rows < 1 || a.cols < 1 || a.ld_x < a.cols || a.ld_y < a.rows) {
        return false;
    }
    const size_t weight_alignment = a.weight_dtype == kDtypeF32 ? 4 : 2;
    if (!aligned(a.w, weight_alignment) || !aligned(a.x, 4) || !aligned(a.y, 4)) return false;
    const size_t m = (size_t)a.m;
    const size_t row_bytes = weight_row_bytes(a.weight_dtype, a.cols);
    if (row_bytes == 0) return false;
    const Span weights = {a.w, size_t(a.rows) * row_bytes};
    const Span in = {a.x, ((m - 1) * a.ld_x + a.cols) * 4};
    const Span out = {a.y, ((m - 1) * a.ld_y + a.rows) * 4};
    return apart(out, in) && apart(out, weights);
}

bool valid(const inference_cuda_dense_gemv_args &a, void *stream) {
    return a.m <= kMaxM && a.rows <= kMaxRows && a.cols <= kMaxCols &&
           a.ld_x <= kMaxLd && a.ld_y <= kMaxLd && valid_operands(a, stream);
}

constexpr int kWmmaTile = 32;
constexpr int kWmmaStride = kWmmaTile + 8;
constexpr int kWmmaThreads = 128;

// Native GGUF blocks are decoded once per tile and rounded to F16. External
// tensor scales remain the caller's responsibility, as in the F32 matvec.
template <int Type, int Block, int Bytes>
__device__ __forceinline__ __half wmma_weight(const uint8_t *row, int col) {
    const uint8_t *block = row + size_t(col / Block) * Bytes;
    const int i = col % Block;
    if constexpr (Type == 40) {
        const int group = i / 16;
        const unsigned pair = block[4 + 8 * group + i % 8];
        return __float2half_rn(gguf_fp4((pair >> (4 * ((i / 8) % 2))) & 15) *
                                 gguf_scale(block[group]));
    } else {
        return __float2half_rn(fifi_gguf::decode<Type>(block, i));
    }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
using WmmaAccumulator = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>;

template <int Type, int Block, int Bytes>
__device__ void accumulate_packed_tile(inference_cuda_dense_gemv_args a,
                                       size_t token, size_t output, int warp,
                                       __half *x_tile, __half *w_tile, WmmaAccumulator &accum) {
    using namespace nvcuda::wmma;
    const auto *x = static_cast<const float *>(a.x);
    const auto *w = static_cast<const uint8_t *>(a.w);
    const size_t row_bytes = size_t(a.cols / Block) * Bytes;
    fill_fragment(accum, 0.0f);
    for (int base = 0; base < a.cols; base += kWmmaTile) {
        for (int i = threadIdx.x; i < kWmmaTile * kWmmaTile; i += kWmmaThreads) {
            const int row = i / kWmmaTile;
            const int col = base + i % kWmmaTile;
            const int at = row * kWmmaStride + i % kWmmaTile;
            x_tile[at] = token + row < size_t(a.m) ?
                __float2half_rn(x[(token + row) * a.ld_x + col]) : __float2half_rn(0.0f);
            w_tile[at] = output + row < size_t(a.rows) ?
                wmma_weight<Type, Block, Bytes>(w + (output + row) * row_bytes, col) :
                __float2half_rn(0.0f);
        }
        __syncthreads();
#pragma unroll
        for (int step = 0; step < kWmmaTile; step += 16) {
            fragment<matrix_a, 16, 16, 16, __half, row_major> left;
            fragment<matrix_b, 16, 16, 16, __half, col_major> right;
            load_matrix_sync(left, x_tile + (warp / 2) * 16 * kWmmaStride + step, kWmmaStride);
            load_matrix_sync(right, w_tile + (warp % 2) * 16 * kWmmaStride + step, kWmmaStride);
            mma_sync(accum, left, right, accum);
        }
        __syncthreads();
    }
}
#endif

template <int Type, int Block, int Bytes>
__global__ void gguf_wmma_kernel(inference_cuda_dense_gemv_args a) {
    const size_t output = size_t(blockIdx.x) * kWmmaTile;
    auto *y = static_cast<float *>(a.y);
#if __CUDA_ARCH__ >= 700
    __shared__ __align__(32) __half x_tile[kWmmaTile * kWmmaStride];
    __shared__ __align__(32) __half w_tile[kWmmaTile * kWmmaStride];
    __shared__ __align__(32) float results[kWmmaTile * kWmmaTile];
    const int warp = threadIdx.x / 32;
#endif
    // Grid Y is hardware-limited. Return rows in the original order even if
    // one CTA must process multiple token tiles.
    for (size_t token = size_t(blockIdx.y) * kWmmaTile; token < size_t(a.m);
         token += size_t(gridDim.y) * kWmmaTile) {
#if __CUDA_ARCH__ >= 700
        WmmaAccumulator accum;
        accumulate_packed_tile<Type, Block, Bytes>(a, token, output, warp, x_tile, w_tile, accum);
        nvcuda::wmma::store_matrix_sync(
            results + (warp / 2) * 16 * kWmmaTile + (warp % 2) * 16, accum,
            kWmmaTile, nvcuda::wmma::mem_row_major);
        __syncthreads();
#endif
        for (int i = threadIdx.x; i < kWmmaTile * kWmmaTile; i += kWmmaThreads) {
            const size_t t = token + i / kWmmaTile;
            const size_t r = output + i % kWmmaTile;
            if (t < size_t(a.m) && r < size_t(a.rows)) {
#if __CUDA_ARCH__ >= 700
                y[t * a.ld_y + r] = results[i];
#else
                // This path requires Tensor Cores. An unsupported raw launch
                // must not leave stale results in the output.
                y[t * a.ld_y + r] = __int_as_float(0x7fc00000);
#endif
            }
        }
        __syncthreads();
    }
}

int launch_wmma(inference_cuda_dense_gemv_args a, void *stream, int *out_stage) {
    const int kind = a.weight_dtype == kDtypeGgufNvfp4 ? 40 :
        a.weight_dtype - fifi_gguf::kDtypeBase;
    const unsigned token_tiles = (a.m - 1) / kWmmaTile + 1;
    const dim3 grid((a.rows - 1) / kWmmaTile + 1, token_tiles < 65535 ? token_tiles : 65535);
    const auto cuda_stream = (cudaStream_t)stream;
    switch (kind) {
#define WMMA_CASE(id, block, bytes) case id: \
        gguf_wmma_kernel<id, block, bytes><<<grid, kWmmaThreads, 0, cuda_stream>>>(a); \
        break;
        WMMA_CASE(16, 256, 66)
        WMMA_CASE(17, 256, 74)
        WMMA_CASE(19, 256, 50)
        WMMA_CASE(22, 256, 82)
        WMMA_CASE(29, 256, 56)
        WMMA_CASE(40, 64, 36)
        WMMA_CASE(42, 64, 18)
#undef WMMA_CASE
    default:
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

template <typename W>
void launch(const inference_cuda_dense_gemv_args &a, cudaStream_t s, Failure &failure) {
    const dim3 grid((a.rows + kWarps - 1) / kWarps, a.m);
    const size_t shared = (size_t)a.cols * sizeof(float);
    const bool vector = a.cols % (sizeof(uint4) / sizeof(W)) == 0 && aligned(a.w, 16);
    if (a.m == 1 && ((a.cols == 4608 && a.rows == 64 && a.weight_dtype == kDtypeF32) ||
                     (a.cols == 2304 && (a.rows == 24 || a.rows == 32) && a.weight_dtype == kDtypeBf16)) &&
        decode_split(a) && vector && aligned(a.x,16)) {
        if constexpr (std::is_same_v<W,float>) decode_controller_down_exact<<<a.rows,256,0,s>>>(a);
        else decode_bf16_exact<<<a.rows,32,0,s>>>(a);
    } else if (vector) {
        dense_gemv_k<W, true><<<grid, kWarps * 32, shared, s>>>(
            (const W *)a.w, (const float *)a.x, (float *)a.y, a.rows, a.cols, a.ld_x, a.ld_y);
    } else {
        dense_gemv_k<W, false><<<grid, kWarps * 32, shared, s>>>(
            (const W *)a.w, (const float *)a.x, (float *)a.y, a.rows, a.cols, a.ld_x, a.ld_y);
    }
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
}

template <typename Index>
void launch_packed(const inference_cuda_dense_gemv_args &a, cudaStream_t stream,
                   Failure &failure, Index index) {
    const dim3 grid((a.rows + kWarps - 1) / kWarps, a.m);
    if (decode_split(a) && launch_decode_split(a, index, stream)) {
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        return;
    }
    if (a.weight_dtype == kDtypeGgufNvfp4) {
        gguf_nvfp4_gemv<Index><<<grid, kWarps * 32, 0, stream>>>(a, index);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        return;
    }
    switch (a.weight_dtype - fifi_gguf::kDtypeBase) {
#define LAUNCH(id, block, bytes) \
    case id: \
        gguf_packed_gemv<id, block, bytes, Index><<<grid, kWarps * 32, 0, stream>>>(a, index); \
        break;
        FIFI_GGUF_GEMV_TYPES(LAUNCH)
#undef LAUNCH
    }
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
}

struct MoeSizes {
    size_t gate_stride;
    size_t down_stride;
};

bool good_scale(float value) { return std::isfinite(value) && value > 0.0f; }

bool valid_moe(const inference_cuda_gguf_moe_args &a,
               const inference_cuda_gguf_moe_matrices *matrices, void *stream, MoeSizes &sizes) {
    if (!stream || !a.x || !a.scratch || !a.y) return false;
    if (a.hidden < 1 || a.hidden > kMaxCols || a.middle < 1 || a.middle > kMaxCols ||
        a.experts < 1 || a.experts > kMaxRows ||
        a.count < 1 || a.count > INFERENCE_CUDA_GGUF_MOE_MAX_PICKS ||
        !good_scale(a.gate_cap) || !good_scale(a.up_cap)) return false;
    if (a.gate_dtype < kDtypeGgufNvfp4 || a.down_dtype < kDtypeGgufNvfp4 ||
        a.gate_dtype == kDtypeF32 || a.down_dtype == kDtypeF32) return false;
    sizes = {2 * size_t(a.middle) * weight_row_bytes(a.gate_dtype, a.hidden),
             size_t(a.hidden) * weight_row_bytes(a.down_dtype, a.middle)};
    if (!sizes.gate_stride || !sizes.down_stride || !aligned(a.x, 4) ||
        !aligned(a.scratch, 4) || !aligned(a.y, 4)) return false;

    const Span x = {a.x, size_t(a.hidden) * sizeof(float)};
    const Span scratch = {a.scratch, size_t(a.count) * (3 * a.middle + a.hidden) * sizeof(float)};
    const Span result = {a.y, size_t(a.hidden) * sizeof(float)};
    if (!apart(scratch, result) || !apart(x, scratch) || !apart(x, result)) return false;
    const auto weights_fit = [&](Span gate, Span down) {
        const Span weights[] = {gate, down};
        for (const Span &weight : weights) {
            if (!weight.p || !aligned(weight.p, 2) ||
                !apart(weight, scratch) || !apart(weight, result)) return false;
        }
        return true;
    };
    if (!matrices && !weights_fit({a.gate, sizes.gate_stride * a.experts},
                                  {a.down, sizes.down_stride * a.experts})) return false;
    if (a.device_ids || a.device_weights || a.gate_scales || a.down_scales) {
        if (matrices || a.count > a.experts) return false;
        const size_t picks = size_t(a.count) * sizeof(float);
        const size_t scales = size_t(a.experts) * sizeof(float);
        const Span inputs[] = {{a.device_ids, picks}, {a.device_weights, picks},
                               {a.gate_scales, scales}, {a.down_scales, scales}};
        for (const Span &input : inputs) {
            if (!input.p || !aligned(input.p, 4) || !apart(input, scratch) ||
                !apart(input, result)) return false;
        }
        return true;
    }
    for (int i = 0; i < a.count; ++i) {
        const auto pick = a.picks[i];
        if (pick.expert >= uint32_t(a.experts) || !good_scale(pick.gate_scale) ||
            !good_scale(pick.down_scale) || !std::isfinite(pick.weight)) return false;
        if (matrices && !weights_fit({matrices->gate[i], sizes.gate_stride},
                                     {matrices->down[i], sizes.down_stride})) return false;
    }
    return true;
}

template <bool DevicePicks>
__global__ void activate_moe(inference_cuda_gguf_moe_args a, float *middle) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= a.middle) return;
    const size_t offset = size_t(blockIdx.y) * a.middle;
    float scale;
    if constexpr (DevicePicks) {
        const unsigned id = unsigned(a.device_ids[blockIdx.y]);
        scale = a.gate_scales[id < unsigned(a.experts) ? id : 0];
    } else {
        scale = a.picks[blockIdx.y].gate_scale;
    }
    const float gate = a.scratch[2 * offset + col] * scale;
    const float up = a.scratch[2 * offset + a.middle + col] * scale;
    middle[offset + col] = clipped_expert_activation(gate, up, a.gate_cap, a.up_cap);
}

template <bool DevicePicks>
__global__ void sum_moe(inference_cuda_gguf_moe_args a, const float *down) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= a.hidden) return;
    float out = 0.0f;
    for (int slot = 0; slot < a.count; ++slot) {
        float row = down[size_t(slot) * a.hidden + col];
        float weight;
        if constexpr (DevicePicks) {
            const unsigned id = unsigned(a.device_ids[slot]);
            if (id >= unsigned(a.experts)) {
                a.y[col] = __int_as_float(0x7fc00000);
                return;
            }
            row = __fmul_rn(row, a.down_scales[id]);
            weight = a.device_weights[slot];
        } else {
            row = __fmul_rn(row, a.picks[slot].down_scale);
            weight = a.picks[slot].weight;
        }
        out = __fadd_rn(out, __fmul_rn(weight, row));
    }
    a.y[col] = out;
}

bool launch_inkling_q2_down(const inference_cuda_gguf_moe_args &a, DeviceSelectedWeights index,
                            const float *middle, float *down, bool split, cudaStream_t stream,
                            Failure &failure) {
    if (!inkling_q2_down_shape(a.hidden, a.middle, a.count, a.down_dtype)) return false;
    const dim3 grid(kInklingDownRows / kQ2RowsPerCta, kInklingDownPicks);
    if (split && aligned(middle, 16)) {
        decode_q2_pair<<<grid, kWarps * 32, 0, stream>>>(
            static_cast<const uint8_t *>(a.down), middle, down, index);
    } else {
        inkling_q2_down<<<grid, kWarps * 32, 0, stream>>>(
            static_cast<const uint8_t *>(a.down), middle, down, index);
    }
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    return true;
}

template <bool DevicePicks = false, typename Index>
void launch_moe(const inference_cuda_gguf_moe_args &a, Index gate, Index selected_down,
                cudaStream_t stream, Failure &failure, int *schedule) {
    float *middle = a.scratch + size_t(a.count) * 2 * a.middle;
    float *down = middle + size_t(a.count) * a.middle;
    const int request = schedule ? *schedule : 0;
    constexpr int threads = 256;
    if (request & (INFERENCE_CUDA_GGUF_MOE_ACTIVATE_ONLY | INFERENCE_CUDA_GGUF_MOE_SUM_ONLY)) {
        if (request & INFERENCE_CUDA_GGUF_MOE_ACTIVATE_ONLY) {
            const dim3 grid((a.middle + threads - 1) / threads, a.count);
            activate_moe<DevicePicks><<<grid, threads, 0, stream>>>(a, middle);
        } else {
            sum_moe<DevicePicks><<<(a.hidden + threads - 1) / threads, threads, 0, stream>>>(a, down);
        }
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        *schedule = 0;
        return;
    }
    const int gate_request = request & INFERENCE_CUDA_GGUF_MOE_GATE_MASK;
    const bool split = (request & INFERENCE_CUDA_GGUF_MOE_DECODE_SPLIT) != 0;
    const int flags = split ? INFERENCE_CUDA_DENSE_GEMV_DECODE_SPLIT : 0;
    int selected_gate = 0;
    if constexpr (DevicePicks) {
        if (gate_request == 2 && launch_inkling_gate_dot(a, gate, middle, split, stream, failure)) {
            selected_gate = 2;
        } else if (gate_request != 0 && launch_inkling_iq2_gate(a, gate, stream, failure)) {
            selected_gate = 1;
        }
    }
    if (schedule) *schedule = selected_gate;
    if (!selected_gate) {
        launch_packed({a.gate, a.x, a.scratch, a.count, 2 * a.middle, a.hidden,
                       0, 2 * a.middle, a.gate_dtype, flags}, stream, failure, gate);
    }
    if (!failure.ok()) return;

    const dim3 grid((a.middle + threads - 1) / threads, a.count);
    activate_moe<DevicePicks><<<grid, threads, 0, stream>>>(a, middle);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (!failure.ok()) return;

    bool exact_down = false;
    if constexpr (DevicePicks) {
        exact_down = (request & INFERENCE_CUDA_GGUF_MOE_EXACT_Q2_DOWN) &&
            launch_inkling_q2_down(a, selected_down, middle, down, split, stream, failure);
        if (schedule && exact_down && failure.ok()) *schedule |= INFERENCE_CUDA_GGUF_MOE_EXACT_Q2_DOWN;
    }
    if (!exact_down) {
        launch_packed({a.down, middle, down, a.count, a.hidden, a.middle,
                       a.middle, a.hidden, a.down_dtype, flags}, stream, failure, selected_down);
    }
    if (!failure.ok()) return;

    sum_moe<DevicePicks><<<(a.hidden + threads - 1) / threads, threads, 0, stream>>>(a, down);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
}

}  // namespace

extern "C" int inference_cuda_dense_gemv_launch(const inference_cuda_dense_gemv_args *args,
                                                void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid(*args, stream)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        const auto cuda_stream = (cudaStream_t)stream;
        if (args->weight_dtype == kDtypeF16) {
            launch<__half>(*args, cuda_stream, failure);
        } else if (args->weight_dtype == kDtypeBf16) {
            launch<__nv_bfloat16>(*args, cuda_stream, failure);
        } else if (args->weight_dtype == kDtypeF32) {
            launch<float>(*args, cuda_stream, failure);
        } else {
            launch_packed(*args, cuda_stream, failure, SharedWeights{});
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_decode_attention_group(const inference_cuda_dense_gemv_args *calls,
        int count, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!calls || (count != 6 && count != 8)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        DecodeAttentionGroup group{};
        int blocks=0;
        for (int i=0;i<count;++i) {
            const auto &a=calls[i];
            int n=decode_attention_blocks(a);
            if (!n || !valid(a,stream)) return INFERENCE_CUDA_INVALID_ARGUMENT;
            group.calls[i]=a;
            group.limits[i]=blocks+=n;
        }
        for (int i=count;i<8;++i) group.limits[i]=blocks;
        for (int i=0;i<count;++i) {
            Span dst={calls[i].y,size_t(calls[i].rows)*4};
            for (int j=0;j<count;++j) if (j!=i) {
                const auto &b=calls[j];
                if (!apart(dst,{b.y,size_t(b.rows)*4}) || !apart(dst,{b.x,2304*4}) ||
                    !apart(dst,{b.w,size_t(b.rows)*weight_row_bytes(b.weight_dtype,2304)}))
                    return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
        }
        Failure f;
        decode_attention_group<<<blocks,128,0,(cudaStream_t)stream>>>(group);
        f.record(cudaGetLastError(),INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage,f.stage);
        return (int)f.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_decode_control_pair(const inference_cuda_dense_gemv_args *a,
        const inference_cuda_dense_gemv_args *b, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        const auto eligible = [stream](const inference_cuda_dense_gemv_args *p) {
            return p && p->weight_dtype == kDtypeF32 && p->m == 1 && p->rows == 2304 &&
                   p->cols == 64 && aligned(p->w, 16) && valid(*p, stream);
        };
        if (!eligible(a) || !eligible(b) || a->x != b->x ||
            !apart({a->y,2304*4},{b->y,2304*4}) ||
            !apart({a->y,2304*4},{b->w,2304*64*4}) ||
            !apart({b->y,2304*4},{a->w,2304*64*4})) return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure f;
        decode_control_pair<<<dim3(144,2),256,0,(cudaStream_t)stream>>>(*a,*b);
        f.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, f.stage);
        return (int)f.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

namespace {

#include "resident/inkling_q8_visit.cuh"
#include "resident/inkling_exact_visit_tile.cuh"
#include "resident/inkling_qkv_visit.cuh"
#include "resident/inkling_prefetch_select.cuh"
#include "resident/inkling_prefetch_large.cuh"
#include "resident/inkling_prefetch_staged.cuh"
#include "resident/inkling_split_visit_tile.cuh"
#include "resident/inkling_iq2_visit_tile.cuh"
#include "resident/inkling_controller_pair.cuh"
#include "resident/inkling_f32_visit.cuh"
#include "resident/inkling_panel_mma.cuh"
#include "resident/inkling_expert_group.cuh"
#ifdef FIFI_VISIT_PROBE
#include "resident/inkling_stream_visit.cuh"
#include "resident/inkling_splitn_visit.cuh"
#include "resident/inkling_grid_split.cuh"
#include "resident/inkling_warp_visit.cuh"
#include "resident/inkling_quartet.cuh"
#include "resident/inkling_q8_residual_imma.cuh"
#include "resident/inkling_q6_imma.cuh"
#include "resident/inkling_expert_octet.cuh"
#include "resident/inkling_interleave_visit.cuh"
#include "resident/inkling_token_visit.cuh"
#include "resident/inkling_simt_visit.cuh"
#include "resident/inkling_panel_wide.cuh"
#include "resident/inkling_weight_panel.cuh"
#include "resident/inkling_group64.cuh"
#include "resident/inkling_occupancy_tile.cuh"
#endif

// Two adjacent input rows share a decoded Q8_0 weight. Keep lane c on
// column block*32+c and preserve the baseline FMA order within each row.
__global__ void inkling_q8_reuse_pair(inference_cuda_dense_gemv_args a) {
    extern __shared__ float2 paired[];
    const int first = blockIdx.y * 2;
    const bool has_second = first + 1 < a.m;
    const auto *x = static_cast<const float *>(a.x) + size_t(first) * a.ld_x;
    const auto *second = has_second ? x + a.ld_x : x;
    for (int c = threadIdx.x; c < a.cols; c += blockDim.x) {
        paired[c] = make_float2(x[c], second[c]);
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (row >= a.rows) return;
    const int blocks = a.cols / 32;
    const auto *weights = static_cast<const uint8_t *>(a.w) + size_t(row) * blocks * 34;
    float lo = 0.0f, hi = 0.0f;
    for (int k = 0; k < blocks; ++k) {
        const float w = fifi_gguf::decode<8>(weights + k * 34, lane);
        const float2 v = paired[k * 32 + lane];
        lo = fmaf(w, v.x, lo);
        hi = fmaf(w, v.y, hi);
    }
    lo = warp_sum(lo);
    hi = warp_sum(hi);
    if (lane == 0) {
        auto *y = static_cast<float *>(a.y) + size_t(first) * a.ld_y + row;
        y[0] = lo;
        if (has_second) y[a.ld_y] = hi;
    }
}

// Only the 2304 x 3072 attention-output projection uses this path. Stage
// a third of K at a time (16 KiB/CTA), retaining four independent F32 chains.
__global__ void inkling_q8_reuse_four(inference_cuda_dense_gemv_args a) {
    constexpr int chunk = 1024;
    constexpr int cols = 3072;
    constexpr int block_bytes = 34;
    __shared__ float4 staged[chunk];
    const int first = blockIdx.y * 4;
    const int live = min(a.m - first, 4);
    const auto *x = static_cast<const float *>(a.x) + size_t(first) * a.ld_x;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
    const auto *w = static_cast<const uint8_t *>(a.w) + size_t(row) * (cols / 32) * block_bytes;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int base = 0; base < cols; base += chunk) {
        for (int i = threadIdx.x; i < chunk; i += blockDim.x) {
            const float *p = x + base + i;
            staged[i] = make_float4(p[0], live > 1 ? p[a.ld_x] : 0.f,
                                     live > 2 ? p[2 * a.ld_x] : 0.f,
                                     live > 3 ? p[3 * a.ld_x] : 0.f);
        }
        __syncthreads();
        for (int k = 0; k < chunk / 32; ++k) {
            const float value = fifi_gguf::decode<8>(w + (base / 32 + k) * block_bytes, lane);
            const float4 input = staged[k * 32 + lane];
            a0 = fmaf(value, input.x, a0);
            a1 = fmaf(value, input.y, a1);
            a2 = fmaf(value, input.z, a2);
            a3 = fmaf(value, input.w, a3);
        }
        __syncthreads();
    }
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    a2 = warp_sum(a2);
    a3 = warp_sum(a3);
    if (lane == 0) {
        auto *y = static_cast<float *>(a.y) + size_t(first) * a.ld_y + row;
        y[0] = a0;
        if (live > 1) y[a.ld_y] = a1;
        if (live > 2) y[2 * a.ld_y] = a2;
        if (live > 3) y[3 * a.ld_y] = a3;
    }
}

// Restrict production to the measured attention prefill M windows.
constexpr bool q8_row_reuse_shape(int dtype, int m, int rows, int cols) {
    if (dtype != fifi_gguf::kDtypeBase + 8) return false;
    return (cols == 2304 && (rows == 1024 || rows == 3072) && m >= 7 && m <= 12) ||
           (cols == 3072 && rows == 2304 && m >= 63 && m <= 64);
}

constexpr int kQ8 = fifi_gguf::kDtypeBase + 8;
static_assert(q8_row_reuse_shape(kQ8, 7, 1024, 2304) &&
              q8_row_reuse_shape(kQ8, 12, 1024, 2304) &&
              q8_row_reuse_shape(kQ8, 7, 3072, 2304) &&
              q8_row_reuse_shape(kQ8, 12, 3072, 2304) &&
              q8_row_reuse_shape(kQ8, 63, 2304, 3072) &&
              q8_row_reuse_shape(kQ8, 64, 2304, 3072));
static_assert(!q8_row_reuse_shape(kQ8 + 1, 7, 1024, 2304) &&
              !q8_row_reuse_shape(kQ8, 7, 1023, 2304) &&
              !q8_row_reuse_shape(kQ8, 7, 1024, 2336) &&
              !q8_row_reuse_shape(kQ8, 1, 1024, 2304) &&
              !q8_row_reuse_shape(kQ8, 2, 1024, 2304) &&
              !q8_row_reuse_shape(kQ8, 6, 1024, 2304) &&
              !q8_row_reuse_shape(kQ8, 13, 1024, 2304) &&
              !q8_row_reuse_shape(kQ8, 6, 3072, 2304) &&
              !q8_row_reuse_shape(kQ8, 13, 3072, 2304) &&
              !q8_row_reuse_shape(kQ8, 62, 2304, 3072) &&
              !q8_row_reuse_shape(kQ8, 65, 2304, 3072));

// The Inkling tile kernels use SM75 tensor-core instructions; every newer card runs them.
cudaError_t current_is_sm75_or_newer(bool &sm75) {
    sm75 = false;
    int device = 0, major = 0, minor = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err == cudaSuccess) {
        err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    }
    if (err == cudaSuccess) {
        err = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        sm75 = err == cudaSuccess && (major > 7 || (major == 7 && minor >= 5));
    }
    return err;
}

}  // namespace

extern "C" int inference_cuda_inkling_ragged_pair_launch(const inference_cuda_dense_gemv_args *args,void *stream,int *stage,int *used){
    if(!args || !used)return INFERENCE_CUDA_INVALID_ARGUMENT;*used=0;report_stage(stage,INFERENCE_CUDA_STAGE_NONE);
    const auto a=args[0],b=args[1];if(!valid(a,stream)||!valid(b,stream))return INFERENCE_CUDA_INVALID_ARGUMENT;
    const int c=a.weight_dtype-256;auto bin=[c](int m){return m==4?0:m>=5 && m<=8?1:m>=9 && m<=(c==16?16:32)?2:-1;};int slot=bin(a.m);
    if(slot<0 || slot!=bin(b.m) || a.weight_dtype!=b.weight_dtype || a.rows!=2304 || b.rows!=2304 || a.cols!=b.cols ||
        !((c==16 && a.cols==2304)||(c==42 && a.cols==1152)) || !aligned(a.x,16)||!aligned(b.x,16)||a.ld_x%4||b.ld_x%4)return 0;
    auto y=[](const inference_cuda_dense_gemv_args &i){return Span{i.y,4*(size_t(i.m-1)*i.ld_y+i.rows)};};
    auto x=[](const inference_cuda_dense_gemv_args &i){return Span{i.x,4*(size_t(i.m-1)*i.ld_x+i.cols)};};
    size_t wb=2304*weight_row_bytes(a.weight_dtype,a.cols);
    if(!apart(y(a),y(b))||!apart(y(a),x(b))||!apart(y(b),x(a))||!apart(y(a),{b.w,wb})||!apart(y(b),{a.w,wb}))return 0;
    bool sm75;auto err=current_is_sm75_or_newer(sm75);if(err==cudaSuccess){if(!sm75)return 0;int bm=slot==0?4:slot==1 || c==42?8:16;
        InklingExpertGroup g{};g.calls[0]=a;g.calls[1]=b;g.ends[0]=(a.m+bm-1)/bm;g.ends[1]=g.ends[0]+(b.m+bm-1)/bm;auto s=static_cast<cudaStream_t>(stream);
#define RAG2(C,M,R) inkling_ragged_exact<C,M,R><<<dim3(2304/(8*R),g.ends[1]),256,0,s>>>(g)
        if(c==16){if(slot==0){RAG2(16,4,2);}else if(slot==1){RAG2(16,8,2);}else{RAG2(16,16,2);}}
        else{if(slot==0){RAG2(42,4,2);}else if(slot==1){RAG2(42,8,2);}else{RAG2(42,8,4);}}
#undef RAG2
        err=cudaGetLastError();}
    if(err==cudaSuccess)*used=1;else report_stage(stage,INFERENCE_CUDA_STAGE_LAUNCH);return int(err);}

extern "C" int inference_cuda_inkling_down_launch(const inference_cuda_dense_gemv_args *args,void *stream,int *stage){
    if(!args || !valid(*args,stream) || args->weight_dtype!=kDtypeF32 || args->rows!=64 || args->cols!=4608)return INFERENCE_CUDA_INVALID_ARGUMENT;
    const auto a=*args;const auto s=static_cast<cudaStream_t>(stream);
    if(!aligned(a.w,16) || !aligned(a.x,16) || a.ld_x%4)return inference_cuda_dense_gemv_launch(args,stream,stage);
    bool sm75;auto e=current_is_sm75_or_newer(sm75);if(e==cudaSuccess){if(!sm75)return inference_cuda_dense_gemv_launch(args,stream,stage);
        switch(a.m){case 7:case 8:case 21:inkling_f32_rank_direct<2><<<dim3(8,(a.m+1)/2),256,0,s>>>(a);break;
            case 63:case 64:inkling_f32_visit<8,1><<<dim3(8,(a.m+7)/8),256,0,s>>>(a);break;
            case 127:case 128:case 255:case 256:inkling_f32_visit<8,2><<<dim3(4,(a.m+7)/8),256,0,s>>>(a);break;
            default:return inference_cuda_dense_gemv_launch(args,stream,stage);}e=cudaGetLastError();}
    report_stage(stage,e==cudaSuccess?INFERENCE_CUDA_STAGE_NONE:INFERENCE_CUDA_STAGE_LAUNCH);return int(e);}

extern "C" int inference_cuda_inkling_panel_launch(const inference_cuda_dense_gemv_args *args,void *scratch,
    size_t scratch_bytes, void *stream,int *stage,int *used){
    if(!used || !args || !valid(*args,stream))return INFERENCE_CUDA_INVALID_ARGUMENT;
    *used=0; report_stage(stage,INFERENCE_CUDA_STAGE_NONE);
    const auto a=*args;const int kind=a.weight_dtype==kDtypeGgufNvfp4?40:a.weight_dtype-fifi_gguf::kDtypeBase;
    if(a.rows!=2304 || a.m<=32 || a.m>512 || (kind==40?a.cols!=1152:((kind!=17 && kind!=19 && kind!=22) || a.cols!=2304)) ||
        !aligned(a.x,16) || a.ld_x%4 || !scratch || !aligned(scratch,16))return 0;
    const size_t size=size_t(a.rows)*a.cols*2;Span tmp{scratch,size};
    if(scratch_bytes<size || !apart(tmp,{a.w,size_t(a.rows)*weight_row_bytes(a.weight_dtype,a.cols)}) ||
        !apart(tmp,{a.x,4*(size_t(a.m-1)*a.ld_x+a.cols)}) || !apart(tmp,{a.y,4*(size_t(a.m-1)*a.ld_y+a.rows)}))return 0;
    auto s=static_cast<cudaStream_t>(stream);auto pack=static_cast<__half *>(scratch);bool sm75;auto err=current_is_sm75_or_newer(sm75);
    if(err==cudaSuccess){if(!sm75)return 0; const int grid=a.rows*a.cols/2048;
        switch(kind){case 17:inkling_decode_panels<17,2304,256,74><<<grid,256,0,s>>>(a,pack);break;
            case 19:inkling_decode_panels<19,2304,256,50><<<grid,256,0,s>>>(a,pack);break;
            case 22:inkling_decode_panels<22,2304,256,82><<<grid,256,0,s>>>(a,pack);break;
            case 40:inkling_decode_panels<40,1152,64,36><<<grid,256,0,s>>>(a,pack);break;}
        err=cudaGetLastError();if(err==cudaSuccess){dim3 g(72,(a.m+31)/32);
            if(kind==40)inkling_panel_mma<1152><<<g,128,0,s>>>(a,pack);else inkling_panel_mma<2304><<<g,128,0,s>>>(a,pack);
            err=cudaGetLastError();}}
    if(err==cudaSuccess)*used=1;else report_stage(stage,INFERENCE_CUDA_STAGE_LAUNCH);
    return int(err);
}

extern "C" int inference_cuda_inkling_qkv_visit_launch(const inference_cuda_dense_gemv_args *args,
    int count,void *stream,int *stage,int *used){
    if(!used || !args || count<2 || count>4)return INFERENCE_CUDA_INVALID_ARGUMENT;
    *used=0;report_stage(stage,INFERENCE_CUDA_STAGE_NONE);
    InklingQkvGroup group{}; unsigned tiles=0;
    for(int i=0;i<count;i++){
        const auto &a=args[i];if(!valid(a,stream))return INFERENCE_CUDA_INVALID_ARGUMENT;
        if(a.weight_dtype!=kQ8 || a.m!=args[0].m || a.m<7 || a.m>12 || a.cols!=2304 || (a.rows!=1024 && a.rows!=3072))return 0;
        Span ya{a.y,4*(size_t(a.m-1)*a.ld_y+a.rows)};
        for(int j=0;j<count;j++)if(j!=i){const auto &b=args[j];if(!valid(b,stream))return INFERENCE_CUDA_INVALID_ARGUMENT;
            Span xb{b.x,4*(size_t(b.m-1)*b.ld_x+b.cols)},wb{b.w,size_t(b.rows)*fifi_gguf::row_bytes(b.weight_dtype,b.cols)};
            if(!apart(ya,xb)||!apart(ya,wb))return 0;
            Span yb{b.y,4*(size_t(b.m-1)*b.ld_y+b.rows)};
            if(!apart(ya,yb)){
                if(a.ld_y!=b.ld_y)return 0;
                size_t stride=size_t(a.ld_y)*4;
                size_t delta=(uintptr_t(b.y)%stride+stride-uintptr_t(a.y)%stride)%stride;
                if(delta<size_t(a.rows)*4 || stride-delta<size_t(b.rows)*4)return 0;
            }
        }
        group.a[i]=a;group.end[i]=(tiles+=a.rows/32);
    }
    bool sm75;auto err=current_is_sm75_or_newer(sm75);if(err==cudaSuccess){if(!sm75)return 0;
        const bool prefetch=count==2 && args[0].m==8 && args[0].rows==1024 && args[1].rows==1024 &&
            aligned(args[0].x,16) && aligned(args[1].x,16) && args[0].ld_x%4==0 && args[1].ld_x%4==0;
        if(prefetch){group.end[0]*=2;group.end[1]*=2;inkling_qkv_prefetch<<<dim3(tiles*2,1),256,0,static_cast<cudaStream_t>(stream)>>>(group);}
        else inkling_qkv_visit<<<dim3(tiles,(args[0].m+7)/8),256,0,static_cast<cudaStream_t>(stream)>>>(group);
        err=cudaGetLastError();}
    if(err==cudaSuccess)*used=1;else report_stage(stage,INFERENCE_CUDA_STAGE_LAUNCH);return int(err);
}

// Shape admission is duplicated at the FFI boundary; no unchecked tail loads.
extern "C" int inference_cuda_inkling_packed_visit_launch(const inference_cuda_dense_gemv_args *args,
    void *stream, int *stage, int variant) {
    if (!args || !valid(*args,stream) || args->m<2 || (args->m>64 && !(args->m==127 || args->m==128 || args->m==255 || args->m==256))) return INFERENCE_CUDA_INVALID_ARGUMENT;
    const auto a=*args;
    const int codec=a.weight_dtype-fifi_gguf::kDtypeBase;
    const bool attn=(a.cols==2304 && (a.rows==384 || a.rows==1024 || a.rows==3072)) || (a.cols==3072 && a.rows==2304);
    const bool expert=a.rows==2304 && a.cols==2304 && a.m<=16;
    if (!(attn && (codec==8 || codec==12 || codec==13 || codec==14 || codec==18)) &&
        !(expert && (codec==17 || codec==19 || codec==22))) return INFERENCE_CUDA_INVALID_ARGUMENT;
    bool sm75; auto err=current_is_sm75_or_newer(sm75);
    if (err==cudaSuccess && !sm75) return inference_cuda_dense_gemv_launch(args,stream,stage);
    if(err==cudaSuccess){
        const auto s=static_cast<cudaStream_t>(stream);
        if(variant==3 && (inkling_staged_select(a,codec,s) || inkling_prefetch_select(a,codec,s) || inkling_large_select(a,codec,s))){
            err=cudaGetLastError();report_stage(stage,err==cudaSuccess?INFERENCE_CUDA_STAGE_NONE:INFERENCE_CUDA_STAGE_LAUNCH);return int(err);
        }
        if(variant==3 && expert && (codec==17 || codec==19 || codec==22)){
            dim3 g(a.rows/32,(a.m+7)/8);
            if(codec==17)inkling_exact_visit_tile<17,256,74,4,false,true><<<g,256,0,s>>>(a);
            else if(codec==19)inkling_exact_visit_tile<19,256,50,4,false,true><<<g,256,0,s>>>(a);
            else inkling_exact_visit_tile<22,256,82,4,false,true><<<g,256,0,s>>>(a);
            err=cudaGetLastError();report_stage(stage,err==cudaSuccess?INFERENCE_CUDA_STAGE_NONE:INFERENCE_CUDA_STAGE_LAUNCH);return int(err);
        }
        if(variant==3)variant=a.rows<=1024 && a.m<=8?1:2;
        const dim3 small(a.rows/8,(a.m+7)/8), large(a.rows/32,(a.m+7)/8);
#define VISIT_CASE(C,B,S) case C: if(variant==1) inkling_exact_visit_tile<C,B,S,1><<<small,256,0,s>>>(a); else inkling_exact_visit_tile<C,B,S><<<large,256,0,s>>>(a); break
        switch(codec){ VISIT_CASE(8,32,34); VISIT_CASE(12,256,144); VISIT_CASE(13,256,176); VISIT_CASE(14,256,210);
            VISIT_CASE(17,256,74); VISIT_CASE(18,256,98); VISIT_CASE(19,256,50); VISIT_CASE(22,256,82); }
#undef VISIT_CASE
        err=cudaGetLastError();
    }
    report_stage(stage,err==cudaSuccess?INFERENCE_CUDA_STAGE_NONE:INFERENCE_CUDA_STAGE_LAUNCH);return int(err);
}

extern "C" int inference_cuda_inkling_controller_pair_launch(const inference_cuda_dense_gemv_args *a,
    const inference_cuda_dense_gemv_args *b, void *stream, int *stage, int *used) {
    if (!used || !a || !b || !valid(*a,stream) || !valid(*b,stream)) return INFERENCE_CUDA_INVALID_ARGUMENT;
    *used = 0;
    report_stage(stage, INFERENCE_CUDA_STAGE_NONE);
    if (a->weight_dtype!=kDtypeF32 || b->weight_dtype!=kDtypeF32 || a->m!=b->m || a->m<2 || (a->m>64 && !(a->m==127 || a->m==128 || a->m==255 || a->m==256)) ||
        a->rows!=2304 || b->rows!=2304 || a->cols!=64 || b->cols!=64 || a->x!=b->x || a->ld_x!=b->ld_x ||
        !aligned(a->w,16) || !aligned(b->w,16)) return 0;
    Span ya{a->y,4*(size_t(a->m-1)*a->ld_y+2304)}, yb{b->y,4*(size_t(b->m-1)*b->ld_y+2304)};
    const bool interleaved = a->ld_y == 4608 && b->ld_y == 4608 && uintptr_t(b->y)>uintptr_t(a->y) &&
        uintptr_t(b->y)-uintptr_t(a->y)==2304*4;
    if ((!interleaved && !apart(ya,yb)) || !apart(ya,{b->w,2304*64*4}) || !apart(yb,{a->w,2304*64*4})) return 0;
    inkling_controller_pair<<<dim3(144,(a->m+7)/8,2),256,0,static_cast<cudaStream_t>(stream)>>>(*a,*b);
    const auto err = cudaGetLastError();
    if (err == cudaSuccess) *used = 1; else report_stage(stage, INFERENCE_CUDA_STAGE_LAUNCH);
    return int(err);
}

extern "C" int inference_cuda_inkling_iq2_visit_launch(const inference_cuda_dense_gemv_args *args, void *stream, int *out_stage) {
    if (!args || !valid(*args, stream) || args->rows != 2304 || args->m > 512) return INFERENCE_CUDA_INVALID_ARGUMENT;
    const int codec = args->weight_dtype == kDtypeGgufNvfp4 ? 40 : args->weight_dtype - fifi_gguf::kDtypeBase;
    if (!(codec==40 ? args->cols==1152 : (codec==17 || codec==19 || codec==22) && args->cols==2304))
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    bool sm75;
    auto err = current_is_sm75_or_newer(sm75);
    if (err == cudaSuccess) {
        if (!sm75) return inference_cuda_gguf_wmma_launch(args, stream, out_stage);
        const dim3 grid(72,(args->m+31)/32);
        const auto s = static_cast<cudaStream_t>(stream);
        switch(codec){
        #define OCTET_CASE(C,K,B,S) case C: \
            if (aligned(args->x,16) && args->ld_x%4==0) \
                inkling_iq2_visit_tile<C,K,B,S,true,32,128,C==19><<<grid,512,0,s>>>(*args); \
            else inkling_iq2_visit_tile<C,K,B,S,false><<<grid,512,0,s>>>(*args); \
            break
        OCTET_CASE(17,2304,256,74); OCTET_CASE(19,2304,256,50);
        OCTET_CASE(22,2304,256,82); OCTET_CASE(40,1152,64,36);
#undef OCTET_CASE
        }
        err = cudaGetLastError();
    }
    report_stage(out_stage, err == cudaSuccess ? INFERENCE_CUDA_STAGE_NONE : INFERENCE_CUDA_STAGE_LAUNCH);
    return int(err);
}

#ifdef FIFI_VISIT_PROBE
extern "C" int ragged_exact_probe(const inference_cuda_dense_gemv_args *args,int count,int bin,cudaStream_t s){
    InklingExpertGroup g{};int c=args[0].weight_dtype-256,bm=bin==0?4:bin==1 || c==42?8:16,tiles=0;
    for(int i=0;i<count;i++){g.calls[i]=args[i];g.ends[i]=(tiles+=(args[i].m+bm-1)/bm);}
#define RAG(C,M,R) inkling_ragged_exact<C,M,R><<<dim3(2304/(8*R),tiles),256,0,s>>>(g)
    if(c==16){if(bin==0){RAG(16,4,2);}else if(bin==1){RAG(16,8,2);}else{RAG(16,16,2);}}
    else if(c==42){if(bin==0){RAG(42,4,2);}else if(bin==1){RAG(42,8,2);}else{RAG(42,8,4);}}else return int(cudaErrorInvalidValue);
#undef RAG
    return int(cudaGetLastError());}
extern "C" int expert_group_probe(const inference_cuda_dense_gemv_args *args,int count,cudaStream_t s){
    InklingExpertGroup group{};int tiles=0;for(int i=0;i<count;i++){group.calls[i]=args[i];group.ends[i]=(tiles+=(args[i].m+31)/32);}
    dim3 grid(72,tiles);
    switch(args[0].weight_dtype){
    case 273:inkling_expert_group<17,2304,256,74><<<grid,512,0,s>>>(group);break;
    case 275:inkling_expert_group<19,2304,256,50><<<grid,512,0,s>>>(group);break;
    case 278:inkling_expert_group<22,2304,256,82><<<grid,512,0,s>>>(group);break;
    case 2:inkling_expert_group<40,1152,64,36><<<grid,512,0,s>>>(group);break;default:return -1;}
    return int(cudaGetLastError());
}
extern "C" int ablate_weight_probe(const inference_cuda_dense_gemv_args *args,cudaStream_t s){auto a=*args;dim3 g(a.rows/32,(a.m+7)/8);
    inkling_exact_visit_tile<8,32,34,4,true><<<g,256,0,s>>>(a);return int(cudaGetLastError());
}
extern "C" int weight_panel_probe(const inference_cuda_dense_gemv_args *args,void *tmp,cudaStream_t s,int mode){auto a=*args;auto p=static_cast<InklingWeightPanel *>(tmp);
#define WPL(M,N) inkling_weight_panel<M,N><<<dim3(a.rows/N,(a.m + M - 1) / M),256,0,s>>>(a,p)
    switch(mode){case 0:inkling_weight_repack<<<(a.rows*a.cols+255)/256,256,0,s>>>(a,p);break;case 1:WPL(8,16);break;case 2:WPL(8,32);break;case 3:WPL(16,32);break;default:return int(cudaErrorInvalidValue);}
#undef WPL
    return int(cudaGetLastError());}
extern "C" int down_visit_probe(const inference_cuda_dense_gemv_args *args,cudaStream_t s,int arm){auto a=*args;
#define DOWN(M,R) inkling_f32_visit<M,R><<<dim3(8/R,(a.m + M - 1) / M),256,0,s>>>(a)
#define DIRECT(M) inkling_f32_rank_direct<M><<<dim3(8,(a.m + M - 1) / M),256,0,s>>>(a)
    switch(arm){case 1:DOWN(2,1);break;case 2:DOWN(4,1);break;case 3:DOWN(8,1);break;case 4:DOWN(8,2);break;case 5:DIRECT(2);break;case 6:DIRECT(4);break;case 7:DIRECT(8);break;default:return int(cudaErrorInvalidValue);}
#undef DOWN
#undef DIRECT
    return int(cudaGetLastError());}
extern "C" int larger_visit_probe(const inference_cuda_dense_gemv_args *args,cudaStream_t s){int stage=0;return inference_cuda_inkling_packed_visit_launch(args,s,&stage,3);}
extern "C" int probe_prefetch_bm(const inference_cuda_dense_gemv_args *a){int m=8;inkling_prefetch_select(*a,a->weight_dtype-256,nullptr,&m);return m;}
extern "C" int qkv_prefetch_probe(const inference_cuda_dense_gemv_args *args,int count,cudaStream_t s){InklingQkvGroup g{};int tiles=0;
    for(int i=0;i<count;i++){g.a[i]=args[i];g.end[i]=(tiles+=args[i].rows/16);}inkling_qkv_prefetch<<<dim3(tiles,(args[0].m+7)/8),256,0,s>>>(g);return int(cudaGetLastError());}
extern "C" int codebook_exact_probe(const inference_cuda_dense_gemv_args *args,cudaStream_t s){auto a=*args;dim3 g(a.rows/32,(a.m+7)/8);
#define CL(C,S) case 256+C: inkling_exact_visit_tile<C,256,S,4,false,true><<<g,256,0,s>>>(a);break
    switch(a.weight_dtype){CL(16,66);CL(17,74);CL(19,50);CL(22,82);default:return int(cudaErrorInvalidValue);}
#undef CL
    return int(cudaGetLastError());}
extern "C" int codebook_visit_probe(const inference_cuda_dense_gemv_args *args,cudaStream_t s){auto a=*args;dim3 g(a.rows/32,(a.m+31)/32);
#define CODES(C,S) case 256+C: inkling_iq2_visit_tile<C,2304,256,S,true,32,128,true><<<g,512,0,s>>>(a); break
    switch(a.weight_dtype){CODES(17,74);CODES(19,50);CODES(22,82);default:return int(cudaErrorInvalidValue);}
#undef CODES
    return int(cudaGetLastError());}
extern "C" int octet_big_probe(const inference_cuda_dense_gemv_args *args,cudaStream_t s,int mode){
    auto a=*args;int bm=mode==1?64:32;dim3 g(72,(a.m+bm-1)/bm);
#define BIG(C,K,B,S,BK) case C: if(mode==1) inkling_iq2_visit_tile<C==2?40:C - 256,K,B,S,true,64,128><<<g,512,0,s>>>(a); else inkling_iq2_visit_tile<C==2?40:C - 256,K,B,S,true,32,BK><<<g,512,0,s>>>(a); break
    switch(a.weight_dtype){BIG(273,2304,256,74,256);BIG(275,2304,256,50,256);BIG(278,2304,256,82,256);BIG(2,1152,64,36,128);default:return -1;}
#undef BIG
    return int(cudaGetLastError());
}
extern "C" int interleave_probe(const inference_cuda_dense_gemv_args *args,void *tmp,cudaStream_t s,int arm){
    auto a=*args;auto *q=static_cast<signed char *>(tmp);float *sc=reinterpret_cast<float *>(q+a.rows*a.cols);bool q8=a.weight_dtype==kQ8;
    if(!arm){int grid=a.rows*a.cols/256;if(q8) inkling_interleave_encode<8><<<grid,256,0,s>>>(a,q,sc);else inkling_interleave_encode<14><<<grid,256,0,s>>>(a,q,sc);}
    else {int bm=arm==1?8:16;dim3 g(a.rows/8,(a.m+bm-1)/bm);
#define INTER(C) if(arm==1)inkling_interleave_visit<C,8><<<g,256,0,s>>>(a,q,sc);else inkling_interleave_visit<C,16><<<g,256,0,s>>>(a,q,sc)
        if(q8){INTER(8);}else{INTER(14);}
#undef INTER
    }
    return int(cudaGetLastError());
}
extern "C" int expert_octet_probe(const inference_cuda_dense_gemv_args *args,cudaStream_t s,int bm){
    auto a=*args;dim3 grid(a.rows/16,(a.m+bm-1)/bm);
#define EXP(C,S) case 256+C: if(bm==2)inkling_expert_octet<C,S,2><<<grid,64,0,s>>>(a); else inkling_expert_octet<C,S,4><<<grid,64,0,s>>>(a); break
    switch(a.weight_dtype){EXP(16,66);EXP(17,74);EXP(19,50);EXP(22,82);default:return -1;}
#undef EXP
    return int(cudaGetLastError());
}
extern "C" int q6_imma_probe(const inference_cuda_dense_gemv_args *args,void *tmp,cudaStream_t s,int variant){auto a=*args;auto x=static_cast<ResidualBlock *>(tmp);
    inkling_residual_pack<<<(a.m*a.cols/32+7)/8,256,0,s>>>(a,x);auto e=cudaGetLastError();if(e!=cudaSuccess)return int(e);
#define Q6I(M,N,K) inkling_q6_imma<M,N,K><<<dim3(a.rows/(N*8),(a.m+M*8-1)/(M*8)),256,0,s>>>(a,x)
    switch(variant){case 1:Q6I(1,1,8);break;case 2:Q6I(1,2,4);break;case 3:Q6I(2,2,2);break;case 4:Q6I(2,4,1);break;default:return int(cudaErrorInvalidValue);}
#undef Q6I
    return int(cudaGetLastError());}
extern "C" int residual_probe(const inference_cuda_dense_gemv_args *args,void *tmp,cudaStream_t s){
    auto a=*args;int blocks=a.m*a.cols/32;auto *q=static_cast<ResidualBlock *>(tmp);
    static_assert(sizeof(ResidualBlock)==68);
    inkling_residual_pack<<<(blocks+7)/8,256,0,s>>>(a,q);auto rc=cudaGetLastError();if(rc!=cudaSuccess)return int(rc);
    inkling_residual_imma<<<dim3(a.rows/8,(a.m+7)/8),256,0,s>>>(a,q);return int(cudaGetLastError());
}
extern "C" int panel_probe(const inference_cuda_dense_gemv_args *args,void *tmp,cudaStream_t s,int cached){
    auto a=*args;auto *pack=static_cast<__half *>(tmp);int grid=a.rows*a.cols/2048;
    if(cached!=1){
        switch(a.weight_dtype){
        case 256+17:inkling_decode_panels<17,2304,256,74><<<grid,256,0,s>>>(a,pack);break;
        case 256+19:inkling_decode_panels<19,2304,256,50><<<grid,256,0,s>>>(a,pack);break;
        case 256+22:inkling_decode_panels<22,2304,256,82><<<grid,256,0,s>>>(a,pack);break;
        case kDtypeGgufNvfp4:inkling_decode_panels<40,1152,64,36><<<grid,256,0,s>>>(a,pack);break;
        default:return -1;}
        int rc=int(cudaGetLastError());if(rc)return rc;
    }
    if(cached>=2){
#define PW(K,M,N) inkling_panel_wide<K,M,N><<<dim3(a.rows/N,(a.m + M - 1) / M),128,0,s>>>(a,pack)
#define PWS(K) if(cached==2){PW(K,32,64);}else if(cached==3){PW(K,64,32);}else{PW(K,64,64);}
        if(a.cols==1152){PWS(1152);}else{PWS(2304);}
#undef PWS
#undef PW
        return int(cudaGetLastError());}
    dim3 g(a.rows/32,(a.m+31)/32);if(a.cols==1152)inkling_panel_mma<1152><<<g,128,0,s>>>(a,pack);else inkling_panel_mma<2304><<<g,128,0,s>>>(a,pack);
    return int(cudaGetLastError());
}
extern "C" int octet_vector_probe(const inference_cuda_dense_gemv_args *args,cudaStream_t s){
    const auto a=*args;const dim3 grid(72,(a.m+31)/32);
    if(!aligned(a.x,16) || a.ld_x%4)return -1;
    switch(a.weight_dtype){
    case 256+17: inkling_iq2_visit_tile<17,2304,256,74,true><<<grid,512,0,s>>>(a);break;
    case 256+19: inkling_iq2_visit_tile<19,2304,256,50,true><<<grid,512,0,s>>>(a);break;
    case 256+22: inkling_iq2_visit_tile<22,2304,256,82,true><<<grid,512,0,s>>>(a);break;
    case kDtypeGgufNvfp4: inkling_iq2_visit_tile<40,1152,64,36,true><<<grid,512,0,s>>>(a);break;
    default:return -1;
    }
    return int(cudaGetLastError());
}
extern "C" int stream_visit_probe(const inference_cuda_dense_gemv_args *args,cudaStream_t s,int variant){
    auto a=*args;
    if(variant>=16 && variant<=19){
#define OCC(C,B,S,WN,WM,R,M,K) inkling_occupancy_tile<C,B,S,WN,WM,R,M,K><<<dim3(a.rows/(WN*R),(a.m + WM * M - 1) / (WM * M)),WN*WM*32,0,s>>>(a)
#define OCCS(C,B,S) if(variant==16){OCC(C,B,S,4,2,4,4,64);}else if(variant==17){OCC(C,B,S,8,2,2,4,128);}else if(variant==18){OCC(C,B,S,4,4,4,4,64);}else{OCC(C,B,S,8,2,2,8,64);}
        if(a.weight_dtype==kQ8){OCCS(8,32,34);}else{OCCS(14,256,210);}
#undef OCCS
#undef OCC
        return int(cudaGetLastError());
    }
    if(variant==14 || variant==15){
#define GROUP64(C,B,S) if(variant==14)inkling_group64<C,B,S,32,128><<<dim3(a.rows/32,(a.m+31)/32),1024,0,s>>>(a);else inkling_group64<C,B,S,64,128><<<dim3(a.rows/32,(a.m+63)/64),1024,0,s>>>(a)
        if(a.weight_dtype==kQ8){GROUP64(8,32,34);}else{GROUP64(14,256,210);}
#undef GROUP64
        return int(cudaGetLastError());
    }
    if(variant==12 || variant==13){
#define SIMT(C,B,S) if(variant==12)inkling_simt_visit<C,B,S,16,32><<<dim3(a.rows/32,(a.m+15)/16),128,0,s>>>(a);else inkling_simt_visit<C,B,S,64,64><<<dim3(a.rows/64,(a.m+63)/64),128,0,s>>>(a)
        if(a.weight_dtype==kQ8){SIMT(8,32,34);}else{SIMT(14,256,210);}
#undef SIMT
        return int(cudaGetLastError());
    }
    if(variant==10 || variant==11){
#define TOK(C,B,S) if(variant==10)inkling_token_visit<C,B,S,8><<<dim3(a.rows/32,(a.m+7)/8),256,0,s>>>(a);else inkling_token_visit<C,B,S,32><<<dim3(a.rows/8,(a.m+31)/32),256,0,s>>>(a)
        if(a.weight_dtype==kQ8){TOK(8,32,34);} else{TOK(14,256,210);}
#undef TOK
        return int(cudaGetLastError());
    }
    if(variant==7 || variant==8){
#define QUART(C,B,S,K) if(variant==7) inkling_quartet<C,B,S,K,2><<<dim3(a.rows/8,(a.m+7)/8),64,0,s>>>(a);else inkling_quartet<C,B,S,K,8><<<dim3(a.rows/32,(a.m+7)/8),256,0,s>>>(a)
        if(a.weight_dtype!=kQ8){QUART(14,256,210,2304);}else if(a.cols==2304){QUART(8,32,34,2304);}else{QUART(8,32,34,3072);}
#undef QUART
        return int(cudaGetLastError());
    }
    if(variant==5 || variant==6){
#define WARP(C,B,S,K) if(variant==5) inkling_warp_visit<C,B,S,K,2,8><<<dim3(a.rows/8,(a.m+7)/8),128,0,s>>>(a);else inkling_warp_visit<C,B,S,K,4,8><<<dim3(a.rows/16,(a.m+7)/8),128,0,s>>>(a)
        if(a.weight_dtype!=kQ8){WARP(14,256,210,2304);}else if(a.cols==2304){WARP(8,32,34,2304);}else{WARP(8,32,34,3072);}
#undef WARP
        return int(cudaGetLastError());
    }
    if(variant==4){
#define GRID(C,B,S) if(a.m<=8) inkling_grid_split<C,B,S,8><<<dim3(a.rows/16,1),512,0,s>>>(a); else inkling_grid_split<C,B,S,16><<<dim3(a.rows/16,(a.m+15)/16),512,0,s>>>(a)
        if(a.weight_dtype==kQ8){GRID(8,32,34);}else{GRID(14,256,210);}
#undef GRID
        return int(cudaGetLastError());
    }
    if(variant==3){
#define TC(C,B,S) if(a.m<=16) inkling_splitn_visit<C,B,S,16><<<a.rows/16,256,0,s>>>(a); else inkling_splitn_visit<C,B,S,64><<<a.rows/16,256,0,s>>>(a)
        if(a.weight_dtype==kQ8){TC(8,32,34);}else{TC(14,256,210);}
#undef TC
        return int(cudaGetLastError());
    }
#define STREAM(C,B,S) if(a.m>16){if(variant==1) inkling_stream_visit<C,B,S,1,64><<<a.rows/8,256,0,s>>>(a); else inkling_stream_visit<C,B,S,2,64><<<a.rows/16,256,0,s>>>(a);} else if(a.m>8) inkling_stream_visit<C,B,S,2,16><<<a.rows/16,256,0,s>>>(a); else if(variant==1) inkling_stream_visit<C,B,S,1,8><<<a.rows/8,256,0,s>>>(a); else inkling_stream_visit<C,B,S,2,8><<<a.rows/16,256,0,s>>>(a)
    if(a.weight_dtype==kQ8){STREAM(8,32,34);} else {STREAM(14,256,210);}
#undef STREAM
    return int(cudaGetLastError());
}
extern "C" int split_visit_probe(const inference_cuda_dense_gemv_args *args, cudaStream_t s) {
    auto a = *args;
    dim3 grid(a.rows/32, (a.m+15)/16);
    switch (a.weight_dtype - fifi_gguf::kDtypeBase) {
    case 8: inkling_split_visit_tile<8,32,34><<<grid,512,0,s>>>(a); break;
    case 14: inkling_split_visit_tile<14,256,210><<<grid,512,0,s>>>(a); break;
    case 17: inkling_split_visit_tile<17,256,74><<<grid,512,0,s>>>(a); break;
    default: return -1;
    }
    return int(cudaGetLastError());
}
#endif

extern "C" int inference_cuda_gguf_q8_row_reuse_width_launch(
    const inference_cuda_dense_gemv_args *args, void *stream, int *out_stage, int *out_width,
    int prefer_four) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (out_width) *out_width = 0;
        if (!args || !valid(*args, stream)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const auto &a = *args;
        const bool visit_shape = prefer_four == -1 && a.weight_dtype == kQ8 &&
            (a.m == 7 || a.m == 8 || a.m == 63 || a.m == 64) &&
            ((a.cols == 2304 && (a.rows == 1024 || a.rows == 3072)) ||
             (a.cols == 3072 && a.rows == 2304));
        const int codec = a.weight_dtype - fifi_gguf::kDtypeBase;
        const bool expert_visit = prefer_four == -1 && a.rows == 2304 && a.m >= 2 && a.m <= 32 &&
            ((codec == 16 && a.cols == 2304) || (codec == 42 && a.cols == 1152));
        const bool q6_visit = prefer_four == -1 && a.weight_dtype == fifi_gguf::kDtypeBase + 14 &&
            exact_expert_tile(14, a.m, a.rows, a.cols) == 2;
        if (!visit_shape && !q6_visit && !expert_visit && !q8_row_reuse_shape(a.weight_dtype, a.m, a.rows, a.cols)) {
            return inference_cuda_dense_gemv_launch(args, stream, out_stage);
        }
        Failure failure;
        bool sm75;
        // Rust selects the stream's adapter under the dispatch guard before entering here.
        failure.record(current_is_sm75_or_newer(sm75), INFERENCE_CUDA_STAGE_LAUNCH);
        if (!failure.ok()) {
            report_stage(out_stage, failure.stage);
            return (int)failure.err;
        }
        if (!sm75) return inference_cuda_dense_gemv_launch(args, stream, out_stage);
        constexpr int output_rows = 2304;
        static_assert(output_rows % kWarps == 0);
        const bool four = prefer_four && a.rows == output_rows && a.cols == 3072;
        const bool visit = prefer_four == -1;
        const int width = visit ? a.m : four ? 4 : 2;
        const dim3 grid((a.rows + kWarps - 1) / kWarps, (a.m + width - 1) / width);
        const auto s = static_cast<cudaStream_t>(stream);
        if (visit) {
            dim3 tiles(a.rows / 32, (a.m + 7) / 8);
            if(expert_visit && a.m>=4 && aligned(a.x,16) && a.ld_x%4==0 && (codec==42 || a.m<=16)){
#define EXP(C,M,R) inkling_prefetch<C,M,R,128,true><<<dim3(a.rows/(8*R),(a.m + M - 1) / M),256,0,s>>>(a)
                if(codec==16){if(a.m==4){EXP(16,4,2);}else if(a.m<=8){EXP(16,8,2);}else{EXP(16,16,2);}}
                else{if(a.m==4){EXP(42,4,2);}else if(a.m<=8){EXP(42,8,2);}else{EXP(42,8,4);}}
#undef EXP
            } else if (expert_visit) {
                dim3 expert_tiles(a.rows/8, (a.m+7)/8);
                if (codec == 16) inkling_exact_visit_tile<16,256,66,1><<<expert_tiles,256,0,s>>>(a);
                else inkling_exact_visit_tile<42,64,18,1><<<expert_tiles,256,0,s>>>(a);
            } else if (!inkling_staged_select(a,q6_visit?14:8,s) && !inkling_prefetch_select(a,q6_visit?14:8,s)) {
                if(q6_visit) inkling_exact_visit_tile<14,256,210><<<tiles,256,0,s>>>(a);
                else inkling_exact_visit_tile<8,32,34><<<tiles,256,0,s>>>(a);
            }
        } else if (four) {
            inkling_q8_reuse_four<<<grid, kWarps * 32, 0, s>>>(a);
        } else {
            inkling_q8_reuse_pair<<<grid, kWarps * 32, size_t(a.cols) * sizeof(float2), s>>>(a);
        }
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.ok() && out_width) *out_width = width;
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_gguf_q8_row_reuse_launch(
    const inference_cuda_dense_gemv_args *args, void *stream, int *out_stage, int *out_pair) {
    int width = 0;
    const int error = inference_cuda_gguf_q8_row_reuse_width_launch(args, stream, out_stage, &width, 0);
    if (out_pair) *out_pair = width == 2;
    return error;
}

extern "C" int inference_cuda_gguf_exact_expert_tile_launch(
    const inference_cuda_dense_gemv_args *args, void *stream, int *out_stage, int *out_tile) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (out_tile) *out_tile = 0;
        if (!args || !valid(*args, stream)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const auto &a = *args;
        const int type = a.weight_dtype - fifi_gguf::kDtypeBase;
        const int tile = exact_expert_tile(type, a.m, a.rows, a.cols);
        if (tile == 0) return inference_cuda_dense_gemv_launch(args, stream, out_stage);
        const dim3 grid((a.rows + kWarps - 1) / kWarps, (a.m + tile - 1) / tile);
        const size_t shared = size_t(tile) * a.cols * sizeof(float);
        const auto s = (cudaStream_t)stream;
        if (type == 14) {
            gguf_exact_expert_tile<14, 256, 210, 2><<<grid, kWarps * 32, shared, s>>>(a);
        } else if (type == 16) {
            gguf_exact_expert_tile<16, 256, 66, 2><<<grid, kWarps * 32, shared, s>>>(a);
        } else if (type == 20) {
            if (tile == 4) gguf_exact_expert_tile<20, 32, 18, 4><<<grid, kWarps * 32, shared, s>>>(a);
            else gguf_exact_expert_tile<20, 32, 18, 2><<<grid, kWarps * 32, shared, s>>>(a);
        } else if (type == 8) {
            if (tile == 4) gguf_exact_expert_tile<8, 32, 34, 4><<<grid, kWarps * 32, shared, s>>>(a);
            else gguf_exact_expert_tile<8, 32, 34, 2><<<grid, kWarps * 32, shared, s>>>(a);
        } else if (tile == 4) {
            gguf_exact_expert_tile<42, 64, 18, 4><<<grid, kWarps * 32, shared, s>>>(a);
        } else {
            gguf_exact_expert_tile<42, 64, 18, 2><<<grid, kWarps * 32, shared, s>>>(a);
        }
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.ok() && out_tile) *out_tile = tile;
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_gguf_wmma_launch(const inference_cuda_dense_gemv_args *args,
                                                void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!args || !valid_operands(*args, stream) || args->cols % kWmmaTile) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        return launch_wmma(*args, stream, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_gguf_moe_launch(const inference_cuda_gguf_moe_args *args,
                                               const inference_cuda_gguf_moe_matrices *matrices,
                                               void *stream, int *out_stage,
                                               int *schedule) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        MoeSizes sizes;
        if (!args || !valid_moe(*args, matrices, stream, sizes)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        constexpr int phases = INFERENCE_CUDA_GGUF_MOE_ACTIVATE_ONLY | INFERENCE_CUDA_GGUF_MOE_SUM_ONLY;
        if (schedule && (*schedule & phases) == phases) return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        if (matrices) {
            SeparateWeights gate{}, down{};
            for (int slot = 0; slot < args->count; ++slot) {
                gate.matrices[slot] = matrices->gate[slot];
                down.matrices[slot] = matrices->down[slot];
            }
            launch_moe(*args, gate, down, (cudaStream_t)stream, failure, schedule);
        } else if (args->device_ids) {
            const auto experts = unsigned(args->experts);
            const DeviceSelectedWeights gate{sizes.gate_stride, args->device_ids, experts};
            const DeviceSelectedWeights down{sizes.down_stride, args->device_ids, experts};
            launch_moe<true>(*args, gate, down, (cudaStream_t)stream, failure, schedule);
        } else {
            SelectedWeights gate = {sizes.gate_stride, {}};
            for (int slot = 0; slot < args->count; ++slot) gate.ids[slot] = args->picks[slot].expert;
            SelectedWeights down = gate;
            down.stride = sizes.down_stride;
            launch_moe(*args, gate, down, (cudaStream_t)stream, failure, schedule);
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
