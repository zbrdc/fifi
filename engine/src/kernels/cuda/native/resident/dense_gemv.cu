// Dense matvec for half-precision or packed GGUF weights (see resident/dense_gemv.h).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
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

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }

struct Span {
    const void *p;
    size_t bytes;
};

bool apart(const Span &a, const Span &b) {
    const uintptr_t x = (uintptr_t)a.p, y = (uintptr_t)b.p;
    return x <= y ? a.bytes <= y - x : b.bytes <= x - y;
}

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
        // 8 weights (16 bytes) per lane per step; cols % 8 == 0 and row is 16-byte aligned.
        const uint4 *wide = reinterpret_cast<const uint4 *>(row);
        for (int v = lane; v < cols / 8; v += 32) {
            const uint4 raw = wide[v];
            const W *e = reinterpret_cast<const W *>(&raw);
#pragma unroll
            for (int j = 0; j < 8; ++j) acc = fmaf(widen(e[j]), staged[v * 8 + j], acc);
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

template <typename Index>
__global__ void gguf_nvfp4_gemv(inference_cuda_dense_gemv_args a, Index index) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (row >= a.rows) return;
    const float *x = static_cast<const float *>(a.x) + size_t(blockIdx.y) * a.ld_x;
    const int blocks = a.cols / 64;
    const auto *w = static_cast<const uint8_t *>(a.w) + index.offset(blockIdx.y) +
                    size_t(row) * blocks * 36;
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

template <int Type, int Block, int Bytes, typename Index>
__global__ void gguf_packed_gemv(inference_cuda_dense_gemv_args a, Index index) {
    extern __shared__ float staged[];
    const auto *x = static_cast<const float *>(a.x) + size_t(blockIdx.y) * a.ld_x;
    for (int col = threadIdx.x; col < a.cols; col += blockDim.x) staged[col] = x[col];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (row >= a.rows) return;
    const int blocks = a.cols / Block;
    const auto *w = static_cast<const uint8_t *>(a.w) + index.offset(blockIdx.y) +
                    size_t(row) * blocks * Bytes;
    float sum = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        const auto *weights = w + block * Bytes;
#pragma unroll
        for (int col = lane; col < Block; col += 32) {
            sum = fmaf(fifi_gguf::decode<Type>(weights, col), staged[block * Block + col], sum);
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) static_cast<float *>(a.y)[size_t(blockIdx.y) * a.ld_y + row] = sum;
}

size_t weight_row_bytes(int dtype, int cols) {
    switch (dtype) {
    case kDtypeF16:
    case kDtypeBf16:
        return size_t(cols) * 2;
    case kDtypeGgufNvfp4:
        return cols % 64 == 0 ? size_t(cols / 64) * 36 : 0;
    default:
        return fifi_gguf::row_bytes(dtype, cols);
    }
}

bool valid(const inference_cuda_dense_gemv_args &a, void *stream) {
    if (stream == NULL || a.w == NULL || a.x == NULL || a.y == NULL) return false;
    if (a.m < 1 || a.m > kMaxM || a.rows < 1 || a.rows > kMaxRows || a.cols < 1 ||
        a.cols > kMaxCols) {
        return false;
    }
    if (a.ld_x < a.cols || a.ld_x > kMaxLd || a.ld_y < a.rows || a.ld_y > kMaxLd) return false;
    if (!aligned(a.w, 2) || !aligned(a.x, 4) || !aligned(a.y, 4)) return false;
    const size_t m = (size_t)a.m;
    const size_t row_bytes = weight_row_bytes(a.weight_dtype, a.cols);
    if (row_bytes == 0) return false;
    const Span weights = {a.w, size_t(a.rows) * row_bytes};
    const Span in = {a.x, ((m - 1) * a.ld_x + a.cols) * 4};
    const Span out = {a.y, ((m - 1) * a.ld_y + a.rows) * 4};
    return apart(out, in) && apart(out, weights);
}

template <typename W>
void launch(const inference_cuda_dense_gemv_args &a, cudaStream_t s, Failure &failure) {
    const dim3 grid((a.rows + kWarps - 1) / kWarps, a.m);
    const size_t shared = (size_t)a.cols * sizeof(float);
    const bool vector = a.cols % 8 == 0 && aligned(a.w, 16);
    if (vector) {
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
    const size_t shared = size_t(a.cols) * sizeof(float);
    if (a.weight_dtype == kDtypeGgufNvfp4) {
        gguf_nvfp4_gemv<Index><<<grid, kWarps * 32, 0, stream>>>(a, index);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        return;
    }
    switch (a.weight_dtype - fifi_gguf::kDtypeBase) {
#define LAUNCH(id, block, bytes) \
    case id: \
        gguf_packed_gemv<id, block, bytes, Index><<<grid, kWarps * 32, shared, stream>>>(a, index); \
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

bool valid_moe(const inference_cuda_gguf_moe_args &a, void *stream, MoeSizes &sizes) {
    if (!stream || !a.gate || !a.down || !a.x || !a.scratch || !a.y) return false;
    if (a.hidden < 1 || a.hidden > kMaxCols || a.middle < 1 || a.middle > kMaxCols ||
        a.experts < 1 || a.experts > kMaxRows ||
        a.count < 1 || a.count > INFERENCE_CUDA_GGUF_MOE_MAX_PICKS ||
        !good_scale(a.gate_cap) || !good_scale(a.up_cap)) return false;
    if (a.gate_dtype < kDtypeGgufNvfp4 || a.down_dtype < kDtypeGgufNvfp4) return false;
    sizes = {2 * size_t(a.middle) * weight_row_bytes(a.gate_dtype, a.hidden),
             size_t(a.hidden) * weight_row_bytes(a.down_dtype, a.middle)};
    if (!sizes.gate_stride || !sizes.down_stride ||
        !aligned(a.gate, 2) || !aligned(a.down, 2) || !aligned(a.x, 4) ||
        !aligned(a.scratch, 4) || !aligned(a.y, 4)) return false;

    const Span inputs[] = {{a.gate, sizes.gate_stride * a.experts},
                           {a.down, sizes.down_stride * a.experts},
                           {a.x, size_t(a.hidden) * sizeof(float)}};
    const Span scratch = {a.scratch, size_t(a.count) * (3 * a.middle + a.hidden) * sizeof(float)};
    const Span result = {a.y, size_t(a.hidden) * sizeof(float)};
    if (!apart(scratch, result)) return false;
    for (const Span &input : inputs) {
        if (!apart(input, scratch) || !apart(input, result)) return false;
    }
    for (int i = 0; i < a.count; ++i) {
        const auto pick = a.picks[i];
        if (pick.expert >= uint32_t(a.experts) || !good_scale(pick.gate_scale) ||
            !good_scale(pick.down_scale) || !std::isfinite(pick.weight)) return false;
    }
    return true;
}

__global__ void activate_moe(inference_cuda_gguf_moe_args a, float *middle) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= a.middle) return;
    const size_t offset = size_t(blockIdx.y) * a.middle;
    const float scale = a.picks[blockIdx.y].gate_scale;
    const float gate = a.scratch[2 * offset + col] * scale;
    const float up = a.scratch[2 * offset + a.middle + col] * scale;
    const float g = a.gate_cap * tanhf(gate / a.gate_cap);
    const float u = a.up_cap * tanhf(up / a.up_cap);
    // Report an overflow from the gate projection even if the soft cap would hide it.
    middle[offset + col] = isfinite(gate) && isfinite(up)
                               ? g * (1.0f / (1.0f + expf(-gate))) * u
                               : __int_as_float(0x7fc00000);
}

__global__ void sum_moe(inference_cuda_gguf_moe_args a, const float *down) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= a.hidden) return;
    float out = 0.0f;
    for (int slot = 0; slot < a.count; ++slot) {
        const auto pick = a.picks[slot];
        const float scaled = __fmul_rn(down[size_t(slot) * a.hidden + col], pick.down_scale);
        out = __fadd_rn(out, __fmul_rn(pick.weight, scaled));
    }
    a.y[col] = out;
}

void launch_moe(const inference_cuda_gguf_moe_args &a, MoeSizes sizes, cudaStream_t stream,
                Failure &failure) {
    float *middle = a.scratch + size_t(a.count) * 2 * a.middle;
    float *down = middle + size_t(a.count) * a.middle;
    SelectedWeights index = {sizes.gate_stride, {}};
    for (int slot = 0; slot < a.count; ++slot) index.ids[slot] = a.picks[slot].expert;
    launch_packed({a.gate, a.x, a.scratch, a.count, 2 * a.middle, a.hidden,
                   0, 2 * a.middle, a.gate_dtype}, stream, failure, index);
    if (!failure.ok()) return;

    constexpr int threads = 256;
    const dim3 grid((a.middle + threads - 1) / threads, a.count);
    activate_moe<<<grid, threads, 0, stream>>>(a, middle);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (!failure.ok()) return;

    index.stride = sizes.down_stride;
    launch_packed({a.down, middle, down, a.count, a.hidden, a.middle,
                   a.middle, a.hidden, a.down_dtype}, stream, failure, index);
    if (!failure.ok()) return;

    sum_moe<<<(a.hidden + threads - 1) / threads, threads, 0, stream>>>(a, down);
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
        } else {
            launch_packed(*args, cuda_stream, failure, SharedWeights{});
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_gguf_moe_launch(const inference_cuda_gguf_moe_args *args,
                                               void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        MoeSizes sizes;
        if (!args || !valid_moe(*args, stream, sizes)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        launch_moe(*args, sizes, (cudaStream_t)stream, failure);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
