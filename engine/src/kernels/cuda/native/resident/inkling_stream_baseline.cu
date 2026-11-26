// Depth-residual stream of Inkling 2 on the device: mix, seal and add.
// The contract is resident/inkling_stream.h. Every arithmetic intrinsic is _rn: the file compiles
// under --use_fast_math, which must not turn a divide or a square root into an approximation.
#include <math.h>
#include <stdint.h>

#include "resident/inkling_stream.h"

#include "abi.h"
#include "device.h"

namespace {
constexpr int threads = 256;
constexpr int warps = threads / 32;
constexpr int max_blocks = 63;
constexpr int max_width = 8192;
constexpr int max_rows = max_blocks + 1;

// Ordered sum of one warp, lane 0 holds the result.
__device__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

// Sum of the 256 per-thread partials: thread 0 adds them in lane order.
__device__ float block_sum(float value, float *scratch) {
    const int lane = threadIdx.x;
    scratch[lane] = value;
    __syncthreads();
    if (lane == 0) {
        float total = 0.0f;
        for (int i = 0; i < threads; ++i) total = __fadd_rn(total, scratch[i]);
        scratch[0] = total;
    }
    __syncthreads();
    const float total = scratch[0];
    __syncthreads();
    return total;
}

// Rust rms_norm_with_scale: scale = (dot(x, x) / len + eps).sqrt().recip().
__device__ float rms_scale(const float *row, int width, float eps, float *scratch, int &invalid) {
    const int lane = threadIdx.x;
    float squares = 0.0f;
    for (int i = lane; i < width; i += threads) {
        const float value = row[i];
        invalid |= !isfinite(value);
        squares = __fadd_rn(squares, __fmul_rn(value, value));
    }
    const float total = block_sum(squares, scratch);
    __shared__ float shared;
    if (lane == 0) {
        const float mean = __fdiv_rn(total, static_cast<float>(width));
        const float root = __fsqrt_rn(__fadd_rn(mean, eps));
        const float scale = __frcp_rn(root);
        invalid |= !isfinite(total) || !isfinite(root) || !isfinite(scale);
        shared = scale;
    }
    __syncthreads();
    const float scale = shared;
    __syncthreads();
    return scale;
}

// Row r of the mix order: the sealed rows newest first, then the current row.
__device__ const float *mix_row(const float *state, int blocks, int rows, int r, int width) {
    if (r == rows - 1) return state;
    return state + (static_cast<size_t>(blocks) - r) * width;
}

// Quiet NaN in the health word of a block whose inputs or outputs were not finite.
__device__ void report(float *health, int invalid) {
    if (__syncthreads_or(invalid) != 0 && threadIdx.x == 0) health[0] = __int_as_float(0x7fc00000);
}

// Depth mix in one block: the scale, the scores and the weights are block-wide reductions in a
// fixed order, and one block cannot hide a memory latency behind its threads. Each lane
// therefore issues several loads ahead of its dependent adds (scores 16, mixed channels 9,
// squares 9); a lane still adds the same elements in the same order, so every rounding is that
// of the kernel that loaded one element per add. Scratch timing with one to 63 rows on a
// RTX 2080 Ti: 7.95 to 12.5 us down to 5.5 to 9.6 us for 2 to 8 rows, output bit for bit.
constexpr int kScoreLoads = 16;
constexpr int kChannelLoads = 9;
constexpr int kSquareLoads = 9;

__global__ void inkling_stream_mix(const float *state, const float *query, float *out,
                                   float *health, int first_block, int blocks, int width,
                                   float eps) {
    __shared__ float scratch[threads];
    __shared__ float scores[max_rows];
    __shared__ float weights[max_rows];
    __shared__ float shared_scale;
    const int lane = threadIdx.x;
    const int warp = lane >> 5;
    const int sub = lane & 31;
    const int rows = blocks - first_block + 1;
    int invalid = 0;
    // Rust rms_norm_with_scale on the current row: scale = (dot(x, x) / len + eps).sqrt().recip().
    {
        float squares = 0.0f;
        int i = lane;
        for (; i + threads * (kSquareLoads - 1) < width; i += threads * kSquareLoads) {
            float value[kSquareLoads];
            #pragma unroll
            for (int u = 0; u < kSquareLoads; ++u) value[u] = state[i + threads * u];
            #pragma unroll
            for (int u = 0; u < kSquareLoads; ++u) {
                invalid |= !isfinite(value[u]);
                squares = __fadd_rn(squares, __fmul_rn(value[u], value[u]));
            }
        }
        for (; i < width; i += threads) {
            const float value = state[i];
            invalid |= !isfinite(value);
            squares = __fadd_rn(squares, __fmul_rn(value, value));
        }
        const float total = block_sum(squares, scratch);
        if (lane == 0) {
            const float mean = __fdiv_rn(total, static_cast<float>(width));
            const float root = __fsqrt_rn(__fadd_rn(mean, eps));
            const float scale = __frcp_rn(root);
            invalid |= !isfinite(total) || !isfinite(root) || !isfinite(scale);
            shared_scale = scale;
        }
        __syncthreads();
    }
    const float current_scale = shared_scale;
    __syncthreads();
    // The host scores dot(row, query) * (len as f32).sqrt().recip().
    const float norm = __frcp_rn(__fsqrt_rn(static_cast<float>(width)));
    for (int r = warp; r < rows; r += warps) {
        const float *row = mix_row(state, blocks, rows, r, width);
        const float scale = r == rows - 1 ? current_scale : 1.0f;
        float sum = -0.0f;
        int i = sub;
        for (; i + 32 * (kScoreLoads - 1) < width; i += 32 * kScoreLoads) {
            float value[kScoreLoads], q[kScoreLoads];
            #pragma unroll
            for (int u = 0; u < kScoreLoads; ++u) {
                value[u] = row[i + 32 * u];
                q[u] = query[i + 32 * u];
            }
            #pragma unroll
            for (int u = 0; u < kScoreLoads; ++u) {
                invalid |= !isfinite(value[u]) || !isfinite(q[u]);
                sum = __fadd_rn(sum, __fmul_rn(__fmul_rn(value[u], scale), q[u]));
            }
        }
        for (; i < width; i += 32) {
            const float value = row[i];
            const float q = query[i];
            invalid |= !isfinite(value) || !isfinite(q);
            sum = __fadd_rn(sum, __fmul_rn(__fmul_rn(value, scale), q));
        }
        sum = warp_sum(sum);
        if (sub == 0) scores[r] = __fmul_rn(sum, norm);
    }
    __syncthreads();
    if (lane == 0) {
        float peak = -INFINITY;
        for (int r = 0; r < rows; ++r) peak = fmaxf(peak, scores[r]);
        float mass = 0.0f;
        for (int r = 0; r < rows; ++r) {
            // The host takes f32::exp of (score - peak) through libm. exp in double, rounded,
            // is within a half ulp of it; --use_fast_math leaves the double function alone.
            const float weight =
                static_cast<float>(exp(static_cast<double>(__fsub_rn(scores[r], peak))));
            weights[r] = weight;
            mass = __fadd_rn(mass, weight);
        }
        shared_scale = mass;
    }
    __syncthreads();
    const float mass = shared_scale;
    __syncthreads();
    int i = lane;
    for (; i + threads * (kChannelLoads - 1) < width; i += threads * kChannelLoads) {
        float value[kChannelLoads];
        #pragma unroll
        for (int u = 0; u < kChannelLoads; ++u) value[u] = 0.0f;
        for (int r = 0; r < rows; ++r) {
            const float *row = mix_row(state, blocks, rows, r, width);
            float x[kChannelLoads];
            #pragma unroll
            for (int u = 0; u < kChannelLoads; ++u) {
                const float raw = row[i + threads * u];
                x[u] = r == rows - 1 ? __fmul_rn(raw, current_scale) : raw;
            }
            #pragma unroll
            for (int u = 0; u < kChannelLoads; ++u) {
                value[u] = __fadd_rn(value[u], __fmul_rn(weights[r], x[u]));
            }
        }
        #pragma unroll
        for (int u = 0; u < kChannelLoads; ++u) {
            value[u] = __fdiv_rn(value[u], mass);
            out[i + threads * u] = value[u];
            invalid |= !isfinite(value[u]);
        }
    }
    for (; i < width; i += threads) {
        float value = 0.0f;
        for (int r = 0; r < rows; ++r) {
            const float x = r == rows - 1
                                ? __fmul_rn(state[i], current_scale)
                                : mix_row(state, blocks, rows, r, width)[i];
            value = __fadd_rn(value, __fmul_rn(weights[r], x));
        }
        value = __fdiv_rn(value, mass);
        out[i] = value;
        invalid |= !isfinite(value);
    }
    report(health, invalid);
}

__global__ void inkling_stream_seal(float *state, const float *next, float *health, int blocks,
                                    int width, float eps) {
    __shared__ float scratch[threads];
    const int lane = threadIdx.x;
    int invalid = 0;
    const float scale = rms_scale(state, width, eps, scratch, invalid);
    float *sealed = state + (static_cast<size_t>(blocks) + 1) * width;
    for (int i = lane; i < width; i += threads) {
        const float value = __fmul_rn(state[i], scale);
        const float replacement = next[i];
        sealed[i] = value;
        state[i] = replacement;
        invalid |= !isfinite(value) || !isfinite(replacement);
    }
    report(health, invalid);
}

__global__ void inkling_stream_add(float *state, const float *row, float *health, int width) {
    const int lane = threadIdx.x;
    int invalid = 0;
    for (int i = lane; i < width; i += threads) {
        const float value = __fadd_rn(state[i], row[i]);
        state[i] = value;
        invalid |= !isfinite(value);
    }
    report(health, invalid);
}

bool aligned(const void *pointer) { return reinterpret_cast<uintptr_t>(pointer) % alignof(float) == 0; }

__global__ void inkling_stream_norm(const float *src, float *dst, float *health, int width,
                                    float eps) {
    __shared__ float scratch[threads];
    const int lane = threadIdx.x;
    int invalid = 0;
    const float scale = rms_scale(src, width, eps, scratch, invalid);
    for (int i = lane; i < width; i += threads) {
        const float value = __fmul_rn(src[i], scale);
        dst[i] = value;
        invalid |= !isfinite(value);
    }
    report(health, invalid);
}

bool overlaps(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(a);
    const uintptr_t y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < a_bytes : x - y < b_bytes;
}

bool valid_shape(int width, float eps) {
    return width >= 1 && width <= max_width && isfinite(eps) && eps > 0.0f;
}
}  // namespace

extern "C" int inference_cuda_inkling_baseline_stream_mix(const float *state, const float *query, float *out,
                                                 float *health, int first_block, int blocks,
                                                 int width, float eps, void *stream,
                                                 int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!state || !query || !out || !health || !stream || !aligned(state) || !aligned(query) ||
        !aligned(out) || !aligned(health) || !valid_shape(width, eps) || first_block < 0 ||
        blocks < first_block || blocks > max_blocks) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const size_t row = static_cast<size_t>(width) * sizeof(float);
    const size_t state_bytes = (static_cast<size_t>(blocks) + 1) * row;
    if (overlaps(out, row, state, state_bytes) || overlaps(out, row, query, row) ||
        overlaps(health, sizeof(float), state, state_bytes) ||
        overlaps(health, sizeof(float), query, row) || overlaps(health, sizeof(float), out, row) ||
        overlaps(state, state_bytes, query, row)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    inkling_stream_mix<<<1, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        state, query, out, health, first_block, blocks, width, eps);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}

extern "C" int inference_cuda_inkling_baseline_stream_seal(float *state, const float *next, float *health,
                                                  int blocks, int width, float eps, void *stream,
                                                  int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!state || !next || !health || !stream || !aligned(state) || !aligned(next) ||
        !aligned(health) || !valid_shape(width, eps) || blocks < 0 || blocks >= max_blocks) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const size_t row = static_cast<size_t>(width) * sizeof(float);
    const size_t state_bytes = (static_cast<size_t>(blocks) + 2) * row;
    if (overlaps(state, state_bytes, next, row) || overlaps(health, sizeof(float), state, state_bytes) ||
        overlaps(health, sizeof(float), next, row)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    inkling_stream_seal<<<1, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        state, next, health, blocks, width, eps);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}

extern "C" int inference_cuda_inkling_baseline_stream_add(float *state, const float *row, float *health,
                                                 int width, void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!state || !row || !health || !stream || !aligned(state) || !aligned(row) ||
        !aligned(health) || !valid_shape(width, 1.0f)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const size_t row_bytes = static_cast<size_t>(width) * sizeof(float);
    if (overlaps(state, row_bytes, row, row_bytes) ||
        overlaps(health, sizeof(float), state, row_bytes) ||
        overlaps(health, sizeof(float), row, row_bytes)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    inkling_stream_add<<<1, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        state, row, health, width);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}

extern "C" int inference_cuda_inkling_baseline_stream_norm(const float *src, float *dst, float *health,
                                                   int width, float eps, void *stream,
                                                   int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!src || !dst || !health || !stream || !aligned(src) || !aligned(dst) ||
        !aligned(health) || !valid_shape(width, eps)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const size_t row = static_cast<size_t>(width) * sizeof(float);
    if (overlaps(dst, row, src, row) || overlaps(health, sizeof(float), src, row) ||
        overlaps(health, sizeof(float), dst, row)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    inkling_stream_norm<<<1, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        src, dst, health, width, eps);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}
