// Depth-residual stream of Inkling 2 on the device: mix, seal and add.
// The contract is resident/inkling_stream.h. Every arithmetic intrinsic is _rn, so no build flag
// can contract a multiply and an add or turn a divide or a square root into an approximation.
#include <math.h>
#include <stdint.h>

#include "resident/inkling_stream.h"
#include "resident/lane_sum.cuh"

#include "abi.h"
#include "device.h"

namespace {
constexpr int threads = 256;
constexpr int warps = threads / 32;
constexpr int max_blocks = 63;
constexpr int max_width = 8192;
constexpr int max_rows = max_blocks + 1;
// The most states one launch takes: one block per state.
constexpr int max_states = 65535;

static_assert(threads == kLaneThreads, "the stream kernels reduce rows in the lane order");

// Row r of the mix order: the sealed rows newest first, then the current row.
__device__ const float *mix_row(const float *state, int blocks, int rows, int r, int width) {
    if (r == rows - 1) return state;
    return state + (static_cast<size_t>(blocks) - r) * width;
}

// e^x from IEEE double additions and multiplications only: the host's portable_exp
// (kernels/portable_exp.rs), bit for bit, whatever the host's libm.
__device__ float portable_exp(float x) {
    if (isnan(x)) return x;
    if (!(x > -110.0f)) return 0.0f;
    if (x > 89.0f) return INFINITY;
    // 2^r = sum of (r ln 2)^i / i! for |r| <= 1/2, to the ninth power.
    const long long terms[10] = {0x3ff0000000000000LL, 0x3fe62e42fefa39efLL, 0x3fcebfbdff82c58fLL,
                                 0x3fac6b08d704a0c0LL, 0x3f83b2ab6fba4e77LL, 0x3f55d87fe78a6731LL,
                                 0x3f2430912f86c787LL, 0x3eeffcbfc588b0c7LL, 0x3eb62c0223a5c824LL,
                                 0x3e7b5253d395e7c4LL};
    // Adding and removing 1.5 * 2^52 rounds a double of magnitude below 2^51 to an integer.
    const double shift = 6755399441055744.0;
    const double z = __dmul_rn(static_cast<double>(x), __longlong_as_double(0x3ff71547652b82feLL));
    const double k = __dadd_rn(__dadd_rn(z, shift), -shift);
    const double r = __dadd_rn(z, -k);
    double p = __longlong_as_double(terms[9]);
#pragma unroll
    for (int i = 8; i >= 0; --i) p = __dadd_rn(__dmul_rn(p, r), __longlong_as_double(terms[i]));
    const double scale = __longlong_as_double((static_cast<long long>(k) + 1023) << 52);
    return __double2float_rn(__dmul_rn(p, scale));
}

// Quiet NaN in the health word of a block whose inputs or outputs were not finite.
__device__ void report(float *health, int invalid) {
    if (__syncthreads_or(invalid) != 0 && threadIdx.x == 0) health[0] = __int_as_float(0x7fc00000);
}

// Depth mix in one block: the scale, the scores and the weights are block-wide reductions in the
// lane order (`lane_sum.cuh`), and one block (all of a decode token's mix) cannot hide a memory
// latency behind other blocks. Each lane therefore keeps many loads in flight ahead of its
// dependent adds: a warp scores kScoreRows rows at once, kScoreLoads columns of each, and a lane
// mixes kChannelLoads columns of kMixRows rows at once. Every sum still adds the same elements in
// the same order, so the loads in flight change the time, not the bits.
constexpr int kScoreLoads = 16;
constexpr int kScoreRows = 2;
constexpr int kChannelLoads = 9;
constexpr int kMixRows = 4;

// `Stream::mix(first_block, query)` of one state into `out`, in the lane order.
__device__ __forceinline__ void weighted_mix(const float *state, const float *query, float *out,
                                             int first_block, int blocks, int width, float eps,
                                             float *scratch, int &invalid) {
    __shared__ float scores[max_rows];
    __shared__ float weights[max_rows];
    __shared__ float shared_mass;
    const int lane = threadIdx.x;
    const int warp = lane >> 5;
    const int sub = lane & 31;
    const int rows = blocks - first_block + 1;
    // Rust rms_norm_with_scale on the current row: scale = (dot(x, x) / len + eps).sqrt().recip().
    const float current_scale = lane_rms_scale(state, width, eps, scratch, invalid);
    // The host scores dot(row, query) * (len as f32).sqrt().recip(). Warp w scores rows w,
    // w + warps, ...; each row's sum runs over the lane's columns in order, then the warp folds it.
    const float norm = __frcp_rn(__fsqrt_rn(static_cast<float>(width)));
    for (int base = warp; base < rows; base += warps * kScoreRows) {
        const float *row[kScoreRows];
        float scale[kScoreRows], sum[kScoreRows];
        bool live[kScoreRows];
        #pragma unroll
        for (int j = 0; j < kScoreRows; ++j) {
            const int r = base + warps * j;
            live[j] = r < rows;
            row[j] = live[j] ? mix_row(state, blocks, rows, r, width) : state;
            scale[j] = r == rows - 1 ? current_scale : 1.0f;
            sum[j] = -0.0f;
        }
        int i = sub;
        for (; i + 32 * (kScoreLoads - 1) < width; i += 32 * kScoreLoads) {
            float q[kScoreLoads], value[kScoreRows][kScoreLoads];
            #pragma unroll
            for (int u = 0; u < kScoreLoads; ++u) q[u] = query[i + 32 * u];
            #pragma unroll
            for (int j = 0; j < kScoreRows; ++j) {
                #pragma unroll
                for (int u = 0; u < kScoreLoads; ++u) {
                    value[j][u] = live[j] ? row[j][i + 32 * u] : 0.0f;
                }
            }
            #pragma unroll
            for (int j = 0; j < kScoreRows; ++j) {
                if (!live[j]) continue;
                #pragma unroll
                for (int u = 0; u < kScoreLoads; ++u) {
                    invalid |= !isfinite(value[j][u]) || !isfinite(q[u]);
                    sum[j] = __fadd_rn(sum[j], __fmul_rn(__fmul_rn(value[j][u], scale[j]), q[u]));
                }
            }
        }
        for (; i < width; i += 32) {
            const float q = query[i];
            #pragma unroll
            for (int j = 0; j < kScoreRows; ++j) {
                if (!live[j]) continue;
                const float value = row[j][i];
                invalid |= !isfinite(value) || !isfinite(q);
                sum[j] = __fadd_rn(sum[j], __fmul_rn(__fmul_rn(value, scale[j]), q));
            }
        }
        #pragma unroll
        for (int j = 0; j < kScoreRows; ++j) {
            if (!live[j]) continue;
            const float total = lane_warp_sum(sum[j]);
            if (sub == 0) scores[base + warps * j] = __fmul_rn(total, norm);
        }
    }
    __syncthreads();
    if (lane == 0) {
        float peak = -INFINITY;
        for (int r = 0; r < rows; ++r) peak = fmaxf(peak, scores[r]);
        float mass = 0.0f;
        for (int r = 0; r < rows; ++r) {
            const float weight = portable_exp(__fsub_rn(scores[r], peak));
            weights[r] = weight;
            mass = __fadd_rn(mass, weight);
        }
        shared_mass = mass;
    }
    __syncthreads();
    const float mass = shared_mass;
    // Row r's value of column c: the current row (last) scaled, a sealed row as stored.
    const auto mixed = [&](int r, int c) {
        const float raw = mix_row(state, blocks, rows, r, width)[c];
        return r == rows - 1 ? __fmul_rn(raw, current_scale) : raw;
    };
    int i = lane;
    for (; i + threads * (kChannelLoads - 1) < width; i += threads * kChannelLoads) {
        float value[kChannelLoads];
        #pragma unroll
        for (int u = 0; u < kChannelLoads; ++u) value[u] = 0.0f;
        int r = 0;
        for (; r + kMixRows <= rows; r += kMixRows) {
            float x[kMixRows][kChannelLoads];
            #pragma unroll
            for (int k = 0; k < kMixRows; ++k) {
                #pragma unroll
                for (int u = 0; u < kChannelLoads; ++u) x[k][u] = mixed(r + k, i + threads * u);
            }
            #pragma unroll
            for (int k = 0; k < kMixRows; ++k) {
                #pragma unroll
                for (int u = 0; u < kChannelLoads; ++u) {
                    value[u] = __fadd_rn(value[u], __fmul_rn(weights[r + k], x[k][u]));
                }
            }
        }
        for (; r < rows; ++r) {
            float x[kChannelLoads];
            #pragma unroll
            for (int u = 0; u < kChannelLoads; ++u) x[u] = mixed(r, i + threads * u);
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
        int r = 0;
        for (; r + kMixRows <= rows; r += kMixRows) {
            float x[kMixRows];
            #pragma unroll
            for (int k = 0; k < kMixRows; ++k) x[k] = mixed(r + k, i);
            #pragma unroll
            for (int k = 0; k < kMixRows; ++k) {
                value = __fadd_rn(value, __fmul_rn(weights[r + k], x[k]));
            }
        }
        for (; r < rows; ++r) value = __fadd_rn(value, __fmul_rn(weights[r], mixed(r, i)));
        value = __fdiv_rn(value, mass);
        out[i] = value;
        invalid |= !isfinite(value);
    }
}

// One block's mix: the body of `inkling_stream_mix` (a block per state) and of each job of
// `inkling_stream_mix_jobs` (a block per job), so both give the same bits. Without `query` the
// mix is the raw current row; with `gamma` the output is `norm(mix) * gamma`, the norm's scale in
// the lane order like every other row norm, so a fused norm has the bits of a separate one.
__device__ __forceinline__ void mix_one(const float *state, const float *query, const float *gamma,
                                        float *out, float *health, int first_block, int blocks,
                                        int width, float eps, float norm_eps) {
    __shared__ float scratch[kLaneWarps];
    const int lane = threadIdx.x;
    int invalid = 0;
    if (query) {
        weighted_mix(state, query, out, first_block, blocks, width, eps, scratch, invalid);
    } else {
        for (int i = lane; i < width; i += threads) {
            const float value = state[i];
            invalid |= !isfinite(value);
            out[i] = value;
        }
    }
    if (gamma) {
        // Each lane reads back only the columns it wrote.
        const float scale = lane_rms_scale(out, width, norm_eps, scratch, invalid);
        for (int i = lane; i < width; i += threads) {
            const float value = __fmul_rn(__fmul_rn(out[i], scale), gamma[i]);
            out[i] = value;
            invalid |= !isfinite(value);
        }
    }
    report(health, invalid);
}

__global__ void inkling_stream_mix(const float *state, const float *query, const float *gamma,
                                   float *out, float *health, int first_block, int blocks,
                                   int width, float eps, float norm_eps, int stride,
                                   int out_stride) {
    mix_one(state + static_cast<size_t>(blockIdx.x) * stride * width, query, gamma,
            out + static_cast<size_t>(blockIdx.x) * out_stride, health, first_block, blocks, width,
            eps, norm_eps);
}

// Independent mixes in one launch, a block per job: the bank queries and the attention query of
// a visit read different stream states, so they need not wait on each other. The lists travel
// in the kernel's parameter space; only the row pointers address device memory.
constexpr int max_mix_jobs = 8;

struct MixJobs {
    const float *state[max_mix_jobs];
    const float *query[max_mix_jobs];
    float *out[max_mix_jobs];
    int first_block[max_mix_jobs];
    int blocks[max_mix_jobs];
};

__global__ void inkling_stream_mix_jobs(MixJobs jobs, float *health, int width, float eps) {
    const int j = blockIdx.x;
    mix_one(jobs.state[j], jobs.query[j], nullptr, jobs.out[j], health, jobs.first_block[j],
            jobs.blocks[j], width, eps, 1.0f);
}

__global__ void inkling_stream_seal(float *state, const float *next, float *health, int blocks,
                                    int width, float eps, int stride) {
    state += static_cast<size_t>(blockIdx.x) * stride * width;
    next += static_cast<size_t>(blockIdx.x) * width;
    __shared__ float scratch[kLaneWarps];
    const int lane = threadIdx.x;
    int invalid = 0;
    const float scale = lane_rms_scale(state, width, eps, scratch, invalid);
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

__global__ void inkling_stream_add(float *state, const float *row, float *health, int width,
                                   int stride) {
    state += static_cast<size_t>(blockIdx.x) * stride * width;
    row += static_cast<size_t>(blockIdx.x) * width;
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

// One block's norm, `dst = src * rms_scale(src)`: a row of `inkling_stream_norm_rows`.
__device__ __forceinline__ void norm_one(const float *src, float *dst, float *health, int width,
                                         float eps) {
    __shared__ float scratch[kLaneWarps];
    const int lane = threadIdx.x;
    int invalid = 0;
    const float scale = lane_rms_scale(src, width, eps, scratch, invalid);
    for (int i = lane; i < width; i += threads) {
        const float value = __fmul_rn(src[i], scale);
        dst[i] = value;
        invalid |= !isfinite(value);
    }
    report(health, invalid);
}

// Independent norms of one width in one launch, a block per row: the gate halves of a visit's
// controllers. The lists travel in the kernel's parameter space.
constexpr int max_norm_rows = 8;

struct NormRows {
    const float *src[max_norm_rows];
    float *dst[max_norm_rows];
    float *health[max_norm_rows];
};

__global__ void inkling_stream_norm_rows(NormRows rows, int width, float eps) {
    const int r = blockIdx.x;
    norm_one(rows.src[r], rows.dst[r], rows.health[r], width, eps);
}

bool overlaps(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(a);
    const uintptr_t y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < a_bytes : x - y < b_bytes;
}

bool valid_shape(int width, float eps) {
    return width >= 1 && width <= max_width && isfinite(eps) && eps > 0.0f;
}

// `rows` states of `used` rows each, `stride >= used` rows apart; 0 when the shape is invalid.
size_t states_bytes(int rows, int stride, int used, int width) {
    if (rows < 1 || rows > max_states || stride < used) return 0;
    return ((static_cast<size_t>(rows) - 1) * stride + used) * width * sizeof(float);
}
}  // namespace

extern "C" int inference_cuda_inkling_stream_mix(const float *state, const float *query,
                                                 const float *gamma, float *out, float *health,
                                                 int first_block, int blocks, int width,
                                                 float mix_eps, float norm_eps, int rows,
                                                 int stride, int out_stride, void *stream,
                                                 int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!state || !out || !health || !stream || !aligned(state) || !aligned(query) ||
        !aligned(gamma) || !aligned(out) || !aligned(health) ||
        !valid_shape(width, query ? mix_eps : 1.0f) || !valid_shape(width, gamma ? norm_eps : 1.0f) ||
        out_stride < width || first_block < 0 || blocks < first_block || blocks > max_blocks) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const size_t row = static_cast<size_t>(width) * sizeof(float);
    const size_t state_bytes = states_bytes(rows, stride, blocks + 1, width);
    const size_t out_bytes =
        ((static_cast<size_t>(rows) - 1) * out_stride + width) * sizeof(float);
    if (state_bytes == 0 || overlaps(out, out_bytes, state, state_bytes) ||
        (query && (overlaps(out, out_bytes, query, row) ||
                   overlaps(health, sizeof(float), query, row))) ||
        (gamma && (overlaps(out, out_bytes, gamma, row) ||
                   overlaps(health, sizeof(float), gamma, row))) ||
        overlaps(health, sizeof(float), state, state_bytes) ||
        overlaps(health, sizeof(float), out, out_bytes)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    inkling_stream_mix<<<rows, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        state, query, gamma, out, health, first_block, blocks, width, mix_eps, norm_eps, stride,
        out_stride);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}

extern "C" int inference_cuda_inkling_stream_mix_jobs(const void *const *rows, const int *extents,
                                                      int count, float *health, int width,
                                                      float eps, void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    // `rows` holds `state, query, out` per job and `extents` holds `first_block, blocks`.
    if (!rows || !extents || !health || !stream || count < 1 || count > max_mix_jobs ||
        !aligned(health) || !valid_shape(width, eps)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const size_t row = static_cast<size_t>(width) * sizeof(float);
    MixJobs jobs{};
    for (int j = 0; j < count; ++j) {
        const float *state = static_cast<const float *>(rows[3 * j]);
        const float *query = static_cast<const float *>(rows[3 * j + 1]);
        float *out = static_cast<float *>(const_cast<void *>(rows[3 * j + 2]));
        const int first_block = extents[2 * j], blocks = extents[2 * j + 1];
        if (!state || !query || !out || !aligned(state) || !aligned(query) || !aligned(out) ||
            first_block < 0 || blocks < first_block || blocks > max_blocks) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        jobs.state[j] = state;
        jobs.query[j] = query;
        jobs.out[j] = out;
        jobs.first_block[j] = first_block;
        jobs.blocks[j] = blocks;
    }
    // Every output and the health word are apart from every row a job reads and from each other.
    for (int j = 0; j < count; ++j) {
        const size_t state_bytes = static_cast<size_t>(jobs.blocks[j] + 1) * row;
        if (overlaps(health, sizeof(float), jobs.state[j], state_bytes) ||
            overlaps(health, sizeof(float), jobs.query[j], row) ||
            overlaps(health, sizeof(float), jobs.out[j], row)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        for (int k = 0; k < count; ++k) {
            const size_t read_bytes = static_cast<size_t>(jobs.blocks[k] + 1) * row;
            if (overlaps(jobs.out[j], row, jobs.state[k], read_bytes) ||
                overlaps(jobs.out[j], row, jobs.query[k], row) ||
                (k != j && overlaps(jobs.out[j], row, jobs.out[k], row))) {
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
        }
    }
    inkling_stream_mix_jobs<<<count, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        jobs, health, width, eps);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}

extern "C" int inference_cuda_inkling_stream_seal(float *state, const float *next, float *health,
                                                  int blocks, int width, float eps, int rows,
                                                  int stride, void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!state || !next || !health || !stream || !aligned(state) || !aligned(next) ||
        !aligned(health) || !valid_shape(width, eps) || blocks < 0 || blocks >= max_blocks) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const size_t row = static_cast<size_t>(width) * sizeof(float);
    const size_t state_bytes = states_bytes(rows, stride, blocks + 2, width);
    const size_t next_bytes = static_cast<size_t>(rows) * row;
    if (state_bytes == 0 || overlaps(state, state_bytes, next, next_bytes) ||
        overlaps(health, sizeof(float), state, state_bytes) ||
        overlaps(health, sizeof(float), next, next_bytes)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    inkling_stream_seal<<<rows, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        state, next, health, blocks, width, eps, stride);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}

extern "C" int inference_cuda_inkling_stream_add(float *state, const float *row, float *health,
                                                 int width, int rows, int stride, void *stream,
                                                 int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!state || !row || !health || !stream || !aligned(state) || !aligned(row) ||
        !aligned(health) || !valid_shape(width, 1.0f)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const size_t state_bytes = states_bytes(rows, stride, 1, width);
    const size_t row_bytes = static_cast<size_t>(rows) * width * sizeof(float);
    if (state_bytes == 0 || overlaps(state, state_bytes, row, row_bytes) ||
        overlaps(health, sizeof(float), state, state_bytes) ||
        overlaps(health, sizeof(float), row, row_bytes)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    inkling_stream_add<<<rows, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        state, row, health, width, stride);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}

extern "C" int inference_cuda_inkling_stream_norm_rows(const void *const *ranges, int count,
                                                        int width, float eps, void *stream,
                                                        int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    // `ranges` holds `src, dst, health` per row.
    if (!ranges || !stream || count < 1 || count > max_norm_rows || !valid_shape(width, eps)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    NormRows rows{};
    for (int r = 0; r < count; ++r) {
        const void *const *row = ranges + 3 * r;
        if (!row[0] || !row[1] || !row[2] || !aligned(row[0]) || !aligned(row[1]) ||
            !aligned(row[2])) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        rows.src[r] = static_cast<const float *>(row[0]);
        rows.dst[r] = static_cast<float *>(const_cast<void *>(row[1]));
        rows.health[r] = static_cast<float *>(const_cast<void *>(row[2]));
    }
    // Every output is apart from every source and every other output, and every health word
    // from every source and output. Health words may be one word: each writer stores the same
    // quiet NaN.
    const size_t row = static_cast<size_t>(width) * sizeof(float);
    for (int r = 0; r < count; ++r) {
        for (int k = 0; k < count; ++k) {
            if (overlaps(rows.dst[r], row, rows.src[k], row) ||
                (k != r && overlaps(rows.dst[r], row, rows.dst[k], row)) ||
                overlaps(rows.health[r], sizeof(float), rows.src[k], row) ||
                overlaps(rows.health[r], sizeof(float), rows.dst[k], row)) {
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
        }
    }
    inkling_stream_norm_rows<<<count, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        rows, width, eps);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}
