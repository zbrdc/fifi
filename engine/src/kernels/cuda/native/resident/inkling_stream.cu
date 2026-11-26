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

// Sum of the 256 per-thread partials: each warp adds its lanes in tree order, thread 0 adds
// the eight warp partials in warp order. (Before: thread 0 added all 256 in lane order.)
__device__ float block_sum(float value, float *scratch) {
    const int lane = threadIdx.x;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    if (lane % 32 == 0) scratch[lane >> 5] = value;
    __syncthreads();
    if (lane == 0) {
        float total = 0.0f;
        for (int i = 0; i < warps; ++i) total = __fadd_rn(total, scratch[i]);
        scratch[0] = total;
    }
    __syncthreads();
    const float total = scratch[0];
    __syncthreads();
    return total;
}

// Rust rms_norm_with_scale: scale = (dot(x, x) / len + eps).sqrt().recip(). Four squares are
// in flight per thread; every thread finishes the same arithmetic on the block sum, so the
// scale needs no broadcast.
constexpr int kSquareLoads = 4;

__device__ float rms_scale(const float *row, int width, float eps, float *scratch, int &invalid) {
    float squares = 0.0f;
    int i = threadIdx.x;
    for (; i + threads * (kSquareLoads - 1) < width; i += threads * kSquareLoads) {
        float value[kSquareLoads];
        #pragma unroll
        for (int u = 0; u < kSquareLoads; ++u) value[u] = row[i + threads * u];
        #pragma unroll
        for (int u = 0; u < kSquareLoads; ++u) {
            invalid |= !isfinite(value[u]);
            squares = __fadd_rn(squares, __fmul_rn(value[u], value[u]));
        }
    }
    for (; i < width; i += threads) {
        const float value = row[i];
        invalid |= !isfinite(value);
        squares = __fadd_rn(squares, __fmul_rn(value, value));
    }
    const float total = block_sum(squares, scratch);
    const float mean = __fdiv_rn(total, static_cast<float>(width));
    const float root = __fsqrt_rn(__fadd_rn(mean, eps));
    const float scale = __frcp_rn(root);
    invalid |= !isfinite(total) || !isfinite(root) || !isfinite(scale);
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
constexpr int kMixSquareLoads = 9;

// Shared words of one mix block, in this order: the reduction scratch, the scores, the weights,
// then one broadcast word (the current row's scale, later the mass).
constexpr int kMixWords = threads + 2 * max_rows + 1;
constexpr int kMixShared = kMixWords * sizeof(float);
constexpr int kMaxMixJobs = 8;

// Independent mixes of one visit: the bank queries and the attention query read different
// stream states, so one block each runs them in parallel in one launch. A job may add a row
// into the current row of its state first and norm its output (`mix_one`); null means no.
struct MixJobs {
    const float *state[kMaxMixJobs];
    const float *query[kMaxMixJobs];
    float *out[kMaxMixJobs];
    float *health[kMaxMixJobs];
    const float *addend[kMaxMixJobs];
    const float *gamma[kMaxMixJobs];
    float *normed[kMaxMixJobs];
    float norm_eps[kMaxMixJobs];
    int first_block[kMaxMixJobs];
    int blocks[kMaxMixJobs];
};

// `addend` is null or a row added into the current row of `state` before anything else: the
// host's `add_current`, one rounded add, the bits of inkling_stream_add. The mix then reads the
// sum, and the current row keeps it. `gamma` and `normed` are null or the mix also writes
// `normed[i] = (out[i] * scale) * gamma[i]` with the host's weighted RMSNorm scale of `out`
// (`rms_norm_with_scale` with `norm_eps`), the row inkling_weighted_norm would produce.
__device__ void mix_one(const float *state, const float *addend, const float *query, float *out,
                        const float *gamma, float *normed, float norm_eps, float *health,
                        int first_block, int blocks, int width, float eps, float *shared) {
    float *const scratch = shared;
    float *const scores = shared + threads;
    float *const weights = shared + threads + max_rows;
    float *const shared_scale = shared + threads + 2 * max_rows;
    const int lane = threadIdx.x;
    const int warp = lane >> 5;
    const int sub = lane & 31;
    const int rows = blocks - first_block + 1;
    int invalid = 0;
    // The host's add_current of `addend` into the current row comes first; every later read
    // of the row sees the sum (the block sums below are barriers).
    if (addend) {
        float *const current = const_cast<float *>(state);
        for (int i = lane; i < width; i += threads) {
            current[i] = __fadd_rn(current[i], addend[i]);
        }
        __syncthreads();
    }
    // Rust rms_norm_with_scale on the current row: scale = (dot(x, x) / len + eps).sqrt().recip().
    {
        float squares = 0.0f;
        int i = lane;
        for (; i + threads * (kMixSquareLoads - 1) < width; i += threads * kMixSquareLoads) {
            float value[kMixSquareLoads];
            #pragma unroll
            for (int u = 0; u < kMixSquareLoads; ++u) value[u] = state[i + threads * u];
            #pragma unroll
            for (int u = 0; u < kMixSquareLoads; ++u) {
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
            *shared_scale = scale;
        }
        __syncthreads();
    }
    const float current_scale = *shared_scale;
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
    // The softmax over at most max_rows scores: the peak, then one exp per lane in parallel
    // (rows lane, lane + 32, ...), then the mass as a block sum. The host takes f32::exp of
    // (score - peak) through libm; exp in double, rounded, is within a half ulp of it and
    // --use_fast_math leaves the double function alone.
    {
        float peak = -INFINITY;
        for (int r = lane; r < rows; r += threads) peak = fmaxf(peak, scores[r]);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, offset));
        }
        if (lane % 32 == 0) scratch[lane >> 5] = peak;
        __syncthreads();
        if (lane == 0) {
            float top = -INFINITY;
            for (int w = 0; w < warps; ++w) top = fmaxf(top, scratch[w]);
            *shared_scale = top;
        }
        __syncthreads();
        const float top = *shared_scale;
        float mass = 0.0f;
        for (int r = lane; r < rows; r += threads) {
            // The host takes f32::exp of (score - peak) through libm. exp in double, rounded,
            // is within a half ulp of it; --use_fast_math leaves the double function alone.
            const float weight = static_cast<float>(exp(static_cast<double>(__fsub_rn(scores[r], top))));
            weights[r] = weight;
            mass = __fadd_rn(mass, weight);
        }
        mass = block_sum(mass, scratch);
        if (lane == 0) *shared_scale = mass;
        __syncthreads();
    }
    const float mass = *shared_scale;
    __syncthreads();
    // The weighted mean. Four rows are in flight per thread: the loads of the next rows are
    // issued before the adds of the previous ones, so a row costs its bytes, not a latency.
    constexpr int kRows = 4;
    int i = lane;
    for (; i + threads * (kChannelLoads - 1) < width; i += threads * kChannelLoads) {
        float value[kChannelLoads];
        #pragma unroll
        for (int u = 0; u < kChannelLoads; ++u) value[u] = 0.0f;
        int r = 0;
        for (; r + kRows <= rows; r += kRows) {
            float x[kRows][kChannelLoads];
            #pragma unroll
            for (int v = 0; v < kRows; ++v) {
                const float *row = mix_row(state, blocks, rows, r + v, width);
                #pragma unroll
                for (int u = 0; u < kChannelLoads; ++u) x[v][u] = row[i + threads * u];
            }
            #pragma unroll
            for (int v = 0; v < kRows; ++v) {
                const float scaled = r + v == rows - 1 ? current_scale : 1.0f;
                #pragma unroll
                for (int u = 0; u < kChannelLoads; ++u) {
                    const float raw = x[v][u];
                    invalid |= !isfinite(raw);
                    value[u] = __fadd_rn(value[u], __fmul_rn(weights[r + v], __fmul_rn(raw, scaled)));
                }
            }
        }
        for (; r < rows; ++r) {
            const float *row = mix_row(state, blocks, rows, r, width);
            const float scale = r == rows - 1 ? current_scale : 1.0f;
            float x[kChannelLoads];
            #pragma unroll
            for (int u = 0; u < kChannelLoads; ++u) {
                const float raw = row[i + threads * u];
                invalid |= !isfinite(raw);
                x[u] = __fmul_rn(raw, scale);
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
    if (normed) {
        // The host's weighted RMSNorm of the output, from the row just written.
        const float scale = rms_scale(out, width, norm_eps, scratch, invalid);
        int i = lane;
        for (; i + threads * (kSquareLoads - 1) < width; i += threads * kSquareLoads) {
            float value[kSquareLoads], weight[kSquareLoads];
            #pragma unroll
            for (int u = 0; u < kSquareLoads; ++u) {
                value[u] = out[i + threads * u];
                weight[u] = gamma[i + threads * u];
            }
            #pragma unroll
            for (int u = 0; u < kSquareLoads; ++u) {
                const float result = __fmul_rn(__fmul_rn(value[u], scale), weight[u]);
                normed[i + threads * u] = result;
                invalid |= !isfinite(result);
            }
        }
        for (; i < width; i += threads) {
            const float result = __fmul_rn(__fmul_rn(out[i], scale), gamma[i]);
            normed[i] = result;
            invalid |= !isfinite(result);
        }
    }
    report(health, invalid);
}

__global__ void inkling_stream_mix(const float *state, const float *query, float *out,
                                   float *health, int first_block, int blocks, int width,
                                   float eps) {
    extern __shared__ float shared[];
    mix_one(state, nullptr, query, out, nullptr, nullptr, 0.0f, health, first_block, blocks,
            width, eps, shared);
}

__global__ void inkling_stream_mix_jobs(MixJobs jobs, int width, float eps) {
    extern __shared__ float shared[];
    const int j = blockIdx.x;
    mix_one(jobs.state[j], jobs.addend[j], jobs.query[j], jobs.out[j], jobs.gamma[j],
            jobs.normed[j], jobs.norm_eps[j], jobs.health[j], jobs.first_block[j], jobs.blocks[j],
            width, eps, shared);
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

// `out[i] = src[i] * scale` with four loads in flight per thread.
__device__ void scale_row(const float *src, float *dst, int width, float scale, int &invalid) {
    int i = threadIdx.x;
    for (; i + threads * (kSquareLoads - 1) < width; i += threads * kSquareLoads) {
        float value[kSquareLoads];
        #pragma unroll
        for (int u = 0; u < kSquareLoads; ++u) value[u] = src[i + threads * u];
        #pragma unroll
        for (int u = 0; u < kSquareLoads; ++u) {
            const float result = __fmul_rn(value[u], scale);
            invalid |= !isfinite(result);
            dst[i + threads * u] = result;
        }
    }
    for (; i < width; i += threads) {
        const float result = __fmul_rn(src[i], scale);
        invalid |= !isfinite(result);
        dst[i] = result;
    }
}

// Norm rows of one launch: independent norms of one width, one block each (the gate halves of
// a visit's controllers).
constexpr int kMaxNormRows = 8;

struct NormRows {
    const float *src[kMaxNormRows];
    float *dst[kMaxNormRows];
    float *health[kMaxNormRows];
};

__global__ void inkling_stream_norm_rows(NormRows rows, int width, float eps) {
    __shared__ float scratch[threads];
    const int r = blockIdx.x;
    int invalid = 0;
    const float scale = rms_scale(rows.src[r], width, eps, scratch, invalid);
    scale_row(rows.src[r], rows.dst[r], width, scale, invalid);
    report(rows.health[r], invalid);
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

extern "C" int inference_cuda_inkling_stream_mix(const float *state, const float *query, float *out,
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
    inkling_stream_mix<<<1, threads, kMixShared, static_cast<cudaStream_t>(stream)>>>(
        state, query, out, health, first_block, blocks, width, eps);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}

extern "C" int inference_cuda_inkling_stream_mix_jobs(const void *const *ranges, const int *shape,
                                                       const float *norm_eps, int count, int width,
                                                       float eps, void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    // `ranges` holds `state, query, out, health, addend, gamma, normed` per job, `shape` its
    // `first_block, blocks` and `norm_eps` its epsilon when it norms its output; the last three
    // pointers are null or (for gamma and normed) both set.
    if (!ranges || !shape || !stream || count < 1 || count > kMaxMixJobs || !valid_shape(width, eps)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    MixJobs jobs{};
    for (int j = 0; j < count; ++j) {
        const void *const *r = ranges + 7 * j;
        const int first_block = shape[2 * j], blocks = shape[2 * j + 1];
        if (!r[0] || !r[1] || !r[2] || !r[3] || !aligned(r[0]) || !aligned(r[1]) ||
            !aligned(r[2]) || !aligned(r[3]) || first_block < 0 || blocks < first_block ||
            blocks > max_blocks || (r[4] && !aligned(r[4])) ||
            (!r[5] != !r[6]) || (r[5] && (!aligned(r[5]) || !aligned(r[6]))) ||
            (!norm_eps && (r[5] || r[6])) || !isfinite(norm_eps ? norm_eps[j] : 0.0f)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        jobs.state[j] = static_cast<const float *>(r[0]);
        jobs.query[j] = static_cast<const float *>(r[1]);
        jobs.out[j] = static_cast<float *>(const_cast<void *>(r[2]));
        jobs.health[j] = static_cast<float *>(const_cast<void *>(r[3]));
        jobs.addend[j] = static_cast<const float *>(r[4]);
        jobs.gamma[j] = static_cast<const float *>(r[5]);
        jobs.normed[j] = static_cast<float *>(const_cast<void *>(r[6]));
        jobs.norm_eps[j] = norm_eps ? norm_eps[j] : 0.0f;
        jobs.first_block[j] = first_block;
        jobs.blocks[j] = blocks;
    }
    const size_t row = static_cast<size_t>(width) * sizeof(float);
    const size_t health = sizeof(float);
    for (int j = 0; j < count; ++j) {
        const size_t state_bytes = (static_cast<size_t>(jobs.blocks[j]) + 1) * row;
        // The job's written ranges (its output, its normalized row, its health word and, with
        // an addend, the current row of its state) share no byte with any of its other ranges.
        const float *const written[] = {jobs.out[j], jobs.normed[j], jobs.health[j]};
        const size_t written_bytes[] = {row, row, health};
        const float *const others[] = {jobs.state[j], jobs.query[j], jobs.addend[j], jobs.gamma[j]};
        const size_t other_bytes[] = {state_bytes, row, row, row};
        for (int w = 0; w < 3; ++w) {
            if (!written[w]) continue;
            for (int r = 0; r < 4; ++r) {
                if (others[r] && overlaps(written[w], written_bytes[w], others[r], other_bytes[r]))
                    return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            for (int v = w + 1; v < 3; ++v) {
                if (written[v] && overlaps(written[w], written_bytes[w], written[v], written_bytes[v]))
                    return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
        }
        if (jobs.addend[j] &&
            overlaps(jobs.state[j], state_bytes, jobs.addend[j], row)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        // No job writes into a row another job reads; the health word may be shared (every
        // writer stores the same quiet NaN) and the outputs must be distinct rows.
        for (int k = j + 1; k < count; ++k) {
            const size_t other_state = (static_cast<size_t>(jobs.blocks[k]) + 1) * row;
            const bool shared_health = jobs.health[j] == jobs.health[k];
            const float *const writes[2][4] = {
                {jobs.out[j], jobs.normed[j], jobs.addend[j] ? jobs.state[j] : nullptr, jobs.health[j]},
                {jobs.out[k], jobs.normed[k], jobs.addend[k] ? jobs.state[k] : nullptr, jobs.health[k]}};
            const size_t write_bytes[2][4] = {{row, row, row, health}, {row, row, row, health}};
            const float *const reads[2][4] = {
                {jobs.state[j], jobs.query[j], jobs.addend[j], jobs.gamma[j]},
                {jobs.state[k], jobs.query[k], jobs.addend[k], jobs.gamma[k]}};
            const size_t read_bytes[2][4] = {{state_bytes, row, row, row}, {other_state, row, row, row}};
            for (int side = 0; side < 2; ++side) {
                const int a = side, b = 1 - side;
                for (int w = 0; w < 4; ++w) {
                    if (!writes[a][w]) continue;
                    for (int r = 0; r < 4; ++r) {
                        if (!reads[b][r]) continue;
                        // A current row written by one job is read by the other job only if
                        // both name the same state rows: two addends into one row, refused.
                        if (reads[b][r] == writes[a][w]) {
                            if (w == 2 && r == 0) return INFERENCE_CUDA_INVALID_ARGUMENT;
                            continue;
                        }
                        if (overlaps(writes[a][w], write_bytes[a][w], reads[b][r], read_bytes[b][r]))
                            return INFERENCE_CUDA_INVALID_ARGUMENT;
                    }
                    for (int v = 0; v < 4; ++v) {
                        // The shared health word: both writers store the same value.
                        if (!writes[b][v] || (shared_health && w == 3 && v == 3)) continue;
                        if (writes[a][w] == writes[b][v]) continue;
                        if (overlaps(writes[a][w], write_bytes[a][w], writes[b][v], write_bytes[b][v]))
                            return INFERENCE_CUDA_INVALID_ARGUMENT;
                    }
                }
            }
        }
    }
    inkling_stream_mix_jobs<<<count, threads, kMixShared, static_cast<cudaStream_t>(stream)>>>(
        jobs, width, eps);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}

extern "C" int inference_cuda_inkling_stream_seal(float *state, const float *next, float *health,
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

extern "C" int inference_cuda_inkling_stream_add(float *state, const float *row, float *health,
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

// Bookkeeping rows of one launch: the status words of every visit slot are zeroed before the
// chain and gathered into the journal after it, one block per slot instead of one copy each.
constexpr int kMaxWordRows = 64;

// The pointer lists and lengths live on the host. Capture them in CUDA's kernel
// parameter space; only the row pointers themselves refer to device memory.
struct WordRows {
    const float *source[kMaxWordRows];
    float *destination[kMaxWordRows];
    int words[kMaxWordRows];
};

template<bool Copy>
__global__ void inkling_word_rows(WordRows rows) {
    const int r = blockIdx.x;
    for (int i = threadIdx.x; i < rows.words[r]; i += blockDim.x) {
        rows.destination[r][i] = Copy ? rows.source[r][i] : 0.0f;
    }
}

extern "C" int inference_cuda_inkling_zero_words(const void *const *rows, const int *words,
                                                 int count, void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!rows || !words || !stream || count < 1 || count > kMaxWordRows) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    WordRows packed{};
    for (int r = 0; r < count; ++r) {
        if (!rows[r] || !aligned(rows[r]) || words[r] < 1) return INFERENCE_CUDA_INVALID_ARGUMENT;
        packed.destination[r] = static_cast<float *>(const_cast<void *>(rows[r]));
        packed.words[r] = words[r];
    }
    inkling_word_rows<false><<<count, threads, 0, static_cast<cudaStream_t>(stream)>>>(packed);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}

extern "C" int inference_cuda_inkling_copy_words(const void *const *source, void *const *destination,
                                                 const int *words, int count, void *stream,
                                                 int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!source || !destination || !words || !stream || count < 1 || count > kMaxWordRows) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    WordRows packed{};
    for (int r = 0; r < count; ++r) {
        if (!source[r] || !destination[r] || !aligned(source[r]) || !aligned(destination[r]) ||
            words[r] < 1 || overlaps(source[r], words[r] * sizeof(float), destination[r],
                                     words[r] * sizeof(float))) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        packed.source[r] = static_cast<const float *>(source[r]);
        packed.destination[r] = static_cast<float *>(destination[r]);
        packed.words[r] = words[r];
    }
    inkling_word_rows<true><<<count, threads, 0, static_cast<cudaStream_t>(stream)>>>(packed);
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
    if (!ranges || !stream || count < 1 || count > kMaxNormRows || !valid_shape(width, eps)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    NormRows rows{};
    for (int r = 0; r < count; ++r) {
        const void *const *pair = ranges + 3 * r;
        if (!pair[0] || !pair[1] || !pair[2] || !aligned(pair[0]) || !aligned(pair[1]) ||
            !aligned(pair[2])) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        rows.src[r] = static_cast<const float *>(pair[0]);
        rows.dst[r] = static_cast<float *>(const_cast<void *>(pair[1]));
        rows.health[r] = static_cast<float *>(const_cast<void *>(pair[2]));
    }
    const size_t row = static_cast<size_t>(width) * sizeof(float);
    const size_t health = sizeof(float);
    for (int r = 0; r < count; ++r) {
        if (overlaps(rows.dst[r], row, rows.src[r], row) ||
            overlaps(rows.health[r], health, rows.src[r], row) ||
            overlaps(rows.health[r], health, rows.dst[r], row)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        // Rows are independent: no row writes into another row's source or output, and health
        // words are distinct or the same word (every writer stores the same quiet NaN).
        for (int k = r + 1; k < count; ++k) {
            if (overlaps(rows.dst[r], row, rows.src[k], row) ||
                overlaps(rows.dst[k], row, rows.src[r], row) ||
                overlaps(rows.dst[r], row, rows.dst[k], row) ||
                (rows.health[r] != rows.health[k] &&
                 (overlaps(rows.dst[r], row, rows.health[k], health) ||
                  overlaps(rows.dst[k], row, rows.health[r], health) ||
                  overlaps(rows.health[r], health, rows.src[k], row) ||
                  overlaps(rows.health[k], health, rows.src[r], row)))) {
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
