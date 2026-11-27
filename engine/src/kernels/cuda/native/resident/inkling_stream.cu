// Depth-residual stream of Inkling 2 on the device: mix, seal and add.
// The contract is resident/inkling_stream.h. Every arithmetic intrinsic is _rn, so no build flag
// can contract a multiply and an add or turn a divide or a square root into an approximation.
#include <math.h>
#include <stdint.h>

#include "resident/inkling_stream.h"
#include "resident/ordered_sum.cuh"
#include "resident/inkling_stream_norm.cuh"

#include "abi.h"
#include "device.h"

namespace {
constexpr int threads = 256;
static_assert(threads == kNormThreads, "the gate norm adds one partial per thread");
constexpr int warps = threads / 32;
constexpr int max_blocks = 63;
constexpr int max_width = 8192;
constexpr int max_rows = max_blocks + 1;
// The most states one launch takes: one block per state.
constexpr int max_states = 65535;

// Ordered sum of one warp, lane 0 holds the result.
__device__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

// `block_sum` and `rms_scale` (Rust rms_norm_with_scale: scale = (dot(x, x) / len
// + eps).sqrt().recip()) live in resident/inkling_stream_norm.cuh, which the fused decode
// controller launch shares.

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

// Depth mix in one block: the scale, the scores and the weights are block-wide reductions in a
// fixed order, and one block cannot hide a memory latency behind its threads. Each lane
// therefore issues several loads ahead of its dependent adds (scores 16, mixed channels 9,
// squares 9); a lane still adds the same elements in the same order, so every rounding is that
// of the kernel that loaded one element per add. Scratch timing with one to 63 rows on a
// RTX 2080 Ti: 7.95 to 12.5 us down to 5.5 to 9.6 us for 2 to 8 rows, output bit for bit.
constexpr int kScoreLoads = 16;
constexpr int kChannelLoads = 9;
constexpr int kSquareLoads = 9;

// What a mix of the decode glue (FIFI_INKLING_CUDA_DECODE_GLUE) folds in; all null for a plain
// mix. `addend` is a row added into the current row of the state before anything else: the
// host's `add_current`, one rounded add, the bits of inkling_stream_add; the mix then reads the
// sum and the current row keeps it. `gamma` and `normed` make the mix also write
// `normed[i] = (out[i] * scale) * gamma[i]`, the host's weighted RMSNorm of its own output with
// `norm_eps` (`rms_norm_with_scale`), the row inkling_weighted_norm would produce.
struct MixExtras {
    const float *addend;
    const float *gamma;
    float *normed;
    float norm_eps;
};

// One block's mix: the body of `inkling_stream_mix` (a block per state) and of each job of
// `inkling_stream_mix_jobs` (a block per job), so both give the same bits.
__device__ __forceinline__ void mix_one(const float *state, const float *query, float *out,
                                        float *health, int first_block, int blocks, int width,
                                        float eps, MixExtras extras) {
    __shared__ float scratch[threads];
    __shared__ float scores[max_rows];
    __shared__ float weights[max_rows];
    __shared__ float shared_scale;
    const int lane = threadIdx.x;
    const int warp = lane >> 5;
    const int sub = lane & 31;
    const int rows = blocks - first_block + 1;
    int invalid = 0;
    // The host's add comes first; the block sum below is the barrier after which every lane's
    // later reads of the current row see the sum.
    if (extras.addend) {
        float *const current = const_cast<float *>(state);
        for (int i = lane; i < width; i += threads) {
            const float value = __fadd_rn(current[i], extras.addend[i]);
            current[i] = value;
            invalid |= !isfinite(value);
        }
        __syncthreads();
    }
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
            const float weight = portable_exp(__fsub_rn(scores[r], peak));
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
    if (extras.normed) {
        // The weighted norm of the row this block just wrote: a lane reads back the elements it
        // wrote, so the sum of squares needs no further barrier. The scale is the host's within
        // floating-point error, as the depth stream's reductions are.
        const float scale = rms_scale(out, width, extras.norm_eps, scratch, invalid);
        int i = lane;
        for (; i + threads * (kSquareLoads - 1) < width; i += threads * kSquareLoads) {
            float value[kSquareLoads], weight[kSquareLoads];
            #pragma unroll
            for (int u = 0; u < kSquareLoads; ++u) {
                value[u] = out[i + threads * u];
                weight[u] = extras.gamma[i + threads * u];
            }
            #pragma unroll
            for (int u = 0; u < kSquareLoads; ++u) {
                const float result = __fmul_rn(__fmul_rn(value[u], scale), weight[u]);
                extras.normed[i + threads * u] = result;
                invalid |= !isfinite(result);
            }
        }
        for (; i < width; i += threads) {
            const float result = __fmul_rn(__fmul_rn(out[i], scale), extras.gamma[i]);
            extras.normed[i] = result;
            invalid |= !isfinite(result);
        }
    }
    report(health, invalid);
}

__global__ void inkling_stream_mix(const float *state, const float *query, float *out,
                                   float *health, int first_block, int blocks, int width,
                                   float eps, int stride) {
    mix_one(state + static_cast<size_t>(blockIdx.x) * stride * width, query,
            out + static_cast<size_t>(blockIdx.x) * width, health, first_block, blocks, width, eps,
            MixExtras{});
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
    MixExtras extras[max_mix_jobs];
};

__global__ void inkling_stream_mix_jobs(MixJobs jobs, float *health, int width, float eps) {
    const int j = blockIdx.x;
    mix_one(jobs.state[j], jobs.query[j], jobs.out[j], health, jobs.first_block[j], jobs.blocks[j],
            width, eps, jobs.extras[j]);
}

__global__ void inkling_stream_seal(float *state, const float *next, float *health, int blocks,
                                    int width, float eps, int stride) {
    state += static_cast<size_t>(blockIdx.x) * stride * width;
    next += static_cast<size_t>(blockIdx.x) * width;
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

// Host-ordered variants of the mix and the seal (`Stream::mix`, `Stream::seal` and
// `CpuForward::norm` of the host): every reduction adds in the host's element order on one lane,
// so the results are the host's bits. One block per state; all lanes stage the products and the
// normalized current row and write the outputs, and the states of a prompt keep the card busy.
constexpr int ordered_max_width = 4096;

// Dynamic shared bytes of an ordered mix of `rows` mix rows: the current row, then the staged
// products of one sum per mix row.
size_t ordered_mix_shared_bytes(int width, int rows) {
    return (static_cast<size_t>(width) + ordered_staged_floats(rows)) * sizeof(float);
}
// The mix kernel's static shared floats: the scores and weights of every row and two scalars.
constexpr int ordered_mix_static_floats = 2 * max_rows + 2;
static_assert((ordered_max_width + ordered_staged_floats(max_rows) + ordered_mix_static_floats) *
                      sizeof(float) <=
                  48 * 1024,
              "the widest ordered mix of the most rows must fit the default shared memory");
// `ordered_dots` gives sum k to lane k, so a block needs a lane per sum.
static_assert(max_rows <= threads, "an ordered mix needs a lane for each of its sums");

// out = norm(mix) * gamma when `gamma` is set, else the mix. Without a query the mix is the raw
// current row (the first visit's input). Output rows are `out_stride` floats apart.
__global__ void inkling_stream_mix_ordered(const float *state, const float *query,
                                           const float *gamma, float *out, float *health,
                                           int first_block, int blocks, int width, float mix_eps,
                                           float norm_eps, int stride, int out_stride) {
    state += static_cast<size_t>(blockIdx.x) * stride * width;
    out += static_cast<size_t>(blockIdx.x) * out_stride;
    extern __shared__ float ordered[];
    // The current row, normalized, then the mix in place: each lane reads its own columns of the
    // current row before it writes them.
    float *mixed = ordered;
    float *staged = ordered + width;
    __shared__ float scores[max_rows];
    __shared__ float weights[max_rows];
    __shared__ float scalars[2];
    const int lane = threadIdx.x;
    int invalid = 0;
    if (query) {
        const int rows = blocks - first_block + 1;
        // Sum 0 is the current row's squares; sum k > 0 scores sealed mix row k - 1.
        const float first = ordered_dots(
            [&](int k, int i) {
                return k == 0 ? __fmul_rn(state[i], state[i])
                              : __fmul_rn(mix_row(state, blocks, rows, k - 1, width)[i], query[i]);
            },
            rows, width, staged);
        if (lane == 0) {
            scalars[0] = ordered_rms_scale(first, width, mix_eps);
        } else if (lane < rows) {
            scores[lane - 1] = first;
        }
        __syncthreads();
        const float scale = scalars[0];
        for (int i = lane; i < width; i += threads) {
            const float value = state[i];
            invalid |= !isfinite(value);
            mixed[i] = __fmul_rn(value, scale);
        }
        __syncthreads();
        const float current_score =
            ordered_dots([&](int, int i) { return __fmul_rn(mixed[i], query[i]); }, 1, width,
                         staged);
        if (lane == 0) {
            scores[rows - 1] = current_score;
            const float inverse = __frcp_rn(__fsqrt_rn(static_cast<float>(width)));
            float peak = -INFINITY;
            for (int r = 0; r < rows; ++r) {
                scores[r] = __fmul_rn(scores[r], inverse);
                peak = fmaxf(peak, scores[r]);
            }
            float mass = 0.0f;
            for (int r = 0; r < rows; ++r) {
                const float weight = portable_exp(__fsub_rn(scores[r], peak));
                weights[r] = weight;
                mass = __fadd_rn(mass, weight);
            }
            scalars[1] = mass;
            invalid |= !isfinite(mass) || !isfinite(peak);
        }
        __syncthreads();
        const float mass = scalars[1];
        for (int c = lane; c < width; c += threads) {
            float sum = 0.0f;
            for (int r = 0; r < rows - 1; ++r) {
                sum = __fadd_rn(sum, __fmul_rn(weights[r], mix_row(state, blocks, rows, r, width)[c]));
            }
            sum = __fadd_rn(sum, __fmul_rn(weights[rows - 1], mixed[c]));
            mixed[c] = __fdiv_rn(sum, mass);
        }
    } else {
        for (int c = lane; c < width; c += threads) mixed[c] = state[c];
    }
    __syncthreads();
    if (gamma) {
        const float squares =
            ordered_dots([&](int, int i) { return __fmul_rn(mixed[i], mixed[i]); }, 1, width,
                         staged);
        if (lane == 0) scalars[0] = ordered_rms_scale(squares, width, norm_eps);
        __syncthreads();
        const float scale = scalars[0];
        for (int c = lane; c < width; c += threads) {
            const float value = __fmul_rn(__fmul_rn(mixed[c], scale), gamma[c]);
            out[c] = value;
            invalid |= !isfinite(value);
        }
    } else {
        for (int c = lane; c < width; c += threads) {
            out[c] = mixed[c];
            invalid |= !isfinite(mixed[c]);
        }
    }
    report(health, invalid);
}

__global__ void inkling_stream_seal_ordered(float *state, const float *next, float *health,
                                            int blocks, int width, float eps, int stride) {
    state += static_cast<size_t>(blockIdx.x) * stride * width;
    next += static_cast<size_t>(blockIdx.x) * width;
    __shared__ float shared_scale;
    __shared__ float staged[ordered_staged_floats(1)];
    const int lane = threadIdx.x;
    const float squares =
        ordered_dots([&](int, int i) { return __fmul_rn(state[i], state[i]); }, 1, width, staged);
    if (lane == 0) shared_scale = ordered_rms_scale(squares, width, eps);
    __syncthreads();
    const float scale = shared_scale;
    float *sealed = state + (static_cast<size_t>(blocks) + 1) * width;
    int invalid = !isfinite(scale);
    for (int i = lane; i < width; i += threads) {
        const float value = __fmul_rn(state[i], scale);
        const float replacement = next[i];
        sealed[i] = value;
        state[i] = replacement;
        invalid |= !isfinite(value) || !isfinite(replacement);
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

// `rows` states of `used` rows each, `stride >= used` rows apart; 0 when the shape is invalid.
size_t states_bytes(int rows, int stride, int used, int width) {
    if (rows < 1 || rows > max_states || stride < used) return 0;
    return ((static_cast<size_t>(rows) - 1) * stride + used) * width * sizeof(float);
}
}  // namespace

extern "C" int inference_cuda_inkling_stream_mix(const float *state, const float *query, float *out,
                                                 float *health, int first_block, int blocks,
                                                 int width, float eps, int rows, int stride,
                                                 void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!state || !query || !out || !health || !stream || !aligned(state) || !aligned(query) ||
        !aligned(out) || !aligned(health) || !valid_shape(width, eps) || first_block < 0 ||
        blocks < first_block || blocks > max_blocks) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const size_t row = static_cast<size_t>(width) * sizeof(float);
    const size_t state_bytes = states_bytes(rows, stride, blocks + 1, width);
    const size_t out_bytes = static_cast<size_t>(rows) * row;
    if (state_bytes == 0 || overlaps(out, out_bytes, state, state_bytes) ||
        overlaps(out, out_bytes, query, row) ||
        overlaps(health, sizeof(float), state, state_bytes) ||
        overlaps(health, sizeof(float), query, row) ||
        overlaps(health, sizeof(float), out, out_bytes) ||
        overlaps(state, state_bytes, query, row)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    inkling_stream_mix<<<rows, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        state, query, out, health, first_block, blocks, width, eps, stride);
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

// The decode glue's job lists: `rows` holds `state, query, out, addend, gamma, normed` per job
// (the last three may be null; `gamma` and `normed` are both null or both set), `extents` holds
// `first_block, blocks` and `norm_eps` the epilogue's eps per job.
bool read_glue_jobs(const void *const *rows, const int *extents, const float *norm_eps,
                    int count, MixJobs &jobs) {
    for (int j = 0; j < count; ++j) {
        const void *const *r = rows + 6 * j;
        const auto *state = static_cast<const float *>(r[0]);
        const auto *query = static_cast<const float *>(r[1]);
        auto *out = static_cast<float *>(const_cast<void *>(r[2]));
        const auto *addend = static_cast<const float *>(r[3]);
        const auto *gamma = static_cast<const float *>(r[4]);
        auto *normed = static_cast<float *>(const_cast<void *>(r[5]));
        const int first_block = extents[2 * j], blocks = extents[2 * j + 1];
        const bool norm = gamma != nullptr;
        if (!state || !query || !out || !aligned(state) || !aligned(query) || !aligned(out) ||
            (addend && !aligned(addend)) || (!norm && normed) ||
            (norm && (!normed || !aligned(gamma) || !aligned(normed))) ||
            first_block < 0 || blocks < first_block || blocks > max_blocks ||
            (norm && !valid_shape(1, norm_eps[j]))) {
            return false;
        }
        jobs.state[j] = state;
        jobs.query[j] = query;
        jobs.out[j] = out;
        jobs.first_block[j] = first_block;
        jobs.blocks[j] = blocks;
        jobs.extras[j] = {addend, gamma, normed, norm ? norm_eps[j] : 0.0f};
    }
    return true;
}

// Every row a job writes (its output, its normed row, the current row it adds into, the health
// word) shares no byte with a row any job reads or writes, except the health word every job
// writes and, within one job, the output its norm reads, the normed row itself and the current
// row that is part of the state it mixes.
bool glue_jobs_apart(const MixJobs &jobs, int count, const void *health, size_t row) {
    for (int j = 0; j < count; ++j) {
        const MixExtras &x = jobs.extras[j];
        const void *writes[4] = {jobs.out[j], x.normed, x.addend ? jobs.state[j] : nullptr, health};
        const size_t write_bytes[4] = {row, row, row, sizeof(float)};
        for (int k = 0; k < count; ++k) {
            const MixExtras &y = jobs.extras[k];
            const void *reads[6] = {jobs.state[k], jobs.query[k], y.addend, y.gamma, jobs.out[k],
                                    y.normed};
            const size_t read_bytes[6] = {static_cast<size_t>(jobs.blocks[k] + 1) * row, row, row,
                                          row, row, row};
            for (int w = 0; w < 4; ++w) {
                if (!writes[w]) continue;
                for (int r = 0; r < 6; ++r) {
                    if (!reads[r]) continue;
                    if (k == j && ((w == 0 && r == 4) || (w == 1 && r == 5) || (w == 2 && r == 0)))
                        continue;
                    if (overlaps(writes[w], write_bytes[w], reads[r], read_bytes[r])) return false;
                }
            }
        }
    }
    return true;
}

extern "C" int inference_cuda_inkling_stream_mix_jobs_glue(const void *const *rows,
                                                            const int *extents,
                                                            const float *norm_eps, int count,
                                                            float *health, int width, float eps,
                                                            void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!rows || !extents || !norm_eps || !health || !stream || count < 1 ||
        count > max_mix_jobs || !aligned(health) || !valid_shape(width, eps)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    MixJobs jobs{};
    if (!read_glue_jobs(rows, extents, norm_eps, count, jobs) ||
        !glue_jobs_apart(jobs, count, health, static_cast<size_t>(width) * sizeof(float))) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
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

extern "C" int inference_cuda_inkling_stream_mix_ordered(
    const float *state, const float *query, const float *gamma, float *out, float *health,
    int first_block, int blocks, int width, float mix_eps, float norm_eps, int rows, int stride,
    int out_stride, void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!state || !out || !health || !stream || !aligned(state) || !aligned(query) ||
        !aligned(gamma) || !aligned(out) || !aligned(health) || width > ordered_max_width ||
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
    const size_t shared = ordered_mix_shared_bytes(width, query ? blocks - first_block + 1 : 1);
    inkling_stream_mix_ordered<<<rows, threads, shared, static_cast<cudaStream_t>(stream)>>>(
        state, query, gamma, out, health, first_block, blocks, width, mix_eps, norm_eps, stride,
        out_stride);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}

extern "C" int inference_cuda_inkling_stream_seal_ordered(float *state, const float *next,
                                                          float *health, int blocks, int width,
                                                          float eps, int rows, int stride,
                                                          void *stream, int *out_stage) {
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
    inkling_stream_seal_ordered<<<rows, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        state, next, health, blocks, width, eps, stride);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}
