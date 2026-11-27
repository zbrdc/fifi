#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

#include "abi.h"
#include "device.h"
#include "resident/ordered_sum.cuh"

namespace {
constexpr int kWidth = 128, kTaps = 4;
// Largest row the Rust wrappers pass to inkling_weighted_norm.
constexpr int kMaxNormWidth = 4096;

// The weight rows and the K/V conv histories are the caller's ranges: nothing is staged. The
// kernel shifts `past_k` / `past_v` in place.
struct Args {
    float *q, *relative, *raw_k, *raw_v, *k, *v, *sinks, *gates;
    const float *q_scale, *rel_scale, *kernel_k, *kernel_v;
    float *past_k, *past_v;
    uint32_t *status;
    int queries, kv, dims, history;
    float qk_limit, rel_limit, eps;
};

// One block per head and plane. Use the host's reduction and tap order; these
// short reductions run once per head, after parallel channel preparation.
__global__ void inkling_local_prep(Args a) {
    __shared__ float squares[kWidth], scale;
    const int d = threadIdx.x;
    int head = blockIdx.x;
    const int kv_words = a.kv * kWidth;
    const float *q_scale = a.q_scale;
    const float *rel_scale = a.rel_scale;
    float *output;
    float x = 0.f;
    bool valid = true;
    int width = kWidth;
    float limit = a.qk_limit;
    bool rms = false;
    if (head < a.queries) {
        const int at = head * kWidth + d;
        output = a.q + at;
        x = *output * q_scale[at];
        valid = isfinite(a.gates[at]) && (d != 0 || isfinite(a.sinks[head]));
    } else if ((head -= a.queries) < a.queries) {
        width = a.dims;
        limit = a.rel_limit;
        const int at = head * width + d;
        output = a.relative + head * width + min(d, width - 1);
        if (d < width) x = *output * rel_scale[at];
    } else {
        head -= a.queries;
        rms = head >= a.kv;
        const int plane = rms ? 1 : 0;
        const int at = (head - plane * a.kv) * kWidth + d;
        output = (rms ? a.v : a.k) + at;
        const float raw = (rms ? a.raw_v : a.raw_k)[at];
        const float *w = (plane ? a.kernel_v : a.kernel_k) + at * kTaps;
        float *past = plane ? a.past_v : a.past_k;
        float conv = __fmul_rn(w[kTaps - 1], raw);
        for (int lag = 1; lag <= a.history; ++lag) {
            conv = __fadd_rn(conv, __fmul_rn(w[kTaps - 1 - lag], past[(lag - 1) * kv_words + at]));
        }
        // Each thread owns one channel of the mirror. Read every old tap
        // before shifting the history for that channel.
        for (int lag = min(a.history, kTaps - 2); lag > 0; --lag) {
            past[lag * kv_words + at] = past[(lag - 1) * kv_words + at];
        }
        past[at] = raw;
        x = __fmul_rn(__fadd_rn(conv, raw), 1.f / sqrtf(float(kTaps + 1)));
    }
    squares[d] = x * x;
    __syncthreads();
    if (d == 0) {
        float sum = 0.f;
        for (int i = 0; i < width; ++i) sum += squares[i];
        scale = rms ? 1.f / sqrtf(sum / width + a.eps)
                    : fminf(1.f, limit / (sqrtf(sum) + 1.e-6f));
        valid &= isfinite(sum);
    }
    __syncthreads();
    const float result = x * scale;
    const int invalid = __syncthreads_or(!valid || !isfinite(x) || !isfinite(result));
    if (d < width) *output = result;
    // Only one block writes status; per-head prep flags are distinct words.
    if (d == 0) {
        if (blockIdx.x == 0) *a.status = 0;
        a.status[1 + blockIdx.x] = invalid;
    }
}
// Static output convolution. Raw projected rows are immutable and become the
// next tile's history. Initial history is right-aligned, oldest first.
__global__ void inkling_output_conv(const float *raw, const float *past, const float *weight,
                                    float *result, uint32_t *status, int rows, int width,
                                    int history, float scale) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    bool bad = false;
    if (channel < width) {
        const int at = row * width + channel;
        const float *w = weight + channel * kTaps;
        float sum = 0.f;
        for (int lag = 0; lag <= min(kTaps - 1, row + history); ++lag) {
            const int source = row - lag;
            const float value = source >= 0 ? raw[source * width + channel]
                                            : past[(kTaps - 1 + source) * width + channel];
            sum = __fadd_rn(sum, __fmul_rn(w[kTaps - 1 - lag], value));
        }
        const float value = __fmul_rn(__fadd_rn(sum, raw[at]), scale);
        result[at] = value;
        bad = !isfinite(value);
    }
    const int invalid = __syncthreads_or(bad);
    if (threadIdx.x == 0 && invalid) atomicOr(status + row, 1u);
}
struct KvRowsArgs {
    const float *raw, *past, *weights, *coefficients;
    float *out;
    uint32_t *status;
    int rows, kv, history, dynamic;
    int raw_stride, raw_k, raw_v, out_stride, out_k, out_v;
    float limit, eps;
};

// One block per row, plane and head of 128 channels. Raw K and V rows of the tile and the
// visit's raw history (right-aligned, oldest first) feed the causal convolution in the host's
// tap order; K heads are then L2-clipped and V heads RMS-normalized with the host's sequential
// sum. Rows write their own slots, so the raw rows are never overwritten.
__global__ void inkling_kv_prep_rows(KvRowsArgs a) {
    __shared__ float squares[kWidth], scale;
    const int d = threadIdx.x, row = blockIdx.y;
    const int heads = a.kv / kWidth;
    const int plane = blockIdx.x / heads, head = blockIdx.x % heads;
    const int at = head * kWidth + d;
    const int raw_at = plane ? a.raw_v : a.raw_k;
    const float *weights = a.weights + plane * kTaps * a.kv;
    float conv = 0.f;
    for (int lag = 0; lag <= min(kTaps - 1, row + a.history); ++lag) {
        const int source = row - lag;
        const float x = source >= 0
            ? a.raw[(size_t)source * a.raw_stride + raw_at + at]
            : a.past[(plane * (kTaps - 1) + kTaps - 1 + source) * a.kv + at];
        const float w = a.dynamic
            ? __fadd_rn(weights[lag * a.kv + at],
                        a.coefficients[((size_t)row * 2 + plane) * heads * kTaps + head * kTaps + lag])
            : weights[at * kTaps + kTaps - 1 - lag];
        conv = __fadd_rn(conv, __fmul_rn(w, x));
    }
    const float raw = a.raw[(size_t)row * a.raw_stride + raw_at + at];
    // Explicitly rounded steps, so a fast-math build cannot change the host's order.
    const float norm = __frcp_rn(__fsqrt_rn(float(a.dynamic ? 2 * kTaps + 1 : kTaps + 1)));
    const float x = __fmul_rn(__fadd_rn(conv, raw), norm);
    squares[d] = __fmul_rn(x, x);
    __syncthreads();
    bool valid = true;
    if (d == 0) {
        float sum = 0.f;
        for (int i = 0; i < kWidth; ++i) sum = __fadd_rn(sum, squares[i]);
        scale = plane ? __frcp_rn(__fsqrt_rn(__fadd_rn(__fdiv_rn(sum, float(kWidth)), a.eps)))
                      : fminf(1.f, __fdiv_rn(a.limit, __fadd_rn(__fsqrt_rn(sum), 1.e-6f)));
        valid = isfinite(sum);
    }
    __syncthreads();
    const float result = __fmul_rn(x, scale);
    a.out[(size_t)row * a.out_stride + (plane ? a.out_v : a.out_k) + at] = result;
    const int invalid = __syncthreads_or(!valid || !isfinite(x) || !isfinite(result));
    if (d == 0 && invalid) atomicOr(a.status + row, 1u);
}

// The squares in element order from -0.0. `ordered_dot` issues sixteen loads and products ahead
// of their adds, so the adding lane does not wait on a shared-memory load per add.
__device__ float ordered_row_scale(const float *x, int width, float eps) {
    const float squares = ordered_dot(x, x, width);
    const float root = __fsqrt_rn(__fadd_rn(__fdiv_rn(squares, float(width)), eps));
    return isfinite(root) ? __frcp_rn(root) : __int_as_float(0x7fc00000);
}

// The ordered scale sums every square in scalar order, so one lane adds and the chain of
// dependent adds is the floor. The other lanes bring the row into shared memory first: one
// lane loading from global memory serializes a memory latency behind every add. The values,
// their order and every rounding are those of the sum from global memory.
__global__ void inkling_weighted_norm(const float *input, const float *gamma,
    float *output, float *status, int width, float eps) {
    __shared__ float staged[kMaxNormWidth];
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const float *x = input + row * width;
    for (int i = lane; i < width; i += blockDim.x) staged[i] = x[i];
    __syncthreads();
    __shared__ float scale;
    if (lane == 0) scale = ordered_row_scale(staged, width, eps);
    __syncthreads();
    bool bad = !isfinite(scale);
    for (int i = lane; i < width; i += blockDim.x) {
        const float value = __fmul_rn(__fmul_rn(x[i], scale), gamma[i]);
        output[row * width + i] = value;
        bad |= !isfinite(value);
    }
    const int failed = __syncthreads_or(bad);
    if (lane == 0) status[row] = failed ? 1.f : 0.f;
}

// Parallel form of the weighted norm for the launch-only chain
// (`inference_cuda_inkling_weighted_norm_reduced`, selected by FIFI_INKLING_CUDA_DECODE_GLUE).
// A lane keeps its kNormKeep samples of the row in registers, adds their squares in row order,
// the warp adds its lanes in tree order and every lane adds the eight warp partials in warp
// order. The sum is therefore not the host's ordered sum: the scale agrees with the host within
// floating-point error, as the depth stream kernels do. The row is read once; the one-block
// kernel pays one memory latency per pass instead of a dependent add per element.
constexpr int kNormThreads = 256;
constexpr int kNormWarps = kNormThreads / 32;
constexpr int kNormKeep = kMaxNormWidth / kNormThreads;
static_assert(kNormKeep * kNormThreads == kMaxNormWidth, "the register tile covers the widest row");

__global__ void inkling_weighted_norm_reduced(const float *input, const float *gamma,
    float *output, float *status, int width, float eps) {
    __shared__ float partials[kNormWarps];
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const float *x = input + row * width;
    float value[kNormKeep];
    int invalid = 0;
    float squares = 0.f;
#pragma unroll
    for (int u = 0; u < kNormKeep; ++u) {
        const int i = lane + u * kNormThreads;
        value[u] = i < width ? x[i] : 0.f;
        invalid |= i < width && !isfinite(value[u]);
        squares = __fadd_rn(squares, __fmul_rn(value[u], value[u]));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        squares = __fadd_rn(squares, __shfl_xor_sync(0xffffffffu, squares, offset));
    }
    if ((lane & 31) == 0) partials[lane >> 5] = squares;
    __syncthreads();
    // The same sum in every lane: the scale is one value for the whole block.
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kNormWarps; ++w) total = __fadd_rn(total, partials[w]);
    const float root = __fsqrt_rn(__fadd_rn(__fdiv_rn(total, float(width)), eps));
    invalid |= !isfinite(total) || !isfinite(root);
    const float scale = isfinite(root) ? __frcp_rn(root) : __int_as_float(0x7fc00000);
#pragma unroll
    for (int u = 0; u < kNormKeep; ++u) {
        const int i = lane + u * kNormThreads;
        if (i >= width) continue;
        const float result = __fmul_rn(__fmul_rn(value[u], scale), gamma[i]);
        output[row * width + i] = result;
        invalid |= !isfinite(result);
    }
    const int failed = __syncthreads_or(invalid);
    if (lane == 0) status[row] = failed ? 1.f : 0.f;
}
} // namespace

extern "C" int inference_cuda_inkling_weighted_norm(void *const *p, int rows, int width,
    float eps, void *stream, int *out_stage) {
    inkling_weighted_norm<<<rows, 128, 0, (cudaStream_t)stream>>>(
        static_cast<const float *>(p[0]), static_cast<const float *>(p[1]),
        static_cast<float *>(p[2]), static_cast<float *>(p[3]), width, eps);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

extern "C" int inference_cuda_inkling_weighted_norm_reduced(void *const *p, int rows, int width,
    float eps, void *stream, int *out_stage) {
    inkling_weighted_norm_reduced<<<rows, kNormThreads, 0, (cudaStream_t)stream>>>(
        static_cast<const float *>(p[0]), static_cast<const float *>(p[1]),
        static_cast<float *>(p[2]), static_cast<float *>(p[3]), width, eps);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

extern "C" int inference_cuda_inkling_output_conv(void *const *p, int rows, int width,
    int history, float scale, void *stream, int *out_stage) {
    constexpr int block = 128;
    inkling_output_conv<<<dim3((width + block - 1) / block, rows), block, 0, (cudaStream_t)stream>>>(
        static_cast<const float *>(p[0]), static_cast<const float *>(p[1]),
        static_cast<const float *>(p[2]), static_cast<float *>(p[3]),
        static_cast<uint32_t *>(p[4]), rows, width, history, scale);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

// p: raw rows, history, taps (K then V weights, then per-row coefficients when dynamic), out,
// status. shape: rows, kv, history, dynamic, raw stride, raw K, raw V, out stride, out K, out V
// (floats).
extern "C" int inference_cuda_inkling_kv_prep_rows(void *const *p, const int *shape,
    float limit, float eps, void *stream, int *out_stage) {
    const float *taps = static_cast<const float *>(p[2]);
    KvRowsArgs args{static_cast<const float *>(p[0]), static_cast<const float *>(p[1]),
                    taps, taps + 2 * kTaps * shape[1],
                    static_cast<float *>(p[3]), static_cast<uint32_t *>(p[4]),
                    shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6],
                    shape[7], shape[8], shape[9], limit, eps};
    inkling_kv_prep_rows<<<dim3(2 * (args.kv / kWidth), args.rows), kWidth, 0,
                           (cudaStream_t)stream>>>(args);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

extern "C" int inference_cuda_inkling_local_prep(void *const *p, int queries, int kv,
    int dims, int history, float qk_limit, float rel_limit, float eps, void *stream,
    int *out_stage) {
    Args args{static_cast<float *>(p[0]), static_cast<float *>(p[1]),
              static_cast<float *>(p[2]), static_cast<float *>(p[3]),
              static_cast<float *>(p[4]), static_cast<float *>(p[5]),
              static_cast<float *>(p[6]), static_cast<float *>(p[7]),
              static_cast<const float *>(p[8]), static_cast<const float *>(p[9]),
              static_cast<const float *>(p[10]), static_cast<const float *>(p[11]),
              static_cast<float *>(p[12]), static_cast<float *>(p[13]),
              static_cast<uint32_t *>(p[14]),
              queries, kv, dims, history, qk_limit, rel_limit, eps};
    inkling_local_prep<<<2 * (queries + kv), kWidth, 0, (cudaStream_t)stream>>>(args);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}
