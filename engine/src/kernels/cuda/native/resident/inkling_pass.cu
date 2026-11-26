// Row-group glue of the device-resident Inkling prefill pass: the depth stream of every prompt
// row (mix, seal, add) and the K/V preparation of a visit's rows.
// The contract is resident/inkling_pass.h. Every arithmetic intrinsic is _rn and expf is the
// CUDA C library function: the file compiles under --use_fast_math, which must not turn a divide
// or a square root into an approximation.
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"

#include "resident/inkling_pass.h"

namespace {
constexpr int threads = 256;
constexpr int warps = threads / 32;
constexpr int max_blocks = 8;
constexpr int max_width = 8192;
constexpr int max_mix = max_blocks + 1;
constexpr int head = 128;
constexpr int taps = 4;

// Tree sum of one warp, lane 0 holds the result.
__device__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

// Sum of the 256 per-thread partials: warp trees, then the eight warp sums in warp order.
__device__ float block_sum(float value, float *scratch) {
    const int lane = threadIdx.x;
    const int warp = lane >> 5;
    value = warp_sum(value);
    if ((lane & 31) == 0) scratch[warp] = value;
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

// Rust rms_norm_with_scale: scale = (dot(x, x) / len + eps).sqrt().recip().
__device__ float rms_scale(const float *row, int width, float eps, float *scratch, int &invalid) {
    float squares = 0.0f;
    for (int c = (int)threadIdx.x; c < width; c += threads) {
        const float value = row[c];
        invalid |= !isfinite(value);
        squares = __fadd_rn(squares, __fmul_rn(value, value));
    }
    const float total = block_sum(squares, scratch);
    __shared__ float shared;
    if (threadIdx.x == 0) {
        const float variance = __fadd_rn(__fdiv_rn(total, static_cast<float>(width)), eps);
        const float scale = __frcp_rn(__fsqrt_rn(variance));
        invalid |= !isfinite(total) || !isfinite(variance) || !isfinite(scale);
        shared = scale;
    }
    __syncthreads();
    const float scale = shared;
    __syncthreads();
    return scale;
}

__device__ void report(float *health, int row, int invalid) {
    if (__syncthreads_or(invalid) != 0 && threadIdx.x == 0) health[row] = __int_as_float(0x7fc00000);
}

// `Stream::mix(first, query)` of every row: one block per prompt row. Mix row j is sealed row
// (blocks - 1 - j) for j < blocks - first, then the normalized current row. The scores are
// warp reductions, the weights and their mass follow the host's row order, and each channel
// accumulates its weighted rows in that order before the divide by the mass.
__global__ void inkling_pass_mix(const float *sealed, const float *current, const float *query,
                                 float *out, float *health, int rows, int blocks, int first,
                                 int width, int out_stride, float eps) {
    __shared__ float scratch[warps];
    __shared__ float scores[max_mix];
    __shared__ float weights[max_mix];
    __shared__ float shared[2];
    const int r = blockIdx.x;
    const int lane = threadIdx.x;
    const int warp = lane >> 5;
    const int sub = lane & 31;
    const int count = blocks - first + 1;
    const float *rows_ptr[max_mix];
    for (int j = 0; j < count; ++j) {
        rows_ptr[j] = j == count - 1
                          ? current + static_cast<size_t>(r) * width
                          : sealed + (static_cast<size_t>(blocks - 1 - j) * rows + r) * width;
    }
    int invalid = 0;
    const float current_scale = rms_scale(rows_ptr[count - 1], width, eps, scratch, invalid);
    const float scale = __frcp_rn(__fsqrt_rn(static_cast<float>(width)));
    for (int j = warp; j < count; j += warps) {
        const float *row = rows_ptr[j];
        float partial = 0.0f;
        for (int c = sub; c < width; c += 32) {
            float value = row[c];
            if (j == count - 1) value = __fmul_rn(value, current_scale);
            invalid |= !isfinite(value) || !isfinite(query[c]);
            partial = __fadd_rn(partial, __fmul_rn(value, query[c]));
        }
        partial = warp_sum(partial);
        if (sub == 0) scores[j] = __fmul_rn(partial, scale);
    }
    __syncthreads();
    if (lane == 0) {
        float peak = -INFINITY;
        for (int j = 0; j < count; ++j) peak = fmaxf(peak, scores[j]);
        float mass = 0.0f;
        for (int j = 0; j < count; ++j) {
            const float weight = expf(scores[j] - peak);
            weights[j] = weight;
            mass = __fadd_rn(mass, weight);
        }
        invalid |= !isfinite(peak) || !isfinite(mass) || mass <= 0.0f;
        shared[0] = mass;
    }
    __syncthreads();
    const float mass = shared[0];
    float *result = out + static_cast<size_t>(r) * out_stride;
    for (int c = lane; c < width; c += threads) {
        float accumulator = 0.0f;
        for (int j = 0; j < count; ++j) {
            float value = rows_ptr[j][c];
            if (j == count - 1) value = __fmul_rn(value, current_scale);
            accumulator = __fadd_rn(accumulator, __fmul_rn(weights[j], value));
        }
        const float value = __fdiv_rn(accumulator, mass);
        result[c] = value;
        invalid |= !isfinite(value);
    }
    report(health, r, invalid);
}

// `Stream::seal(next)` of every row: the normalized current row becomes sealed row `blocks`
// and the current row takes `next`.
__global__ void inkling_pass_seal(float *sealed, float *current, const float *next,
                                  float *health, int rows, int blocks, int width, float eps) {
    __shared__ float scratch[warps];
    const int r = blockIdx.x;
    const int lane = threadIdx.x;
    float *row = current + static_cast<size_t>(r) * width;
    int invalid = 0;
    const float scale = rms_scale(row, width, eps, scratch, invalid);
    float *target = sealed + (static_cast<size_t>(blocks) * rows + r) * width;
    const float *source = next + static_cast<size_t>(r) * width;
    for (int c = lane; c < width; c += threads) {
        const float normalized = __fmul_rn(row[c], scale);
        const float value = source[c];
        target[c] = normalized;
        invalid |= !isfinite(normalized) || !isfinite(value);
    }
    __syncthreads();  // every lane has read the old current row before it is replaced
    for (int c = lane; c < width; c += threads) row[c] = source[c];
    report(health, r, invalid);
}

// `Stream::add_current(row)` of every row.
__global__ void inkling_pass_add(float *current, const float *row, float *health, int rows,
                                 int width) {
    const int r = blockIdx.x;
    const int lane = threadIdx.x;
    float *target = current + static_cast<size_t>(r) * width;
    const float *source = row + static_cast<size_t>(r) * width;
    int invalid = 0;
    for (int c = lane; c < width; c += threads) {
        const float value = __fadd_rn(target[c], source[c]);
        target[c] = value;
        invalid |= !isfinite(value);
    }
    report(health, r, invalid);
}

struct PrepArgs {
    const float *raw;
    float *side;
    const float *history_k, *history_v, *kernel_k, *kernel_v, *dynamic;
    uint32_t *status;
    int rows, raw_stride, side_stride, key_at, value_at, dynamic_stride, kv_heads;
    float qk_limit, eps;
    bool dynamic_taps;
};

// `prepare_attention_row` of every row for the K and V planes. One block of one head per row
// and plane. The raw projections of the whole visit stay immutable (`raw`: row r holds its raw
// K at row offset 0 and its raw V at `channels`), because the rows after r read them as
// convolution history; the prepared rows go to the attention sidecar. The convolution keeps
// the host's lag order; the history plane holds the taps - 1 raw rows before the group, oldest
// first.
__global__ void inkling_pass_prep(PrepArgs a) {
    __shared__ float warp_sums[head / 32];
    __shared__ float shared;
    const int r = blockIdx.x;
    const bool value_plane = blockIdx.y >= a.kv_heads;
    const int channel = (blockIdx.y - (value_plane ? a.kv_heads : 0)) * head + threadIdx.x;
    const int channels = a.kv_heads * head;
    const int lane = threadIdx.x;
    const int raw_at = value_plane ? channels : 0;
    const float *raw = a.raw + static_cast<size_t>(r) * a.raw_stride + raw_at + channel;
    float *out = a.side + static_cast<size_t>(r) * a.side_stride +
                 (value_plane ? a.value_at : a.key_at) + channel;
    const float *kernel = value_plane ? a.kernel_v : a.kernel_k;
    const float *history = value_plane ? a.history_v : a.history_k;
    const float *dynamic =
        a.dynamic_taps ? a.dynamic + static_cast<size_t>(r) * a.dynamic_stride +
                             (value_plane ? a.kv_heads * taps : 0) + (channel / head) * taps
                       : nullptr;
    float sum = 0.0f;
    for (int lag = 0; lag < taps; ++lag) {
        const int source = r - lag;
        const float x = source >= 0 ? raw[static_cast<ptrdiff_t>(source - r) * a.raw_stride]
                                    : history[(taps - 1 + source) * channels + channel];
        const float weight =
            dynamic ? __fadd_rn(kernel[lag * channels + channel], dynamic[lag])
                    : kernel[channel * taps + taps - 1 - lag];
        sum = __fadd_rn(sum, __fmul_rn(weight, x));
    }
    // The host norm counts the dynamic taps as a second kernel: (2 * taps + 1) with them.
    const float norm = __frcp_rn(__fsqrt_rn(
        static_cast<float>(a.dynamic_taps ? 2 * taps + 1 : taps + 1)));
    const float convolved = __fmul_rn(__fadd_rn(sum, raw[0]), norm);
    const int warp = lane >> 5;
    float squares = warp_sum(__fmul_rn(convolved, convolved));
    if ((lane & 31) == 0) warp_sums[warp] = squares;
    __syncthreads();
    int invalid = !isfinite(convolved);
    if (lane == 0) {
        float total = 0.0f;
        for (int i = 0; i < head / 32; ++i) total = __fadd_rn(total, warp_sums[i]);
        float scale;
        if (value_plane) {
            scale = __frcp_rn(__fsqrt_rn(__fadd_rn(__fdiv_rn(total, static_cast<float>(head)), a.eps)));
        } else {
            scale = fminf(__fdiv_rn(a.qk_limit, __fadd_rn(__fsqrt_rn(total), 1.0e-6f)), 1.0f);
        }
        invalid |= !isfinite(total) || !isfinite(scale);
        shared = scale;
    }
    __syncthreads();
    const float result = __fmul_rn(convolved, shared);
    out[0] = result;
    invalid |= !isfinite(result);
    // Prep ORs into the status plane the attention tile checks, like the query clip.
    if (__syncthreads_or(invalid) != 0 && lane == 0) atomicOr(a.status + r, 1u);
}

// `control_input_with_scales` of every row: gate = [norm(input) | norm(message)] with `eps`.
__global__ void inkling_pass_gate(const float *in, const float *msg, float *gate, float *health,
                                  int rows, int width, int in_stride, int msg_stride,
                                  int gate_stride, float eps) {
    __shared__ float scratch[warps];
    __shared__ float shared[2];
    const int r = blockIdx.x;
    const int lane = threadIdx.x;
    const float *input = in + static_cast<size_t>(r) * in_stride;
    const float *message = msg + static_cast<size_t>(r) * msg_stride;
    float *out = gate + static_cast<size_t>(r) * gate_stride;
    int invalid = 0;
    const float input_scale = rms_scale(input, width, eps, scratch, invalid);
    const float message_scale = rms_scale(message, width, eps, scratch, invalid);
    if (lane == 0) {
        shared[0] = input_scale;
        shared[1] = message_scale;
    }
    __syncthreads();
    for (int c = lane; c < width; c += threads) {
        const float a = __fmul_rn(input[c], shared[0]);
        const float b = __fmul_rn(message[c], shared[1]);
        out[c] = a;
        out[width + c] = b;
        invalid |= !isfinite(a) || !isfinite(b);
    }
    report(health, r, invalid);
}

// `apply_control` then `norm` of every row: x = (in - tanh(forget[c]) * in) + tanh(gain[c]) *
// message with `control` (gain and forget are rows of `width`), else x = in; out =
// x * rms_scale(x, eps) * gamma. `out` may be `in`.
__global__ void inkling_pass_apply(const float *in, const float *msg, const float *gain,
                                   const float *forget, const float *gamma, float *out,
                                   float *health, int rows, int width, int stride, float eps,
                                   int control) {
    __shared__ float scratch[warps];
    __shared__ float scale_shared;
    const int r = blockIdx.x;
    const int lane = threadIdx.x;
    float *row = out + static_cast<size_t>(r) * stride;
    int invalid = 0;
    if (control) {
        const float *input = in + static_cast<size_t>(r) * stride;
        const float *message = msg + static_cast<size_t>(r) * stride;
        const float *g = gain + static_cast<size_t>(r) * stride;
        const float *f = forget + static_cast<size_t>(r) * stride;
        for (int c = lane; c < width; c += threads) {
            const float tg = tanhf(g[c]), tf = tanhf(f[c]);
            const float kept = __fsub_rn(input[c], __fmul_rn(tf, input[c]));
            const float x = __fadd_rn(kept, __fmul_rn(tg, message[c]));
            row[c] = x;
            invalid |= !isfinite(x) || !isfinite(g[c]) || !isfinite(f[c]);
        }
    } else if (row != in + static_cast<size_t>(r) * stride) {
        const float *input = in + static_cast<size_t>(r) * stride;
        for (int c = lane; c < width; c += threads) row[c] = input[c];
    }
    __syncthreads();
    const float scale = rms_scale(row, width, eps, scratch, invalid);
    if (lane == 0) scale_shared = scale;
    __syncthreads();
    const float s = scale_shared;
    for (int c = lane; c < width; c += threads) {
        const float value = __fmul_rn(__fmul_rn(row[c], s), gamma[c]);
        row[c] = value;
        invalid |= !isfinite(value);
    }
    report(health, r, invalid);
}

// The last three rows of (the `past` valid rows of `old`, oldest first, then the `rows` rows of
// `src`), oldest first, into `out`. One block per (channel tile, output row).
__global__ void inkling_pass_tails(const float *src, int rows, int stride, int width, int past,
                                   const float *old, float *out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y;
    if (c >= width) return;
    const int j = rows + past - taps + 1 + i;
    const float value =
        j < past ? old[(taps - 1 - past + j) * width + c] : src[static_cast<size_t>(j - past) * stride + c];
    out[i * width + c] = value;
}

// Position words of one visit's attention input sidecar: row r gets `first + r` at `pos_at`.
__global__ void inkling_pass_positions(uint32_t *side, int rows, int stride, int pos_at,
                                       int first) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) side[static_cast<size_t>(r) * stride + pos_at] = static_cast<uint32_t>(first + r);
}

__global__ void inkling_pass_zero(uint32_t *words, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) words[i] = 0u;
}
}  // namespace

extern "C" int inference_cuda_inkling_pass_mix(const float *sealed, const float *current,
                                               const float *query, float *out, float *health,
                                               int rows, int blocks, int first, int width,
                                               int out_stride, float eps, void *stream,
                                               int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!sealed || !current || !query || !out || !health || !stream || rows < 1 || blocks < 0 ||
        blocks >= max_blocks || first < 0 || first > blocks || width < 1 || width > max_width ||
        out_stride < width || out_stride > max_width || !isfinite(eps) || eps <= 0.0f)
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    inkling_pass_mix<<<rows, threads, 0, (cudaStream_t)stream>>>(sealed, current, query, out,
                                                                 health, rows, blocks, first,
                                                                 width, out_stride, eps);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

extern "C" int inference_cuda_inkling_pass_seal(float *sealed, float *current, const float *next,
                                                float *health, int rows, int blocks, int width,
                                                float eps, void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!sealed || !current || !next || !health || !stream || rows < 1 || blocks < 0 ||
        blocks >= max_blocks || width < 1 || width > max_width || !isfinite(eps) || eps <= 0.0f)
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    inkling_pass_seal<<<rows, threads, 0, (cudaStream_t)stream>>>(sealed, current, next, health,
                                                                  rows, blocks, width, eps);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

extern "C" int inference_cuda_inkling_pass_add(float *current, const float *row, float *health,
                                               int rows, int width, void *stream,
                                               int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!current || !row || !health || !stream || rows < 1 || width < 1 || width > max_width)
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    inkling_pass_add<<<rows, threads, 0, (cudaStream_t)stream>>>(current, row, health, rows,
                                                                 width);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

extern "C" int inference_cuda_inkling_pass_prep(void *const *p, int rows, int raw_stride,
                                                int side_stride, int key_at, int value_at,
                                                int dynamic_stride, int kv_heads, float qk_limit,
                                                float eps, int dynamic_taps, void *stream,
                                                int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    // p: raw, side, history_k, history_v, kernel_k, kernel_v, dynamic (may be null), status.
    const int channels = kv_heads * head;
    if (!p || !p[0] || !p[1] || !p[2] || !p[3] || !p[4] || !p[5] || !p[7] || !stream ||
        rows < 1 || kv_heads < 1 || channels > max_width || raw_stride < 2 * channels ||
        side_stride < channels || key_at < 0 || value_at < 0 ||
        key_at + channels > side_stride || value_at + channels > side_stride ||
        side_stride > 65536 || !isfinite(qk_limit) || qk_limit <= 0.0f || !isfinite(eps) ||
        eps <= 0.0f)
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    PrepArgs args;
    args.raw = static_cast<const float *>(p[0]);
    args.side = static_cast<float *>(p[1]);
    args.history_k = static_cast<const float *>(p[2]);
    args.history_v = static_cast<const float *>(p[3]);
    args.kernel_k = static_cast<const float *>(p[4]);
    args.kernel_v = static_cast<const float *>(p[5]);
    args.dynamic = static_cast<const float *>(p[6]);
    args.status = static_cast<uint32_t *>(p[7]);
    args.rows = rows;
    args.raw_stride = raw_stride;
    args.side_stride = side_stride;
    args.key_at = key_at;
    args.value_at = value_at;
    args.dynamic_stride = dynamic_stride;
    args.kv_heads = kv_heads;
    args.qk_limit = qk_limit;
    args.eps = eps;
    args.dynamic_taps = dynamic_taps != 0;
    if (args.dynamic_taps && (!args.dynamic || dynamic_stride < kv_heads * taps))
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    inkling_pass_prep<<<dim3(rows, 2 * kv_heads), head, 0, (cudaStream_t)stream>>>(args);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

extern "C" int inference_cuda_inkling_pass_gate(const float *in, const float *msg, float *gate,
                                                float *health, int rows, int width,
                                                int in_stride, int msg_stride, int gate_stride,
                                                float eps, void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!in || !msg || !gate || !health || !stream || rows < 1 || width < 1 || width > max_width ||
        in_stride < width || msg_stride < width || gate_stride < 2 * width ||
        gate_stride > max_width || !isfinite(eps) || eps <= 0.0f)
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    inkling_pass_gate<<<rows, threads, 0, (cudaStream_t)stream>>>(in, msg, gate, health, rows,
                                                                 width, in_stride, msg_stride,
                                                                 gate_stride, eps);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

extern "C" int inference_cuda_inkling_pass_apply(const float *in, const float *msg,
                                                 const float *gain, const float *forget,
                                                 const float *gamma, float *out, float *health,
                                                 int rows, int width, int stride, float eps,
                                                 int control, void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!in || !gamma || !out || !health || !stream || rows < 1 || width < 1 || width > max_width ||
        stride < width || stride > max_width || !isfinite(eps) || eps <= 0.0f ||
        (control && (!msg || !gain || !forget)))
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    inkling_pass_apply<<<rows, threads, 0, (cudaStream_t)stream>>>(
        in, msg, gain, forget, gamma, out, health, rows, width, stride, eps, control != 0);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

extern "C" int inference_cuda_inkling_pass_tails(const float *src, int rows, int stride,
                                                 int width, int past, const float *old,
                                                 float *out, void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!src || !old || !out || !stream || rows < 1 || stride < width || width < 1 ||
        width > max_width || past < 0 || past >= taps)
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    inkling_pass_tails<<<dim3((width + threads - 1) / threads, taps - 1), threads, 0,
                          (cudaStream_t)stream>>>(src, rows, stride, width, past, old, out);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

extern "C" int inference_cuda_inkling_pass_positions(uint32_t *side, int rows, int stride,
                                                     int pos_at, int first, void *stream,
                                                     int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!side || !stream || rows < 1 || stride < 1 || pos_at < 0 || pos_at >= stride || first < 0)
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    inkling_pass_positions<<<(rows + threads - 1) / threads, threads, 0, (cudaStream_t)stream>>>(
        side, rows, stride, pos_at, first);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

extern "C" int inference_cuda_inkling_pass_zero(uint32_t *words, int count, void *stream,
                                                int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!words || !stream || count < 1 || count > (1 << 20))
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    inkling_pass_zero<<<(count + threads - 1) / threads, threads, 0, (cudaStream_t)stream>>>(
        words, count);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}
