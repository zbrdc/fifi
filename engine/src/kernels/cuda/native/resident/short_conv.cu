// Causal short convolution on the device, with its history resident.
// The contract is resident/short_conv.h. Every arithmetic intrinsic is _rn: the file compiles
// under --use_fast_math, which must not turn a divide or a square root into an approximation.
#include <math.h>
#include <stdint.h>

#include "resident/short_conv.h"

#include "abi.h"
#include "device.h"

namespace {
constexpr int threads = 256;
constexpr int max_width = 8192;
constexpr int min_taps = 2;
constexpr int max_taps = 8;

bool aligned(const void *p) { return (uintptr_t)p % sizeof(float) == 0; }

bool overlaps(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? y - x < a_bytes : x - y < b_bytes;
}

bool valid_shape(int width, int taps) {
    return width >= 1 && width <= max_width && taps >= min_taps && taps <= max_taps;
}

// `out` is exactly `input`, or shares no byte with it: a partial overlap would race.
bool out_matches_input(const void *out, const void *input, size_t bytes) {
    return out == input || !overlaps(out, bytes, input, bytes);
}

// Quiet NaN in the health word of a launch whose inputs or outputs were not finite.
__device__ void report(float *health, int invalid) {
    if (__syncthreads_or(invalid) != 0 && threadIdx.x == 0) health[0] = __int_as_float(0x7fc00000);
}

// Channel c of the host causal convolution, in its summation order: the accumulator starts
// at 0 and each lag adds its rounded product, then the input is added and the norm applied.
__device__ void convolve(const float *input, const float *kernel, float *history, float *out,
                         float *state, int c, int width, int taps, float norm, int &invalid) {
    // `out` may be `input`: x is read before out[c] is written.
    const float x = input[c];
    const float *k = kernel + static_cast<size_t>(c) * taps;
    invalid |= !isfinite(x) || !isfinite(k[taps - 1]);
    float acc = __fadd_rn(0.0f, __fmul_rn(k[taps - 1], x));
    // This thread owns channel c across every lag. Read each old sample before replacing
    // it with the raw sample one position newer (never the convolved output).
    float newer = x;
    for (int lag = 1; lag < taps; ++lag) {
        float &sample = history[static_cast<size_t>(lag - 1) * width + c];
        const float h = sample;
        sample = newer;
        newer = h;
        const float w = k[taps - 1 - lag];
        invalid |= !isfinite(h) || !isfinite(w);
        acc = __fadd_rn(acc, __fmul_rn(w, h));
    }
    const float value = __fmul_rn(__fadd_rn(acc, x), norm);
    out[c] = value;
    invalid |= !isfinite(value);
    // `state`: the host's add_current of the convolved row, one rounded add.
    if (state) {
        const float sum = __fadd_rn(state[c], value);
        state[c] = sum;
        invalid |= !isfinite(sum);
    }
}

__global__ void short_conv_kernel(const float *input, const float *kernel, float *history,
                                    float *out, float *state, float *health, int width, int taps) {
    // The host norm, (taps + 1) as f32 then sqrt then recip, each step rounded to nearest even.
    const float norm = __frcp_rn(__fsqrt_rn(__fadd_rn(static_cast<float>(taps), 1.0f)));
    int invalid = 0;
    // Blocks tile the channels: each one advances its own channels' history in place.
    const int first = blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (int c = first + threadIdx.x; c < width; c += stride) {
        convolve(input, kernel, history, out, state, c, width, taps, norm, invalid);
    }
    report(health, invalid);
}
}  // namespace

extern "C" int inference_cuda_short_conv(const float *input, const float *kernel,
                                           float *history, float *out, float *state,
                                           float *health, int width, int taps, void *stream,
                                           int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!input || !kernel || !history || !out || !health || !stream || !aligned(input) ||
        !aligned(kernel) || !aligned(history) || !aligned(out) || !aligned(health) ||
        !valid_shape(width, taps)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const size_t row = static_cast<size_t>(width) * sizeof(float);
    const size_t kernel_bytes = static_cast<size_t>(taps) * row;
    const size_t history_bytes = (static_cast<size_t>(taps) - 1) * row;
    const size_t health_bytes = sizeof(float);
    // `history`, `kernel` and `health` share no byte with any other range; `out` is `input`
    // or apart from it.
    if (!out_matches_input(out, input, row) ||
        overlaps(history, history_bytes, kernel, kernel_bytes) ||
        overlaps(history, history_bytes, out, row) ||
        overlaps(history, history_bytes, input, row) ||
        overlaps(history, history_bytes, health, health_bytes) ||
        overlaps(kernel, kernel_bytes, out, row) ||
        overlaps(kernel, kernel_bytes, input, row) ||
        overlaps(kernel, kernel_bytes, health, health_bytes) ||
        overlaps(health, health_bytes, out, row) ||
        overlaps(health, health_bytes, input, row)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    // `state` is null or the current row of a stream: apart from every other range.
    if (state &&
        (!aligned(state) || overlaps(state, row, input, row) || overlaps(state, row, out, row) ||
         overlaps(state, row, history, history_bytes) ||
         overlaps(state, row, kernel, kernel_bytes) ||
         overlaps(state, row, health, health_bytes))) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    // One channel per thread: with one block a thread walked up to nine channels and every
    // walk serialized its loads. A channel's accumulation order does not depend on the block
    // that owns it. Every block that saw a non-finite value writes the same NaN.
    const int blocks = (width + threads - 1) / threads;
    short_conv_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        input, kernel, history, out, state, health, width, taps);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}
