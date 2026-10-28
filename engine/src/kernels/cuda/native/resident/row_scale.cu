/*
 * Entry of row_scale on an explicit stream. Glue only: the kernel is below; every exported
 * function has C linkage, takes and returns C types only, and wraps its body in try/catch,
 * like the projector adapter.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <stddef.h>
#include <stdint.h>
#include <cmath>

#include "abi.h"
#include "device.h"
#include "resident/row_scale.h"

namespace {

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }
bool apart(const void *a, size_t ab, const void *b, size_t bb) {
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? ab <= y - x : bb <= x - y;
}

__global__ void row_scale_kernel(float *out, const float *x, float scale, long long total) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = x[i] * scale;
}

}  // namespace

extern "C" int inference_cuda_row_scale_launch(const inference_cuda_row_scale_args *args,
                                               void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const bool finite_scale = std::isfinite(args->scale);
        if (args->out == NULL || args->x == NULL || !finite_scale) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (args->m < 1 || args->m > 65536 || args->n < 1 || args->n > (1 << 22) ||
            (long long)args->m * args->n > (1LL << 32)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (!aligned(args->out, sizeof(float)) || !aligned(args->x, sizeof(float))) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const long long total = (long long)args->m * args->n;
        const size_t bytes = (size_t)total * sizeof(float);
        if (!apart(args->out, bytes, args->x, bytes)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int block = 256;
        const unsigned grid = (unsigned)((total + block - 1) / block);
        Failure failure;
        row_scale_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
            (float *)args->out, (const float *)args->x, args->scale, total);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

namespace {
constexpr unsigned kPow2Threads = 128;

__global__ void row_pow2_guard(inference_cuda_row_pow2_args a) {
    __shared__ float maxima[kPow2Threads];
    __shared__ unsigned invalid[kPow2Threads];
    __shared__ int shift;
    const size_t base = size_t(blockIdx.x) * a.columns;
    float maximum = 0.0f;
    unsigned bad = 0;
    for (size_t col = threadIdx.x; col < a.columns; col += blockDim.x) {
        const float value = a.x[base + col];
        if (isfinite(value)) maximum = fmaxf(maximum, fabsf(value));
        else bad = 1;
    }
    maxima[threadIdx.x] = maximum;
    invalid[threadIdx.x] = bad;
    __syncthreads();
    for (unsigned step = kPow2Threads / 2; step; step /= 2) {
        if (threadIdx.x < step) {
            maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + step]);
            invalid[threadIdx.x] |= invalid[threadIdx.x + step];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const unsigned bits = __float_as_uint(maxima[0]);
        const int ceiling = int((bits >> 23) & 255u) - 127 + ((bits & 0x7fffffu) != 0);
        shift = maxima[0] > 1024.0f ? ceiling - 10 : 0;
        a.scales[blockIdx.x] = __int_as_float(unsigned(127 + shift) << 23);
        if (invalid[0]) atomicOr(a.flag, 1u);
    }
    __syncthreads();
    const float inverse = __int_as_float(unsigned(127 - shift) << 23);
    for (size_t col = threadIdx.x; col < a.columns; col += blockDim.x) {
        a.y[base + col] = invalid[0] ? 0.0f : __fmul_rn(a.x[base + col], inverse);
    }
}

__global__ void row_pow2_restore(inference_cuda_row_pow2_args a) {
    const size_t row = blockIdx.x, base = row * a.columns;
    const float scale = a.scales[row];
    const unsigned bits = __float_as_uint(scale);
    const bool valid = (bits & 0x807fffffu) == 0 &&
        bits >= (127u << 23) && bits <= (245u << 23);
    for (size_t col = threadIdx.x; col < a.columns; col += blockDim.x) {
        const float value = a.x[base + col];
        float result = __fmul_rn(value, scale);
        if (!valid || !isfinite(value) || !isfinite(result)) {
            atomicOr(a.flag, 1u);
            result = 0.0f;
        }
        a.y[base + col] = result;
    }
}

bool valid_pow2(const inference_cuda_row_pow2_args *a) {
    if (!a || a->reserved != 0 || (a->mode != 0 && a->mode != 1) ||
        a->rows < 1 || a->rows > 65536 || a->columns < 1 || a->columns > (1u << 22) ||
        a->columns > SIZE_MAX / a->rows) return false;
    const size_t values = a->rows * a->columns;
    if (values > (1ull << 32) || values > SIZE_MAX / sizeof(float)) return false;
    const size_t bytes = values * sizeof(float);
    const void *ptrs[4] = {a->x, a->y, a->scales, a->flag};
    const size_t have[4] = {a->x_bytes, a->y_bytes, a->scales_bytes, a->flag_bytes};
    const size_t need[4] = {bytes, bytes, a->rows * sizeof(float), sizeof(unsigned)};
    for (unsigned i = 0; i < 4; ++i) {
        if (!ptrs[i] || !aligned(ptrs[i], 4) || have[i] < need[i] ||
            reinterpret_cast<uintptr_t>(ptrs[i]) > UINTPTR_MAX - need[i]) return false;
        for (unsigned j = 0; j < i; ++j)
            if (!apart(ptrs[i], need[i], ptrs[j], need[j])) return false;
    }
    return true;
}
} // namespace

extern "C" int inference_cuda_row_pow2_launch(const inference_cuda_row_pow2_args *args,
                                               void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!stream || !valid_pow2(args)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        if (args->mode == 0)
            row_pow2_guard<<<unsigned(args->rows), kPow2Threads, 0, (cudaStream_t)stream>>>(*args);
        else
            row_pow2_restore<<<unsigned(args->rows), kPow2Threads, 0, (cudaStream_t)stream>>>(*args);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

namespace {
__global__ void bf16_publish_rows(inference_cuda_bf16_publish_args a) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= a.rows * a.columns) return;
    float value = __bfloat162float(__float2bfloat16_rn(a.x[i]));
    if (!isfinite(value)) {
        atomicOr(a.flag, 1u);
        value = 0.0f;
    }
    a.y[i] = value;
}

bool valid_bf16_publish(const inference_cuda_bf16_publish_args *a) {
    if (!a || a->rows < 1 || a->rows > 65536 || a->columns < 1 ||
        a->columns > (1u << 22) || a->columns > SIZE_MAX / a->rows) return false;
    const size_t values = a->rows * a->columns;
    if (values > (1ull << 32) || values > SIZE_MAX / sizeof(float)) return false;
    const size_t bytes = values * sizeof(float);
    const void *ptrs[3] = {a->x, a->y, a->flag};
    const size_t have[3] = {a->x_bytes, a->y_bytes, a->flag_bytes};
    const size_t need[3] = {bytes, bytes, sizeof(unsigned)};
    for (unsigned i = 0; i < 3; ++i) {
        if (!ptrs[i] || !aligned(ptrs[i], 4) || have[i] < need[i] ||
            reinterpret_cast<uintptr_t>(ptrs[i]) > UINTPTR_MAX - need[i]) return false;
        for (unsigned j = 0; j < i; ++j)
            if (!apart(ptrs[i], need[i], ptrs[j], need[j])) return false;
    }
    return true;
}
} // namespace

extern "C" int inference_cuda_bf16_publish_rows(const inference_cuda_bf16_publish_args *args,
                                                void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!stream || !valid_bf16_publish(args)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const unsigned grid = unsigned((args->rows * args->columns + 255) / 256);
        bf16_publish_rows<<<grid, 256, 0, (cudaStream_t)stream>>>(*args);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
