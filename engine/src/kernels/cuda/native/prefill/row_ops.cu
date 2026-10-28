#include "row_ops.h"
#include "status.h"

#include <cuda_runtime.h>
#include <initializer_list>
#include <stdint.h>

#include "../abi.h"
#include "../device.h"

namespace {
constexpr unsigned kThreads = 256;

struct Region {
    const void *pointer;
    size_t available;
    size_t needed;
    bool written;
};

bool float_bytes(size_t rows, size_t columns, size_t *bytes) {
    if (rows < 1 || rows > 65536 || columns < 1 || columns > (size_t(1) << 22)) {
        return false;
    }
    const size_t count = rows * columns;
    if (count > (size_t(1) << 32)) return false;
    *bytes = count * sizeof(float);
    return true;
}

bool valid_regions(std::initializer_list<Region> regions) {
    for (auto a = regions.begin(); a != regions.end(); ++a) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(a->pointer);
        if (!a->pointer || address % 4 || a->available < a->needed ||
            a->needed > UINTPTR_MAX - address) return false;
        for (auto b = regions.begin(); b != a; ++b) {
            const uintptr_t other = reinterpret_cast<uintptr_t>(b->pointer);
            const bool apart = address + a->needed <= other || other + b->needed <= address;
            if ((a->written || b->written) && !apart) return false;
        }
    }
    return true;
}

bool valid_unary(const inference_cuda_row_unary_args *a) {
    size_t bytes;
    return a && float_bytes(a->rows, a->columns, &bytes) && valid_regions({
        {a->x, a->x_bytes, bytes, false}, {a->out, a->out_bytes, bytes, true},
        {a->flag, a->flag_bytes, 4, true}});
}

bool valid_mul(const inference_cuda_row_mul_args *a) {
    size_t bytes, rhs_bytes;
    return a && (a->rhs_columns == 1 || a->rhs_columns == a->columns) &&
        float_bytes(a->rows, a->columns, &bytes) &&
        float_bytes(a->rows, a->rhs_columns, &rhs_bytes) && valid_regions({
            {a->x, a->x_bytes, bytes, false}, {a->rhs, a->rhs_bytes, rhs_bytes, false},
            {a->out, a->out_bytes, bytes, true}, {a->flag, a->flag_bytes, 4, true}});
}

bool valid_gather(const inference_cuda_row_gather_args *a) {
    size_t source, target;
    return a && float_bytes(a->source_rows, a->columns, &source) &&
        float_bytes(a->rows, a->columns, &target) && valid_regions({
            {a->x, a->x_bytes, source, false}, {a->indices, a->indices_bytes, a->rows * 4, false},
            {a->out, a->out_bytes, target, true}, {a->flag, a->flag_bytes, 4, true}});
}

bool valid_rank_sum(const inference_cuda_row_rank_sum_args *a) {
    size_t source, target;
    return a && a->rank >= 1 && a->rank <= 32 &&
        float_bytes(a->rows, a->columns, &target) &&
        float_bytes(a->rows * a->rank, a->columns, &source) && valid_regions({
            {a->x, a->x_bytes, source, false}, {a->out, a->out_bytes, target, true},
            {a->flag, a->flag_bytes, 4, true}});
}

__device__ void store_value(float value, bool inputs_finite, float *out, unsigned *flag) {
    if (inputs_finite && isfinite(value)) {
        *out = value;
    } else {
        *out = 0.0f;
        atomicOr(flag, 1u);
    }
}

__global__ void sigmoid_k(inference_cuda_row_unary_args a, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float x = a.x[i];
    store_value(1.0f / (1.0f + expf(-x)), isfinite(x), a.out + i, a.flag);
}

__global__ void mul_k(inference_cuda_row_mul_args a, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const size_t row = i / a.columns;
    const size_t column = a.rhs_columns == 1 ? 0 : i % a.columns;
    const float x = a.x[i];
    const float rhs = a.rhs[row * a.rhs_columns + column];
    store_value(__fmul_rn(x, rhs), isfinite(x) && isfinite(rhs), a.out + i, a.flag);
}

__global__ void gather_k(inference_cuda_row_gather_args a, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int32_t source = a.indices[i / a.columns];
    if (source < 0 || size_t(source) >= a.source_rows) {
        a.out[i] = 0.0f;
        atomicOr(a.flag, unsigned(INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION));
        return;
    }
    const float value = a.x[size_t(source) * a.columns + i % a.columns];
    store_value(value, isfinite(value), a.out + i, a.flag);
}

__global__ void rank_sum_k(inference_cuda_row_rank_sum_args a, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const size_t row = i / a.columns;
    const size_t column = i % a.columns;
    float sum = 0.0f;
    bool finite = true;
    for (size_t rank = 0; rank < a.rank; ++rank) {
        const float value = a.x[(row * a.rank + rank) * a.columns + column];
        finite = finite && isfinite(value);
        sum = __fadd_rn(sum, value);
    }
    store_value(sum, finite, a.out + i, a.flag);
}

unsigned blocks(size_t count) { return unsigned((count + kThreads - 1) / kThreads); }

int launch_status(int *out_stage) {
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return int(failure.err);
}
} // namespace

extern "C" int inference_cuda_row_sigmoid_launch(const inference_cuda_row_unary_args *a,
                                                 void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!stream || !valid_unary(a)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const size_t count = a->rows * a->columns;
        sigmoid_k<<<blocks(count), kThreads, 0, (cudaStream_t)stream>>>(*a, count);
        return launch_status(out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_row_mul_launch(const inference_cuda_row_mul_args *a,
                                             void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!stream || !valid_mul(a)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const size_t count = a->rows * a->columns;
        mul_k<<<blocks(count), kThreads, 0, (cudaStream_t)stream>>>(*a, count);
        return launch_status(out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_row_gather_launch(const inference_cuda_row_gather_args *a,
                                                void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!stream || !valid_gather(a)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const size_t count = a->rows * a->columns;
        gather_k<<<blocks(count), kThreads, 0, (cudaStream_t)stream>>>(*a, count);
        return launch_status(out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_row_rank_sum_launch(const inference_cuda_row_rank_sum_args *a,
                                                  void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!stream || !valid_rank_sum(a)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const size_t count = a->rows * a->columns;
        rank_sum_k<<<blocks(count), kThreads, 0, (cudaStream_t)stream>>>(*a, count);
        return launch_status(out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
