#include "nemotron_gemv.h"
#include "device.h"

#include <cmath>
#include <cstring>
#include <stdint.h>

namespace {

struct Needs {
    size_t weight;
    size_t x;
    size_t y;
};

bool checked_needs(size_t rows, size_t columns, Needs &need) {
    if (rows == 0 || columns == 0 || columns > SIZE_MAX / sizeof(float) ||
        rows > SIZE_MAX / sizeof(float))
        return false;
    need.x = columns * sizeof(float);
    need.y = rows * sizeof(float);
    if (rows > SIZE_MAX / need.x) return false;
    need.weight = rows * need.x;
    return true;
}

bool region_usable(const void *pointer, size_t bytes, size_t need) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    return pointer != nullptr && address % alignof(float) == 0 && bytes >= need &&
           bytes <= UINTPTR_MAX - address;
}

// region_usable has checked both sums before this comparison.
bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return reinterpret_cast<uintptr_t>(a) + a_bytes <= reinterpret_cast<uintptr_t>(b) ||
           reinterpret_cast<uintptr_t>(b) + b_bytes <= reinterpret_cast<uintptr_t>(a);
}

bool device_regions_usable(const void *weight, size_t weight_bytes,
                           const inference_cuda_workspace &ws, const Needs &need) {
    return region_usable(weight, weight_bytes, need.weight) &&
           region_usable(ws.x, ws.x_bytes, need.x) &&
           region_usable(ws.y, ws.y_bytes, need.y) &&
           apart(weight, weight_bytes, ws.x, ws.x_bytes) &&
           apart(weight, weight_bytes, ws.y, ws.y_bytes) &&
           apart(ws.x, ws.x_bytes, ws.y, ws.y_bytes);
}

__global__ void gemv_rows(const float *weight, const float *x, float *y,
                          size_t rows, size_t columns) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows; row += stride) {
        float sum = 0.0f;
        for (size_t column = 0; column < columns; ++column)
            sum = __fadd_rn(sum, __fmul_rn(weight[row * columns + column], x[column]));
        y[row] = sum;
    }
}

constexpr unsigned kArgmaxThreads = 256;
constexpr size_t kArgmaxScratchBytes = 16;

// Reduces the finished row to the lowest index holding its largest finite value.
// A row holding any non-finite value sets the flag instead of a winner.
__global__ void argmax_row(const float *y, size_t rows, void *scratch) {
    __shared__ float best_value[kArgmaxThreads];
    __shared__ unsigned int best_index[kArgmaxThreads];
    __shared__ int any_nonfinite[kArgmaxThreads];
    const unsigned int tid = threadIdx.x;
    float value = -INFINITY;
    unsigned int index = UINT32_MAX;
    int nonfinite = 0;
    for (size_t row = tid; row < rows; row += blockDim.x) {
        const float current = y[row];
        if (!isfinite(current)) {
            nonfinite = 1;
            continue;
        }
        if (current > value) {
            value = current;
            index = static_cast<unsigned int>(row);
        }
    }
    best_value[tid] = value;
    best_index[tid] = index;
    any_nonfinite[tid] = nonfinite;
    __syncthreads();
    for (unsigned int stride = kArgmaxThreads / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            const float other_value = best_value[tid + stride];
            const unsigned int other_index = best_index[tid + stride];
            if (any_nonfinite[tid + stride]) any_nonfinite[tid] = 1;
            if (other_value > best_value[tid] ||
                (other_value == best_value[tid] &&
                 other_index < best_index[tid])) {
                best_value[tid] = other_value;
                best_index[tid] = other_index;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        unsigned int *words = static_cast<unsigned int *>(scratch);
        words[0] = best_index[0];
        float *values = static_cast<float *>(scratch);
        values[1] = best_value[0];
        words[2] = any_nonfinite[0] ? 1u : 0u;
    }
}

}  // namespace

extern "C" int inference_cuda_nemotron_gemv_f32_ws(
    const void *weight, size_t weight_bytes, size_t rows, size_t columns,
    const float *x, size_t x_bytes, float *y, size_t y_bytes,
    const inference_cuda_workspace *workspace, int *out_stage) {
    Failure failure;
    bool device_started = false;
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        Needs need;
        if (workspace == nullptr || !checked_needs(rows, columns, need) ||
            !region_usable(x, x_bytes, need.x) || !region_usable(y, y_bytes, need.y) ||
            !device_regions_usable(weight, weight_bytes, *workspace, need))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        for (size_t column = 0; column < columns; ++column)
            if (!std::isfinite(x[column])) return INFERENCE_CUDA_INVALID_ARGUMENT;

        (void)cudaGetLastError();
        device_started = true;
        failure.record(cudaMemcpy(workspace->x, x, need.x, cudaMemcpyHostToDevice),
                       INFERENCE_CUDA_STAGE_COPY);
        if (failure.ok()) {
            const size_t blocks_needed = (rows - 1) / 128 + 1;
            const unsigned blocks = static_cast<unsigned>(blocks_needed < 65535
                                                             ? blocks_needed : 65535);
            gemv_rows<<<blocks, 128>>>(static_cast<const float *>(weight),
                                      static_cast<const float *>(workspace->x),
                                      static_cast<float *>(workspace->y), rows, columns);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        failure.record(cudaDeviceSynchronize(), INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        if (failure.ok())
            failure.record(cudaMemcpy(y, workspace->y, need.y, cudaMemcpyDeviceToHost),
                           INFERENCE_CUDA_STAGE_COPY);
        (void)cudaGetLastError();
        report_stage(out_stage, failure.stage);
        return static_cast<int>(failure.err);
    } catch (...) {
        if (device_started) {
            // Cleanup must not replace an earlier CUDA failure or the exception.
            (void)cudaDeviceSynchronize();
            (void)cudaGetLastError();
        }
        report_stage(out_stage, failure.stage);
        return failure.ok() ? INFERENCE_CUDA_NATIVE_EXCEPTION : static_cast<int>(failure.err);
    }
}

extern "C" int inference_cuda_nemotron_gemv_argmax_f32_ws(
    const void *weight, size_t weight_bytes, size_t rows, size_t columns,
    const float *x, size_t x_bytes, void *scratch, size_t scratch_bytes,
    void *out_host, size_t out_host_bytes,
    const inference_cuda_workspace *workspace, int *out_stage) {
    Failure failure;
    bool device_started = false;
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        Needs need;
        if (workspace == nullptr || out_host == nullptr || out_host_bytes < 8 ||
            rows > UINT32_MAX || !checked_needs(rows, columns, need) ||
            !region_usable(x, x_bytes, need.x) ||
            !region_usable(scratch, scratch_bytes, kArgmaxScratchBytes) ||
            !device_regions_usable(weight, weight_bytes, *workspace, need) ||
            !apart(weight, weight_bytes, scratch, scratch_bytes) ||
            !apart(scratch, scratch_bytes, workspace->x, workspace->x_bytes) ||
            !apart(scratch, scratch_bytes, workspace->y, workspace->y_bytes))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        for (size_t column = 0; column < columns; ++column)
            if (!std::isfinite(x[column])) return INFERENCE_CUDA_INVALID_ARGUMENT;

        (void)cudaGetLastError();
        device_started = true;
        failure.record(cudaMemcpy(workspace->x, x, need.x, cudaMemcpyHostToDevice),
                       INFERENCE_CUDA_STAGE_COPY);
        if (failure.ok()) {
            const size_t blocks_needed = (rows - 1) / 128 + 1;
            const unsigned blocks = static_cast<unsigned>(blocks_needed < 65535
                                                             ? blocks_needed : 65535);
            gemv_rows<<<blocks, 128>>>(static_cast<const float *>(weight),
                                      static_cast<const float *>(workspace->x),
                                      static_cast<float *>(workspace->y), rows, columns);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        if (failure.ok()) {
            argmax_row<<<1, kArgmaxThreads>>>(
                static_cast<const float *>(workspace->y), rows, scratch);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        if (failure.ok())
            failure.record(cudaDeviceSynchronize(), INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        if (failure.ok()) {
            unsigned char reported[kArgmaxScratchBytes] = {};
            failure.record(cudaMemcpy(reported, scratch, sizeof(reported),
                                      cudaMemcpyDeviceToHost),
                           INFERENCE_CUDA_STAGE_COPY);
            unsigned int flag = 0;
            if (failure.ok()) {
                std::memcpy(&flag, reported + 8, sizeof(flag));
                if (flag != 0) return INFERENCE_CUDA_INVALID_ARGUMENT;
                std::memcpy(out_host, reported, 8);
            }
        }
        (void)cudaGetLastError();
        report_stage(out_stage, failure.stage);
        return static_cast<int>(failure.err);
    } catch (...) {
        if (device_started) {
            (void)cudaDeviceSynchronize();
            (void)cudaGetLastError();
        }
        report_stage(out_stage, failure.stage);
        return failure.ok() ? INFERENCE_CUDA_NATIVE_EXCEPTION : static_cast<int>(failure.err);
    }
}
