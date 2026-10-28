/* NVFP4 batched matvec: one launch and one synchronization for a layer's
 * selected experts. The per-element arithmetic is a line-for-line copy of the
 * single-expert kernel's byte path, so the batch is bit for bit identical. */
#include "nvfp4_gemv_batch.h"
#include "device.h"

#include <cmath>
#include <cstdint>

namespace {

/* kernels/nvfp4.rs GROUP_SIZE: sixteen elements share one E4M3FN scale byte. */
constexpr size_t kGroupSize = 16;
constexpr size_t kGroupBytes = kGroupSize / 2;
constexpr unsigned kMatvecThreads = 128;

__constant__ float kE2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

/* E4M3FN of one scale byte, identical to nvfp4_gemv.cu: bias 7, subnormal
 * fraction * 2^-9, NaN only at exponent 15 with fraction 7, sign negating. */
__device__ float e4m3_value(unsigned char code) {
    const unsigned exponent = (code >> 3) & 15u;
    const unsigned fraction = code & 7u;
    float magnitude;
    if (exponent == 0u) {
        magnitude = static_cast<float>(fraction) / 512.0f;
    } else if (exponent == 15u && fraction == 7u) {
        magnitude = __int_as_float(0x7fc00000u);
    } else {
        magnitude = __int_as_float(((exponent + 120u) << 23) | (fraction << 20));
    }
    return (code & 128u) != 0u ? -magnitude : magnitude;
}

/* One element of the host WeightBlock::matvec sequence: one rounded multiply
 * for the decoded weight, one rounded multiply by the input, one rounded add.
 * __fmul_rn and __fadd_rn never fuse, so a non-finite decoded weight only
 * flags the batch and the touched row is never written. */
__device__ __forceinline__ void accumulate_step(float magnitude, float scale,
                                                float xv, float &sum, bool &bad) {
    const float value = __fmul_rn(magnitude, scale);
    bad |= !isfinite(value);
    sum = __fadd_rn(sum, __fmul_rn(value, xv));
}

/* Byte walk of one row, exactly the single-expert kernel's narrow path:
 * groups in order, bytes in order, even nibble before odd nibble. */
__device__ __forceinline__ void row_dot(const unsigned char *row_packed,
                                        const unsigned char *row_scales,
                                        float weight_scale_2,
                                        const float *x, size_t columns,
                                        const float *e2m1, float &sum,
                                        bool &bad) {
    const size_t per_row = columns / kGroupSize;
    for (size_t group = 0; group < per_row; ++group) {
        const unsigned char *group_bytes = row_packed + group * kGroupBytes;
        const float scale = __fmul_rn(e4m3_value(row_scales[group]), weight_scale_2);
        const float *xg = x + group * kGroupSize;
        for (size_t byte = 0; byte < kGroupBytes; ++byte) {
            const unsigned char packed = group_bytes[byte];
            accumulate_step(e2m1[packed & 15u], scale, xg[byte * 2], sum, bad);
            accumulate_step(e2m1[(packed >> 4) & 15u], scale, xg[byte * 2 + 1], sum, bad);
        }
    }
}

__global__ void nvfp4_gemv_batch_rows(
    const unsigned char *const *packed_ptrs, const unsigned char *const *scale_ptrs,
    const float *weight_scales, const float *x, float *y,
    unsigned int *nonfinite, size_t n, size_t rows, size_t columns) {
    extern __shared__ float e2m1[];
    for (unsigned i = 0; i < 16; ++i)
        e2m1[i] = kE2M1[i];
    __syncthreads();
    const size_t total = n * rows;
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
         index += stride) {
        const size_t expert = index / rows;
        const size_t row = index % rows;
        const size_t per_row = columns / kGroupSize;
        float sum = 0.0f;
        bool bad = false;
        row_dot(packed_ptrs[expert] + row * per_row * kGroupBytes,
                scale_ptrs[expert] + row * per_row, weight_scales[expert],
                x + expert * columns, columns, e2m1, sum, bad);
        if (bad)
            atomicOr(nonfinite, 1u);
        else
            y[index] = sum;
    }
}

struct Needs {
    size_t pointers;
    size_t x;
    size_t y;
    size_t flag;
};

bool checked_needs(size_t n, size_t rows, size_t columns, Needs &need) {
    if (n == 0 || rows == 0 || columns == 0 || columns % kGroupSize != 0 ||
        n > SIZE_MAX / rows || rows > SIZE_MAX / columns ||
        n > SIZE_MAX / sizeof(void *) || n > SIZE_MAX / sizeof(float) ||
        columns > SIZE_MAX / sizeof(float) || rows > SIZE_MAX / sizeof(float))
        return false;
    need.pointers = n * sizeof(void *);
    need.x = n * columns * sizeof(float);
    need.y = n * rows * sizeof(float);
    need.flag = sizeof(unsigned int);
    return true;
}

bool usable(const void *pointer, size_t bytes, size_t need, size_t alignment) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    return pointer != nullptr && address % alignment == 0 && bytes >= need &&
           bytes <= UINTPTR_MAX - address;
}

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return reinterpret_cast<uintptr_t>(a) + a_bytes <= reinterpret_cast<uintptr_t>(b) ||
           reinterpret_cast<uintptr_t>(b) + b_bytes <= reinterpret_cast<uintptr_t>(a);
}

}  // namespace

extern "C" int inference_cuda_nvfp4_gemv_batch_ws(
    const void *const *packed_ptrs, size_t packed_ptrs_bytes,
    const void *const *scale_ptrs, size_t scale_ptrs_bytes,
    const float *weight_scales, size_t weight_scales_bytes,
    size_t n, size_t rows, size_t columns,
    const float *x, size_t x_bytes, float *y, size_t y_bytes,
    const inference_cuda_workspace *workspace, int *out_stage) {
    Failure failure;
    bool device_started = false;
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        Needs need;
        if (workspace == nullptr || !checked_needs(n, rows, columns, need) ||
            !usable(packed_ptrs, packed_ptrs_bytes, need.pointers, alignof(void *)) ||
            !usable(scale_ptrs, scale_ptrs_bytes, need.pointers, alignof(void *)) ||
            !usable(x, x_bytes, need.x, alignof(float)) ||
            !usable(y, y_bytes, need.y, alignof(float)) ||
            !usable(workspace->x, workspace->x_bytes, need.x, alignof(float)) ||
            !usable(workspace->y, workspace->y_bytes, need.y, alignof(float)) ||
            !usable(workspace->xf, workspace->xf_bytes, need.flag, alignof(unsigned int)) ||
            !apart(workspace->x, need.x, workspace->y, need.y) ||
            !apart(workspace->x, need.x, workspace->xf, need.flag) ||
            !apart(workspace->y, need.y, workspace->xf, need.flag))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        // weight_scales is validated host-side by the safe wrapper before its
        // upload; the kernel flags any non-finite decoded product regardless.
        for (size_t i = 0; i < need.x / sizeof(float); ++i)
            if (!std::isfinite(x[i])) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (!usable(weight_scales, weight_scales_bytes, n * sizeof(float), alignof(float)))
            return INFERENCE_CUDA_INVALID_ARGUMENT;

        (void)cudaGetLastError();
        device_started = true;
        failure.record(cudaMemcpy(workspace->x, x, need.x, cudaMemcpyHostToDevice),
                       INFERENCE_CUDA_STAGE_COPY);
        if (failure.ok())
            failure.record(cudaMemset(workspace->xf, 0, need.flag),
                           INFERENCE_CUDA_STAGE_COPY);
        if (failure.ok()) {
            const size_t total = n * rows;
            const size_t blocks_needed = (total - 1) / kMatvecThreads + 1;
            const unsigned blocks =
                static_cast<unsigned>(blocks_needed < 65535 ? blocks_needed : 65535);
            nvfp4_gemv_batch_rows<<<blocks, kMatvecThreads, 16 * sizeof(float)>>>(
                reinterpret_cast<const unsigned char *const *>(packed_ptrs),
                reinterpret_cast<const unsigned char *const *>(scale_ptrs),
                reinterpret_cast<const float *>(weight_scales),
                static_cast<const float *>(workspace->x),
                static_cast<float *>(workspace->y),
                static_cast<unsigned int *>(workspace->xf), n, rows, columns);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        failure.record(cudaDeviceSynchronize(), INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        if (failure.ok()) {
            unsigned int flag = 0;
            failure.record(cudaMemcpy(&flag, workspace->xf, need.flag,
                                      cudaMemcpyDeviceToHost),
                           INFERENCE_CUDA_STAGE_COPY);
            if (failure.ok() && flag != 0) return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (failure.ok())
            failure.record(cudaMemcpy(y, workspace->y, need.y, cudaMemcpyDeviceToHost),
                           INFERENCE_CUDA_STAGE_COPY);
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
