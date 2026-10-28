/* BF16 dense matvec: the device widens the stored 16-bit codes to F32 in
 * registers (exact: sixteen zero bits appended), so the host never expands a
 * dense matrix (M2 rule). The arithmetic is bit for bit the host reference:
 * the loader widens each code exactly, and the reference stack then runs the
 * F32 matvec sequence, one rounded multiply and add per element, no fused
 * multiply-add. */
#include "bf16_gemv.h"
#include "device.h"

#include <cmath>
#include <cstdint>

namespace {

constexpr unsigned kMatvecThreads = 128;

/* Widen one BF16 code: bit for bit the host loader's
 * f32::from_bits(u32::from(u16::from_le_bytes(pair)) << 16). */
__device__ __forceinline__ float bf16_value(unsigned short code) {
    return __int_as_float(static_cast<unsigned>(code) << 16);
}

__global__ void bf16_gemv_rows(const unsigned short *codes,
                               const float *x, float *y,
                               unsigned int *nonfinite, size_t rows,
                               size_t columns) {
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
         row += stride) {
        const unsigned short *row_codes = codes + row * columns;
        float sum = 0.0f;
        bool bad = false;
        for (size_t column = 0; column < columns; ++column) {
            const float value = bf16_value(row_codes[column]);
            bad |= !isfinite(value);
            sum = __fadd_rn(sum, __fmul_rn(value, x[column]));
        }
        if (bad)
            atomicOr(nonfinite, 1u);
        else
            y[row] = sum;
    }
}

struct Needs {
    size_t codes;
    size_t x;
    size_t y;
    size_t flag;
};

bool checked_needs(size_t rows, size_t columns, Needs &need) {
    if (rows == 0 || columns == 0 || rows > SIZE_MAX / columns ||
        columns > SIZE_MAX / 2 || rows > SIZE_MAX / 2 ||
        columns > SIZE_MAX / sizeof(float) || rows > SIZE_MAX / sizeof(float))
        return false;
    need.codes = rows * columns * 2;
    need.x = columns * sizeof(float);
    need.y = rows * sizeof(float);
    need.flag = sizeof(unsigned int);
    return true;
}

bool usable(const void *pointer, size_t bytes, size_t need, size_t alignment) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    return pointer != nullptr && address % alignment == 0 && bytes >= need &&
           bytes <= UINTPTR_MAX - address;
}

// usable has checked both sums before this comparison.
bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return reinterpret_cast<uintptr_t>(a) + a_bytes <= reinterpret_cast<uintptr_t>(b) ||
           reinterpret_cast<uintptr_t>(b) + b_bytes <= reinterpret_cast<uintptr_t>(a);
}

bool bf16_regions_usable(const void *codes, size_t codes_bytes,
                         const inference_cuda_workspace &ws, const Needs &need) {
    // The host reference stores exact byte counts, so the extent must match.
    if (codes_bytes != need.codes) return false;
    return usable(codes, codes_bytes, need.codes, alignof(unsigned short)) &&
           usable(ws.x, ws.x_bytes, need.x, alignof(float)) &&
           usable(ws.y, ws.y_bytes, need.y, alignof(float)) &&
           usable(ws.xf, ws.xf_bytes, need.flag, alignof(unsigned int)) &&
           apart(codes, codes_bytes, ws.x, need.x) &&
           apart(codes, codes_bytes, ws.y, need.y) &&
           apart(codes, codes_bytes, ws.xf, need.flag) &&
           apart(ws.x, need.x, ws.y, need.y) &&
           apart(ws.x, need.x, ws.xf, need.flag) &&
           apart(ws.y, need.y, ws.xf, need.flag);
}

}  // namespace

extern "C" int inference_cuda_bf16_gemv_ws(
    const void *codes, size_t codes_bytes,
    size_t rows, size_t columns,
    const float *x, size_t x_bytes, float *y, size_t y_bytes,
    const inference_cuda_workspace *workspace, int *out_stage) {
    Failure failure;
    bool device_started = false;
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        Needs need;
        if (workspace == nullptr || !checked_needs(rows, columns, need) ||
            !bf16_regions_usable(codes, codes_bytes, *workspace, need) ||
            !usable(x, x_bytes, need.x, alignof(float)) ||
            !usable(y, y_bytes, need.y, alignof(float)))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        for (size_t column = 0; column < columns; ++column)
            if (!std::isfinite(x[column])) return INFERENCE_CUDA_INVALID_ARGUMENT;

        (void)cudaGetLastError();
        device_started = true;
        failure.record(cudaMemcpy(workspace->x, x, need.x, cudaMemcpyHostToDevice),
                       INFERENCE_CUDA_STAGE_COPY);
        if (failure.ok())
            failure.record(cudaMemset(workspace->xf, 0, need.flag),
                           INFERENCE_CUDA_STAGE_COPY);
        if (failure.ok()) {
            const size_t blocks_needed = (rows - 1) / kMatvecThreads + 1;
            const unsigned blocks = static_cast<unsigned>(blocks_needed < 65535
                                                             ? blocks_needed : 65535);
            bf16_gemv_rows<<<blocks, kMatvecThreads>>>(
                static_cast<const unsigned short *>(codes),
                static_cast<const float *>(workspace->x),
                static_cast<float *>(workspace->y),
                static_cast<unsigned int *>(workspace->xf), rows, columns);
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
            // Cleanup must not replace an earlier CUDA failure or the exception.
            (void)cudaDeviceSynchronize();
            (void)cudaGetLastError();
        }
        report_stage(out_stage, failure.stage);
        return failure.ok() ? INFERENCE_CUDA_NATIVE_EXCEPTION : static_cast<int>(failure.err);
    }
}

extern "C" int inference_cuda_bf16_gemv_launch(
    const void *codes, size_t codes_bytes,
    size_t rows, size_t columns,
    const float *x, size_t x_bytes,
    const inference_cuda_workspace *workspace, void *stream, int *out_stage) {
    Failure failure;
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        Needs need;
        if (workspace == nullptr || !checked_needs(rows, columns, need) ||
            !bf16_regions_usable(codes, codes_bytes, *workspace, need) ||
            !usable(x, x_bytes, need.x, alignof(float)) ||
            !apart(x, need.x, workspace->y, need.y))
            return INFERENCE_CUDA_INVALID_ARGUMENT;

        (void)cudaGetLastError();
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        failure.record(cudaMemsetAsync(workspace->xf, 0, need.flag, cuda_stream),
                       INFERENCE_CUDA_STAGE_COPY);
        if (failure.ok()) {
            const size_t blocks_needed = (rows - 1) / kMatvecThreads + 1;
            const unsigned blocks = static_cast<unsigned>(blocks_needed < 65535
                                                             ? blocks_needed : 65535);
            bf16_gemv_rows<<<blocks, kMatvecThreads, 0, cuda_stream>>>(
                static_cast<const unsigned short *>(codes),
                static_cast<const float *>(x),
                static_cast<float *>(workspace->y),
                static_cast<unsigned int *>(workspace->xf), rows, columns);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return static_cast<int>(failure.err);
    } catch (...) {
        report_stage(out_stage, failure.stage);
        return failure.ok() ? INFERENCE_CUDA_NATIVE_EXCEPTION : static_cast<int>(failure.err);
    }
}
