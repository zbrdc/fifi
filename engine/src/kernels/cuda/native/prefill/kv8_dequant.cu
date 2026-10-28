#include "abi.h"
#include "device.h"
#include "prefill/kv8_dequant.h"

#include <cuda_fp16.h>
#include <stdint.h>

namespace {
constexpr unsigned kThreads = 256;
constexpr int kMaxCapacity = (1 << 30) - 1;

bool valid_shape(const inference_cuda_kv8_dequant_args *a) {
    return a != nullptr && a->capacity > 0 && a->capacity <= kMaxCapacity &&
           a->rows > 0 && a->rows <= a->capacity && a->n_kv_head > 0 &&
           a->n_kv_head <= 64 && a->head_dim >= 32 && a->head_dim <= 1024 &&
           a->head_dim % 32 == 0;
}

bool span_ok(const void *p, size_t bytes, uintptr_t alignment) {
    const uintptr_t at = reinterpret_cast<uintptr_t>(p);
    return p != nullptr && at % alignment == 0 && bytes > 0 && bytes <= UINTPTR_MAX - at;
}

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(a);
    const uintptr_t y = reinterpret_cast<uintptr_t>(b);
    return x + a_bytes <= y || y + b_bytes <= x;
}

bool valid_call(const inference_cuda_kv8_dequant_args *a) {
    if (!valid_shape(a)) return false;
    const size_t codes = size_t(a->capacity) * a->n_kv_head * a->head_dim;
    const size_t scales = codes / 32 * 2;
    const size_t out = size_t(a->rows) * a->n_kv_head * a->head_dim * 2;
    return span_ok(a->codes, codes, 4) && span_ok(a->scales, scales, 4) &&
           span_ok(a->out, out, 16) && apart(a->out, out, a->codes, codes) &&
           apart(a->out, out, a->scales, scales);
}

__global__ void kv8_dequant_k(const int8_t *__restrict__ codes,
                              const __half *__restrict__ scales,
                              __half *__restrict__ out, int capacity, int rows,
                              int heads, int dim, int source_row_origin) {
    const uint64_t count = uint64_t(rows) * heads * dim;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const uint64_t row = i / (uint64_t(heads) * dim);
        const uint64_t head = (i / dim) % heads;
        const uint64_t col = i % dim;
        const uint64_t source_row = head * capacity + source_row_origin + row;
        const float scale = __half2float(scales[source_row * (dim / 32) + col / 32]);
        out[i] = __float2half_rn(float(codes[source_row * dim + col]) * scale);
    }
}

Failure launch(const inference_cuda_kv8_dequant_args *a, int source_row_origin, void *stream) {
    const uint64_t count = uint64_t(a->rows) * a->n_kv_head * a->head_dim;
    const uint64_t needed = (count + kThreads - 1) / kThreads;
    const unsigned blocks = unsigned(needed < 65535 ? needed : 65535);
    kv8_dequant_k<<<blocks, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const int8_t *>(a->codes), static_cast<const __half *>(a->scales),
        static_cast<__half *>(a->out), a->capacity, a->rows, a->n_kv_head, a->head_dim,
        source_row_origin);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    return failure;
}
}  // namespace

extern "C" int inference_cuda_kv8_dequant_launch(const inference_cuda_kv8_dequant_args *a,
                                                void *stream, int *out_stage) {
    return inference_cuda_kv8_dequant_segment_launch(a, 0, stream, out_stage);
}

extern "C" int inference_cuda_kv8_dequant_segment_launch(
    const inference_cuda_kv8_dequant_args *a, int source_row_origin, void *stream,
    int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (stream == nullptr || !valid_call(a) || source_row_origin < 0 ||
            int64_t(source_row_origin) + a->rows > a->capacity)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const Failure failure = launch(a, source_row_origin, stream);
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
