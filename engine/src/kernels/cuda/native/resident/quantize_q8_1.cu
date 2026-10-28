/*
 * Entries of the vendored quantize_row_q8_1 (MATVEC-PAR S1 c3a) on an explicit stream. Glue only:
 * no kernel and no device logic of its own; every exported function has C linkage, takes and
 * returns C types only, and wraps its body in try/catch, like the projector adapter.
 */
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/quantize_q8_1.h"

// The vendored headers keep static host helpers their own objects reference; in this glue TU they
// are unreferenced, which -Wall reports although nothing here is wrong.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "quantize.cuh"
#pragma GCC diagnostic pop

namespace {

/* k is padded up to this before the launch (MATRIX_ROW_PADDING, common.cuh). */
const int64_t kMatrixRowPadding = 512;

bool aligned(const void *p, size_t alignment) { return (uintptr_t)p % alignment == 0; }

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return (uintptr_t)a + a_bytes <= (uintptr_t)b || (uintptr_t)b + b_bytes <= (uintptr_t)a;
}

}  // namespace

extern "C" int inference_cuda_quantize_q8_1_launch(const inference_cuda_quantize_q8_1_args *args,
                                                   void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int k = args->k;
        const int ncols = args->ncols;
        if (args->x == NULL || args->y == NULL || k < 32 || k % QK8_1 != 0 ||
            ncols < 1 || ncols > 8 || !aligned(args->x, sizeof(float)) ||
            !aligned(args->y, sizeof(float))) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        // The kernel writes y while it reads x; the guard needs the byte counts and must not
        // wrap (ncols * padded blocks * sizeof(block_q8_1)).
        const int64_t ne0 = (k + kMatrixRowPadding - 1) / kMatrixRowPadding * kMatrixRowPadding;
        const int64_t blocks = ne0 / QK8_1;
        if (blocks > INT64_MAX / (int64_t)sizeof(block_q8_1) ||
            blocks * (int64_t)sizeof(block_q8_1) > INT64_MAX / ncols ||
            !apart(args->y, (size_t)(blocks * (int64_t)sizeof(block_q8_1) * ncols), args->x,
                   (size_t)((int64_t)k * ncols * (int64_t)sizeof(float)))) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        // One launch over the padded width; the kernel writes the zero tail itself.
        Failure failure;
        failure.record(quantize_row_q8_1_cuda(
                           (const float *)args->x, NULL, args->y, GGML_TYPE_F32,
                           k, /*s01=*/k, /*s02=*/(int64_t)k * ncols, /*s03=*/(int64_t)k * ncols,
                           ne0, ncols, /*ne2=*/1, /*ne3=*/1, (cudaStream_t)stream),
                       INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
