/* Test-only bounded readback. Serial double reduction favors auditability over speed. */
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include "abi.h"
#include "device.h"
#include "resident/top_logits.h"

namespace {
__global__ void top_logits_kernel(const float *x, int n, unsigned *ids, float *lp,
                                  unsigned *status) {
    *status = 0;
    float best[5] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
    for (int k = 0; k < 5; ++k) ids[k] = UINT32_MAX;
    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) { *status = 1; return; }
        for (int k = 0; k < 5; ++k) {
            if (x[i] > best[k]) {
                for (int j = 4; j > k; --j) { best[j] = best[j-1]; ids[j] = ids[j-1]; }
                best[k] = x[i]; ids[k] = (unsigned)i; break;
            }
        }
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += exp((double)x[i] - (double)best[0]);
    const double normalizer = log(sum);
    for (int k = 0; k < 5; ++k) lp[k] = (float)((double)best[k] - best[0] - normalizer);
}
}
extern "C" int inference_cuda_top_logits_launch(const inference_cuda_top_logits_args *a,
                                                void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!a || !stream || !a->out || !a->x || a->n < 5 || a->n > 4194304)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const uintptr_t out = (uintptr_t)a->out, x = (uintptr_t)a->x;
        if (out % 4 || x % 4) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const size_t bytes = (size_t)a->n * 4;
        if (!(out <= x ? 44 <= x - out : bytes <= out - x))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        top_logits_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(
            (const float *)a->x, a->n, (unsigned *)a->out,
            (float *)a->out + 5, (unsigned *)a->out + 10);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) { return INFERENCE_CUDA_NATIVE_EXCEPTION; }
}
