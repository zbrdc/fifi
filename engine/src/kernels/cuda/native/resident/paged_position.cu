#include <cuda_runtime.h>
#include <stdint.h>
#include "abi.h"
#include "device.h"
#include "resident/paged_position.h"

namespace {
constexpr int kMaxWord = (1 << 30) - 1;

__global__ void paged_position_k(inference_cuda_paged_position_args a) {
    const int pos = a.params->pos;
    const bool valid = a.params->status == 0 && pos >= 0 && pos < a.capacity &&
                       pos == a.expected && a.params->kv_len == pos + 1 &&
                       a.params->kv_slot == pos;
    *a.local = valid ? pos - a.base : -1;
    if (!valid) {
        atomicOr(&a.params->status, INFERENCE_CUDA_STATUS_INVALID_STEP_STATE);
    }
}

bool valid(const inference_cuda_paged_position_args *a, void *stream) {
    if (a == NULL || stream == NULL || a->params == NULL || a->local == NULL ||
        a->capacity <= 0 || a->capacity > kMaxWord || a->expected < 0 ||
        a->expected >= a->capacity || a->base < 0 || a->base > a->expected ||
        a->cells <= 0 || a->cells > kMaxWord || a->base % a->cells != 0 ||
        a->expected - a->base >= a->cells) {
        return false;
    }
    const uintptr_t p = (uintptr_t)a->params, l = (uintptr_t)a->local;
    return p % 4 == 0 && l % 4 == 0 && p <= UINTPTR_MAX - 32 && l <= UINTPTR_MAX - 4 &&
           (p + 32 <= l || l + 4 <= p);
}
}  // namespace

extern "C" int inference_cuda_paged_position_launch(const inference_cuda_paged_position_args *a,
                                                    void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid(a, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        paged_position_k<<<1, 1, 0, (cudaStream_t)stream>>>(*a);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
