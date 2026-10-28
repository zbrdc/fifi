/* Entry of embed_q8 on an explicit stream. Glue only. */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/embed_q8.h"

namespace {

constexpr int kBlock = 32;  // values per Q8_0 block

struct BlockQ8_0 {
    unsigned short d;  // fp16 scale bits
    signed char q[kBlock];
};

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }
bool apart(const void *a, size_t ab, const void *b, size_t bb) {
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? ab <= y - x : bb <= x - y;
}

/* One block of 32 threads per block column; blockIdx.x = row, blockIdx.y = block index. */
__global__ void embed_q8_kernel(float *out, const BlockQ8_0 *w, const int *ids, int n_rows, int k) {
    const int row = blockIdx.x;
    const int id = ids[row];
    const int b = blockIdx.y;
    const int lane = threadIdx.x;
    const long long at = (long long)row * k + b * kBlock + lane;
    if (id < 0 || id >= n_rows) {
        out[at] = nanf("");
        return;
    }
    const BlockQ8_0 blk = w[(long long)id * (k / kBlock) + b];
    out[at] = __half2float(__ushort_as_half(blk.d)) * (float)blk.q[lane];
}

}  // namespace

extern "C" int inference_cuda_embed_q8_launch(const inference_cuda_embed_q8_args *args,
                                              void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->out == NULL || args->w == NULL || args->ids == NULL)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->m < 1 || args->m > 4096 || args->n_rows < 1 || args->k < 32 ||
            args->k > 32768 || args->k % kBlock != 0)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (!aligned(args->out, 16) || !aligned(args->w, 4) || !aligned(args->ids, 4))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const size_t out_bytes = (size_t)args->m * (size_t)args->k * sizeof(float);
        const size_t w_bytes = (size_t)args->n_rows * (size_t)(args->k / kBlock) * sizeof(BlockQ8_0);
        const size_t ids_bytes = (size_t)args->m * sizeof(int);
        if (!apart(args->out, out_bytes, args->w, w_bytes) ||
            !apart(args->out, out_bytes, args->ids, ids_bytes))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        embed_q8_kernel<<<dim3((unsigned)args->m, (unsigned)(args->k / kBlock)), kBlock, 0,
                          (cudaStream_t)stream>>>((float *)args->out, (const BlockQ8_0 *)args->w,
                                                  (const int *)args->ids, args->n_rows, args->k);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
