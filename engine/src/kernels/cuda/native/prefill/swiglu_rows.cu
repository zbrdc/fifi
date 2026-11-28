/* Argument validation and the one launch of swiglu_rows (contract: prefill/swiglu_rows.h).
 * The entry follows attn_prep_chunk.cu (validate everything before any CUDA call, try/catch,
 * C types only), and the element is x / (1 + expf(-x)) * u. Built in CUDA_SOURCES without
 * -use_fast_math so prefill SwiGLU uses the same IEEE expf and division as decode swiglu_q8. */
#include "prefill/swiglu_rows.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"

namespace {

constexpr int kThreads = 128;   // float4 columns per block
constexpr int kMaxRows = 32768; // 8 x the 4,096-row chunk cap; below the grid's y limit (65,535)
constexpr int kMaxWidth = 32768;

bool aligned16(const void *p) { return ((uintptr_t)p & 15) == 0; }

bool valid_shape(const inference_cuda_swiglu_rows_args *a) {
    const bool rows = a->rows >= 1 && a->rows <= kMaxRows;
    const bool width = a->n_ff >= 4 && a->n_ff <= kMaxWidth && a->n_ff % 4 == 0;
    const bool up = a->up_off >= a->n_ff && a->up_off % 4 == 0;
    const bool in = (long long)a->ld_in >= (long long)a->up_off + a->n_ff && a->ld_in % 4 == 0;
    const bool out = a->ld_out >= a->n_ff && a->ld_out % 4 == 0;
    return rows && width && up && in && out;
}

/* Bytes of the two extents; valid after valid_shape, which bounds every factor, so the
 * 64-bit products cannot wrap. */
size_t gu_bytes(const inference_cuda_swiglu_rows_args *a) {
    return ((size_t)(a->rows - 1) * (size_t)a->ld_in + (size_t)a->up_off + (size_t)a->n_ff) * 4;
}

size_t act_bytes(const inference_cuda_swiglu_rows_args *a) {
    return ((size_t)(a->rows - 1) * (size_t)a->ld_out + (size_t)a->n_ff) * 4;
}

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return (uintptr_t)a + a_bytes <= (uintptr_t)b || (uintptr_t)b + b_bytes <= (uintptr_t)a;
}

bool valid_swiglu_rows(const inference_cuda_swiglu_rows_args *a) {
    if (a == NULL || a->gu == NULL || a->act == NULL || !aligned16(a->gu) ||
        !aligned16(a->act) || !valid_shape(a)) {
        return false;
    }
    return apart(a->act, act_bytes(a), a->gu, gu_bytes(a));
}

// One element: the silu of the gate times the up value.
__device__ __forceinline__ float swiglu1(float g, float u) {
    return (g / (1.0f + expf(-g))) * u;
}

// Block (x, r): thread i handles float4 column c4 = x * kThreads + i of row r, n4 = n_ff / 4.
__global__ void swiglu_rows_k(const float *__restrict__ gu, float *__restrict__ act, int n4,
                              int ld_in, int up_off, int ld_out) {
    const int c4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (c4 >= n4) return;
    const float *in = gu + (int64_t)blockIdx.y * ld_in;
    const float4 g = reinterpret_cast<const float4 *>(in)[c4];
    const float4 u = reinterpret_cast<const float4 *>(in + up_off)[c4];
    const float4 y = make_float4(swiglu1(g.x, u.x), swiglu1(g.y, u.y), swiglu1(g.z, u.z),
                                 swiglu1(g.w, u.w));
    reinterpret_cast<float4 *>(act + (int64_t)blockIdx.y * ld_out)[c4] = y;
}

Failure launch_rows(const inference_cuda_swiglu_rows_args *a, void *stream) {
    const int n4 = a->n_ff / 4;
    const dim3 grid((n4 + kThreads - 1) / kThreads, a->rows);
    swiglu_rows_k<<<grid, kThreads, 0, (cudaStream_t)stream>>>(
        (const float *)a->gu, (float *)a->act, n4, a->ld_in, a->up_off, a->ld_out);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    return failure;
}

}  // namespace

extern "C" int inference_cuda_swiglu_rows_launch(const inference_cuda_swiglu_rows_args *a,
                                                 void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid_swiglu_rows(a)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        /* The launch status is the only status: the kernel has no failure of its own. */
        const Failure failure = launch_rows(a, stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
