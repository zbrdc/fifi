/*
 * RD-4e: MoE combine of the resident Qwen3.6 decode step (decision 0047).
 *
 * Design guidance only, no code copied. Ideas from MegaCapybara v1.35 static RE,
 * research/megacapybara-re-lessons.md section 6.1 C6 (stop-fusing rule: keep the combine a small
 * separate kernel in stage 1) and C8 (sum the picks in a fixed order in registers so that the result
 * does not depend on scheduling). The code is original CUDA C for sm_86 (sm_72 compiles), written from
 * the shapes of this repository and the graph of the pinned llama.cpp (qwen35moe, build_moe_ffn).
 *
 * Semantics: see moe_combine.h. The weights and the shared gate are RD-5's outputs and are used as
 * given. Every operation is one intrinsic (__fmul_rn, __fadd_rn), so neither the release compiler nor
 * the debug device build (-G -O0) can contract a product and a sum into a fused multiply-add, and the
 * bits equal the host emulation of the same sequence. The first product is the start of the sum (no
 * 0 + p0), so a -0 stays -0 as in ggml.
 *
 * One thread owns four consecutive columns of one token (a float4), reads the top_k weights of the
 * token (a broadcast), the top_k rows of down and, when given, the shared row, and writes one float4.
 * The picks are summed in registers in pick order 0, 1, ..., top_k - 1, then the shared term last, then
 * the residual. No shared memory, no barrier, no atomic; the same input twice gives equal bytes.
 */
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/moe_combine.h"

namespace {

constexpr int kThreads = 128;
constexpr int kMaxTokens = 4096;
constexpr int kMaxN = 65536;
constexpr int kMaxTop = 32;
constexpr int kMaxLd = 1 << 26;

__device__ __forceinline__ float4 load4(const float *p) { return *reinterpret_cast<const float4 *>(p); }

__device__ __forceinline__ float4 mul4(float w, float4 v) {
    return make_float4(__fmul_rn(w, v.x), __fmul_rn(w, v.y), __fmul_rn(w, v.z), __fmul_rn(w, v.w));
}

__device__ __forceinline__ float4 add4(float4 a, float4 b) {
    return make_float4(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y), __fadd_rn(a.z, b.z), __fadd_rn(a.w, b.w));
}

/* res and out may be the same buffer (in place), so they are not marked restrict. */
__global__ void __launch_bounds__(kThreads)
    moe_combine_k(const float *res, const float *__restrict__ down, const float *__restrict__ weights,
                  const float *__restrict__ shared_gate, const float *__restrict__ shared, float *out,
                  int n, int top_k, int ld_down) {
    const int col = (blockIdx.x * kThreads + threadIdx.x) * 4;
    if (col >= n) return;
    const size_t t = blockIdx.y;
    const float *w = weights + t * top_k;
    const float *pick = down + t * top_k * (size_t)ld_down + col;
    float4 s = mul4(w[0], load4(pick));
#pragma unroll 8
    for (int i = 1; i < top_k; ++i) {
        s = add4(s, mul4(w[i], load4(pick + (size_t)i * ld_down)));
    }
    if (shared != NULL) {
        s = add4(s, mul4(shared_gate[t], load4(shared + t * n + col)));
    }
    *reinterpret_cast<float4 *>(out + t * n + col) = add4(load4(res + t * n + col), s);
}

bool aligned_to(const void *p, size_t n) { return ((uintptr_t)p % n) == 0; }

bool apart(const void *a, size_t abytes, const void *b, size_t bbytes) {
    const uintptr_t pa = (uintptr_t)a, pb = (uintptr_t)b;
    return pa + abytes <= pb || pb + bbytes <= pa;
}

bool shapes_ok(const inference_cuda_moe_combine_args *a) {
    const bool m_ok = a->m >= 1 && a->m <= kMaxTokens;
    const bool n_ok = a->n >= 4 && a->n <= kMaxN && a->n % 4 == 0;
    const bool top_ok = a->top_k >= 1 && a->top_k <= kMaxTop;
    const bool ld_ok = a->ld_down >= a->n && a->ld_down <= kMaxLd && a->ld_down % 4 == 0;
    return m_ok && n_ok && top_ok && ld_ok;
}

bool pointers_ok(const inference_cuda_moe_combine_args *a) {
    const bool present = a->res != NULL && a->down != NULL && a->weights != NULL && a->out != NULL;
    const bool paired = (a->shared == NULL) == (a->shared_gate == NULL);
    const bool wide = aligned_to(a->res, 16) && aligned_to(a->down, 16) && aligned_to(a->out, 16) &&
                      (a->shared == NULL || aligned_to(a->shared, 16));
    const bool narrow = aligned_to(a->weights, 4) && (a->shared_gate == NULL || aligned_to(a->shared_gate, 4));
    return present && paired && wide && narrow;
}

/* out shares no byte with any input, except that it may be exactly res. Extents in bytes. */
bool overlaps_ok(const inference_cuda_moe_combine_args *a) {
    const size_t rows = (size_t)a->m * a->n * 4;
    const size_t down = (((size_t)a->m * a->top_k - 1) * a->ld_down + a->n) * 4;
    const bool in_place = (const void *)a->out == (const void *)a->res;
    bool ok = in_place || apart(a->out, rows, a->res, rows);
    ok = ok && apart(a->out, rows, a->down, down);
    ok = ok && apart(a->out, rows, a->weights, (size_t)a->m * a->top_k * 4);
    if (a->shared != NULL) {
        ok = ok && apart(a->out, rows, a->shared, rows) && apart(a->out, rows, a->shared_gate, (size_t)a->m * 4);
    }
    return ok;
}

/* Every term of this test has a twin in the Rust wrapper (the table "C and Rust refusals" of RD-4e.md). */
bool valid(const inference_cuda_moe_combine_args *a, void *stream) {
    return a != NULL && stream != NULL && shapes_ok(a) && pointers_ok(a) && overlaps_ok(a);
}

}  // namespace

extern "C" int inference_cuda_moe_combine_launch(const inference_cuda_moe_combine_args *a, void *stream,
                                                 int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid(a, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const dim3 grid((a->n / 4 + kThreads - 1) / kThreads, a->m);
        moe_combine_k<<<grid, kThreads, 0, (cudaStream_t)stream>>>(a->res, a->down, a->weights, a->shared_gate,
                                                                    a->shared, a->out, a->n, a->top_k, a->ld_down);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
