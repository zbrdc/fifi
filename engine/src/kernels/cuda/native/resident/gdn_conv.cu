// Launch shim for the vendored conv_silu_k (kern::conv_silu in ops.cu): it proves the arguments
// the kernel does not check and makes one extern "C" call. The kernel is vendored xavier-qwen
// code. The one kernel of this file, conv_silu_rows_k, is the prefill variant of it: the same
// per-element arithmetic, split over rows so that a long prefill launch fills the device.
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "ops.h"
#include "resident/gdn_conv.h"
#include "resident/gdn_mutants.h"

namespace {

constexpr int kMaxM = 32768;
constexpr int kMaxChannels = 65536;
constexpr int kMaxLd = 1 << 20;

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }

// True when [a, a + a_bytes) and [b, b + b_bytes) share no byte.
bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    uintptr_t pa = (uintptr_t)a, pb = (uintptr_t)b;
    return pa + a_bytes <= pb || pb + b_bytes <= pa;
}

// Mirrors the Rust wrapper term by term; a term added here needs its twin there. The Rust side
// refuses everything this refuses, so a C refusal means a wrapper defect.
bool valid(const inference_cuda_gdn_conv_args &a, void *stream) {
    if (a.proj == NULL || a.w == NULL || a.state == NULL || a.out == NULL || stream == NULL) {
        return false;
    }
    if (a.m < 1 || a.m > kMaxM || a.channels < 1 || a.channels > kMaxChannels ||
        a.ld < a.channels || a.ld > kMaxLd) {
        return false;
    }
    if (!aligned(a.proj, 4) || !aligned(a.state, 4) || !aligned(a.out, 4) || !aligned(a.w, 16)) {
        return false;
    }
    const size_t n = (size_t)a.channels;
    const size_t proj_bytes = (((size_t)a.m - 1) * (size_t)a.ld + n) * sizeof(float);
    const size_t w_bytes = n * 4 * sizeof(float);
    const size_t state_bytes = 3 * n * sizeof(float);
    const size_t out_bytes = (size_t)a.m * n * sizeof(float);
    // state and out are written; proj and w are read. Reads may share bytes with each other.
    return apart(a.state, state_bytes, a.out, out_bytes) &&
           apart(a.state, state_bytes, a.proj, proj_bytes) &&
           apart(a.state, state_bytes, a.w, w_bytes) && apart(a.out, out_bytes, a.proj, proj_bytes) &&
           apart(a.out, out_bytes, a.w, w_bytes);
}

// The snapshot range of the strided entry: [snap, snap + snap_bytes) shares no byte with any range
// the kernel reads or writes, and holds the m - 1 snapshots of 12 * channels bytes at snap_stride.
bool valid_snap(const inference_cuda_gdn_conv_snap_args &s) {
    const inference_cuda_gdn_conv_args &a = s.conv;
    const size_t n = (size_t)a.channels;
    const size_t row = 3 * n * sizeof(float);
    if (s.snap == NULL || a.m < 2 || !aligned(s.snap, 4) || s.snap_stride % 4 != 0 ||
        s.snap_stride < row) {
        return false;
    }
    const size_t reach = ((size_t)a.m - 2) * s.snap_stride + row;
    if (s.snap_bytes < reach) {
        return false;
    }
    const size_t proj_bytes = (((size_t)a.m - 1) * (size_t)a.ld + n) * sizeof(float);
    return apart(s.snap, reach, a.state, row) && apart(s.snap, reach, a.out, (size_t)a.m * n * sizeof(float)) &&
           apart(s.snap, reach, a.proj, proj_bytes) && apart(s.snap, reach, a.w, n * 4 * sizeof(float));
}

constexpr int kBlock = 256;

// silu as in the vendored ops.cu (the same expression, so the same instructions under the same
// build flags); the device tests compare the two kernels bit for bit.
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + __expf(-x)); }

// The prefill variant of conv_silu_k. One thread owns channel c for the rows [t0, t0 + R) of its
// CTA row, t0 = blockIdx.y * R. The window entering row t is the three inputs of rows t - 3 .. t - 1,
// which are proj rows (or, before row 0, the state), so the rows need no carried register chain:
// a thread loads its R inputs and the three before them, then computes the R outputs with the
// expression of conv_silu_k, term for term. Every output element therefore sees the same four
// products in the same order as in the original kernel.
//
// The state after the m rows is the last three inputs (or the state shifted, for m < 3). Only the
// CTA row 0 reads the state (R >= 3 gives every other CTA row a full window in proj), and the same
// thread of that CTA row writes it back, after reading it, so no CTA reads what another writes.
template <int R>
__global__ void __launch_bounds__(kBlock)
conv_silu_rows_k(const float *__restrict__ proj, int ld, int m, int n,
                 const float4 *__restrict__ w, float *state, float *__restrict__ out GDN_MUTANT_PARAM) {
#ifndef INFERENCE_CUDA_GDN_MUTANTS
    constexpr int mutant = 0;  // a mutant branch is the constant false and compiles away
#endif
    static_assert(R >= 3, "a CTA row after the first must find its window in proj");
    const int c = blockIdx.x * kBlock + threadIdx.x;
    if (mutant == 21 ? c > n : c >= n) return;
    const int t0 = blockIdx.y * R;
    const int rows = min(R, m - t0);
    float x[R + 3];
    // x[0..2]: the inputs of rows t0 - 3 .. t0 - 1; a negative row is a state row (3 + row).
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int r = t0 - 3 + k;
        x[k] = r >= 0 ? proj[(size_t)r * ld + c]
                      : state[(size_t)(mutant == 3 ? -r - 1 : 3 + r) * n + c];
    }
#pragma unroll
    for (int j = 0; j < R; ++j) {
        // A dead row (j >= rows) leaves x[3 + j] unset: the compute loop below skips it, so it is never read.
        if (j < rows + (mutant == 19 ? 1 : 0)) x[3 + j] = proj[(size_t)(t0 + j) * ld + c];
    }
    const float4 wc = w[c];
#pragma unroll
    for (int j = 0; j < R; ++j) {
        if (j < rows) {
            // conv_silu_k's expression, with the contraction it gets from the compiler written
            // out: the first product rounded, then three fused multiply-adds in tap order. As a
            // plain expression the compiler fuses the second product first in this kernel, which
            // rounds differently.
            float v = __fmaf_rn(x[j + 3], wc.w,
                                __fmaf_rn(x[j + 2], wc.z,
                                          __fmaf_rn(x[j + 1], wc.y, __fmul_rn(x[j], wc.x))));
            if (mutant == 1) v = x[j] * wc.x + x[j + 1] * wc.y + x[j + 2] * wc.z + x[j + 3] * wc.w;
            if (!(mutant == 5 && j == rows - 1)) out[(size_t)(t0 + j) * n + c] = silu_f(v);
        }
    }
    if (mutant == 4 ? blockIdx.y == 0 : blockIdx.y != 0) return;
    float f0, f1, f2;
    if (m >= 3) {
        f0 = proj[(size_t)(m - 3) * ld + c];
        f1 = proj[(size_t)(m - 2) * ld + c];
        f2 = proj[(size_t)(m - 1) * ld + c];
    } else if (m == 2) {
        f0 = x[2];
        f1 = proj[c];
        f2 = proj[(size_t)ld + c];
    } else {
        f0 = x[1];
        f1 = x[2];
        f2 = proj[c];
    }
    if (mutant == 2) {
        const float swapped = f0;
        f0 = f1;
        f1 = swapped;
    }
    state[c] = f0;
    state[n + c] = f1;
    state[2 * n + c] = f2;
}

template <int R>
cudaError_t launch_rows(const inference_cuda_gdn_conv_args &a, cudaStream_t stream) {
    const dim3 grid((a.channels + kBlock - 1) / kBlock, (a.m + R - 1) / R);
    conv_silu_rows_k<R><<<grid, kBlock, 0, stream>>>(
        (const float *)a.proj, a.ld, a.m, a.channels, (const float4 *)a.w, (float *)a.state,
        (float *)a.out GDN_MUTANT_ARG);
    return cudaGetLastError();
}

}  // namespace

extern "C" int inference_cuda_gdn_conv_launch(const inference_cuda_gdn_conv_args *args,
                                              void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid(*args, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        // snap stays null: this entry does not use the speculative-rollback snapshots.
        kern::ConvArgs call = {(const float *)args->proj, args->ld,   args->m,
                               args->channels,            (const float *)args->w,
                               (float *)args->state,      NULL,       (float *)args->out};
        Failure failure;
        failure.record(kern::conv_silu(call, (cudaStream_t)stream), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_gdn_conv_snap_launch(const inference_cuda_gdn_conv_snap_args *args,
                                                   void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid(args->conv, stream) || !valid_snap(*args)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const inference_cuda_gdn_conv_args &a = args->conv;
        kern::ConvArgs call = {(const float *)a.proj, a.ld,  a.m,
                               a.channels,            (const float *)a.w,
                               (float *)a.state,      (float *)args->snap, (float *)a.out,
                               args->snap_stride};
        Failure failure;
        failure.record(kern::conv_silu(call, (cudaStream_t)stream), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_gdn_conv_rows_launch(const inference_cuda_gdn_conv_args *args,
                                                   void *stream, int *out_stage,
                                                   int rows_per_thread) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid(*args, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        cudaError_t launched;
        switch (rows_per_thread) {
        case 16: launched = launch_rows<16>(*args, (cudaStream_t)stream); break;
        case 32: launched = launch_rows<32>(*args, (cudaStream_t)stream); break;
        case 64: launched = launch_rows<64>(*args, (cudaStream_t)stream); break;
        default: return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        failure.record(launched, INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

#ifdef INFERENCE_CUDA_GDN_MUTANTS
std::atomic<int> g_gdn_mutant{0};

extern "C" int inference_cuda_gdn_mutant_select(int id) {
    if (id < 0 || id > 31) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    g_gdn_mutant.store(id, std::memory_order_relaxed);
    return INFERENCE_CUDA_OK;
}
#endif
