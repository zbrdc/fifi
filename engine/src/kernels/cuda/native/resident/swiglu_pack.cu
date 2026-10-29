/*
 * Launch entry of the vendored swiglu_pack, with the padded width k_pad.
 *
 * The activation is fused into the pack, as in add_norm_pack. The kernel is vendored xavier-qwen
 * code (MIT); this file validates, zeroes the padded tails of gu and makes one kern:: call on the
 * given stream; the kernel itself writes the zero tail entry of the fragment-order xf that the
 * padding-token lanes read, so no whole-buffer memset of xf is queued here.
 */
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/swiglu_pack.h"

#include "kernels.h"
#include "ops.h"

namespace {

constexpr int kMaxRows = 32768;   // native.rs MAX_DIM
constexpr int kMaxWidth = 32768;  // kern::kMaxK

bool aligned(const void *p, size_t alignment) { return (uintptr_t)p % alignment == 0; }

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return (uintptr_t)a + a_bytes <= (uintptr_t)b || (uintptr_t)b + b_bytes <= (uintptr_t)a;
}

bool shape_ok(const inference_cuda_swiglu_pack_args *in) {
    return in->m >= 1 && in->m <= kMaxRows && in->k_pad >= 256 && in->k_pad <= kMaxWidth &&
           in->k_pad % 256 == 0 && in->n_ff >= 1 && in->n_ff <= in->k_pad;
}

bool valid(const inference_cuda_swiglu_pack_args *in, void *stream, size_t *xf_bytes) {
    if (in == NULL || stream == NULL || in->gu == NULL || in->xf == NULL || !shape_ok(in) ||
        !aligned(in->gu, 4) || !aligned(in->xf, sizeof(uint4))) {
        return false;
    }
    *xf_bytes = in->m <= kern::kMaxTokens ? kern::xf_vecs() * sizeof(uint4)
                                          : (size_t)in->m * (size_t)in->k_pad * sizeof(__half);
    const size_t gu_bytes = (size_t)in->m * 2 * (size_t)in->k_pad * sizeof(float);
    return apart(in->gu, gu_bytes, in->xf, *xf_bytes);
}

/* Zeroes columns n_ff .. k_pad - 1 of the gate half and of the up half of every row. */
void zero_tails(const inference_cuda_swiglu_pack_args *in, cudaStream_t s, Failure &failure) {
    if (in->n_ff == in->k_pad) return;
    const size_t pitch = 2 * (size_t)in->k_pad * sizeof(float);
    const size_t width = (size_t)(in->k_pad - in->n_ff) * sizeof(float);
    char *base = (char *)in->gu;
    failure.record(cudaMemset2DAsync(base + (size_t)in->n_ff * sizeof(float), pitch, 0, width,
                                     (size_t)in->m, s),
                   INFERENCE_CUDA_STAGE_LAUNCH);
    if (failure.ok()) {
        failure.record(cudaMemset2DAsync(base + ((size_t)in->k_pad + in->n_ff) * sizeof(float),
                                         pitch, 0, width, (size_t)in->m, s),
                       INFERENCE_CUDA_STAGE_LAUNCH);
    }
}

}  // namespace

extern "C" int inference_cuda_swiglu_pack_launch(const inference_cuda_swiglu_pack_args *in,
                                                 void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        size_t xf_bytes = 0;
        if (!valid(in, stream, &xf_bytes)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const cudaStream_t s = (cudaStream_t)stream;
        Failure failure;
        zero_tails(in, s, failure);
        // The kernel writes the zero tail entry the padding lanes read; the rest of
        // the padding bytes is never read by any consumer, so no whole-buffer memset runs here.
        if (failure.ok()) {
            failure.record(
                kern::swiglu_pack((const float *)in->gu, in->m, in->k_pad, (uint4 *)in->xf, s),
                INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
