/*
 * Launch entry of the vendored add_rmsnorm_pack, patched to also store the f32 normalized row
 * (norm_out).
 *
 * The kernel fuses the residual add, the RMSNorm and the fp16 pack so the activations make
 * one trip through memory. The kernel is vendored xavier-qwen code (MIT); this file validates
 * and makes one kern:: call on the given stream; the kernel itself writes the zero tail entry
 * that the padding-token lanes read, so no whole-buffer memset is queued here.
 */
#include <cuda_runtime.h>
#include <cmath>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/add_norm_pack.h"

#include "kernels.h"
#include "ops.h"

namespace {

constexpr int kMaxRows = 32768;  // native.rs MAX_DIM
constexpr int kMaxWidth = 32768;  // kern::kMaxK

bool aligned(const void *p, size_t alignment) { return (uintptr_t)p % alignment == 0; }

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return (uintptr_t)a + a_bytes <= (uintptr_t)b || (uintptr_t)b + b_bytes <= (uintptr_t)a;
}

struct Extent {
    const void *p;
    size_t bytes;
};

bool shape_ok(const inference_cuda_add_norm_pack_args *in) {
    return in->m >= 1 && in->m <= kMaxRows && in->n >= 256 && in->n <= kMaxWidth &&
           in->n % 256 == 0 && std::isfinite(in->eps) && in->eps > 0.0f;
}

bool pointers_ok(const inference_cuda_add_norm_pack_args *in) {
    return in->h != NULL && in->w != NULL && in->xf != NULL && aligned(in->h, 4) &&
           aligned(in->w, 4) && aligned(in->xf, sizeof(uint4)) &&
           (in->delta == NULL || aligned(in->delta, 4)) &&
           (in->norm_out == NULL || aligned(in->norm_out, 16));
}

/* The written ranges (h, xf, norm_out) overlap no other range; w and delta may share bytes. */
bool disjoint(const inference_cuda_add_norm_pack_args *in, size_t xf_bytes) {
    const size_t row_bytes = (size_t)in->m * (size_t)in->n * sizeof(float);
    Extent written[3] = {{in->h, row_bytes}, {in->xf, xf_bytes}, {in->norm_out, row_bytes}};
    Extent read[2] = {{in->delta, row_bytes}, {in->w, (size_t)in->n * sizeof(float)}};
    for (int i = 0; i < 3; ++i) {
        if (written[i].p == NULL) continue;
        for (int j = i + 1; j < 3; ++j) {
            if (written[j].p != NULL && !apart(written[i].p, written[i].bytes, written[j].p, written[j].bytes))
                return false;
        }
        for (int j = 0; j < 2; ++j) {
            if (read[j].p != NULL && !apart(written[i].p, written[i].bytes, read[j].p, read[j].bytes))
                return false;
        }
    }
    return true;
}

bool valid(const inference_cuda_add_norm_pack_args *in, void *stream, size_t *xf_bytes) {
    if (in == NULL || stream == NULL || !shape_ok(in) || !pointers_ok(in)) return false;
    *xf_bytes = in->m <= kern::kMaxTokens ? kern::xf_vecs() * sizeof(uint4)
                                          : (size_t)in->m * (size_t)in->n * sizeof(__half);
    return disjoint(in, *xf_bytes);
}

}  // namespace

extern "C" int inference_cuda_add_norm_pack_launch(const inference_cuda_add_norm_pack_args *in,
                                                   void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        size_t xf_bytes = 0;
        if (!valid(in, stream, &xf_bytes)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const cudaStream_t s = (cudaStream_t)stream;
        // The kernel writes the zero tail entry the padding lanes read; the rest of the padding
        // bytes is never read by any consumer, so no whole-buffer memset runs here.
        Failure failure;
        failure.record(kern::add_rmsnorm_pack((float *)in->h, (const float *)in->delta,
                                              (const float *)in->w, in->m, in->n, in->eps,
                                              (uint4 *)in->xf, (float *)in->norm_out, s),
                       INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
