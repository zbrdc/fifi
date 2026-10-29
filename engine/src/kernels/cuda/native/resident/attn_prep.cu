/* Argument validation and the one launch of the vendored attn_prep (with its poison-guard
 * patch). No kernel is written here.
 * Contract: resident/attn_prep.h. */
#include "resident/attn_prep.h"

#include <math.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "ops.h"

namespace {

constexpr int kHead = 256;  // the vendored kernel's only head size (kern::kHeadDim)
constexpr int kMaxHeads = 256;
constexpr int kMaxKvHeads = 64;
constexpr int kMaxGridY = 65535;
constexpr int kMaxRows = (1 << 30) - 1;
// Mask 4 of StepParams.status (invalid step state); equals INFERENCE_CUDA_STATUS_INVALID_STEP_STATE
// of resident/step_params.h. Repeated here so that this file does not depend on that header.
constexpr int kStatusInvalidStepState = 4;

bool aligned_to(const void *p, uintptr_t a) { return ((uintptr_t)p & (a - 1)) == 0; }

bool valid_pointers(const inference_cuda_attn_prep_args *a) {
    return a->proj != NULL && a->q_norm != NULL && a->k_norm != NULL && a->k_cache != NULL &&
           a->v_cache != NULL && a->q != NULL && a->pos != NULL && a->status != NULL && aligned_to(a->proj, 4) &&
           aligned_to(a->q_norm, 4) && aligned_to(a->k_norm, 4) && aligned_to(a->pos, 4) &&
           aligned_to(a->status, 4) &&
           aligned_to(a->q, 16) && aligned_to(a->k_cache, 16) && aligned_to(a->v_cache, 16);
}

/* The configurations the vendored kernel does not support (the Mellum ones) are refused here. */
bool supported_configuration(const inference_cuda_attn_prep_args *a) {
    return a->head_dim == kHead && a->has_gate == 1 && a->slot_mode == INFERENCE_CUDA_SLOT_POSITION &&
           a->slot_ring == 0;
}

bool valid_shape(const inference_cuda_attn_prep_args *a) {
    const bool heads = a->n_head >= 1 && a->n_head <= kMaxHeads && a->n_kv_head >= 1 &&
                       a->n_kv_head <= kMaxKvHeads;
    const bool tokens = a->m >= 1 && a->m <= kMaxGridY;
    const bool rope = a->n_rot >= 2 && a->n_rot <= kHead &&
                      (a->n_rot & (a->n_rot - 1)) == 0;  // a power of two
    const bool cache = a->rows >= a->m && a->rows <= kMaxRows;
    const long long row_width = ((long long)a->n_head * 2 + (long long)a->n_kv_head * 2) * kHead;
    return heads && tokens && rope && cache && a->ld >= row_width;
}

bool valid_scalars(const inference_cuda_attn_prep_args *a) {
    return isfinite(a->eps) && a->eps > 0.0f && isfinite(a->rope_base) && a->rope_base > 1.0f;
}

bool valid_attn_prep(const inference_cuda_attn_prep_args *a) {
    return a != NULL && valid_pointers(a) && supported_configuration(a) && valid_shape(a) &&
           valid_scalars(a);
}

}  // namespace

extern "C" int inference_cuda_attn_prep_launch(const inference_cuda_attn_prep_args *a, void *stream,
                                               int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid_attn_prep(a)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        kern::AttnArgs k = {};
        k.proj = (const float *)a->proj;
        k.ld = a->ld;
        k.n_head = a->n_head;
        k.n_kv_head = a->n_kv_head;
        k.n_rot = a->n_rot;
        k.q_norm = (const float *)a->q_norm;
        k.k_norm = (const float *)a->k_norm;
        k.eps = a->eps;
        k.rope_base = a->rope_base;
        k.k_cache = (__half *)a->k_cache;
        k.v_cache = (__half *)a->v_cache;
        k.q = (float *)a->q;
        k.m = a->m;
        k.pos_dev = (const int *)a->pos;
        k.rows = a->rows;
        k.status_dev = (int *)a->status;
        k.status_bits = kStatusInvalidStepState;
        /* The launch status is the only status: attn_prep returns cudaGetLastError(). */
        Failure failure;
        failure.record(kern::attn_prep(k, (cudaStream_t)stream), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

/* The f32 rows entry of the same kernel (the Int8G32 KV handoff), for its Rust wrapper.
 * Contract: resident/attn_prep.h. */
namespace {

constexpr int kMaxDecodeTokens = 16;  // kv8_append's decode bound: rows 1..=16 only

/* Every rule of the fp16 entry's valid_attn_prep with the rows buffers in place of the caches, the
 * cell count in place of the cache rows and the decode bound in place of the grid-y bound, one
 * conjunct per guard. The null test comes first, as its own early return, because the pointer
 * conjuncts read fields of *a; the fp16 entry leads with the same test as its first conjunct. */
bool valid_attn_prep_rows(const inference_cuda_attn_prep_rows_args *a) {
    if (a == NULL) return false;
    const bool pointers = a->proj != NULL && a->q_norm != NULL && a->k_norm != NULL &&
                          a->k_rows != NULL && a->v_rows != NULL && a->q != NULL && a->pos != NULL &&
                          a->status != NULL && aligned_to(a->proj, 4) && aligned_to(a->q_norm, 4) &&
                          aligned_to(a->k_norm, 4) && aligned_to(a->pos, 4) &&
                          aligned_to(a->status, 4) && aligned_to(a->q, 16) &&
                          aligned_to(a->k_rows, 16) && aligned_to(a->v_rows, 16);
    const bool config = a->head_dim == kHead && a->has_gate == 1 &&
                        a->slot_mode == INFERENCE_CUDA_SLOT_POSITION && a->slot_ring == 0;
    const bool heads = a->n_head >= 1 && a->n_head <= kMaxHeads && a->n_kv_head >= 1 &&
                       a->n_kv_head <= kMaxKvHeads;
    const bool tokens = a->m >= 1 && a->m <= kMaxDecodeTokens;
    const bool rope = a->n_rot >= 2 && a->n_rot <= kHead &&
                      (a->n_rot & (a->n_rot - 1)) == 0;  // a power of two
    const bool cells = a->cells >= a->m && a->cells <= kMaxRows;
    const long long row_width = ((long long)a->n_head * 2 + (long long)a->n_kv_head * 2) * kHead;
    return pointers && config && heads && tokens && rope && cells &&
           a->ld >= row_width && isfinite(a->eps) && a->eps > 0.0f && isfinite(a->rope_base) &&
           a->rope_base > 1.0f;
}

}  // namespace

extern "C" int inference_cuda_attn_prep_rows_launch(const inference_cuda_attn_prep_rows_args *a,
                                                    void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid_attn_prep_rows(a)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        kern::AttnArgs k = {};
        k.proj = (const float *)a->proj;
        k.ld = a->ld;
        k.n_head = a->n_head;
        k.n_kv_head = a->n_kv_head;
        k.n_rot = a->n_rot;
        k.q_norm = (const float *)a->q_norm;
        k.k_norm = (const float *)a->k_norm;
        k.eps = a->eps;
        k.rope_base = a->rope_base;
        k.k_rows = (float *)a->k_rows;
        k.v_rows = (float *)a->v_rows;
        k.q = (float *)a->q;
        k.m = a->m;
        k.pos_dev = (const int *)a->pos;
        k.rows = a->cells;
        k.status_dev = (int *)a->status;
        k.status_bits = kStatusInvalidStepState;
        /* The launch status is the only status: attn_prep returns cudaGetLastError(). */
        Failure failure;
        failure.record(kern::attn_prep(k, (cudaStream_t)stream), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
