/*
 * Entries of kern::matmul_expert on an explicit stream. Glue only: no kernel and no device logic.
 * Wraps the vendored kern::matmul_expert (kernels.h) and asks mma884::split_k (mma884.cuh) whether a
 * shape splits along k, so the split rule has one owner. Every exported function has C linkage, takes
 * and returns C types only, and wraps its body in try/catch, like the projector adapter.
 */
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "kernels.h"
#include "mma884.cuh"
#include "repack.h"
#include "resident/matmul_expert.h"

namespace {

/* kern::matmul_expert takes at most this many tokens (kernels.h). */
const int kMaxExpertTokens = 8;

bool tiled_type(int ggml_type) {
    return ggml_type == INFERENCE_CUDA_TYPE_Q4_K || ggml_type == INFERENCE_CUDA_TYPE_Q6_K ||
           ggml_type == INFERENCE_CUDA_TYPE_Q8_0;
}

bool aligned(const void *p, size_t alignment) { return (uintptr_t)p % alignment == 0; }

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return (uintptr_t)a + a_bytes <= (uintptr_t)b || (uintptr_t)b + b_bytes <= (uintptr_t)a;
}

bool expert_shape_ok(int m, int n_rows, int k) {
    return m >= 1 && m <= kMaxExpertTokens && n_rows > 0 && n_rows % 8 == 0 && k > 0 &&
           k % mma884::kTileK == 0 && k <= kern::kMaxK &&
           mma884::split_k(n_rows / 8, k / mma884::kTileK) == 1;
}

}  // namespace

extern "C" int inference_cuda_expert_shape_ok(int m, int n_rows, int k) {
    try {
        return expert_shape_ok(m, n_rows, k) ? INFERENCE_CUDA_OK : INFERENCE_CUDA_INVALID_ARGUMENT;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_matmul_expert_launch(int ggml_type, int m, const void *stack,
                                                   const int *ids_dev, int slot, int count,
                                                   const void *xf_dev, float *y_dev, int ldy,
                                                   int n_rows, int k, void *stream,
                                                   int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (stack == NULL || ids_dev == NULL || xf_dev == NULL || y_dev == NULL || stream == NULL ||
            !tiled_type(ggml_type) || !expert_shape_ok(m, n_rows, k) || slot < 0 || count < 1 ||
            ldy < n_rows || !aligned(stack, sizeof(uint4)) || !aligned(xf_dev, sizeof(uint4)) ||
            !aligned(ids_dev, sizeof(int)) || !aligned(y_dev, sizeof(float))) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        // y is written while xf, the experts' tiles and the ids are read; the guard needs the byte
        // counts, and count * expert_bytes must not wrap.
        const size_t expert_bytes = tiled_vecs((GgmlType)ggml_type, n_rows, k) * sizeof(uint4);
        if (expert_bytes == 0 || (size_t)count > SIZE_MAX / expert_bytes) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const size_t stack_bytes = expert_bytes * (size_t)count;
        const size_t xf_bytes = kern::xf_vecs() * sizeof(uint4);
        const size_t ids_bytes = ((size_t)slot + 1) * sizeof(int);
        const size_t y_bytes = (((size_t)(m - 1) * (size_t)ldy) + (size_t)n_rows) * sizeof(float);
        if (!apart(y_dev, y_bytes, xf_dev, xf_bytes) || !apart(y_dev, y_bytes, stack, stack_bytes) ||
            !apart(y_dev, y_bytes, ids_dev, ids_bytes)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        const kern::ExpertSelect select = {ids_dev, slot, count};
        failure.record(kern::matmul_expert((GgmlType)ggml_type, m, (const uint4 *)stack, select,
                                           (const uint4 *)xf_dev, y_dev, ldy, n_rows, k,
                                           (cudaStream_t)stream),
                       INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

#ifdef INFERENCE_CUDA_FAULT_QUALIFICATION
extern "C" int inference_cuda_fault_expert_rejections(unsigned int *out_mask, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (out_mask == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_mask = 0;
        int id = 0;  // never read: a rejected call launches nothing
        struct Case {
            GgmlType type;
            int m;
            kern::ExpertSelect select;
            int n_rows;
            int k;
        };
        const kern::ExpertSelect good = {&id, 0, 4};
        const kern::ExpertSelect no_ids = {static_cast<const int *>(NULL), 0, 4};
        const Case cases[] = {
            {GgmlType::Q4_K, 1, good, 0, 256},               // 0: n_rows == 0 (split 1 at k = 256)
            {GgmlType::Q4_K, 1, good, -8, 256},              // 1: n_rows < 0
            {GgmlType::Q4_K, 1, good, 8, 0},                 // 2: k == 0
            {GgmlType::Q4_K, 1, good, 8, -256},              // 3: k < 0
            {GgmlType::Q4_K, 1, good, 32768, 33024},         // 4: k above kMaxK, 4096 warps so split 1
            {GgmlType::Q4_K, 1, no_ids, 8, 256},             // 5: null ids
            {GgmlType::Q4_K, 1, {&id, 0, 0}, 8, 256},        // 6: count 0
            {GgmlType::Q4_K, 1, {&id, -1, 4}, 8, 256},       // 7: slot < 0
            {GgmlType::Q4_K, 0, good, 8, 256},               // 8: m == 0
            {GgmlType::Q4_K, 9, good, 8, 256},               // 9: m > 8
            {GgmlType::Q4_K, 1, good, 12, 256},              // 10: n_rows % 8
            {GgmlType::Q4_K, 1, good, 8, 300},               // 11: k % 256
            {(GgmlType)99, 1, good, 8, 256},                 // 12: untiled type
            {GgmlType::Q4_K, 1, good, 64, 4096},             // 13: splits along k
        };
        Failure failure;
        for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
            const Case &c = cases[i];
            if (kern::matmul_expert(c.type, c.m, NULL, c.select, NULL, NULL, c.n_rows, c.n_rows,
                                    c.k, NULL) == cudaErrorInvalidValue) {
                *out_mask |= 1u << i;
            }
        }
        failure.record(cudaDeviceSynchronize(), INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
#endif
