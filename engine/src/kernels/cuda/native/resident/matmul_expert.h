/*
 * Launch entry of kern::matmul_expert (RD-0b): the tiled matmul against one expert of a stack, with the
 * expert id read from device memory when the kernel runs. Launch-only, like abi_resident.h: nothing here
 * allocates, copies or synchronizes; a fault in queued work surfaces at the next synchronize. This header
 * is the per-kernel header of the layout in kernels/cuda/resident/mod.rs; abi_resident.h stays closed.
 */
#ifndef INFERENCE_CUDA_RESIDENT_MATMUL_EXPERT_H
#define INFERENCE_CUDA_RESIDENT_MATMUL_EXPERT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* y[t * ldy + r] = (x W^T)[t][r] for m <= 8 tokens against expert ids_dev[slot] of `stack`
 * (kern::matmul_expert), on stream. stack is the 16-byte-aligned tiled weight of `count` experts of
 * n_rows x k each, back to back. xf_dev is the fragment-order buffer that inference_cuda_pack_launch
 * wrote, 16-byte aligned and xf-sized for m <= 16. ids_dev is a 4-byte-aligned device array of at
 * least slot + 1 ints that the kernel reads when it runs, so a captured graph needs no host-known id.
 * An id outside [0, count) writes NaN to y and reads no weight. y_dev is 4-byte aligned, ldy >= n_rows,
 * and must not overlap xf, the count experts of the stack, or ids_dev[0..slot]. stream must not be
 * NULL: the entry never falls back to the default stream. The shape must be one that
 * inference_cuda_expert_shape_ok accepts. A rejected argument returns INFERENCE_CUDA_INVALID_ARGUMENT
 * before any CUDA call; otherwise the launch status of the kernel. */
int inference_cuda_matmul_expert_launch(int ggml_type, int m, const void *stack, const int *ids_dev,
                                        int slot, int count, const void *xf_dev, float *y_dev,
                                        int ldy, int n_rows, int k, void *stream, int *out_stage);
/* Host only, no CUDA call: INFERENCE_CUDA_OK when m tokens against an expert of n_rows x k are a shape
 * that inference_cuda_matmul_expert_launch accepts (m in [1, 8], a tiled shape, and no split along k),
 * else INFERENCE_CUDA_INVALID_ARGUMENT. */
int inference_cuda_expert_shape_ok(int m, int n_rows, int k);
#ifdef INFERENCE_CUDA_FAULT_QUALIFICATION
/* Fault qualification only, host only: calls kern::matmul_expert directly with fourteen invalid
 * argument sets (null pointers elsewhere, nothing launches) and sets bit i of *out_mask for each case
 * that returned cudaErrorInvalidValue. Afterwards it synchronizes, so a case that launched anyway
 * shows as a fault. */
int inference_cuda_fault_expert_rejections(unsigned int *out_mask, int *out_stage);
#endif

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_MATMUL_EXPERT_H */
