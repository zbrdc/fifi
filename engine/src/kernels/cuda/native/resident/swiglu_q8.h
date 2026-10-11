/*
 * MATVEC-PAR S3a K2: the batched SwiGLU + q8_1 pack entry (native/resident/swiglu_q8.h, per-kernel
 * header of the layout in kernels/cuda/resident/mod.rs). Launch-only, like the other resident
 * headers: nothing here allocates, copies, synchronizes or memsets; a fault in queued work
 * surfaces at the next synchronize. abi_resident.h stays closed.
 */
#ifndef INFERENCE_CUDA_RESIDENT_SWIGLU_Q8_H
#define INFERENCE_CUDA_RESIDENT_SWIGLU_Q8_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* For each of `slots` expert-slot rows of `gu` (slot s at gu + s * y_slot_stride ELEMENTS, the
 * A2 [slot][gate|up] layout of the step's ExpertGu): q8 row s = the block_q8_1 quantization of
 * silu(gate) * up over the n_ff live values, with the vendored quantize_q8_1 block rule (per 32
 * values amax, d = amax/127, q = amax == 0 ? 0 : roundf(x/d), ds = half2(d, sum x) with the warp
 * butterfly sum) computed in IEEE arithmetic — this entry's translation unit is in the
 * warnings-into-errors group WITHOUT -use_fast_math, unlike the vendored quantize.cu object —
 * except silu itself, which keeps swiglu_pack_k's fast form x / (1 + __expf(-x)) (the E2 numerics
 * domain of the swiglu_pack doc; __expf stays by decision). Each q8 row is padded to
 * MATRIX_ROW_PADDING (512) values; the tail blocks past n_ff are written as zeros, so
 * q8 holds slots * pad(n_ff, 512)/32 blocks of 36 bytes back to back. ONE kernel, grid = slots,
 * 256 threads; gu is read only (no tail zeroing: the kernel never reads past n_ff of each half).
 * n_ff is a multiple of 32 in 32..32768, slots in 1..8, y_slot_stride >= 2*n_ff. gu and q8 are
 * 4-byte aligned and must not overlap. stream must not be NULL. Returns
 * INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call for a bad argument. */
typedef struct inference_cuda_swiglu_q8_args {
    const void *gu;       /* f32 [slots][y_slot_stride], read only */
    void *q8;             /* block_q8_1 out: slots * pad(n_ff, 512)/32 * 36 bytes */
    int n_ff;             /* live values per slot half, a multiple of 32 */
    int y_slot_stride;    /* elements between slot rows, >= 2 * n_ff */
    int slots;            /* expert slots, 1..8 */
} inference_cuda_swiglu_q8_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_swiglu_q8_args) == 32 && offsetof(inference_cuda_swiglu_q8_args, gu) == 0 && offsetof(inference_cuda_swiglu_q8_args, q8) == 8 && offsetof(inference_cuda_swiglu_q8_args, n_ff) == 16 && offsetof(inference_cuda_swiglu_q8_args, y_slot_stride) == 20 && offsetof(inference_cuda_swiglu_q8_args, slots) == 24,
              "inference_cuda_swiglu_q8_args layout");
#else
_Static_assert(sizeof(inference_cuda_swiglu_q8_args) == 32 && offsetof(inference_cuda_swiglu_q8_args, gu) == 0 && offsetof(inference_cuda_swiglu_q8_args, q8) == 8 && offsetof(inference_cuda_swiglu_q8_args, n_ff) == 16 && offsetof(inference_cuda_swiglu_q8_args, y_slot_stride) == 20 && offsetof(inference_cuda_swiglu_q8_args, slots) == 24,
               "inference_cuda_swiglu_q8_args layout");
#endif

int inference_cuda_swiglu_q8_launch(const inference_cuda_swiglu_q8_args *args, void *stream,
                                    int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_SWIGLU_Q8_H */
