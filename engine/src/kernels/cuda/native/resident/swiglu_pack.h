/*
 * SwiGLU + fp16 pack entry (native/resident/swiglu_pack.h, per-kernel header).
 * Launch-only, like the entries of abi_resident.h. Include after abi.h in C++; Rust mirrors the
 * struct in native/resident/swiglu_pack.rs (child module of native/resident.rs).
 */
#ifndef INFERENCE_CUDA_SWIGLU_PACK_H
#define INFERENCE_CUDA_SWIGLU_PACK_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* For each of m rows: xf = fp16( silu(gate) * up ) in the layout kern::pack_f32 writes (fragment
 * order for m <= 16, with the data lanes and the zero tail entry written by the kernel itself
 * and the unread bytes left untouched; row-major fp16 [m][k_pad]
 * above). gu is f32 [m][2 * k_pad]: row t holds the gate in columns 0 .. k_pad - 1 and the up
 * values in columns k_pad .. 2 * k_pad - 1; the gate and up matmuls write only the first n_ff
 * columns of each half. For n_ff < k_pad the entry first zeroes columns n_ff .. k_pad - 1 of both
 * halves of every row (two cudaMemset2DAsync on the stream), so silu(0) * 0 = 0 fills the pack
 * tail; for n_ff == k_pad nothing is zeroed. gu is 4-byte aligned and modified only in those
 * tails; xf is 16-byte aligned, 65537 * 16 bytes for m <= 16 and m * k_pad * 2 bytes above. m in
 * 1..32768, k_pad a multiple of 256 in 256..32768, n_ff in 1..k_pad. gu and xf must not overlap.
 * stream must not be NULL. Returns INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call for a bad
 * argument; enqueues at most two 2-D memsets and one kernel; never synchronizes. */
typedef struct inference_cuda_swiglu_pack_args {
    void *gu;
    void *xf;
    int m;
    int n_ff;
    int k_pad;
} inference_cuda_swiglu_pack_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_swiglu_pack_args) == 32 && offsetof(inference_cuda_swiglu_pack_args, gu) == 0 && offsetof(inference_cuda_swiglu_pack_args, xf) == 8 && offsetof(inference_cuda_swiglu_pack_args, m) == 16 && offsetof(inference_cuda_swiglu_pack_args, n_ff) == 20 && offsetof(inference_cuda_swiglu_pack_args, k_pad) == 24,
              "inference_cuda_swiglu_pack_args layout");
#else
_Static_assert(sizeof(inference_cuda_swiglu_pack_args) == 32 && offsetof(inference_cuda_swiglu_pack_args, gu) == 0 && offsetof(inference_cuda_swiglu_pack_args, xf) == 8 && offsetof(inference_cuda_swiglu_pack_args, m) == 16 && offsetof(inference_cuda_swiglu_pack_args, n_ff) == 20 && offsetof(inference_cuda_swiglu_pack_args, k_pad) == 24,
               "inference_cuda_swiglu_pack_args layout");
#endif

int inference_cuda_swiglu_pack_launch(const inference_cuda_swiglu_pack_args *args, void *stream,
                                      int *out_stage);

#ifdef __cplusplus
}
#endif

#endif
