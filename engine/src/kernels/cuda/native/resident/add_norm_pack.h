/*
 * Add + RMSNorm + fp16 pack entry (native/resident/add_norm_pack.h, per-kernel header).
 * Launch-only, like the entries of abi_resident.h. Include after abi.h in C++; Rust mirrors the
 * struct in native/resident/add_norm_pack.rs (child module of native/resident.rs).
 */
#ifndef INFERENCE_CUDA_ADD_NORM_PACK_H
#define INFERENCE_CUDA_ADD_NORM_PACK_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* For each of m rows of n values: h = h + delta (when delta is not NULL; h is updated in place),
 * then norm = h * rsqrt(mean(h^2) + eps) * w, rounded to nearest even into the fp16 matmul input
 * xf (the layout kern::pack_f32 writes: fragment order for m <= 16, with the data lanes and the
 * zero tail entry the kernel itself writes and the bytes no consumer reads left untouched;
 * row-major fp16 [m][n] for m > 16). When norm_out is not NULL it also receives the f32 value
 * that xf rounds ([m][n], the router input). All pointers are device pointers. h, delta, w:
 * f32, 4-byte aligned; norm_out: f32, 16-byte aligned (the router reads it as its input and
 * needs 16); xf: 16-byte aligned, 65537 * 16 bytes for m <= 16 and m * n * 2 bytes above. n
 * is a multiple of 256 in 256..32768, m in 1..32768, eps finite and > 0. h, xf and norm_out are
 * written; no range may overlap another (w and delta may). stream must not be NULL (no
 * default-stream path). Returns INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call for a bad
 * argument; enqueues at most one kernel; never synchronizes. */
typedef struct inference_cuda_add_norm_pack_args {
    void *h;
    const void *delta;
    const void *w;
    void *xf;
    void *norm_out;
    int m;
    int n;
    float eps;
} inference_cuda_add_norm_pack_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_add_norm_pack_args) == 56 && offsetof(inference_cuda_add_norm_pack_args, h) == 0 && offsetof(inference_cuda_add_norm_pack_args, delta) == 8 && offsetof(inference_cuda_add_norm_pack_args, w) == 16 && offsetof(inference_cuda_add_norm_pack_args, xf) == 24 && offsetof(inference_cuda_add_norm_pack_args, norm_out) == 32 && offsetof(inference_cuda_add_norm_pack_args, m) == 40 && offsetof(inference_cuda_add_norm_pack_args, n) == 44 && offsetof(inference_cuda_add_norm_pack_args, eps) == 48,
              "inference_cuda_add_norm_pack_args layout");
#else
_Static_assert(sizeof(inference_cuda_add_norm_pack_args) == 56 && offsetof(inference_cuda_add_norm_pack_args, h) == 0 && offsetof(inference_cuda_add_norm_pack_args, delta) == 8 && offsetof(inference_cuda_add_norm_pack_args, w) == 16 && offsetof(inference_cuda_add_norm_pack_args, xf) == 24 && offsetof(inference_cuda_add_norm_pack_args, norm_out) == 32 && offsetof(inference_cuda_add_norm_pack_args, m) == 40 && offsetof(inference_cuda_add_norm_pack_args, n) == 44 && offsetof(inference_cuda_add_norm_pack_args, eps) == 48,
               "inference_cuda_add_norm_pack_args layout");
#endif

int inference_cuda_add_norm_pack_launch(const inference_cuda_add_norm_pack_args *args,
                                        void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif
