/*
 * Launch entry of softcap for the Gemma 4 Humanizer head: one kernel that computes
 * out = cap * tanh(x / cap) element-wise in f32 (cap 30 in the pinned model).
 * Launch-only: nothing here allocates, copies or synchronizes.
 */
#ifndef INFERENCE_CUDA_RESIDENT_SOFTCAP_H
#define INFERENCE_CUDA_RESIDENT_SOFTCAP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_softcap_args {
    void *out;       /* f32 [n], written */
    const void *x;   /* f32 [n], read */
    float cap;       /* finite and above 0 */
    int n;           /* values, a multiple of 32, 32..4194304 */
} inference_cuda_softcap_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_softcap_args) == 24 && offsetof(inference_cuda_softcap_args, out) == 0 && offsetof(inference_cuda_softcap_args, x) == 8 && offsetof(inference_cuda_softcap_args, cap) == 16 && offsetof(inference_cuda_softcap_args, n) == 20, "layout");
#else
_Static_assert(sizeof(inference_cuda_softcap_args) == 24 && offsetof(inference_cuda_softcap_args, out) == 0 && offsetof(inference_cuda_softcap_args, x) == 8 && offsetof(inference_cuda_softcap_args, cap) == 16 && offsetof(inference_cuda_softcap_args, n) == 20, "layout");
#endif

/* One launch on stream (must not be NULL). out is written; x is read.
 * Returns INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call for a bad argument,
 * else the launch status. */
int inference_cuda_softcap_launch(const inference_cuda_softcap_args *args, void *stream,
                                  int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_SOFTCAP_H */
