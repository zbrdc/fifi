/*
 * Launch entry of gelu_ffn for the Gemma 4 Humanizer resident step: one kernel that
 * computes out = 0.5*g*(1 + tanh(sqrt(2/pi)*(g + 0.044715*g^3)))*u row-wise in f32.
 * Launch-only: nothing here allocates, copies or synchronizes.
 */
#ifndef INFERENCE_CUDA_RESIDENT_GELU_FFN_H
#define INFERENCE_CUDA_RESIDENT_GELU_FFN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_gelu_ffn_args {
    void *out;        /* f32 [m][n], written */
    const void *gate; /* f32 [m][n], read */
    const void *up;   /* f32 [m][n], read */
    int m;            /* rows, 1..65535 */
    int n;            /* values per row, a multiple of 32, 32..65536 */
} inference_cuda_gelu_ffn_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_gelu_ffn_args) == 32 && offsetof(inference_cuda_gelu_ffn_args, out) == 0 && offsetof(inference_cuda_gelu_ffn_args, gate) == 8 && offsetof(inference_cuda_gelu_ffn_args, up) == 16 && offsetof(inference_cuda_gelu_ffn_args, m) == 24 && offsetof(inference_cuda_gelu_ffn_args, n) == 28, "layout");
#else
_Static_assert(sizeof(inference_cuda_gelu_ffn_args) == 32 && offsetof(inference_cuda_gelu_ffn_args, out) == 0 && offsetof(inference_cuda_gelu_ffn_args, gate) == 8 && offsetof(inference_cuda_gelu_ffn_args, up) == 16 && offsetof(inference_cuda_gelu_ffn_args, m) == 24 && offsetof(inference_cuda_gelu_ffn_args, n) == 28, "layout");
#endif

/* One launch on stream (must not be NULL). out is written; gate and up are read.
 * Returns INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call for a bad argument,
 * else the launch status. */
int inference_cuda_gelu_ffn_launch(const inference_cuda_gelu_ffn_args *args, void *stream,
                                   int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_GELU_FFN_H */
