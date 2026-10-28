/*
 * Launch entry of rope_gemma for the Gemma 4 Humanizer resident step: split-half RoPE
 * applied in place to the q and k rows of one step. Local mode: head_dim 256, base 1e4,
 * freq[i] = base^(-2i/head_dim). Global mode: head_dim 512, base 1e6,
 * freq[i] = base^(-2i/head_dim) for i < 64 and 0 for i >= 64. An optional f32 factor table
 * divides the frequency; a factor >= 1e29 forces the angle to zero. The base position is one
 * int32 device word; row t rotates at base + t. Launch-only: nothing here allocates, copies or synchronizes.
 */
#ifndef INFERENCE_CUDA_RESIDENT_ROPE_GEMMA_H
#define INFERENCE_CUDA_RESIDENT_ROPE_GEMMA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_rope_gemma_args {
    void *q;              /* f32 [rows_q][n_q_head][head_dim], rotated in place */
    void *k;              /* f32 [rows_k][n_kv_head][head_dim], rotated in place */
    const void *pos;      /* one int32 device word, the base position (row t uses base + t) */
    const void *factors;  /* f32 [head_dim/2] or NULL */
    int rows_q;           /* 1..16 */
    int rows_k;           /* 1..16 */
    int n_q_head;         /* 16 */
    int n_kv_head;        /* 1 (global) or 8 (local) */
    int head_dim;         /* 256 (local) or 512 (global) */
    int global_mode;      /* 0 local, 1 global */
} inference_cuda_rope_gemma_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_rope_gemma_args) == 56 && offsetof(inference_cuda_rope_gemma_args, q) == 0 && offsetof(inference_cuda_rope_gemma_args, k) == 8 && offsetof(inference_cuda_rope_gemma_args, pos) == 16 && offsetof(inference_cuda_rope_gemma_args, factors) == 24 && offsetof(inference_cuda_rope_gemma_args, rows_q) == 32 && offsetof(inference_cuda_rope_gemma_args, rows_k) == 36 && offsetof(inference_cuda_rope_gemma_args, n_q_head) == 40 && offsetof(inference_cuda_rope_gemma_args, n_kv_head) == 44 && offsetof(inference_cuda_rope_gemma_args, head_dim) == 48 && offsetof(inference_cuda_rope_gemma_args, global_mode) == 52, "layout");
#else
_Static_assert(sizeof(inference_cuda_rope_gemma_args) == 56 && offsetof(inference_cuda_rope_gemma_args, q) == 0 && offsetof(inference_cuda_rope_gemma_args, k) == 8 && offsetof(inference_cuda_rope_gemma_args, pos) == 16 && offsetof(inference_cuda_rope_gemma_args, factors) == 24 && offsetof(inference_cuda_rope_gemma_args, rows_q) == 32 && offsetof(inference_cuda_rope_gemma_args, rows_k) == 36 && offsetof(inference_cuda_rope_gemma_args, n_q_head) == 40 && offsetof(inference_cuda_rope_gemma_args, n_kv_head) == 44 && offsetof(inference_cuda_rope_gemma_args, head_dim) == 48 && offsetof(inference_cuda_rope_gemma_args, global_mode) == 52, "layout");
#endif

/* One launch on stream (must not be NULL). q and k are rotated in place. Returns
 * INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call for a bad argument, else the
 * launch status. */
int inference_cuda_rope_gemma_launch(const inference_cuda_rope_gemma_args *args, void *stream,
                                     int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_ROPE_GEMMA_H */
