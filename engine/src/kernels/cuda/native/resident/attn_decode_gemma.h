/*
 * Launch entry of attn_decode_gemma for the Gemma 4 Humanizer resident step: one
 * single-pass decode attention over the KV8 planes (int8 codes, fp16 scale bits per
 * 32-value group). Scale is a parameter (Gemma uses 1.0); GQA maps query head h to
 * kv head h*n_kv_head/n_q_head; head_dim is 256 (local, window 1024) or 512 (global,
 * window 0); the device kv_len word is authoritative: token t sees keys 0..kv_len+t-1
 * cut to the last window positions; out is NaN when kv_len is out of range. No output
 * gate. v0 is single-pass: no split-K. Launch-only; faults surface at synchronize.
 */
#ifndef INFERENCE_CUDA_RESIDENT_ATTN_DECODE_GEMMA_H
#define INFERENCE_CUDA_RESIDENT_ATTN_DECODE_GEMMA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_attn_decode_gemma_args {
    void *out;               /* f32 [m][n_q_head][head_dim], written */
    void *status;            /* u32 word, OR-ed with 4 on an invalid kv_len, or NULL */
    const void *q;           /* f32 [m][n_q_head][head_dim] */
    const void *k_codes;     /* i8 [n_kv_head][cells][head_dim] */
    const void *k_scales;    /* f16 bits [n_kv_head][cells][head_dim/32] */
    const void *v_codes;     /* i8 [n_kv_head][cells][head_dim] */
    const void *v_scales;    /* f16 bits [n_kv_head][cells][head_dim/32] */
    const void *kv_len;      /* one i32 device word */
    int m;                   /* tokens, 1..16 */
    int n_q_head;            /* 16 */
    int n_kv_head;           /* 1 (global) or 8 (local) */
    int head_dim;            /* 256 or 512 */
    int cells;               /* KV capacity per head */
    int window;              /* 0 all keys, else the last W keys */
    float scale;             /* |scale| < 1e30, Gemma uses 1.0 */
} inference_cuda_attn_decode_gemma_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_attn_decode_gemma_args) == 96 && offsetof(inference_cuda_attn_decode_gemma_args, out) == 0 && offsetof(inference_cuda_attn_decode_gemma_args, status) == 8 && offsetof(inference_cuda_attn_decode_gemma_args, q) == 16 && offsetof(inference_cuda_attn_decode_gemma_args, k_codes) == 24 && offsetof(inference_cuda_attn_decode_gemma_args, k_scales) == 32 && offsetof(inference_cuda_attn_decode_gemma_args, v_codes) == 40 && offsetof(inference_cuda_attn_decode_gemma_args, v_scales) == 48 && offsetof(inference_cuda_attn_decode_gemma_args, kv_len) == 56 && offsetof(inference_cuda_attn_decode_gemma_args, m) == 64 && offsetof(inference_cuda_attn_decode_gemma_args, n_q_head) == 68 && offsetof(inference_cuda_attn_decode_gemma_args, n_kv_head) == 72 && offsetof(inference_cuda_attn_decode_gemma_args, head_dim) == 76 && offsetof(inference_cuda_attn_decode_gemma_args, cells) == 80 && offsetof(inference_cuda_attn_decode_gemma_args, window) == 84 && offsetof(inference_cuda_attn_decode_gemma_args, scale) == 88, "layout");
#else
_Static_assert(sizeof(inference_cuda_attn_decode_gemma_args) == 96 && offsetof(inference_cuda_attn_decode_gemma_args, out) == 0 && offsetof(inference_cuda_attn_decode_gemma_args, status) == 8 && offsetof(inference_cuda_attn_decode_gemma_args, q) == 16 && offsetof(inference_cuda_attn_decode_gemma_args, k_codes) == 24 && offsetof(inference_cuda_attn_decode_gemma_args, k_scales) == 32 && offsetof(inference_cuda_attn_decode_gemma_args, v_codes) == 40 && offsetof(inference_cuda_attn_decode_gemma_args, v_scales) == 48 && offsetof(inference_cuda_attn_decode_gemma_args, kv_len) == 56 && offsetof(inference_cuda_attn_decode_gemma_args, m) == 64 && offsetof(inference_cuda_attn_decode_gemma_args, n_q_head) == 68 && offsetof(inference_cuda_attn_decode_gemma_args, n_kv_head) == 72 && offsetof(inference_cuda_attn_decode_gemma_args, head_dim) == 76 && offsetof(inference_cuda_attn_decode_gemma_args, cells) == 80 && offsetof(inference_cuda_attn_decode_gemma_args, window) == 84 && offsetof(inference_cuda_attn_decode_gemma_args, scale) == 88, "layout");
#endif

/* One launch on stream (must not be NULL). Returns INFERENCE_CUDA_INVALID_ARGUMENT
 * before any CUDA call for a bad argument, else the launch status. */
int inference_cuda_attn_decode_gemma_launch(
    const inference_cuda_attn_decode_gemma_args *args, void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_ATTN_DECODE_GEMMA_H */
