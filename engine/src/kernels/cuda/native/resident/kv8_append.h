/* RD-8k: the KV8 append entry (8-bit KV of decision 0047 item 6, the frozen format below).
 * Launch-only, like the entries of abi_resident.h. Include after abi.h in C++; Rust mirrors the
 * structs in native/resident/kv8_append.rs (child module of native/resident.rs).
 *
 * Format (frozen; do not change): K and V are symmetric int8 with one fp16 scale per 32 values,
 * `scale = amax / 127` rounded to fp16 (half even); each code is round_half_even(x / scale_f16)
 * clamped to [-127, 127], quantized against the STORED fp16 scale so that the decode and the
 * chunk appends dequantize identically. A group whose stored scale is 0 — an all-zero group, or
 * an amax below what an fp16 scale can carry — has codes 0 as well (dequantizing gives 0, never
 * NaN, and the quantizer never divides by 0). Codes are int8 [kv head][cell][head_dim] and scales are fp16
 * [kv head][cell][head_dim / 32], K and V each with their own arrays, `cell = position`.
 *
 * Two entries. The decode append takes m rows (1..16) of k and v, f32 [m][n_kv_head][head_dim]
 * each (k after the norm and the rope), and writes row *pos + t of both planes, reading the
 * device i32 word pos (StepParams.pos, offset 4). Poison rule (as attn_prep, RD-8a): the kernel
 * tests the word against the cells before any other memory access; if *pos < 0 or
 * *pos + m > cells (64-bit sum) no code and no scale is written and mask 4
 * (INFERENCE_CUDA_STATUS_INVALID_STEP_STATE of resident/step_params.h) is OR-ed into *status by
 * one thread. The chunk append takes t rows of prefill at the HOST base position pos0
 * (0 <= pos0, pos0 + t <= cells), checked here before any launch.
 *
 * All geometry is run-time (C20): n_kv_head 1..64, head_dim a multiple of 32 in 32..1024
 * (one warp quantizes one group of 32), cells 1..2^30 - 1. Both entries enqueue one kernel on
 * stream (NULL: the default stream; the Rust wrapper always passes one) and return
 * INFERENCE_CUDA_INVALID_ARGUMENT before any launch for a bad argument.
 */
#ifndef INFERENCE_CUDA_RESIDENT_KV8_APPEND_H
#define INFERENCE_CUDA_RESIDENT_KV8_APPEND_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The decode append: m rows at the device position word. */
typedef struct inference_cuda_kv8_append_args {
    const void *k;        /* f32 [m][n_kv_head][head_dim] */
    const void *v;        /* f32 [m][n_kv_head][head_dim] */
    void *k_codes;        /* int8 [n_kv_head][cells][head_dim] */
    void *k_scales;       /* fp16 [n_kv_head][cells][head_dim / 32] */
    void *v_codes;
    void *v_scales;
    const void *pos;      /* one device i32: the position of token 0 */
    void *status;         /* one device i32: receives mask 4 on a refused launch */
    int m;                /* tokens, 1..16 */
    int n_kv_head;        /* 1..64 */
    int head_dim;         /* a multiple of 32, 32..1024 */
    int cells;            /* plane cells per head (the KV capacity), 1..2^30 - 1 */
} inference_cuda_kv8_append_args;

/* The prefill chunk append: t rows at the host base position pos0. */
typedef struct inference_cuda_kv8_chunk_args {
    const void *k;        /* f32 [t][n_kv_head][head_dim] */
    const void *v;        /* f32 [t][n_kv_head][head_dim] */
    void *k_codes;
    void *k_scales;
    void *v_codes;
    void *v_scales;
    int pos0;             /* the position of token 0, 0..cells - t */
    int t;                /* rows, 1..65535 (grid y) */
    int n_kv_head;        /* 1..64 */
    int head_dim;         /* a multiple of 32, 32..1024 */
    int cells;            /* plane cells per head, 1..2^30 - 1 */
} inference_cuda_kv8_chunk_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_kv8_append_args) == 80 && offsetof(inference_cuda_kv8_append_args, k_codes) == 16 && offsetof(inference_cuda_kv8_append_args, pos) == 48 && offsetof(inference_cuda_kv8_append_args, status) == 56 && offsetof(inference_cuda_kv8_append_args, m) == 64,
              "inference_cuda_kv8_append_args layout");
static_assert(sizeof(inference_cuda_kv8_chunk_args) == 72 && offsetof(inference_cuda_kv8_chunk_args, k_codes) == 16 && offsetof(inference_cuda_kv8_chunk_args, pos0) == 48 && offsetof(inference_cuda_kv8_chunk_args, t) == 52,
              "inference_cuda_kv8_chunk_args layout");
#else
_Static_assert(sizeof(inference_cuda_kv8_append_args) == 80 && offsetof(inference_cuda_kv8_append_args, k_codes) == 16 && offsetof(inference_cuda_kv8_append_args, pos) == 48 && offsetof(inference_cuda_kv8_append_args, status) == 56 && offsetof(inference_cuda_kv8_append_args, m) == 64,
               "inference_cuda_kv8_append_args layout");
_Static_assert(sizeof(inference_cuda_kv8_chunk_args) == 72 && offsetof(inference_cuda_kv8_chunk_args, k_codes) == 16 && offsetof(inference_cuda_kv8_chunk_args, pos0) == 48 && offsetof(inference_cuda_kv8_chunk_args, t) == 52,
               "inference_cuda_kv8_chunk_args layout");
#endif

/* One launch (grid n_kv_head by m, head_dim threads). */
int inference_cuda_kv8_append_launch(const inference_cuda_kv8_append_args *args, void *stream,
                                     int *out_stage);
/* One launch (grid n_kv_head by t, head_dim threads). */
int inference_cuda_kv8_chunk_launch(const inference_cuda_kv8_chunk_args *args, void *stream,
                                    int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_KV8_APPEND_H */
