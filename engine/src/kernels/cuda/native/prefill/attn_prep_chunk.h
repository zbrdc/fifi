/* CPF-ATTN-A K1: launch entry of attn_prep_chunk, the chunk KV write of the prefill attention
 * (decision 0049 item 2), one header per kernel file (the lane C ruling the resident tree
 * records); abi_resident.h stays closed.
 *
 * What it computes, per token t of m and per head: the RMSNorm of the token's q and k rows
 * with the learned weights, the NEOX rope over the first n_rot dimensions at the position
 * p0 + t (p0 and m are LAUNCH ARGUMENTS, decision 0049 item 1: prefill reads and writes no
 * StepParams field and no device pos word), the f32 q output, and the k row (after the norm
 * and the rope) plus the unchanged v row written either to the fp16 cache planes or to f32
 * rows. The math is a byte-for-byte re-expression of the vendored attn_prep (RD-4V,
 * vendor/xavier-qwen/src/ops.cu attn_prep_k; the vendored form reads the position from the
 * device word *pos_dev, ops.cu:166): identical expressions and reduction order, pinned
 * bit-equal by the wrapper tests (test 4).
 *
 * Layout (all f32 unless stated; device pointers):
 *   proj     [m][ld], ld >= (n_head * 2 + n_kv_head * 2) * head_dim. Per q head at column
 *            h * 2 * head_dim: [q head_dim | gate head_dim]; the gate half is NOT read here;
 *            then n_kv_head k heads of head_dim, then n_kv_head v heads of head_dim.
 *   q_norm, k_norm  head_dim floats each.
 *   q        [m][n_head][head_dim] out: RMSNorm * weight, then rope. 16-byte aligned.
 *   mode INFERENCE_CUDA_PREP_APPEND_FP16 (0): k_cache and v_cache fp16
 *            [rows][n_kv_head][head_dim], 16-byte aligned, position-major (KvMode::Fp16).
 *            Row p0 + t of both caches receives token t, rounded to fp16 to nearest even;
 *            k_rows and v_rows must be NULL.
 *   mode INFERENCE_CUDA_PREP_ROWS_OUT (1): k_rows and v_rows f32 [m][n_kv_head][head_dim],
 *            4-byte aligned: the post-rope k rows and the unchanged v rows of the whole
 *            chunk, the input contract of kv8_append's chunk entry (IMPL-KV8-A); k_cache and
 *            v_cache must be NULL.
 *   status   NULL, or one device i32: the prefill status word of decision 0049 item 10 (NOT
 *            StepParams.status). It is untouched by a valid launch.
 *
 * Poison rule (decision 0049 item 2: the device checks p0/m again; item 10: bit 8): if
 * p0 < 0 or (long long)p0 + m > rows, no cache row and no k/v row is written, every q
 * element is NaN, and bit 8 (INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION, 0x100) is OR-ed
 * into *status by one thread (integer atomic) when status != NULL. The answer is
 * block-uniform, so the early return happens before the first barrier. A valid launch
 * writes neither NaN nor status.
 *
 * Configurations: head_dim 256 only (the compiled head dim; every geometry value stays a
 * run-time argument, lane C condition 8, and a value the kernel refuses is
 * INFERENCE_CUDA_INVALID_ARGUMENT before any launch, never an assertion inside the kernel).
 * The launch is asynchronous on stream (NULL: the default stream; the Rust wrapper always
 * passes one).
 */
#ifndef INFERENCE_CUDA_PREFILL_ATTN_PREP_CHUNK_H
#define INFERENCE_CUDA_PREFILL_ATTN_PREP_CHUNK_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INFERENCE_CUDA_PREP_APPEND_FP16 0
#define INFERENCE_CUDA_PREP_ROWS_OUT 1
/* Bit 8 of the prefill status word (decision 0049 item 10): a prefill chunk fault, a bound
 * violation here. It maps to Backend with operation "prefill_chunk". */
#define INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION 0x100

typedef struct inference_cuda_attn_prep_chunk_args {
    const void *proj;
    const void *q_norm;
    const void *k_norm;
    void *k_cache;   /* APPEND_FP16 mode */
    void *v_cache;
    float *q;        /* [m][n_head][head_dim] out */
    float *k_rows;   /* ROWS_OUT mode */
    float *v_rows;
    int *status;     /* optional; bit 8 on a refused launch */
    int ld;
    int p0;          /* the position of token 0, a launch argument */
    int m;           /* tokens, 1..65535 (grid y) */
    int n_head;      /* 1..256 */
    int n_kv_head;   /* 1..64 */
    int head_dim;    /* 256 */
    int n_rot;       /* a power of two, 2..256 (RP1 domain, REPLY-53) */
    int rows;        /* cache rows, 1..2^30 - 1, with p0 + m <= rows */
    int mode;        /* INFERENCE_CUDA_PREP_APPEND_FP16 or INFERENCE_CUDA_PREP_ROWS_OUT */
    float eps;       /* finite, > 0 */
    float rope_base; /* finite, > 1 */
} inference_cuda_attn_prep_chunk_args;

int inference_cuda_attn_prep_chunk_launch(const inference_cuda_attn_prep_chunk_args *args,
                                          void *stream, int *out_stage);

#ifdef INFERENCE_CUDA_FAULT_QUALIFICATION
/* Qualified fault path (decision 0049 item 2's device re-check): validates every rule of the
 * entry EXCEPT the p0/m bound, then overrides p0 so the bound is violated, so the KERNEL's
 * own device guard (no KV row, NaN q, bit 8) is the code under test. The validating entry
 * above can never reach it with bad bounds; this entry changes no rule of that entry. */
int inference_cuda_attn_prep_chunk_fault_bad_bounds_launch(
    const inference_cuda_attn_prep_chunk_args *args, void *stream, int *out_stage);
#endif

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_attn_prep_chunk_args) == 120 &&
              offsetof(inference_cuda_attn_prep_chunk_args, proj) == 0 &&
              offsetof(inference_cuda_attn_prep_chunk_args, k_cache) == 24 &&
              offsetof(inference_cuda_attn_prep_chunk_args, q) == 40 &&
              offsetof(inference_cuda_attn_prep_chunk_args, k_rows) == 48 &&
              offsetof(inference_cuda_attn_prep_chunk_args, status) == 64 &&
              offsetof(inference_cuda_attn_prep_chunk_args, ld) == 72 &&
              offsetof(inference_cuda_attn_prep_chunk_args, p0) == 76 &&
              offsetof(inference_cuda_attn_prep_chunk_args, m) == 80 &&
              offsetof(inference_cuda_attn_prep_chunk_args, head_dim) == 92 &&
              offsetof(inference_cuda_attn_prep_chunk_args, rows) == 100 &&
              offsetof(inference_cuda_attn_prep_chunk_args, mode) == 104 &&
              offsetof(inference_cuda_attn_prep_chunk_args, eps) == 108,
              "inference_cuda_attn_prep_chunk_args layout");
#else
_Static_assert(sizeof(inference_cuda_attn_prep_chunk_args) == 120 &&
               offsetof(inference_cuda_attn_prep_chunk_args, proj) == 0 &&
               offsetof(inference_cuda_attn_prep_chunk_args, k_cache) == 24 &&
               offsetof(inference_cuda_attn_prep_chunk_args, q) == 40 &&
               offsetof(inference_cuda_attn_prep_chunk_args, k_rows) == 48 &&
               offsetof(inference_cuda_attn_prep_chunk_args, status) == 64 &&
               offsetof(inference_cuda_attn_prep_chunk_args, ld) == 72 &&
               offsetof(inference_cuda_attn_prep_chunk_args, p0) == 76 &&
               offsetof(inference_cuda_attn_prep_chunk_args, m) == 80 &&
               offsetof(inference_cuda_attn_prep_chunk_args, head_dim) == 92 &&
               offsetof(inference_cuda_attn_prep_chunk_args, rows) == 100 &&
               offsetof(inference_cuda_attn_prep_chunk_args, mode) == 104 &&
               offsetof(inference_cuda_attn_prep_chunk_args, eps) == 108,
               "inference_cuda_attn_prep_chunk_args layout");
#endif

#endif
