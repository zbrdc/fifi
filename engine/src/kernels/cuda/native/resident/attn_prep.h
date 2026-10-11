/* RD-8a: launch entry of the vendored attn_prep (RD-4V) with the device position word:
 * per-head RMSNorm of q and k with learned weights, NEOX rope over the first n_rot dimensions at
 * the position *pos + t of token t, the q output, and the KV append at the same rows.
 * One header per kernel file (lane C ruling on O2); abi_resident.h stays closed.
 *
 * Layout (all f32 unless stated; device pointers):
 *   proj     [m][ld], ld >= n_head * 2 * 256 + 2 * n_kv_head * 256. Per q head h at column h * 512:
 *            [q 256 | gate 256]; then n_kv_head k heads of 256, then n_kv_head v heads of 256.
 *            The gate half is NOT read here; attn_decode (RD-8b) reads it from the same row.
 *   q_norm, k_norm  256 floats each.
 *   q        [m][n_head][256] out: RMSNorm * weight, then rope. 16-byte aligned (RD-8b input).
 *   k_cache, v_cache  fp16 [rows][n_kv_head][256], 16-byte aligned. Row *pos + t of both caches
 *            receives token t (k after norm and rope, v unchanged), rounded to fp16 to nearest even.
 *   pos      one device i32: the position of token 0 (StepParams.pos, offset 4). kv_slot equals pos
 *            (REPLY-44) and is not read: the row is derived from this word.
 * Poison rule (REPLY-47 and the lead addendum): the kernel tests the word against the cache
 * before any memory access. If *pos < 0 or *pos + m > rows (64-bit sum), no KV row is written,
 * every q element is NaN, and mask 4 (INFERENCE_CUDA_STATUS_INVALID_STEP_STATE of step_params.h,
 * REPLY-47) is OR-ed into *status by one thread (integer atomic). No partial append happens
 * while no other stream writes the pos word. A valid launch writes neither NaN nor status.
 *
 * Configurations: only head_dim 256 with an output gate and slot_mode 0 (row = position) run.
 * has_gate = 0, head_dim = 128 and slot_mode 1 (ring, Mellum) belong to MEL-K1 and are refused
 * with INFERENCE_CUDA_INVALID_ARGUMENT before any launch, as is every bad shape below.
 * The launch is asynchronous on stream (NULL: the default stream; the Rust wrapper always passes one).
 */
#ifndef INFERENCE_CUDA_RESIDENT_ATTN_PREP_H
#define INFERENCE_CUDA_RESIDENT_ATTN_PREP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INFERENCE_CUDA_SLOT_POSITION 0
#define INFERENCE_CUDA_SLOT_RING 1

typedef struct inference_cuda_attn_prep_args {
    const void *proj;
    const void *q_norm;
    const void *k_norm;
    void *k_cache;
    void *v_cache;
    void *q;
    const void *pos;
    void *status;   /* StepParams.status (offset 16), 4-aligned: receives mask 4 on a refused launch */
    int ld;
    int m;          /* tokens, 1..65535 (grid y) */
    int n_head;     /* 1..256 */
    int n_kv_head;  /* 1..64 */
    int head_dim;   /* 256 */
    int n_rot;      /* a power of two, 2..256 (RP1 domain, REPLY-53) */
    int rows;       /* cache rows, m..2^30 - 1 */
    int has_gate;   /* 1 */
    int slot_mode;  /* INFERENCE_CUDA_SLOT_POSITION */
    int slot_ring;  /* 0 */
    float eps;      /* finite, > 0 */
    float rope_base;/* finite, > 1 */
} inference_cuda_attn_prep_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_attn_prep_args) == 112 && offsetof(inference_cuda_attn_prep_args, pos) == 48 && offsetof(inference_cuda_attn_prep_args, status) == 56 && offsetof(inference_cuda_attn_prep_args, ld) == 64 && offsetof(inference_cuda_attn_prep_args, slot_ring) == 100 && offsetof(inference_cuda_attn_prep_args, eps) == 104 && offsetof(inference_cuda_attn_prep_args, rope_base) == 108,
              "inference_cuda_attn_prep_args layout");
#else
_Static_assert(sizeof(inference_cuda_attn_prep_args) == 112 && offsetof(inference_cuda_attn_prep_args, pos) == 48 && offsetof(inference_cuda_attn_prep_args, status) == 56 && offsetof(inference_cuda_attn_prep_args, ld) == 64 && offsetof(inference_cuda_attn_prep_args, slot_ring) == 100 && offsetof(inference_cuda_attn_prep_args, eps) == 104 && offsetof(inference_cuda_attn_prep_args, rope_base) == 108,
               "inference_cuda_attn_prep_args layout");
#endif

/* One launch (grid n_head + 2 * n_kv_head by m, 256 threads). */
int inference_cuda_attn_prep_launch(const inference_cuda_attn_prep_args *args, void *stream,
                                    int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_ATTN_PREP_H */
