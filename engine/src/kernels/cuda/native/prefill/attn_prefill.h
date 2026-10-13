/* CPF-ATTN-A K2: launch entry of attn_prefill, the flash-style chunked-prefill attention
 * (decision 0049 items 1, 2 and 10; the CPF-5a/5b kernel of design section 3.7), one header
 * per kernel file (the lane C ruling the resident tree records); abi_resident.h stays closed.
 *
 * What it computes, per token t of m and query head h (kv head kh = h / 8, exactly 8 query
 * heads per kv head) over the keys j <= P with P = p0 + t (causal by ABSOLUTE position):
 *   out[t][h][:] = sigmoid(gate[t][h][:]) * sum_j softmax_j(scale * qhat . K[j][kh]) * V[j][kh]
 * with qhat = q rounded to fp16 (q is f32), K and V read from the KV cache, the probabilities
 * rounded to fp16 for the P*V product and everything else accumulated in f32 (PA1's reference
 * form: "on fp16 Q, K and V"). Two launches on the caller's stream, the attn_decode structure
 * scaled by a query-tile dimension:
 *   attn_prefill_part_k    grid (max_splits, n_kv_head, q_tiles), 128 threads, q_tile = 8:
 *                          a block owns one query tile (8 tokens x 8 heads = 64 rows) of one
 *                          kv head and one key range; online softmax over key tiles of 32;
 *                          one partial per row goes to part.
 *   attn_prefill_combine_k grid (n_head, m), 256 threads: merges the partials of the active
 *                          splits in split-index order (exact block max, fixed-order sums),
 *                          divides and applies the sigmoid output gate when proj is given.
 * The device split rule is attn_decode's split_plan verbatim, evaluated per q_tile from its
 * own key count p0 + tile_end; every block and the combine recompute the same deterministic
 * plan from the launch arguments. kv_len is a HOST value in prefill (no device word, 0049
 * item 1), so capture is trivial and eager is the mode.
 *
 * Layout (device pointers):
 *   q        f32 [m][n_head][head_dim] (attn_prep_chunk's q), 16-byte aligned.
 *   proj     NULL (no gate) or f32 [m][ld], ld >= n_head * 2 * head_dim: head h's gate is at
 *            column h * 2 * head_dim + head_dim + d, 4-byte aligned.
 *   k_cache, v_cache  fp16 [cache_rows][n_kv_head][head_dim], 16-byte aligned (kv_mode 0).
 *   The four kv8_* pointers and kv8_capacity are the Int8G32 planes of state_layout.rs; they
 *            are NULL and 0 while kv_mode is 0 (the KV8 loader lands with IMPL-KV8-A).
 *   part     f32 scratch of inference_cuda_attn_prefill_part_floats(m, n_kv_head, head_dim,
 *            max_splits) floats, 16-byte aligned, no initialization needed; only the active
 *            splits of each cell are written.
 *   out      f32 [m][n_head * head_dim], 16-byte aligned; NaN on a refused launch.
 *   status   NULL, or one device i32: the prefill status word of decision 0049 item 10.
 *
 * Window and ring (the attn_decode rules, included for MEL-P1): window = 0 attends to all
 * keys; window = W > 0 attends to the last min(P + 1, W) positions. ring = 0 is a linear
 * cache (row = position); ring = R > 0 (needs window > 0 and window + m - 1 <= R <=
 * cache_rows) puts position p in row p % R.
 *
 * Poison rule (decision 0049 item 2: the device checks p0/m again; item 10: bit 8): if
 * p0 < 0 or (long long)p0 + m > cache_rows, the part kernel writes no partial, the combine
 * kernel writes NaN to out, and each ORs bit 8
 * (INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION) into *status when status != NULL. A valid
 * launch touches neither.
 *
 * The part kernel uses 96 KB of dynamic shared memory (the Q tile 32 KB, double-buffered K
 * and V tiles 2 x 16 KB each), above the 48 KB default: the entry opts in with
 * cudaFuncSetAttribute once per process (the vendored gemm.cuh pattern). sm_72
 * (__CUDA_ARCH__ < 800) compiles an fp32 SIMT arm with the same semantics; it is
 * compile-only in this packet (decision 0049 item 14; no board run).
 *
 * Configurations: head_dim 256 and exactly 8 query heads per kv head are the compiled cases;
 * every geometry value stays a run-time argument (lane C condition 8) and a value the entry
 * refuses returns INFERENCE_CUDA_INVALID_ARGUMENT before any launch, never an assertion
 * inside a kernel. The launch is asynchronous on stream (NULL: the default stream; the Rust
 * wrapper always passes one) and never synchronizes.
 */
#ifndef INFERENCE_CUDA_PREFILL_ATTN_PREFILL_H
#define INFERENCE_CUDA_PREFILL_ATTN_PREFILL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* kv_mode 0: the fp16 planes k_cache/v_cache. kv_mode 1 (the Int8G32 loader) arrives with
 * IMPL-KV8-A and is refused while the four kv8 pointers must be NULL. */
#define INFERENCE_CUDA_PREFILL_KV_FP16 0
#define INFERENCE_CUDA_PREFILL_KV_KV8 1
/* Bit 8 of the prefill status word (decision 0049 item 10): a prefill chunk fault, a bound
 * violation here (the same bit attn_prep_chunk.h defines). */
#define INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION 0x100

typedef struct inference_cuda_attn_prefill_args {
    const void *q;
    const void *proj; /* NULL = no gate */
    const void *k_cache;
    const void *v_cache;
    const void *kv8_k_codes;  /* NULL while kv_mode is 0 */
    const void *kv8_k_scales;
    const void *kv8_v_codes;
    const void *kv8_v_scales;
    void *part;
    void *out;
    int *status; /* optional; bit 8 on a refused launch */
    int ld;
    int p0;      /* the position of token 0, a launch argument */
    int m;       /* tokens, 1..65535 */
    int n_head;  /* == 8 * n_kv_head */
    int n_kv_head;
    int head_dim;     /* 256 */
    int cache_rows;   /* 1..2^30 - 1, with p0 + m <= cache_rows */
    int max_splits;   /* 1..128 */
    int min_keys;     /* a multiple of 32, 32..2^20 */
    int window;       /* 0: all keys; > 0: the last window positions */
    int ring;         /* 0: linear; > 0: row = position % ring */
    int kv_mode;      /* INFERENCE_CUDA_PREFILL_KV_FP16 */
    int kv8_capacity; /* 0 while kv_mode is 0 */
    float scale;      /* finite, in (0, 1e30) */
} inference_cuda_attn_prefill_args;

/* Floats of `part`: ceil(m / 8) * 8 * n_kv_head * max_splits * 8 * (2 + head_dim). */
size_t inference_cuda_attn_prefill_part_floats(int m, int n_kv_head, int head_dim,
                                               int max_splits);
int inference_cuda_attn_prefill_launch(const inference_cuda_attn_prefill_args *args,
                                       void *stream, int *out_stage);

#ifdef INFERENCE_CUDA_FAULT_QUALIFICATION
/* Qualified fault path (decision 0049 item 2's device re-check): validates every rule of the
 * entry EXCEPT the p0/m bound, then overrides p0 so the bound is violated, so the KERNELS'
 * own device guards (no partial, NaN out, bit 8) are the code under test. The validating
 * entry above can never reach them with bad bounds. */
int inference_cuda_attn_prefill_fault_bad_bounds_launch(
    const inference_cuda_attn_prefill_args *args, void *stream, int *out_stage);
#endif

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_attn_prefill_args) == 144 &&
              offsetof(inference_cuda_attn_prefill_args, q) == 0 &&
              offsetof(inference_cuda_attn_prefill_args, k_cache) == 16 &&
              offsetof(inference_cuda_attn_prefill_args, kv8_k_codes) == 32 &&
              offsetof(inference_cuda_attn_prefill_args, part) == 64 &&
              offsetof(inference_cuda_attn_prefill_args, out) == 72 &&
              offsetof(inference_cuda_attn_prefill_args, status) == 80 &&
              offsetof(inference_cuda_attn_prefill_args, ld) == 88 &&
              offsetof(inference_cuda_attn_prefill_args, p0) == 92 &&
              offsetof(inference_cuda_attn_prefill_args, m) == 96 &&
              offsetof(inference_cuda_attn_prefill_args, head_dim) == 108 &&
              offsetof(inference_cuda_attn_prefill_args, cache_rows) == 112 &&
              offsetof(inference_cuda_attn_prefill_args, max_splits) == 116 &&
              offsetof(inference_cuda_attn_prefill_args, window) == 124 &&
              offsetof(inference_cuda_attn_prefill_args, kv_mode) == 132 &&
              offsetof(inference_cuda_attn_prefill_args, scale) == 140,
              "inference_cuda_attn_prefill_args layout");
#else
_Static_assert(sizeof(inference_cuda_attn_prefill_args) == 144 &&
               offsetof(inference_cuda_attn_prefill_args, q) == 0 &&
               offsetof(inference_cuda_attn_prefill_args, k_cache) == 16 &&
               offsetof(inference_cuda_attn_prefill_args, kv8_k_codes) == 32 &&
               offsetof(inference_cuda_attn_prefill_args, part) == 64 &&
               offsetof(inference_cuda_attn_prefill_args, out) == 72 &&
               offsetof(inference_cuda_attn_prefill_args, status) == 80 &&
               offsetof(inference_cuda_attn_prefill_args, ld) == 88 &&
               offsetof(inference_cuda_attn_prefill_args, p0) == 92 &&
               offsetof(inference_cuda_attn_prefill_args, m) == 96 &&
               offsetof(inference_cuda_attn_prefill_args, head_dim) == 108 &&
               offsetof(inference_cuda_attn_prefill_args, cache_rows) == 112 &&
               offsetof(inference_cuda_attn_prefill_args, max_splits) == 116 &&
               offsetof(inference_cuda_attn_prefill_args, window) == 124 &&
               offsetof(inference_cuda_attn_prefill_args, kv_mode) == 132 &&
               offsetof(inference_cuda_attn_prefill_args, scale) == 140,
               "inference_cuda_attn_prefill_args layout");
#endif

#endif
