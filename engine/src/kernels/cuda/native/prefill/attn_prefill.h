/* CPF-ATTN-A K2: launch entry of attn_prefill, the flash-style chunked-prefill attention
 * (decision 0049 items 1, 2 and 10; the CPF-5a/5b kernel of design section 3.7), one header
 * per kernel file (the lane C ruling the resident tree records); abi_resident.h stays closed.
 *
 * What it computes, per token t of m and query head h (kv head kh = h / G, G selected
 * by n_head / n_kv_head) over the keys j <= P with P = p0 + t (causal by ABSOLUTE position):
 *   out[t][h][:] = sigmoid(gate[t][h][:]) * sum_j softmax_j(scale * qhat . K[j][kh]) * V[j][kh]
 * with qhat = q rounded to fp16 (q is f32), K and V read from the KV cache, the probabilities
 * rounded to fp16 for the P*V product and everything else accumulated in f32 (PA1's reference
 * form: "on fp16 Q, K and V"). Two launches on the caller's stream, the attn_decode structure
 * scaled by a query-tile dimension:
 *   attn_prefill_part_dsplit_k  grid (q_tiles, n_kv_head, max_splits), 512 threads, q_tile 8:
 *                          a block owns one query tile (8 tokens x 8 heads = 64 rows) of one
 *                          kv head and one key range, the heaviest causal tiles first; its 16
 *                          warps hold 4 row groups x 4 head_dim quarters, and the quarter
 *                          scores are summed in one fixed order; online softmax over key tiles
 *                          of 32; one partial per row goes to part.
 *   attn_prefill_combine_k grid (n_head, m), 256 threads: merges the partials of the active
 *                          splits in split-index order (exact block max, fixed-order sums),
 *                          divides and applies the sigmoid output gate when proj is given.
 * The device split rule is attn_decode's split_plan verbatim, evaluated per q_tile over its
 * key range: the union of its rows' windows (all p0 + tile_end keys with no window); every
 * block and the combine recompute the same deterministic plan from the launch arguments.
 * kv_len is a HOST value in prefill (no device word, 0049 item 1), so capture is trivial
 * and eager is the mode.
 *
 * Layout (device pointers):
 *   q        f32 [m][n_head][head_dim] (attn_prep_chunk's q), 16-byte aligned.
 *   proj     NULL (no gate) or f32 [m][ld], ld >= n_head * 2 * head_dim: head h's gate is at
 *            column h * 2 * head_dim + head_dim + d, 4-byte aligned.
 *   sinks    NULL or published f32 [n_head], 4-byte aligned: raw unscaled query-head logits,
 *            denominator mass once after merging all parts, with no value vector.
 *   k_cache, v_cache  fp16 [cache_rows][n_kv_head][head_dim], 16-byte aligned (kv_mode 0).
 *   The four kv8_* pointers and kv8_capacity are the Int8G32 planes of state_layout.rs; they
 *            are NULL and 0 while kv_mode is 0 (the KV8 loader lands with IMPL-KV8-A).
 *   part     f32 scratch of inference_cuda_attn_prefill_part_floats(m, n_kv_head, head_dim,
 *            max_splits, G) floats, 16-byte aligned, no initialization needed; only the active
 *            splits of each cell are written.
 *   out      f32 [m][n_head * head_dim], 16-byte aligned; NaN on a refused launch.
 *   status   NULL, or one device i32: the prefill status word of decision 0049 item 10.
 *
 * Window and ring (the attn_decode rules, included for MEL-P1): window = 0 attends to all
 * keys; window = W > 0 attends to the last min(P + 1, W) positions. ring = 0 is a linear
 * cache (row = position); ring = R > 0 (needs window > 0 and window + m - 1 <= R <=
 * cache_rows) puts position p in row p % R. Ring positions are absolute: p0 + m <= 2^30 - 1,
 * independent of physical cache_rows. The immutable union of all m query windows must be
 * staged before the call; do not overwrite it with the final W-row tail until consumption.
 *
 * Poison rule (decision 0049 item 2: the device checks p0/m again; item 10: mask 8): if
 * p0 < 0 or (long long)p0 + m exceeds cache_rows (linear) or 2^30 - 1 (ring/segmented),
 * the part kernel writes no partial, the combine
 * kernel writes NaN to out, and each ORs mask 8
 * (INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION) into *status when status != NULL. A valid
 * launch touches neither.
 *
 * The D256 MMA part uses 96 KB of dynamic shared memory; D128 uses 64 KB (two cp.async
 * K/V stages and a 32 KB score exchange). The entry opts in above the 48 KB default with
 * cudaFuncSetAttribute once per variant (the vendored gemm.cuh pattern), one block per SM.
 * Code compiled for sm_72 (__CUDA_ARCH__ < 800; the entry selects by the loaded code's PTX
 * arch) runs an fp32 SIMT part kernel instead, attn_prefill_part_simt_k (grid (max_splits,
 * n_kv_head, q_tiles), 128 threads), requesting 96/48 KB for D256/D128; it computes the
 * same result within the PA1 bound, not bit for bit. It is compile-only in this packet
 * (decision 0049 item 14; no board run).
 *
 * Legacy F16 NaN isolation is per 64-row query tile, not per row: a NaN V cell inside a tile's key range
 * reaches every row of that tile (a row that does not see the key still multiplies p = 0 by
 * it), and a NaN q row reaches only its own row. A cell outside every tile's key range is
 * never read (the loader zero-fills past the range).
 *
 * Legacy F16: head_dim 128/256 and GQA 1/8. Tagged Q8 affine: D64/128/256 and
 * any positive integral GQA within the 64MiB partial budget; F32 Q and accumulators,
 * independent F16/BF16/F32 scalar planes, optional bias, no temporary F16 cache.
 * Q8 reads only keys visible to each row; signed/G32/F16-scale/no-bias is
 * Q8_0-equivalent separated planes, not the raw interleaved GGUF block layout.
 * In both paths,
 * every geometry value stays a run-time argument (lane C condition 8) and a value the entry
 * refuses returns INFERENCE_CUDA_INVALID_ARGUMENT before any launch, never an assertion
 * inside a kernel. The launch is asynchronous on stream (NULL: the default stream; the Rust
 * wrapper always passes one) and never synchronizes.
 */
#ifndef INFERENCE_CUDA_PREFILL_ATTN_PREFILL_H
#define INFERENCE_CUDA_PREFILL_ATTN_PREFILL_H

#include <stddef.h>

#include "prefill/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* kv_mode 0: the fp16 planes k_cache/v_cache. kv_mode 1 (the Int8G32 loader) arrives with
 * IMPL-KV8-A and is refused while the four kv8 pointers must be NULL. */
#define INFERENCE_CUDA_PREFILL_KV_FP16 0
#define INFERENCE_CUDA_PREFILL_KV_KV8 1

/* Shared tagged-plane transport; only Q8 affine is implemented. Scalar dtype codes:
 * F16=0, BF16=1, F32=2. A NULL pointer alone means absent: strides are zero and
 * dtype is ignored (zero also encodes F16). 64-bit ABI: stride fields 24..88,
 * i.e. [24, 88); all strides are in bytes. */
#define INFERENCE_CUDA_PREFILL_KV_VIEW 2
#define INFERENCE_CUDA_ATTN_STORAGE_F16 0
#define INFERENCE_CUDA_ATTN_STORAGE_BF16 1
#define INFERENCE_CUDA_ATTN_STORAGE_F32 2
#define INFERENCE_CUDA_ATTN_STORAGE_Q8_AFFINE 3
#define INFERENCE_CUDA_ATTN_STORAGE_Q4_AFFINE 4

typedef struct inference_cuda_attn_kv_view {
    const void *payload;
    const void *scales;
    const void *bias;
    size_t payload_time_stride_bytes;
    size_t payload_head_stride_bytes;
    size_t scale_time_stride_bytes;
    size_t scale_head_stride_bytes;
    size_t scale_group_stride_bytes;
    size_t bias_time_stride_bytes;
    size_t bias_head_stride_bytes;
    size_t bias_group_stride_bytes;
    int storage;
    int group_size;
    int scale_dtype;
    int bias_dtype;
    int q8_signed;
    int q4_low_even;
} inference_cuda_attn_kv_view;

typedef struct inference_cuda_attn_kv_pair {
    inference_cuda_attn_kv_view k;
    inference_cuda_attn_kv_view v;
} inference_cuda_attn_kv_pair;

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
    int *status; /* optional; mask 8 on a refused launch */
    int ld;
    int p0;      /* the position of token 0, a launch argument */
    int m;       /* tokens, 1..65535 */
    int n_head;      /* legacy GQA1/8; Q8: positive integral multiple of n_kv_head */
    int n_kv_head;
    int head_dim;     /* legacy 128/256; Q8 64/128/256 */
    int cache_rows;   /* 1..2^30 - 1 physical rows; Full linear: p0 + m <= cache_rows;
                      * ring/segmented: p0 + m <= 2^30 - 1 */
    int max_splits;   /* 1..128 */
    int min_keys;     /* a multiple of 32, 32..2^20 */
    int window;       /* 0: all keys; > 0: the last window positions */
    int ring;         /* 0: linear; > 0: row = position % ring */
    int kv_mode;      /* INFERENCE_CUDA_PREFILL_KV_FP16 or INFERENCE_CUDA_PREFILL_KV_VIEW */
    int kv8_capacity; /* 0 while kv_mode is 0 */
    float scale;      /* finite, in (0, 1e30) */
    const void *sinks; /* NULL or raw f32 [n_head], no GQA folding */
    const inference_cuda_attn_kv_pair *kv_views; /* NULL for F16; Q8 host metadata copied at enqueue */
} inference_cuda_attn_prefill_args;

/* Floats of `part`: ceil(m / 8) * 8 * n_kv_head * max_splits * group * (2 + head_dim).
 * Zero means an unsupported geometry. */
size_t inference_cuda_attn_prefill_part_floats(int m, int n_kv_head, int head_dim,
                                               int max_splits, int group);
int inference_cuda_attn_prefill_launch(const inference_cuda_attn_prefill_args *args,
                                       void *stream, int *out_stage);

/* Bounded staging: cache_rows is the physical stage extent, p0 stays absolute, window
 * and ring must be zero, max_splits is N total segment slots (partials <=64MiB).
 * PART reads stage rows 0..key_rows and writes only part_index, including neutral rows
 * with no visible keys. Absolute key_origin+key_rows <=p0+m; part_index lies in0..N-1.
 * COMBINE reads all N initialized slots in index order and applies the gate once.
 * Caller queues all parts and then combine on the same stream before reusing the partials.
 * Neither entry allocates, transfers or synchronizes. The legacy entry stays unchanged. */
int inference_cuda_attn_prefill_part_launch(const inference_cuda_attn_prefill_args *args,
                                            int key_origin, int key_rows, int part_index,
                                            void *stream, int *out_stage);
int inference_cuda_attn_prefill_combine_launch(const inference_cuda_attn_prefill_args *args,
                                               void *stream, int *out_stage);

#ifdef INFERENCE_CUDA_FAULT_QUALIFICATION
/* Qualified fault path (decision 0049 item 2's device re-check): validates every rule of the
 * entry EXCEPT the p0/m bound, then forces p0 = -1 for both linear and ring calls. The KERNELS'
 * own device guards (no partial, NaN out, mask 8) are the code under test. The validating
 * entry above can never reach them with bad bounds. */
int inference_cuda_attn_prefill_fault_bad_bounds_launch(
    const inference_cuda_attn_prefill_args *args, void *stream, int *out_stage);
#endif

#ifdef INFERENCE_CUDA_Q8_REFERENCE
/* Fixture builds only (A ruling 01a18101): the q8 part kernel with the pre-hoist per-element
 * math, so one ELF can compare old vs new byte for byte. Requires q8 views like the Q8 path. */
int inference_cuda_attn_prefill_q8_reference_launch(
    const inference_cuda_attn_prefill_args *args, void *stream, int *out_stage);
#endif

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#define INFERENCE_CUDA_ATTN_ABI_ASSERT static_assert
#define INFERENCE_CUDA_ATTN_ABI_ALIGNOF alignof
#else
#define INFERENCE_CUDA_ATTN_ABI_ASSERT _Static_assert
#define INFERENCE_CUDA_ATTN_ABI_ALIGNOF _Alignof
#endif

INFERENCE_CUDA_ATTN_ABI_ASSERT(sizeof(inference_cuda_attn_kv_view) == 112 &&
    INFERENCE_CUDA_ATTN_ABI_ALIGNOF(inference_cuda_attn_kv_view) == 8 &&
    offsetof(inference_cuda_attn_kv_view, payload) == 0 &&
    offsetof(inference_cuda_attn_kv_view, scales) == 8 &&
    offsetof(inference_cuda_attn_kv_view, bias) == 16 &&
    offsetof(inference_cuda_attn_kv_view, payload_time_stride_bytes) == 24 &&
    offsetof(inference_cuda_attn_kv_view, payload_head_stride_bytes) == 32 &&
    offsetof(inference_cuda_attn_kv_view, scale_time_stride_bytes) == 40 &&
    offsetof(inference_cuda_attn_kv_view, scale_head_stride_bytes) == 48 &&
    offsetof(inference_cuda_attn_kv_view, scale_group_stride_bytes) == 56 &&
    offsetof(inference_cuda_attn_kv_view, bias_time_stride_bytes) == 64 &&
    offsetof(inference_cuda_attn_kv_view, bias_head_stride_bytes) == 72 &&
    offsetof(inference_cuda_attn_kv_view, bias_group_stride_bytes) == 80 &&
    offsetof(inference_cuda_attn_kv_view, storage) == 88 &&
    offsetof(inference_cuda_attn_kv_view, group_size) == 92 &&
    offsetof(inference_cuda_attn_kv_view, scale_dtype) == 96 &&
    offsetof(inference_cuda_attn_kv_view, bias_dtype) == 100 &&
    offsetof(inference_cuda_attn_kv_view, q8_signed) == 104 &&
    offsetof(inference_cuda_attn_kv_view, q4_low_even) == 108,
    "inference_cuda_attn_kv_view layout");

INFERENCE_CUDA_ATTN_ABI_ASSERT(sizeof(inference_cuda_attn_kv_pair) == 224 &&
    INFERENCE_CUDA_ATTN_ABI_ALIGNOF(inference_cuda_attn_kv_pair) == 8 &&
    offsetof(inference_cuda_attn_kv_pair, k) == 0 &&
    offsetof(inference_cuda_attn_kv_pair, v) == 112,
    "inference_cuda_attn_kv_pair layout");

INFERENCE_CUDA_ATTN_ABI_ASSERT(sizeof(inference_cuda_attn_prefill_args) == 160 &&
    INFERENCE_CUDA_ATTN_ABI_ALIGNOF(inference_cuda_attn_prefill_args) == 8 &&
    offsetof(inference_cuda_attn_prefill_args, q) == 0 &&
    offsetof(inference_cuda_attn_prefill_args, proj) == 8 &&
    offsetof(inference_cuda_attn_prefill_args, k_cache) == 16 &&
    offsetof(inference_cuda_attn_prefill_args, v_cache) == 24 &&
    offsetof(inference_cuda_attn_prefill_args, kv8_k_codes) == 32 &&
    offsetof(inference_cuda_attn_prefill_args, kv8_k_scales) == 40 &&
    offsetof(inference_cuda_attn_prefill_args, kv8_v_codes) == 48 &&
    offsetof(inference_cuda_attn_prefill_args, kv8_v_scales) == 56 &&
    offsetof(inference_cuda_attn_prefill_args, part) == 64 &&
    offsetof(inference_cuda_attn_prefill_args, out) == 72 &&
    offsetof(inference_cuda_attn_prefill_args, status) == 80 &&
    offsetof(inference_cuda_attn_prefill_args, ld) == 88 &&
    offsetof(inference_cuda_attn_prefill_args, p0) == 92 &&
    offsetof(inference_cuda_attn_prefill_args, m) == 96 &&
    offsetof(inference_cuda_attn_prefill_args, n_head) == 100 &&
    offsetof(inference_cuda_attn_prefill_args, n_kv_head) == 104 &&
    offsetof(inference_cuda_attn_prefill_args, head_dim) == 108 &&
    offsetof(inference_cuda_attn_prefill_args, cache_rows) == 112 &&
    offsetof(inference_cuda_attn_prefill_args, max_splits) == 116 &&
    offsetof(inference_cuda_attn_prefill_args, min_keys) == 120 &&
    offsetof(inference_cuda_attn_prefill_args, window) == 124 &&
    offsetof(inference_cuda_attn_prefill_args, ring) == 128 &&
    offsetof(inference_cuda_attn_prefill_args, kv_mode) == 132 &&
    offsetof(inference_cuda_attn_prefill_args, kv8_capacity) == 136 &&
    offsetof(inference_cuda_attn_prefill_args, scale) == 140 &&
    offsetof(inference_cuda_attn_prefill_args, sinks) == 144 &&
    offsetof(inference_cuda_attn_prefill_args, kv_views) == 152,
    "inference_cuda_attn_prefill_args layout");

#undef INFERENCE_CUDA_ATTN_ABI_ALIGNOF
#undef INFERENCE_CUDA_ATTN_ABI_ASSERT

#endif
