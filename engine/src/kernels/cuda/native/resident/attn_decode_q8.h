/*
 * The KV8 split-KV decode attention entry (native/resident/attn_decode_q8.h, per-kernel
 * header). Launch-only, like the entries of abi_resident.h. Include after abi.h in C++; Rust
 * mirrors the struct in native/resident/attn_decode_q8.rs (child module of native/resident.rs).
 *
 * The same contract as attn_decode.h, over the 8-bit KV planes that kv8_append writes: q is f32
 * [m][n_head][head_dim] (attn_prep's q; m <= 16), query heads grouped 8 per kv head
 * (n_head == 8 * n_kv_head), head_dim 128 or 256, proj NULL (no output gate) or the f32
 * projection row [m][ld] whose head h gate is at column h * 2 * head_dim + head_dim. K and V are
 * the code and scale planes of the frozen KV8 format, one pair per side:
 *   k_codes, v_codes     int8 [n_kv_head][cells][head_dim]
 *   k_scales, v_scales   fp16 [n_kv_head][cells][head_dim / 32]
 * with cell = position (cell = cache row of attn_decode). The kernels read them in contiguous
 * 64-cell tiles and dequantize in registers (code * the f32 value of the stored fp16 scale);
 * there is no dequantized scratch and no arena term beyond `part`.
 *
 * kv_len is a device i32 word: token t has kv_len + t keys. window = 0: all of them; window =
 * W > 0: the last min(kv_len + t, W) positions. ring = 0: position p is cell p, and kv_len must
 * lie in [1, cells - (m - 1)]; ring = R > 0 (needs window > 0 and R >= window + m - 1, R <=
 * cells): position p is cell p % R. A kv_len outside its range makes the output NaN. part is f32
 * scratch of at least inference_cuda_attn_decode_q8_part_floats(m, n_kv_head, head_dim,
 * max_splits) floats (the layout of attn_decode), 16-byte aligned, no initialization needed.
 * out is f32 [m][n_head * head_dim]. max_splits (1..128) is the grid size; min_keys is a
 * MULTIPLE OF 64 (the tile of the KV8 kernel; attn_decode takes 32) and at most 2^20; scale
 * multiplies the scores. Enqueues two kernels on stream (NULL: the default stream): the partial
 * kernel and a fixed-order combine (no float atomic). Returns INFERENCE_CUDA_INVALID_ARGUMENT
 * before any launch for a bad argument.
 */
#ifndef INFERENCE_CUDA_ATTN_DECODE_Q8_H
#define INFERENCE_CUDA_ATTN_DECODE_Q8_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_attn_decode_q8_args {
    const void *q;
    const void *proj;      /* NULL (no gate) or the f32 projection rows */
    const void *k_codes;
    const void *k_scales;
    const void *v_codes;
    const void *v_scales;
    const void *kv_len;    /* device i32: keys visible to token 0 */
    void *part;
    void *out;
    int ld;
    int m;
    int n_head;
    int n_kv_head;
    int head_dim;
    int cells;             /* plane cells per head (the KV capacity) */
    int max_splits;
    int min_keys;
    int window;
    int ring;
    float scale;
} inference_cuda_attn_decode_q8_args;

size_t inference_cuda_attn_decode_q8_part_floats(int m, int n_kv_head, int head_dim,
                                                 int max_splits);
int inference_cuda_attn_decode_q8_launch(const inference_cuda_attn_decode_q8_args *args,
                                         void *stream, int *out_stage);
/* One raw, unscaled F32 sink per query head, read-only and disjoint from out and part.
 * Only the final combine adds its zero-value term. The original entry means no sinks. */
int inference_cuda_attn_decode_q8_sinks_launch(const inference_cuda_attn_decode_q8_args *args,
    const float *sinks, void *stream, int *out_stage);


// Paged m1 linear entries use the same struct. Both take a positive visible count.
// Partial uses q/KV/part only; combine uses part/proj/out only. Unused fields may be NULL.
// Let N be the global visible count, C the page capacity, and L the logical capacity:
// 1 <= N <= L <= 64*C. Partial uses cells=min_keys=C and visible_keys=min(C, N-group*C).
// Combine uses cells=L, min_keys=C and visible_keys=N. Populate groups [0, ceil(N/C))
// exactly once for one query/scale/geometry/scratch generation. Preceding pages hold C
// keys; the last holds the remainder. Keep q stable and exclude other scratch writers.
// Order every producer before combine on its stream or with explicit joins. These entries
// do not track group coverage or data provenance; the checked Rust pass enforces composition.
int inference_cuda_attn_decode_q8_partial_launch(const inference_cuda_attn_decode_q8_args *args,
    int visible_keys, int group, void *stream, int *out_stage);
int inference_cuda_attn_decode_q8_combine_launch(const inference_cuda_attn_decode_q8_args *args,
    int visible_keys, void *stream, int *out_stage);

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_attn_decode_q8_args) == 120 && offsetof(inference_cuda_attn_decode_q8_args, proj) == 8 && offsetof(inference_cuda_attn_decode_q8_args, k_scales) == 24 && offsetof(inference_cuda_attn_decode_q8_args, kv_len) == 48 && offsetof(inference_cuda_attn_decode_q8_args, part) == 56 && offsetof(inference_cuda_attn_decode_q8_args, out) == 64 && offsetof(inference_cuda_attn_decode_q8_args, ld) == 72,
              "inference_cuda_attn_decode_q8_args layout");
#else
_Static_assert(sizeof(inference_cuda_attn_decode_q8_args) == 120 && offsetof(inference_cuda_attn_decode_q8_args, proj) == 8 && offsetof(inference_cuda_attn_decode_q8_args, k_scales) == 24 && offsetof(inference_cuda_attn_decode_q8_args, kv_len) == 48 && offsetof(inference_cuda_attn_decode_q8_args, part) == 56 && offsetof(inference_cuda_attn_decode_q8_args, out) == 64 && offsetof(inference_cuda_attn_decode_q8_args, ld) == 72,
               "inference_cuda_attn_decode_q8_args layout");
#endif

#ifdef __cplusplus
}
#endif

#endif
