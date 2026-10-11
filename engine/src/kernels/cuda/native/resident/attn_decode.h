/*
 * RD-8b: split-KV decode attention entry (native/resident/attn_decode.h, per-kernel header).
 * Launch-only, like the entries of abi_resident.h. Include after abi.h in C++; Rust mirrors the
 * struct in native/resident/attn_decode.rs (child module of native/resident.rs).
 */
#ifndef INFERENCE_CUDA_ATTN_DECODE_H
#define INFERENCE_CUDA_ATTN_DECODE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Decode attention for a batch of m tokens (m <= 16) over an fp16 KV cache, query heads
 * grouped 8 per kv head (n_head == 8 * n_kv_head), head_dim 128 or 256. q is f32
 * [m][n_head][head_dim] (attn_prep's q); k_cache and v_cache are fp16
 * [cache_rows][n_kv_head][head_dim], 16-byte aligned. proj is NULL (no output gate) or the f32
 * projection row [m][ld] (ld >= n_head * 2 * head_dim) whose head h gate is at column
 * h * 2 * head_dim + head_dim. kv_len is a device i32 word: token t has kv_len + t keys
 * (positions 0 .. kv_len + t - 1), so attn_prep must have appended up to position
 * kv_len + m - 2 first. window = 0: token t attends to all its keys; window = W > 0: only to the
 * last min(kv_len + t, W) positions. ring = 0: position p is cache row p, and kv_len must lie in
 * [1, cache_rows - (m - 1)]; ring = R > 0 (needs window > 0 and R >= window + m - 1): position p
 * is cache row p % R and any kv_len >= 1 is valid. A kv_len outside its range makes the output
 * NaN. part is f32 scratch of at least inference_cuda_attn_decode_part_floats(m, n_kv_head,
 * head_dim, max_splits) floats, 16-byte aligned, no initialization needed. out is f32
 * [m][n_head * head_dim] = attention output (times sigmoid(gate) when proj is given).
 * max_splits (1..128) is the grid size; min_keys (a multiple of 32) is the fewest keys worth one
 * split; scale multiplies the scores. Enqueues two kernels on stream (NULL: the default
 * stream); returns INFERENCE_CUDA_INVALID_ARGUMENT before any launch for a bad argument. */
typedef struct inference_cuda_attn_decode_args {
    const void *q;
    const void *proj;
    const void *k_cache;
    const void *v_cache;
    const void *kv_len;
    void *part;
    void *out;
    int ld;
    int m;
    int n_head;
    int n_kv_head;
    int head_dim;
    int cache_rows;
    int max_splits;
    int min_keys;
    int window;
    int ring;
    float scale;
} inference_cuda_attn_decode_args;
size_t inference_cuda_attn_decode_part_floats(int m, int n_kv_head, int head_dim, int max_splits);
int inference_cuda_attn_decode_launch(const inference_cuda_attn_decode_args *args, void *stream,
                                      int *out_stage);

#ifdef __cplusplus
}
#endif

#endif
