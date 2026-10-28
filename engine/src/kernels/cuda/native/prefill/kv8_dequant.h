/* Expand a stored KV8 prefix into the time-major FP16 cache read by attn_prefill.
 *
 * codes: int8 [n_kv_head][capacity][head_dim].
 * scales: fp16 [n_kv_head][capacity][head_dim / 32], one stored scale per 32 codes.
 * out: fp16 [rows][n_kv_head][head_dim]. No suffix or padding byte is written.
 * Each value is RNE_f16(float(code) * float(stored_scale)); inf/NaN propagate, no clipping.
 *
 * Rules, checked before any CUDA call: capacity in 1..2^30-1; rows in 1..capacity;
 * n_kv_head in 1..64; head_dim a multiple of 32 in 32..1024. Pointers and stream are
 * non-NULL; codes/scales are 4-byte aligned and out is 16-byte aligned. Written out
 * shares no byte with either read extent. Address extents must not wrap.
 * The launch is asynchronous on the supplied stream; no allocation, copy, status word
 * or synchronization. Source head stride is capacity, never the active row count.
 */
#ifndef INFERENCE_CUDA_PREFILL_KV8_DEQUANT_H
#define INFERENCE_CUDA_PREFILL_KV8_DEQUANT_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_kv8_dequant_args {
    const void *codes;
    const void *scales;
    void *out;
    int capacity;
    int rows;
    int n_kv_head;
    int head_dim;
} inference_cuda_kv8_dequant_args;

int inference_cuda_kv8_dequant_launch(const inference_cuda_kv8_dequant_args *args,
                                      void *stream, int *out_stage);

/* Same layout and extents, but out row r reads absolute source row source_row_origin+r.
 * The nonnegative origin plus rows must fit capacity; output still starts at stage row0. */
int inference_cuda_kv8_dequant_segment_launch(const inference_cuda_kv8_dequant_args *args,
                                              int source_row_origin, void *stream,
                                              int *out_stage);

#ifdef __cplusplus
}
static_assert(sizeof(inference_cuda_kv8_dequant_args) == 40 &&
              offsetof(inference_cuda_kv8_dequant_args, codes) == 0 &&
              offsetof(inference_cuda_kv8_dequant_args, scales) == 8 &&
              offsetof(inference_cuda_kv8_dequant_args, out) == 16 &&
              offsetof(inference_cuda_kv8_dequant_args, capacity) == 24 &&
              offsetof(inference_cuda_kv8_dequant_args, rows) == 28 &&
              offsetof(inference_cuda_kv8_dequant_args, n_kv_head) == 32 &&
              offsetof(inference_cuda_kv8_dequant_args, head_dim) == 36,
              "inference_cuda_kv8_dequant_args layout");
#else
_Static_assert(sizeof(inference_cuda_kv8_dequant_args) == 40,
               "inference_cuda_kv8_dequant_args layout");
#endif
#endif
