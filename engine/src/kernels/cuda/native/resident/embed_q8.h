/*
 * Launch entry of embed_q8: gather m rows of a raw Q8_0 table (n_rows rows of k
 * values, 34-byte blocks) into m f32 rows. Launch-only.
 */
#ifndef INFERENCE_CUDA_RESIDENT_EMBED_Q8_H
#define INFERENCE_CUDA_RESIDENT_EMBED_Q8_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_embed_q8_args {
    void *out;         /* f32 [m][k], written, 16-byte aligned */
    const void *w;     /* Q8_0 blocks [n_rows][k/32], read */
    const void *ids;   /* i32 [m], read */
    int m;             /* rows, 1..4096 */
    int n_rows;        /* table rows, 1..268435456 */
    int k;             /* values per row, a multiple of 32, 32..32768 */
} inference_cuda_embed_q8_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_embed_q8_args) == 40 && offsetof(inference_cuda_embed_q8_args, out) == 0 && offsetof(inference_cuda_embed_q8_args, w) == 8 && offsetof(inference_cuda_embed_q8_args, ids) == 16 && offsetof(inference_cuda_embed_q8_args, m) == 24 && offsetof(inference_cuda_embed_q8_args, n_rows) == 28 && offsetof(inference_cuda_embed_q8_args, k) == 32, "layout");
#else
_Static_assert(sizeof(inference_cuda_embed_q8_args) == 40 && offsetof(inference_cuda_embed_q8_args, out) == 0 && offsetof(inference_cuda_embed_q8_args, w) == 8 && offsetof(inference_cuda_embed_q8_args, ids) == 16 && offsetof(inference_cuda_embed_q8_args, m) == 24 && offsetof(inference_cuda_embed_q8_args, n_rows) == 28 && offsetof(inference_cuda_embed_q8_args, k) == 32, "layout");
#endif

/* One launch on stream. An id outside 0..n_rows writes a NaN row; the wrapper
 * refuses host-known bad ids before the call. Returns INFERENCE_CUDA_INVALID_ARGUMENT
 * before any CUDA call for a bad argument, else the launch status. */
int inference_cuda_embed_q8_launch(const inference_cuda_embed_q8_args *args, void *stream,
                                   int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_EMBED_Q8_H */
