/* CPF-V4 G2: the dense GEMM entry of the vendored-MMQ glue (decision 0049 items 3 and 11; plan
 * section B.4 "G2"). One launch of the vendored `mul_mat_q_case<type>` (mmq.cuh) computes
 *   dst[j*ld + col_off + i] = sum_k w[i][k] * x[j][k]      i < n, j < m
 * from the Native weight blocks (GGUF layout, row-major, read as they lie: the stock loaders)
 * and the q8_1 columns of `inference_cuda_mmq_quantize` (the D4/DS4 scale layout must be the
 * one of the weight's type, mmq.cuh:61-101). The entry only enqueues on the caller's stream,
 * never synchronizes and allocates nothing. One header per kernel file; no shared ABI header.
 *
 * Fill (`fill_dense_args` in the .cu, pinned field by field by the CPU test through
 * inference_cuda_mmq_dense_fill): ids_dst = expert_bounds = NULL (the kernel's dense branch
 * is exactly `ids_dst == NULL`, mmq.cuh:961/:1049), ncols_max = ncols_opt = m, one channel,
 * one sample, zero strides, `stride_row_x = k / qk` blocks, `nrows_dst = ld`, and
 * `dst = base + col_off` so a GEMM can write its slice of a fused [m][ld] destination.
 *
 * Sizing (plan section B.3): w holds n*(k/qk) blocks; y `q8_1_bytes(k, m)` =
 * (k/128)*m*144 + 18,432 (the GEMM's y-tile copies may read the pad, never write it); dst
 * ((m-1)*ld + col_off + n) floats; fixup at most nsm*128*128 floats (mmq.cuh:1334-1338),
 * which the Rust wrapper enforces as a floor. The C entry also refuses, as an invalid
 * argument, a fixup range below the exact bytes the launch demands for the tile width it
 * picks on the device it reads (`blocks * J * I` floats, 0 when the launch tiles): the C
 * ABI is a boundary of its own (MSG-85 B2). That rule needs the device facts, so only the
 * launching entry applies it, after the facts query (a CUDA attribute read, the one CUDA call
 * before it) and before anything is enqueued; the vendored require (mmq.cuh:1409) stays as
 * the backstop. Plan section B.0 (c) said a short range reaches that require; since lane C's
 * ruling it meets this refusal first (status -1, not cudaErrorInvalidValue).
 *
 * Refusals (INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call; the Rust wrapper mirrors
 * every rule so the native entry cannot latch the backend): a weight type with no dense case
 * (anything but Q8_0, Q4_K, Q6_K), k off the 256-value K grid or above 32768 (the k loop reads
 * 256 values per iteration with no k clamp, mmq.cuh:856; checked before the qk rule, which it
 * implies), m outside 1..=2048 (the chunk cap), n below 1, ld below col_off + n, a destination
 * reach `(m-1)*ld + n` above INT_MAX (the kernels index dst in 32-bit int), a tile grid
 * of 2^30 or more (`ceil(m/8)*ceil(n/128)*k/qk`, bounded with the smallest J, mmq.cuh:1402),
 * a NULL or misaligned pointer, and a written range (dst, fixup) that overlaps another range.
 */
#ifndef INFERENCE_CUDA_PREFILL_MMQ_DENSE_H
#define INFERENCE_CUDA_PREFILL_MMQ_DENSE_H

#include <stddef.h>
#include <stdint.h>

#include "prefill/mmq_facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_mmq_dense_args {
    const void *w;      /* Native blocks [n][k/qk], 16-byte aligned */
    const void *y;      /* block_q8_1_mmq columns [k/128][m], 16-byte aligned */
    void *dst;          /* f32 destination base, 4-byte aligned */
    void *fixup;        /* stream-k fixup range, 16-byte aligned; NULL only with fixup_bytes 0 */
    size_t fixup_bytes; /* bytes of the fixup range */
    int type;           /* ggml_type of the weight: Q8_0 = 8, Q4_K = 12, Q6_K = 14 */
    int k;              /* values per weight row: 256, 512, ..., 32768 */
    int n;              /* weight rows = output values per column */
    int m;              /* activation columns: 1..=2048 */
    int ld;             /* floats between two dst columns: >= col_off + n */
    int col_off;        /* floats before this GEMM's slice inside a dst column */
} inference_cuda_mmq_dense_args;

/* The vendored mmq_args as the entry fills it, field for field minus the device facts. Only
 * the CPU fill pin reads it; the launch never goes through it. */
typedef struct inference_cuda_mmq_args_view {
    const void *x;
    const void *y;
    const void *ids_dst;
    const void *expert_bounds;
    void *dst;
    const void *y_scale;
    void *fixup;
    size_t fixup_bytes;
    int64_t ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, nrows_dst;
    int64_t nchannels_x, nchannels_y, stride_channel_x, stride_channel_y, stride_channel_dst;
    int64_t nsamples_x, nsamples_y, stride_sample_x, stride_sample_y, stride_sample_dst;
    int64_t ncols_max, ncols_opt;
    int type_x;
    int reserved; /* zero */
} inference_cuda_mmq_args_view;

/* Queues the dense GEMM on stream. Returns INFERENCE_CUDA_OK, an argument refusal, the CUDA
 * error of the facts query or the launch (out_stage INFERENCE_CUDA_STAGE_LAUNCH), or
 * INFERENCE_CUDA_NATIVE_EXCEPTION. */
int inference_cuda_mmq_dense(const inference_cuda_mmq_dense_args *a, void *stream, int *out_stage);

/* The bytes of stream-k fixup range the vendored launch demands for `a` on a device with
 * `facts`: the tile width it picks, the tiling-mode test, then `blocks * J * I` floats, 0
 * when the launch tiles, when no tile width fits the shared memory, for a NULL pointer, a
 * type with no dense case, m outside 1..=2048, n below 1 or an SM count of 0. Every other
 * argument is computed, not validated: the entry's own rules refuse them before the
 * launching entry calls this with the facts it queried. Pure: no CUDA call, and only the two
 * structs are read (no pointer inside `a` is followed), so a host test can sweep it against
 * the Rust sizing over any device facts. */
size_t inference_cuda_mmq_dense_fixup_need(const inference_cuda_mmq_dense_args *a,
                                           const inference_cuda_mmq_facts *facts);

/* The entry minus the device: validates `a` as inference_cuda_mmq_dense does before it reads
 * the device (every refusal of the list above except the fixup size), then writes the fill
 * to *out (host only, no CUDA call, addresses compared and never dereferenced). Returns
 * INFERENCE_CUDA_OK or INFERENCE_CUDA_INVALID_ARGUMENT for a NULL pointer or such a refusal. */
int inference_cuda_mmq_dense_fill(const inference_cuda_mmq_dense_args *a,
                                  inference_cuda_mmq_args_view *out);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_mmq_dense_args) == 64 &&
                  offsetof(inference_cuda_mmq_dense_args, w) == 0 &&
                  offsetof(inference_cuda_mmq_dense_args, y) == 8 &&
                  offsetof(inference_cuda_mmq_dense_args, dst) == 16 &&
                  offsetof(inference_cuda_mmq_dense_args, fixup) == 24 &&
                  offsetof(inference_cuda_mmq_dense_args, fixup_bytes) == 32 &&
                  offsetof(inference_cuda_mmq_dense_args, type) == 40 &&
                  offsetof(inference_cuda_mmq_dense_args, k) == 44 &&
                  offsetof(inference_cuda_mmq_dense_args, n) == 48 &&
                  offsetof(inference_cuda_mmq_dense_args, m) == 52 &&
                  offsetof(inference_cuda_mmq_dense_args, ld) == 56 &&
                  offsetof(inference_cuda_mmq_dense_args, col_off) == 60,
              "inference_cuda_mmq_dense_args layout");
static_assert(sizeof(inference_cuda_mmq_args_view) == 216 &&
                  offsetof(inference_cuda_mmq_args_view, fixup_bytes) == 56 &&
                  offsetof(inference_cuda_mmq_args_view, ncols_x) == 64 &&
                  offsetof(inference_cuda_mmq_args_view, ncols_opt) == 200 &&
                  offsetof(inference_cuda_mmq_args_view, type_x) == 208,
              "inference_cuda_mmq_args_view layout");
#else
_Static_assert(sizeof(inference_cuda_mmq_dense_args) == 64, "dense args size");
_Static_assert(sizeof(inference_cuda_mmq_args_view) == 216, "args view size");
#endif

#endif /* INFERENCE_CUDA_PREFILL_MMQ_DENSE_H */
