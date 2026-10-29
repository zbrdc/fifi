/* The q8_1 quantize entries of the vendored-MMQ glue. Every MMQ GEMM input is requantized
 * from f32 into `block_q8_1_mmq` (mmq.cuh:28-47: 128 values, 144 B, k-block-major columns,
 * `ib = k_block*ne1 + column`, quantize.cu:530-533), with the D4/DS4 scale layout selected by
 * the WEIGHT type the GEMM reads (mmq.cuh:61-101: Q8_0/Q6_K/Q5_0 take D4, Q4_K takes DS4).
 * One header per kernel file; no shared ABI header.
 *
 * Two entries, both asynchronous on the caller's stream, never synchronizing, allocating
 * nothing and reading no status word:
 *   inference_cuda_mmq_quantize         quantize.cu:576-604 `quantize_mmq_q8_1_cuda`.
 *     x f32 [rows][row_stride] (16-byte aligned base, float4 loads, quantize.cu:485-492,
 *     zero past ne00) -> y, one column per source row, or - with ids != NULL - one column
 *     per map entry: column j quantizes source row ids[j] (the gather of the down-projection
 *     input; the map VALUES are trusted device indices, guarded upstream by the ids builder).
 *   inference_cuda_mmq_quantize_scatter quantize.cu:607-634 `quantize_scatter_mmq_q8_1_cuda`.
 *     Each token is quantized once and the block written to all `top_k` compact rows its
 *     inverse-map slots name (quantize.cu:524-530); the map values are trusted (the ids
 *     builder guards them).
 *
 * Sizing: y holds `q8_1_bytes(k, ncols)` bytes = `(kpad/128)*ncols*144 + 18,432` with the
 * ONE constant pad (the GEMM's y-tile copies read past the last column; quantize never writes
 * the pad); the source holds `(ncols - 1)*row_stride + k` floats, ids `ncols` ints,
 * ids_src1_inv `m*top_k` ints.
 *
 * Refusals (INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call; the Rust wrapper mirrors
 * every rule so the native entry cannot latch the backend): k off the 256-value K grid or
 * above 32768 (MMQ_ITER_K: the GEMM reads 256-value k strides with no k clamp, so a buffer it
 * cannot read safely is never built), row_stride not a multiple of 4 or below k or above 2^30
 * (float4 row alignment), ncols/m/top_k outside the domain (1..=2048 columns plain and
 * tokens, `m*top_k` in 1..=16384 gathered and scattered, top_k in 1..=32), a weight type with
 * no q8_1 ds layout (MMQ_Q8_1_DS_LAYOUT_INVALID), a NULL or misaligned pointer, and an output
 * range that overlaps an input range.
 */
#ifndef INFERENCE_CUDA_PREFILL_MMQ_QUANT_H
#define INFERENCE_CUDA_PREFILL_MMQ_QUANT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The plain call: `ncols` output columns over `ncols` source rows (ids == NULL), or the
 * gathered call (ids != NULL): `ncols` output columns over `ncols` source rows read through
 * the permutation `ids` (at the resident stage 15 ncols = m * top_k). */
typedef struct inference_cuda_mmq_quantize_args {
    const void *x;   /* f32 [rows][row_stride], 16-byte aligned */
    const void *ids; /* NULL, or i32 [ncols] gather map, 4-byte aligned */
    void *y;         /* block_q8_1_mmq, 16-byte aligned */
    int type;        /* ggml_type discriminant of the consuming GEMM's weight */
    int k;           /* values per row quantized: 256, 512, ..., 32768 */
    int row_stride;  /* floats per source row: multiple of 4, k..=2^30 */
    int ncols;       /* output columns: 1..=2048 (ids == NULL), 1..=16384 (ids != NULL) */
    int packing;     /* 0: vendor, 1: CPU-reference Q8_0/Q4_K packing */
} inference_cuda_mmq_quantize_args;

/* The scatter call: m source rows, m*top_k compact output columns. */
typedef struct inference_cuda_mmq_quantize_scatter_args {
    const void *x;              /* f32 [m][row_stride], 16-byte aligned */
    const void *ids_src1_inv;   /* i32 [m*top_k] inverse map, 4-byte aligned */
    void *y;                    /* block_q8_1_mmq, 16-byte aligned */
    int type;                   /* ggml_type discriminant of the consuming GEMM's weight */
    int k;                      /* values per row quantized: 256, 512, ..., 32768 */
    int row_stride;             /* floats per source row: multiple of 4, k..=2^30 */
    int m;                      /* source rows (tokens): 1..=2048 */
    int top_k;                  /* compact rows per token: 1..=32, m*top_k <= 16384 */
    int packing;                /* 0: vendor, 1: CPU-reference Q8_0/Q4_K packing */
} inference_cuda_mmq_quantize_scatter_args;

/* Queues the gather/plain quantize on stream. Returns INFERENCE_CUDA_OK, an argument
 * refusal, the CUDA error of the launch (out_stage INFERENCE_CUDA_STAGE_LAUNCH), or
 * INFERENCE_CUDA_NATIVE_EXCEPTION. */
int inference_cuda_mmq_quantize(const inference_cuda_mmq_quantize_args *a, void *stream,
                                int *out_stage);

/* Queues the scatter quantize on stream; the same status contract. */
int inference_cuda_mmq_quantize_scatter(const inference_cuda_mmq_quantize_scatter_args *a,
                                        void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_mmq_quantize_args) == 48 &&
                  offsetof(inference_cuda_mmq_quantize_args, x) == 0 &&
                  offsetof(inference_cuda_mmq_quantize_args, ids) == 8 &&
                  offsetof(inference_cuda_mmq_quantize_args, y) == 16 &&
                  offsetof(inference_cuda_mmq_quantize_args, type) == 24 &&
                  offsetof(inference_cuda_mmq_quantize_args, k) == 28 &&
                  offsetof(inference_cuda_mmq_quantize_args, row_stride) == 32 &&
                  offsetof(inference_cuda_mmq_quantize_args, ncols) == 36 &&
                  offsetof(inference_cuda_mmq_quantize_args, packing) == 40,
              "inference_cuda_mmq_quantize_args layout");
static_assert(sizeof(inference_cuda_mmq_quantize_scatter_args) == 48 &&
                  offsetof(inference_cuda_mmq_quantize_scatter_args, x) == 0 &&
                  offsetof(inference_cuda_mmq_quantize_scatter_args, ids_src1_inv) == 8 &&
                  offsetof(inference_cuda_mmq_quantize_scatter_args, y) == 16 &&
                  offsetof(inference_cuda_mmq_quantize_scatter_args, type) == 24 &&
                  offsetof(inference_cuda_mmq_quantize_scatter_args, k) == 28 &&
                  offsetof(inference_cuda_mmq_quantize_scatter_args, row_stride) == 32 &&
                  offsetof(inference_cuda_mmq_quantize_scatter_args, m) == 36 &&
                  offsetof(inference_cuda_mmq_quantize_scatter_args, top_k) == 40 &&
                  offsetof(inference_cuda_mmq_quantize_scatter_args, packing) == 44,
              "inference_cuda_mmq_quantize_scatter_args layout");
#else
_Static_assert(sizeof(inference_cuda_mmq_quantize_args) == 48, "quantize args size");
_Static_assert(sizeof(inference_cuda_mmq_quantize_scatter_args) == 48, "scatter args size");
#endif

#endif /* INFERENCE_CUDA_PREFILL_MMQ_QUANT_H */
