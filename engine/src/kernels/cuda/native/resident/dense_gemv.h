/* Dense matvec for floating-point and packed GGUF weights, per-kernel header
 * (native/resident/dense_gemv.h). Included by dense_gemv.cu and read by the Rust FFI mirror;
 * nothing else.
 *
 * For each of m activation rows t and each weight row r:
 *   y[t * ld_y + r] = sum_c w[r][c] * x[t * ld_x + c]
 * with w a dense row-major [rows][cols] matrix in its stored dtype (0: binary16, 1: bfloat16,
 * 2: original GGUF NVFP4 blocks of 36 bytes per 64 columns, 3: binary32,
 * 256 + GGUF type ID: supported original packed GGUF blocks, see gguf_gemv.cuh),
 * widened exactly to f32 in registers, x and y f32. Products and sums are f32: a lane adds its
 * columns in ascending order with fused multiply-adds, and the 32 lanes of a warp combine by a
 * fixed xor butterfly, so a result depends on cols and the alignment of w, never on m, rows or
 * the other rows. `ld_y` lets the result land inside a wider row; a result element is written
 * exactly once. Callers apply any separate weight multipliers.
 *
 * Limits: m in [1, 32768], rows in [1, 1048576], cols in [1, 12288] (a row of x is staged in
 * shared memory), ld_x in [cols, 1048576], ld_y in [rows, 1048576]. x and y are 4-byte aligned,
 * w 4-byte aligned for binary32 and 2-byte aligned otherwise. Floating-point weights use
 * vector loads when every row is 16-byte aligned. y overlaps
 * neither x nor w (x and w are read only). Enqueues one kernel on stream (not NULL) and returns
 * INFERENCE_CUDA_INVALID_ARGUMENT before any launch for a bad argument. */
#ifndef INFERENCE_CUDA_RESIDENT_DENSE_GEMV_H
#define INFERENCE_CUDA_RESIDENT_DENSE_GEMV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_dense_gemv_args {
    const void *w;
    const void *x;
    void *y;
    int m;
    int rows;
    int cols;
    int ld_x;
    int ld_y;
    int weight_dtype;
    /* INFERENCE_CUDA_DENSE_GEMV_* bits chosen by the caller's context. */
    int flags;
} inference_cuda_dense_gemv_args;

/* The single-row split kernels: one CTA per output with K across four warps. */
enum { INFERENCE_CUDA_DENSE_GEMV_DECODE_SPLIT = 1 };

int inference_cuda_dense_gemv_launch(const inference_cuda_dense_gemv_args *args, void *stream,
                                     int *out_stage);

/* Internal independent single-row Inkling attention projections (6 or 8). */
int inference_cuda_decode_attention_group(const inference_cuda_dense_gemv_args *calls,
                                          int count, void *stream, int *out_stage);

/* Internal opt-in paired controller projections. Independent F32 matrices and
 * disjoint outputs, both M=1, K=64, N=2304, sharing the same input. */
int inference_cuda_decode_control_pair(const inference_cuda_dense_gemv_args *a,
                                       const inference_cuda_dense_gemv_args *b,
                                       void *stream, int *out_stage);

/* Opt-in: pair independent F32 activation rows sharing one dense Q8_0 matrix.
 * Same operand checks, strides and arithmetic as packed GEMV. Selects an active
 * SM75 device only, for (N,K)=(1024,2304) or (3072,2304) at M=7..12, or
 * (2304,3072) at M=63..64. Only the last shape honors prefer_four. Other valid
 * calls use ordinary GEMV. prefer_four=-1 selects exact all-row visit instead.
 * Optional out_width reports 2, 4, m for a visit, or 0. No retry. */
int inference_cuda_gguf_q8_row_reuse_width_launch(const inference_cuda_dense_gemv_args *args,
                                                  void *stream, int *out_stage, int *out_width,
                                                  int prefer_four);

/* Compatibility entry used by the standalone pair probe; out_pair is boolean. */
int inference_cuda_gguf_q8_row_reuse_launch(const inference_cuda_dense_gemv_args *args,
                                            void *stream, int *out_stage, int *out_pair);

/* Opt-in: args describe one already-grouped expert matrix and
 * its F32 activation rows. Same layout, checks and per-row arithmetic as
 * packed GEMV. Tiles IQ2_XXS (16) and Q2_0 (42) only, for 2 <= m <= 512 and
 * rows, cols <= 2304. Other valid calls use dense GEMV, never WMMA or MMQ.
 * Optional host out_tile reports 2 or 4 after a tiled enqueue, 0 otherwise. */
int inference_cuda_gguf_exact_expert_tile_launch(const inference_cuda_dense_gemv_args *args,
                                                 void *stream, int *out_stage, int *out_tile);

/* Tiled FP16-operand FP32-accumulating GEMM with the same external F32 activation
 * and output layouts. This opt-in entry uses Tensor Cores and accepts the
 * original packed blocks of IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, NVFP4 and Q2_0.
 * Columns must be a multiple of their block size. Weight and input half rounding
 * can change the output from the F32 matvec. No global weight staging. */
int inference_cuda_gguf_wmma_launch(const inference_cuda_dense_gemv_args *args, void *stream,
                                     int *out_stage);

/* Combine two independent exact expert calls; used=0 declines without enqueue. */
int inference_cuda_inkling_ragged_pair_launch(const inference_cuda_dense_gemv_args *two,void *stream,int *stage,int *used);

/* Exact F32 SM75 controller down at measured visit widths. */
int inference_cuda_inkling_down_launch(const inference_cuda_dense_gemv_args *args,void *stream,int *stage);

/* Decode an expert to borrowed aligned half panels then run WMMA. used=0
 * declines without enqueue; a launch error never requests a fallback. */
int inference_cuda_inkling_panel_launch(const inference_cuda_dense_gemv_args *args,void *scratch,
    size_t scratch_bytes,void *stream,int *stage,int *used);

/* Adjacent independent SM75 Q8 projections. used=0 declines without enqueue. */
int inference_cuda_inkling_qkv_visit_launch(const inference_cuda_dense_gemv_args *args,int count,void *stream,int *stage,int *used);

/* Exact multicodec packed tile: variant 1 uses eight weight rows per CTA. */
int inference_cuda_inkling_packed_visit_launch(const inference_cuda_dense_gemv_args *args, void *stream, int *stage, int variant);

/* F32 gain/forget heads. used=0 means an ineligible pair, no enqueue. */
int inference_cuda_inkling_controller_pair_launch(const inference_cuda_dense_gemv_args *a,
    const inference_cuda_dense_gemv_args *b, void *stream, int *stage, int *used);

/* Opt-in IQ2_XS gate/up tile with the deployed WMMA accumulation order. */
int inference_cuda_inkling_iq2_visit_launch(const inference_cuda_dense_gemv_args *args, void *stream,
                                            int *out_stage);

/* Contiguous GGUF expert stacks. Rank order is the order of picks. Scratch holds
 * count * (3 * middle + hidden) f32 values. The launch queues the two indexed
 * projections, clipped gate activation and weighted sum on one stream. */
enum { INFERENCE_CUDA_GGUF_MOE_MAX_PICKS = 8 };
typedef struct inference_cuda_gguf_moe_pick {
    uint32_t expert;
    float gate_scale;
    float down_scale;
    float weight;
} inference_cuda_gguf_moe_pick;

typedef struct inference_cuda_gguf_moe_args {
    const void *gate;
    const void *down;
    const float *x;
    float *scratch;
    float *y;
    int hidden;
    int middle;
    int experts;
    int count;
    int gate_dtype;
    int down_dtype;
    float gate_cap;
    float up_cap;
    inference_cuda_gguf_moe_pick picks[INFERENCE_CUDA_GGUF_MOE_MAX_PICKS];
    /* All present only for device-produced picks over contiguous stacks. Scales are [experts]. */
    const int *device_ids;
    const float *device_weights;
    const float *gate_scales;
    const float *down_scales;
} inference_cuda_gguf_moe_args;

/* Optional matrix addresses in routing rank order, for a mixture of cached
 * and staged experts. A null matrix list selects the contiguous stacks. */
typedef struct inference_cuda_gguf_moe_matrices {
    const void *gate[INFERENCE_CUDA_GGUF_MOE_MAX_PICKS];
    const void *down[INFERENCE_CUDA_GGUF_MOE_MAX_PICKS];
} inference_cuda_gguf_moe_matrices;

/* Optional in/out schedule: gate bits 0–1 encode packed (0), grouped IQ2 (1),
 * or integer dot (2); bit 2 requests the exact Q2_0 down schedule independently.
 * Reports only the kernels enqueued. NULL always uses the baseline. */
enum {
    INFERENCE_CUDA_GGUF_MOE_GATE_MASK = 3,
    INFERENCE_CUDA_GGUF_MOE_EXACT_Q2_DOWN = 4,
    /* Request only: the gate, down and packed launches take the split kernels. */
    INFERENCE_CUDA_GGUF_MOE_DECODE_SPLIT = 8,
    /* Request only, one of the two: launch only the activation, which reads gate and up
     * products a caller wrote at the start of the scratch, or only the weighted sum, which
     * reads down products a caller wrote after the middle rows. */
    INFERENCE_CUDA_GGUF_MOE_ACTIVATE_ONLY = 16,
    INFERENCE_CUDA_GGUF_MOE_SUM_ONLY = 32
};
int inference_cuda_gguf_moe_launch(const inference_cuda_gguf_moe_args *args,
                                  const inference_cuda_gguf_moe_matrices *matrices,
                                  void *stream, int *out_stage, int *schedule);

#ifdef __cplusplus
}
static_assert(sizeof(inference_cuda_dense_gemv_args) == 56 &&
              offsetof(inference_cuda_dense_gemv_args, flags) == 48,
              "dense GEMV args layout");
static_assert(sizeof(inference_cuda_gguf_moe_args) == 232 &&
              offsetof(inference_cuda_gguf_moe_args, device_ids) == 200 &&
              offsetof(inference_cuda_gguf_moe_args, device_weights) == 208 &&
              offsetof(inference_cuda_gguf_moe_args, gate_scales) == 216 &&
              offsetof(inference_cuda_gguf_moe_args, down_scales) == 224,
              "GGUF expert args layout");
#endif

#endif
