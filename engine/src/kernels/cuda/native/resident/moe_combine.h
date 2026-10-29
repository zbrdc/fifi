/*
 * Launch entry of the MoE combine (native/resident/moe_combine.h, the per-kernel header of the
 * layout in kernels/cuda/resident/mod.rs; abi_resident.h stays closed). Launch-only: nothing here
 * allocates, copies or synchronizes; a fault in queued work surfaces at the next synchronize.
 * Include after abi.h in C++. Rust mirrors the struct in native/resident/moe_combine.rs.
 *
 * For every token t of m and column j of n:
 *     s   = w[t][0] * down[t][0][j]                       (rounded product)
 *     s   = s + w[t][i] * down[t][i][j]   for i = 1 .. top_k - 1, in pick order
 *     s   = s + shared_gate[t] * shared[t][j]             only when shared_gate / shared are given
 *     out[t][j] = res[t][j] + s
 * Every product and every sum is rounded alone (no fused multiply-add) and the order is fixed: this
 * is the sequence of the pinned llama.cpp qwen35moe graph (ggml_mul of the expert outputs by the
 * weights, ggml_add of the picks in pick order, ggml_add of the gated shared expert, ggml_add of the
 * residual). The weights and the gate are applied AS GIVEN: the router already softmaxes and
 * renormalizes the weights and applies the sigmoid to the shared gate; this kernel applies no
 * second sigmoid or softmax and no renormalization. NaN propagates: a faulty router row (ids -1,
 * weights 0) makes matmul_expert write NaN to every down row, and 0 * NaN is NaN, so the output
 * row is NaN; no weight is tested for zero and nothing is clamped. (A router fault whose down rows
 * are finite would give res; the kernel trusts the NaN of matmul_expert.)
 *
 *   res          f32 [m][n], 16-byte aligned
 *   down         f32 rows (t * top_k + i) of n values at a stride of ld_down floats (ld_down >= n,
 *                multiple of 4), 16-byte aligned: the i-th pick of token t is row t * top_k + i.
 *                The decode step uses m = 1: the top_k outputs of matmul_expert, in pick (rank)
 *                order.
 *   weights      f32 [m][top_k], 4-byte aligned (router output, rank order)
 *   shared_gate  f32 [m] (router output, already a sigmoid), 4-byte aligned, or NULL
 *   shared       f32 [m][n], 16-byte aligned, or NULL; shared_gate and shared are both given or both NULL
 *   out          f32 [m][n], 16-byte aligned, written in full. It is either exactly res (in place) or
 *                shares no byte with res; it shares no byte with any other input.
 *   m in [1, 4096], n a multiple of 4 in [4, 65536], top_k in [1, 32], ld_down in [n, 2^26].
 * stream must not be NULL. A rejected argument returns INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA
 * call; otherwise the launch status of the kernel (one launch).
 */
#ifndef INFERENCE_CUDA_RESIDENT_MOE_COMBINE_H
#define INFERENCE_CUDA_RESIDENT_MOE_COMBINE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_moe_combine_args {
    const float *res;
    const float *down;
    const float *weights;
    const float *shared_gate;
    const float *shared;
    float *out;
    int m;
    int n;
    int top_k;
    int ld_down;
} inference_cuda_moe_combine_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_moe_combine_args) == 64 && offsetof(inference_cuda_moe_combine_args, res) == 0 &&
                  offsetof(inference_cuda_moe_combine_args, down) == 8 && offsetof(inference_cuda_moe_combine_args, weights) == 16 &&
                  offsetof(inference_cuda_moe_combine_args, shared_gate) == 24 && offsetof(inference_cuda_moe_combine_args, shared) == 32 &&
                  offsetof(inference_cuda_moe_combine_args, out) == 40 && offsetof(inference_cuda_moe_combine_args, m) == 48 &&
                  offsetof(inference_cuda_moe_combine_args, n) == 52 && offsetof(inference_cuda_moe_combine_args, top_k) == 56 &&
                  offsetof(inference_cuda_moe_combine_args, ld_down) == 60,
              "inference_cuda_moe_combine_args layout");
#else
_Static_assert(sizeof(inference_cuda_moe_combine_args) == 64 && offsetof(inference_cuda_moe_combine_args, res) == 0 &&
                   offsetof(inference_cuda_moe_combine_args, down) == 8 && offsetof(inference_cuda_moe_combine_args, weights) == 16 &&
                   offsetof(inference_cuda_moe_combine_args, shared_gate) == 24 && offsetof(inference_cuda_moe_combine_args, shared) == 32 &&
                   offsetof(inference_cuda_moe_combine_args, out) == 40 && offsetof(inference_cuda_moe_combine_args, m) == 48 &&
                   offsetof(inference_cuda_moe_combine_args, n) == 52 && offsetof(inference_cuda_moe_combine_args, top_k) == 56 &&
                   offsetof(inference_cuda_moe_combine_args, ld_down) == 60,
               "inference_cuda_moe_combine_args layout");
#endif

int inference_cuda_moe_combine_launch(const inference_cuda_moe_combine_args *args, void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_MOE_COMBINE_H */
