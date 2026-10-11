// MIT License, Copyright (c) 2026 Dan Campos (see ../LICENSE).
// Restored from xavier-qwen 2f9183c src/kernels.h (RD-4V); provenance.json lists the line
// ranges taken unchanged and the patches applied since.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace kern {

constexpr int kHeadDim = 256;   // full-attention head size
constexpr int kStateDim = 128;  // DeltaNet head size (keys and values)

// Every entry runs on `stream` (nullptr: the default stream) and returns the launch status; none
// allocates, synchronizes or ends the process. Shapes are not checked: the caller owns them.

// h += delta (when delta is non-null), then xf = RMSNorm(h) * w. Needs n % 256 == 0. When norm_out
// is non-null it also receives the f32 row RMSNorm(h) * w [m][n] (the router input), equal to the
// output of rmsnorm() on the updated h.
cudaError_t add_rmsnorm_pack(float* h, const float* delta, const float* w, int m, int n,
                             float eps, uint4* xf, float* norm_out = nullptr,
                             cudaStream_t stream = nullptr);

// out = RMSNorm(x) * w for x [m][n].
cudaError_t rmsnorm(const float* x, const float* w, int m, int n, float eps, float* out,
                    cudaStream_t stream = nullptr);

// xf = silu(gate) * up for gu [m][2 * n_ff] holding gate then up. Needs n_ff % 256 == 0.
cudaError_t swiglu_pack(const float* gu, int m, int n_ff, uint4* xf,
                        cudaStream_t stream = nullptr);

// Causal depthwise conv (kernel 4) + silu over the first `channels` columns of
// proj [m][ld]; state [3][channels] holds the previous inputs, oldest first.
// When snap is non-null, snap[t] receives the state after token t < m - 1.
struct ConvArgs {
  const float* proj;
  int ld, m, channels;
  const float* w;
  float* state;
  float* snap;  // [m - 1][3][channels] or null
  float* out;   // [m][channels]
};
// w is [channels][4] f32, 16-byte aligned (one float4 per channel).
cudaError_t conv_silu(const ConvArgs& a, cudaStream_t stream = nullptr);

// Gated DeltaNet recurrence for one layer (autoregressive over the m tokens).
struct DeltaArgs {
  const float* conv;  // [m][conv_ld]: q (n_k_heads x 128) | k | v (n_v_heads x 128)
  int conv_ld;
  const float* proj;  // in-projection rows [m][ld] holding z, beta and alpha
  int ld, z_off, beta_off, alpha_off;
  const float* a;        // ssm_a = -exp(A_log), per v head
  const float* dt_bias;  // per v head
  const float* norm_w;   // ssm_norm, 128
  float* state;          // [n_v_heads][128 (k)][128 (v)]
  float* snap;           // null, or [m - 1][state]: state after token t < m - 1
  float* out;            // [m][n_v_heads * 128]
  int m, n_k_heads, n_v_heads;
  float eps;
};
// Value head h reads key head h % n_k_heads (the GGUF layout, tiled), so n_v_heads % n_k_heads
// must be 0; the gate is silu(z).
cudaError_t gated_delta(const DeltaArgs& a, cudaStream_t stream = nullptr);

// Attention preparation for one layer (attn_prep): Q/K RMSNorm, NEOX rope on the first n_rot dims,
// KV cache write. It reads proj, ld, n_head, n_kv_head, n_rot, q_norm, k_norm, eps, rope_base,
// k_cache, v_cache, q, pos_dev, pos0 and m; the other fields belong to the attention kernels.
struct AttnArgs {
  const float* proj;  // [m][ld]: per q head [q 256 | gate 256], then k, then v
  int ld, n_head, n_kv_head, n_rot;
  const float* q_norm;
  const float* k_norm;
  float eps, rope_base, scale;
  __half* k_cache;  // this layer: [ctx][n_kv_head][256]
  __half* v_cache;
  float* q;    // scratch [m][n_head][256]
  float* out;   // [m][n_head * 256]
  float* part;  // scratch [m <= kMaxTokens][n_head][kAttnSplits][kAttnPartLd]
  int pos0, m;
  // Null, or a device word holding the position of token 0 (replaces pos0). The kernel reads it
  // when it runs, so one captured graph serves every position.
  const int* pos_dev = nullptr;
};
cudaError_t attn_prep(const AttnArgs& a, cudaStream_t stream = nullptr);

}  // namespace kern
