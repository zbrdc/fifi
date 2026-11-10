/* Launch entry of gdn_scan_seq, the sequential gated-delta scan of one chunk, per-kernel
 * header; abi_resident.h stays closed.
 *
 * What it computes, per value head h (key head kh = h % n_k_heads, tiled, the GGUF layout;
 * D = head_dim = 128), per token t of m, in this order (the vendored element order,
 * ops.cu:132-139, with the two forms this path deliberately differs in marked). `form` 0 is
 * this description exactly (the Qwen3.6 GGUF form); each form bit swaps one rule:
 *   GROUPED_HEADS  kh = h / (n_v_heads / n_k_heads) (consecutive value heads share a key
 *                  head, the Hugging Face layout) instead of the tiled h % n_k_heads;
 *   ADD_EPS_L2     the q/k L2 norms take rsqrt(sum(x^2) + eps) (the Hugging Face l2norm)
 *                  instead of the ggml rsqrt(fmax(sum(x^2), eps^2)); the q scale rsqrt(D)
 *                  stays;
 *   SIGMOID_GATE   the output gate is sigmoid(z) = 1 / (1 + expf(-z)) instead of silu(z).
 * Qwen3.8 Flash-Next uses all three. Each form is its own template instance, so the form-0
 * kernels are the same code as before the bits existed.
 *   1. qv/kv: the head's D values of the conv row t (layout q | k | v of gdn_delta.h);
 *      vv: the head's D values of the v plane.
 *   2. q/k L2 norms in the GGML form (norm.cu:273), NOT the decode rsqrt(sum + eps):
 *      qhat = q * rsqrt(fmax(sum(q^2), eps^2)) * rsqrt(D) (the norm and the attention scale
 *      folded into one multiplier); khat = k * rsqrt(fmax(sum(k^2), eps^2)).
 *   3. gates, in kernel: decay = exp(softplus(alpha[h] + dt_bias[h]) * a[h]) with a = ssm_a
 *      AS STORED (already -exp(A_log); no second exp), softplus(x) = x > 20 ? x :
 *      log1pf(expf(x)); beta = 1 / (1 + expf(-beta_logit)). expf, never __expf: this file
 *      is in the exactness-critical group.
 *   4. state update: S *= decay; sk = S^T k (against the DECAYED state);
 *      d = (v - sk) * beta; S += k d^T; o = S^T q. State f32 [n_v_heads][128 key i][128
 *      value j], j fastest (gdn_delta.h), read and written in place, never fp16.
 *   5. o written PRE-NORM to o_out[t][h*D + j] (gdn_scan_seq_k); the RMSNorm over all D
 *      columns cannot finish inside a slice, so gdn_out_gate_k finishes it in place:
 *      o_out = o * rsqrt(mean(o^2) + eps) * norm_w[j] * silu(z), z = proj[t][z_off + h*D +
 *      j], silu(z) = z / (1 + expf(-z)), NOT sigmoid.
 *
 * Layouts (f32 unless stated): conv [m][conv_ld] with conv_ld == (2*n_k_heads +
 * n_v_heads)*128 EXACTLY (the conv output has no row padding; any other conv_ld is
 * refused); proj [m][ld] holding z (n_v_heads*128 at z_off), the beta logits (n_v_heads at
 * beta_off) and the alpha logits (n_v_heads at alpha_off); a, dt_bias [n_v_heads]; norm_w
 * [128]; state [n_v_heads][128][128]; o_out [m][n_v_heads*128], written by gdn_scan_seq_k
 * and rewritten in place by gdn_out_gate_k (one buffer).
 *
 * p0 and m are LAUNCH ARGUMENTS (no StepParams field is read or written); the scan math is
 * position independent, so p0 enters only the guard. Poison rule (the device re-check of the
 * launch arguments): on p0 < 0 or (long long)p0 + m > ctx_tokens the kernel loads no state,
 * updates nothing, writes NaN to its o_out columns for every t, and ORs mask 8
 * (INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION) into *status (integer atomic, one
 * thread) when status != NULL; gdn_out_gate_k repeats the check and NaNs its rows. The check
 * is block-uniform and precedes the first barrier.
 *
 * Limits: m 1..32768; head_dim 128 (the compiled case; a geometry value is a run-time
 * argument, and any other value is INFERENCE_CUDA_INVALID_ARGUMENT before any launch, never
 * an assertion inside a kernel); n_k_heads 1..64, n_v_heads
 * 1..128 and a multiple of n_k_heads; ld >= max(z_off + n_v*128, beta_off + n_v, alpha_off
 * + n_v) and <= 2^22, offsets >= 0; ctx_tokens 1..2^30 - 1 with p0 + m <= ctx_tokens (the
 * sum in 64 bits); eps exactly the f32 1e-6, used both for the q/k L2 norms and for the
 * gated RMSNorm (Qwen3.6 and Flash-Next both configure rms_norm_eps = 1e-6; a model with
 * another value needs a second eps field); form only the bits below. Every range 4-byte
 * aligned (no float4 loads here); status NULL or 4-byte aligned. state and o_out must overlap
 * no other range, and the status word, when given, must overlap no range either. Enqueues
 * two kernels on stream (not NULL); never synchronizes; no device word is read.
 */
#ifndef INFERENCE_CUDA_PREFILL_GDN_SCAN_SEQ_H
#define INFERENCE_CUDA_PREFILL_GDN_SCAN_SEQ_H

#include <stddef.h>

#include "prefill/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define INFERENCE_CUDA_GDN_FORM_GROUPED_HEADS 1
#define INFERENCE_CUDA_GDN_FORM_ADD_EPS_L2 2
#define INFERENCE_CUDA_GDN_FORM_SIGMOID_GATE 4
#define INFERENCE_CUDA_GDN_FORM_ALL 7

typedef struct inference_cuda_gdn_scan_seq_args {
    const void *conv;
    const void *proj;
    const void *a;
    const void *dt_bias;
    const void *norm_w;
    void *state;
    void *o_out;
    int *status;     /* optional; mask 8 on a refused launch */
    int m;           /* tokens of the chunk, 1..32768 */
    int p0;          /* the position of token 0, a launch argument */
    int ctx_tokens;  /* the context length the guard checks p0 + m against */
    int conv_ld;     /* == (2*n_k_heads + n_v_heads)*128, exactly */
    int ld;          /* proj row length */
    int z_off;
    int beta_off;
    int alpha_off;
    int n_k_heads;   /* 1..64 */
    int n_v_heads;   /* 1..128, a multiple of n_k_heads */
    int head_dim;    /* 128 */
    float eps;       /* exactly 1e-6f */
    int form;        /* INFERENCE_CUDA_GDN_FORM_* bits; 0 is the Qwen3.6 GGUF form */
} inference_cuda_gdn_scan_seq_args;

int inference_cuda_gdn_scan_seq_launch(const inference_cuda_gdn_scan_seq_args *args,
                                       void *stream, int *out_stage);

/* The same launch with the scan kernel gdn_scan_seq_cols_k, which owns block_cols (8, 16 or 32;
 * any other value is INVALID_ARGUMENT) value columns per block instead of 32 with the value-column
 * arithmetic unchanged: state and o_out bytes equal those of the entry above for every valid
 * call. The gated output RMSNorm is the same kernel. Arguments, limits and overlap rules are
 * those of the entry above. */
int inference_cuda_gdn_scan_seq_cols_launch(const inference_cuda_gdn_scan_seq_args *args,
                                            void *stream, int *out_stage, int block_cols);

/* The same contract, limits, guards and written bytes as the entry above, with the scan
 * kernel gdn_scan_chunk_k: the chunk-parallel WY form of the same recurrence (its scalar
 * reference is kernels/gdn_reference.rs chunk_step), which replaces the per-token state
 * chain with a kC-token forward substitution, a masked output sum and one state update per
 * sub-chunk. The rounding differs from the sequential kernels' (deeper dot trees inside the
 * solve); the state and o_out values are pinned to the same f64 reference by the Rust
 * wrapper's tests with their own derived bounds. The gated output RMSNorm is the same
 * kernel. Selected by the wrapper's FIFI_GDN_SCAN_CHUNK switch, never by this entry's
 * arguments. */
int inference_cuda_gdn_scan_chunk_launch(const inference_cuda_gdn_scan_seq_args *args,
                                          void *stream, int *out_stage);

#ifdef INFERENCE_CUDA_FAULT_QUALIFICATION
/* Qualified fault path (the device re-check): validates every rule of the entry EXCEPT the
 * p0/m bound, then overrides p0 so the bound is violated, so the KERNELS' own device guard
 * (no state load, NaN rows, mask 8) is the code under test. The validating entry above can
 * never reach it with bad bounds; this entry changes no rule of that entry. */
int inference_cuda_gdn_scan_seq_fault_bad_bounds_launch(
    const inference_cuda_gdn_scan_seq_args *args, void *stream, int *out_stage);
#endif

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_gdn_scan_seq_args) == 120 &&
                  offsetof(inference_cuda_gdn_scan_seq_args, conv) == 0 &&
                  offsetof(inference_cuda_gdn_scan_seq_args, state) == 40 &&
                  offsetof(inference_cuda_gdn_scan_seq_args, status) == 56 &&
                  offsetof(inference_cuda_gdn_scan_seq_args, m) == 64 &&
                  offsetof(inference_cuda_gdn_scan_seq_args, p0) == 68 &&
                  offsetof(inference_cuda_gdn_scan_seq_args, ctx_tokens) == 72 &&
                  offsetof(inference_cuda_gdn_scan_seq_args, conv_ld) == 76 &&
                  offsetof(inference_cuda_gdn_scan_seq_args, ld) == 80 &&
                  offsetof(inference_cuda_gdn_scan_seq_args, z_off) == 84 &&
                  offsetof(inference_cuda_gdn_scan_seq_args, n_k_heads) == 96 &&
                  offsetof(inference_cuda_gdn_scan_seq_args, head_dim) == 104 &&
                  offsetof(inference_cuda_gdn_scan_seq_args, eps) == 108 &&
                  offsetof(inference_cuda_gdn_scan_seq_args, form) == 112,
              "inference_cuda_gdn_scan_seq_args layout");
#else
_Static_assert(sizeof(inference_cuda_gdn_scan_seq_args) == 120 &&
                   offsetof(inference_cuda_gdn_scan_seq_args, conv) == 0 &&
                   offsetof(inference_cuda_gdn_scan_seq_args, state) == 40 &&
                   offsetof(inference_cuda_gdn_scan_seq_args, status) == 56 &&
                   offsetof(inference_cuda_gdn_scan_seq_args, m) == 64 &&
                   offsetof(inference_cuda_gdn_scan_seq_args, p0) == 68 &&
                   offsetof(inference_cuda_gdn_scan_seq_args, ctx_tokens) == 72 &&
                   offsetof(inference_cuda_gdn_scan_seq_args, conv_ld) == 76 &&
                   offsetof(inference_cuda_gdn_scan_seq_args, ld) == 80 &&
                   offsetof(inference_cuda_gdn_scan_seq_args, z_off) == 84 &&
                   offsetof(inference_cuda_gdn_scan_seq_args, n_k_heads) == 96 &&
                   offsetof(inference_cuda_gdn_scan_seq_args, head_dim) == 104 &&
                   offsetof(inference_cuda_gdn_scan_seq_args, eps) == 108 &&
                   offsetof(inference_cuda_gdn_scan_seq_args, form) == 112,
               "inference_cuda_gdn_scan_seq_args layout");
#endif

#endif
