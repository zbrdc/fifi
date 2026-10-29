/* The gated DeltaNet recurrence step with the gated RMSNorm, per-kernel header
 * (native/resident/gdn_delta.h). Included by gdn_delta.cu and read by the Rust FFI mirror; nothing
 * else. Wraps the vendored kern::gated_delta (engine/vendor/xavier-qwen/src/ops.cu) and adds no
 * kernel.
 *
 * Per v head h (kernel layout; D = 128): key head kh = h % n_k_heads (tiled, the GGUF layout);
 * q, k = the head's 128 values of the conv row, each x * rsqrt(sum(x^2) + eps), q also times
 * 1/sqrt(128); decay = exp(softplus(alpha[h] + dt_bias[h]) * a[h]) with a = ssm_a as stored
 * (= -exp(A_log)); beta = sigmoid(beta_logit[h]); S[i][j] *= decay; sk[j] = sum_i S[i][j] k[i];
 * d[j] = (v[j] - sk[j]) * beta; S[i][j] += k[i] d[j]; o[j] = sum_i S[i][j] q[i]; then
 * out[h][j] = o[j] * rsqrt(mean(o^2) + eps) * norm_w[j] * silu(z[h][j]).
 * State: f32 [n_v_heads][128 key i][128 value j], j fastest, in place. ggml keeps the transpose:
 * [head][128 value j][128 key i], the key index fastest, so a state import or export transposes.
 * conv is f32 [m][conv_ld] laid out q (n_k_heads x 128) | k (n_k_heads x 128) | v (n_v_heads x
 * 128), i.e. the output of the conv step with conv_ld >= (2 n_k_heads + n_v_heads) * 128. proj is
 * f32 [m][ld] holding z (n_v_heads x 128 at z_off), the beta logits (n_v_heads at beta_off) and
 * the alpha logits (n_v_heads at alpha_off). a, dt_bias are f32 [n_v_heads], norm_w f32 [128].
 * out is f32 [m][n_v_heads * 128]. eps must be 1e-6. Limits: m in [1, 32768], n_k_heads in
 * [1, 64], n_v_heads in [1, 128] and a multiple of n_k_heads. state and out must overlap no other
 * range. Enqueues one kernel on stream (not NULL); returns INFERENCE_CUDA_INVALID_ARGUMENT before
 * any launch for a bad argument. The kernel reads no position, length or slot word. */
#ifndef INFERENCE_CUDA_RESIDENT_GDN_DELTA_H
#define INFERENCE_CUDA_RESIDENT_GDN_DELTA_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_gdn_delta_args {
    const void *conv;
    const void *proj;
    const void *a;
    const void *dt_bias;
    const void *norm_w;
    void *state;
    void *out;
    int m;
    int conv_ld;
    int ld;
    int z_off;
    int beta_off;
    int alpha_off;
    int n_k_heads;
    int n_v_heads;
    float eps;
} inference_cuda_gdn_delta_args;

int inference_cuda_gdn_delta_launch(const inference_cuda_gdn_delta_args *args, void *stream,
                                    int *out_stage);

#ifdef __cplusplus
}
#endif

#endif
