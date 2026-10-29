/* The DeltaNet causal conv step, per-kernel header (native/resident/gdn_conv.h). Included by
 * gdn_conv.cu and read by the Rust FFI mirror; nothing else. Wraps the vendored kern::conv_silu
 * (engine/vendor/xavier-qwen/src/ops.cu) and adds no kernel.
 *
 * out[t][c] = silu(w[c][0]*s0 + w[c][1]*s1 + w[c][2]*s2 + w[c][3]*x) with x = proj[t][c] and
 * (s0, s1, s2) the three previous inputs of channel c, oldest first; the window then shifts. The
 * state is planar f32 [3][channels] (row 0 the oldest input) and is read and written in place.
 * ggml keeps the same ring as [channel][tap] (ne0 = 3 fastest), the transpose of this layout: a state
 * import or export (layout tag qwen36-gguf-xq-1 keeps the device order) must transpose it.
 * proj is f32 [m][ld] (only the first `channels` columns of a row are read), w is the GGUF
 * ssm_conv1d tensor as stored, f32 [channels][4] (tap 0 multiplies the oldest input), 16-byte
 * aligned. out is f32 [m][channels]. Limits: m in [1, 32768], channels in [1, 65536],
 * ld in [channels, 1048576]. state and out must overlap no other range (proj and w are read only).
 * Enqueues one kernel on stream (not NULL); returns INFERENCE_CUDA_INVALID_ARGUMENT before any
 * launch for a bad argument. The kernel reads no position, length or slot word. */
#ifndef INFERENCE_CUDA_RESIDENT_GDN_CONV_H
#define INFERENCE_CUDA_RESIDENT_GDN_CONV_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_gdn_conv_args {
    const void *proj;
    const void *w;
    void *state;
    void *out;
    int m;
    int ld;
    int channels;
} inference_cuda_gdn_conv_args;

int inference_cuda_gdn_conv_launch(const inference_cuda_gdn_conv_args *args, void *stream,
                                   int *out_stage);

#ifdef __cplusplus
}
#endif

#endif
