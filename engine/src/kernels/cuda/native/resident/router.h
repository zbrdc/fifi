/* RD-5: the device router, per-kernel header (native/resident/router.h). Included by router.cu
 * and by the Rust FFI mirror's reader; nothing else. x is the f32 normalized row [m][hidden] (the
 * norm_out of the add+RMSNorm kernel), w the router weight f32 [n_expert][hidden] (expert-major,
 * 16-byte aligned), w_shared the shared-expert gate f32 [hidden], or NULL together with
 * shared_gate for a model without a shared expert. logits is scratch f32 [m][n_expert + 1] (the
 * last column is unused without a shared gate). Outputs: ids i32 [m][top_k] (-1 on a faulty row),
 * weights f32 [m][top_k], shared_gate f32 [m] = sigmoid(dot(x, w_shared)) (NULL: not computed).
 * status is a device i32 word; bit 1 is OR-ed in when any row is faulty (a non-finite logit, or a
 * softmax or selected sum that is not finite and positive, or a non-finite gate dot). Limits: m in
 * [1, 4096], hidden a multiple of 4 up to 65536, n_expert in [1, 256], top_k in
 * [1, min(n_expert, 32)]. Enqueues two kernels on stream (NULL: the default stream); returns
 * INFERENCE_CUDA_INVALID_ARGUMENT before any launch for a bad argument. */
#ifndef INFERENCE_CUDA_RESIDENT_ROUTER_H
#define INFERENCE_CUDA_RESIDENT_ROUTER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_router_args {
    const void *x;
    const void *w;
    const void *w_shared;
    void *logits;
    void *ids;
    void *weights;
    void *shared_gate;
    void *status;
    int m;
    int hidden;
    int n_expert;
    int top_k;
} inference_cuda_router_args;

int inference_cuda_router_launch(const inference_cuda_router_args *args, void *stream,
                                 int *out_stage);

#ifdef __cplusplus
}
#endif

#endif
