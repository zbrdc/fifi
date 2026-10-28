/* Qualification only: five ids, five log probabilities, and a status word. */
#ifndef INFERENCE_CUDA_TOP_LOGITS_H
#define INFERENCE_CUDA_TOP_LOGITS_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct inference_cuda_top_logits_args {
    void *out; /* 44 bytes: u32 ids[5], f32 logprobs[5], u32 status */
    const void *x; /* f32 [n] */
    int n;
} inference_cuda_top_logits_args;
#ifdef __cplusplus
static_assert(sizeof(inference_cuda_top_logits_args) == 24 &&
              offsetof(inference_cuda_top_logits_args, n) == 16, "layout");
#endif
int inference_cuda_top_logits_launch(const inference_cuda_top_logits_args *args,
                                    void *stream, int *out_stage);
#ifdef __cplusplus
}
#endif
#endif
