/*
 * Launch entry of expert_activation.cu: the device-side relu-squared of the
 * chunk expert stream. Launch-only; a fault in queued work surfaces at the
 * next synchronize. CUDA-only header (no C translation unit includes it), so
 * the extern "C" block and layout assert run unconditionally.
 */
#ifndef INFERENCE_CUDA_RESIDENT_EXPERT_ACTIVATION_H
#define INFERENCE_CUDA_RESIDENT_EXPERT_ACTIVATION_H

#include <stddef.h>

extern "C" {

typedef struct inference_cuda_element_args {
    void *out;        /* f32 [values], written */
    const void *x;    /* f32 [values], read */
    long long values; /* 1..2^32 */
} inference_cuda_element_args;

int inference_cuda_relu_squared_launch(const inference_cuda_element_args *args, void *stream,
                                       int *out_stage);

static_assert(sizeof(inference_cuda_element_args) == 24 &&
                  offsetof(inference_cuda_element_args, out) == 0 &&
                  offsetof(inference_cuda_element_args, x) == 8 &&
                  offsetof(inference_cuda_element_args, values) == 16,
              "layout");
}

#endif
