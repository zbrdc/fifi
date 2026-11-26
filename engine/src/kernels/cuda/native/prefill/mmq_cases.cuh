/*
 * The per-type parts of the dense and grouped MMQ entries, over the one type list of
 * prefill/mmq_types.h: the block shape a weight row is checked against and the vendored
 * `mul_mat_q_case` a launch dispatches to. Include after mmq.cuh.
 */
#ifndef INFERENCE_CUDA_MMQ_CASES_CUH
#define INFERENCE_CUDA_MMQ_CASES_CUH

#include "prefill/mmq_types.h"

/* The values and bytes per weight block of a listed type; {0, 0} for every other int. */
struct MmqBlockShape {
    int values;
    int bytes;
};

static inline MmqBlockShape mmq_block_shape(int type) {
    switch (type) {
#define INFERENCE_CUDA_MMQ_SHAPE(T, B) \
    case T:                            \
        return {(int)ggml_cuda_type_traits<T>::qk, (int)sizeof(B)};
        INFERENCE_CUDA_MMQ_TYPES(INFERENCE_CUDA_MMQ_SHAPE)
#undef INFERENCE_CUDA_MMQ_SHAPE
        default:
            return {0, 0};
    }
}

/* The vendored launch of a listed type; cudaErrorInvalidValue for any other (unreachable
 * behind each entry's validation). */
static inline cudaError_t mmq_launch_case(const mmq_args &args, cudaStream_t stream) {
    switch (args.type_x) {
#define INFERENCE_CUDA_MMQ_LAUNCH(T, B) \
    case T:                             \
        return mul_mat_q_case<T>(args, stream);
        INFERENCE_CUDA_MMQ_TYPES(INFERENCE_CUDA_MMQ_LAUNCH)
#undef INFERENCE_CUDA_MMQ_LAUNCH
        default:
            return cudaErrorInvalidValue;
    }
}

#endif
