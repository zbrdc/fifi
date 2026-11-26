/*
 * The weight types of the vendored MMQ cases this tree instantiates, one
 * vendor/ggml-cuda-mmq/src/ggml-cuda/template-instances/mmq-instance-*.cu per type in build.rs,
 * as X(ggml type, block struct). The dense, grouped and quantize entries read this one list, and
 * the Rust side's `MmqType` names the same types. NVFP4 has its own dense entry (mmq_nvfp4.h) and
 * is not listed. The names come from the vendored ggml.h shim and ggml-common.h.
 */
#ifndef INFERENCE_CUDA_MMQ_TYPES_H
#define INFERENCE_CUDA_MMQ_TYPES_H

#define INFERENCE_CUDA_MMQ_TYPES(X)      \
    X(GGML_TYPE_Q4_0, block_q4_0)        \
    X(GGML_TYPE_Q5_0, block_q5_0)        \
    X(GGML_TYPE_Q8_0, block_q8_0)        \
    X(GGML_TYPE_Q2_0, block_q2_0)        \
    X(GGML_TYPE_Q2_K, block_q2_K)        \
    X(GGML_TYPE_Q3_K, block_q3_K)        \
    X(GGML_TYPE_Q4_K, block_q4_K)        \
    X(GGML_TYPE_Q5_K, block_q5_K)        \
    X(GGML_TYPE_Q6_K, block_q6_K)        \
    X(GGML_TYPE_IQ1_S, block_iq1_s)      \
    X(GGML_TYPE_IQ2_XXS, block_iq2_xxs)  \
    X(GGML_TYPE_IQ2_XS, block_iq2_xs)    \
    X(GGML_TYPE_IQ2_S, block_iq2_s)      \
    X(GGML_TYPE_IQ3_XXS, block_iq3_xxs)  \
    X(GGML_TYPE_IQ3_S, block_iq3_s)      \
    X(GGML_TYPE_IQ4_NL, block_iq4_nl)    \
    X(GGML_TYPE_IQ4_XS, block_iq4_xs)

#endif
