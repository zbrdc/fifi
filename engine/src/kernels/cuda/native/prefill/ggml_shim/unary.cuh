// Fifi shadow header for the vendored llama.cpp MMQ sources (engine/vendor/ggml-cuda-mmq; decision 0049, CPF-V2).
// This is glue, not a patched vendor file: it stands in for ggml/src/ggml-cuda/unary.cuh.
// Text marked "derived" is copied from that file at llama.cpp acecd56032ddc34bada14a2d978f110d9c987095 (MIT, Copyright (c) 2023-2026
// The ggml authors; engine/vendor/ggml-cuda-mmq/LICENSE has the notice).
//
// It carries only the four scalar device helpers that the GLU switch of the vendored mul_mat_vec_q
// kernels names (MATVEC-PAR S1 c1): none of the ggml_tensor host operations of the upstream header
// are vendored, and nothing else here defines them.
#pragma once

#include "common.cuh"

// derived: copied verbatim from unary.cuh at the pin; the upstream host declarations between
// them are not carried.
__device__ __forceinline__ float ggml_cuda_op_silu_single(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float ggml_cuda_op_gelu_single(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__device__ __forceinline__ float ggml_cuda_op_swiglu_oai_single(float x, float g, float alpha = 1.702f, float limit = 7.0f) {
    x = fminf(x, limit);
    g = fmaxf(fminf(g, limit), -limit);

    float out_glu = x / (1.0f + expf(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    return out_glu;
}

__device__ __forceinline__ float ggml_cuda_op_swiglu_clamp_single(float gate, float up, float limit) {
    gate = fminf(gate, limit);
    up = fmaxf(fminf(up, limit), -limit);

    return ggml_cuda_op_silu_single(gate) * up;
}
