// Fifi shadow header for the vendored llama.cpp MMQ sources (engine/vendor/ggml-cuda-mmq; decision 0049, CPF-V2).
// This is glue, not a patched vendor file: it stands in for ggml/include/ggml.h.
// Text marked "derived" is copied from that file at llama.cpp acecd56032ddc34bada14a2d978f110d9c987095 (MIT, Copyright (c) 2023-2026
// The ggml authors; engine/vendor/ggml-cuda-mmq/LICENSE has the notice).
//
// It carries only what the kept sources use: the ggml_type numbers (the GGUF tensor type ids, so a type tag means
// what ggml means), GGML_UNUSED, GGML_UNUSED_VARS and GGML_PAD. It defines no abort or assert macro on purpose:
// the vendored sources return errors (patch V-2), so a use of one is a compile error.
#pragma once

#include <cstddef>
#include <cstdint>

#define GGML_UNUSED(x) (void)(x) // derived
#ifdef __CUDACC__
template<typename... Args>
__host__ __device__ constexpr inline void ggml_unused_vars_impl(Args&&...) noexcept {} // derived
#define GGML_UNUSED_VARS(...) ggml_unused_vars_impl(__VA_ARGS__)                       // derived
#else
#define GGML_UNUSED_VARS(...) do { (void)sizeof((__VA_ARGS__, 0)); } while(0)          // derived
#endif // __CUDACC__

#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1)) // derived; n must be a power of two

// derived: copied verbatim modulo indentation (8 spaces became 4); the numbers are the GGUF tensor type ids
enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // GGML_TYPE_Q4_0_4_8 = 32,
    // GGML_TYPE_Q4_0_8_8 = 33,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    // GGML_TYPE_IQ4_NL_4_4 = 36,
    // GGML_TYPE_IQ4_NL_4_8 = 37,
    // GGML_TYPE_IQ4_NL_8_8 = 38,
    GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
    GGML_TYPE_NVFP4   = 40, // NVFP4 (4 blocks, E4M3 scale)
    GGML_TYPE_Q1_0    = 41,
    GGML_TYPE_Q2_0    = 42,
    GGML_TYPE_COUNT   = 43,
};
