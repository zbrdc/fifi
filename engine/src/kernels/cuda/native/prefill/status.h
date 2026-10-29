/* The prefill status word's shared mask, one definition for every prefill kernel:
 * attn_prefill.h, attn_prep_chunk.h and gdn_scan_seq.h include this file instead of defining
 * their own. INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION is mask 8 (0x8) of the prefill
 * status word — a prefill chunk fault, a bound violation in any of the kernels; the family
 * book reads the word with `status_error` (family/qwen36/cuda/book.rs), which maps mask 8 to
 * Backend "prefill_chunk" with precedence 4 > 8 > 1 > 2. The Rust mirror is
 * `crate::kernels::cuda::native::prefill::PREFILL_STATUS_BOUND_VIOLATION`, also 0x8; the
 * device bound-violation tests pin the C value against it at run time. */
#ifndef INFERENCE_CUDA_PREFILL_STATUS_H
#define INFERENCE_CUDA_PREFILL_STATUS_H

#define INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION 0x8

/* The C++ TUs (the .cu files, nvcc -std=c++17) pin the value at compile time. */
#ifdef __cplusplus
static_assert(INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION == 0x8,
              "INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION is mask 8 (0x8) of the prefill "
              "status word (book.rs status_error maps it to prefill_chunk)");
#endif

#endif /* INFERENCE_CUDA_PREFILL_STATUS_H */
