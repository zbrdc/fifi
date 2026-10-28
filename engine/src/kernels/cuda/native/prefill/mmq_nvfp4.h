/* Dense NVFP4-only addition: the legacy 64-byte dense ABI stays unchanged.
 * matrix is type40, contiguous dst [M][N], original GGUF blocks [N][K/64].
 * y_scale requires M+127 immutable, finite positive FP32 projection factors:
 * stream-k temporary writes include whole J<=128 tiles past M. The owning Rust
 * plan validates/uploads that capacity. Vendor writeback applies each once.
 * N*(K/64) must fit a signed 32-bit native weight-block index.
 * All dense shape/range/fixup rules apply, including output/scratch disjoint
 * from y_scale. The pure fill entry compares addresses, never dereferences them.
 */
#ifndef INFERENCE_CUDA_PREFILL_MMQ_NVFP4_H
#define INFERENCE_CUDA_PREFILL_MMQ_NVFP4_H
#include "prefill/mmq_dense.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct inference_cuda_mmq_nvfp4_args {
    inference_cuda_mmq_dense_args matrix;
    const float *y_scale;
} inference_cuda_mmq_nvfp4_args;

int inference_cuda_mmq_nvfp4_dense(const inference_cuda_mmq_nvfp4_args *a,
                                  void *stream, int *out_stage);
int inference_cuda_mmq_nvfp4_dense_fill(const inference_cuda_mmq_nvfp4_args *a,
                                       inference_cuda_mmq_args_view *out);
#ifdef __cplusplus
}
static_assert(sizeof(inference_cuda_mmq_nvfp4_args) == 72 &&
              offsetof(inference_cuda_mmq_nvfp4_args, y_scale) == 64,
              "NVFP4 dense args layout");
#else
_Static_assert(sizeof(inference_cuda_mmq_nvfp4_args) == 72, "NVFP4 dense args layout");
#endif
#endif
