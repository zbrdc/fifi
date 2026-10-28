/*
 * Device facts of the vendored MMQ entries (CPF-V4 V4-0, decision 0049 items 3 and 5). The V-3b
 * launches read a fifi_device_facts (vendored common.cuh): device ordinal, architecture, SM count
 * and opt-in shared memory per block. This translation unit is the one producer for the MMQ glue,
 * queried the device_info.cu way: no allocation, no synchronization, no stream work.
 *
 * The architecture is the BUILT one whose cubin runs on the device (resident/mmvq_arch.h), so an
 * sm_89 GPU runs the sm_86 cubins and a device no built cubin runs on is refused with the runtime's
 * own cudaErrorNoKernelImageForDevice. The Rust side reads the same facts through the C entry:
 * the fixup range of a launch is sized from nsm (CPF-V4 G1), never from a literal.
 */
#ifndef INFERENCE_CUDA_PREFILL_MMQ_FACTS_H
#define INFERENCE_CUDA_PREFILL_MMQ_FACTS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The facts as plain C, field for field the vendored fifi_device_facts plus one reserved word. */
typedef struct inference_cuda_mmq_facts {
    int device;   /* CUDA device ordinal */
    int cc;       /* built architecture, 100 * major + 10 * minor (860 for sm_86) */
    int nsm;      /* streaming multiprocessors */
    int reserved; /* zero */
    size_t smpbo; /* opt-in shared memory per block, in bytes */
} inference_cuda_mmq_facts;

/* Fills out for the current device. Returns INFERENCE_CUDA_OK, INFERENCE_CUDA_INVALID_ARGUMENT
 * for a NULL out, the CUDA error code of a failed query, or INFERENCE_CUDA_NATIVE_EXCEPTION. On
 * any failure out is zeroed. */
int inference_cuda_mmq_device_facts(inference_cuda_mmq_facts *out);

#ifdef __cplusplus
}

#include <cuda_runtime.h>

struct fifi_device_facts;

/* The same query for the MMQ glue translation units, which pass the result to the vendored
 * launches unchanged. */
cudaError_t inference_cuda_mmq_query_facts(fifi_device_facts *out);
#endif

#endif /* INFERENCE_CUDA_PREFILL_MMQ_FACTS_H */
