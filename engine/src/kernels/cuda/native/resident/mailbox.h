/*
 * Host-mapped page-locked memory for the CPU-expert mailbox (native/resident/mailbox.rs wraps it;
 * the layout is in kernels/cuda/context/mailbox.rs). Two entries behind the C ABI; abi_resident.h
 * stays closed. Include after abi.h in C++.
 *
 *   inference_cuda_mapped_alloc  `bytes` of zeroed page-locked host memory mapped into the device address space
 *                                (cudaHostAllocMapped): the host pointer and the pointer the device uses. Freed with
 *                                inference_cuda_pinned_free, which waits for the device first.
 *   inference_cuda_mailbox_caps  canMapHostMemory, unifiedAddressing and kernelExecTimeout (the display watchdog: 1 =
 *                                a kernel running past the driver's limit is killed) of the current device, 0 or 1.
 */
#ifndef INFERENCE_CUDA_MAILBOX_H
#define INFERENCE_CUDA_MAILBOX_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int inference_cuda_mapped_alloc(size_t bytes, void **out_host, void **out_device, int *out_stage);
int inference_cuda_mailbox_caps(int *out_can_map_host, int *out_unified_addressing, int *out_kernel_exec_timeout,
                                int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_MAILBOX_H */
