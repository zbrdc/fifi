/*
 * TCAP C1: mapped page-locked host memory and the device attributes the CPU-expert mailbox depends on
 * (semantics: see mailbox.h). Original CUDA C; design from ledger/plans/TCAP-DESIGN.md section 3.1.
 *
 * Like the lifecycle entries of stream.cu: C linkage, C types only, the body in try/catch, the thread's
 * pending non-sticky CUDA error cleared before and after the calls.
 */
#include <cuda_runtime.h>
#include <stddef.h>
#include <string.h>

#include "abi.h"
#include "device.h"
#include "resident/mailbox.h"

namespace {

int finish(const Failure &failure, int *out_stage) {
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

}  // namespace

extern "C" int inference_cuda_mapped_alloc(size_t bytes, void **out_host, void **out_device, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (out_host == NULL || out_device == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_host = NULL;
        *out_device = NULL;
        if (bytes == 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        Failure failure;
        void *host = NULL;
        void *device = NULL;
        const cudaError_t allocated = cudaHostAlloc(&host, bytes, cudaHostAllocMapped);
        if (allocated != cudaSuccess) {
            host = NULL;  // the runtime leaves the pointer unspecified on failure: never free it
        }
        failure.record(allocated, INFERENCE_CUDA_STAGE_ALLOCATE);
        if (failure.ok()) {
            failure.record(cudaHostGetDevicePointer(&device, host, 0), INFERENCE_CUDA_STAGE_ALLOCATE);
        }
        if (failure.ok()) {
            memset(host, 0, bytes);
            *out_host = host;
            *out_device = device;
        } else if (host != NULL) {
            // Nothing was queued on the memory yet, so no device wait is needed before the free.
            (void)cudaFreeHost(host);
        }
        (void)cudaGetLastError();
        return finish(failure, out_stage);
    } catch (...) {
        // Nothing between the allocation and the publication of the pointers can throw, so no host pointer is leaked here.
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_mailbox_caps(int *out_can_map_host, int *out_unified_addressing,
                                           int *out_kernel_exec_timeout, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        int *const outs[3] = {out_can_map_host, out_unified_addressing, out_kernel_exec_timeout};
        const cudaDeviceAttr attrs[3] = {cudaDevAttrCanMapHostMemory, cudaDevAttrUnifiedAddressing,
                                         cudaDevAttrKernelExecTimeout};
        for (int *out : outs) {
            if (out == NULL) {
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            *out = 0;
        }
        (void)cudaGetLastError();
        Failure failure;
        int device = 0;
        failure.record(cudaGetDevice(&device), INFERENCE_CUDA_STAGE_ALLOCATE);
        for (int i = 0; i < 3 && failure.ok(); ++i) {
            int value = 0;
            failure.record(cudaDeviceGetAttribute(&value, attrs[i], device), INFERENCE_CUDA_STAGE_ALLOCATE);
            *outs[i] = value != 0;
        }
        (void)cudaGetLastError();
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
