/*
 * Device-property queries of the CUDA adapter. Host only: nothing here
 * allocates, copies or launches. Like abi.h, a function returns an int status
 * (INFERENCE_CUDA_OK, a negative adapter status, or a positive cudaError_t) and
 * clears the thread's pending non-sticky CUDA error before it returns.
 */
#ifndef INFERENCE_CUDA_ABI_DEVICE_H
#define INFERENCE_CUDA_ABI_DEVICE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Writes 1 to *out_integrated when the calling thread's current device shares
 * physical memory with the host (cudaDevAttrIntegrated: a Jetson), else 0.
 * *out_integrated is 0 on failure. */
int inference_cuda_device_integrated(int *out_integrated);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_ABI_DEVICE_H */
