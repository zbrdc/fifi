#include <cuda_runtime.h>

#include "abi.h"
#include "abi_device.h"

extern "C" int inference_cuda_device_integrated(int *out_integrated) {
    try {
        if (out_integrated == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_integrated = 0;
        (void)cudaGetLastError();
        int device = 0;
        int value = 0;
        cudaError_t err = cudaGetDevice(&device);
        if (err == cudaSuccess) {
            err = cudaDeviceGetAttribute(&value, cudaDevAttrIntegrated, device);
        }
        (void)cudaGetLastError();
        if (err != cudaSuccess) {
            return (int)err;
        }
        *out_integrated = value != 0 ? 1 : 0;
        return INFERENCE_CUDA_OK;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
