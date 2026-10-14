/*
 * CUDA smoke adapter: device count and f32 vector add behind the C ABI in abi.h.
 *
 * Every exported function has C linkage, takes and returns C types only, and wraps
 * its body in try/catch. An exception that unwinds into Rust is undefined behavior,
 * so each one becomes INFERENCE_CUDA_NATIVE_EXCEPTION.
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>

#include "abi.h"
#include "device.h"

/* KV8-B B4' (lane C amendment 2, lazy module loading): CUDA loads a kernel's module lazily
 * by default (12.2+), and a module first needed mid-prompt can fail to load after a cache
 * that fills the VRAM has been admitted. The adapter asks for EAGER module loading before
 * CUDA ever initializes: constructors of the executable and of its static archives run
 * before main, before the Rust runtime or any server or test-harness thread exists, and
 * CUDA reads CUDA_MODULE_LOADING at its first initialization, which happens only after
 * main starts - so the environment is written while the process is provably
 * single-threaded, and no Rust `set_var` is used. The overwrite argument is 0, so an
 * operator's explicit value (LAZY, EAGER) always wins. This TU defines
 * inference_cuda_device_count, which the Rust side calls at every context build, so the
 * linker can never drop this object from the archive. */
__attribute__((constructor)) static void set_eager_module_loading(void) {
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);
}

#define VECTOR_ADD_THREADS 256u
/* The grid is capped so the launch stays valid on every supported arch.
 * The kernel loops with a grid stride to cover longer inputs. */
#define VECTOR_ADD_MAX_BLOCKS 65535u

/* Host buffers of one vector add call. The caller validated len. */
struct VectorAddHost {
    const float *lhs;
    const float *rhs;
    float *out;
    size_t len;
};

/* Device buffers of one vector add call, each holding len floats. */
struct VectorAddDevice {
    DeviceBuffer<float> lhs;
    DeviceBuffer<float> rhs;
    DeviceBuffer<float> out;
};

__global__ static void vector_add_f32_kernel(const float *lhs,
                                             const float *rhs, float *out,
                                             size_t len) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < len;
         i += stride) {
        out[i] = lhs[i] + rhs[i];
    }
}

/* Queues the kernel and returns the launch status. It does not wait for the
 * kernel to finish. */
static cudaError_t launch_vector_add_f32(const VectorAddDevice &dev,
                                         size_t len) {
    /* Ceiling division without the addition, which could overflow. */
    size_t blocks = len / VECTOR_ADD_THREADS + (len % VECTOR_ADD_THREADS != 0);
    if (blocks > VECTOR_ADD_MAX_BLOCKS) {
        blocks = VECTOR_ADD_MAX_BLOCKS;
    }
    vector_add_f32_kernel<<<(unsigned int)blocks, VECTOR_ADD_THREADS>>>(
        dev.lhs.ptr, dev.rhs.ptr, dev.out.ptr, len);
    return cudaGetLastError();
}

/* Allocates, copies the inputs in, runs the kernel, waits for it, and copies
 * the result out. The device buffers stay owned by dev. Each step runs only
 * while no earlier step failed. */
static void vector_add_f32_on_device(const VectorAddHost &host,
                                     VectorAddDevice &dev, Failure &failure) {
    const size_t bytes = host.len * sizeof(float);
    failure.record(dev.lhs.allocate(bytes), INFERENCE_CUDA_STAGE_ALLOCATE);
    if (failure.ok()) {
        failure.record(dev.rhs.allocate(bytes), INFERENCE_CUDA_STAGE_ALLOCATE);
    }
    if (failure.ok()) {
        failure.record(dev.out.allocate(bytes), INFERENCE_CUDA_STAGE_ALLOCATE);
    }
    if (failure.ok()) {
        failure.record(
            cudaMemcpy(dev.lhs.ptr, host.lhs, bytes, cudaMemcpyHostToDevice),
            INFERENCE_CUDA_STAGE_COPY);
    }
    if (failure.ok()) {
        failure.record(
            cudaMemcpy(dev.rhs.ptr, host.rhs, bytes, cudaMemcpyHostToDevice),
            INFERENCE_CUDA_STAGE_COPY);
    }
    if (failure.ok()) {
        failure.record(launch_vector_add_f32(dev, host.len),
                       INFERENCE_CUDA_STAGE_LAUNCH);
    }
    if (failure.ok()) {
        /* Wait for the kernel so that execution errors surface here and not
         * in the result copy. */
        failure.record(cudaDeviceSynchronize(), INFERENCE_CUDA_STAGE_SYNCHRONIZE);
    }
    if (failure.ok()) {
        failure.record(
            cudaMemcpy(host.out, dev.out.ptr, bytes, cudaMemcpyDeviceToHost),
            INFERENCE_CUDA_STAGE_COPY);
    }
}

extern "C" int inference_cuda_device_count(int *out_count) {
    try {
        if (out_count == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        (void)cudaGetLastError();
        if (err == cudaErrorNoDevice) {
            *out_count = 0;
            return INFERENCE_CUDA_OK;
        }
        if (err != cudaSuccess) {
            return (int)err;
        }
        *out_count = count;
        return INFERENCE_CUDA_OK;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_vector_add_f32(const float *lhs,
                                             const float *rhs, float *out,
                                             size_t len, int *out_stage) {
    try {
        if (out_stage != NULL) {
            *out_stage = INFERENCE_CUDA_STAGE_NONE;
        }
        if (lhs == NULL || rhs == NULL || out == NULL || len == 0 ||
            len > SIZE_MAX / sizeof(float)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        /* Clear a stale error so this call reports only its own failure. */
        (void)cudaGetLastError();
        const VectorAddHost host = {lhs, rhs, out, len};
        VectorAddDevice dev;
        Failure failure;
        vector_add_f32_on_device(host, dev, failure);
        /* On an early failure, queued copies or the kernel may still use the
         * buffers. Wait for them before freeing. On success this returns at
         * once, because the result copy already waited. */
        failure.record(cudaDeviceSynchronize(), INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        failure.record(dev.lhs.finish(), INFERENCE_CUDA_STAGE_ALLOCATE);
        failure.record(dev.rhs.finish(), INFERENCE_CUDA_STAGE_ALLOCATE);
        failure.record(dev.out.finish(), INFERENCE_CUDA_STAGE_ALLOCATE);
        (void)cudaGetLastError();
        if (out_stage != NULL) {
            *out_stage = failure.stage;
        }
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" const char *inference_cuda_status_message(int status) {
    try {
        if (status == INFERENCE_CUDA_OK) {
            return "success";
        }
        if (status == INFERENCE_CUDA_INVALID_ARGUMENT) {
            return "invalid argument";
        }
        if (status == INFERENCE_CUDA_NATIVE_EXCEPTION) {
            return "native exception";
        }
        if (status < 0) {
            return "unknown status";
        }
        return cudaGetErrorString((cudaError_t)status);
    } catch (...) {
        return "native exception";
    }
}

/* Writes the driver's module loading mode to *out_mode: the CUmoduleLoadingMode of cuda.h
 * (CU_MODULE_EAGER_LOADING 1, CU_MODULE_LAZY_LOADING 2), read through cuModuleGetLoadingMode
 * resolved with cudaGetDriverEntryPoint, so the adapter keeps linking the runtime only. The
 * observation point of the eager-loading pin of KV8-B B4'. */
extern "C" int inference_cuda_module_loading_mode(int *out_mode) {
    try {
        if (out_mode == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        CUresult (*get_mode)(CUmoduleLoadingMode *);
        cudaError_t err = cudaGetDriverEntryPoint("cuModuleGetLoadingMode",
                                                 (void **)&get_mode, cudaEnableDefault);
        if (err != cudaSuccess) {
            return (int)err;
        }
        CUmoduleLoadingMode mode;
        err = (cudaError_t)get_mode(&mode);
        if (err != cudaSuccess) {
            return (int)err;
        }
        *out_mode = (int)mode;
        return INFERENCE_CUDA_OK;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
