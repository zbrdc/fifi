/*
 * Helpers shared by the CUDA adapter files: first-failure tracking and an
 * owning device buffer. C++ only; nothing here crosses the C ABI.
 */
#ifndef INFERENCE_CUDA_DEVICE_H
#define INFERENCE_CUDA_DEVICE_H

#include <cuda_runtime.h>

#include <stddef.h>

#include "abi.h"

/* First failure of a device call and its stage. Later failures are dropped
 * so that cleanup can still run after the first one. The first failure
 * prints to stderr once so a latched terminal state is self-explaining. */
#include <cstdio>

static inline const char *stage_name(int stage) {
    switch (stage) {
    case INFERENCE_CUDA_STAGE_NONE: return "none";
    case INFERENCE_CUDA_STAGE_ALLOCATE: return "allocate";
    case INFERENCE_CUDA_STAGE_COPY: return "copy";
    case INFERENCE_CUDA_STAGE_LAUNCH: return "launch";
    case INFERENCE_CUDA_STAGE_SYNCHRONIZE: return "synchronize";
    default: return "unknown";
    }
}

struct Failure {
    cudaError_t err = cudaSuccess;
    int stage = INFERENCE_CUDA_STAGE_NONE;

    bool ok() const { return err == cudaSuccess; }

    void record(cudaError_t next, int next_stage) {
        if (ok() && next != cudaSuccess) {
            err = next;
            stage = next_stage;
            // First-error capture: name the CUDA error and stage so the
            // terminal latch is self-explaining instead of silent.
            std::fprintf(stderr,
                         "CUDA FIRST ERROR: %s (code %d) at stage %s\n",
                         cudaGetErrorString(next), static_cast<int>(next),
                         stage_name(next_stage));
        }
    }
};

/* Owns one device allocation. The destructor frees it on every exit path,
 * including an exception. finish() frees it early and reports the status;
 * release() hands the allocation to the caller. */
template <typename T> struct DeviceBuffer {
    T *ptr = NULL;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer() { (void)finish(); }

    cudaError_t allocate(size_t bytes) {
        cudaError_t err = cudaMalloc((void **)&ptr, bytes);
        // cudaMalloc leaves ptr unspecified on failure; finish must not free it.
        if (err != cudaSuccess) {
            ptr = NULL;
        }
        return err;
    }

    cudaError_t finish() {
        cudaError_t err = ptr != NULL ? cudaFree(ptr) : cudaSuccess;
        ptr = NULL;
        return err;
    }

    T *release() {
        T *owned = ptr;
        ptr = NULL;
        return owned;
    }
};

/* Writes stage to *out_stage when the caller asked for it. */
static inline void report_stage(int *out_stage, int stage) {
    if (out_stage != NULL) {
        *out_stage = stage;
    }
}

#endif /* INFERENCE_CUDA_DEVICE_H */
