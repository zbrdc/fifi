/*
 * The device facts of the vendored MMQ entries (see mmq_facts.h). The first translation unit of
 * the MMQ glue build group (the vendor group's flags, warnings as errors). The query follows
 * device_info.cu and the MMVQ wrappers' pattern in this tree.
 */
#include <cuda_runtime.h>

#include "abi.h"

// The vendored headers keep static host helpers their own objects reference; in this glue TU they
// are unreferenced, which -Wall reports although nothing here is wrong.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "common.cuh"
#pragma GCC diagnostic pop

#include "prefill/mmq_facts.h"
#include "resident/mmvq_arch.h"

// The live query. Called once per process by the cached wrapper below, and on demand by the
// introspection entry. The four attributes cannot change while the process holds one device.
static cudaError_t mmq_facts_live(fifi_device_facts *out) {
    int device = 0, major = 0, minor = 0, nsm = 0, smpbo = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err == cudaSuccess) {
        err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    }
    if (err == cudaSuccess) {
        err = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    }
    if (err == cudaSuccess) {
        err = cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, device);
    }
    if (err == cudaSuccess) {
        err = cudaDeviceGetAttribute(&smpbo, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    }
    if (err != cudaSuccess) {
        return err;
    }
    const int built = inference_cuda_mmvq_built_arch(100 * major + 10 * minor);
    if (built == 0) {
        return cudaErrorNoKernelImageForDevice;
    }
    *out = fifi_device_facts{device, built, nsm, (size_t)smpbo};
    return cudaSuccess;
}

// Cached: every MMQ entry used to run this query, which is cudaGetDevice plus four
// cudaDeviceGetAttribute calls. A 128-row chunk makes about 470 MMQ entries, so the cache
// removes roughly 2,350 driver calls per chunk from the enqueue path.
cudaError_t inference_cuda_mmq_query_facts(fifi_device_facts *out) {
    static const fifi_device_facts cached = [] {
        fifi_device_facts facts{};
        const cudaError_t err = mmq_facts_live(&facts);
        return err == cudaSuccess ? facts : fifi_device_facts{};
    }();
    if (cached.cc == 0) {
        // The first query failed. Repeat it so the caller sees the device error, not a zero fact.
        return mmq_facts_live(out);
    }
    *out = cached;
    return cudaSuccess;
}

extern "C" int inference_cuda_mmq_device_facts(inference_cuda_mmq_facts *out) {
    try {
        if (out == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out = inference_cuda_mmq_facts{};
        (void)cudaGetLastError();
        fifi_device_facts facts{};
        const cudaError_t err = mmq_facts_live(&facts);
        (void)cudaGetLastError();
        if (err != cudaSuccess) {
            return (int)err;
        }
        *out = inference_cuda_mmq_facts{facts.device, facts.cc, facts.nsm, 0, facts.smpbo};
        return INFERENCE_CUDA_OK;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
