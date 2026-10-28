/*
 * CUDA resident-decode lifecycle: streams, events and page-locked host memory
 * behind the C ABI in abi_resident.h.
 *
 * Every exported function has C linkage, takes and returns C types only, and
 * wraps its body in try/catch, like the projector adapter. Lifecycle entries
 * clear the thread's pending non-sticky CUDA error before and after their
 * calls; the launch entries (event record, stream wait, the two host copies)
 * and the two synchronizes do not, so they return their own status.
 */
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "abi_resident.h"
#include "device.h"

namespace {

/* Reports the stage of the first failure and returns its status. */
int finish(const Failure &failure, int *out_stage) {
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

}  // namespace

extern "C" int inference_cuda_stream_create(void **out_stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (out_stream == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_stream = NULL;
        (void)cudaGetLastError();
        Failure failure;
        cudaStream_t stream = NULL;
        failure.record(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                       INFERENCE_CUDA_STAGE_ALLOCATE);
        if (failure.ok()) {
            *out_stream = (void *)stream;
        }
        (void)cudaGetLastError();
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_stream_destroy(void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (stream == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        Failure failure;
        failure.record(cudaStreamSynchronize((cudaStream_t)stream),
                       INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        // A stream with unknown completion is not destroyed (decision 0022).
        if (failure.ok()) {
            failure.record(cudaStreamDestroy((cudaStream_t)stream), INFERENCE_CUDA_STAGE_ALLOCATE);
        }
        (void)cudaGetLastError();
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_stream_synchronize(void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (stream == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        failure.record(cudaStreamSynchronize((cudaStream_t)stream),
                       INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_event_create(void **out_event, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (out_event == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_event = NULL;
        (void)cudaGetLastError();
        Failure failure;
        cudaEvent_t event = NULL;
        failure.record(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                       INFERENCE_CUDA_STAGE_ALLOCATE);
        if (failure.ok()) {
            *out_event = (void *)event;
        }
        (void)cudaGetLastError();
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_event_destroy(void *event, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (event == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        Failure failure;
        // No synchronize: an event destroyed while recorded is released when it completes.
        failure.record(cudaEventDestroy((cudaEvent_t)event), INFERENCE_CUDA_STAGE_ALLOCATE);
        (void)cudaGetLastError();
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_event_record(void *event, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (event == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        failure.record(cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream),
                       INFERENCE_CUDA_STAGE_LAUNCH);
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

/* The graph-timer trio (packet B1): a timing-enabled event, an external-record that a stream
 * capture turns into an event-record node of the captured graph, and the host-side elapsed
 * query. The external-record flag and the node behavior are DeepInfer's graph-timer idea
 * (ledger deepinfer-774ce24/B-kernels-moe-attention-quant.md lines 40 and 71); the elapsed
 * entry follows the bench trio of matvec_q.cu. */
extern "C" int inference_cuda_timing_event_create(void **out_event, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (out_event == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_event = NULL;
        (void)cudaGetLastError();
        Failure failure;
        cudaEvent_t event = NULL;
        failure.record(cudaEventCreateWithFlags(&event, 0), INFERENCE_CUDA_STAGE_ALLOCATE);
        if (failure.ok()) {
            *out_event = (void *)event;
        }
        (void)cudaGetLastError();
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_timing_event_record_external(void *event, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (event == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        failure.record(
            cudaEventRecordWithFlags((cudaEvent_t)event, (cudaStream_t)stream,
                                     cudaEventRecordExternal),
            INFERENCE_CUDA_STAGE_LAUNCH);
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_timing_event_elapsed_ms(void *start, void *end, float *out_ms,
                                               int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (start == NULL || end == NULL || out_ms == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        failure.record(cudaEventElapsedTime(out_ms, (cudaEvent_t)start, (cudaEvent_t)end),
                       INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_stream_wait_event(void *stream, void *event, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (event == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        failure.record(cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)event, 0),
                       INFERENCE_CUDA_STAGE_LAUNCH);
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_event_synchronize(void *event, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (event == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        failure.record(cudaEventSynchronize((cudaEvent_t)event), INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_pinned_alloc(size_t bytes, void **out_host, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (out_host == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_host = NULL;
        if (bytes == 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        Failure failure;
        void *host = NULL;
        failure.record(cudaHostAlloc(&host, bytes, cudaHostAllocDefault),
                       INFERENCE_CUDA_STAGE_ALLOCATE);
        if (failure.ok()) {
            *out_host = host;
        }
        (void)cudaGetLastError();
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_pinned_free(void *host, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (host == NULL) {
            return INFERENCE_CUDA_OK;
        }
        (void)cudaGetLastError();
        Failure failure;
        // A copy queued on any stream may still read or write the buffer. If the wait
        // fails the completion is unknown, so the memory is not freed.
        failure.record(cudaDeviceSynchronize(), INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        if (failure.ok()) {
            failure.record(cudaFreeHost(host), INFERENCE_CUDA_STAGE_ALLOCATE);
        }
        (void)cudaGetLastError();
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_copy_d2h_async(void *host_pinned, const void *device, size_t bytes,
                                             void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (host_pinned == NULL || device == NULL || stream == NULL || bytes == 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        failure.record(cudaMemcpyAsync(host_pinned, device, bytes, cudaMemcpyDeviceToHost,
                                       (cudaStream_t)stream),
                       INFERENCE_CUDA_STAGE_COPY);
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_copy_d2d_async(void *device_dst, const void *device_src, size_t bytes,
                                             void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (device_dst == NULL || device_src == NULL || stream == NULL || bytes == 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const uintptr_t dst = (uintptr_t)device_dst;
        const uintptr_t src = (uintptr_t)device_src;
        if ((dst >= src ? dst - src : src - dst) < bytes) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        failure.record(cudaMemcpyAsync(device_dst, device_src, bytes, cudaMemcpyDeviceToDevice,
                                       (cudaStream_t)stream),
                       INFERENCE_CUDA_STAGE_COPY);
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_copy_h2d_async(void *device, const void *host, size_t bytes,
                                             void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (device == NULL || host == NULL || stream == NULL || bytes == 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        failure.record(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice,
                                       (cudaStream_t)stream),
                       INFERENCE_CUDA_STAGE_COPY);
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
