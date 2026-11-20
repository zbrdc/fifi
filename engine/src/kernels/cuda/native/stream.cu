/*
 * CUDA resident-decode lifecycle: streams, events and page-locked host memory
 * behind the C ABI in abi_resident.h.
 *
 * Every exported function has C linkage, takes and returns C types only, and
 * wraps its body in try/catch, like the projector adapter. Lifecycle entries
 * peek, report and consume a pending non-sticky error at entry (the
 * sticky-prior rule stated in abi.h), then drain the slot after their own
 * recorded status; the launch entries (event record, stream wait, the
 * two host copies), stream_synchronize and event_synchronize do not touch
 * the slot, so they return their own status. Compute entries follow the
 * same peek-report-consume rule.
 */
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "abi_resident.h"
#include "device.h"

namespace {

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
        const cudaError_t sticky = cudaPeekAtLastError();
        if (sticky != cudaSuccess && sticky_prior(sticky, out_stage))
            return static_cast<int>(sticky);
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
        const cudaError_t sticky = cudaPeekAtLastError();
        if (sticky != cudaSuccess && sticky_prior(sticky, out_stage))
            return static_cast<int>(sticky);
        Failure failure;
        failure.record(cudaStreamSynchronize((cudaStream_t)stream),
                       INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        // A stream with unknown completion is not destroyed.
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
        const cudaError_t sticky = cudaPeekAtLastError();
        if (sticky != cudaSuccess && sticky_prior(sticky, out_stage))
            return static_cast<int>(sticky);
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
        const cudaError_t sticky = cudaPeekAtLastError();
        if (sticky != cudaSuccess && sticky_prior(sticky, out_stage))
            return static_cast<int>(sticky);
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

/* The copy fence's completion question. cudaEventQuery answers cudaSuccess once the
 * event's recorded work is done and cudaErrorNotReady while the stream is still behind
 * it; both are ordinary answers, everything else is an error. An event that was never
 * recorded answers cudaSuccess, which is why the Rust side refuses to build an unrecorded
 * fence in the first place. */
extern "C" int inference_cuda_event_query(void *event, int *out_done, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (event == NULL || out_done == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_done = 0;
        cudaError_t state = cudaEventQuery((cudaEvent_t)event);
        if (state == cudaErrorNotReady) {
            return INFERENCE_CUDA_OK;
        }
        Failure failure;
        failure.record(state, INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.err == INFERENCE_CUDA_OK) {
            *out_done = 1;
        }
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

/* The graph-timer trio: a timing-enabled event, an external-record that a stream capture
 * turns into an event-record node of the captured graph, and the host-side elapsed query. The
 * external-record flag and the node behavior are DeepInfer's graph-timer idea; the elapsed
 * entry follows the bench trio of matvec_q.cu. */
extern "C" int inference_cuda_timing_event_create(void **out_event, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (out_event == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_event = NULL;
        const cudaError_t sticky = cudaPeekAtLastError();
        if (sticky != cudaSuccess && sticky_prior(sticky, out_stage))
            return static_cast<int>(sticky);
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

/* Page-locks host memory the caller owns (cudaHostRegister). Registered ranges DMA
 * without driver staging; the caller unregisters before freeing the memory. */
extern "C" int inference_cuda_host_register(void *host, size_t bytes, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (host == NULL || bytes == 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const cudaError_t sticky = cudaPeekAtLastError();
        if (sticky != cudaSuccess && sticky_prior(sticky, out_stage))
            return static_cast<int>(sticky);
        Failure failure;
        failure.record(cudaHostRegister(host, bytes, cudaHostRegisterDefault),
                       INFERENCE_CUDA_STAGE_ALLOCATE);
        (void)cudaGetLastError();
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

/* Releases a page lock taken by inference_cuda_host_register. */
extern "C" int inference_cuda_host_unregister(void *host, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (host == NULL) {
            return INFERENCE_CUDA_OK;
        }
        const cudaError_t sticky = cudaPeekAtLastError();
        if (sticky != cudaSuccess && sticky_prior(sticky, out_stage))
            return static_cast<int>(sticky);
        Failure failure;
        failure.record(cudaHostUnregister(host), INFERENCE_CUDA_STAGE_ALLOCATE);
        (void)cudaGetLastError();
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
        const cudaError_t sticky = cudaPeekAtLastError();
        if (sticky != cudaSuccess && sticky_prior(sticky, out_stage))
            return static_cast<int>(sticky);
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
        const cudaError_t sticky = cudaPeekAtLastError();
        if (sticky != cudaSuccess && sticky_prior(sticky, out_stage))
            return static_cast<int>(sticky);
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

/* Device-side fill (see abi_resident.h): the state clears ride this instead of
 * pageable host chunks, so no cuStreamSynchronize storm sits between the
 * enqueued fills. */
extern "C" int inference_cuda_memset_async(void *device_dst, size_t bytes, int value,
                                           void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (device_dst == NULL || stream == NULL || bytes == 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        failure.record(cudaMemsetAsync(device_dst, value, bytes, (cudaStream_t)stream),
                       INFERENCE_CUDA_STAGE_COPY);
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
