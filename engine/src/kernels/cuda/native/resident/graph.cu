/*
 * RD-2d: stream capture, instantiate and launch of executable graphs behind the C ABI in graph.h
 * (decision 0047). One graph per step (CUDA-RD rev 2.3); the code uses the ordinary CUDA
 * runtime capture and graph API.
 *
 * Every exported function has C linkage, takes and returns C types only, and wraps its body in
 * try/catch, like the other adapter files. capture_begin and capture_end report their failures at
 * INFERENCE_CUDA_STAGE_CAPTURE; the Rust side keeps the closed list of statuses that do not latch
 * the backend. graph_launch is launch-only. The only synchronize here is the one in
 * graph_exec_destroy (decision 0022: an executable graph is destroyed only after the device is
 * known to be idle).
 */
#include <cuda_runtime.h>

#include <stddef.h>

#include "abi.h"
#include "device.h"
#include "resident/graph.h"

namespace {

/* Reports the stage of the first failure and returns its status. */
int finish(const Failure &failure, int *out_stage) {
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

/* Instantiates graph into *exec and reads its node count; records the first failure. */
void instantiate(cudaGraph_t graph, cudaGraphExec_t *exec, size_t *nodes, Failure &failure) {
    failure.record(cudaGraphGetNodes(graph, NULL, nodes), INFERENCE_CUDA_STAGE_CAPTURE);
    if (failure.ok()) {
        failure.record(cudaGraphInstantiate(exec, graph, 0), INFERENCE_CUDA_STAGE_CAPTURE);
    }
}

#ifdef INFERENCE_CUDA_FAULT_QUALIFICATION
/* armed == 0 writes nothing: that launch only loads the kernel (CUDA 12 loads modules lazily at the
 * first launch, which a capture forbids), so a replay fault can be captured. */
__global__ void write_through_null_graph(float *target, int armed) {
    if (armed != 0) {
        target[threadIdx.x] = 1.0f;
    }
}

int launch_null_write(void *stream, int armed, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (stream == NULL) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    Failure failure;
    write_through_null_graph<<<1, 32, 0, (cudaStream_t)stream>>>(NULL, armed);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    return finish(failure, out_stage);
}
#endif

}  // namespace

extern "C" int inference_cuda_capture_begin(void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (stream == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        // A stale non-sticky error must not be taken for the capture's own.
        (void)cudaGetLastError();
        Failure failure;
        failure.record(cudaStreamBeginCapture((cudaStream_t)stream, cudaStreamCaptureModeThreadLocal),
                       INFERENCE_CUDA_STAGE_CAPTURE);
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_capture_end(void *stream, int instantiate_graph, void **out_exec,
                                          size_t *out_nodes, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (stream == NULL || out_exec == NULL || out_nodes == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_exec = NULL;
        *out_nodes = 0;
        Failure failure;
        cudaGraph_t graph = NULL;
        cudaGraphExec_t exec = NULL;
        size_t nodes = 0;
        failure.record(cudaStreamEndCapture((cudaStream_t)stream, &graph),
                       INFERENCE_CUDA_STAGE_CAPTURE);
        if (graph != NULL) {
            if (failure.ok() && instantiate_graph != 0) {
                instantiate(graph, &exec, &nodes, failure);
            }
            // The executable graph owns what it needs; the captured graph is not kept.
            failure.record(cudaGraphDestroy(graph), INFERENCE_CUDA_STAGE_CAPTURE);
        }
        // "Clean end" means the stream has left capture mode, whatever the end returned.
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        const cudaError_t query = cudaStreamIsCapturing((cudaStream_t)stream, &capture);
        if (query != cudaSuccess || capture != cudaStreamCaptureStatusNone) {
            if (exec != NULL) {
                (void)cudaGraphExecDestroy(exec);
            }
            (void)cudaGetLastError();
            report_stage(out_stage, INFERENCE_CUDA_STAGE_CAPTURE);
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (failure.ok()) {
            *out_exec = (void *)exec;
            *out_nodes = nodes;
        } else if (exec != NULL) {
            (void)cudaGraphExecDestroy(exec);
        }
        // The capture statuses are not sticky: leave nothing for the next call to pick up.
        (void)cudaGetLastError();
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_stream_is_capturing(void *stream, int *out_capturing,
                                                  int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (stream == NULL || out_capturing == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_capturing = 0;
        Failure failure;
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        failure.record(cudaStreamIsCapturing((cudaStream_t)stream, &capture),
                       INFERENCE_CUDA_STAGE_CAPTURE);
        if (failure.ok()) {
            *out_capturing = capture != cudaStreamCaptureStatusNone;
        }
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_graph_launch(void *exec, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (exec == NULL || stream == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        failure.record(cudaGraphLaunch((cudaGraphExec_t)exec, (cudaStream_t)stream),
                       INFERENCE_CUDA_STAGE_LAUNCH);
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_graph_exec_destroy(void *exec, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (exec == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        Failure failure;
        failure.record(cudaDeviceSynchronize(), INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        // A replay with unknown completion is not destroyed (decision 0022).
        if (failure.ok()) {
            failure.record(cudaGraphExecDestroy((cudaGraphExec_t)exec),
                           INFERENCE_CUDA_STAGE_ALLOCATE);
        }
        (void)cudaGetLastError();
        return finish(failure, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

#ifdef INFERENCE_CUDA_FAULT_QUALIFICATION
extern "C" int inference_cuda_fault_null_write_warm(void *stream, int *out_stage) {
    try {
        return launch_null_write(stream, 0, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_fault_null_write_launch(void *stream, int *out_stage) {
    try {
        return launch_null_write(stream, 1, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
#endif
