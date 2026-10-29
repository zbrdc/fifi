/* Stream capture, executable graphs and their launch. One header per kernel file;
 * abi_resident.h stays closed.
 *
 * A capture is begun and ended by the same thread, on a non-blocking stream, in thread-local
 * capture mode: every launch, async memset or copy, event record and event wait that the caller
 * queues on the stream between the two calls is recorded into a graph instead of running. The
 * graph is instantiated by inference_cuda_capture_end and the executable graph is replayed with
 * inference_cuda_graph_launch. Graph update (cudaGraphExecUpdate) is not used: what changes
 * between replays is buffer content, never an address.
 *
 * Statuses. The capture statuses 900..906 (cudaErrorStreamCapture*) and an allocation failure of
 * the instantiate are reported with stage INFERENCE_CUDA_STAGE_CAPTURE; the Rust side decides which
 * of them leave the backend usable. Every entry returns INFERENCE_CUDA_INVALID_ARGUMENT before any
 * CUDA call for a NULL argument.
 */
#ifndef INFERENCE_CUDA_RESIDENT_GRAPH_H
#define INFERENCE_CUDA_RESIDENT_GRAPH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Stage of a failure in begin or end of a capture, or in instantiating it. */
#define INFERENCE_CUDA_STAGE_CAPTURE 5

/* Begins a thread-local capture on stream (not NULL). Clears the pending non-sticky error first. */
int inference_cuda_capture_begin(void *stream, int *out_stage);

/* Ends the capture of stream. With instantiate != 0 the captured graph is instantiated:
 * *out_exec gets the executable graph and *out_nodes its node count (event records and waits are
 * edges, not nodes). With instantiate == 0 *out_exec is NULL and *out_nodes is 0. The captured
 * graph is destroyed on every path. After the call the stream is no longer capturing; if the
 * runtime leaves it capturing (or the query fails) the entry destroys what it made and returns
 * INFERENCE_CUDA_INVALID_ARGUMENT, an adapter failure. On a recorded failure *out_exec is NULL.
 * The capture statuses are not sticky, so the last-error slot is cleared at the end. */
int inference_cuda_capture_end(void *stream, int instantiate, void **out_exec, size_t *out_nodes,
                               int *out_stage);

/* Writes 1 to *out_capturing when stream is capturing, else 0. */
int inference_cuda_stream_is_capturing(void *stream, int *out_capturing, int *out_stage);

/* Launch-only: queues exec on stream (not NULL), behind the earlier work of the stream and of the
 * same exec. A fault in the replay surfaces at the next synchronize. No error clearing. */
int inference_cuda_graph_launch(void *exec, void *stream, int *out_stage);

/* Synchronizes the whole device, then destroys exec; when the synchronize fails nothing is
 * destroyed and the status is returned. Clears the last error before and after. */
int inference_cuda_graph_exec_destroy(void *exec, int *out_stage);

#ifdef INFERENCE_CUDA_FAULT_QUALIFICATION
/* Fault qualification only, launch-only: queues a kernel on stream that writes through a null
 * pointer. The fault surfaces at the next synchronize, like a fault in a replayed graph. */
int inference_cuda_fault_null_write_launch(void *stream, int *out_stage);
/* Fault qualification only, launch-only: queues the same kernel with the write disarmed, so that the
 * first (module loading) launch happens outside a capture. Nothing faults. */
int inference_cuda_fault_null_write_warm(void *stream, int *out_stage);
#endif

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_GRAPH_H */
