/*
 * Resident-decode entries (CUDA-RD). Launch-only: none of these allocates,
 * copies synchronously, or synchronizes unless its name or comment says so. A
 * fault in queued work surfaces at the next synchronize. Handles (stream,
 * event, pinned memory) are opaque and owned by Rust; each is destroyed once
 * with its destroy function. Lifecycle entries (create, destroy, alloc, free)
 * clear the thread's pending non-sticky CUDA error first, like abi.h; launch
 * entries do not, so the status they return is their own.
 * Per-kernel launch entries live in their own headers, native/resident/<n>.h, one per
 * kernel; this header stays closed (see kernels/cuda/resident/mod.rs).
 */
#ifndef INFERENCE_CUDA_ABI_RESIDENT_H
#define INFERENCE_CUDA_ABI_RESIDENT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Creates a non-blocking stream. */
int inference_cuda_stream_create(void **out_stream, int *out_stage);
/* Synchronizes, then destroys the stream. If the synchronize fails nothing is
 * destroyed (decision 0022) and the status is returned. */
int inference_cuda_stream_destroy(void *stream, int *out_stage);
int inference_cuda_stream_synchronize(void *stream, int *out_stage);

/* Events with timing disabled. */
int inference_cuda_event_create(void **out_event, int *out_stage);
int inference_cuda_event_destroy(void *event, int *out_stage);
int inference_cuda_event_record(void *event, void *stream, int *out_stage);
/* Makes later work on stream wait for event (fork and join). */
int inference_cuda_stream_wait_event(void *stream, void *event, int *out_stage);
/* Blocks the host until the event completes. */
int inference_cuda_event_synchronize(void *event, int *out_stage);

/* Page-locked host memory (cudaHostAlloc, default flags). bytes must not be 0. */
int inference_cuda_pinned_alloc(size_t bytes, void **out_host, int *out_stage);
/* Waits for the device, then frees; if the wait fails nothing is freed. */
int inference_cuda_pinned_free(void *host, int *out_stage);
/* Asynchronous copy from device memory into pinned host memory on stream. With
 * pageable host memory (the ranged reads of a device buffer) the call returns only
 * after the copy completed, which still runs behind the work queued on stream. */
int inference_cuda_copy_d2h_async(void *host_pinned, const void *device, size_t bytes,
                                  void *stream, int *out_stage);
/* Asynchronous copy of bytes from host memory into device memory on stream. From
 * pageable host memory the driver has staged the bytes when the call returns, so
 * the host buffer may be reused at once; the copy itself runs behind the work
 * queued on stream. bytes must not be 0. */
int inference_cuda_copy_h2d_async(void *device, const void *host, size_t bytes, void *stream,
                                  int *out_stage);

/* xf = fp16 pack of x [m][k] (kern::pack_f32), on stream. stream must not be
 * NULL: the entry never falls back to the default stream. x_dev and xf_dev are
 * device pointers; xf_dev is 16-byte aligned and has the size
 * inference_cuda_linear_workspace_bytes gives for m and k; x_dev is 4-byte
 * aligned; k is a multiple of 256, m and k lie in [1, 32768]. The two buffers
 * must not overlap in the bytes the call uses. For m <= 16 the fragment-order
 * buffer is zeroed first (one cudaMemsetAsync of the whole buffer), because the
 * kernels read its padding positions and its last 16 bytes; a row-major xf
 * (m > 16) is fully written. Enqueues up to two operations and returns the
 * launch status of the first that fails. A rejected argument returns
 * INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call. */
int inference_cuda_pack_launch(const float *x_dev, int m, int k, void *xf_dev, void *stream,
                               int *out_stage);
/* y[t * ldy + r] = (x W^T)[t][r] (kern::matmul) for the tiled weight at the
 * device pointer weight (16-byte aligned; a run of whole 8-row groups is itself
 * a tiled weight), reading the xf that inference_cuda_pack_launch wrote for the
 * same m and k. ldy >= n_rows. stream must not be NULL. y_dev (m * n_rows floats
 * at ldy = n_rows, 4-byte aligned) must not overlap the weight tiles or xf_dev,
 * which is 16-byte aligned and xf-sized as above; xf_dev and the weight tiles
 * must not overlap either. Enqueues the split-K memset (when the shape splits)
 * and the kernel on stream. A rejected argument returns
 * INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call. */
int inference_cuda_matmul_launch(int ggml_type, int m, const void *weight, const void *xf_dev,
                                 float *y_dev, int ldy, int n_rows, int k, void *stream,
                                 int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_ABI_RESIDENT_H */
