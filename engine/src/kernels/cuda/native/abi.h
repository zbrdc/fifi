/*
 * C ABI between the Rust CUDA module and the native CUDA adapter.
 *
 * Every function returns an int status. INFERENCE_CUDA_OK means success.
 * A positive status is a cudaError_t value from the CUDA runtime.
 * Pointers refer to host memory, except the opaque device handles that
 * inference_cuda_upload and inference_cuda_alloc return. Rust owns each
 * handle and releases it exactly once with inference_cuda_free. Two more
 * exceptions: the weight pointer of inference_cuda_linear_ws, which may also be
 * a 16-byte-aligned address inside an allocation, and the three device regions
 * of its inference_cuda_workspace, which the caller owns. The adapter keeps no
 * pointer after a call returns.
 *
 * Each function clears the calling thread's pending CUDA runtime error
 * before it returns, so a later call reports only its own failure. Sticky
 * device errors (for example cudaErrorIllegalAddress) cannot be cleared;
 * the Rust caller treats them as terminal for the process.
 */
#ifndef INFERENCE_CUDA_ABI_H
#define INFERENCE_CUDA_ABI_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INFERENCE_CUDA_OK 0
#define INFERENCE_CUDA_INVALID_ARGUMENT (-1)
/* A C++ exception was caught inside the adapter. It never crosses this ABI. */
#define INFERENCE_CUDA_NATIVE_EXCEPTION (-2)

/* Writes the number of CUDA devices to *out_count.
 * A host with no GPU (cudaErrorNoDevice) writes 0 and returns
 * INFERENCE_CUDA_OK. A driver that is missing or too old for the runtime
 * returns its cudaError_t, such as cudaErrorInsufficientDriver, so a
 * misconfigured GPU host fails instead of looking GPU-free. Callers must not
 * run kernels when the count is 0. */
int inference_cuda_device_count(int *out_count);

/* Stage of the first failure in a device call. The Rust caller turns the
 * stage into the operation name of its error. Freeing device memory counts
 * as the allocate stage. */
#define INFERENCE_CUDA_STAGE_NONE 0
#define INFERENCE_CUDA_STAGE_ALLOCATE 1
#define INFERENCE_CUDA_STAGE_COPY 2
#define INFERENCE_CUDA_STAGE_LAUNCH 3
#define INFERENCE_CUDA_STAGE_SYNCHRONIZE 4

/* Computes out[i] = lhs[i] + rhs[i] for i in [0, len) on the first device.
 * The call blocks until the device has finished and the result is copied
 * back. The caller owns all buffers, each with at least len floats.
 * When out_stage is not NULL, it receives the stage of the first failure,
 * or INFERENCE_CUDA_STAGE_NONE on success and on adapter-level failures
 * (invalid argument, native exception). */
int inference_cuda_vector_add_f32(const float *lhs, const float *rhs,
                                  float *out, size_t len, int *out_stage);

/* Writes the free and total memory of the first device, in bytes. */
int inference_cuda_memory_info(size_t *out_free, size_t *out_total);

/* Allocates bytes of device memory, copies them from host, and writes the
 * new handle to *out_device. On failure *out_device is NULL and nothing
 * stays allocated. bytes must not be 0. */
int inference_cuda_upload(const void *host, size_t bytes, void **out_device,
                          int *out_stage);

/* Allocates bytes of uninitialized device memory and writes the handle to
 * *out_device (NULL on failure). bytes must not be 0. Fill the handle with
 * inference_cuda_upload_at and release it with inference_cuda_free. */
int inference_cuda_alloc(size_t bytes, void **out_device, int *out_stage);

/* Copies bytes from host to device + offset, after which the device has
 * finished. The range [offset, offset + bytes) must lie inside the
 * device_bytes of the handle and bytes must not be 0. A failure leaves the
 * handle valid; the caller still frees it. */
int inference_cuda_upload_at(void *device, size_t device_bytes, size_t offset,
                             const void *host, size_t bytes, int *out_stage);

/* Copies bytes from the device handle to host, after the device finished. */
int inference_cuda_download(const void *device, void *host, size_t bytes,
                            int *out_stage);

/* Waits for the device, then frees the handle. A NULL handle is a no-op.
 * The handle is invalid afterwards, even when the call fails. */
int inference_cuda_free(void *device, int *out_stage);

/* Host only, no CUDA call: posix_fadvise(fd, offset, len,
 * POSIX_FADV_DONTNEED). Returns INFERENCE_CUDA_INVALID_ARGUMENT for a
 * negative argument or a failed call; *out_errno holds the error number of
 * a failed call and 0 otherwise. Decision 0040 item 5 measures whether
 * dropping the page cache of uploaded ranges frees unified memory. */
int inference_cuda_drop_page_cache(int fd, long long offset, long long len,
                                   int *out_errno);

/* GGML type codes the tiled weight functions accept. Q5_0 is accepted only by
 * inference_cuda_linear_ws (MEL-C1b); its tiles are built on the host, not by repack. */
#define INFERENCE_CUDA_TYPE_Q5_0 6
#define INFERENCE_CUDA_TYPE_Q8_0 8
#define INFERENCE_CUDA_TYPE_Q4_K 12
#define INFERENCE_CUDA_TYPE_Q6_K 14

/* Host only, no CUDA call: rearranges n_rows rows of k values in GGUF
 * blocks into the tiled layout of the vendored matmul. dst_bytes must equal
 * the tiled size (8-row x 256-k tiles of 1152 bytes for Q4_K, 1792 for
 * Q6_K, 2176 for Q8_0); n_rows must be a multiple of 8 and k a multiple of
 * 256. */
int inference_cuda_repack(int ggml_type, const unsigned char *src, int n_rows,
                          int k, void *dst, size_t dst_bytes);

/* Caller-owned device scratch of one inference_cuda_linear_ws call. Each
 * region is a device address and its size in bytes. x holds m * k floats,
 * xf the fp16 buffer of the vendored matmul and y m * n_rows floats; the
 * sizes come from inference_cuda_linear_workspace_bytes. The regions must
 * not overlap, x and y must be 4-byte aligned and xf 16-byte aligned. */
typedef struct inference_cuda_workspace {
    void *x;
    size_t x_bytes;
    void *xf;
    size_t xf_bytes;
    void *y;
    size_t y_bytes;
} inference_cuda_workspace;

/* Host only, no CUDA call: writes the bytes that the x, xf and y regions of
 * a linear call with m tokens, n_rows rows and k values per row need. xf is
 * a fixed fragment-order buffer up to 16 tokens and m * k halves above.
 * m and k lie in [1, 32768]; n_rows is a positive multiple of 8 and k a
 * multiple of 256. */
int inference_cuda_linear_workspace_bytes(int m, int k, int n_rows, size_t *out_x,
                                          size_t *out_xf, size_t *out_y);

/* y[t * n_rows + r] = sum_i x[t * k + i] * W[r][i] for t < m, with W the
 * tiled weight of ggml_type that starts at the device pointer weight. The
 * pointer is a handle or a 16-byte-aligned address inside one: the tiles are
 * group-major, so a run of whole 8-row groups is itself a tiled weight. The
 * caller keeps the n_rows x k tiles inside the allocation. x is converted to
 * fp16 and products are summed in fp32. x holds m * k floats and y
 * m * n_rows floats, both host memory; workspace holds the device scratch and
 * is not freed or retained. Nothing is allocated or freed. The call copies x
 * in, packs, multiplies, waits for the device, and copies y back.
 * Q5_0 (type 6, MEL-C1b): k is a multiple of 128, m is at most 16, and the
 * workspace is sized by inference_cuda_linear_workspace_bytes for k rounded
 * up to a multiple of 256 (x holds m rows of that width); the call zeroes the
 * tail of each staged row. */
int inference_cuda_linear_ws(int ggml_type, const void *weight, int n_rows, int k,
                             const float *x, int m, float *y,
                             const inference_cuda_workspace *workspace, int *out_stage);

#ifdef INFERENCE_CUDA_FAULT_QUALIFICATION
/* Qualification only, host only: how many cudaMalloc and cudaFree calls the
 * adapter (projector.cu) has made so far, by every entry that allocates. The
 * counters only grow, so a test compares two readings around a call. */
int inference_cuda_allocation_calls(unsigned long long *out_mallocs,
                                    unsigned long long *out_frees);

/* Fault qualification only: a kernel writes through an invalid device
 * address, so the context holds a sticky cudaErrorIllegalAddress. */
int inference_cuda_fault_illegal_address(int *out_stage);

/* Fault qualification only: calls the vendored pack and matmul launch
 * functions directly, without allocating, and returns their status. After
 * a sticky fault it shows that the vendored code returns the error instead
 * of ending the process. */
int inference_cuda_fault_vendor_launch(int *out_stage);
#endif

/* Returns a static, NUL-terminated description of a status code. */
const char *inference_cuda_status_message(int status);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_ABI_H */
