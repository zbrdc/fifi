/*
 * CUDA projector adapter: device memory, weight repacking and the tiled
 * matmul behind the C ABI in abi.h. The matmul and repack code is vendored
 * from xavier-qwen (engine/vendor/xavier-qwen, see provenance.json).
 *
 * Every exported function has C linkage, takes and returns C types only, and
 * wraps its body in try/catch, like the smoke adapter.
 */
#include <cuda_runtime.h>

#include <stdint.h>
#include <string.h>

#include <fcntl.h>

#include "abi.h"
#include "abi_resident.h"

#ifdef INFERENCE_CUDA_FAULT_QUALIFICATION
#include <atomic>

/* Qualification builds count the cudaMalloc and cudaFree calls that this file
 * makes, directly or through device.h, so a test can prove that a linear call
 * allocates nothing. The macros must come before device.h. The DeviceBuffer
 * rename keeps this file's template apart from the one smoke.cu compiles,
 * so the two bodies never share a symbol. */
namespace {
std::atomic<unsigned long long> malloc_calls(0);
std::atomic<unsigned long long> free_calls(0);

cudaError_t counted_malloc(void **ptr, size_t bytes) {
    malloc_calls.fetch_add(1);
    return cudaMalloc(ptr, bytes);
}

cudaError_t counted_free(void *ptr) {
    free_calls.fetch_add(1);
    return cudaFree(ptr);
}
}  // namespace

#define cudaMalloc counted_malloc
#define cudaFree counted_free
#define DeviceBuffer CountedDeviceBuffer
#endif

#include "device.h"
#include "kernels.h"
#include "repack.h"
#include "resident/matmul_q5_0.h"

namespace {

/* Longest row and largest batch the vendored kernels accept. */
const int kMaxDim = kern::kMaxK;

bool tiled_type(int ggml_type) {
    return ggml_type == INFERENCE_CUDA_TYPE_Q4_K || ggml_type == INFERENCE_CUDA_TYPE_Q6_K ||
           ggml_type == INFERENCE_CUDA_TYPE_Q8_0;
}

bool tiled_shape(int n_rows, int k) {
    return n_rows > 0 && n_rows % 8 == 0 && k > 0 && k % 256 == 0 && k <= kMaxDim;
}

/* ggml type code of a call; Q5_0 has its own kernel and is not a vendored GgmlType. */
struct LinearCall {
    int ggml_type;
    const uint4 *weight;
    int n_rows;
    int k;
    /* Values per row of the staged x and the packed xf: k rounded up to the 256 grid for Q5_0
     * (K = 896 is not a multiple of 256), whose tail is zero; k itself for the vendored types. */
    int k_x;
    const float *x;
    int m;
    float *y;
};

bool is_q5_0(int ggml_type) { return ggml_type == INFERENCE_CUDA_TYPE_Q5_0; }

int padded_k(int ggml_type, int k) { return is_q5_0(ggml_type) ? (k + 255) / 256 * 256 : k; }

/* The type and shape rule of inference_cuda_linear_ws. */
bool linear_shape_ok(int ggml_type, int n_rows, int k, int m) {
    if (is_q5_0(ggml_type)) {
        return inference_cuda_q5_0_shape_ok(m, n_rows, k, 0) == INFERENCE_CUDA_OK;
    }
    return tiled_type(ggml_type) && tiled_shape(n_rows, k) && m >= 1 && m <= kMaxDim;
}

struct Needs {
    size_t x;
    size_t xf;
    size_t y;
};

/* Bytes of the x, xf and y regions for m tokens. Fragment order (m <= kMaxTokens)
 * has a fixed size xf; row-major needs m * k halves. */
Needs linear_needs(int m, int k, int n_rows) {
    Needs needs;
    needs.x = (size_t)m * (size_t)k * sizeof(float);
    needs.xf = m <= kern::kMaxTokens ? kern::xf_vecs() * sizeof(uint4)
                                     : (size_t)m * (size_t)k * sizeof(__half);
    needs.y = (size_t)m * (size_t)n_rows * sizeof(float);
    return needs;
}

bool aligned(const void *p, size_t alignment) { return (uintptr_t)p % alignment == 0; }

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return (uintptr_t)a + a_bytes <= (uintptr_t)b || (uintptr_t)b + b_bytes <= (uintptr_t)a;
}

/* The memory safety guard: every region exists, holds the call, is aligned for the
 * kernels, and no two regions overlap. Nothing here touches the device. */
bool workspace_usable(const inference_cuda_workspace &ws, const Needs &need) {
    if (ws.x == NULL || ws.xf == NULL || ws.y == NULL) return false;
    if (ws.x_bytes < need.x || ws.xf_bytes < need.xf || ws.y_bytes < need.y) return false;
    if (!aligned(ws.x, sizeof(float)) || !aligned(ws.xf, sizeof(uint4)) ||
        !aligned(ws.y, sizeof(float)))
        return false;
    if (!apart(ws.x, need.x, ws.xf, need.xf)) return false;
    if (!apart(ws.x, need.x, ws.y, need.y)) return false;
    if (!apart(ws.xf, need.xf, ws.y, need.y)) return false;
    return true;
}

/* Copies the host x (m rows of k floats) into the device staging of m rows of k_x floats. The
 * Q5_0 tail of every row (k..k_x-1) is zeroed first: the staged row is what the pack reads. */
cudaError_t stage_x(const LinearCall &call, const inference_cuda_workspace &ws, const Needs &need) {
    if (call.k_x == call.k) {
        return cudaMemcpy(ws.x, call.x, need.x, cudaMemcpyHostToDevice);
    }
    cudaError_t err = cudaMemset(ws.x, 0, need.x);
    if (err != cudaSuccess) return err;
    return cudaMemcpy2D(ws.x, (size_t)call.k_x * sizeof(float), call.x,
                        (size_t)call.k * sizeof(float), (size_t)call.k * sizeof(float),
                        (size_t)call.m, cudaMemcpyHostToDevice);
}

cudaError_t matmul_of(const LinearCall &call, const inference_cuda_workspace &ws) {
    if (is_q5_0(call.ggml_type)) {
        return q5_0::matmul(call.m, call.weight, (const uint4 *)ws.xf, (float *)ws.y, call.n_rows,
                            call.n_rows, call.k, NULL);
    }
    return kern::matmul((GgmlType)call.ggml_type, call.m, call.weight, (const uint4 *)ws.xf,
                        (float *)ws.y, call.n_rows, call.n_rows, call.k);
}

void run_linear(const LinearCall &call, const inference_cuda_workspace &ws, const Needs &need,
                Failure &failure) {
    failure.record(stage_x(call, ws, need), INFERENCE_CUDA_STAGE_COPY);
    // Padding tokens of the fragment-order buffer read its zeroed tail.
    if (failure.ok())
        failure.record(cudaMemset(ws.xf, 0, need.xf), INFERENCE_CUDA_STAGE_COPY);
    if (failure.ok())
        failure.record(kern::pack_f32((const float *)ws.x, call.m, call.k_x, (uint4 *)ws.xf),
                       INFERENCE_CUDA_STAGE_LAUNCH);
    if (failure.ok()) failure.record(matmul_of(call, ws), INFERENCE_CUDA_STAGE_LAUNCH);
    failure.record(cudaDeviceSynchronize(), INFERENCE_CUDA_STAGE_SYNCHRONIZE);
    if (failure.ok())
        failure.record(cudaMemcpy(call.y, ws.y, need.y, cudaMemcpyDeviceToHost),
                       INFERENCE_CUDA_STAGE_COPY);
}

#ifdef INFERENCE_CUDA_FAULT_QUALIFICATION
__global__ void write_through_null(float *target) { target[threadIdx.x] = 1.0f; }
#endif

}  // namespace

extern "C" int inference_cuda_drop_page_cache(int fd, long long offset, long long len,
                                              int *out_errno) {
    try {
        if (out_errno == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_errno = 0;
        if (fd < 0 || offset < 0 || len < 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        // posix_fadvise returns the error number instead of setting errno.
        const int err = posix_fadvise(fd, static_cast<off_t>(offset), static_cast<off_t>(len),
                                      POSIX_FADV_DONTNEED);
        if (err != 0) {
            *out_errno = err;
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        return INFERENCE_CUDA_OK;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_memory_info(size_t *out_free, size_t *out_total) {
    try {
        if (out_free == NULL || out_total == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        cudaError_t err = cudaMemGetInfo(out_free, out_total);
        (void)cudaGetLastError();
        return (int)err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_alloc(size_t bytes, void **out_device, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (out_device == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_device = NULL;
        if (bytes == 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        Failure failure;
        DeviceBuffer<unsigned char> buffer;
        failure.record(buffer.allocate(bytes), INFERENCE_CUDA_STAGE_ALLOCATE);
        if (failure.ok()) {
            *out_device = buffer.release();
        }
        (void)cudaGetLastError();
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_upload_at(void *device, size_t device_bytes, size_t offset,
                                        const void *host, size_t bytes, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (device == NULL || host == NULL || bytes == 0 || offset > device_bytes ||
            bytes > device_bytes - offset) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        Failure failure;
        failure.record(cudaMemcpy((unsigned char *)device + offset, host, bytes,
                                  cudaMemcpyHostToDevice),
                       INFERENCE_CUDA_STAGE_COPY);
        failure.record(cudaDeviceSynchronize(), INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        (void)cudaGetLastError();
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
extern "C" int inference_cuda_upload_pinned(void *device, size_t device_bytes,
                                            void *pinned, size_t pinned_bytes,
                                            const void *host, size_t bytes,
                                            int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (device == NULL || pinned == NULL || host == NULL || bytes == 0 ||
            bytes > pinned_bytes || bytes > device_bytes) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        Failure failure;
        // Host copy-in; the DMA below reads only these staged bytes.
        memcpy(pinned, host, bytes);
        failure.record(cudaMemcpyAsync(device, pinned, bytes,
                                       cudaMemcpyHostToDevice, NULL),
                       INFERENCE_CUDA_STAGE_COPY);
        // Stream-scoped completion, not a device-wide wait: only the legacy
        // default stream touches this slot, and every kernel of this ABI
        // runs on it too.
        if (failure.ok())
            failure.record(cudaStreamSynchronize(NULL),
                           INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        (void)cudaGetLastError();
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_upload(const void *host, size_t bytes, void **out_device,
                                     int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (out_device == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_device = NULL;
        if (host == NULL || bytes == 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        Failure failure;
        DeviceBuffer<unsigned char> buffer;
        failure.record(buffer.allocate(bytes), INFERENCE_CUDA_STAGE_ALLOCATE);
        if (!failure.ok()) {
            (void)cudaGetLastError();
            report_stage(out_stage, failure.stage);
            return (int)failure.err;
        }
        failure.record(cudaMemcpy(buffer.ptr, host, bytes, cudaMemcpyHostToDevice),
                       INFERENCE_CUDA_STAGE_COPY);
        const cudaError_t sync_status = cudaDeviceSynchronize();
        failure.record(sync_status, INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        if (sync_status != cudaSuccess) {
            // Unknown completion: retain the allocation for terminal teardown.
            (void)buffer.release();
        } else if (failure.ok()) {
            *out_device = buffer.release();
        } else {
            failure.record(buffer.finish(), INFERENCE_CUDA_STAGE_ALLOCATE);
        }
        (void)cudaGetLastError();
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_download(const void *device, void *host, size_t bytes,
                                       int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (device == NULL || host == NULL || bytes == 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        Failure failure;
        failure.record(cudaDeviceSynchronize(), INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        if (failure.ok())
            failure.record(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost),
                           INFERENCE_CUDA_STAGE_COPY);
        (void)cudaGetLastError();
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_device_synchronize(int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        (void)cudaGetLastError();
        Failure failure;
        failure.record(cudaDeviceSynchronize(), INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        (void)cudaGetLastError();
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_free(void *device, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (device == NULL) {
            return INFERENCE_CUDA_OK;
        }
        (void)cudaGetLastError();
        Failure failure;
        // Failed completion must retain the allocation for process teardown.
        failure.record(cudaDeviceSynchronize(), INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        if (failure.ok())
            failure.record(cudaFree(device), INFERENCE_CUDA_STAGE_ALLOCATE);
        (void)cudaGetLastError();
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_repack(int ggml_type, const unsigned char *src, int n_rows, int k,
                                     void *dst, size_t dst_bytes) {
    try {
        if (src == NULL || dst == NULL || !tiled_type(ggml_type) || !tiled_shape(n_rows, k)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        // repack writes uint4 values, so dst must be 16-byte aligned.
        if ((uintptr_t)dst % sizeof(uint4) != 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        GgmlType type = (GgmlType)ggml_type;
        if (dst_bytes != tiled_vecs(type, n_rows, k) * sizeof(uint4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (!repack(type, src, n_rows, k, (uint4 *)dst)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        return INFERENCE_CUDA_OK;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_linear_workspace_bytes(int m, int k, int n_rows, size_t *out_x,
                                                     size_t *out_xf, size_t *out_y) {
    try {
        if (out_x == NULL || out_xf == NULL || out_y == NULL || m < 1 || m > kMaxDim ||
            !tiled_shape(n_rows, k)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const Needs needs = linear_needs(m, k, n_rows);
        *out_x = needs.x;
        *out_xf = needs.xf;
        *out_y = needs.y;
        return INFERENCE_CUDA_OK;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_linear_ws(int ggml_type, const void *weight, int n_rows, int k,
                                        const float *x, int m, float *y,
                                        const inference_cuda_workspace *workspace,
                                        int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (weight == NULL || x == NULL || y == NULL || workspace == NULL ||
            !linear_shape_ok(ggml_type, n_rows, k, m)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        // The matmul reads uint4 tiles, so an offset view must stay 16-byte aligned.
        if ((uintptr_t)weight % sizeof(uint4) != 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int k_x = padded_k(ggml_type, k);
        const Needs need = linear_needs(m, k_x, n_rows);
        if (!workspace_usable(*workspace, need)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        Failure failure;
        LinearCall call = {ggml_type, (const uint4 *)weight, n_rows, k, k_x, x, m, y};
        run_linear(call, *workspace, need, failure);
        (void)cudaGetLastError();
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_pack_launch(const float *x_dev, int m, int k, void *xf_dev,
                                          void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (x_dev == NULL || xf_dev == NULL || stream == NULL || m < 1 || m > kMaxDim || k < 1 ||
            k > kMaxDim || k % 256 != 0 || !aligned(xf_dev, sizeof(uint4)) ||
            !aligned(x_dev, sizeof(float))) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        // The memset below writes the whole xf, so an x that overlapped it would be clobbered.
        const Needs need = linear_needs(m, k, 0);
        if (!apart(x_dev, need.x, xf_dev, need.xf)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        cudaStream_t launch_stream = (cudaStream_t)stream;
        // Fragment order is a fixed-size buffer whose padding positions the matmul reads.
        if (m <= kern::kMaxTokens) {
            failure.record(cudaMemsetAsync(xf_dev, 0, need.xf, launch_stream),
                           INFERENCE_CUDA_STAGE_LAUNCH);
        }
        if (failure.ok()) {
            failure.record(kern::pack_f32(x_dev, m, k, (uint4 *)xf_dev, launch_stream),
                           INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_matmul_launch(int ggml_type, int m, const void *weight,
                                            const void *xf_dev, float *y_dev, int ldy, int n_rows,
                                            int k, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (weight == NULL || xf_dev == NULL || y_dev == NULL || stream == NULL ||
            !tiled_type(ggml_type) || !tiled_shape(n_rows, k) || m < 1 || m > kMaxDim ||
            ldy < n_rows || !aligned(weight, sizeof(uint4)) || !aligned(xf_dev, sizeof(uint4)) ||
            !aligned(y_dev, sizeof(float))) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        // y is written (the split-K memset and the kernel) while the weight and xf are read.
        const Needs need = linear_needs(m, k, n_rows);
        const size_t weight_bytes = tiled_vecs((GgmlType)ggml_type, n_rows, k) * sizeof(uint4);
        if (!apart(y_dev, need.y, xf_dev, need.xf) || !apart(y_dev, need.y, weight, weight_bytes) ||
            !apart(xf_dev, need.xf, weight, weight_bytes)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        failure.record(kern::matmul((GgmlType)ggml_type, m, (const uint4 *)weight,
                                    (const uint4 *)xf_dev, y_dev, ldy, n_rows, k,
                                    (cudaStream_t)stream),
                       INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

#ifdef INFERENCE_CUDA_FAULT_QUALIFICATION
extern "C" int inference_cuda_allocation_calls(unsigned long long *out_mallocs,
                                               unsigned long long *out_frees) {
    try {
        if (out_mallocs == NULL || out_frees == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_mallocs = malloc_calls.load();
        *out_frees = free_calls.load();
        return INFERENCE_CUDA_OK;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_fault_illegal_address(int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        (void)cudaGetLastError();
        Failure failure;
        write_through_null<<<1, 32>>>(NULL);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        failure.record(cudaDeviceSynchronize(), INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_fault_vendor_launch(int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        Failure failure;
        // Null buffers are never dereferenced: after a sticky fault every
        // launch fails before a kernel runs. m = 17 takes the gemm path and
        // its shared-memory opt-in, m = 1 the mma884 path.
        failure.record(kern::pack_f32(NULL, 1, 256, NULL), INFERENCE_CUDA_STAGE_LAUNCH);
        failure.record(kern::matmul(GgmlType::Q4_K, 17, NULL, NULL, NULL, 8, 8, 256),
                       INFERENCE_CUDA_STAGE_LAUNCH);
        failure.record(kern::matmul(GgmlType::Q4_K, 1, NULL, NULL, NULL, 8, 8, 256),
                       INFERENCE_CUDA_STAGE_LAUNCH);
        failure.record(kern::matmul(GgmlType::Q8_0, 17, NULL, NULL, NULL, 8, 8, 256),
                       INFERENCE_CUDA_STAGE_LAUNCH);
        failure.record(kern::matmul(GgmlType::Q8_0, 1, NULL, NULL, NULL, 8, 8, 256),
                       INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
#endif
