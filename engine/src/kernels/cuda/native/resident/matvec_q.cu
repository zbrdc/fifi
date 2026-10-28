/*
 * Entries of the vendored MMVQ matvec (MATVEC-PAR S1 c3b) on an explicit stream. Glue only: it
 * fills a fifi_mmvq_args (the V-MV1 host entry's argument set) for the dense one-sample view
 * and validates before any launch; every exported function has C linkage, takes and returns C
 * types only, and wraps its body in try/catch, like the projector adapter.
 */
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"

// The vendored headers keep static host helpers their own objects reference; in this glue TU they
// are unreferenced, which -Wall reports although nothing here is wrong.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "mmvq.cuh"
#pragma GCC diagnostic pop

#include "resident/matvec_q.h"
#include "resident/mmvq_arch.h"

namespace {

/* k is padded up to this before the launch (MATRIX_ROW_PADDING, common.cuh). */
const int64_t kMatrixRowPadding = 512;

/* The kernel loads the weight rows of one block as a group (rows_per_block: 1, 2, or up to 4 for
 * the small-k launch) and never receives nrows, so a matrix whose row count is not a multiple of
 * the group is read up to 3 rows past its end; the wrappers prove only rows x row_bytes. Limitation
 * until a guarded kernel exists (note for S3): rows must be a multiple of this, as every Qwen3.6
 * shape is. KUBDAN-MSG-64 item 4. */
const int kRowsGroup = 4;

bool aligned(const void *p, size_t alignment) { return (uintptr_t)p % alignment == 0; }

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return (uintptr_t)a + a_bytes <= (uintptr_t)b || (uintptr_t)b + b_bytes <= (uintptr_t)a;
}

bool mmvq_type(int type) {
    return type == INFERENCE_CUDA_TYPE_Q4_K || type == INFERENCE_CUDA_TYPE_Q5_0 ||
           type == INFERENCE_CUDA_TYPE_Q6_K || type == INFERENCE_CUDA_TYPE_Q8_0;
}

/* The fifi_device_facts the V-MV1 entry reads (V-3b pattern), queried the device_info.cu way:
 * no allocation, no sync, no stream work. */
cudaError_t query_device_facts(fifi_device_facts *out) {
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
    /* An sm_89 GPU runs the sm_86 cubin: the facts carry the built architecture, and a
     * device no built cubin runs on is refused with the runtime's own code. */
    const int built = inference_cuda_mmvq_built_arch(100 * major + 10 * minor);
    if (built == 0) {
        return cudaErrorNoKernelImageForDevice;
    }
    *out = fifi_device_facts{device, built, nsm, (size_t)smpbo};
    return cudaSuccess;
}

/* The dense view of one matvec: one channel, one sample, columns = activation rows. */
cudaError_t launch_dense(const fifi_mmvq_args & a, cudaStream_t stream) {
    return fifi_mmvq_launch(a, stream);
}

}  // namespace

extern "C" int inference_cuda_matvec_q_launch(const inference_cuda_matvec_q_args *args,
                                              void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int k = args->k;
        const int rows = args->rows;
        const int ncols = args->ncols;
        if (args->w == NULL || args->x_q8 == NULL || args->y == NULL || !mmvq_type(args->type) ||
            k <= 0 || rows < 1 || rows % kRowsGroup != 0 || ncols < 1 || ncols > 8 ||
            !aligned(args->w, 4) || !aligned(args->x_q8, 4) || !aligned(args->y, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        /* The kernel writes y while it reads w and x_q8; the guard needs the byte counts and
         * must not wrap. */
        const ggml_type type = (ggml_type)args->type;
        const int qk = type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K ? 256 : 32;
        if (k % qk != 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const size_t block_bytes = type == GGML_TYPE_Q4_K   ? 144
                                   : type == GGML_TYPE_Q6_K ? 210
                                   : type == GGML_TYPE_Q8_0 ? 34
                                                            : 22;
        const int64_t stride_row_x = k / qk;
        const int64_t blocks_y =
            (k + kMatrixRowPadding - 1) / kMatrixRowPadding * kMatrixRowPadding / 32;
        if (stride_row_x > INT64_MAX / (int64_t)rows ||
            stride_row_x * rows > INT64_MAX / (int64_t)block_bytes ||
            blocks_y > INT64_MAX / 36 || blocks_y * 36 > INT64_MAX / ncols) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const size_t w_bytes = (size_t)(stride_row_x * rows * (int64_t)block_bytes);
        const size_t x_bytes = (size_t)(blocks_y * 36 * ncols);
        const size_t y_bytes = (size_t)rows * (size_t)ncols * sizeof(float);
        if (!apart(args->y, y_bytes, args->x_q8, x_bytes) ||
            !apart(args->y, y_bytes, args->w, w_bytes)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        fifi_mmvq_args a{};
        a.type = type;
        a.vx = args->w;
        a.vy_q8_1 = args->x_q8;
        a.ids = NULL;
        a.dst = args->y;
        a.ncols_x = k;
        a.nrows_x = rows;
        a.ncols_dst = ncols;
        a.nchannels_x = 1;
        a.nchannels_y = 1;
        a.nchannels_dst = 1;
        a.stride_row_x = stride_row_x;
        a.stride_col_y = blocks_y;
        a.stride_col_dst = rows; /* the kernel's row bound and y's column stride */
        a.stride_channel_x = 1;
        a.stride_channel_y = 1;
        a.stride_channel_dst = 1;
        fifi_device_facts dev{};
        const cudaError_t facts = query_device_facts(&dev);
        if (facts != cudaSuccess) {
            report_stage(out_stage, INFERENCE_CUDA_STAGE_LAUNCH);
            return (int)facts;
        }
        a.dev = dev;
        Failure failure;
        failure.record(launch_dense(a, (cudaStream_t)stream), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

/* Test support of the seam-1 microbench (MATVEC-PAR S1 test 11): a raw timing event recorded on
 * a stream, the elapsed milliseconds between two of them, and the destroy. The context tree's
 * DeviceEvent carries no elapsed query and stays closed for this packet, so the bench times its
 * queued launches through these three entries. */
extern "C" int inference_cuda_event_record_create(void *stream, void **out_event, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (stream == NULL || out_event == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        cudaEvent_t event = NULL;
        cudaError_t err = cudaEventCreate(&event);
        if (err == cudaSuccess) {
            err = cudaEventRecord(event, (cudaStream_t)stream);
        }
        if (err != cudaSuccess) {
            (void)cudaEventDestroy(event);
            report_stage(out_stage, INFERENCE_CUDA_STAGE_LAUNCH);
            return (int)err;
        }
        *out_event = event;
        return INFERENCE_CUDA_OK;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_event_elapsed_ms(void *start, void *end, float *out_ms,
                                               int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (start == NULL || end == NULL || out_ms == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const cudaError_t err = cudaEventElapsedTime(out_ms, (cudaEvent_t)start, (cudaEvent_t)end);
        report_stage(out_stage, err == cudaSuccess ? INFERENCE_CUDA_STAGE_NONE
                                                   : INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        return (int)err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_event_destroy_test(void *event, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (event == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        report_stage(out_stage, INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        return (int)cudaEventDestroy((cudaEvent_t)event);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

/* Test support: free and total device memory, so the microbench can skip the 417 MB lm_head
 * shape when the device is nearly full (MATVEC-PAR S1 test 11). */
extern "C" int inference_cuda_mem_info(size_t *out_free, size_t *out_total, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (out_free == NULL || out_total == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        size_t free_bytes = 0, total_bytes = 0;
        const cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err == cudaSuccess) {
            *out_free = free_bytes;
            *out_total = total_bytes;
        }
        report_stage(out_stage, err == cudaSuccess ? INFERENCE_CUDA_STAGE_NONE
                                                   : INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        return (int)err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
