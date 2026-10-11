/*
 * Entries of the vendored MMVQ expert matvec (MATVEC-PAR S1 c3c) on an explicit stream. Glue
 * only: it fills the ids view of a fifi_mmvq_args (ncols_dst 1, nchannels_dst n_used, the id
 * range nchannels_x n_expert) and validates before any launch; C linkage, C types only,
 * try/catch around the body, like the projector adapter.
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

#include "resident/matvec_expert.h"
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

/* The fifi_device_facts the V-MV1 entry reads (V-3b pattern), queried the device_info.cu way. */
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

}  // namespace

extern "C" int inference_cuda_matvec_expert_launch(const inference_cuda_matvec_expert_args *args,
                                                   void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int k = args->k;
        const int rows = args->rows;
        const int n_expert = args->n_expert;
        const int n_used = args->n_used;
        const int per_slot = args->x_per_slot;
        if (args->w_stack == NULL || args->ids == NULL || args->x_q8 == NULL || args->y == NULL ||
            !mmvq_type(args->type) || k <= 0 || rows < 1 || rows % kRowsGroup != 0 ||
            n_expert < 1 || n_used < 1 || n_used > 8 || n_used > n_expert ||
            (per_slot != 0 && per_slot != 1) ||
            args->y_slot_stride < rows || !aligned(args->w_stack, 4) ||
            !aligned(args->ids, 4) || !aligned(args->x_q8, 4) || !aligned(args->y, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const ggml_type type = (ggml_type)args->type;
        const int qk = type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K ? 256 : 32;
        const size_t block_bytes = type == GGML_TYPE_Q4_K   ? 144
                                   : type == GGML_TYPE_Q6_K ? 210
                                   : type == GGML_TYPE_Q8_0 ? 34
                                                            : 22;
        if (k % qk != 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int64_t stride_row_x = k / qk;
        const int64_t expert_blocks = stride_row_x * rows;
        const int64_t blocks_y =
            (k + kMatrixRowPadding - 1) / kMatrixRowPadding * kMatrixRowPadding / 32;
        const int64_t x_rows = per_slot ? n_used : 1;
        /* Overlap guards with byte counts that must not wrap. */
        if (expert_blocks > INT64_MAX / (int64_t)n_expert ||
            expert_blocks * n_expert > INT64_MAX / (int64_t)block_bytes ||
            blocks_y > INT64_MAX / 36 || blocks_y * 36 > INT64_MAX / x_rows ||
            (int64_t)args->y_slot_stride > INT64_MAX / n_used) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const size_t w_bytes = (size_t)(expert_blocks * n_expert * (int64_t)block_bytes);
        const size_t x_bytes = (size_t)(blocks_y * 36 * x_rows);
        const size_t y_bytes =
            (size_t)((int64_t)(n_used - 1) * args->y_slot_stride + rows) * sizeof(float);
        if (!apart(args->y, y_bytes, args->x_q8, x_bytes) ||
            !apart(args->y, y_bytes, args->w_stack, w_bytes) ||
            !apart(args->y, y_bytes, args->ids, (size_t)n_used * sizeof(int32_t))) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        fifi_mmvq_args a{};
        a.type = type;
        a.vx = args->w_stack;
        a.vy_q8_1 = args->x_q8;
        a.ids = (const int32_t *)args->ids;
        a.dst = args->y;
        a.ncols_x = k;
        a.nrows_x = rows;
        a.ncols_dst = 1;
        a.nchannels_x = n_expert; /* the A1 id range */
        a.nchannels_y = per_slot ? n_used : 1;
        a.nchannels_dst = n_used;
        a.stride_row_x = stride_row_x;
        a.stride_col_y = blocks_y;
        a.stride_col_dst = rows; /* the kernel's row bound */
        a.stride_channel_x = expert_blocks;
        a.stride_channel_y = per_slot ? blocks_y : 1;
        a.stride_channel_dst = args->y_slot_stride;
        fifi_device_facts dev{};
        const cudaError_t facts = query_device_facts(&dev);
        if (facts != cudaSuccess) {
            report_stage(out_stage, INFERENCE_CUDA_STAGE_LAUNCH);
            return (int)facts;
        }
        a.dev = dev;
        Failure failure;
        failure.record(fifi_mmvq_launch(a, (cudaStream_t)stream), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
