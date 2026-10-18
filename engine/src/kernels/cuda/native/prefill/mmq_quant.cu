/*
 * The q8_1 quantize entries of the vendored-MMQ glue (see prefill/mmq_quant.h). CPF-V4 G4,
 * decision 0049 item 11. Design guidance only, no code copied: the validation follows the
 * pattern of the prefill kernels of this tree; the launches are the vendored entries
 * (quantize.cu), called, not transcribed.
 */
#include <cstdint>

#include "abi.h"
#include "device.h"

// The vendored headers keep static host helpers their own objects reference; in this glue TU
// they are unreferenced, which -Wall reports although nothing here is wrong (mmq_facts.cu).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "quantize.cuh"
#pragma GCC diagnostic pop

#include "prefill/mmq_quant.h"

namespace {

// Q4_K reference codes use a signed scale per 256 values, repacked into DS4 groups.
struct Q8kMmqCall {
    const float *x;
    const int32_t *ids;
    block_q8_1_mmq *y;
    int k;
    int row_stride;
    int rows;
    int top_k;
};

__device__ __forceinline__ bool prefer_q8k_max(float a, int ai, float b, int bi) {
    const float aa = fabsf(a), ab = fabsf(b);
    return aa > ab || (aa == ab && ai < bi);
}

template <bool Scatter>
__global__ void q8k_mmq_kernel(Q8kMmqCall a) {
    const int lane = threadIdx.x;
    const int row = blockIdx.x;
    const int source_row = !Scatter && a.ids != nullptr ? a.ids[row] : row;
    const int64_t start = (int64_t)source_row * a.row_stride + (int64_t)blockIdx.y * 256;
    float values[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) values[j] = a.x[start + lane + 32*j];
    float maximum = values[0];
    int first = lane;
#pragma unroll
    for (int j = 1; j < 8; ++j) {
        const int index = lane + 32*j;
        if (prefer_q8k_max(values[j], index, maximum, first)) {
            maximum = values[j];
            first = index;
        }
    }
    // Equal magnitudes retain the first value, including its sign, as Q8_K requires.
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(0xffffffff, maximum, offset);
        const int index = __shfl_down_sync(0xffffffff, first, offset);
        if (lane + offset < 32 && prefer_q8k_max(other, index, maximum, first)) {
            maximum = other;
            first = index;
        }
    }
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    const float inverse = maximum == 0.0f ? 0.0f : __fdiv_rn(-127.0f, maximum);
    const float scale = maximum == 0.0f ? 0.0f : __fdiv_rn(1.0f, inverse);
    const float stored_scale = __half2float(__float2half_rn(scale));
    const int copies = Scatter ? a.top_k : 1;
    const int columns = a.rows * copies;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int q = min(127, __float2int_rn(__fmul_rn(values[j], inverse)));
        int sum = q;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        const half2 ds = make_half2(stored_scale, __fmul_rn(stored_scale, (float)sum));
        for (int slot = 0; slot < copies; ++slot) {
            const int destination = Scatter ? a.ids[(int64_t)row*a.top_k + slot] : row;
            const int64_t block = ((int64_t)blockIdx.y*2 + j/4)*columns + destination;
            a.y[block].qs[32*(j%4) + lane] = (int8_t)q;
            if (lane == 0) a.y[block].ds4[j%4] = ds;
        }
    }
}

template <bool Scatter>
int launch_q8k_mmq(Q8kMmqCall a, void *stream, int *out_stage) {
    Failure failure;
    q8k_mmq_kernel<Scatter><<<dim3(a.rows, a.k/256), 32, 0, (cudaStream_t)stream>>>(a);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}


/* The K grid of the MMQ GEMM (plan section B.3, MSG-85 M2): `MMQ_ITER_K` k values per loop
 * iteration, read with no k clamp, so the quantize refuses to build a buffer the GEMM
 * cannot read safely. The extent bound matches the tree's `MAX_DIM`. */
constexpr int kKGrid = 256;
constexpr int kKMax = 32768;
constexpr long long kStrideMax = 1LL << 30;
/* The chunk cap (G1's NCOLS_LIMIT) and its gathered form (`m * top_k` at top_k 8). */
constexpr int kRowLimit = 2048;
constexpr long long kRowsMax = 16384;
constexpr int kTopKMax = 32;

bool aligned_to(const void *p, uintptr_t a) { return ((uintptr_t)p & (a - 1)) == 0; }

/* True when the two used spans share no byte. */
bool apart(const void *a, long long a_bytes, const void *b, long long b_bytes) {
    const uintptr_t pa = (uintptr_t)a, pb = (uintptr_t)b;
    return pa + (uintptr_t)a_bytes <= pb || pb + (uintptr_t)b_bytes <= pa;
}

/* The k and row rules both entries share (the K-grid first, so its refusal row fires for
 * every type; k % 4 is the vendored FIFI_REQUIRE mirror). */
bool k_ok(int k, int row_stride) {
    return k > 0 && k <= kKMax && k % kKGrid == 0 && k % 4 == 0 && row_stride >= k &&
           row_stride % 4 == 0 && row_stride <= kStrideMax;
}

/* NIT-7 (review): only the four types the glue serves reach the vendored switch — the
 * wrapper's MmqType is exactly these, and no other int is ever cast to ggml_type. Each one
 * has a q8_1 ds layout (mmq.cuh:61-101), so the vendored default refusal stays dead. */
bool served_type(int type) {
    return type == 8 /* Q8_0 */ || type == 6 /* Q5_0 */ || type == 12 /* Q4_K */ ||
           type == 14 /* Q6_K */;
}

/* The wrapper mirrors every rule here, so the native entry cannot return INVALID_ARGUMENT
 * (which would latch the backend); a C caller gets the same refusal before any launch. */
bool valid_quantize(const inference_cuda_mmq_quantize_args *a) {
    if (a == NULL || a->x == NULL || a->y == NULL) return false;
    if (a->ids != NULL && !aligned_to(a->ids, 4)) return false;
    if (!aligned_to(a->x, 16) || !aligned_to(a->y, 16)) return false;
    if (!served_type(a->type) || !k_ok(a->k, a->row_stride)) return false;
    if (a->packing != 0 && a->packing != 1) return false;
    /* ids == NULL: the plain dense call, ncols = M. ids != NULL: the gathered down call,
     * ncols = ne_get_rows (the map's values are trusted device indices, guarded by N1). */
    if (a->ncols < 1 || a->ncols > (a->ids == NULL ? kRowLimit : kRowsMax)) return false;
    const long long blocks_per_col = (a->k + 127) / 128;
    const long long y_bytes = blocks_per_col * (long long)a->ncols * 144;
    const long long x_bytes = ((long long)a->ncols - 1) * a->row_stride * 4 + 4LL * a->k;
    if (!apart(a->y, y_bytes, a->x, x_bytes)) return false;
    if (a->ids != NULL && !apart(a->y, y_bytes, a->ids, 4LL * a->ncols)) return false;
    return true;
}

bool valid_scatter(const inference_cuda_mmq_quantize_scatter_args *a) {
    if (a == NULL || a->x == NULL || a->y == NULL || a->ids_src1_inv == NULL) return false;
    if (!aligned_to(a->x, 16) || !aligned_to(a->y, 16) || !aligned_to(a->ids_src1_inv, 4))
        return false;
    if (!served_type(a->type) || !k_ok(a->k, a->row_stride)) return false;
    if (a->packing != 0 && a->packing != 1) return false;
    if (a->m < 1 || a->m > kRowLimit || a->top_k < 1 || a->top_k > kTopKMax) return false;
    const long long nrows = (long long)a->m * a->top_k;
    if (nrows < 1 || nrows > kRowsMax) return false;
    const long long blocks_per_col = (a->k + 127) / 128;
    const long long y_bytes = blocks_per_col * nrows * 144;
    const long long x_bytes = ((long long)a->m - 1) * a->row_stride * 4 + 4LL * a->k;
    if (!apart(a->y, y_bytes, a->x, x_bytes)) return false;
    if (!apart(a->y, y_bytes, a->ids_src1_inv, 4LL * nrows)) return false;
    return true;
}

/* `k` rounded to the 128-value block grid (the vendored `ne0 % QK8_1_MMQ == 0`). */
long long k_padded(int k) { return ((long long)k + 127) / 128 * 128; }

}  // namespace

extern "C" int inference_cuda_mmq_quantize(const inference_cuda_mmq_quantize_args *a,
                                           void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid_quantize(a)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (a->packing == 1 && a->type == 12) {
            return launch_q8k_mmq<false >({
                (const float *)a->x, (const int32_t *)a->ids,
                (block_q8_1_mmq *)a->y, a->k, a->row_stride, a->ncols, 1}, stream, out_stage);
        }
        Failure failure;
        failure.record(quantize_mmq_q8_1_cuda_policy(
                           (const float *)a->x, (const int32_t *)a->ids, a->y,
                           (ggml_type)a->type, a->k, (int64_t)a->row_stride, /*s02=*/0,
                           /*s03=*/0, k_padded(a->k), (int64_t)a->ncols, /*ne2=*/1, /*ne3=*/1,
                           (cudaStream_t)stream, a->packing == 1),
                       INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_mmq_quantize_scatter(
    const inference_cuda_mmq_quantize_scatter_args *a, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid_scatter(a)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (a->packing == 1 && a->type == 12) {
            return launch_q8k_mmq<true >({
                (const float *)a->x, (const int32_t *)a->ids_src1_inv,
                (block_q8_1_mmq *)a->y, a->k, a->row_stride, a->m, a->top_k}, stream, out_stage);
        }
        Failure failure;
        failure.record(quantize_scatter_mmq_q8_1_cuda_policy(
                           (const float *)a->x, (const int32_t *)a->ids_src1_inv, a->y,
                           (ggml_type)a->type, a->k, (int64_t)a->row_stride, k_padded(a->k),
                           (int64_t)a->m, /*nrows_dst=*/(int64_t)a->m * a->top_k, a->top_k,
                           (cudaStream_t)stream, a->packing == 1),
                       INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
