/*
 * The dense GEMM entry of the vendored-MMQ glue (see prefill/mmq_dense.h). CPF-V4 G2, decision
 * 0049 item 11. No vendored code is copied: the fill is derived from the vendored kernel source
 * (mmq.cuh) and the validation follows the pattern of the quantize entries; the launch is the
 * vendored `mul_mat_q_case`, called, not transcribed. `fixup_need` re-expresses the tile-width
 * selection of vendored mmq.cuh:1444-1461 and the stream-k block rule of mmq.cuh:1395-1412
 * (llama.cpp pin acecd56032ddc34bada14a2d978f110d9c987095, MIT; THIRD_PARTY_NOTICES.md), on
 * top of the vendored config helpers; a test hash-pins both passages and cross-checks it
 * against the Rust sizing.
 */
#include <climits>
#include <cstdint>

#include "abi.h"
#include "device.h"

// The vendored headers keep static host helpers their own objects reference; in this glue TU
// they are unreferenced, which -Wall reports although nothing here is wrong (mmq_facts.cu).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "mmq.cuh"
#pragma GCC diagnostic pop

#include "prefill/mmq_dense.h"
#include "prefill/mmq_facts.h"

namespace {

/* The K grid (plan section B.3, MSG-85 M2): `MMQ_ITER_K` values per k-loop iteration, read
 * with no k clamp. The extent bound is the tree's `MAX_DIM`. */
constexpr int kKGrid = MMQ_ITER_K;
constexpr int kKMax = 32768;
/* The chunk cap (G1's NCOLS_LIMIT), the smallest tile width the launch can pick and the I
 * tile of every kept row; the `2^30` bound is the vendored `variable kbc` guard. */
constexpr int kColsMax = 2048;
constexpr int kTileJMin = 8;
constexpr int kTileI = 128;
constexpr long long kGridLimit = 1LL << 30;
/* `QK8_1_MMQ` values per q8_1 block of 144 B, and the constant pad (G1's `Q8_1_PAD`). */
constexpr long long kQ8BlockBytes = 144;
constexpr long long kQ8Pad = 128 * kQ8BlockBytes;

/* The values and bytes per weight block of the three dense cases; 0 for every other type. */
struct BlockShape {
    int values;
    int bytes;
};

BlockShape block_shape(int type) {
    switch (type) {
        case GGML_TYPE_Q8_0:
            return {(int)ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qk, (int)sizeof(block_q8_0)};
        case GGML_TYPE_Q4_K:
            return {(int)ggml_cuda_type_traits<GGML_TYPE_Q4_K>::qk, (int)sizeof(block_q4_K)};
        case GGML_TYPE_Q6_K:
            return {(int)ggml_cuda_type_traits<GGML_TYPE_Q6_K>::qk, (int)sizeof(block_q6_K)};
        default:
            return {0, 0};
    }
}

bool aligned_to(const void *p, uintptr_t a) { return ((uintptr_t)p & (a - 1)) == 0; }

/* True when the two used spans share no byte. */
bool apart(const void *a, long long a_bytes, const void *b, long long b_bytes) {
    const uintptr_t pa = (uintptr_t)a, pb = (uintptr_t)b;
    return pa + (uintptr_t)a_bytes <= pb || pb + (uintptr_t)b_bytes <= pa;
}

long long ceil_div(long long a, long long b) { return (a + b - 1) / b; }

/* The shape rules: the K grid before every other k rule, then m, n, ld, the destination reach
 * and the tile grid. The Rust wrapper checks the same set in its own order (the grid, the int
 * conversions, then the reach); only which tag wins for a call that breaks two differs. */
bool shape_ok(const inference_cuda_mmq_dense_args *a, BlockShape shape) {
    if (shape.values == 0 || a->k % kKGrid != 0 || a->k <= 0 || a->k > kKMax) return false;
    if (a->m < 1 || a->m > kColsMax || a->n < 1) return false;
    if (a->col_off < 0 || (long long)a->ld < (long long)a->col_off + a->n) return false;
    /* The vendored kernels index the destination in 32-bit int (mmq.cuh:907/953/1041/1130/1273):
     * the last element, (m-1)*ld + n - 1, must stay below 2^31 or the offset wraps. */
    if ((long long)(a->m - 1) * a->ld + a->n > INT_MAX) return false;
    const long long grid =
        ceil_div(a->m, kTileJMin) * ceil_div(a->n, kTileI) * (a->k / shape.values);
    return grid < kGridLimit;
}

/* The wrapper mirrors every rule here, so the native entry cannot return INVALID_ARGUMENT
 * (which would latch the backend); a C caller gets the same refusal before any launch. */
bool valid_dense(const inference_cuda_mmq_dense_args *a) {
    if (a == NULL || a->w == NULL || a->y == NULL || a->dst == NULL) return false;
    const BlockShape shape = block_shape(a->type);
    if (!shape_ok(a, shape)) return false;
    if ((a->fixup == NULL) != (a->fixup_bytes == 0)) return false;
    if (!aligned_to(a->w, 16) || !aligned_to(a->y, 16) || !aligned_to(a->dst, 4)) return false;
    if (a->fixup != NULL && !aligned_to(a->fixup, 16)) return false;
    const long long w_bytes = (long long)a->n * (a->k / shape.values) * shape.bytes;
    const long long y_bytes = ceil_div(a->k, 128) * a->m * kQ8BlockBytes + kQ8Pad;
    const long long dst_bytes = ((long long)(a->m - 1) * a->ld + a->col_off + a->n) * 4;
    if (!apart(a->dst, dst_bytes, a->w, w_bytes) || !apart(a->dst, dst_bytes, a->y, y_bytes)) {
        return false;
    }
    if (a->fixup == NULL) return true;
    const long long f_bytes = (long long)a->fixup_bytes;
    return apart(a->fixup, f_bytes, a->w, w_bytes) && apart(a->fixup, f_bytes, a->y, y_bytes) &&
           apart(a->fixup, f_bytes, a->dst, dst_bytes);
}

/* The dense fill, one pure function of the call (no device facts). */
mmq_args fill_dense_args(const inference_cuda_mmq_dense_args &a) {
    mmq_args args{};
    args.x = (const char *)a.w;
    args.type_x = (ggml_type)a.type;
    args.y = (const int *)a.y;
    args.ids_dst = nullptr;
    args.expert_bounds = nullptr;
    args.dst = (float *)a.dst + a.col_off;
    args.y_scale = nullptr;
    args.ncols_x = a.k;
    args.nrows_x = a.n;
    args.ncols_dst = a.m;
    args.stride_row_x = a.k / block_shape(a.type).values;
    args.ncols_y = a.m;
    args.nrows_dst = a.ld;
    args.nchannels_x = args.nchannels_y = 1;
    args.nsamples_x = args.nsamples_y = 1;
    args.ncols_max = a.m;
    args.ncols_opt = a.m;
    args.fixup = (float *)a.fixup;
    args.fixup_bytes = a.fixup_bytes;
    return args;
}

/* The bytes of stream-k fixup range the launch of `a` demands on `dev` (mmq.cuh:1349-1410):
 * the vendored tile-width choice (the first J with the fewest column tiles whose shared
 * memory fits), the tiling-mode test (efficiency >= 90 %) and the exact need
 * `blocks * J * I` floats, 0 when the launch tiles or the case has no stream-k row. The
 * vendored launch repeats the check with cudaErrorInvalidValue, so a divergence here can
 * only cost a refusal. */
size_t fixup_need(const inference_cuda_mmq_dense_args &a, const fifi_device_facts &dev) {
    // nsm > 0 is the vendored fifi_check_device's refusal; here it only keeps the divisions safe.
    if (dev.nsm <= 0) {
        return 0;
    }
    const ggml_type type = (ggml_type)a.type;
    const bool fallback = a.n % kTileI != 0;
    int best_j = 0, best_tiles = INT_MAX;
    for (int j = kTileJMin; j <= 128 && best_tiles > 1; j += kTileJMin) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, j, fallback, dev.cc);
        if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, dev.cc) > dev.smpbo) {
            continue;
        }
        const int tiles = (a.m + config.J - 1) / config.J;
        if (tiles < best_tiles) {
            best_j = j;
            best_tiles = tiles;
        }
    }
    if (best_j == 0 || !ggml_cuda_mmq_get_stream_k(type, best_j, fallback, dev.cc)) {
        return 0;
    }
    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, best_j, fallback, dev.cc);
    const long long tiles = (long long)ceil_div(a.m, config.J) * ceil_div(a.n, config.I);
    const long long waves = ceil_div(tiles, dev.nsm);
    const bool tiling = GGML_CUDA_CC_IS_NVIDIA(dev.cc) && 100 * tiles / (dev.nsm * waves) >= 90;
    const long long blocks = tiling ? tiles : dev.nsm;
    return tiles % blocks != 0 ? (size_t)blocks * config.J * config.I * sizeof(float) : 0;
}

cudaError_t launch_dense(const mmq_args &args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q8_0:
            return mul_mat_q_case<GGML_TYPE_Q8_0>(args, stream);
        case GGML_TYPE_Q4_K:
            return mul_mat_q_case<GGML_TYPE_Q4_K>(args, stream);
        case GGML_TYPE_Q6_K:
            return mul_mat_q_case<GGML_TYPE_Q6_K>(args, stream);
        default:  // unreachable behind valid_dense
            return cudaErrorInvalidValue;
    }
}

inference_cuda_mmq_args_view view_of(const mmq_args &m) {
    inference_cuda_mmq_args_view v{};
    v.x = m.x;
    v.y = m.y;
    v.ids_dst = m.ids_dst;
    v.expert_bounds = m.expert_bounds;
    v.dst = m.dst;
    v.y_scale = m.y_scale;
    v.fixup = m.fixup;
    v.fixup_bytes = m.fixup_bytes;
    v.ncols_x = m.ncols_x;
    v.nrows_x = m.nrows_x;
    v.ncols_dst = m.ncols_dst;
    v.stride_row_x = m.stride_row_x;
    v.ncols_y = m.ncols_y;
    v.nrows_dst = m.nrows_dst;
    v.nchannels_x = m.nchannels_x;
    v.nchannels_y = m.nchannels_y;
    v.stride_channel_x = m.stride_channel_x;
    v.stride_channel_y = m.stride_channel_y;
    v.stride_channel_dst = m.stride_channel_dst;
    v.nsamples_x = m.nsamples_x;
    v.nsamples_y = m.nsamples_y;
    v.stride_sample_x = m.stride_sample_x;
    v.stride_sample_y = m.stride_sample_y;
    v.stride_sample_dst = m.stride_sample_dst;
    v.ncols_max = m.ncols_max;
    v.ncols_opt = m.ncols_opt;
    v.type_x = (int)m.type_x;
    return v;
}

}  // namespace

extern "C" size_t inference_cuda_mmq_dense_fixup_need(const inference_cuda_mmq_dense_args *a,
                                                      const inference_cuda_mmq_facts *facts) {
    try {
        if (a == NULL || facts == NULL || block_shape(a->type).values == 0 || a->m < 1 ||
            a->m > kColsMax || a->n < 1) {
            return 0;
        }
        const fifi_device_facts dev = {facts->device, facts->cc, facts->nsm, facts->smpbo};
        return fixup_need(*a, dev);
    } catch (...) {
        return 0;
    }
}

extern "C" int inference_cuda_mmq_dense(const inference_cuda_mmq_dense_args *a, void *stream,
                                        int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid_dense(a)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        mmq_args args = fill_dense_args(*a);
        failure.record(inference_cuda_mmq_query_facts(&args.dev), INFERENCE_CUDA_STAGE_LAUNCH);
        const inference_cuda_mmq_facts facts = {args.dev.device, args.dev.cc, args.dev.nsm, 0,
                                                args.dev.smpbo};
        if (failure.ok() && a->fixup_bytes < inference_cuda_mmq_dense_fixup_need(a, &facts)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (failure.ok()) {
            failure.record(launch_dense(args, (cudaStream_t)stream), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_mmq_dense_fill(const inference_cuda_mmq_dense_args *a,
                                             inference_cuda_mmq_args_view *out) {
    try {
        if (out == NULL || !valid_dense(a)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out = view_of(fill_dense_args(*a));
        return INFERENCE_CUDA_OK;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
