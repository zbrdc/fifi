/*
 * CPF-V4 G3: argument validation and the launches of the mmid entries of the MMQ glue. Contract:
 * prefill/mmq_mmid.h. The ids guard (plan rev 3 section
 * C.2 N1) is our own kernel; the vendored entries are called, not transcribed.
 */
#include "prefill/mmq_mmid.h"

#include <cuda_runtime.h>

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"

// The vendored headers keep static host helpers their own objects reference; in this glue TU they
// are unreferenced, which -Wall reports although nothing here is wrong.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "mmid.cuh"
#include "mmq.cuh"
#pragma GCC diagnostic pop

#include "prefill/mmq_facts.h"

namespace {

constexpr int kGuardThreads = 128;
constexpr int64_t kMaxK = 32768;
/* The stream-k limit of mmq.cuh:1402, "variable kbc must not overflow". */
constexpr int64_t kGridLimit = (int64_t)1 << 30;
/* mmq.cuh:1399 computes 100 * ntiles_dst in int (the tile-efficiency percent of the tiling
 * decision), so ntiles_dst must stay at or below INT_MAX / 100. ntiles at the smallest tile
 * (J = 8, I = 128) is the largest the launch can pick, and valid_limits computes exactly
 * that form, so this bound holds for every J the launcher chooses. */
constexpr int64_t kTilesLimit = INT_MAX / 100;
/* Rule M1 (CPF-V4 G1, MSG-85 B1): the one constant pad on every q8_1 buffer, 128 blocks. */
constexpr int64_t kQ81Block = 144;
constexpr int64_t kQ81Pad = 128 * kQ81Block;
/* The documented fixup maximum per SM, J * I = 128 * 128 floats (mmq.cuh:1334-1338). */
constexpr int64_t kFixupFloatsPerSm = 128 * 128;

bool aligned4(const void *p) { return ((uintptr_t)p & 3u) == 0; }

/* True when [a, a + a_bytes) and [b, b + b_bytes) share no byte. */
bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return (uintptr_t)a + a_bytes <= (uintptr_t)b || (uintptr_t)b + b_bytes <= (uintptr_t)a;
}

/* The shape rules every mmid entry shares (mmq_mmid.h). */
bool valid_ids_shape(int n_experts, int m, int top_k) {
    return n_experts >= 1 && n_experts <= INFERENCE_CUDA_MMQ_MMID_MAX_EXPERTS && m >= 1 &&
           m <= INFERENCE_CUDA_MMQ_MMID_MAX_TOKENS && top_k >= 1 &&
           top_k <= INFERENCE_CUDA_MMQ_MMID_MAX_TOP_K && top_k <= n_experts;
}

/* N1, one thread per row: at most 32 ids and 496 compares, no float operation. */
__global__ void mmq_ids_guard_k(const int *__restrict__ ids, int *__restrict__ ids_g, int *status,
                                int n_experts, int m, int top_k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) {
        return;
    }
    const int *in = ids + (size_t)row * top_k;
    bool fault_form = true; /* every slot -1 so far */
    bool flagged = false;   /* an id out of range, or a repeat */
    for (int s = 0; s < top_k; ++s) {
        const int id = in[s];
        fault_form = fault_form && id == -1;
        flagged = flagged || id < 0 || id >= n_experts;
        for (int p = 0; p < s; ++p) {
            flagged = flagged || in[p] == id;
        }
    }
    int *out = ids_g + (size_t)row * top_k;
    for (int s = 0; s < top_k; ++s) {
        out[s] = flagged ? s : in[s];
    }
    if (flagged && status != NULL) {
        atomicOr(status, fault_form ? INFERENCE_CUDA_MMQ_IDS_ROUTER_FAULT
                                    : INFERENCE_CUDA_MMQ_IDS_BAD_ROW);
    }
}

/* Pointers and disjointness of the guard: ids and ids_g hold m * top_k ints each. */
bool valid_guard(const inference_cuda_mmq_ids_guard_args *a) {
    if (a == NULL || !valid_ids_shape(a->n_experts, a->m, a->top_k) || a->ids == NULL ||
        a->ids_g == NULL || !aligned4(a->ids) || !aligned4(a->ids_g)) {
        return false;
    }
    const size_t rows = (size_t)a->m * (size_t)a->top_k * sizeof(int);
    if (!apart(a->ids, rows, a->ids_g, rows)) {
        return false;
    }
    return a->status == NULL || (aligned4(a->status) && apart(a->status, 4, a->ids, rows) &&
                                 apart(a->status, 4, a->ids_g, rows));
}

cudaError_t launch_guard(const int *ids, int *ids_g, int *status, int n_experts, int m, int top_k,
                         cudaStream_t stream) {
    const int blocks = (m + kGuardThreads - 1) / kGuardThreads;
    mmq_ids_guard_k<<<blocks, kGuardThreads, 0, stream>>>(ids, ids_g, status, n_experts, m,
                                                          top_k);
    return cudaGetLastError();
}

/* Pointers and disjointness of the builder: ids_g, ids_src1, the written m * top_k ints of
 * ids_dst, expert_bounds and the optional status word, each apart from ids and from each other. */
bool valid_build(const inference_cuda_mmq_build_ids_args *a) {
    if (a == NULL || !valid_ids_shape(a->n_experts, a->m, a->top_k) || a->ids == NULL ||
        !aligned4(a->ids)) {
        return false;
    }
    const size_t rows = (size_t)a->m * (size_t)a->top_k * sizeof(int);
    const void *const out[5] = {a->ids_g, a->ids_src1, a->ids_dst, a->expert_bounds, a->status};
    const size_t bytes[5] = {rows, rows, rows, ((size_t)a->n_experts + 1) * sizeof(int),
                             sizeof(int)};
    for (int i = 0; i < 5; ++i) {
        if (out[i] == NULL) {
            if (i < 4) {
                return false; /* only the status word is optional */
            }
            continue;
        }
        bool ok = aligned4(out[i]) && apart(out[i], bytes[i], a->ids, rows);
        for (int j = 0; j < i; ++j) {
            ok = ok && (out[j] == NULL || apart(out[i], bytes[i], out[j], bytes[j]));
        }
        if (!ok) {
            return false;
        }
    }
    return true;
}

/* Bytes of one block of a grouped weight type, 0 for a type with no arm. Both take qk = 256. */
int64_t grouped_block_bytes(int64_t type) {
    if (type == INFERENCE_CUDA_TYPE_Q4_K) {
        return 144;
    }
    return type == INFERENCE_CUDA_TYPE_Q6_K ? 210 : 0;
}

/* The grouped mmq_args form of mmq_mmid.h, before any product is formed. */
bool valid_grouped(const inference_cuda_mmq_mmid_args *a) {
    const int64_t k = a->ncols_x, m = a->ncols_max, ne = a->ncols_dst, e = a->nchannels_x;
    const bool tokens = a->ncols_opt <= m && m >= 1 && m <= INFERENCE_CUDA_MMQ_MMID_MAX_TOKENS &&
                        a->ncols_y == ne && ne >= m && ne % m == 0;
    const bool experts = a->nchannels_y == e && e >= 1 && e <= INFERENCE_CUDA_MMQ_MMID_MAX_EXPERTS;
    const bool picks = tokens && experts && ne / m <= INFERENCE_CUDA_MMQ_MMID_MAX_TOP_K &&
                       ne / m <= e;
    const bool weight = grouped_block_bytes(a->type_x) != 0 && k % 256 == 0 && k >= 256 &&
                        k <= kMaxK && a->stride_row_x == k / 256 && a->nrows_x >= 1 &&
                        a->nrows_x <= INT_MAX;
    const bool flat = a->nsamples_x == 1 && a->nsamples_y == 1 && a->stride_sample_x == 0 &&
                      a->stride_sample_y == 0 && a->stride_sample_dst == 0 &&
                      a->stride_channel_y == 0 && a->stride_channel_dst == 0;
    return picks && weight && flat && a->nrows_dst >= a->nrows_x && a->nrows_dst <= INT_MAX;
}

/* The int limits of the vendored kernels; every product is bounded by valid_grouped first. */
bool valid_limits(const inference_cuda_mmq_mmid_args *a) {
    const int64_t kb = a->ncols_x / 256, n = a->nrows_x, e = a->nchannels_x;
    const int64_t tiles = (a->ncols_max + 7) / 8 * ((n + 127) / 128) * e;
    const int64_t stride = a->stride_channel_x;
    if (tiles > kTilesLimit || tiles * kb >= kGridLimit || stride < n * kb || stride > INT_MAX) {
        return false;
    }
    return (e - 1) * stride + n * kb <= INT_MAX &&
           (a->ncols_dst - 1) * a->nrows_dst + n <= INT_MAX;
}

/* One range of the launch: its pointer, the bytes it must hold, its capacity, its alignment. */
struct Extent {
    const void *p;
    size_t need;
    size_t have;
    uintptr_t align;
};

/* Capacities, alignment and disjointness, after valid_limits. Extents: x, y, ids_dst,
 * expert_bounds, dst, fixup; dst is apart from all others, fixup from the first four. */
bool valid_ranges(const inference_cuda_mmq_mmid_args *a, int nsm) {
    const int64_t kb = a->ncols_x / 256, ne = a->ncols_dst;
    const int64_t blocks = (a->nchannels_x - 1) * a->stride_channel_x + a->nrows_x * kb;
    const Extent r[6] = {
        {a->x, (size_t)(blocks * grouped_block_bytes(a->type_x)), a->x_bytes, 16},
        {a->y, (size_t)(a->ncols_x / 128 * ne * kQ81Block + kQ81Pad), a->y_bytes, 16},
        {a->ids_dst, (size_t)(ne + a->ncols_max + 127) * sizeof(int), a->ids_dst_bytes, 4},
        {a->expert_bounds, (size_t)(a->nchannels_x + 1) * sizeof(int), a->expert_bounds_bytes, 4},
        {a->dst, (size_t)((ne - 1) * a->nrows_dst + a->nrows_x) * sizeof(float), a->dst_bytes, 4},
        {a->fixup, (size_t)nsm * kFixupFloatsPerSm * sizeof(float), a->fixup_bytes, 16},
    };
    for (int i = 0; i < 6; ++i) {
        bool ok = r[i].p != NULL && r[i].have >= r[i].need && (uintptr_t)r[i].p % r[i].align == 0;
        for (int j = 0; j < i && i >= 4; ++j) {
            ok = ok && apart(r[i].p, r[i].need, r[j].p, r[j].need);
        }
        if (!ok) {
            return false;
        }
    }
    return nsm > 0;
}

/* The vendored arguments of a valid call: the geometry field for field, the facts as queried. */
mmq_args to_mmq_args(const inference_cuda_mmq_mmid_args *a, const fifi_device_facts &facts) {
    mmq_args m{};
    m.x = (const char *)a->x;
    m.type_x = (ggml_type)a->type_x;
    m.y = (const int *)a->y;
    m.ids_dst = a->ids_dst;
    m.expert_bounds = a->expert_bounds;
    m.dst = a->dst;
    m.y_scale = nullptr; /* NVFP4 only */
    m.ncols_x = a->ncols_x;
    m.nrows_x = a->nrows_x;
    m.ncols_dst = a->ncols_dst;
    m.stride_row_x = a->stride_row_x;
    m.ncols_y = a->ncols_y;
    m.nrows_dst = a->nrows_dst;
    m.nchannels_x = a->nchannels_x;
    m.nchannels_y = a->nchannels_y;
    m.stride_channel_x = a->stride_channel_x;
    m.stride_channel_y = a->stride_channel_y;
    m.stride_channel_dst = a->stride_channel_dst;
    m.nsamples_x = a->nsamples_x;
    m.nsamples_y = a->nsamples_y;
    m.stride_sample_x = a->stride_sample_x;
    m.stride_sample_y = a->stride_sample_y;
    m.stride_sample_dst = a->stride_sample_dst;
    m.ncols_max = a->ncols_max;
    m.ncols_opt = a->ncols_opt;
    m.fixup = a->fixup;
    m.fixup_bytes = a->fixup_bytes;
    m.dev = facts;
    return m;
}

}  // namespace

extern "C" int inference_cuda_mmq_ids_guard(const inference_cuda_mmq_ids_guard_args *a,
                                            void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid_guard(a)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        failure.record(launch_guard(a->ids, a->ids_g, a->status, a->n_experts, a->m, a->top_k,
                                    (cudaStream_t)stream),
                       INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_mmq_build_ids(const inference_cuda_mmq_build_ids_args *a,
                                            void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid_build(a)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        fifi_device_facts facts{};
        const cudaError_t queried = inference_cuda_mmq_query_facts(&facts);
        if (queried != cudaSuccess) {
            return (int)queried;
        }
        if ((size_t)a->m * sizeof(int) > facts.smpbo) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const cudaStream_t s = (cudaStream_t)stream;
        Failure failure;
        failure.record(launch_guard(a->ids, a->ids_g, a->status, a->n_experts, a->m, a->top_k, s),
                       INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.ok()) {
            /* nchannels_y = 1 and sis1 = 1 are the pin's arguments for one activation row per
             * token (mmq.cu:193-201, dedup_bcast); the inverse branch reads neither. */
            const cudaError_t sorted = ggml_cuda_launch_mm_ids_helper(
                a->ids_g, a->ids_src1, a->ids_dst, a->expert_bounds, a->n_experts, a->m, a->top_k,
                1, a->top_k, 1, true, facts, s);
            failure.record(sorted, INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_mmq_mmid_launch(const inference_cuda_mmq_mmid_args *a, void *stream,
                                              int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (a == NULL || !valid_grouped(a) || !valid_limits(a)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        fifi_device_facts facts{};
        const cudaError_t queried = inference_cuda_mmq_query_facts(&facts);
        if (queried != cudaSuccess) {
            return (int)queried;
        }
        if (!valid_ranges(a, facts.nsm)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const mmq_args args = to_mmq_args(a, facts);
        const cudaStream_t s = (cudaStream_t)stream;
        Failure failure;
        failure.record(a->type_x == INFERENCE_CUDA_TYPE_Q4_K
                           ? mul_mat_q_case<GGML_TYPE_Q4_K>(args, s)
                           : mul_mat_q_case<GGML_TYPE_Q6_K>(args, s),
                       INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
