/* CPF-ATTN-A K1: argument validation and the one launch of attn_prep_chunk, the chunk KV
 * write of the prefill attention (decision 0049 item 2). Design guidance only, no code
 * copied: the math is a byte-for-byte re-expression of the vendored attn_prep (RD-4V, the
 * user's own xavier-qwen code, vendor/xavier-qwen/src/ops.cu attn_prep_k) — the same
 * per-head RMSNorm expression and 256-thread block-reduction order (ops.cu:184-185), the
 * same NEOX rope over the first n_rot dimensions (ops.cu:188-194), the same fp16
 * nearest-even cache writes (ops.cu:179/198) and the same launch shape (ops.cu:231-234) —
 * re-stated here because the vendored helpers sit in an anonymous namespace of ops.cu. The
 * vendored kernel stays decode-only (it reads the device pos word, ops.cu:166, and ORs mask
 * 4 of StepParams.status, attn_prep.cu:23); this chunk form takes p0 as a launch argument
 * (0049 item 1), writes post-rope f32 rows for the KV8 writer in the RowsOut mode, and ORs
 * mask 8 of the prefill status word (0049 item 10). Test 4 of the wrapper pins the bit
 * equality against the vendored entry. Contract: prefill/attn_prep_chunk.h. */
#include "prefill/attn_prep_chunk.h"

#include <cuda_fp16.h>

#include <math.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"

namespace {

constexpr int kHeadDim = 256;  // the only head size (the wrapper refuses every other value)
constexpr int kMaxHeads = 256;
constexpr int kMaxKvHeads = 64;
constexpr int kMaxGridY = 65535;
constexpr int kMaxRows = (1 << 30) - 1;

struct ChunkArgs {
    const float *proj;
    const float *q_norm;
    const float *k_norm;
    __half *k_cache;  // APPEND_FP16 mode
    __half *v_cache;
    float *q;       // [m][n_head][D] out
    float *k_rows;  // ROWS_OUT mode: [m][n_kv_head][D]
    float *v_rows;
    int *status;  // optional prefill status word (decision 0049 item 10)
    int ld, p0, m, n_head, n_kv_head, n_rot, rows, mode;
    float eps, rope_base;
};

bool aligned_to(const void *p, uintptr_t a) { return ((uintptr_t)p & (a - 1)) == 0; }

/* Pointers of one mode, exactly where the mode leaves them: the unused pair must be NULL
 * (the Rust side models this with an enum, so an inconsistent struct can only come from C). */
bool valid_pointers(const inference_cuda_attn_prep_chunk_args *a) {
    if (a->proj == NULL || a->q_norm == NULL || a->k_norm == NULL || a->q == NULL) {
        return false;
    }
    if (!aligned_to(a->proj, 4) || !aligned_to(a->q_norm, 4) || !aligned_to(a->k_norm, 4) ||
        !aligned_to(a->q, 16) || (a->status != NULL && !aligned_to(a->status, 4))) {
        return false;
    }
    if (a->mode == INFERENCE_CUDA_PREP_APPEND_FP16) {
        return a->k_cache != NULL && a->v_cache != NULL && a->k_rows == NULL &&
               a->v_rows == NULL && aligned_to(a->k_cache, 16) && aligned_to(a->v_cache, 16);
    }
    if (a->mode == INFERENCE_CUDA_PREP_ROWS_OUT) {
        return a->k_rows != NULL && a->v_rows != NULL && a->k_cache == NULL &&
               a->v_cache == NULL && aligned_to(a->k_rows, 4) && aligned_to(a->v_rows, 4);
    }
    return false;
}

/* Every shape rule except the p0 bound, so the fault entry can reuse the check while it
 * deliberately violates that bound. */
bool valid_shape_sans_bounds(const inference_cuda_attn_prep_chunk_args *a) {
    const bool heads = a->n_head >= 1 && a->n_head <= kMaxHeads && a->n_kv_head >= 1 &&
                       a->n_kv_head <= kMaxKvHeads;
    const bool tokens = a->m >= 1 && a->m <= kMaxGridY;
    const bool rope = a->n_rot >= 2 && a->n_rot <= kHeadDim &&
                      (a->n_rot & (a->n_rot - 1)) == 0;  // a power of two: RP1's domain
    const bool cache = a->rows >= 1 && a->rows <= kMaxRows;
    const long long row_width =
        ((long long)a->n_head * 2 + (long long)a->n_kv_head * 2) * kHeadDim;
    return a->head_dim == kHeadDim && heads && tokens && rope && cache && a->ld >= row_width;
}

/* The p0 bound the wrapper, this entry and the kernel all check (0049 item 2): p0 >= 0 and
 * p0 + m <= rows, the sum in 64 bits. */
bool valid_bounds(const inference_cuda_attn_prep_chunk_args *a) {
    return a->p0 >= 0 && (long long)a->p0 + a->m <= a->rows;
}

bool valid_scalars(const inference_cuda_attn_prep_chunk_args *a) {
    return isfinite(a->eps) && a->eps > 0.0f && isfinite(a->rope_base) && a->rope_base > 1.0f;
}

bool valid_attn_prep_chunk(const inference_cuda_attn_prep_chunk_args *a) {
    return a != NULL && valid_pointers(a) && valid_shape_sans_bounds(a) && valid_bounds(a) &&
           valid_scalars(a);
}

__device__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Sum of v over the block, returned to every thread. The vendored reduction (ops.cu:20-29):
// identical order and barriers, re-stated because the vendored helper is file-local.
__device__ float block_sum(float v) {
    __shared__ float red[32];
    v = warp_sum(v);
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();  // earlier callers may still be reading red
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = lane < (int)blockDim.x / 32 ? red[lane] : 0.f;
    return warp_sum(v);
}

// Block (head, token), thread per dim: RMSNorm, NEOX rope over the first n_rot dims at the
// position p0 + t. Heads are the n_head queries, then n_kv_head keys, then values — the
// vendored attn_prep_k with p0 a launch argument and the outputs chosen by the mode.
__global__ void attn_prep_chunk_k(ChunkArgs a) {
    constexpr int D = kHeadDim;
    __shared__ float x[D];
    int head = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    // The device re-check of the launch arguments (decision 0049 item 2), the vendored form
    // (ops.cu:166-172): the answer is the same for every block, so the early return is block
    // uniform and happens before the first barrier. A refused launch writes no cache row and
    // no k/v row, sets q to NaN so the attention that reads it cannot yield a plausible
    // value, and ORs mask 8 into the optional prefill status word (item 10).
    if ((long long)a.p0 < 0 || (long long)a.p0 + a.m > a.rows) {
        if (a.status && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
            atomicOr(a.status, INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION);
        if (head < a.n_head) a.q[((size_t)t * a.n_head + head) * D + d] = __int_as_float(0x7fc00000);
        return;
    }
    int pos = a.p0 + t;
    const float *p = a.proj + (size_t)t * a.ld;
    int kv_base = a.n_head * 2 * D;
    if (head >= a.n_head + a.n_kv_head) {
        int vh = head - a.n_head - a.n_kv_head;
        float v = p[kv_base + (a.n_kv_head + vh) * D + d];
        if (a.mode == INFERENCE_CUDA_PREP_ROWS_OUT)
            a.v_rows[((size_t)t * a.n_kv_head + vh) * D + d] = v;
        else
            a.v_cache[((size_t)pos * a.n_kv_head + vh) * D + d] = __float2half(v);
        return;
    }
    bool is_q = head < a.n_head;
    float v = is_q ? p[head * 2 * D + d] : p[kv_base + (head - a.n_head) * D + d];
    float rstd = rsqrtf(block_sum(v * v) / D + a.eps);
    x[d] = v * rstd * (is_q ? a.q_norm : a.k_norm)[d];
    __syncthreads();
    float out = x[d];
    int half = a.n_rot / 2;
    if (d < a.n_rot) {
        int i = d % half;
        float theta = pos * powf(a.rope_base, -2.f * i / a.n_rot), sn, cs;
        sincosf(theta, &sn, &cs);
        out = d < half ? x[d] * cs - x[d + half] * sn : x[d - half] * sn + x[d] * cs;
    }
    if (is_q) {
        a.q[((size_t)t * a.n_head + head) * D + d] = out;
    } else if (a.mode == INFERENCE_CUDA_PREP_ROWS_OUT) {
        a.k_rows[((size_t)t * a.n_kv_head + head - a.n_head) * D + d] = out;
    } else {
        a.k_cache[((size_t)pos * a.n_kv_head + head - a.n_head) * D + d] = __float2half(out);
    }
}

ChunkArgs to_chunk_args(const inference_cuda_attn_prep_chunk_args *a) {
    ChunkArgs k = {};
    k.proj = (const float *)a->proj;
    k.q_norm = (const float *)a->q_norm;
    k.k_norm = (const float *)a->k_norm;
    k.k_cache = (__half *)a->k_cache;
    k.v_cache = (__half *)a->v_cache;
    k.q = (float *)a->q;
    k.k_rows = (float *)a->k_rows;
    k.v_rows = (float *)a->v_rows;
    k.status = (int *)a->status;
    k.ld = a->ld;
    k.p0 = a->p0;
    k.m = a->m;
    k.n_head = a->n_head;
    k.n_kv_head = a->n_kv_head;
    k.n_rot = a->n_rot;
    k.rows = a->rows;
    k.mode = a->mode;
    k.eps = a->eps;
    k.rope_base = a->rope_base;
    return k;
}

Failure launch_chunk(const ChunkArgs &k, void *stream) {
    Failure failure;
    attn_prep_chunk_k<<<dim3(k.n_head + 2 * k.n_kv_head, k.m), kHeadDim, 0, (cudaStream_t)stream>>>(
        k);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    return failure;
}

}  // namespace

extern "C" int inference_cuda_attn_prep_chunk_launch(
    const inference_cuda_attn_prep_chunk_args *a, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid_attn_prep_chunk(a)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const ChunkArgs k = to_chunk_args(a);
        /* The launch status is the only status: the kernel returns cudaGetLastError(). */
        const Failure failure = launch_chunk(k, stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

#ifdef INFERENCE_CUDA_FAULT_QUALIFICATION
extern "C" int inference_cuda_attn_prep_chunk_fault_bad_bounds_launch(
    const inference_cuda_attn_prep_chunk_args *a, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        /* Every rule except the p0 bound holds, so the pointers and shapes the guard's NaN-q
         * write touches are valid; only the bound is violated, by force. The kernel's device
         * re-check (the guard above) is the code under test. */
        if (a == NULL || !valid_pointers(a) || !valid_shape_sans_bounds(a) ||
            !valid_scalars(a)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        ChunkArgs k = to_chunk_args(a);
        k.p0 = a->rows;  // rows + m > rows: the bound violation, with p0 still in range of int
        const Failure failure = launch_chunk(k, stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
#endif
