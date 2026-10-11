/*
 * RD-4d: the step parameter kernels of the resident Qwen3.6 decode step (decision 0047):
 * step_begin, step_advance and the greedy argmax with token chaining.
 *
 * Design guidance only, no code copied. Ideas from MegaCapybara v1.35 static RE (lessons C3: per-launch
 * data in a small device table, so one captured graph serves every position; C1: the sampler writes
 * the next input on the device; here greedy only, no device sampler, decision 0047 item 8). The code
 * is original CUDA C for sm_86 (sm_72 compiles), written from the shapes of this repository.
 *
 * Argmax rule. It is the rule of family/sampler.rs::argmax (the largest value, the lowest index
 * on a tie) for every row that Sampler::sample accepts. Sampler::sample first runs check_logits
 * and fails on any value that is not finite (NaN, +infinity, -infinity). The device cannot
 * raise that error, so it sets status mask 2 for the same predicate and the host raises it. The
 * token of such a row is still defined, so that the kernel is a total function: a NaN never
 * wins (it fails both comparisons of beats(), so it can never become a candidate), +infinity
 * wins like a value, -infinity is the smallest value, and a row of only NaN gives -1. The
 * host does not use that token.
 *
 * beats() is a strict total order on (value, index) pairs with distinct indices, so the best
 * pair is the same for every reduction shape: the result does not depend on the order of the
 * threads, warps or blocks. There is no float arithmetic, no float atomic, and the only atomic
 * is an integer OR of one bit.
 *
 * Step words. A valid step state is 0 <= pos <= kPosMax, kv_len == pos + 1 and kv_slot == pos
 * (kPosMax = 2^30 - 2, so kv_len never exceeds 2^30 - 1). step_begin clears the status and sets
 * mask 4 (invalid step state) when the state is not valid. step_advance moves all three words by m when the
 * state is valid and pos + m <= kPosMax, so kv_len == pos + 1 holds after every advance. In any
 * other case it writes -1 into all three words: a word is never clamped to row 0, and the next
 * step_begin reports the poison through the status. The host checks pos + m <= rows before every
 * launch (REPLY-44), so this path is reached only by a corrupt word.
 *
 * Two launches on one stream: argmax_partial_k (kBlocks blocks of kThreads threads; a
 * grid-stride loop, then one candidate and one flag per block) and argmax_final_k (one block of
 * kBlocks threads, one partial per thread). One block of 1024 threads measured 32 to 33 us
 * against 4.45 us for this shape on the same cold row (STAGE1-KERNELS-PLAN-files).
 */

#include <cuda_runtime.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/step_params.h"

namespace {

constexpr int kThreads = 256;
constexpr int kBlocks = 256;  // also the block size of argmax_final_k: one partial per thread
constexpr int kWarps = kThreads / 32;
constexpr int kNone = 0x7fffffff;      // index of "no candidate"; larger than every valid index
constexpr int kMaxN = 1 << 30;         // keeps every index below kNone
constexpr int kWordMax = (1 << 30) - 1;  // largest kv_len; a step word never reaches 2^30
constexpr int kPosMax = kWordMax - 1;    // largest pos, so that kv_len = pos + 1 fits
constexpr int kPoison = -1;              // written into pos, kv_len and kv_slot by a refused advance
constexpr int kMaxAdvance = 4096;
constexpr int kMaskNonFinite = INFERENCE_CUDA_STATUS_NONFINITE_LOGITS;
constexpr int kMaskInvalidState = INFERENCE_CUDA_STATUS_INVALID_STEP_STATE;
static_assert(kBlocks == kThreads, "argmax_final_k reads one partial per thread");

/* True when (v, i) is better than (bv, bi): larger value, or equal value and lower index. A NaN
 * fails both tests on either side. */
__device__ __forceinline__ bool beats(float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
}

/* After the call every lane of the warp holds the best pair of the warp. */
__device__ __forceinline__ void warp_best(float &v, int &i) {
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, i, o);
        if (beats(ov, oi, v, i)) {
            v = ov;
            i = oi;
        }
    }
}

/* Thread 0 holds the best pair of the block after the call. Every thread of the block must call
 * it (it contains a barrier). The block has kThreads threads. */
__device__ void block_best(float &v, int &i) {
    __shared__ float sv[kWarps];
    __shared__ int si[kWarps];
    warp_best(v, i);
    if ((threadIdx.x & 31) == 0) {
        sv[threadIdx.x >> 5] = v;
        si[threadIdx.x >> 5] = i;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        v = sv[0];
        i = si[0];
        for (int k = 1; k < kWarps; ++k) {
            if (beats(sv[k], si[k], v, i)) {
                v = sv[k];
                i = si[k];
            }
        }
    }
}

__global__ void __launch_bounds__(kThreads)
    argmax_partial_k(const float *x, int n, float *pv, int *pi, int *pbad) {
    float bv = -INFINITY;
    int bi = kNone;
    int bad = 0;
    const int stride = gridDim.x * kThreads;
    for (int i = blockIdx.x * kThreads + threadIdx.x; i < n; i += stride) {
        const float v = x[i];
        bad |= !isfinite(v);
        if (beats(v, i, bv, bi)) {
            bv = v;
            bi = i;
        }
    }
    const int any_bad = __syncthreads_or(bad);
    block_best(bv, bi);
    if (threadIdx.x == 0) {
        pv[blockIdx.x] = bv;
        pi[blockIdx.x] = bi;
        pbad[blockIdx.x] = any_bad != 0;
    }
}

__global__ void __launch_bounds__(kBlocks) argmax_final_k(const float *pv, const int *pi, const int *pbad,
                                                         inference_cuda_step_params *p, int *result) {
    float bv = pv[threadIdx.x];
    int bi = pi[threadIdx.x];
    const int any_bad = __syncthreads_or(pbad[threadIdx.x]);
    block_best(bv, bi);
    if (threadIdx.x == 0) {
        const int token = bi == kNone ? -1 : bi;
        const int bits = any_bad != 0 ? kMaskNonFinite : 0;
        // atomicOr with 0 still returns the current word, so the result pair sees the final status.
        const int before = atomicOr(&p->status, bits);
        p->token_id = token;  // token chaining: the next replay embeds this token
        if (result != NULL) {
            result[0] = token;
            result[1] = before | bits;
        }
    }
}

/* The state a step may start from (see the header comment). pos is tested first, so pos + 1 does
 * not overflow. */
__device__ __forceinline__ bool step_state_valid(int pos, int kv_len, int kv_slot) {
    bool ok = pos >= 0;
    ok = ok && pos <= kPosMax;
    ok = ok && kv_len == pos + 1;
    ok = ok && kv_slot == pos;
    return ok;
}

__global__ void step_begin_k(inference_cuda_step_params *p) {
    const int cleared = step_state_valid(p->pos, p->kv_len, p->kv_slot) ? 0 : kMaskInvalidState;
    p->status = cleared;
}

__global__ void step_advance_k(inference_cuda_step_params *p, int m) {
    const long long add = m;
    const bool ok = step_state_valid(p->pos, p->kv_len, p->kv_slot) && (long long)p->pos + add <= kPosMax;
    if (!ok) {
        p->pos = p->kv_len = p->kv_slot = kPoison;
        return;
    }
    p->pos = (int)((long long)p->pos + add);
    p->kv_len = (int)((long long)p->kv_len + add);
    p->kv_slot = (int)((long long)p->kv_slot + add);  // kv_slot is reserved and equals pos (REPLY-44)
}

bool aligned_to(const void *p, size_t n) { return ((uintptr_t)p % n) == 0; }

bool valid_advance(const inference_cuda_step_advance_args *a) {
    const bool m_low = a != NULL && a->m >= 1;
    const bool m_high = a != NULL && a->m <= kMaxAdvance;
    const bool params_ok = a != NULL && a->params != NULL && aligned_to(a->params, 4);
    return m_low && m_high && params_ok;
}

bool valid_argmax(const inference_cuda_argmax_args *a) {
    return a != NULL && a->logits != NULL && a->part != NULL && a->params != NULL && a->n >= 1 &&
           a->n <= kMaxN && aligned_to(a->logits, 4) && aligned_to(a->part, 4) &&
           aligned_to(a->params, 4) && (a->result == NULL || aligned_to(a->result, 8));
}

int launch_status(int *out_stage) {
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

}  // namespace

extern "C" int inference_cuda_step_begin_launch(void *params, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (params == NULL || !aligned_to(params, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        step_begin_k<<<1, 1, 0, (cudaStream_t)stream>>>((inference_cuda_step_params *)params);
        return launch_status(out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_step_advance_launch(const inference_cuda_step_advance_args *a,
                                                  void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid_advance(a)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        step_advance_k<<<1, 1, 0, (cudaStream_t)stream>>>((inference_cuda_step_params *)a->params,
                                                          a->m);
        return launch_status(out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_argmax_launch(const inference_cuda_argmax_args *a, void *stream,
                                            int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid_argmax(a)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const cudaStream_t s = (cudaStream_t)stream;
        float *pv = (float *)a->part;
        int *pi = (int *)(pv + kBlocks);
        int *pbad = pi + kBlocks;
        Failure failure;
        argmax_partial_k<<<kBlocks, kThreads, 0, s>>>((const float *)a->logits, a->n, pv, pi, pbad);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.ok()) {
            argmax_final_k<<<1, kBlocks, 0, s>>>(pv, pi, pbad, (inference_cuda_step_params *)a->params,
                                                 (int *)a->result);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
