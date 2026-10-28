/* Device sampler of the Gemma 4 Humanizer decode step.
 *
 * The logits row is consumed on the device. One block computes the maximum, the
 * temperature-scaled exponential and its sum, derives the min-p, top-k and top-p
 * cuts, renormalizes over the kept set, draws one
 * uniform from Philox-4x32-10 seeded by (seed, position), and walks the kept set in
 * ascending token order to the chosen id. Only the id, its log probability and a
 * status word leave the device.
 */
#include <cuda_runtime.h>

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/sample_gemma.h"

namespace {

constexpr int kThreads = 256;  // sm_72 cannot launch 1024 threads of this kernel
constexpr int kSteps = 40;  // bisection steps: 2^-40 of the value range

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

/* Deterministic block reductions: every thread walks the same strided order for its
 * lane, then the warp butterflies and the shared tree fix the summation order. */
__device__ float block_max(float v) {
    __shared__ float red[kThreads / 32];
    v = warp_max(v);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = lane < kThreads / 32 ? red[lane] : -INFINITY;
    return warp_max(v);
}

__device__ int warp_sum_int(int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ int warp_min_int(int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const int other = __shfl_xor_sync(0xffffffffu, v, o);
        v = other < v ? other : v;
    }
    return v;
}

__device__ int block_sum_int(int v) {
    __shared__ int red[kThreads / 32];
    v = warp_sum_int(v);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = lane < kThreads / 32 ? red[lane] : 0;
    return warp_sum_int(v);
}

__device__ int block_min_int(int v) {
    __shared__ int red[kThreads / 32];
    v = warp_min_int(v);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = lane < kThreads / 32 ? red[lane] : INT_MAX;
    return warp_min_int(v);
}

__device__ float block_sum(float v) {
    __shared__ float red[kThreads / 32];
    v = warp_sum(v);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = lane < kThreads / 32 ? red[lane] : 0.0f;
    return warp_sum(v);
}

/* Philox-4x32-10, the canonical counter-based generator of the sampling contract. */
__device__ unsigned int mulhilo(unsigned int a, unsigned int b, unsigned int *hi) {
    const unsigned long long p = (unsigned long long)a * b;
    *hi = (unsigned int)(p >> 32);
    return (unsigned int)p;
}

__device__ unsigned int philox4(unsigned int *c, const unsigned int *key, int rounds) {
    unsigned int k[2] = {key[0], key[1]};
    for (int i = 0; i < rounds; ++i) {
        unsigned int hi0 = 0u, hi1 = 0u;
        const unsigned int lo0 = mulhilo(0xD2511F53u, c[0], &hi0);
        const unsigned int lo2 = mulhilo(0xCD9E8D57u, c[2], &hi1);
        const unsigned int next0 = hi1 ^ c[1] ^ k[0];
        const unsigned int next1 = lo2;
        const unsigned int next2 = hi0 ^ c[3] ^ k[1];
        const unsigned int next3 = lo0;
        c[0] = next0;
        c[1] = next1;
        c[2] = next2;
        c[3] = next3;
        k[0] += 0x9E3779B9u;
        k[1] += 0xBB67AE85u;
    }
    return c[0];
}

/* u in [0, 1) from the top 24 bits of one Philox word. */
__device__ float uniform01(unsigned int seed_lo, unsigned int seed_hi, unsigned int pos_lo,
                           unsigned int pos_hi) {
    unsigned int c[4] = {pos_lo, pos_hi, 0u, 0u};
    unsigned int k[2] = {seed_lo, seed_hi};
    const unsigned int w = philox4(c, k, 10);
    return (float)(w >> 8) * (1.0f / 16777216.0f);
}

__global__ void sample_gemma_kernel(const float *logits, float *probs, int n, float temperature,
                                    int top_k, float top_p, float min_p, unsigned int seed_lo,
                                    unsigned int seed_hi, unsigned int pos_lo, unsigned int pos_hi,
                                    unsigned int *token, float *logprob, unsigned int *status) {
    if (blockIdx.x != 0) return;
    const int t = threadIdx.x;
    if (t == 0) status[0] = 0u;
    __syncthreads();
    float local_max = -INFINITY;
    for (int i = t; i < n; i += kThreads) local_max = fmaxf(local_max, logits[i]);
    const float peak = block_max(local_max);
    if (t == 0 && !isfinite(peak)) atomicOr(status, 4u);

    if (!(temperature > 0.0f)) {
        /* Greedy: the lowest id among the maxima. The peak is already the block max of
         * the logits, so each thread keeps its lowest matching index and the block min
         * of those is the answer. */
        int mine = n;
        for (int i = t; i < n; i += kThreads) {
            if (logits[i] == peak && i < mine) mine = i;
        }
        const int best = block_min_int(mine);
        if (t == 0) {
            token[0] = (unsigned int)best;
            logprob[0] = 0.0f;
        }
        return;
    }

    const float inv_temperature = 1.0f / temperature;
    float local_sum = 0.0f;
    for (int i = t; i < n; i += kThreads) {
        const float p = expf((logits[i] - peak) * inv_temperature);
        probs[i] = p;
        local_sum += p;
    }
    const float total = block_sum(local_sum);
    if (t == 0 && !(total > 0.0f)) atomicOr(status, 2u);
    __syncthreads();

    float local_top = 0.0f;
    for (int i = t; i < n; i += kThreads) local_top = fmaxf(local_top, probs[i]);
    const float top = block_max(local_top);
    float threshold = min_p * top;
    int candidates = 0;
    for (int i = t; i < n; i += kThreads) {
        if (probs[i] < threshold) probs[i] = 0.0f;
        if (probs[i] > 0.0f) ++candidates;
    }
    const int eligible = block_sum_int(candidates);
    if (top_k > 0 && top_k < eligible) {
        /* Positive float bits preserve value order, so this finds the exact kth value. */
        unsigned int lo = 0u, hi = __float_as_uint(top);
        while (lo < hi) {
            const unsigned int mid = lo + (hi - lo + 1u) / 2u;
            int count = 0;
            for (int i = t; i < n; i += kThreads) {
                if (probs[i] >= __uint_as_float(mid)) ++count;
            }
            if (block_sum_int(count) >= top_k) lo = mid;
            else hi = mid - 1u;
        }
        threshold = fmaxf(threshold, __uint_as_float(lo));
        int above = 0;
        for (int i = t; i < n; i += kThreads) {
            if (probs[i] > threshold) ++above;
        }
        const int ties = top_k - block_sum_int(above);
        int first = 0, last = n - 1;
        while (first < last) {
            const int mid = first + (last - first) / 2;
            int count = 0;
            for (int i = t; i <= mid; i += kThreads) {
                if (probs[i] == threshold) ++count;
            }
            if (block_sum_int(count) >= ties) last = mid;
            else first = mid + 1;
        }
        for (int i = t; i < n; i += kThreads) {
            if (probs[i] < threshold || (probs[i] == threshold && i > first)) probs[i] = 0.0f;
        }
        __syncthreads();
    }
    if (top_p < 1.0f) {
        float surviving = 0.0f;
        for (int i = t; i < n; i += kThreads) surviving += probs[i];
        const float surviving_mass = block_sum(surviving);
        float lo = 0.0f, hi = top;
        for (int step = 0; step < kSteps; ++step) {
            const float mid = 0.5f * (lo + hi);
            float mass = 0.0f;
            for (int i = t; i < n; i += kThreads) {
                if (probs[i] >= mid) mass += probs[i];
            }
            if (block_sum(mass) >= top_p * surviving_mass) {
                lo = mid;
                threshold = fmaxf(threshold, mid);
            } else {
                hi = mid;
            }
        }
    }

    /* Kept mass and the draw. The walk needs the cumulative mass in ascending index
     * order, so each thread sums one contiguous chunk and thread 0 scans the chunk sums.
     * Only one chunk is then walked serially. */
    __shared__ float s_chunk[kThreads];
    __shared__ float s_scan[kThreads];
    const int chunk = (n + kThreads - 1) / kThreads;
    const int begin = t * chunk;
    const int end = begin + chunk < n ? begin + chunk : n;
    float local_kept = 0.0f;
    for (int i = begin; i < end; ++i) {
        if (probs[i] >= threshold) local_kept += probs[i];
    }
    s_chunk[t] = local_kept;
    __syncthreads();
    if (t == 0) {
        float acc = 0.0f;
        for (int c = 0; c < kThreads; ++c) {
            s_scan[c] = acc;
            acc += s_chunk[c];
        }
        s_chunk[kThreads - 1] = acc;
    }
    __syncthreads();
    const float kept = s_chunk[kThreads - 1];
    if (!(kept > 0.0f)) {
        if (t == 0) {
            atomicOr(status, 2u);
            token[0] = 0u;
            logprob[0] = 0.0f;
        }
        return;
    }
    if (t == 0) {
        const float u = uniform01(seed_lo, seed_hi, pos_lo, pos_hi);
        float walk = u * kept;
        if (walk > kept) walk = kept;
        int chosen = -1;
        float chosen_p = 0.0f;
        for (int c = 0; c < kThreads; ++c) {
            const float mass = c + 1 < kThreads ? s_chunk[c] : kept - s_scan[c];
            if (!(mass > 0.0f)) continue;
            if (c + 1 < kThreads && walk > mass) {
                walk -= mass;
                continue;
            }
            const int b = c * chunk;
            const int e = b + chunk < n ? b + chunk : n;
            for (int i = b; i < e; ++i) {
                if (probs[i] < threshold || !(probs[i] > 0.0f)) continue;
                chosen = i;
                chosen_p = probs[i];
                walk -= probs[i];
                if (walk <= 0.0f) break;
            }
            break;
        }
        if (chosen < 0) {
            for (int i = 0; i < n; ++i) {
                if (probs[i] >= threshold && probs[i] > 0.0f) {
                    chosen = i;
                    chosen_p = probs[i];
                    break;
                }
            }
        }
        token[0] = (unsigned int)chosen;
        logprob[0] = logf(chosen_p / kept);
    }
}

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }

bool apart(const void *a, size_t ab, const void *b, size_t bb) {
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? ab <= y - x : bb <= x - y;
}

}  // namespace

extern "C" int inference_cuda_sample_gemma_launch(const inference_cuda_sample_gemma_args *args,
                                                  void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->token == NULL || args->logprob == NULL || args->status == NULL ||
            args->probs == NULL || args->logits == NULL)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->n < 32 || args->n > 4194304 || args->n % 32 != 0) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (!isfinite(args->temperature)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->top_k < 0 || args->top_k > args->n) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (!(args->top_p > 0.0f) || args->top_p > 1.0f) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->min_p < 0.0f || args->min_p >= 1.0f) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (!aligned(args->token, 4) || !aligned(args->logprob, 4) || !aligned(args->status, 4) ||
            !aligned(args->probs, 4) || !aligned(args->logits, 4))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const size_t bytes = (size_t)args->n * sizeof(float);
        if (!apart(args->probs, bytes, args->logits, bytes) ||
            !apart(args->token, 4, args->probs, bytes) || !apart(args->token, 4, args->logits, bytes) ||
            !apart(args->logprob, 4, args->probs, bytes) || !apart(args->logprob, 4, args->logits, bytes) ||
            !apart(args->status, 4, args->probs, bytes) || !apart(args->status, 4, args->logits, bytes) ||
            !apart(args->status, 4, args->token, 4) || !apart(args->status, 4, args->logprob, 4))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        sample_gemma_kernel<<<1, kThreads, 0, (cudaStream_t)stream>>>(
            (const float *)args->logits, (float *)args->probs, args->n, args->temperature,
            args->top_k, args->top_p, args->min_p, args->seed_lo, args->seed_hi, args->pos_lo,
            args->pos_hi, (unsigned int *)args->token, (float *)args->logprob,
            (unsigned int *)args->status);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
