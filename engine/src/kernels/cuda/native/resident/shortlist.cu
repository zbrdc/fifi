#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include "abi.h"
#include "device.h"
#include "resident/shortlist.h"

namespace {
using Candidate = inference_cuda_shortlist_candidate;
using Part = inference_cuda_shortlist_result;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kTiles = INFERENCE_CUDA_SHORTLIST_TILES;
constexpr int kK = INFERENCE_CUDA_SHORTLIST_K;
constexpr size_t kScratch = INFERENCE_CUDA_SHORTLIST_SCRATCH_BYTES;

__device__ Candidate empty_candidate() { return {-INFINITY, UINT32_MAX}; }

__device__ bool better(Candidate a, Candidate b) {
    return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
}

__device__ Candidate warp_best(Candidate best) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const Candidate other{__shfl_xor_sync(0xffffffffu, best.logit, offset),
                              __shfl_xor_sync(0xffffffffu, best.id, offset)};
        if (better(other, best)) best = other;
    }
    return best;
}

__device__ Candidate block_best(Candidate best, Candidate *warps, Candidate *chosen) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    best = warp_best(best);
    if (lane == 0) warps[warp] = best;
    __syncthreads();
    if (warp == 0) {
        best = warp_best(lane < kWarps ? warps[lane] : empty_candidate());
        if (lane == 0) *chosen = best;
    }
    __syncthreads();
    return *chosen;
}

__global__ void shortlist_parts_k(const float *logits, int n, Part *parts) {
    __shared__ Candidate warps[kWarps], chosen;
    __shared__ unsigned warp_bad[kWarps];
    const size_t first = size_t(n) * blockIdx.x / kTiles;
    const size_t end = size_t(n) * (blockIdx.x + 1) / kTiles;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    Candidate cutoff{INFINITY, 0};
    bool bad = false;
#pragma unroll 1
    for (int rank = 0; rank < kK; ++rank) {
        Candidate best = empty_candidate();
        for (size_t index = first + threadIdx.x; index < end; index += kThreads) {
            const Candidate value{logits[index], unsigned(index)};
            if (!isfinite(value.logit)) {
                bad = true;
            } else if (better(cutoff, value) && better(value, best)) {
                best = value;
            }
        }
        if (rank == 0) {
            const unsigned any_bad = __any_sync(0xffffffffu, bad);
            if (lane == 0) warp_bad[warp] = any_bad;
        }
        cutoff = block_best(best, warps, &chosen);
        if (threadIdx.x == 0) parts[blockIdx.x].entries[rank] = cutoff;
    }
    if (threadIdx.x == 0) {
        unsigned any_bad = 0;
#pragma unroll
        for (int i = 0; i < kWarps; ++i) any_bad |= warp_bad[i];
        parts[blockIdx.x].non_finite = any_bad;
        for (int i = 0; i < 3; ++i) parts[blockIdx.x].pad[i] = 0;
    }
}

__global__ void shortlist_final_k(Part *parts) {
    __shared__ Candidate warps[kWarps], chosen;
    Candidate cutoff{INFINITY, 0};
#pragma unroll 1
    for (int rank = 0; rank < kK; ++rank) {
        Candidate best = empty_candidate();
        for (int index = threadIdx.x; index < kTiles * kK; index += kThreads) {
            const Candidate value = parts[index / kK].entries[index % kK];
            if (value.id != UINT32_MAX && better(cutoff, value) && better(value, best)) best = value;
        }
        cutoff = block_best(best, warps, &chosen);
        if (threadIdx.x == 0) parts[kTiles].entries[rank] = cutoff;
    }
    if (threadIdx.x == 0) {
        unsigned bad = 0;
#pragma unroll
        for (int i = 0; i < kTiles; ++i) bad |= parts[i].non_finite;
        parts[kTiles].non_finite = bad;
        for (int i = 0; i < 3; ++i) parts[kTiles].pad[i] = 0;
    }
}

bool valid(const inference_cuda_shortlist_args *a, void *stream) {
    if (!a || !stream || !a->logits || !a->scratch || a->n < 1 || a->n > (1 << 30)) return false;
    const uintptr_t x = uintptr_t(a->logits), s = uintptr_t(a->scratch);
    if (x % 4 || s % 16) return false;
    const size_t bytes = size_t(a->n) * 4;
    return s <= x ? kScratch <= x - s : bytes <= s - x;
}
}

extern "C" int inference_cuda_shortlist_launch(const inference_cuda_shortlist_args *a,
                                                void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid(a, stream)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        auto parts = static_cast<Part *>(a->scratch);
        shortlist_parts_k<<<kTiles, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const float *>(a->logits), a->n, parts);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.ok()) {
            shortlist_final_k<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(parts);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
