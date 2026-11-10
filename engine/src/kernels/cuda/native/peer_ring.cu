#include <cuda/atomic>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>

#include "abi.h"
#include "abi_ring.h"
#include "peer_ring_policy.h"

namespace {

constexpr int kMaxValues = 1 << 22;
constexpr int kMaxF16Values = 1 << 20;
constexpr int kMaxPeers = 3;
constexpr unsigned kStatusTimeout = 1u;
constexpr unsigned kStatusOverflow = 2u;
constexpr unsigned kFlagOverflow = 1u;
constexpr int kHeaderBytes = 128;
constexpr int kThreads = 256;

int finish(cudaError_t err) {
    (void)cudaGetLastError();
    return err == cudaSuccess ? INFERENCE_CUDA_OK : (int)err;
}

bool aligned16(const void *p) {
    return (reinterpret_cast<unsigned long long>(p) & 15ull) == 0ull;
}

__device__ unsigned *seq_word(void *slot) {
    return reinterpret_cast<unsigned *>(slot);
}

// The producer/consumer pair for the sequence word: a system-scope release
// store after the payload and flags, a system-scope acquire load before
// the payload reads. A plain or volatile word would compile, and would
// pass on one device, but only the system-scope pair orders the payload
// across devices. The generated PTX carries st.release.sys /
// ld.acquire.sys; the build that last changed this checked for them.
using SeqAtomic = cuda::atomic<unsigned, cuda::thread_scope_system>;

__device__ void seq_publish(void *slot, unsigned seq) {
    reinterpret_cast<SeqAtomic *>(seq_word(slot))
        ->store(seq, cuda::memory_order_release);
}

__device__ unsigned seq_observe(const void *slot) {
    return reinterpret_cast<const SeqAtomic *>(slot)->load(
        cuda::memory_order_acquire);
}

__device__ unsigned *flags_word(void *slot) {
    return reinterpret_cast<unsigned *>(reinterpret_cast<char *>(slot) + 4);
}

__device__ void *payload_of(void *slot) {
    return reinterpret_cast<char *>(slot) + kHeaderBytes;
}

__global__ void publish_f32_kernel(const float *src, void *dst_slot,
                                   int count) {
    float *dst = reinterpret_cast<float *>(payload_of(dst_slot));
    for (int i = (int)(blockIdx.x * kThreads + threadIdx.x); i < count;
         i += kThreads * gridDim.x) {
        dst[i] = src[i];
    }
}

/// One thread, ordered after the multi-block f32 data launch on the same
/// stream. Clear the old slot's flags before making the new sequence visible.
__global__ void publish_commit_kernel(void *dst_slot, unsigned seq) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *flags_word(dst_slot) = 0;
        seq_publish(dst_slot, seq);
    }
}

__device__ bool wait_for_slots(const void *slot0, const void *slot1,
                               const void *slot2, int peer_count,
                               unsigned seq,
                               unsigned long long timeout_cycles) {
    const void *slots[kMaxPeers] = {slot0, slot1, slot2};
    const unsigned long long start = clock64();
    for (;;) {
        bool ready = true;
        for (int p = 0; p < peer_count; ++p) {
            const unsigned seen = seq_observe(slots[p]);
            if (seen != seq) {
                ready = false;
                break;
            }
        }
        if (ready) {
            return true;
        }
        if (clock64() - start >= timeout_cycles) {
            return false;
        }
    }
}

template <typename Payload>
__global__ void wait_sum_kernel(float *out, const float *own,
                                const void *slot0, const void *slot1,
                                const void *slot2, int peer_count,
                                int count, unsigned seq,
                                unsigned long long timeout_cycles,
                                unsigned *status) {
    __shared__ int proceed;
    if (threadIdx.x == 0) {
        proceed = 0;
        if (wait_for_slots(slot0, slot1, slot2, peer_count, seq,
                           timeout_cycles)) {
            const void *slots[kMaxPeers] = {slot0, slot1, slot2};
            unsigned flags = 0;
            for (int p = 0; p < peer_count; ++p) {
                flags |= *reinterpret_cast<const volatile unsigned *>(
                    reinterpret_cast<const char *>(slots[p]) + 4);
            }
            __threadfence_system();
            if (flags != 0) {
                atomicOr(status, kStatusOverflow);
            }
            proceed = 1;
        } else {
            atomicOr(status, kStatusTimeout);
        }
    }
    __syncthreads();
    if (proceed == 0) {
        return;
    }
    const void *slots[kMaxPeers] = {slot0, slot1, slot2};
    for (int i = (int)(blockIdx.x * kThreads + threadIdx.x); i < count;
         i += kThreads * gridDim.x) {
        float sum = own[i];
        for (int p = 0; p < peer_count; ++p) {
            const Payload *payload = reinterpret_cast<const Payload *>(
                reinterpret_cast<const char *>(slots[p]) + kHeaderBytes);
            if constexpr (sizeof(Payload) == 2) {
                sum += __half2float(payload[i]);
            } else {
                sum += payload[i];
            }
        }
        out[i] = sum;
    }
}

__global__ void wait_copy_kernel(float *out, const void *slot, int count,
                                 unsigned seq,
                                 unsigned long long timeout_cycles,
                                 unsigned *status) {
    __shared__ int proceed;
    if (threadIdx.x == 0) {
        proceed = 0;
        if (wait_for_slots(slot, nullptr, nullptr, 1, seq, timeout_cycles)) {
            proceed = 1;
        } else {
            atomicOr(status, kStatusTimeout);
        }
    }
    __syncthreads();
    if (proceed == 0) {
        return;
    }
    const float *payload = reinterpret_cast<const float *>(
        reinterpret_cast<const char *>(slot) + kHeaderBytes);
    for (int i = (int)(blockIdx.x * kThreads + threadIdx.x); i < count;
         i += kThreads * gridDim.x) {
        out[i] = payload[i];
    }
}

bool publish_args_ok(const float *src, const void *dst_slot, int count,
                     unsigned seq) {
    return src != nullptr && dst_slot != nullptr && aligned16(src) &&
           aligned16(dst_slot) && count > 0 && count <= kMaxValues &&
           seq != 0;
}

bool wait_args_ok(const float *out, const float *own, const void *s0,
                  const void *s1, const void *s2, int peer_count, int count,
                  unsigned seq, const unsigned *status) {
    if (out == nullptr || own == nullptr || status == nullptr ||
        !aligned16(out) || !aligned16(own) || peer_count < 0 ||
        peer_count > kMaxPeers || count <= 0 || count > kMaxValues ||
        seq == 0) {
        return false;
    }
    const void *slots[kMaxPeers] = {s0, s1, s2};
    for (int p = 0; p < peer_count; ++p) {
        if (slots[p] == nullptr || !aligned16(slots[p])) {
            return false;
        }
    }
    return true;
}

cudaStream_t as_stream(void *stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

/// One block covers this many values before the multi-block form pays
/// for its two extra launches (the eager step's payloads all sit below).
constexpr int kSingleBlockValues = fifi_ring::kValuesPerBlock;
static_assert(kSingleBlockValues == kThreads * 8);

__global__ void publish_f16_small_kernel(const float *src, void *dst_slot,
                                         int count, unsigned seq) {
    __shared__ unsigned overflow;
    if (threadIdx.x == 0) {
        overflow = 0;
    }
    __syncthreads();
    half *dst = reinterpret_cast<half *>(payload_of(dst_slot));
    unsigned local = 0;
    for (int i = (int)threadIdx.x; i < count; i += kThreads) {
        const float v = src[i];
        if (!(fabsf(v) <= 65504.0f)) {
            local = kFlagOverflow;
        }
        dst[i] = __float2half_rn(v);
    }
    if (local != 0) {
        atomicOr(&overflow, local);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *flags_word(dst_slot) = overflow;
        seq_publish(dst_slot, seq);
    }
}

__global__ void publish_f32_small_kernel(const float *src, void *dst_slot,
                                         int count, unsigned seq) {
    float *dst = reinterpret_cast<float *>(payload_of(dst_slot));
    for (int i = (int)threadIdx.x; i < count; i += kThreads) {
        dst[i] = src[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *flags_word(dst_slot) = 0;
        seq_publish(dst_slot, seq);
    }
}

int payload_blocks(int count) {
    if (count <= kSingleBlockValues) {
        return 1;
    }
    // Opt in before process start; the disabled path retains one-block launches.
    static const bool enabled = fifi_ring::multiblock_enabled(
        std::getenv("INFERENCE_CUDA_P2P_RING_MULTIBLOCK"));
    return fifi_ring::payload_blocks(count, enabled);
}

} // namespace

extern "C" int inference_cuda_ring_publish_f16(void *stream,
                                               const float *src,
                                               void *dst_slot, int count,
                                               unsigned seq) {
    try {
        if (!publish_args_ok(src, dst_slot, count, seq) || count > kMaxF16Values) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        cudaStream_t s = as_stream(stream);
        // The destination is peer memory. Keep overflow aggregation in one
        // local block: remote atomics are not available on every P2P link.
        publish_f16_small_kernel<<<1, kThreads, 0, s>>>(src, dst_slot,
                                                        count, seq);
        return finish(cudaGetLastError());
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_ring_publish_f32(void *stream,
                                               const float *src,
                                               void *dst_slot, int count,
                                               unsigned seq) {
    try {
        if (!publish_args_ok(src, dst_slot, count, seq)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        cudaStream_t s = as_stream(stream);
        const int blocks = payload_blocks(count);
        if (blocks == 1) {
            publish_f32_small_kernel<<<1, kThreads, 0, s>>>(src, dst_slot,
                                                            count, seq);
        } else {
            publish_f32_kernel<<<blocks, kThreads, 0, s>>>(src, dst_slot,
                                                           count);
            publish_commit_kernel<<<1, 1, 0, s>>>(dst_slot, seq);
        }
        return finish(cudaGetLastError());
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_ring_wait_sum_f16(
    void *stream, float *out, const float *own, const void *peer_slot0,
    const void *peer_slot1, const void *peer_slot2, int peer_count,
    int count, unsigned seq, unsigned long long timeout_cycles,
    unsigned *status) {
    try {
        if (!wait_args_ok(out, own, peer_slot0, peer_slot1, peer_slot2,
                          peer_count, count, seq, status) || count > kMaxF16Values) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        wait_sum_kernel<half><<<payload_blocks(count), kThreads, 0, as_stream(stream)>>>(
            out, own, peer_slot0, peer_slot1, peer_slot2, peer_count, count,
            seq, timeout_cycles, status);
        return finish(cudaGetLastError());
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_ring_wait_copy_f32(
    void *stream, float *out, const void *peer_slot, int count, unsigned seq,
    unsigned long long timeout_cycles, unsigned *status) {
    try {
        if (out == nullptr || peer_slot == nullptr || status == nullptr ||
            !aligned16(out) || !aligned16(peer_slot) || count <= 0 ||
            count > kMaxValues || seq == 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        wait_copy_kernel<<<payload_blocks(count), kThreads, 0, as_stream(stream)>>>(
            out, peer_slot, count, seq, timeout_cycles, status);
        return finish(cudaGetLastError());
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_ring_wait_sum_f32(
    void *stream, float *out, const float *own, const void *peer_slot0,
    const void *peer_slot1, const void *peer_slot2, int peer_count,
    int count, unsigned seq, unsigned long long timeout_cycles,
    unsigned *status) {
    try {
        if (!wait_args_ok(out, own, peer_slot0, peer_slot1, peer_slot2,
                          peer_count, count, seq, status)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        wait_sum_kernel<float><<<payload_blocks(count), kThreads, 0, as_stream(stream)>>>(
            out, own, peer_slot0, peer_slot1, peer_slot2, peer_count, count,
            seq, timeout_cycles, status);
        return finish(cudaGetLastError());
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
