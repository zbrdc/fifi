/*
 * The tiered classification of one layer. Semantics: see tiered_classify.h; the host oracle is
 * family/qwen36/tiered/classify.rs. Original CUDA C.
 *
 * One block, 256 threads. Thread 0 does the classification (top_k <= 8 picks, a handful of integer compares, in pick
 * order, so the result cannot depend on scheduling) into registers and shared memory; the block then copies the normed
 * row into the mailbox when there is a miss. The mailbox memory is written by plain stores and made visible to the
 * host server by a system-scope fence per writing thread, a block barrier, and a RELEASE store of the sequence number
 * into the request line: a server that acquires the sequence sees the row and the request.
 */
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/tiered_classify.h"

namespace {

constexpr int kThreads = 256;
constexpr int kMaxTop = 8;
constexpr int kMaxExpert = 4096;
constexpr int kMaxHidden = 65536;
constexpr int kMaxLayer = 1023;
constexpr int kRouteInvalid = 4;
constexpr size_t kLogBytes = 8;     /* per pick: i32 id, f32 weight bits */
constexpr size_t kCtrlBytes = 64;
constexpr size_t kRequestBytes = 128;

__device__ __forceinline__ void store_release_system(uint64_t *p, uint64_t v) {
    asm volatile("st.release.sys.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__global__ void __launch_bounds__(kThreads) tiered_classify_k(const inference_cuda_tiered_classify_args a) {
    __shared__ int s_count;
    __shared__ int s_miss_id[kMaxTop];
    __shared__ float s_miss_w[kMaxTop];
    if (threadIdx.x == 0) {
        int word = 0;
        int count = 0;
        for (int i = 0; i < a.top_k; ++i) {
            const int id = a.ids[i];
            const float w = a.weights[i];
            a.log_out[2 * i] = id;
            a.log_out[2 * i + 1] = __float_as_int(w);
            int slot_out = -1;
            float w_out = w;
            if (id != -1) {
                bool invalid = id < -1 || id >= a.n_expert || !isfinite(w);
                for (int j = 0; j < i && !invalid; ++j) {
                    invalid = a.ids[j] == id;
                }
                int slot = 0;
                if (!invalid) {
                    slot = a.slot_row[id];
                    invalid = slot < -1 || slot >= a.slots;
                }
                if (invalid) {
                    word |= kRouteInvalid;
                } else if (slot == -1) {
                    slot_out = 0;
                    w_out = 0.0f;
                    s_miss_id[count] = id;
                    s_miss_w[count] = w;
                    ++count;
                } else {
                    slot_out = slot;
                }
            }
            a.slot_ids_out[i] = slot_out;
            a.weights_out[i] = w_out;
        }
        if (a.layer == 0) {
            *a.word = word;
        } else if (word != 0) {
            *a.word |= word;
        }
        a.ctrl[0] = (uint64_t)count;
        s_count = count;
    }
    __syncthreads();
    const int count = s_count;
    if (count == 0) {
        return;
    }
    const float4 *src = reinterpret_cast<const float4 *>(a.norm_row);
    float4 *dst = reinterpret_cast<float4 *>(a.mbox_row);
    for (int j = threadIdx.x; j < a.hidden / 4; j += kThreads) {
        dst[j] = src[j];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        int32_t *ids_line = reinterpret_cast<int32_t *>(a.mbox_req) + 4;   /* byte 16 */
        float *weights_line = reinterpret_cast<float *>(a.mbox_req) + 12;  /* byte 48 */
        a.mbox_req[1] = ((uint64_t)(uint32_t)count << 32) | (uint64_t)(uint32_t)a.layer;
        for (int k = 0; k < kMaxTop; ++k) {
            ids_line[k] = k < count ? s_miss_id[k] : -1;
            weights_line[k] = k < count ? s_miss_w[k] : 0.0f;
        }
        __threadfence_system();
        const uint64_t seq = a.ctrl[2] + 1;
        a.ctrl[2] = seq;
        a.ctrl[1] = seq;
        store_release_system(a.mbox_req, seq);
    }
}

bool aligned_to(const void *p, size_t n) { return ((uintptr_t)p % n) == 0; }

bool apart(const void *a, size_t abytes, const void *b, size_t bbytes) {
    const uintptr_t pa = (uintptr_t)a, pb = (uintptr_t)b;
    return pa + abytes <= pb || pb + bbytes <= pa;
}

bool shapes_ok(const inference_cuda_tiered_classify_args *a) {
    const bool experts = a->n_expert >= 1 && a->n_expert <= kMaxExpert;
    const bool bank = a->slots >= 1 && a->slots <= a->n_expert;
    const bool top = a->top_k >= 1 && a->top_k <= kMaxTop;
    const bool wide = a->hidden >= 4 && a->hidden <= kMaxHidden && a->hidden % 4 == 0;
    const bool layer = a->layer >= 0 && a->layer <= kMaxLayer;
    return experts && bank && top && wide && layer;
}

bool pointers_ok(const inference_cuda_tiered_classify_args *a) {
    const bool present = a->ids != NULL && a->weights != NULL && a->slot_row != NULL && a->norm_row != NULL &&
                         a->slot_ids_out != NULL && a->weights_out != NULL && a->log_out != NULL && a->ctrl != NULL &&
                         a->word != NULL && a->mbox_req != NULL && a->mbox_row != NULL;
    const bool narrow = aligned_to(a->ids, 4) && aligned_to(a->weights, 4) && aligned_to(a->slot_row, 4) &&
                        aligned_to(a->slot_ids_out, 4) && aligned_to(a->weights_out, 4) && aligned_to(a->word, 4);
    const bool eight = aligned_to(a->log_out, 8) && aligned_to(a->ctrl, 8) && aligned_to(a->mbox_req, 8);
    const bool wide = aligned_to(a->norm_row, 16) && aligned_to(a->mbox_row, 16);
    return present && narrow && eight && wide;
}

struct Region {
    const void *p;
    size_t bytes;
    bool written;
};

/* Every written range shares no byte with any other range; the four inputs may alias one another. */
bool overlaps_ok(const inference_cuda_tiered_classify_args *a) {
    const size_t top = (size_t)a->top_k, hidden = (size_t)a->hidden * 4;
    const Region regions[] = {
        {a->ids, top * 4, false},          {a->weights, top * 4, false},      {a->slot_row, (size_t)a->n_expert * 4, false},
        {a->norm_row, hidden, false},      {a->slot_ids_out, top * 4, true},  {a->weights_out, top * 4, true},
        {a->log_out, top * kLogBytes, true}, {a->ctrl, kCtrlBytes, true},     {a->word, 4, true},
        {a->mbox_req, kRequestBytes, true}, {a->mbox_row, hidden, true},
    };
    const size_t n = sizeof(regions) / sizeof(regions[0]);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if ((regions[i].written || regions[j].written) &&
                !apart(regions[i].p, regions[i].bytes, regions[j].p, regions[j].bytes)) {
                return false;
            }
        }
    }
    return true;
}

/* Every term of this test has a twin in the Rust wrapper (check_classify in tiered_classify.rs). */
bool valid(const inference_cuda_tiered_classify_args *a, void *stream) {
    return a != NULL && stream != NULL && shapes_ok(a) && pointers_ok(a) && overlaps_ok(a);
}

}  // namespace

extern "C" int inference_cuda_tiered_classify_launch(const inference_cuda_tiered_classify_args *a, void *stream,
                                                     int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid(a, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        tiered_classify_k<<<1, kThreads, 0, (cudaStream_t)stream>>>(*a);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
