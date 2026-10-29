/*
 * The tiered wait-add of one layer, the device half of the CPU-expert answer. Semantics: see
 * tiered_wait_add.h; the host half of the protocol is context/mailbox.rs and its port. Original
 * CUDA C.
 *
 * One block, 256 threads. Thread 0 alone waits on the mapped answer line with a bounded poll, so the
 * block occupies its SM but never the host: the decode stream keeps its enqueue-only contract. The
 * wait reads the done word with an ACQUIRE at system scope, so the partial and the code that the
 * server wrote before it released done are visible to the loads that follow; the partial is then read
 * element by element with RELAXED system-scope loads (each 32-bit load is atomic on its own; the
 * acquire of done orders them), never with a plain or a vector load, because the host writes that
 * memory while the CPU stores into it. The sum keeps the v1 arithmetic: __fmul_rn then __fadd_rn, one
 * intrinsic each, so neither the release compiler nor the debug device build can contract them and
 * the bits equal the second moe_combine of the v1 tiered block.
 */
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/tiered_wait_add.h"

namespace {

constexpr int kThreads = 256;
constexpr int kMaxHidden = 65536;
constexpr uint64_t kMaxTimeoutNs = 120000000000ull;
constexpr size_t kCtrlBytes = 64;
constexpr size_t kAnswerBytes = 128;
/* Shared state of the block: no miss (write nothing), ok, server error, timeout. */
constexpr int kNoMiss = 0;
constexpr int kOk = 1;
constexpr int kServerError = 2;
constexpr int kTimeout = 3;
/* The poll sleeps 100 ns first, doubling up to 2000 ns. */
constexpr unsigned kFirstSleepNs = 100;
constexpr unsigned kMaxSleepNs = 2000;

__device__ __forceinline__ uint64_t globaltimer_ns() {
    uint64_t now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now) : : "memory");
    return now;
}

/* Thread 0's wait for the server's answer: polls done at system scope until it equals expect (an older
 * answer is stale and keeps the wait going; an answer AHEAD of expect, done > expect, is a protocol fault
 * and returns the server-error state at once), sleeping between polls, and gives up once %globaltimer has
 * advanced more than timeout_ns since the first poll. A satisfied wait reads the code and returns ok
 * (code 0) or server error; done is never read again after the first time it is satisfied. */
__device__ __forceinline__ int wait_for_answer(const uint64_t *ans, uint64_t expect, uint64_t timeout_ns) {
    const uint64_t started = globaltimer_ns();
    unsigned sleep_ns = kFirstSleepNs;
    bool answered = false;
    while (!answered) {
        uint64_t done;
        asm volatile("ld.acquire.sys.u64 %0, [%1];" : "=l"(done) : "l"(ans) : "memory");
        if (done > expect) {
            return kServerError;
        }
        answered = done == expect;
        if (!answered) {
            if (globaltimer_ns() - started > timeout_ns) {
                return kTimeout;
            }
            __nanosleep(sleep_ns);
            sleep_ns = sleep_ns * 2 > kMaxSleepNs ? kMaxSleepNs : sleep_ns * 2;
        }
    }
    uint32_t code;
    asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(code) : "l"(ans + 1) : "memory");
    return code == 0 ? kOk : kServerError;
}

__global__ void __launch_bounds__(kThreads) tiered_wait_add_k(const inference_cuda_tiered_wait_add_args a) {
    __shared__ int s_state;
    int *const word = a.word;
    if (threadIdx.x == 0) {
        if (a.ctrl[0] == 0) {
            s_state = kNoMiss; /* no miss: nothing is written, not even the word */
        } else {
            s_state = wait_for_answer(static_cast<const uint64_t *>(a.mbox_ans), a.ctrl[1], a.timeout_ns);
        }
    }
    __syncthreads();
    const int state = s_state;
    if (state == kNoMiss) {
        return;
    }
    if (state == kOk) {
        const float *const partial = static_cast<const float *>(a.mbox_partial);
        float *const out = a.resid;
        for (int j = threadIdx.x; j < a.hidden; j += kThreads) {
            float p;
            asm volatile("ld.relaxed.sys.f32 %0, [%1];" : "=f"(p) : "l"(partial + j) : "memory");
            out[j] = __fadd_rn(out[j], __fmul_rn(1.0f, p));
        }
        return;
    }
    if (threadIdx.x == 0) {
        if (state == kServerError) {
            atomicOr(word, 1); /* the CPU_FAILED bit of the tiered word */
        } else {
            atomicOr(word, 2); /* the WAIT_TIMEOUT bit of the tiered word */
        }
    }
    const float quiet_nan = __uint_as_float(0x7FC00000);
    for (int j = threadIdx.x; j < a.hidden; j += kThreads) {
        a.resid[j] = quiet_nan;
    }
}

bool aligned_to(const void *p, size_t n) { return ((uintptr_t)p % n) == 0; }

bool apart(const void *a, size_t abytes, const void *b, size_t bbytes) {
    const uintptr_t pa = (uintptr_t)a, pb = (uintptr_t)b;
    return pa + abytes <= pb || pb + bbytes <= pa;
}

bool shapes_ok(const inference_cuda_tiered_wait_add_args *a) {
    const bool wide = a->hidden >= 4 && a->hidden <= kMaxHidden && a->hidden % 4 == 0;
    const bool bounded = a->timeout_ns >= 1 && a->timeout_ns <= kMaxTimeoutNs;
    return wide && bounded;
}

bool pointers_ok(const inference_cuda_tiered_wait_add_args *a) {
    const bool present =
        a->ctrl != NULL && a->word != NULL && a->mbox_ans != NULL && a->mbox_partial != NULL && a->resid != NULL;
    const bool narrow = aligned_to(a->ctrl, 8) && aligned_to(a->word, 4) && aligned_to(a->mbox_ans, 8);
    const bool wide = aligned_to(a->mbox_partial, 16) && aligned_to(a->resid, 16);
    return present && narrow && wide;
}

struct Region {
    const void *p;
    size_t bytes;
    bool written;
};

/* Every written range (resid, word) shares no byte with any other range; the read ranges may alias. */
bool overlaps_ok(const inference_cuda_tiered_wait_add_args *a) {
    const size_t row = (size_t)a->hidden * 4;
    const Region regions[] = {
        {a->ctrl, kCtrlBytes, false},       {a->word, 4, true},          {a->mbox_ans, kAnswerBytes, false},
        {a->mbox_partial, row, false},      {a->resid, row, true},
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

/* Every term of this test has a twin in the Rust wrapper (check_wait_add in tiered_wait_add.rs). */
bool valid(const inference_cuda_tiered_wait_add_args *a, void *stream) {
    return a != NULL && stream != NULL && shapes_ok(a) && pointers_ok(a) && overlaps_ok(a);
}

}  // namespace

extern "C" int inference_cuda_tiered_wait_add_launch(const inference_cuda_tiered_wait_add_args *a, void *stream,
                                                     int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid(a, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        tiered_wait_add_k<<<1, kThreads, 0, (cudaStream_t)stream>>>(*a);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
