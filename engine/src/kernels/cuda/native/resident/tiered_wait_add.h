/*
 * Launch entry of the tiered wait-add of one layer (native/resident/tiered_wait_add.h, the
 * per-kernel header of the layout in kernels/cuda/resident/mod.rs; abi_resident.h stays closed).
 * Launch-only: nothing here allocates, copies or synchronizes; a fault in queued work surfaces at the
 * next synchronize. Include after abi.h in C++. Rust mirrors the struct in
 * native/resident/tiered_wait_add.rs. This is the device half of the CPU-expert answer; the
 * publisher of the request is tiered_classify and the host half of the protocol is
 * context/mailbox.rs. The arithmetic is the second moe_combine of the v1 tiered
 * block (m 1, top_k 1, weight 1.0), so a served layer is bitwise the v1 step.
 *
 * One block of 256 threads. Thread 0 reads ctrl[0]: when it is 0 (no miss) the kernel writes NOTHING,
 * not the row and not the word, and every thread returns after one barrier. Otherwise thread 0 reads
 * expect = ctrl[1] and waits for the server's answer: it polls the answer line's done word with
 * ld.acquire.sys.u64 (inline PTX, volatile, a "memory" clobber) until done == expect (an older answer,
 * done < expect, is stale and never accepted; an answer ahead, done > expect, is a protocol fault: bit 1 and
 * the NaN row, as for a server error), sleeping between polls with __nanosleep (100 ns first,
 * doubling up to 2000 ns) and giving up when %globaltimer has advanced more than timeout_ns since the
 * first poll. After the wait it reads code (byte 8 of the answer line) with ld.relaxed.sys.u32. The
 * shared state after the barrier: 1 ok (code == 0), 2 server error (code != 0), 3 timeout; then
 *   ok           -> every element j of the row: resid[j] = __fadd_rn(resid[j], __fmul_rn(1.0f, p)),
 *                   p = ld.relaxed.sys.f32 of partial[j] (one scalar 32-bit load per element, never a
 *                   plain and never a vector load; 1.0 x p is exact)
 *   server error -> thread 0 ORs bit 1 (CPU_FAILED) into word, every thread stores the quiet NaN
 *   timeout      -> thread 0 ORs bit 2 (WAIT_TIMEOUT), every thread stores the quiet NaN
 * The quiet NaN (0x7FC00000) over the row makes the next layer's router fault, so a token sees at most
 * one wait timeout; that is why the row is poisoned. The kernel never reads done after the first time
 * it is satisfied and never writes the mailbox.
 *
 *   ctrl          64 bytes, 8-byte aligned: u64 [0] miss count, [1] expected sequence; ordinary device
 *                 memory written earlier on the same stream by tiered_classify
 *   word          i32, 4-byte aligned: the tiered word; this kernel only ORs bits into it
 *   mbox_ans      128 bytes, 8-byte aligned: the answer line of the mailbox, mapped host memory
 *                 (done u64 at byte 0, written by the host with a release store; code u32 at byte 8)
 *   mbox_partial  f32 [hidden], 16-byte aligned: the CPU experts' partial sum, mapped host memory
 *   resid         f32 [hidden], 16-byte aligned, device memory: the residual row, read and written in
 *                 place
 *   hidden        a multiple of 4 from 4 to 65536
 *   timeout_ns    1 to 120,000,000,000: the wait budget; %globaltimer counts nanoseconds
 * Every written range (resid, word) shares no byte with any other range; the read ranges may alias one
 * another. A rejected argument returns INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call; otherwise
 * the launch status of the kernel (one launch).
 */
#ifndef INFERENCE_CUDA_RESIDENT_TIERED_WAIT_ADD_H
#define INFERENCE_CUDA_RESIDENT_TIERED_WAIT_ADD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_tiered_wait_add_args {
    const uint64_t *ctrl;
    int32_t *word;
    const void *mbox_ans;
    const void *mbox_partial;
    float *resid;
    int hidden;
    uint64_t timeout_ns;
} inference_cuda_tiered_wait_add_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_tiered_wait_add_args) == 56 &&
                  offsetof(inference_cuda_tiered_wait_add_args, ctrl) == 0 &&
                  offsetof(inference_cuda_tiered_wait_add_args, word) == 8 &&
                  offsetof(inference_cuda_tiered_wait_add_args, mbox_ans) == 16 &&
                  offsetof(inference_cuda_tiered_wait_add_args, mbox_partial) == 24 &&
                  offsetof(inference_cuda_tiered_wait_add_args, resid) == 32 &&
                  offsetof(inference_cuda_tiered_wait_add_args, hidden) == 40 &&
                  offsetof(inference_cuda_tiered_wait_add_args, timeout_ns) == 48,
              "inference_cuda_tiered_wait_add_args layout");
#else
_Static_assert(sizeof(inference_cuda_tiered_wait_add_args) == 56 &&
                   offsetof(inference_cuda_tiered_wait_add_args, ctrl) == 0 &&
                   offsetof(inference_cuda_tiered_wait_add_args, word) == 8 &&
                   offsetof(inference_cuda_tiered_wait_add_args, mbox_ans) == 16 &&
                   offsetof(inference_cuda_tiered_wait_add_args, mbox_partial) == 24 &&
                   offsetof(inference_cuda_tiered_wait_add_args, resid) == 32 &&
                   offsetof(inference_cuda_tiered_wait_add_args, hidden) == 40 &&
                   offsetof(inference_cuda_tiered_wait_add_args, timeout_ns) == 48,
               "inference_cuda_tiered_wait_add_args layout");
#endif

int inference_cuda_tiered_wait_add_launch(const inference_cuda_tiered_wait_add_args *args, void *stream,
                                          int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_TIERED_WAIT_ADD_H */
