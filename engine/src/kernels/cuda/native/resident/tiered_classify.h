/*
 * Launch entry of the tiered classification of one layer (native/resident/tiered_classify.h,
 * the per-kernel header of the layout in kernels/cuda/resident/mod.rs; abi_resident.h stays closed).
 * Launch-only: nothing here allocates, copies or synchronizes; a fault in queued work surfaces at the
 * next synchronize. Include after abi.h in C++. Rust mirrors the struct in
 * native/resident/tiered_classify.rs. The host oracle is family/qwen36/tiered/classify.rs
 * (`classify_reference`), which this kernel must equal bit for bit.
 *
 * One block of 256 threads. Thread 0 walks the router's picks in rank order (id, weight) and writes, per pick i:
 *   log_out[i]       = (id, weight bits)                        the raw pick, always
 *   id == -1         -> slot_ids_out[i] = -1, weights_out[i] = w   the router fault passes through
 *   refused          -> slot_ids_out[i] = -1, weights_out[i] = w   and bit 4 (ROUTE_INVALID) in this layer's word:
 *                       id < -1, id >= n_expert, an id equal to an earlier pick, a non-finite w, or a
 *                       table slot below -1 or at or above `slots`
 *   resident         -> slot_ids_out[i] = slot_row[id], weights_out[i] = w
 *   miss (slot_row[id] == -1) -> slot_ids_out[i] = 0, weights_out[i] = +0.0f, and (id, w) is appended to the miss list
 * ctrl[0] = the number of misses, always. The tiered word: layer 0 stores this layer's bits (it starts the token),
 * later layers OR theirs in. When the layer has at least one miss: all threads copy norm_row into mbox_row (float4
 * stores), a system-scope fence and a block barrier follow, then thread 0 writes the request line (layer, count, the
 * miss ids and weights, unused entries -1 and 0.0f), fences, takes seq = ctrl[2] + 1, stores ctrl[2] = ctrl[1] = seq and
 * publishes seq with a release store at system scope into mbox_req[0]. With no miss the mailbox is not touched and the
 * sequence does not move.
 *
 *   ids          i32 [top_k], 4-byte aligned: the router's picks (rank order)
 *   weights      f32 [top_k], 4-byte aligned: the router's renormalized weights
 *   slot_row     i32 [n_expert], 4-byte aligned: the layer's row of the slot table (bank slot or -1)
 *   norm_row     f32 [hidden], 16-byte aligned: the normed row the CPU experts read
 *   slot_ids_out i32 [top_k], 4-byte aligned; weights_out f32 [top_k], 4-byte aligned
 *   log_out      8 * top_k bytes, 8-byte aligned: top_k pairs (i32 id, f32 weight bits)
 *   ctrl         64 bytes, 8-byte aligned: u64 [0] miss count, [1] expected sequence, [2] sequence counter
 *   word         i32, 4-byte aligned: the tiered word
 *   mbox_req     128 bytes, 8-byte aligned: u64 [0] seq (written last, release), u64 [1] layer | count << 32,
 *                i32 ids[8] at byte 16, f32 weights[8] at byte 48
 *   mbox_row     f32 [hidden], 16-byte aligned (mapped page-locked memory in production)
 * Every written range (slot_ids_out, weights_out, log_out, ctrl, word, mbox_req, mbox_row) is disjoint from every
 * other range; the four inputs may alias each other.
 * n_expert in [1, 4096], slots in [1, n_expert], top_k in [1, 8], hidden a multiple of 4 in [4, 65536], layer in
 * [0, 1023]. stream must not be NULL. A rejected argument returns INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA
 * call; otherwise the launch status of the kernel (one launch).
 */
#ifndef INFERENCE_CUDA_RESIDENT_TIERED_CLASSIFY_H
#define INFERENCE_CUDA_RESIDENT_TIERED_CLASSIFY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_tiered_classify_args {
    const int32_t *ids;
    const float *weights;
    const int32_t *slot_row;
    const float *norm_row;
    int32_t *slot_ids_out;
    float *weights_out;
    int32_t *log_out;
    uint64_t *ctrl;
    int32_t *word;
    uint64_t *mbox_req;
    float *mbox_row;
    int n_expert;
    int slots;
    int top_k;
    int hidden;
    int layer;
} inference_cuda_tiered_classify_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_tiered_classify_args) == 112 && offsetof(inference_cuda_tiered_classify_args, ids) == 0 &&
                  offsetof(inference_cuda_tiered_classify_args, weights) == 8 &&
                  offsetof(inference_cuda_tiered_classify_args, slot_row) == 16 &&
                  offsetof(inference_cuda_tiered_classify_args, norm_row) == 24 &&
                  offsetof(inference_cuda_tiered_classify_args, slot_ids_out) == 32 &&
                  offsetof(inference_cuda_tiered_classify_args, weights_out) == 40 &&
                  offsetof(inference_cuda_tiered_classify_args, log_out) == 48 &&
                  offsetof(inference_cuda_tiered_classify_args, ctrl) == 56 &&
                  offsetof(inference_cuda_tiered_classify_args, word) == 64 &&
                  offsetof(inference_cuda_tiered_classify_args, mbox_req) == 72 &&
                  offsetof(inference_cuda_tiered_classify_args, mbox_row) == 80 &&
                  offsetof(inference_cuda_tiered_classify_args, n_expert) == 88 &&
                  offsetof(inference_cuda_tiered_classify_args, slots) == 92 &&
                  offsetof(inference_cuda_tiered_classify_args, top_k) == 96 &&
                  offsetof(inference_cuda_tiered_classify_args, hidden) == 100 &&
                  offsetof(inference_cuda_tiered_classify_args, layer) == 104,
              "inference_cuda_tiered_classify_args layout");
#else
_Static_assert(sizeof(inference_cuda_tiered_classify_args) == 112 && offsetof(inference_cuda_tiered_classify_args, ids) == 0 &&
                   offsetof(inference_cuda_tiered_classify_args, weights) == 8 &&
                   offsetof(inference_cuda_tiered_classify_args, slot_row) == 16 &&
                   offsetof(inference_cuda_tiered_classify_args, norm_row) == 24 &&
                   offsetof(inference_cuda_tiered_classify_args, slot_ids_out) == 32 &&
                   offsetof(inference_cuda_tiered_classify_args, weights_out) == 40 &&
                   offsetof(inference_cuda_tiered_classify_args, log_out) == 48 &&
                   offsetof(inference_cuda_tiered_classify_args, ctrl) == 56 &&
                   offsetof(inference_cuda_tiered_classify_args, word) == 64 &&
                   offsetof(inference_cuda_tiered_classify_args, mbox_req) == 72 &&
                   offsetof(inference_cuda_tiered_classify_args, mbox_row) == 80 &&
                   offsetof(inference_cuda_tiered_classify_args, n_expert) == 88 &&
                   offsetof(inference_cuda_tiered_classify_args, slots) == 92 &&
                   offsetof(inference_cuda_tiered_classify_args, top_k) == 96 &&
                   offsetof(inference_cuda_tiered_classify_args, hidden) == 100 &&
                   offsetof(inference_cuda_tiered_classify_args, layer) == 104,
               "inference_cuda_tiered_classify_args layout");
#endif

int inference_cuda_tiered_classify_launch(const inference_cuda_tiered_classify_args *args, void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_TIERED_CLASSIFY_H */
