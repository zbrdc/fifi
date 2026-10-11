/* RD-4d: the per-step parameter block of the resident decode step, and the kernels that begin
 * a step, choose the greedy token and advance the position. One header per kernel file
 * (lane C ruling on O2); abi_resident.h stays closed.
 *
 * inference_cuda_step_params is the device image of StepParams: 32 bytes, 256-byte aligned in
 * the arena (the kernels need only 4-byte alignment). Other kernels take a 4-byte DeviceRange
 * at one field offset, not the struct:
 *   token_id  offset  0  read by the embedding gather; written by argmax (token chaining)
 *   pos       offset  4  read by attn_prep (rope angle)
 *   kv_len    offset  8  read by attn_decode; kv_len == pos + 1 once the step has appended
 *   kv_slot   offset 12  reserved; equals pos (REPLY-44)
 *   status    offset 16  flags by mask: 1 router fault (RD-5); 2 non-finite logits row (RD-4d);
 *                        4 invalid step state (RD-4d, step_begin; a defect when it appears, REPLY-47)
 * A valid state is 0 <= pos <= 2^30 - 2, kv_len == pos + 1, kv_slot == pos.
 * token_id and status stay the device source of truth; the result pair of argmax is a copy.
 *
 * All three launch entries are launch-only and enqueue on stream (NULL: the default stream).
 * Each returns INFERENCE_CUDA_INVALID_ARGUMENT before any launch for a bad argument.
 */
#ifndef INFERENCE_CUDA_RESIDENT_STEP_PARAMS_H
#define INFERENCE_CUDA_RESIDENT_STEP_PARAMS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INFERENCE_CUDA_STATUS_ROUTER_FAULT 1
#define INFERENCE_CUDA_STATUS_NONFINITE_LOGITS 2
#define INFERENCE_CUDA_STATUS_INVALID_STEP_STATE 4

typedef struct inference_cuda_step_params {
    int token_id;
    int pos;
    int kv_len;
    int kv_slot;
    int status;
    int pad[3];
} inference_cuda_step_params;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_step_params) == 32, "StepParams is 32 bytes");
#else
_Static_assert(sizeof(inference_cuda_step_params) == 32, "StepParams is 32 bytes");
#endif

/* Clears params->status: the status is zeroed at each step start (REPLY-44), then mask 4
 * (invalid step state) is set when the state is not valid (see above). First kernel of a step; the bits of the step
 * before were published to the host (result pair of argmax) at its end. One launch. */
int inference_cuda_step_begin_launch(void *params, void *stream, int *out_stage);

/* Last kernel of a step: pos += m, kv_len += m, and kv_slot += m (kv_slot is reserved and
 * equals pos, REPLY-44) when the state is valid and pos + m <= 2^30 - 2, so kv_len == pos + 1
 * holds after every advance. Otherwise all three words become -1 (never clamped to row 0) and
 * the next step_begin sets mask 4. token_id and status are never touched (argmax is the last
 * kernel that writes status). m in [1, 4096]. One launch. */
typedef struct inference_cuda_step_advance_args {
    void *params;
    int m;
} inference_cuda_step_advance_args;
int inference_cuda_step_advance_launch(const inference_cuda_step_advance_args *args, void *stream,
                                       int *out_stage);
/* The Rust mirror (native/resident/step_params.rs) pins the same size and offsets with const asserts. */
#ifdef __cplusplus
static_assert(sizeof(inference_cuda_step_advance_args) == 16 && offsetof(inference_cuda_step_advance_args, params) == 0 && offsetof(inference_cuda_step_advance_args, m) == 8,
              "inference_cuda_step_advance_args layout");
#else
_Static_assert(sizeof(inference_cuda_step_advance_args) == 16 && offsetof(inference_cuda_step_advance_args, params) == 0 && offsetof(inference_cuda_step_advance_args, m) == 8,
               "inference_cuda_step_advance_args layout");
#endif

/* Greedy argmax over one logits row of n f32 values.
 *   token rule: the largest value, the lowest index on a tie; a NaN never wins; +infinity wins
 *     like any value; -infinity is the smallest value (an all -infinity row gives index 0, as
 *     the host argmax of family/sampler.rs does); a row with no comparable value (all NaN)
 *     gives -1.
 *   status rule: mask 2 is OR-ed into params->status when ANY of the n values is not finite
 *     (NaN, +infinity or -infinity). This is the predicate of check_logits in sampler.rs,
 *     which fails the host sampler on such a row. The token of such a row is defined above
 *     but the host must not use it.
 *   token chaining: params->token_id = the token (also -1), so that the next replay of the
 *     step graph embeds it. No other word of params is written.
 *   result (may be NULL, else 8-byte aligned): two ints {token, status}, where status is
 *     params->status after the OR. One 8-byte device-to-host copy gives the host both words. It
 *     is a copy: params->token_id and params->status stay the source of truth. It equals the
 *     final status only if no kernel writes status after argmax.
 * part is scratch of 3072 bytes (ARGMAX_PART_BYTES), 4-byte aligned, needs no initialization.
 * n in [1, 2^30]. Enqueues two kernels (256 blocks of 256 threads, then one block). */
typedef struct inference_cuda_argmax_args {
    const void *logits;
    void *part;
    void *params;
    void *result;
    int n;
} inference_cuda_argmax_args;
int inference_cuda_argmax_launch(const inference_cuda_argmax_args *args, void *stream,
                                 int *out_stage);
#ifdef __cplusplus
static_assert(sizeof(inference_cuda_argmax_args) == 40 && offsetof(inference_cuda_argmax_args, logits) == 0 && offsetof(inference_cuda_argmax_args, part) == 8 && offsetof(inference_cuda_argmax_args, params) == 16 && offsetof(inference_cuda_argmax_args, result) == 24 && offsetof(inference_cuda_argmax_args, n) == 32,
              "inference_cuda_argmax_args layout");
#else
_Static_assert(sizeof(inference_cuda_argmax_args) == 40 && offsetof(inference_cuda_argmax_args, logits) == 0 && offsetof(inference_cuda_argmax_args, part) == 8 && offsetof(inference_cuda_argmax_args, params) == 16 && offsetof(inference_cuda_argmax_args, result) == 24 && offsetof(inference_cuda_argmax_args, n) == 32,
               "inference_cuda_argmax_args layout");
#endif

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_STEP_PARAMS_H */
