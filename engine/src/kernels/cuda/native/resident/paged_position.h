/* Derive a page-local append position from authoritative device StepParams.
 * params: 32 bytes, local: 4 bytes; both 4-byte aligned, disjoint device ranges.
 * expected is a consistency assertion, never the source of the local position.
 * On inconsistent pos/kv_len/kv_slot/status, set status mask4 and local=-1.
 * Otherwise local=GPUpos-base; global fields and all other bytes stay unchanged.
 * Launch-only on an explicit non-null stream; invalid arguments enqueue nothing.
 */
#ifndef INFERENCE_CUDA_RESIDENT_PAGED_POSITION_H
#define INFERENCE_CUDA_RESIDENT_PAGED_POSITION_H
#include <stddef.h>
#include "resident/step_params.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct inference_cuda_paged_position_args {
    inference_cuda_step_params *params;
    int *local;
    int expected;
    int base;
    int cells;
    int capacity;
} inference_cuda_paged_position_args;
#ifdef __cplusplus
static_assert(sizeof(inference_cuda_paged_position_args) == 32 &&
              offsetof(inference_cuda_paged_position_args, params) == 0 &&
              offsetof(inference_cuda_paged_position_args, local) == 8 &&
              offsetof(inference_cuda_paged_position_args, expected) == 16 &&
              offsetof(inference_cuda_paged_position_args, base) == 20 &&
              offsetof(inference_cuda_paged_position_args, cells) == 24 &&
              offsetof(inference_cuda_paged_position_args, capacity) == 28,
              "paged_position ABI layout");
#else
_Static_assert(sizeof(inference_cuda_paged_position_args) == 32 &&
               offsetof(inference_cuda_paged_position_args, local) == 8 &&
               offsetof(inference_cuda_paged_position_args, capacity) == 28,
               "paged_position ABI layout");
#endif
int inference_cuda_paged_position_launch(const inference_cuda_paged_position_args *args,
                                        void *stream, int *out_stage);
#ifdef __cplusplus
}
#endif
#endif
