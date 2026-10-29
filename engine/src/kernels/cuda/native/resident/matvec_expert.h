/*
 * Launch entry of the vendored MMVQ expert matvec: one launch for the used slots of an expert
 * stack, the per-slot expert ids read on the DEVICE when the kernel runs, through
 * fifi_mmvq_launch of the trimmed vendored mmvq.cu (ids path, the id range guard inside).
 * Launch-only: nothing here allocates, copies or synchronizes. Per-kernel header of the layout
 * in kernels/cuda/resident/mod.rs; abi_resident.h stays closed.
 */
#ifndef INFERENCE_CUDA_RESIDENT_MATVEC_EXPERT_H
#define INFERENCE_CUDA_RESIDENT_MATVEC_EXPERT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_matvec_expert_args {
    int type;         /* INFERENCE_CUDA_TYPE_Q4_K, Q5_0, Q6_K or Q8_0 */
    const void *w_stack; /* native blocks [n_expert][rows][k / qk] */
    const void *ids;  /* device i32 [n_used], read when the kernel runs */
    const void *x_q8; /* block_q8_1: one shared padded row (x_per_slot 0) or n_used rows */
    int x_per_slot;   /* 0: one shared vector (gate/up), 1: one per slot (down) */
    float *y;         /* f32 out [n_used][y_slot_stride], rows used per slot */
    int y_slot_stride;/* elements between slots, >= rows (the step's ExpertGu is [slot][gate|up]) */
    int k;            /* a multiple of the type's block size */
    int rows;         /* a multiple of 4 (the kernel reads whole row groups) */
    int n_expert;
    int n_used;       /* 1..8 */
} inference_cuda_matvec_expert_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_matvec_expert_args) == 72 && offsetof(inference_cuda_matvec_expert_args, type) == 0 && offsetof(inference_cuda_matvec_expert_args, w_stack) == 8 && offsetof(inference_cuda_matvec_expert_args, ids) == 16 && offsetof(inference_cuda_matvec_expert_args, x_q8) == 24 && offsetof(inference_cuda_matvec_expert_args, x_per_slot) == 32 && offsetof(inference_cuda_matvec_expert_args, y) == 40 && offsetof(inference_cuda_matvec_expert_args, y_slot_stride) == 48 && offsetof(inference_cuda_matvec_expert_args, k) == 52 && offsetof(inference_cuda_matvec_expert_args, rows) == 56 && offsetof(inference_cuda_matvec_expert_args, n_expert) == 60 && offsetof(inference_cuda_matvec_expert_args, n_used) == 64,
              "inference_cuda_matvec_expert_args layout");
#else
_Static_assert(sizeof(inference_cuda_matvec_expert_args) == 72 && offsetof(inference_cuda_matvec_expert_args, type) == 0 && offsetof(inference_cuda_matvec_expert_args, w_stack) == 8 && offsetof(inference_cuda_matvec_expert_args, ids) == 16 && offsetof(inference_cuda_matvec_expert_args, x_q8) == 24 && offsetof(inference_cuda_matvec_expert_args, x_per_slot) == 32 && offsetof(inference_cuda_matvec_expert_args, y) == 40 && offsetof(inference_cuda_matvec_expert_args, y_slot_stride) == 48 && offsetof(inference_cuda_matvec_expert_args, k) == 52 && offsetof(inference_cuda_matvec_expert_args, rows) == 56 && offsetof(inference_cuda_matvec_expert_args, n_expert) == 60 && offsetof(inference_cuda_matvec_expert_args, n_used) == 64,
               "inference_cuda_matvec_expert_args layout");
#endif

/* One launch on stream (must not be NULL): slot s computes y[s * y_slot_stride .. s *
 * y_slot_stride + rows) against expert ids[s]'s weights and x_q8 row (s if x_per_slot else 0).
 * The ids are read on the device at execution time, so a captured graph replay follows a
 * rewritten ids word; an id outside [0, n_expert) (the router writes -1 on a fault) makes the
 * kernel write the quiet NaN 0x7fc00000 to that slot's rows and read no weight (the id range
 * guard). Returns INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call for a bad argument,
 * else the launch status. */
int inference_cuda_matvec_expert_launch(const inference_cuda_matvec_expert_args *args,
                                        void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_MATVEC_EXPERT_H */
