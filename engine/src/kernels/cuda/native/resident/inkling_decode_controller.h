/* Fused controller chain of one Inkling 2 decode visit (FIFI_INKLING_CUDA_DECODE_GLUE), per-kernel
 * header (native/resident/inkling_decode_controller.h). Included by inkling_decode_controller.cu
 * and read by the Rust FFI mirror; nothing else.
 *
 * A controller role of width W and rank R reads two gate halves `input[W]` and `message[W]`,
 * normalizes each (`rms_norm_with_scale` with `gate_eps`, the bits of inkling_stream_norm_rows)
 * into `gate[2W] = [input / rms | message / rms]`, reduces `gate` through the down matrix
 * `down[R][2W]` into `reduced[R]` (the bits of decode_controller_down_exact), projects `reduced`
 * through `gain_w[W][R]` and `forget_w[W][R]` into `gain[W]` and `forget[W]` (the bits of
 * decode_control_pair; R is 64), and writes
 * `out = rmsnorm((input - tanh(forget) * input) + tanh(gain) * message) * gamma` with `eps`
 * (the bits of inkling_controller_norm_rows).
 *
 * inference_cuda_decode_controller_gate_down runs the gate norms and the down reduction of every
 * role: grid (ceil(R / 8), roles), 256 threads; every block of a role normalizes the halves
 * itself (the same bits in each) and one warp reduces one row. The health word of a half takes a
 * quiet NaN when a value of the gate was not finite, and is left alone otherwise. The kernel also
 * clears the role's ticket, which the pair launch of the same role counts with.
 *
 * inference_cuda_decode_controller_pair_norm runs both projections of every role: grid (W / 16,
 * 2, roles), 256 threads; the last block of a role to finish then applies the controller norm of
 * that role, over the gain and forget rows every block wrote, and clears the ticket again.
 * `health[role]` takes a quiet NaN when a value was not finite and is zeroed otherwise, as
 * inkling_controller_norm_rows does.
 *
 * `tickets` holds one u32 per role and shares no byte with any row. Limits: roles in [1, 4],
 * W a multiple of 16 in [16, 4096], R a multiple of 8 in [8, 1024] for the gate launch and
 * R == 64 for the pair launch. Every row is 16-byte aligned. The written rows (`gate`, `reduced`,
 * `gain`, `forget`, `out`, the health and ticket words) share no byte with each other or with a
 * read row. Each entry takes the stream explicitly, returns INFERENCE_CUDA_INVALID_ARGUMENT for a
 * bad argument before any CUDA call, never falls back to the default stream and does not
 * synchronize. */
#ifndef INFERENCE_CUDA_RESIDENT_INKLING_DECODE_CONTROLLER_H
#define INFERENCE_CUDA_RESIDENT_INKLING_DECODE_CONTROLLER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INFERENCE_CUDA_CONTROLLER_MAX_ROLES 4

typedef struct inference_cuda_controller_gate_role {
    const float *input;
    const float *message;
    float *gate;
    const float *down;
    float *reduced;
    /* Health of the input half, then of the message half. */
    float *health[2];
} inference_cuda_controller_gate_role;

typedef struct inference_cuda_controller_gate_args {
    inference_cuda_controller_gate_role role[INFERENCE_CUDA_CONTROLLER_MAX_ROLES];
    unsigned *tickets;
    int roles;
    int width;
    int rank;
    float gate_eps;
} inference_cuda_controller_gate_args;

typedef struct inference_cuda_controller_apply_role {
    const float *input;
    const float *message;
    const float *gamma;
    const float *reduced;
    const float *gain_w;
    const float *forget_w;
    float *gain;
    float *forget;
    float *out;
    float *health;
} inference_cuda_controller_apply_role;

typedef struct inference_cuda_controller_apply_args {
    inference_cuda_controller_apply_role role[INFERENCE_CUDA_CONTROLLER_MAX_ROLES];
    unsigned *tickets;
    int roles;
    int rank;
    int width;
    float eps;
} inference_cuda_controller_apply_args;

int inference_cuda_decode_controller_gate_down(const inference_cuda_controller_gate_args *args,
                                               void *stream, int *out_stage);

int inference_cuda_decode_controller_pair_norm(const inference_cuda_controller_apply_args *args,
                                               void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_INKLING_DECODE_CONTROLLER_H */
