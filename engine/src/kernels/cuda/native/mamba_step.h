/* Mamba-2 decode-step kernels for the device chain (NEM35_DEVICE_CHAIN):
 * the conv1d step and the selective SSM update for ONE token row, both
 * operating entirely on device-resident state. The math mirrors
 * forward/mamba.rs exactly: sequential F32 accumulation, softplus with the
 * threshold-20 shortcut, the two-branch SiLU, and the lower dt clamp. */
#ifndef INFERENCE_CUDA_MAMBA_STEP_H
#define INFERENCE_CUDA_MAMBA_STEP_H

#include "abi.h"

extern "C" {

/* Conv1d decode step for conv_dim channels with a kernel of conv_kernel
 * taps. conv_state is [conv_dim, conv_kernel] device memory, updated in
 * place: each channel's window shifts left one tap and xbc (the device
 * xBC row of [conv_dim]) enters at the newest end. weight is the
 * [conv_dim, conv_kernel] depthwise kernel, bias is [conv_dim]. out
 * receives [conv_dim] of silu(dot(state, weight) + bias). nonfinite is a
 * 4-byte device flag raised when any output is not finite. Launch-only:
 * no copy, no allocation, no synchronization; the caller queues it on
 * stream (NULL = default) and retains every owner until completion. */
int inference_cuda_mamba_conv1d_step_launch(
    float *conv_state, size_t conv_state_bytes,
    const float *xbc, size_t xbc_bytes,
    const float *weight, size_t weight_bytes,
    const float *bias, size_t bias_bytes,
    float *out, size_t out_bytes,
    unsigned int *nonfinite,
    size_t conv_dim, size_t conv_kernel,
    void *stream, int *out_stage);

/* Selective SSM decode step for num_heads heads of head_dim each with
 * state_size per head. ssm_state is [num_heads, head_dim, state_size]
 * device memory updated in place. x is [num_heads * head_dim] (the SiLU'd
 * conv x partition), b and c are [n_groups * state_size], dt is
 * [num_heads], dt_bias/a_log/d are [num_heads]. out receives
 * [num_heads * head_dim] of y = C.state + D*x after the per-head state
 * decay-and-accumulate. time_step_min is the lower dt clamp. nonfinite
 * is the same 4-byte rejection flag. Launch-only contract as above. */
int inference_cuda_mamba_ssm_step_launch(
    float *ssm_state, size_t ssm_state_bytes,
    const float *x, size_t x_bytes,
    const float *b, size_t b_bytes,
    const float *c, size_t c_bytes,
    const float *dt, size_t dt_bytes,
    const float *dt_bias, size_t dt_bias_bytes,
    const float *a_log, size_t a_log_bytes,
    const float *d, size_t d_bytes,
    float *out, size_t out_bytes,
    unsigned int *nonfinite,
    size_t num_heads, size_t head_dim, size_t state_size, size_t n_groups,
    float time_step_min,
    void *stream, int *out_stage);

/* SiLU-gated grouped RMS norm for the mamba chain: out[i] =
 * scan[i] * silu(gate[i]), then per-group RMS normalization with the
 * mixer norm weight, exactly as forward/mamba.rs lines 167-175. One
 * thread per group of group_size elements. Launch-only contract. */
int inference_cuda_mamba_gated_norm_launch(
    const float *scan, size_t scan_bytes,
    const float *gate, size_t gate_bytes,
    const float *weight, size_t weight_bytes,
    float *out, size_t out_bytes,
    unsigned int *nonfinite,
    size_t total, size_t group_size, float eps,
    void *stream, int *out_stage);

/* Residual add for the mamba chain: out[i] = y[i] + residual[i].
 * Launch-only contract. */
int inference_cuda_mamba_residual_launch(
    const float *y, size_t y_bytes,
    const float *residual, size_t residual_bytes,
    float *out, size_t out_bytes,
    unsigned int *nonfinite,
    size_t total,
    void *stream, int *out_stage);
}

#endif /* INFERENCE_CUDA_MAMBA_STEP_H */
