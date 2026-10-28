/* Mamba-2 decode-step kernels: see mamba_step.h for the contract. The
 * arithmetic mirrors forward/mamba.rs exactly — sequential F32 rounding
 * per multiply and add, the threshold-20 softplus, the two-branch SiLU —
 * so a gate run can diff the device output against the host step. */
#include "mamba_step.h"
#include "device.h"

#include <cmath>
#include <cstdint>

namespace {

constexpr unsigned kStepThreads = 128;

/* The host silu of forward/mamba.rs: x >= 0 uses 1/(1+exp(-x)); x < 0 uses
 * exp(x)/(1+exp(x)), which is overflow-safe for large negative x. */
__device__ __forceinline__ float silu_device(float x) {
    if (x >= 0.0f) {
        return __fmul_rn(x, __frcp_rn(__fadd_rn(1.0f, expf(-x))));
    }
    const float ex = expf(x);
    return __fmul_rn(x, __fdiv_rn(ex, __fadd_rn(1.0f, ex)));
}

__global__ void conv1d_step_kernel(
    float *conv_state, const float *xbc, const float *weight,
    const float *bias, float *out, unsigned int *nonfinite,
    size_t conv_dim, size_t kernel) {
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t ch = blockIdx.x * blockDim.x + threadIdx.x; ch < conv_dim;
         ch += stride) {
        float *state = conv_state + ch * kernel;
        for (size_t i = 0; i + 1 < kernel; ++i)
            state[i] = state[i + 1];
        state[kernel - 1] = xbc[ch];
        const float *w = weight + ch * kernel;
        float sum = 0.0f;
        for (size_t i = 0; i < kernel; ++i)
            sum = __fadd_rn(sum, __fmul_rn(state[i], w[i]));
        sum = __fadd_rn(sum, bias[ch]);
        const float value = silu_device(sum);
        if (!isfinite(value))
            atomicOr(nonfinite, 1u);
        else
            out[ch] = value;
    }
}

__global__ void ssm_step_kernel(
    float *ssm_state, const float *x, const float *b, const float *c,
    const float *dt, const float *dt_bias, const float *a_log,
    const float *d, float *out, unsigned int *nonfinite,
    size_t num_heads, size_t head_dim, size_t state_size, size_t n_groups,
    float time_step_min) {
    const size_t stride = blockDim.x * gridDim.x;
    const size_t total = num_heads * head_dim;
    const size_t heads_per_group = num_heads / n_groups;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += stride) {
        const size_t h = i / head_dim;
        const size_t dim = i % head_dim;
        /* Per-head scalars; every thread of a head recomputes them, which
         * keeps the kernel dependency-free at the cost of three exp calls. */
        const float raw = __fadd_rn(dt[h], dt_bias[h]);
        const float sp = raw > 20.0f ? raw : logf(__fadd_rn(1.0f, expf(raw)));
        const float step = fmaxf(sp, time_step_min);
        const float a = -expf(a_log[h]);
        const float decay = expf(__fmul_rn(step, a));
        const size_t group = h / heads_per_group;
        float *state = ssm_state + i * state_size;
        const float *bg = b + group * state_size;
        const float *cg = c + group * state_size;
        const float xi = x[i];
        float y = 0.0f;
        for (size_t n = 0; n < state_size; ++n) {
            const float updated =
                __fadd_rn(__fmul_rn(state[n], decay),
                          __fmul_rn(__fmul_rn(step, bg[n]), xi));
            state[n] = updated;
            y = __fadd_rn(y, __fmul_rn(updated, cg[n]));
        }
        const float value = __fadd_rn(y, __fmul_rn(d[h], xi));
        if (!isfinite(value))
            atomicOr(nonfinite, 1u);
        else
            out[i] = value;
    }
}

bool step_usable(const void *p, size_t bytes, size_t need) {
    return p != nullptr && bytes >= need &&
           reinterpret_cast<uintptr_t>(p) % alignof(float) == 0;
}

unsigned grid_for(size_t count) {
    const size_t blocks = (count + kStepThreads - 1) / kStepThreads;
    return static_cast<unsigned>(blocks < 65535 ? blocks : 65535);
}

}  // namespace

extern "C" int inference_cuda_mamba_conv1d_step_launch(
    float *conv_state, size_t conv_state_bytes,
    const float *xbc, size_t xbc_bytes,
    const float *weight, size_t weight_bytes,
    const float *bias, size_t bias_bytes,
    float *out, size_t out_bytes,
    unsigned int *nonfinite,
    size_t conv_dim, size_t conv_kernel,
    void *stream, int *out_stage) {
    Failure failure;
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (conv_dim == 0 || conv_kernel == 0 ||
            !step_usable(conv_state, conv_state_bytes, conv_dim * conv_kernel * sizeof(float)) ||
            !step_usable(xbc, xbc_bytes, conv_dim * sizeof(float)) ||
            !step_usable(weight, weight_bytes, conv_dim * conv_kernel * sizeof(float)) ||
            !step_usable(bias, bias_bytes, conv_dim * sizeof(float)) ||
            !step_usable(out, out_bytes, conv_dim * sizeof(float)) ||
            nonfinite == nullptr)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        (void)cudaGetLastError();
        cudaStream_t s = static_cast<cudaStream_t>(stream);
        failure.record(cudaMemsetAsync(nonfinite, 0, 4, s),
                       INFERENCE_CUDA_STAGE_COPY);
        if (failure.ok())
            conv1d_step_kernel<<<grid_for(conv_dim), kStepThreads, 0, s>>>(
                conv_state, xbc, weight, bias, out, nonfinite, conv_dim,
                conv_kernel);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return static_cast<int>(failure.err);
    } catch (...) {
        report_stage(out_stage, failure.stage);
        return failure.ok() ? INFERENCE_CUDA_NATIVE_EXCEPTION
                            : static_cast<int>(failure.err);
    }
}

extern "C" int inference_cuda_mamba_ssm_step_launch(
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
    void *stream, int *out_stage) {
    Failure failure;
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        const size_t total = num_heads * head_dim;
        if (num_heads == 0 || head_dim == 0 || state_size == 0 ||
            n_groups == 0 || num_heads % n_groups != 0 ||
            !step_usable(ssm_state, ssm_state_bytes,
                         total * state_size * sizeof(float)) ||
            !step_usable(x, x_bytes, total * sizeof(float)) ||
            !step_usable(b, b_bytes, n_groups * state_size * sizeof(float)) ||
            !step_usable(c, c_bytes, n_groups * state_size * sizeof(float)) ||
            !step_usable(dt, dt_bytes, num_heads * sizeof(float)) ||
            !step_usable(dt_bias, dt_bias_bytes, num_heads * sizeof(float)) ||
            !step_usable(a_log, a_log_bytes, num_heads * sizeof(float)) ||
            !step_usable(d, d_bytes, num_heads * sizeof(float)) ||
            !step_usable(out, out_bytes, total * sizeof(float)) ||
            nonfinite == nullptr || !std::isfinite(time_step_min))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        (void)cudaGetLastError();
        cudaStream_t s = static_cast<cudaStream_t>(stream);
        failure.record(cudaMemsetAsync(nonfinite, 0, 4, s),
                       INFERENCE_CUDA_STAGE_COPY);
        if (failure.ok())
            ssm_step_kernel<<<grid_for(total), kStepThreads, 0, s>>>(
                ssm_state, x, b, c, dt, dt_bias, a_log, d, out, nonfinite,
                num_heads, head_dim, state_size, n_groups, time_step_min);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return static_cast<int>(failure.err);
    } catch (...) {
        report_stage(out_stage, failure.stage);
        return failure.ok() ? INFERENCE_CUDA_NATIVE_EXCEPTION
                            : static_cast<int>(failure.err);
    }
}

namespace {

__global__ void gated_norm_kernel(
    const float *scan, const float *gate, const float *weight, float *out,
    unsigned int *nonfinite, size_t total, size_t group_size, float eps) {
    const size_t groups = total / group_size;
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t g = blockIdx.x * blockDim.x + threadIdx.x; g < groups;
         g += stride) {
        float *o = out + g * group_size;
        const float *s = scan + g * group_size;
        const float *gt = gate + g * group_size;
        const float *w = weight + g * group_size;
        float variance = 0.0f;
        for (size_t i = 0; i < group_size; ++i) {
            o[i] = __fmul_rn(s[i], silu_device(gt[i]));
            variance = __fadd_rn(variance, __fmul_rn(o[i], o[i]));
        }
        variance = __fadd_rn(__fdiv_rn(variance, static_cast<float>(group_size)), eps);
        const float scale = __frsqrt_rn(variance);
        for (size_t i = 0; i < group_size; ++i) {
            const float value = __fmul_rn(__fmul_rn(o[i], scale), w[i]);
            if (!isfinite(value)) {
                atomicOr(nonfinite, 1u);
                return;
            }
            o[i] = value;
        }
    }
}

__global__ void residual_kernel(
    const float *y, const float *residual, float *out,
    unsigned int *nonfinite, size_t total) {
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += stride) {
        const float value = __fadd_rn(y[i], residual[i]);
        if (!isfinite(value))
            atomicOr(nonfinite, 1u);
        else
            out[i] = value;
    }
}

}  // namespace

extern "C" int inference_cuda_mamba_gated_norm_launch(
    const float *scan, size_t scan_bytes,
    const float *gate, size_t gate_bytes,
    const float *weight, size_t weight_bytes,
    float *out, size_t out_bytes,
    unsigned int *nonfinite,
    size_t total, size_t group_size, float eps,
    void *stream, int *out_stage) {
    Failure failure;
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (total == 0 || group_size == 0 || total % group_size != 0 ||
            !std::isfinite(eps) ||
            !step_usable(scan, scan_bytes, total * sizeof(float)) ||
            !step_usable(gate, gate_bytes, total * sizeof(float)) ||
            !step_usable(weight, weight_bytes, total * sizeof(float)) ||
            !step_usable(out, out_bytes, total * sizeof(float)) ||
            nonfinite == nullptr)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        (void)cudaGetLastError();
        cudaStream_t s = static_cast<cudaStream_t>(stream);
        failure.record(cudaMemsetAsync(nonfinite, 0, 4, s),
                       INFERENCE_CUDA_STAGE_COPY);
        if (failure.ok())
            gated_norm_kernel<<<grid_for(total / group_size), kStepThreads, 0, s>>>(
                scan, gate, weight, out, nonfinite, total, group_size, eps);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return static_cast<int>(failure.err);
    } catch (...) {
        report_stage(out_stage, failure.stage);
        return failure.ok() ? INFERENCE_CUDA_NATIVE_EXCEPTION
                            : static_cast<int>(failure.err);
    }
}

extern "C" int inference_cuda_mamba_residual_launch(
    const float *y, size_t y_bytes,
    const float *residual, size_t residual_bytes,
    float *out, size_t out_bytes,
    unsigned int *nonfinite,
    size_t total,
    void *stream, int *out_stage) {
    Failure failure;
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (total == 0 ||
            !step_usable(y, y_bytes, total * sizeof(float)) ||
            !step_usable(residual, residual_bytes, total * sizeof(float)) ||
            !step_usable(out, out_bytes, total * sizeof(float)) ||
            nonfinite == nullptr)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        (void)cudaGetLastError();
        cudaStream_t s = static_cast<cudaStream_t>(stream);
        failure.record(cudaMemsetAsync(nonfinite, 0, 4, s),
                       INFERENCE_CUDA_STAGE_COPY);
        if (failure.ok())
            residual_kernel<<<grid_for(total), kStepThreads, 0, s>>>(
                y, residual, out, nonfinite, total);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return static_cast<int>(failure.err);
    } catch (...) {
        report_stage(out_stage, failure.stage);
        return failure.ok() ? INFERENCE_CUDA_NATIVE_EXCEPTION
                            : static_cast<int>(failure.err);
    }
}
