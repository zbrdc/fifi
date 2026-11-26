// Independent cross-pass controller roles, one block per role.
#include <cuda_runtime.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/lane_sum.cuh"

namespace {
constexpr int threads = 256;
static_assert(threads == kLaneThreads, "the controller gate norm reduces in the lane order");

// Packed [gate outputs | raw triples | scale pairs], all in role order.
__global__ void inkling_rebuild_gates(float *workspace, int width, int roles) {
    const int role = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * width) return;
    const float *raw = workspace + 2 * roles * width + role * 3 * width;
    const float *scales = workspace + 5 * roles * width + 2 * role;
    workspace[role * 2 * width + i] = __fmul_rn(raw[i], scales[i >= width]);
}

// One CTA per gate half. Keep the original pair for applying gain/forget;
// broadcasting gamma here avoids uploading the same weight for every row. The gate is the
// host's `control_input_from` half with the squares in the lane order (`lane_sum.cuh`), the order
// of every resident Inkling row norm. Unlike the host, a half whose squares or variance overflow
// (the host then scales by zero) fails the health word, so the visit stops with an error instead
// of carrying a degenerate gate.
__global__ void inkling_prepare_controller(const float *pairs, const float *gamma,
                                           float *triples, float *gates, float *health,
                                           int width, float eps) {
    const int half = blockIdx.x;
    const int role = half / 2;
    const int side = half % 2;
    const float *src = pairs + half * width;
    float *raw = triples + (3 * role + side) * width;
    float *gate = gates + half * width;
    float *raw_gamma = triples + (3 * role + 2) * width;
    const int lane = threadIdx.x;
    __shared__ float scratch[kLaneWarps];
    int invalid = 0;
    for (int i = lane; i < width; i += threads) {
        const float x = src[i];
        raw[i] = x;
        if (side == 0) raw_gamma[i] = gamma[i];
    }
    const float scale = lane_rms_scale(src, width, eps, scratch, invalid);
    for (int i = lane; i < width; i += threads) {
        const float x = __fmul_rn(src[i], scale);
        gate[i] = x;
        invalid |= !isfinite(x);
    }
    const int failed = __syncthreads_or(invalid);
    if (lane == 0) health[half] = failed ? __int_as_float(0x7fc00000) : 0.0f;
}

// One role over the caller's rows: `out = rmsnorm((input - tanh(forget) * input) + tanh(gain)
// * message) * gamma`, the host's controller step, every operation rounded.
__device__ void controller_norm_one(const float *input, const float *message, const float *gamma,
                                    const float *gain, const float *forget, float *out,
                                    float *health, int width, float eps) {
    const int lane = threadIdx.x;

    __shared__ float sums[threads];
    float squares = 0.0f;
    bool invalid = false;
    for (int i = lane; i < width; i += threads) {
        const float kept = __fsub_rn(input[i], __fmul_rn(tanhf(forget[i]), input[i]));
        const float x = __fadd_rn(kept, __fmul_rn(tanhf(gain[i]), message[i]));
        invalid |= !isfinite(gain[i]) || !isfinite(forget[i]) || !isfinite(x);
        out[i] = x;
        squares += x * x;
    }
    sums[lane] = squares;
    __syncthreads();
    for (int stride = threads / 2; stride != 0; stride /= 2) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        __syncthreads();
    }
    const float variance = sums[0] / width + eps;
    const float scale = 1.0f / sqrtf(variance);
    invalid |= !isfinite(variance) || !isfinite(scale);
    for (int i = lane; i < width; i += threads) {
        out[i] = __fmul_rn(__fmul_rn(out[i], scale), gamma[i]);
        invalid |= !isfinite(out[i]);
    }
    const int failed = __syncthreads_or(invalid);
    if (lane == 0) health[0] = failed ? __int_as_float(0x7fc00000) : 0.0f;
}

__global__ void inkling_controller_norm(float *workspace, float *health, int width, float eps) {
    const int role = blockIdx.x;
    const int roles = gridDim.x;
    const float *input = workspace + role * 3 * width;
    const float *message = input + width;
    const float *gamma = message + width;
    const float *gain = workspace + (3 * roles + role * 2) * width;
    const float *forget = gain + width;
    float *out = workspace + (5 * roles + role) * width;
    controller_norm_one(input, message, gamma, gain, forget, out, health + role, width, eps);
}

// The rows of at most kMaxRowRoles roles, passed by value: a host array of device pointers
// cannot be read from the device, and 96 pointers fit the parameter space.
constexpr int kMaxRowRoles = 16;
struct RoleRows {
    const float *row[6 * kMaxRowRoles];
};

// One block per role; `row[6 * role]` holds the role's `input, message, gamma, gain, forget,
// out`: the caller's rows, nothing is packed into a workspace.
__global__ void inkling_controller_norm_rows(RoleRows rows, float *health, int width, float eps) {
    const float **r = rows.row + 6 * blockIdx.x;
    controller_norm_one(r[0], r[1], r[2], r[3], r[4], const_cast<float *>(r[5]),
                        health + blockIdx.x, width, eps);
}

// Same packed roles and health contract as the generic path. The exact-model
// first tile matched host f32 tanh with FP64 tanh narrowed to f32; this is not
// a guarantee of identical transcendental results on other inputs or hosts.
__global__ void inkling_controller_norm_cpu_parity(float *workspace, float *health, int width, float eps) {
    const int role = blockIdx.x;
    const int roles = gridDim.x;
    const int lane = threadIdx.x;
    const float *input = workspace + role * 3 * width;
    const float *message = input + width;
    const float *gamma = message + width;
    const float *gain = workspace + (3 * roles + role * 2) * width;
    const float *forget = gain + width;
    float *out = workspace + (5 * roles + role) * width;

    extern __shared__ float x[];
    __shared__ float scale;
    bool invalid = false;
    for (int i = lane; i < width; i += threads) {
        const float f = forget[i], g = gain[i], in = input[i];
        const float tf = __double2float_rn(tanh(static_cast<double>(f)));
        const float tg = __double2float_rn(tanh(static_cast<double>(g)));
        const float kept = __fsub_rn(in, __fmul_rn(tf, in));
        const float value = __fadd_rn(kept, __fmul_rn(tg, message[i]));
        // Infinities in the coefficients can be masked by tanh saturation.
        invalid |= !isfinite(f) || !isfinite(g) || !isfinite(value);
        x[i] = value;
    }
    __syncthreads();
    if (lane == 0) {
        // Rust dot(x, x): ordered f32 multiply, then f32 sum (whose seed is -0).
        float squares = -0.0f;
        for (int i = 0; i < width; ++i) {
            const float value = x[i];
            squares = __fadd_rn(squares, __fmul_rn(value, value));
        }
        const float mean = __fdiv_rn(squares, static_cast<float>(width));
        const float root = __fsqrt_rn(__fadd_rn(mean, eps));
        scale = __frcp_rn(root);
        // A square, sum, or epsilon addition can overflow even if the output is finite.
        invalid |= !isfinite(squares) || !isfinite(root) || !isfinite(scale);
    }
    __syncthreads();
    for (int i = lane; i < width; i += threads) {
        const float value = __fmul_rn(__fmul_rn(x[i], scale), gamma[i]);
        out[i] = value;
        invalid |= !isfinite(value);
    }
    const int failed = __syncthreads_or(invalid);
    if (lane == 0) health[role] = failed ? __int_as_float(0x7fc00000) : 0.0f;
}
}  // namespace

extern "C" int inference_cuda_inkling_rebuild_gates(float *workspace, int width, int roles,
                                                    void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!workspace || !stream || uintptr_t(workspace) % alignof(float) ||
        width < 1 || width > 4096 || roles < 1 || roles > 1536)
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    const dim3 grid((2 * width + threads - 1) / threads, roles);
    inkling_rebuild_gates<<<grid, threads, 0, (cudaStream_t)stream>>>(workspace, width, roles);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return int(failure.err);
}

extern "C" int inference_cuda_inkling_prepare_controller(
    const float *pairs, const float *gamma, float *triples, float *gates, float *health,
    int width, int roles, float eps, void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!pairs || !gamma || !triples || !gates || !health || !stream ||
        width < 1 || width > 4096 || roles < 1 || roles > 1536 ||
        !isfinite(eps) || eps <= 0.0f)
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    inkling_prepare_controller<<<2 * roles, threads, 0, (cudaStream_t)stream>>>(
        pairs, gamma, triples, gates, health, width, eps);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return int(failure.err);
}

extern "C" int inference_cuda_inkling_controller_norm(float *workspace, float *health,
                                                      int width, int roles, float eps, int cpu_parity,
                                                      void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (workspace == nullptr || health == nullptr || stream == nullptr || width < 1 ||
        width > 4096 || roles < 1 || roles > 1536 || !isfinite(eps) || eps <= 0.0f ||
        (cpu_parity != 0 && cpu_parity != 1))
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    const uintptr_t work = (uintptr_t)workspace, status = (uintptr_t)health;
    if (work % 4 != 0 || status % 4 != 0 ||
        (work <= status ? status - work < 6 * (size_t)roles * width * sizeof(float)
                        : work - status < (size_t)roles * sizeof(float)))
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    if (cpu_parity) {
        inkling_controller_norm_cpu_parity<<<roles, threads, width * sizeof(float), (cudaStream_t)stream>>>(
            workspace, health, width, eps);
    } else {
        inkling_controller_norm<<<roles, threads, 0, (cudaStream_t)stream>>>(workspace, health, width, eps);
    }
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

extern "C" int inference_cuda_inkling_controller_norm_rows(const void *const *rows, float *health,
                                                           int width, int roles, float eps,
                                                           void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!rows || !health || !stream || width < 1 || width > 4096 || roles < 1 ||
        roles > kMaxRowRoles || !isfinite(eps) || eps <= 0.0f)
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    const size_t row_bytes = static_cast<size_t>(width) * sizeof(float);
    const size_t roles_bytes = static_cast<size_t>(roles) * sizeof(float);
    const uintptr_t status = reinterpret_cast<uintptr_t>(health);
    if (status % alignof(float) != 0) return INFERENCE_CUDA_INVALID_ARGUMENT;
    for (int role = 0; role < roles; ++role) {
        for (int i = 0; i < 6; ++i) {
            const uintptr_t a = reinterpret_cast<uintptr_t>(rows[6 * role + i]);
            if (!rows[6 * role + i] || a % alignof(float) != 0) return INFERENCE_CUDA_INVALID_ARGUMENT;
            if (a <= status ? status - a < row_bytes : a - status < roles_bytes) return INFERENCE_CUDA_INVALID_ARGUMENT;
            // Every output row shares no byte with any row of any role; health with none.
            for (int other = 0; other < roles; ++other) {
                for (int j = 0; j < 6; ++j) {
                    const uintptr_t b = reinterpret_cast<uintptr_t>(rows[6 * other + j]);
                    if ((i == 5 || j == 5) && (other != role || i != j) &&
                        (a <= b ? b - a < row_bytes : a - b < row_bytes))
                        return INFERENCE_CUDA_INVALID_ARGUMENT;
                }
            }
        }
    }
    RoleRows packed{};
    for (int i = 0; i < 6 * roles; ++i) packed.row[i] = static_cast<const float *>(rows[i]);
    inkling_controller_norm_rows<<<roles, threads, 0, (cudaStream_t)stream>>>(packed, health, width, eps);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return int(failure.err);
}
