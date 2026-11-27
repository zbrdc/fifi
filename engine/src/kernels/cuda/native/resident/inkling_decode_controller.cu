// Fused controller chain of one Inkling 2 decode visit (FIFI_INKLING_CUDA_DECODE_GLUE): the gate
// norms and the down reduction in one launch, both controller projections and the controller
// norm in another. The contract is resident/inkling_decode_controller.h. Every kernel reproduces
// the arithmetic of the launches it replaces, operation for operation, so a visit's bits do not
// depend on the switch.
#include <cuda_runtime.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/inkling_decode_controller.h"
#include "resident/inkling_controller_norm.cuh"
#include "resident/inkling_stream_norm.cuh"
#include "resident/inkling_warp_sum.cuh"

namespace {

constexpr int threads = 256;
constexpr int warps = threads / 32;
constexpr int max_width = 4096;
constexpr int pair_rank = 64;
constexpr int pair_rows = 16;

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }

struct Span {
    const void *p;
    size_t bytes;
};

bool apart(const Span &a, const Span &b) {
    const uintptr_t x = (uintptr_t)a.p, y = (uintptr_t)b.p;
    return x <= y ? a.bytes <= y - x : b.bytes <= x - y;
}

bool finite_positive(float eps) { return isfinite(eps) && eps > 0.0f; }

// `writes` shares no byte with `reads`, with `tickets`, with `others` and with each other.
bool apart_all(const Span *writes, int writes_n, const Span *reads, int reads_n, Span tickets,
               const Span *others, int others_n) {
    for (int i = 0; i < writes_n; ++i) {
        if (!apart(writes[i], tickets)) return false;
        for (int j = 0; j < reads_n; ++j) {
            if (!apart(writes[i], reads[j])) return false;
        }
        for (int j = 0; j < others_n; ++j) {
            if (!apart(writes[i], others[j])) return false;
        }
        for (int j = 0; j < i; ++j) {
            if (!apart(writes[i], writes[j])) return false;
        }
    }
    return true;
}

// One block of a role: both gate norms (every block of the role computes the same bits) and
// `warps` rows of the down reduction, one warp per row.
__global__ void decode_controller_gate_down(inference_cuda_controller_gate_args a) {
    const inference_cuda_controller_gate_role &r = a.role[blockIdx.y];
    extern __shared__ __align__(16) float staged[];  // [input | message], then the normed gate
    __shared__ float scratch[kNormThreads];
    const int lane = threadIdx.x;
    const int width = a.width;
    // The pair launch of this role counts from zero; its last block clears the ticket again.
    if (blockIdx.x == 0 && lane == 0) a.tickets[blockIdx.y] = 0u;
    for (int i = lane; i < 2 * width; i += threads) {
        staged[i] = i < width ? r.input[i] : r.message[i - width];
    }
    int failed[2] = {0, 0};
    float scale[2];
    for (int half = 0; half < 2; ++half) {
        scale[half] = rms_scale(staged + half * width, width, a.gate_eps, scratch, failed[half]);
    }
    for (int half = 0; half < 2; ++half) {
        int invalid = 0;
        for (int i = lane; i < width; i += threads) {
            const float value = __fmul_rn(staged[half * width + i], scale[half]);
            staged[half * width + i] = value;
            r.gate[half * width + i] = value;
            invalid |= !isfinite(value);
        }
        failed[half] |= __syncthreads_or(invalid);
    }
    if (lane == 0 && (failed[0] | failed[1])) {
        for (int half = 0; half < 2; ++half) {
            if (failed[half]) r.health[half][0] = __int_as_float(0x7fc00000);
        }
    }
    // decode_controller_down_exact: 32 chains over the float4 words in ascending order, then the
    // xor butterfly of dense_gemv.cu's warp_sum.
    const int row = blockIdx.x * warps + (lane >> 5);
    if (row >= a.rank) return;
    const int words = 2 * width / 4;
    const float4 *w = reinterpret_cast<const float4 *>(r.down) + static_cast<size_t>(row) * words;
    float sum = 0.0f;
#pragma unroll 3
    for (int i = lane & 31; i < words; i += 32) {
        const float4 v = w[i];
        const float4 u = *reinterpret_cast<const float4 *>(staged + 4 * i);
        sum = fmaf(v.x, u.x, sum);
        sum = fmaf(v.y, u.y, sum);
        sum = fmaf(v.z, u.z, sum);
        sum = fmaf(v.w, u.w, sum);
    }
    sum = warp_sum(sum);
    if ((lane & 31) == 0) r.reduced[row] = sum;
}

// One block: 16 rows of one projection of one role (decode_control_pair), then the last block of
// the role applies the controller norm over every gain and forget row.
__global__ void decode_controller_pair_norm(inference_cuda_controller_apply_args a) {
    const inference_cuda_controller_apply_role &r = a.role[blockIdx.z];
    const bool gain_half = blockIdx.y == 0;
    const int lane = threadIdx.x % 16;
    const int row = blockIdx.x * pair_rows + threadIdx.x / 16;
    const float *weights = gain_half ? r.gain_w : r.forget_w;
    const float4 *w = reinterpret_cast<const float4 *>(weights) + row * 16;
    const float4 v = w[lane];
    const float *x = r.reduced + lane * 4;
    float s = fmaf(v.x, x[0], 0.f);
    s = fmaf(v.y, x[1], s);
    s = fmaf(v.z, x[2], s);
    s = fmaf(v.w, x[3], s);
#pragma unroll
    for (int offset = 8; offset; offset /= 2) s += __shfl_xor_sync(0xffffffff, s, offset, 16);
    if (!lane) (gain_half ? r.gain : r.forget)[row] = s;
    // Every block of the role publishes its rows, then takes a ticket; the block that draws the
    // last one sees them all (the threadfence reduction pattern) and runs the norm.
    __shared__ int last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned ticket = atomicAdd(&a.tickets[blockIdx.z], 1u);
        last = ticket == gridDim.x * 2u - 1u;
        if (last) a.tickets[blockIdx.z] = 0u;
    }
    __syncthreads();
    if (!last) return;
    controller_norm_one(r.input, r.message, r.gamma, r.gain, r.forget, r.out, r.health, a.width,
                        a.eps);
}

}  // namespace

extern "C" int inference_cuda_decode_controller_gate_down(
    const inference_cuda_controller_gate_args *args, void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!args || !stream || args->roles < 1 || args->roles > INFERENCE_CUDA_CONTROLLER_MAX_ROLES ||
        args->width < 16 || args->width > max_width || args->width % 16 != 0 ||
        args->rank < 8 || args->rank > 1024 || args->rank % 8 != 0 ||
        !finite_positive(args->gate_eps) || !aligned(args->tickets, 4))
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    const size_t row = static_cast<size_t>(args->width) * sizeof(float);
    const size_t gate = 2 * row;
    const size_t reduced = static_cast<size_t>(args->rank) * sizeof(float);
    const Span tickets = {args->tickets, static_cast<size_t>(args->roles) * 4};
    for (int role = 0; role < args->roles; ++role) {
        const inference_cuda_controller_gate_role &r = args->role[role];
        if (!r.input || !r.message || !r.gate || !r.down || !r.reduced || !r.health[0] ||
            !r.health[1] || !aligned(r.input, 4) || !aligned(r.message, 4) ||
            !aligned(r.gate, 16) || !aligned(r.down, 16))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const Span reads[] = {{r.input, row}, {r.message, row}, {r.down, gate * args->rank}};
        // The gate, the reduced row and the two health words; the health words may be one word,
        // since each writer stores the same quiet NaN.
        const Span words[] = {
            {r.gate, gate}, {r.reduced, reduced}, {r.health[0], 4}, {r.health[1], 4}};
        Span others[6 * INFERENCE_CUDA_CONTROLLER_MAX_ROLES];
        int others_n = 0;
        for (int o = 0; o < role; ++o) {
            const inference_cuda_controller_gate_role &p = args->role[o];
            others[others_n++] = {p.input, row};
            others[others_n++] = {p.message, row};
            others[others_n++] = {p.gate, gate};
            others[others_n++] = {p.reduced, reduced};
            others[others_n++] = {p.health[0], 4};
            others[others_n++] = {p.health[1], 4};
        }
        for (int i = 0; i < 4; ++i) {
            if (!apart(words[i], tickets)) return INFERENCE_CUDA_INVALID_ARGUMENT;
            for (int j = 0; j < 3; ++j) {
                if (!apart(words[i], reads[j])) return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            for (int j = 0; j < others_n; ++j) {
                if (!apart(words[i], others[j])) return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            for (int j = 0; j < i; ++j) {
                if (!(i == 3 && j == 2) && !apart(words[i], words[j]))
                    return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
        }
    }
    inference_cuda_controller_gate_args a = *args;
    decode_controller_gate_down<<<dim3((a.rank + warps - 1) / warps, a.roles), threads, gate,
                                     static_cast<cudaStream_t>(stream)>>>(a);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}

extern "C" int inference_cuda_decode_controller_pair_norm(
    const inference_cuda_controller_apply_args *args, void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!args || !stream || args->roles < 1 || args->roles > INFERENCE_CUDA_CONTROLLER_MAX_ROLES ||
        args->width < 16 || args->width > max_width || args->width % 16 != 0 ||
        args->rank != pair_rank || !finite_positive(args->eps) || !aligned(args->tickets, 4))
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    const size_t row = static_cast<size_t>(args->width) * sizeof(float);
    const size_t reduced = static_cast<size_t>(pair_rank) * sizeof(float);
    const Span tickets = {args->tickets, static_cast<size_t>(args->roles) * 4};
    for (int role = 0; role < args->roles; ++role) {
        const inference_cuda_controller_apply_role &r = args->role[role];
        if (!r.input || !r.message || !r.gamma || !r.reduced || !r.gain_w || !r.forget_w ||
            !r.gain || !r.forget || !r.out || !r.health || !aligned(r.reduced, 4) ||
            !aligned(r.gain_w, 16) || !aligned(r.forget_w, 16))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const Span reads[] = {
            {r.input, row},        {r.message, row},       {r.gamma, row},
            {r.reduced, reduced},  {r.gain_w, row * pair_rank}, {r.forget_w, row * pair_rank}};
        const Span writes[] = {{r.gain, row}, {r.forget, row}, {r.out, row}, {r.health, 4}};
        Span others[8 * INFERENCE_CUDA_CONTROLLER_MAX_ROLES];
        int others_n = 0;
        for (int o = 0; o < role; ++o) {
            const inference_cuda_controller_apply_role &p = args->role[o];
            others[others_n++] = {p.input, row};
            others[others_n++] = {p.message, row};
            others[others_n++] = {p.gamma, row};
            others[others_n++] = {p.reduced, reduced};
            others[others_n++] = {p.gain, row};
            others[others_n++] = {p.forget, row};
            others[others_n++] = {p.out, row};
            others[others_n++] = {p.health, 4};
        }
        if (!apart_all(writes, 4, reads, 6, tickets, others, others_n))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    inference_cuda_controller_apply_args a = *args;
    decode_controller_pair_norm<<<dim3(a.width / pair_rows, 2, a.roles), threads, 0,
                                  static_cast<cudaStream_t>(stream)>>>(a);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}
