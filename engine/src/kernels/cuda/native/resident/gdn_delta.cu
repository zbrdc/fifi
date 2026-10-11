// RD-7b: launch shim for the vendored gated_delta_k (kern::gated_delta in ops.cu). No kernel is
// added; this file proves the arguments the kernel does not check and makes one extern "C" call.
// Design guidance only, no code copied: nothing here comes from MegaCapybara (no
// MegaCapybara-derived design); the kernel is the user's own xavier-qwen code restored in RD-4V.
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "ops.h"
#include "resident/gdn_delta.h"

namespace {

constexpr int kMaxM = 32768;
// Policy limits, not kernel bounds (the kernel launches n_v_heads blocks): Qwen3.6 and Qwen3-Next have 16 key and
// 32 value heads; raise these here, and in GDN_MAX_* of the wrapper, for a larger model.
constexpr int kMaxKHeads = 64;
constexpr int kMaxVHeads = 128;
constexpr float kEps = 1e-6f;  // one eps serves both norms; the GGUF rms eps is the f32 1e-6
constexpr size_t D = kern::kStateDim;

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }

struct Span {
    const void *p;
    size_t bytes;
};

bool apart(const Span &a, const Span &b) {
    uintptr_t pa = (uintptr_t)a.p, pb = (uintptr_t)b.p;
    return pa + a.bytes <= pb || pb + b.bytes <= pa;
}

bool shapes_ok(const inference_cuda_gdn_delta_args &a) {
    if (a.m < 1 || a.m > kMaxM || a.n_k_heads < 1 || a.n_k_heads > kMaxKHeads || a.n_v_heads < 1 ||
        a.n_v_heads > kMaxVHeads || a.n_v_heads % a.n_k_heads != 0 || a.eps != kEps) {
        return false;
    }
    // The offsets are bounded from above by ld (checked below, ld <= 1 << 22), so they have no limit of their own.
    if (a.z_off < 0 || a.beta_off < 0 || a.alpha_off < 0) {
        return false;
    }
    const long conv_cols = (2L * a.n_k_heads + a.n_v_heads) * (long)D;
    const long z_end = (long)a.z_off + (long)a.n_v_heads * (long)D;
    const long b_end = (long)a.beta_off + a.n_v_heads, al_end = (long)a.alpha_off + a.n_v_heads;
    const long need_ld = z_end > b_end ? (z_end > al_end ? z_end : al_end) : (b_end > al_end ? b_end : al_end);
    return a.conv_ld >= conv_cols && a.conv_ld <= (1 << 20) && a.ld >= need_ld && a.ld <= (1 << 22);
}

// Mirrors the Rust wrapper term by term (RD-7b.md "C and Rust refusals"); a term added here needs a
// row there. The Rust side refuses everything this refuses, so a C refusal means a wrapper defect.
bool valid(const inference_cuda_gdn_delta_args &a, void *stream) {
    if (a.conv == NULL || a.proj == NULL || a.a == NULL || a.dt_bias == NULL || a.norm_w == NULL ||
        a.state == NULL || a.out == NULL || stream == NULL || !shapes_ok(a)) {
        return false;
    }
    if (!aligned(a.conv, 4) || !aligned(a.proj, 4) || !aligned(a.a, 4) || !aligned(a.dt_bias, 4) ||
        !aligned(a.norm_w, 4) || !aligned(a.state, 4) || !aligned(a.out, 4)) {
        return false;
    }
    const size_t m = (size_t)a.m, nv = (size_t)a.n_v_heads, nk = (size_t)a.n_k_heads;
    const size_t f = sizeof(float);
    const size_t conv_cols = (2 * nk + nv) * D;
    const size_t z_end = (size_t)a.z_off + nv * D, b_end = (size_t)a.beta_off + nv, al_end = (size_t)a.alpha_off + nv;
    size_t proj_cols = z_end > b_end ? z_end : b_end;
    proj_cols = proj_cols > al_end ? proj_cols : al_end;
    const Span state = {a.state, nv * D * D * f}, out = {a.out, m * nv * D * f};
    const Span reads[] = {{a.conv, ((m - 1) * (size_t)a.conv_ld + conv_cols) * f},
                          {a.proj, ((m - 1) * (size_t)a.ld + proj_cols) * f},
                          {a.a, nv * f},
                          {a.dt_bias, nv * f},
                          {a.norm_w, D * f}};
    if (!apart(state, out)) {
        return false;
    }
    for (const Span &r : reads) {
        if (!apart(state, r) || !apart(out, r)) {
            return false;
        }
    }
    return true;
}

}  // namespace

extern "C" int inference_cuda_gdn_delta_launch(const inference_cuda_gdn_delta_args *args,
                                               void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid(*args, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        // snap stays null: the speculative-rollback snapshots are not used at stage 1.
        kern::DeltaArgs call = {};
        call.conv = (const float *)args->conv;
        call.conv_ld = args->conv_ld;
        call.proj = (const float *)args->proj;
        call.ld = args->ld;
        call.z_off = args->z_off;
        call.beta_off = args->beta_off;
        call.alpha_off = args->alpha_off;
        call.a = (const float *)args->a;
        call.dt_bias = (const float *)args->dt_bias;
        call.norm_w = (const float *)args->norm_w;
        call.state = (float *)args->state;
        call.snap = NULL;
        call.out = (float *)args->out;
        call.m = args->m;
        call.n_k_heads = args->n_k_heads;
        call.n_v_heads = args->n_v_heads;
        call.eps = args->eps;
        Failure failure;
        failure.record(kern::gated_delta(call, (cudaStream_t)stream), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
