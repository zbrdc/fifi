/* Host-only selection of the f16 attention part arm for the current model and GPU. */
#ifndef INFERENCE_CUDA_ATTN_PREFILL_PLAN_H
#define INFERENCE_CUDA_ATTN_PREFILL_PLAN_H

#include <stddef.h>

namespace attn_prefill_plan {

enum class Arm { none, simt, dsplit };

/* Unset, zero and unrecognized values retain the legacy launch path. */
constexpr bool adaptive_enabled(const char *value) {
    return value && value[0] == '1' && value[1] == '\0';
}

/* optin_limit is the device's cudaDevAttrMaxSharedMemoryPerBlockOptin, already
 * the per-block opt-in limit. static_bytes is each kernel's actual reported
 * static shared allocation. Subtract before comparing to avoid overflow. */
constexpr bool fits(size_t optin_limit, size_t static_bytes, size_t dynamic_bytes) {
    return static_bytes <= optin_limit && dynamic_bytes <= optin_limit - static_bytes;
}

constexpr bool fits_threads(int device_max, int kernel_max, int requested) {
    return requested > 0 && device_max >= requested && kernel_max >= requested;
}

constexpr Arm select(int dsplit_ptx, size_t optin_limit, int device_threads,
                     size_t dsplit_static, size_t dsplit_dynamic, int dsplit_threads,
                     size_t simt_static, size_t simt_dynamic, int simt_threads) {
    return dsplit_ptx >= 80 && fits_threads(device_threads, dsplit_threads, 512) &&
                   fits(optin_limit, dsplit_static, dsplit_dynamic)
               ? Arm::dsplit
           : fits_threads(device_threads, simt_threads, 128) &&
                     fits(optin_limit, simt_static, simt_dynamic)
               ? Arm::simt : Arm::none;
}

/* Fixed Q and key tiles, with model head dimension deciding the byte count. */
constexpr size_t simt_bytes(int rows, int tile, int head_dim) {
    return size_t(rows + 2 * tile) * size_t(head_dim) * 2;
}
constexpr size_t dsplit_bytes(int tile, int head_dim, int threads) {
    return size_t(4 * tile) * size_t(head_dim) * 2 + size_t(threads / 32) * 2048;
}

/* Compile these model/capacity boundaries with the actual CUDA translation
 * unit. They cover loaded PTX, the reported opt-in cap and static bytes. */
static_assert(simt_bytes(64, 32, 128) == 32768, "D128 SIMT layout");
static_assert(simt_bytes(64, 32, 256) == 65536, "D256 SIMT layout");
static_assert(!adaptive_enabled(nullptr), "unset keeps legacy");
static_assert(!adaptive_enabled(""), "empty keeps legacy");
static_assert(!adaptive_enabled("0"), "zero keeps legacy");
static_assert(!adaptive_enabled("true"), "unknown keeps legacy");
static_assert(!adaptive_enabled("10"), "non-exact one keeps legacy");
static_assert(adaptive_enabled("1"), "one enables adaptation");
static_assert(dsplit_bytes(32, 128, 512) == 65536, "D128 dsplit layout");
static_assert(dsplit_bytes(32, 256, 512) == 98304, "D256 dsplit layout");
static_assert(select(75, 65536, 1024, 0, 98304, 1024, 0, 65536, 1024) == Arm::simt, "Turing D256");
static_assert(select(80, 101376, 1024, 0, 98304, 1024, 0, 65536, 1024) == Arm::dsplit, "fast D256");
static_assert(select(80, 65536, 1024, 0, 98304, 1024, 0, 65536, 1024) == Arm::simt, "smaller cap");
static_assert(select(80, 65536, 1024, 0, 65536, 1024, 0, 32768, 1024) == Arm::dsplit, "D128 fast");
static_assert(select(75, 32768, 1024, 0, 65536, 1024, 0, 32768, 1024) == Arm::simt, "D128 SIMT");
static_assert(select(80, 98304, 1024, 1, 98304, 1024, 0, 65536, 1024) == Arm::simt, "static bytes");
static_assert(select(75, 65536, 1024, 0, 98304, 1024, 1, 65536, 1024) == Arm::none, "no fit");
static_assert(select(75, 65535, 1024, 0, 98304, 1024, 0, 65536, 1024) == Arm::none, "below cap");
static_assert(select(80, 101376, 1024, 0, 98304, 511, 0, 65536, 1024) == Arm::simt, "fast threads");
static_assert(select(80, 101376, 127, 0, 98304, 1024, 0, 65536, 1024) == Arm::none, "device threads");
static_assert(select(75, 65536, 1024, 0, 98304, 1024, 0, 65536, 127) == Arm::none, "SIMT threads");

}  // namespace attn_prefill_plan
#endif
