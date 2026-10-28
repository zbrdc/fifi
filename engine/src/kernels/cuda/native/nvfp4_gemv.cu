#include "nvfp4_gemv.h"
#include "device.h"

#include <cmath>
#include <stdint.h>

namespace {

/* kernels/nvfp4.rs GROUP_SIZE: sixteen elements share one E4M3FN scale byte. */
constexpr size_t kGroupSize = 16;
constexpr size_t kGroupBytes = kGroupSize / 2;
constexpr unsigned kMatvecThreads = 128;

/* Exact E2M1 values of the host E2M1_VALUES row, sign included. Every entry
 * is exactly representable in F32, so the table is bit for bit the host row
 * and the sign bit is folded into the lookup instead of a per-element select.
 * Copied once per block into shared memory: a banked shared load replaces the
 * old dynamically indexed local-memory array, which cost a local load plus
 * address and sign arithmetic per element. */
__constant__ float kE2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

/* E4M3FN of one scale byte, built from the bit fields exactly as nvfp4.rs
 * builds E4M3_VALUES: bias 7, subnormal fraction * 2^-9, NaN only at
 * exponent 15 with fraction 7, and the sign bit negating the magnitude.
 * Only finiteness and sign of a value are observable here, so the NaN
 * payload does not matter; the normal range reconstructs the same bits. */
__device__ float e4m3_value(unsigned char code) {
    const unsigned exponent = (code >> 3) & 15u;
    const unsigned fraction = code & 7u;
    float magnitude;
    if (exponent == 0u) {
        magnitude = static_cast<float>(fraction) / 512.0f;
    } else if (exponent == 15u && fraction == 7u) {
        magnitude = __int_as_float(0x7fc00000u);
    } else {
        magnitude = __int_as_float(((exponent + 120u) << 23) | (fraction << 20));
    }
    return (code & 128u) != 0u ? -magnitude : magnitude;
}

/* One element of the host WeightBlock::matvec sequence: one rounded multiply
 * for the decoded weight, one rounded multiply by the input, one rounded add
 * to the single accumulator. __fmul_rn and __fadd_rn never fuse, so a
 * non-finite decoded weight only sets the row's rejection flag: arithmetic
 * continues on a row whose sum will never be written, which keeps the loop
 * free of per-element branches. The set of rejected matrices is unchanged,
 * because the flag is per row and y is left untouched when it is set. */
__device__ __forceinline__ void accumulate_step(float magnitude, float scale,
                                                float xv, float &sum, bool &bad) {
    const float value = __fmul_rn(magnitude, scale);
    bad |= !isfinite(value);
    sum = __fadd_rn(sum, __fmul_rn(value, xv));
}

/* Wide group load path. Little-endian word order puts column base + k in
 * nibble k of the word pair, and float2 pair j holds x[base + 2j] and
 * x[base + 2j + 1], so this walk is exactly the host column order: groups in
 * order, bytes in order, even nibble before odd nibble. Selected by the host
 * only when the packed row starts stay 4-byte aligned and x stays 8-byte
 * aligned, so no alignment is assumed that the old kernel did not accept. */
__device__ __forceinline__ void group_wide(const unsigned char *group_bytes,
                                           const float *x, size_t base, float scale,
                                           float &sum, bool &bad, const float *e2m1) {
    const unsigned lo = *reinterpret_cast<const unsigned *>(group_bytes);
    const unsigned hi = *reinterpret_cast<const unsigned *>(group_bytes + 4);
    const float2 *pairs = reinterpret_cast<const float2 *>(x + base);
    const float2 a = pairs[0];
    const float2 b = pairs[1];
    const float2 c = pairs[2];
    const float2 d = pairs[3];
    const float2 e = pairs[4];
    const float2 f = pairs[5];
    const float2 g = pairs[6];
    const float2 h = pairs[7];
    accumulate_step(e2m1[lo & 15u], scale, a.x, sum, bad);
    accumulate_step(e2m1[(lo >> 4) & 15u], scale, a.y, sum, bad);
    accumulate_step(e2m1[(lo >> 8) & 15u], scale, b.x, sum, bad);
    accumulate_step(e2m1[(lo >> 12) & 15u], scale, b.y, sum, bad);
    accumulate_step(e2m1[(lo >> 16) & 15u], scale, c.x, sum, bad);
    accumulate_step(e2m1[(lo >> 20) & 15u], scale, c.y, sum, bad);
    accumulate_step(e2m1[(lo >> 24) & 15u], scale, d.x, sum, bad);
    accumulate_step(e2m1[(lo >> 28) & 15u], scale, d.y, sum, bad);
    accumulate_step(e2m1[hi & 15u], scale, e.x, sum, bad);
    accumulate_step(e2m1[(hi >> 4) & 15u], scale, e.y, sum, bad);
    accumulate_step(e2m1[(hi >> 8) & 15u], scale, f.x, sum, bad);
    accumulate_step(e2m1[(hi >> 12) & 15u], scale, f.y, sum, bad);
    accumulate_step(e2m1[(hi >> 16) & 15u], scale, g.x, sum, bad);
    accumulate_step(e2m1[(hi >> 20) & 15u], scale, g.y, sum, bad);
    accumulate_step(e2m1[(hi >> 24) & 15u], scale, h.x, sum, bad);
    accumulate_step(e2m1[(hi >> 28) & 15u], scale, h.y, sum, bad);
}

/* Narrow group load path for pointers the wide path may not dereference.
 * Identical arithmetic and order; only the load widths differ. */
__device__ __forceinline__ void group_narrow(const unsigned char *group_bytes,
                                             const float *x, size_t base, float scale,
                                             float &sum, bool &bad, const float *e2m1) {
#pragma unroll
    for (size_t offset = 0; offset < kGroupBytes; ++offset) {
        const unsigned byte = group_bytes[offset];
        accumulate_step(e2m1[byte & 15u], scale, x[base + offset * 2], sum, bad);
        accumulate_step(e2m1[byte >> 4], scale, x[base + offset * 2 + 1], sum, bad);
    }
}

/* Walks one row in the column order of WeightBlock::matvec. Group scales are
 * decoded and folded once per group, exactly as the host does, and a bad
 * scale stops the row: the flag makes the call fail either way, so skipping
 * the rest only avoids work whose result is discarded. */
__device__ void row_dot(const unsigned char *packed, const unsigned char *scales,
                        float weight_scale_2, const float *x, size_t per_row,
                        const float *e2m1, bool wide, float *out, bool *bad_out) {
    float sum = 0.0f;
    bool bad = false;
    for (size_t group = 0; group < per_row; ++group) {
        const float group_scale = e4m3_value(scales[group]);
        if (!isfinite(group_scale) || signbit(group_scale)) {
            bad = true;
            break;
        }
        const float scale = __fmul_rn(group_scale, weight_scale_2);
        if (!isfinite(scale)) {
            bad = true;
            break;
        }
        const unsigned char *bytes = packed + group * kGroupBytes;
        const size_t base = group * kGroupSize;
        if (wide)
            group_wide(bytes, x, base, scale, sum, bad, e2m1);
        else
            group_narrow(bytes, x, base, scale, sum, bad, e2m1);
    }
    *out = sum;
    *bad_out = bad;
}

/* Thread mapping: one thread owns one output row and walks it alone, so a
 * block covers blockDim.x consecutive output rows and the grid-stride
 * mapping row = blockIdx.x * blockDim.x + threadIdx.x (+ stride) visits
 * every row exactly once for any blockDim and gridDim. No thread shares a
 * row, so no cross-thread reduction exists that could reorder the F32
 * accumulation away from the host column order, and no partial sums are
 * combined across threads. A rejected row raises the flag instead of
 * writing y, mirroring the host error paths. */
__global__ void nvfp4_gemv_rows(const unsigned char *__restrict__ packed,
                                const unsigned char *__restrict__ scales,
                                float weight_scale_2, const float *__restrict__ x,
                                float *__restrict__ y, unsigned int *__restrict__ nonfinite,
                                size_t rows, size_t columns, unsigned int wide) {
    __shared__ float e2m1[kGroupSize];
    if (threadIdx.x < kGroupSize) e2m1[threadIdx.x] = kE2M1[threadIdx.x];
    __syncthreads();
    const size_t per_row = columns / kGroupSize;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows; row += stride) {
        float sum = 0.0f;
        bool bad = false;
        row_dot(packed + row * per_row * kGroupBytes, scales + row * per_row,
                weight_scale_2, x, per_row, e2m1, wide != 0u, &sum, &bad);
        if (bad)
            atomicOr(nonfinite, 1u);
        else
            y[row] = sum;
    }
}

struct Needs {
    size_t packed;
    size_t scale;
    size_t x;
    size_t y;
    size_t flag;
};

bool checked_needs(size_t rows, size_t columns, Needs &need) {
    if (rows == 0 || columns == 0 || columns % kGroupSize != 0 ||
        rows > SIZE_MAX / columns || columns > SIZE_MAX / sizeof(float) ||
        rows > SIZE_MAX / sizeof(float))
        return false;
    const size_t per_row = columns / kGroupSize;
    if (rows > SIZE_MAX / per_row) return false;
    need.packed = rows * columns / 2;
    need.scale = rows * per_row;
    need.x = columns * sizeof(float);
    need.y = rows * sizeof(float);
    need.flag = sizeof(unsigned int);
    return true;
}

bool usable(const void *pointer, size_t bytes, size_t need, size_t alignment) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    return pointer != nullptr && address % alignment == 0 && bytes >= need &&
           bytes <= UINTPTR_MAX - address;
}

// usable has checked both sums before this comparison.
bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return reinterpret_cast<uintptr_t>(a) + a_bytes <= reinterpret_cast<uintptr_t>(b) ||
           reinterpret_cast<uintptr_t>(b) + b_bytes <= reinterpret_cast<uintptr_t>(a);
}

bool nvfp4_regions_usable(const void *packed, size_t packed_bytes,
                          const void *scale, size_t scale_bytes,
                          const inference_cuda_workspace &ws, const Needs &need) {
    // The host reference stores exact byte counts, so the extents must match.
    if (packed_bytes != need.packed || scale_bytes != need.scale) return false;
    return usable(packed, packed_bytes, need.packed, alignof(unsigned char)) &&
           usable(scale, scale_bytes, need.scale, alignof(unsigned char)) &&
           usable(ws.x, ws.x_bytes, need.x, alignof(float)) &&
           usable(ws.y, ws.y_bytes, need.y, alignof(float)) &&
           usable(ws.xf, ws.xf_bytes, need.flag, alignof(unsigned int)) &&
           apart(packed, packed_bytes, scale, scale_bytes) &&
           apart(packed, packed_bytes, ws.x, need.x) &&
           apart(packed, packed_bytes, ws.y, need.y) &&
           apart(packed, packed_bytes, ws.xf, need.flag) &&
           apart(scale, scale_bytes, ws.x, need.x) &&
           apart(scale, scale_bytes, ws.y, need.y) &&
           apart(scale, scale_bytes, ws.xf, need.flag) &&
           apart(ws.x, need.x, ws.y, need.y) &&
           apart(ws.x, need.x, ws.xf, need.flag) &&
           apart(ws.y, need.y, ws.xf, need.flag);
}

}  // namespace

extern "C" int inference_cuda_nvfp4_gemv_ws(
    const void *packed, size_t packed_bytes,
    const void *scale, size_t scale_bytes, float weight_scale_2,
    size_t rows, size_t columns,
    const float *x, size_t x_bytes, float *y, size_t y_bytes,
    const inference_cuda_workspace *workspace, int *out_stage) {
    Failure failure;
    bool device_started = false;
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        Needs need;
        if (workspace == nullptr || !std::isfinite(weight_scale_2) ||
            weight_scale_2 <= 0.0f || !checked_needs(rows, columns, need) ||
            !nvfp4_regions_usable(packed, packed_bytes, scale, scale_bytes,
                                  *workspace, need) ||
            !usable(x, x_bytes, need.x, alignof(float)) ||
            !usable(y, y_bytes, need.y, alignof(float)))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        for (size_t column = 0; column < columns; ++column)
            if (!std::isfinite(x[column])) return INFERENCE_CUDA_INVALID_ARGUMENT;

        // Row starts stay multiples of eight bytes from packed, so 4-byte
        // group loads are safe whenever the base is 4-byte aligned; the x
        // pairs need 8. Anything else keeps the old byte loads with the same
        // arithmetic, so no input the old kernel accepted is rejected here.
        const bool wide = reinterpret_cast<uintptr_t>(packed) % 4u == 0 &&
                          reinterpret_cast<uintptr_t>(workspace->x) % 8u == 0;

        (void)cudaGetLastError();
        device_started = true;
        failure.record(cudaMemcpy(workspace->x, x, need.x, cudaMemcpyHostToDevice),
                       INFERENCE_CUDA_STAGE_COPY);
        if (failure.ok())
            failure.record(cudaMemset(workspace->xf, 0, need.flag),
                           INFERENCE_CUDA_STAGE_COPY);
        if (failure.ok()) {
            const size_t blocks_needed = (rows - 1) / kMatvecThreads + 1;
            const unsigned blocks = static_cast<unsigned>(blocks_needed < 65535
                                                             ? blocks_needed : 65535);
            nvfp4_gemv_rows<<<blocks, kMatvecThreads>>>(
                static_cast<const unsigned char *>(packed),
                static_cast<const unsigned char *>(scale), weight_scale_2,
                static_cast<const float *>(workspace->x),
                static_cast<float *>(workspace->y),
                static_cast<unsigned int *>(workspace->xf), rows, columns,
                wide ? 1u : 0u);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        failure.record(cudaDeviceSynchronize(), INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        if (failure.ok()) {
            unsigned int flag = 0;
            failure.record(cudaMemcpy(&flag, workspace->xf, need.flag,
                                      cudaMemcpyDeviceToHost),
                           INFERENCE_CUDA_STAGE_COPY);
            if (failure.ok() && flag != 0) return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (failure.ok())
            failure.record(cudaMemcpy(y, workspace->y, need.y, cudaMemcpyDeviceToHost),
                           INFERENCE_CUDA_STAGE_COPY);
        (void)cudaGetLastError();
        report_stage(out_stage, failure.stage);
        return static_cast<int>(failure.err);
    } catch (...) {
        if (device_started) {
            // Cleanup must not replace an earlier CUDA failure or the exception.
            (void)cudaDeviceSynchronize();
            (void)cudaGetLastError();
        }
        report_stage(out_stage, failure.stage);
        return failure.ok() ? INFERENCE_CUDA_NATIVE_EXCEPTION : static_cast<int>(failure.err);
    }
}
