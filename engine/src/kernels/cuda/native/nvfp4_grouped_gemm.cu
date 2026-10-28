#include "nvfp4_grouped_gemm.h"
#include "device.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cstdint>

namespace {
constexpr unsigned kThreads = 128;
constexpr unsigned kM = 16, kN = 64, kK = 16;

__device__ float fp4(unsigned code) {
    const unsigned m = code & 7u;
    const float magnitude = m < 2u ? float(m) * 0.5f :
        __int_as_float(((126u + (m >> 1)) << 23) | ((m & 1u) << 22));
    return (code & 8u) ? -magnitude : magnitude;
}

__device__ float fp8(unsigned code) {
    const unsigned exponent = (code >> 3) & 15u, fraction = code & 7u;
    if (exponent == 0u) return float(fraction) / 512.0f;
    if (exponent == 15u && fraction == 7u) return __int_as_float(0x7fc00000u);
    return __int_as_float(((120u + exponent) << 23) | (fraction << 20));
}

__device__ half operand(float value, unsigned *bad) {
    const half result = __float2half_rn(value);
    if (!isfinite(value) || !isfinite(__half2float(result))) {
        atomicOr(bad, 1u);
        return __float2half_rn(0.0f);
    }
    return result;
}

__device__ bool same_expert(size_t a, size_t b, const unsigned char *const *packed,
                            const unsigned char *const *scales, const float *global) {
    return packed[a] == packed[b] && scales[a] == scales[b] && global[a] == global[b];
}

__global__ void grouped(const unsigned char *const *packed,
                        const unsigned char *const *scales, const float *global,
                        const float *x, float *y, unsigned *bad,
                        size_t n, size_t rows, size_t columns) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    __shared__ __align__(32) half a[kM * kK];
    __shared__ __align__(32) half b[kN * kK];
    __shared__ __align__(32) float c[kM * kN];
    const unsigned warp = threadIdx.x / 32;
    const size_t output_base = size_t(blockIdx.x) * kN;
    for (size_t first = blockIdx.y; first < n; first += gridDim.y) {
        // Uniform CTA decisions keep all warp collectives and barriers converged.
        if (first && same_expert(first, first - 1, packed, scales, global)) continue;
        size_t end = first + 1;
        while (end < n && same_expert(first, end, packed, scales, global)) ++end;
        for (size_t token = first; token < end; token += kM) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum;
            wmma::fill_fragment(accum, 0.0f);
            for (size_t base = 0; base < columns; base += kK) {
                for (unsigned i = threadIdx.x; i < kM * kK; i += blockDim.x) {
                    const size_t pair = token + i / kK;
                    a[i] = pair < end ? operand(x[pair * columns + base + i % kK], bad) :
                        __float2half_rn(0.0f);
                }
                for (unsigned i = threadIdx.x; i < kN * kK; i += blockDim.x) {
                    const size_t row = output_base + i / kK;
                    float value = 0.0f;
                    if (row < rows) {
                        const unsigned code = scales[first][row * (columns / 16) + base / 16];
                        const unsigned byte = packed[first][row * (columns / 2) + (base + i % kK) / 2];
                        const unsigned nibble = (byte >> ((i & 1u) * 4u)) & 15u;
                        const float scale = fp8(code) * global[first];
                        if ((code & 128u) || !isfinite(scale) || !(global[first] > 0.0f)) atomicOr(bad, 1u);
                        value = fp4(nibble) * scale;
                    }
                    b[i] = operand(value, bad);
                }
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> left;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> right;
                wmma::load_matrix_sync(left, a, 16);
                wmma::load_matrix_sync(right, b + warp * 256, 16);
                wmma::mma_sync(accum, left, right, accum);
                __syncthreads();
            }
            wmma::store_matrix_sync(c + warp * 256, accum, 16, wmma::mem_row_major);
            __syncthreads();
            for (unsigned i = threadIdx.x; i < kM * kN; i += blockDim.x) {
                const size_t pair = token + (i % 256) / 16;
                const size_t row = output_base + (i / 256) * 16 + i % 16;
                if (pair < end && row < rows) {
                    if (!isfinite(c[i])) atomicOr(bad, 1u);
                    y[pair * rows + row] = c[i];
                }
            }
            __syncthreads();
        }
    }
#else
    if (threadIdx.x == 0) atomicOr(bad, 1u);
#endif
}

struct Needs { size_t pointers, multipliers, x, y; };
bool needs(size_t n, size_t rows, size_t columns, Needs &v) {
    if (!n || !rows || !columns || columns % 16 ||
        n > SIZE_MAX / sizeof(void *) || n > SIZE_MAX / sizeof(float) ||
        rows > SIZE_MAX / columns || n > SIZE_MAX / columns ||
        n * columns > SIZE_MAX / sizeof(float) || n > SIZE_MAX / rows ||
        n * rows > SIZE_MAX / sizeof(float) || (rows - 1) / kN + 1 > 2147483647u)
        return false;
    v = {n * sizeof(void *), n * sizeof(float), n * columns * sizeof(float), n * rows * sizeof(float)};
    return true;
}
bool usable(const void *p, size_t bytes, size_t required, size_t alignment) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    return p && a % alignment == 0 && bytes >= required && bytes <= UINTPTR_MAX - a;
}
bool apart(const void *a, size_t na, const void *b, size_t nb) {
    return reinterpret_cast<uintptr_t>(a) + na <= reinterpret_cast<uintptr_t>(b) ||
           reinterpret_cast<uintptr_t>(b) + nb <= reinterpret_cast<uintptr_t>(a);
}
bool valid(const void *p, size_t np, const void *s, size_t ns, const void *g, size_t ng,
           const inference_cuda_workspace *w, const Needs &v) {
    if (!w || !usable(p,np,v.pointers,alignof(void *)) || !usable(s,ns,v.pointers,alignof(void *)) ||
        !usable(g,ng,v.multipliers,alignof(float)) || !usable(w->x,w->x_bytes,v.x,alignof(float)) ||
        !usable(w->y,w->y_bytes,v.y,alignof(float)) || !usable(w->xf,w->xf_bytes,4,alignof(unsigned)) ||
        !apart(w->x,v.x,w->y,v.y) || !apart(w->x,v.x,w->xf,4) || !apart(w->y,v.y,w->xf,4))
        return false;
    const void *tables[] = {p,s,g};
    const size_t sizes[] = {v.pointers,v.pointers,v.multipliers};
    for (unsigned i=0;i<3;++i)
        if (!apart(tables[i],sizes[i],w->x,v.x) || !apart(tables[i],sizes[i],w->y,v.y) ||
            !apart(tables[i],sizes[i],w->xf,4)) return false;
    return apart(p,v.pointers,s,v.pointers) && apart(p,v.pointers,g,v.multipliers) &&
           apart(s,v.pointers,g,v.multipliers);
}
cudaError_t enqueue(const void *const *p, const void *const *s, const float *g,
                    size_t n, size_t rows, size_t columns,
                    const inference_cuda_workspace *w, cudaStream_t stream) {
    const cudaError_t cleared = cudaMemsetAsync(w->xf,0,4,stream);
    if (cleared != cudaSuccess) return cleared;
    const dim3 grid(unsigned((rows-1)/kN+1),unsigned(n < 65535 ? n : 65535));
    grouped<<<grid,kThreads,0,stream>>>(
        reinterpret_cast<const unsigned char *const *>(p),
        reinterpret_cast<const unsigned char *const *>(s),g,
        static_cast<const float *>(w->x),static_cast<float *>(w->y),
        static_cast<unsigned *>(w->xf),n,rows,columns);
    return cudaGetLastError();
}
} // namespace

extern "C" int inference_cuda_nvfp4_grouped_gemm_launch(
    const void *const *p,size_t np,const void *const *s,size_t ns,const float *g,size_t ng,
    size_t n,size_t rows,size_t columns,const inference_cuda_workspace *w,void *stream,int *stage) {
    report_stage(stage,INFERENCE_CUDA_STAGE_NONE);
    Needs v;
    if (!needs(n,rows,columns,v) || !valid(p,np,s,ns,g,ng,w,v)) return INFERENCE_CUDA_INVALID_ARGUMENT;
    (void)cudaGetLastError();
    const cudaError_t status = enqueue(p,s,g,n,rows,columns,w,reinterpret_cast<cudaStream_t>(stream));
    if (status != cudaSuccess) report_stage(stage,INFERENCE_CUDA_STAGE_LAUNCH);
    return int(status);
}

extern "C" int inference_cuda_nvfp4_grouped_gemm_ws(
    const void *const *p,size_t np,const void *const *s,size_t ns,const float *g,size_t ng,
    size_t n,size_t rows,size_t columns,const float *x,size_t nx,float *y,size_t ny,
    const inference_cuda_workspace *w,int *stage) {
    Failure failure;
    bool started = false;
    report_stage(stage,INFERENCE_CUDA_STAGE_NONE);
    try {
        Needs v;
        if (!needs(n,rows,columns,v) || !valid(p,np,s,ns,g,ng,w,v) ||
            !usable(x,nx,v.x,alignof(float)) || !usable(y,ny,v.y,alignof(float)))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        for (size_t i=0;i<v.x/sizeof(float);++i)
            if (!std::isfinite(x[i])) return INFERENCE_CUDA_INVALID_ARGUMENT;
        (void)cudaGetLastError();
        started = true;
        failure.record(cudaMemcpy(w->x,x,v.x,cudaMemcpyHostToDevice),INFERENCE_CUDA_STAGE_COPY);
        if (failure.ok()) failure.record(enqueue(p,s,g,n,rows,columns,w,nullptr),INFERENCE_CUDA_STAGE_LAUNCH);
        failure.record(cudaDeviceSynchronize(),INFERENCE_CUDA_STAGE_SYNCHRONIZE);
        if (failure.ok()) {
            unsigned bad = 0;
            failure.record(cudaMemcpy(&bad,w->xf,4,cudaMemcpyDeviceToHost),INFERENCE_CUDA_STAGE_COPY);
            if (failure.ok() && bad) return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (failure.ok()) failure.record(cudaMemcpy(y,w->y,v.y,cudaMemcpyDeviceToHost),INFERENCE_CUDA_STAGE_COPY);
        (void)cudaGetLastError();
        report_stage(stage,failure.stage);
        return int(failure.err);
    } catch (...) {
        if (started) { (void)cudaDeviceSynchronize(); (void)cudaGetLastError(); }
        report_stage(stage,failure.stage);
        return failure.ok() ? INFERENCE_CUDA_NATIVE_EXCEPTION : int(failure.err);
    }
}
