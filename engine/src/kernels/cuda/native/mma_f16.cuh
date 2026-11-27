/* The f16 tensor-core primitives the attention and scan kernels share: the shared-memory
 * address of a pointer, the 8x8 ldmatrix loads and the m16n8k16 MMA with f32 accumulation.
 * They exist for sm_75 and later only, so no older device pass sees one; a caller guards its
 * uses the same way. */
#ifndef INFERENCE_CUDA_MMA_F16_CUH
#define INFERENCE_CUDA_MMA_F16_CUH

#if __CUDA_ARCH__ >= 750
namespace inference {
namespace mma {

__device__ __forceinline__ unsigned smem_addr(const void *p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

/* Four 8x8 b16 matrices; lanes 8i..8i+7 give the row addresses of matrix i, and lane j gets
 * row j/4, columns 2(j%4) and 2(j%4)+1 of each. */
__device__ __forceinline__ void ldsm_x4(unsigned addr, unsigned (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

/* The same, transposed: lane j gets rows 2(j%4), 2(j%4)+1 of column j/4. */
__device__ __forceinline__ void ldsm_x4_trans(unsigned addr, unsigned (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

/* D (16x8 f32) += A (16x16 f16, row) * B (16x8 f16, col). Lane j = 4g + t holds A rows g and
 * g + 8 at columns 2t, 2t + 1 (a[0], a[1]) and 2t + 8, 2t + 9 (a[2], a[3]), B column g at
 * rows 2t, 2t + 1 (b0) and 2t + 8, 2t + 9 (b1), and D rows g (d[0], d[1]) and g + 8 (d[2],
 * d[3]) at columns 2t, 2t + 1. Turing issues it as two m16n8k8 halves. */
__device__ __forceinline__ void mma_16816(float *d, const unsigned *a, unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    // PTX m16n8k8 stores the first eight columns in A[0:2], B[0],
    // and the next eight in A[2:4], B[1]; C/D has the same four f32 lanes.
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%7,%8}, {%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b0), "r"(a[2]), "r"(a[3]), "r"(b1));
#endif
}

}  // namespace mma
}  // namespace inference
#endif

#endif /* INFERENCE_CUDA_MMA_F16_CUH */
