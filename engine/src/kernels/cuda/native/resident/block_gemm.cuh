/* Tensor-core batched GEMM kernels for block_ops (sm_80 and later), with scalar fallbacks for
 * older architectures. Included by block_ops.cu inside its anonymous namespace after the kMla
 * constants and warp_sum helper.
 *
 * 1. f16_gemm_tc_kernel: Y[n, rows] = X[n, cols] * W[rows, cols]^T (X f32, W f16, Y f32).
 *    X tiles round to f16 on load into shared memory; products accumulate in f32 via m16n8k16.
 * 2. tf32_batched_gemm_kernel: per-head strided f32 GEMM C[h] = A[h] * B[h]^T (plus optional
 *    trailing rope copy when rope > 0), converting A and B to tf32 on load into shared memory and
 *    accumulating in f32 via m16n8k8 (full f32 exponent range, 10-bit mantissa). Serves both
 *    MLA_Q_ABSORB_CHUNK and MLA_PROJECT_CHUNK. */

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ unsigned f32_to_tf32(float v) {
    unsigned r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v));
    return r;
}

__device__ __forceinline__ void mma_1688_tf32(float *d, const unsigned *a, unsigned b0,
                                              unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#endif

/* ---- F16 GEMM (X f32, W f16 -> Y f32) ------------------------------------------------ */

constexpr int kF16GemmBm = 32;
constexpr int kF16GemmBn = 32;
constexpr int kF16GemmBk = 32;
/* Stride 40 halfs = 80 bytes: bank-conflict-free for 8x8 b16 ldmatrix across 8 rows. */
constexpr int kF16GemmStride = 40;

__global__ void __launch_bounds__(128) f16_gemm_tc_kernel(const float *__restrict__ x,
                                                          const __half *__restrict__ w,
                                                          float *__restrict__ y, int n, int rows,
                                                          int cols) {
#if __CUDA_ARCH__ >= 800
    using inference::mma::ldsm_x4;
    using inference::mma::mma_16816;
    using inference::mma::smem_addr;

    __shared__ __align__(16) __half xs[kF16GemmBm][kF16GemmStride];
    __shared__ __align__(16) __half ws[kF16GemmBn][kF16GemmStride];

    const int n0 = (int)blockIdx.x * kF16GemmBn;
    const int t0 = (int)blockIdx.y * kF16GemmBm;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = (warp >> 1) * 16; /* 0 or 16 */
    const int wn = (warp & 1) * 16;  /* 0 or 16 */
    const int mi = lane >> 3, rr = lane & 7;
    const int load_r = tid >> 2;
    const int load_c = (tid & 3) * 8;
    const int gx_r = t0 + load_r;
    const int gw_r = n0 + load_r;
    const bool x_aligned = ((cols & 3) == 0) && (((uintptr_t)x & 15) == 0);
    const bool w_aligned = ((cols & 7) == 0) && (((uintptr_t)w & 15) == 0);

    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

    for (int k = 0; k < cols; k += kF16GemmBk) {
        const bool full_k = (k + kF16GemmBk <= cols);
        if (full_k && x_aligned) {
            if (gx_r < n) {
                const float4 *xp =
                    reinterpret_cast<const float4 *>(x + (size_t)gx_r * cols + k + load_c);
                const float4 v0 = xp[0], v1 = xp[1];
                __half2 *dst = reinterpret_cast<__half2 *>(&xs[load_r][load_c]);
                dst[0] = __floats2half2_rn(v0.x, v0.y);
                dst[1] = __floats2half2_rn(v0.z, v0.w);
                dst[2] = __floats2half2_rn(v1.x, v1.y);
                dst[3] = __floats2half2_rn(v1.z, v1.w);
            } else {
                *reinterpret_cast<uint4 *>(&xs[load_r][load_c]) = make_uint4(0, 0, 0, 0);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int gc = k + load_c + j;
                const float v = (gx_r < n && gc < cols) ? x[(size_t)gx_r * cols + gc] : 0.f;
                xs[load_r][load_c + j] = __float2half_rn(v);
            }
        }

        if (full_k && w_aligned) {
            if (gw_r < rows) {
                *reinterpret_cast<uint4 *>(&ws[load_r][load_c]) =
                    *reinterpret_cast<const uint4 *>(w + (size_t)gw_r * cols + k + load_c);
            } else {
                *reinterpret_cast<uint4 *>(&ws[load_r][load_c]) = make_uint4(0, 0, 0, 0);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int gc = k + load_c + j;
                ws[load_r][load_c + j] =
                    (gw_r < rows && gc < cols) ? w[(size_t)gw_r * cols + gc] : __float2half_rn(0.f);
            }
        }
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            unsigned af[4], bf[4];
            ldsm_x4(smem_addr(&xs[wm + 8 * (mi & 1) + rr][ks * 16 + 8 * (mi >> 1)]), af);
            ldsm_x4(smem_addr(&ws[wn + 8 * (mi >> 1) + rr][ks * 16 + 8 * (mi & 1)]), bf);
            mma_16816(acc[0], af, bf[0], bf[1]);
            mma_16816(acc[1], af, bf[2], bf[3]);
        }
        __syncthreads();
    }

    const int g = lane >> 2, c = (lane & 3) * 2;
    const int r0 = t0 + wm + g;
    const int r1 = r0 + 8;
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
        const int col = n0 + wn + nb * 8 + c;
        if (r0 < n) {
            if (col < rows) y[(size_t)r0 * rows + col] = acc[nb][0];
            if (col + 1 < rows) y[(size_t)r0 * rows + col + 1] = acc[nb][1];
        }
        if (r1 < n) {
            if (col < rows) y[(size_t)r1 * rows + col] = acc[nb][2];
            if (col + 1 < rows) y[(size_t)r1 * rows + col + 1] = acc[nb][3];
        }
    }
#else
    __trap();
#endif
}

__global__ void __launch_bounds__(256) f16_gemm_fallback_kernel(const float *__restrict__ x,
                                                                const __half *__restrict__ w,
                                                                float *__restrict__ y, int n,
                                                                int rows, int cols) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r = (int)blockIdx.x * 8 + warp;
    const int t = (int)blockIdx.y;
    if (r >= rows || t >= n) return;
    const float *xr = x + (size_t)t * cols;
    const __half *wr = w + (size_t)r * cols;
    float acc = 0.f;
    for (int c = lane; c < cols; c += 32) acc = fmaf(__half2float(wr[c]), xr[c], acc);
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t)t * rows + r] = acc;
}

/* ---- Batched TF32 GEMM for MLA_Q_ABSORB_CHUNK and MLA_PROJECT_CHUNK ------------------- */

constexpr int kTf32Bm = 32;
constexpr int kTf32Bn = 64;
constexpr int kTf32Bk = 16;
/* Stride 20 uint32 words = 80 bytes: bank-conflict-free for 8x8 b16 ldmatrix across 8 rows. */
constexpr int kTf32Stride = 20;

/* Computes for each head h in 0..heads:
 *   C[t][h][0..N-1] = A[t][h][0..K-1] * B[h][0..N-1][0..K-1]^T
 * and when rope > 0, copies C[t][h][N..N+rope-1] = A[t][h][K..K+rope-1].
 * Requires K % 16 == 0 and 16-byte aligned B. */
__global__ void __launch_bounds__(256)
tf32_batched_gemm_kernel(const float *__restrict__ a_base, const float *__restrict__ b_base,
                         float *__restrict__ c_base, int n, int N, int K, int rope,
                         int a_head_stride, int a_tok_stride, int c_head_stride, int c_tok_stride) {
#if __CUDA_ARCH__ >= 800
    using inference::mma::ldsm_x4;
    using inference::mma::smem_addr;

    __shared__ __align__(16) unsigned as[kTf32Bm][kTf32Stride];
    __shared__ __align__(16) unsigned bs[kTf32Bn][kTf32Stride];

    const int n0 = (int)blockIdx.x * kTf32Bn;
    const int t0 = (int)blockIdx.y * kTf32Bm;
    const int h = (int)blockIdx.z;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = (warp >> 2) * 16; /* 0 or 16 */
    const int wn = (warp & 3) * 16;  /* 0, 16, 32, 48 */
    const int mi = lane >> 3, rr = lane & 7;

    const float *a_head = a_base + (size_t)h * a_head_stride;
    const float *b_head = b_base + (size_t)h * N * K;
    float *c_head = c_base + (size_t)h * c_head_stride;

    if (rope > 0 && blockIdx.x == 0) {
        const int cnt = min(kTf32Bm, n - t0);
        const int total = cnt * rope;
        for (int idx = tid; idx < total; idx += 256) {
            const int r = idx / rope, j = idx % rope;
            c_head[(size_t)(t0 + r) * c_tok_stride + N + j] =
                a_head[(size_t)(t0 + r) * a_tok_stride + K + j];
        }
    }

    const int ar = tid >> 3, ac = (tid & 7) * 2;
    const int br = tid >> 2, bc = (tid & 3) * 4;
    const int gar = t0 + ar;
    const int gbr = n0 + br;
    const bool a_vec = (((uintptr_t)a_base & 7) == 0) && ((a_head_stride & 1) == 0) &&
                       ((a_tok_stride & 1) == 0);

    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

    for (int k = 0; k < K; k += kTf32Bk) {
        if (gar < n) {
            const float *ap = a_head + (size_t)gar * a_tok_stride + k + ac;
            if (a_vec) {
                const float2 v = *reinterpret_cast<const float2 *>(ap);
                as[ar][ac] = f32_to_tf32(v.x);
                as[ar][ac + 1] = f32_to_tf32(v.y);
            } else {
                as[ar][ac] = f32_to_tf32(ap[0]);
                as[ar][ac + 1] = f32_to_tf32(ap[1]);
            }
        } else {
            as[ar][ac] = 0u;
            as[ar][ac + 1] = 0u;
        }

        if (gbr < N) {
            const float4 v =
                *reinterpret_cast<const float4 *>(b_head + (size_t)gbr * K + k + bc);
            bs[br][bc] = f32_to_tf32(v.x);
            bs[br][bc + 1] = f32_to_tf32(v.y);
            bs[br][bc + 2] = f32_to_tf32(v.z);
            bs[br][bc + 3] = f32_to_tf32(v.w);
        } else {
            bs[br][bc] = 0u;
            bs[br][bc + 1] = 0u;
            bs[br][bc + 2] = 0u;
            bs[br][bc + 3] = 0u;
        }
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            unsigned af[4], bf[4];
            ldsm_x4(smem_addr(&as[wm + 8 * (mi & 1) + rr][ks * 8 + 4 * (mi >> 1)]), af);
            ldsm_x4(smem_addr(&bs[wn + 8 * (mi >> 1) + rr][ks * 8 + 4 * (mi & 1)]), bf);
            mma_1688_tf32(acc[0], af, bf[0], bf[1]);
            mma_1688_tf32(acc[1], af, bf[2], bf[3]);
        }
        __syncthreads();
    }

    const int g = lane >> 2, c = (lane & 3) * 2;
    const int r0 = t0 + wm + g;
    const int r1 = r0 + 8;
    const bool c_vec = (((uintptr_t)c_base & 7) == 0) && ((c_head_stride & 1) == 0) &&
                       ((c_tok_stride & 1) == 0);
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
        const int col = n0 + wn + nb * 8 + c;
        if (r0 < n) {
            float *dst0 = c_head + (size_t)r0 * c_tok_stride + col;
            if (c_vec && col + 1 < N) {
                *reinterpret_cast<float2 *>(dst0) = make_float2(acc[nb][0], acc[nb][1]);
            } else {
                if (col < N) dst0[0] = acc[nb][0];
                if (col + 1 < N) dst0[1] = acc[nb][1];
            }
        }
        if (r1 < n) {
            float *dst1 = c_head + (size_t)r1 * c_tok_stride + col;
            if (c_vec && col + 1 < N) {
                *reinterpret_cast<float2 *>(dst1) = make_float2(acc[nb][2], acc[nb][3]);
            } else {
                if (col < N) dst1[0] = acc[nb][2];
                if (col + 1 < N) dst1[1] = acc[nb][3];
            }
        }
    }
#else
    __trap();
#endif
}
