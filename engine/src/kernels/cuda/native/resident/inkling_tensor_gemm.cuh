// SM75: each of eight warps owns a 16x16 accumulator. Two 32-column
// activation tiles can now share one weight decode; tail warps skip MMA.
template<int Codec>
__global__ void inkling_tensor_gemm(inference_cuda_dense_gemv_args a) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda::wmma;
    constexpr int K = Codec == 40 ? 1152 : 2304;
    constexpr int Block = Codec == 40 ? 64 : 256;
    constexpr int Bytes = Codec == 40 ? 36 : Codec == 17 ? 74 : 82;
    constexpr int BM = 64, BN = 32, BK = 64, Pitch = BK + 8;
    __shared__ __align__(32) __half x[BM * Pitch], w[BN * Pitch];
    __shared__ __align__(32) float out[BM * BN];
    struct Empty {};
    __shared__ std::conditional_t<Codec == 40, Empty, InklingCodebook<Codec>> codes;
    if constexpr (Codec != 40) codes.init();
    const int t = threadIdx.x, warp = t / 32;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    WmmaAccumulator acc;
    fill_fragment(acc, 0.f);
    for (int k = 0; k < K; k += BK) {
        const int n = t / (BK / 8), c = (t % (BK / 8)) * 8;
        const auto *wr = static_cast<const uint8_t *>(a.w) + size_t(n0 + n) * (K / Block * Bytes);
        __align__(16) __half octet[8];
        if constexpr (Codec == 40) inkling_decode_octet<Codec,Block,Bytes>(wr, k + c, octet);
        else codes.octet(wr, k + c, octet);
        *reinterpret_cast<uint4 *>(w + n * Pitch + c) = *reinterpret_cast<const uint4 *>(octet);
        for (int i = t; i < BM * BK / 2; i += 256) {
            const int row = i / (BK / 2), col = (i % (BK / 2)) * 2;
            __half2 v = __float2half2_rn(0.f);
            if (m0 + row < a.m) {
                const float *src = static_cast<const float *>(a.x) + size_t(m0 + row) * a.ld_x + k + col;
                v = __floats2half2_rn(src[0], src[1]);
            }
            *reinterpret_cast<__half2 *>(x + row * Pitch + col) = v;
        }
        __syncthreads();
        if (m0 + (warp / 2) * 16 < a.m) {
#pragma unroll
            for (int j = 0; j < BK; j += 16) {
                fragment<matrix_a,16,16,16,__half,row_major> l;
                fragment<matrix_b,16,16,16,__half,col_major> r;
                load_matrix_sync(l, x + (warp / 2) * 16 * Pitch + j, Pitch);
                load_matrix_sync(r, w + (warp % 2) * 16 * Pitch + j, Pitch);
                mma_sync(acc, l, r, acc);
            }
        }
        __syncthreads();
    }
    store_matrix_sync(out + (warp / 2) * 16 * BN + (warp % 2) * 16, acc, BN, mem_row_major);
    __syncthreads();
    for (int i = t; i < BM * BN; i += 256)
        if (m0 + i / BN < a.m) static_cast<float *>(a.y)[size_t(m0 + i / BN) * a.ld_y + n0 + i % BN] = out[i];
#endif
}

bool inkling_try_tensor_gemm(const inference_cuda_dense_gemv_args &a, int codec, cudaStream_t stream) {
    // Narrow IQ groups reuse little across the second half of this tile.
    if (!((codec == 40 && a.m >= 32) || ((codec == 17 || codec == 22) && a.m > 32 && a.m <= 64))) return false;
    if (!(a.flags & INFERENCE_CUDA_DENSE_GEMV_PREFILL_TENSOR_GEMM)) return false;
    const dim3 grid(72, (a.m + 63) / 64);
    if (codec == 40) inkling_tensor_gemm<40><<<grid,256,0,stream>>>(a);
    else if (codec == 17) inkling_tensor_gemm<17><<<grid,256,0,stream>>>(a);
    else inkling_tensor_gemm<22><<<grid,256,0,stream>>>(a);
    return true;
}
