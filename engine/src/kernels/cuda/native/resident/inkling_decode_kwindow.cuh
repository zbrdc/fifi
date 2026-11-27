// Four contiguous samples from each of two windows. Their quant codes share
// the same low-nibble bytes, so load both windows' codes in a single word.
__device__ __forceinline__ unsigned kw_read32(const uint8_t *p) {
    const auto *v = reinterpret_cast<const uint16_t *>(p);
    return unsigned(v[0]) | (unsigned(v[1]) << 16);
}

template<int Type, int Bytes, int Cols, int Kw, bool OneRow=false>
__device__ __forceinline__ void decode_kw_rows(inference_cuda_dense_gemv_args a, int block) {
    __shared__ float part[2][Kw];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = OneRow ? block : 2*block + warp/Kw;
    const int pair = lane/8, c = 4*(lane%8);
    const auto *w = static_cast<const uint8_t *>(a.w) + size_t(row)*(Cols/256)*Bytes;
    const auto *x = static_cast<const float4 *>(a.x);
    float acc = 0;
#pragma unroll
    for (int k = warp%Kw; k < Cols/256 && (!OneRow || warp<Kw); k += Kw) {
        const auto *b = w + k*Bytes;
        unsigned lo, hi = 0;
        if constexpr (Type == 14) {
            // Q6 halves pair windows (0,2), (1,3), (4,6), (5,7).
            // The 210-byte stride guarantees two-byte, not four-byte alignment.
            lo = kw_read32(b + (pair/2)*64 + (pair%2)*32 + c);
            hi = kw_read32(b + 128 + (pair/2)*32 + c);
        } else {
            lo = *reinterpret_cast<const unsigned *>(b + (Type==13 ? 48 : 16) + 32*pair + c);
            if constexpr (Type == 13) hi = *reinterpret_cast<const unsigned *>(b+16+c);
        }
#pragma unroll
        for (int j=0; j<2; ++j) {
            unsigned codes = (lo >> (4*j)) & 0x0f0f0f0fu;
            int win;
            float scale, minimum;
            if constexpr (Type == 14) {
                win = (pair/2)*4 + pair%2 + 2*j;
                codes |= ((hi >> (2*(win%4))) & 0x03030303u) << 4;
                codes = __vsub4(codes,0x20202020u);
                scale = __half2float(*reinterpret_cast<const __half *>(b+208)) * float(int8_t(b[192+2*win+c/16]));
                minimum = 0;
            } else {
                win = 2*pair + j;
                if constexpr (Type == 13) codes |= ((hi >> win) & 0x01010101u) << 4;
                unsigned s, m;
                scale_min_k4(b + 4, win, s, m);
                scale = __half2float(*reinterpret_cast<const __half *>(b))*float(s);
                minimum = __half2float(*reinterpret_cast<const __half *>(b+2))*float(m);
            }
            float4 v = x[64*k + 8*win + c/4];
            float dot = 0, sum = 0;
#pragma unroll
            for (int i=0; i<4; ++i) {
                const float u = reinterpret_cast<float *>(&v)[i];
                dot = fmaf(float(int8_t(codes>>(8*i))), u, dot);
                if constexpr (Type != 14) sum += u;
            }
            acc = fmaf(dot,scale,acc);
            if constexpr (Type != 14) acc = fmaf(-minimum,sum,acc);
        }
    }
    acc = warp_sum(acc);
    if (!lane) part[warp/Kw][warp%Kw] = acc;
    __syncthreads();
    if (warp%Kw==0 && !lane && (!OneRow || !warp)) {
        const auto *p = part[warp/Kw];
        float sum = p[0]+p[1];
        if constexpr (Kw==3) sum += p[2];
        static_cast<float *>(a.y)[row] = sum;
    }
}

template<int Type, int Bytes, int Cols, int Kw>
__global__ void decode_kwindow(inference_cuda_dense_gemv_args a) {
    decode_kw_rows<Type,Bytes,Cols,Kw>(a,blockIdx.x);
}

template<int Type, int Bytes>
void launch_kwindow(inference_cuda_dense_gemv_args a, cudaStream_t stream) {
    if (a.rows == 384) decode_kwindow<Type,Bytes,2304,3><<<a.rows/2,192,0,stream>>>(a);
    else if (a.cols == 2304) decode_kwindow<Type,Bytes,2304,2><<<a.rows/2,128,0,stream>>>(a);
    else decode_kwindow<Type,Bytes,3072,2><<<a.rows/2,128,0,stream>>>(a);
}
