/* Tiled unabsorbed multi-head latent attention kernels for prefill chunks (sm_80 and later, with
 * scalar fallbacks for older architectures). Included by block_ops.cu inside its anonymous
 * namespace after mla_attn_chunk.cuh and block_gemm.cuh.
 *
 * Instead of absorbing W_k into Q (operating over 576 score + 512 value = 1,088 dims per head),
 * chunks expand a tile of cached 512-dim latents on the fly into per-head f16 keys (128 dims) and
 * values (128 dims), then run FlashAttention-2 split-Q over [Q_nope(128) | Q_rope(64)] against
 * [K_nope(128) | K_rope_shared(64)] and V(128) (320 total dims: 3.4x fewer attention FLOPs while
 * keeping the compact 576-f16 KV cache unchanged).
 *
 * 1. mla_kv_expand_kernel<TransposeW>:
 *    - TransposeW = true:  K[h][t][d] = sum_r cache[k_base + t][r] * W_k[h][r][d]
 *    - TransposeW = false: V[h][t][d] = sum_r cache[k_base + t][r] * W_v[h][d][r]
 *    Tile BM=64 x BN=64 x BK=32, 8 warps (256 threads), m16n8k16 fp16->f32 -> f16 output.
 *
 * 2. mla_attn_unabsorbed_tile_kernel:
 *    One key-tile pass over k_len expanded keys [k_base .. k_base + k_len).
 *    Tile Br=64 x Bc=32, 4 warps (128 threads, 16 query rows per warp), K/V cp.async pipelined in
 *    20,480 B static shared memory (K_nope 8 KiB + K_rope 4 KiB + V 8 KiB). Q stays in registers;
 *    softmax runs purely in-warp (2 shfl_xor reductions, zero P smem, zero named barriers).
 *    Normalizes O /= l_run on the last tile visible to each query block and skips later tiles. */

constexpr int kUnabExpBm = 64;
constexpr int kUnabExpBn = 64;
constexpr int kUnabExpBk = 32;
/* Stride 40 halfs (80 B = 20 banks): bank-conflict-free for 8x8 b16 ldmatrix across 8 rows. */
constexpr int kUnabExpStride = 40;

constexpr int kUnabBr = 64;
constexpr int kUnabBc = 32;
constexpr int kUnabThreads = 128;
constexpr int kUnabNope = 128;
constexpr int kUnabRope = 64;
constexpr int kUnabV = 128;
constexpr int kUnabQDim = kUnabNope + kUnabRope;

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ unsigned unab_swizzle16(int row, int chunk) {
    return (unsigned)(row * (kUnabNope * 2) + ((chunk ^ (row & 7)) << 4));
}

__device__ __forceinline__ unsigned unab_swizzle8(int row, int chunk) {
    return (unsigned)(row * (kUnabRope * 2) + ((chunk ^ (row & 7)) << 4));
}

__device__ inline void unab_issue_k(const __half *k_nope_h, const __half *cache, int k_base,
                                    int base, int k_hi, unsigned k_nope_smem,
                                    unsigned k_rope_smem) {
    /* K_nope: 32 rows x 16 chunks (16 B each) = 512 chunks -> 4 per thread. */
#pragma unroll
    for (int i = threadIdx.x; i < kUnabBc * (kUnabNope / 8); i += kUnabThreads) {
        const int key = i >> 4, ch = i & 15;
        const bool valid = base + key < k_hi;
        const __half *src = k_nope_h + (size_t)(base + (valid ? key : 0)) * kUnabNope + ch * 8;
        cp_async16(k_nope_smem + unab_swizzle16(key, ch), src, valid ? 16 : 0);
    }
    /* K_rope: 32 rows x 8 chunks (16 B each) = 256 chunks -> 2 per thread, from shared cache. */
#pragma unroll
    for (int i = threadIdx.x; i < kUnabBc * (kUnabRope / 8); i += kUnabThreads) {
        const int key = i >> 3, ch = i & 7;
        const bool valid = base + key < k_hi;
        const __half *src =
            cache + (size_t)(k_base + base + (valid ? key : 0)) * kMlaWidth + kMlaRank + ch * 8;
        cp_async16(k_rope_smem + unab_swizzle8(key, ch), src, valid ? 16 : 0);
    }
    cp_async_commit();
}

__device__ inline void unab_issue_v(const __half *v_h, int base, int k_hi, unsigned v_smem) {
    /* V: 32 rows x 16 chunks (16 B each) = 512 chunks -> 4 per thread. */
#pragma unroll
    for (int i = threadIdx.x; i < kUnabBc * (kUnabV / 8); i += kUnabThreads) {
        const int key = i >> 4, ch = i & 15;
        const bool valid = base + key < k_hi;
        const __half *src = v_h + (size_t)(base + (valid ? key : 0)) * kUnabV + ch * 8;
        cp_async16(v_smem + unab_swizzle16(key, ch), src, valid ? 16 : 0);
    }
    cp_async_commit();
}
#endif

/* ---- Kernel 1: Expand cached 512-dim latents into per-head f16 K_nope or V ---------------- */

template <bool TransposeW>
__global__ void __launch_bounds__(256, 2)
mla_kv_expand_kernel(const __half *__restrict__ cache, const float *__restrict__ w,
                     __half *__restrict__ out, int k_len, int k_base) {
#if __CUDA_ARCH__ >= 800
    __shared__ __align__(16) __half as[kUnabExpBm][kUnabExpStride];
    /* TransposeW=true: bs_t[32][64] (swizzled 8 chunks/row); TransposeW=false: bs_n[64][40]. */
    __shared__ __align__(16) __half bs_raw[kUnabExpBn * kUnabExpStride];

    const int n0 = (int)blockIdx.x * kUnabExpBn;
    const int t0 = (int)blockIdx.y * kUnabExpBm;
    const int h = (int)blockIdx.z;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = (warp >> 1) * 16; /* 0, 16, 32, 48 */
    const int wn = (warp & 1) * 32;  /* 0, 32 */
    const int mi = lane >> 3, rr = lane & 7;

    const int d_out = kUnabNope; /* 128 for both K_nope and V */
    const float *w_head = w + (size_t)h * kMlaRank * d_out;
    __half *out_head = out + (size_t)h * k_len * d_out;

    const int ar = tid >> 2;
    const int ac = (tid & 3) * 8;
    const int gar = t0 + ar;

    float acc[4][4];
#pragma unroll
    for (int nb = 0; nb < 4; ++nb)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[nb][e] = 0.f;

    for (int k = 0; k < kMlaRank; k += kUnabExpBk) {
        if (gar < k_len) {
            *reinterpret_cast<uint4 *>(&as[ar][ac]) =
                *reinterpret_cast<const uint4 *>(cache + (size_t)(k_base + gar) * kMlaWidth + k + ac);
        } else {
            *reinterpret_cast<uint4 *>(&as[ar][ac]) = make_uint4(0, 0, 0, 0);
        }

        if constexpr (TransposeW) {
            /* w_head[r][d], r in [k..k+32), d in [n0..n0+64): 2 float4 loads per thread. */
            const int bc = (tid & 15) * 4;
            const int ch = bc >> 3;
            const int sub = bc & 4;
#pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                const int br = (tid >> 4) + 16 * pass;
                const float4 v = *reinterpret_cast<const float4 *>(
                    w_head + (size_t)(k + br) * d_out + n0 + bc);
                const int dst_col = ((ch ^ (br & 7)) << 3) + sub;
                const __half2 h0 = __floats2half2_rn(v.x, v.y);
                const __half2 h1 = __floats2half2_rn(v.z, v.w);
                *reinterpret_cast<uint2 *>(&bs_raw[br * kUnabExpBn + dst_col]) = make_uint2(
                    *reinterpret_cast<const unsigned *>(&h0),
                    *reinterpret_cast<const unsigned *>(&h1));
            }
        } else {
            /* w_head[d][r], d in [n0..n0+64), r in [k..k+32): 2 float4 loads per thread. */
            const int bc = (tid & 7) * 4;
#pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                const int br = (tid >> 3) + 32 * pass;
                const float4 v = *reinterpret_cast<const float4 *>(
                    w_head + (size_t)(n0 + br) * kMlaRank + k + bc);
                const __half2 h0 = __floats2half2_rn(v.x, v.y);
                const __half2 h1 = __floats2half2_rn(v.z, v.w);
                *reinterpret_cast<uint2 *>(&bs_raw[br * kUnabExpStride + bc]) = make_uint2(
                    *reinterpret_cast<const unsigned *>(&h0),
                    *reinterpret_cast<const unsigned *>(&h1));
            }
        }
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            unsigned af[4];
            ldsm_x4(smem_addr(&as[wm + 8 * (mi & 1) + rr][16 * ks + 8 * (mi >> 1)]), af);
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                unsigned bf[4];
                if constexpr (TransposeW) {
                    const int b_row = 16 * ks + 8 * (mi & 1) + rr;
                    const int b_ch = (((wn + 16 * np) >> 3) + (mi >> 1)) ^ (b_row & 7);
                    ldsm_x4_trans(smem_addr(&bs_raw[b_row * kUnabExpBn + (b_ch << 3)]), bf);
                } else {
                    const int b_row = wn + 16 * np + 8 * (mi >> 1) + rr;
                    const int b_col = 16 * ks + 8 * (mi & 1);
                    ldsm_x4(smem_addr(&bs_raw[b_row * kUnabExpStride + b_col]), bf);
                }
                mma_16816(acc[2 * np], af, bf[0], bf[1]);
                mma_16816(acc[2 * np + 1], af, bf[2], bf[3]);
            }
        }
        __syncthreads();
    }

    const int g = lane >> 2, c = (lane & 3) * 2;
    const int r0 = t0 + wm + g;
    const int r1 = r0 + 8;
#pragma unroll
    for (int nb = 0; nb < 4; ++nb) {
        const int col = n0 + wn + 8 * nb + c;
        if (r0 < k_len) {
            *reinterpret_cast<__half2 *>(out_head + (size_t)r0 * d_out + col) =
                __floats2half2_rn(acc[nb][0], acc[nb][1]);
        }
        if (r1 < k_len) {
            *reinterpret_cast<__half2 *>(out_head + (size_t)r1 * d_out + col) =
                __floats2half2_rn(acc[nb][2], acc[nb][3]);
        }
    }
#else
    __trap();
#endif
}

__global__ void __launch_bounds__(128)
mla_kv_expand_fallback_kernel(const __half *__restrict__ cache, const float *__restrict__ w,
                              __half *__restrict__ out, int k_len, int k_base, int transpose_w) {
    const int t = blockIdx.x, h = blockIdx.y;
    const int d = threadIdx.x; /* 0..127 */
    const __half *lat = cache + (size_t)(k_base + t) * kMlaWidth;
    const float *w_head = w + (size_t)h * kMlaRank * kUnabNope;
    float acc = 0.f;
    if (transpose_w) {
        for (int r = 0; r < kMlaRank; ++r)
            acc = fmaf(__half2float(lat[r]), w_head[(size_t)r * kUnabNope + d], acc);
    } else {
        const float *wr = w_head + (size_t)d * kMlaRank;
        for (int r = 0; r < kMlaRank; ++r) acc = fmaf(__half2float(lat[r]), wr[r], acc);
    }
    out[((size_t)h * k_len + t) * kUnabNope + d] = __float2half_rn(acc);
}

/* ---- Kernel 2: Tiled unabsorbed FlashAttention over one expanded KV tile ------------------ */

__global__ void __launch_bounds__(kUnabThreads, 2)
mla_attn_unabsorbed_tile_kernel(const float *__restrict__ qb, const __half *__restrict__ k_nope,
                                const __half *__restrict__ cache, const __half *__restrict__ v,
                                float *__restrict__ out, float *__restrict__ ml, int n, int heads,
                                int pos0, int k_base, int k_len, float scale) {
#if __CUDA_ARCH__ >= 800
    __shared__ __align__(16) unsigned char smem[kUnabBc * (kUnabNope + kUnabRope + kUnabV) * 2];

    const int tiles_q = (n + kUnabBr - 1) / kUnabBr;
    const int tile_q = tiles_q - 1 - (int)blockIdx.x;
    const int h = (int)blockIdx.y;
    const int t0 = tile_q * kUnabBr;
    const int t_end = min(n, t0 + kUnabBr);
    if (k_base >= pos0 + t_end) return;

    const int k_hi = min(k_len, pos0 + t_end - k_base);
    const bool first_tile = (k_base == 0);
    const bool last_for_block = (k_base + k_len >= pos0 + t_end);

    const __half *k_nope_h = k_nope + (size_t)h * k_len * kUnabNope;
    const __half *v_h = v + (size_t)h * k_len * kUnabV;

    const unsigned k_nope_smem = smem_addr(smem);
    const unsigned k_rope_smem = k_nope_smem + kUnabBc * kUnabNope * 2;
    const unsigned v_smem = k_rope_smem + kUnabBc * kUnabRope * 2;

    /* Kick off first K tile load before loading Q into registers. */
    unab_issue_k(k_nope_h, cache, k_base, 0, k_hi, k_nope_smem, k_rope_smem);

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, c = (lane & 3) * 2;
    const int tok[2] = {t0 + 16 * warp + g, t0 + 16 * warp + g + 8};
    const int p_row[2] = {pos0 + tok[0], pos0 + tok[1]};

    /* Load Q [16 x 192] per warp into A fragments (f16). */
    unsigned qf[kUnabQDim / 16][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float *q = qb + ((size_t)tok[i] * heads + h) * kUnabQDim + c;
#pragma unroll
        for (int ks = 0; ks < kUnabQDim / 16; ++ks) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                float2 val = make_float2(0.f, 0.f);
                if (tok[i] < t_end)
                    val = *reinterpret_cast<const float2 *>(q + 16 * ks + 8 * hh);
                const __half2 hv = __floats2half2_rn(val.x, val.y);
                qf[ks][i + 2 * hh] = *reinterpret_cast<const unsigned *>(&hv);
            }
        }
    }

    float o[kUnabV / 8][4];
    float m_run[2] = {-INFINITY, -INFINITY};
    float l_run[2] = {0.f, 0.f};
    if (first_tile) {
#pragma unroll
        for (int j = 0; j < kUnabV / 8; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) o[j][e] = 0.f;
    } else {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            if (tok[i] < t_end) {
                const float2 st =
                    *reinterpret_cast<const float2 *>(ml + ((size_t)tok[i] * heads + h) * 2);
                m_run[i] = st.x;
                l_run[i] = st.y;
                const float *src = out + ((size_t)tok[i] * heads + h) * kUnabV + c;
#pragma unroll
                for (int j = 0; j < kUnabV / 8; ++j) {
                    const float2 ov = *reinterpret_cast<const float2 *>(src + 8 * j);
                    o[j][2 * i] = ov.x;
                    o[j][2 * i + 1] = ov.y;
                }
            } else {
#pragma unroll
                for (int j = 0; j < kUnabV / 8; ++j) {
                    o[j][2 * i] = 0.f;
                    o[j][2 * i + 1] = 0.f;
                }
            }
        }
    }

    const int mi = lane >> 3, rr = lane & 7;
    const int k_key = 8 * (mi >> 1) + rr, k_c = mi & 1;
    const int v_key = 8 * (mi & 1) + rr, v_c = mi >> 1;

    for (int base = 0; base < k_hi; base += kUnabBc) {
        /* Wait for K(base) and ensure prior V consumers finished before issuing V(base). */
        cp_async_wait_all();
        __syncthreads();
        unab_issue_v(v_h, base, k_hi, v_smem);

        /* S = Q @ K^T: 12 k16 steps (8 nope + 4 rope) x 4 n8 key blocks. */
        float sq[4][4];
#pragma unroll
        for (int nb = 0; nb < 4; ++nb)
#pragma unroll
            for (int e = 0; e < 4; ++e) sq[nb][e] = 0.f;
#pragma unroll
        for (int ks = 0; ks < kUnabNope / 16; ++ks) {
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                unsigned kb[4];
                ldsm_x4(k_nope_smem + unab_swizzle16(16 * np + k_key, 2 * ks + k_c), kb);
                mma_16816(sq[2 * np], qf[ks], kb[0], kb[1]);
                mma_16816(sq[2 * np + 1], qf[ks], kb[2], kb[3]);
            }
        }
#pragma unroll
        for (int ks = 0; ks < kUnabRope / 16; ++ks) {
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                unsigned kb[4];
                ldsm_x4(k_rope_smem + unab_swizzle8(16 * np + k_key, 2 * ks + k_c), kb);
                mma_16816(sq[2 * np], qf[kUnabNope / 16 + ks], kb[0], kb[1]);
                mma_16816(sq[2 * np + 1], qf[kUnabNope / 16 + ks], kb[2], kb[3]);
            }
        }

        /* In-warp online softmax: C fragment layout matches P A fragment layout directly. */
        const bool full = (base + kUnabBc <= k_hi) && (k_base + base + kUnabBc - 1 <= pos0 + t0);
        float bm[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
            const int lk = base + 8 * nb + c;
            const int gk = k_base + lk;
            const bool v00 = full || (lk < k_hi && gk <= p_row[0]);
            const bool v01 = full || (lk + 1 < k_hi && gk + 1 <= p_row[0]);
            const bool v10 = full || (lk < k_hi && gk <= p_row[1]);
            const bool v11 = full || (lk + 1 < k_hi && gk + 1 <= p_row[1]);
            sq[nb][0] = v00 ? sq[nb][0] * scale : -INFINITY;
            sq[nb][1] = v01 ? sq[nb][1] * scale : -INFINITY;
            sq[nb][2] = v10 ? sq[nb][2] * scale : -INFINITY;
            sq[nb][3] = v11 ? sq[nb][3] * scale : -INFINITY;
            bm[0] = fmaxf(bm[0], fmaxf(sq[nb][0], sq[nb][1]));
            bm[1] = fmaxf(bm[1], fmaxf(sq[nb][2], sq[nb][3]));
        }
#pragma unroll
        for (int x = 1; x <= 2; x <<= 1) {
            bm[0] = fmaxf(bm[0], __shfl_xor_sync(0xffffffffu, bm[0], x));
            bm[1] = fmaxf(bm[1], __shfl_xor_sync(0xffffffffu, bm[1], x));
        }
        const float m_new[2] = {fmaxf(m_run[0], bm[0]), fmaxf(m_run[1], bm[1])};
        const float m_ref[2] = {m_new[0] == -INFINITY ? 0.f : m_new[0],
                                m_new[1] == -INFINITY ? 0.f : m_new[1]};
        const float corr[2] = {m_run[0] == -INFINITY ? 0.f : expf(m_run[0] - m_ref[0]),
                               m_run[1] == -INFINITY ? 0.f : expf(m_run[1] - m_ref[1])};

        unsigned pa[2][4];
        float ps[2] = {0.f, 0.f};
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int nb = 2 * ks + hh;
                const __half2 h0 =
                    __floats2half2_rn(expf(sq[nb][0] - m_ref[0]), expf(sq[nb][1] - m_ref[0]));
                const __half2 h1 =
                    __floats2half2_rn(expf(sq[nb][2] - m_ref[1]), expf(sq[nb][3] - m_ref[1]));
                pa[ks][2 * hh] = *reinterpret_cast<const unsigned *>(&h0);
                pa[ks][2 * hh + 1] = *reinterpret_cast<const unsigned *>(&h1);
                const float2 p0 = __half22float2(h0);
                const float2 p1 = __half22float2(h1);
                ps[0] += p0.x + p0.y;
                ps[1] += p1.x + p1.y;
            }
        }
#pragma unroll
        for (int x = 1; x <= 2; x <<= 1) {
            ps[0] += __shfl_xor_sync(0xffffffffu, ps[0], x);
            ps[1] += __shfl_xor_sync(0xffffffffu, ps[1], x);
        }
        l_run[0] = l_run[0] * corr[0] + ps[0];
        l_run[1] = l_run[1] * corr[1] + ps[1];
        m_run[0] = m_new[0];
        m_run[1] = m_new[1];

#pragma unroll
        for (int j = 0; j < kUnabV / 8; ++j) {
            o[j][0] *= corr[0];
            o[j][1] *= corr[0];
            o[j][2] *= corr[1];
            o[j][3] *= corr[1];
        }

        /* Wait for V(base) and ensure all warps finished reading K(base) before prefetching next K. */
        cp_async_wait_all();
        __syncthreads();
        if (base + kUnabBc < k_hi) {
            unab_issue_k(k_nope_h, cache, k_base, base + kUnabBc, k_hi, k_nope_smem, k_rope_smem);
        }

        /* O += P̂ @ V: 2 k16 steps x 8 pairs of n8 value blocks = 128 dims. */
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
            for (int jp = 0; jp < kUnabV / 16; ++jp) {
                unsigned vb[4];
                ldsm_x4_trans(v_smem + unab_swizzle16(16 * ks + v_key, 2 * jp + v_c), vb);
                mma_16816(o[2 * jp], pa[ks], vb[0], vb[1]);
                mma_16816(o[2 * jp + 1], pa[ks], vb[2], vb[3]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        if (tok[i] >= t_end) continue;
        const float inv_l = last_for_block ? (1.f / l_run[i]) : 1.f;
        float *dst = out + ((size_t)tok[i] * heads + h) * kUnabV + c;
#pragma unroll
        for (int j = 0; j < kUnabV / 8; ++j) {
            *reinterpret_cast<float2 *>(dst + 8 * j) =
                make_float2(o[j][2 * i] * inv_l, o[j][2 * i + 1] * inv_l);
        }
        if (!last_for_block && c == 0) {
            *reinterpret_cast<float2 *>(ml + ((size_t)tok[i] * heads + h) * 2) =
                make_float2(m_run[i], l_run[i]);
        }
    }
#else
    __trap();
#endif
}

__global__ void __launch_bounds__(128)
mla_attn_unabsorbed_fallback_kernel(const float *__restrict__ qb, const __half *__restrict__ k_nope,
                                    const __half *__restrict__ cache, const __half *__restrict__ v,
                                    float *__restrict__ out, float *__restrict__ ml, int heads,
                                    int pos0, int k_base, int k_len, float scale) {
    const int t = blockIdx.x, h = blockIdx.y;
    const int d = threadIdx.x; /* 0..127 */
    const int p_row = pos0 + t;
    if (k_base > p_row) return;
    const int k_hi = min(k_len, p_row + 1 - k_base);
    const bool first_tile = (k_base == 0);
    const bool last_for_row = (k_base + k_len > p_row);

    __shared__ float q_smem[kUnabQDim];
    __shared__ float s_red[4];
    const float *q = qb + ((size_t)t * heads + h) * kUnabQDim;
    q_smem[d] = q[d];
    if (d < kUnabRope) q_smem[kUnabNope + d] = q[kUnabNope + d];
    __syncthreads();

    float m = first_tile ? -INFINITY : ml[((size_t)t * heads + h) * 2];
    float l = first_tile ? 0.f : ml[((size_t)t * heads + h) * 2 + 1];
    float acc = first_tile ? 0.f : out[((size_t)t * heads + h) * kUnabV + d];

    const __half *kn_h = k_nope + (size_t)h * k_len * kUnabNope;
    const __half *v_h = v + (size_t)h * k_len * kUnabV;
    for (int k = 0; k < k_hi; ++k) {
        float part = q_smem[d] * __half2float(kn_h[(size_t)k * kUnabNope + d]);
        if (d < kUnabRope) {
            part += q_smem[kUnabNope + d] *
                    __half2float(cache[(size_t)(k_base + k) * kMlaWidth + kMlaRank + d]);
        }
        const float sc = block_sum(part, s_red) * scale;
        const float m_new = fmaxf(m, sc);
        const float corr = m == -INFINITY ? 0.f : expf(m - m_new);
        const float p = expf(sc - m_new);
        l = l * corr + p;
        m = m_new;
        acc = acc * corr + p * __half2float(v_h[(size_t)k * kUnabV + d]);
    }
    const float inv_l = last_for_row ? (1.f / l) : 1.f;
    out[((size_t)t * heads + h) * kUnabV + d] = acc * inv_l;
    if (!last_for_row && d == 0) {
        ml[((size_t)t * heads + h) * 2] = m;
        ml[((size_t)t * heads + h) * 2 + 1] = l;
    }
}
