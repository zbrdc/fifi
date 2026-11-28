/* The chunk form of the latent attention's score, on tensor cores (sm_80 and later). Included
 * by block_ops.cu inside its namespace, after the kMla constants.
 *
 * One block takes a query tile of 8 token rows x 8 heads = 64 rows, one key split, and walks
 * the split in tiles of 32 cached rows with the online-softmax recurrence. A cached row holds
 * the key ([latent | rotary], 576 f16) and the value (its latent part, the first 512), so one
 * shared-memory plane of a tile serves both products:
 *   S = Q̂ K^T   64 x 32 over 576 dimensions,
 *   O = O corr + P̂ V   64 x 512 over the 32 keys.
 * Q̂ is the absorbed query rounded to f16 once; P̂ is the tile's probability rounded to f16, and
 * the normalizer sums the rounded values, so the weights of a row still sum to one. Products
 * accumulate in f32.
 *
 * The 16 warps are 4 row groups (two token rows x 8 heads each) x 4 quarters of the
 * dimensions. Warp (rg, dq) keeps its quarter of Q̂ in registers (144 dimensions), computes its
 * quarter of S for all 32 keys, and owns O for 128 latent dimensions. The four quarter partials
 * of S are summed through shared memory in quarter order, so the four warps of a row group hold
 * the same statistics and P̂. Warp dq of a row group finishes the scores of key block dq (keys
 * 8 dq .. +7 of the tile).
 *
 * Shared memory: two stages of 32 rows x 1,152 B that cp.async fills one tile ahead, three
 * 512 B score slots per warp, and 256 B of row maxima per row group: 99,328 B. Chunk j (16 B)
 * of row k of a stage sits at k * 1,152 + ((j ^ (k & 7)) << 4), so the eight row addresses of
 * an 8x8 ldmatrix load fall in eight different bank groups. After the maxima are exchanged a row
 * group's score slots are dead until the next tile's block barrier; P̂ (16 rows of 80 B) and
 * the row-sum partials reuse them.
 *
 * The partials have the layout of mla_attn_kernel (per token, head group, split and head:
 * acc[512], running max, running sum), so mla_reduce_kernel combines either. */

constexpr int kMlaChunkTokens = 8;   /* token rows per query tile */
constexpr int kMlaChunkThreads = 512;
constexpr int kMlaRowBytes = kMlaWidth * 2;
constexpr int kMlaRowChunks = kMlaRowBytes / 16;
constexpr int kMlaStageBytes = kMlaTile * kMlaRowBytes;
constexpr int kMlaScoreQuarter = kMlaWidth / 4;  /* score dimensions of one warp */
constexpr int kMlaValueQuarter = kMlaRank / 4;   /* latent dimensions of one warp */
constexpr int kMlaSlotBytes = 32 * 16;           /* one float4 per lane */
constexpr int kMlaGroupSlots = 3 * 4;            /* score slots of one row group */
constexpr int kMlaProbRow = 80;                  /* bytes of one P̂ row: 32 f16 and a bank shift */
constexpr int kMlaChunkSmem =
    2 * kMlaStageBytes + 4 * kMlaGroupSlots * kMlaSlotBytes + 4 * 16 * 4 * 4;
static_assert(kMlaTile == 32 && kMlaGroup == 8, "the tile is 4 key blocks of 8, the group 8 heads");
static_assert(kMlaRowChunks % 8 == 0, "the swizzle stays inside a row");
static_assert(kMlaScoreQuarter % 16 == 0 && kMlaValueQuarter % 16 == 0, "whole MMA steps");
static_assert(16 * kMlaProbRow + 16 * 4 * 4 <= kMlaGroupSlots * kMlaSlotBytes,
              "P̂ and the sum partials fit a row group's score slots");
static_assert(kMlaChunkSmem == 99328, "chunk score shared memory");

#if __CUDA_ARCH__ >= 800
using inference::mma::bar_row_group;
using inference::mma::cp_async16;
using inference::mma::cp_async_commit;
using inference::mma::cp_async_wait_all;
using inference::mma::ldsm_x4;
using inference::mma::ldsm_x4_trans;
using inference::mma::mma_16816;
using inference::mma::smem_addr;

__device__ __forceinline__ unsigned mla_swizzle(int key, int chunk) {
    return key * kMlaRowBytes + ((chunk ^ (key & 7)) << 4);
}

/* Issues the cached rows base .. base + 31 into the stage at byte `stage`. A row at or past
 * `hi` copies zero bytes (zeros, no global read) from the tile's first row, so a row no query
 * may read never enters the tile. */
__device__ inline void mla_issue_tile(const __half *cache, int base, int hi, unsigned stage) {
#pragma unroll
    for (int i = threadIdx.x; i < kMlaTile * kMlaRowChunks; i += kMlaChunkThreads) {
        const int key = i / kMlaRowChunks, chunk = i % kMlaRowChunks;
        const bool valid = base + key < hi;
        const __half *src = cache + (size_t)(base + (valid ? key : 0)) * kMlaWidth + chunk * 8;
        cp_async16(stage + mla_swizzle(key, chunk), src, valid ? 16 : 0);
    }
    cp_async_commit();
}
#endif

/* Grid (token tiles, head groups, splits), the deepest tiles first. `n` token rows at positions
 * pos0 .., split sp over the cached rows [sp * keys_per_split, (sp + 1) * keys_per_split) cut
 * at the tile's last position. An image older than sm_80 traps: the launch entry never picks
 * this kernel there. */
__global__ void __launch_bounds__(kMlaChunkThreads, 1)
mla_attn_chunk_kernel(const float *qeff, const __half *cache, float *part, int n, int heads,
                      int pos0, int splits, int keys_per_split, float scale) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) unsigned char smem[];
    const int tiles = (n + kMlaChunkTokens - 1) / kMlaChunkTokens;
    const int tile = tiles - 1 - (int)blockIdx.x, grp = blockIdx.y, sp = blockIdx.z;
    const int groups = heads / kMlaGroup;
    const int t0 = tile * kMlaChunkTokens, t_end = min(n, t0 + kMlaChunkTokens);
    const int lo = sp * keys_per_split;
    const int hi = min(lo + keys_per_split, pos0 + t_end);
    const unsigned stages = smem_addr(smem);
    if (lo < hi) mla_issue_tile(cache, lo, hi, stages);

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rg = warp >> 2, dq = warp & 3;
    const int g = lane >> 2, c = (lane & 3) * 2;
    const int tok[2] = {t0 + 2 * rg, t0 + 2 * rg + 1};
    float4 *const xs = reinterpret_cast<float4 *>(smem + 2 * kMlaStageBytes) +
                       rg * (kMlaGroupSlots * 32);
    float *const xmax = reinterpret_cast<float *>(smem + 2 * kMlaStageBytes +
                                                  4 * kMlaGroupSlots * kMlaSlotBytes) +
                        rg * 64;
    unsigned char *const xp = reinterpret_cast<unsigned char *>(xs);
    float *const xsum = reinterpret_cast<float *>(xp + 16 * kMlaProbRow);

    /* Q̂ as A fragments: rows g (token tok[0]) and g + 8 (token tok[1]) of head 8 grp + g, at
     * dimensions 144 dq + 16 ks + {c, c + 1, c + 8, c + 9}. A token past the chunk holds zeros. */
    unsigned qf[kMlaScoreQuarter / 16][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float *q = qeff + ((size_t)tok[i] * heads + grp * kMlaGroup + g) * kMlaWidth +
                         kMlaScoreQuarter * dq + c;
#pragma unroll
        for (int ks = 0; ks < kMlaScoreQuarter / 16; ++ks)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                float2 v = make_float2(0.f, 0.f);
                if (tok[i] < t_end) v = *reinterpret_cast<const float2 *>(q + 16 * ks + 8 * hh);
                const __half2 h = __floats2half2_rn(v.x, v.y);
                qf[ks][i + 2 * hh] = *reinterpret_cast<const unsigned *>(&h);
            }
    }
    const int p_row[2] = {pos0 + tok[0], pos0 + tok[1]};
    /* ldmatrix lane roles: matrix mi = lane >> 3, row rr = lane & 7. K: key 16 np + 8 (mi >> 1)
     * + rr, chunk 18 dq + 2 ks + (mi & 1). V: key 16 ks + 8 (mi & 1) + rr, chunk 16 dq + 2 jp +
     * (mi >> 1). */
    const int mi = lane >> 3, rr = lane & 7;
    const int k_key = 8 * (mi >> 1) + rr, k_chunk = (kMlaScoreQuarter / 8) * dq + (mi & 1);
    const int v_key = 8 * (mi & 1) + rr, v_chunk = (kMlaValueQuarter / 8) * dq + (mi >> 1);

    float o[kMlaValueQuarter / 8][4];
#pragma unroll
    for (int j = 0; j < kMlaValueQuarter / 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[j][e] = 0.f;
    float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};

    int it = 0;
    for (int base = lo; base < hi; base += kMlaTile, ++it) {
        cp_async_wait_all();
        __syncthreads();
        if (base + kMlaTile < hi)
            mla_issue_tile(cache, base + kMlaTile, hi, stages + ((it + 1) & 1) * kMlaStageBytes);
        const unsigned kst = stages + (it & 1) * kMlaStageBytes;

        /* This quarter's S: 9 k16 steps x 4 key blocks. */
        float sq[4][4];
#pragma unroll
        for (int nb = 0; nb < 4; ++nb)
#pragma unroll
            for (int e = 0; e < 4; ++e) sq[nb][e] = 0.f;
#pragma unroll
        for (int ks = 0; ks < kMlaScoreQuarter / 16; ++ks)
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                unsigned kb[4];
                ldsm_x4(kst + mla_swizzle(16 * np + k_key, k_chunk + 2 * ks), kb);
                mma_16816(sq[2 * np], qf[ks], kb[0], kb[1]);
                mma_16816(sq[2 * np + 1], qf[ks], kb[2], kb[3]);
            }

        /* The exchange. Warp dq keeps key block dq in registers and stores the three blocks its
         * siblings own; slot (q, nb) of a row group is 3 q + nb - (nb > q). */
        float4 own = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
            const float4 v = make_float4(sq[nb][0], sq[nb][1], sq[nb][2], sq[nb][3]);
            if (nb == dq)
                own = v;
            else
                xs[(3 * dq + nb - (nb > dq ? 1 : 0)) * 32 + lane] = v;
        }
        bar_row_group(rg);  // the score partials are in place
        /* The own block over the four quarters in quarter order, ((S0 + S1) + S2) + S3. */
        float4 sum4 = dq == 0 ? own : xs[dq * 32 + lane - 32];
#pragma unroll
        for (int q = 1; q < 4; ++q) {
            const float4 v = q == dq ? own : xs[(3 * q + dq - (dq > q ? 1 : 0)) * 32 + lane];
            sum4.x += v.x;
            sum4.y += v.y;
            sum4.z += v.z;
            sum4.w += v.w;
        }
        /* The lane holds keys pos, pos + 1 of rows g and g + 8. A row sees the keys up to its
         * own position; a tile wholly before the block's first row skips the mask. */
        const int pos = base + 8 * dq + c;
        const bool full = base + kMlaTile - 1 <= pos0 + t0;
        const float s2[4] = {full || pos <= p_row[0] ? sum4.x * scale : -INFINITY,
                             full || pos + 1 <= p_row[0] ? sum4.y * scale : -INFINITY,
                             full || pos <= p_row[1] ? sum4.z * scale : -INFINITY,
                             full || pos + 1 <= p_row[1] ? sum4.w * scale : -INFINITY};
        float bm[2] = {fmaxf(s2[0], s2[1]), fmaxf(s2[2], s2[3])};
#pragma unroll
        for (int x = 1; x <= 2; x <<= 1) {
            bm[0] = fmaxf(bm[0], __shfl_xor_sync(0xffffffffu, bm[0], x));
            bm[1] = fmaxf(bm[1], __shfl_xor_sync(0xffffffffu, bm[1], x));
        }
        if (c == 0) {
            xmax[g * 4 + dq] = bm[0];
            xmax[(g + 8) * 4 + dq] = bm[1];
        }
        bar_row_group(rg);  // the block maxima are in place; the score slots are dead
        const float4 mq0 = *reinterpret_cast<const float4 *>(xmax + g * 4);
        const float4 mq1 = *reinterpret_cast<const float4 *>(xmax + (g + 8) * 4);
        const float m_new[2] = {
            fmaxf(m_run[0], fmaxf(fmaxf(mq0.x, mq0.y), fmaxf(mq0.z, mq0.w))),
            fmaxf(m_run[1], fmaxf(fmaxf(mq1.x, mq1.y), fmaxf(mq1.z, mq1.w)))};
        /* No key yet: exponentiate against 0, so p and corr are 0 and not NaN. */
        const float m_ref[2] = {m_new[0] == -INFINITY ? 0.f : m_new[0],
                                m_new[1] == -INFINITY ? 0.f : m_new[1]};
        const __half2 ph[2] = {
            __floats2half2_rn(expf(s2[0] - m_ref[0]), expf(s2[1] - m_ref[0])),
            __floats2half2_rn(expf(s2[2] - m_ref[1]), expf(s2[3] - m_ref[1]))};
        const float2 pr[2] = {__half22float2(ph[0]), __half22float2(ph[1])};
        float ps[2] = {pr[0].x + pr[0].y, pr[1].x + pr[1].y};
#pragma unroll
        for (int x = 1; x <= 2; x <<= 1) {
            ps[0] += __shfl_xor_sync(0xffffffffu, ps[0], x);
            ps[1] += __shfl_xor_sync(0xffffffffu, ps[1], x);
        }
        if (c == 0) {
            xsum[g * 4 + dq] = ps[0];
            xsum[(g + 8) * 4 + dq] = ps[1];
        }
        *reinterpret_cast<__half2 *>(xp + g * kMlaProbRow + 16 * dq + 2 * c) = ph[0];
        *reinterpret_cast<__half2 *>(xp + (g + 8) * kMlaProbRow + 16 * dq + 2 * c) = ph[1];
        const float corr[2] = {expf(m_run[0] - m_ref[0]), expf(m_run[1] - m_ref[1])};
        bar_row_group(rg);  // the sum partials and P̂ are in place
        const float4 sq0 = *reinterpret_cast<const float4 *>(xsum + g * 4);
        const float4 sq1 = *reinterpret_cast<const float4 *>(xsum + (g + 8) * 4);
        l_run[0] = l_run[0] * corr[0] + (((sq0.x + sq0.y) + sq0.z) + sq0.w);
        l_run[1] = l_run[1] * corr[1] + (((sq1.x + sq1.y) + sq1.z) + sq1.w);
        m_run[0] = m_new[0];
        m_run[1] = m_new[1];

        /* O = O corr + P̂ V over this warp's 128 latent dimensions: P̂ (16 rows x 32 keys) as A
         * fragments (matrix mi: rows 8 (mi & 1) .., keys 16 ks + 8 (mi >> 1) ..), V by the
         * transposed load, two blocks of 8 dimensions per load. */
#pragma unroll
        for (int j = 0; j < kMlaValueQuarter / 8; ++j) {
            o[j][0] *= corr[0];
            o[j][1] *= corr[0];
            o[j][2] *= corr[1];
            o[j][3] *= corr[1];
        }
        const unsigned p_lane = smem_addr(xp) + (rr + 8 * (mi & 1)) * kMlaProbRow + 16 * (mi >> 1);
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            unsigned pa[4];
            ldsm_x4(p_lane + 32 * ks, pa);
#pragma unroll
            for (int jp = 0; jp < kMlaValueQuarter / 16; ++jp) {
                unsigned vb[4];
                ldsm_x4_trans(kst + mla_swizzle(16 * ks + v_key, v_chunk + 2 * jp), vb);
                mma_16816(o[2 * jp], pa, vb[0], vb[1]);
                mma_16816(o[2 * jp + 1], pa, vb[2], vb[3]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        if (tok[i] >= t_end) continue;
        float *dst = part + ((((size_t)tok[i] * groups + grp) * splits + sp) * kMlaGroup + g) *
                                kMlaPartStride;
#pragma unroll
        for (int j = 0; j < kMlaValueQuarter / 8; ++j)
            *reinterpret_cast<float2 *>(dst + kMlaValueQuarter * dq + 8 * j + c) =
                make_float2(o[j][2 * i], o[j][2 * i + 1]);
        if (dq == 0 && c == 0) {
            dst[kMlaRank] = isnan(l_run[i]) ? NAN : m_run[i];
            dst[kMlaRank + 1] = l_run[i];
        }
    }
#else
    __trap();
#endif
}

/* Single-row shared-KV latent attention on tensor cores (sm_80 and later). One block covers up to
 * 64 heads of the query token across one key split (4 row groups x 16 heads), so up to 64 heads
 * share every KV tile load from DRAM while S = Q̂ K^T and O += P̂ V run on m16n8k16 tensor cores
 * with the same f16 rounding and f32 accumulation as mla_attn_chunk_kernel. Reads the current
 * position from pos_ptr when non-null for graph-captured decode steps. */
__global__ void __launch_bounds__(kMlaChunkThreads, 1)
mla_attn_decode_tc_kernel(const float *qeff, const __half *cache, float *part,
                          const int *pos_ptr, int heads, int splits, int pos_host, int cells,
                          float scale) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) unsigned char smem[];
    const int ht = blockIdx.x, sp = blockIdx.y;
    const int pos = pos_ptr != NULL ? *pos_ptr : pos_host;
    const int visible = (pos >= 0 && pos < cells) ? (pos + 1) : 0;
    const int per_split =
        visible > 0
            ? (((visible + splits - 1) / splits + kMlaTile - 1) / kMlaTile * kMlaTile)
            : 0;
    const int lo = sp * per_split;
    const int hi = min(lo + per_split, visible);
    const unsigned stages = smem_addr(smem);
    if (lo < hi) mla_issue_tile(cache, lo, hi, stages);

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rg = warp >> 2, dq = warp & 3;
    const int g = lane >> 2, c = (lane & 3) * 2;
    const int head[2] = {ht * 64 + 16 * rg + g, ht * 64 + 16 * rg + g + 8};
    float4 *const xs = reinterpret_cast<float4 *>(smem + 2 * kMlaStageBytes) +
                       rg * (kMlaGroupSlots * 32);
    float *const xmax = reinterpret_cast<float *>(smem + 2 * kMlaStageBytes +
                                                  4 * kMlaGroupSlots * kMlaSlotBytes) +
                        rg * 64;
    unsigned char *const xp = reinterpret_cast<unsigned char *>(xs);
    float *const xsum = reinterpret_cast<float *>(xp + 16 * kMlaProbRow);

    unsigned qf[kMlaScoreQuarter / 16][4];
    if (lo < hi) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float *q = qeff + (size_t)head[i] * kMlaWidth + kMlaScoreQuarter * dq + c;
#pragma unroll
            for (int ks = 0; ks < kMlaScoreQuarter / 16; ++ks)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    float2 v = make_float2(0.f, 0.f);
                    if (head[i] < heads) v = *reinterpret_cast<const float2 *>(q + 16 * ks + 8 * hh);
                    const __half2 h = __floats2half2_rn(v.x, v.y);
                    qf[ks][i + 2 * hh] = *reinterpret_cast<const unsigned *>(&h);
                }
        }
    }
    const int mi = lane >> 3, rr = lane & 7;
    const int k_key = 8 * (mi >> 1) + rr, k_chunk = (kMlaScoreQuarter / 8) * dq + (mi & 1);
    const int v_key = 8 * (mi & 1) + rr, v_chunk = (kMlaValueQuarter / 8) * dq + (mi >> 1);

    float o[kMlaValueQuarter / 8][4];
#pragma unroll
    for (int j = 0; j < kMlaValueQuarter / 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[j][e] = 0.f;
    float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};

    int it = 0;
    for (int base = lo; base < hi; base += kMlaTile, ++it) {
        cp_async_wait_all();
        __syncthreads();
        if (base + kMlaTile < hi)
            mla_issue_tile(cache, base + kMlaTile, hi, stages + ((it + 1) & 1) * kMlaStageBytes);
        const unsigned kst = stages + (it & 1) * kMlaStageBytes;

        float sq[4][4];
#pragma unroll
        for (int nb = 0; nb < 4; ++nb)
#pragma unroll
            for (int e = 0; e < 4; ++e) sq[nb][e] = 0.f;
#pragma unroll
        for (int ks = 0; ks < kMlaScoreQuarter / 16; ++ks)
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                unsigned kb[4];
                ldsm_x4(kst + mla_swizzle(16 * np + k_key, k_chunk + 2 * ks), kb);
                mma_16816(sq[2 * np], qf[ks], kb[0], kb[1]);
                mma_16816(sq[2 * np + 1], qf[ks], kb[2], kb[3]);
            }

        float4 own = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
            const float4 v = make_float4(sq[nb][0], sq[nb][1], sq[nb][2], sq[nb][3]);
            if (nb == dq)
                own = v;
            else
                xs[(3 * dq + nb - (nb > dq ? 1 : 0)) * 32 + lane] = v;
        }
        bar_row_group(rg);
        float4 sum4 = dq == 0 ? own : xs[dq * 32 + lane - 32];
#pragma unroll
        for (int q = 1; q < 4; ++q) {
            const float4 v = q == dq ? own : xs[(3 * q + dq - (dq > q ? 1 : 0)) * 32 + lane];
            sum4.x += v.x;
            sum4.y += v.y;
            sum4.z += v.z;
            sum4.w += v.w;
        }
        const int kpos = base + 8 * dq + c;
        const bool full = base + kMlaTile <= hi;
        const float s2[4] = {full || kpos < hi ? sum4.x * scale : -INFINITY,
                             full || kpos + 1 < hi ? sum4.y * scale : -INFINITY,
                             full || kpos < hi ? sum4.z * scale : -INFINITY,
                             full || kpos + 1 < hi ? sum4.w * scale : -INFINITY};
        float bm[2] = {fmaxf(s2[0], s2[1]), fmaxf(s2[2], s2[3])};
#pragma unroll
        for (int x = 1; x <= 2; x <<= 1) {
            bm[0] = fmaxf(bm[0], __shfl_xor_sync(0xffffffffu, bm[0], x));
            bm[1] = fmaxf(bm[1], __shfl_xor_sync(0xffffffffu, bm[1], x));
        }
        if (c == 0) {
            xmax[g * 4 + dq] = bm[0];
            xmax[(g + 8) * 4 + dq] = bm[1];
        }
        bar_row_group(rg);
        const float4 mq0 = *reinterpret_cast<const float4 *>(xmax + g * 4);
        const float4 mq1 = *reinterpret_cast<const float4 *>(xmax + (g + 8) * 4);
        const float m_new[2] = {
            fmaxf(m_run[0], fmaxf(fmaxf(mq0.x, mq0.y), fmaxf(mq0.z, mq0.w))),
            fmaxf(m_run[1], fmaxf(fmaxf(mq1.x, mq1.y), fmaxf(mq1.z, mq1.w)))};
        const float m_ref[2] = {m_new[0] == -INFINITY ? 0.f : m_new[0],
                                m_new[1] == -INFINITY ? 0.f : m_new[1]};
        const __half2 ph[2] = {
            __floats2half2_rn(expf(s2[0] - m_ref[0]), expf(s2[1] - m_ref[0])),
            __floats2half2_rn(expf(s2[2] - m_ref[1]), expf(s2[3] - m_ref[1]))};
        const float2 pr[2] = {__half22float2(ph[0]), __half22float2(ph[1])};
        float ps[2] = {pr[0].x + pr[0].y, pr[1].x + pr[1].y};
#pragma unroll
        for (int x = 1; x <= 2; x <<= 1) {
            ps[0] += __shfl_xor_sync(0xffffffffu, ps[0], x);
            ps[1] += __shfl_xor_sync(0xffffffffu, ps[1], x);
        }
        if (c == 0) {
            xsum[g * 4 + dq] = ps[0];
            xsum[(g + 8) * 4 + dq] = ps[1];
        }
        *reinterpret_cast<__half2 *>(xp + g * kMlaProbRow + 16 * dq + 2 * c) = ph[0];
        *reinterpret_cast<__half2 *>(xp + (g + 8) * kMlaProbRow + 16 * dq + 2 * c) = ph[1];
        const float corr[2] = {expf(m_run[0] - m_ref[0]), expf(m_run[1] - m_ref[1])};
        bar_row_group(rg);
        const float4 sq0 = *reinterpret_cast<const float4 *>(xsum + g * 4);
        const float4 sq1 = *reinterpret_cast<const float4 *>(xsum + (g + 8) * 4);
        l_run[0] = l_run[0] * corr[0] + (((sq0.x + sq0.y) + sq0.z) + sq0.w);
        l_run[1] = l_run[1] * corr[1] + (((sq1.x + sq1.y) + sq1.z) + sq1.w);
        m_run[0] = m_new[0];
        m_run[1] = m_new[1];

#pragma unroll
        for (int j = 0; j < kMlaValueQuarter / 8; ++j) {
            o[j][0] *= corr[0];
            o[j][1] *= corr[0];
            o[j][2] *= corr[1];
            o[j][3] *= corr[1];
        }
        const unsigned p_lane = smem_addr(xp) + (rr + 8 * (mi & 1)) * kMlaProbRow + 16 * (mi >> 1);
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            unsigned pa[4];
            ldsm_x4(p_lane + 32 * ks, pa);
#pragma unroll
            for (int jp = 0; jp < kMlaValueQuarter / 16; ++jp) {
                unsigned vb[4];
                ldsm_x4_trans(kst + mla_swizzle(16 * ks + v_key, v_chunk + 2 * jp), vb);
                mma_16816(o[2 * jp], pa, vb[0], vb[1]);
                mma_16816(o[2 * jp + 1], pa, vb[2], vb[3]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        if (head[i] >= heads) continue;
        const int grp = head[i] / kMlaGroup, hg = head[i] % kMlaGroup;
        float *dst = part + (((size_t)grp * splits + sp) * kMlaGroup + hg) * kMlaPartStride;
#pragma unroll
        for (int j = 0; j < kMlaValueQuarter / 8; ++j)
            *reinterpret_cast<float2 *>(dst + kMlaValueQuarter * dq + 8 * j + c) =
                make_float2(o[j][2 * i], o[j][2 * i + 1]);
        if (dq == 0 && c == 0) {
            dst[kMlaRank] = isnan(l_run[i]) ? NAN : m_run[i];
            dst[kMlaRank + 1] = l_run[i];
        }
    }
#else
    __trap();
#endif
}
