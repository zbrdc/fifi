// q8_flash_prefill_mma -- fused causal prefill attention over the Q8 KV cache of the fifi Qwen3.6 engine.
// KERNEL BODY ONLY (MLX fast metal_kernel generates the signature from the input / output names and the
// template arguments; the grid is in THREADS).  Spike r45-mma, validated on Apple M4 Max with MLX 0.32.2 and 0.32.3.
//
// Cache layout it reads (engine/src/family/qwen36/resident/attn.rs, Q8Tensor; Q8_GROUP = 64, Q8_BITS = 8):
//   kc, vc            [1, KVH, cap, D / 4]    uint32, 4 unsigned 8-bit codes per word, first value in the LOW byte
//   ks_row, kb_row,
//   vs_row, vb_row    [1, KVH, cap, D / GSZ]  affine scale / bias per group of GSZ values (activation dtype, read as float);
//                                             value = scale * code + bias (MLX affine quantisation)
//   q_in              [1, QH, rows, D]        T (bfloat16_t), row-major, head h reads KV head h / 8
// Shape contract: QH / KVH == 8 (REPS), D % 16 == 0, D % GSZ == 0, GSZ % 16 == 0, BK % 8 == 0, LD >= D, LD % 8 == 0.
//
// Structure (FlashAttention-2 style, simdgroup_matrix 8x8 MMA, float accumulation):
//  * One simdgroup owns ONE query token and its 8 heads of one KV head: the 8 rows of every 8x8
//    fragment are the 8 query heads that share that KV head, so all 8 rows have the same causal limit.
//  * One threadgroup = NSG simdgroups = NSG consecutive query tokens of one KV head (grid y = KV head).
//    The K tile and the V tile (BK keys x D) are dequantised from the Q8 codes ONCE per threadgroup into
//    threadgroup memory (K as half: scale * code + bias in half arithmetic, one rounding; V as half or bf16,
//    see VBF) and used by all NSG simdgroups, so the dequantisation is shared by 8 * NSG query rows.
//  * Per tile and simdgroup: S = Q K^T (D/8 x BK/8 MMAs; the Q fragments are loaded from device memory at
//    every use, which keeps the register state to the 32 float O fragments: a register-resident Q spills),
//    causal mask only for tiles that can reach past a row's limit, online softmax in float (exp2, scale
//    folded in), then O += P V with P in half (Q8) or float (FLOAT). The 4 lanes that share a fragment
//    row hold its softmax statistics, so row max / row sum are two simd_shuffle_xor (masks 1 and 8).
//  * Threadgroups of the latest (heaviest) query tokens are scheduled first.
//
// Launch (LAUNCH.md): grid = (ceil(rows / NSG) * NSG * 32, KVH, 1) in THREADS, threadgroup = (NSG * 32, 1, 1),
// always set explicitly. Q8 threadgroup memory is Kbuf / Vbuf: 2 * BK * LD * 2 bytes = 16,896 B.
// Template args: T (dtype of q_in and out: bfloat16_t), D (256), GSZ (64), NSG (8), BK (16), LD (264),
//   VBF (1: V tile in bf16, matches the error floor of MLX quantized_matmul; 0: V tile in half, slightly more
//   accurate on real data, about 0.5% faster).   Recommended: T = query dtype, D 256, GSZ 64, NSG 8, BK 16, LD 264, VBF 1.
// Inputs, in this order: q_in kc ks_row kb_row vc vs_row vb_row rows length cap scale sinks window.   Output: out.
//   sinks [1, QH] float32: RAW sink logits per QUERY head (not QK-scaled; the family publishes them
//           - Flint: 32 distinct learned values; GQA K/V sharing does NOT share the denominator offset).
//           Absent FLOAT sinks are -infinity; Q8 retains its validated -1e30 sentinel.
//   window int32: 0 = full causal; > 0 = the sliding window width (row t sees [t-window+1, t]).
//   Template KVQ8: 1 = the Q8 affine cache (codes + scales/biases, the validated path);
//   0 = FLOAT KV planes - kc/vc are [1, KVH, cap, D] in matching query dtype T (quant slots are dummies),
//   with K staged as T and V as float, serving the large-batch float dispatch.
//   rows, length, cap: int32 scalars (length = valid keys including this chunk, cap = allocated positions of the
//   capacity buffers, which are passed WHOLE; query row t sees keys 0 .. length - rows + t); scale: float32 (1 / sqrt(D)).
//   out [1, QH, rows, D] T.
constexpr int NTH = NSG * 32;
constexpr int NFD = D / 8;
constexpr int NFK = BK / 8;
constexpr int NCH = D / 16;
constexpr int NWORD = D / 4;
constexpr int NGRP = D / GSZ;
constexpr int REPS = 8;

const int sg = (int)simdgroup_index_in_threadgroup;
const int lane = (int)thread_index_in_simdgroup;
const int tid = sg * 32 + lane;
const int nblk = (rows + NSG - 1) / NSG;
const int tb = nblk - 1 - (int)threadgroup_position_in_grid.x;
const int kv = (int)threadgroup_position_in_grid.y;
const int t = tb * NSG + sg;
const bool active = t < rows;
const int q_off = length - rows;
const int lim = q_off + t;
const int kLo = (window > 0) ? max(0, lim - (window - 1)) : 0;  // first key this row may see (SWA)                                   // last key this simdgroup may see
const int kend = q_off + min(tb * NSG + NSG, rows);          // keys this threadgroup needs
const int ntiles = (kend + BK - 1) / BK;
const int tile_lo = (window > 0) ? max(0, q_off + tb * NSG - (window - 1)) / BK : 0;  // skip pre-window tiles
const int first_masked = q_off + tb * NSG;                   // keys above it can be masked for some simdgroup

// element coordinates of a thread inside an 8x8 simdgroup_matrix fragment: (fm, fn) and (fm, fn + 1)
const short qid = lane / 4;
const short fm = (qid & 4) + ((lane / 2) % 4);
const short fn = (qid & 2) * 2 + (lane % 2) * 2;

// Keep FLOAT K in T, never half (BF16 2^17 would overflow). Q8 is unchanged.
using KT = conditional_t<KVQ8 != 0, half, T>;
using VT = conditional_t<KVQ8 != 0, conditional_t<VBF != 0, bfloat16_t, half>, float>;
// FLOAT P/V stay F32 to avoid tile-origin-dependent BF16 probability rounding
// and half underflow of tiny probabilities. Q8 keeps its validated half P.
using PT = conditional_t<KVQ8 != 0, half, float>;
threadgroup KT Kbuf[BK * LD];
threadgroup VT Vbuf[BK * LD];

simdgroup_matrix<float, 8, 8> Of[NFD];
#pragma clang loop unroll(full)
for (int j = 0; j < NFD; j++) {
  Of[j] = simdgroup_matrix<float, 8, 8>(0.0f);
}
const float log2e = 1.4426950408889634f;
const float raw_sink = (const float)sinks[kv * REPS + fm];  // PER QUERY HEAD, never QK-scaled
// FLOAT subtracts its maximum before log2 conversion to avoid finite-range overflow.
const float sink_score = KVQ8 != 0 ? raw_sink * log2e : raw_sink;
const float exp_scale = KVQ8 != 0 ? 1.0f : log2e;
float m_run = sink_score;
// One lane per fragment row (the xor-1/xor-8 reduction group) seeds the sink mass, so the
// reduced denominator carries it exactly once; only Q8 uses a magnitude cutoff.
const bool sink_lane = ((lane & 1) == 0) && ((lane & 8) == 0);
const bool has_sink = KVQ8 != 0 ? sink_score > -1.0e29f : sink_score != -INFINITY;
float l_run = (has_sink && sink_lane) ? 1.0f : 0.0f;
const float masked_score = KVQ8 != 0 ? -1.0e30f : -INFINITY;
const float score_scale = KVQ8 != 0 ? scale * log2e : scale;
const device T* qtok = q_in + ((size_t)(kv * REPS) * rows + (size_t)(active ? t : 0)) * D;
const ulong qstride = (ulong)rows * D;                       // distance between the 8 heads of one token

for (int tile = tile_lo; tile < ntiles; tile++) {
  const int k0 = tile * BK;
  // ---- KV tile load: Q8 dequant (KVQ8 != 0) or the direct bf16 plane copy ----
  if (KVQ8 != 0) {
// ---- cooperative dequantisation of the K tile, then of the V tile (two loops keep the live raw
  // registers of only one side at a time); positions past `length` are zero ----
  for (int c = tid; c < BK * NCH; c += NTH) {
    const int row = c / NCH;
    const int ch = c % NCH;
    const int key = k0 + row;
    if (key < length) {
      const size_t rbase = (size_t)kv * cap + key;
      const int g = (ch * 16) / GSZ;
      const uint4 raw = *(const device uint4*)(kc + rbase * NWORD + ch * 4);
      const half sc = (half)(float)ks_row[rbase * NGRP + g];
      const half bi = (half)(float)kb_row[rbase * NGRP + g];
      #pragma clang loop unroll(full)
      for (int w = 0; w < 4; w++) {
        *(threadgroup half4*)(Kbuf + row * LD + ch * 16 + w * 4) =
            fma(half4(as_type<uchar4>(raw[w])), half4(sc), half4(bi));
      }
    } else {
      #pragma clang loop unroll(full)
      for (int w = 0; w < 4; w++) {
        *(threadgroup half4*)(Kbuf + row * LD + ch * 16 + w * 4) = half4(0.0h);
      }
    }
  }
  for (int c = tid; c < BK * NCH; c += NTH) {
    const int row = c / NCH;
    const int ch = c % NCH;
    const int key = k0 + row;
    if (key < length) {
      const size_t rbase = (size_t)kv * cap + key;
      const int g = (ch * 16) / GSZ;
      const uint4 raw = *(const device uint4*)(vc + rbase * NWORD + ch * 4);
      const half sc = (half)(float)vs_row[rbase * NGRP + g];
      const half bi = (half)(float)vb_row[rbase * NGRP + g];
      #pragma clang loop unroll(full)
      for (int w = 0; w < 4; w++) {
        if (VBF != 0) {
          const float4 f = fma(float4(as_type<uchar4>(raw[w])), (float)sc, (float)bi);
          Vbuf[row * LD + ch * 16 + w * 4 + 0] = (VT)f[0];
          Vbuf[row * LD + ch * 16 + w * 4 + 1] = (VT)f[1];
          Vbuf[row * LD + ch * 16 + w * 4 + 2] = (VT)f[2];
          Vbuf[row * LD + ch * 16 + w * 4 + 3] = (VT)f[3];
        } else {
          *(threadgroup half4*)(Vbuf + row * LD + ch * 16 + w * 4) =
              fma(half4(as_type<uchar4>(raw[w])), half4(sc), half4(bi));
        }
      }
    } else {
      #pragma clang loop unroll(full)
      for (int w = 0; w < 4; w++) {
        Vbuf[row * LD + ch * 16 + w * 4 + 0] = (VT)0.0f;
        Vbuf[row * LD + ch * 16 + w * 4 + 1] = (VT)0.0f;
        Vbuf[row * LD + ch * 16 + w * 4 + 2] = (VT)0.0f;
        Vbuf[row * LD + ch * 16 + w * 4 + 3] = (VT)0.0f;
      }
    }
  }
  } else {
    // FLOAT KV: kc/vc are [1, KVH, cap, D] planes in T; K stages as T, V as float.
    // Quant slots carry dummies. Explicit casts also type-check the dead Q8 arm.
    for (int c = tid; c < BK * D; c += NTH) {
      const int row = c / D;
      const int col = c % D;
      const int key = k0 + row;
      const device T* kplane = (const device T*)kc;
      if (key < length) {
        Kbuf[row * LD + col] = (KT)kplane[((size_t)kv * cap + key) * D + col];
      } else {
        Kbuf[row * LD + col] = (KT)0.0f;
      }
    }
    for (int c = tid; c < BK * D; c += NTH) {
      const int row = c / D;
      const int col = c % D;
      const int key = k0 + row;
      const device T* vplane = (const device T*)vc;
      if (key < length) {
        Vbuf[row * LD + col] = (VT)vplane[((size_t)kv * cap + key) * D + col];
      } else {
        Vbuf[row * LD + col] = (VT)0.0f;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (active) {
    // ---- S = Q K^T for this token's 8 heads x BK keys ----
    simdgroup_matrix<float, 8, 8> S[NFK];
    #pragma clang loop unroll(full)
    for (int kb = 0; kb < NFK; kb++) {
      S[kb] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
    #pragma clang loop unroll(full)
    for (int dj = 0; dj < NFD; dj++) {
      simdgroup_matrix<T, 8, 8> qf;
      simdgroup_load(qf, qtok + dj * 8, qstride);
      #pragma clang loop unroll(full)
      for (int kb = 0; kb < NFK; kb++) {
        simdgroup_matrix<KT, 8, 8> kf;
        simdgroup_load(kf, Kbuf + kb * 8 * LD + dj * 8, LD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(S[kb], qf, kf, S[kb]);
      }
    }
    // ---- scale, causal mask (only tiles that can reach past a limit), online softmax ----
    const bool need_mask = (k0 + BK - 1) > first_masked;
    float mt = masked_score;
    #pragma clang loop unroll(full)
    for (int kb = 0; kb < NFK; kb++) {
      #pragma clang loop unroll(full)
      for (int e = 0; e < 2; e++) {
        float s = S[kb].thread_elements()[e] * score_scale;
        if ((need_mask && (k0 + kb * 8 + fn + e > lim)) ||
            (window > 0 && (k0 + kb * 8 + fn + e < kLo))) {
          s = masked_score;
        }
        S[kb].thread_elements()[e] = s;
        mt = max(mt, s);
      }
    }
    mt = max(mt, simd_shuffle_xor(mt, (ushort)1));
    mt = max(mt, simd_shuffle_xor(mt, (ushort)8));
    const float m_new = max(m_run, mt);
    // An empty FLOAT tile with no sink must not evaluate -infinity - -infinity.
    const bool empty_float_tile = KVQ8 == 0 && m_new == -INFINITY;
    const float alpha = (KVQ8 == 0 && m_run == m_new) ? 1.0f : fast::exp2((m_run - m_new) * exp_scale);
    m_run = m_new;
    simdgroup_matrix<PT, 8, 8> Ph[NFK];
    float lsum = 0.0f;
    #pragma clang loop unroll(full)
    for (int kb = 0; kb < NFK; kb++) {
      const float p0 = empty_float_tile ? 0.0f : fast::exp2((S[kb].thread_elements()[0] - m_new) * exp_scale);
      const float p1 = empty_float_tile ? 0.0f : fast::exp2((S[kb].thread_elements()[1] - m_new) * exp_scale);
      lsum += p0 + p1;
      Ph[kb].thread_elements()[0] = (PT)p0;
      Ph[kb].thread_elements()[1] = (PT)p1;
    }
    l_run = l_run * alpha + lsum;
    #pragma clang loop unroll(full)
    for (int dj = 0; dj < NFD; dj++) {
      Of[dj].thread_elements()[0] *= alpha;
      Of[dj].thread_elements()[1] *= alpha;
    }
    // ---- O += P V ----
    #pragma clang loop unroll(full)
    for (int kb = 0; kb < NFK; kb++) {
      #pragma clang loop unroll(full)
      for (int dj = 0; dj < NFD; dj++) {
        simdgroup_matrix<VT, 8, 8> vf;
        simdgroup_load(vf, Vbuf + kb * 8 * LD + dj * 8, LD);
        simdgroup_multiply_accumulate(Of[dj], Ph[kb], vf, Of[dj]);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (active) {
  float lt = l_run;
  lt += simd_shuffle_xor(lt, (ushort)1);
  lt += simd_shuffle_xor(lt, (ushort)8);
  const float inv = 1.0f / lt;
  device T* orow = out + ((size_t)(kv * REPS + fm) * rows + t) * D + fn;
  #pragma clang loop unroll(full)
  for (int j = 0; j < NFD; j++) {
    orow[8 * j] = (T)(Of[j].thread_elements()[0] * inv);
    orow[8 * j + 1] = (T)(Of[j].thread_elements()[1] * inv);
  }
}
