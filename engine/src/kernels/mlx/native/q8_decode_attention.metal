    // The fused 8-bit KV decode attention (packet FUSED-KV8-a; the pass-1 structure R4 of
    // packet FUSED-KV8-EFF-2). Pass 1: one simdgroup computes RPH of the REP query heads of
    // one KV head over one block of keys, and one threadgroup holds SGS = REP / RPH
    // simdgroups that all work on the SAME block of keys (blk = threadgroup_position_in_grid.x),
    // each simdgroup loading the same K and V rows - the repeats hit the cache. RPH is the
    // largest divisor of REP that is at most 4 (REP 8 runs RPH 4 with SGS 2; REP 4 runs
    // 4 and 1; REP 6 runs 3 and 2; REP 5 and 7 run 1 with SGS = REP; REP 1, 2, 3 run
    // RPH = REP with SGS 1), so the per-lane live state is q/o[RPH][DPT] - 64 floats at
    // REP 8, D 256 - instead of the q/o[REP][DPT] = 128 floats of the first fused kernel,
    // whose roughly 170-register pass 1 (an estimate; the toolchain reports no register
    // counts) was latency bound: 102 GB/s at 131,072 keys against a 320 GB/s load-only
    // floor. Cutting the per-lane state to RPH heads was measured in the FUSED-KV8-EFF-1
    // standalone spike (its REPORT.md under the fused-kv8-eff/eff1 scratch directory; Mac,
    // p50 of the attention of 10 layers per decode token, QH 16, KVH 2, D 256, REP 8, with
    // the correctness gate of 290/290 entries and per-head arithmetic identical to the
    // first kernel): 8,192 keys 1.148 -> 0.835 ms with the half block count; 32,768 keys
    // 4.012 -> 2.671 ms (1.50x); 131,072 keys 13.949 -> 9.683 ms (1.44x, 147 GB/s);
    // 262,000 keys 26.187 -> 18.889 ms (1.39x). The register bound is evidenced by that
    // report's measured slope over the per-lane state (8/4/2/1 heads per simdgroup:
    // 13.95 / 10.27 / 10.41 / 17.22 ms at 131,072 keys - halving the state pays 1.36x,
    // halving again pays nothing, and one head per simdgroup loses 24 percent to the
    // per-key dependency chain), not by a toolchain report. Each lane owns D / 32
    // consecutive dims, loads its slice of the 8-bit key and value code words once per row
    // (little-endian bytes: the first value sits in the lowest byte of a word), and forms
    // the key dot product through the per-group factoring
    // `scale_g * sum(q_i * code_i) + bias_g * sum(q_i)`, all in F32, beside an online
    // softmax per head (running max m, running sum dd, running mix o, rescaled by
    // `exp(m - new_m)` at every key). Every block - including an EMPTY tail block, which
    // occurs whenever the block-count rule leaves blocks whose first key is at or past
    // `len` (len 130 gives NB 64 and a block size of 3, so blocks 44 to 63 hold no key) -
    // writes its unnormalized F32 partials po [QH, nb, D], psum [QH, nb], and pmax
    // [QH, nb]; an empty block writes max -FLT_MAX, sum 0, and a zero partial, and pass 2
    // weights it by `exp(-FLT_MAX - m*) = 0` under MLX's default safe math. Pass 2: one
    // simdgroup merges the nb block partials of one head by log-sum-exp around the global
    // max m* (simd_max), so no exponent ever overflows, and casts to the query dtype T.
    // The template arguments are T (the query dtype: float or bfloat16_t), D, REP, and
    // GSZ, plus RPH and SGS of pass 1; the scalars length, cap, nb, and scale are kernel
    // INPUTS, so one compiled program serves every length. This file holds the two kernel
    // bodies as plain Metal text separated by one marker line (the constant
    // Q8_DECODE_PASS_MARKER of kernels/shape.rs); resident_ops.rs includes it with
    // include_str! and splits it at that line. MLX generates the kernel signatures from
    // the input and output names of the launcher (q_in, kc, ks_row, kb_row, vc, vs_row,
    // vb_row, length, cap, nb, scale -> po, psum, pmax; then po, psum, pmax, nb -> out).

    // ---- PASS 1 ----
    // RPH of the REP query heads of this KV head per simdgroup; the SGS simdgroups of the
    // threadgroup all cover the SAME block of this KV head (the y rows of the grid carry
    // the KV heads: threadgroup y / HS is the KV head, y % HS the head-row slot). The
    // launcher picks RPH as a divisor of REP, so HS is 1, `row` is 0, and the `blk >= nb`
    // guard below never fires under the shipped grid (NB * 32 threads over 32); both stay
    // as cheap invariants for a launch that changes the grid.
    const int grp = (int)threadgroup_position_in_grid.x;
    const int sg = (int)simdgroup_index_in_threadgroup;
    const int lane = (int)thread_index_in_simdgroup;
    const int blk = grp;
    constexpr int HS = REP / (RPH * SGS);
    const int kv = (int)threadgroup_position_in_grid.y / HS;
    const int row = (int)threadgroup_position_in_grid.y % HS;
    const int h0 = kv * REP + row * SGS * RPH + sg * RPH;
    if (blk >= nb) {
      return;
    }

    constexpr int DPT = D / 32;         // dims per lane
    constexpr int NWORD = D / 4;        // uint32 words per key row
    constexpr int NGRP = D / GSZ;       // groups per key row
    constexpr int NW = (DPT + 3) / 4;   // words touched by a lane

    const int bs = (length + nb - 1) / nb;
    const int s0 = blk * bs;
    const int s1 = min(s0 + bs, length);

    // per-head buffers, head stride CAP (never a prefix copy)
    const device uint32_t* kc_h = kc + (size_t)kv * cap * NWORD;
    const device uint32_t* vc_h = vc + (size_t)kv * cap * NWORD;
    const size_t s_ofs = (size_t)kv * cap * NGRP;  // scales/biases head offset

    const int g_lane = (lane * DPT) / GSZ;  // lane slice inside one group
    const int off = lane * DPT;

    // query in registers (pre-scaled), per-lane group partial of sum(q)
    float q[RPH][DPT];
    float qsum[RPH];
    for (int j = 0; j < RPH; j++) {
      float s = 0.0f;
      for (int i = 0; i < DPT; i++) {
        float v = (float)scale * q_in[((size_t)(h0 + j)) * D + off + i];
        q[j][i] = v;
        s += v;
      }
      qsum[j] = s;
    }

    float o[RPH][DPT];
    float m[RPH];
    float dd[RPH];
    for (int j = 0; j < RPH; j++) {
      for (int i = 0; i < DPT; i++) {
        o[j][i] = 0.0f;
      }
      m[j] = -3.4028234663852886e+38f;  // -FLT_MAX; safe math: exp(...) == 0
      dd[j] = 0.0f;
    }

    for (int t = s0; t < s1; t++) {
      // --- key row: load the lane's words once, extract codes by shift/mask
      uint32_t kw[NW];
      #pragma clang loop unroll(full)
      for (int w = 0; w < NW; w++) {
        kw[w] = kc_h[(size_t)t * NWORD + (off >> 2) + w];
      }
      const float ksc = ks_row[s_ofs + (size_t)t * NGRP + g_lane];
      const float kbi = kb_row[s_ofs + (size_t)t * NGRP + g_lane];

      // --- value row: dequantize the lane's slice once, share it over the RPH heads
      uint32_t vw[NW];
      #pragma clang loop unroll(full)
      for (int w = 0; w < NW; w++) {
        vw[w] = vc_h[(size_t)t * NWORD + (off >> 2) + w];
      }
      const float vsc = vs_row[s_ofs + (size_t)t * NGRP + g_lane];
      const float vbi = vb_row[s_ofs + (size_t)t * NGRP + g_lane];
      float wv[DPT];
      #pragma clang loop unroll(full)
      for (int i = 0; i < DPT; i++) {
        uint32_t word = vw[(i + (off & 3)) >> 2];
        int code = (int)((word >> (8 * ((off + i) & 3))) & 0xFFu);
        wv[i] = fma(vsc, (float)code, vbi);
      }

      // --- key dot product through the per-group factoring, per head
      float a[RPH];
      #pragma clang loop unroll(full)
      for (int j = 0; j < RPH; j++) {
        float acc = 0.0f;
        #pragma clang loop unroll(full)
        for (int i = 0; i < DPT; i++) {
          uint32_t word = kw[(i + (off & 3)) >> 2];
          int code = (int)((word >> (8 * ((off + i) & 3))) & 0xFFu);
          acc = fma(q[j][i], (float)code, acc);
        }
        a[j] = fma(ksc, acc, kbi * qsum[j]);
      }

      // --- online softmax per head (lane partials combined by simd_sum)
      #pragma clang loop unroll(full)
      for (int j = 0; j < RPH; j++) {
        const float score = simd_sum(a[j]);
        const float new_m = max(m[j], score);
        const float factor = exp(m[j] - new_m);
        const float p = exp(score - new_m);
        m[j] = new_m;
        dd[j] = dd[j] * factor + p;
        #pragma clang loop unroll(full)
        for (int i = 0; i < DPT; i++) {
          o[j][i] = o[j][i] * factor + p * wv[i];
        }
      }
    }

    // --- write this block's partial (unnormalized) for every head of this simdgroup
    #pragma clang loop unroll(full)
    for (int j = 0; j < RPH; j++) {
      size_t h = (size_t)(h0 + j);
      device float* po_b = po + (h * nb + blk) * D + off;
      #pragma clang loop unroll(full)
      for (int i = 0; i < DPT; i++) {
        po_b[i] = o[j][i];
      }
      if (lane == 0) {
        psum[h * nb + blk] = dd[j];
        pmax[h * nb + blk] = m[j];
      }
    }
// ---- PASS 2 ----
    // The merge of the NB block partials of one head by log-sum-exp; the output carries
    // the query dtype T.
    const int h = (int)threadgroup_position_in_grid.x;
    const int lane = (int)thread_index_in_simdgroup;
    constexpr int DPT = D / 32;
    const int off = lane * DPT;

    float mstar = -3.4028234663852886e+38f;
    for (int b = lane; b < nb; b += 32) {
      mstar = max(mstar, pmax[(size_t)h * nb + b]);
    }
    mstar = simd_max(mstar);

    float denom = 0.0f;
    for (int b = lane; b < nb; b += 32) {
      float w = exp(pmax[(size_t)h * nb + b] - mstar);
      denom += w * psum[(size_t)h * nb + b];
    }
    denom = simd_sum(denom);

    float o[DPT];
    for (int i = 0; i < DPT; i++) {
      o[i] = 0.0f;
    }
    for (int b = 0; b < nb; b++) {
      float w = exp(pmax[(size_t)h * nb + b] - mstar);
      const size_t row0 = ((size_t)h * nb + b) * D + off;
      #pragma clang loop unroll(full)
      for (int i = 0; i < DPT; i++) {
        o[i] += w * po[row0 + i];
      }
    }
    for (int i = 0; i < DPT; i++) {
      out[(size_t)h * D + off + i] = static_cast<T>(o[i] / denom);
    }
