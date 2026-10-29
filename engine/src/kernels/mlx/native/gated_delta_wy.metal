// gated_delta_chunk -- chunked (WY / UT-transform) gated delta rule, C = 16 rows per chunk, simdgroup_matrix MMA with f32
// operands and f32 accumulation, ONE kernel.  KERNEL BODY ONLY (MLX fast metal_kernel generates the signature from the
// input / output names and the template arguments; the grid is in THREADS).
// Inputs, in this order: q k v g beta state_in T.   Outputs: y state_out.   Exactly the contract of the vendored mlx-lm
// kernel gated_delta_step: all F32, B = 1;
//   q [1,T,Hk,Dk] after L2 norm and the 1/sqrt(Dk) scale, k [1,T,Hk,Dk] after L2 norm, v [1,T,Hv,Dv],
//   g [1,T,Hv] decay FACTOR, beta [1,T,Hv], state_in [1,Hv,Dv,Dk] (key index fastest), T int scalar input,
//   y [1,T,Hv,Dv], state_out [1,Hv,Dv,Dk].   Value head h reads key head h / (Hv / Hk).
// Template args: Dk Dv Hk Hv NS.  NS = simdgroups per threadgroup: 16 recommended (one threadgroup per head); 8 gives
// bitwise identical results 1-3% slower, 4 is 20% slower.  Dv % (8 * NS) == 0; Dk % 64 == 0; NS must divide 16 * Dk / 4 / 32.
// Launch: grid (32 * Dv / 8, Hv, 1) threads, threadgroup (32 * NS, 1, 1).  Static threadgroup memory 20,224 bytes.
// Pointers must be 16-byte aligned (MLX buffers are).  Any T >= 1; the T % 16 trailing rows run as recurrent steps.
//
// Math, per chunk of C rows, with E[t][i] = g_{i+1} ... g_t, gam_t = g_0 ... g_t, wgt_t = g_{t+1} ... g_{C-1} and the state
// S0 at the start of the chunk:
//   rhs   = beta (v - gam (K S0^T)),   solve (I + A) delta = rhs,   A[t][i] = beta_t E[t][i] (k_t . k_i)  (i < t)
//   y     = gam (Q S0^T) + M delta,    M[t][i] = E[t][i] (q_t . k_i)  (i <= t)
//   S0^T <- gam_{C-1} S0^T + K^T (wgt . delta)
// The unit-lower-triangular solve is two 8 x 8 blocks with the explicit inverses D0, D1 of the diagonal blocks:
//   delta0 = D0 rhs0, delta1 = D1 (rhs1 - A10 delta0).
// Design references (read only, nothing copied): Yang et al. arXiv 2406.06484 (chunkwise parallel DeltaNet, WY
// representation, UT transform), Yang, Kautz, Hatamizadeh arXiv 2412.06464 (Gated DeltaNet, decayed chunk form).
//
// Structure: one simdgroup owns an 8-column strip of the value dimension of one value head and walks all chunks in order;
// its state strip lives transposed ([Dk x 8], key index down the rows) in NKB = Dk / 8 simdgroup matrices (32 registers).
// The NS simdgroups of a threadgroup are NS consecutive strips of one head (the 128 / (8 * NS) threadgroups of a head each
// recompute the chunk factors; that redundancy is 2x at NS = 8 and absent at NS = 16).  K, Q, g, beta of a chunk are staged in threadgroup memory
// by all threads (the next chunk is prefetched into registers while the current one is computed).  Per chunk:
//   1. simdgroups 0-5 each build one 8 x 8 tile of K K^T / Q K^T from the staged K and Q (16 MMAs), apply the decay
//      products E (fma-compensated, one f32 rounding) and beta, and write A10 / M tiles to threadgroup memory; simdgroups
//      0 and 2 also invert their unit-lower diagonal block A00 / A11 (8 lanes, forward substitution); simdgroup 6 forms
//      gam and wgt.   2. all simdgroups: G = K S0^T and P = Q S0^T (64 MMAs).   3. barrier.   4. rhs, the two-block solve,
//      y = gam P + M delta, then the state update with K^T (wgt . delta) (38 MMAs).   5. barrier, stage the next chunk, barrier.
// Three threadgroup barriers per chunk.  The 8 x 8 operand tiles are threadgroup loads (the same tiles from device memory
// were 2.7x slower).  G and P use four accumulator chains each (key blocks kb % 4): that cut the kernel's error on real
// layers from 1.0-1.5x to 0.8-1.0x of the recurrent kernel's at 3% more time.  
constexpr int C = 16;
constexpr int NKB = Dk / 8;
constexpr int SCR = 544;
constexpr ulong RS = (ulong)Hk * Dk;        // row stride of q and k  ([T, Hk, Dk])
constexpr ulong VS = (ulong)Hv * Dv;        // row stride of v and y  ([T, Hv, Dv])
constexpr int LD = 136;                     // staged row stride in floats (padded against bank conflicts)
constexpr int NTH = NS * 32;
constexpr int NKQ = (C * Dk / 4) / NTH;     // float4 per thread for each of K and Q
#define CP_STEP(hi, lo, gj) { const float p_ = (hi) * (gj); lo = fma(lo, (gj), fma((hi), (gj), -p_)); hi = p_; }
threadgroup float Ksm[C * LD];
threadgroup float Qsm[C * LD];
threadgroup float Ssm[SCR];
threadgroup float Atmp[2][64];
threadgroup float Gsm[C];
threadgroup float Bsm[C];
const int lane = (int)thread_index_in_simdgroup;
const int sgi = (int)simdgroup_index_in_threadgroup;
const int h = (int)threadgroup_position_in_grid.y;
const int dv0 = ((int)threadgroup_position_in_grid.x * NS + sgi) * 8;
const int hk = h / (Hv / Hk);
// element coordinates of a thread inside an 8x8 simdgroup_matrix fragment: (fm, fn) and (fm, fn + 1)
const short qid = lane / 4;
const short fm = (qid & 4) + ((lane / 2) % 4);
const short fn = (qid & 2) * 2 + (lane % 2) * 2;
const int nfull = T / C;
const int tid = sgi * 32 + lane;

simdgroup_matrix<float, 8, 8> S[NKB];
const device float* sin_p = state_in + ((size_t)h * Dv + dv0) * Dk;
#pragma clang loop unroll(full)
for (int kb = 0; kb < NKB; kb++) {
  simdgroup_load(S[kb], sin_p + kb * 8, Dk, ulong2(0, 0), true);
}

float4 pk[NKQ], pq[NKQ];
float pg = 0.0f;
auto fetch = [&](int cc) {
  const device float* kg = k + ((size_t)cc * C * Hk + hk) * Dk;
  const device float* qg = q + ((size_t)cc * C * Hk + hk) * Dk;
  #pragma clang loop unroll(full)
  for (int j = 0; j < NKQ; j++) {
    const int i = tid + j * NTH;
    const int r = i / (Dk / 4);
    const int c4 = i % (Dk / 4);
    pk[j] = *(const device float4*)(kg + (size_t)r * RS + c4 * 4);
    pq[j] = *(const device float4*)(qg + (size_t)r * RS + c4 * 4);
  }
  if (tid < C) {
    pg = g[((size_t)cc * C + tid) * Hv + h];
  } else if (tid < 2 * C) {
    pg = beta[((size_t)cc * C + tid - C) * Hv + h];
  }
};
auto commit = [&]() {
  #pragma clang loop unroll(full)
  for (int j = 0; j < NKQ; j++) {
    const int i = tid + j * NTH;
    const int r = i / (Dk / 4);
    const int c4 = i % (Dk / 4);
    *(threadgroup float4*)(Ksm + r * LD + c4 * 4) = pk[j];
    *(threadgroup float4*)(Qsm + r * LD + c4 * 4) = pq[j];
  }
  if (tid < C) {
    Gsm[tid] = pg;
  } else if (tid < 2 * C) {
    Bsm[tid - C] = pg;
  }
};

if (nfull > 0) {
  fetch(0);
  commit();
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (int c = 0; c < nfull; c++) {
  if (c + 1 < nfull) {
    fetch(c + 1);                            // prefetch into registers; stored after this chunk is done
  }
  const threadgroup float* sc = Ssm;
  const float beta0 = Bsm[fm];
  const float beta1 = Bsm[8 + fm];

  // ---- chunk factors (state independent), split over the simdgroups ----
  for (int tk = sgi; tk < 7; tk += NS) {
    if (tk == 6) {
      if (lane < C) {
        float gh = 1.0f, gl_ = 0.0f, wh = 1.0f, wl = 0.0f;
        #pragma clang loop unroll(full)
        for (int j = 0; j < C; j++) {
          if (j <= lane) {
            CP_STEP(gh, gl_, Gsm[j]);
          } else {
            CP_STEP(wh, wl, Gsm[j]);
          }
        }
        Ssm[512 + lane] = gh + gl_;
        Ssm[528 + lane] = wh + wl;
      }
    } else {
      const int kind = tk % 3;                      // 0: (0,0)  1: (1,0)  2: (1,1)
      const int rb = kind == 0 ? 0 : 1;
      const int cb = kind == 2 ? 1 : 0;
      const threadgroup float* abase = (tk < 3 ? Ksm : Qsm) + rb * 8 * LD;
      simdgroup_matrix<float, 8, 8> acc(0.0f), acc2(0.0f);
      #pragma clang loop unroll(full)
      for (int kb = 0; kb < NKB; kb += 2) {
        simdgroup_matrix<float, 8, 8> a, b, a2, b2;
        simdgroup_load(a, abase + kb * 8, LD);
        simdgroup_load(b, Ksm + cb * 8 * LD + kb * 8, LD, ulong2(0, 0), true);
        simdgroup_load(a2, abase + (kb + 1) * 8, LD);
        simdgroup_load(b2, Ksm + cb * 8 * LD + (kb + 1) * 8, LD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(acc, a, b, acc);
        simdgroup_multiply_accumulate(acc2, a2, b2, acc2);
      }
      #pragma clang loop unroll(full)
      for (int e = 0; e < 2; e++) {
        acc.thread_elements()[e] += acc2.thread_elements()[e];
      }
      simdgroup_matrix<float, 8, 8> out;
      const int t = rb * 8 + fm;
      const int i1 = cb * 8 + fn + 1;                 // columns of this lane: i1 - 1 and i1
      float gr[C];
      #pragma clang loop unroll(full)
      for (int j = 0; j < C; j++) {
        gr[j] = Gsm[j];
      }
      // E[t][i1] and E[t][i1 - 1] as (hi + lo) pairs.  Same-block tiles: the product of g_{i+1} ... g_t.  Tile (1,0):
      // F(i) H(t), F(i) = g_{i+1} ... g_7, H(t) = g_8 ... g_t.  E[t][i1 - 1] = g_{i1} E[t][i1] for i1 - 1 < t.
      float e1h = 1.0f, e1l = 0.0f;
      if (kind == 1) {
        float fh = 1.0f, fl = 0.0f, hh = 1.0f, hl = 0.0f;
        #pragma clang loop unroll(full)
        for (int j = 2; j < 8; j++) {
          if (j > i1) {
            CP_STEP(fh, fl, gr[j]);
          }
        }
        #pragma clang loop unroll(full)
        for (int j = 8; j < C; j++) {
          if (j <= t) {
            CP_STEP(hh, hl, gr[j]);
          }
        }
        const float p_ = fh * hh;
        e1l = fma(fh, hh, -p_) + (fh * hl + fl * hh);
        e1h = p_;
      } else {
        const int base = kind == 0 ? 0 : 8;
        #pragma clang loop unroll(full)
        for (int j = 1; j < 8; j++) {
          if (j + base > i1 && j + base <= t) {
            CP_STEP(e1h, e1l, gr[j + base]);
          }
        }
      }
      const float e1 = e1h + e1l;
      float e0h = e1h, e0l = e1l;
      const float gi1 = Gsm[i1];
      CP_STEP(e0h, e0l, gi1);
      // the diagonal column (i == t) has E = 1; for kind 1 every column is below the diagonal
      const float e0 = (kind == 1 || t > i1 - 1) ? (e0h + e0l) : 1.0f;
      #pragma clang loop unroll(full)
      for (int e = 0; e < 2; e++) {
        const int i = i1 - 1 + e;
        const float ee = e == 0 ? e0 : e1;
        const float v_ = acc.thread_elements()[e] * ee;
        if (tk < 3) {
          out.thread_elements()[e] = (i < t) ? ((tk == 1 ? -1.0f : 1.0f) * (rb == 0 ? beta0 : beta1) * v_) : 0.0f;
        } else {
          out.thread_elements()[e] = (i <= t) ? v_ : 0.0f;
        }
      }
      if (tk == 1) {
        simdgroup_store(out, Ssm + 256 + 8 * C, C);
      } else if (tk >= 3) {
        simdgroup_store(out, Ssm + rb * 8 * C + cb * 8, C);
      } else {
        threadgroup float* at = &Atmp[kind / 2][0];
        simdgroup_store(out, at, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 8) {
          float x[8];
          #pragma clang loop unroll(full)
          for (int r = 0; r < 8; r++) {
            float acc2 = (r == lane) ? 1.0f : 0.0f;
            #pragma clang loop unroll(full)
            for (int j = 0; j < 8; j++) {
              if (j < r) {
                acc2 -= at[r * 8 + j] * x[j];
              }
            }
            x[r] = acc2;
            Ssm[256 + (rb * 8 + r) * C + rb * 8 + lane] = acc2;
          }
        }
      }
    }
  }

  // G = K S0^T and P = Q S0^T  ([C x 8] each, two 8-row tiles).  Four accumulator chains per product (key blocks
  // kb % 4) shorten the dependent MMA chain and cut the rounding error of the 128-term sums (real layers: kernel error
  // 1.0-1.5x -> 0.8-1.0x of the recurrent kernel's).
  simdgroup_matrix<float, 8, 8> Ga[4], Gb[4], Pa[4], Pb[4];
  #pragma clang loop unroll(full)
  for (int u = 0; u < 4; u++) {
    Ga[u] = simdgroup_matrix<float, 8, 8>(0.0f);
    Gb[u] = simdgroup_matrix<float, 8, 8>(0.0f);
    Pa[u] = simdgroup_matrix<float, 8, 8>(0.0f);
    Pb[u] = simdgroup_matrix<float, 8, 8>(0.0f);
  }
  #pragma clang loop unroll(full)
  for (int kb = 0; kb < NKB; kb++) {
    const int u = kb & 3;
    simdgroup_matrix<float, 8, 8> a0, a1, b0, b1;
    simdgroup_load(a0, Ksm + kb * 8, LD);
    simdgroup_load(a1, Ksm + 8 * LD + kb * 8, LD);
    simdgroup_load(b0, Qsm + kb * 8, LD);
    simdgroup_load(b1, Qsm + 8 * LD + kb * 8, LD);
    simdgroup_multiply_accumulate(Ga[u], a0, S[kb], Ga[u]);
    simdgroup_multiply_accumulate(Gb[u], a1, S[kb], Gb[u]);
    simdgroup_multiply_accumulate(Pa[u], b0, S[kb], Pa[u]);
    simdgroup_multiply_accumulate(Pb[u], b1, S[kb], Pb[u]);
  }
  simdgroup_matrix<float, 8, 8> G0, G1, P0, P1;
  #pragma clang loop unroll(full)
  for (int e = 0; e < 2; e++) {
    G0.thread_elements()[e] = (Ga[0].thread_elements()[e] + Ga[1].thread_elements()[e]) + (Ga[2].thread_elements()[e] + Ga[3].thread_elements()[e]);
    G1.thread_elements()[e] = (Gb[0].thread_elements()[e] + Gb[1].thread_elements()[e]) + (Gb[2].thread_elements()[e] + Gb[3].thread_elements()[e]);
    P0.thread_elements()[e] = (Pa[0].thread_elements()[e] + Pa[1].thread_elements()[e]) + (Pa[2].thread_elements()[e] + Pa[3].thread_elements()[e]);
    P1.thread_elements()[e] = (Pb[0].thread_elements()[e] + Pb[1].thread_elements()[e]) + (Pb[2].thread_elements()[e] + Pb[3].thread_elements()[e]);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);   // chunk factors in Ssm are complete
  const float gam0 = sc[512 + fm];
  const float gam1 = sc[512 + 8 + fm];
  const float wgt0 = sc[528 + fm];
  const float wgt1 = sc[528 + 8 + fm];
  const float gc = sc[512 + 15];
  // rhs = beta_t * (v_t - gam_t * (S0 k_t))
  const device float* vp = v + ((size_t)c * C * Hv + h) * Dv + dv0 + fn;
  simdgroup_matrix<float, 8, 8> r0, r1;
  #pragma clang loop unroll(full)
  for (int e = 0; e < 2; e++) {
    r0.thread_elements()[e] = beta0 * (vp[(size_t)fm * VS + e] - gam0 * G0.thread_elements()[e]);
    r1.thread_elements()[e] = beta1 * (vp[(size_t)(8 + fm) * VS + e] - gam1 * G1.thread_elements()[e]);
  }
  // blocked forward substitution: delta0 = D0 r0 ; delta1 = D1 (r1 + N10 delta0), D = inv(I + A diag block), N10 = -A10
  simdgroup_matrix<float, 8, 8> D0, D1, N10, M00, M10, M11;
  simdgroup_load(D0, sc + 256, C);
  simdgroup_load(N10, sc + 256 + 8 * C, C);
  simdgroup_load(D1, sc + 256 + 8 * C + 8, C);
  simdgroup_load(M00, sc, C);
  simdgroup_load(M10, sc + 8 * C, C);
  simdgroup_load(M11, sc + 8 * C + 8, C);
  simdgroup_matrix<float, 8, 8> d0(0.0f), d1(0.0f);
  simdgroup_multiply_accumulate(d0, D0, r0, d0);
  simdgroup_multiply_accumulate(r1, N10, d0, r1);
  simdgroup_multiply_accumulate(d1, D1, r1, d1);

  // y = gam * (Q S0^T) + M delta
  #pragma clang loop unroll(full)
  for (int e = 0; e < 2; e++) {
    P0.thread_elements()[e] *= gam0;
    P1.thread_elements()[e] *= gam1;
  }
  simdgroup_multiply_accumulate(P0, M00, d0, P0);
  simdgroup_multiply_accumulate(P1, M10, d0, P1);
  simdgroup_multiply_accumulate(P1, M11, d1, P1);
  device float* yp = y + ((size_t)c * C * Hv + h) * Dv + dv0;
  simdgroup_store(P0, yp, VS);
  simdgroup_store(P1, yp + 8 * VS, VS);

  // S0^T <- gam_{C-1} S0^T + K^T diag(wgt) delta
  #pragma clang loop unroll(full)
  for (int e = 0; e < 2; e++) {
    d0.thread_elements()[e] *= wgt0;
    d1.thread_elements()[e] *= wgt1;
  }
  #pragma clang loop unroll(full)
  for (int kb = 0; kb < NKB; kb++) {
    simdgroup_matrix<float, 8, 8> t0, t1;
    simdgroup_load(t0, Ksm + kb * 8, LD, ulong2(0, 0), true);
    simdgroup_load(t1, Ksm + 8 * LD + kb * 8, LD, ulong2(0, 0), true);
    S[kb].thread_elements()[0] *= gc;
    S[kb].thread_elements()[1] *= gc;
    simdgroup_multiply_accumulate(S[kb], t0, d0, S[kb]);
    simdgroup_multiply_accumulate(S[kb], t1, d1, S[kb]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);   // every simdgroup is done reading the staged chunk
  if (c + 1 < nfull) {
    commit();
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ---- trailing tokens: scalar recurrent steps on the fragment layout ----
// Lane (fm, fn) holds S[dk = kb*8 + fm][dv = dv0 + fn + e]; the 8 lanes sharing fn differ in the lane bits 1, 2 and 4.
for (int t = nfull * C; t < T; t++) {
  const float gt = g[(size_t)t * Hv + h];
  const float bt = beta[(size_t)t * Hv + h];
  const device float* kt = k + ((size_t)t * Hk + hk) * Dk + fm;
  const device float* qt = q + ((size_t)t * Hk + hk) * Dk + fm;
  const device float* vt = v + ((size_t)t * Hv + h) * Dv + dv0 + fn;
  float kv0 = 0.0f;
  float kv1 = 0.0f;
  #pragma clang loop unroll(full)
  for (int kb = 0; kb < NKB; kb++) {
    const float kk = kt[kb * 8];
    S[kb].thread_elements()[0] *= gt;
    S[kb].thread_elements()[1] *= gt;
    kv0 += S[kb].thread_elements()[0] * kk;
    kv1 += S[kb].thread_elements()[1] * kk;
  }
  kv0 += simd_shuffle_xor(kv0, (ushort)2);
  kv0 += simd_shuffle_xor(kv0, (ushort)4);
  kv0 += simd_shuffle_xor(kv0, (ushort)16);
  kv1 += simd_shuffle_xor(kv1, (ushort)2);
  kv1 += simd_shuffle_xor(kv1, (ushort)4);
  kv1 += simd_shuffle_xor(kv1, (ushort)16);
  const float dl0 = (vt[0] - kv0) * bt;
  const float dl1 = (vt[1] - kv1) * bt;
  float o0 = 0.0f;
  float o1 = 0.0f;
  #pragma clang loop unroll(full)
  for (int kb = 0; kb < NKB; kb++) {
    const float kk = kt[kb * 8];
    S[kb].thread_elements()[0] += kk * dl0;
    S[kb].thread_elements()[1] += kk * dl1;
    const float qq = qt[kb * 8];
    o0 += S[kb].thread_elements()[0] * qq;
    o1 += S[kb].thread_elements()[1] * qq;
  }
  o0 += simd_shuffle_xor(o0, (ushort)2);
  o0 += simd_shuffle_xor(o0, (ushort)4);
  o0 += simd_shuffle_xor(o0, (ushort)16);
  o1 += simd_shuffle_xor(o1, (ushort)2);
  o1 += simd_shuffle_xor(o1, (ushort)4);
  o1 += simd_shuffle_xor(o1, (ushort)16);
  if ((lane & 22) == 0) {
    device float* yt = y + ((size_t)t * Hv + h) * Dv + dv0 + fn;
    yt[0] = o0;
    yt[1] = o1;
  }
}

device float* sout_p = state_out + ((size_t)h * Dv + dv0) * Dk;
#pragma clang loop unroll(full)
for (int kb = 0; kb < NKB; kb++) {
  simdgroup_store(S[kb], sout_p + kb * 8, Dk, ulong2(0, 0), true);
}
