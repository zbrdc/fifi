// MIT License, Copyright (c) 2026 Dan Campos (see ../LICENSE).
// Restored from xavier-qwen 2f9183c src/kernels.cu (RD-4V); provenance.json lists the line
// ranges taken unchanged and the patches applied since.

#include "kernels.h"
#include "ops.h"

#include "frag_layout.cuh"

namespace kern {
namespace {

__device__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Sum of v over the block, returned to every thread.
__device__ float block_sum(float v) {
  __shared__ float red[32];
  v = warp_sum(v);
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  __syncthreads();  // earlier callers may still be reading red
  if (lane == 0) red[warp] = v;
  __syncthreads();
  v = lane < (int)blockDim.x / 32 ? red[lane] : 0.f;
  return warp_sum(v);
}

__device__ float silu(float x) { return x / (1.f + __expf(-x)); }
__device__ float sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ float softplus(float x) { return x > 20.f ? x : log1pf(__expf(x)); }

__global__ void add_rmsnorm_pack_k(float* h, const float* delta, const float* w, int n,
                                   float eps, bool rowmajor, uint4* xf, float* norm_out) {
  int tok = blockIdx.x;
  float* row = h + (size_t)tok * n;
  float ss = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = row[i];
    if (delta) {
      v += delta[(size_t)tok * n + i];
      row[i] = v;
    }
    ss += v * v;
  }
  float rstd = rsqrtf(block_sum(ss) / n + eps);
  // write_xf calls f once per element, so norm_out holds the f32 value that xf rounds to fp16.
  write_xf(xf, n, tok, rowmajor, [&](int i) {
    float v = row[i] * rstd * w[i];
    if (norm_out) norm_out[(size_t)tok * n + i] = v;
    return v;
  });
}

__global__ void swiglu_pack_k(const float* gu, int n_ff, bool rowmajor, uint4* xf) {
  const float* g = gu + (size_t)blockIdx.x * 2 * n_ff;
  write_xf(xf, n_ff, blockIdx.x, rowmajor, [&](int i) { return silu(g[i]) * g[n_ff + i]; });
}

__global__ void rmsnorm_k(const float* x, const float* w, int n, float eps, float* out) {
  const float* row = x + (size_t)blockIdx.x * n;
  float ss = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) ss += row[i] * row[i];
  float rstd = rsqrtf(block_sum(ss) / n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    out[(size_t)blockIdx.x * n + i] = row[i] * rstd * w[i];
}

__global__ void conv_silu_k(ConvArgs a) {
  int c = blockIdx.x * blockDim.x + threadIdx.x, n = a.channels;
  if (c >= n) return;
  float s0 = a.state[c], s1 = a.state[n + c], s2 = a.state[2 * n + c];
  float4 wc = reinterpret_cast<const float4*>(a.w)[c];
  for (int t = 0; t < a.m; ++t) {
    float x = a.proj[(size_t)t * a.ld + c];
    float v = s0 * wc.x + s1 * wc.y + s2 * wc.z + x * wc.w;
    a.out[(size_t)t * n + c] = silu(v);
    s0 = s1;
    s1 = s2;
    s2 = x;
    if (a.snap && t < a.m - 1) {
      float* sn = a.snap + (size_t)t * 3 * n;
      sn[c] = s0;
      sn[n + c] = s1;
      sn[2 * n + c] = s2;
    }
  }
  a.state[c] = s0;
  a.state[n + c] = s1;
  a.state[2 * n + c] = s2;
}

// One block per v head; thread j owns column j of the head's 128x128 state
// S[i][j] (i over keys), kept in registers across the m tokens.
__global__ void __launch_bounds__(kStateDim) gated_delta_k(DeltaArgs a) {
  constexpr int D = kStateDim;
  __shared__ float qs[D], ks[D];
  int h = blockIdx.x, j = threadIdx.x, kh = h % a.n_k_heads;
  float* st = a.state + (size_t)h * D * D;
  float s[D];
#pragma unroll
  for (int i = 0; i < D; ++i) s[i] = st[i * D + j];

  for (int t = 0; t < a.m; ++t) {
    const float* c = a.conv + (size_t)t * a.conv_ld;
    const float* p = a.proj + (size_t)t * a.ld;
    float qv = c[kh * D + j], kv = c[(a.n_k_heads + kh) * D + j];
    float vv = c[2 * a.n_k_heads * D + h * D + j];
    // L2 norms, and q's 1/sqrt(D) attention scale.
    float qn = rsqrtf(block_sum(qv * qv) + a.eps) * rsqrtf((float)D);
    float kn = rsqrtf(block_sum(kv * kv) + a.eps);
    qs[j] = qv * qn;
    ks[j] = kv * kn;
    __syncthreads();
    float decay = __expf(softplus(p[a.alpha_off + h] + a.dt_bias[h]) * a.a[h]);
    float beta = sigmoid(p[a.beta_off + h]);
    float sk = 0.f;
#pragma unroll
    for (int i = 0; i < D; ++i) {
      s[i] *= decay;
      sk += s[i] * ks[i];
    }
    float d = (vv - sk) * beta, o = 0.f;
#pragma unroll
    for (int i = 0; i < D; ++i) {
      s[i] += ks[i] * d;
      o += s[i] * qs[i];
    }
    float rstd = rsqrtf(block_sum(o * o) / D + a.eps);
    float z = p[a.z_off + h * D + j];
    a.out[(size_t)t * a.n_v_heads * D + h * D + j] = o * rstd * a.norm_w[j] * silu(z);
    if (a.snap && t < a.m - 1) {
      float* sn = a.snap + ((size_t)t * a.n_v_heads + h) * D * D;
#pragma unroll
      for (int i = 0; i < D; ++i) sn[i * D + j] = s[i];
    }
  }
#pragma unroll
  for (int i = 0; i < D; ++i) st[i * D + j] = s[i];
}

// Block (head, token), thread per dim: RMSNorm, NEOX rope over the first n_rot
// dims. Heads are the n_head queries, then n_kv_head keys, then values.
__global__ void attn_prep_k(AttnArgs a) {
  constexpr int D = kHeadDim;
  __shared__ float x[D];
  int head = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
  int pos = (a.pos_dev ? *a.pos_dev : a.pos0) + t;
  const float* p = a.proj + (size_t)t * a.ld;
  int kv_base = a.n_head * 2 * D;
  if (head >= a.n_head + a.n_kv_head) {
    int vh = head - a.n_head - a.n_kv_head;
    float v = p[kv_base + (a.n_kv_head + vh) * D + d];
    a.v_cache[((size_t)pos * a.n_kv_head + vh) * D + d] = __float2half(v);
    return;
  }
  bool is_q = head < a.n_head;
  float v = is_q ? p[head * 2 * D + d] : p[kv_base + (head - a.n_head) * D + d];
  float rstd = rsqrtf(block_sum(v * v) / D + a.eps);
  x[d] = v * rstd * (is_q ? a.q_norm : a.k_norm)[d];
  __syncthreads();
  float out = x[d];
  int half = a.n_rot / 2;
  if (d < a.n_rot) {
    int i = d % half;
    float theta = pos * powf(a.rope_base, -2.f * i / a.n_rot), sn, cs;
    sincosf(theta, &sn, &cs);
    out = d < half ? x[d] * cs - x[d + half] * sn : x[d - half] * sn + x[d] * cs;
  }
  if (is_q)
    a.q[((size_t)t * a.n_head + head) * D + d] = out;
  else
    a.k_cache[((size_t)pos * a.n_kv_head + head - a.n_head) * D + d] = __float2half(out);
}

}  // namespace

cudaError_t add_rmsnorm_pack(float* h, const float* delta, const float* w, int m, int n,
                             float eps, uint4* xf, float* norm_out, cudaStream_t stream) {
  add_rmsnorm_pack_k<<<m, 256, 0, stream>>>(h, delta, w, n, eps, m > kMaxTokens, xf, norm_out);
  return cudaGetLastError();
}

cudaError_t rmsnorm(const float* x, const float* w, int m, int n, float eps, float* out,
                    cudaStream_t stream) {
  rmsnorm_k<<<m, 256, 0, stream>>>(x, w, n, eps, out);
  return cudaGetLastError();
}

cudaError_t swiglu_pack(const float* gu, int m, int n_ff, uint4* xf, cudaStream_t stream) {
  swiglu_pack_k<<<m, 256, 0, stream>>>(gu, n_ff, m > kMaxTokens, xf);
  return cudaGetLastError();
}

cudaError_t conv_silu(const ConvArgs& a, cudaStream_t stream) {
  conv_silu_k<<<(a.channels + 255) / 256, 256, 0, stream>>>(a);
  return cudaGetLastError();
}

cudaError_t gated_delta(const DeltaArgs& a, cudaStream_t stream) {
  gated_delta_k<<<a.n_v_heads, kStateDim, 0, stream>>>(a);
  return cudaGetLastError();
}

// The attn_prep_k launch of upstream attention() (2f9183c kernels.cu:845) as an entry of its own.
cudaError_t attn_prep(const AttnArgs& a, cudaStream_t stream) {
  attn_prep_k<<<dim3(a.n_head + 2 * a.n_kv_head, a.m), kHeadDim, 0, stream>>>(a);
  return cudaGetLastError();
}


}  // namespace kern
