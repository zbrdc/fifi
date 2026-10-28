#include "repack.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "mma884.cuh"

using mma884::frag_row;
using mma884::kTileK;
using mma884::quadpair;

namespace {

// ggml block layouts (byte offsets).
constexpr int kQ8Block = 34;    // fp16 d, int8 qs[32]
constexpr int kQ4KBlock = 144;  // fp16 d, fp16 dmin, scales[12], qs[128]
constexpr int kQ6KBlock = 210;  // ql[128], qh[64], int8 scales[16], fp16 d

// Each pack writes one tile from rows[r], the first ggml block of row r that
// falls in the tile.
void pack_q4k(const uint8_t* const rows[8], uint4* tile) {
  for (int r = 0; r < 8; ++r) memcpy(&tile[r], rows[r], 16);
  for (int h = 0; h < 2; ++h) {
    for (int lane = 0; lane < 32; ++lane) {
      const uint8_t* qs = rows[frag_row(lane)] + 16;
      memcpy(&tile[8 + 32 * h + lane], qs + 32 * quadpair(lane) + 16 * h, 16);
    }
  }
}

void pack_q8_0(const uint8_t* const rows[8], uint4* tile) {
  auto* dtab = reinterpret_cast<uint8_t*>(tile);
  for (int r = 0; r < 8; ++r)
    for (int b = 0; b < 8; ++b) memcpy(dtab + 2 * (r * 8 + b), rows[r] + b * kQ8Block, 2);
  for (int j = 0; j < 4; ++j) {
    for (int lane = 0; lane < 32; ++lane) {
      int kk = 64 * quadpair(lane) + 16 * j;
      const uint8_t* block = rows[frag_row(lane)] + (kk / 32) * kQ8Block;
      memcpy(&tile[8 + 32 * j + lane], block + 2 + kk % 32, 16);
    }
  }
}

// Unsigned 6-bit q of element i, following ggml's dequantize_row_q6_K.
int q6(const uint8_t* block, int i) {
  int half = i / 128, quarter = (i % 128) / 32, l = i % 32;
  uint8_t ql = block[64 * half + l + 32 * (quarter & 1)];
  uint8_t qh = block[128 + 32 * half + l];
  int lo4 = quarter < 2 ? ql & 0xF : ql >> 4;
  return lo4 | ((qh >> (2 * quarter)) & 3) << 4;
}

void pack_q6k_lane(const uint8_t* block, int lane, uint8_t* bytes) {
  int k0 = 64 * quadpair(lane);
  uint8_t* ql = bytes + 16 * lane;
  uint8_t* ql2 = bytes + 16 * (32 + lane);
  uint8_t* qh = bytes + 16 * (64 + lane);
  // fp16 d * scale of the lane's four sub-blocks, rounded once from the exact fp32 product.
  __half d;
  memcpy(&d, block + 208, 2);
  for (int i = 0; i < 4; ++i) {
    __half s = __float2half_rn(__half2float(d) * (int8_t)block[192 + 4 * (k0 / 64) + i]);
    memcpy(bytes + 16 * 96 + 8 * lane + 2 * i, &s, 2);
  }
  memset(qh, 0, 16);
  for (int wi = 0; wi < 8; ++wi) {
    for (int v = 0; v < 4; ++v) {
      int lo = q6(block, k0 + 4 * wi + v), hi = q6(block, k0 + 32 + 4 * wi + v);
      (wi < 4 ? ql : ql2)[4 * (wi % 4) + v] = (lo & 0xF) | (hi & 0xF) << 4;
      qh[4 * (wi / 2) + v] |= ((lo >> 4) | (hi >> 4) << 2) << (4 * (wi % 2));
    }
  }
}

void pack_q6k(const uint8_t* const rows[8], uint4* tile) {
  auto* bytes = reinterpret_cast<uint8_t*>(tile);
  for (int lane = 0; lane < 32; ++lane) pack_q6k_lane(rows[frag_row(lane)], lane, bytes);
}

constexpr int kQ4_0Block = 18;   // fp16 d, qs[16]: low nibbles are 0..15, high 16..31
constexpr int kQ5KBlock = 176;   // fp16 d, fp16 dmin, scales[12], qh[32], qs[128]

float half_at(const uint8_t* p) {
  __half h;
  memcpy(&h, p, 2);
  return __half2float(h);
}

// ggml's quantize_row_q8_0_ref for one block of 32.
void quantize_q8_0(const float* x, uint8_t* block) {
  float amax = 0;
  for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
  float d = amax / 127, id = d != 0 ? 1 / d : 0;
  __half dh = __float2half_rn(d);
  memcpy(block, &dh, 2);
  for (int i = 0; i < 32; ++i) block[2 + i] = (uint8_t)(int8_t)std::lround(x[i] * id);
}

// Exact: x = d * (q - 8) is already a Q8_0 value with the same d.
void q4_0_to_q8_0(const uint8_t* src, uint8_t* block) {
  memcpy(block, src, 2);
  for (int i = 0; i < 16; ++i) {
    block[2 + i] = (uint8_t)(int8_t)((src[2 + i] & 0xF) - 8);
    block[18 + i] = (uint8_t)(int8_t)((src[2 + i] >> 4) - 8);
  }
}

// ggml's get_scale_min_k4.
void scale_min_k4(int j, const uint8_t* q, int* sc, int* m) {
  if (j < 4) {
    *sc = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *sc = (q[j + 4] & 0xF) | (q[j - 4] >> 6) << 4;
    *m = (q[j + 4] >> 4) | (q[j] >> 6) << 4;
  }
}

// ggml's dequantize_row_q5_K for one block of 256.
void dequant_q5k(const uint8_t* block, float* y) {
  float d = half_at(block), dmin = half_at(block + 2);
  const uint8_t* scales = block + 4;
  const uint8_t* qh = block + 16;
  const uint8_t* ql = block + 48;
  for (int j = 0; j < 4; ++j) {
    for (int h = 0; h < 2; ++h) {
      int sc, m;
      scale_min_k4(2 * j + h, scales, &sc, &m);
      int bit = 1 << (2 * j + h);
      for (int l = 0; l < 32; ++l) {
        int q = (h == 0 ? ql[32 * j + l] & 0xF : ql[32 * j + l] >> 4) + (qh[l] & bit ? 16 : 0);
        y[64 * j + 32 * h + l] = d * sc * q - dmin * m;
      }
    }
  }
}

// Weights kb..kb+255 of a float row (F32, F16 or Q5_K) into x.
void dequant_256(GgmlType type, const uint8_t* src, int kb, float* x) {
  if (type == GgmlType::Q5_K) return dequant_q5k(src + (size_t)kb / 256 * kQ5KBlock, x);
  for (int i = 0; i < 256; ++i) {
    float f;
    if (type == GgmlType::F32) memcpy(&f, src + 4 * ((size_t)kb + i), 4);
    else f = half_at(src + 2 * ((size_t)kb + i));
    x[i] = f;
  }
}

// One row of k weights to Q8_0 blocks at dst.
void row_to_q8_0(GgmlType type, const uint8_t* src, int k, uint8_t* dst) {
  if (type == GgmlType::Q4_0) {
    for (int b = 0; b < k / 32; ++b) q4_0_to_q8_0(src + (size_t)b * kQ4_0Block, dst + (size_t)b * kQ8Block);
    return;
  }
  float x[256];
  for (int kb = 0; kb < k; kb += 256) {
    dequant_256(type, src, kb, x);
    for (int i = 0; i < 8; ++i) quantize_q8_0(x + 32 * i, dst + (size_t)(kb / 32 + i) * kQ8Block);
  }
}

}  // namespace

bool converts_to_q8_0(GgmlType type) {
  return type == GgmlType::F32 || type == GgmlType::F16 || type == GgmlType::Q4_0 ||
         type == GgmlType::Q5_K;
}

std::vector<uint8_t> to_q8_0(GgmlType type, const uint8_t* src, size_t n_rows, int k) {
  if (!converts_to_q8_0(type) || k % 256 != 0) return {};
  size_t in_bytes = ggml_row_bytes(type, k), out_bytes = ggml_row_bytes(GgmlType::Q8_0, k);
  std::vector<uint8_t> out(n_rows * out_bytes);
  for (size_t r = 0; r < n_rows; ++r) row_to_q8_0(type, src + r * in_bytes, k, out.data() + r * out_bytes);
  return out;
}

size_t ggml_row_bytes(GgmlType type, int k) {
  switch (type) {
    case GgmlType::F32: return 4 * (size_t)k;
    case GgmlType::F16: return 2 * (size_t)k;
    case GgmlType::Q4_0: return (size_t)k / 32 * kQ4_0Block;
    case GgmlType::Q8_0: return (size_t)k / 32 * kQ8Block;
    case GgmlType::Q4_K: return (size_t)k / 256 * kQ4KBlock;
    case GgmlType::Q5_K: return (size_t)k / 256 * kQ5KBlock;
    case GgmlType::Q6_K: return (size_t)k / 256 * kQ6KBlock;
  }
  return 0;
}

bool is_tiled(GgmlType type) {
  return type == GgmlType::Q8_0 || type == GgmlType::Q4_K || type == GgmlType::Q6_K;
}

size_t tiled_vecs(GgmlType type, int n_rows, int k) {
  size_t tiles = (size_t)(n_rows / 8) * (k / kTileK);
  switch (type) {
    case GgmlType::Q8_0: return tiles * mma884::Q8_0::kTileVecs;
    case GgmlType::Q4_K: return tiles * mma884::Q4_K::kTileVecs;
    case GgmlType::Q6_K: return tiles * mma884::Q6_K::kTileVecs;
    default: return 0;
  }
}

bool repack(GgmlType type, const uint8_t* src, int n_rows, int k, uint4* dst) {
  void (*pack)(const uint8_t* const[8], uint4*) = nullptr;
  if (type == GgmlType::Q8_0) pack = pack_q8_0;
  if (type == GgmlType::Q4_K) pack = pack_q4k;
  if (type == GgmlType::Q6_K) pack = pack_q6k;
  if (!pack || n_rows % 8 != 0 || k % kTileK != 0) return false;
  size_t row_bytes = ggml_row_bytes(type, k), tile_bytes = ggml_row_bytes(type, kTileK);
  size_t tile_vecs = tiled_vecs(type, 8, kTileK);
  int nb = k / kTileK;
  for (int g = 0; g < n_rows / 8; ++g) {
    for (int ib = 0; ib < nb; ++ib) {
      const uint8_t* rows[8];
      for (int r = 0; r < 8; ++r) rows[r] = src + (size_t)(g * 8 + r) * row_bytes + ib * tile_bytes;
      pack(rows, dst + ((size_t)g * nb + ib) * tile_vecs);
    }
  }
  return true;
}
