// GGUF dequantization oracle driver.
//
// This file is original project code. It links against ggml-quants.c from the
// pinned llama.cpp commit and uses that library as the reference output. The
// ggml sources stay outside the repository.
//
// Usage: oracle <out-dir> <seed>
//        oracle --convert <name> <in.blocks> <out.f32>
// For each type the program writes <name>.blocks (raw stored bytes) and
// <name>.f32 (the ggml CPU dequantization, little-endian F32 bits).

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-impl.h"
#include "ggml-quants.h"

// Stubs for the few ggml.c symbols that ggml-quants.c references. The
// dequantizers do not call them.
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "ggml_abort %s:%d %s\n", file, line, fmt);
    abort();
}
size_t ggml_row_size(enum ggml_type type, int64_t ne) { (void) type; (void) ne; abort(); }
const char * ggml_type_name(enum ggml_type type) { (void) type; return "stub"; }
size_t ggml_type_size(enum ggml_type type) { (void) type; abort(); }

// Wrappers call the real prototypes from ggml-quants.h. The pointer cast only
// changes the block type; the byte layout is the one the header declares.
typedef void (*deq_fn)(const void *, float *, int64_t);
#define WRAP(name, blk, fn) \
    static void name(const void * x, float * y, int64_t k) { fn((const blk *) x, y, k); }
WRAP(w_q2_0, block_q2_0, dequantize_row_q2_0)
WRAP(w_q4_k, block_q4_K, dequantize_row_q4_K)
WRAP(w_q5_k, block_q5_K, dequantize_row_q5_K)
WRAP(w_q6_k, block_q6_K, dequantize_row_q6_K)
WRAP(w_q8_0, block_q8_0, dequantize_row_q8_0)
WRAP(w_iq2_xxs, block_iq2_xxs, dequantize_row_iq2_xxs)
WRAP(w_iq2_xs, block_iq2_xs, dequantize_row_iq2_xs)
WRAP(w_iq2_s, block_iq2_s, dequantize_row_iq2_s)
WRAP(w_iq3_xxs, block_iq3_xxs, dequantize_row_iq3_xxs)
WRAP(w_iq3_s, block_iq3_s, dequantize_row_iq3_s)
WRAP(w_iq4_nl, block_iq4_nl, dequantize_row_iq4_nl)
WRAP(w_iq4_xs, block_iq4_xs, dequantize_row_iq4_xs)

static uint64_t rng_state;
static uint64_t rng_next(void) {
    // xorshift64*
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 0x2545F4914F6CDD1DULL;
}
static void rng_seed(uint64_t seed, uint64_t salt) {
    rng_state = (seed ^ (salt * 0x9E3779B97F4A7C15ULL)) | 1;
    for (int i = 0; i < 8; i++) rng_next();
}
static void rng_fill(uint8_t * p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t) (rng_next() >> 56);
}

static void put16(uint8_t * p, uint16_t v) { p[0] = v & 0xff; p[1] = v >> 8; }
static void put32(uint8_t * p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (v >> (8 * i)) & 0xff; }

typedef struct {
    int fill; // 0 zero, 1 all 0xff, 2 random
    uint16_t d, dmin;
} edge_t;

static const edge_t EDGES[] = {
    {0, 0x0000, 0x0000}, {0, 0x3c00, 0x3c00}, {1, 0x3c00, 0x3c00}, {1, 0x7bff, 0x7bff},
    {2, 0x0001, 0x0001}, {2, 0x03ff, 0x8001}, {2, 0x7c00, 0x3c00}, {2, 0xfc00, 0xfc00},
    {2, 0x7e00, 0x3c00}, {2, 0x7d55, 0xfd01}, {2, 0x8000, 0x8000}, {2, 0xbc00, 0x3555},
    {2, 0x3c00, 0x7e01},
};
#define N_EDGES (int) (sizeof(EDGES) / sizeof(EDGES[0]))

// Sweep blocks make every grid index and sign index appear at least once.
// `s` is the sweep block number. Each function writes the structured fields
// of one block. The scale field stays at the random fill value.
static void sweep_iq2_xxs(uint8_t * b, int s) {
    // d(2) then 8 groups of: 4 grid bytes, then u32 (4 x 7-bit signs, 4-bit scale).
    for (int g = 0; g < 8; g++) {
        uint8_t * q = b + 2 + 8 * g;
        uint32_t signs = 0;
        for (int l = 0; l < 4; l++) {
            int idx = s * 32 + g * 4 + l;
            q[l] = idx & 0xff;
            signs |= (uint32_t) ((idx * 5 + s) & 127) << (7 * l);
        }
        put32(q + 4, signs | ((uint32_t) ((g + s) & 15) << 28));
    }
}
static void sweep_iq2_xs(uint8_t * b, int s) {
    // d(2), 32 x u16 qs (9-bit grid, 7-bit sign), 8 scale bytes.
    for (int i = 0; i < 32; i++) {
        int idx = s * 32 + i;
        uint16_t v = (uint16_t) ((idx & 511) | (((idx * 3 + s) & 127) << 9));
        put16(b + 2 + 2 * i, v);
    }
}
static void sweep_iq2_s(uint8_t * b, int s) {
    // d(2), qs[64] (low 8 bits of grid, then 32 sign bytes), qh[8], scales[8].
    // Grid index = qs[l] | ((qh[ib32] << (8-2l)) & 0x300), so qh holds 2 bits per l.
    uint8_t * qs = b + 2;
    uint8_t * qh = b + 2 + 64;
    for (int ib = 0; ib < 8; ib++) {
        uint8_t h = 0;
        for (int l = 0; l < 4; l++) {
            int idx = (s * 32 + ib * 4 + l) & 1023;
            qs[ib * 4 + l] = idx & 0xff;
            h |= (uint8_t) (((idx >> 8) & 3) << (2 * l));
            qs[32 + ib * 4 + l] = (uint8_t) (idx * 7 + s); // sign byte
        }
        qh[ib] = h;
    }
}
static void sweep_iq3_xxs(uint8_t * b, int s) {
    // d(2), qs[64] grid bytes, then 8 x u32 (4 x 7-bit signs, 4-bit scale).
    for (int i = 0; i < 64; i++) b[2 + i] = (uint8_t) (s * 64 + i);
    for (int g = 0; g < 8; g++) {
        uint32_t signs = 0;
        for (int l = 0; l < 4; l++) signs |= (uint32_t) ((g * 4 + l + s * 32) & 127) << (7 * l);
        put32(b + 2 + 64 + 4 * g, signs | ((uint32_t) ((g + s) & 15) << 28));
    }
}
static void sweep_iq3_s(uint8_t * b, int s) {
    // d(2), qs[64], qh[8], signs[32], scales[4].
    // Grid index = qs[k] | 9th bit from qh. For pair (ib32/2 = p, half h), l in 0..3:
    // qs[8*(2p+h)... ] — the 9th bits sit in qh[2p+h] bit (8-2l) for grid1 and (7-2l) for grid2.
    uint8_t * qs = b + 2;
    uint8_t * qh = b + 2 + 64;
    for (int blk = 0; blk < 8; blk++) {
        uint8_t h = 0;
        for (int k = 0; k < 8; k++) {
            int idx = (s * 64 + blk * 8 + k) & 511;
            qs[blk * 8 + k] = idx & 0xff;
            int l = k / 2;
            int shift = (k % 2 == 0) ? (8 - 2 * l) : (7 - 2 * l);
            // (qh << shift) & 256 picks bit (8 - shift) of qh.
            if (idx & 256) h |= (uint8_t) (1u << (8 - shift));
        }
        qh[blk] = h;
    }
}

typedef struct {
    const char * name;
    int id;
    int block_bytes;
    int block_elems;
    deq_fn fn;
    int off_d;    // byte offset of the main F16 scale
    int off_dmin; // byte offset of the second F16 scale, or -1
    int n_random;
    int n_sweep;
    void (*sweep)(uint8_t *, int);
    // Leave NaN-scale edges out. The engine rejects a NaN Q8_0 scale before any
    // device call, and the NaN sign of the output depends on the host.
    int skip_nan_d;
} type_t;

static const type_t TYPES[] = {
    {"q2_0", 42, 18, 64, w_q2_0, 0, -1, 24, 0, NULL},
    {"q4_k", 12, 144, 256, w_q4_k, 0, 2, 24, 0, NULL},
    {"q5_k", 13, 176, 256, w_q5_k, 0, 2, 24, 0, NULL},
    {"q6_k", 14, 210, 256, w_q6_k, 208, -1, 24, 0, NULL},
    {"q8_0", 8, 34, 32, w_q8_0, 0, -1, 24, 0, NULL, 1},
    {"iq2_xxs", 16, 66, 256, w_iq2_xxs, 0, -1, 16, 8, sweep_iq2_xxs},
    {"iq2_xs", 17, 74, 256, w_iq2_xs, 0, -1, 16, 16, sweep_iq2_xs},
    {"iq2_s", 22, 82, 256, w_iq2_s, 0, -1, 16, 32, sweep_iq2_s},
    {"iq3_xxs", 18, 98, 256, w_iq3_xxs, 0, -1, 16, 4, sweep_iq3_xxs},
    {"iq3_s", 21, 110, 256, w_iq3_s, 0, -1, 16, 8, sweep_iq3_s},
    {"iq4_nl", 20, 18, 32, w_iq4_nl, 0, -1, 24, 0, NULL},
    {"iq4_xs", 23, 136, 256, w_iq4_xs, 0, -1, 24, 0, NULL},
};
#define N_TYPES (int) (sizeof(TYPES) / sizeof(TYPES[0]))

static void write_file(const char * dir, const char * name, const char * ext, const void * data, size_t n) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s.%s", dir, name, ext);
    FILE * f = fopen(path, "wb");
    if (!f || fwrite(data, 1, n, f) != n) { fprintf(stderr, "write %s failed\n", path); exit(1); }
    fclose(f);
}

static int is_nan_half(uint16_t h) { return (h & 0x7c00) == 0x7c00 && (h & 0x03ff) != 0; }

static int skips_edge(const type_t * t, int e) { return t->skip_nan_d && is_nan_half(EDGES[e].d); }

static int edge_count(const type_t * t) {
    int n = 0;
    for (int e = 0; e < N_EDGES; e++) n += !skips_edge(t, e);
    return n;
}

static void run_block_type(const char * dir, uint64_t seed, const type_t * t) {
    int nblocks = edge_count(t) + t->n_random + t->n_sweep;
    size_t in_len = (size_t) nblocks * t->block_bytes;
    size_t out_n = (size_t) nblocks * t->block_elems;
    uint8_t * in = calloc(in_len, 1);
    float * out = calloc(out_n, sizeof(float));
    rng_seed(seed, (uint64_t) t->id);
    int bi = 0;
    for (int e = 0; e < N_EDGES; e++) {
        if (skips_edge(t, e)) continue;
        uint8_t * b = in + (size_t) bi++ * t->block_bytes;
        if (EDGES[e].fill == 1) memset(b, 0xff, t->block_bytes);
        else if (EDGES[e].fill == 2) rng_fill(b, t->block_bytes);
        put16(b + t->off_d, EDGES[e].d);
        if (t->off_dmin >= 0) put16(b + t->off_dmin, EDGES[e].dmin);
    }
    for (int r = 0; r < t->n_random; r++, bi++) {
        uint8_t * b = in + (size_t) bi * t->block_bytes;
        rng_fill(b, t->block_bytes);
        // Finite scales: random sign, exponent 0..30 range keeps most outputs finite.
        uint16_t d = (uint16_t) (((rng_next() >> 60) & 1) << 15 | (((rng_next() >> 59) % 24) + 4) << 10 | ((rng_next() >> 54) & 0x3ff));
        put16(b + t->off_d, d);
        if (t->off_dmin >= 0) {
            uint16_t m = (uint16_t) (((rng_next() >> 60) & 1) << 15 | (((rng_next() >> 59) % 24) + 4) << 10 | ((rng_next() >> 54) & 0x3ff));
            put16(b + t->off_dmin, m);
        }
    }
    for (int s = 0; s < t->n_sweep; s++, bi++) {
        uint8_t * b = in + (size_t) bi * t->block_bytes;
        rng_fill(b, t->block_bytes);
        put16(b + t->off_d, 0x3c00 + (uint16_t) (s % 7) * 0x0400); // finite, varied
        t->sweep(b, s);
    }
    t->fn(in, out, (int64_t) out_n);
    write_file(dir, t->name, "blocks", in, in_len);
    write_file(dir, t->name, "f32", out, out_n * sizeof(float));
    printf("%s id=%d blocks=%d bytes=%zu elems=%zu\n", t->name, t->id, nblocks, in_len, out_n);
    free(in);
    free(out);
}

static uint16_t rnd16(void) { return (uint16_t) (rng_next() >> 48); }

static void run_scalar_types(const char * dir, uint64_t seed) {
    enum { N = 1024 };
    // F32: store edge and random bit patterns; the oracle is the identity.
    {
        rng_seed(seed, 0);
        uint32_t * in = calloc(N, 4);
        const uint32_t edge[] = {0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x00000001, 0x007fffff,
                                 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001, 0xffc12345};
        for (int i = 0; i < N; i++) in[i] = i < (int) (sizeof edge / 4) ? edge[i] : (uint32_t) (rng_next() >> 32);
        write_file(dir, "f32", "blocks", in, N * 4);
        write_file(dir, "f32", "f32", in, N * 4);
        printf("f32 id=0 blocks=%d\n", N);
        free(in);
    }
    // F16: every exponent pattern, signs, and mantissa edges, then random.
    {
        rng_seed(seed, 1);
        uint16_t * in = calloc(N, 2);
        float * out = calloc(N, 4);
        for (int i = 0; i < N; i++) {
            if (i < 64) in[i] = (uint16_t) ((i & 31) << 10 | ((i >> 5) ? 0x8000 : 0) | 0x155);
            else if (i < 128) in[i] = (uint16_t) ((i & 31) << 10 | ((i >> 5) & 1 ? 0x8000 : 0) | ((i % 2) ? 0x3ff : 0x001));
            else in[i] = rnd16();
        }
        in[0] = 0x0000; in[1] = 0x8000; in[2] = 0x7c00; in[3] = 0xfc00; in[4] = 0x7e00; in[5] = 0x7c01;
        in[6] = 0xfd55; in[7] = 0x0001; in[8] = 0x03ff; in[9] = 0x0400; in[10] = 0x7bff;
        for (int i = 0; i < N; i++) out[i] = GGML_FP16_TO_FP32(in[i]);
        write_file(dir, "f16", "blocks", in, N * 2);
        write_file(dir, "f16", "f32", out, N * 4);
        printf("f16 id=1 blocks=%d\n", N);
        free(in); free(out);
    }
    // BF16
    {
        rng_seed(seed, 30);
        uint16_t * in = calloc(N, 2);
        float * out = calloc(N, 4);
        for (int i = 0; i < N; i++) in[i] = rnd16();
        in[0] = 0x0000; in[1] = 0x8000; in[2] = 0x7f80; in[3] = 0xff80; in[4] = 0x7fc0; in[5] = 0x7f81;
        in[6] = 0xffc1; in[7] = 0x0001; in[8] = 0x007f; in[9] = 0x0080; in[10] = 0x7f7f;
        for (int i = 0; i < N; i++) { ggml_bf16_t h; h.bits = in[i]; out[i] = GGML_BF16_TO_FP32(h); }
        write_file(dir, "bf16", "blocks", in, N * 2);
        write_file(dir, "bf16", "f32", out, N * 4);
        printf("bf16 id=30 blocks=%d\n", N);
        free(in); free(out);
    }
}

// Extra K-quant fixtures: 64 blocks per type in which every 6-bit scale and
// minimum value (0..63) occurs in every one of the 8 slots. d = 1.0, dmin = 0.5.
static void pack_scales(uint8_t * q, const int * sc, const int * mn) {
    for (int j = 0; j < 4; j++) {
        q[j]     = (uint8_t) ((sc[j] & 63) | ((sc[j + 4] >> 4) << 6));
        q[j + 4] = (uint8_t) ((mn[j] & 63) | ((mn[j + 4] >> 4) << 6));
    }
    for (int j = 4; j < 8; j++) q[j + 4] = (uint8_t) ((sc[j] & 15) | ((mn[j] & 15) << 4));
}

static void run_kscales(const char * dir, uint64_t seed) {
    for (int i = 0; i < N_TYPES; i++) {
        const type_t * t = &TYPES[i];
        if (strcmp(t->name, "q4_k") && strcmp(t->name, "q5_k")) continue;
        enum { NB = 64 };
        size_t in_len = (size_t) NB * t->block_bytes, out_n = (size_t) NB * t->block_elems;
        uint8_t * in = calloc(in_len, 1);
        float * out = calloc(out_n, sizeof(float));
        rng_seed(seed, 100 + (uint64_t) t->id);
        for (int b = 0; b < NB; b++) {
            uint8_t * blk = in + (size_t) b * t->block_bytes;
            rng_fill(blk, t->block_bytes);
            int sc[8], mn[8];
            for (int j = 0; j < 8; j++) { sc[j] = (b + 8 * j) % 64; mn[j] = (5 * b + 8 * j + 3) % 64; }
            put16(blk, 0x3c00);
            put16(blk + 2, 0x3800);
            pack_scales(blk + 4, sc, mn);
        }
        t->fn(in, out, (int64_t) out_n);
        char name[64];
        snprintf(name, sizeof name, "%s_scales", t->name);
        write_file(dir, name, "blocks", in, in_len);
        write_file(dir, name, "f32", out, out_n * sizeof(float));
        printf("%s blocks=%d\n", name, NB);
        free(in); free(out);
    }
}

// Converts an existing block file: oracle --convert <name> <in.blocks> <out.f32>.
static int run_convert(const char * name, const char * in_path, const char * out_path) {
    FILE * f = fopen(in_path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", in_path); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t * in = malloc((size_t) n);
    if (fread(in, 1, (size_t) n, f) != (size_t) n) { fprintf(stderr, "read failed\n"); return 1; }
    fclose(f);
    size_t out_n; float * out;
    if (!strcmp(name, "f32") || !strcmp(name, "f16") || !strcmp(name, "bf16")) {
        size_t w = !strcmp(name, "f32") ? 4 : 2;
        out_n = (size_t) n / w; out = calloc(out_n, 4);
        for (size_t i = 0; i < out_n; i++) {
            if (w == 4) memcpy(&out[i], in + 4 * i, 4);
            else {
                uint16_t v = (uint16_t) (in[2 * i] | in[2 * i + 1] << 8);
                if (!strcmp(name, "f16")) out[i] = GGML_FP16_TO_FP32(v);
                else { ggml_bf16_t h; h.bits = v; out[i] = GGML_BF16_TO_FP32(h); }
            }
        }
    } else {
        const type_t * t = NULL;
        for (int i = 0; i < N_TYPES; i++) if (!strcmp(TYPES[i].name, name)) t = &TYPES[i];
        if (!t || n % t->block_bytes) { fprintf(stderr, "bad type or length\n"); return 1; }
        out_n = (size_t) (n / t->block_bytes) * t->block_elems; out = calloc(out_n, 4);
        t->fn(in, out, (int64_t) out_n);
    }
    FILE * g = fopen(out_path, "wb");
    if (!g || fwrite(out, 4, out_n, g) != out_n) { fprintf(stderr, "write failed\n"); return 1; }
    fclose(g);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 5 && !strcmp(argv[1], "--convert")) return run_convert(argv[2], argv[3], argv[4]);
    if (argc != 3) { fprintf(stderr, "usage: oracle <out-dir> <seed> | oracle --convert <name> <in> <out>\n"); return 2; }
    uint64_t seed = strtoull(argv[2], NULL, 0);
    for (int i = 0; i < N_TYPES; i++) run_block_type(argv[1], seed, &TYPES[i]);
    run_scalar_types(argv[1], seed);
    run_kscales(argv[1], seed);
    return 0;
}
