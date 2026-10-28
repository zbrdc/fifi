//! Pure sizing of every buffer the vendored MMQ launches read or write (CPF-V4 G1, decision
//! 0049 item 11): the q8_1 scratch (M1), the ids ranges (M2), the grouped column rule (M3),
//! the fixup range, the J replica, and the under-pad windows of the wrong upstream helper.
//! Host-only, pure, checked throughout: an overflowing product is a `BadRequest` (the
//! pattern of `prefill::attn_prefill`). Terminology, binding
//! on G2-G4: `K` is the input width in values, `Kpad` rounds it to 128, `blocks_per_col =
//! Kpad/128` (one 144-B `block_q8_1_mmq` per 128 values), and the buffer holds
//! `blocks_per_col * ncols` blocks k-block-major (`ib = k_block*ne1 + column`,
//! quantize.cu:530-533): `ncols = M` dense, `ncols = ne_get_rows = M*top_k` grouped.
//!
//! The J/config tables are MIT-derived constants transcribed from the vendored tree at
//! llama.cpp pin `acecd56032ddc34bada14a2d978f110d9c987095` (mmq-config-ampere.cuh,
//! mmq-config-pascal-dp4a.cuh; engine/vendor/ggml-cuda-mmq/LICENSE holds the notice,
//! THIRD_PARTY_NOTICES.md names this file). No other code is copied, and the cross-check
//! test re-parses both headers, so a vendor patch that changes a CASE row fails CI without
//! a device. Not padded, with reasons (rule M2): x-tile loads
//! (only weight ROWS are clamped and only with fallback on, mmq-load-tiles.cuh:498-499,
//! :736-737; the k extent never is — the K-grid precondition the entries own, not a rule
//! here); write-back stores (bounded by `j_max`/`i_max`); the
//! fixup range (write-only until the fixup kernel reads it back). NVFP4 plans separately
//! pad `y_scale` with J_MAX-1 factors because stream-k writes full temporary tiles.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "staged for CPF-V4's glue entries and CPF-1 (decision 0049)"
    )
)]

use crate::EngError;

/// Values per `block_q8_1_mmq` (`QK8_1_MMQ`, mmq.cuh:25) and its bytes (:28-47, :57-58).
const QK8_1_MMQ: usize = 128;
pub(crate) const Q8_1_BLOCK_BYTES: usize = 144;
/// The PAD's rounding granularity, a deliberate over-bound (KUBDAN-MSG-85 B1, rev 3): the
/// y-tile copies' own loop steps by `nwarps * warp_size = 256` ints, so a copy spans
/// `round_up(144 * J, 1024)` bytes (mmq.cuh:859-865) — the pad rounds `128 * 144` up to
/// multiples of 2048 instead, the plan's one-line safety margin over the exact need.
pub(crate) const SRC1_LOAD_CHUNK: usize = 2048;
/// Rule M1 (rev 3, MSG-85 B1): the one constant pad on every q8_1 buffer, dense and
/// grouped, `round_up(128 * 144, 2048)`. The two unconditional y-tile copies read
/// `round_up(144 * J, 1024)` bytes from the first column of a J-column tile with no column
/// bound (mmq.cuh:859-865, :875-881), so the worst need over every reachable J is
/// `S(128) - 144 = 18,288 B`; a grouped top expert with one column reaches it whatever J
/// the launch picked, so no `ncols_opt` function is safe (worst-row margin 144 B).
pub(crate) const Q8_1_PAD: usize = 128 * Q8_1_BLOCK_BYTES;
/// The chunk-column domain of every sizing function (0049 item 9 caps chunks at 2,048
/// rows); wrappers refuse more through [`check_ncols_domain`].
pub(crate) const NCOLS_LIMIT: usize = 2048;
/// The architectures the engine compiles (V-4, mmq.cuh:221-222): both at least Volta.
const COMPILED_ARCHS: [i32; 2] = [720, 860];
const CC_VOLTA: i32 = 700;
const CC_TURING: i32 = 750;
pub(super) const J_MAX: usize = 128;
/// `MMQ_TILE_NE_K` and the quantization widths of the sram strides (mmq.cuh:116, :126-140).
const TILE_K: usize = 32;
const QI8_0: usize = 8;
const QI6_K: usize = 32;

/// A `BadRequest` that names the sizing rule that fired (the `mmq_size` tag).
fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("mmq_size: {tag}: {detail}"))
}

fn pad_up(n: usize, m: usize) -> usize {
    n.div_ceil(m) * m
}

/// The MMQ weight type tags: per-tensor dispatch, never role.
/// The widest Q4_K tile the vendored table implements as a tiling config (patch V-5). Above
/// this width the Q4_K rows are stream-k configs.
pub(crate) const Q4K_TILING_MAX_J: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MmqType {
    Q80,
    Q50,
    Q4K,
    Q6K,
    Nvfp4,
}

/// The q8_1 scale layout a weight of an [`MmqType`] reads (mmq.cuh:61-101): the D4 scale
/// form for Q8_0, Q5_0, Q6_K and NVFP4, the DS4 form for Q4_K. One q8_1 buffer is quantized once
/// and read by every GEMM of its group, so the layout is a property of the weight type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Q8_1Layout {
    D4,
    Ds4,
}

impl MmqType {
    /// The scale layout of the q8_1 activations this weight type reads.
    pub(crate) fn q8_1_layout(self) -> Q8_1Layout {
        match self {
            Self::Q80 | Self::Q50 | Self::Q6K | Self::Nvfp4 => Q8_1Layout::D4,
            Self::Q4K => Q8_1Layout::Ds4,
        }
    }
}

/// The two vendored config tables; `cc` selects one the way `ggml_cuda_mmq_get_config`
/// does (mmq.cuh:223-228): the highest COMPILED arch at or below `cc` (common.cuh:159-169)
/// at least Volta takes ampere, anything below falls through to pascal-dp4a.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MmqConfig {
    Ampere,
    PascalDp4a,
}

/// The per-table fields the sizing reads; the cross-check test verifies every CASE row of
/// our served types against them, plus `occupancy`/`K_vram`/`stream_k` (test-only literals).
struct MmqTableConfig {
    nthreads: usize,
    /// The launch's `I` tile width.
    i_tile: usize,
    /// `[fallback = false, fallback = true]`.
    j_sets: [&'static [usize]; 2],
}

const AMPERE_J_FALSE: &[usize] = &[8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128];
const AMPERE_J_TRUE: &[usize] = &[8, 16, 32, 64, 128];
const PASCAL_J_FALSE: &[usize] = &[8, 16, 24, 32, 40, 48, 64];
const PASCAL_J_TRUE: &[usize] = &[8, 16, 32, 64];

impl MmqConfig {
    /// `ggml_cuda_highest_compiled_arch` (common.cuh:159-169) over [`COMPILED_ARCHS`]: the
    /// highest compiled arch at or below `cc`, else -1 (no cubin runs on it).
    fn highest_compiled_arch(cc: i32) -> i32 {
        COMPILED_ARCHS
            .into_iter()
            .filter(|&a| a <= cc)
            .max()
            .unwrap_or(-1)
    }

    pub(crate) fn select(cc: i32) -> Self {
        match Self::highest_compiled_arch(cc) >= CC_VOLTA {
            true => Self::Ampere,
            false => Self::PascalDp4a,
        }
    }

    fn config(self) -> MmqTableConfig {
        match self {
            Self::Ampere => MmqTableConfig {
                nthreads: 256,
                i_tile: 128,
                j_sets: [AMPERE_J_FALSE, AMPERE_J_TRUE],
            },
            Self::PascalDp4a => MmqTableConfig {
                nthreads: 256,
                i_tile: 64,
                j_sets: [PASCAL_J_FALSE, PASCAL_J_TRUE],
            },
        }
    }

    /// `ggml_cuda_mmq_get_sram_stride` (mmq.cuh:126-140) for the layout our types use: the
    /// Q8_0 layout for Q8_0/Q5_0, Q8_1 for Q4_K, Q6_K for Q6_K, NVFP4 for NVFP4.
    fn sram_stride(self, mmq_type: MmqType) -> usize {
        match mmq_type {
            MmqType::Q80 | MmqType::Q50 => 2 * TILE_K + 2 * TILE_K / QI8_0 + 4,
            MmqType::Q4K => 2 * TILE_K + 2 * TILE_K / 8 + 4,
            MmqType::Q6K => 2 * TILE_K + TILE_K / QI6_K + TILE_K / 8 + 7,
            MmqType::Nvfp4 => 2 * TILE_K + TILE_K / 2 + 4,
        }
    }

    /// Table membership, the CASE chain's fall-through (`config.type == GGML_TYPE_COUNT`).
    fn contains(self, fallback: bool, j: usize) -> bool {
        self.config().j_sets[usize::from(fallback)].contains(&j)
    }

    /// Whether the table's row for this (type, J) runs stream-k. Patch V-5 turns the Q4_K rows for
    /// J at or below [`Q4K_TILING_MAX_J`] into tiling configs, which need no fixup; every other
    /// Ampere row is stream-k, and the pascal table has no stream-k rows.
    pub(crate) fn stream_k(self, mmq_type: MmqType, j: usize) -> bool {
        self == Self::Ampere && !(mmq_type == MmqType::Q4K && j <= Q4K_TILING_MAX_J)
    }

    /// The x-tile term of `mmq_get_nbytes_shared` (mmq.cuh:1342-1346 via :375-381): the MMA
    /// layout stride on Turing and newer, the dp4a tile sizes below (mmq.cuh:330-352).
    fn nbytes_shared_x(self, cc: i32, mmq_type: MmqType) -> usize {
        let i = self.config().i_tile;
        if Self::highest_compiled_arch(cc) >= CC_TURING {
            return i * self.sram_stride(mmq_type) * 4;
        }
        // [qs, dm, sc] of MMQ_DP4A_TXS_* (mmq.cuh:330-352) at tile width i, TILE_K = 32.
        let dims = match mmq_type {
            MmqType::Q80 | MmqType::Q50 => [i * 64 + i, i * 8 + i / 4, 0],
            MmqType::Q4K => [i * 32 + i, i, i * 4 + i / 8],
            MmqType::Q6K => [i * 64 + i, i + i / 32, i * 4 + i / 8],
            MmqType::Nvfp4 => [i * 64 + i, i * 16 + i / 2, 0],
        };
        (dims[0] + dims[1] + dims[2]) * 4
    }

    /// `mmq_get_nbytes_shared` (mmq.cuh:1342-1346); `None` off the table.
    fn nbytes_shared(self, mmq_type: MmqType, fallback: bool, cc: i32, j: usize) -> Option<usize> {
        let cfg = self.config();
        if !self.contains(fallback, j) {
            return None;
        }
        let nbs_y = pad_up(j * Q8_1_BLOCK_BYTES, cfg.nthreads * 4);
        Some(j * 4 + self.nbytes_shared_x(cc, mmq_type) + nbs_y)
    }
}

/// Bytes the y-tile copies read from the first column of a J-column tile (mmq.cuh:859).
pub(crate) fn y_tile_span_bytes(nthreads: usize, j: usize) -> usize {
    pad_up(j * Q8_1_BLOCK_BYTES, nthreads * 4)
}

/// The tail r of `n` columns under tile width J (rev 3): `n - J*floor((n-1)/J)`, so
/// `r` is in `1..=J` and equals J when `n % J == 0`.
pub(crate) fn tail_r(n: usize, j: usize) -> usize {
    n - j * ((n - 1) / j)
}

/// The J the launch picks (mmq.cuh:1444-1461), modelled exactly: table membership, the
/// `smpbo` skip, the strict-`<` update, the stop at one tile. `None` is the launch's
/// `cudaErrorInvalidValue` ("no tile width fits"). A predictor for tests and refusal rows,
/// never a safety input: the capacity rules are safe for any table J.
pub(crate) fn j_best(
    mmq_type: MmqType,
    fallback: bool,
    cc: i32,
    smpbo: usize,
    ncols_opt: usize,
) -> Option<usize> {
    let table = MmqConfig::select(cc);
    let mut best = 0;
    let mut ntiles_best = usize::MAX;
    let mut j = 8;
    while j <= J_MAX && ntiles_best > 1 {
        let ntiles = ncols_opt.div_ceil(j);
        if let Some(nbytes) = table.nbytes_shared(mmq_type, fallback, cc, j)
            && nbytes <= smpbo
            && ntiles < ntiles_best
        {
            best = j;
            ntiles_best = ntiles;
        }
        j += 8;
    }
    (best != 0).then_some(best)
}

/// The wrong upstream helper, kept only to classify its under-pad windows
/// (`ggml_cuda_mmq_get_J_max`, mmq.cuh:320-329): the largest multiple of 8 at or below
/// `min(ncols, 512)` in the table.
pub(crate) fn legacy_j_max(fallback: bool, cc: i32, ncols: usize) -> usize {
    let table = MmqConfig::select(cc);
    let mut ret = ncols.min(512);
    ret -= ret % 8;
    while ret > 0 && !table.contains(fallback, ret) {
        ret -= 8;
    }
    ret
}

/// The `ncols_opt` where a q8_1 buffer sized by the old rule (`pad = legacy_j_max * 144`,
/// the pin's mmq.cu:136-137 and :206) under-pads the launch's need (`y_tile_span - r*144`),
/// recomputed over the whole domain 1..=2048 (finding 9), never taken from a sweep.
pub(crate) fn overread_window(
    mmq_type: MmqType,
    fallback: bool,
    cc: i32,
    smpbo: usize,
) -> Vec<usize> {
    let nthreads = MmqConfig::select(cc).config().nthreads;
    (1..=NCOLS_LIMIT)
        .filter(|&n| match j_best(mmq_type, fallback, cc, smpbo, n) {
            Some(j) => {
                y_tile_span_bytes(nthreads, j)
                    > (legacy_j_max(fallback, cc, n) + tail_r(n, j)) * Q8_1_BLOCK_BYTES
            }
            None => false,
        })
        .collect()
}

/// Rev 2's upper bound J (`min(128, 8*ceil(n/8))`), kept for the coupling row and its
/// mutant: it is NOT the launch's J in the table gaps.
pub(crate) fn j_ub(n: usize) -> usize {
    J_MAX.min(8 * n.div_ceil(8))
}

/// Rule M1 (rev 3, MSG-85 B1): `blocks_per_col * ncols * 144 + Q8_1_PAD`, the one constant
/// pad on every q8_1 buffer, dense (`ncols = M`) and grouped (`ncols = ne_get_rows`).
pub(crate) fn q8_1_bytes(k: usize, ncols: usize) -> Result<usize, EngError> {
    if k == 0 || ncols == 0 {
        return Err(rule("zero extent", "K and ncols must be at least 1"));
    }
    k.div_ceil(QK8_1_MMQ)
        .checked_mul(ncols)
        .and_then(|blocks| blocks.checked_mul(Q8_1_BLOCK_BYTES))
        .and_then(|bytes| bytes.checked_add(Q8_1_PAD))
        .ok_or_else(|| rule("size overflow", "the q8_1 scratch size overflows"))
}

/// Rule M2 (rev 3, MSG-85 B2), the shipped closed form: `ids_dst` holds at least
/// `ne_get_rows + ncols_max + 127` ints (`9M + 127` at Qwen's top_k 8, `ncols_max = M`),
/// because the stream-k fixup gather (mmq.cuh:1295-1302) has no `jt*J >= col_diff` exit.
pub(crate) fn ids_dst_ints(ne_get_rows: usize, ncols_max: usize) -> Result<usize, EngError> {
    ne_get_rows
        .checked_add(ncols_max)
        .and_then(|ints| ints.checked_add(127))
        .ok_or_else(|| rule("size overflow", "the ids_dst capacity overflows"))
}

/// The exact bound of an ids gather that may read `base + j - 1`: the main kernels stop at
/// `col_high + J - 2 <= ne_get_rows + J - 2` past the `jt*J >= col_diff` exits (:972,
/// :1060), while the fixup gather reaches `ne_get_rows + ncols_max + J - 2` (mmq.cuh:1295-
/// 1302, no exit). Exported for the arena's byte accounting and the refusal tests.
pub(crate) fn ids_gather_ints(base: usize, j: usize) -> Result<usize, EngError> {
    let sum = base
        .checked_add(j)
        .ok_or_else(|| rule("size overflow", "the ids gather bound overflows"))?;
    sum.checked_sub(1).ok_or_else(|| {
        rule(
            "zero extent",
            "ids_gather_ints(0, 0) underflows: the gather base and J are both 0",
        )
    })
}

/// `ids_src1` is written only at valid compact/token indices (mmid.cu:106-117).
pub(crate) fn ids_src1_ints(ne_get_rows: usize) -> usize {
    ne_get_rows
}

/// `expert_bounds` capacity: the `zt + 1` read of every grouped path (mmq.cuh:962-963,
/// :1296-1297) needs one int past the last expert.
pub(crate) fn expert_bounds_ints(n_experts: usize) -> Result<usize, EngError> {
    n_experts
        .checked_add(1)
        .ok_or_else(|| rule("size overflow", "the expert_bounds capacity overflows"))
}

/// The fixup rule: the documented maximum, `nsm * J * I` floats with the ampere table's
/// `J = I = 128` (mmq.cuh:1334-1338), sized once per arena from the runtime device facts.
pub(crate) fn fixup_floats_max(nsm: usize) -> Result<usize, EngError> {
    nsm.checked_mul(J_MAX)
        .and_then(|floats| floats.checked_mul(J_MAX))
        .ok_or_else(|| rule("size overflow", "the fixup capacity overflows"))
}

/// The exact need of one launch, `blocks_stream_k * J * I` floats (mmq.cuh:1408), for the
/// refusal test and a later arena tightening.
pub(crate) fn fixup_floats_exact(
    blocks: usize,
    j: usize,
    i_tile: usize,
) -> Result<usize, EngError> {
    blocks
        .checked_mul(j)
        .and_then(|floats| floats.checked_mul(i_tile))
        .ok_or_else(|| rule("size overflow", "the exact fixup bound overflows"))
}

/// The ncols domain guard: every rule above holds for `ncols` in `1..=2048`, and a caller
/// that raises the chunk cap must extend this domain first (finding 22).
pub(crate) fn check_ncols_domain(ncols: usize) -> Result<(), EngError> {
    if ncols == 0 || ncols > NCOLS_LIMIT {
        return Err(rule(
            "chunk cap",
            &format!("ncols {ncols} is outside 1..={NCOLS_LIMIT}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
