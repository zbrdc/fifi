//! v1 multi-GPU placement planner, frozen at model load.
//!
//! Policy lives here as data plus pure functions; mechanism (run
//! validation, exact load-plan byte accounting) stays in `shard` and the
//! loader. A future user-configurable policy arrives as another
//! [`PlacementPolicy`] without touching the execution path.
//!
//! Honesty rule: device properties (SM count, memory bandwidth, VRAM) are
//! hardware facts, not benchmark results. Every receipt states the
//! evidence class behind its choice; no estimate here is ever labeled a
//! measurement.

use crate::EngError;
use crate::family::gguf_model::{Inventory, Role};
use crate::family::qwen36::cuda::shard::{
    ATTN_Q_PER_KV_HEAD, ShardLoad, ShardRuns, ShardTable, VALUE_BLOCK, shard_loads,
};
use crate::family::qwen36::host::HostShape;
use crate::kernels::cuda::peer::DeviceFacts;
use crate::weights::GgufType;

const MIB: u64 = 1024 * 1024;
/// Mirrors `state_layout`'s per-region alignment.
const REGION_ALIGN: u64 = 256;

fn refuse(message: String) -> EngError {
    EngError::BadRequest(format!("multi-GPU placement: {message}"))
}

fn mib(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / MIB as f64)
}

fn align_up(value: u64) -> u64 {
    value.div_ceil(REGION_ALIGN) * REGION_ALIGN
}

/// What one card offers, as probed (or injected by tests). `free_bytes`
/// is what cudaMemGetInfo reports at plan time; `memory_bandwidth_bytes`
/// is bus-width/8 x 2 x memory-clock: a device property, not a measured
/// benchmark. `can_access` is this device's peer-access row over the
/// planned set; the diagonal is true by definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeviceProfile {
    pub(crate) device: usize,
    pub(crate) name: String,
    pub(crate) total_bytes: u64,
    pub(crate) free_bytes: u64,
    pub(crate) multiprocessors: u32,
    pub(crate) memory_bandwidth_bytes: u64,
    pub(crate) can_access: Vec<bool>,
}

impl DeviceProfile {
    /// One profile from one probed device's facts, in probe order. The
    /// bandwidth figure is the double-data-rate peak of the two raw facts
    /// (bus width in bits, memory clock in Hz): a device property, never a
    /// measurement. A device that reports no bus width or clock (nothing
    /// real does) yields zero bandwidth, which the planner treats as
    /// indistinguishable — the conservative even deal, never a refusal.
    pub(crate) fn from_facts(facts: &DeviceFacts) -> Self {
        let memory_bandwidth_bytes =
            u64::from(facts.bus_width_bits) / 8 * 2 * facts.memory_clock_hz;
        Self {
            device: facts.device,
            name: facts.name.clone(),
            total_bytes: facts.total_bytes,
            free_bytes: facts.free_bytes,
            multiprocessors: facts.multiprocessors,
            memory_bandwidth_bytes,
            can_access: facts.can_access.clone(),
        }
    }
}

/// Every profile of a probed set, in the set's order.
pub(crate) fn profiles_from_facts(facts: &[DeviceFacts]) -> Vec<DeviceProfile> {
    facts.iter().map(DeviceProfile::from_facts).collect()
}

#[cfg(test)]
mod facts_tests {
    use super::*;

    fn facts(device: usize) -> DeviceFacts {
        DeviceFacts {
            device,
            cc_major: 8,
            cc_minor: 6,
            can_access: vec![true, true],
            name: format!("probe {device}"),
            total_bytes: 24 * 1024 * 1024 * 1024,
            free_bytes: 20 * 1024 * 1024 * 1024,
            multiprocessors: 82,
            bus_width_bits: 384,
            memory_clock_hz: 9_500_000_000,
        }
    }

    #[test]
    fn a_profile_carries_the_probe_facts_and_derives_the_ddr_peak_bandwidth() {
        // 384 bit / 8 * 2 * 9.5 GHz = 912 GB/s: the RTX 3090's property peak.
        let profile = DeviceProfile::from_facts(&facts(0));
        assert_eq!(profile.name, "probe 0");
        assert_eq!(profile.free_bytes, 20 * 1024 * 1024 * 1024);
        assert_eq!(profile.memory_bandwidth_bytes, 912_000_000_000);
        assert_eq!(profile.can_access, vec![true, true]);
    }

    #[test]
    fn a_device_without_bus_facts_yields_zero_bandwidth_not_a_refusal() {
        let mut bare = facts(1);
        bare.bus_width_bits = 0;
        assert_eq!(DeviceProfile::from_facts(&bare).memory_bandwidth_bytes, 0);
    }

    #[test]
    fn profiles_keep_the_probe_order() {
        let profiles = profiles_from_facts(&[facts(0), facts(1)]);
        assert_eq!(
            profiles.iter().map(|p| p.device).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }
}

/// Planning knobs with defaults; the seam for future user configuration
/// of reserves and, later, measured performance tables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlacementPolicy {
    /// Held back on every card after all commitments.
    pub(crate) safety_reserve_bytes: u64,
    /// Per-context allowance for the CUDA context, plan arena, exchange
    /// mailboxes and gather buffers. An allowance, not a measurement.
    pub(crate) cuda_overhead_bytes: u64,
}

impl Default for PlacementPolicy {
    fn default() -> Self {
        Self {
            safety_reserve_bytes: 512 * MIB,
            cuda_overhead_bytes: 256 * MIB,
        }
    }
}

/// Everything the planner needs, borrowed. `quant` is the type the file
/// is planned as; the load-time check owns the file's real types.
pub(crate) struct PlacementRequest<'a> {
    pub(crate) shape: &'a HostShape,
    pub(crate) inventory: &'a Inventory,
    pub(crate) quant: GgufType,
    /// KV capacity positions; the Int8G32 cache is the only qualified KV
    /// layout, so its plane math is what v1 budgets.
    pub(crate) kv_capacity_rows: usize,
    /// Delta state slots (one per live sequence plus headroom).
    pub(crate) delta_slots: usize,
    pub(crate) profiles: &'a [DeviceProfile],
    pub(crate) policy: &'a PlacementPolicy,
}

/// One card's committed bytes under the chosen plan. `planned_bytes` is
/// everything except the safety reserve.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CardCommitment {
    pub(crate) device: usize,
    pub(crate) name: String,
    pub(crate) free_bytes: u64,
    pub(crate) sliced_weights_bytes: u64,
    pub(crate) replicated_weights_bytes: u64,
    pub(crate) embedding_bytes: u64,
    pub(crate) kv_cache_bytes: u64,
    pub(crate) delta_state_bytes: u64,
    pub(crate) cuda_overhead_bytes: u64,
    pub(crate) safety_reserve_bytes: u64,
    pub(crate) planned_bytes: u64,
    pub(crate) memory_bandwidth_bytes: u64,
    /// This card's fraction of the sliceable weight mass.
    pub(crate) share: Option<f64>,
    pub(crate) clamped: bool,
    pub(crate) peer_access: Vec<bool>,
}

/// Machine-readable record of why the plan is what it is.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PlacementReceipt {
    pub(crate) evidence: &'static str,
    pub(crate) model: &'static str,
    pub(crate) chosen: &'static str,
    pub(crate) reason: String,
    pub(crate) refusals: Vec<String>,
    /// Dealt dimensions whose legal grain (quantization blocks, the
    /// decode kernel's 8-head query groups) cannot express the chosen
    /// shares: geometry, not bandwidth, fixed their split.
    pub(crate) geometry_capped: Vec<String>,
    pub(crate) policy_safety_reserve_bytes: u64,
    pub(crate) policy_cuda_overhead_bytes: u64,
    pub(crate) cards: Vec<CardCommitment>,
}

impl PlacementReceipt {
    /// Minimal JSON writer; the engine has no serde dependency and the
    /// receipt is the only structured surface here.
    pub(crate) fn to_json(&self) -> String {
        let mut out = String::from("{");
        let field = |out: &mut String, key: &str, value: String, first: &mut bool| {
            if !*first {
                out.push(',');
            }
            *first = false;
            out.push_str(&json_str(key));
            out.push(':');
            out.push_str(&value);
        };
        let mut first = true;
        field(&mut out, "evidence", json_str(self.evidence), &mut first);
        field(&mut out, "model", json_str(self.model), &mut first);
        field(&mut out, "chosen", json_str(self.chosen), &mut first);
        field(&mut out, "reason", json_str(&self.reason), &mut first);
        field(
            &mut out,
            "refusals",
            format!(
                "[{}]",
                self.refusals
                    .iter()
                    .map(|r| json_str(r))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            &mut first,
        );
        field(
            &mut out,
            "geometry_capped",
            format!(
                "[{}]",
                self.geometry_capped
                    .iter()
                    .map(|d| json_str(d))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            &mut first,
        );
        field(
            &mut out,
            "policy",
            format!(
                "{{\"safety_reserve_bytes\":{},\"cuda_overhead_bytes\":{}}}",
                self.policy_safety_reserve_bytes, self.policy_cuda_overhead_bytes
            ),
            &mut first,
        );
        let cards = self
            .cards
            .iter()
            .map(|c| {
                let mut s = String::from("{");
                let mut f = true;
                field(&mut s, "device", c.device.to_string(), &mut f);
                field(&mut s, "name", json_str(&c.name), &mut f);
                field(&mut s, "free_bytes", c.free_bytes.to_string(), &mut f);
                field(
                    &mut s,
                    "sliced_weights_bytes",
                    c.sliced_weights_bytes.to_string(),
                    &mut f,
                );
                field(
                    &mut s,
                    "replicated_weights_bytes",
                    c.replicated_weights_bytes.to_string(),
                    &mut f,
                );
                field(
                    &mut s,
                    "embedding_bytes",
                    c.embedding_bytes.to_string(),
                    &mut f,
                );
                field(
                    &mut s,
                    "kv_cache_bytes",
                    c.kv_cache_bytes.to_string(),
                    &mut f,
                );
                field(
                    &mut s,
                    "delta_state_bytes",
                    c.delta_state_bytes.to_string(),
                    &mut f,
                );
                field(
                    &mut s,
                    "cuda_overhead_bytes",
                    c.cuda_overhead_bytes.to_string(),
                    &mut f,
                );
                field(
                    &mut s,
                    "safety_reserve_bytes",
                    c.safety_reserve_bytes.to_string(),
                    &mut f,
                );
                field(&mut s, "planned_bytes", c.planned_bytes.to_string(), &mut f);
                field(
                    &mut s,
                    "memory_bandwidth_bytes",
                    c.memory_bandwidth_bytes.to_string(),
                    &mut f,
                );
                field(
                    &mut s,
                    "share",
                    c.share.map(json_f64).unwrap_or_else(|| "null".to_owned()),
                    &mut f,
                );
                field(&mut s, "clamped", c.clamped.to_string(), &mut f);
                field(
                    &mut s,
                    "peer_access",
                    format!(
                        "[{}]",
                        c.peer_access
                            .iter()
                            .map(|b| b.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    ),
                    &mut f,
                );
                s.push('}');
                s
            })
            .collect::<Vec<_>>()
            .join(",");
        field(&mut out, "cards", format!("[{cards}]"), &mut first);
        out.push('}');
        out
    }
}

fn json_str(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_f64(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.4}")
    } else {
        "null".to_owned()
    }
}

/// The chosen runs, the table they validate into, and the receipt.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PlacementPlan {
    pub(crate) runs: ShardRuns,
    pub(crate) table: ShardTable,
    pub(crate) receipt: PlacementReceipt,
}

// ---- exact byte accounting ------------------------------------------------

/// One full-attention layer's KV stride, `[K | K scales | V | V scales]`,
/// mirroring `state_layout`'s KvPool math for the Int8G32 layout.
pub(super) fn kv_layer_stride_bytes(shape: &HostShape) -> Result<u64, EngError> {
    let attn = &shape.attn;
    if !attn.head_dim.is_multiple_of(32) {
        return Err(refuse(format!(
            "attention head_dim {} has no Int8G32 scale groups",
            attn.head_dim
        )));
    }
    let rows = attn.heads_kv as u64 * attn.head_dim as u64;
    let scales = rows / 32 * 2;
    Ok(align_up(rows + scales + rows + scales))
}

pub(super) fn kv_cache_bytes(
    shape: &HostShape,
    layers: usize,
    capacity: usize,
) -> Result<u64, EngError> {
    let stride = kv_layer_stride_bytes(shape)?;
    stride
        .checked_mul(layers as u64)
        .and_then(|v| v.checked_mul(capacity as u64))
        .ok_or_else(|| refuse("KV cache size overflows u64".to_owned()))
}

/// The DeltaNet recurrent+conv state pool, FULL-SIZE per card: that is
/// what ResidentState allocates today (rigs verify this), so that is what
/// v1 budgets. Mirrors `state_layout`'s DeltaPool math.
fn delta_state_bytes(shape: &HostShape, layers: usize, slots: usize) -> Result<u64, EngError> {
    let d = &shape.delta;
    let rec = (d.value_heads * d.key_dim * d.value_dim) as u64 * 4;
    let conv = ((d.conv_kernel - 1) * d.conv_dim()) as u64 * 4;
    let stride = align_up(rec) + align_up(conv);
    stride
        .checked_mul(layers as u64)
        .and_then(|v| v.checked_mul(slots as u64))
        .ok_or_else(|| refuse("delta state size overflows u64".to_owned()))
}

/// The type a tensor is planned at: the file quant where the role allows
/// it, else the role's first allowed type (norms and per-head vectors are
/// never block-quantized). The load-time check owns the file's real
/// types; this is the planning assumption the receipt stands behind.
pub(crate) fn plan_type(allowed: &[GgufType], quant: GgufType) -> GgufType {
    if allowed.contains(&quant) {
        quant
    } else {
        allowed[0]
    }
}

/// Exact weight bytes one card holds under `table`: what its load plan
/// covers, at each tensor's planning type. Returns (sliced, replicated,
/// embedding).
fn card_weight_bytes(
    table: &ShardTable,
    inventory: &Inventory,
    quant: GgufType,
    card: usize,
) -> Result<(u64, u64, u64), EngError> {
    let lengths: Vec<(String, u64, [u64; 2])> = inventory
        .tensors()
        .iter()
        .map(|tensor| {
            let bytes = tensor.byte_len(plan_type(tensor.allowed_types(), quant))?;
            let dims = [tensor.dims[0], tensor.dims.get(1).copied().unwrap_or(1)];
            Ok((tensor.name.clone(), bytes, dims))
        })
        .collect::<Result<_, EngError>>()?;
    let mut sliced = 0u64;
    let mut replicated = 0u64;
    let mut embedding = 0u64;
    for (name, load) in shard_loads(table, inventory, card)? {
        let (bytes, dims) = lengths
            .iter()
            .find(|(n, _, _)| n == &name)
            .map(|(_, b, d)| (*b, *d))
            .ok_or_else(|| refuse(format!("inventory lost tensor {name}")))?;
        match load {
            ShardLoad::Whole => replicated += bytes,
            ShardLoad::CardZeroOnly => embedding += u64::from(card == 0) * bytes,
            ShardLoad::Sliced(spec) => {
                // Only one axis is ever sliced; the other covers fully.
                let cols = (spec.cols.end - spec.cols.start) as u128;
                let rows = spec
                    .rows
                    .iter()
                    .map(|r| (r.end - r.start) as u128)
                    .sum::<u128>();
                let full = dims[0] as u128 * dims[1] as u128;
                sliced += (bytes as u128 * cols * rows / full) as u64;
            }
        }
    }
    Ok((sliced, replicated, embedding))
}

/// Layer counts straight from the inventory: the file is the truth.
fn layer_counts(inventory: &Inventory) -> (usize, usize) {
    let mut full = 0;
    let mut delta = 0;
    for tensor in inventory.tensors() {
        match tensor.role {
            Role::AttnK => full += 1,
            Role::SsmA => delta += 1,
            _ => {}
        }
    }
    (full, delta)
}

// ---- run construction -----------------------------------------------------

struct Dim {
    total: usize,
    block: usize,
    what: &'static str,
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// The native matvec serves dense rows in multiples of four
/// (`check_native_dense`), so a dealt dimension whose rows load as native
/// dense weights moves in blocks of at least four rows.
const NATIVE_ROWS_MULTIPLE: usize = 4;

/// The legal grain of each dealt dimension. `k_align` is the quantization
/// block along K: the column-sharded matrices (delta out, attention o,
/// expert and shared down) must cut on K multiples of it, which forces
/// heads to move in groups of `k_align / gcd(k_align, head width)` and
/// inter cuts onto `lcm(VALUE_BLOCK, k_align)` multiples. Dimensions whose
/// rows resolve through the native matvec (value heads and their beta /
/// alpha vectors, the vocabulary head shards) additionally move in
/// `NATIVE_ROWS_MULTIPLE` row groups.
fn dims(shape: &HostShape, k_align: usize) -> [Dim; 6] {
    let head_grain = |width: usize| {
        if k_align <= 1 {
            1
        } else {
            (k_align / gcd(k_align, width)).max(1)
        }
    };
    let col_grain = if k_align <= 1 {
        VALUE_BLOCK
    } else {
        VALUE_BLOCK / gcd(VALUE_BLOCK, k_align) * k_align
    };
    // The decode kernel serves 8 query heads per K/V head, so query
    // heads move in whole 8-groups on top of the quantization grain.
    let q_grain = head_grain(shape.attn.head_dim);
    let q_block = q_grain * ATTN_Q_PER_KV_HEAD / gcd(q_grain, ATTN_Q_PER_KV_HEAD);
    let native_grain =
        |grain: usize| grain * NATIVE_ROWS_MULTIPLE / gcd(grain, NATIVE_ROWS_MULTIPLE);
    // The GDN delta kernel pairs value head h with key head h % n_k_heads,
    // and the keys REPLICATE (every card holds all key channels), so a
    // card's value heads must be a multiple of the full key-head count.
    let v_head_block = {
        let g = native_grain(head_grain(shape.delta.value_dim));
        g * shape.delta.key_heads / gcd(g, shape.delta.key_heads)
    };
    [
        Dim {
            total: shape.delta.value_heads,
            block: v_head_block,
            what: "DeltaNet value heads",
        },
        Dim {
            total: shape.attn.heads_q,
            block: q_block,
            what: "attention query heads",
        },
        Dim {
            total: shape.attn.heads_kv * shape.attn.head_dim,
            block: VALUE_BLOCK,
            what: "K/V rows (following their query groups)",
        },
        Dim {
            total: shape.moe.inter,
            block: col_grain,
            what: "expert intermediate",
        },
        Dim {
            total: shape.moe.shared_inter,
            block: col_grain,
            what: "shared intermediate",
        },
        Dim {
            total: shape.vocab,
            block: NATIVE_ROWS_MULTIPLE,
            what: "vocabulary",
        },
    ]
}

fn runs_from_counts(counts: [&[usize]; 6]) -> ShardRuns {
    let mut starts = [0usize; 6];
    let mut runs = [
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];
    for (di, per_card) in counts.iter().enumerate() {
        for count in per_card.iter() {
            let end = starts[di] + count;
            runs[di].push(starts[di]..end);
            starts[di] = end;
        }
    }
    let [
        value_heads,
        query_heads,
        kv_rows,
        inter,
        shared_inter,
        vocab,
    ] = runs;
    ShardRuns {
        value_heads,
        query_heads,
        kv_rows,
        inter,
        shared_inter,
        vocab,
    }
}

/// A card's K/V row count follows its query-head groups: the card
/// holding a group stores, and decodes against, that group's K/V head.
fn kv_counts_from_q(q: &[usize], head_dim: usize) -> Vec<usize> {
    q.iter()
        .map(|heads| heads / ATTN_Q_PER_KV_HEAD * head_dim)
        .collect()
}

/// Largest-remainder deal of `blocks` over `shares` (which sum to ~1),
/// every card keeping at least one block. Lower card index wins ties.
fn deal_blocks(blocks: usize, shares: &[f64]) -> Option<Vec<usize>> {
    if blocks < shares.len() {
        return None;
    }
    let exact: Vec<f64> = shares.iter().map(|s| s * blocks as f64).collect();
    let mut counts: Vec<usize> = exact.iter().map(|x| x.floor() as usize).collect();
    let mut left = blocks - counts.iter().sum::<usize>();
    let mut order: Vec<usize> = (0..shares.len()).collect();
    order.sort_by(|&a, &b| {
        exact[b]
            .fract()
            .partial_cmp(&exact[a].fract())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut ring = 0;
    while left > 0 {
        counts[order[ring % order.len()]] += 1;
        left -= 1;
        ring += 1;
    }
    if counts.contains(&0) {
        // Guard: largest remainder can starve a small-share card even
        // when blocks >= cards; give every card one, then re-deal.
        counts = counts.into_iter().map(|c| c.max(1)).collect();
        let spent: usize = counts.iter().sum();
        if spent > blocks {
            return None;
        }
    }
    Some(counts)
}

// ---- candidates and scoring ------------------------------------------------

/// Shares proportional to bandwidth, clamped by capacity: over-budget
/// cards hold exactly what fits and the remainder re-deals over the rest.
fn waterfill(caps: &[u64], mass: u64, bw: &[u64]) -> Option<(Vec<f64>, Vec<bool>)> {
    let n = caps.len();
    let mut clamped = vec![false; n];
    let mut shares: Vec<f64> = {
        let total: u64 = bw.iter().sum();
        if total == 0 {
            return None;
        }
        bw.iter().map(|b| *b as f64 / total as f64).collect()
    };
    for _ in 0..n {
        let mut changed = false;
        for i in 0..n {
            if clamped[i] || shares[i] <= 0.0 {
                continue;
            }
            let budget = caps[i] as f64 / mass as f64;
            if shares[i] > budget {
                shares[i] = budget;
                clamped[i] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        let free_bw: f64 = bw
            .iter()
            .zip(clamped.iter())
            .filter(|(_, c)| !**c)
            .map(|(b, _)| *b as f64)
            .sum();
        if free_bw <= 0.0 {
            break;
        }
        let held: f64 = shares
            .iter()
            .zip(clamped.iter())
            .filter(|(_, c)| **c)
            .map(|(s, _)| *s)
            .sum();
        let free: f64 = shares
            .iter()
            .zip(clamped.iter())
            .filter(|(_, c)| !**c)
            .map(|(s, _)| *s)
            .sum();
        if free <= 0.0 {
            break;
        }
        // Rescale the unclamped shares so the total returns to 1.
        let scale = (1.0 - held) / free;
        for i in 0..n {
            if !clamped[i] {
                shares[i] *= scale;
            }
        }
    }
    let sum: f64 = shares.iter().sum();
    if sum > 1.0 + 1e-9 {
        return None;
    }
    Some((shares, clamped))
}

/// Decode makespan estimate: the slowest card's sliced mass over its
/// bandwidth. Device-properties evidence; never presented as a benchmark.
fn makespan(sliced: &[u64], bw: &[u64]) -> f64 {
    sliced
        .iter()
        .zip(bw)
        .map(|(s, b)| {
            if *b == 0 {
                f64::INFINITY
            } else {
                *s as f64 / *b as f64
            }
        })
        .fold(0.0, f64::max)
}

const EVIDENCE_DEVICE_PROPERTIES: &str = "device-properties";
const MODEL_TEXT: &str = "decode makespan = max_i(sliced_bytes_i / memory_bandwidth_i); \
                          device properties only, no measured benchmark";

struct Candidate {
    name: &'static str,
    shares: Vec<f64>,
    clamped: Vec<bool>,
}

/// One candidate through block rounding, exact accounting, and repair:
/// returns (table, runs, per-card sliced bytes) or a refusal string.
fn realize(
    req: &PlacementRequest,
    cand: &Candidate,
    caps: &[u64],
    dim_masses: &[u64; 6],
) -> Result<(ShardTable, ShardRuns, Vec<u64>), String> {
    let dims = dims(req.shape, req.quant.block_elems() as usize);
    let mut counts: Vec<Vec<usize>> = Vec::with_capacity(dims.len());
    for (di, dim) in dims.iter().enumerate() {
        // K/V rows are not dealt independently: the card holding a
        // query group stores, and decodes against, that group's K/V
        // head, so dim 2 always follows dim 1.
        if di == 2 {
            counts.push(kv_counts_from_q(&counts[1], req.shape.attn.head_dim));
            continue;
        }
        let blocks = dim.total / dim.block;
        if !dim.total.is_multiple_of(dim.block) {
            return Err(format!(
                "{} width {} is not a whole number of {}-value blocks",
                dim.what, dim.total, dim.block
            ));
        }
        let per_card = deal_blocks(blocks, &cand.shares).ok_or_else(|| {
            format!(
                "{} has {blocks} blocks for {} cards; a card would hold none",
                dim.what,
                cand.shares.len()
            )
        })?;
        counts.push(per_card.into_iter().map(|c| c * dim.block).collect());
    }
    // Repair loop: rebuild, measure exactly, move one block off any card
    // that exceeds its budget to the card with the most slack.
    for _ in 0..4096 {
        let refs: [&[usize]; 6] = [
            &counts[0], &counts[1], &counts[2], &counts[3], &counts[4], &counts[5],
        ];
        let runs = runs_from_counts(refs);
        let table = ShardTable::from_runs(req.shape, req.quant.block_elems() as usize, &runs)
            .map_err(|e| format!("runs rejected: {e:?}"))?;
        let sliced: Vec<u64> = (0..cand.shares.len())
            .map(|card| {
                card_weight_bytes(&table, req.inventory, req.quant, card).map(|(s, _, _)| s)
            })
            .collect::<Result<_, _>>()
            .map_err(|e| format!("accounting failed: {e:?}"))?;
        let over = (0..sliced.len())
            .filter(|&i| sliced[i] > caps[i])
            .max_by_key(|&i| sliced[i].saturating_sub(caps[i]));
        let Some(over) = over else {
            return Ok((table, runs, sliced));
        };
        let slack = |i: usize| caps[i].saturating_sub(sliced[i]);
        let recipient = (0..sliced.len())
            .filter(|&i| i != over)
            .max_by_key(|&i| slack(i));
        let Some(recipient) = recipient else {
            return Err(format!(
                "card {over} exceeds its budget by {} and no card has slack",
                mib(sliced[over] - caps[over])
            ));
        };
        // Shed from the fattest dim where the card has more than one grain.
        let mut moved = false;
        let order = {
            let mut idx: Vec<usize> = (0..6).collect();
            idx.sort_by(|&a, &b| dim_masses[b].cmp(&dim_masses[a]));
            idx
        };
        for &di in &order {
            if di == 2 {
                continue;
            }
            if counts[di][over] > dims[di].block {
                counts[di][over] -= dims[di].block;
                counts[di][recipient] += dims[di].block;
                if di == 1 {
                    counts[2] = kv_counts_from_q(&counts[1], req.shape.attn.head_dim);
                }
                moved = true;
                break;
            }
        }
        if !moved {
            return Err(format!(
                "card {over} exceeds its budget by {} and every dim is at one grain",
                mib(sliced[over] - caps[over])
            ));
        }
    }
    Err("repair loop did not settle in 4096 moves".to_owned())
}

/// Chooses the placement: bandwidth-waterfilled if the evidence
/// distinguishes it, otherwise today's conservative even deal.
pub(crate) fn plan_placement(req: &PlacementRequest) -> Result<PlacementPlan, EngError> {
    let profiles = req.profiles;
    let n = profiles.len();
    if n == 0 {
        return Err(refuse("no devices given".to_owned()));
    }
    if profiles.len() > 4 {
        return Err(refuse(format!(
            "{} devices exceed the four-card exchange maximum",
            profiles.len()
        )));
    }
    // Peer feasibility: the qualified exchange transport is the peer
    // ring; v1 refuses a set whose pairs lack mutual direct access
    // instead of silently falling back to an unqualified transport.
    for a in 0..n {
        if profiles[a].can_access.len() != n {
            return Err(refuse(format!(
                "device {} has a peer row of {} entries for {n} devices",
                profiles[a].device,
                profiles[a].can_access.len()
            )));
        }
        for b in 0..n {
            if a != b && !profiles[a].can_access[b] {
                return Err(refuse(format!(
                    "devices {} and {} have no direct peer access; the qualified exchange needs it",
                    profiles[a].device, profiles[b].device
                )));
            }
        }
    }
    let (full_layers, delta_layers) = layer_counts(req.inventory);
    if full_layers + delta_layers == 0 {
        return Err(refuse("the inventory holds no layers".to_owned()));
    }
    let kv = kv_cache_bytes(req.shape, full_layers, req.kv_capacity_rows)?;
    let delta = delta_state_bytes(req.shape, delta_layers, req.delta_slots)?;

    // Per-card budgets for sliced weights: free minus every fixed cost,
    // the reserve, and card 0's embedding. The even table is only the
    // accounting reference here.
    let even_table = ShardTable::new(req.shape, n, req.quant.block_elems() as usize)?;
    let per_card: Vec<(u64, u64, u64)> = (0..n)
        .map(|card| card_weight_bytes(&even_table, req.inventory, req.quant, card))
        .collect::<Result<_, EngError>>()?;
    let caps: Vec<u64> = (0..n)
        .map(|i| {
            let (_, replicated, embed) = per_card[i];
            profiles[i].free_bytes.saturating_sub(
                replicated
                    + kv
                    + delta
                    + req.policy.cuda_overhead_bytes
                    + req.policy.safety_reserve_bytes
                    + if i == 0 { embed } else { 0 },
            )
        })
        .collect();
    for i in 0..n {
        if caps[i] == 0 {
            return Err(refuse(format!(
                "device {} ({}) has no room for sliced weights: free {} against replicated {} + KV {} + delta {} + overhead {} + reserve {}{}",
                profiles[i].device,
                profiles[i].name,
                mib(profiles[i].free_bytes),
                mib(per_card[i].1),
                mib(kv),
                mib(delta),
                mib(req.policy.cuda_overhead_bytes),
                mib(req.policy.safety_reserve_bytes),
                if i == 0 {
                    " + embedding".to_owned()
                } else {
                    String::new()
                },
            )));
        }
    }

    // Sliceable mass: the even table's per-card slices sum to the total.
    let mass: u64 = per_card.iter().map(|(s, _, _)| *s).sum();
    // Global per-dim masses, only to order repair moves.
    let mut dim_masses = [0u64; 6];
    for tensor in req.inventory.tensors() {
        let bytes = tensor
            .byte_len(plan_type(tensor.allowed_types(), req.quant))
            .unwrap_or(0);
        let di = match tensor.role {
            Role::Output => 5,
            Role::AttnQ | Role::AttnOutput => 1,
            Role::AttnK | Role::AttnV => 2,
            Role::FfnGateExps | Role::FfnUpExps | Role::FfnDownExps => 3,
            Role::FfnGateShexp | Role::FfnUpShexp | Role::FfnDownShexp => 4,
            _ => 0,
        };
        dim_masses[di] += bytes;
    }

    let bw: Vec<u64> = profiles.iter().map(|p| p.memory_bandwidth_bytes).collect();
    let mut refusals: Vec<String> = Vec::new();
    let mut chosen: Option<(Candidate, ShardTable, ShardRuns, Vec<u64>, String)> = None;

    if n == 1 {
        let cand = Candidate {
            name: "single-card",
            shares: vec![1.0],
            clamped: vec![false],
        };
        let (table, runs, sliced) = realize(req, &cand, &caps, &dim_masses).map_err(refuse)?;
        let reason = "single card: whole ranges, no placement decision to make".to_owned();
        return Ok(finish(
            req, cand, table, runs, sliced, kv, delta, reason, refusals,
        ));
    }

    let indistinguishable = bw.iter().all(|b| *b == bw[0]);
    if !indistinguishable {
        if let Some((shares, clamped)) = waterfill(&caps, mass, &bw) {
            let cand = Candidate {
                name: "bandwidth-waterfill",
                shares,
                clamped,
            };
            match realize(req, &cand, &caps, &dim_masses) {
                Ok((table, runs, sliced)) => {
                    let even = Candidate {
                        name: "conservative-even",
                        shares: vec![1.0 / n as f64; n],
                        clamped: vec![false; n],
                    };
                    let even_ok = realize(req, &even, &caps, &dim_masses).ok();
                    let a = makespan(&sliced, &bw);
                    let b = even_ok
                        .as_ref()
                        .map(|(_, _, s)| makespan(s, &bw))
                        .unwrap_or(f64::INFINITY);
                    if a < b * (1.0 - 1e-9) {
                        let reason = format!(
                            "bandwidth-waterfill wins the makespan model ({:.3} vs {:.3} relative); \
                             shares follow memory bandwidth, clamped by per-card budgets",
                            1.0,
                            b / a
                        );
                        chosen = Some((cand, table, runs, sliced, reason));
                    } else if let Some((t2, r2, s2)) = even_ok {
                        let reason = format!(
                            "conservative even deal chosen: the makespan model did not separate the \
                             candidates ({:.3} vs {:.3} relative); ties go to the conservative plan",
                            1.0,
                            b / a
                        );
                        refusals.push(format!(
                            "bandwidth-waterfill feasible but not separated ({} vs {} relative)",
                            json_f64(1.0),
                            json_f64(b / a)
                        ));
                        chosen = Some((even, t2, r2, s2, reason));
                    } else {
                        refusals.push(
                            "conservative-even deal does not fit; keeping bandwidth-waterfill"
                                .to_owned(),
                        );
                        let reason =
                            "bandwidth-waterfill: the even deal does not fit these capacities"
                                .to_owned();
                        chosen = Some((cand, table, runs, sliced, reason));
                    }
                }
                Err(why) => refusals.push(format!("bandwidth-waterfill refused: {why}")),
            }
        } else {
            refusals.push(
                "bandwidth-waterfill infeasible: the sliceable mass cannot fit the budgets"
                    .to_owned(),
            );
        }
    }

    if chosen.is_none() {
        let cand = Candidate {
            name: "conservative-even",
            shares: vec![1.0 / n as f64; n],
            clamped: vec![false; n],
        };
        match realize(req, &cand, &caps, &dim_masses) {
            Ok((table, runs, sliced)) => {
                let reason = if indistinguishable {
                    "conservative even deal chosen: the devices report equal memory bandwidth, so \
                     the evidence cannot distinguish a better split"
                        .to_owned()
                } else {
                    "conservative even deal chosen: no faster candidate was feasible".to_owned()
                };
                chosen = Some((cand, table, runs, sliced, reason));
            }
            Err(why) => refusals.push(format!("conservative-even refused: {why}")),
        }
    }

    let Some((cand, table, runs, sliced, reason)) = chosen else {
        return Err(refuse(format!(
            "no feasible placement over {:?}: {}",
            profiles.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            refusals.join(" | ")
        )));
    };
    Ok(finish(
        req, cand, table, runs, sliced, kv, delta, reason, refusals,
    ))
}

#[allow(clippy::too_many_arguments)]
fn finish(
    req: &PlacementRequest,
    cand: Candidate,
    table: ShardTable,
    runs: ShardRuns,
    sliced: Vec<u64>,
    kv: u64,
    delta: u64,
    reason: String,
    refusals: Vec<String>,
) -> PlacementPlan {
    let total_sliced: u64 = sliced.iter().sum();
    let cards: Vec<CardCommitment> = (0..req.profiles.len())
        .map(|i| {
            let (s, r, e) =
                card_weight_bytes(&table, req.inventory, req.quant, i).unwrap_or((0, 0, 0));
            let planned = s + r + e + kv + delta + req.policy.cuda_overhead_bytes;
            // A card is capacity-clamped when waterfill clamped it OR its
            // realized share came in below the candidate's target: repair
            // moves land at the same budget-bound endpoint either way.
            let realized = if total_sliced > 0 {
                sliced[i] as f64 / total_sliced as f64
            } else {
                0.0
            };
            let clamped = cand.clamped.get(i).copied().unwrap_or(false)
                || (cand.shares.len() > 1 && realized < cand.shares[i] - 1e-9);
            CardCommitment {
                device: req.profiles[i].device,
                name: req.profiles[i].name.clone(),
                free_bytes: req.profiles[i].free_bytes,
                sliced_weights_bytes: s,
                replicated_weights_bytes: r,
                embedding_bytes: e,
                kv_cache_bytes: kv,
                delta_state_bytes: delta,
                cuda_overhead_bytes: req.policy.cuda_overhead_bytes,
                safety_reserve_bytes: req.policy.safety_reserve_bytes,
                planned_bytes: planned,
                memory_bandwidth_bytes: req.profiles[i].memory_bandwidth_bytes,
                share: if cand.shares.len() > 1 {
                    Some(sliced[i] as f64 / sliced.iter().sum::<u64>() as f64)
                } else {
                    None
                },
                clamped,
                peer_access: req.profiles[i].can_access.clone(),
            }
        })
        .collect();
    // Dims whose realized split deviates from the candidate's target
    // beyond rounding: the legal grain (quantization blocks, the decode
    // kernel's whole 8-head query groups) — not bandwidth — fixed the
    // split. Surfaced so a receipt never presents a bandwidth-proportional
    // deal where the execution geometry forced another one. Five points
    // of share is far beyond any block rounding the fine-grained dims can
    // produce, and well inside the 0.18 a forced two-group 50/50 sits from
    // a 0.68 bandwidth target.
    let dims = dims(req.shape, req.quant.block_elems() as usize);
    let run_lens: [Vec<usize>; 6] = [
        runs.value_heads.iter().map(|r| r.len()).collect(),
        runs.query_heads.iter().map(|r| r.len()).collect(),
        runs.kv_rows.iter().map(|r| r.len()).collect(),
        runs.inter.iter().map(|r| r.len()).collect(),
        runs.shared_inter.iter().map(|r| r.len()).collect(),
        runs.vocab.iter().map(|r| r.len()).collect(),
    ];
    let mut geometry_capped: Vec<String> = Vec::new();
    if cand.shares.len() > 1 {
        for (di, dim) in dims.iter().enumerate() {
            if di == 2 || dim.total == 0 {
                continue; // K/V rows follow their query groups
            }
            let total: usize = run_lens[di].iter().sum();
            if total == 0 {
                continue;
            }
            let capped = (0..cand.shares.len()).any(|i| {
                let realized = run_lens[di][i] as f64 / total as f64;
                (realized - cand.shares[i]).abs() > 0.05
            });
            if capped {
                geometry_capped.push(dim.what.to_owned());
            }
        }
    }
    let reason = if geometry_capped.is_empty() {
        reason
    } else {
        format!(
            "{reason}; geometry-capped: {} (the legal grain cannot express the \
             bandwidth split)",
            geometry_capped.join(", ")
        )
    };
    let receipt = PlacementReceipt {
        evidence: if cand.name == "conservative-even" || cand.name == "single-card" {
            "conservative"
        } else {
            EVIDENCE_DEVICE_PROPERTIES
        },
        model: MODEL_TEXT,
        chosen: cand.name,
        reason,
        refusals,
        geometry_capped,
        policy_safety_reserve_bytes: req.policy.safety_reserve_bytes,
        policy_cuda_overhead_bytes: req.policy.cuda_overhead_bytes,
        cards,
    };
    PlacementPlan {
        runs,
        table,
        receipt,
    }
}
