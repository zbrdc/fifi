//! The MoE block of ONE layer of the EXL3 step at ONE token (`m = 1`), and the block of a K-row
//! verify window ([`run_exl3_moe_rows`]), which is the same block run once per row where the rows
//! cannot share a launch.
//!
//! The router writes the ids, the pick weights and the shared gate on the device, as in the GGUF
//! block. The block has two ways to turn the ids into launches. Built without graph capture it
//! reads the eight ids back, which waits for the stream once per layer, and queues grouped
//! launches over the picked members' ranges. Built with graph capture it queues id-indexed
//! launches that read `ids[slot]` on the device when the kernels run, so it makes no host read and
//! can be recorded in a graph. Both queue the same groups: the gate and up projections of every pick (and of the shared expert) in one group,
//! the SwiGLU, the down projections in one group, and the combine. Every projection reads the F32
//! row and keeps F32 intermediates; the sticky arithmetic flag is read by the step once per token.
//!
//! The order of the pairs of the gate and up group is the layout of `ExpertGu`: pick `s` owns
//! the pairs `2 s` (gate) and `2 s + 1` (up), so each pick's `[gate | up]` row is contiguous for
//! the SwiGLU. The grouped launch requires the rotated inputs and the partials of its pairs
//! contiguous in group order (`Scratch` places them); the out extents may be scattered rows.
//!
//! The block is shared with Qwen3.8 Flash-Next, which picks 10 of 512 experts: an id-indexed
//! launch takes at most 16 pairs (`CudaContext::exl3_indexed_capacity`; 20 under the opt-in
//! `FIFI_EXL3_INDEXED_PAIRS=20`), so a group of more (Flash-Next's 20 gate and up pairs) is
//! queued as consecutive launches of at most that many in group order. Qwen3.6's 8 picks make
//! one launch, as before.

pub(in crate::family) mod merge;

use super::{
    layer_table::LayerTable,
    moe::{Launch, MoeDims, RouteBufs},
};
use crate::{
    EngError,
    family::{
        gguf_model::Role,
        project::cuda::{
            CudaWeight, CudaWeights,
            exl3::{Exl3Bank, Exl3Io, Exl3Weight},
        },
    },
    kernels::cuda::{
        DeviceBuffer, DeviceEvent, DeviceRange,
        prefill::swiglu_rows::{SwigluRowsCall, SwigluRowsShape},
        resident::{
            moe_combine::{CombineCall, CombineOut, CombineShape, SharedTerm},
            router::{RouterCall, RouterOut, RouterShape, RouterWeights},
        },
    },
};

const F32: usize = size_of::<f32>();

/// The most rows one down-projection group of the rows kernel takes (`grouped_max_rows`,
/// exl3.cu `grouped_project_rows`); a wider window splits at this width.
const GROUPED_MAX_ROWS: usize = 8;
const I32: usize = size_of::<i32>();

pub(in crate::family) struct Exl3MoeWeights<'w> {
    /// The raw F32 `[n_expert][hidden]` router matrix.
    pub(super) router: &'w DeviceBuffer,
    /// The raw F32 `[hidden]` shared-gate vector.
    pub(super) router_shared: &'w DeviceBuffer,
    pub(super) gate: &'w Exl3Bank,
    pub(super) up: &'w Exl3Bank,
    pub(super) down: &'w Exl3Bank,
    pub(super) shared_gate: &'w Exl3Weight,
    pub(super) shared_up: &'w Exl3Weight,
    pub(super) shared_down: &'w Exl3Weight,
}

impl<'w> Exl3MoeWeights<'w> {
    pub(super) fn resolve(
        weights: &'w CudaWeights,
        table: &LayerTable,
        layer: usize,
        moe: &MoeDims,
    ) -> Result<Self, EngError> {
        Self::resolve_by(weights, layer, moe, |role| table.name(layer, role))
    }

    /// The block of `layer` of a banked load: the routed banks hold `members` slots, not the
    /// full expert count, and the host miss resolver feeds the launches slot indices.
    pub(super) fn resolve_banked(
        weights: &'w CudaWeights,
        table: &LayerTable,
        layer: usize,
        moe: &MoeDims,
        members: usize,
    ) -> Result<Self, EngError> {
        Self::resolve_by_members(weights, layer, moe, |role| table.name(layer, role), members)
    }

    /// The block of `layer` in `weights`, where `name` gives the key of each role's tensor.
    pub(in crate::family) fn resolve_by<'n>(
        weights: &'w CudaWeights,
        layer: usize,
        moe: &MoeDims,
        name: impl Fn(Role) -> Result<&'n str, EngError>,
    ) -> Result<Self, EngError> {
        Self::resolve_by_members(weights, layer, moe, name, moe.n_expert)
    }

    /// [`resolve_by`] with the routed banks' member count named by the caller: a banked load
    /// resolves against its slot count.
    fn resolve_by_members<'n>(
        weights: &'w CudaWeights,
        layer: usize,
        moe: &MoeDims,
        name: impl Fn(Role) -> Result<&'n str, EngError>,
        members: usize,
    ) -> Result<Self, EngError> {
        let raw = |role, values: usize| -> Result<&'w DeviceBuffer, EngError> {
            match weights.get(name(role)?) {
                Some(CudaWeight::Raw { buffer, .. }) if buffer.bytes() == values * F32 => {
                    Ok(buffer)
                }
                _ => Err(refuse(format!(
                    "layer {layer} {role:?} is not a raw F32 tensor of {values} values"
                ))),
            }
        };
        let bank = |role, shape: [usize; 2]| -> Result<&'w Exl3Bank, EngError> {
            match weights.get(name(role)?) {
                Some(CudaWeight::Exl3Bank { bank, .. })
                    if bank.count() == members && bank.shape() == shape =>
                {
                    Ok(bank)
                }
                _ => Err(refuse(format!(
                    "layer {layer} {role:?} is not a bank of {members} experts of {shape:?}"
                ))),
            }
        };
        let single = |role, shape: [usize; 2]| -> Result<&'w Exl3Weight, EngError> {
            match weights.get(name(role)?) {
                Some(CudaWeight::Exl3 { weight, .. }) if weight.shape == shape => Ok(weight),
                _ => Err(refuse(format!(
                    "layer {layer} {role:?} is not an EXL3 matrix of {shape:?}"
                ))),
            }
        };
        let (h, i, s) = (moe.hidden, moe.inter, moe.shared_inter);
        Ok(Self {
            router: raw(Role::FfnGateInp, moe.n_expert * h)?,
            router_shared: raw(Role::FfnGateInpShexp, h)?,
            gate: bank(Role::FfnGateExps, [h, i])?,
            up: bank(Role::FfnUpExps, [h, i])?,
            down: bank(Role::FfnDownExps, [i, h])?,
            shared_gate: single(Role::FfnGateShexp, [h, s])?,
            shared_up: single(Role::FfnUpShexp, [h, s])?,
            shared_down: single(Role::FfnDownShexp, [s, h])?,
        })
    }
}

pub(in crate::family) struct Exl3MoeBufs<'b> {
    pub(in crate::family) route: RouteBufs<'b>,
    /// f32 `[top_k][gate | up]`.
    pub(in crate::family) expert_gu: DeviceRange<'b>,
    /// f32 `[top_k][inter]`: the SwiGLU of the picks.
    pub(in crate::family) expert_act: DeviceRange<'b>,
    /// f32 `[top_k][hidden]`: the down outputs in pick order.
    pub(in crate::family) expert_down: DeviceRange<'b>,
    /// f32 `[gate | up]` of the shared expert.
    pub(in crate::family) shared_gu: DeviceRange<'b>,
    /// f32 `[shared_inter]`.
    pub(in crate::family) shared_act: DeviceRange<'b>,
    pub(in crate::family) shared_out: DeviceRange<'b>,
    /// The rotated inputs, the split-K partials and the arithmetic flag of the projections.
    pub(in crate::family) scratch: Scratch<'b>,
}

/// The shared scratch of the EXL3 projections (see `Slot::Exl3Xh`, `Exl3Z`, `Exl3Flag`).
#[derive(Clone, Copy)]
pub(in crate::family) struct Scratch<'b> {
    pub(in crate::family) xh: DeviceRange<'b>,
    pub(in crate::family) z: DeviceRange<'b>,
    pub(in crate::family) flag: DeviceRange<'b>,
}

fn refuse(detail: String) -> EngError {
    EngError::BadRequest(format!("EXL3 moe block: {detail}"))
}

fn row<'b>(
    range: &DeviceRange<'b>,
    width: usize,
    index: usize,
) -> Result<DeviceRange<'b>, EngError> {
    range.buffer().range(range.offset() + index * width, width)
}

/// The `width` rows of `range` starting at row `r0` of `row_bytes` each.
fn span<'b>(
    range: &DeviceRange<'b>,
    row_bytes: usize,
    r0: usize,
    width: usize,
) -> Result<DeviceRange<'b>, EngError> {
    range
        .buffer()
        .range(range.offset() + r0 * row_bytes, width * row_bytes)
}

/// The first `bytes` bytes of `range`.
fn leading<'b>(range: &DeviceRange<'b>, bytes: usize) -> Result<DeviceRange<'b>, EngError> {
    range.buffer().range(range.offset(), bytes)
}

fn whole(buffer: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(0, buffer.bytes())
}

/// The ids of the picks, read back once the router has run; a faulty router row writes -1.
/// Reads `count` picks back, which waits for the stream: the one drain the block makes. A
/// fault-marked or out-of-range id is the router's own error (non-finite logits).
fn read_picks(
    launch: &Launch<'_>,
    ids: &DeviceRange<'_>,
    count: usize,
    n_expert: usize,
) -> Result<Vec<usize>, EngError> {
    let mut raw = vec![0u8; count * I32];
    #[cfg(test)]
    let began = std::time::Instant::now();
    ids.buffer()
        .read_at(launch.stream, ids.offset(), &mut raw)?;
    #[cfg(test)]
    crate::family::qwen36::cuda::exl3_bank::probe::readback(began.elapsed());
    raw.as_chunks::<I32>()
        .0
        .iter()
        .map(|bytes| {
            usize::try_from(i32::from_le_bytes(*bytes))
                .ok()
                .filter(|&id| id < n_expert)
                .ok_or_else(|| {
                    EngError::ModelOutput(
                        "EXL3 moe block: the router picked no valid expert (non-finite logits)"
                            .into(),
                    )
                })
        })
        .collect()
}

struct Pair<'a> {
    view: crate::family::project::cuda::Exl3MatrixView<'a>,
    x: DeviceRange<'a>,
    out: DeviceRange<'a>,
}

/// `FIFI_EXL3_GROUP_SPLIT`: the k split of the banked grouped launches (2, 4 or 8;
/// default 2). 4/8 change the partial fold order for more blocks in flight - the
/// fill-limited decode lever - so outputs differ from 2 in the last ulp; keep 2
/// wherever the byte pins must hold.
static GROUP_SPLIT_OVERRIDE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Test-only override of [`group_split`] (the env is process-wide and OnceLock-cached).
#[cfg(test)]
pub(super) fn set_group_split_override(split: u32) {
    GROUP_SPLIT_OVERRIDE.store(split, std::sync::atomic::Ordering::Relaxed);
}

pub(super) fn group_split() -> u32 {
    let set = GROUP_SPLIT_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if matches!(set, 2 | 4 | 8) {
        return set;
    }
    static SPLIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *SPLIT.get_or_init(|| {
        std::env::var("FIFI_EXL3_GROUP_SPLIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v| matches!(v, 2 | 4 | 8))
            .unwrap_or(2)
    })
}

/// Queues one grouped launch of `pairs`, all of one shape: pair `p` gets row `p` of the scratch
/// partial and rotated-input areas.
fn grouped(launch: &Launch<'_>, pairs: &[Pair<'_>], scratch: &Scratch<'_>) -> Result<(), EngError> {
    let [k, n] = pairs
        .first()
        .map(|pair| pair.view.shape)
        .ok_or_else(|| refuse("a grouped launch without pairs".into()))?;
    let split = group_split();
    let banks = (split as usize - 1).max(1);
    let mut calls = Vec::with_capacity(pairs.len());
    for (index, pair) in pairs.iter().enumerate() {
        let io = Exl3Io {
            x: pair.x,
            xh: row(&scratch.xh, k * F32, index)?,
            z: row(&scratch.z, n * F32 * banks, index)?,
            out: pair.out,
            flag: scratch.flag,
            rows: 1,
            fused: false,
        };
        let mut call = pair.view.linear_call(&io);
        call.split_k = split;
        calls.push(call);
    }
    if split == 2 {
        return launch.ctx.exl3_grouped_step(launch.stream, &calls);
    }
    launch.ctx.exl3_grouped_split(launch.stream, &calls)
}

/// [`grouped`] as one launch over the whole set: the C side of the grouped kernels takes
/// scattered out rows (each group's extent is validated per call), so a resident/refilling
/// split - or a hit pick inside a miss run - no longer needs one launch per contiguous slice.
pub(super) fn grouped_runs(
    launch: &Launch<'_>,
    pairs: &[Pair<'_>],
    scratch: &Scratch<'_>,
) -> Result<(), EngError> {
    grouped(launch, pairs, scratch)
}

/// One pair of an id-indexed launch: the bank, the pick slot whose id chooses the member, the row
/// the pair reads and where its rows go.
struct IndexedPair<'a> {
    bank: &'a Exl3Bank,
    slot: u32,
    x: DeviceRange<'a>,
    out: DeviceRange<'a>,
}

/// Queues the id-indexed group `pairs`, all of one shape, as consecutive launches of at most
/// `exl3_indexed_capacity` pairs in group order. Each launch starts again at row 0 of the
/// scratch; the stream runs it after the previous launch has finished with those rows.
fn grouped_indexed(
    launch: &Launch<'_>,
    pairs: &[IndexedPair<'_>],
    ids: DeviceRange<'_>,
    scratch: &Scratch<'_>,
) -> Result<(), EngError> {
    if pairs.is_empty() {
        return Err(refuse("an indexed launch without pairs".into()));
    }
    for chunk in pairs.chunks(launch.ctx.exl3_indexed_capacity()) {
        indexed_launch(launch, chunk, ids, scratch)?;
    }
    Ok(())
}

/// Queues one id-indexed launch of `pairs`, all of one shape: pair `p` gets row `p` of the scratch
/// partial and rotated-input areas, and its member is `ids[slot]` read on the device.
fn indexed_launch(
    launch: &Launch<'_>,
    pairs: &[IndexedPair<'_>],
    ids: DeviceRange<'_>,
    scratch: &Scratch<'_>,
) -> Result<(), EngError> {
    let [k, n] = pairs
        .first()
        .map(|pair| pair.bank.shape())
        .ok_or_else(|| refuse("an indexed launch without pairs".into()))?;
    let mut calls = Vec::with_capacity(pairs.len());
    for (index, pair) in pairs.iter().enumerate() {
        let io = Exl3Io {
            x: pair.x,
            xh: row(&scratch.xh, k * F32, index)?,
            z: row(&scratch.z, n * F32, index)?,
            out: pair.out,
            flag: scratch.flag,
            rows: 1,
            fused: false,
        };
        calls.push(pair.bank.span()?.indexed_call(&io, ids, pair.slot));
    }
    launch
        .ctx
        .exl3_grouped_indexed(launch.stream, &calls, launch.ctx.exl3_indexed_cols(n)?)
}

fn route(
    launch: &Launch<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Exl3MoeBufs<'_>,
    dims: &MoeDims,
) -> Result<(), EngError> {
    let route = &bufs.route;
    launch.ctx.router(
        launch.stream,
        RouterCall {
            shape: RouterShape {
                m: dims.rows,
                hidden: dims.hidden,
                n_expert: dims.n_expert,
                top_k: dims.top_k,
            },
            x: route.norm_row,
            weights: RouterWeights {
                w: whole(weights.router)?,
                w_shared: Some(whole(weights.router_shared)?),
            },
            logits: route.logits,
            out: RouterOut {
                ids: route.ids,
                weights: route.weights,
                shared_gate: Some(route.shared_gate),
                status: route.status,
            },
        },
    )
}

/// The SwiGLU of `rows` rows of `[gate | up]` into `rows` rows of `width` values.
fn swiglu(
    launch: &Launch<'_>,
    gu: DeviceRange<'_>,
    act: DeviceRange<'_>,
    (rows, width): (usize, usize),
) -> Result<(), EngError> {
    launch.ctx.swiglu_rows(
        launch.stream,
        SwigluRowsCall {
            shape: SwigluRowsShape {
                rows,
                n_ff: width,
                up_off: width,
                ld_in: 2 * width,
                ld_out: width,
            },
            gu,
            act,
        },
    )
}

/// The routed experts of the host-picks arm: `picks` holds one slot (banked) or expert
/// (resident) per pick. `gate`, when present, carries the overlapped refills of a banked
/// call: the pick indices whose slots are refilling on the copy stream and the event those
/// picks' launches must first wait. The picks outside `gate.1` are already resident, so
/// their gate/up pairs launch before the wait and their compute hides the refill DMA; the
/// swiglu, the down pairs and the combine still wait (they need every pick's output), which
/// is why only the gate/up phase splits.
fn routed(
    launch: &Launch<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Exl3MoeBufs<'_>,
    (picks, dims): (&[usize], &MoeDims),
    gate: Option<(&DeviceEvent, &[usize])>,
) -> Result<(), EngError> {
    let (h, i) = (dims.hidden, dims.inter);
    let mut resident = Vec::with_capacity(2 * picks.len());
    let mut refilling = Vec::new();
    for (index, &slot) in picks.iter().enumerate() {
        let gated = gate.is_some_and(|(_, missed)| missed.contains(&index));
        for (half, bank) in [weights.gate, weights.up].into_iter().enumerate() {
            let pair = Pair {
                view: bank.view(slot)?,
                x: bufs.route.norm_row,
                out: row(&bufs.expert_gu, i * F32, 2 * index + half)?,
            };
            if gated {
                refilling.push(pair);
            } else {
                resident.push(pair);
            }
        }
    }
    if !resident.is_empty() {
        grouped_runs(launch, &resident, &bufs.scratch)?;
    }
    if !refilling.is_empty() {
        if let Some((event, _)) = gate {
            launch.stream.wait(event)?;
        }
        grouped_runs(launch, &refilling, &bufs.scratch)?;
    }
    swiglu(launch, bufs.expert_gu, bufs.expert_act, (dims.top_k, i))?;
    let mut group = Vec::with_capacity(picks.len());
    for (slot, &expert) in picks.iter().enumerate() {
        group.push(Pair {
            view: weights.down.view(expert)?,
            x: row(&bufs.expert_act, i * F32, slot)?,
            out: row(&bufs.expert_down, h * F32, slot)?,
        });
    }
    grouped(launch, &group, &bufs.scratch)
}

/// The routed experts with the picks read on the device: the same groups and the same arithmetic
/// as [`routed`], but the member of every pair comes from the router's id array when the kernels
/// run, so the block waits for nothing and can be captured in a graph.
pub(super) fn routed_on_device(
    launch: &Launch<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Exl3MoeBufs<'_>,
    dims: &MoeDims,
) -> Result<(), EngError> {
    let (h, i) = (dims.hidden, dims.inter);
    let slots = u32::try_from(dims.top_k).map_err(|_| refuse("top_k does not fit u32".into()))?;
    let ids = bufs.route.ids;
    let mut group = Vec::with_capacity(2 * dims.top_k);
    for slot in 0..slots {
        for (half, bank) in [weights.gate, weights.up].into_iter().enumerate() {
            group.push(IndexedPair {
                bank,
                slot,
                x: bufs.route.norm_row,
                out: row(&bufs.expert_gu, i * F32, 2 * slot as usize + half)?,
            });
        }
    }
    grouped_indexed(launch, &group, ids, &bufs.scratch)?;
    swiglu(launch, bufs.expert_gu, bufs.expert_act, (dims.top_k, i))?;
    let mut group = Vec::with_capacity(dims.top_k);
    for slot in 0..slots {
        group.push(IndexedPair {
            bank: weights.down,
            slot,
            x: row(&bufs.expert_act, i * F32, slot as usize)?,
            out: row(&bufs.expert_down, h * F32, slot as usize)?,
        });
    }
    grouped_indexed(launch, &group, ids, &bufs.scratch)
}

/// The shared expert: gate and up in one group, the SwiGLU, the down projection.
pub(super) fn shared(
    launch: &Launch<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Exl3MoeBufs<'_>,
    dims: &MoeDims,
) -> Result<(), EngError> {
    let matrices = SharedWeights {
        gate: weights.shared_gate,
        up: weights.shared_up,
        down: weights.shared_down,
    };
    shared_of(launch, &matrices, bufs, dims)
}

/// The shared expert of [`shared`] over its three matrices alone, for a block whose routed experts
/// do not come from an [`Exl3Bank`] (the Flash-Next bank of records).
pub(in crate::family) fn shared_of(
    launch: &Launch<'_>,
    weights: &SharedWeights<'_>,
    bufs: &Exl3MoeBufs<'_>,
    dims: &MoeDims,
) -> Result<(), EngError> {
    let s = dims.shared_inter;
    let mut group = Vec::with_capacity(2);
    for (half, weight) in [weights.gate, weights.up].into_iter().enumerate() {
        group.push(Pair {
            view: weight.view()?,
            x: bufs.route.norm_row,
            out: row(&bufs.shared_gu, s * F32, half)?,
        });
    }
    grouped(launch, &group, &bufs.scratch)?;
    swiglu(launch, bufs.shared_gu, bufs.shared_act, (1, s))?;
    let down = [Pair {
        view: weights.down.view()?,
        x: bufs.shared_act,
        out: bufs.shared_out,
    }];
    grouped(launch, &down, &bufs.scratch)
}

/// The three matrices of the shared expert.
pub(in crate::family) struct SharedWeights<'w> {
    pub(in crate::family) gate: &'w Exl3Weight,
    pub(in crate::family) up: &'w Exl3Weight,
    pub(in crate::family) down: &'w Exl3Weight,
}

/// The ranges of the shared expert of a `rows`-row window. `norm_row` holds `rows` rows of the
/// hidden width; `gu` holds `rows` rows of `[gate | up]` (`2 * shared_inter` values each), `act`
/// `rows` rows of `shared_inter` and `out` `rows` rows of the hidden width.
pub(in crate::family) struct SharedRowsIo<'b> {
    pub(in crate::family) norm_row: DeviceRange<'b>,
    pub(in crate::family) gu: DeviceRange<'b>,
    pub(in crate::family) act: DeviceRange<'b>,
    pub(in crate::family) out: DeviceRange<'b>,
    pub(in crate::family) scratch: Scratch<'b>,
}

/// The shared expert of a K-row verify window, `rows` in 2 to 8.
///
/// Gate and up are `2 * rows` one-row groups in ONE grouped launch, ordered
/// `[gate_0, up_0, gate_1, up_1, ...]` like the routed block (row `r` owns the pairs `2 r` and
/// `2 r + 1`), so each row's `[gate | up]` is contiguous and ONE `swiglu_rows` over `rows` rows
/// reads it with `ld_in = 2 s`, `up_off = s`, the layout of the one-row block: no relaxation of the
/// SwiGLU's validation. Every pair is exactly the one-row computation (a group of one row, the
/// one-tile wide body). The down projection is one group of `rows` rows, the rows kernel. Two
/// groups of `rows` rows each (`[gate rows][up rows]`) would land as `[gate K rows][up K rows]`
/// and need `up_off = K s` with `ld_in = s`, which `valid_shape` refuses.
///
/// The one-row block [`shared`] is untouched; the test compares this one to `rows` runs of it.
pub(in crate::family) fn shared_rows(
    launch: &Launch<'_>,
    weights: &SharedWeights<'_>,
    io: &SharedRowsIo<'_>,
    (hidden, shared_inter, rows): (usize, usize, usize),
) -> Result<(), EngError> {
    if !(2..=16).contains(&rows) {
        return Err(refuse(format!(
            "the K-row shared expert takes 2 to 16 rows, got {rows}"
        )));
    }
    let s = shared_inter;
    let mut group = Vec::with_capacity(2 * rows);
    for r in 0..rows {
        for (half, weight) in [weights.gate, weights.up].into_iter().enumerate() {
            group.push(Pair {
                view: weight.view()?,
                x: row(&io.norm_row, hidden * F32, r)?,
                out: row(&io.gu, s * F32, 2 * r + half)?,
            });
        }
    }
    grouped(launch, &group, &io.scratch)?;
    swiglu(launch, io.gu, io.act, (rows, s))?;
    // The down projection's rows kernel takes one group of at most `GROUPED_MAX_ROWS` rows
    // (`grouped_project_rows`, exl3.cu). Gate and up above are one-row groups, so only the
    // down splits: `ceil(rows / 8)` row-disjoint groups, each exactly the computation the
    // one-group launch ran for those rows. A window of at most 8 rows keeps the single call.
    let view = weights.down.view()?;
    let [k, n] = view.shape;
    for r0 in (0..rows).step_by(GROUPED_MAX_ROWS) {
        let width = (rows - r0).min(GROUPED_MAX_ROWS);
        let down = Exl3Io {
            x: span(&io.act, s * F32, r0, width)?,
            xh: span(&io.scratch.xh, k * F32, r0, width)?,
            z: span(&io.scratch.z, n * F32, r0, width)?,
            out: span(&io.out, n * F32, r0, width)?,
            flag: io.scratch.flag,
            rows: width,
            fused: false,
        };
        launch
            .ctx
            .exl3_grouped_step(launch.stream, &[view.linear_call(&down)])?;
    }
    Ok(())
}

/// `resid_out = resid_in + (picks in pick order + gate * shared)`, as the GGUF block combines.
fn combine(launch: &Launch<'_>, bufs: &Exl3MoeBufs<'_>, dims: &MoeDims) -> Result<(), EngError> {
    combine_ranges(
        launch,
        &bufs.route,
        (bufs.expert_down, bufs.shared_out),
        dims,
    )
}

/// The combine over the routed `down` rows and the `shared` row, wherever the block put them.
fn combine_ranges(
    launch: &Launch<'_>,
    route: &RouteBufs<'_>,
    (down, shared): (DeviceRange<'_>, DeviceRange<'_>),
    dims: &MoeDims,
) -> Result<(), EngError> {
    launch.ctx.moe_combine(
        launch.stream,
        CombineCall {
            shape: CombineShape {
                m: dims.rows,
                n: dims.hidden,
                top_k: dims.top_k,
                ld_down: dims.hidden,
            },
            res: route.resid_in,
            down,
            weights: route.weights,
            shared: Some(SharedTerm {
                gate: route.shared_gate,
                row: shared,
            }),
            out: CombineOut::To(route.resid_out),
        },
    )
}

/// The block of one layer on the one stream. With `device_ids` the picks stay on the device and
/// the block makes no host read; otherwise it waits for the stream once, to read the ids. With
/// `bank` the read picks map through the layer's slot table, whose missed slots this fills from
/// the host store first, and the launches then read the slot members.
pub(in crate::family) fn run_exl3_moe(
    launch: &Launch<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Exl3MoeBufs<'_>,
    dims: &MoeDims,
    device_ids: bool,
    mut bank: Option<&mut super::exl3_bank::LayerResolver<'_, '_>>,
    overlap: bool,
) -> Result<(), EngError> {
    if dims.rows != 1 {
        return Err(refuse(format!(
            "the block takes one row, got {}",
            dims.rows
        )));
    }
    route(launch, weights, bufs, dims)?;
    if device_ids {
        if bank.is_some() {
            return Err(refuse(
                "the banked block reads the picks on the host; capture is refused while banked"
                    .into(),
            ));
        }
        routed_on_device(launch, weights, bufs, dims)?;
    } else {
        let picks = read_picks(launch, &bufs.route.ids, dims.top_k, dims.n_expert)?;
        let mut gated = None;
        let picks = match bank.as_deref_mut() {
            Some(bank) => {
                #[cfg(test)]
                let began = std::time::Instant::now();
                let split = bank.resolve_split(launch.stream, &picks, overlap);
                #[cfg(test)]
                crate::family::qwen36::cuda::exl3_bank::probe::resolve(began.elapsed());
                // A resolve that fails mid-refill leaves copies on the copy stream with no
                // event recorded, so no wait would ever order them (fence_copy_stream).
                let (slots, missed, event) = match split {
                    Ok(split) => split,
                    Err(error) => {
                        bank.fence_copy_stream();
                        return Err(error);
                    }
                };
                gated = event.map(|event| (event, missed));
                slots
            }
            None => picks,
        };
        let gate = gated
            .as_ref()
            .map(|(event, missed)| (*event, missed.as_slice()));
        if let Err(error) = routed(launch, weights, bufs, (&picks, dims), gate) {
            if let Some(bank) = bank.as_deref_mut() {
                // The block queues no wait on its way out with an error; its refills keep
                // writing with no fence (LayerResolver::fence_copy_stream).
                bank.fence_copy_stream();
            }
            return Err(error);
        }
    }
    shared(launch, weights, bufs, dims)?;
    combine(launch, bufs, dims)
}

/// The ranges of row `index` of the K-row buffers, as the one-row block takes them: every
/// per-row buffer offset by `index` rows. The scratch and the status word are shared (the rows run
/// in stream order).
fn row_bufs<'b>(
    bufs: &Exl3MoeBufs<'b>,
    dims: &MoeDims,
    index: usize,
) -> Result<Exl3MoeBufs<'b>, EngError> {
    let (h, i, s, k) = (dims.hidden, dims.inter, dims.shared_inter, dims.top_k);
    let route = &bufs.route;
    Ok(Exl3MoeBufs {
        route: RouteBufs {
            resid_in: row(&route.resid_in, h * F32, index)?,
            resid_out: row(&route.resid_out, h * F32, index)?,
            norm_row: row(&route.norm_row, h * F32, index)?,
            logits: row(&route.logits, (dims.n_expert + 1) * F32, index)?,
            ids: row(&route.ids, k * I32, index)?,
            weights: row(&route.weights, k * F32, index)?,
            shared_gate: row(&route.shared_gate, F32, index)?,
            status: route.status,
        },
        expert_gu: row(&bufs.expert_gu, k * 2 * i * F32, index)?,
        expert_act: row(&bufs.expert_act, k * i * F32, index)?,
        expert_down: row(&bufs.expert_down, k * h * F32, index)?,
        shared_gu: row(&bufs.shared_gu, 2 * s * F32, index)?,
        shared_act: row(&bufs.shared_act, s * F32, index)?,
        shared_out: row(&bufs.shared_out, h * F32, index)?,
        scratch: bufs.scratch,
    })
}

/// The router of a K-row block: one launch over every row, its picks to the id array.
fn route_rows(
    launch: &Launch<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Exl3MoeBufs<'_>,
    rows: usize,
    dims: &MoeDims,
) -> Result<(), EngError> {
    let route = &bufs.route;
    launch.ctx.router_rows(
        launch.stream,
        RouterCall {
            shape: RouterShape {
                m: rows,
                hidden: dims.hidden,
                n_expert: dims.n_expert,
                top_k: dims.top_k,
            },
            x: route.norm_row,
            weights: RouterWeights {
                w: whole(weights.router)?,
                w_shared: Some(whole(weights.router_shared)?),
            },
            logits: route.logits,
            out: RouterOut {
                ids: route.ids,
                weights: route.weights,
                shared_gate: Some(route.shared_gate),
                status: route.status,
            },
        },
    )
}

/// The block of one layer of a K-row verify window (`dims.rows` rows, 2 to 8) on the one stream.
/// `bufs` holds the K-row buffers: row `r` of each per-row buffer is at `r` rows from its start.
///
/// The router runs once for all rows (the decode kernel at every row count: below 8 rows one
/// launch, from 8 rows chunks of 4). The routed experts differ per row, so each row runs its own
/// chain of id-indexed launches on its own ranges, exactly the launches of the one-row block with
/// other pointers; the results of every row stay in their own buffers until the combine. The
/// shared expert and the combine take all rows at once. The picks stay on the device, so the block
/// waits for nothing.
pub(in crate::family) fn run_exl3_moe_rows(
    launch: &Launch<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Exl3MoeBufs<'_>,
    dims: &MoeDims,
) -> Result<(), EngError> {
    let rows = dims.rows;
    if !(2..=8).contains(&rows) {
        return Err(refuse(format!(
            "the K-row block takes 2 to 8 rows, got {rows}"
        )));
    }
    route_rows(launch, weights, bufs, rows, dims)?;
    let one = MoeDims { rows: 1, ..*dims };
    for index in 0..rows {
        routed_on_device(launch, weights, &row_bufs(bufs, dims, index)?, &one)?;
    }
    let route = &bufs.route;
    shared_rows(
        launch,
        &SharedWeights {
            gate: weights.shared_gate,
            up: weights.shared_up,
            down: weights.shared_down,
        },
        &SharedRowsIo {
            norm_row: route.norm_row,
            gu: bufs.shared_gu,
            act: bufs.shared_act,
            out: bufs.shared_out,
            scratch: bufs.scratch,
        },
        (dims.hidden, dims.shared_inter, rows),
    )?;
    combine(launch, bufs, dims)
}

/// The block of one layer of a K-row verify window on a banked engine: the router, the shared
/// expert and the combine are the resident body's. The routed experts differ: the picks of
/// every row are read back once — the one drain of the layer — then each row runs the host-pick
/// chain of the one-row banked block over its own ranges, its picks mapped through the layer's
/// slot table (refilling missed slots from the host store first), so the grouped launches read
/// the slot members of the bank views exactly as the resident body reads its expert members.
/// The body waits for the host inside every layer, so a banked step never captures it.
pub(in crate::family) fn run_exl3_moe_rows_banked(
    launch: &Launch<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Exl3MoeBufs<'_>,
    dims: &MoeDims,
    bank: &mut super::exl3_bank::LayerResolver<'_, '_>,
) -> Result<(), EngError> {
    let rows = dims.rows;
    if !(2..=8).contains(&rows) {
        return Err(refuse(format!(
            "the K-row block takes 2 to 8 rows, got {rows}"
        )));
    }
    route_rows(launch, weights, bufs, rows, dims)?;
    let picks = read_picks(launch, &bufs.route.ids, rows * dims.top_k, dims.n_expert)?;
    let one = MoeDims { rows: 1, ..*dims };
    for (index, row) in picks.chunks(dims.top_k).enumerate() {
        #[cfg(test)]
        let began = std::time::Instant::now();
        let mapped = bank.resolve(launch.stream, row)?;
        #[cfg(test)]
        crate::family::qwen36::cuda::exl3_bank::probe::resolve(began.elapsed());
        routed(
            launch,
            weights,
            &row_bufs(bufs, dims, index)?,
            (&mapped, &one),
            None,
        )?;
    }
    let route = &bufs.route;
    shared_rows(
        launch,
        &SharedWeights {
            gate: weights.shared_gate,
            up: weights.shared_up,
            down: weights.shared_down,
        },
        &SharedRowsIo {
            norm_row: route.norm_row,
            gu: bufs.shared_gu,
            act: bufs.shared_act,
            out: bufs.shared_out,
            scratch: bufs.scratch,
        },
        (dims.hidden, dims.shared_inter, rows),
    )?;
    combine(launch, bufs, dims)
}
