//! Pure host planning of the opt-in EXL3 MoE block that queues the shared expert inside the
//! routed launches (`FIFI_EXL3_SHARED_IN_ROUTED`): the switch, the pairs of each launch, and the
//! byte layout of the working buffer. No device, no allocation. The CUDA side
//! (`family/exl3/moe/merge.rs`, which holds the layout contract in its module doc) runs exactly the
//! plan built here, and the host tests (`exl3_merge_plan_tests.rs`) pin it.

use std::ffi::OsStr;

use crate::{EngError, kernels::single_row::SingleRow};

const F32: usize = size_of::<f32>();

/// The sizes of one MoE block the plan needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) struct PlanDims {
    /// Token rows; the merged block takes one.
    pub(in crate::family) rows: usize,
    pub(in crate::family) hidden: usize,
    /// Experts picked per row.
    pub(in crate::family) top_k: usize,
    /// Width of one routed expert.
    pub(in crate::family) inter: usize,
    /// Width of the shared expert.
    pub(in crate::family) shared_inter: usize,
}

/// The environment switch of the shared expert in the routed launches. On an EXL3 Qwen3.6 load,
/// unset means ON as a default (a load that cannot run merged falls back to the reference block and
/// says why), `0` means OFF, and `1` means ON as an explicit request (a load that cannot run merged
/// is refused). On any other weight format unset and `0` mean off and `1` is refused. Process-wide,
/// read once, strict: any other value stops the session start. Same bytes as the reference block
/// (device-tested), fewer launches.
pub(in crate::family) const SHARED_IN_ROUTED_VAR: &str = "FIFI_EXL3_SHARED_IN_ROUTED";

/// The most pairs of the merged gate/up launch: the 20-pair kernels' capacity.
const PAIR_LIMIT: usize = 20;

/// Parses [`SHARED_IN_ROUTED_VAR`]: `None` when unset, else whether it asks for the merged block.
pub(in crate::family) fn shared_in_routed_from_env(
    value: Option<&OsStr>,
) -> Result<Option<bool>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value.to_str().map(str::trim) {
        Some("0") => Ok(Some(false)),
        Some("1") => Ok(Some(true)),
        _ => Err(format!(
            "{SHARED_IN_ROUTED_VAR} must be unset, \"0\" or \"1\""
        )),
    }
}

/// The switch from the environment, or the refusal of a bad value. Read once per process.
pub(in crate::family) fn shared_in_routed_explicit() -> Result<Option<bool>, EngError> {
    static EXPLICIT: std::sync::OnceLock<Result<Option<bool>, String>> = std::sync::OnceLock::new();
    EXPLICIT
        .get_or_init(|| {
            shared_in_routed_from_env(std::env::var_os(SHARED_IN_ROUTED_VAR).as_deref())
        })
        .clone()
        .map_err(EngError::BadRequest)
}

/// Where the mode of the merged block came from, for the load line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) enum Source {
    /// The variable is unset and the load can run the block.
    Default,
    /// The variable is set: `1` (on) or `0` (off).
    Explicit,
    /// The variable is unset and the load cannot run the block: the reference block runs.
    Fallback,
    /// The variable is unset and `draft_verify = krow` took the block off for this run.
    Krow,
}

impl Source {
    fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Explicit => "explicit",
            Self::Fallback => "fallback",
            Self::Krow => "krow",
        }
    }
}

/// The load-line suffix of an EXL3 load: the resolved mode and where it came from, e.g.
/// ` exl3_shared_in_routed=on source=default`. (A GGUF load prints nothing.) The `=on` token is
/// what the GPU scripts count.
pub(in crate::family) fn load_suffix(running: bool, source: Source) -> String {
    format!(
        " exl3_shared_in_routed={} source={}",
        if running { "on" } else { "off" },
        source.name()
    )
}

/// What a MoE block must satisfy for the shared expert to ride in the routed launches.
pub(in crate::family) fn check_dims(dims: &PlanDims) -> Result<(), String> {
    if dims.rows != 1 {
        return Err(format!("the block takes one row, got {}", dims.rows));
    }
    if dims.top_k == 0 || 2 * dims.top_k + 2 > PAIR_LIMIT {
        return Err(format!(
            "{} picks need {} gate and up pairs in one launch, at most {PAIR_LIMIT} fit",
            dims.top_k,
            2 * dims.top_k + 2
        ));
    }
    if dims.shared_inter != dims.inter {
        return Err(format!(
            "the shared expert is {} wide, the routed experts {}: the pairs of one launch share k and n",
            dims.shared_inter, dims.inter
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) enum Proj {
    Gate,
    Up,
    Down,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) enum Member {
    /// The expert the router wrote in `ids[slot]`.
    Routed(u32),
    /// The shared expert: a bank of one member, its ids word is the zero word.
    Shared,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) enum Input {
    /// The normalized row of the block.
    NormRow,
    /// Row `r` of the swiglu output.
    ActRow(usize),
}

/// One pair of a launch. Its scratch row is its index in the launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) struct PlannedPair {
    pub(in crate::family) proj: Proj,
    pub(in crate::family) member: Member,
    pub(in crate::family) input: Input,
    /// Row of the launch's output area the pair writes (`gu` half-row or `down` row).
    pub(in crate::family) out_row: usize,
}

/// The three steps of the merged block, besides the router and the combine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::family) struct BlockPlan {
    pub(in crate::family) gate_up: Vec<PlannedPair>,
    pub(in crate::family) swiglu_rows: usize,
    pub(in crate::family) down: Vec<PlannedPair>,
}

/// The kernels of one grouped or id-indexed launch: input rotate, projection, output rotate.
#[cfg(test)]
const KERNELS_PER_LAUNCH: usize = 3;

#[cfg(test)]
impl BlockPlan {
    /// Kernel launches of the block besides the router and the combine.
    pub(in crate::family) fn kernel_launches(&self) -> usize {
        // The gate/up group and the down group, and the one swiglu between them.
        2 * KERNELS_PER_LAUNCH + 1
    }
}

/// The merged block of a layer that picks `top_k` experts.
pub(in crate::family) fn block_plan(top_k: usize) -> BlockPlan {
    let mut gate_up = Vec::with_capacity(2 * top_k + 2);
    for slot in 0..top_k {
        for (half, proj) in [Proj::Gate, Proj::Up].into_iter().enumerate() {
            gate_up.push(PlannedPair {
                proj,
                member: Member::Routed(slot as u32),
                input: Input::NormRow,
                out_row: 2 * slot + half,
            });
        }
    }
    for (half, proj) in [Proj::Gate, Proj::Up].into_iter().enumerate() {
        gate_up.push(PlannedPair {
            proj,
            member: Member::Shared,
            input: Input::NormRow,
            out_row: 2 * top_k + half,
        });
    }
    let mut down: Vec<PlannedPair> = (0..top_k)
        .map(|slot| PlannedPair {
            proj: Proj::Down,
            member: Member::Routed(slot as u32),
            input: Input::ActRow(slot),
            out_row: slot,
        })
        .collect();
    down.push(PlannedPair {
        proj: Proj::Down,
        member: Member::Shared,
        input: Input::ActRow(top_k),
        out_row: top_k,
    });
    BlockPlan {
        gate_up,
        swiglu_rows: top_k + 1,
        down,
    }
}

/// Kernel launches of the reference block besides the router and the combine: routed gate/up,
/// swiglu, routed down, shared gate/up, swiglu, shared down.
#[cfg(test)]
pub(in crate::family) const fn reference_kernel_launches() -> usize {
    4 * KERNELS_PER_LAUNCH + 2
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) struct Span {
    pub(in crate::family) offset: usize,
    pub(in crate::family) bytes: usize,
}

/// Byte placement of the merged working buffer; every span starts on a 256-byte boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) struct MergeLayout {
    /// Rotated inputs: row `p` of `k` values for pair `p` of the running launch.
    pub(in crate::family) xh: Span,
    /// Split-K partials: row `p` of `n` values.
    pub(in crate::family) z: Span,
    /// `[top_k + 1][gate | up]`, each half `i` wide.
    pub(in crate::family) gu: Span,
    /// `[top_k + 1][i]`.
    pub(in crate::family) act: Span,
    /// `[top_k + 1][h]`.
    pub(in crate::family) down: Span,
    /// Four bytes holding 0, the `ids` word of the shared pairs; nothing writes it.
    pub(in crate::family) zero: Span,
    pub(in crate::family) total: usize,
}

impl MergeLayout {
    pub(in crate::family) fn new(dims: &PlanDims) -> Result<Self, String> {
        check_dims(dims)?;
        let (h, i, k) = (dims.hidden, dims.inter, dims.top_k);
        let pairs = 2 * k + 2;
        let floats = |count: usize| -> Result<usize, String> {
            count
                .checked_mul(F32)
                .ok_or_else(|| "a merged span overflows".to_owned())
        };
        let sizes = [
            floats((pairs * h).max((k + 1) * i))?,
            floats((pairs * i).max((k + 1) * h))?,
            floats((k + 1) * 2 * i)?,
            floats((k + 1) * i)?,
            floats((k + 1) * h)?,
            4,
        ];
        let mut offset = 0usize;
        let mut spans = [Span {
            offset: 0,
            bytes: 0,
        }; 6];
        for (span, bytes) in spans.iter_mut().zip(sizes) {
            *span = Span { offset, bytes };
            offset += bytes.next_multiple_of(256);
        }
        let [xh, z, gu, act, down, zero] = spans;
        Ok(Self {
            xh,
            z,
            gu,
            act,
            down,
            zero,
            total: offset,
        })
    }
}

/// Which `ids` word a pair reads its member from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) enum IdWord {
    /// The router's `TopkId` array.
    Router,
    /// The zero word of the working buffer: the member id of a one-member bank.
    Zero,
}

/// Byte ranges of a pair inside the working buffer. `input` is `None` for the normalized row,
/// which lives in the step arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) struct PairSpans {
    pub(in crate::family) input: Option<Span>,
    pub(in crate::family) output: Span,
}

impl PlannedPair {
    /// The `ids` word and slot the pair's member is read from: pick `s` reads the router's
    /// `ids[s]`, the shared pair reads word 0 of the zero word, so its member is member 0.
    pub(in crate::family) fn id_word(&self) -> (IdWord, u32) {
        match self.member {
            Member::Routed(slot) => (IdWord::Router, slot),
            Member::Shared => (IdWord::Zero, 0),
        }
    }

    /// Where the pair reads its row and writes its result in the working buffer.
    pub(in crate::family) fn spans(&self, dims: &PlanDims, layout: &MergeLayout) -> PairSpans {
        let row = |base: Span, width: usize, index: usize| Span {
            offset: base.offset + index * width * F32,
            bytes: width * F32,
        };
        PairSpans {
            input: match self.input {
                Input::NormRow => None,
                Input::ActRow(index) => Some(row(layout.act, dims.inter, index)),
            },
            output: match self.proj {
                Proj::Gate | Proj::Up => row(layout.gu, dims.inter, self.out_row),
                Proj::Down => row(layout.down, dims.hidden, self.out_row),
            },
        }
    }
}

impl MergeLayout {
    /// The combine's two terms in the `down` area: the routed rows `0..top_k` in pick order and
    /// the shared row `top_k`.
    pub(in crate::family) fn combine_spans(&self, dims: &PlanDims) -> (Span, Span) {
        let routed = Span {
            offset: self.down.offset,
            bytes: dims.top_k * dims.hidden * F32,
        };
        let shared = Span {
            offset: self.down.offset + routed.bytes,
            bytes: dims.hidden * F32,
        };
        (routed, shared)
    }
}

/// What the plan needs to know of one shared matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) struct SharedFacts {
    pub(in crate::family) name: &'static str,
    pub(in crate::family) bias: bool,
    pub(in crate::family) bits_twice: u8,
}

/// Everything that must hold for the shared expert to ride in the routed launches, or the first
/// refusal, naming the switch: the block (rows, picks, widths), the single-row kernel (`wide1`,
/// one tile per block) and, per layer, the shared gate, up and down (no bias, 4 or 6 bits).
pub(in crate::family) fn check_merge(
    dims: &PlanDims,
    kernel: SingleRow,
    layers: &[[SharedFacts; 3]],
) -> Result<(), String> {
    let fail = |detail: String| format!("{SHARED_IN_ROUTED_VAR}: {detail}");
    check_dims(dims).map_err(fail)?;
    if kernel != SingleRow::Wide(1) {
        return Err(fail(format!(
            "it needs the single-row kernel wide1 (one tile per block), the session runs {kernel:?}"
        )));
    }
    for (layer, facts) in layers.iter().enumerate() {
        for fact in facts {
            if fact.bias {
                return Err(fail(format!("layer {layer} {} has a bias", fact.name)));
            }
            if !matches!(fact.bits_twice, 8 | 12) {
                return Err(fail(format!(
                    "layer {layer} {} is {} bits twice, the merged launch supports 8 and 12",
                    fact.name, fact.bits_twice
                )));
            }
        }
    }
    Ok(())
}

/// What the switch asks of a load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) enum Intent {
    /// Reference block: `0`, or any switch value on a weight format with no such block.
    Off,
    /// The default of an EXL3 load (variable unset): merged if the load can run it, else the
    /// reference block with a logged reason. Never a failure.
    Default,
    /// `1` on an EXL3 load: merged, or the start is refused.
    Explicit,
}

/// The switch as the session start resolves it, before any weight is read: a bad value is
/// refused for every weight format, `1` is refused unless the weights are EXL3 (only the EXL3 MoE
/// block has a shared expert to move), and unset on other formats is off and NOT refused.
/// `value` is [`shared_in_routed_explicit`].
pub(in crate::family) fn resolve_switch(
    value: Result<Option<bool>, EngError>,
    exl3_weights: bool,
) -> Result<Intent, EngError> {
    match (value?, exl3_weights) {
        (Some(true), false) => Err(EngError::BadRequest(format!(
            "{SHARED_IN_ROUTED_VAR}=1 applies to EXL3 weights only; unset it or use an EXL3 pack"
        ))),
        (Some(false), _) | (None, false) => Ok(Intent::Off),
        (None, true) => Ok(Intent::Default),
        (Some(true), true) => Ok(Intent::Explicit),
    }
}

/// What a load does with the merged block, given the intent and the result of [`check_merge`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::family) enum Decision {
    /// Run the merged block.
    Merge,
    /// Run the reference block, nothing to say.
    Reference,
    /// Run the reference block and log `reason`: the default could not be honoured.
    Fallback(String),
    /// Refuse the start with `reason`: an explicit request could not be honoured.
    Refuse(String),
}

/// The default must never turn a loadable model into a failure; an explicit `1` keeps the hard
/// refusal. `Off` never looks at the check.
pub(in crate::family) fn decide(intent: Intent, check: Result<(), String>) -> Decision {
    match (intent, check) {
        (Intent::Off, _) => Decision::Reference,
        (_, Ok(())) => Decision::Merge,
        (Intent::Default, Err(reason)) => Decision::Fallback(reason),
        (Intent::Explicit, Err(reason)) => Decision::Refuse(reason),
    }
}

/// What a load does about the merged block once the intent and the check are known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::family) struct Resolved {
    /// Allocate the working buffer and run the merged block.
    pub(in crate::family) merge: bool,
    pub(in crate::family) source: Source,
    /// Set when the default could not be honoured: the reason to log.
    pub(in crate::family) fallback_reason: Option<String>,
}

/// [`decide`] turned into what the load does: the refusal of an explicit `1` is the error, the
/// fallback of the default is a successful reference run with its reason.
pub(in crate::family) fn resolve(
    intent: Intent,
    check: Result<(), String>,
) -> Result<Resolved, EngError> {
    match decide(intent, check) {
        Decision::Merge => Ok(Resolved {
            merge: true,
            source: if intent == Intent::Explicit {
                Source::Explicit
            } else {
                Source::Default
            },
            fallback_reason: None,
        }),
        Decision::Reference => Ok(Resolved {
            merge: false,
            source: Source::Explicit,
            fallback_reason: None,
        }),
        Decision::Fallback(reason) => Ok(Resolved {
            merge: false,
            source: Source::Fallback,
            fallback_reason: Some(reason),
        }),
        Decision::Refuse(reason) => Err(EngError::BadRequest(reason)),
    }
}
