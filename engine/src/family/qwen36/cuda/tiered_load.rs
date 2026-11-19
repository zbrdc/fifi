//! The load path of the tiered step: which tensors a tiered load uploads, what the bank sizes are
//! checked against, and the host side built over the mapped GGUF.
//!
//! WHAT: [`names_without_experts`] drops the routed expert stacks from the device tensor list (the
//! banks hold the hot experts instead), [`admit_banks`] refuses a bank that does not fit the free
//! device memory through the admission arithmetic of `tiered/budget.rs`, and [`build_tiered`]
//! allocates the banks and builds the [`TieredHost`] over a read-only mapping of the checkpoint.
//! WHY: the 20 GB checkpoint cannot be resident in 6 GiB; everything but the experts still is.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use super::dims::{StepDims, TieredPlan};
use super::step::QwenResidentStep;
use crate::EngError;
use crate::family::device_budget::KvPlan;
use crate::family::mapped_file::MappedFile;
use crate::family::qwen36::settings::SlotRequest;
use crate::family::qwen36::tiered::cpu_pool::{CpuPool, ExpertKernel, MappedRegion, PoolError};
use crate::family::qwen36::tiered::guess::{GuessError, LayerRouter, RouterGuess};
use crate::family::qwen36::tiered::host::TieredHost;
use crate::family::qwen36::tiered::locator::{ExpertLocator, ExpertTensorNames};
use crate::family::qwen36::tiered::{PolicyMode, TieredConfig};
use crate::family::tier_budget::{ExpertSize, VramBudget, plan_vram};
use crate::kernels::cuda::{CudaContext, ExpertBank, ExpertBankGeometry};
use crate::kernels::expert_cpu::ExpertShape;
use crate::kernels::expert_cpu_q8k::DotBackend;
use crate::weights::GgufHeader;

/// The expert kernel of the tier's CPU pool: the per-32 group grid of the device path, not llama.cpp's
/// Q8_K. Gate (b) failed with Q8_K at 80 percent CPU-served picks (p95 0.686 against the bound 0.651) and
/// passed with this grid (0.478). The choice is pinned by the pure test below and, on the device, by the
/// rule-C bounds; [`tier_pool`] is where the load and those tests share it.
fn tier_kernel() -> ExpertKernel {
    ExpertKernel::Q8g(DotBackend::detect())
}

/// The CPU pool of the tier: `threads` workers running [`tier_kernel`] over experts of `shape`. The load and
/// the device rule-C tests both build their pool here, and a source pin keeps `build_tiered` on this call.
pub(super) fn tier_pool(threads: usize, shape: ExpertShape) -> Result<CpuPool, PoolError> {
    CpuPool::with_kernel(threads, shape, tier_kernel())
}

/// The three routed expert stacks of a layer, by the suffix of their GGUF names.
const EXPERT_SUFFIXES: [&str; 3] = [
    ".ffn_gate_exps.weight",
    ".ffn_up_exps.weight",
    ".ffn_down_exps.weight",
];

/// The device tensor names of a tiered load: every name but the routed expert stacks.
pub(super) fn names_without_experts<'n>(names: &[&'n str]) -> Vec<&'n str> {
    names
        .iter()
        .copied()
        .filter(|name| !EXPERT_SUFFIXES.iter().any(|suffix| name.ends_with(suffix)))
        .collect()
}

pub(super) fn geometry_of(dims: &StepDims) -> ExpertBankGeometry {
    ExpertBankGeometry {
        hidden: dims.shape.hidden,
        inter: dims.shape.moe.inter,
        top_k: dims.shape.moe.top_k,
    }
}

/// Refuses a plan whose banks do not fit `free` device bytes (after the weights, the KV cache
/// and the step state are resident), through `plan_vram`: the minimum bank is the plan's own
/// slots per layer, the desktop reserve is the loader's headroom. The message names the numbers.
pub(super) fn admit_banks(
    free: u64,
    geometry: ExpertBankGeometry,
    layers: usize,
    request: SlotRequest,
    experts: usize,
    reserve: u64,
) -> Result<usize, EngError> {
    let minimum = match request {
        SlotRequest::Fit => geometry.top_k,
        SlotRequest::Exactly(count) => count,
    };
    if minimum < geometry.top_k || minimum > experts {
        return Err(EngError::MemoryPlan(format!(
            "tiered: {minimum} slots per layer is outside the model range {}..={experts}",
            geometry.top_k
        )));
    }
    let minimum_bank = ExpertBank::bytes_for(geometry, geometry.top_k)?;
    let bytes_per_slot = minimum_bank / geometry.top_k as u64;
    let budget = VramBudget {
        usable: free,
        non_expert: 0,
        state: 0,
        arena: 0,
        desktop_reserve: reserve,
    };
    // The KV cache is already allocated by the step: it takes no part in this split.
    let kv = KvPlan {
        positions: 0,
        bytes_per_position: 1,
    };
    let expert = ExpertSize {
        bytes: bytes_per_slot,
        layers,
    };
    let capacity = plan_vram(&budget, &kv, &expert, minimum, 0).map_err(|error| {
        EngError::MemoryPlan(format!(
            "tiered: {minimum} slots per layer of {layers} layers do not fit {free} free device bytes with {reserve} reserved: {error:?}"
        ))
    })?;
    Ok(match request {
        SlotRequest::Fit => capacity.bank_slots_per_layer.min(experts),
        SlotRequest::Exactly(count) => count,
    })
}

pub(super) fn build_tiered(
    ctx: &CudaContext,
    (header, gguf): (&GgufHeader, &Path),
    dims: &StepDims,
    plan: &TieredPlan,
    slots: usize,
) -> Result<(Vec<ExpertBank>, TieredHost), EngError> {
    let moe = &dims.shape.moe;
    let layers = dims.shape.kinds.len();
    let shape = ExpertShape {
        hidden: dims.shape.hidden,
        inter: moe.inter,
    };
    let checkpoint = |what: String| EngError::Checkpoint(format!("tiered: {what}"));
    let locator = ExpertLocator::new(
        header,
        &ExpertTensorNames::llama_cpp(),
        layers,
        moe.experts,
        shape,
    )
    .map_err(|error| checkpoint(format!("{error:?}")))?;
    let mapped = MappedFile::open(gguf, header.file_len())
        .map_err(|error| checkpoint(format!("mapping {}: {error}", gguf.display())))?;
    let region: Arc<dyn MappedRegion> = Arc::new(mapped);
    let pool = tier_pool(plan.threads, shape)
        .map_err(|error| checkpoint(format!("cpu pool: {error:?}")))?;
    let config = TieredConfig {
        layers,
        experts: moe.experts,
        top_k: moe.top_k,
        bank_slots_per_layer: slots,
        mode: PolicyMode::Adaptive,
        swaps_per_token: f64::from(plan.swaps_milli) / 1000.0,
        decay: f64::from(plan.decay_milli) / 1000.0,
    };
    let timeout = Duration::from_millis(u64::from(plan.timeout_ms));
    // The guess reads the same mapping the host will own, so build it first.
    let guess = if guess_enabled(std::env::var_os(GUESS_VAR).as_deref()) {
        Some(
            build_router_guess(header, &region, dims)
                .map_err(|error| checkpoint(format!("router guess: {error:?}")))?,
        )
    } else {
        None
    };
    let mut host = TieredHost::new(config, locator, region, pool, timeout)
        .map_err(|error| checkpoint(error.to_string()))?;
    if let Some(guess) = guess {
        host.set_guess(guess, moe.top_k);
    }
    let geometry = geometry_of(dims);
    let banks = (0..layers)
        .map(|_| ExpertBank::new(ctx, geometry, slots))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((banks, host))
}

/// Admits the banks against the free device memory (the weights, the KV cache and the step state
/// are resident), builds the banks and the host and attaches them to `step`: the one tail shared by
/// the serve start and the gate. `source` is the checkpoint header and its path.
pub(super) fn attach_tiered_parts(
    step: &mut QwenResidentStep<'_>,
    ctx: &CudaContext,
    (source, dims): ((&GgufHeader, &Path), &StepDims),
    plan: &TieredPlan,
) -> Result<usize, EngError> {
    let free = u64::try_from(ctx.memory_info()?.free).unwrap_or(u64::MAX);
    let slots = admit_banks(
        free,
        geometry_of(dims),
        dims.shape.kinds.len(),
        plan.slots,
        dims.shape.moe.experts,
        ADMISSION_RESERVE,
    )?;
    let (banks, host) = build_tiered(ctx, source, dims, plan, slots)?;
    step.attach_tiered(banks, host)?;
    if plan.preload {
        step.preload_tiered()?;
    }
    Ok(slots)
}

/// The device bytes kept free beside the banks: the loader's headroom (1 GiB).
const ADMISSION_RESERVE: u64 = 1 << 30;

/// Whether the router-guess expert prefetch runs: `Q36_TIERED_GUESS=1`. The flag exists
/// so the prefetch (new behaviour in the tiered token path) ships dark until it is
/// qualified on a device; anything but `1` keeps today's behaviour.
const GUESS_VAR: &str = "Q36_TIERED_GUESS";
fn guess_enabled(value: Option<&std::ffi::OsStr>) -> bool {
    value == Some(std::ffi::OsStr::new("1"))
}

/// Builds the [`RouterGuess`] of the whole model straight from the checkpoint mapping:
/// every layer's `ffn_gate_inp` router matrix and `post_attention_layernorm` weight are
/// plain F32 in this family's files, so no dequantization is involved. `Err` names the
/// first tensor that is missing or not the expected F32 `[experts, hidden]` shape.
fn build_router_guess(
    header: &GgufHeader,
    region: &Arc<dyn MappedRegion>,
    dims: &StepDims,
) -> Result<RouterGuess, GuessError> {
    let (hidden, experts) = (dims.shape.hidden, dims.shape.moe.experts);
    let eps = crate::family::spec::QWEN3_6_35B_A3B.rms_norm_eps;
    let bytes = region.bytes();
    let read_f32 = |tensor: &crate::weights::GgufTensor, len: usize, what: &str| {
        let range = tensor.abs_range();
        let start = usize::try_from(range.start).map_err(|_| {
            GuessError::BadInput(format!(
                "tiered guess: {what} offset {} overflows usize",
                range.start
            ))
        })?;
        let end = start
            .checked_add(tensor.byte_len() as usize)
            .ok_or_else(|| {
                GuessError::BadInput(format!("tiered guess: {what} length overflows usize"))
            })?;
        let slice = bytes.get(start..end).ok_or_else(|| {
            GuessError::BadInput(format!(
                "tiered guess: {what} range {start}..{end} is outside the file"
            ))
        })?;
        if tensor.ggml_type() != crate::weights::GgufType::F32
            || tensor.element_count() != len as u64
        {
            return Err(GuessError::BadInput(format!(
                "tiered guess: {what} is not F32 of {len} values"
            )));
        }
        let (words, _) = slice.as_chunks::<4>();
        Ok(words
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect::<Vec<f32>>()
            .into())
    };
    let mut routers = Vec::with_capacity(dims.shape.kinds.len());
    for layer in 0..dims.shape.kinds.len() {
        let router = header
            .tensor(&format!("blk.{layer}.ffn_gate_inp.weight"))
            .ok_or_else(|| {
                GuessError::BadInput(format!(
                    "tiered guess: blk.{layer}.ffn_gate_inp.weight is missing"
                ))
            })?;
        let norm = header
            .tensor(&format!("blk.{layer}.post_attention_norm.weight"))
            .ok_or_else(|| {
                GuessError::BadInput(format!(
                    "tiered guess: blk.{layer}.post_attention_norm.weight is missing"
                ))
            })?;
        routers.push(LayerRouter {
            router: read_f32(router, experts * hidden, "the router matrix")?,
            norm_weight: read_f32(norm, hidden, "the post-attention norm weight")?,
        });
    }
    RouterGuess::new(routers, hidden, experts, eps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::qwen36::cuda::common_tests::dims;

    #[test]
    fn the_tier_kernel_is_the_per_32_group_grid_never_q8_k() {
        assert!(
            matches!(tier_kernel(), ExpertKernel::Q8g(_)),
            "{:?}",
            tier_kernel()
        );
        let shape = ExpertShape {
            hidden: 256,
            inter: 256,
        };
        let pool = tier_pool(2, shape).expect("a pool");
        assert!(
            matches!(pool.kernel(), ExpertKernel::Q8g(_)),
            "{:?}",
            pool.kernel()
        );
        assert_eq!(pool.threads(), 2);
    }

    #[test]
    fn a_tiered_load_drops_exactly_the_routed_expert_stacks() {
        let names = [
            "token_embd.weight",
            "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_up_exps.weight",
            "blk.0.ffn_down_exps.weight",
            "blk.0.ffn_gate_shexp.weight",
            "blk.39.ffn_down_exps.weight",
            "output.weight",
        ];
        assert_eq!(
            names_without_experts(&names),
            [
                "token_embd.weight",
                "blk.0.ffn_gate_inp.weight",
                "blk.0.ffn_gate_shexp.weight",
                "output.weight"
            ]
        );
    }

    #[test]
    fn the_bank_geometry_comes_from_the_step_dims() {
        let dims = dims();
        let geometry = geometry_of(&dims);
        assert_eq!(geometry.hidden, dims.shape.hidden);
        assert_eq!(geometry.inter, dims.shape.moe.inter);
        assert_eq!(geometry.top_k, dims.shape.moe.top_k);
    }

    #[test]
    fn auto_slots_fit_whole_layer_sets_above_the_reserve_up_to_the_model_limit() {
        for geometry in [
            ExpertBankGeometry {
                hidden: 2048,
                inter: 512,
                top_k: 8,
            },
            ExpertBankGeometry {
                hidden: 1024,
                inter: 256,
                top_k: 4,
            },
        ] {
            for layers in [12, 40] {
                let minimum = geometry.top_k;
                let experts = 32;
                let reserve = 1 << 30;
                let slot_set = ExpertBank::bytes_for(geometry, minimum).unwrap() / minimum as u64
                    * layers as u64;
                let fit =
                    |free| admit_banks(free, geometry, layers, SlotRequest::Fit, experts, reserve);
                assert!(fit(reserve + slot_set * minimum as u64 - 1).is_err());
                for (sets, extra, want) in [
                    (minimum, 0, minimum),
                    (18, slot_set - 1, 18),
                    (19, 0, 19),
                    (64, 0, experts),
                ] {
                    let free = reserve + slot_set * sets as u64 + extra;
                    assert_eq!(fit(free).unwrap(), want);
                    assert!(
                        ExpertBank::bytes_for(geometry, want).unwrap() * layers as u64
                            <= free - reserve
                    );
                }
                assert_eq!(
                    admit_banks(
                        reserve + 28 * slot_set,
                        geometry,
                        layers,
                        SlotRequest::Exactly(16),
                        experts,
                        reserve
                    )
                    .unwrap(),
                    16
                );
                assert!(
                    admit_banks(
                        u64::MAX,
                        geometry,
                        layers,
                        SlotRequest::Exactly(experts + 1),
                        experts,
                        reserve
                    )
                    .is_err()
                );
            }
        }
    }

    #[test]
    fn a_plan_that_does_not_fit_is_refused_naming_the_budget() {
        let geometry = geometry_of(&dims());
        let top_k = geometry.top_k as u64;
        let slot_set = ExpertBank::bytes_for(geometry, geometry.top_k).unwrap() / top_k * 40;
        let plan = SlotRequest::Exactly;
        let reserve = 1 << 30;
        // 40 slots per layer need 40 sets; a card with room for 39 plus the reserve refuses.
        let free = reserve + 39 * slot_set;
        let refusal = admit_banks(free, geometry, 40, plan(40), 256, reserve);
        match refusal {
            Err(EngError::MemoryPlan(message)) => {
                assert!(message.contains("40 slots per layer"), "{message}");
                assert!(message.contains(&free.to_string()), "{message}");
                assert!(message.contains("minimum expert bank"), "{message}");
            }
            other => panic!("39 slots of room must refuse 40: {other:?}"),
        }
        assert!(admit_banks(free, geometry, 40, plan(39), 256, reserve).is_ok());
        assert!(
            admit_banks(
                reserve + 40 * slot_set,
                geometry,
                40,
                plan(40),
                256,
                reserve
            )
            .is_ok()
        );
    }

    #[test]
    fn the_guess_flag_is_exactly_one() {
        use super::{GUESS_VAR, guess_enabled};
        use std::ffi::OsStr;
        assert_eq!(GUESS_VAR, "Q36_TIERED_GUESS");
        assert!(guess_enabled(Some(OsStr::new("1"))));
        assert!(!guess_enabled(Some(OsStr::new("0"))));
        assert!(!guess_enabled(Some(OsStr::new("true"))));
        assert!(!guess_enabled(None));
    }
}
