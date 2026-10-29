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
use crate::family::qwen36::tiered::budget::{ExpertSize, KvPlan, VramBudget, plan_vram};
use crate::family::qwen36::tiered::cpu_pool::{CpuPool, ExpertKernel, MappedRegion, PoolError};
use crate::family::qwen36::tiered::host::TieredHost;
use crate::family::qwen36::tiered::locator::{ExpertLocator, ExpertTensorNames};
use crate::family::qwen36::tiered::mapped_file::MappedFile;
use crate::family::qwen36::tiered::{PolicyMode, TieredConfig};
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

/// The bank geometry of the step dims.
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
    plan: &TieredPlan,
    reserve: u64,
) -> Result<(), EngError> {
    let top_k = u64::try_from(geometry.top_k).unwrap_or(u64::MAX);
    let slice = ExpertBank::bytes_for(geometry, geometry.top_k)? / (3 * top_k);
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
        bytes: slice * 3,
        layers,
    };
    plan_vram(&budget, &kv, &expert, plan.slots_per_layer, 0).map(drop).map_err(|error| {
        EngError::MemoryPlan(format!(
            "tiered: {} slots per layer of {layers} layers do not fit {free} free device bytes with {reserve} reserved: {error:?}",
            plan.slots_per_layer
        ))
    })
}

/// Allocates the banks and builds the host over a read-only mapping of the checkpoint `gguf`.
pub(super) fn build_tiered(
    ctx: &CudaContext,
    (header, gguf): (&GgufHeader, &Path),
    dims: &StepDims,
    plan: &TieredPlan,
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
        bank_slots_per_layer: plan.slots_per_layer,
        mode: PolicyMode::Adaptive,
        swaps_per_token: f64::from(plan.swaps_milli) / 1000.0,
        decay: f64::from(plan.decay_milli) / 1000.0,
    };
    let timeout = Duration::from_millis(u64::from(plan.timeout_ms));
    let host = TieredHost::new(config, locator, region, pool, timeout)
        .map_err(|error| checkpoint(error.to_string()))?;
    let geometry = geometry_of(dims);
    let banks = (0..layers)
        .map(|_| ExpertBank::new(ctx, geometry, plan.slots_per_layer))
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
) -> Result<(), EngError> {
    let free = u64::try_from(ctx.memory_info()?.free).unwrap_or(u64::MAX);
    admit_banks(
        free,
        geometry_of(dims),
        dims.shape.kinds.len(),
        plan,
        ADMISSION_RESERVE,
    )?;
    let (banks, host) = build_tiered(ctx, source, dims, plan)?;
    step.attach_tiered(banks, host)?;
    if plan.preload {
        step.preload_tiered()?;
    }
    Ok(())
}

/// The device bytes kept free beside the banks: the loader's headroom (1 GiB).
const ADMISSION_RESERVE: u64 = 1 << 30;

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
    fn a_plan_that_does_not_fit_is_refused_naming_the_budget() {
        let geometry = geometry_of(&dims());
        let top_k = geometry.top_k as u64;
        let slot_set = ExpertBank::bytes_for(geometry, geometry.top_k).unwrap() / top_k * 40;
        let plan = |slots| TieredPlan::new(slots, 6);
        let reserve = 1 << 30;
        // 40 slots per layer need 40 sets; a card with room for 39 plus the reserve refuses.
        let free = reserve + 39 * slot_set;
        let refusal = admit_banks(free, geometry, 40, &plan(40), reserve);
        match refusal {
            Err(EngError::MemoryPlan(message)) => {
                assert!(message.contains("40 slots per layer"), "{message}");
                assert!(message.contains(&free.to_string()), "{message}");
                assert!(message.contains("minimum expert bank"), "{message}");
            }
            other => panic!("39 slots of room must refuse 40: {other:?}"),
        }
        assert!(admit_banks(free, geometry, 40, &plan(39), reserve).is_ok());
        assert!(admit_banks(reserve + 40 * slot_set, geometry, 40, &plan(40), reserve).is_ok());
    }
}
