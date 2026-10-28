use super::{Result, WIDTH, bad};
use crate::EngError;
use crate::family::flint::{kv_plan::KvPlan, settings::PREFILL_MAX_ROWS};
use crate::kernels::cuda::{
    ArenaTensor, CudaContext, DeviceArena, DeviceRange, plan_arena,
    prefill::attn_prefill::{
        CPF_ATTN_MAX_SPLITS, CPF_ATTN_PART_CAP, CPF_ATTN_SM_COUNT, CpfAttnGeometry, CpfAttnPlan,
        cpf_attn_launch_plan, cpf_attn_part_bytes,
    },
    xf_bytes,
};

pub(super) struct HotScratch(DeviceArena);

impl HotScratch {
    pub(super) fn new(ctx: &CudaContext, plan: &KvPlan) -> Result<Self> {
        Ok(Self(ctx.arena(&tensors(plan)?)?))
    }

    pub(super) fn bytes(plan: &KvPlan) -> Result<u64> {
        u64::try_from(plan_arena(&tensors(plan)?)?.total)
            .map_err(|_| EngError::MemoryPlan("Flint hot scratch byte charge overflows".into()))
    }

    pub(super) fn geometry() -> CpfAttnGeometry {
        CpfAttnGeometry {
            m: 1,
            n_head: 32,
            n_kv_head: 4,
            head_dim: 128,
        }
    }

    pub(super) fn norm(&mut self, rows: usize) -> Result<DeviceRange<'_>> {
        let bytes = xf_bytes(rows, WIDTH).ok_or_else(|| bad("EXL3 norm scratch"))?;
        let slot = self.0.tensor(0)?;
        if bytes > slot.bytes() {
            return Err(EngError::MemoryPlan(
                "EXL3 norm exceeds persistent scratch".into(),
            ));
        }
        slot.buffer().range(slot.offset(), bytes)
    }

    pub(super) fn part(&mut self) -> Result<DeviceRange<'_>> {
        self.0.tensor(1)
    }

    pub(super) fn router(&mut self) -> Result<DeviceRange<'_>> {
        self.0.tensor(2)
    }

    #[cfg(test)]
    pub(super) fn dirty(&mut self, stream: &crate::kernels::cuda::DeviceStream) -> Result<()> {
        self.0
            .buffer()
            .write_at(stream, 0, &vec![0xff; self.0.bytes()])
    }
}

fn tensors(plan: &KvPlan) -> Result<[ArenaTensor; 3]> {
    let norm = (1..=plan.context_rows().min(PREFILL_MAX_ROWS)).try_fold(0, |max, rows| {
        xf_bytes(rows, WIDTH)
            .map(|bytes| max.max(bytes))
            .ok_or_else(|| bad("EXL3 norm scratch"))
    })?;
    let geometry = HotScratch::geometry();
    let least = cpf_attn_launch_plan(geometry, 1, 0, CPF_ATTN_SM_COUNT)?;
    let per_split = cpf_attn_part_bytes(
        geometry,
        CpfAttnPlan {
            max_splits: 1,
            ..least
        },
    )?;
    // A raised launch count must be fully active; for m=1 its key count is <= context.
    let active_bound = plan.context_rows().div_ceil(least.min_keys);
    let split_cap = CPF_ATTN_MAX_SPLITS.min(CPF_ATTN_PART_CAP / per_split);
    let part = cpf_attn_part_bytes(
        geometry,
        CpfAttnPlan {
            max_splits: least.max_splits.max(active_bound.min(split_cap)),
            ..least
        },
    )?;
    let router = 256usize
        .checked_mul(WIDTH)
        .and_then(|cells| cells.checked_mul(4))
        .ok_or_else(|| bad("EXL3 router scratch bytes overflow"))?;
    // All three slots are disjoint, even though their stream use is sequential.
    Ok([norm, part, router].map(|bytes| ArenaTensor {
        bytes,
        first: 0,
        last: 0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hot_scratch_capacity_covers_every_admitted_query() {
        for context in [1, 8, 513, 576, 4096, 16384, 262_144] {
            let plan = KvPlan::new(context, PREFILL_MAX_ROWS, true).unwrap();
            let slots = tensors(&plan).unwrap();
            for rows in 1..=context.min(PREFILL_MAX_ROWS) {
                assert!(xf_bytes(rows, WIDTH).unwrap() <= slots[0].bytes);
            }
            for position in 1..=context {
                for window in [0, 513] {
                    let launch = cpf_attn_launch_plan(
                        HotScratch::geometry(),
                        position,
                        window,
                        CPF_ATTN_SM_COUNT,
                    )
                    .unwrap();
                    assert!(
                        cpf_attn_part_bytes(HotScratch::geometry(), launch).unwrap()
                            <= slots[1].bytes,
                        "context={context} position={position} window={window}"
                    );
                }
            }
            assert_eq!(slots[2].bytes, 256 * WIDTH * 4);
        }
    }

    #[test]
    fn hot_scratch_charge_is_the_exact_disjoint_arena_plan() {
        for (context, part_bytes, total_bytes) in [
            (1, 5_457_920, 8_603_664),
            (576, 5_457_920, 8_603_664),
            (262_144, 17_039_360, 20_185_104),
        ] {
            let kv = KvPlan::new(context, PREFILL_MAX_ROWS, true).unwrap();
            let slots = tensors(&kv).unwrap();
            let plan = plan_arena(&slots).unwrap();
            // Independent ABI-sized pins also reject a consistently oversized plan.
            assert_eq!(
                slots.each_ref().map(|slot| slot.bytes),
                [1_048_592, part_bytes, 2_097_152]
            );
            assert_eq!(plan.total, total_bytes);
            assert_eq!(HotScratch::bytes(&kv).unwrap(), plan.total as u64);
            println!(
                "hot scratch context={context} slots={:?} offsets={:?} charged_bytes={}",
                slots.each_ref().map(|slot| slot.bytes),
                plan.offsets,
                plan.total
            );
            for i in 0..slots.len() {
                for j in i + 1..slots.len() {
                    assert!(
                        plan.offsets[i] + slots[i].bytes <= plan.offsets[j]
                            || plan.offsets[j] + slots[j].bytes <= plan.offsets[i]
                    );
                }
            }
        }
    }
}
