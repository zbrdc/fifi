//! The routed host walk used by both real dispatch and the request observer.

use super::bad;

use crate::{
    EngError,
    family::flint::cuda_ffn::plan::{FfnBlock, Grouping},
    kernels::cuda::{ArenaTensor, plan_arena, prefill::row_ops::RowShape},
};

pub(super) const MAX_ROWS: usize = 8;
const _: () = assert!(MAX_ROWS == crate::family::flint::settings::PREFILL_MAX_ROWS);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(super) enum Slot {
    Xh,
    Z,
    Out,
    Gate,
    Up,
    SigmoidRaw,
    Sigmoid,
    SiluRaw,
    Silu,
    ActivatedRaw,
    Activated,
    Down,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Loan {
    pub(super) slot: Slot,
    pub(super) offset: usize,
    pub(super) bytes: usize,
}

/// The same disjoint plan is charged, allocated, and observed on the host.
pub(super) struct Layout {
    pairs: usize,
    hidden: usize,
    intermediate: usize,
    pub(super) tensors: [ArenaTensor; 12],
}

impl Layout {
    pub(super) fn new(rows: usize, hidden: usize, intermediate: usize) -> Result<Self, EngError> {
        if !(1..=MAX_ROWS).contains(&rows) || hidden == 0 || intermediate == 0 {
            return Err(bad("routed workspace dimensions are invalid"));
        }
        let pairs = rows
            .checked_mul(super::TOP_K)
            .filter(|&pairs| pairs <= 48)
            .ok_or_else(|| bad("routed workspace pair count overflows"))?;
        let bytes = |columns| {
            RowShape {
                rows: pairs,
                columns,
            }
            .bytes()
            .ok_or_else(|| bad("routed workspace bytes overflow"))
        };
        let tensor = |bytes| ArenaTensor {
            bytes,
            first: 0,
            last: 0,
        };
        let mut tensors = [tensor(bytes(intermediate)?); 12];
        tensors[..3].fill(tensor(bytes(hidden.max(intermediate))?));
        tensors[Slot::Down as usize] = tensor(bytes(hidden)?);
        Ok(Self {
            pairs,
            hidden,
            intermediate,
            tensors,
        })
    }

    pub(super) fn bytes(&self) -> Result<u64, EngError> {
        u64::try_from(plan_arena(&self.tensors)?.total)
            .map_err(|_| bad("routed workspace charge overflows"))
    }

    pub(super) fn loan(&self, slot: Slot, first: usize, shape: RowShape) -> Result<Loan, EngError> {
        let valid_width = match slot {
            Slot::Xh | Slot::Z | Slot::Out => {
                shape.columns == self.hidden || shape.columns == self.intermediate
            }
            Slot::Down => shape.columns == self.hidden,
            _ => shape.columns == self.intermediate,
        };
        if !valid_width
            || shape.rows == 0
            || first
                .checked_add(shape.rows)
                .is_none_or(|end| end > self.pairs)
        {
            return Err(bad("routed workspace loan exceeds its logical slot"));
        }
        let offset = first
            .checked_mul(shape.columns)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| bad("routed workspace offset overflows"))?;
        let bytes = shape
            .bytes()
            .ok_or_else(|| bad("routed workspace loan overflows"))?;
        if offset
            .checked_add(bytes)
            .is_none_or(|end| end > self.tensors[slot as usize].bytes)
        {
            return Err(bad("routed workspace loan exceeds its physical slot"));
        }
        Ok(Loan {
            slot,
            offset,
            bytes,
        })
    }
}

pub(super) struct Expert<M> {
    pub(super) matrices: [M; 3],
    pub(super) hidden: usize,
    pub(super) intermediate: usize,
}

#[derive(Clone, Copy)]
pub(super) struct Span<T> {
    pub(super) tensor: T,
    pub(super) offset: usize,
    pub(super) bytes: usize,
}

pub(super) struct Projection<M, T> {
    pub(super) weight: M,
    pub(super) x: T,
}

/// Only the routed FFN effects; the concrete Frame still owns completion and health.
pub(super) trait Effects {
    type Tensor: Copy;
    type Matrix: Copy;

    fn packed_pointwise(&self) -> bool {
        false
    }

    fn allocate(&mut self, shape: RowShape) -> Result<Self::Tensor, EngError>;
    fn view(&mut self, span: Span<Self::Tensor>) -> Result<Self::Tensor, EngError>;
    fn stage(
        &mut self,
        slot: Slot,
        first: usize,
        shape: RowShape,
    ) -> Result<Self::Tensor, EngError>;
    fn grouped(&mut self, calls: &[Projection<Self::Matrix, Self::Tensor>])
    -> Result<(), EngError>;
    fn indices(&mut self, values: &[i32]) -> Result<Self::Tensor, EngError>;
    fn expert(&mut self, id: usize) -> Result<Expert<Self::Matrix>, EngError>;
    fn copy(&mut self, source: Span<Self::Tensor>, out: Span<Self::Tensor>)
    -> Result<(), EngError>;
    fn gather(
        &mut self,
        x: Self::Tensor,
        indices: Self::Tensor,
        out: Self::Tensor,
        shape: RowShape,
    ) -> Result<(), EngError>;
    fn project(&mut self, weight: Self::Matrix, x: Self::Tensor) -> Result<(), EngError>;
    fn publish_projection(&mut self, out: Self::Tensor, shape: RowShape) -> Result<(), EngError>;
    fn sigmoid(
        &mut self,
        x: Self::Tensor,
        out: Self::Tensor,
        shape: RowShape,
    ) -> Result<(), EngError>;
    fn multiply(
        &mut self,
        x: [Self::Tensor; 2],
        out: Self::Tensor,
        shape: RowShape,
    ) -> Result<(), EngError>;
    fn publish(
        &mut self,
        x: Self::Tensor,
        out: Self::Tensor,
        shape: RowShape,
    ) -> Result<(), EngError>;
}

fn project<E: Effects>(
    effects: &mut E,
    weight: E::Matrix,
    x: E::Tensor,
    shape: RowShape,
) -> Result<E::Tensor, EngError> {
    effects.project(weight, x)?;
    // Keep the legacy order: enqueue the projection before allocating publication.
    let out = effects.allocate(shape)?;
    effects.publish_projection(out, shape)?;
    Ok(out)
}

fn multiply<E: Effects>(
    effects: &mut E,
    x: [E::Tensor; 2],
    shape: RowShape,
) -> Result<E::Tensor, EngError> {
    let raw = effects.allocate(shape)?;
    effects.multiply(x, raw, shape)?;
    let out = effects.allocate(shape)?;
    effects.publish(raw, out, shape)?;
    Ok(out)
}

fn expert<E: Effects>(
    effects: &mut E,
    id: usize,
    x: E::Tensor,
    input: RowShape,
) -> Result<E::Tensor, EngError> {
    let Expert {
        matrices: [gate, up, down],
        hidden,
        intermediate,
    } = effects.expert(id)?;
    if hidden != input.columns {
        return Err(bad("routed expert hidden width disagrees"));
    }
    let rows = input.rows;
    let shape = RowShape {
        rows,
        columns: intermediate,
    };
    let gate = project(effects, gate, x, shape)?;
    let up = project(effects, up, x, shape)?;
    let raw = effects.allocate(shape)?;
    effects.sigmoid(gate, raw, shape)?;
    let sigmoid = effects.allocate(shape)?;
    effects.publish(raw, sigmoid, shape)?;
    let silu = multiply(effects, [gate, sigmoid], shape)?;
    let activated = multiply(effects, [silu, up], shape)?;
    project(
        effects,
        down,
        activated,
        RowShape {
            rows,
            columns: hidden,
        },
    )
}

struct Group<E: Effects> {
    first: usize,
    rows: usize,
    expert: Expert<E::Matrix>,
    input: E::Tensor,
}

fn project_groups<E: Effects>(
    effects: &mut E,
    groups: &[Group<E>],
    role: usize,
    slot: Slot,
    packed_rows: Option<usize>,
) -> Result<(), EngError> {
    let calls: Vec<_> = groups
        .iter()
        .map(|group| Projection {
            weight: group.expert.matrices[role],
            x: group.input,
        })
        .collect();
    effects.grouped(&calls)?;
    let count = if packed_rows.is_some() {
        1
    } else {
        groups.len()
    };
    for group in groups.iter().take(count) {
        let shape = RowShape {
            rows: packed_rows.unwrap_or(group.rows),
            columns: if role == 2 {
                group.expert.hidden
            } else {
                group.expert.intermediate
            },
        };
        let raw = effects.stage(Slot::Out, group.first, shape)?;
        let out = effects.stage(slot, group.first, shape)?;
        effects.publish(raw, out, shape)?;
    }
    Ok(())
}

fn grouped_inputs<E: Effects>(
    effects: &mut E,
    block: FfnBlock,
    ids: &[i32],
    grouping: &Grouping,
    x: E::Tensor,
) -> Result<Vec<Group<E>>, EngError> {
    let pairs = RowShape {
        rows: block.pairs,
        columns: block.hidden,
    };
    let gathered = if block.rows == 1 {
        x
    } else {
        let indices = effects.indices(&grouping.tokens)?;
        let out = effects.allocate(pairs)?;
        effects.gather(x, indices, out, pairs)?;
        out
    };
    let row_bytes = block.hidden * 4;
    let mut groups = Vec::new();
    let mut first = 0;
    while first < block.pairs {
        let id = ids[if block.rows == 1 {
            first
        } else {
            grouping.order[first]
        }];
        let mut end = first + 1;
        if block.rows != 1 {
            while end < block.pairs && ids[grouping.order[end]] == id {
                end += 1;
            }
        }
        let expert = effects.expert(id as usize)?;
        if expert.hidden != block.hidden
            || groups
                .first()
                .is_some_and(|group: &Group<E>| group.expert.intermediate != expert.intermediate)
        {
            return Err(bad("grouped routed expert dimensions disagree"));
        }
        let input = effects.view(Span {
            tensor: gathered,
            offset: if block.rows == 1 {
                0
            } else {
                first * row_bytes
            },
            bytes: (end - first) * row_bytes,
        })?;
        groups.push(Group {
            first,
            rows: end - first,
            expert,
            input,
        });
        first = end;
    }
    Ok(groups)
}

pub(super) fn run<E: Effects>(
    effects: &mut E,
    block: FfnBlock,
    ids: &[i32],
    grouping: &Grouping,
    x: E::Tensor,
) -> Result<E::Tensor, EngError> {
    // The grouped SIMT kernels retain the existing dispatcher for larger row blocks.
    // Future-only: row_shape currently caps production at MAX_ROWS; legacy is a host-test control.
    // The equality pin above prevents a higher serving cap silently activating this unqualified path.
    if block.rows > MAX_ROWS {
        return legacy(effects, block, ids, grouping, x);
    }
    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    let _scope = crate::kernels::cuda::CudaContext::test_exl3_routed_scope();
    let mut groups = grouped_inputs(effects, block, ids, grouping, x)?;
    let packed_rows = effects.packed_pointwise().then_some(block.pairs);
    project_groups(effects, &groups, 0, Slot::Gate, packed_rows)?;
    project_groups(effects, &groups, 1, Slot::Up, packed_rows)?;
    let count = if packed_rows.is_some() {
        1
    } else {
        groups.len()
    };
    for group in groups.iter().take(count) {
        let shape = RowShape {
            rows: packed_rows.unwrap_or(group.rows),
            columns: group.expert.intermediate,
        };
        let gate = effects.stage(Slot::Gate, group.first, shape)?;
        let raw = effects.stage(Slot::SigmoidRaw, group.first, shape)?;
        effects.sigmoid(gate, raw, shape)?;
        let sigmoid = effects.stage(Slot::Sigmoid, group.first, shape)?;
        effects.publish(raw, sigmoid, shape)?;
        for (lhs, rhs, raw, out) in [
            (Slot::Gate, Slot::Sigmoid, Slot::SiluRaw, Slot::Silu),
            (Slot::Silu, Slot::Up, Slot::ActivatedRaw, Slot::Activated),
        ] {
            let lhs = effects.stage(lhs, group.first, shape)?;
            let rhs = effects.stage(rhs, group.first, shape)?;
            let raw = effects.stage(raw, group.first, shape)?;
            effects.multiply([lhs, rhs], raw, shape)?;
            let out = effects.stage(out, group.first, shape)?;
            effects.publish(raw, out, shape)?;
        }
    }
    for group in &mut groups {
        let shape = RowShape {
            rows: group.rows,
            columns: group.expert.intermediate,
        };
        group.input = effects.stage(Slot::Activated, group.first, shape)?;
    }
    project_groups(effects, &groups, 2, Slot::Down, packed_rows)?;
    let pairs = RowShape {
        rows: block.pairs,
        columns: block.hidden,
    };
    let down = effects.stage(Slot::Down, 0, pairs)?;
    let out = effects.allocate(pairs)?;
    if block.rows == 1 {
        let bytes = pairs
            .bytes()
            .ok_or_else(|| bad("routed output bytes overflow"))?;
        effects.copy(
            Span {
                tensor: down,
                offset: 0,
                bytes,
            },
            Span {
                tensor: out,
                offset: 0,
                bytes,
            },
        )?;
    } else {
        let inverse = effects.indices(&grouping.inverse)?;
        effects.gather(down, inverse, out, pairs)?;
    }
    Ok(out)
}

pub(super) fn legacy<E: Effects>(
    effects: &mut E,
    block: FfnBlock,
    ids: &[i32],
    grouping: &Grouping,
    x: E::Tensor,
) -> Result<E::Tensor, EngError> {
    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    let _scope = crate::kernels::cuda::CudaContext::test_exl3_routed_scope();
    let pairs = RowShape {
        rows: block.pairs,
        columns: block.hidden,
    };
    let down = effects.allocate(pairs)?;
    let row_bytes = block.hidden * 4;
    if block.rows == 1 {
        for (rank, &id) in ids.iter().enumerate() {
            let output = expert(
                effects,
                id as usize,
                x,
                RowShape {
                    rows: 1,
                    columns: block.hidden,
                },
            )?;
            effects.copy(
                Span {
                    tensor: output,
                    offset: 0,
                    bytes: row_bytes,
                },
                Span {
                    tensor: down,
                    offset: rank * row_bytes,
                    bytes: row_bytes,
                },
            )?;
        }
        return Ok(down);
    }
    let tokens = effects.indices(&grouping.tokens)?;
    let gathered = effects.allocate(pairs)?;
    effects.gather(x, tokens, gathered, pairs)?;
    let mut start = 0;
    while start < block.pairs {
        let id = ids[grouping.order[start]];
        let mut end = start + 1;
        while end < block.pairs && ids[grouping.order[end]] == id {
            end += 1;
        }
        let rows = end - start;
        let input = effects.allocate(RowShape {
            rows,
            columns: block.hidden,
        })?;
        effects.copy(
            Span {
                tensor: gathered,
                offset: start * row_bytes,
                bytes: rows * row_bytes,
            },
            Span {
                tensor: input,
                offset: 0,
                bytes: rows * row_bytes,
            },
        )?;
        let output = expert(
            effects,
            id as usize,
            input,
            RowShape {
                rows,
                columns: block.hidden,
            },
        )?;
        effects.copy(
            Span {
                tensor: output,
                offset: 0,
                bytes: rows * row_bytes,
            },
            Span {
                tensor: down,
                offset: start * row_bytes,
                bytes: rows * row_bytes,
            },
        )?;
        start = end;
    }
    let inverse = effects.indices(&grouping.inverse)?;
    let out = effects.allocate(pairs)?;
    effects.gather(down, inverse, out, pairs)?;
    Ok(out)
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn grouped_workspace_charge_matches_its_disjoint_allocation_plan() {
        for (rows, expected) in [(1, 294_912), (8, 2_359_296)] {
            let layout = Layout::new(rows, 2048, 512).unwrap();
            let plan = plan_arena(&layout.tensors).unwrap();
            assert_eq!(layout.bytes().unwrap(), expected);
            assert_eq!(plan.total as u64, expected);
            for i in 0..12 {
                for j in i + 1..12 {
                    assert!(
                        plan.offsets[i] + layout.tensors[i].bytes <= plan.offsets[j]
                            || plan.offsets[j] + layout.tensors[j].bytes <= plan.offsets[i]
                    );
                }
            }
        }
        let layout = Layout::new(8, 2048, 512).unwrap();
        assert_eq!(
            layout.tensors.each_ref().map(|tensor| tensor.bytes),
            [
                393_216, 393_216, 393_216, 98_304, 98_304, 98_304, 98_304, 98_304, 98_304, 98_304,
                98_304, 393_216,
            ]
        );
        let shape = RowShape {
            rows: 1,
            columns: 512,
        };
        let last = layout.loan(Slot::Gate, 47, shape).unwrap();
        assert_eq!((last.offset, last.bytes), (96_256, 2048));
        assert!(layout.loan(Slot::Gate, 48, shape).is_err());
        assert!(layout.loan(Slot::Gate, usize::MAX, shape).is_err());
        assert!(
            layout
                .loan(
                    Slot::Gate,
                    0,
                    RowShape {
                        rows: 1,
                        columns: 2048
                    }
                )
                .is_err()
        );
        assert!(Layout::new(0, 2048, 512).is_err());
        assert!(Layout::new(9, 2048, 512).is_err());
        assert!(Layout::new(8, usize::MAX, 512).is_err());
    }
}

#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod tests;
