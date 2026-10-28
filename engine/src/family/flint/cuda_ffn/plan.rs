//! Checked block geometry and stable expert-major routing metadata.

use crate::EngError;

pub(super) const TOP_K: usize = 6;

#[derive(Clone, Copy, Debug)]
pub(super) enum FfnShape {
    Dense {
        hidden: usize,
        intermediate: usize,
    },
    Moe {
        hidden: usize,
        intermediate: usize,
        shared_intermediate: usize,
        experts: usize,
    },
}

#[derive(Clone, Copy, Debug)]
pub(super) struct FfnBlock {
    pub(super) rows: usize,
    pub(super) hidden: usize,
    pub(super) experts: usize,
    pub(super) pairs: usize,
}

fn bad(message: &str) -> EngError {
    EngError::BadRequest(format!("Flint CUDA FFN: {message}"))
}

fn width(value: usize) -> Result<(), EngError> {
    if !(32..=32768).contains(&value) || !value.is_multiple_of(32) {
        return Err(bad("matrix widths must be multiples of 32 in 32..32768"));
    }
    Ok(())
}

fn values(rows: usize, columns: usize) -> Result<usize, EngError> {
    rows.checked_mul(columns)
        .filter(|&count| count <= (1usize << 32))
        .ok_or_else(|| bad("block element count exceeds the shared row geometry"))
}

impl FfnBlock {
    pub(super) fn new(rows: usize, shape: FfnShape) -> Result<Self, EngError> {
        if !(1..=4096).contains(&rows) {
            return Err(bad("block rows must be in 1..4096"));
        }
        let (hidden, intermediate, shared_intermediate, experts) = match shape {
            FfnShape::Dense {
                hidden,
                intermediate,
            } => (hidden, intermediate, 0, 0),
            FfnShape::Moe {
                hidden,
                intermediate,
                shared_intermediate,
                experts,
            } => {
                if !(TOP_K..=256).contains(&experts) {
                    return Err(bad("expert count must be in 6..256"));
                }
                width(shared_intermediate)?;
                (hidden, intermediate, shared_intermediate, experts)
            }
        };
        width(hidden)?;
        width(intermediate)?;
        let pairs = rows * if experts == 0 { 1 } else { TOP_K };
        let input_values = values(rows, hidden)?;
        values(pairs, hidden)?;
        values(pairs, intermediate)?;
        if experts != 0 {
            if input_values > (1 << 22) {
                return Err(bad("shared row add exceeds its admitted element count"));
            }
            values(rows, shared_intermediate)?;
        }
        Ok(Self {
            rows,
            hidden,
            experts,
            pairs,
        })
    }
}

#[derive(Debug)]
pub(super) struct Grouping {
    pub(super) order: Vec<usize>,
    pub(super) tokens: Vec<i32>,
    pub(super) inverse: Vec<i32>,
}

impl Grouping {
    pub(super) fn new(ids: &[i32], rows: usize, experts: usize) -> Result<Self, EngError> {
        if !(1..=4096).contains(&rows) || !(TOP_K..=256).contains(&experts) {
            return Err(bad("invalid routing metadata geometry"));
        }
        if ids.len() != rows * TOP_K {
            return Err(bad("routing IDs do not cover the complete block"));
        }
        if ids.iter().any(|&id| id < 0 || id as usize >= experts) {
            return Err(bad("routing expert ID lies outside the bank"));
        }
        for token in ids.as_chunks::<TOP_K>().0 {
            for rank in 0..TOP_K {
                if token[..rank].contains(&token[rank]) {
                    return Err(bad("routing selected the same expert twice"));
                }
            }
        }
        let mut order: Vec<_> = (0..ids.len()).collect();
        order.sort_by_key(|&pair| (ids[pair], pair));
        let mut inverse = vec![0; ids.len()];
        let mut tokens = Vec::with_capacity(ids.len());
        for (sorted, &original) in order.iter().enumerate() {
            inverse[original] = sorted as i32;
            tokens.push((original / TOP_K) as i32);
        }
        Ok(Self {
            order,
            tokens,
            inverse,
        })
    }
}
