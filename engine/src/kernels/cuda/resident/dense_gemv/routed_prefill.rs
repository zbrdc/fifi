//! Device-only routed stage for the resident Inkling Mini stack geometry.
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceRange, DeviceStream,
        context::{Claim, require_apart, require_fit},
        native::resident::dense_gemv::moe::launch_routed_prefill,
    },
};

/// Contiguous row-major operands. `ids` and `weights` have eight i32/f32 slots per token.
/// `status` is a caller-cleared device i32; `scratch` is reusable in stream order.
#[derive(Clone, Copy)]
pub(crate) struct DeviceRoutedPrefill<'a> {
    pub rows: usize,
    pub x: DeviceRange<'a>,
    pub ids: DeviceRange<'a>,
    pub weights: DeviceRange<'a>,
    pub out: DeviceRange<'a>,
    pub scratch: DeviceRange<'a>,
    pub status: DeviceRange<'a>,
}

pub(crate) fn routed_prefill_scratch_bytes(rows: usize) -> Option<usize> {
    if !(1..=4096).contains(&rows) {
        return None;
    }
    let slots = rows * 8;
    Some(4 * (130 + slots * 2 + 3 * (slots.div_ceil(16) + 127) + slots * (2304 + 1152)))
}
impl CudaContext {
    pub(crate) fn routed_prefill(
        &self,
        stream: &DeviceStream,
        r: DeviceRoutedPrefill<'_>,
        stacks: [DeviceRange<'_>; 2],
        scales: [DeviceRange<'_>; 2],
        kinds: [u32; 2],
        caps: [f32; 2],
    ) -> Result<(), EngError> {
        let bad = || EngError::BadRequest("unsupported CUDA routed prefill geometry".into());
        let scratch = routed_prefill_scratch_bytes(r.rows).ok_or_else(bad)?;
        let gb = match kinds[0] {
            16 => 66,
            17 => 74,
            19 => 50,
            22 => 82,
            _ => return Err(bad()),
        };
        let db = match kinds[1] {
            40 => 36,
            42 => 18,
            _ => return Err(bad()),
        };
        if !caps.iter().all(|v| v.is_finite() && *v > 0.) {
            return Err(bad());
        }
        let ranges = [
            stacks[0], stacks[1], scales[0], scales[1], r.x, r.ids, r.weights, r.out, r.scratch,
            r.status,
        ];
        let sizes = [
            128 * 2304 * 9 * gb,
            128 * 2304 * 18 * db,
            512,
            512,
            r.rows * 2304 * 4,
            r.rows * 32,
            r.rows * 32,
            r.rows * 2304 * 4,
            scratch,
            4,
        ];
        for (i, range) in ranges.iter().enumerate() {
            require_fit("routed operand", range, sizes[i], if i < 2 { 2 } else { 4 })?;
            if i >= 7 {
                for j in 0..i {
                    require_apart(
                        &Claim::of("output", range, sizes[i]),
                        &Claim::of("operand", &ranges[j], sizes[j]),
                    )?;
                }
            }
        }
        let [a, b, c, d, e, f, g, h, i, j] = ranges.map(|r| self.raw_range(&r));
        let raw = [a?, b?, c?, d?, e?, f?, g?, h?, i?, j?];
        self.launch_on(stream, |s| {
            launch_routed_prefill(s, raw, r.rows, kinds, caps)
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn routed_scratch_spans_metadata_and_both_live_matrices() {
        assert_eq!(routed_prefill_scratch_bytes(0), None);
        assert_eq!(routed_prefill_scratch_bytes(4097), None);
        for rows in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 513, 4096] {
            let s: usize = 8 * rows;
            let tiles = s.div_ceil(16) + 127;
            assert_eq!(
                routed_prefill_scratch_bytes(rows),
                Some(4 * (129 + s + s + 1) + 12 * tiles + s * (2304 + 1152) * 4)
            );
        }
    }
}
