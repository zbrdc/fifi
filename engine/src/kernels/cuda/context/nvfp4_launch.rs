//! Explicit-stream NVFP4 grouped launch facade. Setup allocates/uploads immutable
//! descriptors separately; enqueue allocates/copies/frees nothing and never waits.
//! `n` is expert/token pairs, not unique experts or weight output rows. Repeated
//! consecutive descriptors are grouped by the existing kernel, including n=1.
//! This approximate FP16-operand path still requires the prefill quality gate.
//!
//! Each outstanding enqueue needs its own status slot. Completion must be checked
//! before consuming output. An unchecked slot cannot be reused, and dropping one
//! waits and checks its flag, poisoning only on an error. Payload/IO
//! owners follow resident lifetime rules: frees wait for queued work. Callers
//! order all other writes with the explicit stream/event API, as for matmul.
//! Capture is deliberately refused: replay could reset an unchecked shared flag.

use super::launch::{Claim, require_apart, require_fit};
use super::stream::{DeviceEvent, DeviceStream, same_adapter};
use super::{CudaContext, DeviceBuffer, DeviceRange, device_error, graph};
use crate::EngError;
use crate::kernels::cuda::{TERMINAL, native};
use native::nvfp4_launch as ffi;
use std::collections::HashSet;
use std::sync::atomic::Ordering;

/// A shape-checked live packed matrix and its canonical per-16 E4M3FN scales.
#[derive(Clone, Copy)]
pub(crate) struct Nvfp4WeightRange<'a> {
    packed: DeviceRange<'a>,
    scales: DeviceRange<'a>,
    factor: f32,
    shape: [usize; 2],
}

/// Private immutable descriptor tables; payload owners stay borrowed for its life.
pub(crate) struct Nvfp4GroupedPlan<'a> {
    weights: Vec<Nvfp4WeightRange<'a>>,
    packed: DeviceBuffer,
    scales: DeviceBuffer,
    factors: DeviceBuffer,
    shape: [usize; 2],
    #[cfg(test)]
    descriptor_bytes: usize,
}

/// A single in-flight result's private rejection flag; never shared across enqueues.
#[must_use = "check completion before consuming output"]
pub(crate) struct Nvfp4LaunchStatus<'s> {
    stream: &'s DeviceStream,
    flag: DeviceBuffer,
    pending: Option<(u64, u64)>,
}

fn bad(message: &str) -> EngError {
    EngError::BadRequest(format!("cuda NVFP4 launch: {message}"))
}

impl Nvfp4WeightRange<'_> {
    pub(crate) fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl CudaContext {
    pub(crate) fn nvfp4_weight_range<'a>(
        &self,
        packed: DeviceRange<'a>,
        scales: DeviceRange<'a>,
        factor: f32,
        shape: [usize; 2],
    ) -> Result<Nvfp4WeightRange<'a>, EngError> {
        self.ensure_usable()?;
        self.raw_range(&packed)?;
        self.raw_range(&scales)?;
        let (p, s, _, _) = ffi::geometry(1, shape).ok_or_else(|| bad("invalid matrix geometry"))?;
        if !factor.is_finite() || factor <= 0.0 {
            return Err(bad("global scale must be finite and positive"));
        }
        // Byte payloads have no stronger alignment requirement in the ABI.
        require_fit("packed weight", &packed, p, 1)?;
        require_fit("weight scales", &scales, s, 1)?;
        require_apart(
            &Claim::of("packed weight", &packed, p),
            &Claim::of("weight scales", &scales, s),
        )?;
        Ok(Nvfp4WeightRange {
            packed,
            scales,
            factor,
            shape,
        })
    }

    /// Setup only. Admission can use `nvfp4_descriptor_bytes` before uploading.
    pub(crate) fn nvfp4_grouped_plan<'a>(
        &self,
        weights: &[Nvfp4WeightRange<'a>],
    ) -> Result<Nvfp4GroupedPlan<'a>, EngError> {
        self.ensure_usable()?;
        graph::ensure_closed("NVFP4 descriptor setup")?;
        let shape = weights.first().ok_or_else(|| bad("empty pair list"))?.shape;
        let (p, s, _, _) =
            ffi::geometry(weights.len(), shape).ok_or_else(|| bad("extents overflow"))?;
        #[cfg(test)]
        let descriptor_bytes = Self::nvfp4_descriptor_bytes(weights.len(), shape)?;
        #[cfg(not(test))]
        Self::nvfp4_descriptor_bytes(weights.len(), shape)?;
        let mut packed = Vec::new();
        let mut scales = Vec::new();
        let mut factors = Vec::new();
        let mut owned_weights = Vec::new();
        let mut unique_weights = Vec::new();
        let mut seen = HashSet::new();
        packed
            .try_reserve_exact(weights.len())
            .map_err(|_| bad("host descriptor allocation failed"))?;
        scales
            .try_reserve_exact(weights.len())
            .map_err(|_| bad("host descriptor allocation failed"))?;
        factors
            .try_reserve_exact(weights.len() * 4)
            .map_err(|_| bad("host descriptor allocation failed"))?;
        owned_weights
            .try_reserve_exact(weights.len())
            .map_err(|_| bad("host descriptor allocation failed"))?;
        unique_weights
            .try_reserve_exact(weights.len())
            .map_err(|_| bad("host descriptor allocation failed"))?;
        seen.try_reserve(weights.len())
            .map_err(|_| bad("host descriptor allocation failed"))?;
        for weight in weights {
            if weight.shape != shape {
                return Err(bad("all pairs must have the same matrix shape"));
            }
            packed.push(self.raw_range(&weight.packed)?);
            scales.push(self.raw_range(&weight.scales)?);
            factors.extend_from_slice(&weight.factor.to_ne_bytes());
            let key = (
                std::ptr::from_ref(weight.packed.buffer()),
                weight.packed.offset(),
                std::ptr::from_ref(weight.scales.buffer()),
                weight.scales.offset(),
            );
            if seen.insert(key) {
                unique_weights.push(weight);
            }
            owned_weights.push(*weight);
        }
        // shortcut: O(unique matrices²); use interval sweeping if distinct counts become a bottleneck.
        for weight in &unique_weights {
            for other in &unique_weights {
                require_apart(
                    &Claim::of("packed weight", &weight.packed, p),
                    &Claim::of("weight scales", &other.scales, s),
                )?;
            }
        }
        // Wrap each successful native allocation immediately for failure cleanup.
        let packed_table = {
            let _dispatch = self.enter()?;
            self.wrap(ffi::upload_pointers(&packed).map_err(device_error)?)
        };
        let scale_table = {
            let _dispatch = self.enter()?;
            self.wrap(ffi::upload_pointers(&scales).map_err(device_error)?)
        };
        let factors = self.upload(&factors)?;
        Ok(Nvfp4GroupedPlan {
            weights: owned_weights,
            packed: packed_table,
            scales: scale_table,
            factors,
            shape,
            #[cfg(test)]
            descriptor_bytes,
        })
    }

    pub(crate) fn nvfp4_descriptor_bytes(n: usize, shape: [usize; 2]) -> Result<usize, EngError> {
        let (_, _, pointers, _) =
            ffi::geometry(n, shape).ok_or_else(|| bad("invalid geometry or overflow"))?;
        pointers
            .checked_mul(2)
            .and_then(|v| v.checked_add(n.checked_mul(4)?))
            .ok_or_else(|| bad("descriptor bytes overflow"))
    }

    /// Setup only; adds exactly four device bytes per concurrent launch.
    pub(crate) fn nvfp4_launch_status<'s>(
        &self,
        stream: &'s DeviceStream,
    ) -> Result<Nvfp4LaunchStatus<'s>, EngError> {
        self.ensure_usable()?;
        same_adapter(&self.lease, stream.lease(), "stream")?;
        Ok(Nvfp4LaunchStatus {
            stream,
            flag: self.allocate(4)?,
            pending: None,
        })
    }

    /// Enqueue only. Inputs are [pairs][columns], outputs [pairs][rows] F32.
    /// Use distinct status slots for multiple outstanding calls. No result is
    /// numerically valid until `status.check` succeeds.
    pub(crate) fn nvfp4_grouped_enqueue(
        &self,
        stream: &DeviceStream,
        plan: &Nvfp4GroupedPlan<'_>,
        x: DeviceRange<'_>,
        y: DeviceRange<'_>,
        status: &mut Nvfp4LaunchStatus<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        graph::ensure_closed("NVFP4 checked launch")?;
        same_adapter(&self.lease, stream.lease(), "stream")?;
        if status.stream.id() != stream.id() {
            return Err(bad("status belongs to another stream"));
        }
        if status.pending.is_some() {
            return Err(bad("previous status has not been checked"));
        }
        let sequence = stream.next_enqueue()?;
        let (p, s, pointers, io) =
            ffi::geometry(plan.weights.len(), plan.shape).ok_or_else(|| bad("invalid geometry"))?;
        let flag = status.flag.range(0, 4)?;
        let tables = [
            plan.packed.range(0, pointers)?,
            plan.scales.range(0, pointers)?,
            plan.factors.range(0, plan.weights.len() * 4)?,
        ];
        let ranges = [tables[0], tables[1], tables[2], x, y, flag];
        let needs = [pointers, pointers, plan.weights.len() * 4, io.x, io.y, 4];
        let aligns = [align_of::<usize>(), align_of::<usize>(), 4, 4, 4, 4];
        for i in 0..ranges.len() {
            self.raw_range(&ranges[i])?;
            require_fit("launch region", &ranges[i], needs[i], aligns[i])?;
            for j in 0..i {
                require_apart(
                    &Claim::of("launch region", &ranges[i], needs[i]),
                    &Claim::of("launch region", &ranges[j], needs[j]),
                )?;
            }
            for weight in &plan.weights {
                self.raw_range(&weight.packed)?;
                self.raw_range(&weight.scales)?;
                require_apart(
                    &Claim::of("launch region", &ranges[i], needs[i]),
                    &Claim::of("packed weight", &weight.packed, p),
                )?;
                require_apart(
                    &Claim::of("launch region", &ranges[i], needs[i]),
                    &Claim::of("weight scales", &weight.scales, s),
                )?;
            }
        }
        let call = ffi::Launch {
            packed: self.raw_range(&tables[0])?,
            scales: self.raw_range(&tables[1])?,
            factors: self.raw_range(&tables[2])?,
            x: self.raw_range(&x)?,
            y: self.raw_range(&y)?,
            flag: self.raw_range(&flag)?,
            n: plan.weights.len(),
            shape: plan.shape,
        };
        self.launch_on(stream, |raw| ffi::launch(raw, &call))?;
        status.pending = Some((stream.id(), sequence));
        Ok(())
    }
}

#[cfg(test)]
impl Nvfp4GroupedPlan<'_> {
    pub(crate) fn descriptor_bytes(&self) -> usize {
        self.descriptor_bytes
    }
}

fn covers(pending: (u64, u64), mark: Option<(u64, u64)>) -> bool {
    mark.is_some_and(|(stream, count)| stream == pending.0 && count >= pending.1)
}

impl Nvfp4LaunchStatus<'_> {
    #[cfg(test)]
    pub(crate) fn device_bytes(&self) -> usize {
        4
    }

    /// Blocking completion outside enqueue. The event must be recorded on the
    /// launch stream after this enqueue. A transport failure leaves pending set;
    /// a checked numerical rejection returns BadRequest and invalidates output.
    pub(crate) fn check(
        &mut self,
        stream: &DeviceStream,
        event: &DeviceEvent,
    ) -> Result<(), EngError> {
        crate::kernels::cuda::ensure_usable(&TERMINAL)?;
        let pending = self
            .pending
            .ok_or_else(|| bad("status has no pending launch"))?;
        same_adapter(&self.flag.lease, stream.lease(), "stream")?;
        same_adapter(&self.flag.lease, event.lease(), "event")?;
        if stream.id() != pending.0 || !covers(pending, event.mark()) {
            return Err(bad("completion event does not cover this launch"));
        }
        event.synchronize()?;
        let mut bytes = [0u8; 4];
        self.flag.read_at(stream, 0, &mut bytes)?;
        self.pending = None;
        if u32::from_ne_bytes(bytes) != 0 {
            return Err(bad(
                "invalid scale, nonfinite value or FP16 operand overflow; output invalid",
            ));
        }
        Ok(())
    }
}

impl Drop for Nvfp4LaunchStatus<'_> {
    fn drop(&mut self) {
        if self.pending.is_none() {
            return;
        }
        // Cancellation follows buffer ownership rules: wait/check before release.
        // A healthy completed cancellation is not a terminal adapter failure.
        // No error may disappear merely because its ticket was dropped.
        let mut bytes = [0u8; 4];
        if self.flag.read_at(self.stream, 0, &mut bytes).is_err() || u32::from_ne_bytes(bytes) != 0
        {
            TERMINAL.store(true, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CudaContext, covers};
    #[test]
    fn completion_must_cover_same_stream_and_generation() {
        assert!(!covers((7, 3), None));
        assert!(!covers((7, 3), Some((8, 4))));
        assert!(!covers((7, 3), Some((7, 2))));
        assert!(covers((7, 3), Some((7, 3))));
        assert!(covers((7, 3), Some((7, 4))));
    }
    #[test]
    fn admission_counts_only_immutable_tables() {
        assert_eq!(
            CudaContext::nvfp4_descriptor_bytes(22, [65, 32]).unwrap(),
            22 * (2 * size_of::<usize>() + 4)
        );
        assert!(CudaContext::nvfp4_descriptor_bytes(usize::MAX, [1, 16]).is_err());
    }

    #[test]
    fn independent_launch_statuses_reject_overflow_and_allow_retry() {
        use crate::EngError;
        use crate::kernels::cuda::{
            Nvfp4GroupedPlan, Nvfp4LaunchStatus, Nvfp4WeightRange, testkit,
        };
        if !testkit::device_present("independent_launch_statuses_reject_overflow_and_allow_retry") {
            return;
        }
        let result = (|| -> Result<_, EngError> {
            let ctx = CudaContext::new()?;
            let stream = ctx.stream()?;
            let early = ctx.event()?;
            let done = ctx.event()?;
            let packed_a = ctx.upload(&[0x21; 256])?;
            let packed_b = ctx.upload(&[0x43; 256])?;
            let scales = ctx.upload(&[0x38; 32])?;
            let a: Nvfp4WeightRange<'_> = ctx.nvfp4_weight_range(
                packed_a.range(0, 256)?,
                scales.range(0, 32)?,
                1.0,
                [16, 32],
            )?;
            let b = ctx.nvfp4_weight_range(
                packed_b.range(0, 256)?,
                scales.range(0, 32)?,
                1.0,
                [16, 32],
            )?;
            let scaled_a = ctx.nvfp4_weight_range(
                packed_a.range(0, 256)?,
                scales.range(0, 32)?,
                2.0,
                [16, 32],
            )?;
            let alias_arena = ctx.allocate(1024)?;
            let alias_a = ctx.nvfp4_weight_range(
                alias_arena.range(0, 256)?,
                alias_arena.range(600, 32)?,
                1.0,
                [16, 32],
            )?;
            let alias_b = ctx.nvfp4_weight_range(
                alias_arena.range(256, 256)?,
                alias_arena.range(0, 32)?,
                1.0,
                [16, 32],
            )?;
            let overlap = ctx.nvfp4_grouped_plan(&[alias_a, alias_b]).map(|_| ());
            let plan: Nvfp4GroupedPlan<'_> = ctx.nvfp4_grouped_plan(&[a, scaled_a, b])?;
            let values = [vec![1.0f32; 32], vec![2.0; 32], vec![0.5; 32]].concat();
            let bytes = values
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>();
            let good = ctx.upload(&bytes)?;
            let bad_bytes = [f32::MAX; 96]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>();
            let bad = ctx.upload(&bad_bytes)?;
            let first = ctx.allocate(3 * 16 * 4)?;
            let second = ctx.allocate(3 * 16 * 4)?;
            let mut rejected: Nvfp4LaunchStatus<'_> = ctx.nvfp4_launch_status(&stream)?;
            let mut accepted = ctx.nvfp4_launch_status(&stream)?;
            let budget = [
                plan.descriptor_bytes(),
                rejected.device_bytes(),
                accepted.device_bytes(),
            ];
            ctx.nvfp4_grouped_enqueue(
                &stream,
                &plan,
                bad.range(0, 384)?,
                first.range(0, 192)?,
                &mut rejected,
            )?;
            stream.record(&early)?;
            let busy = ctx.nvfp4_grouped_enqueue(
                &stream,
                &plan,
                good.range(0, 384)?,
                first.range(0, 192)?,
                &mut rejected,
            );
            ctx.nvfp4_grouped_enqueue(
                &stream,
                &plan,
                good.range(0, 384)?,
                second.range(0, 192)?,
                &mut accepted,
            )?;
            stream.record(&done)?;
            let stale = accepted.check(&stream, &early);
            let overflow = rejected.check(&stream, &done);
            accepted.check(&stream, &done)?;
            ctx.ensure_usable()?;
            ctx.nvfp4_grouped_enqueue(
                &stream,
                &plan,
                good.range(0, 384)?,
                first.range(0, 192)?,
                &mut rejected,
            )?;
            stream.record(&done)?;
            rejected.check(&stream, &done)?;
            {
                let mut cancelled = ctx.nvfp4_launch_status(&stream)?;
                ctx.nvfp4_grouped_enqueue(
                    &stream,
                    &plan,
                    good.range(0, 384)?,
                    first.range(0, 192)?,
                    &mut cancelled,
                )?;
            }
            ctx.ensure_usable()?;
            let mut one = vec![0u8; 192];
            let mut two = vec![0u8; 192];
            first.read_at(&stream, 0, &mut one)?;
            second.read_at(&stream, 0, &mut two)?;
            Ok((overlap, busy, stale, overflow, one, two, budget))
        })();
        let (overlap, busy, stale, overflow, one, two, budget) =
            result.expect("owned grouped launch");
        for refusal in [overlap, busy, stale, overflow] {
            assert!(
                matches!(refusal, Err(EngError::BadRequest(_))),
                "{refusal:?}"
            );
        }
        let expected = [vec![24.0f32; 16], vec![96.0; 16], vec![28.0; 16]]
            .concat()
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>();
        assert_eq!(one, expected);
        assert_eq!(two, expected);
        assert_eq!(budget, [3 * (2 * size_of::<usize>() + 4), 4, 4]);
    }

    #[test]
    #[cfg(feature = "cuda-fault-qualification")]
    fn unchecked_rejection_latches_terminal_in_child() {
        use crate::EngError;
        use crate::kernels::cuda::testkit;
        const PATH: &str = "kernels::cuda::context::nvfp4_launch::tests::unchecked_rejection_latches_terminal_in_child";
        if !testkit::latching_test_in_child(PATH, &["unchecked grouped rejection checked"]) {
            return;
        }
        if !testkit::device_present("unchecked_rejection_latches_terminal_in_child") {
            return;
        }
        let observed = (|| -> Result<_, EngError> {
            let ctx = CudaContext::new()?;
            let stream = ctx.stream()?;
            let packed = ctx.upload(&[0x21; 256])?;
            let scales = ctx.upload(&[0x38; 32])?;
            let weight =
                ctx.nvfp4_weight_range(packed.range(0, 256)?, scales.range(0, 32)?, 1.0, [16, 32])?;
            let plan = ctx.nvfp4_grouped_plan(&[weight])?;
            let bytes = [f32::MAX; 32]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>();
            let input = ctx.upload(&bytes)?;
            let output = ctx.allocate(64)?;
            let mut status = ctx.nvfp4_launch_status(&stream)?;
            ctx.nvfp4_grouped_enqueue(
                &stream,
                &plan,
                input.range(0, 128)?,
                output.range(0, 64)?,
                &mut status,
            )?;
            drop(status);
            Ok(ctx.ensure_usable())
        })();
        assert!(matches!(
            observed.expect("owned unchecked rejection"),
            Err(EngError::Backend { .. })
        ));
        println!("unchecked grouped rejection checked");
    }
}
