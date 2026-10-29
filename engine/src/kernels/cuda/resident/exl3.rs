//! Shared EXL3 launches, staged off the default family paths. No allocation,
//! transfer or synchronization occurs here; retain owners until stream completion.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "EXL3 M1, family routing stays off")
)]
use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::exl3 as ffi;
use crate::weights::exl3::Codebook;

fn midrows_enabled(rows: usize, legacy: Option<&std::ffi::OsStr>) -> bool {
    (64..=144).contains(&rows) && legacy != Some(std::ffi::OsStr::new("1"))
}

/// Decode the admitted packed trellis to input-major [k,n] binary16 bits.
pub(crate) struct Exl3PrepCall<'a> {
    pub(crate) trellis: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) bits_twice: u8,
    pub(crate) book: Codebook,
}
impl CudaContext {
    pub(crate) fn exl3_prep(
        &self,
        stream: &DeviceStream,
        call: Exl3PrepCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let trellis = self.raw_range(&call.trellis)?;
        let out = self.raw_range(&call.out)?;
        let (packed, dense) = ffi::geometry(call.k, call.n, call.bits_twice, call.book)
            .ok_or_else(|| {
                EngError::BadRequest("EXL3 prep: invalid geometry or bitrate/codebook".into())
            })?;
        require_fit("EXL3 trellis", &call.trellis, packed, 2)?;
        require_fit("EXL3 W-prime", &call.out, dense, 2)?;
        require_apart(
            &Claim::of("EXL3 trellis", &call.trellis, packed),
            &Claim::of("EXL3 W-prime", &call.out, dense),
        )?;
        let raw = ffi::PrepRaw {
            trellis,
            out,
            k: call.k,
            n: call.n,
            bits_twice: call.bits_twice,
            book: call.book,
        };
        self.launch_on(stream, |s| ffi::prep(s, &raw))
    }
}
/// F32 x:[rows,k], scales:[k]/[n], optional bias:[n], out:[rows,n].
/// XH:[rows,k] is caller scratch. Z holds split_k-1 [rows,n] banks for rows<=8,
/// otherwise one bank. All F32 working spans are disjoint; split_k must be 2/4/8.
/// Initialize flag before use; it is sticky and invalid arithmetic does not poison context.
/// Standalone callers check bit0 after completion before consuming ANY output.
/// The private Flint Frame may enqueue scratch-only VALUE dependencies while retaining
/// all owners and the flag. It rejects every nonzero status before any host interpretation,
/// address/index/geometry decision or external output/state publication/ownership transfer.
pub(crate) struct Exl3LinearCall<'a> {
    pub(crate) trellis: DeviceRange<'a>,
    pub(crate) suh: DeviceRange<'a>,
    pub(crate) svh: DeviceRange<'a>,
    pub(crate) bias: Option<DeviceRange<'a>>,
    pub(crate) x: DeviceRange<'a>,
    pub(crate) xh: DeviceRange<'a>,
    pub(crate) z: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) flag: DeviceRange<'a>,
    pub(crate) rows: usize,
    pub(crate) split_k: u32,
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) bits_twice: u8,
    pub(crate) book: Codebook,
}
impl CudaContext {
    pub(crate) fn exl3_linear(
        &self,
        stream: &DeviceStream,
        call: Exl3LinearCall<'_>,
    ) -> Result<(), EngError> {
        self.exl3_linear_mode(stream, call, false)
    }

    /// Opt-in rows9..=144: legacy below64, tensor cores otherwise; FIFI_EXL3_LEGACY=1 rolls back.
    /// Working spans and accumulator stay F32.
    /// FP16-rounded operands and F32 accumulation can round differently from legacy F32.
    /// This opt-in path requires real-case numerical qualification, not assumed FP32 equivalence.
    pub(crate) fn exl3_linear_midrows(
        &self,
        stream: &DeviceStream,
        call: Exl3LinearCall<'_>,
    ) -> Result<(), EngError> {
        self.exl3_linear_mode(stream, call, true)
    }

    fn exl3_linear_raw<'a>(
        &self,
        call: &Exl3LinearCall<'a>,
        midrows: bool,
    ) -> Result<ffi::LinearRaw<'a>, EngError> {
        self.ensure_usable()?;
        let needs = ffi::linear_extents(
            call.rows,
            call.k,
            call.n,
            call.bits_twice,
            call.book,
            call.split_k,
        )
        .filter(|_| !midrows || (9..=144).contains(&call.rows))
        .ok_or_else(|| {
            EngError::BadRequest(
                "EXL3 linear: invalid geometry, bitrate/codebook or split-K".into(),
            )
        })?;
        let ranges = [
            Some(&call.trellis),
            Some(&call.suh),
            Some(&call.svh),
            call.bias.as_ref(),
            Some(&call.x),
            Some(&call.xh),
            Some(&call.z),
            Some(&call.out),
            Some(&call.flag),
        ];
        let names = [
            "EXL3 trellis",
            "EXL3 suh",
            "EXL3 svh",
            "EXL3 bias",
            "EXL3 x",
            "EXL3 xh",
            "EXL3 z",
            "EXL3 out",
            "EXL3 flag",
        ];
        for (i, range) in ranges.iter().enumerate() {
            let Some(range) = range else { continue };
            require_fit(names[i], range, needs[i], if i == 0 { 2 } else { 4 })?;
            for (j, other) in ranges[..i].iter().enumerate() {
                if let Some(other) = other
                    && (i >= 5 || j >= 5)
                {
                    require_apart(
                        &Claim::of(names[i], range, needs[i]),
                        &Claim::of(names[j], other, needs[j]),
                    )?;
                }
            }
        }
        Ok(ffi::LinearRaw {
            trellis: self.raw_range(&call.trellis)?,
            suh: self.raw_range(&call.suh)?,
            svh: self.raw_range(&call.svh)?,
            bias: call.bias.as_ref().map(|r| self.raw_range(r)).transpose()?,
            x: self.raw_range(&call.x)?,
            xh: self.raw_range(&call.xh)?,
            z: self.raw_range(&call.z)?,
            out: self.raw_range(&call.out)?,
            flag: self.raw_range(&call.flag)?,
            rows: call.rows,
            split_k: call.split_k,
            k: call.k,
            n: call.n,
            bits_twice: call.bits_twice,
            book: call.book,
        })
    }

    /// Every local group stays on the split-2 small-row arithmetic path.
    /// XH/Z/out slices must be contiguous in group order; inputs may share read-only storage.
    /// `gather` selects the shared-table gather variant; the launch plan is unchanged.
    pub(crate) fn exl3_grouped(
        &self,
        stream: &DeviceStream,
        calls: &[Exl3LinearCall<'_>],
        gather: bool,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        if calls.is_empty()
            || calls.len() > 48
            || calls
                .iter()
                .any(|call| !(1..=8).contains(&call.rows) || call.split_k != 2)
            || calls.iter().map(|call| call.rows).sum::<usize>() > 48
        {
            return Err(EngError::BadRequest(
                "EXL3 grouped needs 1..=48 pairs in local groups1..=8".into(),
            ));
        }
        let raw = calls
            .iter()
            .map(|call| self.exl3_linear_raw(call, false))
            .collect::<Result<Vec<_>, _>>()?;
        self.launch_on(stream, |s| ffi::grouped(s, &raw, gather))
    }

    fn exl3_linear_mode(
        &self,
        stream: &DeviceStream,
        call: Exl3LinearCall<'_>,
        midrows: bool,
    ) -> Result<(), EngError> {
        let raw = self.exl3_linear_raw(&call, midrows)?;
        let midrows =
            midrows && midrows_enabled(call.rows, std::env::var_os("FIFI_EXL3_LEGACY").as_deref());
        self.launch_on(stream, |s| {
            if midrows {
                ffi::linear_midrows(s, &raw)
            } else {
                ffi::linear(s, &raw)
            }
        })
    }
}
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
impl CudaContext {
    /// Runs only the host scheduling emitter; no context or driver call is made.
    pub(crate) fn test_exl3_emissions(
        shape: [usize; 3],
        split_k: u32,
        midrows: bool,
        fail_at: u32,
    ) -> Result<(Vec<[u32; 7]>, bool), EngError> {
        ffi::launch_observer::emissions(shape, split_k, midrows, fail_at)
            .map_err(|_| EngError::BadRequest("EXL3 observer geometry is invalid".into()))
    }

    pub(crate) fn test_exl3_grouped_emissions(
        shape: [usize; 3],
        fail_at: u32,
    ) -> Result<(Vec<[u32; 7]>, bool), EngError> {
        ffi::launch_observer::grouped_emissions(shape, fail_at)
            .map_err(|_| EngError::BadRequest("EXL3 grouped observer geometry is invalid".into()))
    }

    /// Counts successful native enqueues only inside a routed expert scope.
    pub(crate) fn test_routed_exl3_launches<R>(run: impl FnOnce() -> R) -> (R, u64) {
        ffi::launch_observer::capture(run)
    }

    /// Which grouped variant a captured leg actually ran: none, all shuffle,
    /// all gather, or mixed.
    pub(crate) fn test_routed_grouped_variant() -> Option<&'static str> {
        ffi::launch_observer::grouped_variant()
    }

    pub(crate) fn test_exl3_routed_scope() -> impl Drop {
        ffi::launch_observer::routed_scope()
    }
}

#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod observer_tests {
    use super::*;

    #[test]
    fn host_emitter_uses_the_production_grids_and_stops_after_failure() {
        let (stages, complete) =
            CudaContext::test_exl3_emissions([8, 2048, 512], 2, false, 0).unwrap();
        assert!(complete);
        assert_eq!(
            stages,
            [
                [0, 128, 1, 1, 128, 1, 1],
                [1, 32, 8, 2, 256, 1, 1],
                [4, 32, 1, 1, 128, 1, 1],
            ]
        );
        for fail_at in 1..=3 {
            let (emitted, complete) =
                CudaContext::test_exl3_emissions([8, 2048, 512], 2, false, fail_at).unwrap();
            assert!(!complete);
            assert_eq!(emitted, stages[..fail_at as usize]);
        }
        for (rows, midrows, projection) in [
            (1, false, [1, 32, 1, 2, 256, 1, 1]),
            (7, false, [1, 32, 7, 2, 256, 1, 1]),
            (9, false, [3, 32, 1, 1, 16, 16, 1]),
            (9, true, [2, 32, 1, 1, 128, 1, 1]),
            (144, true, [2, 32, 3, 1, 128, 1, 1]),
        ] {
            let (emitted, complete) =
                CudaContext::test_exl3_emissions([rows, 2048, 512], 2, midrows, 0).unwrap();
            assert!(complete);
            assert_eq!(emitted[1], projection);
        }
        for shape in [[0, 2048, 512], [8, 2047, 512], [65537, 2048, 512]] {
            assert!(CudaContext::test_exl3_emissions(shape, 2, false, 0).is_err());
        }
        let (grouped, complete) =
            CudaContext::test_exl3_grouped_emissions([48, 2048, 512], 0).unwrap();
        assert!(complete);
        assert_eq!(
            grouped,
            [
                [0, 16, 48, 1, 128, 1, 1],
                [1, 32, 48, 2, 256, 1, 1],
                [4, 4, 48, 1, 128, 1, 1],
            ]
        );
        for fail_at in 1..=3 {
            let (emitted, complete) =
                CudaContext::test_exl3_grouped_emissions([48, 2048, 512], fail_at).unwrap();
            assert!(!complete);
            assert_eq!(emitted, grouped[..fail_at as usize]);
        }
        for shape in [[0, 2048, 512], [49, 2048, 512], [48, 2047, 512]] {
            assert!(CudaContext::test_exl3_grouped_emissions(shape, 0).is_err());
        }
    }

    #[test]
    fn host_observation_cannot_increment_the_actual_launch_counter() {
        let (_, actual) = CudaContext::test_routed_exl3_launches(|| {
            let _scope = CudaContext::test_exl3_routed_scope();
            let (emitted, complete) =
                CudaContext::test_exl3_emissions([8, 2048, 512], 2, false, 0).unwrap();
            assert!(complete);
            assert_eq!(emitted.len(), 3);
            let (grouped, complete) =
                CudaContext::test_exl3_grouped_emissions([48, 2048, 512], 0).unwrap();
            assert!(complete);
            assert_eq!(grouped.len(), 3);
        });
        assert_eq!(actual, 0);
        assert!(
            std::panic::catch_unwind(|| {
                CudaContext::test_routed_exl3_launches(|| {
                    let _scope = CudaContext::test_exl3_routed_scope();
                    panic!("exercise capture unwind");
                });
            })
            .is_err()
        );
        let (_, actual) = CudaContext::test_routed_exl3_launches(|| {});
        assert_eq!(actual, 0);
    }
}

#[cfg(test)]
mod linear_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod validation_tests;
