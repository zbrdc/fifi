//! The CPU/GPU hand-off of one MoE block.

use std::time::{Duration, Instant};

use crate::EngError;

/// Wall time of the two concurrent halves of one block's expert work.
#[derive(Clone, Copy, Debug, Default)]
pub struct Overlap {
    /// The host-expert closure, measured on its own thread.
    pub routed: Duration,
    /// The device-side closure, measured on the calling thread.
    pub shared: Duration,
    /// How long the caller then waited for the host closure after the device side finished.
    pub exposed: Duration,
}

/// Runs `host` on a scoped thread while the calling thread runs `device`, then joins.
///
/// The device context is thread-confined, so `device` stays on the caller and only the host
/// kernel moves. A panic in `host` comes back as an error, never as unwinding through the caller.
pub fn run_overlapped<H, D>(
    host: impl FnOnce() -> H + Send,
    device: impl FnOnce() -> D,
) -> Result<(H, D, Overlap), EngError>
where
    H: Send,
{
    let mut timing = Overlap::default();
    let (joined, device_value) = std::thread::scope(|scope| {
        let handle = scope.spawn(|| {
            let started = Instant::now();
            let value = host();
            (value, started.elapsed())
        });
        let started = Instant::now();
        let device_value = device();
        timing.shared = started.elapsed();
        let waited = Instant::now();
        let joined = handle.join();
        timing.exposed = waited.elapsed();
        (joined, device_value)
    });
    let (host_value, routed) = joined.map_err(|_| EngError::Backend {
        operation: "expert_tier_overlap",
        message: "the host-expert thread panicked".to_owned(),
    })?;
    timing.routed = routed;
    Ok((host_value, device_value, timing))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_halves_run_and_the_device_side_hides_the_host_side() {
        let (host, device, timing) = run_overlapped(
            || {
                std::thread::sleep(Duration::from_millis(60));
                7
            },
            || {
                std::thread::sleep(Duration::from_millis(80));
                "device"
            },
        )
        .expect("overlap");
        assert_eq!((host, device), (7, "device"));
        assert!(timing.routed >= Duration::from_millis(60));
        assert!(timing.shared >= Duration::from_millis(80));
        // The host side finished first, so almost none of it is exposed.
        assert!(timing.exposed < Duration::from_millis(30), "{timing:?}");
    }

    #[test]
    fn a_slower_host_side_is_exposed_by_the_difference() {
        let (_, _, timing) = run_overlapped(
            || std::thread::sleep(Duration::from_millis(120)),
            || std::thread::sleep(Duration::from_millis(20)),
        )
        .expect("overlap");
        assert!(timing.exposed >= Duration::from_millis(60), "{timing:?}");
    }

    #[test]
    fn a_panicking_host_side_is_an_error_not_an_unwind() {
        let outcome = run_overlapped(|| -> u32 { panic!("kernel bug") }, || 1);
        assert!(matches!(outcome, Err(EngError::Backend { .. })));
    }
}
