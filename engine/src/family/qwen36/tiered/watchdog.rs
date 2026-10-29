//! The watchdog clamp of the tiered mailbox wait.
//!
//! WHAT: pure arithmetic that turns a requested CPU-server timeout into the
//! pair (server timeout, device deadline) the tiered step runs with, clamped
//! strictly below an assumed display-watchdog limit when the device reports
//! one. WHY: a graph-captured `tiered_wait_add` kernel spins on host-mapped
//! memory until the CPU server answers, and on a device whose display watchdog
//! is on, a kernel that outruns the driver's unnamed limit is killed with the
//! whole context — terminal — while the CUDA runtime exposes only the on/off
//! boolean. So the device deadline is held a margin below the assumed limit,
//! the server's own timeout fires first by a second margin, and where the
//! arithmetic leaves no room the caller refuses the in-graph wait and falls
//! back to the path without it. Both of our GPUs report no watchdog; the clamp
//! ships anyway. Deterministic: no clock read, no device query — `watchdog_on`
//! arrives as a parameter. This module is staged for the CUDA adapter of the
//! mailbox.

/// Longest time a kernel may run before the display watchdog kills it, ASSUMED
/// when the watchdog is on (the runtime only says on or off). Milliseconds.
/// WHY: the driver publishes no number, so a conservative guess below typical
/// display limits is the only safe ceiling.
pub const ASSUMED_WATCHDOG_MS: u64 = 2000;

/// Distance kept below that limit, in milliseconds. WHY: the assumption can be
/// wrong, so the device deadline never touches [`ASSUMED_WATCHDOG_MS`].
pub const WATCHDOG_MARGIN_MS: u64 = 500;

/// How much longer than the server's timeout the device waits, so that the
/// server's own timeout fires first. Milliseconds. WHY: a typed
/// server-side error is recoverable poison, while a device deadline miss is the
/// last line of defence.
pub const WAIT_MARGIN_MS: u64 = 1000;

/// The smallest server-timeout CAP worth running with, in milliseconds. WHY: when
/// the assumed watchdog limit leaves less than this for the server's timeout, the
/// in-graph wait is refused ([`Refusal::NoRoom`]) and the caller falls back to
/// the path without it. It applies to the cap, not to a request: a request below
/// it is kept as asked, because the caller asked for it.
pub const MIN_SERVER_TIMEOUT_MS: u64 = 100;

/// The pair of deadlines the tiered step runs with, in milliseconds: the CPU
/// server's own timeout and the device deadline of `tiered_wait_add`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Timeouts {
    /// The CPU server's own timeout in milliseconds; it fires at least
    /// `wait_margin_ms` before the device deadline, so the server's typed error
    /// usually wins the race.
    pub server_ms: u64,
    /// The `tiered_wait_add` deadline in milliseconds: how long the kernel
    /// spins before it sets `WAIT_TIMEOUT` and NaNs the row.
    pub device_ms: u64,
}

/// Why a requested server timeout cannot be run with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The caller asked for a zero timeout; nothing can be waited for.
    ZeroTimeout,
    /// The assumed watchdog limit leaves less than `min_server_ms` of server
    /// timeout under it, so the in-graph wait is refused in favour of the
    /// fallback path. Carries the largest server timeout in milliseconds that
    /// would have fit.
    NoRoom {
        /// The largest server timeout that fits under the assumed watchdog
        /// limit, in milliseconds; smaller than `min_server_ms`.
        server_cap_ms: u64,
    },
}

/// Pure; the constants above are the limits.
///
/// WHY: attach-time code knows only the watchdog boolean and the requested
/// server timeout, and this function turns them into the deadline pair without
/// touching the device or the clock, so it can be reasoned about and tested on
/// any host.
pub fn clamp_timeouts(requested_server_ms: u64, watchdog_on: bool) -> Result<Timeouts, Refusal> {
    clamp_with(Limits::DEFAULT, requested_server_ms, watchdog_on)
}

/// The same with explicit limits (so tests can reach `NoRoom`):
/// `(assumed_watchdog_ms, watchdog_margin_ms, wait_margin_ms, min_server_ms)`
/// packed as [`Limits`]. Semantics: a zero request is refused; with the watchdog
/// off the request is kept and the device waits one `wait_margin_ms` longer;
/// with the watchdog on both deadlines sit at least `watchdog_margin_ms` below
/// `assumed_watchdog_ms` (saturating, so no limit combination can panic), the
/// server timeout at most `server_cap = assumed - watchdog_margin -
/// wait_margin`, and a request below `min_server_ms` is kept as asked rather
/// than rounded up, because the caller asked for it.
pub(crate) fn clamp_with(
    limits: Limits,
    requested_server_ms: u64,
    watchdog_on: bool,
) -> Result<Timeouts, Refusal> {
    if requested_server_ms == 0 {
        return Err(Refusal::ZeroTimeout);
    }
    if !watchdog_on {
        return Ok(Timeouts {
            server_ms: requested_server_ms,
            device_ms: requested_server_ms.saturating_add(limits.wait_margin_ms),
        });
    }
    let device_cap_ms = limits
        .assumed_watchdog_ms
        .saturating_sub(limits.watchdog_margin_ms);
    let server_cap_ms = device_cap_ms.saturating_sub(limits.wait_margin_ms);
    if server_cap_ms < limits.min_server_ms {
        return Err(Refusal::NoRoom { server_cap_ms });
    }
    let server_ms = requested_server_ms.min(server_cap_ms);
    Ok(Timeouts {
        server_ms,
        // Exact while `wait_margin_ms` fits under the watchdog limit, which the
        // accepted region above guarantees for any sane `Limits`.
        device_ms: server_ms.saturating_add(limits.wait_margin_ms),
    })
}

/// The explicit limits [`clamp_with`] runs with; [`Limits::DEFAULT`] is the set
/// the engine ships with.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Limits {
    /// The assumed watchdog limit in milliseconds ([`ASSUMED_WATCHDOG_MS`]).
    pub assumed_watchdog_ms: u64,
    /// The distance kept below it in milliseconds ([`WATCHDOG_MARGIN_MS`]).
    pub watchdog_margin_ms: u64,
    /// The distance the device deadline sits beyond the server timeout in
    /// milliseconds ([`WAIT_MARGIN_MS`]).
    pub wait_margin_ms: u64,
    /// The smallest server-timeout cap in milliseconds ([`MIN_SERVER_TIMEOUT_MS`]): a watchdog cap below it
    /// refuses the in-graph wait; a request below it is kept as asked.
    pub min_server_ms: u64,
}

impl Limits {
    /// The shipped limits: the four module constants. WHY: one place ties the
    /// defaults together, so `clamp_timeouts` and the tests read the same
    /// numbers.
    pub const DEFAULT: Limits = Limits {
        assumed_watchdog_ms: ASSUMED_WATCHDOG_MS,
        watchdog_margin_ms: WATCHDOG_MARGIN_MS,
        wait_margin_ms: WAIT_MARGIN_MS,
        min_server_ms: MIN_SERVER_TIMEOUT_MS,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_a_watchdog_the_server_timeout_is_kept_and_the_device_waits_one_margin_longer() {
        let t = clamp_timeouts(60_000, false).unwrap();
        assert_eq!(
            t,
            Timeouts {
                server_ms: 60_000,
                device_ms: 61_000
            }
        );
        let t = clamp_timeouts(1, false).unwrap();
        assert_eq!(
            t,
            Timeouts {
                server_ms: 1,
                device_ms: 1_001
            }
        );
        // u64::MAX saturates instead of panicking, so no input can crash attach.
        let t = clamp_timeouts(u64::MAX, false).unwrap();
        assert_eq!(
            t,
            Timeouts {
                server_ms: u64::MAX,
                device_ms: u64::MAX
            }
        );
    }

    #[test]
    fn with_a_watchdog_both_timeouts_sit_below_the_assumed_limit_minus_the_margin() {
        // server_cap is 500, so every request at or above it clamps to 500/1500.
        for requested in [60_000u64, 1_500, 501, 500] {
            let t = clamp_timeouts(requested, true).unwrap();
            assert_eq!(t.server_ms, 500, "server for {requested}");
            assert_eq!(t.device_ms, 1_500, "device for {requested}");
            assert_eq!(t.device_ms, t.server_ms + WAIT_MARGIN_MS);
        }
        // A request below the cap is kept as asked; only the cap rounds down.
        let t = clamp_timeouts(300, true).unwrap();
        assert_eq!(
            t,
            Timeouts {
                server_ms: 300,
                device_ms: 1_300
            }
        );
    }

    #[test]
    fn the_server_timeout_always_fires_before_the_device_deadline_by_the_margin() {
        let mut requests: Vec<u64> = (1..=130_000).step_by(7).collect();
        requests.extend(499..=502);
        requests.extend(1_499..=1_502);
        for &requested in &requests {
            for watchdog_on in [false, true] {
                let t = clamp_timeouts(requested, watchdog_on).unwrap();
                assert!(
                    t.device_ms >= t.server_ms + WAIT_MARGIN_MS,
                    "requested {requested}, watchdog {watchdog_on}: {t:?}"
                );
                if watchdog_on {
                    assert!(
                        t.device_ms <= ASSUMED_WATCHDOG_MS - WATCHDOG_MARGIN_MS,
                        "requested {requested}: {t:?}"
                    );
                    assert!(t.device_ms < ASSUMED_WATCHDOG_MS, "{t:?}");
                }
            }
        }
    }

    #[test]
    fn a_zero_timeout_is_refused() {
        for watchdog_on in [false, true] {
            assert_eq!(
                clamp_timeouts(0, watchdog_on),
                Err(Refusal::ZeroTimeout),
                "watchdog {watchdog_on}"
            );
        }
    }

    #[test]
    fn a_limit_that_leaves_no_room_refuses_option_e() {
        // device_cap 1000, so server_cap 0: nothing fits under the watchdog.
        let no_room = Limits {
            assumed_watchdog_ms: 1_500,
            watchdog_margin_ms: 500,
            wait_margin_ms: 1_000,
            min_server_ms: 100,
        };
        for &requested in &[60_000u64, 100, 1] {
            assert_eq!(
                clamp_with(no_room, requested, true),
                Err(Refusal::NoRoom { server_cap_ms: 0 }),
                "requested {requested}"
            );
        }
        // Exactly min_server_ms of room is accepted, at the boundary.
        let exact = Limits {
            assumed_watchdog_ms: 1_600,
            watchdog_margin_ms: 500,
            wait_margin_ms: 1_000,
            min_server_ms: 100,
        };
        assert_eq!(
            clamp_with(exact, 60_000, true).unwrap(),
            Timeouts {
                server_ms: 100,
                device_ms: 1_100
            }
        );
    }

    #[test]
    fn the_default_limits_leave_room() {
        assert!(clamp_timeouts(60_000, true).is_ok());
        // The wrapper is exactly the default limits, so the shipped constants
        // can never refuse the in-graph wait.
        for &requested in &[60_000u64, MIN_SERVER_TIMEOUT_MS, 1] {
            assert_eq!(
                clamp_timeouts(requested, true),
                clamp_with(Limits::DEFAULT, requested, true)
            );
        }
    }
}
