//! Load policy of the CUDA projector for an integrated GPU.
//!
//! On a Jetson the device memory is system RAM. `cudaMemGetInfo` counts the
//! page cache as free, yet the allocator must reclaim that cache while it
//! allocates, and a warm cache of a 20 GiB checkpoint made a whole-model load
//! fail once in two runs on the Xavier (`cuda allocate: out of memory`).
//! A discrete GPU has its own memory, so none of this applies to it.
//!
//! An integrated device therefore gets four rules:
//! 1. evict the model file before admission, and again after the scale scan
//!    (the scan reads every Q6_K and Q8_0 tensor and refills the cache);
//! 2. drop each tensor's pages right after its upload, even when the caller
//!    passed no file (a file the caller did pass wins, so its choice and its
//!    failures stay visible);
//! 3. retry one tensor's upload at most [`OOM_RETRIES`] times after a device
//!    out-of-memory, evicting the file and pausing before each retry;
//! 4. nothing else: no thread, no timer and no native call besides
//!    `posix_fadvise`.
//!
//! A discrete GPU keeps the caller's options and never retries.
//!
//! Ownership of a retried attempt: the device buffer of a tensor is allocated
//! once, before any read of that tensor, so the out-of-memory happens at the
//! start of the upload. The failed attempt's `DeviceBuffer` drops on the error
//! path (it frees and `LIVE_BYTES` returns), the native allocation leaves
//! nothing on failure, and the weights under construction hold only completed
//! tensors. Only `MemoryPlan` errors of the allocate stage are retried; a
//! `Backend` error is terminal or a caller fault and never is.

use std::{cell::Cell, fs::File, time::Duration};

use crate::{EngError, kernels::cuda};

/// Device out-of-memory retries per tensor on an integrated GPU. Warm/keep
/// failed once in two Xavier loads and the rerun passed; the failing request
/// was one tensor while about 17 GiB of cache was reclaimable, so the cause is
/// a race with reclaim. After the whole-file eviction one retry is the expected
/// recovery and the second covers a race with other files' cache. More retries
/// mostly hide real exhaustion. The budget is per tensor, not per load.
pub(super) const OOM_RETRIES: u32 = 2;
/// Pause before a retry: a fraction of one tensor's upload time (0.2 to 0.6
/// GiB/s on the Xavier), enough for reclaim to make progress.
pub(super) const OOM_RETRY_PAUSE: Duration = Duration::from_millis(100);

/// What a load does differently on an integrated GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LoadPolicy {
    /// Evict the whole model file before admission and after the scale scan.
    pub(super) evict_file: bool,
    /// Drop each tensor's pages after its upload even when the caller passed no file.
    pub(super) drop_each_tensor: bool,
    /// Retries of one tensor's upload after a device out-of-memory.
    pub(super) oom_retries: u32,
    /// Pause before each retry.
    pub(super) retry_pause: Duration,
}

impl LoadPolicy {
    pub(super) const DISCRETE: Self = Self {
        evict_file: false,
        drop_each_tensor: false,
        oom_retries: 0,
        retry_pause: Duration::ZERO,
    };
    pub(super) const INTEGRATED: Self = Self {
        evict_file: true,
        drop_each_tensor: true,
        oom_retries: OOM_RETRIES,
        retry_pause: OOM_RETRY_PAUSE,
    };

    pub(super) fn for_device(integrated: bool) -> Self {
        if integrated {
            Self::INTEGRATED
        } else {
            Self::DISCRETE
        }
    }

    /// The file to drop after each tensor: the caller's file when it passed one
    /// (its choice wins), else the model file when the policy forces the drop.
    pub(super) fn drop_file<'a>(
        &self,
        own: &'a File,
        requested: Option<&'a File>,
    ) -> Option<&'a File> {
        requested.or_else(|| self.drop_each_tensor.then_some(own))
    }
}

/// True for the one failure the loader retries: `native_error`'s non-terminal
/// device out-of-memory at the allocate stage. Host chunk failures, "does not
/// fit", admission and every `Backend` error do not match.
pub(super) fn is_device_out_of_memory(error: &EngError) -> bool {
    matches!(error, EngError::MemoryPlan(text)
        if text.starts_with("cuda allocate: ") && text.ends_with("(status 2)"))
}

/// Drops every cached page of `file` (`posix_fadvise` with length 0 reaches
/// the end of the file).
pub(super) fn evict_model(file: &File) -> Result<(), EngError> {
    cuda::drop_page_cache(file, 0, 0)
}

/// The recovery between two attempts: evict, then give reclaim a moment.
pub(super) fn recover(file: &File, pause: Duration) -> Result<(), EngError> {
    evict_model(file)?;
    std::thread::sleep(pause);
    Ok(())
}

/// Runs `attempt`. After a device out-of-memory it runs `between` and tries
/// again, at most `policy.oom_retries` more times; every retry is counted in
/// `retries`. Any other error, an exhausted budget and a failing `between`
/// end the call with that error. `attempt` must own nothing when it fails.
pub(super) fn retry_on_oom<T>(
    policy: &LoadPolicy,
    retries: &Cell<u32>,
    mut attempt: impl FnMut() -> Result<T, EngError>,
    mut between: impl FnMut() -> Result<(), EngError>,
) -> Result<T, EngError> {
    let mut used = 0;
    loop {
        match attempt() {
            Err(error) if used < policy.oom_retries && is_device_out_of_memory(&error) => {
                between()?;
                used += 1;
                retries.set(retries.get() + 1);
            }
            outcome => return outcome,
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::{fs, io::Write, path::Path, process::Command};

    use super::*;

    const OUT_OF_MEMORY: &str = "cuda allocate: out of memory (status 2)";

    fn out_of_memory() -> EngError {
        EngError::MemoryPlan(OUT_OF_MEMORY.into())
    }

    fn backend(operation: &'static str) -> EngError {
        EngError::Backend {
            operation,
            message: "test".into(),
        }
    }

    /// Bytes of `path` in the page cache (`fincore`), or `None` when the tool fails.
    pub(in crate::family::project::cuda) fn resident_bytes(path: &Path) -> Option<u64> {
        let output = Command::new("fincore")
            .args(["--bytes", "--noheadings", "--output", "RES,SIZE"])
            .arg(path)
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8(output.stdout).ok()?;
        text.split_whitespace().next()?.parse().ok()
    }

    #[test]
    fn the_policy_follows_the_integrated_attribute() {
        assert_eq!(LoadPolicy::for_device(true), LoadPolicy::INTEGRATED);
        assert_eq!(LoadPolicy::for_device(false), LoadPolicy::DISCRETE);
        let integrated = LoadPolicy::INTEGRATED;
        assert!(integrated.evict_file && integrated.drop_each_tensor);
        assert_eq!(integrated.oom_retries, 2);
        assert_eq!(integrated.retry_pause, Duration::from_millis(100));
        let discrete = LoadPolicy::DISCRETE;
        assert!(!discrete.evict_file && !discrete.drop_each_tensor);
        assert_eq!(discrete.oom_retries, 0);
        assert_eq!(discrete.retry_pause, Duration::ZERO);
    }

    #[test]
    fn the_callers_file_wins_and_an_integrated_policy_fills_a_missing_one() {
        let own = File::open("/dev/null").expect("open own");
        let theirs = File::open("/dev/null").expect("open theirs");
        let pick = |policy: LoadPolicy, requested: Option<&File>| {
            policy.drop_file(&own, requested).map(std::ptr::from_ref)
        };
        let own_ptr = Some(std::ptr::from_ref(&own));
        let their_ptr = Some(std::ptr::from_ref(&theirs));
        assert_eq!(pick(LoadPolicy::INTEGRATED, Some(&theirs)), their_ptr);
        assert_eq!(pick(LoadPolicy::INTEGRATED, None), own_ptr);
        assert_eq!(pick(LoadPolicy::DISCRETE, Some(&theirs)), their_ptr);
        assert_eq!(pick(LoadPolicy::DISCRETE, None), None);
    }

    #[test]
    fn only_a_device_allocation_failure_is_retriable() {
        let cases = [
            (out_of_memory(), true),
            (
                EngError::MemoryPlan(
                    "cuda allocate: host source chunk of tensor w: memory allocation of 8 bytes failed"
                        .into(),
                ),
                false,
            ),
            (
                EngError::MemoryPlan("cuda allocate: tensor w does not fit".into()),
                false,
            ),
            (
                EngError::MemoryPlan("cuda admission: need 5 bytes, 4 free".into()),
                false,
            ),
            (backend("execute"), false),
            (backend("cuda copy"), false),
            (EngError::BadRequest("cuda allocate: x (status 2)".into()), false),
        ];
        for (error, expected) in cases {
            assert_eq!(is_device_out_of_memory(&error), expected, "{error:?}");
        }
    }

    #[test]
    fn a_persistent_out_of_memory_is_attempted_retries_plus_one_times() {
        let retries = Cell::new(0);
        let (mut attempts, mut betweens) = (0, 0);
        let outcome: Result<u32, EngError> = retry_on_oom(
            &LoadPolicy::INTEGRATED,
            &retries,
            || {
                attempts += 1;
                Err(out_of_memory())
            },
            || {
                betweens += 1;
                Ok(())
            },
        );
        assert_eq!(outcome, Err(out_of_memory()));
        assert_eq!((attempts, betweens, retries.get()), (3, 2, 2));
    }

    #[test]
    fn a_retry_that_succeeds_returns_its_value_after_one_recovery() {
        let retries = Cell::new(0);
        let (mut attempts, mut betweens) = (0, 0);
        let outcome = retry_on_oom(
            &LoadPolicy::INTEGRATED,
            &retries,
            || {
                attempts += 1;
                if attempts == 1 {
                    Err(out_of_memory())
                } else {
                    Ok(7)
                }
            },
            || {
                betweens += 1;
                Ok(())
            },
        );
        assert_eq!(outcome, Ok(7));
        assert_eq!((attempts, betweens, retries.get()), (2, 1, 1));
    }

    #[test]
    fn other_errors_and_a_discrete_policy_are_never_retried() {
        let cases: [(LoadPolicy, fn() -> EngError); 3] = [
            (LoadPolicy::INTEGRATED, || backend("execute")),
            (LoadPolicy::INTEGRATED, || backend("cuda copy")),
            (LoadPolicy::DISCRETE, out_of_memory),
        ];
        for (policy, make_error) in cases {
            let retries = Cell::new(0);
            let (mut attempts, mut betweens) = (0, 0);
            let outcome: Result<u32, EngError> = retry_on_oom(
                &policy,
                &retries,
                || {
                    attempts += 1;
                    Err(make_error())
                },
                || {
                    betweens += 1;
                    Ok(())
                },
            );
            assert_eq!(outcome, Err(make_error()));
            assert_eq!((attempts, betweens, retries.get()), (1, 0, 0));
        }
    }

    #[test]
    fn a_failing_recovery_ends_the_retry_with_its_error() {
        let retries = Cell::new(0);
        let mut attempts = 0;
        let outcome: Result<u32, EngError> = retry_on_oom(
            &LoadPolicy::INTEGRATED,
            &retries,
            || {
                attempts += 1;
                Err(out_of_memory())
            },
            || Err(backend("cuda adapter")),
        );
        assert_eq!(outcome, Err(backend("cuda adapter")));
        assert_eq!((attempts, retries.get()), (1, 0));
    }

    #[test]
    fn evict_model_drops_every_page_of_the_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("evict.bin");
        let mut file = File::create(&path).expect("create");
        file.write_all(&vec![0x5a; 1 << 20]).expect("write");
        file.sync_all().expect("sync");
        let file = File::open(&path).expect("open");
        // Control: the raw call must evict on this filesystem, else the test
        // cannot tell a no-op from a working eviction (tmpfs keeps its pages).
        fs::read(&path).expect("warm");
        if resident_bytes(&path).is_none_or(|bytes| bytes == 0) {
            println!(
                "SKIPPED evict_model_drops_every_page_of_the_file: fincore shows no cached pages"
            );
            return;
        }
        cuda::drop_page_cache(&file, 0, 0).expect("control drop");
        if resident_bytes(&path) != Some(0) {
            println!(
                "SKIPPED evict_model_drops_every_page_of_the_file: the filesystem keeps its pages"
            );
            return;
        }
        fs::read(&path).expect("warm again");
        assert!(resident_bytes(&path).is_some_and(|bytes| bytes > 0));
        evict_model(&file).expect("evict");
        assert_eq!(resident_bytes(&path), Some(0));
    }
}
