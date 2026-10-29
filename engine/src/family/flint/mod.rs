//! Flint metadata, pinned startup, and the initial MLX execution path.

pub(crate) mod chunk_plan;
mod config;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod cuda_executor;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) mod cuda_ffn;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) mod cuda_kv;
#[cfg(test)]
mod cuda_kv_tests;
// The prefix module's consumers are the native-cuda executor and its
// device gates; portable test builds do not compile it (
// a consumer-less compile is dead-code under clippy -D warnings).
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) mod cuda_prefix;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod cuda_prefix_tests;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod cuda_sample;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) mod exl3_resident;
pub(crate) mod factory;
pub(crate) mod settings;
// Staged CUDA source views; the mixed-residency entry has no serving caller yet.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
#[cfg_attr(not(test), allow(dead_code))]
mod gguf_weights;
#[cfg(test)]
mod kv_plan_tests;
#[cfg(test)]
mod mlx_qualification;
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
mod mlx_state;
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
mod precision;
#[cfg(test)]
mod tests;
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
pub(super) mod weights;

pub use config::check_config;

pub(crate) const FAMILY: &str = "flint";

/// Pure opt-in polarity for [`prefix_reuse_opt_in`]: ONLY the explicit
/// enable words turn the knob on; an absent variable, non-UTF-8, and every
/// other value stay off. Opt-in (not default-on): the server keeps Flint
/// prefix reuse excluded from the config flag until qualification lifts it.
/// Compiled only where a consumer exists (test binaries, the MLX target,
/// and the CUDA store since slice P1) so portable lib builds stay
/// warning-clean under -D warnings.
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
    all(feature = "native-cuda", target_os = "linux")
))]
pub(crate) fn prefix_reuse_env(value: std::result::Result<String, std::env::VarError>) -> bool {
    match value {
        Err(_) => false,
        Ok(value) => crate::tokenizer::env_allows_value(Some(&value)),
    }
}

/// The fixed OS floor subtracted from physical memory when deriving the
/// Flint prefix retention cap: the same spirit as the pinned-RAM
/// rule - the engine never plans against memory the OS needs. 8 GiB: macOS
/// anonymous compression + windowserver + the harness measured well under
/// this on the 64 GiB Mac.
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
pub(in crate::family::flint) const PREFIX_OS_FLOOR_BYTES: u64 = 8 << 30;

/// The derived retention cap: physical memory minus the loaded-weights
/// charge minus the OS floor. `None` when physical memory is
/// unknown - callers must treat that as NO cap derivable (fail closed),
/// never as "no limit".
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
pub(in crate::family::flint) fn derived_retention_cap(
    physical: Option<u64>,
    weights: u64,
) -> Option<u64> {
    physical.map(|p| p.saturating_sub(weights.saturating_add(PREFIX_OS_FLOOR_BYTES)))
}

/// Retention admission: the worst case a save commits is the new entry's
/// charge, the transient overlap with the entry it displaces, and the
/// copy-on-write the continuing session needs on the shared planes (the
/// full retained-plane bytes again). All three must fit the cap BEFORE the
/// store is touched.
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
pub(in crate::family::flint) fn retention_admitted(
    new_charge: u64,
    retained: u64,
    cap: u64,
) -> bool {
    new_charge.saturating_mul(2).saturating_add(retained) <= cap
}

/// FIFI_FLINT_PREFIX_BUDGET_BYTES, parsed STRICTLY: a set-but-unparsable OR
/// non-UTF-8 value is an Err naming it — fail closed at startup — never a
/// silent "no cap". Absent -> `Ok(None)`: the cap is derived.
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
pub(in crate::family::flint) fn retention_cap_from_env(
    raw: Option<&std::ffi::OsStr>,
) -> Result<Option<u64>, String> {
    match raw {
        None => Ok(None),
        Some(text) => {
            let text = text.to_str().ok_or_else(|| {
                "FIFI_FLINT_PREFIX_BUDGET_BYTES is set but not UTF-8 text".to_string()
            })?;
            text.parse::<u64>().map(Some).map_err(|_| {
                format!(
                    "FIFI_FLINT_PREFIX_BUDGET_BYTES is set but not a plain byte count: {text:?}"
                )
            })
        }
    }
}

/// The cap `open` admits saves against (the derivation must be CONDITIONAL
/// on the reuse opt-in, or default-OFF serving gains two new startup
/// failures - a sysctl probe and a strict env parse). Opt-in
/// OFF -> `Ok(0)`: no probe, no parse, and every save is refused (none can
/// run on the serving path anyway). Opt-in ON -> the strict env value or
/// the derived headroom. The cap covers THIS engine only: other engines in
/// the same process share the headroom but not this budget.
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
pub(in crate::family::flint) fn retention_cap_for(
    opt_in: bool,
    env_raw: Option<&std::ffi::OsStr>,
    physical: impl FnOnce() -> Option<u64>,
    weights: impl FnOnce() -> Result<u64, String>,
) -> Result<u64, String> {
    if !opt_in {
        return Ok(0);
    }
    match retention_cap_from_env(env_raw)? {
        // A zero budget would arm a store that refuses EVERY save while
        // prefill still splits - a silent misconfiguration, so reject it
        // at startup naming the var.
        Some(0) => Err(
            "FIFI_FLINT_PREFIX_BUDGET_BYTES=0 would refuse every save while prefill still splits; unset it to derive a cap or set a positive byte count"
                .to_string(),
        ),
        Some(cap) => Ok(cap),
        None => {
            let physical = physical()
                .ok_or_else(|| "physical memory unknown (sysctl hw.memsize failed)".to_string())?;
            let weights = weights()?;
            derived_retention_cap(Some(physical), weights)
                .filter(|cap| *cap > 0)
                .ok_or_else(|| {
                    "derived retention cap is zero (physical <= weights + OS floor)".to_string()
                })
        }
    }
}

/// The server's entry flag: the frozen opt-in
/// resolved against the engine's kv profile when the config carries one.
/// `None` is the MLX target (its config carries no CUDA options), where
/// the env alone arms the store; a non-f16 CUDA profile resolves OFF so a
/// q8 entry logs and arms "off" and its boundary requests serve cold.
pub(crate) fn prefix_reuse_entry_enabled(profile: Option<&str>) -> bool {
    prefix_reuse_opt_in() && profile.is_none_or(|profile| profile == "f16")
}

/// FIFI_FLINT_PREFIX_REUSE, read once per process. The mirror of the
/// default-on family switches but inverted: absence stays OFF. True only
/// on backends whose Flint executor carries a prefix store: MLX and CUDA.
/// Other builds compile to false so env+flag can never arm no-op save/restore
/// entries that would only split every turn's prefill.
#[cfg(any(
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
    all(feature = "native-cuda", target_os = "linux")
))]
pub(crate) fn prefix_reuse_opt_in() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| prefix_reuse_env(std::env::var("FIFI_FLINT_PREFIX_REUSE")))
}

#[cfg(not(any(
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
    all(feature = "native-cuda", target_os = "linux")
)))]
pub(crate) fn prefix_reuse_opt_in() -> bool {
    false
}

/// The CUDA per-engine resolution of the frozen opt-in against the kv
/// profile: a non-f16 profile resolves the opt-in to OFF with
/// one load-line note per process — the server keeps serving cold and says
/// why, instead of failing every boundary request at restore time.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) fn prefix_reuse_enabled_for(profile: &str) -> bool {
    if !prefix_reuse_opt_in() {
        return false;
    }
    if profile == "f16" {
        return true;
    }
    static NOTED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    NOTED.get_or_init(|| {
        eprintln!(
            "flint prefix reuse: off (kv_profile {profile}: cuda snapshots are f16-only; serving cold)"
        );
    });
    false
}

pub mod kv_plan;
pub mod math;
pub mod reference;

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod mlx_executor;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod mlx_forward;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub mod mlx_math;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod mlx_model;
#[cfg(all(
    test,
    feature = "native-mlx",
    target_os = "macos",
    target_arch = "aarch64"
))]
mod mlx_prefix_tests;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod mlx_route;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod precision_native;

#[cfg(all(
    test,
    feature = "native-mlx",
    target_os = "macos",
    target_arch = "aarch64"
))]
mod mlx_packed_tests;
