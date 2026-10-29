mod checkpoint_json;

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub mod cache;
#[cfg(test)]
mod exec_fake;
pub mod expert_tier;
pub mod family;
pub mod kernels;
pub mod memory;
pub mod sched;
pub mod server;
pub mod tensor;
pub mod tokenizer;
pub mod weights;

/// Serializes the in-crate tests that hold MLX adapters. The process admits two adapter leases, so
/// every library test module shares this one lock. A panicked test does not poison it for the
/// others.
#[cfg(all(
    test,
    feature = "native-mlx",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub(crate) fn native_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngError {
    Checkpoint(String),
    /// Invalid model output fails one job; native backend/status failures remain terminal.
    ModelOutput(String),
    BadRequest(String),
    MemoryPlan(String),
    ContextLimit(String),
    /// The request's prompt exceeds the token ceiling the engine can honor
    /// right now: `limit_tokens` is the largest prompt admissible at this
    /// instant, after eviction — the effective pool-derived ceiling, at most
    /// the configured context limit. `prompt_tokens` is the request's total
    /// prompt. The invariant is `prompt_tokens > limit_tokens`. The Anthropic
    /// surface reports it in the exact over-length form Claude Code matches
    /// to auto-compact.
    PromptTooLong {
        prompt_tokens: u64,
        limit_tokens: u64,
    },
    Backend {
        operation: &'static str,
        message: String,
    },
}

impl EngError {
    /// The wire form of the over-length refusal: the exact string Claude Code
    /// matches to auto-compact. One source for every surface that renders it.
    pub fn prompt_too_long_message(prompt_tokens: u64, limit_tokens: u64) -> String {
        format!("prompt is too long: {prompt_tokens} tokens > {limit_tokens} maximum")
    }
}
