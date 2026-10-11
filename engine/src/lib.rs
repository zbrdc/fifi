mod checkpoint_json;

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub mod cache;
#[cfg(test)]
mod exec_fake;
pub mod family;
pub mod kernels;
pub mod sched;
pub mod server;
pub mod tensor;
pub mod tokenizer;
pub mod weights;

/// Serializes the in-crate tests that hold MLX adapters. The process admits two adapter leases
/// (decision 0024), so every library test module shares this one lock. A panicked test does not
/// poison it for the others.
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

#[derive(Debug, PartialEq, Eq)]
pub enum EngError {
    Checkpoint(String),
    BadRequest(String),
    MemoryPlan(String),
    ContextLimit(String),
    Backend {
        operation: &'static str,
        message: String,
    },
}
