//! Loader-to-stack wiring for one real generation over a pinned checkpoint directory.

use std::{
    fs,
    path::Path,
    time::{Duration, Instant},
};

use super::{
    NemotronTextSpec, admission, cpu_experts,
    exec::{self, NemotronExecSpec, NemotronStack, NemotronState, StoredWeight, WeightSource},
    loader::{self, NemotronLoader},
};
use crate::{EngError, memory::MemoryPolicy, tensor::Tensor};

pub use super::cpu_experts::{CpuLayerTiming, time_cpu_experts};

/// The pinned generation config lists eos ids [2, 11]; the tokenizer bundle resolves 11.
const EOS_ID: u32 = 11;

/// Cumulative per-token wall timestamps of the current run, filled only when
/// NEM35_TOKEN_TIMING=1. Index 0 is the prefill prediction (time to first
/// token); each later entry is the cumulative time after emitting that token.
/// Measurement only: enabling it changes no model math.
static TOKEN_TIMING: std::sync::Mutex<Option<Vec<f64>>> = std::sync::Mutex::new(None);
static RUN_STARTED: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);

fn token_timing_enabled() -> bool {
    static ONCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONCE.get_or_init(|| std::env::var_os("NEM35_TOKEN_TIMING").is_some_and(|v| v == "1"))
}

fn start_token_timing(started: Instant) {
    if !token_timing_enabled() {
        return;
    }
    if let (Ok(mut stamps), Ok(mut run)) = (TOKEN_TIMING.lock(), RUN_STARTED.lock()) {
        *stamps = Some(Vec::new());
        *run = Some(started);
    }
}

fn record_token_timing(now: Instant) {
    if !token_timing_enabled() {
        return;
    }
    if let (Ok(mut stamps), Ok(run)) = (TOKEN_TIMING.lock(), RUN_STARTED.lock())
        && let (Some(values), Some(started)) = (stamps.as_mut(), *run)
    {
        values.push((now - started).as_secs_f64());
    }
}

/// With NEM35_TOKEN_TIMING=1, prints each emitted token to stderr as it appears, so a run that is
/// killed at its time limit still leaves the tokens it reached. Measurement only.
fn trace_token(index: usize, id: u32) {
    if !token_timing_enabled() {
        return;
    }
    let seconds = RUN_STARTED
        .lock()
        .ok()
        .and_then(|run| *run)
        .map_or(f64::NAN, |started| started.elapsed().as_secs_f64());
    eprintln!("nem35_token {index} {id} {seconds:.3}");
}

/// Cumulative per-token timestamps of the last run; `None` unless NEM35_TOKEN_TIMING=1.
pub fn token_timing_seconds() -> Option<Vec<f64>> {
    TOKEN_TIMING.lock().ok().and_then(|guard| guard.clone())
}

/// Per-token deltas: entry i is the seconds spent producing token i.
/// The first entry is the time to first token, including the prefill prediction.
pub fn token_deltas(cumulative: &[f64]) -> Vec<f64> {
    let mut previous = 0.0;
    cumulative
        .iter()
        .map(|&stamp| {
            let delta = stamp - previous;
            previous = stamp;
            delta
        })
        .collect()
}

/// The adapter that feeds `NemotronStack` from `NemotronLoader`.
/// Dense names pass through unchanged; expert fetches map onto loader keys.
pub struct LoaderWeightSource {
    loader: NemotronLoader,
}

impl LoaderWeightSource {
    pub fn open(directory: &Path, cache_entries: usize) -> Result<Self, EngError> {
        Ok(Self {
            loader: NemotronLoader::open(directory, cache_entries)?,
        })
    }
}

impl WeightSource for LoaderWeightSource {
    fn dense(&mut self, name: &str) -> Result<Tensor, EngError> {
        // The retained input scale is unused: the reference stack executes dequantized F32.
        Ok(self.loader.dense_weight(name)?.0)
    }

    fn dense_stored(&mut self, name: &str) -> Result<StoredWeight, EngError> {
        self.loader.dense_stored(name)
    }

    fn expert(
        &mut self,
        layer: u32,
        expert: exec::ExpertIndex,
        projection: exec::ExpertProjection,
    ) -> Result<Tensor, EngError> {
        self.loader.expert_block(
            usize::try_from(layer).map_err(|_| bad("layer index exceeds host limits"))?,
            match expert {
                exec::ExpertIndex::Routed(index) => loader::ExpertIndex::Routed(index),
                exec::ExpertIndex::Shared => loader::ExpertIndex::Shared,
            },
            match projection {
                exec::ExpertProjection::Up => loader::ExpertProjection::Up,
                exec::ExpertProjection::Down => loader::ExpertProjection::Down,
            },
        )
    }

    fn has_packed_experts(&self) -> bool {
        true
    }

    fn has_mapped_experts(&self) -> bool {
        true
    }

    fn expert_layer_mapping(&mut self, layer: u32) -> Result<loader::ExpertLayerMapping, EngError> {
        self.loader.expert_layer_mapping(
            usize::try_from(layer).map_err(|_| bad("layer index exceeds host limits"))?,
        )
    }

    fn release_expert_layer(&mut self, layer: u32) -> Result<(), EngError> {
        self.loader.release_expert_layer(
            usize::try_from(layer).map_err(|_| bad("layer index exceeds host limits"))?,
        )
    }

    fn release_routed_experts(&mut self, layer: u32, experts: &[u32]) -> Result<(), EngError> {
        self.loader.release_routed_experts(
            usize::try_from(layer).map_err(|_| bad("layer index exceeds host limits"))?,
            experts,
        )
    }

    fn cpu_expert_layer(
        &mut self,
        layer: u32,
    ) -> Result<crate::kernels::nvfp4_cpu::LayerExperts<'static>, EngError> {
        let mapping = self.loader.expert_layer_mapping(
            usize::try_from(layer).map_err(|_| bad("layer index exceeds host limits"))?,
        )?;
        cpu_experts::layer_experts_from_mapping(&mapping)
    }

    fn expert_packed(
        &mut self,
        layer: u32,
        expert: exec::ExpertIndex,
        projection: exec::ExpertProjection,
    ) -> Result<exec::PackedExpert, EngError> {
        self.loader.expert_packed(
            usize::try_from(layer).map_err(|_| bad("layer index exceeds host limits"))?,
            match expert {
                exec::ExpertIndex::Routed(index) => loader::ExpertIndex::Routed(index),
                exec::ExpertIndex::Shared => loader::ExpertIndex::Shared,
            },
            match projection {
                exec::ExpertProjection::Up => loader::ExpertProjection::Up,
                exec::ExpertProjection::Down => loader::ExpertProjection::Down,
            },
        )
    }
}

#[derive(Debug)]
pub struct Generated {
    pub prompt: Vec<u32>,
    pub produced: Vec<u32>,
    pub elapsed: Duration,
    /// Time from the start of the run until the stack was built (config, admission, checkpoint
    /// index, state, device context, and the hybrid placement when it is on), before the prefill.
    pub setup: Duration,
    /// Placement and per-token split when `NEM35_CPU_EXPERTS=1` ran; `None` on the sequential path.
    pub hybrid: Option<exec::HybridReport>,
}

/// Runs one greedy generation after probing Linux `MemAvailable` and admitting its memory plan.
pub fn generate(
    model_dir: &Path,
    prompt_tokens: &[u32],
    max_tokens: usize,
    cache_entries: usize,
) -> Result<Generated, EngError> {
    generate_with_admission(
        model_dir,
        prompt_tokens,
        max_tokens,
        cache_entries,
        probed_available_bytes()?,
    )
}

/// Runs one greedy generation with a caller-supplied host-available byte budget.
/// The admission ceiling is the larger of the 20 GiB floor and the policy
/// share of the budget, capped at the measured available bytes; it does not
/// reduce the requested cache or capacity.
pub fn generate_with_admission(
    model_dir: &Path,
    prompt_tokens: &[u32],
    max_tokens: usize,
    cache_entries: usize,
    available_bytes: u64,
) -> Result<Generated, EngError> {
    generate_with_constructor(
        model_dir,
        prompt_tokens,
        max_tokens,
        cache_entries,
        available_bytes,
        admission::admit_checkpoint,
        |spec, source, state, ()| match exec::HybridConfig::from_env()? {
            Some(config) => {
                NemotronStack::<LoaderWeightSource>::new_hybrid(spec, source, state, config)
            }
            None => NemotronStack::<LoaderWeightSource>::new(spec, source, state),
        },
    )
}

/// The CUDA run. `NEM35_CPU_EXPERTS=1` serves the routed experts from the host CPU (see
/// `exec::HybridConfig`); otherwise the sequential device path runs exactly as before.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub fn generate_cuda(
    model_dir: &Path,
    prompt_tokens: &[u32],
    max_tokens: usize,
    cache_entries: usize,
) -> Result<Generated, EngError> {
    let hybrid = exec::HybridConfig::from_env()?;
    generate_with_constructor(
        model_dir,
        prompt_tokens,
        max_tokens,
        cache_entries,
        probed_available_bytes()?,
        admission::cuda::admit_cuda_serving,
        move |spec, source, state, admitted| match hybrid {
            Some(config) => NemotronStack::<LoaderWeightSource>::new_cuda_hybrid(
                spec, source, state, admitted, config,
            ),
            None => NemotronStack::<LoaderWeightSource>::new_cuda(spec, source, state, admitted),
        },
    )
}

fn generate_with_constructor<A>(
    model_dir: &Path,
    prompt_tokens: &[u32],
    max_tokens: usize,
    cache_entries: usize,
    available_bytes: u64,
    admit: impl FnOnce(
        &Path,
        &NemotronTextSpec,
        &NemotronExecSpec,
        u32,
        usize,
        MemoryPolicy,
        u64,
    ) -> Result<A, EngError>,
    constructor: impl FnOnce(
        NemotronExecSpec,
        LoaderWeightSource,
        NemotronState,
        A,
    ) -> Result<NemotronStack<LoaderWeightSource>, EngError>,
) -> Result<Generated, EngError> {
    let started = Instant::now();
    let config = fs::read(model_dir.join("config.json"))
        .map_err(|error| bad(format!("read config.json: {error}")))?;
    let descriptor = NemotronTextSpec::from_config(&config)?;
    let spec = NemotronExecSpec::from_descriptor(&descriptor)?;
    let capacity = prompt_tokens
        .len()
        .checked_add(max_tokens)
        .and_then(|tokens| tokens.checked_add(8))
        .and_then(|tokens| u32::try_from(tokens).ok())
        .ok_or_else(|| bad("context capacity exceeds u32"))?;
    let admitted = admit(
        model_dir,
        &descriptor,
        &spec,
        capacity,
        cache_entries,
        MemoryPolicy::default_share(),
        available_bytes,
    )?;
    let source = LoaderWeightSource::open(model_dir, cache_entries)?;
    let state = NemotronState::new(&spec, capacity)?;
    let mut stack = constructor(spec, source, state, admitted)?;
    let setup = started.elapsed();
    start_token_timing(started);
    stack.prefill(prompt_tokens)?;
    record_token_timing(Instant::now());
    let produced = emit_greedy(&mut stack, max_tokens)?;
    Ok(Generated {
        prompt: prompt_tokens.to_vec(),
        produced,
        elapsed: started.elapsed(),
        setup,
        hybrid: stack.hybrid_report(),
    })
}

fn probed_available_bytes() -> Result<u64, EngError> {
    admission::host_available_bytes().ok_or_else(|| {
        EngError::MemoryPlan(
            "Nemotron admission: host availability probe requires Linux; use generate_with_admission with an explicit available-byte budget".into(),
        )
    })
}

/// Emission seam: any source exposing the exec prediction contract.
/// Splitting the loop out makes the runner token order unit-testable.
pub(crate) trait PredictionSource {
    fn last_prediction(&self) -> Option<u32>;
    fn next_token(&mut self) -> Result<u32, EngError>;
}

impl<S: WeightSource> PredictionSource for NemotronStack<S> {
    fn last_prediction(&self) -> Option<u32> {
        NemotronStack::last_prediction(self)
    }
    fn next_token(&mut self) -> Result<u32, EngError> {
        NemotronStack::next_token(self)
    }
}

/// Emits the stored prefill prediction first, then advances, stopping at EOS.
fn emit_greedy<P: PredictionSource>(
    predictor: &mut P,
    max_tokens: usize,
) -> Result<Vec<u32>, EngError> {
    let mut produced = Vec::new();
    if let Some(first) = predictor.last_prediction() {
        if max_tokens == 0 || first == EOS_ID {
            return Ok(produced);
        }
        produced.push(first);
        trace_token(0, first);
    }
    while produced.len() < max_tokens {
        let id = predictor.next_token()?;
        record_token_timing(Instant::now());
        if id == EOS_ID {
            break;
        }
        trace_token(produced.len(), id);
        produced.push(id);
    }
    Ok(produced)
}
fn bad(message: impl Into<String>) -> EngError {
    EngError::Checkpoint(format!("Nemotron runner: {}", message.into()))
}

#[cfg(test)]
mod emit_tests {
    #[test]
    fn token_deltas_turn_cumulative_stamps_into_per_token_seconds() {
        assert_eq!(super::token_deltas(&[]), Vec::<f64>::new());
        assert_eq!(super::token_deltas(&[10.0]), vec![10.0]);
        assert_eq!(
            super::token_deltas(&[10.0, 12.5, 13.0]),
            vec![10.0, 2.5, 0.5]
        );
    }

    use super::*;

    /// Scripted predictor: predictions[0] is the stored prefill prediction;
    /// each next_token() returns the following entry.
    struct Fake {
        predictions: Vec<u32>,
        advanced: usize,
    }

    impl PredictionSource for Fake {
        fn last_prediction(&self) -> Option<u32> {
            self.predictions.first().copied()
        }
        fn next_token(&mut self) -> Result<u32, EngError> {
            self.advanced += 1;
            self.predictions
                .get(self.advanced)
                .copied()
                .ok_or_else(|| bad("script exhausted"))
        }
    }

    #[test]
    fn emits_the_prefill_prediction_first_and_stops_at_eos() {
        let mut f = Fake {
            predictions: vec![7, 9, 11, 20],
            advanced: 0,
        };
        assert_eq!(emit_greedy(&mut f, 4).unwrap(), vec![7, 9]);
        assert_eq!(f.advanced, 2);
    }

    #[test]
    fn zero_budget_emits_nothing() {
        let mut f = Fake {
            predictions: vec![7, 9],
            advanced: 0,
        };
        assert!(emit_greedy(&mut f, 0).unwrap().is_empty());
        assert_eq!(f.advanced, 0);
    }

    #[test]
    fn budget_one_emits_only_the_prefill_prediction() {
        let mut f = Fake {
            predictions: vec![7, 9],
            advanced: 0,
        };
        assert_eq!(emit_greedy(&mut f, 1).unwrap(), vec![7]);
        assert_eq!(f.advanced, 0);
    }

    #[test]
    fn immediate_eos_returns_empty_without_advancing() {
        let mut f = Fake {
            predictions: vec![11, 7],
            advanced: 0,
        };
        assert!(emit_greedy(&mut f, 8).unwrap().is_empty());
        assert_eq!(f.advanced, 0);
    }
}
