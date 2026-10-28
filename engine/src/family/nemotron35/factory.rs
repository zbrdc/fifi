//! Engine factory for one nemotron35 entry of the enginesrv config.
//!
//! It follows the interface of the Flash-Next and Q36 factories and returns the
//! same StartedEngine. Options are checked before any file is read. The engine
//! thread then loads the pinned tokenizer bundle, the checkpoint metadata, and
//! the serving admission before it constructs the stack.
//!
//! The cuda backend runs the head on the device and accepts greedy requests
//! only. The executor refuses a sampled request until the canonical device
//! sampler is accepted, and no request falls back to host sampling.

use super::{
    NemotronTextSpec, admission,
    exec::{NemotronExecSpec, NemotronStack, NemotronState},
    executor::NemotronExecutor,
    runner::LoaderWeightSource,
};
use crate::{
    EngError,
    family::flash_next::factory::StartedEngine,
    memory::MemoryPolicy,
    sched::worker::{ReadyNote, spawn_engine_noted},
};
use serde::Deserialize;
use std::{fs, path::Path};

/// The family name of a Nemotron engine entry.
pub(crate) const FAMILY: &str = "nemotron35";

/// The options of one Nemotron engine entry.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NemotronOptions {
    backend: String,
    context_limit: u32,
    cache_entries: usize,
    prefill_chunk: u32,
}

impl NemotronOptions {
    /// Rejects every unsupported setting before a file or a device is touched.
    fn check(&self) -> Result<(), EngError> {
        match self.backend.as_str() {
            "cpu_reference" => {}
            "cuda" if cuda_build() => {}
            "cuda" => {
                return Err(config_error(
                    "the cuda backend needs a Linux build with the native-cuda feature",
                ));
            }
            other => {
                return Err(config_error(format!(
                    "backend {other:?} is unsupported; use cpu_reference or cuda"
                )));
            }
        }
        if self.context_limit == 0 {
            return Err(config_error("context_limit must be positive"));
        }
        if self.cache_entries == 0 {
            return Err(config_error("cache_entries must be positive"));
        }
        if self.prefill_chunk == 0 || self.prefill_chunk > self.context_limit {
            return Err(config_error(
                "prefill_chunk must be positive and at most context_limit",
            ));
        }
        Ok(())
    }
}

const fn cuda_build() -> bool {
    cfg!(all(feature = "native-cuda", target_os = "linux"))
}

enum StackAdmission {
    Cpu,
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    Cuda(admission::cuda::CudaAdmission),
}

/// Starts the engine of one config entry.
pub(crate) fn start_engine(
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&NemotronOptions>,
    ready_note: ReadyNote,
) -> Result<StartedEngine, EngError> {
    let options = options.ok_or_else(|| config_error("this entry needs options"))?;
    options.check()?;
    let bundle = crate::tokenizer::nemotron::load(tokenizer_dir)?;
    let eos = bundle.eos_token_ids().to_vec();
    let bundle_vocab = bundle.vocab_size();
    let piece = usize::try_from(options.prefill_chunk)
        .map_err(|_| config_error("prefill_chunk exceeds host limits"))?;
    let context_limit = options.context_limit;
    let owned = options.clone();
    let model = model_dir.to_path_buf();
    let engine = spawn_engine_noted(piece, ready_note, move || {
        let config = fs::read(model.join("config.json"))
            .map_err(|error| EngError::Checkpoint(format!("nemotron35 config.json: {error}")))?;
        let descriptor = NemotronTextSpec::from_config(&config)?;
        if descriptor.geometry().vocab_size != bundle_vocab {
            return Err(config_error(
                "the tokenizer vocabulary differs from the checkpoint vocabulary",
            ));
        }
        let spec = NemotronExecSpec::from_descriptor(&descriptor)?;
        let available = admission::host_available_bytes().ok_or_else(|| {
            EngError::MemoryPlan(
                "nemotron35 admission: available host memory is unavailable".to_string(),
            )
        })?;
        let admitted = match owned.backend.as_str() {
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            "cuda" => StackAdmission::Cuda(admission::cuda::admit_cuda_serving(
                &model,
                &descriptor,
                &spec,
                owned.context_limit,
                owned.cache_entries,
                MemoryPolicy::default_share(),
                available,
            )?),
            _ => {
                admission::admit_serving(
                    &model,
                    &descriptor,
                    &spec,
                    owned.context_limit,
                    owned.cache_entries,
                    MemoryPolicy::default_share(),
                    available,
                )?;
                StackAdmission::Cpu
            }
        };
        let state = NemotronState::new(&spec, owned.context_limit)?;
        let source = LoaderWeightSource::open(&model, owned.cache_entries)?;
        let stack = build_stack(admitted, spec, source, state)?;
        NemotronExecutor::new(stack, eos)
    })?;
    Ok(StartedEngine {
        engine,
        bundle,
        context_limit,
    })
}

fn build_stack(
    admitted: StackAdmission,
    spec: NemotronExecSpec,
    source: LoaderWeightSource,
    state: NemotronState,
) -> Result<NemotronStack<LoaderWeightSource>, EngError> {
    match admitted {
        StackAdmission::Cpu => NemotronStack::new(spec, source, state),
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        StackAdmission::Cuda(admission) => cuda_stack(spec, source, state, admission),
    }
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn cuda_stack(
    spec: NemotronExecSpec,
    source: LoaderWeightSource,
    state: NemotronState,
    admission: admission::cuda::CudaAdmission,
) -> Result<NemotronStack<LoaderWeightSource>, EngError> {
    NemotronStack::new_cuda(spec, source, state, admission)
}

fn config_error(message: impl Into<String>) -> EngError {
    EngError::BadRequest(format!("nemotron35 config: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::{FAMILY, NemotronOptions, start_engine};
    use crate::EngError;
    use std::path::Path;

    fn options(json: &str) -> NemotronOptions {
        serde_json::from_str(json).expect("valid options")
    }

    fn check(json: &str) -> String {
        match options(json).check() {
            Err(EngError::BadRequest(text)) => text,
            Err(other) => panic!("expected BadRequest, got {other:?}"),
            Ok(()) => panic!("expected the options to be rejected"),
        }
    }

    #[test]
    fn family_name_is_stable() {
        assert_eq!(FAMILY, "nemotron35");
    }

    #[test]
    fn valid_options_pass_and_unknown_or_missing_fields_fail_to_parse() {
        let json =
            r#"{"backend":"cpu_reference","context_limit":64,"cache_entries":1,"prefill_chunk":1}"#;
        let parsed = options(json);
        parsed.check().expect("valid options pass");
        let unknown = r#"{"backend":"cpu_reference","context_limit":64,"cache_entries":1,"prefill_chunk":1,"extra":2}"#;
        assert!(serde_json::from_str::<NemotronOptions>(unknown).is_err());
        let missing = r#"{"backend":"cpu_reference","context_limit":64}"#;
        assert!(serde_json::from_str::<NemotronOptions>(missing).is_err());
    }

    #[test]
    fn unknown_backends_name_the_supported_choices() {
        let text =
            check(r#"{"backend":"host","context_limit":64,"cache_entries":1,"prefill_chunk":1}"#);
        assert!(text.contains("unsupported"), "{text}");
        assert!(text.contains("cpu_reference"), "{text}");
    }

    #[test]
    fn the_cuda_backend_needs_a_native_build() {
        let json = r#"{"backend":"cuda","context_limit":64,"cache_entries":1,"prefill_chunk":1}"#;
        let parsed = options(json);
        if cfg!(all(feature = "native-cuda", target_os = "linux")) {
            parsed.check().expect("a native build accepts cuda");
        } else {
            assert!(check(json).contains("native-cuda"));
        }
    }

    #[test]
    fn zero_and_inverted_bounds_are_rejected() {
        let zero_context =
            r#"{"backend":"cpu_reference","context_limit":0,"cache_entries":1,"prefill_chunk":1}"#;
        assert!(check(zero_context).contains("context_limit"));
        let zero_cache =
            r#"{"backend":"cpu_reference","context_limit":64,"cache_entries":0,"prefill_chunk":1}"#;
        assert!(check(zero_cache).contains("cache_entries"));
        let zero_chunk =
            r#"{"backend":"cpu_reference","context_limit":64,"cache_entries":1,"prefill_chunk":0}"#;
        assert!(check(zero_chunk).contains("prefill_chunk"));
        let over_chunk = r#"{"backend":"cpu_reference","context_limit":64,"cache_entries":1,"prefill_chunk":65}"#;
        assert!(check(over_chunk).contains("prefill_chunk"));
    }

    #[test]
    fn missing_options_and_bad_options_stop_before_any_file_read() {
        let absent = Path::new("/nonexistent/model");
        let tokenizer = Path::new("/nonexistent/tokenizer");
        let error = match start_engine(absent, tokenizer, None, None) {
            Err(error) => error,
            Ok(_) => panic!("options are required"),
        };
        assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
        let json =
            r#"{"backend":"cpu_reference","context_limit":0,"cache_entries":1,"prefill_chunk":1}"#;
        let parsed = options(json);
        let error = match start_engine(absent, tokenizer, Some(&parsed), None) {
            Err(error) => error,
            Ok(_) => panic!("bounds are checked first"),
        };
        assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
    }
}
