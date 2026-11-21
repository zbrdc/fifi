//! Both reference backends use the shared token session and sampler.

use std::path::Path;

#[cfg(target_os = "linux")]
use super::DEFAULT_TEXT_CONTEXT;
use super::{
    TextBackend,
    cpu_forward::{CpuForward, PrefixState},
};
use crate::{
    EngError,
    family::{
        executor::{ExecutorOps, GenerationState},
        session::{
            DecodeSession, Limits, PrefixStore, SaveKind, StoreLimits, TokenStepper, check_prompt,
        },
    },
    sched::worker::{Engine, ReadyNote, spawn_engine_noted},
    tokenizer::{DecodeBatch, TokenizerBundle},
};

pub(crate) fn start_engine(
    backend: TextBackend,
    model_dir: &Path,
    tokenizer_dir: &Path,
    settings: TextSettings,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle), EngError> {
    let bundle = crate::tokenizer::inkling2::load(tokenizer_dir)?;
    let vocab_size = bundle.vocab_size();
    let eos = bundle.eos_token_ids().to_vec();
    let model_dir = model_dir.to_path_buf();
    let requested_rows = prefill_rows(backend)?;
    // Leave cancellation and prefix boundaries with the shared worker.
    let engine = spawn_engine_noted(1, ready_note, move |stages| {
        stages.begin("weights");
        let forward = CpuForward::open(&model_dir, settings.context as usize)?;
        let host_reserve = settings.host_reserve(&forward)?;
        let prefix = PrefixStore::with_limits(settings.prefix_limits());
        forward.validate_tokenizer(vocab_size, &eos)?;
        let forward = match backend {
            TextBackend::Cpu => forward,
            TextBackend::Cuda => {
                stages.begin("CUDA projections");
                cuda_forward(forward, host_reserve)?
            }
        };
        let prefill_rows = if supports_grouped_prefill(&forward) {
            requested_rows
        } else {
            1
        };
        Ok((
            TextExecutor {
                forward,
                eos,
                prefix,
                prefill_rows,
            },
            DecodeSession::default(),
        ))
    })?;
    Ok((engine, bundle))
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn cuda_forward(mut forward: CpuForward, host_reserve: u64) -> Result<CpuForward, EngError> {
    forward.enable_cuda(host_reserve)?;
    Ok(forward)
}

#[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
fn cuda_forward(_forward: CpuForward, _host_reserve: u64) -> Result<CpuForward, EngError> {
    Err(EngError::BadRequest(
        "Inkling CUDA serving requires a 64-bit Linux build with native-cuda".into(),
    ))
}

fn prefill_rows(backend: TextBackend) -> Result<usize, EngError> {
    const NAME: &str = "FIFI_INKLING_CUDA_PREFILL_ROWS";
    if backend != TextBackend::Cuda {
        return Ok(1);
    }
    let Some(value) = std::env::var_os(NAME) else {
        return Ok(1);
    };
    value
        .to_str()
        .and_then(|text| text.parse::<usize>().ok())
        .filter(|rows| (1..=super::MAX_PREFILL_ROWS).contains(rows))
        .ok_or_else(|| super::bad(format!("{NAME} must be in 1..={}", super::MAX_PREFILL_ROWS)))
}

fn supports_grouped_prefill(forward: &CpuForward) -> bool {
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    return forward.supports_grouped_prefill();
    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
    {
        let _ = forward;
        false
    }
}

impl TokenStepper for CpuForward {
    const OPERATION: &'static str = "inkling2_text";

    fn feed(&mut self, token: u32) -> Result<(), EngError> {
        self.token(token, false).map(|_| ())
    }

    fn feed_rows(&mut self, tokens: &[u32]) -> Result<(), EngError> {
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if tokens.len() > 1 && self.supports_grouped_prefill() {
            return self.prefill_grouped(tokens);
        }
        tokens.iter().try_for_each(|&token| self.feed(token))
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
        self.token(token, true)?.ok_or_else(|| EngError::Backend {
            operation: Self::OPERATION,
            message: "the output projection returned no logits".into(),
        })
    }

    fn fed(&self) -> usize {
        self.position()
    }

    fn reset(&mut self) -> Result<(), EngError> {
        Self::reset(self);
        Ok(())
    }
}

pub(crate) struct TextSettings {
    pub(crate) context: u32,
    pub(crate) prefix_reuse: bool,
}

impl TextSettings {
    fn prefix_limits(&self) -> StoreLimits {
        if self.prefix_reuse {
            StoreLimits {
                roots: 1,
                entries: 2,
            }
        } else {
            StoreLimits {
                roots: 0,
                entries: 0,
            }
        }
    }

    fn host_reserve(&self, forward: &CpuForward) -> Result<u64, EngError> {
        let limits = self.prefix_limits();
        // Count every saved state at full size irrespective of shared KV rows.
        let reserve = forward.host_state_bytes()? * (1 + limits.roots + limits.entries) as u64;
        #[cfg(target_os = "linux")]
        if self.context > DEFAULT_TEXT_CONTEXT || self.prefix_reuse {
            const LABEL: &str = "Inkling text state";
            crate::family::host_admit::HostAdmission::probe(LABEL, "Inkling")?
                .admit(LABEL, reserve)?;
        }
        Ok(reserve)
    }
}

struct TextExecutor {
    prefix: PrefixStore<PrefixState, ()>,
    forward: CpuForward,
    eos: Vec<u32>,
    prefill_rows: usize,
}

impl ExecutorOps for TextExecutor {
    type Session = DecodeSession;

    fn check_prompt(&self, prompt_tokens: usize) -> Result<(), EngError> {
        check_prompt(prompt_tokens, self.forward.context)
    }

    fn history<'s>(&self, session: &'s Self::Session) -> &'s [u32] {
        session.history()
    }

    fn reset(&mut self, session: &mut Self::Session) -> Result<(), EngError> {
        session.reset(&mut self.forward)
    }

    fn prefill(&mut self, session: &mut Self::Session, tokens: &[u32]) -> Result<(), EngError> {
        let context = self.forward.context;
        session.prefill(&mut self.forward, tokens, context)
    }

    fn decode(
        &mut self,
        session: &mut Self::Session,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch, EngError> {
        let limits = Limits {
            eos: &self.eos,
            sampler_vocab: self.forward.vocab_size() as usize,
            context_limit: self.forward.context,
        };
        session.decode(&mut self.forward, generation, &limits)
    }

    fn boundary(&mut self, _session: &mut Self::Session) {}

    fn prefill_piece(&self) -> Option<usize> {
        Some(self.prefill_rows)
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    fn prefill_bank_report(&self) -> Option<crate::sched::job::PrefillBankReport> {
        match &self.forward.weights {
            super::cpu_weights::Weights::Gguf(weights) => weights.prefill_bank_report(),
            _ => None,
        }
    }

    fn prefix_limits(&self) -> StoreLimits {
        self.prefix.limits()
    }

    fn save_prefix(
        &mut self,
        session: &Self::Session,
        key: &[u32],
        kind: SaveKind,
    ) -> Result<bool, EngError> {
        session.save_prefix(
            &mut self.prefix,
            key,
            (),
            self.forward.position(),
            kind,
            || Ok(self.forward.snapshot()),
        )
    }

    fn restore_prefix(
        &mut self,
        session: &mut Self::Session,
        prompt: &[u32],
    ) -> Result<usize, EngError> {
        session.restore_prefix(&mut self.prefix, prompt, (), |state| {
            self.forward.restore(state);
            Ok(())
        })
    }
}
