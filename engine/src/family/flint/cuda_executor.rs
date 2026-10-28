//! Single-session CUDA adapter. The shared session owns generation and publication.

use super::{
    cuda_ffn::device::runtime::{Decoder, State, VOCAB},
    cuda_sample::SamplingScratch,
    exl3_resident::{FlintExl3Resident, FlintResidentLimits},
    settings::{FlintOptions, late_health_selected},
};
use crate::{
    EngError,
    family::{
        executor::{ExecutorOps, GenerationState},
        project::cuda::{Reserve, served_admission},
        sampler::Sampler,
        session::{DecodeSession, Limits, StoreLimits, TokenStepper, check_prompt},
    },
    kernels::cuda::{CudaContext, DeviceStream, plan},
    tokenizer::DecodeBatch,
};
use std::path::Path;

type Result<T> = std::result::Result<T, EngError>;

const ALLOC_CACHE_VAR: &str = "FIFI_FLINT_CUDA_ALLOC_CACHE";

fn alloc_cache_selected() -> bool {
    crate::tokenizer::env_allows_default_on(ALLOC_CACHE_VAR)
}

#[cfg(test)]
#[path = "cuda_executor_route_trace_tests.rs"]
mod route_trace_tests;
#[cfg(test)]
#[path = "cuda_executor_tests.rs"]
mod tests;

pub(super) struct FlintExecutor {
    ctx: CudaContext,
    stream: DeviceStream,
    model: FlintExl3Resident,
    sampling: SamplingScratch,
    eos: Vec<u32>,
    vocab: usize,
    context_limit: usize,
    prefill_chunk_size: usize,
}

pub(super) struct FlintSession {
    state: State,
    core: DecodeSession,
}

struct Step<'a> {
    executor: &'a FlintExecutor,
    state: &'a mut State,
}

impl TokenStepper for Step<'_> {
    const OPERATION: &'static str = "flint_cuda";

    fn feed(&mut self, token: u32) -> Result<()> {
        self.executor
            .decoder()?
            .step(self.state, token, |_, _| Ok(()))
    }

    fn feed_rows(&mut self, tokens: &[u32]) -> Result<()> {
        let rows = self.executor.prefill_chunk_size;
        if rows == 1 {
            for &token in tokens {
                self.feed(token)?;
            }
        } else {
            let decoder = self.executor.decoder()?;
            for chunk in tokens.chunks(rows) {
                decoder.feed_rows(self.state, chunk)?;
            }
        }
        Ok(())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.executor
            .decoder()?
            .step(self.state, token, |_, logits| Ok(logits.to_vec()))
    }

    fn next_token(&mut self, token: u32, sampler: &mut Sampler, vocab: usize) -> Result<u32> {
        let executor = self.executor;
        executor.decoder()?.step(self.state, token, |head, logits| {
            executor.sampling.sample(
                &executor.ctx,
                &executor.stream,
                head,
                logits,
                sampler,
                vocab,
            )
        })
    }

    fn fed(&self) -> usize {
        self.state.fed()
    }

    fn reset(&mut self) -> Result<()> {
        self.state.reset(&self.executor.stream)?;
        self.executor.ctx.drain_alloc_cache()?;
        self.executor
            .model
            .recover_expert_cache(&self.executor.ctx, &self.executor.stream)?;
        Ok(())
    }
}

fn work_reserve(plan: &super::kv_plan::KvPlan, chunk_rows: u32, cache_bytes: u64) -> Result<u64> {
    // Six routed + one shared expert retain <16 MiB per extra row at width <=32768.
    ((512 + 16 * (u64::from(chunk_rows) - 1)) << 20)
        .checked_add(State::scratch_bytes(plan)?)
        .and_then(|bytes| bytes.checked_add(cache_bytes))
        .ok_or_else(|| EngError::MemoryPlan("Flint workspace reserve overflows".into()))
}

impl FlintExecutor {
    pub(super) fn open(
        root: &Path,
        options: &FlintOptions,
        eos: Vec<u32>,
        vocab: usize,
    ) -> Result<(Self, FlintSession)> {
        if vocab == 0
            || vocab > VOCAB
            || eos.is_empty()
            || eos.iter().any(|&id| id as usize >= vocab)
        {
            return Err(EngError::Checkpoint(
                "Flint tokenizer vocabulary or EOS set is invalid".into(),
            ));
        }
        let kv = options.plan()?;
        let cache_enabled = alloc_cache_selected();
        let ctx = CudaContext::with_admission_cache(&served_admission(), cache_enabled)?;
        let stream = ctx.stream()?;
        let free = ctx.memory_info()?.free as u64;
        let cache_bytes = CudaContext::alloc_cache_reserve(cache_enabled);
        let reserve = Reserve {
            kv_cache: kv.bytes(kv.storage().row_bytes())? as u64,
            work: work_reserve(&kv, options.prefill_chunk_size, cache_bytes)?,
            headroom: 512 << 20,
        };
        let model = FlintExl3Resident::load_with_prefill(
            &ctx,
            &stream,
            root,
            FlintResidentLimits {
                max_staging_bytes: 2 << 30,
                max_weight_bytes: free,
                reserve,
            },
            options.prefill_chunk_size as usize,
        )?;
        // Weights are now live; recheck the remaining demand before session allocation.
        plan::admit(
            &plan::Demand {
                tensors: 0,
                kv_cache: reserve.kv_cache,
                work: reserve.work,
                headroom: reserve.headroom,
            },
            ctx.memory_info()?.free as u64,
        )?;
        Decoder::new(&ctx, &stream, &model)?;
        let state = State::new(&ctx, kv)?;
        let sampling = SamplingScratch::new(&ctx)?;
        eprintln!(
            "Flint CUDA V/{}-write {}: context={} kv={} kv_bytes={} work_reserve={} headroom={} weights={} alloc_cache={} late_health={}",
            options.kv_profile.to_ascii_uppercase(),
            if options.kv_profile == "q8" {
                "native"
            } else {
                "bridge"
            },
            options.context_limit,
            options.kv_profile,
            reserve.kv_cache,
            reserve.work,
            reserve.headroom,
            model.device_weight_bytes(),
            if cache_enabled { "on" } else { "off" },
            if late_health_selected(std::env::var_os("FIFI_FLINT_LATE_HEALTH").as_deref()) {
                "on"
            } else {
                "off"
            }
        );
        Ok((
            Self {
                ctx,
                stream,
                model,
                sampling,
                eos,
                vocab,
                context_limit: options.context_limit as usize,
                prefill_chunk_size: options.prefill_chunk_size as usize,
            },
            FlintSession {
                state,
                core: DecodeSession::default(),
            },
        ))
    }

    fn decoder(&self) -> Result<Decoder<'_>> {
        Decoder::new(&self.ctx, &self.stream, &self.model)
    }
}

impl ExecutorOps for FlintExecutor {
    type Session = FlintSession;

    fn check_prompt(&self, tokens: usize) -> Result<()> {
        check_prompt(tokens, self.context_limit)
    }

    fn history<'s>(&self, session: &'s FlintSession) -> &'s [u32] {
        session.core.history()
    }

    fn reset(&mut self, session: &mut FlintSession) -> Result<()> {
        session.core.reset(&mut Step {
            executor: self,
            state: &mut session.state,
        })
    }

    fn prefill(&mut self, session: &mut FlintSession, tokens: &[u32]) -> Result<()> {
        session.core.prefill(
            &mut Step {
                executor: self,
                state: &mut session.state,
            },
            tokens,
            self.context_limit,
        )
    }

    fn decode(
        &mut self,
        session: &mut FlintSession,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch> {
        let limits = Limits {
            eos: &self.eos,
            sampler_vocab: self.vocab,
            context_limit: self.context_limit,
        };
        session.core.decode(
            &mut Step {
                executor: self,
                state: &mut session.state,
            },
            generation,
            &limits,
        )
    }

    fn boundary(&mut self, _session: &mut FlintSession) {}

    fn prefill_piece(&self) -> Option<usize> {
        Some(self.prefill_chunk_size)
    }

    fn prefix_limits(&self) -> StoreLimits {
        StoreLimits {
            roots: 0,
            entries: 0,
        }
    }
}
