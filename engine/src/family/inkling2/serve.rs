//! Both reference backends use the shared token session and sampler.

use crate::family::inkling2::switches::Switch;
use std::path::Path;

use super::{
    TextBackend,
    cpu_forward::{CpuForward, PrefixState},
    phase_profile::{Step, Work},
};
use crate::{
    EngError,
    family::{
        executor::{ExecutorOps, GenerationState},
        sampler::Sampler,
        session::{
            DecodeSession, Limits, PrefixStore, SaveKind, StoreLimits, TokenStepper, check_prompt,
            sample_logits,
        },
        speculate::{driver::Speculator, dspark::DsparkDrafter},
    },
    sched::worker::{Engine, ReadyNote, spawn_engine_noted},
    tokenizer::{DecodeBatch, TokenizerBundle},
    weights::dspark_gguf::DsparkGguf,
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
        let mut forward = CpuForward::open(&model_dir, settings.context as usize)?;
        let mut settings = settings;
        let draft_workspace = settings
            .draft
            .as_ref()
            .map_or(0, |draft| draft.workspace_bytes);
        if let Some(draft) = &mut settings.draft {
            if draft.width != forward.hidden() || draft.vocab != vocab_size as usize {
                return Err(super::bad(
                    "draft hidden width or vocabulary differs from the target",
                ));
            }
            if draft.cuda && backend != TextBackend::Cuda {
                return Err(super::bad("CUDA DSpark requires a CUDA Inkling target"));
            }
            forward.prepare_draft(std::mem::take(&mut draft.taps), draft.max_drafts)?;
        }
        let grouped_draft_prefill_requested = backend == TextBackend::Cuda
            && forward.draft_window_enabled()
            && std::env::var("FIFI_INKLING_DSPARK_GROUPED_PREFILL").as_deref() == Ok("1");
        let planning_rows = if forward.draft_window_enabled() && !grouped_draft_prefill_requested {
            1
        } else {
            requested_rows
        };
        let grouped_verify_requested = backend == TextBackend::Cuda
            && forward.draft_window_enabled()
            && Switch::GroupedVerify.on();
        let plan = settings.host_prefill_plan(
            &forward,
            planning_rows,
            draft_workspace,
            grouped_verify_requested,
        )?;
        let prefix = PrefixStore::with_limits(settings.prefix_limits());
        forward.validate_tokenizer(vocab_size, &eos)?;
        let mut forward = match backend {
            TextBackend::Cpu => forward,
            TextBackend::Cuda => {
                stages.begin("CUDA projections");
                cuda_forward(forward, plan.host_reserve)?
            }
        };
        let speculator = settings
            .draft
            .take()
            .map(|draft| start_draft(draft, &mut forward))
            .transpose()?;
        if grouped_verify_requested {
            let enabled = plan.grouped_verify && enable_grouped_verify(&mut forward);
            eprintln!(
                "Inkling CUDA grouped verify {}",
                serde_json::json!({"enabled": enabled, "host_workspace_admitted": plan.grouped_verify})
            );
        }
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if grouped_draft_prefill_requested && plan.prefill_rows > 1 {
            forward.enable_grouped_draft_prefill();
        }
        let prefill_rows = if supports_grouped_prefill(&forward) {
            plan.prefill_rows
        } else {
            1
        };
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        forward.plan_prompt_chunk(prefill_rows);
        if requested_rows > 1 {
            eprintln!(
                "Inkling CUDA prefill rows {}",
                serde_json::json!({
                    "requested": requested_rows,
                    "admitted": prefill_rows,
                    "context": settings.context,
                    "host_reserve_bytes": plan.host_reserve,
                })
            );
        }
        Ok((
            TextExecutor {
                forward,
                eos,
                prefix,
                prefill_rows,
                speculator,
            },
            DecodeSession::default(),
        ))
    })?;
    Ok((engine, bundle))
}

fn start_draft(
    draft: InklingDraft,
    forward: &mut CpuForward,
) -> Result<Speculator<DsparkDrafter>, EngError> {
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    let cap = draft.model.config().final_logit_softcap;
    let drafter = DsparkDrafter::new(draft.model);
    let drafter = if draft.cuda {
        cuda_draft(drafter, forward)?
    } else {
        drafter
    };
    if draft.cuda_head {
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        forward.enable_cuda_draft_head(cap)?;
        #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
        return Err(super::bad("CUDA draft head requires native-cuda on Linux"));
    }
    let speculator = Speculator::new(drafter, draft.max_drafts);
    Ok(if draft.adaptive_drafts {
        eprintln!("Inkling DSpark adaptive draft budget enabled");
        speculator.with_adaptive_drafts()
    } else {
        speculator
    })
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn cuda_draft(mut drafter: DsparkDrafter, forward: &CpuForward) -> Result<DsparkDrafter, EngError> {
    let cuda = forward
        .draft_cuda_context()
        .ok_or_else(|| super::bad("CUDA DSpark requires a GGUF CUDA Inkling target"))?;
    let target_credit = forward.draft_target_reserve_credit();
    drafter.enable_cuda(cuda, forward.context, |draft_credit| {
        super::cuda_projection::state_headroom(target_credit.saturating_add(draft_credit))
    })?;
    Ok(drafter)
}

#[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
fn cuda_draft(_: DsparkDrafter, _: &CpuForward) -> Result<DsparkDrafter, EngError> {
    Err(super::bad(
        "CUDA DSpark requires a 64-bit Linux build with native-cuda",
    ))
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

/// The prompt rows a CUDA prefill group requests unless `FIFI_INKLING_CUDA_PREFILL_ROWS` sets
/// them; the host memory plan admits fewer when the state does not fit.
const CUDA_PREFILL_ROWS: usize = 512;

pub(super) fn prefill_rows(backend: TextBackend) -> Result<usize, EngError> {
    const NAME: &str = "FIFI_INKLING_CUDA_PREFILL_ROWS";
    if backend != TextBackend::Cuda {
        return Ok(1);
    }
    let Some(value) = std::env::var_os(NAME) else {
        return Ok(CUDA_PREFILL_ROWS);
    };
    value
        .to_str()
        .and_then(|text| text.parse::<usize>().ok())
        .filter(|&rows| rows != 0)
        .ok_or_else(|| super::bad(format!("{NAME} must be a positive row count")))
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

fn enable_grouped_verify(forward: &mut CpuForward) -> bool {
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    return forward.enable_grouped_verify();
    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
    {
        let _ = forward;
        false
    }
}

impl CpuForward {
    pub(super) fn sample_target(
        &self,
        logits: &[f32],
        sampler: &mut Sampler,
        sampler_vocab: usize,
    ) -> Result<u32, EngError> {
        let _span = self.phase_span(Work::Sampling);
        sample_logits(Self::OPERATION, logits, sampler, sampler_vocab)
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

    fn next_token(
        &mut self,
        token: u32,
        sampler: &mut Sampler,
        sampler_vocab: usize,
    ) -> Result<u32, EngError> {
        let logits = self.logits(token)?;
        self.sample_target(&logits, sampler, sampler_vocab)
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
    pub(crate) draft: Option<InklingDraft>,
}

pub(crate) struct InklingDraft {
    pub(super) model: DsparkGguf,
    pub(super) cuda: bool,
    pub(super) cuda_head: bool,
    pub(super) adaptive_drafts: bool,
    pub(super) taps: Vec<usize>,
    pub(super) max_drafts: usize,
    pub(super) width: usize,
    pub(super) vocab: usize,
    pub(super) workspace_bytes: u64,
}

/// The longest context whose host state is small enough to start without a host memory plan.
#[cfg(target_os = "linux")]
const UNPLANNED_TEXT_CONTEXT: u32 = 128;

struct HostPrefillPlan {
    prefill_rows: usize,
    host_reserve: u64,
    grouped_verify: bool,
}

impl HostPrefillPlan {
    fn serial(host_reserve: u64) -> Self {
        Self {
            prefill_rows: 1,
            host_reserve,
            grouped_verify: false,
        }
    }
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

    fn host_prefill_plan(
        &self,
        forward: &CpuForward,
        requested_rows: usize,
        draft_workspace: u64,
        grouped_verify_requested: bool,
    ) -> Result<HostPrefillPlan, EngError> {
        let limits = self.prefix_limits();
        // Count every saved state at full size irrespective of shared KV rows.
        let state_bytes = forward
            .host_state_bytes()?
            .checked_mul((1 + limits.roots + limits.entries) as u64)
            .ok_or_else(|| EngError::MemoryPlan("Inkling host state size overflows".into()))?;
        // The draft proposal finishes before the target opens its verification window.
        let verify_bytes = forward.window_workspace_bytes()?;
        let window_bytes = verify_bytes.max(draft_workspace);
        let host_reserve = state_bytes
            .checked_add(window_bytes)
            .ok_or_else(|| EngError::MemoryPlan("Inkling host state size overflows".into()))?;
        let plan = HostPrefillPlan::serial(host_reserve);
        #[cfg(target_os = "linux")]
        {
            const LABEL: &str = "Inkling text state";
            let state_needs_admission =
                self.context > UNPLANNED_TEXT_CONTEXT || self.prefix_reuse || window_bytes != 0;
            if state_needs_admission || requested_rows > 1 {
                let admission = match crate::family::mem::HostFacts::read(LABEL, "Inkling") {
                    Ok(admission) => admission,
                    Err(error) if !state_needs_admission => {
                        eprintln!("Inkling grouped prefill unavailable: {error:?}");
                        return Ok(plan);
                    }
                    Err(error) => return Err(error),
                };
                if state_needs_admission {
                    admission.admit_payload(host_reserve, LABEL)?;
                }
                #[cfg(feature = "native-cuda")]
                {
                    let mut plan = plan;
                    let capacity = admission.payload_spare();
                    if grouped_verify_requested {
                        let extra = forward.grouped_verify_workspace_bytes()?;
                        if let Some(host_reserve) = fit_grouped_verify(
                            state_bytes,
                            draft_workspace,
                            verify_bytes,
                            extra,
                            capacity,
                        ) {
                            plan.grouped_verify = true;
                            plan.host_reserve = host_reserve;
                        }
                    }
                    if requested_rows > 1 {
                        // Pinned and encoded attention staging can outlive prefill.
                        let fixed =
                            super::cuda_projection::gguf::attention::prefill::host_reserve();
                        let base = plan.host_reserve.saturating_add(fixed);
                        let (prefill_rows, host_reserve) = fit_prefill(
                            requested_rows,
                            forward.context,
                            base,
                            forward.prefill_workspace_bytes_per_row()?,
                            capacity,
                        );
                        if prefill_rows > 1 {
                            plan.prefill_rows = prefill_rows;
                            plan.host_reserve = host_reserve;
                        }
                    }
                    return Ok(plan);
                }
            }
        }
        let _ = (requested_rows, grouped_verify_requested);
        Ok(plan)
    }
}

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
fn fit_grouped_verify(
    state: u64,
    draft: u64,
    verify: u64,
    extra: u64,
    capacity: u64,
) -> Option<u64> {
    let reserve = state.checked_add(verify.checked_add(extra)?.max(draft))?;
    (reserve <= capacity).then_some(reserve)
}

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
fn fit_prefill(
    requested: usize,
    context: usize,
    state_bytes: u64,
    bytes_per_row: u64,
    capacity: u64,
) -> (usize, u64) {
    let fit = capacity
        .saturating_sub(state_bytes)
        .checked_div(bytes_per_row)
        .unwrap_or(0);
    let rows = requested
        .min(context)
        .min(usize::try_from(fit).unwrap_or(usize::MAX));
    if rows < 2 {
        return (1, state_bytes);
    }
    // The product is bounded by the capacity above. Include it in the
    // reserve seen by the expert RAM warmer.
    (rows, state_bytes + rows as u64 * bytes_per_row)
}

struct TextExecutor {
    prefix: PrefixStore<PrefixState, ()>,
    forward: CpuForward,
    eos: Vec<u32>,
    prefill_rows: usize,
    speculator: Option<Speculator<DsparkDrafter>>,
}

impl TextExecutor {
    fn finish_profile_step(&self, step: Step, tokens: usize) {
        if let Some(profile) = &self.forward.phase_profile {
            profile.finish(step, self.forward.position(), tokens);
        }
    }

    fn report_profile(&mut self, completed: bool) {
        if let Some(profile) = &mut self.forward.phase_profile
            && let Some(receipt) = profile.take_receipt(completed)
        {
            eprintln!("Inkling GGUF phases {receipt}");
        }
    }

    fn report_speculation(&mut self) {
        if let Some(speculator) = &mut self.speculator
            && speculator.stats.used()
        {
            eprintln!("inkling: {}", std::mem::take(&mut speculator.stats).line());
        }
    }
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
        self.report_profile(false);
        session.reset(&mut self.forward)
    }

    fn prefill(&mut self, session: &mut Self::Session, tokens: &[u32]) -> Result<(), EngError> {
        let step = self
            .forward
            .phase_profile
            .as_ref()
            .map(|profile| profile.begin_prefill(self.forward.position()));
        let context = self.forward.context;
        let result = session.prefill(&mut self.forward, tokens, context);
        if let Some(step) = step {
            self.finish_profile_step(step, 0);
            if result.is_err() {
                self.report_profile(false);
            }
        }
        result
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
        let step = self
            .forward
            .phase_profile
            .as_ref()
            .map(|profile| profile.begin_decode(self.forward.position()));
        let batch = match &mut self.speculator {
            Some(speculator) => speculator.decode(session, &mut self.forward, generation, &limits),
            None => session.decode(&mut self.forward, generation, &limits),
        };
        if let Some(step) = step {
            self.finish_profile_step(step, batch.as_ref().map_or(0, |batch| batch.tokens.len()));
        }
        if (self.speculator.is_some() || self.forward.phase_profile.is_some())
            && (batch.as_ref().is_ok_and(|batch| batch.finish.is_some()) || batch.is_err())
        {
            self.report_speculation();
            self.report_profile(batch.is_ok());
        }
        batch
    }

    fn boundary(&mut self, _session: &mut Self::Session) {
        self.report_speculation();
    }

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
            || self.forward.snapshot(),
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

#[cfg(test)]
mod tests {
    use super::{fit_grouped_verify, fit_prefill};

    #[test]
    fn grouped_prefill_fits_requested_width_and_context() {
        assert_eq!(fit_prefill(512, 768, 100, 20, 20_000), (512, 10_340));
        assert_eq!(fit_prefill(usize::MAX, 128, 100, 20, 20_000), (128, 2_660));
    }

    #[test]
    fn grouped_prefill_reserves_workspace_and_falls_back_under_pressure() {
        assert_eq!(fit_prefill(512, 768, 200, 80, 2_999), (34, 2_920));
        assert_eq!(fit_prefill(512, 768, 200, 80, 359), (1, 200));
        assert_eq!(fit_prefill(512, 768, 200, 80, 199), (1, 200));
    }

    #[test]
    fn grouped_verify_uses_only_the_larger_speculation_workspace() {
        assert_eq!(fit_grouped_verify(200, 80, 60, 40, 300), Some(300));
        assert_eq!(fit_grouped_verify(200, 80, 60, 40, 299), None);
        assert_eq!(fit_grouped_verify(200, 150, 60, 40, 350), Some(350));
        assert_eq!(fit_grouped_verify(200, 0, u64::MAX, 40, u64::MAX), None);
    }
}
