//! Nemotron `ExecutorOps` adapter. It consumes tokens before publishing them.
//!
//! Entry-count weight caching and context capacity do not bound total serving bytes.
//! Retained pristine tensors, live replacements, cached weights, and forward scratch coexist.

use std::{marker::PhantomData, sync::atomic::Ordering};

use super::exec::{NemotronStack, WeightSource};
use crate::{
    EngError,
    family::executor::{ExecutorOps, GenerationState},
    tokenizer::{DecodeBatch, DecodeFinish},
};

pub(crate) struct NemotronSession<S> {
    stack: NemotronStack<S>,
    history: Vec<u32>,
    finished: bool,
}

impl<S: WeightSource> NemotronSession<S> {
    fn clear(&mut self) {
        self.stack.reset();
        self.history.clear();
        self.finished = false;
    }
}

pub(crate) struct NemotronExecutor<S> {
    eos: Vec<u32>,
    vocab_size: u32,
    context_limit: usize,
    source: PhantomData<fn() -> S>,
}

impl<S: WeightSource> NemotronExecutor<S> {
    pub(crate) fn new(
        stack: NemotronStack<S>,
        eos: Vec<u32>,
    ) -> Result<(Self, NemotronSession<S>), EngError> {
        let vocab_size = stack.vocab_size();
        let context_limit = usize::try_from(stack.context_capacity())
            .map_err(|_| bad("context capacity exceeds host limits"))?;
        if eos.is_empty() || eos.iter().any(|&id| id >= vocab_size) {
            return Err(bad("EOS ids must be non-empty and within vocabulary"));
        }
        if stack.state().position() != 0 || stack.logits().is_some() {
            return Err(bad("executor requires a pristine stack"));
        }
        Ok((
            Self {
                eos,
                vocab_size,
                context_limit,
                source: PhantomData,
            },
            NemotronSession {
                stack,
                history: Vec::new(),
                finished: false,
            },
        ))
    }

    fn cancel(
        st: &mut NemotronSession<S>,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch, EngError> {
        st.clear();
        let text = generation.decoder.finish()?;
        Ok(DecodeBatch {
            tokens: Vec::new(),
            text,
            finish: Some(DecodeFinish::Cancelled),
        })
    }

    fn prefill_inner(&self, st: &mut NemotronSession<S>, toks: &[u32]) -> Result<(), EngError> {
        if st.finished || toks.is_empty() {
            return Err(bad(
                "prefill requires an unfinished session and non-empty tokens",
            ));
        }
        if toks.iter().any(|&id| id >= self.vocab_size) {
            return Err(bad("prompt token exceeds vocabulary"));
        }
        let length = st
            .history
            .len()
            .checked_add(toks.len())
            .ok_or_else(|| bad("prompt history length overflow"))?;
        if length >= self.context_limit {
            return Err(EngError::ContextLimit(
                "Nemotron prompt must be shorter than context_limit".into(),
            ));
        }
        st.history.try_reserve(toks.len()).map_err(|_| memory())?;
        st.stack.prefill(toks)?;
        st.history.extend_from_slice(toks);
        Ok(())
    }

    fn decode_inner(
        &self,
        st: &mut NemotronSession<S>,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch, EngError> {
        if st.finished || st.history.is_empty() {
            return Err(bad("decode requires prefill and an unfinished session"));
        }
        if generation.cancel.load(Ordering::SeqCst) {
            return Self::cancel(st, generation);
        }
        if generation.remaining_tokens() == Some(0) || st.history.len() >= self.context_limit {
            let text = generation.decoder.finish()?;
            st.finished = true;
            return Ok(DecodeBatch {
                tokens: Vec::new(),
                text,
                finish: Some(DecodeFinish::Length),
            });
        }
        let token = match st.stack.logits() {
            Some(logits) => generation.sampler.sample(logits)?,
            None => {
                if !generation.sampler.is_greedy() {
                    return Err(bad(
                        "CUDA device sampling requires greedy decoding until the canonical device sampler is accepted",
                    ));
                }
                st.stack
                    .last_prediction()
                    .ok_or_else(|| bad("prefill prediction is missing"))?
            }
        };
        st.history.try_reserve(1).map_err(|_| memory())?;
        let mut tokens = Vec::new();
        tokens.try_reserve_exact(1).map_err(|_| memory())?;
        st.stack.step(token)?;
        if generation.cancel.load(Ordering::SeqCst) {
            return Self::cancel(st, generation);
        }
        let eos = self.eos.contains(&token);
        let finish = if eos {
            Some(DecodeFinish::Stop)
        } else if generation.remaining_tokens() == Some(1)
            || st.history.len() == self.context_limit - 1
        {
            Some(DecodeFinish::Length)
        } else {
            None
        };
        let mut text = if eos {
            String::new()
        } else {
            generation.decoder.push(token)?
        };
        if finish.is_some() {
            text.push_str(&generation.decoder.finish()?);
        }
        st.history.push(token);
        if !eos {
            generation.record_generated(1);
        }
        st.finished = finish.is_some();
        tokens.push(token);
        Ok(DecodeBatch {
            tokens,
            text,
            finish,
        })
    }
}

impl<S: WeightSource> ExecutorOps for NemotronExecutor<S> {
    type Session = NemotronSession<S>;

    fn history<'s>(&self, st: &'s Self::Session) -> &'s [u32] {
        &st.history
    }

    fn reset(&mut self, st: &mut Self::Session) -> Result<(), EngError> {
        st.clear();
        Ok(())
    }

    fn prefill(&mut self, st: &mut Self::Session, toks: &[u32]) -> Result<(), EngError> {
        let result = self.prefill_inner(st, toks);
        if result
            .as_ref()
            .is_err_and(|e| !matches!(e, EngError::Backend { .. }))
        {
            self.reset(st)?;
        }
        result
    }

    fn decode(
        &mut self,
        st: &mut Self::Session,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch, EngError> {
        let result = self.decode_inner(st, generation);
        if result
            .as_ref()
            .is_err_and(|e| !matches!(e, EngError::Backend { .. }))
        {
            self.reset(st)?;
        }
        result
    }

    fn boundary(&mut self, _st: &mut Self::Session) {}
}

fn bad(message: &str) -> EngError {
    EngError::BadRequest(format!("Nemotron executor: {message}"))
}

fn memory() -> EngError {
    EngError::MemoryPlan("Nemotron executor: cannot reserve token publication".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        family::{
            nemotron35::{
                NemotronBlockKind, NemotronGeometry, NemotronMambaGeometry, NemotronMoeGeometry,
                exec::{ExpertIndex, ExpertProjection, NemotronExecSpec, NemotronState},
            },
            sampler::{Sampler, SamplingParams},
        },
        sched::{
            job::{Job, JobEvent},
            worker::spawn_engine,
        },
        tensor::{Dtype, Tensor},
        tokenizer::{IncrementalDecoder, Tokenizer},
    };
    use std::sync::{Arc, Mutex, atomic::AtomicBool};

    #[derive(Default)]
    struct Control {
        dense: usize,
        experts: usize,
        fail: Option<EngError>,
        fail_at: Option<usize>,
        cancel: Option<Arc<AtomicBool>>,
    }

    struct Source {
        control: Arc<Mutex<Control>>,
        uniform: bool,
    }

    fn tensor(shape: &[u32], data: &[f32]) -> Tensor {
        Tensor::try_from_host(
            Dtype::F32,
            shape.to_vec(),
            data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap()
    }

    impl WeightSource for Source {
        fn dense(&mut self, name: &str) -> Result<Tensor, EngError> {
            self.control.lock().unwrap().dense += 1;
            let suffix = name.strip_prefix("language_model.backbone.");
            Ok(match suffix {
                Some("embeddings.weight") => tensor(
                    &[4, 4],
                    &[
                        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
                    ],
                ),
                Some("norm_f.weight" | "layers.0.norm.weight") => tensor(&[4], &[1.; 4]),
                // Row e reads hidden[e]: token t (identity embedding) routes
                // to expert t, so a second prompt token fetches a fresh expert.
                Some("layers.0.mixer.gate.weight") => {
                    tensor(&[2, 4], &[1., 0., 0., 0., 0., 1., 0., 0.])
                }
                Some("layers.0.mixer.gate.e_score_correction_bias") => tensor(&[2], &[0.; 2]),
                Some("layers.0.mixer.fc1_latent_proj.weight") => tensor(&[2, 4], &[0.; 8]),
                Some("layers.0.mixer.fc2_latent_proj.weight") => tensor(&[4, 2], &[0.; 8]),
                None if name == "language_model.lm_head.weight" => {
                    // Independent cycle: input 0 predicts 1, then 2, then 3, then 0.
                    let head = if self.uniform {
                        [0.; 16]
                    } else {
                        [
                            0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
                        ]
                    };
                    tensor(&[4, 4], &head)
                }
                _ => {
                    return Err(EngError::Checkpoint(format!(
                        "unknown fixture weight {name}"
                    )));
                }
            })
        }

        fn expert(
            &mut self,
            layer: u32,
            index: ExpertIndex,
            projection: ExpertProjection,
        ) -> Result<Tensor, EngError> {
            assert_eq!(layer, 0);
            let mut control = self.control.lock().unwrap();
            control.experts += 1;
            if control.fail_at.is_none_or(|at| at == control.experts)
                && let Some(error) = control.fail.take()
            {
                return Err(error);
            }
            if let Some(flag) = control.cancel.take() {
                flag.store(true, Ordering::SeqCst);
            }
            let width = if index == ExpertIndex::Shared { 4 } else { 2 };
            let shape = match projection {
                ExpertProjection::Up => [1, width],
                ExpertProjection::Down => [width, 1],
            };
            Ok(tensor(&shape, &vec![0.; width as usize]))
        }
    }

    fn stack(capacity: u32, control: Arc<Mutex<Control>>, uniform: bool) -> NemotronStack<Source> {
        let spec = NemotronExecSpec::new(
            NemotronGeometry {
                hidden_size: 4,
                vocab_size: 4,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                head_dim: 4,
                max_position_embeddings: 8,
                layer_norm_epsilon: 1.0,
            },
            NemotronMambaGeometry {
                groups: 1,
                state_size: 1,
                conv_kernel: 2,
                heads: 1,
                head_dim: 4,
                expand: 1,
                inner_size: 4,
                chunk_size: 1,
            },
            NemotronMoeGeometry {
                intermediate_size: 1,
                moe_intermediate_size: 1,
                latent_size: 2,
                shared_expert_intermediate_size: 1,
                routed_experts: 2,
                shared_experts: 1,
                experts_per_token: 1,
                n_group: 1,
                n_groups: 1,
                topk_group: 1,
                norm_topk_prob: true,
                routed_scaling_factor: 1.0,
            },
            vec![NemotronBlockKind::Moe],
        )
        .unwrap();
        let state = NemotronState::new(&spec, capacity).unwrap();
        NemotronStack::new(spec, Source { control, uniform }, state).unwrap()
    }

    fn adapter(
        capacity: u32,
        eos: &[u32],
        uniform: bool,
    ) -> (
        NemotronExecutor<Source>,
        NemotronSession<Source>,
        Arc<Mutex<Control>>,
    ) {
        let control = Arc::new(Mutex::new(Control::default()));
        let (exec, st) =
            NemotronExecutor::new(stack(capacity, control.clone(), uniform), eos.to_vec()).unwrap();
        (exec, st, control)
    }

    fn generation() -> GenerationState {
        let tokenizer = Tokenizer::from_bytes(br#"{"version":"1.0","added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"a":0,"b":1,"c":2,"d":3},"unk_token":"a"}}"#).unwrap();
        GenerationState::new(IncrementalDecoder::new(&tokenizer))
    }

    fn empty(st: &NemotronSession<Source>, exec: &NemotronExecutor<Source>) {
        assert!(exec.history(st).is_empty());
        assert_eq!(st.stack.state().position(), 0);
        assert!(st.stack.logits().is_none());
        assert_eq!(st.stack.last_prediction(), None);
    }

    #[test]
    fn first_token_uses_prefill_logits_and_each_publication_is_consumed() {
        let (mut exec, mut st, _) = adapter(8, &[3], false);
        exec.prefill(&mut st, &[0]).unwrap();
        let mut job = generation();
        let first = exec.decode(&mut st, &mut job).unwrap();
        assert_eq!(
            first,
            DecodeBatch {
                tokens: vec![1],
                text: "b".into(),
                finish: None
            }
        );
        assert_eq!(exec.history(&st), &[0, 1]);
        assert_eq!(st.stack.state().position(), 2);
        assert_eq!(st.stack.last_prediction(), Some(2));
        assert_eq!(exec.decode(&mut st, &mut job).unwrap().tokens, [2]);
        assert_eq!(exec.history(&st), &[0, 1, 2]);
    }

    #[test]
    fn seeded_sampling_and_filters_use_the_job_sampler() {
        for (top_p, top_k, expected) in [(1.0, 0, 3), (1.0, 2, 1), (0.0, 0, 0)] {
            let (mut exec, mut st, _) = adapter(8, &[2], true);
            exec.prefill(&mut st, &[0]).unwrap();
            let mut job = generation();
            job.sampler = Sampler::new(SamplingParams::new(1.0, top_p, top_k).unwrap(), 0);
            let batch = exec.decode(&mut st, &mut job).unwrap();
            assert_eq!(batch.tokens, [expected], "top_p={top_p}, top_k={top_k}");
            assert_eq!(exec.history(&st), &[0, expected]);
        }
    }

    #[test]
    fn zero_and_one_output_limits_do_not_overconsume() {
        for limit in [0, 1] {
            let (mut exec, mut st, control) = adapter(8, &[3], false);
            exec.prefill(&mut st, &[0]).unwrap();
            st.stack.clear_expert_cache();
            let before = control.lock().unwrap().experts;
            let mut job = generation().with_max_tokens(limit);
            let batch = exec.decode(&mut st, &mut job).unwrap();
            assert_eq!(batch.finish, Some(DecodeFinish::Length));
            assert_eq!(batch.tokens, if limit == 0 { vec![] } else { vec![1] });
            assert_eq!(st.stack.state().position(), limit + 1);
            assert_eq!(job.remaining_tokens(), Some(0));
            assert_eq!(control.lock().unwrap().experts, before + limit as usize * 4);
        }
    }

    #[test]
    fn every_eos_is_consumed_but_not_detokenized_or_counted_against_budget() {
        for (prompt, eos) in [(0, 1), (1, 2)] {
            let (mut exec, mut st, _) = adapter(2, &[1, 2], false);
            exec.prefill(&mut st, &[prompt]).unwrap();
            let mut job = generation().with_max_tokens(1);
            let batch = exec.decode(&mut st, &mut job).unwrap();
            assert_eq!(
                batch,
                DecodeBatch {
                    tokens: vec![eos],
                    text: String::new(),
                    finish: Some(DecodeFinish::Stop)
                }
            );
            assert_eq!(exec.history(&st), &[prompt, eos]);
            assert_eq!(st.stack.state().position(), 2);
            assert_eq!(job.remaining_tokens(), Some(1));
        }
    }

    #[test]
    fn final_context_position_is_available_for_a_consumed_output() {
        let (mut exec, mut st, _) = adapter(3, &[3], false);
        exec.prefill(&mut st, &[0]).unwrap();
        exec.prefill(&mut st, &[1]).unwrap();
        let batch = exec.decode(&mut st, &mut generation()).unwrap();
        assert_eq!(batch.tokens, [2]);
        assert_eq!(batch.finish, Some(DecodeFinish::Length));
        assert_eq!(exec.history(&st), &[0, 1, 2]);
        assert_eq!(st.stack.state().position(), 3);
    }

    #[test]
    fn cancellation_before_decode_and_during_forward_resets_without_publication() {
        for during in [false, true] {
            let (mut exec, mut st, control) = adapter(8, &[3], false);
            exec.prefill(&mut st, &[0]).unwrap();
            st.stack.clear_expert_cache();
            let mut job = generation().with_max_tokens(2);
            if during {
                control.lock().unwrap().cancel = Some(job.cancel.clone());
            } else {
                job.cancel.store(true, Ordering::SeqCst);
            }
            let batch = exec.decode(&mut st, &mut job).unwrap();
            assert_eq!(
                batch,
                DecodeBatch {
                    tokens: vec![],
                    text: String::new(),
                    finish: Some(DecodeFinish::Cancelled)
                }
            );
            empty(&st, &exec);
            assert_eq!(job.remaining_tokens(), Some(2));
            exec.prefill(&mut st, &[0]).unwrap();
            assert_eq!(exec.decode(&mut st, &mut generation()).unwrap().tokens, [1]);
        }
    }

    #[test]
    fn failed_forward_preserves_error_and_resets_without_publication() {
        let (mut exec, mut st, control) = adapter(8, &[3], false);
        exec.prefill(&mut st, &[0]).unwrap();
        st.stack.clear_expert_cache();
        control.lock().unwrap().fail =
            Some(EngError::Checkpoint("injected forward failure".into()));
        let mut job = generation().with_max_tokens(1);
        assert_eq!(
            exec.decode(&mut st, &mut job),
            Err(EngError::Checkpoint("injected forward failure".into()))
        );
        assert_eq!(job.remaining_tokens(), Some(1));
        assert_eq!(job.decoder.finish().unwrap(), "");
        empty(&st, &exec);
        exec.prefill(&mut st, &[0]).unwrap();
        assert_eq!(exec.decode(&mut st, &mut generation()).unwrap().tokens, [1]);
    }

    #[test]
    fn partial_chunk_failure_discards_all_previous_prompt_consumption() {
        let (mut exec, mut st, control) = adapter(8, &[3], false);
        {
            let mut control = control.lock().unwrap();
            // Fail on the fifth expert fetch: token 0 publishes first (its
            // four projections, calls 1-4); token 1's forward then re-fetches
            // its routed pair (routed experts stream every step — the stack
            // caches only shared), so call 5 fails after published state
            // exists to discard. The per-token gate rows keep the routing
            // deterministic rather than tie-broken.
            control.fail = Some(EngError::Checkpoint("second prompt token failed".into()));
            control.fail_at = Some(5);
        }
        assert_eq!(
            exec.prefill(&mut st, &[0, 1]),
            Err(EngError::Checkpoint("second prompt token failed".into()))
        );
        empty(&st, &exec);
        exec.prefill(&mut st, &[0]).unwrap();
        assert_eq!(exec.decode(&mut st, &mut generation()).unwrap().tokens, [1]);
    }

    #[test]
    fn backend_failure_is_unchanged_and_does_not_reset_the_failed_session() {
        let (mut exec, mut st, control) = adapter(8, &[3], false);
        exec.prefill(&mut st, &[0]).unwrap();
        st.stack.clear_expert_cache();
        control.lock().unwrap().fail = Some(EngError::Backend {
            operation: "fixture",
            message: "native failed".into(),
        });
        assert_eq!(
            exec.decode(&mut st, &mut generation()),
            Err(EngError::Backend {
                operation: "fixture",
                message: "native failed".into()
            })
        );
        assert_eq!(exec.history(&st), &[0]);
        assert_eq!(st.stack.state().position(), 1);
    }

    #[test]
    fn invalid_prefill_and_call_order_clear_the_entire_session() {
        let (mut exec, mut st, _) = adapter(3, &[3], false);
        assert!(matches!(
            exec.decode(&mut st, &mut generation()),
            Err(EngError::BadRequest(_))
        ));
        for tokens in [vec![], vec![4], vec![0, 1]] {
            exec.prefill(&mut st, &[0]).unwrap();
            assert!(exec.prefill(&mut st, &tokens).is_err());
            empty(&st, &exec);
        }
        exec.prefill(&mut st, &[0]).unwrap();
        exec.decode(&mut st, &mut generation().with_max_tokens(0))
            .unwrap();
        assert!(exec.decode(&mut st, &mut generation()).is_err());
        empty(&st, &exec);
    }

    #[test]
    fn decoder_failure_after_forward_does_not_publish_history_or_budget() {
        let (mut exec, mut st, _) = adapter(8, &[3], false);
        exec.prefill(&mut st, &[0]).unwrap();
        let tokenizer = Tokenizer::from_bytes(
            br#"{"version":"1.0","model":{"type":"WordLevel","vocab":{"a":0},"unk_token":"a"}}"#,
        )
        .unwrap();
        let mut job = GenerationState::new(IncrementalDecoder::new(&tokenizer)).with_max_tokens(1);
        assert!(matches!(
            exec.decode(&mut st, &mut job),
            Err(EngError::Checkpoint(_))
        ));
        empty(&st, &exec);
        assert_eq!(job.remaining_tokens(), Some(1));
    }

    #[test]
    fn reset_reuses_weights_and_matches_fresh_execution_without_prefix_reuse() {
        let (mut exec, mut st, control) = adapter(8, &[3], false);
        exec.prefill(&mut st, &[0, 1]).unwrap();
        exec.decode(&mut st, &mut generation()).unwrap();
        let weights = control.lock().unwrap().dense;
        exec.reset(&mut st).unwrap();
        empty(&st, &exec);
        assert!(
            !exec
                .save_prefix(&st, &[0], crate::family::session::SaveKind::Tip)
                .unwrap()
        );
        assert_eq!(exec.restore_prefix(&mut st, &[0, 1]).unwrap(), 0);
        exec.boundary(&mut st);
        exec.prefill(&mut st, &[1]).unwrap();
        let actual = exec.decode(&mut st, &mut generation()).unwrap();
        let (mut fresh_exec, mut fresh, _) = adapter(8, &[3], false);
        fresh_exec.prefill(&mut fresh, &[1]).unwrap();
        assert_eq!(
            actual,
            fresh_exec.decode(&mut fresh, &mut generation()).unwrap()
        );
        assert_eq!(control.lock().unwrap().dense, weights);
    }

    #[test]
    fn worker_repeated_jobs_reuse_weights_and_usage_counts_consumed_eos() {
        let control = Arc::new(Mutex::new(Control::default()));
        let source_control = control.clone();
        let engine = spawn_engine(1, move || {
            NemotronExecutor::new(stack(8, source_control, false), vec![2])
        })
        .unwrap();
        let mut weights = 0;
        for _ in 0..2 {
            let (job, mut handle) = Job::new(vec![0], generation().with_max_tokens(2)).unwrap();
            engine.handle().try_submit(job).unwrap();
            let mut batches = Vec::new();
            let mut usage = None;
            while let Some(event) = handle.events.blocking_recv() {
                match event {
                    JobEvent::Batch(batch) => batches.push(batch),
                    JobEvent::Finished(value) => usage = Some(value),
                    JobEvent::Error(error) => panic!("job failed: {error:?}"),
                }
            }
            assert_eq!(batches[0].tokens, [1]);
            assert_eq!(batches[0].text, "b");
            assert_eq!(
                batches[1],
                DecodeBatch {
                    tokens: vec![2],
                    text: String::new(),
                    finish: Some(DecodeFinish::Stop)
                }
            );
            let usage = usage.unwrap();
            assert_eq!(usage.prompt_tokens, 1);
            assert_eq!(
                usage.completion_tokens, 2,
                "worker usage includes consumed EOS"
            );
            assert_eq!(usage.reused_tokens, 0);
            let current = control.lock().unwrap().dense;
            if weights != 0 {
                assert_eq!(current, weights);
            }
            weights = current;
        }
        engine.shutdown();
    }
}
