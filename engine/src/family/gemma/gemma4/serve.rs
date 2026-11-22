//! The serving wrap of the resident Bolt executor.
//!
//! One session serves a request at a time; `max_seqs` above one (the default plans it from free
//! memory) serves several sessions in one decode window. The token side is the shared `DecodeSession`
//! (`family::session`): it owns the committed tokens, the pending token and
//! the finish rules (cancel, length, end of sequence, context limit). The
//! model side is a `TokenStepper` that feeds prompt rows cache-only and reads
//! the capped logits of a decode step; host sampling draws from that row.
//! Prompt rows run the cache-writing layer prefix only (the shared layers
//! after the prefix write no cache cells and feed nothing but the final norm);
//! the pending token's decode runs the head.

use super::config;
use crate::{EngError, capability::probe::CompiledBackends, family::gguf_select};
use std::fs;
use std::path::Path;

/// Build eligibility only. A CUDA build still has to pass the Bolt checkpoint checks at load.
pub(crate) fn check_backend(model: &Path, backends: &CompiledBackends) -> Result<(), EngError> {
    if backends.cuda {
        Ok(())
    } else {
        Err(cuda_required(model))
    }
}

pub(crate) fn cuda_required(model: &Path) -> EngError {
    EngError::BadRequest(format!(
        "gemma4 serves only the Nemotron Bolt E4B on the native-cuda build; \
         this build has no CUDA engine (model directory {})",
        model.display()
    ))
}

/// Shared Bolt identity gate, not a tensor or memory check. A supplied config is
/// authoritative; only a missing config lets the selected GGUF header stand in for it.
pub(crate) fn check_checkpoint(model: &Path) -> Result<(), EngError> {
    let directory = gguf_select::directory_of(model);
    let result = bolt_configured(&directory).or_else(|error| {
        if directory.join("config.json").exists() {
            return Err(error);
        }
        gguf_select::check_header_identity(
            model,
            "gemma4",
            crate::family::spec::NEMOTRON_BOLT_E4B_GGUF_KEYS,
            Some(crate::family::spec::NEMOTRON_BOLT_E4B_GGUF_SOFTCAP),
        )
    });
    result.map_err(|cause| {
        EngError::BadRequest(format!(
            "gemma4 serves only the Nemotron Bolt E4B on the native-cuda build; \
             the model directory {} failed the Bolt configuration check: {}",
            directory.display(),
            cause.describe()
        ))
    })
}

fn bolt_configured(model_dir: &Path) -> Result<(), EngError> {
    let json = fs::read(model_dir.join("config.json"))
        .map_err(|error| EngError::Checkpoint(format!("gemma4 config.json: {error}")))?;
    config::check_bolt_config(&json)
}

#[cfg(feature = "native-cuda")]
fn bad(message: &str) -> EngError {
    EngError::BadRequest(format!("gemma4 serve: {message}"))
}

#[cfg(test)]
mod tests {
    use super::bolt_configured;
    use crate::family::gguf_select::candidates;

    const BOLT_CONFIG: &[u8] = include_bytes!("../../../../testdata/gemma4/bolt-config.json");

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("gemma4-serve-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    #[test]
    fn a_model_path_with_no_gguf_refuses_and_several_are_ranked() {
        let dir = scratch("none");
        assert!(candidates(&dir).is_err());
        std::fs::write(dir.join("bolt.gguf"), b"xx").expect("write");
        assert_eq!(candidates(&dir).expect("one").1.len(), 1);
        std::fs::write(dir.join("other.gguf"), b"xxxx").expect("write");
        let (_, files) = candidates(&dir).expect("two");
        assert_eq!(files[0].path, dir.join("other.gguf"));
        assert_eq!(files[1].path, dir.join("bolt.gguf"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_config_gate_reads_the_directory_of_a_gguf_file_path() {
        let dir = scratch("file-path");
        std::fs::write(dir.join("config.json"), BOLT_CONFIG).expect("write config");
        std::fs::write(dir.join("bolt-q4.gguf"), b"weights").expect("write weights");
        std::fs::write(dir.join("bolt-q8.gguf"), b"weights-bigger").expect("write weights");
        let (found, files) = candidates(&dir.join("bolt-q4.gguf")).expect("file path");
        assert_eq!(found, dir);
        assert_eq!(files.len(), 1);
        bolt_configured(&found).expect("the Bolt config gate passes beside the file");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_bare_bolt_header_passes_but_cannot_override_an_invalid_config() {
        use crate::family::spec::{NEMOTRON_BOLT_E4B_GGUF_KEYS, NEMOTRON_BOLT_E4B_GGUF_SOFTCAP};
        use crate::tokenizer::gguf::tests::{
            gguf_file, kv_f32_for_tests, kv_string_for_tests, kv_u32_for_tests,
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("bolt.gguf");
        let mut metadata = vec![kv_string_for_tests("general.architecture", "gemma4")];
        metadata.extend(
            NEMOTRON_BOLT_E4B_GGUF_KEYS
                .iter()
                .map(|(key, value)| kv_u32_for_tests(key, *value)),
        );
        let (softcap, value) = NEMOTRON_BOLT_E4B_GGUF_SOFTCAP;
        metadata.push(kv_f32_for_tests(softcap, value));
        std::fs::write(&file, gguf_file(&metadata)).expect("write Bolt header");
        for model in [dir.path(), &file] {
            super::check_checkpoint(model).expect("accept the Bolt header");
        }

        std::fs::write(dir.path().join("config.json"), "{").expect("write invalid config");
        let error = super::check_checkpoint(&file).expect_err("do not fall back to the header");
        assert!(error.describe().contains("configuration JSON"), "{error:?}");
    }

    #[test]
    fn bolt_configured_reads_the_pinned_config_shape() {
        let dir = scratch("config");
        assert!(bolt_configured(&dir).is_err());
        std::fs::write(dir.join("config.json"), BOLT_CONFIG).expect("write");
        assert!(bolt_configured(&dir).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_bolt_directory_passes_the_config_gate_and_finds_its_gguf() {
        // Checks the two helpers start_gemma4 dispatches on: bolt_configured
        // accepts the pinned Bolt config, and the candidates are the single weights
        // file. The server's engine start itself needs a device and is not run here.
        let dir = scratch("dispatch");
        std::fs::write(dir.join("config.json"), BOLT_CONFIG).expect("write config");
        std::fs::write(dir.join("bolt-e4b-q8_0.gguf"), b"weights").expect("write weights");
        bolt_configured(&dir).expect("the Bolt config gate passes");
        let (_, files) = candidates(&dir).expect("the weights file is found");
        assert_eq!(files[0].path, dir.join("bolt-e4b-q8_0.gguf"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(feature = "native-cuda")]
mod device {
    use std::path::Path;

    use super::super::budget;
    use super::super::config::MaxSeqs;
    use super::super::kv_tier::{self, BoltFamily, TIER_LABEL};
    use super::super::resident::bolt::{self, Bolt, BoltSeq, BoltSnapshot, VOCAB};
    use crate::{
        EngError,
        family::{
            engine_announce::{Announcement, announce_failure, await_announced},
            executor::{ExecutorOps, GenerationState},
            kv_disk::{
                self as kv_cache, DiskOwner, DiskParts, HostStoreDisk, KvCacheSettings,
                PUMP_EMIT_BYTES,
            },
            qwen::flash_next::factory::StartedEngine,
            seq_plan::{SeqInputs, plan_seqs},
            session::{
                DecodeSession, Limits, PrefixIdentity, PrefixStore, SaveKind, ShadowLcp,
                StoreLimits, TokenStepper,
            },
            window::{self, WindowRows},
        },
        sched::{
            job::{PrefixHit, PrefixReport, PrefixTier},
            worker::{ReadyNote, spawn_engine_noted},
        },
        tokenizer::DecodeBatch,
    };

    /// The token side is the shared `DecodeSession`; `pos` is the model cache
    /// position (the number of tokens fed); `seq` holds the KV planes and position
    /// words of this session. The finish rules live in `family::session`, not here.
    pub(crate) struct BoltSession {
        core: DecodeSession,
        pos: usize,
        seq: BoltSeq,
    }

    pub(crate) struct BoltServe {
        model: Bolt,
        eos: Vec<u32>,
        /// One-tip prefix store (no roots, no periodic points). An entry is a copy of the saving
        /// session's cache cells below its position, held in host memory: a restore writes it
        /// into any session, so reuse works whatever the number of sessions and whichever
        /// pooled session serves the next request. Host memory, not a device slab, because the
        /// device memory belongs to the sessions' planes: an entry is at most one session's
        /// cache (952 MiB at 32,768 cells, 30,464 bytes a position) and costs a transfer over
        /// the bus (about 125 MB for a 4,000-token prefix), where recomputing the prefix costs
        /// its whole prefill.
        prefix: PrefixStore<BoltSnapshot, PrefixIdentity>,
        /// The load-time identity half of the prefix path: constant for one loaded Bolt, so
        /// live behaviour is unchanged; a persisted or cross-step store refuses entries another
        /// load wrote. Empty for now: the Bolt loader reads the GGUF header digest at open
        /// but does not retain it, and threading it means widening `bolt::load`'s return; this
        /// seam takes it when that return widens.
        identity: PrefixIdentity,
        /// The sessions the engine serves at once; the worker reads it after start.
        max_seqs: usize,
        /// The prompt rows the worker feeds per tick while sessions share the model.
        prompt_rows: usize,
        /// The kind of the entry the last restore used, the prompt tokens it shared beyond the
        /// restored prefix and the shadow lookup, for `prefix_report`.
        prefix_hit: PrefixHit,
        prefix_uncached: u32,
        prefix_shadow: Option<ShadowLcp>,
        /// The on-disk tier behind the host store, when it runs.
        disk: Option<HostStoreDisk>,
        /// The tier facts of the last restore, for `prefix_report`.
        prefix_tier: PrefixTier,
    }

    /// The store's explicit one-tip shape (no roots, no periodic points),
    /// shared by the executor, the start wiring and the tests.
    const ONE_TIP: StoreLimits = StoreLimits {
        roots: 0,
        entries: 1,
    };

    /// The model side of one session: feeds prompt rows cache-only and reads
    /// the capped logits of a decode step. The session never holds it.
    pub(crate) struct Step<'a> {
        model: &'a mut Bolt,
        seq: &'a BoltSeq,
        pos: &'a mut usize,
    }

    impl TokenStepper for Step<'_> {
        const OPERATION: &'static str = "gemma4_bolt";

        fn feed(&mut self, token: u32) -> Result<(), EngError> {
            self.model
                .prefill_batch(self.seq, &[token], *self.pos, false)?;
            *self.pos += 1;
            Ok(())
        }

        fn feed_rows(&mut self, tokens: &[u32]) -> Result<(), EngError> {
            for chunk in tokens.chunks(Bolt::PREFILL_BATCH) {
                self.model
                    .prefill_batch(self.seq, chunk, *self.pos, false)?;
                *self.pos += chunk.len();
            }
            Ok(())
        }

        fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
            self.model.decode(self.seq, token, *self.pos)?;
            *self.pos += 1;
            self.model.capped_logits()
        }

        /// The next token is drawn on the device from the capped logits: one word leaves the card.
        fn next_token(
            &mut self,
            token: u32,
            sampler: &mut crate::family::sampler::Sampler,
            sampler_vocab: usize,
        ) -> Result<u32, EngError> {
            if sampler_vocab != VOCAB {
                return Err(EngError::Backend {
                    operation: Self::OPERATION,
                    message: "the sampler vocabulary is not the head width".to_owned(),
                });
            }
            if sampler.capture_armed() {
                let logits = self.logits(token)?;
                return crate::family::decode_step::draw::host_draw(
                    Self::OPERATION,
                    &logits,
                    sampler,
                    sampler_vocab,
                );
            }
            let drawn = self
                .model
                .decode_draw(self.seq, token, *self.pos, sampler)?;
            *self.pos += 1;
            sampler.record_device_draw(drawn);
            Ok(drawn)
        }

        fn fed(&self) -> usize {
            *self.pos
        }

        fn reset(&mut self) -> Result<(), EngError> {
            // A replay from position 0 overwrites every cache cell it uses.
            *self.pos = 0;
            Ok(())
        }
    }

    impl ExecutorOps for BoltServe {
        type Session = BoltSession;

        fn max_seqs(&self) -> usize {
            self.max_seqs
        }

        /// Bolt's batching is measured: idle prompts take whole pieces, and a pooled session that
        /// idles for the default interval is released.
        fn batch_policy(&self) -> crate::family::executor::BatchPolicy {
            budget::bolt_batch_policy()
        }

        /// Whether the device has room for one more session's planes and the headroom. A probe
        /// that cannot read the device says yes, and the allocation decides.
        fn can_admit_session(&self) -> bool {
            let Some(cost) = budget::bolt_seq_cost() else {
                return true;
            };
            let per_session = cost
                .per_position
                .saturating_mul(self.model.cells() as u64)
                .saturating_add(cost.fixed)
                .saturating_add(budget::BOLT_HEADROOM_BYTES);
            self.model
                .ctx
                .memory_info()
                .map_or(true, |info| info.free as u64 >= per_session)
        }

        /// The entries live in host memory and the sessions on the device: dropping an entry does
        /// not make room for a session.
        fn eviction_frees_session_memory(&self) -> bool {
            false
        }

        fn prompt_rows_per_tick(&self) -> Option<usize> {
            (self.max_seqs > 1).then_some(self.prompt_rows)
        }

        fn new_session(&mut self) -> Result<Self::Session, EngError> {
            Ok(BoltSession {
                core: DecodeSession::default(),
                pos: 0,
                seq: self.model.new_seq()?,
            })
        }

        fn decode_window(
            &mut self,
            sessions: &mut [&mut Self::Session],
            generations: &mut [&mut GenerationState],
        ) -> Result<Vec<Result<crate::tokenizer::DecodeBatch, EngError>>, EngError> {
            window::decode_window(self, sessions, generations)
        }

        fn history<'s>(&self, st: &'s Self::Session) -> &'s [u32] {
            st.core.history()
        }

        fn reset(&mut self, st: &mut Self::Session) -> Result<(), EngError> {
            // The restore facts of the report do not survive a job.
            self.prefix_hit = PrefixHit::Miss;
            self.prefix_uncached = 0;
            self.prefix_shadow = None;
            self.prefix_tier = PrefixTier::default();
            st.core.reset(&mut Step {
                model: &mut self.model,
                seq: &st.seq,
                pos: &mut st.pos,
            })
        }

        fn check_prompt(&self, prompt_tokens: usize) -> Result<(), EngError> {
            crate::family::session::check_prompt(prompt_tokens, self.model.cells())
        }

        /// The store's explicit one-tip shape, so the worker's save-point
        /// planning sees what the store really keeps.
        fn prefix_limits(&self) -> StoreLimits {
            ONE_TIP
        }

        fn prefill(&mut self, st: &mut Self::Session, toks: &[u32]) -> Result<(), EngError> {
            let cells = self.model.cells();
            st.core.prefill(
                &mut Step {
                    model: &mut self.model,
                    seq: &st.seq,
                    pos: &mut st.pos,
                },
                toks,
                cells,
            )
        }

        fn decode(
            &mut self,
            st: &mut Self::Session,
            generation: &mut GenerationState,
        ) -> Result<DecodeBatch, EngError> {
            let limits = Limits {
                eos: &self.eos,
                sampler_vocab: VOCAB,
                context_limit: self.model.cells(),
            };
            st.core.decode(
                &mut Step {
                    model: &mut self.model,
                    seq: &st.seq,
                    pos: &mut st.pos,
                },
                generation,
                &limits,
            )
        }

        fn boundary(&mut self, _st: &mut Self::Session) {}

        /// Saves a copy of the session's cache cells below its position as an entry of the
        /// store. The session rules (the pending token, the strict key match) are
        /// `DecodeSession`'s; the model state at the save point is the cells `0..pos`. A host
        /// that cannot hold the copy refuses with a memory-plan error, which the store treats as
        /// a denied save.
        fn save_prefix(
            &mut self,
            st: &Self::Session,
            key: &[u32],
            kind: SaveKind,
        ) -> Result<bool, EngError> {
            let identity = self.identity.clone();
            let model = &self.model;
            let saved =
                st.core
                    .save_prefix(&mut self.prefix, key, identity, st.pos, kind, || {
                        model.snapshot_seq(&st.seq, st.pos)
                    })?;
            if saved {
                self.spill_after_save();
            }
            Ok(saved)
        }

        /// Writes the saved entry into the (reset) session and moves its position to the entry's.
        /// The cells above stay as they are; every read is bounded by the position word and the
        /// tail prefill that follows overwrites them. Any session can take any entry.
        fn restore_prefix(
            &mut self,
            st: &mut Self::Session,
            prompt: &[u32],
        ) -> Result<usize, EngError> {
            // The host store missed or holds a shorter prefix: the disk may hold a longer one.
            self.promote_from_disk((), prompt);
            let identity = self.identity.clone();
            let model = &self.model;
            let BoltSession { core, pos, seq } = st;
            let covered = core.restore_prefix(&mut self.prefix, prompt, identity, |saved| {
                model.restore_seq(seq, saved)?;
                *pos = saved.fed();
                Ok(())
            })?;
            // The restore facts of the report, read now: the store moves on with the job's saves.
            self.prefix_hit = self.prefix.restored_kind().into();
            self.prefix_uncached = self.prefix.uncached_shared(prompt, covered);
            self.prefix_shadow = self.prefix.shadow_report();
            self.prefix_tier = self.restore_tier_of(covered, None);
            Ok(covered)
        }

        /// The entries the store holds, the host bytes of their copies, and the facts of the
        /// last restore, for the request log and the usage report.
        fn prefix_report(&self) -> Option<PrefixReport> {
            let stored = self.prefix.roots_len() + self.prefix.entries_len();
            Some(PrefixReport {
                entries: u32::try_from(stored).unwrap_or(u32::MAX),
                bytes: self
                    .prefix
                    .snapshots()
                    .map(|snapshot| snapshot.bytes() as u64)
                    .sum(),
                hit: self.prefix_hit,
                uncached_shared: self.prefix_uncached,
                shadow: self.prefix_shadow,
                restore_drops: self.prefix.restore_drop_count(),
                evictions: self.prefix.eviction_count(),
                save_side: self.prefix.save_side_totals(),
                tier: self.prefix_tier,
            })
        }

        /// Whether the disk tier has saves the engine thread should advance while it waits.
        fn background_pending(&self) -> bool {
            DiskOwner::disk_pending(self)
        }

        /// Copies a bounded part of the pending saves to the writer: a few MiB per call, which a
        /// batching worker makes between ticks and an idle one every 2 ms.
        fn background_step(&mut self) {
            self.pump_disk(PUMP_EMIT_BYTES);
        }

        /// A graceful stop: writes the tips that exist only in host memory, within the tier's
        /// bound.
        fn background_drain(&mut self) {
            self.drain_disk();
        }
    }

    impl DiskOwner for BoltServe {
        type Snapshot = BoltSnapshot;
        type Path = PrefixIdentity;
        type Family<'a> = BoltFamily<'a>;
        type Restore = ();

        fn disk_ref(&self) -> Option<&HostStoreDisk> {
            self.disk.as_ref()
        }

        fn disk_mut(&mut self) -> Option<&mut HostStoreDisk> {
            self.disk.as_mut()
        }

        fn disk_parts(&mut self, _restore: Option<()>) -> Option<DiskParts<'_, BoltFamily<'_>>> {
            let model = &self.model;
            let disk = self.disk.as_mut()?;
            Some(DiskParts {
                disk,
                family: BoltFamily {
                    layer_bytes: Box::new(|cells| model.snapshot_layer_bytes(cells)),
                    context_cells: model.cells(),
                },
                store: &mut self.prefix,
            })
        }

        fn restore_path(&self) -> PrefixIdentity {
            self.identity.clone()
        }
    }

    impl WindowRows for BoltServe {
        type Stepper<'a> = Step<'a>;
        /// The planes are preallocated, so a position costs no growth.
        type Permit = ();
        const OPERATION: &'static str = "gemma4_bolt";

        fn split<'a>(
            &'a mut self,
            session: &'a mut BoltSession,
        ) -> (&'a mut DecodeSession, Step<'a>, Limits<'a>) {
            let BoltSession { core, pos, seq } = session;
            let cells = self.model.cells();
            (
                core,
                Step {
                    model: &mut self.model,
                    seq,
                    pos,
                },
                Limits {
                    eos: &self.eos,
                    sampler_vocab: VOCAB,
                    context_limit: cells,
                },
            )
        }

        fn check_row(&self, session: &BoltSession, token: u32) -> Result<(), EngError> {
            let cells = self.model.cells();
            if session.pos >= cells {
                return Err(EngError::ContextLimit(format!(
                    "position {} reaches the {cells} cell context",
                    session.pos
                )));
            }
            if token as usize >= VOCAB {
                return Err(EngError::BadRequest(
                    "token is outside the embedding table".to_owned(),
                ));
            }
            Ok(())
        }

        fn row_growth(&self, _session: &BoltSession) -> Result<u64, EngError> {
            Ok(0)
        }

        fn reserve_growth(&self, _bytes: u64) -> Result<(), EngError> {
            Ok(())
        }

        /// The logits rows of a window with a request that records its draw: only the host walk
        /// has the candidates.
        fn step_rows(
            &mut self,
            sessions: &mut [&mut BoltSession],
            tokens: &[u32],
        ) -> Result<Vec<Vec<f32>>, EngError> {
            let seqs: Vec<&BoltSeq> = sessions.iter().map(|session| &session.seq).collect();
            let positions: Vec<usize> = sessions.iter().map(|session| session.pos).collect();
            let rows = self.model.decode_rows(&seqs, tokens, &positions)?;
            for session in sessions.iter_mut() {
                session.pos += 1;
            }
            Ok(rows)
        }

        fn draws_on_device(&self) -> bool {
            true
        }

        /// One token of every session through one step, and the next token of each drawn from
        /// its own sampler.
        fn step_rows_drawn(
            &mut self,
            sessions: &mut [&mut BoltSession],
            tokens: &[u32],
            samplers: &[&crate::family::sampler::Sampler],
        ) -> Result<Vec<u32>, EngError> {
            let seqs: Vec<&BoltSeq> = sessions.iter().map(|session| &session.seq).collect();
            let positions: Vec<usize> = sessions.iter().map(|session| session.pos).collect();
            let drawn = self
                .model
                .decode_rows_draw(&seqs, tokens, &positions, samplers)?;
            for session in sessions.iter_mut() {
                session.pos += 1;
            }
            Ok(drawn)
        }
    }

    /// The free device bytes the planner may spend on extra sessions: what remains once the
    /// first session is allocated.
    fn plan_sessions(
        model: &Bolt,
        seqs: MaxSeqs,
    ) -> Result<crate::family::seq_plan::SeqPlan, EngError> {
        let free = u64::try_from(model.ctx.memory_info()?.free).unwrap_or(u64::MAX);
        let inputs = SeqInputs {
            free,
            headroom: budget::BOLT_HEADROOM_BYTES,
            cost: budget::bolt_seq_cost()
                .ok_or_else(|| EngError::MemoryPlan("the Bolt sequence cost overflows".into()))?,
            context: model.cells() as u64,
            row_cap: Bolt::PREFILL_BATCH,
            ceiling: Bolt::PREFILL_BATCH,
        };
        plan_seqs(
            inputs,
            match seqs {
                MaxSeqs::Count(count) => Some(count),
                MaxSeqs::Auto => None,
            },
        )
    }

    /// `model_path` is the model directory or one `.gguf` file in it. A directory with several
    /// `.gguf` files serves the highest-precision one that the device memory admits; `seqs`
    /// is the explicit session count or the count the free memory funds. Several sessions turn
    /// the decode graph off, and the start line says so; prefix reuse works at every count.
    pub(crate) fn start_bolt_engine(
        model_path: &Path,
        tokenizer_dir: &Path,
        ready_note: ReadyNote,
        seqs: MaxSeqs,
        prompt_rows: Option<usize>,
        kv_cache: Option<KvCacheSettings>,
    ) -> Result<StartedEngine, EngError> {
        let tier_plan = kv_cache::plan_tier(kv_cache.as_ref(), "gemma4", TIER_LABEL)?;
        let (_, files) = crate::family::gguf_select::candidates(model_path)?;
        let bundle = crate::tokenizer::gemma::load(tokenizer_dir)?;
        let eos = bundle.eos_token_ids().to_vec();
        if eos.is_empty() || eos.iter().any(|&id| id as usize >= VOCAB) {
            return Err(super::bad(
                "EOS ids must be non-empty and within vocabulary",
            ));
        }
        if bundle.vocab_size() as usize != VOCAB {
            return Err(super::bad(
                "the tokenizer vocabulary differs from the checkpoint vocabulary",
            ));
        }
        // The engine thread plans the KV cells from the free card and announces them before the
        // weights are uploaded; the server's request bound needs the figure before the engine is
        // ready.
        let (announce, resolved) = std::sync::mpsc::sync_channel::<Announcement<usize>>(1);
        let engine = spawn_engine_noted(Bolt::PREFILL_BATCH, ready_note, move |_| {
            let loaded = bolt::load_choosing(&files, seqs, &|cells| {
                let _ = announce.try_send(Ok(cells));
            });
            announce_failure(&loaded, Some(&announce));
            let mut model = loaded?;
            let first = model.new_seq()?;
            let ask = budget::session_ask(seqs, model.graph_mode.active());
            if let Some(note) = &ask.note {
                eprintln!("gemma4 bolt: {note}");
            }
            let plan = plan_sessions(&model, ask.seqs)?;
            eprintln!("gemma4 bolt: {}", plan.describe());
            if plan.seqs == 1 && prompt_rows.is_some() {
                eprintln!(
                    "gemma4 bolt: prompt_rows has no effect with one session: it sets how many prompt rows are fed per tick while several sessions share the model"
                );
            }
            if plan.seqs > 1 {
                model.reserve_window(plan.seqs)?;
                model.use_decode_graph(false);
                eprintln!(
                    "gemma4 bolt: {} sessions share the model, so the decode graph is off",
                    plan.seqs
                );
            }
            // What names this load to an entry kept outside the process: the checkpoint, the
            // pinned tokenizer and template, the layout of a snapshot and the numeric path.
            let load = kv_tier::load_identity(
                &model.weights_identity,
                &model.snapshot_layer_bytes(1),
                bolt::attn_split_wanted()?.then(|| model.attn_splits()),
            );
            let identity = PrefixIdentity {
                weights: load.weights.as_str().into(),
                tokenizer: load.tokenizer.as_str().into(),
                template: load.template.as_str().into(),
            };
            let disk = tier_plan.as_ref().and_then(|(resolved, origin)| {
                HostStoreDisk::open(TIER_LABEL, resolved, origin, &load, 1)
            });
            let mut serve = BoltServe {
                model,
                eos: eos.clone(),
                prefix: PrefixStore::with_limits(ONE_TIP),
                identity,
                max_seqs: plan.seqs,
                prompt_rows: prompt_rows.unwrap_or(budget::BOLT_AUTO_PROMPT_ROWS),
                prefix_hit: PrefixHit::Miss,
                prefix_uncached: 0,
                prefix_shadow: None,
                disk,
                prefix_tier: PrefixTier::default(),
            };
            serve.log_host_budget();
            if plan.seqs > 1 {
                eprintln!(
                    "gemma4 bolt: {} prompt rows per tick while sessions share the model ({})",
                    serve.prompt_rows,
                    if prompt_rows.is_some() {
                        "set"
                    } else {
                        "automatic"
                    }
                );
            }
            Ok((
                serve,
                BoltSession {
                    core: DecodeSession::default(),
                    pos: 0,
                    seq: first,
                },
            ))
        })?;
        let cells = await_announced(&resolved, "Bolt", || engine.handle().failure())?;
        Ok(StartedEngine {
            engine,
            bundle,
            context_limit: u32::try_from(cells)
                .map_err(|_| EngError::MemoryPlan("the Bolt cell count exceeds u32".into()))?,
        })
    }

    #[cfg(test)]
    mod prefix_tests {
        use super::*;

        /// Panics when the pinned Bolt file is not configured: an explicit
        /// `--ignored` run without `BOLT_Q8_FILE` is a setup error, not a pass.
        fn pinned_bolt() -> std::path::PathBuf {
            std::env::var("BOLT_Q8_FILE")
                .map(std::path::PathBuf::from)
                .expect("BOLT_Q8_FILE must point at the pinned Bolt q8 file")
        }

        /// Greedy argmax of the capped logits after one decode step.
        fn greedy_step(model: &mut Bolt, seq: &BoltSeq, token: u32, pos: usize) -> u32 {
            model.decode(seq, token, pos).expect("decode");
            let logits = model.capped_logits().expect("capped logits");
            logits
                .iter()
                .enumerate()
                .fold((0u32, logits[0]), |(best, value), (index, &other)| {
                    if other > value {
                        (index as u32, other)
                    } else {
                        (best, value)
                    }
                })
                .0
        }

        /// Feeds a prompt cache-only, holding nothing pending: the model-level
        /// control for cold runs.
        fn feed(model: &mut Bolt, seq: &BoltSeq, prompt: &[u32]) {
            for (index, &token) in prompt.iter().enumerate() {
                model
                    .prefill_batch(seq, &[token], index, false)
                    .expect("prefill");
            }
        }

        /// A true cold replay of a restore past a decode. Turn 1 decodes 48
        /// greedy tokens past the save point (cells P..P+48 now hold turn-1
        /// tokens); the restore rewinds to P, the tail prefill overwrites only
        /// the first cells of that span with DIFFERENT tokens, and the drawn
        /// chain must equal the chain a cold run of prompt + tail draws after a
        /// reset. A cache read above the position word (or a ring-buffer
        /// window) would leak the surviving turn-1 cells into the restored run
        /// and break the equality. The prompt is 3,000 tokens so turn 1 crosses
        /// the 512-cell sliding window.
        #[test]
        #[ignore = "requires the pinned Bolt file and a device"]
        fn a_restore_past_a_decode_matches_the_cold_replay() {
            let mut model: Bolt = bolt::load(&pinned_bolt()).expect("load");
            let seq = model.new_seq().expect("sequence");
            let prompt: Vec<u32> = (0..3_000u32).map(|index| 100 + (index % 977)).collect();
            feed(&mut model, &seq, &prompt);
            // The save point mirrors the session's: `prompt.len() - 1` tokens
            // fed, the last token held pending.
            let saved = prompt.len() - 1;
            // Turn 1: 48 greedy tokens past the save point.
            let mut token = prompt[saved];
            for pos in (saved..).take(48) {
                token = greedy_step(&mut model, &seq, token, pos);
            }
            // The restored branch: rewind to the save point, feed the held
            // token plus an extension whose tokens differ from turn 1's drawn
            // tokens (one PREFILL_BATCH of rows), then draw greedily.
            let extension: Vec<u32> = (0..7).map(|index| 90_000 + index).collect();
            let mut tail: Vec<u32> = vec![prompt[saved]];
            tail.extend_from_slice(&extension);
            let draw = |model: &mut Bolt| -> Vec<u32> {
                model
                    .prefill_batch(&seq, &tail, saved, false)
                    .expect("tail prefill");
                let mut token = *tail.last().expect("tail");
                let mut drawn = Vec::new();
                for pos in (saved + tail.len()..).take(24) {
                    token = greedy_step(model, &seq, token, pos);
                    drawn.push(token);
                }
                drawn
            };
            let restored = draw(&mut model);
            // The cold control: reset (position 0), re-feed the prompt, then
            // the same tail and greedy continuation.
            feed(&mut model, &seq, &prompt);
            let cold = draw(&mut model);
            assert_eq!(restored, cold, "the restore must match the cold replay");
            assert!(
                restored.windows(2).any(|pair| pair[0] != pair[1]),
                "the drawn chain must not be constant"
            );
        }

        /// The graphed decode must draw the identical greedy chain as the
        /// eager one. Run 1 draws eagerly; the re-feed from position 0
        /// overwrites every cell the first run read below its position word;
        /// run 2 arms the graph (the first clean step warms, the second
        /// captures, the rest replay) and must reproduce run 1's chain over a
        /// mid-run prefill too — the restore flow feeds rows while the graph
        /// exists, and the next decode must still replay it.
        #[test]
        #[ignore = "requires the pinned Bolt file and a device"]
        fn a_graphed_greedy_run_matches_the_eager_run() {
            let mut model: Bolt = bolt::load(&pinned_bolt()).expect("load");
            let seq = model.new_seq().expect("sequence");
            assert!(
                !model.graph_mode.active(),
                "the graph stays off without FIFI_BOLT_DECODE_GRAPH"
            );
            // 1,000 prompt rows and 96 more drawn: every decode reads past
            // the 512-cell sliding window of the local layers (Bolt's
            // LOCAL_WINDOW is the spec's sliding_window), so the windowed
            // attention of a replay is exercised over its full span.
            let prompt: Vec<u32> = (0..1_000u32).map(|index| 100 + (index % 977)).collect();
            let extension: Vec<u32> = (0..8u32).map(|index| 90_000 + index).collect();
            let run = |model: &mut Bolt, prompt: &[u32]| -> Vec<u32> {
                feed(model, &seq, prompt);
                let saved = prompt.len() - 1;
                let mut token = prompt[saved];
                let mut drawn = Vec::new();
                for pos in (saved..).take(10) {
                    token = greedy_step(model, &seq, token, pos);
                    drawn.push(token);
                }
                model
                    .prefill_batch(&seq, &extension, saved + 10, false)
                    .expect("mid-run prefill");
                let mut token = *extension.last().expect("extension");
                for pos in (saved + 10 + extension.len()..).take(86) {
                    token = greedy_step(model, &seq, token, pos);
                    drawn.push(token);
                }
                drawn
            };
            let eager = run(&mut model, &prompt);
            // Poison: a divergent eager run leaves OTHER values in every cell
            // the graphed run will read, so a replay that reads or writes a
            // stale cell (the capture-time position, say) draws a different
            // chain instead of matching the identical leftovers.
            let poison: Vec<u32> = (0..1_000u32).map(|index| 200_000 + (index % 891)).collect();
            let spoiled = run(&mut model, &poison);
            assert_ne!(eager, spoiled, "the poison run must diverge");
            model.use_decode_graph(true);
            let graphed = run(&mut model, &prompt);
            assert_eq!(eager, graphed, "the graphed chain must be byte-identical");
            assert!(
                model.graph_mode.replays() > 80,
                "the steps must have replayed (replays: {})",
                model.graph_mode.replays()
            );
            assert_eq!(model.graph_mode.refusal(), None, "no refusal may occur");
            let mut status = [0u8; 4];
            model
                .ctx
                .download(&model.scratch.status, &mut status)
                .expect("status word");
            assert_eq!(
                u32::from_le_bytes(status),
                0,
                "no KV8 append may have refused mid-run"
            );
            assert!(
                eager.windows(2).any(|pair| pair[0] != pair[1]),
                "the drawn chain must not be constant"
            );
        }

        /// The graphed decode with the split attention armed must still draw
        /// the identical greedy chain as the eager split run: the capture
        /// replays the part+combine pair exactly. The same poison discipline
        /// as the unsplit parity test, at a shorter draw to keep the suite
        /// time bounded.
        #[test]
        #[ignore = "requires the pinned Bolt file and a device"]
        fn a_graphed_split_run_matches_the_eager_split_run() {
            let mut model: Bolt = bolt::load(&pinned_bolt()).expect("load");
            let seq = model.new_seq().expect("sequence");
            let prompt: Vec<u32> = (0..1_000u32).map(|index| 100 + (index % 977)).collect();
            // The chain AND the logits bits per step: matching the greedy
            // chain alone is weaker than the pin this test owes, and the
            // bits make every replayed buffer write checkable.
            let run = |model: &mut Bolt, prompt: &[u32]| -> (Vec<u32>, Vec<Vec<u32>>) {
                feed(model, &seq, prompt);
                let saved = prompt.len() - 1;
                let mut token = prompt[saved];
                let mut drawn = Vec::new();
                let mut bits = Vec::new();
                for pos in (saved..).take(24) {
                    model.decode(&seq, token, pos).expect("decode");
                    let logits = model.capped_logits().expect("capped logits");
                    bits.push(logits.iter().map(|value| value.to_bits()).collect());
                    token = logits
                        .iter()
                        .enumerate()
                        .fold((0u32, logits[0]), |(best, value), (index, &other)| {
                            if other > value {
                                (index as u32, other)
                            } else {
                                (best, value)
                            }
                        })
                        .0;
                    drawn.push(token);
                }
                (drawn, bits)
            };
            // Capture a decode graph against the unsplit state first.
            model.use_decode_graph(true);
            let (unsplit, _) = run(&mut model, &prompt);
            assert!(model.graph_captured(), "the graph must be captured");
            let replays_unsplit = model.graph_mode.replays();
            // Arming the split must drop that graph: its capture points at the
            // pre-split launches, and a replay after the part buffer appears
            // (or the buffer frees on disarm) would read freed or dead state.
            model.use_attn_split(true).expect("arm split");
            assert!(!model.graph_captured(), "arming must invalidate the graph");
            model.use_decode_graph(false);
            let (eager, eager_bits) = run(&mut model, &prompt);
            assert!(
                eager.windows(2).any(|pair| pair[0] != pair[1]),
                "the drawn chain must not be constant"
            );
            let poison: Vec<u32> = (0..1_000u32).map(|index| 200_000 + (index % 891)).collect();
            let (spoiled, _) = run(&mut model, &poison);
            assert_ne!(eager, spoiled, "the poison run must diverge");
            model.use_attn_split(false).expect("disarm split");
            assert!(
                !model.split_armed(),
                "the disarm itself must free the partial buffer"
            );
            model.use_attn_split(true).expect("re-arm split");
            model.use_decode_graph(true);
            let (graphed, graphed_bits) = run(&mut model, &prompt);
            assert_eq!(eager, graphed, "the graphed split chain must match");
            assert_eq!(
                eager_bits, graphed_bits,
                "every replayed logits word must be bit-identical to the eager split run"
            );
            assert!(
                model.graph_captured(),
                "the graph must have recaptured after the re-arm"
            );
            assert!(
                model.graph_mode.replays() > replays_unsplit + 20,
                "the recaptured graph must replay the split steps (delta {})",
                model.graph_mode.replays() - replays_unsplit
            );
            assert!(model.graph_mode.replays() > 20, "must have replayed");
            assert_eq!(model.graph_mode.refusal(), None, "no refusal may occur");
            let _ = unsplit;
        }

        /// A serve with room for three sessions and no tokenizer: the tests drive the executor
        /// seam and the model directly.
        fn serve_of(model: Bolt) -> BoltServe {
            BoltServe {
                model,
                eos: vec![0],
                prefix: PrefixStore::with_limits(ONE_TIP),
                identity: PrefixIdentity::default(),
                max_seqs: 3,
                prompt_rows: budget::BOLT_AUTO_PROMPT_ROWS,
                prefix_hit: PrefixHit::Miss,
                prefix_uncached: 0,
                prefix_shadow: None,
                disk: None,
                prefix_tier: PrefixTier::default(),
            }
        }

        fn prompt(len: usize, salt: u32) -> Vec<u32> {
            (0..len as u32)
                .map(|index| 100 + salt + (index % 977))
                .collect()
        }

        /// The capped logits bits of decoding the pending last token of `session`.
        fn next_logits(serve: &mut BoltServe, session: &BoltSession, token: u32) -> Vec<u32> {
            serve
                .model
                .decode(&session.seq, token, session.pos)
                .expect("decode");
            serve
                .model
                .capped_logits()
                .expect("logits")
                .iter()
                .map(|value| value.to_bits())
                .collect()
        }

        /// A prefix saved by one session restores into a different, dirty session and the
        /// continuation matches a cold run bit for bit, with the source session having moved on
        /// (decoded and reset) since the save.
        #[test]
        #[ignore = "requires the pinned Bolt file and a device"]
        fn a_saved_prefix_restores_into_another_session_and_matches_a_cold_run() {
            restore_matches_a_cold_run(300);
        }

        /// The same past the 512-cell sliding window of the local layers and with planes larger
        /// than the 8 MiB zero chunk (20,000 positions: 10 MB per global plane range, 5 MB per
        /// local one). The largest range a snapshot can copy is 16 MB (32,768 cells of 512), under
        /// the 64 MiB staging chunk of a device copy, so one copy never splits a range.
        #[test]
        #[ignore = "requires the pinned Bolt file and a device"]
        fn a_long_saved_prefix_restores_bit_exact_past_the_window_and_the_copy_chunk() {
            restore_matches_a_cold_run(20_000);
        }

        fn restore_matches_a_cold_run(base_len: usize) {
            let mut serve = serve_of(bolt::load(&pinned_bolt()).expect("load"));
            let mut source = serve.new_session().expect("source");
            let mut target = serve.new_session().expect("target");
            let mut cold = serve.new_session().expect("cold");
            let base = prompt(base_len, 0);
            let extended: Vec<u32> = base.iter().copied().chain(prompt(40, 5_000)).collect();

            serve.reset(&mut source).expect("reset");
            serve.prefill(&mut source, &base).expect("prefill");
            assert!(
                serve
                    .save_prefix(&source, &base, SaveKind::Tip)
                    .expect("save")
            );
            // The source moves on: it decodes past the save point and is reset for another job.
            let mut token = *base.last().expect("prompt");
            for step in 0..8 {
                serve
                    .model
                    .decode(&source.seq, token, source.pos + step)
                    .expect("decode");
                token = 200 + step as u32;
            }
            serve.reset(&mut source).expect("reset");
            serve
                .prefill(&mut source, &prompt(64, 9_000))
                .expect("other prompt");

            // The target held other data; the restore covers the whole saved key.
            serve.reset(&mut target).expect("reset");
            serve
                .prefill(&mut target, &prompt(500, 7_000))
                .expect("dirty");
            serve.reset(&mut target).expect("reset");
            let covered = serve
                .restore_prefix(&mut target, &extended)
                .expect("restore");
            assert_eq!(covered, base.len(), "the whole saved prompt restores");
            assert_eq!(
                target.pos,
                base.len() - 1,
                "the cache position is the entry's"
            );
            serve
                .prefill(&mut target, &extended[covered..])
                .expect("tail prefill");

            serve.reset(&mut cold).expect("reset");
            serve.prefill(&mut cold, &extended).expect("cold prefill");
            assert_eq!(target.pos, cold.pos);
            let last = *extended.last().expect("prompt");
            let restored = next_logits(&mut serve, &target, last);
            let fresh = next_logits(&mut serve, &cold, last);
            assert_eq!(
                restored, fresh,
                "a restored session continues like a cold one"
            );
        }

        /// A different prompt saves over the tip, and the old key no longer restores: the store
        /// holds one entry and answers only for keys the prompt extends.
        #[test]
        #[ignore = "requires the pinned Bolt file and a device"]
        fn the_store_answers_only_for_a_key_the_prompt_extends() {
            let mut serve = serve_of(bolt::load(&pinned_bolt()).expect("load"));
            let mut first = serve.new_session().expect("first");
            let first_key = prompt(64, 0);
            let other_key = prompt(64, 300);
            serve.reset(&mut first).expect("reset");
            serve.prefill(&mut first, &first_key).expect("prefill");
            assert!(
                serve
                    .save_prefix(&first, &first_key, SaveKind::Tip)
                    .expect("save")
            );
            serve.reset(&mut first).expect("reset");
            serve
                .prefill(&mut first, &other_key)
                .expect("prefill other");
            assert!(
                serve
                    .save_prefix(&first, &other_key, SaveKind::Tip)
                    .expect("save other")
            );

            let mut second = serve.new_session().expect("second");
            serve.reset(&mut second).expect("reset");
            let stale: Vec<u32> = first_key.iter().copied().chain(500..512).collect();
            assert_eq!(serve.restore_prefix(&mut second, &stale).expect("stale"), 0);
            assert_eq!(second.pos, 0, "a miss leaves the session as reset");
            let fresh: Vec<u32> = other_key.iter().copied().chain(600..612).collect();
            assert_eq!(
                serve.restore_prefix(&mut second, &fresh).expect("fresh"),
                other_key.len()
            );
        }
    }
}

#[cfg(feature = "native-cuda")]
pub(crate) use device::start_bolt_engine;
