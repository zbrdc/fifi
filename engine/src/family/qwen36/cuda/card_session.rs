//! The multi-device serving session: the eager [`CardStack`] token driver
//! behind the shared [`DecodeSession`] seam, and the start-level enum that
//! keeps one executor type over the single-device and multi-device steps.
//!
//! The default is the eager whole-token driver ([`run_card_token`]):
//! per token, every card runs its mixer and MoE blocks with the ring
//! exchanges inside, the tail head projection covers each card's vocabulary
//! shard, and the host reads back the status words and the logits shards.
//! The failure model of `multi.rs` is enforced at the read-back: every
//! card's step-status word and ring status word fold BEFORE a token is
//! handed out, and any nonzero word poisons the session irrevocably - no
//! corrupted token is returned, and after a failure every later call
//! refuses by name.
//!
//! Experimental chunked prefill is opt-in; prefix reuse is disabled while it
//! is active, pending numeric-path and turn-chain qualification. Speculation,
//! tiered experts, host-paged KV, and EXL3 are refused at start.
//!
//! Lifetime: like `QwenResidentStep`, the owners the views borrow - the
//! [`CardStack`], the per-card [`CudaWeights`] - are leaked for the process
//! lifetime by the start, so every view the session holds is `'static`.

use std::sync::Arc;

use super::card_step::CardStack;
use super::card_token::run_card_token;
use super::dims::{StepDims, StepOptions};
use super::layer_table::LayerTable;
use super::mixer::{CardKindOps, MixerParts, MixerViews, Mixers};
use super::mixer_geometry::{DenseSpec, MixerGeometry, check_native_dense};
use super::mixer_ops::CardOp;
use super::mixer_weights::MixerWeights;
use super::moe::MoeLayerWeights;
use super::multi::fold_ring_status;
use super::prefix::CudaPrefillPath;
use super::resources::{GlobalNames, MoeSpecs, check_native_embed, moe_layer_weights, raw_buffer};
use super::state::export::ExportedState;
use super::state::{ResidentState, StepField};
use super::step_plan::Slot;

use crate::EngError;
use crate::family::AttentionKind;
use crate::family::executor::{ExecutorOps, GenerationState};
use crate::family::gguf_model::{Inventory, Role};
use crate::family::project::cuda::{CudaWeight, CudaWeights, WeightLayout};
use crate::family::qwen36::cuda::rules::check_token_id;
use crate::family::session::{
    DecodeSession, Limits, PrefixSlot, SaveKind, StoreLimits, TokenStepper, check_prompt,
};
use crate::kernels::cuda::resident::step_params::StepParams;
use crate::kernels::cuda::{DeviceBuffer, NativeWeight};

/// The whole eager distributed step: the stack, the per-card weight views,
/// the card op lists, and the token book.
pub(crate) struct CardSession {
    stack: &'static CardStack,
    /// The per-card resident states, owned mutably beside the shared
    /// stack: the single-device split (leaked shared resources, step-owned
    /// state) applied to the cards.
    states: Vec<ResidentState>,
    /// The per-card mixers, one per card over every layer.
    mixers: Vec<Mixers<'static>>,
    /// The MoE weights, one set per layer: `[layer][card]`.
    moe: Vec<Vec<MoeLayerWeights<'static>>>,
    /// The card op lists, one per card.
    card_ops: Vec<CardKindOps>,
    kinds: Vec<AttentionKind>,
    ordinals: Vec<usize>,
    kv_runs: Vec<std::ops::Range<usize>>,
    embed: &'static NativeWeight,
    output_norms: Vec<&'static DeviceBuffer>,
    heads: Vec<&'static NativeWeight>,
    /// The ascending global row starts of the cards' head shards.
    logits_starts: Vec<usize>,
    /// The full head row the concat of the shards rebuilds.
    vocab: usize,
    eps: f32,
    context_limit: usize,
    fed: usize,
    poisoned: bool,
    /// The prefix store of the session: one entry is one `ExportedState`
    /// per card (the card's sliced KV planes plus its DeltaNet state), so
    /// an entry's host cost is one replica of cache plus two delta pools.
    prefix: PrefixSlot<Vec<ExportedState>, CudaPrefillPath>,
    /// The numeric path every entry of this session saves and restores under.
    prefix_path: CudaPrefillPath,
    /// The chunk resources, built at start when Q36_P2P_CHUNK is set:
    /// the session is the stepper the worker drives, so the chunk
    /// dispatch lives here.
    chunk: Option<super::card_prefill::CardChunk>,
    /// Set by a prefix import: the word block was zeroed, so the next feed
    /// writes the full step params (token id AND position) again.
    full_params_next: bool,
    /// Experimental: prompt-only feeds need state, not the vocabulary row.
    skip_prefill_head: bool,
    eos: Arc<[u32]>,
    sampler_vocab: usize,
}

impl CardSession {
    /// Resolves every view of the session from the leaked shared `stack`,
    /// the `states` moved out beside it, and the loaded per-card weights.
    /// Every shape check runs against the CARD's geometry, so a wrongly
    /// sliced tensor refuses here, by name.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        stack: &'static CardStack,
        states: Vec<ResidentState>,
        weights: &'static [&'static CudaWeights],
        table: &'static LayerTable,
        inventory: &Inventory,
        dims: &StepDims,
        options: &StepOptions,
        eos: Arc<[u32]>,
        sampler_vocab: usize,
        prefix_limits: StoreLimits,
        chunk_rows: Option<usize>,
    ) -> Result<Self, EngError> {
        let cards = stack.shards().len();
        if weights.len() != cards {
            return Err(EngError::BadRequest(format!(
                "card session: {cards} cards, {} weight sets",
                weights.len()
            )));
        }
        let shard = stack.table();
        let layers: Vec<usize> = (0..table.len()).collect();
        let mut kinds = Vec::with_capacity(table.len());
        let mut ordinals = Vec::with_capacity(table.len());
        for layer in 0..table.len() {
            let slot = table.slot(layer)?;
            kinds.push(slot.kind);
            ordinals.push(slot.ordinal);
        }
        let names = GlobalNames::new(inventory)?;
        let native = WeightLayout::Native;
        let mut mixers = Vec::with_capacity(cards);
        let mut card_ops = Vec::with_capacity(cards);
        let mut output_norms = Vec::with_capacity(cards);
        let mut heads = Vec::with_capacity(cards);
        let mut logits_starts = Vec::with_capacity(cards);
        let mut moe = Vec::with_capacity(table.len());
        for _ in 0..table.len() {
            moe.push(Vec::with_capacity(cards));
        }
        for (card, &weights_card_of) in weights.iter().enumerate() {
            // The mixer views of every layer at this card's geometry.
            let geometry = MixerGeometry::for_card(dims, shard, card)?;
            let weights_card =
                MixerWeights::new(&geometry, table, weights_card_of, &layers, native)?;
            let ops = CardKindOps::new(shard, card, &geometry, options.kv_mode)?;
            let views =
                MixerViews::for_card(&stack.plans()[card], stack.shards()[card].arena(), &ops)?;
            let parts = MixerParts {
                geometry,
                table,
                weights: weights_card,
                views,
                layout: native,
                kv_mode: options.kv_mode,
                // Cards hold native weights only: the tiled-experts hybrid has no card form.
                moe_tiled: false,
                paged: None,
                // Card sessions run dense: no sparse exec is constructed on this
                // path, so the sparse chain can never launch here.
                sparse: None,
            };
            mixers.push(Mixers::new(&stack.contexts()[card], parts)?);
            card_ops.push(ops);
            // The MoE weights of every layer on this card, at its narrowed
            // expert widths.
            let specs = MoeSpecs::for_card(dims, shard, card)?;
            for (layer, set) in moe.iter_mut().enumerate() {
                set.push(moe_layer_weights(
                    table,
                    weights_card_of,
                    &specs,
                    layer,
                    native,
                )?);
            }
            // The replicated final norm, at the same width as the
            // single-device global.
            output_norms.push(raw_buffer(weights[card], &names.norm, dims.shape.hidden)?);
            // The card's vocabulary-shard head: the single-device check at
            // the card's head width.
            let head_rows = shard.head_rows(card)?;
            logits_starts.push(head_rows.start);
            let Some(CudaWeight::Native { weight, .. }) = weights[card].get(names.head.as_str())
            else {
                return Err(EngError::Checkpoint(format!(
                    "card weights: {} is not a native tensor",
                    names.head
                )));
            };
            check_native_dense(
                &names.head,
                (weight.n_rows(), weight.k()),
                &DenseSpec {
                    role: Role::Output,
                    rows: head_rows.len(),
                    k: dims.shape.hidden,
                },
            )?;
            heads.push(weight);
        }
        // Card 0's embedding table: the only card that holds one.
        let Some(CudaWeight::Native { weight, .. }) = weights[0].get(names.embed.as_str()) else {
            return Err(EngError::Checkpoint(format!(
                "card weights: {} is not a native tensor",
                names.embed
            )));
        };
        check_native_dense(
            &names.embed,
            (weight.n_rows(), weight.k()),
            &DenseSpec {
                role: Role::TokenEmbd,
                rows: dims.shape.vocab,
                k: dims.shape.hidden,
            },
        )?;
        check_native_embed(&names.embed, weight.quant())?;
        let embed = weight;
        let kv_runs = (0..cards)
            .map(|card| shard.kv_rows(card))
            .collect::<Result<Vec<_>, EngError>>()?;
        // Every card's state carries the same KV capacity (one
        // `ResidentState::allocate` shape); the session limit is that
        // capacity, the same rule the single-device step uses.
        let context_limit = states[0].layout().kv.capacity;
        let vocab = logits_starts
            .last()
            .copied()
            .zip(heads.last().map(|head| head.n_rows()))
            .and_then(|(start, rows)| start.checked_add(rows))
            .ok_or_else(|| {
                EngError::Checkpoint("card session: the head shards overflow usize".into())
            })?;
        if states.len() != cards {
            return Err(EngError::BadRequest(format!(
                "card session: {cards} cards, {} states",
                states.len()
            )));
        }
        let chunk = Self::build_chunk_if_asked(
            stack, weights, inventory, dims, table, options, chunk_rows,
        )?;
        let effective_limits = chunk_safe_prefix_limits(prefix_limits, chunk.is_some());
        if chunk.is_some() && effective_limits != prefix_limits {
            eprintln!("q36 cuda: prefix reuse disabled for experimental P2P chunked prefill");
        }
        Ok(Self {
            stack,
            states,
            mixers,
            moe,
            card_ops,
            kinds,
            ordinals,
            kv_runs,
            embed,
            output_norms,
            heads,
            logits_starts,
            vocab,
            eps: dims.shape.eps,
            context_limit,
            fed: 0,
            poisoned: false,
            skip_prefill_head: std::env::var_os("Q36_P2P_SKIP_PREFILL_HEAD")
                .is_some_and(|value| value == "1"),
            prefix: PrefixSlot::with_limits(effective_limits),
            prefix_path: CudaPrefillPath::Steps(options.kv_mode),
            chunk,
            full_params_next: false,
            eos,
            sampler_vocab,
        })
    }

    pub(crate) fn context_limit(&self) -> usize {
        self.context_limit
    }

    #[cfg(test)]
    pub(super) fn set_skip_prefill_head_for_test(&mut self, enabled: bool) {
        self.skip_prefill_head = enabled;
    }

    #[cfg(test)]
    pub(super) fn set_chunk_for_test(&mut self, chunk: super::card_prefill::CardChunk) {
        self.chunk = Some(chunk);
    }

    #[cfg(test)]
    pub(super) fn set_sampler_vocab_for_test(&mut self, sampler_vocab: usize) {
        assert!((1..=self.vocab).contains(&sampler_vocab));
        self.sampler_vocab = sampler_vocab;
    }

    fn build_chunk_if_asked(
        stack: &'static CardStack,
        weights: &'static [&'static crate::family::project::cuda::CudaWeights],
        inventory: &crate::family::gguf_model::Inventory,
        dims: &super::dims::StepDims,
        table: &'static LayerTable,
        options: &super::dims::StepOptions,
        chunk_rows: Option<usize>,
    ) -> Result<Option<super::card_prefill::CardChunk>, EngError> {
        let Some(rows) = chunk_rows else {
            return Ok(None);
        };
        let embed_name: &'static str = Box::leak(
            super::resources::GlobalNames::new(inventory)?
                .embed
                .into_boxed_str(),
        );
        match super::card_prefill::build_card_chunk(
            stack,
            dims,
            table,
            super::card_prefill::CardChunkInputs {
                weights,
                embed_name,
            },
            options,
            rows,
        ) {
            Ok(chunk) => {
                eprintln!("q36 cuda: chunked prefill engaged at {rows} rows");
                Ok(Some(chunk))
            }
            Err(error) => {
                eprintln!("q36 cuda: chunked prefill refused: {error:?}");
                Ok(None)
            }
        }
    }

    #[cfg(test)]
    pub(super) fn states_mut(&mut self) -> &mut [ResidentState] {
        &mut self.states
    }

    #[cfg(test)]
    pub(super) fn note_chunk_rows(&mut self, rows: usize) {
        self.fed += rows;
        // The chunk's revalidate left the states needing a start block,
        // exactly like a prefix import: the next feed writes the full step
        // params (token id AND position).
        self.full_params_next = true;
    }

    /// Card 0's embedding table: the reference ChunkWeights carries.
    #[cfg(test)]
    pub(super) fn embed_for_test(&self) -> &'static crate::kernels::cuda::NativeWeight {
        self.embed
    }

    /// The chunk-driver seam: this card's mixer views and its per-layer
    /// MoE views, transposed to the [card][layer] order ChunkWeights takes.
    #[cfg(test)]
    pub(super) fn chunk_views_for_test(
        &self,
        card: usize,
    ) -> (
        &super::mixer_weights::MixerWeights<'static>,
        Vec<super::moe::MoeLayerWeights<'static>>,
    ) {
        let moe = self.moe.iter().map(|set| set[card]).collect::<Vec<_>>();
        (self.mixers[card].weights(), moe)
    }

    /// The prefix rig's comparison vehicle: one export per card of the
    /// committed state (sliced KV planes plus DeltaNet slots).
    #[cfg(test)]
    pub(super) fn export_states_for_test(
        &self,
    ) -> Result<Vec<crate::family::qwen36::cuda::state::export::ExportedState>, EngError> {
        self.states
            .iter()
            .enumerate()
            .map(|(card, state)| state.export_state(&self.stack.streams()[card]))
            .collect()
    }

    /// The one launch path of a token: run the whole distributed step, fold
    /// the status words, and optionally read the concatenated logits row.
    /// `fed` advances only when the token ran clean.
    fn run_token(&mut self, token: u32, read_logits: bool) -> Result<Vec<f32>, EngError> {
        if self.poisoned {
            return Err(EngError::Backend {
                operation: "qwen36_cards",
                message: "the multi-device session is poisoned after an earlier failure".to_owned(),
            });
        }
        check_token_id(token, self.sampler_vocab)?;
        if self.fed >= self.context_limit {
            return Err(EngError::BadRequest(format!(
                "card session: position {} reaches the context limit {}",
                self.fed, self.context_limit
            )));
        }
        if self.fed == 0 || self.full_params_next {
            // The cleared word block takes the full params: the token id
            // and the position. After a fresh reset the position is 0; after
            // a prefix import the word block was zeroed by the import and
            // the position restarts at the restored token count.
            let pos = i32::try_from(self.fed).map_err(|_| {
                EngError::BadRequest("card session: position overflows i32".to_owned())
            })?;
            let params = StepParams::new(card_token_id(token)?, pos)?;
            for card in 0..self.stack.shards().len() {
                self.states[card].write_step_params(&self.stack.streams()[card], &params)?;
            }
            self.full_params_next = false;
        } else {
            // The next token comes from the HOST draw over the folded
            // logits row, so the global id is injected explicitly on every
            // card; the position advances on device.
            self.write_token_words(token)?;
        }
        let cards = self.stack.shards().len();
        let mut layer_ops: Vec<Vec<&[CardOp]>> = Vec::with_capacity(self.kinds.len());
        for &kind in &self.kinds {
            let list = (0..cards)
                .map(|card| self.card_ops[card].of(kind))
                .collect::<Result<Vec<_>, EngError>>()?;
            layer_ops.push(list);
        }
        for state in &mut self.states {
            state.enqueue_step(1)?;
        }
        let ops: Vec<&[&[CardOp]]> = layer_ops.iter().map(|list| list.as_slice()).collect();
        let moe_sets: Vec<&[MoeLayerWeights]> = self.moe.iter().map(|set| set.as_slice()).collect();
        run_card_token(
            self.stack,
            &self.states,
            &self.kinds,
            &self.ordinals,
            &ops,
            &self.kv_runs,
            &self.mixers,
            &moe_sets,
            self.embed,
            &self.output_norms,
            &self.heads,
            self.eps,
            self.fed,
            read_logits || !self.skip_prefill_head,
        )?;
        // The read-back: every stream joined, then every status word folded
        // BEFORE anything is handed out.
        for card in 0..cards {
            self.stack.streams()[card].synchronize()?;
        }
        self.fold_statuses()?;
        // The step committed one clean position on every card; the books
        // carry the count the prefix export reads.
        for state in &mut self.states {
            state.commit_step(1, 0)?;
        }
        let logits = if read_logits {
            self.read_logits()?
        } else {
            Vec::new()
        };
        self.fed += 1;
        Ok(logits)
    }

    /// Writes the global token id into every card's `TokenId` word.
    fn write_token_words(&self, token: u32) -> Result<(), EngError> {
        let bytes = card_token_id(token)?.to_le_bytes();
        for card in 0..self.stack.shards().len() {
            let stream = &self.stack.streams()[card];
            let word = self.states[card].step_field(StepField::TokenId)?;
            word.buffer().write_at(stream, word.offset(), &bytes)?;
        }
        Ok(())
    }

    /// Reads and folds every card's step-status and ring-status words. Any
    /// nonzero word poisons the session and returns a `Backend` error
    /// naming the card and the words.
    fn fold_statuses(&mut self) -> Result<(), EngError> {
        let cards = self.stack.shards().len();
        let mut step_words = Vec::with_capacity(cards);
        let mut ring_words = Vec::with_capacity(cards);
        for card in 0..cards {
            let stream = &self.stack.streams()[card];
            let mut bytes = [0u8; 4];
            let step = self.states[card].step_status()?;
            step.buffer().read_at(stream, step.offset(), &mut bytes)?;
            step_words.push(u32::from_le_bytes(bytes));
            self.stack.shards()[card]
                .ring_word()
                .read_at(stream, 0, &mut bytes)?;
            ring_words.push(u32::from_le_bytes(bytes));
        }
        let folded = fold_ring_status(&ring_words);
        let failed = step_words.iter().position(|&word| word != 0);
        if folded != 0 || failed.is_some() {
            self.poisoned = true;
            let card = failed.unwrap_or(0);
            return Err(EngError::Backend {
                operation: "qwen36_cards",
                message: format!(
                    "card {card} reported step status {:#x}, ring fold {folded:#x} (step words \
                     {step_words:?}, ring words {ring_words:?}); the session is poisoned",
                    step_words[card],
                ),
            });
        }
        Ok(())
    }

    /// Reads the cards' logits shards and rebuilds the full head row by
    /// ascending shard order.
    fn read_logits(&self) -> Result<Vec<f32>, EngError> {
        let cards = self.stack.shards().len();
        let mut row = Vec::with_capacity(self.vocab);
        for card in 0..cards {
            let index = self.stack.plans()[card]
                .index(Slot::Logits)
                .ok_or_else(|| {
                    EngError::Checkpoint("card session: the plan holds no logits slot".into())
                })?;
            let view = self.stack.shards()[card].tensor(index)?;
            let mut bytes = vec![0u8; view.bytes()];
            view.buffer()
                .read_at(&self.stack.streams()[card], view.offset(), &mut bytes)?;
            row.extend(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f32::from_le_bytes(*chunk)),
            );
        }
        debug_assert_eq!(row.len(), self.vocab);
        Ok(row)
    }
}

/// The token id as the step word wants it: non-negative and i32-wide.
fn card_token_id(token: u32) -> Result<i32, EngError> {
    i32::try_from(token)
        .map_err(|_| EngError::BadRequest(format!("token id {token} does not fit an i32")))
}

/// Chunked and per-token states are numerically different. Keep prefix reuse
/// off until the chunk path has its own qualified identity and turn-chain gate.
fn chunk_safe_prefix_limits(limits: StoreLimits, chunk_enabled: bool) -> StoreLimits {
    if chunk_enabled {
        StoreLimits {
            roots: 0,
            entries: 0,
        }
    } else {
        limits
    }
}

impl TokenStepper for CardSession {
    const OPERATION: &'static str = "qwen36_cards";

    fn feed(&mut self, token: u32) -> Result<(), EngError> {
        self.run_token(token, false).map(|_| ())
    }

    fn feed_rows(&mut self, tokens: &[u32]) -> Result<(), EngError> {
        let Some(chunk) = self.chunk.as_ref() else {
            return tokens.iter().try_for_each(|&t| self.feed(t));
        };
        if tokens.len() < 2 || self.poisoned {
            return tokens.iter().try_for_each(|&t| self.feed(t));
        }
        // Match the single-device chunk path: validate the sampler vocabulary and
        // signed kernel-index range for the whole batch before any card changes
        // state. The embedding table can contain rows beyond the sampler vocabulary.
        let _ = super::prefill_driver::chunk_ids(tokens, self.sampler_vocab)?;
        let (rows, tail) = tokens.split_at(tokens.len() - 1);
        let [last] = tail else {
            unreachable!("len checked above");
        };
        for piece in rows.chunks(chunk.rows()) {
            if let Err(error) = super::card_prefill::run_card_chunk(
                self.stack,
                &mut self.states,
                chunk,
                piece,
                self.fed,
            ) {
                self.poisoned = true;
                for state in &mut self.states {
                    state.poison();
                }
                return Err(error);
            }
            self.fed += piece.len();
        }
        self.full_params_next = true;
        self.feed(*last)
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
        self.run_token(token, true)
    }

    fn fed(&self) -> usize {
        self.fed
    }

    fn reset(&mut self) -> Result<(), EngError> {
        // The streams are joined inside; the fed book restarts.
        self.stack.reset_states_and_ring(&mut self.states)?;
        self.fed = 0;
        self.poisoned = false;
        self.full_params_next = false;
        Ok(())
    }
}

impl ExecutorOps for CardSession {
    type Session = DecodeSession;

    fn check_prompt(&self, prompt_tokens: usize) -> Result<(), EngError> {
        check_prompt(prompt_tokens, self.context_limit)
    }

    fn history<'s>(&self, st: &'s DecodeSession) -> &'s [u32] {
        st.history()
    }

    fn reset(&mut self, st: &mut DecodeSession) -> Result<(), EngError> {
        st.reset(self)
    }

    fn prefill(&mut self, st: &mut DecodeSession, toks: &[u32]) -> Result<(), EngError> {
        st.prefill(self, toks, self.context_limit)
    }

    /// Advertise the admitted chunk size only when experimental card chunking is active.
    /// Otherwise the worker keeps its existing per-token-path fallback.
    fn prefill_piece(&self) -> Option<usize> {
        self.chunk.as_ref().map(|chunk| chunk.rows())
    }

    fn decode(
        &mut self,
        st: &mut DecodeSession,
        generation: &mut GenerationState,
    ) -> Result<crate::tokenizer::DecodeBatch, EngError> {
        // The ids are shared, so the limits do not borrow the session that
        // `decode` borrows mutably.
        let eos = Arc::clone(&self.eos);
        let limits = Limits {
            eos: &eos,
            sampler_vocab: self.sampler_vocab,
            context_limit: self.context_limit,
        };
        st.decode(self, generation, &limits)
    }

    fn boundary(&mut self, _st: &mut DecodeSession) {}

    fn prefix_limits(&self) -> StoreLimits {
        self.prefix.limits()
    }

    /// Snapshots every card's state (its sliced KV planes plus its DeltaNet
    /// slots) into the host store under the shared session rules. An entry
    /// is one `ExportedState` per card, so restoring hands each card back
    /// its own slice; the recorded slice identity refuses a mismatched deal.
    fn save_prefix(
        &mut self,
        st: &DecodeSession,
        key: &[u32],
        kind: SaveKind,
    ) -> Result<bool, EngError> {
        if self.prefix.limits().entries == 0 {
            return Ok(false);
        }
        let fed = self.fed;
        let mut slot = std::mem::take(&mut self.prefix);
        let saved = st.save_prefix(&mut slot, key, self.prefix_path, fed, kind, || {
            self.states
                .iter()
                .enumerate()
                .map(|(card, state)| state.export_state(&self.stack.streams()[card]))
                .collect()
        });
        self.prefix = slot;
        saved
    }

    fn restore_prefix(
        &mut self,
        st: &mut DecodeSession,
        prompt: &[u32],
    ) -> Result<usize, EngError> {
        if self.prefix.limits().entries == 0 {
            return Ok(0);
        }
        let mut slot = std::mem::take(&mut self.prefix);
        let covered = st.restore_prefix(&mut slot, prompt, self.prefix_path, |saved| {
            if saved.len() != self.states.len() {
                return Err(EngError::BadRequest(format!(
                    "card prefix: {} saved states for {} cards",
                    saved.len(),
                    self.states.len()
                )));
            }
            let tokens = saved[0].tokens;
            for (card, (state, exported)) in self.states.iter_mut().zip(saved.iter()).enumerate() {
                state.import_state(&self.stack.streams()[card], exported)?;
            }
            self.fed = tokens;
            self.full_params_next = true;
            Ok(())
        });
        self.prefix = slot;
        covered
    }
}

/// The one CUDA executor type of the start: the single-device resident step
/// or the multi-device card session, switched once at `make`.
// The single-device step is the fat variant by nature (it carries every
// per-layer weight view inline); one instance lives per engine open, so
// the size difference buys nothing by boxing it too.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Q36CudaStep {
    Single(super::step::QwenResidentStep<'static>),
    Cards(Box<CardSession>),
}

impl TokenStepper for Q36CudaStep {
    const OPERATION: &'static str = "qwen36_cuda";

    fn feed(&mut self, token: u32) -> Result<(), EngError> {
        match self {
            Self::Single(step) => step.feed(token),
            Self::Cards(session) => session.feed(token),
        }
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
        match self {
            Self::Single(step) => step.logits(token),
            Self::Cards(session) => session.logits(token),
        }
    }

    fn fed(&self) -> usize {
        match self {
            Self::Single(step) => step.fed(),
            Self::Cards(session) => session.fed(),
        }
    }

    fn reset(&mut self) -> Result<(), EngError> {
        match self {
            Self::Single(step) => TokenStepper::reset(step),
            Self::Cards(session) => TokenStepper::reset(&mut **session),
        }
    }
}

impl ExecutorOps for Q36CudaStep {
    type Session = DecodeSession;

    fn check_prompt(&self, prompt_tokens: usize) -> Result<(), EngError> {
        match self {
            Self::Single(step) => step.check_prompt(prompt_tokens),
            Self::Cards(session) => session.check_prompt(prompt_tokens),
        }
    }

    fn history<'s>(&self, st: &'s DecodeSession) -> &'s [u32] {
        match self {
            Self::Single(step) => step.history(st),
            Self::Cards(session) => session.history(st),
        }
    }

    fn reset(&mut self, st: &mut DecodeSession) -> Result<(), EngError> {
        match self {
            Self::Single(step) => ExecutorOps::reset(step, st),
            Self::Cards(session) => ExecutorOps::reset(&mut **session, st),
        }
    }

    fn prefill(&mut self, st: &mut DecodeSession, toks: &[u32]) -> Result<(), EngError> {
        match self {
            Self::Single(step) => step.prefill(st, toks),
            Self::Cards(session) => session.prefill(st, toks),
        }
    }

    fn prefill_piece(&self) -> Option<usize> {
        match self {
            Self::Single(step) => step.prefill_piece(),
            Self::Cards(session) => session.prefill_piece(),
        }
    }

    fn decode(
        &mut self,
        st: &mut DecodeSession,
        generation: &mut GenerationState,
    ) -> Result<crate::tokenizer::DecodeBatch, EngError> {
        match self {
            Self::Single(step) => step.decode(st, generation),
            Self::Cards(session) => session.decode(st, generation),
        }
    }

    fn prefix_limits(&self) -> StoreLimits {
        match self {
            Self::Single(step) => step.prefix_limits(),
            Self::Cards(session) => session.prefix_limits(),
        }
    }

    fn save_prefix(
        &mut self,
        st: &DecodeSession,
        key: &[u32],
        kind: SaveKind,
    ) -> Result<bool, EngError> {
        match self {
            Self::Single(step) => step.save_prefix(st, key, kind),
            Self::Cards(session) => session.save_prefix(st, key, kind),
        }
    }

    fn restore_prefix(
        &mut self,
        st: &mut DecodeSession,
        prompt: &[u32],
    ) -> Result<usize, EngError> {
        match self {
            Self::Single(step) => step.restore_prefix(st, prompt),
            Self::Cards(session) => session.restore_prefix(st, prompt),
        }
    }

    fn boundary(&mut self, st: &mut DecodeSession) {
        match self {
            Self::Single(step) => step.boundary(st),
            Self::Cards(session) => session.boundary(st),
        }
    }
}

#[cfg(test)]
mod chunk_prefix_tests {
    use super::{StoreLimits, chunk_safe_prefix_limits};

    #[test]
    fn experimental_chunking_refuses_unqualified_prefix_reuse() {
        let configured = StoreLimits {
            roots: 2,
            entries: 3,
        };
        assert_eq!(chunk_safe_prefix_limits(configured, false), configured);
        assert_eq!(
            chunk_safe_prefix_limits(configured, true),
            StoreLimits {
                roots: 0,
                entries: 0,
            },
        );
    }
}
