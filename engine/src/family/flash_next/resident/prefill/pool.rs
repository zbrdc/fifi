//! Prefill expert page pool (decision 0044 items 5 and 7; packet PF-4).
//!
//! A prefill allocates one pool when it starts and drops it after its last chunk. The pool holds
//! two page sets. A set holds one wave of a layer, in this order: the gate, up, and down stacks of
//! the wave's experts (`SetLayout`), then the route page. The route page holds the U32 wave-local
//! rank, the U32 token row, and the F32 gate of each pair of the wave, and the U32 inverse
//! permutation of the chunk (the last wave of a layer only).
//! The pairs of a wave arrive sorted by local rank (`prefill_plan::split_waves`), and
//! `import_wave` proves that order with `require_sorted_ranks` before any page write, because the
//! expert products of the wave run `gather_qmm_sorted` with MLX's `sorted_indices` flag (packet
//! FN-PF-SORTED).
//! The release rule of decision 0041 is unchanged: the last set of a layer is released after the
//! router readback of the next layer (`release_lent`), and `acquire` never waits. A wave before
//! the last is released right after the `evaluate` that covers its graph (packet PF-7b).
//!
//! Every byte that a chunk imports is written by that chunk. The pool never reads the S3 expert
//! cache and never writes to it (decision 0044 item 7).

#[cfg(feature = "native-fault-qualification")]
use super::super::super::stored::record_page_import;
use super::super::super::{
    prefill_plan::{Wave, require_sorted_ranks},
    stored::read_stack_into,
};
use super::super::{
    ops::short_import,
    slots::{ExpertSlots, SetLayout},
    weights::RMoe,
};
use crate::{
    EngError,
    kernels::{HostPages, MlxKernels, PageRange, ResidentPackedWeight},
    tensor::{Dtype, ExecutionTensor},
    weights::StreamedCheckpoint,
};

/// Page sets in the pool. Two suffice: layer `L` reuses the set of layer `L - 2`.
const PREFILL_SETS: usize = 2;

/// Returns a `BadRequest` with `message`.
fn bad(message: &str) -> EngError {
    EngError::BadRequest(message.into())
}

/// The page layout of one prefill set: the stacks of a `SetLayout`, then the route page.
pub(super) struct PrefillLayout {
    /// The three expert stacks. Its own gate page is not used: the route page replaces it.
    pub(super) set: SetLayout,
    /// The U32 wave-local rank of each pair, `[pairs]`. It starts where the gate page of `set`
    /// starts.
    pub(super) ranks: PageRange,
    /// The U32 token row of each pair, `[pairs]`, on the next page boundary.
    pub(super) tokens: PageRange,
    /// The F32 gate of each pair, `[pairs]`, on the next page boundary.
    pub(super) gates: PageRange,
    /// The U32 inverse permutation of the chunk, `[pairs]`, on the next page boundary.
    pub(super) inverse: PageRange,
    /// Bytes from the set start to the end of the inverse permutation, rounded up to a page.
    pub(super) end: usize,
}

impl PrefillLayout {
    /// Places `experts` stacked experts of `w`, then four route ranges of room for `pairs` words
    /// each. A call shortens the shapes to the words it writes.
    ///
    /// The route ranges start after the stacks of `experts`, so their offsets move with the
    /// expert count of the wave. Every range ends inside `end`, and `end` does not exceed the
    /// set size of the pool when `experts` is at most `set_experts`.
    pub(super) fn new(
        ckpt: &StreamedCheckpoint,
        w: &RMoe,
        experts: u32,
        pairs: u32,
        page: usize,
    ) -> Result<Self, EngError> {
        let set = SetLayout::new(ckpt, w, experts, page)?;
        let overflow = || bad("prefill set layout overflows");
        let words = usize::try_from(pairs)
            .ok()
            .and_then(|pairs| pairs.checked_mul(4))
            .filter(|&words| words > 0)
            .ok_or_else(|| bad("prefill set needs at least one pair"))?;
        let next = |at: usize| {
            at.checked_add(words)
                .and_then(|end| end.checked_next_multiple_of(page))
                .ok_or_else(overflow)
        };
        let ranks_at = set.gates.offset;
        let tokens_at = next(ranks_at)?;
        let gates_at = next(tokens_at)?;
        let inverse_at = next(gates_at)?;
        let end = next(inverse_at)?;
        let range = |offset: usize, dtype: Dtype| PageRange {
            offset,
            dtype,
            shape: vec![pairs],
        };
        Ok(Self {
            ranks: range(ranks_at, Dtype::U32),
            tokens: range(tokens_at, Dtype::U32),
            gates: range(gates_at, Dtype::F32),
            inverse: range(inverse_at, Dtype::U32),
            set,
            end,
        })
    }
}

/// The residents of one imported wave.
pub(super) struct WavePages {
    /// The gate, up, and down stacks, `[experts, rows, columns]`, in wave rank order.
    pub(super) stacks: [ResidentPackedWeight; 3],
    /// The wave-local rank of each pair of the wave, U32 `[pairs]`.
    pub(super) ranks: ExecutionTensor,
    /// The token row of each pair of the wave, U32 `[pairs]`.
    pub(super) tokens: ExecutionTensor,
    /// The gate of each pair of the wave, F32 `[pairs]`.
    pub(super) gates: ExecutionTensor,
    /// The inverse permutation of the chunk, U32 `[chunk pairs]`, when the caller asked for it.
    pub(super) inverse: Option<ExecutionTensor>,
}

/// Two page sets for the chunks of one prefill.
pub(super) struct PrefillPool {
    slots: ExpertSlots,
    set_experts: u32,
    pairs: u32,
}

impl PrefillPool {
    /// Maps two sets. Each holds up to `set_experts` experts and `pairs` pairs of any layer in
    /// `layers`, so its size is the largest layout of those layers.
    pub(super) fn new<'a>(
        ckpt: &StreamedCheckpoint,
        layers: impl IntoIterator<Item = &'a RMoe>,
        set_experts: u32,
        pairs: u32,
    ) -> Result<Self, EngError> {
        Self::with_sets(ckpt, layers, set_experts, pairs, PREFILL_SETS)
    }

    /// Like `new` with `count` sets. Tests use it to show the release rule.
    pub(super) fn with_sets<'a>(
        ckpt: &StreamedCheckpoint,
        layers: impl IntoIterator<Item = &'a RMoe>,
        set_experts: u32,
        pairs: u32,
        count: usize,
    ) -> Result<Self, EngError> {
        let page = HostPages::page_size();
        let mut bytes = 0usize;
        for w in layers {
            bytes = bytes.max(PrefillLayout::new(ckpt, w, set_experts, pairs, page)?.end);
        }
        if bytes == 0 {
            return Err(bad("the prefill pool needs at least one layer"));
        }
        Ok(Self {
            slots: ExpertSlots::new(count, bytes)?,
            set_experts,
            pairs,
        })
    }

    /// The most experts that one set holds. A chunk with more distinct experts needs waves.
    pub(super) fn set_experts(&self) -> u32 {
        self.set_experts
    }

    /// Mapped bytes of both sets. They do not change after `new`.
    pub(super) fn byte_len(&self) -> u64 {
        self.slots.byte_len()
    }

    /// Releases every lent set. Call it only after an evaluation that covers every graph that
    /// reads those sets, or when those graphs are dropped and will never be evaluated: right
    /// after the router readback of the next layer, and after the final `evaluate` of a chunk.
    pub(super) fn release_lent(&mut self) {
        self.slots.release_lent();
    }

    /// Releases every lent set and drops the pool. The prefill calls it after its last chunk, so
    /// no caller can leave a set lent. Dropping the pins is memory-safe at any time.
    pub(super) fn finish(mut self) {
        self.slots.release_lent();
    }

    /// Reads the experts and the route page of `wave` into a free set, imports them in place
    /// without a synchronization, and lends the set. With `inverse`, the route page also holds
    /// the inverse permutation of the chunk. The layout is that of the wave's expert count, so the
    /// route ranges move with it and stay inside the set.
    ///
    /// The checks run before any page changes and return `BadRequest`: more experts than
    /// `set_experts`, more pairs than the pool was sized for, lists of different lengths, and
    /// local ranks that fall (`require_sorted_ranks`, the non-decreasing order that
    /// `gather_qmm_sorted` requires; this is the gate that runs before any device work of the
    /// wave, because this import is the single entry point that writes and imports the wave
    /// buffers, whichever caller built the plan). The check runs once per wave at its own import,
    /// so a multi-wave layer may already have imported and run its earlier waves when a later
    /// wave is rejected; that cannot happen while `split_waves` is the only producer of waves,
    /// because it only emits waves with sorted ranks. When no set is free it returns `Backend`
    /// (a release was missed).
    pub(super) fn import_wave(
        &mut self,
        kernels: &mut MlxKernels,
        ckpt: &StreamedCheckpoint,
        w: &RMoe,
        wave: &Wave,
        inverse: Option<&[u32]>,
    ) -> Result<WavePages, EngError> {
        let count = |len: usize| u32::try_from(len).map_err(|_| bad("prefill count exceeds u32"));
        let experts = count(wave.experts.len())?;
        let pairs = count(wave.pairs.len())?;
        let total = count(inverse.map_or(0, <[u32]>::len))?;
        if experts == 0 || experts > self.set_experts {
            return Err(EngError::BadRequest(format!(
                "the wave holds {experts} experts, the pool set holds {}",
                self.set_experts
            )));
        }
        let lists = [wave.local_ranks.len(), wave.tokens.len(), wave.gates.len()];
        if pairs == 0
            || pairs > self.pairs
            || total > self.pairs
            || lists.iter().any(|&len| len != wave.pairs.len())
        {
            return Err(bad("the wave pairs do not fit the pool"));
        }
        require_sorted_ranks(&wave.local_ranks)?;
        let mut layout = PrefillLayout::new(ckpt, w, experts, self.pairs, HostPages::page_size())?;
        layout.ranks.shape = vec![pairs];
        layout.tokens.shape = vec![pairs];
        layout.gates.shape = vec![pairs];
        layout.inverse.shape = vec![total];
        let set = self.slots.acquire()?;
        {
            let bytes = self.slots.write(set)?;
            if layout.end > bytes.len() {
                return Err(bad("the expert stacks do not fit one page set"));
            }
            for (stored, stack) in [&w.gate, &w.up, &w.down]
                .into_iter()
                .zip(&layout.set.stacks)
            {
                read_stack_into(ckpt, stored, &wave.experts, bytes, stack)?;
            }
            put_words(bytes, layout.ranks.offset, le_words(&wave.local_ranks))?;
            put_words(bytes, layout.tokens.offset, le_words(&wave.tokens))?;
            put_words(
                bytes,
                layout.gates.offset,
                wave.gates.iter().map(|gate| gate.to_le_bytes()),
            )?;
            if let Some(inverse) = inverse {
                put_words(bytes, layout.inverse.offset, le_words(inverse))?;
            }
        }
        let mut plain = vec![layout.ranks, layout.tokens, layout.gates];
        if inverse.is_some() {
            plain.push(layout.inverse);
        }
        #[cfg(feature = "native-fault-qualification")]
        let started = std::time::Instant::now();
        let imported =
            kernels.import_packed_pages(self.slots.pages(set)?, &layout.set.packed(), &plain);
        #[cfg(feature = "native-fault-qualification")]
        record_page_import(started.elapsed());
        let (stacks, plain) = imported?;
        self.slots.lend(set);
        let stacks: [ResidentPackedWeight; 3] = stacks.try_into().map_err(|_| short_import())?;
        let mut plain = plain.into_iter();
        let mut next = || plain.next().ok_or_else(short_import);
        Ok(WavePages {
            stacks,
            ranks: next()?,
            tokens: next()?,
            gates: next()?,
            inverse: if inverse.is_some() {
                Some(next()?)
            } else {
                None
            },
        })
    }

    /// Fills every byte of every set with `byte`. Only unlent sets are writable.
    #[cfg(test)]
    pub(super) fn poison(&mut self, byte: u8) {
        for set in 0..PREFILL_SETS {
            if let Ok(bytes) = self.slots.write(set) {
                bytes.fill(byte);
            }
        }
    }
}

/// The little-endian bytes of each word of `list`.
fn le_words(list: &[u32]) -> impl ExactSizeIterator<Item = [u8; 4]> + '_ {
    list.iter().map(|word| word.to_le_bytes())
}

/// Writes the little-endian `words` at `offset` of `bytes`.
fn put_words(
    bytes: &mut [u8],
    offset: usize,
    words: impl ExactSizeIterator<Item = [u8; 4]>,
) -> Result<(), EngError> {
    let region = words
        .len()
        .checked_mul(4)
        .and_then(|len| len.checked_add(offset))
        .and_then(|end| bytes.get_mut(offset..end))
        .ok_or_else(|| bad("the route page passes the end of the set"))?;
    for (slot, word) in region.as_chunks_mut::<4>().0.iter_mut().zip(words) {
        *slot = word;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::super::prefill_plan::{ChunkRoute, split_waves};
    use super::super::super::{
        fixtures::config,
        moe::tests::{checkpoint_from, routed_entries},
        ops::values,
        weights::ResidentWeights,
    };
    use super::*;
    use crate::family::flash_next::load::FlashWeights;

    /// Loads the routed fixture: 4 experts of width 64 and 32, top-k 2, two layers.
    fn load(kernels: &mut MlxKernels) -> (StreamedCheckpoint, ResidentWeights) {
        let ckpt = checkpoint_from(routed_entries());
        let flash = FlashWeights::load(&ckpt, &config()).unwrap();
        let weights = ResidentWeights::import(kernels, &flash).unwrap();
        (ckpt, weights)
    }

    /// Every stack of 3 experts has its payload, scales, and biases in one page each: the payload
    /// is 3 * 1,024 B and the scales and biases are 3 * 128 B per stack. Three stacks use 9
    /// pages. The ranks start at page 9, where the gate page of `SetLayout` would start. Each
    /// route range takes 8 * 4 = 32 B, so one page: tokens at page 10, gates at page 11, inverse
    /// at page 12, and the set ends at page 13.
    #[test]
    fn the_route_page_follows_the_stacks_on_page_boundaries() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (ckpt, weights) = load(&mut kernels);
        let page = HostPages::page_size();
        let layout = PrefillLayout::new(&ckpt, &weights.layers[0].moe, 3, 8, page).unwrap();
        assert_eq!(
            layout.set.stacks[2].end,
            9 * page,
            "end of the three stacks"
        );
        assert_eq!(layout.ranks.offset, 9 * page, "ranks offset");
        assert_eq!(layout.ranks.dtype, Dtype::U32, "ranks dtype");
        assert_eq!(layout.ranks.shape, [8], "ranks shape");
        assert_eq!(layout.tokens.offset, 10 * page, "tokens offset");
        assert_eq!(layout.tokens.dtype, Dtype::U32, "tokens dtype");
        assert_eq!(layout.gates.offset, 11 * page, "gates offset");
        assert_eq!(layout.gates.dtype, Dtype::F32, "gates dtype");
        assert_eq!(layout.gates.shape, [8], "gates shape");
        assert_eq!(layout.inverse.offset, 12 * page, "inverse offset");
        assert_eq!(layout.inverse.dtype, Dtype::U32, "inverse dtype");
        assert_eq!(layout.end, 13 * page, "set end");
    }

    /// The route of the P2 and P4 tests: 8 pairs of 2 choices over the experts 0, 1, and 3.
    fn route() -> ChunkRoute {
        ChunkRoute {
            distinct: vec![0, 1, 3],
            rank_idx: vec![0, 1, 2, 1, 0, 1, 0, 1],
            gates: vec![0.5, 0.25, 0.125, 0.75, 1.0, 0.0, 0.625, 0.375],
            top_k: 2,
        }
    }

    /// The U32 ranks, tokens, and inverse permutation and the F32 gates come back as written, and
    /// the stacks have the rank-order shape `[3, rows, columns]`. One wave holds the whole route,
    /// so its local ranks are the chunk ranks, and pair `p` belongs to token `p / 2`. The wave
    /// lists its pairs in (rank, pair) order (`split_waves`): p0, p4, and p6 of rank 0 (tokens
    /// 0, 2, 3), p1, p3, p5, and p7 of rank 1 (tokens 0, 1, 2, 3), and p2 of rank 2 (token 1),
    /// so the ranks read 0, 0, 0, 1, 1, 1, 1, 2, the tokens 0, 2, 3, 0, 1, 2, 3, 1, and the
    /// gates follow the pairs in that order.
    #[test]
    fn the_route_page_imports_u32_ranks_tokens_inverse_and_f32_gates() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (ckpt, weights) = load(&mut kernels);
        let layers = weights.layers.iter().map(|layer| &layer.moe);
        let mut pool = PrefillPool::new(&ckpt, layers, 4, 8).unwrap();
        let route = route();
        let plan = split_waves(&route, 4).unwrap();
        let inverse = [7u32, 6, 5, 4, 3, 2, 1, 0];
        let pages = pool
            .import_wave(
                &mut kernels,
                &ckpt,
                &weights.layers[0].moe,
                &plan.waves[0],
                Some(&inverse),
            )
            .unwrap();
        let inverse_tensor = pages.inverse.as_ref().unwrap();
        for tensor in [&pages.ranks, &pages.tokens, inverse_tensor] {
            assert_eq!(tensor.dtype(), Dtype::U32, "index dtype");
            assert_eq!(tensor.shape(), [8], "index shape");
        }
        assert_eq!(pages.gates.dtype(), Dtype::F32, "gates dtype");
        assert_eq!(pages.gates.shape(), [8], "gates shape");
        let bytes = |list: &[u32]| {
            list.iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<u8>>()
        };
        for (name, tensor, want) in [
            ("ranks", &pages.ranks, bytes(&[0, 0, 0, 1, 1, 1, 1, 2])),
            ("tokens", &pages.tokens, bytes(&[0, 2, 3, 0, 1, 2, 3, 1])),
            ("inverse", inverse_tensor, bytes(&inverse)),
        ] {
            let host = kernels.readback(tensor).unwrap();
            assert_eq!(host.host_bytes(), Some(want.as_slice()), "{name} bytes");
        }
        let gates = values(&mut kernels, &pages.gates).unwrap();
        assert_eq!(
            gates,
            [0.5, 1.0, 0.625, 0.25, 0.75, 0.0, 0.375, 0.125],
            "gates in (rank, pair) order"
        );
        let shapes: Vec<&[u32]> = pages.stacks.iter().map(|s| s.logical_shape()).collect();
        assert_eq!(
            shapes,
            [&[3, 32, 64][..], &[3, 32, 64][..], &[3, 64, 32][..]],
            "stack shapes"
        );
    }

    /// A wave shorter than the chunk shortens the shapes: the second wave of cap 2 holds the
    /// expert 3 and the one pair of token 1.
    #[test]
    fn a_short_wave_imports_only_its_own_pairs() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (ckpt, weights) = load(&mut kernels);
        let layers = weights.layers.iter().map(|layer| &layer.moe);
        let mut pool = PrefillPool::new(&ckpt, layers, 2, 8).unwrap();
        let plan = split_waves(&route(), 2).unwrap();
        assert_eq!(plan.waves.len(), 2, "two waves");
        let moe_w = &weights.layers[0].moe;
        let first = pool
            .import_wave(&mut kernels, &ckpt, moe_w, &plan.waves[0], None)
            .unwrap();
        assert_eq!(first.ranks.shape(), [7], "first wave pairs");
        assert!(first.inverse.is_none(), "no inverse requested");
        drop(first);
        let second = pool
            .import_wave(&mut kernels, &ckpt, moe_w, &plan.waves[1], None)
            .unwrap();
        assert_eq!(second.ranks.shape(), [1], "second wave pairs");
        assert_eq!(second.stacks[0].logical_shape(), [1, 32, 64], "one expert");
        let host = kernels.readback(&second.tokens).unwrap();
        assert_eq!(
            host.host_bytes(),
            Some(&1u32.to_le_bytes()[..]),
            "token of the pair"
        );
    }

    #[test]
    fn a_wave_beyond_the_set_or_the_pairs_is_rejected_before_any_page_write() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (ckpt, weights) = load(&mut kernels);
        let layers = weights.layers.iter().map(|layer| &layer.moe);
        let moe_w = &weights.layers[0].moe;
        let mut pool = PrefillPool::new(&ckpt, layers, 2, 8).unwrap();
        let wave = split_waves(&route(), 4).unwrap().waves.remove(0);
        let mut bad_request = |wave: &Wave, inverse: Option<&[u32]>| {
            let result = pool.import_wave(&mut kernels, &ckpt, moe_w, wave, inverse);
            matches!(result, Err(EngError::BadRequest(_)))
        };
        assert!(bad_request(&wave, None), "three experts in a set of two");
        let mut small = wave.clone();
        small.experts.truncate(2);
        assert!(!bad_request(&small, None), "two experts fit");
        let mut long = small.clone();
        for list in [&mut long.pairs, &mut long.local_ranks, &mut long.tokens] {
            list.push(0);
        }
        long.gates.push(0.0);
        assert!(bad_request(&long, None), "nine pairs in a pool of eight");
        let mut ragged = small.clone();
        ragged.tokens.pop();
        assert!(
            bad_request(&ragged, None),
            "token list shorter than the pairs"
        );
        assert!(
            bad_request(&small, Some(&[0u32; 9][..])),
            "inverse longer than the pool"
        );
    }

    /// Packet FN-PF-SORTED task 2 and the fix round: the import proves the wave order before any
    /// device work of the wave. A wave whose local ranks fall (one inversion suffices) is a
    /// `BadRequest` of the same pre-write group, whatever built the wave, and a sorted wave
    /// imports into the same pool right after the rejection, which shows that the rejected
    /// import checked the ranks before it acquired a set or wrote a page and left the pool
    /// usable. The checks assert rather than `panic!` on the wrong shape, per decision 0032 (a
    /// panicking native test marks the shared native owner failed and cascades).
    #[test]
    fn an_unsorted_wave_is_rejected_before_any_page_write() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (ckpt, weights) = load(&mut kernels);
        let layers = weights.layers.iter().map(|layer| &layer.moe);
        let mut pool = PrefillPool::new(&ckpt, layers, 4, 8).unwrap();
        let mut wave = split_waves(&route(), 4).unwrap().waves.remove(0);
        assert!(
            wave.local_ranks.is_sorted(),
            "the plan hands over sorted ranks"
        );
        wave.local_ranks.reverse();
        let rejected = pool.import_wave(&mut kernels, &ckpt, &weights.layers[0].moe, &wave, None);
        let failure = rejected.err();
        assert!(
            matches!(&failure, Some(EngError::BadRequest(m)) if m.starts_with("flash-next prefill:")),
            "one inversion is a bad request that names the flash-next prefill: {failure:?}"
        );
        let sorted = split_waves(&route(), 4).unwrap().waves.remove(0);
        let imported = pool.import_wave(&mut kernels, &ckpt, &weights.layers[0].moe, &sorted, None);
        let failure = imported.err();
        assert!(
            failure.is_none(),
            "the pool imports a sorted wave after the rejection: {failure:?}"
        );
    }

    #[test]
    fn the_pool_maps_two_sets_of_the_largest_layer_layout() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (ckpt, weights) = load(&mut kernels);
        let layers = || weights.layers.iter().map(|layer| &layer.moe);
        let pool = PrefillPool::new(&ckpt, layers(), 4, 8).unwrap();
        let page = u64::try_from(HostPages::page_size()).unwrap();
        assert_eq!(pool.byte_len(), 2 * 13 * page, "two sets of 13 pages");
        assert_eq!(pool.set_experts(), 4, "experts per set");
        assert!(
            matches!(
                PrefillPool::new(&ckpt, std::iter::empty(), 4, 8),
                Err(EngError::BadRequest(_))
            ),
            "no layer"
        );
        assert!(
            matches!(
                PrefillPool::new(&ckpt, layers(), 4, 0),
                Err(EngError::BadRequest(_))
            ),
            "no pair"
        );
    }
}
