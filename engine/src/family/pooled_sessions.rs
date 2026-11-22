//! Several sessions on one resident step, time sliced.
//!
//! The step's weights, context, stream, activation arena, prefill arena and prefix store are
//! shared. A session owns only its sequence part (the host's `Part`): KV planes, DeltaNet and convolution
//! state, step words and the graph captured over them. The cache cells come from one pool
//! (`family::cell_pool`): a session leases the cells its request needs when the request is
//! admitted, and a request the pool cannot fund yet queues in the worker.
//!
//! The step runs the part of the session that was used last. Using another session swaps the two
//! parts (a handle exchange, no device work); a window of rows runs the rows one after another,
//! each through the step's own single-row decode, so a row's tokens are the tokens the same
//! request gets alone. An engine that serves one session at a time never builds this type.

#![cfg_attr(
    not(feature = "native-cuda"),
    allow(
        dead_code,
        reason = "the CUDA-build executors serve their sessions through it"
    )
)]

use crate::{
    EngError,
    family::{
        cell_pool::{
            CellLease, CellPool, CellRequest, CellRoom, INITIAL_CELLS, even_share,
            session_cells_or_queue,
        },
        executor::{BatchPolicy, ExecutorOps, GenerationState},
        session::{DecodeSession, SaveKind, StoreLimits},
    },
    sched::job::PrefixReport,
    tokenizer::DecodeBatch,
};
use std::{cell::RefCell, rc::Rc, sync::Arc};

type Result<T> = std::result::Result<T, EngError>;

/// What a session keeps on the engine thread: its cell lease, and its sequence part while the step
/// does not run it.
struct SeqSlot<P> {
    inner: RefCell<SlotInner<P>>,
}

struct SlotInner<P> {
    part: Option<P>,
    lease: CellLease,
}

/// What a resident step gives the sessions executor: the part of it that one sequence owns, and
/// the exchange of that part. The weights, context, stream, arenas and prefix store stay the
/// step's own and shared.
pub(crate) trait PartHost: ExecutorOps<Session = DecodeSession> {
    /// One sequence's state on the device.
    type Part;
    /// The prefix of the host's log lines.
    const LABEL: &'static str;
    /// Whether this step can serve more than one session.
    fn check_sessions(&self) -> Result<()>;
    /// The cells of the context the step was built for, which become the pool.
    fn context_cells(&self) -> usize;
    /// A zeroed part holding `cells` positions.
    fn new_part(&mut self, cells: usize) -> Result<Self::Part>;
    /// Exchanges the part the step runs with `part`.
    fn swap_part(&mut self, part: &mut Self::Part);
    /// The cells `part` holds.
    fn part_cells(part: &Self::Part) -> usize;
    /// Device bytes of a part holding `cells` positions.
    fn part_bytes_at(&self, cells: usize) -> u64;
    /// Free device memory now, when the card can be read.
    fn free_device_bytes(&self) -> Option<u64>;
    /// Finishes a save staged for the part that is about to leave.
    fn finish_save(&mut self) -> Result<()>;
    /// Discards a staged save without publishing it when its owner's part is retired.
    fn discard_save(&mut self);
}

/// How a session's cells are sized until a session can grow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sizing {
    /// A session takes every cell it can reach: the first one the whole context it would get
    /// alone, later ones what is left. A request the pool cannot fund waits for a session to
    /// let go; no live session is shrunk.
    WholePool,
    /// The pool is split among `max_seqs` sessions by what each request needs.
    EvenShare,
}

/// One session of a [`PooledSessions`] engine.
pub(crate) struct PooledSession<P> {
    core: DecodeSession,
    slot: Rc<SeqSlot<P>>,
}

/// The executor of several sessions over one resident step.
pub(crate) struct PooledSessions<E: PartHost> {
    // Declaration order is drop order: the parts (and their graphs) go before the step.
    /// The small part the step runs while no session does. It is `Some` only while a session's
    /// part runs in the step, and then it waits here.
    idle: Option<E::Part>,
    /// The session whose part the step runs now; `None` when the idle part runs.
    resident: Option<Rc<SeqSlot<E::Part>>>,
    pool: Arc<CellPool>,
    /// The most cells one session holds: the context the card plan chose.
    ceiling: usize,
    /// The even share of the pool among the sessions.
    share: usize,
    max_seqs: usize,
    sizing: Sizing,
    /// Whether the last session request found the pool empty and said so.
    waiting: std::cell::Cell<bool>,
    step: E,
}

impl<E: PartHost> PooledSessions<E> {
    /// Takes over a built step and the session the engine opened with. The step's planes were
    /// sized for the whole context and become the pool's cells. With [`Sizing::WholePool`] the
    /// first session keeps those planes, so a lone client runs the state it always ran; with
    /// [`Sizing::EvenShare`] they go back to the card and every session leases by its request.
    pub(crate) fn new(
        mut step: E,
        core: DecodeSession,
        max_seqs: usize,
        sizing: Sizing,
    ) -> Result<(Self, PooledSession<E::Part>)> {
        step.check_sessions()?;
        let ceiling = step.context_cells();
        let pool = CellPool::new(ceiling);
        let initial = INITIAL_CELLS.min(ceiling);
        let share = even_share(ceiling, max_seqs);
        let small = step.part_bytes_at(initial);
        let (idle, resident, slot) = match sizing {
            Sizing::WholePool => {
                let lease = pool.lease_for(ceiling, ceiling)?;
                let idle = step.new_part(initial)?;
                let slot = Rc::new(SeqSlot {
                    inner: RefCell::new(SlotInner { part: None, lease }),
                });
                (Some(idle), Some(Rc::clone(&slot)), slot)
            }
            Sizing::EvenShare => {
                let lease = pool.lease_for(initial, ceiling)?;
                let mut small = step.new_part(initial)?;
                step.swap_part(&mut small);
                drop(small);
                let part = step.new_part(initial)?;
                let slot = Rc::new(SeqSlot {
                    inner: RefCell::new(SlotInner {
                        part: Some(part),
                        lease,
                    }),
                });
                (None, None, slot)
            }
        };
        match sizing {
            Sizing::WholePool => eprintln!(
                "{}: sessions: one pool of {} cells; the first request takes the whole \
                 context, later ones take what is left or wait for a session to finish \
                 (at most {max_seqs}; set max_seqs to split the pool evenly)",
                E::LABEL,
                pool.total()
            ),
            Sizing::EvenShare => eprintln!(
                "{}: sessions {max_seqs}: one pool of {} cells, up to {ceiling} for one \
                 session, even share {share}, {small} B of state per session at start",
                E::LABEL,
                pool.total()
            ),
        }
        let this = Self {
            idle,
            resident,
            pool,
            ceiling,
            share,
            max_seqs,
            sizing,
            waiting: std::cell::Cell::new(false),
            step,
        };
        Ok((this, PooledSession { core, slot }))
    }

    /// Makes `slot`'s part the one the step runs.
    fn activate(&mut self, slot: &Rc<SeqSlot<E::Part>>) -> Result<()> {
        if self
            .resident
            .as_ref()
            .is_some_and(|resident| Rc::ptr_eq(resident, slot))
        {
            return Ok(());
        }
        // A save staged for the part that leaves finishes before the parts change places.
        self.step.finish_save()?;
        let Some(mut part) = slot.inner.borrow_mut().part.take() else {
            return Err(EngError::Backend {
                operation: "qwen36_cuda_sessions",
                message: "the session's sequence part is missing".to_owned(),
            });
        };
        self.step.swap_part(&mut part);
        match self.resident.replace(Rc::clone(slot)) {
            Some(left) => left.inner.borrow_mut().part = Some(part),
            None => self.idle = Some(part),
        }
        Ok(())
    }

    /// Whether the session whose part runs in the step was dropped: the executor's own handle is
    /// then the only one left, and the part and the cells it holds belong to nobody.
    fn orphaned(&self) -> bool {
        self.resident
            .as_ref()
            .is_some_and(|resident| Rc::strong_count(resident) == 1)
    }

    /// Gives the planes and the cells of a dropped resident session back to the card and the
    /// pool. Every way a session can be dropped (a failed setup, a failed window, a pool that
    /// trims idle sessions) leaves it here, so one call at each entry covers them all.
    fn reclaim(&mut self) {
        if self.orphaned() {
            self.deactivate();
        }
    }

    /// Puts the idle part under the step, so that the resident session's part is in its slot.
    fn deactivate(&mut self) {
        let Some(left) = self.resident.take() else {
            return;
        };
        self.step.discard_save();
        let mut idle = self.idle.take().expect("the idle part is present");
        self.step.swap_part(&mut idle);
        left.inner.borrow_mut().part = Some(idle);
    }

    /// Sizes the session's planes to `cells`: the old planes go to the card before the new ones
    /// are allocated, and the new planes start zeroed with no captured graph.
    fn resize(&mut self, session: &mut PooledSession<E::Part>, cells: usize) -> Result<()> {
        let resident = self
            .resident
            .as_ref()
            .is_some_and(|resident| Rc::ptr_eq(resident, &session.slot));
        let held = if resident {
            Some(self.step.context_cells())
        } else {
            session.slot.inner.borrow().part.as_ref().map(E::part_cells)
        };
        if session.slot.inner.borrow().lease.cells() == cells && held == Some(cells) {
            return Ok(());
        }
        // Resizing a different slot must not discard the live owner's save.
        if !resident && self.resident.is_some() {
            self.step.finish_save()?;
        }
        self.deactivate();
        let small = INITIAL_CELLS.min(self.ceiling);
        session
            .slot
            .inner
            .borrow_mut()
            .lease
            .resize_for(cells, self.ceiling)?;
        drop(session.slot.inner.borrow_mut().part.take());
        match self.step.new_part(cells) {
            Ok(part) => {
                session.slot.inner.borrow_mut().part = Some(part);
                Ok(())
            }
            Err(error) => {
                // The planes could not be built: the session keeps the smallest planes it can,
                // and its lease follows them.
                let mut inner = session.slot.inner.borrow_mut();
                inner.lease.resize(small);
                inner.part = self.step.new_part(small).ok();
                Err(error)
            }
        }
    }

    /// Device bytes the planes of one more session need at the start.
    fn free_device_bytes(&self) -> Option<u64> {
        self.step.free_device_bytes()
    }
}

impl<E: PartHost> ExecutorOps for PooledSessions<E> {
    type Session = PooledSession<E::Part>;

    fn max_seqs(&self) -> usize {
        self.max_seqs
    }

    /// The pool must hold a first lease, and the device must hold the planes with room to spare.
    fn can_admit_session(&self) -> bool {
        // A dropped session's cells come back at the next admission.
        if self.pool.free() < INITIAL_CELLS && !self.orphaned() {
            return false;
        }
        let per_session = self.step.part_bytes_at(INITIAL_CELLS);
        self.free_device_bytes()
            .is_none_or(|free| free >= per_session.saturating_mul(2))
    }

    /// Prefix entries live in host memory and the sessions on the device: dropping an entry does
    /// not make room for a session.
    fn eviction_frees_session_memory(&self) -> bool {
        false
    }

    fn batch_policy(&self) -> BatchPolicy {
        BatchPolicy::default()
    }

    fn new_session(&mut self) -> Result<PooledSession<E::Part>> {
        self.reclaim();
        let cells = INITIAL_CELLS.min(self.ceiling);
        let lease = match self.pool.lease_for(cells, self.ceiling) {
            Ok(lease) => lease,
            Err(error) => {
                if !self.waiting.replace(true) {
                    eprintln!(
                        "{}: a request waits for a session: the pool holds {} of {} cells \
                         free and the live sessions hold the rest; it runs when one finishes",
                        E::LABEL,
                        self.pool.free(),
                        self.pool.total()
                    );
                }
                return Err(error);
            }
        };
        self.waiting.set(false);
        let part = self.step.new_part(cells)?;
        Ok(PooledSession {
            core: DecodeSession::default(),
            slot: Rc::new(SeqSlot {
                inner: RefCell::new(SlotInner {
                    part: Some(part),
                    lease,
                }),
            }),
        })
    }

    fn check_prompt(&self, prompt_tokens: usize) -> Result<()> {
        crate::family::session::check_prompt(prompt_tokens, self.ceiling)
    }

    fn prefix_limits(&self) -> StoreLimits {
        ExecutorOps::prefix_limits(&self.step)
    }

    fn history<'s>(&self, st: &'s PooledSession<E::Part>) -> &'s [u32] {
        st.core.history()
    }

    fn reset(&mut self, st: &mut PooledSession<E::Part>) -> Result<()> {
        self.activate(&st.slot)?;
        ExecutorOps::reset(&mut self.step, &mut st.core)
    }

    /// Sizes the session's cells to the request, leased from the pool: a request that names its
    /// generation limit holds the prompt and the limit, one that names none holds the even share,
    /// and a prompt above the share holds what it needs while the pool can fund it. A pool that
    /// cannot fund the prompt yet refuses with a memory-plan error and the worker queues the
    /// request until a session lets go. This runs before any prefix restore.
    fn reset_for_request(
        &mut self,
        st: &mut PooledSession<E::Part>,
        prompt_tokens: usize,
        completion_limit: Option<u32>,
        _has_prefix_boundary: bool,
    ) -> Result<()> {
        self.reclaim();
        let reachable = st.slot.inner.borrow().lease.reachable();
        let room = CellRoom {
            reachable,
            ceiling: self.ceiling,
            share: match self.sizing {
                Sizing::WholePool => reachable,
                Sizing::EvenShare => self.share,
            },
        };
        // Until a session can grow, a whole-pool session holds every cell it can reach whatever
        // limit the request names, as a lone client always had.
        let completion = match self.sizing {
            Sizing::WholePool => None,
            Sizing::EvenShare => completion_limit,
        };
        let request = CellRequest {
            prompt: prompt_tokens,
            completion,
        };
        let cells = session_cells_or_queue(request, room)?;
        self.resize(st, cells)?;
        self.reset(st)
    }

    fn prefill(&mut self, st: &mut PooledSession<E::Part>, toks: &[u32]) -> Result<()> {
        self.activate(&st.slot)?;
        ExecutorOps::prefill(&mut self.step, &mut st.core, toks)
    }

    fn prefill_piece(&self) -> Option<usize> {
        ExecutorOps::prefill_piece(&self.step)
    }

    fn decode(
        &mut self,
        st: &mut PooledSession<E::Part>,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch> {
        self.activate(&st.slot)?;
        ExecutorOps::decode(&mut self.step, &mut st.core, generation)
    }

    /// The rows run one after another on the shared step, each through the single-row decode: a
    /// row's tokens are the ones the same request gets alone. A failed row is its own error; the
    /// step's state of that row is poisoned and the worker resets it.
    fn decode_window(
        &mut self,
        sessions: &mut [&mut PooledSession<E::Part>],
        generations: &mut [&mut GenerationState],
    ) -> Result<Vec<Result<DecodeBatch>>> {
        if sessions.is_empty()
            || sessions.len() != generations.len()
            || sessions.len() > self.max_seqs
        {
            return Err(EngError::BadRequest(
                "a decode window needs equal nonempty slices within max_seqs".into(),
            ));
        }
        Ok(sessions
            .iter_mut()
            .zip(generations.iter_mut())
            .map(|(session, generation)| self.decode(session, generation))
            .collect())
    }

    fn after_emit(&mut self) -> Result<()> {
        ExecutorOps::after_emit(&mut self.step)
    }

    fn boundary(&mut self, st: &mut PooledSession<E::Part>) {
        ExecutorOps::boundary(&mut self.step, &mut st.core);
    }

    fn background_pending(&self) -> bool {
        ExecutorOps::background_pending(&self.step)
    }

    fn background_step(&mut self) {
        ExecutorOps::background_step(&mut self.step);
    }

    fn background_drain(&mut self) {
        ExecutorOps::background_drain(&mut self.step);
    }

    fn save_prefix(
        &mut self,
        st: &PooledSession<E::Part>,
        key: &[u32],
        kind: SaveKind,
    ) -> Result<bool> {
        // The entry is exported from the part the step runs, so the session's part must be it,
        // and the prefix store is laid out for the whole context: a session cut to fewer cells
        // saves nothing.
        let resident = self
            .resident
            .as_ref()
            .is_some_and(|resident| Rc::ptr_eq(resident, &st.slot));
        if !resident || self.step.context_cells() != self.ceiling {
            return Ok(false);
        }
        ExecutorOps::save_prefix(&mut self.step, &st.core, key, kind)
    }

    fn prefix_report(&self) -> Option<PrefixReport> {
        ExecutorOps::prefix_report(&self.step)
    }

    fn restore_prefix(&mut self, st: &mut PooledSession<E::Part>, prompt: &[u32]) -> Result<usize> {
        self.activate(&st.slot)?;
        if self.step.context_cells() != self.ceiling {
            return Ok(0);
        }
        ExecutorOps::restore_prefix(&mut self.step, &mut st.core, prompt)
    }
}

impl<E: PartHost> Drop for PooledSessions<E> {
    fn drop(&mut self) {
        // Every part leaves the step through its slot before the step's own drop synchronizes.
        self.deactivate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host with no device: a part is its cell count, and the step runs one part.
    struct Fake {
        running: usize,
        pending_owner: Option<usize>,
        saved: bool,
    }

    impl ExecutorOps for Fake {
        type Session = DecodeSession;

        fn history<'s>(&self, st: &'s DecodeSession) -> &'s [u32] {
            st.history()
        }

        fn reset(&mut self, _st: &mut DecodeSession) -> Result<()> {
            Ok(())
        }

        fn prefill(&mut self, _st: &mut DecodeSession, _toks: &[u32]) -> Result<()> {
            Ok(())
        }

        fn decode(
            &mut self,
            _st: &mut DecodeSession,
            _generation: &mut GenerationState,
        ) -> Result<DecodeBatch> {
            Err(EngError::BadRequest("the fake host does not decode".into()))
        }

        fn boundary(&mut self, _st: &mut DecodeSession) {}
    }

    impl PartHost for Fake {
        type Part = usize;
        const LABEL: &'static str = "fake";

        fn check_sessions(&self) -> Result<()> {
            Ok(())
        }

        fn context_cells(&self) -> usize {
            self.running
        }

        fn new_part(&mut self, cells: usize) -> Result<usize> {
            Ok(cells)
        }

        fn swap_part(&mut self, part: &mut usize) {
            std::mem::swap(&mut self.running, part);
        }

        fn part_cells(part: &usize) -> usize {
            *part
        }

        fn part_bytes_at(&self, cells: usize) -> u64 {
            cells as u64
        }

        fn free_device_bytes(&self) -> Option<u64> {
            None
        }

        fn finish_save(&mut self) -> Result<()> {
            if let Some(owner) = self.pending_owner.take() {
                assert_eq!(self.running, owner, "save finished on a different part");
                self.saved = true;
            }
            Ok(())
        }

        fn discard_save(&mut self) {
            if let Some(owner) = self.pending_owner.take() {
                assert_eq!(self.running, owner, "save discarded on a different part");
            }
        }
    }

    const POOL: usize = 110 * 1024;

    fn engine(max_seqs: usize, sizing: Sizing) -> (PooledSessions<Fake>, PooledSession<usize>) {
        PooledSessions::new(
            Fake {
                running: POOL,
                pending_owner: None,
                saved: false,
            },
            DecodeSession::default(),
            max_seqs,
            sizing,
        )
        .expect("sessions")
    }

    #[test]
    fn a_session_dropped_after_a_failed_setup_gives_the_whole_pool_back() {
        let (mut sessions, mut first) = engine(8, Sizing::WholePool);
        sessions
            .reset_for_request(&mut first, 100, None, false)
            .expect("the first request takes the pool");
        assert_eq!(sessions.pool.free(), 0);
        // The worker drops a session whose setup failed; its part still runs in the step.
        drop(first);
        assert!(
            sessions.can_admit_session(),
            "the dropped session's cells are reclaimable"
        );
        let mut second = sessions
            .new_session()
            .expect("the next request is admitted");
        sessions
            .reset_for_request(&mut second, 100, None, false)
            .expect("and takes the whole pool");
        assert_eq!(sessions.step.context_cells(), POOL);
    }

    #[test]
    fn reclaim_does_not_carry_a_cancelled_owners_save_to_the_idle_part() {
        let (mut sessions, first) = engine(2, Sizing::WholePool);
        sessions.step.pending_owner = Some(POOL);
        drop(first);
        let mut peer = sessions.new_session().expect("admit peer after reclaim");
        sessions.reset(&mut peer).expect("activate peer");
        sessions.prefill(&mut peer, &[1]).expect("peer progresses");
        assert!(sessions.step.pending_owner.is_none());
        assert!(!sessions.step.saved);
    }

    #[test]
    fn resize_does_not_carry_a_deferred_save_to_replacement_planes() {
        let (mut sessions, mut first) = engine(2, Sizing::WholePool);
        sessions.step.pending_owner = Some(POOL);
        sessions.resize(&mut first, POOL / 2).expect("resize");
        sessions.reset(&mut first).expect("activate replacement");
        assert!(sessions.step.pending_owner.is_none());
        assert!(!sessions.step.saved);
    }

    #[test]
    fn resizing_a_peer_finishes_the_live_owners_save_before_switching() {
        let (mut sessions, mut first) = engine(2, Sizing::EvenShare);
        let mut peer = sessions.new_session().expect("admit peer");
        sessions.resize(&mut first, POOL / 2).expect("size owner");
        sessions.reset(&mut first).expect("activate owner");
        sessions.step.pending_owner = Some(POOL / 2);
        sessions.resize(&mut peer, POOL / 4).expect("resize peer");
        sessions.reset(&mut peer).expect("activate peer");
        assert!(sessions.step.pending_owner.is_none());
        assert!(sessions.step.saved);
    }

    #[test]
    fn a_dropped_share_does_not_refuse_a_long_prompt_on_an_empty_pool() {
        let (mut sessions, mut first) = engine(2, Sizing::EvenShare);
        sessions
            .reset_for_request(&mut first, 100, None, false)
            .expect("the first request takes its share");
        assert_eq!(sessions.step.context_cells(), POOL / 2);
        drop(first);
        let mut lone = sessions.new_session().expect("admitted");
        sessions
            .reset_for_request(&mut lone, 70_000, None, false)
            .expect("a prompt above one share is served while the pool is empty");
        assert!(sessions.step.context_cells() >= 70_000);
    }

    #[test]
    fn a_live_session_keeps_its_cells() {
        let (mut sessions, mut first) = engine(8, Sizing::WholePool);
        sessions
            .reset_for_request(&mut first, 100, None, false)
            .expect("the first request takes the pool");
        assert!(!sessions.orphaned());
        assert!(matches!(
            sessions.new_session(),
            Err(EngError::MemoryPlan(_))
        ));
        assert_eq!(sessions.pool.free(), 0);
    }
}
