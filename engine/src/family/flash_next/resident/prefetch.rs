//! Bounded expert hints only. No storage, kernel, routing, or page-lifetime operation belongs here.

/// Flash-Next selects ten experts per token. A hint can carry at most ten predicted ids.
pub(super) const MAX_EXPERT_IDS: usize = 10;

/// The input token being processed, not the token that the head will emit.
/// The position distinguishes repeated occurrences of the same vocabulary id in this step.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Staging hints is a later slice.")
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TokenIdentity {
    pub(super) position: u32,
    pub(super) id: u32,
}

/// A speculative set for the next layer of the same input token.
/// Each occupied entry is one predicted id; the fixed array bounds the set without allocation.
/// This record has no session identity and must not be consumed across steps or restores.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Staging hints is a later slice.")
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExpertHint {
    pub(super) layer: usize,
    pub(super) token: TokenIdentity,
    pub(super) expert_ids: [Option<u32>; MAX_EXPERT_IDS],
}

/// Substitutable producer seam. It receives no routing output or page-pool access.
pub(super) type Producer = fn(usize, TokenIdentity) -> ExpertHint;

/// STUB: fixed predictions, not a router or a learned predictor. No performance claim.
fn stub(layer: usize, token: TokenIdentity) -> ExpertHint {
    let mut expert_ids = [None; MAX_EXPERT_IDS];
    expert_ids[..3].copy_from_slice(&[Some(0), Some(1), Some(2)]);
    ExpertHint {
        layer,
        token,
        expert_ids,
    }
}

/// Default-off dispatch after a layer's residual update. The producer is replaceable by tests.
pub(super) struct Prefetch {
    pub(super) enabled: bool,
    pub(super) producer: Producer,
}

impl Default for Prefetch {
    fn default() -> Self {
        Self {
            enabled: false,
            producer: stub,
        }
    }
}

impl Prefetch {
    /// Calls neither seam when disabled or after the final layer.
    pub(super) fn after_layer(
        &self,
        layer: usize,
        layers: usize,
        token: TokenIdentity,
        admit: impl FnOnce(ExpertHint),
    ) {
        if !self.enabled {
            return;
        }
        if let Some(next) = layer.checked_add(1).filter(|&next| next < layers) {
            admit((self.producer)(next, token));
        }
    }
}

/// One overwrite-only pending hint beside the page pool. Its capacity never depends on demand.
#[derive(Default)]
pub(super) struct Admission {
    pending: Option<ExpertHint>,
    #[cfg(test)]
    calls: usize,
}

impl Admission {
    /// Records at most one hint, replacing the previous record, and returns immediately.
    /// No storage reads, waits, allocation, page reservations, demand reordering, or routing.
    /// The page pool cannot stage hints yet; staging is a later slice.
    pub(super) fn admit(&mut self, hint: ExpertHint) {
        self.pending = Some(hint);
        #[cfg(test)]
        {
            self.calls = self.calls.saturating_add(1);
        }
    }

    #[cfg(test)]
    pub(super) fn recorded(&self) -> (usize, Option<ExpertHint>) {
        (self.calls, self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_flag_gates_both_seams_and_the_final_layer_has_no_target() {
        let token = TokenIdentity { position: 7, id: 4 };
        let mut admission = Admission::default();
        let mut prefetch = Prefetch {
            enabled: false,
            producer: |_, _| panic!("disabled prefetch must not call the producer"),
        };
        for layer in 0..4 {
            prefetch.after_layer(layer, 4, token, |hint| admission.admit(hint));
        }
        assert_eq!(admission.recorded(), (0, None));

        prefetch.enabled = true;
        prefetch.producer = |layer, token| {
            let mut hint = stub(layer, token);
            hint.expert_ids[0] = Some(9);
            hint
        };
        for layer in 0..4 {
            prefetch.after_layer(layer, 4, token, |hint| admission.admit(hint));
            let (calls, pending) = admission.recorded();
            assert_eq!(calls, (layer + 1).min(3));
            let hint = pending.unwrap();
            assert_eq!(hint.layer, (layer + 1).min(3));
            assert_eq!(hint.token, token);
            assert_eq!(
                hint.expert_ids,
                [
                    Some(9),
                    Some(1),
                    Some(2),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None
                ]
            );
        }

        let mut targets = Vec::new();
        for layer in 0..4 {
            Prefetch {
                enabled: true,
                ..Prefetch::default()
            }
            .after_layer(layer, 4, token, |hint| targets.push(hint.layer));
        }
        assert_eq!(targets, [1, 2, 3]);
        assert_eq!(
            stub(1, token).expert_ids,
            [
                Some(0),
                Some(1),
                Some(2),
                None,
                None,
                None,
                None,
                None,
                None,
                None
            ]
        );
    }
}
