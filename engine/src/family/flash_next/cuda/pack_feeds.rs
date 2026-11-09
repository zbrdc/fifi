//! The host feeds of a token from the real pack: [`PackFeeds`] is the [`TokenFeeds`] of the driver.
//!
//! WHAT: [`TokenFeeds::begin`] reads the token's BF16 embedding row (`embedding::stream_row`, one
//! positioned read of 5 KiB), plans and reads its 16 n-gram rows (`PleFeed::plan` and `gather`) and
//! keeps the gathered rows for the update. [`TokenFeeds::ple_update`] hands the stream at the PLE
//! layer's input to `PleFeed::update`, which is the host reference's PLE step over those rows.
//!
//! WHY the rows are read in `begin`: the driver reads them before it writes anything to the card, so
//! a failed read leaves the state as it was. Overlapping the read with the layers before the PLE
//! layer is a later change (the rows depend on the token history only).

use super::{
    driver::{StartVectors, TokenFeeds},
    embedding::stream_row,
    ple_feed::{Gathered, PleFeed, PleKeyValue},
};
use crate::{EngError, weights::StreamedCheckpoint};

/// The gathered rows of the token being fed, with the token.
struct Pending<T> {
    token: u32,
    rows: T,
}

/// Takes the front pending rows, which must be `token`'s; a chunk consumes its rows in order.
fn take_pending<T>(pending: &mut Vec<Pending<T>>, token: u32) -> Result<T, EngError> {
    if pending.first().is_some_and(|front| front.token == token) {
        return Ok(pending.remove(0).rows);
    }
    Err(EngError::BadRequest(format!(
        "no n-gram rows were read for token {token}"
    )))
}

pub(super) struct PackFeeds {
    source: &'static StreamedCheckpoint,
    ple: PleFeed,
    hidden: usize,
    branches: usize,
    pending: Vec<Pending<Gathered>>,
}

impl PackFeeds {
    pub(super) fn new(
        source: &'static StreamedCheckpoint,
        ple: PleFeed,
        hidden: usize,
        branches: usize,
    ) -> Self {
        Self {
            source,
            ple,
            hidden,
            branches,
            pending: Vec::new(),
        }
    }
}

impl TokenFeeds for PackFeeds {
    fn begin(&mut self, token: u32) -> Result<StartVectors, EngError> {
        self.begin_chunk(&[token])
            .map(|mut starts| starts.pop().expect("one start"))
    }

    /// Row `i` plans against the history it will have when its turn comes (the tokens before it
    /// in the chunk), and every row set stays pending for the ordered update pass. All reads
    /// are host-only, so a failure leaves the feed where it was.
    fn begin_chunk(&mut self, tokens: &[u32]) -> Result<Vec<StartVectors>, EngError> {
        let mut starts = Vec::with_capacity(tokens.len());
        let mut pending = Vec::with_capacity(tokens.len());
        for (i, &token) in tokens.iter().enumerate() {
            let stream = stream_row(self.source, token, self.hidden, self.branches)?;
            let plan = self.ple.plan_after(&tokens[..i], token)?;
            let rows = self.ple.gather(&plan)?;
            let mut ple_embedding = Vec::new();
            for &(_, local) in &plan.rows {
                let values = rows.get(u64::from(local)).ok_or_else(|| {
                    EngError::BadRequest(format!("n-gram row {local} was not gathered"))
                })?;
                ple_embedding.extend_from_slice(values);
            }
            starts.push(StartVectors {
                stream,
                ple_embedding,
            });
            pending.push(Pending { token, rows });
        }
        self.pending = pending;
        Ok(starts)
    }

    fn ple_update(&mut self, token: u32, stream: &[f32]) -> Result<Vec<f32>, EngError> {
        let rows = take_pending(&mut self.pending, token)?;
        self.ple.update(token, &rows, stream)
    }

    fn set_ple_pair(&mut self, pair: PleKeyValue) -> Result<(), EngError> {
        self.ple.set_fixed(pair)
    }

    fn reset(&mut self) {
        self.pending.clear();
        self.ple.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::{Pending, take_pending};
    use crate::EngError;

    #[test]
    fn pending_rows_serve_their_own_token_once() {
        let mut pending = vec![Pending {
            token: 7,
            rows: "rows",
        }];
        assert_eq!(take_pending(&mut pending, 7).unwrap(), "rows");
        assert!(matches!(
            take_pending(&mut pending, 7),
            Err(EngError::BadRequest(_))
        ));
    }

    #[test]
    fn rows_read_for_another_token_are_refused_and_kept() {
        let mut pending = vec![Pending {
            token: 7,
            rows: "rows",
        }];
        assert!(matches!(
            take_pending(&mut pending, 8),
            Err(EngError::BadRequest(text)) if text.contains("token 8")
        ));
        // The queue keeps its row: a mismatched take is a caller bug, not a reason to drop
        // the row the real token still needs.
        assert_eq!(pending.len(), 1);
        assert_eq!(take_pending(&mut pending, 7).unwrap(), "rows");
        let mut empty: Vec<Pending<()>> = Vec::new();
        assert!(take_pending(&mut empty, 1).is_err());
    }
}
