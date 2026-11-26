//! Admission is read-only with respect to the sequence. After commitment there
//! is no fallback even for an enqueue error: a kernel may have appended K/V.
use super::*;
use crate::family::inkling2::switches::Switch;

impl CpuForward {
    pub(in crate::family::inkling2::cpu_forward) fn admit_resident_token(
        &mut self,
    ) -> Result<bool, EngError> {
        if !Switch::ResidentToken.on() || self.draft_window_enabled() || !self.taps.is_empty() {
            return Ok(false);
        }
        let Weights::Gguf(w) = &self.weights else {
            return Ok(false);
        };
        let Some(cuda) = w.cuda() else {
            return Ok(false);
        };
        if let Some(ws) = &self.resident_token {
            return Ok(ws.ready(cuda, &self.state, self.position));
        }
        match TokenWorkspace::open(self, cuda, &self.state) {
            Ok(Some(ws)) => {
                self.resident_token = Some(ws);
                Ok(true)
            }
            Ok(None) | Err(EngError::MemoryPlan(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }
    pub(in crate::family::inkling2::cpu_forward) fn commit_resident_token(
        &mut self,
        id: u32,
    ) -> Result<Vec<f32>, EngError> {
        self.window = None;
        self.failed = true;
        let mut states = std::mem::take(&mut self.state);
        let mut ws = self.resident_token.take().expect("admitted resident token");
        let outcome: Result<Vec<f32>, EngError> = (|| {
            let _span = self.phase_span(Work::ResidentVisit);
            let Weights::Gguf(w) = &self.weights else {
                unreachable!("admitted GGUF token")
            };
            let cuda = w.cuda().expect("admitted CUDA token");
            let profile = self
                .weights
                .cuda_projection()
                .and_then(|p| p.begin_token().map(|start| (p, start)));
            let logits = ws.run(self, cuda, &states, id)?;
            ws.commit(self, cuda, &mut states);
            if let Some((p, start)) = profile {
                p.end_token(start, self.position, true);
            }
            Ok(logits)
        })();
        self.resident_token = Some(ws);
        let logits = outcome.inspect_err(|_| self.invalidate_device_attention())?;
        self.state = states;
        self.position += 1;
        self.failed = false;
        static USED: Once = Once::new();
        USED.call_once(|| {
            eprintln!(
                "Inkling CUDA used whole resident decode token: {} visits, one readback fence",
                self.visits.len()
            )
        });
        Ok(logits)
    }
}

impl CpuForward {
    /// Brings rows that only the device holds after a chain token back to the host state.
    /// Every host-orchestrated consumer calls it first; it is a no-op while no visit is resident.
    pub(in crate::family::inkling2::cpu_forward) fn materialize_resident(
        &mut self,
    ) -> Result<(), EngError> {
        if !self.state.iter().any(VisitState::is_resident) {
            return Ok(());
        }
        self.pass_tails_stale();
        let Weights::Gguf(w) = &self.weights else {
            return Err(bad("resident rows without GGUF weights"));
        };
        let Some(cuda) = w.cuda() else {
            return Err(bad("resident rows without the CUDA projection"));
        };
        let past = self.math.sconv_kernel_size - 1;
        let capacities: Vec<usize> = (0..self.state.len())
            .map(|slot| {
                self.visits
                    .iter()
                    .find(|v| v.state_slot == slot)
                    .map_or(0, |v| self.kv_capacity(v.attention))
            })
            .collect();
        let mut states = std::mem::take(&mut self.state);
        let position = self.position;
        let outcome = self.resident_token.as_mut().map_or(
            Err(bad("resident rows without the token arenas")),
            |ws| {
                ws.materialize(
                    cuda,
                    Cursor {
                        states: &mut states,
                        position,
                        past,
                        capacities: &capacities,
                    },
                )
            },
        );
        self.state = states;
        outcome
    }
}
