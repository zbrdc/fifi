//! Layer-major prompt prefill. Each visit advances its own causal state in token
//! order, then computes the routed expert outputs using the shared CUDA banks.

use super::{CpuForward, LayerVisit, MessageBank, MoeRoute, Stream, VisitState, Weights, add, bad};
use crate::{
    EngError,
    family::inkling2::{
        cuda_projection::gguf::streaming::PrefillRoute, gguf_weights::GgufTextWeights,
    },
};

struct PromptRow {
    stream: Stream,
    prelude: Option<Stream>,
    bank: Option<MessageBank>,
}

impl PromptRow {
    fn new(stream: Stream) -> Self {
        Self {
            stream,
            prelude: None,
            bank: None,
        }
    }

    fn begin_pass(&mut self, pass: usize) -> Result<(), EngError> {
        if pass == 0 {
            self.prelude = Some(self.stream.clone());
        } else {
            let prelude = self
                .prelude
                .as_ref()
                .ok_or_else(|| bad("Inkling prefill lost its core prelude"))?;
            self.bank = Some(MessageBank {
                first_block: prelude.blocks.len() + 1,
                stream: std::mem::replace(&mut self.stream, prelude.clone()),
            });
        }
        Ok(())
    }
}

struct MlpRow {
    route: MoeRoute,
    input: Vec<f32>,
}

impl CpuForward {
    pub(in crate::family::inkling2) fn supports_grouped_prefill(&self) -> bool {
        matches!(&self.weights, Weights::Gguf(weights) if weights.supports_grouped_prefill())
    }

    pub(in crate::family::inkling2) fn prefill_grouped(
        &mut self,
        tokens: &[u32],
    ) -> Result<(), EngError> {
        let Weights::Gguf(weights) = &self.weights else {
            return Err(bad("grouped prefill needs CUDA GGUF weights"));
        };
        if self.failed {
            return Err(bad(
                "a previous Inkling forward invalidated the sequence state",
            ));
        }
        if !(2..=crate::family::inkling2::MAX_PREFILL_ROWS).contains(&tokens.len())
            || tokens.len() > self.context.saturating_sub(self.position)
            || tokens
                .iter()
                .any(|&id| u64::from(id) >= self.spec.unpadded_vocab_size)
        {
            return Err(bad(
                "prefill rows or tokens are outside the Inkling text limits",
            ));
        }
        self.failed = true;
        let mut state = std::mem::take(&mut self.state);
        let mut rows = tokens
            .iter()
            .map(|&id| self.embed(id).map(PromptRow::new))
            .collect::<Result<Vec<_>, _>>()?;
        let core = &self.spec.core;
        let width = core.core_end - core.core_start;
        let end = core.core_start + width * core.num_passes;
        for &visit in &self.visits {
            let slot = visit.state_slot;
            if (core.core_start..end).contains(&slot)
                && (slot - core.core_start).is_multiple_of(width)
            {
                for row in &mut rows {
                    row.begin_pass((slot - core.core_start) / width)?;
                }
            } else if slot == end {
                for row in &mut rows {
                    row.bank = None;
                    row.prelude = None;
                }
            }
            self.prompt_visit(visit, &mut state[slot], &mut rows, weights)?;
        }
        self.state = state;
        self.position += tokens.len();
        self.failed = false;
        Ok(())
    }

    fn prompt_visit(
        &self,
        visit: LayerVisit,
        state: &mut VisitState,
        rows: &mut [PromptRow],
        weights: &GgufTextWeights,
    ) -> Result<(), EngError> {
        let physical = visit.weight_layer;
        let layer = format!("model.llm.layers.{physical}");
        let prefix = format!("{layer}.mlp");
        let mlp = rows
            .iter_mut()
            .map(|row| {
                let input =
                    self.mlp_input(visit, &layer, state, &mut row.stream, row.bank.as_ref())?;
                let route = self.moe_route(physical, &prefix, &input)?;
                Ok(MlpRow { route, input })
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        let outputs = self.prompt_experts(weights, physical, &prefix, &mlp)?;
        for (row, output) in rows.iter_mut().zip(outputs) {
            self.finish_layer(&layer, state, &mut row.stream, output)?;
        }
        Ok(())
    }

    fn prompt_experts(
        &self,
        weights: &GgufTextWeights,
        layer: usize,
        prefix: &str,
        mlp: &[MlpRow],
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let routes: Vec<_> = mlp
            .iter()
            .map(|row| PrefillRoute {
                ids: &row.route.selected,
                input: &row.input,
            })
            .collect();
        let names = [
            format!("{prefix}.experts.w13_weight"),
            format!("{prefix}.experts.w2_weight"),
        ];
        let batch = weights.prefill_experts(
            names.each_ref().map(String::as_str),
            layer as u32,
            &routes,
            |gate, up| self.math.expert_activation(gate, up),
        )?;
        let Some(batch) = batch else {
            return mlp
                .iter()
                .map(|row| self.moe_with_route(layer, prefix, &row.input, &row.route))
                .collect();
        };
        Ok(mlp
            .iter()
            .zip(batch)
            .map(|(input, results)| {
                let mut output = vec![0.; self.hidden()];
                for (&id, contribution) in input.route.selected.iter().zip(results) {
                    add(
                        &mut output,
                        &contribution,
                        input.route.weight(id, self.math.route_scale),
                    );
                }
                output
            })
            .collect())
    }
}

#[test]
#[ignore = "requires a leased CUDA device, the Inkling GGUF and serial reference logits"]
fn grouped_prefill_matches_serial_logits_and_restored_state() {
    use std::{env, fs, path::Path};

    let directory = env::var_os("FIFI_INKLING_MODEL_DIR").expect("set the Inkling GGUF directory");
    let reference =
        env::var_os("FIFI_INKLING_REFERENCE_LOGITS").expect("set the serial logits path");
    let tokens: Vec<u32> = env::var("FIFI_INKLING_TOKEN_IDS")
        .expect("set comma-separated token ids")
        .split(',')
        .map(|value| value.parse().unwrap())
        .collect();
    let expected = fs::read(reference).unwrap();
    let mut model = CpuForward::open(Path::new(&directory), tokens.len()).unwrap();
    model.enable_cuda(0).unwrap();
    assert!(model.supports_grouped_prefill());
    let vocab = model.vocab_size() as usize;
    let row_bytes = vocab * size_of::<f32>();
    assert_eq!(expected.len(), tokens.len() * row_bytes);

    for width in [2, 8, 16] {
        model.reset();
        // The same held prefix is used for two independent continuations.
        for &token in &tokens[..3] {
            model.token(token, false).unwrap();
        }
        let prefix = model.snapshot();
        for _ in 0..2 {
            model.restore(&prefix);
            for chunk in tokens[3..].chunks(width + 1) {
                let (&last, head) = chunk.split_last().unwrap();
                if head.len() > 1 {
                    model.prefill_grouped(head).unwrap();
                } else if let Some(&token) = head.first() {
                    model.token(token, false).unwrap();
                }
                let position = model.position();
                let logits = model.token(last, true).unwrap().unwrap();
                let reference = &expected[position * row_bytes..][..row_bytes];
                assert_eq!(logits.len(), vocab);
                assert!(
                    reference
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(logits)
                        .all(|(&bytes, logit)| bytes == logit.to_le_bytes()),
                    "logit mismatch at position {position}, chunk {width}"
                );
            }
        }
    }
}
