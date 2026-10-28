//! Test-only route recorder for the S3 expert-cache trace (S3 plan section 3.2, packet T0).
//!
//! `step` pushes one row per MoE layer after the router readback. A row holds the host selection
//! and the host probabilities that `moe::select` already used, moved out of the step. The
//! recorder makes no native call, no synchronization, and no readback. Release builds contain no
//! recorder. The portable tests below check the rows, the T0-P1 row checks, and the TSV line.

use super::super::moe::{MoeShape, select};
use crate::EngError;

/// Schema of `trace.tsv` (S3 plan section 3.4).
pub(super) const TRACE_SCHEMA: &str = "s3-route-trace/1";

/// Gate sums of a renormalized selection must be 1 within this bound (T0-P1).
const GATE_SUM_TOLERANCE: f32 = 1e-5;

/// The kind of a traced step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TraceMode {
    /// A prompt token whose logits are unused (`StepMode::Prompt`), written as `p`.
    Prompt,
    /// A token whose greedy output is read back (`StepMode::Generate`), written as `g`.
    Generate,
}

impl TraceMode {
    fn letter(self) -> char {
        match self {
            Self::Prompt => 'p',
            Self::Generate => 'g',
        }
    }
}

/// The step that the next rows belong to. The test sets it before each `step`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TraceStep {
    /// The prompt id from the prompts file.
    pub(super) prompt: u32,
    /// The position of the input token.
    pub(super) pos: u32,
    pub(super) mode: TraceMode,
}

/// One MoE layer of one step.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct RouteRow {
    pub(super) step: TraceStep,
    pub(super) layer: usize,
    /// The selected experts in rank order with their gates, as `moe::select` returned them.
    pub(super) routes: Vec<(u32, f32)>,
    /// The router probabilities `[E]` that the selection came from.
    pub(super) probs: Vec<f32>,
}

/// Rows of the current step. `take` returns them and clears the step, so every step needs its
/// own `begin`.
#[derive(Debug, Default)]
pub(super) struct RouteTrace {
    step: Option<TraceStep>,
    rows: Vec<RouteRow>,
}

impl RouteTrace {
    /// Sets the step of the next rows.
    pub(super) fn begin(&mut self, step: TraceStep) {
        self.step = Some(step);
    }

    /// Appends the selection of `layer`. Without `begin` it returns `BadRequest`.
    pub(super) fn push(
        &mut self,
        layer: usize,
        routes: Vec<(u32, f32)>,
        probs: Vec<f32>,
    ) -> Result<(), EngError> {
        let step = self
            .step
            .ok_or_else(|| EngError::BadRequest("route trace row without a step".into()))?;
        self.rows.push(RouteRow {
            step,
            layer,
            routes,
            probs,
        });
        Ok(())
    }

    /// Returns the rows pushed since the last `take` and clears the step.
    pub(super) fn take(&mut self) -> Vec<RouteRow> {
        self.step = None;
        std::mem::take(&mut self.rows)
    }
}

/// The `trace.tsv` column line for `top_k` experts per row.
pub(super) fn tsv_columns(top_k: usize) -> String {
    let mut columns = vec![
        "prompt".to_string(),
        "mode".to_string(),
        "pos".to_string(),
        "layer".to_string(),
    ];
    for prefix in ["e", "p", "g"] {
        columns.extend((0..top_k).map(|rank| format!("{prefix}{rank}")));
    }
    columns.join("\t")
}

impl RouteRow {
    /// The `trace.tsv` line: prompt, mode, position, layer, the ids in rank order, the raw
    /// probabilities of the ids, and the gates, each value with 9 significant digits.
    pub(super) fn tsv(&self) -> Result<String, String> {
        let mut fields = vec![
            self.step.prompt.to_string(),
            self.step.mode.letter().to_string(),
            self.step.pos.to_string(),
            self.layer.to_string(),
        ];
        fields.extend(self.routes.iter().map(|&(id, _)| id.to_string()));
        for &(id, _) in &self.routes {
            let p = usize::try_from(id)
                .ok()
                .and_then(|index| self.probs.get(index))
                .ok_or_else(|| format!("layer {}: expert {id} has no probability", self.layer))?;
            fields.push(format!("{p:.8e}"));
        }
        fields.extend(self.routes.iter().map(|&(_, gate)| format!("{gate:.8e}")));
        Ok(fields.join("\t"))
    }
}

/// The T0-P1 checks of one step's rows: one row per layer in layer order, all of `step`; each row
/// has `top_k` distinct ids below the expert count, finite probabilities, the exact selection of
/// `moe::select` on its own probabilities, and gates that sum to 1 when the selection renormalizes.
pub(super) fn check_step(
    rows: &[RouteRow],
    shape: &MoeShape,
    layers: usize,
    step: TraceStep,
) -> Result<(), String> {
    if rows.len() != layers {
        return Err(format!(
            "step {step:?}: {} rows for {layers} layers",
            rows.len()
        ));
    }
    for (index, row) in rows.iter().enumerate() {
        let what = format!("step {step:?} row {index}");
        if row.layer != index || row.step != step {
            return Err(format!(
                "{what}: labelled layer {} of {:?}",
                row.layer, row.step
            ));
        }
        if row.routes.len() != shape.top_k() || row.probs.len() != shape.experts() {
            return Err(format!(
                "{what}: {} routes and {} probabilities",
                row.routes.len(),
                row.probs.len()
            ));
        }
        let mut ids: Vec<u32> = row.routes.iter().map(|&(id, _)| id).collect();
        ids.sort_unstable();
        ids.dedup();
        let in_range = ids
            .last()
            .is_some_and(|&id| usize::try_from(id).is_ok_and(|id| id < shape.experts()));
        if ids.len() != shape.top_k() || !in_range {
            return Err(format!("{what}: ids are not distinct expert ids"));
        }
        if row.probs.iter().any(|p| !p.is_finite()) {
            return Err(format!("{what}: non-finite probability"));
        }
        let want = select(shape, &row.probs).map_err(|error| format!("{what}: {error:?}"))?;
        if want != row.routes {
            return Err(format!(
                "{what}: routes {:?} differ from the selection {want:?}",
                row.routes
            ));
        }
        let sum: f32 = row.routes.iter().map(|&(_, gate)| gate).sum();
        if shape.renorm() && (sum - 1.0).abs() > GATE_SUM_TOLERANCE {
            return Err(format!("{what}: gates sum to {sum}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four experts, top 2, renormalized gates.
    fn shape() -> MoeShape {
        MoeShape::new(4, 2, 32, 32, 64, true).unwrap()
    }

    const STEP: TraceStep = TraceStep {
        prompt: 3,
        pos: 17,
        mode: TraceMode::Generate,
    };

    /// Layer 0 selects experts 2 and 0: probabilities 0.5 and 0.25 renormalize to 2/3 and 1/3.
    fn layer0() -> (Vec<(u32, f32)>, Vec<f32>) {
        let probs = vec![0.25, 0.125, 0.5, 0.125];
        (vec![(2, 0.5 / 0.75), (0, 0.25 / 0.75)], probs)
    }

    /// Layer 1 selects experts 1 and 3 with gates 0.75 and 0.25.
    fn layer1() -> (Vec<(u32, f32)>, Vec<f32>) {
        let probs = vec![0.125, 0.375, 0.375, 0.125];
        // Experts 1 and 2 tie at 0.375; the lower index ranks first, so the second pick is 2.
        (vec![(1, 0.5), (2, 0.5)], probs)
    }

    /// One valid step of two layers, pushed through the recorder.
    fn traced_step() -> Vec<RouteRow> {
        let mut trace = RouteTrace::default();
        trace.begin(STEP);
        for (layer, (routes, probs)) in [layer0(), layer1()].into_iter().enumerate() {
            trace.push(layer, routes, probs).unwrap();
        }
        trace.take()
    }

    #[test]
    fn rows_keep_the_step_layer_and_selection_of_each_push() {
        let rows = traced_step();
        let (routes0, probs0) = layer0();
        let (routes1, probs1) = layer1();
        assert_eq!(
            rows,
            vec![
                RouteRow {
                    step: STEP,
                    layer: 0,
                    routes: routes0,
                    probs: probs0,
                },
                RouteRow {
                    step: STEP,
                    layer: 1,
                    routes: routes1,
                    probs: probs1,
                },
            ],
            "the rows in push order"
        );
    }

    #[test]
    fn a_push_after_take_needs_a_new_step() {
        let mut trace = RouteTrace::default();
        trace.begin(STEP);
        let (routes, probs) = layer0();
        trace.push(0, routes.clone(), probs.clone()).unwrap();
        assert_eq!(trace.take().len(), 1, "one row before take");
        let late = trace.push(0, routes, probs);
        assert!(
            matches!(&late, Err(EngError::BadRequest(m)) if m.contains("without a step")),
            "{late:?}"
        );
        assert!(trace.take().is_empty(), "the rejected row is not kept");
    }

    #[test]
    fn check_step_accepts_the_selection_of_every_layer() {
        // The hand-derived selections of `layer0` and `layer1` equal `select` on their probabilities.
        assert_eq!(check_step(&traced_step(), &shape(), 2, STEP), Ok(()));
    }

    #[test]
    fn check_step_rejects_a_missing_layer() {
        let mut rows = traced_step();
        rows.pop();
        let checked = check_step(&rows, &shape(), 2, STEP);
        assert!(
            checked
                .as_ref()
                .is_err_and(|m| m.contains("1 rows for 2 layers")),
            "{checked:?}"
        );
    }

    #[test]
    fn check_step_rejects_routes_of_another_layer() {
        // Layer 1 labelled with the selection of layer 0, as a recorder that pushes the previous
        // layer's selection would write it.
        let mut rows = traced_step();
        rows[1].routes = rows[0].routes.clone();
        let checked = check_step(&rows, &shape(), 2, STEP);
        assert!(
            checked
                .as_ref()
                .is_err_and(|m| m.contains("row 1") && m.contains("differ from the selection")),
            "{checked:?}"
        );
    }

    #[test]
    fn check_step_rejects_a_row_of_another_step() {
        let mut rows = traced_step();
        rows[1].step.pos += 1;
        let checked = check_step(&rows, &shape(), 2, STEP);
        assert!(
            checked
                .as_ref()
                .is_err_and(|m| m.contains("row 1") && m.contains("labelled")),
            "{checked:?}"
        );
    }

    /// The error text of `check_step` for `rows`, or a failure when the rows pass.
    fn rejection(rows: &[RouteRow], shape: &MoeShape) -> String {
        match check_step(rows, shape, 2, STEP) {
            Err(message) => message,
            Ok(()) => panic!("check_step accepted rows that it must reject"),
        }
    }

    #[test]
    fn check_step_rejects_a_duplicate_expert_id() {
        // Layer 1 lists expert 1 twice. Four experts and top 2 need two distinct ids.
        let mut rows = traced_step();
        rows[1].routes = vec![(1, 0.5), (1, 0.5)];
        let message = rejection(&rows, &shape());
        assert!(
            message.contains("row 1") && message.contains("distinct expert ids"),
            "{message}"
        );
    }

    #[test]
    fn check_step_rejects_an_id_equal_to_the_expert_count() {
        // The shape has experts 0 to 3, so id 4 is the first id out of range.
        let mut rows = traced_step();
        rows[0].routes = vec![(4, 0.5), (0, 0.5)];
        let message = rejection(&rows, &shape());
        assert!(
            message.contains("row 0") && message.contains("distinct expert ids"),
            "{message}"
        );
    }

    #[test]
    fn check_step_accepts_the_largest_expert_id_below_the_count() {
        // Id 3 is the last of four experts. The probabilities 0.125, 0.125, 0.25, 0.5 select
        // experts 3 and 2 with gates 0.5 / 0.75 and 0.25 / 0.75.
        let mut rows = traced_step();
        rows[0].probs = vec![0.125, 0.125, 0.25, 0.5];
        rows[0].routes = vec![(3, 0.5 / 0.75), (2, 0.25 / 0.75)];
        assert_eq!(check_step(&rows, &shape(), 2, STEP), Ok(()));
    }

    #[test]
    fn check_step_rejects_a_non_finite_probability() {
        let mut rows = traced_step();
        rows[1].probs[3] = f32::NAN;
        let message = rejection(&rows, &shape());
        assert!(
            message.contains("row 1") && message.contains("non-finite probability"),
            "{message}"
        );
    }

    #[test]
    fn check_step_rejects_gates_that_do_not_sum_to_one() {
        // Two probabilities of 3.0e38 are finite, but their sum 6.0e38 exceeds `f32::MAX`
        // (3.4028235e38) and rounds to infinity. Every gate is then 3.0e38 / inf = 0, `select`
        // returns exactly these routes, and the gates sum to 0 instead of 1.
        let mut rows = traced_step();
        rows[0].probs = vec![3.0e38, 3.0e38, 0.0, 0.0];
        rows[0].routes = vec![(0, 0.0), (1, 0.0)];
        assert_eq!(
            select(&shape(), &rows[0].probs).unwrap(),
            rows[0].routes,
            "the routes are the selection of the probabilities"
        );
        let message = rejection(&rows, &shape());
        assert!(
            message.contains("row 0") && message.contains("gates sum to 0"),
            "{message}"
        );
    }

    #[test]
    fn check_step_leaves_the_gate_sum_open_without_renormalization() {
        // Without renormalization the gates divide by the sum of all probabilities: layer 0 has
        // probabilities 0.25, 0.125, 0.5, 0.125 (sum 1) and top 2 gates 0.5 and 0.25, which sum to
        // 0.75. Layer 1 has 0.125, 0.375, 0.375, 0.125 and gates 0.375 and 0.375 (sum 0.75).
        let open = MoeShape::new(4, 2, 32, 32, 64, false).unwrap();
        let mut rows = traced_step();
        rows[0].routes = vec![(2, 0.5), (0, 0.25)];
        rows[1].routes = vec![(1, 0.375), (2, 0.375)];
        assert_eq!(check_step(&rows, &open, 2, STEP), Ok(()));
    }

    #[test]
    fn tsv_line_writes_ids_raw_probabilities_and_gates() {
        let rows = traced_step();
        assert_eq!(
            tsv_columns(2),
            "prompt\tmode\tpos\tlayer\te0\te1\tp0\tp1\tg0\tg1",
            "column line"
        );
        // Layer 0: ids 2 and 0, raw probabilities 0.5 and 0.25, gates 2/3 and 1/3. The F32 gates
        // are 0.666666686534881591796875 and 0.3333333432674407958984375.
        assert_eq!(
            rows[0].tsv(),
            Ok(
                "3\tg\t17\t0\t2\t0\t5.00000000e-1\t2.50000000e-1\t6.66666687e-1\t3.33333343e-1"
                    .into()
            )
        );
        let prompt = RouteRow {
            step: TraceStep {
                mode: TraceMode::Prompt,
                ..STEP
            },
            ..rows[1].clone()
        };
        assert!(
            prompt
                .tsv()
                .is_ok_and(|line| line.starts_with("3\tp\t17\t1\t1\t2\t")),
            "prompt mode letter"
        );
    }
}

/// The recorder inside `ResidentDecoder::step` on the synthetic checkpoint (Mac only).
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod native {
    use super::super::super::load::FlashWeights;
    use super::super::fixtures::{Entry, config, tensors};
    use super::super::moe::tests::checkpoint_from;
    use super::super::{ResidentDecoder, StepMode};
    use super::*;
    use crate::kernels::MlxKernels;

    const LAYER0_ROUTER: &str = "language_model.model.layers.0.mlp.gate.weight";
    const LAYER1_ROUTER: &str = "language_model.model.layers.1.mlp.gate.weight";

    /// The fixture with two known routers `[4, 64]`. Layer 0: rows 0 and 1 are zero, row 2 is all
    /// ones, and row 3 is all minus ones, so the logits are `[0, 0, s, -s]` for an input sum `s`
    /// and the selection is experts 2 and 0 (`s > 0`) or 3 and 0 (`s < 0`). Layer 1: all zero, so
    /// the probabilities are uniform and the selection is experts 0 and 1 with gates 0.5.
    fn trace_entries() -> Vec<Entry> {
        let mut out = tensors();
        for (name, _, _, bytes) in &mut out {
            let rows: [f32; 4] = match name.as_str() {
                LAYER0_ROUTER => [0.0, 0.0, 1.0, -1.0],
                LAYER1_ROUTER => [0.0; 4],
                _ => continue,
            };
            *bytes = rows
                .iter()
                .flat_map(|&v| [v; 64])
                .flat_map(f32::to_le_bytes)
                .collect();
        }
        out
    }

    #[test]
    fn the_route_trace_holds_every_layer_of_every_step() {
        let _serial = crate::native_test_lock();
        let ckpt = checkpoint_from(trace_entries());
        let flash = FlashWeights::load(&ckpt, &config()).unwrap();
        let shape = flash.shape().moe;
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, &flash).unwrap();
        decoder.route_trace = Some(RouteTrace::default());
        let mut state = decoder.zero_state().unwrap();
        let runs = [
            (1u32, StepMode::Prompt, TraceMode::Prompt),
            (2, StepMode::Generate, TraceMode::Generate),
            (3, StepMode::Generate, TraceMode::Generate),
        ];
        let mut total = 0;
        for (token, mode, trace_mode) in runs {
            let step = TraceStep {
                prompt: 5,
                pos: state.pos(),
                mode: trace_mode,
            };
            decoder.route_trace.as_mut().unwrap().begin(step);
            let ran = decoder.step(&mut state, token, mode);
            assert!(ran.is_ok(), "token {token}: {:?}", ran.err());
            let rows = decoder.route_trace.as_mut().unwrap().take();
            total += rows.len();
            assert_eq!(check_step(&rows, &shape, 2, step), Ok(()), "token {token}");
            let ids = |row: &RouteRow| row.routes.iter().map(|r| r.0).collect::<Vec<u32>>();
            assert_eq!(ids(&rows[1]), [0, 1], "token {token}: layer 1 ids");
            for &(_, gate) in &rows[1].routes {
                assert!(
                    (gate - 0.5).abs() <= 1e-6,
                    "token {token}: layer 1 gate {gate}"
                );
            }
            for &p in &rows[1].probs {
                assert!(
                    (p - 0.25).abs() <= 1e-6,
                    "token {token}: layer 1 probability {p}"
                );
            }
            let first = ids(&rows[0]);
            assert!(
                first == [2, 0] || first == [3, 0],
                "token {token}: layer 0 ids {first:?}"
            );
        }
        assert_eq!(total, 6, "two layers of three steps");
    }
}
