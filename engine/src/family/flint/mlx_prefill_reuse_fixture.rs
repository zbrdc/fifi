//! Single-layer experiment. Every arm gets the same independently routed pieces.

use super::{
    prefill_pair::{PairStaging, StageLoader, StagePeak, wave_experts},
    *,
};
use crate::family::{
    mlx_budget::{DeviceBudget, PRESSURE_MARGIN},
    prefill::mlx::{local_expert_ids, sorted_expert_ids},
    stored::qualification_expert_reads,
};
use crate::kernels::HostPages;
use std::{rc::Rc, time::Instant};

const HIDDEN: u32 = 2048;
const TOP_K: u32 = 6;
const WAVE_SLOTS: usize = PairStaging::Waves64.slots();

struct Piece {
    input: ExecutionTensor,
    residual: ExecutionTensor,
    route: mlx_route::Route,
    experts: Vec<u32>,
}

impl Piece {
    fn prepare(
        k: &mut MlxKernels,
        layer: &Layer,
        index: usize,
        plane: &mut Option<Plane>,
        position: u32,
        rows: u32,
        capacity: u32,
    ) -> Result<Self> {
        // Fix the input across schedules without replaying all preceding layers.
        let mut seed = 17 + position;
        let bytes = (0..rows * HIDDEN)
            .flat_map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let value = (f32::from((seed >> 24) as u8) - 128.0) / 256.0;
                ((value.to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect();
        let input = k.import_tensor(&Tensor::try_from_host(
            Dtype::Bf16,
            vec![rows, HIDDEN],
            bytes,
        )?)?;
        let normalized = k.rms_norm(&input, &layer.input_norm, 1e-6)?;
        let plan = ChunkCachePlan::new(index, position, rows, capacity)?;
        let attended = attention(k, layer, &normalized, plane, &plan)?;
        let residual = k.binary(BinaryOp::Add, &input, &attended)?;
        let input = k.rms_norm(&residual, &layer.post_norm, 1e-6)?;
        let Ffn::Moe { router, .. } = &layer.ffn else {
            panic!("fixture must use a routed layer");
        };
        let logits = k.mm(&input, router)?;
        let route = mlx_route::select(k, &logits)?;
        let host = k.readback(route.ids.ids())?;
        let experts = host
            .host_bytes()
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&bytes| u32::from_le_bytes(bytes))
            .collect::<Vec<_>>();
        assert_eq!(experts.len(), (rows * TOP_K) as usize);
        mlx_route::read_step(k, None, &[route.health.share()])?;
        Ok(Self {
            input,
            residual,
            route,
            experts,
        })
    }
}

fn id_tensor(ids: &[u32]) -> Result<Tensor> {
    Tensor::try_from_host(
        Dtype::U32,
        vec![ids.len() as u32],
        ids.iter().flat_map(|id| id.to_le_bytes()).collect(),
    )
}

#[derive(Default)]
struct Staging {
    peak_mapped_bytes: usize,
    peak_experts: usize,
}

impl Staging {
    fn load(
        &mut self,
        k: &mut MlxKernels,
        source: &super::super::weights::FlintCheckpoint,
        stem: &str,
        experts: &[u32],
        selected: &[u32],
    ) -> Result<(SwiGlu, ExecutionTensor, HostPages)> {
        let ids = id_tensor(&local_expert_ids(selected, experts)?)?;
        let stage = SwiGlu::load_expert_pages(k, source, stem, experts, &ids, true)?;
        self.peak_mapped_bytes = self.peak_mapped_bytes.max(stage.2.len());
        self.peak_experts = self.peak_experts.max(experts.len());
        Ok(stage)
    }
}

fn retire_stage(
    k: &mut MlxKernels,
    anchor: &ExecutionTensor,
    weights: SwiGlu,
    local: ExecutionTensor,
    mut pages: HostPages,
) -> Result<()> {
    drop(weights);
    drop(local);
    pages.release();
    k.evaluate(&[anchor])?;
    assert!(
        !pages.in_use(),
        "the previous stage must retire before the next read"
    );
    Ok(())
}

fn staged_pieces(
    k: &mut MlxKernels,
    source: &super::super::weights::FlintCheckpoint,
    stem: &str,
    pieces: &[Piece],
    staging: &mut Staging,
) -> Result<Vec<ExecutionTensor>> {
    let selected = pieces
        .iter()
        .flat_map(|piece| piece.experts.iter().copied())
        .collect::<Vec<_>>();
    let experts = sorted_expert_ids(&selected);
    let (weights, local, pages) = staging.load(k, source, stem, &experts, &selected)?;
    let mut start = 0;
    let mut outputs = Vec::new();
    for piece in pieces {
        let end = start + piece.experts.len() as u32;
        let ids = k.slice(&local, &[start], &[end])?;
        let ids = k.bound_indices(&ids, experts.len() as u32)?;
        let output = grouped_experts(k, &weights, &piece.input, &ids)?;
        k.evaluate(&[&output])?;
        outputs.push(output);
        start = end;
    }
    retire_stage(k, &pieces[0].input, weights, local, pages)?;
    Ok(outputs)
}

#[test]
#[ignore = "requires the published Flint weights and an admitted Mac GPU window"]
fn two_piece_prefill_reuse_preserves_layer_output() -> Result<()> {
    let _serial = crate::native_test_lock();
    assert_eq!(
        std::env::var("FLINT_QUAL_GO").as_deref(),
        Ok("parent-owned-gpu-window")
    );
    assert!(super::super::mlx_expert_timing::enabled());
    assert!(!prefill_chain_flagged());
    let budget = DeviceBudget::detect()?.expect("set an admitted Metal working-set ceiling");
    let smoke = std::env::var_os("FIFI_FLINT_PREFILL_REUSE_SMOKE").is_some_and(|v| v == "1");
    let rows = if smoke { [32, 28] } else { [1023, 1017] };
    let total: u32 = rows.iter().sum();
    let capacity = total.next_power_of_two();
    let root = crate::tokenizer::flint_tests::fixture_dir();
    let checkpoint = Rc::new(super::super::weights::FlintCheckpoint::open(&root)?);
    let record_bytes = checkpoint.bytes_per_routed_expert();
    let staging_bound = super::super::mlx_model::expert_staging_bytes(record_bytes, rows[0]);
    let attention_scratch = 3 * 32 * 4 * u64::from(rows[0]) * u64::from(total);
    let residual_selected_bytes = u64::from(total) * u64::from(HIDDEN) * 2;
    budget.remaining(
        "Flint two-piece layer fixture",
        &[
            ("expert_staging_bytes", staging_bound),
            ("attention_score_scratch", attention_scratch),
            (
                "hidden_and_routed_buffers",
                (2 + 3 * u64::from(TOP_K)) * residual_selected_bytes,
            ),
            ("pressure_margin_bytes", PRESSURE_MARGIN),
        ],
    )?;
    let mut k = MlxKernels::try_new()?;
    k.allow_bf16(true);
    k.set_cache_limit(0)?;

    for index in [35, 37] {
        let layer = Layer::load(&mut k, &checkpoint, index, true, None)?;
        let Ffn::Moe {
            routed: Routed::Disk { stem, .. },
            shared,
            ..
        } = &layer.ffn
        else {
            panic!("expected a disk-backed MoE layer");
        };
        k.reset_peak_memory()?;
        let mut plane = None;
        let pieces = [
            Piece::prepare(&mut k, &layer, index, &mut plane, 0, rows[0], capacity)?,
            Piece::prepare(
                &mut k, &layer, index, &mut plane, rows[0], rows[1], capacity,
            )?,
        ];
        let preparation_peak = k.memory_stats()?.peak;
        let union = sorted_expert_ids(
            &pieces
                .iter()
                .flat_map(|piece| piece.experts.iter().copied())
                .collect::<Vec<_>>(),
        );
        let separate_records = pieces
            .iter()
            .map(|piece| sorted_expert_ids(&piece.experts).len())
            .sum::<usize>();
        assert!(union.len() < separate_records && union.len() > WAVE_SLOTS);
        let mut reference = None;
        for (mode, records, stages) in [
            ("separate", separate_records, 2),
            ("union", union.len(), 1),
            ("waves64", union.len(), union.len().div_ceil(WAVE_SLOTS)),
        ] {
            k.evaluate(&[&pieces[0].input, &pieces[1].input])?;
            k.reset_peak_memory()?;
            let active = k.memory_stats()?.active;
            let before = qualification_expert_reads();
            let copies = k.qualification_page_copies();
            let mut staging = Staging::default();
            let start = Instant::now();
            let routed = match mode {
                "separate" => {
                    let mut outputs = Vec::new();
                    for piece in &pieces {
                        outputs.extend(staged_pieces(
                            &mut k,
                            &checkpoint,
                            stem,
                            std::slice::from_ref(piece),
                            &mut staging,
                        )?);
                    }
                    outputs
                }
                "union" => staged_pieces(&mut k, &checkpoint, stem, &pieces, &mut staging)?,
                _ => {
                    let mut peak = StagePeak::default();
                    let mut loader = StageLoader {
                        source: &checkpoint,
                        stem,
                        coalesce: true,
                        peak: &mut peak,
                    };
                    let routed = wave_experts(
                        &mut k,
                        &mut loader,
                        [&pieces[0].input, &pieces[1].input],
                        [&pieces[0].experts, &pieces[1].experts],
                        &union,
                    )?;
                    staging.peak_experts = peak.experts;
                    staging.peak_mapped_bytes = peak.mapped_bytes;
                    Vec::from(routed)
                }
            };
            assert_eq!(routed.len(), pieces.len());
            let mut output = Vec::new();
            for (routed, piece) in routed.into_iter().zip(&pieces) {
                let shared = shared.apply(&mut k, &piece.input)?;
                let combined =
                    precision_native::combine(&mut k, &routed, &piece.route.weights, &shared)?;
                let residual = k.binary(BinaryOp::Add, &piece.residual, &combined)?;
                let host = k.readback(&residual)?;
                assert_eq!(host.dtype(), Dtype::Bf16);
                assert_eq!(host.shape(), piece.input.shape());
                let bytes = host.host_bytes().unwrap();
                assert!(
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .all(|&word| { u16::from_le_bytes(word) & 0x7f80 != 0x7f80 }),
                    "fixture output must be finite"
                );
                assert!(bytes.iter().any(|&byte| byte != 0));
                output.extend_from_slice(bytes);
            }
            let elapsed_ns = start.elapsed().as_nanos() as u64;
            let peak = k.memory_stats()?.peak;
            assert!(
                peak.max(preparation_peak) + (staging.peak_mapped_bytes as u64) < budget.bytes()
            );
            let after = qualification_expert_reads();
            let read_bytes = after.read_bytes - before.read_bytes;
            assert_eq!(read_bytes, records as u64 * record_bytes);
            assert_eq!(after.page_imports - before.page_imports, stages as u64);
            assert_eq!(k.qualification_page_copies() - copies, 0);
            if mode == "waves64" {
                assert!(staging.peak_experts <= WAVE_SLOTS);
            }
            let expected = reference.get_or_insert_with(|| output.clone());
            let matches = output == *expected;
            eprintln!(
                "flint_prefill_reuse_fixture {}",
                serde_json::json!({
                    "layer": index, "first_rows": rows[0], "second_rows": rows[1],
                    "mode": mode, "output_matches": matches, "record_reads": records,
                    "read_bytes": read_bytes, "slice_reads": after.slice_reads - before.slice_reads,
                    "read_ns": after.read_ns - before.read_ns, "stages": stages,
                    "elapsed_ns": elapsed_ns, "device_budget_bytes": budget.bytes(),
                    "peak_experts": staging.peak_experts, "peak_mapped_bytes": staging.peak_mapped_bytes,
                    "attention_route_peak_bytes": preparation_peak, "expert_start_active_bytes": active,
                    "expert_peak_bytes": peak,
                })
            );
            assert!(
                matches,
                "layer {index} {mode} changed the BF16 layer output"
            );
        }
    }
    Ok(())
}
