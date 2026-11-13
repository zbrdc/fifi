//! The chunk projections of the EXL3 arms against the one-row path and against fp64.
//!
//! `FIFI_EXL3_CHECKPOINT` names the pack directory. The test loads one DeltaNet layer and one
//! attention layer, projects deterministic F32 rows at several chunk heights through
//! [`Projector`] and compares the rows with (a) the same rows through the one-row EXL3 path and
//! (b) the fp64 evaluation of the stored matrix, where the stored matrix is what the device
//! holds (the attention projections; the DeltaNet value heads are moved at load).
//!
//! A job table that reads every job from the first one's input rows, or an output that ignores its
//! column offset, fails the comparison with the one-row path (433 rows, two jobs and a tail).
//!
//! Row counts straddle every boundary of the tiling: one row, the split-K limit of eight, the
//! SIMT/tensor-core switch at 64, one full tile, one row over a tile and several tiles.

use std::path::Path;

use super::{Exl3Scratch, F32, Output, Projector, TILE_ROWS, write_dense_jobs};
use crate::{
    family::{
        AttentionKind,
        gguf_model::Role,
        project::cuda::{CudaProjector, CudaWeights, Exl3MatrixView, Reserve, WeightLayout},
        qwen36::cuda::{
            common_tests::{dims, inventory},
            exl3_load_device_tests::{matrix_of, relative_l2},
            exl3_source::{PackReader, attention_weights, delta_weights},
            layer_table::LayerTable,
            mixer_geometry::MixerGeometry,
            mixer_weights::MixerWeights,
            moe::Launch,
        },
        qwen36::exl3_pack::layer_name,
    },
    kernels::cuda::{DeviceBuffer, DeviceRange, device_present, resident::router::RouterShape},
    weights::exl3::Checkpoint,
};

/// Row counts of the chunk: the tiling boundaries.
const HEIGHTS: [usize; 12] = [1, 5, 9, 63, 64, 100, 144, 145, 255, 256, 257, 433];
/// Relative L2 against fp64 for the one-row path (F32 operands; measured 1.6e-7).
const SIMT_BOUND: f64 = 3e-6;
/// Relative L2 against fp64 for the job-table projection (fp16-rounded operands, F32 accumulation;
/// measured 2.1e-4).
const TENSOR_BOUND: f64 = 1e-3;

pub(super) fn input(rows: usize, k: usize) -> Vec<f32> {
    (0..rows * k)
        .map(|i| (((i * 7919 + i / k * 31) % 2003) as f32 - 1001.0) / 700.0)
        .collect()
}

pub(super) fn upload(projector: &CudaProjector, values: &[f32]) -> DeviceBuffer {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    projector.context().upload(&bytes).unwrap()
}

pub(super) fn filled(projector: &CudaProjector, byte: u8, bytes: usize) -> DeviceBuffer {
    projector.context().upload(&vec![byte; bytes]).unwrap()
}

pub(super) fn read(projector: &CudaProjector, buffer: &DeviceBuffer, floats: usize) -> Vec<f32> {
    let stream = projector.context().stream().unwrap();
    let mut bytes = vec![0; floats * F32];
    buffer.read_at(&stream, 0, &mut bytes).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

pub(super) fn whole(buffer: &DeviceBuffer) -> DeviceRange<'_> {
    buffer.range(0, buffer.bytes()).unwrap()
}

/// The scratch buffers of the widest stage of the model, owned by the test.
pub(super) struct Scratch {
    xh: DeviceBuffer,
    z: DeviceBuffer,
    y: DeviceBuffer,
    flag: DeviceBuffer,
    jobs: DeviceBuffer,
}

impl Scratch {
    /// Scratch for chunks of up to `rows` rows of the widest projection of the model.
    pub(super) fn new(
        projector: &CudaProjector,
        rows: usize,
        (widest_k, widest_n): (usize, usize),
    ) -> Self {
        Self {
            xh: filled(projector, 0, rows * widest_k * F32),
            z: filled(projector, 0, rows * widest_n * F32),
            y: filled(projector, 0, rows * 32 * F32),
            flag: filled(projector, 0, 4),
            jobs: filled(projector, 0, 16 * rows.div_ceil(64)),
        }
    }

    /// The sticky arithmetic flag word.
    pub(super) fn flag_word(&self, projector: &CudaProjector) -> u32 {
        read(projector, &self.flag, 1)[0].to_bits()
    }

    pub(super) fn view(&self) -> Exl3Scratch<'_> {
        Exl3Scratch {
            xh: whole(&self.xh),
            z: whole(&self.z),
            y: whole(&self.y),
            flag: whole(&self.flag),
            jobs: whole(&self.jobs),
        }
    }

    /// The projector of a chunk of `rows` rows, its job table written.
    pub(super) fn chunk<'a>(
        &'a self,
        launch: &'a crate::family::qwen36::cuda::moe::Launch<'a>,
        rows: usize,
    ) -> Projector<'a> {
        let projector = Projector {
            launch,
            scratch: self.view(),
            rows,
            tile_rows: TILE_ROWS,
        };
        write_dense_jobs(launch, &projector.scratch, rows).unwrap();
        projector
    }
}

/// The rows through the one-row EXL3 path, one launch per row.
fn one_row_path(
    projector: &CudaProjector,
    view: &Exl3MatrixView<'_>,
    scratch: &Scratch,
    x: &DeviceBuffer,
    rows: usize,
) -> Vec<f32> {
    let [k, n] = view.shape;
    let out = filled(projector, 0, rows * n * F32);
    let ctx = projector.context();
    let stream = ctx.stream().unwrap();
    for row in 0..rows {
        let io = crate::family::project::cuda::exl3::Exl3Io {
            x: x.range(row * k * F32, k * F32).unwrap(),
            xh: scratch.xh.range(0, k * F32).unwrap(),
            z: scratch.z.range(0, n * F32).unwrap(),
            out: out.range(row * n * F32, n * F32).unwrap(),
            flag: whole(&scratch.flag),
            rows: 1,
            fused: false,
        };
        ctx.exl3_linear(&stream, view.linear_call(&io)).unwrap();
    }
    stream.synchronize().unwrap();
    read(projector, &out, rows * n)
}

fn fp64(matrix: &crate::weights::exl3::Matrix, x: &[f32], rows: usize) -> Vec<f64> {
    let [k, n] = matrix.shape();
    let x: Vec<f64> = x[..rows * k].iter().map(|&v| f64::from(v)).collect();
    let mut out = vec![0.0; rows * n];
    matrix.project_fp64_into(&x, rows, &mut out).unwrap();
    out
}

fn first_layer(table: &LayerTable, kind: AttentionKind) -> usize {
    (0..table.len())
        .find(|&layer| table.slot(layer).unwrap().kind == kind)
        .unwrap()
}

/// What a test needs once the two layers are on the device.
struct Rig<'a> {
    projector: &'a CudaProjector,
    checkpoint: &'a Checkpoint,
    mixers: &'a MixerWeights<'a>,
    geometry: &'a MixerGeometry,
    launch: &'a Launch<'a>,
    scratch: &'a Scratch,
    stream: &'a crate::kernels::cuda::DeviceStream,
    delta: usize,
    attn: usize,
}

/// Loads one DeltaNet and one attention layer of the pack in `FIFI_EXL3_CHECKPOINT` and runs
/// `test` on them.
fn with_rig(test: impl FnOnce(&Rig<'_>)) {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    assert!(
        device_present("EXL3 chunk projections"),
        "a GPU is required"
    );
    let checkpoint = Checkpoint::open(Path::new(&dir)).expect("open the pack");
    let d = dims();
    let table = LayerTable::new(&d, &inventory()).unwrap();
    let geometry = MixerGeometry::new(&d).unwrap();
    let projector = CudaProjector::new().expect("projector");
    let (delta, attn) = (
        first_layer(&table, AttentionKind::DeltaNet),
        first_layer(&table, AttentionKind::Full),
    );
    let reader = PackReader::new(&projector, &checkpoint, Reserve::default());
    let mut loaded = delta_weights(&reader, &table, delta, &geometry).expect("delta layer");
    loaded.extend(attention_weights(&reader, &table, attn, &geometry).expect("attention layer"));
    let weights = CudaWeights::from_exl3(loaded).unwrap();
    let mixers = MixerWeights::new(
        &geometry,
        &table,
        &weights,
        &[delta, attn],
        WeightLayout::Exl3,
    )
    .unwrap();
    let widest = (geometry.delta.value_width, geometry.delta.ld);
    let scratch = Scratch::new(&projector, 433, widest);
    let ctx = projector.context();
    let stream = ctx.stream().unwrap();
    let launch = Launch {
        ctx,
        stream: &stream,
    };
    test(&Rig {
        projector: &projector,
        checkpoint: &checkpoint,
        mixers: &mixers,
        geometry: &geometry,
        launch: &launch,
        scratch: &scratch,
        stream: &stream,
        delta,
        attn,
    });
}

#[test]
#[ignore = "requires admitted GPU memory (2 GiB) and the pinned EXL3 pack in FIFI_EXL3_CHECKPOINT"]
fn the_chunk_projections_match_the_one_row_path_and_fp64() {
    with_rig(|rig| {
        let Rig {
            projector,
            checkpoint,
            mixers,
            launch,
            scratch,
            stream,
            attn,
            ..
        } = *rig;
        // The attention projections are the stored matrices: fp64 applies to them directly.
        for (role, suffix) in [
            (Role::AttnQ, "self_attn.q_proj"),
            (Role::AttnK, "self_attn.k_proj"),
            (Role::AttnV, "self_attn.v_proj"),
            (Role::AttnOutput, "self_attn.o_proj"),
        ] {
            let weight = mixers.layer_weights(attn).unwrap().exl3(role).unwrap();
            let view = weight.view().unwrap();
            let stored = matrix_of(checkpoint, &layer_name(attn, suffix));
            let [k, n] = view.shape;
            // The rows of every height are the first rows of the tallest: one fp64 evaluation.
            let tallest = HEIGHTS[HEIGHTS.len() - 1];
            let want_all = fp64(&stored, &input(tallest, k), tallest);
            for rows in HEIGHTS {
                let x = input(rows, k);
                let xb = upload(projector, &x);
                let out = filled(projector, 0xff, rows * n * F32);
                scratch
                    .chunk(launch, rows)
                    .project(weight, whole(&xb), Output::Rows(whole(&out)))
                    .unwrap();
                stream.synchronize().unwrap();
                let got = read(projector, &out, rows * n);
                let want = &want_all[..rows * n];
                let single = one_row_path(projector, &view, scratch, &xb, rows);
                let (chunk_error, single_error) =
                    (relative_l2(&got, want), relative_l2(&single, want));
                println!(
                    "{role:?} rows {rows}: chunk {chunk_error:.3e} one-row {single_error:.3e}"
                );
                assert!(
                    chunk_error < TENSOR_BOUND,
                    "{role:?} rows {rows}: chunk {chunk_error}"
                );
                assert!(
                    single_error < SIMT_BOUND,
                    "{role:?} rows {rows}: one-row {single_error}"
                );
            }
        }
        assert_eq!(scratch.flag_word(projector), 0, "arithmetic flag");
    });
}

#[test]
#[ignore = "requires admitted GPU memory (2 GiB) and the pinned EXL3 pack in FIFI_EXL3_CHECKPOINT"]
fn the_projections_land_in_their_columns_and_the_plain_gates_match_the_host() {
    with_rig(|rig| {
        columns_and_plain(rig);
        assert_eq!(rig.scratch.flag_word(rig.projector), 0, "arithmetic flag");
    });
}

/// The DeltaNet row: the two matrices land in their columns of a canary-filled `ProjD`, equal to
/// the contiguous projection, and the two plain gates equal the dot products on the host.
fn columns_and_plain(rig: &Rig<'_>) {
    let Rig {
        projector,
        mixers,
        launch,
        scratch,
        stream,
        geometry: g,
        delta: layer,
        ..
    } = *rig;
    let layer_weights = mixers.layer_weights(layer).unwrap();
    // One row, a part-tile, the 144-row tile of the old path, the 256-row job, one row over it, and
    // two jobs and a tail.
    for rows in [1, 9, 144, 256, 257, 433] {
        let chunk = scratch.chunk(launch, rows);
        let x = input(rows, g.hidden);
        let xb = upload(projector, &x);
        let proj = filled(projector, 0xff, rows * g.delta.ld * F32);
        let mut cases = vec![];
        for (role, n, col_off) in super::delta_proj_cols(g) {
            let plain = matches!(role, Role::SsmBeta | Role::SsmAlpha);
            if plain {
                let w = layer_weights.raw(role).unwrap();
                let shape = RouterShape {
                    m: rows,
                    hidden: g.hidden,
                    n_expert: n,
                    top_k: 1,
                };
                chunk
                    .plain(w, whole(&xb), shape, (whole(&proj), g.delta.ld, col_off))
                    .unwrap();
                cases.push((role, n, col_off, None));
                continue;
            }
            let weight = layer_weights.exl3(role).unwrap();
            let contiguous = filled(projector, 0, rows * n * F32);
            chunk
                .project(weight, whole(&xb), Output::Rows(whole(&contiguous)))
                .unwrap();
            let out = Output::Columns {
                dst: whole(&proj),
                ld: g.delta.ld,
                col_off,
            };
            chunk.project(weight, whole(&xb), out).unwrap();
            cases.push((role, n, col_off, Some(contiguous)));
        }
        stream.synchronize().unwrap();
        let got = read(projector, &proj, rows * g.delta.ld);
        for (role, n, col_off, contiguous) in cases {
            let flat = contiguous.map(|buffer| read(projector, &buffer, rows * n));
            let w = flat.is_none().then(|| {
                let range = layer_weights.raw(role).unwrap();
                let mut bytes = vec![0; range.bytes()];
                range
                    .buffer()
                    .read_at(stream, range.offset(), &mut bytes)
                    .unwrap();
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect::<Vec<f32>>()
            });
            for r in 0..rows {
                let row = &got[r * g.delta.ld + col_off..r * g.delta.ld + col_off + n];
                match (&flat, &w) {
                    (Some(flat), _) => assert_eq!(
                        row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        flat[r * n..(r + 1) * n]
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        "{role:?} rows {rows}: row {r} columns"
                    ),
                    (None, Some(w)) => {
                        let x = &x[r * g.hidden..(r + 1) * g.hidden];
                        let want: Vec<f64> = (0..n)
                            .map(|c| {
                                (0..g.hidden)
                                    .map(|i| f64::from(w[c * g.hidden + i]) * f64::from(x[i]))
                                    .sum()
                            })
                            .collect();
                        let error = relative_l2(row, &want);
                        assert!(error < 1e-5, "{role:?} rows {rows} row {r}: {error}");
                    }
                    (None, None) => unreachable!("a plain gate reads its weight"),
                }
            }
        }
        // Every column of a row that no projection owns keeps the canary.
        let owned: usize = super::delta_proj_cols(g).iter().map(|(_, n, _)| n).sum();
        assert_eq!(owned, g.delta.ld, "the four projections fill the row");
        assert!(got.iter().all(|v| !v.is_nan()), "every column written");
    }
}
