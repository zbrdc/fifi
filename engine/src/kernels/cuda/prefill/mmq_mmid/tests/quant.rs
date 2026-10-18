//! Seam tests of the typed scatter and gather quantize wrappers (CPF-V4 G3 part 2, the rev 3
//! grant's MAJOR item). The scatter test compares the q8_1 buffer of
//! `mmq_quantize_scatter_ids` byte for byte with G4's plain quantize of the token rows laid
//! out at their compact columns (`ids_dst`, the builder's own map), and the gather test the
//! buffer of `mmq_quantize_gather_ids` with the plain quantize of the rows `ids_dst` names. A
//! wrapper that took the wrong map from the `MmidIds` (the forward map for the inverse one, or
//! the other way) lays the columns out differently and fails the comparison; both tests first
//! assert that the two maps differ, so the comparison can tell them apart.
//!
//! Each test runs two `Result`-only device phases (the typed call, then the plain one), asserts
//! after both contexts are gone, and gates first: every output id in its domain and every scale
//! word of the poisoned q8_1 buffers finite, so a skipped store fails before the comparison.

use super::rig::{
    IdBufs, POISON, assert_same_bytes, f32_bytes, normal_rows, scales_finite, small_qwen,
};
use super::{as_i32, ints, random_rows, read};
use crate::EngError;
use crate::kernels::cuda::prefill::mmq_mmid::{MmidQuantCall, MmidShape, mmq_type_of};
use crate::kernels::cuda::prefill::mmq_quant::MmqQuantizeCall;
use crate::kernels::cuda::prefill::mmq_size::{Q8_1_PAD, q8_1_bytes};
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::testkit;
use crate::kernels::cuda::{CudaContext, DeviceRange};

/// What a typed quantize call left on the device, and the maps the builder wrote.
struct TypedRun {
    y: Vec<u8>,
    ids_dst: Vec<i32>,
    ids_src1: Vec<i32>,
}

/// Builds the ids of `rows` and runs the typed wrapper (`gather` false: the scatter) over `x`
/// with the q8_1 buffer poisoned beforehand.
fn typed_phase(
    shape: MmidShape,
    rows: &[i32],
    x: &[f32],
    (quant, k, gather): (QuantType, usize, bool),
) -> Result<TypedRun, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let bufs = IdBufs::new(&ctx, &shape, rows, 0)?;
    let ids = bufs.build(&ctx, &stream, shape)?;
    let y_bytes = q8_1_bytes(k, shape.ne_get_rows())?;
    let x_buf = ctx.upload(&f32_bytes(x))?;
    let y_buf = ctx.upload(&vec![POISON; y_bytes])?;
    let call = MmidQuantCall {
        packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
        quant,
        k,
        row_stride: k,
        x: x_buf.range(0, x.len() * 4)?,
        y: y_buf.range(0, y_bytes)?,
    };
    if gather {
        ctx.mmq_quantize_gather_ids(&stream, &ids, call)?;
    } else {
        ctx.mmq_quantize_scatter_ids(&stream, &ids, call)?;
    }
    stream.synchronize()?;
    Ok(TypedRun {
        y: read(&ctx, &y_buf)?,
        ids_dst: ints(&read(&ctx, &bufs.ids_dst)?),
        ids_src1: ints(&read(&ctx, &bufs.ids_src1)?),
    })
}

/// G4's plain quantize of `ncols` rows of `x`, the oracle of the typed wrappers.
fn plain_phase(quant: QuantType, k: usize, x: &[f32]) -> Result<Vec<u8>, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let ncols = x.len() / k;
    let y_bytes = q8_1_bytes(k, ncols)?;
    let x_buf = ctx.upload(&f32_bytes(x))?;
    let y_buf = ctx.upload(&vec![POISON; y_bytes])?;
    ctx.mmq_quantize(
        &stream,
        MmqQuantizeCall {
            packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
            ty: mmq_type_of(quant),
            k,
            row_stride: k,
            ncols,
            x: x_buf.range(0, x.len() * 4)?,
            gather: None,
            y: y_buf.range(0, y_bytes)?,
        },
    )?;
    stream.synchronize()?;
    read(&ctx, &y_buf)
}

/// The gates both tests share, in order: the maps hold values in `0..ne`, differ from each
/// other, and the q8_1 scale words are finite where the data sits and still poison in the pad.
fn gate_typed_run(run: &TypedRun, ne: usize, k: usize, quant: QuantType) {
    let n = as_i32(ne);
    assert!(
        run.ids_dst[..ne].iter().all(|v| (0..n).contains(v)),
        "HARD GATE ids_dst"
    );
    assert!(
        run.ids_src1[..ne].iter().all(|v| (0..n).contains(v)),
        "HARD GATE ids_src1"
    );
    assert_ne!(
        run.ids_dst[..ne],
        run.ids_src1[..ne],
        "the test ids form an involution, which cannot tell the two maps apart"
    );
    let blocks = ne * (k / 128);
    assert!(
        scales_finite(&run.y, blocks, quant != QuantType::Q4K),
        "HARD GATE: a q8_1 scale word is not finite (an unwritten block)"
    );
    let pad = &run.y[run.y.len() - Q8_1_PAD..];
    assert!(pad.iter().all(|&b| b == POISON), "the q8_1 pad was written");
}

#[test]
fn mmq_quantize_scatter_ids_lays_the_token_rows_out_by_the_builder_map() {
    if !testkit::device_present(
        "mmq_quantize_scatter_ids_lays_the_token_rows_out_by_the_builder_map",
    ) {
        return;
    }
    let shape = small_qwen(40);
    let (ne, k) = (shape.ne_get_rows(), 512);
    let rows = random_rows(&shape, 0x5ca7);
    let x = normal_rows(shape.m, k, 0x5ca8);
    for quant in [QuantType::Q4K, QuantType::Q6K] {
        let run = typed_phase(shape, &rows, &x, (quant, k, false)).expect("the typed scatter");
        gate_typed_run(&run, ne, k, quant);
        // Compact column c holds token `ids_dst[c] / top_k`'s row.
        let laid_out: Vec<f32> = run.ids_dst[..ne]
            .iter()
            .flat_map(|&slot| {
                let token = usize::try_from(slot).expect("gated") / shape.top_k;
                x[token * k..(token + 1) * k].to_vec()
            })
            .collect();
        let want = plain_phase(quant, k, &laid_out).expect("the plain quantize");
        assert_same_bytes(
            &run.y,
            &want,
            &format!("{quant:?}: the scatter against the plain quantize"),
        );
    }
}

#[test]
fn mmq_quantize_gather_ids_reads_the_rows_the_forward_map_names() {
    if !testkit::device_present("mmq_quantize_gather_ids_reads_the_rows_the_forward_map_names") {
        return;
    }
    let shape = small_qwen(40);
    let (ne, k) = (shape.ne_get_rows(), 512);
    let rows = random_rows(&shape, 0x6a7e);
    let act = normal_rows(ne, k, 0x6a7f);
    for quant in [QuantType::Q4K, QuantType::Q6K] {
        let run = typed_phase(shape, &rows, &act, (quant, k, true)).expect("the typed gather");
        gate_typed_run(&run, ne, k, quant);
        // Column j holds row `ids_dst[j]` of the token-slot ordered activations.
        let gathered: Vec<f32> = run.ids_dst[..ne]
            .iter()
            .flat_map(|&slot| {
                let slot = usize::try_from(slot).expect("gated");
                act[slot * k..(slot + 1) * k].to_vec()
            })
            .collect();
        let want = plain_phase(quant, k, &gathered).expect("the plain quantize");
        assert_same_bytes(
            &run.y,
            &want,
            &format!("{quant:?}: the gather against the plain quantize"),
        );
    }
}

/// A Q4_K typed call over contiguous rows of `k` floats.
fn q4k_call<'b>(k: usize, x: DeviceRange<'b>, y: DeviceRange<'b>) -> MmidQuantCall<'b> {
    MmidQuantCall {
        packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
        quant: QuantType::Q4K,
        k,
        row_stride: k,
        x,
        y,
    }
}

/// One refusal row of the typed wrappers: a name, a needle of the expected text, the outcome.
type TypedRow = (&'static str, &'static str, Result<(), EngError>);

#[test]
fn mmq_quantize_ids_wrappers_refuse_ranges_that_do_not_fit() {
    if !testkit::device_present("mmq_quantize_ids_wrappers_refuse_ranges_that_do_not_fit") {
        return;
    }
    let phase = || -> Result<Vec<TypedRow>, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let shape = MmidShape {
            n_experts: 4,
            m: 4,
            top_k: 2,
        };
        let bufs = IdBufs::new(&ctx, &shape, &[0, 1, 1, 2, 2, 3, 3, 0], 0)?;
        let ids = bufs.build(&ctx, &stream, shape)?;
        let (ne, k) = (8, 256);
        let y_need = q8_1_bytes(k, ne)?;
        let (x_scatter, x_gather) = (shape.m * k * 4, ne * k * 4);
        let x_buf = ctx.upload(&vec![0u8; x_gather])?;
        let y_buf = ctx.upload(&vec![POISON; y_need])?;
        let call = q4k_call;
        let (x, y) = (|n: usize| x_buf.range(0, n), |n: usize| y_buf.range(0, n));
        let overlap = ctx.upload(&vec![0u8; y_need])?;
        let over = |n: usize| overlap.range(0, n);
        let scatter = |c| ctx.mmq_quantize_scatter_ids(&stream, &ids, c);
        let gather = |c| ctx.mmq_quantize_gather_ids(&stream, &ids, c);
        let rows: Vec<TypedRow> = vec![
            (
                "scatter base",
                "",
                scatter(call(k, x(x_scatter)?, y(y_need)?)),
            ),
            (
                "scatter x one float short",
                "the CUDA x range holds",
                scatter(call(k, x(x_scatter - 4)?, y(y_need)?)),
            ),
            (
                "scatter y one byte short of rule M1",
                "the CUDA y range holds",
                scatter(call(k, x(x_scatter)?, y(y_need - 1)?)),
            ),
            (
                "scatter y over x",
                "the CUDA y and x ranges overlap",
                scatter(call(k, over(x_scatter)?, over(y_need)?)),
            ),
            (
                "scatter K off the grid",
                "mmq_quant: K-grid",
                scatter(call(384, x(x_scatter)?, y(y_need)?)),
            ),
            ("gather base", "", gather(call(k, x(x_gather)?, y(y_need)?))),
            (
                "gather x one float short",
                "the CUDA x range holds",
                gather(call(k, x(x_gather - 4)?, y(y_need)?)),
            ),
            (
                "gather y one byte short of rule M1",
                "the CUDA y range holds",
                gather(call(k, x(x_gather)?, y(y_need - 1)?)),
            ),
            (
                "gather y over x",
                "the CUDA y and x ranges overlap",
                gather(call(k, over(x_gather)?, over(y_need)?)),
            ),
            (
                "gather K off the grid",
                "mmq_quant: K-grid",
                gather(call(384, x(x_gather)?, y(y_need)?)),
            ),
        ];
        stream.synchronize()?;
        Ok(rows)
    };
    for (name, needle, outcome) in phase().expect("the refusal phase") {
        match (needle.is_empty(), outcome) {
            (true, Ok(())) => {}
            (false, Err(EngError::BadRequest(text))) => {
                assert!(text.contains(needle), "{name}: {text}");
            }
            (_, other) => panic!("{name}: unexpected outcome {other:?}"),
        }
    }
}
