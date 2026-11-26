//! Opt-in routing and diagnostics for Inkling's existing multirow attention projections.

use crate::family::inkling2::switches::Switch;
use std::sync::OnceLock;

use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceStream,
        resident::dense_gemv::{AttentionRowReuse, DenseDtype, DenseGemvCall},
    },
    weights::GgufType,
};

fn flags() -> (bool, bool) {
    static FLAGS: OnceLock<(bool, bool)> = OnceLock::new();
    *FLAGS.get_or_init(|| {
        (
            Switch::PrefillQ8RowReuse.on() || Switch::PrefillQ8Visit.on(),
            Switch::PrefillQ8RowReuseReport.on(),
        )
    })
}

pub(in crate::family::inkling2) fn project_group(
    context: &CudaContext,
    stream: &DeviceStream,
    calls: &[DenseGemvCall<'_>],
    stage: &str,
) -> Result<(), EngError> {
    let q6 = Switch::PrefillQ6RowReuse.on();
    let (q8, report) = flags();
    let counts = if q8 || q6 {
        context.dense_gemv_group_with_attention_reuse(
            stream,
            calls,
            AttentionRowReuse { q8, q6 },
        )?
    } else {
        context.dense_gemv_group(stream, calls)?;
        Default::default()
    };
    if counts.q6_row_pairs != 0 {
        static USED: std::sync::Once = std::sync::Once::new();
        USED.call_once(|| eprintln!("Inkling CUDA used exact Q6_K attention row pairs"));
        if report {
            eprintln!(
                "Inkling CUDA Q6 prefill {stage} rows {} pairs {}",
                calls[0].shape.m, counts.q6_row_pairs
            );
        }
    }
    if report {
        report_counts(calls, stage, counts.q8_row_pairs, counts.q8_grouped_visits);
    }
    Ok(())
}

pub(in crate::family::inkling2) fn project_output(
    context: &CudaContext,
    stream: &DeviceStream,
    call: DenseGemvCall<'_>,
) -> Result<(), EngError> {
    let (reuse, report) = flags();
    let shape = call.shape;
    let q8 = call.dtype == DenseDtype::PackedGguf(GgufType::Q80);
    let width = if reuse {
        let four = Switch::PrefillQ8OutputExact4.on();
        context.dense_gemv_with_q8_row_reuse(stream, call, four)?
    } else {
        context.dense_gemv(stream, call)?;
        0
    };
    if width == 4 {
        static USED: std::sync::Once = std::sync::Once::new();
        USED.call_once(|| eprintln!("Inkling CUDA used exact four-row Q8 attention output"));
        if report {
            eprintln!("Inkling CUDA Q8 output exact4 rows {}", shape.m);
        }
    } else if report && q8 {
        emit("attention_output", shape.m, 1, usize::from(width == 2), 0);
    }
    Ok(())
}

fn report_counts(calls: &[DenseGemvCall<'_>], stage: &str, paired: usize, grouped: usize) {
    // No WMMA/expert opt-in here: every Q8_0 call took the pair or ordinary GEMV entry.
    let mut q8 = calls
        .iter()
        .filter(|call| call.dtype == DenseDtype::PackedGguf(GgufType::Q80));
    if let Some(first) = q8.next() {
        emit(stage, first.shape.m, 1 + q8.count(), paired, grouped);
    }
}

fn emit(stage: &str, rows: usize, total: usize, paired: usize, grouped: usize) {
    let mut record = serde_json::json!({
        "stage": stage,
        "rows": rows,
        "q8_pair_calls": paired,
        "q8_gemv_calls": total - paired - grouped * 2,
    });
    if grouped != 0 {
        record["q8_grouped_visit_launches"] = grouped.into();
    }
    eprintln!("Inkling CUDA Q8 prefill {record}");
}
