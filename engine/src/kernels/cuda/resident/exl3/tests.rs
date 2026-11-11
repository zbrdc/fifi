//! EXL3 device checks: run only in the canonical three-lock GPU window.
use super::Exl3PrepCall;
use crate::kernels::cuda::{CudaContext, testkit};
use crate::tensor::{Dtype, Tensor};
use crate::weights::exl3::Codebook;

#[test]
fn midrows_policy_keeps_small_rows_legacy_and_honors_rollback() {
    use std::ffi::OsStr;
    for (rows, enabled) in [
        (0, false),
        (8, false),
        (9, false),
        (32, false),
        (63, false),
        (64, true),
        (128, true),
        (144, true),
        (145, false),
        (usize::MAX, false),
    ] {
        assert_eq!(super::midrows_enabled(rows, None), enabled, "rows{rows}");
        assert!(!super::midrows_enabled(rows, Some(OsStr::new("1"))));
    }
    for value in ["", "0", "01", "true"] {
        assert!(super::midrows_enabled(64, Some(OsStr::new(value))));
    }
}

#[test]
fn all_codebooks_and_rates_prepare_bit_exact_nonsquare_wprime() {
    if !testkit::device_present("EXL3 W-prime all codebooks/rates") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x4558_4c33);
    let (k, n) = (128usize, 256usize);
    for book in [Codebook::ThreeInst, Codebook::Mcg, Codebook::Mul1] {
        for bits_twice in 2u8..=16 {
            if bits_twice % 2 != 0 && !(book == Codebook::Mul1 && matches!(bits_twice, 3 | 5 | 7)) {
                continue;
            }
            let words = usize::from(bits_twice) * 8;
            let bytes: Vec<u8> = (0..k * n / 256 * words)
                .flat_map(|_| (rng.next_u64() as u16).to_le_bytes())
                .collect();
            let tensor = Tensor::try_from_host(
                Dtype::I16,
                vec![(k / 16) as u32, (n / 16) as u32, words as u32],
                bytes.clone(),
            )
            .expect("trellis");
            let mut expected = vec![0u16; k * n];
            book.reconstruct_into(&tensor, &mut expected)
                .expect("CPU format oracle");
            let trellis = testkit::upload_bytes(&ctx, &bytes);
            let output = testkit::upload_filled(&ctx, 0xff, k * n * 2);
            ctx.exl3_prep(
                &stream,
                Exl3PrepCall {
                    trellis: trellis.range(0, bytes.len()).unwrap(),
                    out: output.range(0, k * n * 2).unwrap(),
                    k,
                    n,
                    bits_twice,
                    book,
                },
            )
            .expect("GPU prep");
            stream.synchronize().expect("drain");
            let actual = testkit::read_bytes(&ctx, &output, k * n * 2);
            let expected: Vec<u8> = expected
                .iter()
                .flat_map(|bits| bits.to_le_bytes())
                .collect();
            assert_eq!(actual, expected, "{book:?}, bits_twice={bits_twice}");
        }
    }
}

/// An explicit value of the variable wins over any context default; an unset variable takes the
/// default of the context, and a context without one runs the shuffle kernel. Two contexts hold
/// their own defaults, whichever registers first.
#[test]
fn the_single_row_default_is_per_context_and_the_variable_wins() {
    use super::{SingleRow, effective_single_row as effective};
    assert_eq!(effective(None, None), SingleRow::Legacy);
    assert_eq!(
        effective(None, Some(SingleRow::Wide(1))),
        SingleRow::Wide(1)
    );
    assert_eq!(
        effective(Some(SingleRow::Legacy), Some(SingleRow::Wide(1))),
        SingleRow::Legacy
    );
    assert_eq!(
        effective(Some(SingleRow::Gather), Some(SingleRow::Wide(1))),
        SingleRow::Gather
    );
    // Entry A registers wide1, entry B registers nothing; the order does not matter.
    let defaults = [Some(SingleRow::Wide(1)), None];
    for order in [[0, 1], [1, 0]] {
        let seen: Vec<SingleRow> = order
            .iter()
            .map(|&i| effective(None, defaults[i]))
            .collect();
        let expected: Vec<SingleRow> = order
            .iter()
            .map(|&i| {
                if i == 0 {
                    SingleRow::Wide(1)
                } else {
                    SingleRow::Legacy
                }
            })
            .collect();
        assert_eq!(seen, expected);
    }
}

#[test]
fn the_single_gather_switch_parses_strictly() {
    use super::SingleRow;
    use std::ffi::OsStr;
    let parse = |value: &str| super::single_gather_from_env(Some(OsStr::new(value)));
    assert_eq!(parse("off"), Ok(SingleRow::Legacy));
    assert_eq!(super::single_gather_from_env(None), Ok(SingleRow::Legacy));
    for (value, kernel) in [
        ("1", SingleRow::Gather),
        (" 1 ", SingleRow::Gather),
        ("wide1", SingleRow::Wide(1)),
        ("wide2", SingleRow::Wide(2)),
        ("wide4", SingleRow::Wide(4)),
    ] {
        assert_eq!(parse(value), Ok(kernel), "{value:?}");
    }
    for bad in [
        "", "0", "true", "on", "11", "01", "gather", "wide", "wide3", "wide8", "Wide1",
    ] {
        let refusal = parse(bad).expect_err("every other value is refused");
        assert!(
            refusal.contains("FIFI_EXL3_SINGLE_GATHER"),
            "{bad}: {refusal}"
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert!(super::single_gather_from_env(Some(OsStr::from_bytes(b"\xff"))).is_err());
    }
}

#[test]
fn the_single_gather_route_takes_only_small_split_two_calls() {
    use super::{SingleRow, single_row_route};
    for kernel in [SingleRow::Gather, SingleRow::Wide(2)] {
        for rows in 1..=8 {
            assert_eq!(single_row_route(kernel, rows, 2), kernel, "rows {rows}");
        }
        for rows in [0, 9, 16, 64, 144] {
            assert_eq!(
                single_row_route(kernel, rows, 2),
                SingleRow::Legacy,
                "rows {rows}"
            );
        }
        for split_k in [0, 1, 3, 4, 8] {
            assert_eq!(
                single_row_route(kernel, 1, split_k),
                SingleRow::Legacy,
                "split_k {split_k}"
            );
        }
    }
    for rows in 1..=8 {
        assert_eq!(
            single_row_route(SingleRow::Legacy, rows, 2),
            SingleRow::Legacy
        );
    }
}

/// The switch is consulted before the legacy kernel is chosen: deleting the branch, or moving it
/// after the legacy launch, must fail here.
#[test]
fn exl3_linear_consults_the_single_row_switch_before_the_legacy_kernel() {
    let source = include_str!("../exl3.rs");
    let start = source
        .find("pub(crate) fn exl3_linear(")
        .expect("exl3_linear");
    let body = &source[start..start + 1300];
    let switch = body
        .find("single_row_route(")
        .expect("the routing predicate");
    let grouped = body.find("self.grouped_by(").expect("the grouped route");
    let legacy = body
        .find("self.exl3_linear_mode(")
        .expect("the legacy route");
    assert!(switch < grouped && grouped < legacy, "{body}");
}

#[test]
fn the_load_line_names_only_an_armed_single_row_kernel() {
    use super::{SingleRow, single_row_label};
    assert_eq!(single_row_label(SingleRow::Legacy), None);
    assert_eq!(single_row_label(SingleRow::Gather), Some("gather"));
    for (cols, name) in [(1, "wide1"), (2, "wide2"), (4, "wide4")] {
        assert_eq!(single_row_label(SingleRow::Wide(cols)), Some(name));
    }
}

/// The grouped kernel follows the switch and the widths: wide only for a wide setting whose tile
/// span divides every width; mapping a wide setting to the gather kernel, or the wrong tile
/// count, fails here.
#[test]
fn the_grouped_kernel_follows_the_setting_and_the_widths() {
    use super::{GroupedKernel, SingleRow, grouped_kernel_for};
    for setting in [SingleRow::Legacy, SingleRow::Gather] {
        assert_eq!(
            grouped_kernel_for(setting, [2048, 512]),
            GroupedKernel::Gather
        );
    }
    for cols in [1, 2, 4] {
        assert_eq!(
            grouped_kernel_for(SingleRow::Wide(cols), [2048, 512, 4096]),
            GroupedKernel::Wide(cols)
        );
    }
    // 96 = 16 * 6: divisible by 1 and 2 tiles, not by 4.
    assert_eq!(
        grouped_kernel_for(SingleRow::Wide(2), [96]),
        GroupedKernel::Wide(2)
    );
    assert_eq!(
        grouped_kernel_for(SingleRow::Wide(4), [96]),
        GroupedKernel::Gather
    );
    // One width that does not divide takes the whole launch to the gather kernel.
    assert_eq!(
        grouped_kernel_for(SingleRow::Wide(4), [2048, 96]),
        GroupedKernel::Gather
    );
}

/// The step entry and the single-row entry both choose through `grouped_kernel_for` and launch
/// through `grouped_by`, which names the wide launch.
#[test]
fn both_grouped_entries_choose_through_the_kernel_function() {
    let source = include_str!("../exl3.rs");
    for entry in [
        "pub(crate) fn exl3_grouped_step(",
        "pub(crate) fn exl3_linear(",
    ] {
        let start = source.find(entry).expect(entry);
        // 1200 chars: exl3_grouped_step grew past 900 when its fused-input arm landed.
        let body = &source[start..start + 1200];
        assert!(body.contains("grouped_kernel_for("), "{entry}: {body}");
        assert!(body.contains("grouped_by("), "{entry}: {body}");
    }
    let by = source.find("fn grouped_by(").expect("grouped_by");
    let body = &source[by..by + 500];
    assert!(
        body.contains("self.exl3_grouped_wide(stream, calls, cols)"),
        "{body}"
    );
    assert!(
        body.contains("self.exl3_grouped(stream, calls, true)"),
        "{body}"
    );
}

#[test]
fn the_warp_rotation_switch_accepts_unset_zero_and_one_only() {
    use std::ffi::OsStr;
    let parse = |value: &str| super::jobs_warp_rotate_from_env(Some(OsStr::new(value)));
    assert_eq!(super::jobs_warp_rotate_from_env(None), Ok(false));
    assert_eq!(parse("0"), Ok(false));
    assert_eq!(parse("1"), Ok(true));
    for bad in ["", "2", "01", "on", "true", "off", " 1", "1 ", "warp"] {
        let refusal = parse(bad).expect_err(bad);
        assert!(
            refusal.contains("FIFI_EXL3_JOBS_WARP_ROTATE"),
            "{bad}: {refusal}"
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert!(super::jobs_warp_rotate_from_env(Some(OsStr::from_bytes(b"\xff"))).is_err());
    }
}

#[test]
fn the_pair_rotation_switch_accepts_unset_zero_and_one_only() {
    use std::ffi::OsStr;
    let parse = |value: &str| super::jobs_pair_rotate_from_env(Some(OsStr::new(value)));
    assert_eq!(super::jobs_pair_rotate_from_env(None), Ok(false));
    assert_eq!(parse("0"), Ok(false));
    assert_eq!(parse("1"), Ok(true));
    for bad in ["", "2", "01", "on", "true", "pair"] {
        let refusal = parse(bad).expect_err(bad);
        assert!(
            refusal.contains("FIFI_EXL3_JOBS_PAIR_ROTATE"),
            "{bad}: {refusal}"
        );
    }
}
