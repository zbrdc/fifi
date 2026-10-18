//! Pure bit comparison and gate status decision. Compiles on every host.

use serde::Serialize;

use crate::run::CaseReport;

#[derive(Serialize)]
pub struct PathResult {
    pub path: &'static str,
    pub bit_differences: u64,
    pub value_differences: u64,
    pub first_differences: Vec<String>,
    pub actual_sha256: String,
}

#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]
pub fn compare(path: &'static str, expected: &[u8], actual: &[u8]) -> PathResult {
    let words = |b: &[u8]| -> Vec<u32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect()
    };
    let (e, a) = (words(expected), words(actual));
    let (mut bits, mut values, mut first) = (0u64, 0u64, Vec::new());
    for (i, (x, y)) in e.iter().zip(&a).enumerate() {
        if x != y {
            bits += 1;
            if f32::from_bits(*x) != f32::from_bits(*y) {
                values += 1;
            }
            if first.len() < 8 {
                first.push(format!("[{i}] expected {x:#010x} actual {y:#010x}"));
            }
        }
    }
    if e.len() != a.len() {
        bits += e.len().abs_diff(a.len()) as u64;
        first.push(format!("length expected {} actual {}", e.len(), a.len()));
    }
    PathResult {
        path,
        bit_differences: bits,
        value_differences: values,
        first_differences: first,
        actual_sha256: crate::fixture::hex(actual),
    }
}

pub struct Verdict {
    pub status: &'static str,
    pub differing_modules: Vec<String>,
}

const NEGATIVE_MODULE: &str = "language_model.model.layers.0.mlp.gate";

/// Decide the gate status from per-case results.
pub fn decide(cases: &[CaseReport], bit_differences: u64, negative_control: bool) -> Verdict {
    let differing: std::collections::BTreeSet<String> = cases
        .iter()
        .filter(|c| c.results.iter().any(|r| r.bit_differences > 0))
        .map(|c| c.module.clone())
        .collect();
    let status = if negative_control {
        let detected = cases
            .iter()
            .filter(|c| c.module == NEGATIVE_MODULE)
            .all(|c| c.results.iter().all(|r| r.bit_differences > 0))
            && differing.iter().all(|m| m == NEGATIVE_MODULE);
        if detected {
            "negative_control_detected"
        } else {
            "negative_control_missed"
        }
    } else if bit_differences == 0 {
        "passed"
    } else {
        "mismatch"
    };
    Verdict {
        status,
        differing_modules: differing.into_iter().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn case(module: &str, diffs: u64) -> CaseReport {
        CaseReport {
            id: "c".into(),
            module: module.into(),
            m: 1,
            expected_kernel: String::new(),
            results: vec![compare(
                "p",
                &bytes(&[1]),
                &bytes(&[if diffs == 0 { 1 } else { 2 }]),
            )],
        }
    }

    #[test]
    fn equal_bits_report_no_difference() {
        let r = compare("p", &bytes(&[1, 2, 3]), &bytes(&[1, 2, 3]));
        assert_eq!((r.bit_differences, r.value_differences), (0, 0));
    }

    #[test]
    fn signed_zero_is_a_bit_difference_not_a_value_difference() {
        let r = compare("p", &bytes(&[0x0000_0000]), &bytes(&[0x8000_0000]));
        assert_eq!((r.bit_differences, r.value_differences), (1, 0));
    }

    #[test]
    fn nan_with_other_payload_is_a_bit_difference() {
        let r = compare("p", &bytes(&[0x7fc0_0000]), &bytes(&[0x7fc0_0001]));
        assert_eq!(r.bit_differences, 1);
    }

    #[test]
    fn length_difference_counts_and_is_recorded() {
        let r = compare("p", &bytes(&[1, 2]), &bytes(&[1]));
        assert_eq!(r.bit_differences, 1);
        assert!(r.first_differences.iter().any(|d| d.starts_with("length")));
    }

    #[test]
    fn first_differences_are_capped_at_eight_plus_length_note() {
        let a = bytes(&[0; 20]);
        let b = bytes(&[1; 20]);
        let r = compare("p", &a, &b);
        assert_eq!(r.bit_differences, 20);
        assert_eq!(r.first_differences.len(), 8);
    }

    #[test]
    fn gate_passes_only_without_differences() {
        assert_eq!(decide(&[case("a", 0)], 0, false).status, "passed");
        assert_eq!(decide(&[case("a", 1)], 1, false).status, "mismatch");
    }

    #[test]
    fn negative_control_requires_only_the_target_module_to_differ() {
        let l0 = NEGATIVE_MODULE;
        let ok = decide(&[case(l0, 1), case("b", 0)], 1, true);
        assert_eq!(ok.status, "negative_control_detected");
        assert_eq!(ok.differing_modules, vec![l0.to_string()]);
        assert_eq!(
            decide(&[case(l0, 0), case("b", 0)], 0, true).status,
            "negative_control_missed"
        );
        assert_eq!(
            decide(&[case(l0, 1), case("b", 1)], 2, true).status,
            "negative_control_missed"
        );
    }
}
