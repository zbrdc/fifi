use super::*;

/// Whether the device products are the exact f32 ones, the decode experts and the prompt's
/// projections and experts. The chain and the host token path agree to rounding only then: the
/// q8_1 products quantize each visit's inputs, so state rows drift apart by a few percent over a
/// long run. The comparisons print why they skip; the test chains run them with the three
/// switches below off.
pub(super) fn exact_products(test: &str) -> bool {
    use crate::family::inkling2::switches::Switch;
    let exact = !Switch::DecodeQ8Experts.on()
        && !Switch::PrefillTensorCoreProjections.on()
        && !Switch::PrefillTensorCoreExperts.on();
    if !exact {
        eprintln!(
            "{test}: skipped, it compares to the f32 products; set FIFI_INKLING_CUDA_DECODE_Q8_EXPERTS=0, FIFI_INKLING_CUDA_PREFILL_TENSOR_CORE_PROJECTIONS=0 and FIFI_INKLING_CUDA_PREFILL_TENSOR_CORE_EXPERTS=0"
        );
    }
    exact
}

#[derive(Default)]
pub(super) struct Error {
    n: usize,
    exact: usize,
    max: f64,
    diff: f64,
    norm: f64,
}
impl Error {
    pub(super) fn rows(&mut self, a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (&a, &b) in a.iter().zip(b) {
            assert!(a.is_finite() && b.is_finite());
            let d = f64::from(a) - f64::from(b);
            self.n += 1;
            self.exact += usize::from(a.to_bits() == b.to_bits());
            self.max = self.max.max(d.abs());
            self.diff += d * d;
            self.norm += f64::from(a).powi(2);
        }
    }
    pub(super) fn max_abs(&self) -> f64 {
        self.max
    }
    pub(super) fn check(self, what: &str, max: f64, l2: f64) {
        let rel = (self.diff / self.norm).sqrt();
        eprintln!(
            "{what}: exact={}/{} max_abs={} rel_l2={rel} limits={max},{l2}",
            self.exact, self.n, self.max
        );
        assert!(self.max <= max && rel <= l2);
    }
}

#[test]
#[ignore = "requires a leased CUDA device and real packed Inkling GGUF"]
fn shadow_snapshot_restore_and_resume() {
    if !exact_products("shadow_snapshot_restore_and_resume") {
        return;
    }
    let path = std::env::var_os("FIFI_INKLING_MODEL_DIR").unwrap();
    let run = |oracle: bool| {
        let mut f = CpuForward::open(std::path::Path::new(&path), 768).unwrap();
        f.enable_cuda(0).unwrap();
        if oracle {
            f.taps = vec![f.visits.len()];
        }
        f.token(1, false).unwrap();
        let mut logits = Vec::new();
        logits.push(f.token(2, true).unwrap().unwrap());
        let saved = f.snapshot().unwrap();
        logits.push(f.token(3, true).unwrap().unwrap());
        logits.push(f.token(4, true).unwrap().unwrap());
        f.restore(&saved);
        assert_eq!(f.state, saved.visits);
        assert!(
            !f.admit_resident_token().unwrap(),
            "invalidated ring must decline before any state change"
        );
        assert_eq!(f.state, saved.visits);
        logits.push(f.token(5, true).unwrap().unwrap());
        assert_eq!(
            f.admit_resident_token().unwrap(),
            !oracle,
            "host fallback rebuilt device cursors"
        );
        logits.push(f.token(6, true).unwrap().unwrap());
        assert_eq!(f.resident_token.is_some(), !oracle);
        // A committed chain token leaves its rows on the device; the host reads them back on demand.
        assert_eq!(f.state.iter().any(VisitState::is_resident), !oracle);
        f.materialize_resident().unwrap();
        assert!(!f.state.iter().any(VisitState::is_resident));
        (logits, saved.visits, f.state)
    };
    let (reference, sa, fa) = run(true);
    let (candidate, sb, fb) = run(false);
    let mut logits = Error::default();
    for (a, b) in reference.iter().zip(&candidate) {
        logits.rows(a, b);
        let arg = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        assert_eq!(arg(a), arg(b));
    }
    logits.check("shadow logits", 0.02, 0.001);
    let mut states = Error::default();
    for (a, b) in sa.iter().chain(&fa).zip(sb.iter().chain(&fb)) {
        assert_eq!(a.position(), b.position());
        let (aa, bb) = (a.shadow_rows(), b.shadow_rows());
        assert_eq!(aa.len(), bb.len());
        for (x, y) in aa.into_iter().zip(bb) {
            states.rows(x, y);
        }
    }
    states.check("snapshot K/V and convolution rows", 0.1, 0.001);
}
