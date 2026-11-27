use super::*;

#[test]
#[ignore = "requires leased CUDA and fully resident Inkling Mini GGUF"]
fn whole_token_logits_agree_after_warm_host_row() {
    if !super::shadow_tests::exact_products("whole_token_logits_agree_after_warm_host_row") {
        return;
    }
    let path = std::env::var_os("FIFI_INKLING_MODEL_DIR").unwrap();
    let load = || {
        let mut f = CpuForward::open(std::path::Path::new(&path), 768).unwrap();
        f.enable_cuda(0).unwrap();
        f.token(1, false).unwrap();
        f
    };
    let oracle = {
        let mut f = load(); // A final feature tap forces ordinary visit dispatch.
        f.taps = vec![f.visits.len()];
        f.token(2, true).unwrap().unwrap()
    };
    let f = load();
    let Weights::Gguf(w) = &f.weights else {
        panic!("GGUF required")
    };
    let cuda = w.cuda().unwrap();
    let mut chain = TokenWorkspace::open(&f, cuda, &f.state)
        .unwrap()
        .expect("whole token was declined");
    assert!(chain.ready(cuda, &f.state, f.position));
    let test = chain.run(&f, cuda, &f.state, 2).unwrap();
    assert_eq!(test.len(), oracle.len());
    let (mut exact, mut max, mut diff, mut norm) = (0, 0f64, 0., 0.);
    for (&a, &b) in oracle.iter().zip(&test) {
        assert!(a.is_finite() && b.is_finite());
        let d = f64::from(a) - f64::from(b);
        exact += usize::from(a.to_bits() == b.to_bits());
        max = max.max(d.abs());
        diff += d * d;
        norm += f64::from(a).powi(2);
    }
    let rel = (diff / norm).sqrt();
    let argmax = |xs: &[f32]| {
        xs.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    eprintln!(
        "whole token logits exact={exact}/{} max_abs={max} rel_l2={rel} tol=0.02,0.001 top={}/{}",
        oracle.len(),
        argmax(&oracle),
        argmax(&test)
    );
    assert_eq!(argmax(&oracle), argmax(&test));
    assert!(max <= 0.02 && rel <= 0.001);
}

/// The logits of the whole-token chain over a short prompt and greedy continuation, kept as
/// bytes. With `FIFI_INKLING_CHAIN_BYTES` naming a missing file the run writes it; naming an
/// existing one, every logit must keep its bits: the proof that a change of how the chain is
/// driven left its numbers alone.
#[test]
#[ignore = "requires leased CUDA and fully resident Inkling Mini GGUF"]
fn the_token_chain_keeps_the_bits_of_its_logits() {
    let Some(bytes_path) = std::env::var_os("FIFI_INKLING_CHAIN_BYTES") else {
        eprintln!(
            "the_token_chain_keeps_the_bits_of_its_logits: skipped, set FIFI_INKLING_CHAIN_BYTES"
        );
        return;
    };
    let path = std::env::var_os("FIFI_INKLING_MODEL_DIR").unwrap();
    let mut f = CpuForward::open(std::path::Path::new(&path), 768).unwrap();
    f.enable_cuda(0).unwrap();
    let argmax = |xs: &[f32]| {
        xs.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32
    };
    let (mut out, mut chained) = (Vec::new(), 0);
    let mut next = 1u32;
    for step in 0..32u32 {
        let id = if step < 8 { 1 + 37 * step } else { next };
        let logits = f.token(id, true).unwrap().expect("logits");
        chained += usize::from(f.resident_token.is_some());
        next = argmax(&logits);
        out.extend(logits.iter().flat_map(|v| v.to_le_bytes()));
    }
    assert!(chained >= 30, "the chain served {chained} of 32 tokens");
    let file = std::path::Path::new(&bytes_path);
    if file.exists() {
        let want = std::fs::read(file).unwrap();
        assert_eq!(want.len(), out.len(), "logit count");
        let first = want.iter().zip(&out).position(|(a, b)| a != b);
        assert!(
            first.is_none(),
            "first differing byte {first:?} (logit {})",
            first.unwrap_or(0) / 4
        );
    } else {
        std::fs::write(file, &out).unwrap();
        eprintln!("wrote {} logits to {}", out.len() / 4, file.display());
    }
}
