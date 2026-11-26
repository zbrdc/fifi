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
