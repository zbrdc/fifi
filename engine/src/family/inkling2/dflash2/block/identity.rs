//! Slow real-checkpoint CPU gate. Check a fresh plain target against 64 verified windows.
use super::*;
use crate::family::{
    inkling2::cpu_forward::CpuForward,
    sampler::Sampler,
    session::TokenStepper,
    speculate::{accept::accept_run, driver::WindowModel, window::Window},
};
#[test]
#[ignore = "slow: real target + drafter, CPU-only"]
fn greedy_64_windows() {
    let p = std::env::var_os("FIFI_INKLING_MODEL_DIR").unwrap();
    let side = std::env::var_os("FIFI_INKLING_SPEC_DRAFT").unwrap();
    let w = super::super::open(Path::new(&side)).unwrap();
    let mut model = Dflash2(w);
    let mut target = CpuForward::open(Path::new(&p), 521).unwrap();
    let mut plain = CpuForward::open(Path::new(&p), 521).unwrap();
    target.prepare_draft(vec![5, 23, 41, 48], 7).unwrap();
    target.feed(1).unwrap();
    plain.feed(1).unwrap();
    let mut anchor = 2;
    let mut solo = anchor;
    let mut sampler = Sampler::greedy();
    let mut accepted = 0;
    let mut history = vec![1, anchor];
    for step in 0..64 {
        let drafts = model
            .propose_conditioned(&history, 7, target.draft_conditioning())
            .unwrap()
            .tokens;
        assert_eq!(drafts.len(), 7);
        let window = Window::new(target.fed(), drafts, vec![]);
        target.run_window(&window.input(anchor)).unwrap();
        let verdict = accept_run(
            window.drafts(),
            8,
            |row| target.draw(row, &mut sampler, 200058),
            |_| false,
        )
        .unwrap();
        target
            .commit_window(&window.commit(&verdict).unwrap())
            .unwrap();
        for &id in &verdict.tokens {
            solo = plain.next_token(solo, &mut sampler, 200058).unwrap();
            assert_eq!(id, solo, "step {step}");
        }
        history.extend_from_slice(&verdict.tokens);
        anchor = solo;
        accepted += verdict.tokens.len() - 1;
        eprintln!(
            "step={} accepted_drafts={} total_accepted_drafts={accepted}",
            step + 1,
            verdict.tokens.len() - 1
        );
    }
    eprintln!(
        "accepted drafts/window={} over 64 windows",
        accepted as f64 / 64.
    );
}
