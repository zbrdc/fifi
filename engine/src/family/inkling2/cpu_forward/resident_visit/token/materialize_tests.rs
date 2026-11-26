//! Proof that device-owned rows are read back intact: after whole chain tokens, a snapshot
//! taken every N tokens must hold the K/V rows and convolution histories of the ordinary
//! host-orchestrated path (flag off), and the logits must agree at every token. The token
//! sequences are the standard serving fixtures (`fixture_ids.py` output in
//! `FIFI_INKLING_FIXTURES`), continued greedily by the ordinary path.
use super::shadow_tests::Error;
use super::*;
use crate::family::inkling2::cpu_forward::PrefixState;

pub(super) struct Fixture {
    pub(super) name: &'static str,
    pub(super) prompt: Vec<u32>,
    pub(super) limit: usize,
}

fn fixtures() -> Vec<Fixture> {
    let path = std::env::var_os("FIFI_INKLING_FIXTURES").expect("FIFI_INKLING_FIXTURES");
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    ["warmup", "context_512", "natural_decode_64"]
        .into_iter()
        .map(|name| Fixture {
            name,
            prompt: json[name]["prompt"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_u64().unwrap() as u32)
                .collect(),
            limit: json[name]["limit"].as_u64().unwrap() as usize,
        })
        .collect()
}

pub(super) fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

#[derive(Default)]
pub(super) struct Run {
    logits: Vec<Vec<f32>>,
    /// Greedy continuation, one token per logits row.
    pub(super) tokens: Vec<u32>,
    /// Host state after every token (rows are shared with the forward's storage).
    snapshots: Vec<PrefixState>,
}

/// Feeds a fixture the way serving does: grouped prefill of every prompt row but the last, then
/// the last prompt token as the first decode step.
pub(super) fn generate(
    f: &mut CpuForward,
    fixture: &Fixture,
    forced: Option<&[u32]>,
    every: usize,
) -> Run {
    let chain = forced.is_some();
    if fixture.prompt.len() > 1 {
        f.prefill_grouped(&fixture.prompt[..fixture.prompt.len() - 1])
            .unwrap();
    }
    let mut run = Run::default();
    let mut id = *fixture.prompt.last().unwrap();
    let mut download = std::time::Duration::ZERO;
    for step in 0..fixture.limit {
        let logits = f.token(id, true).unwrap().unwrap();
        assert_eq!(
            f.resident_token.is_some(),
            chain,
            "{}: step {step}",
            fixture.name
        );
        assert_eq!(
            f.state.iter().any(VisitState::is_resident),
            chain,
            "{}: step {step}: rows must be device-owned exactly after a chain token",
            fixture.name
        );
        if !chain || (step + 1).is_multiple_of(every) || step + 1 == fixture.limit {
            let start = std::time::Instant::now();
            let snapshot = f.snapshot().unwrap();
            download += start.elapsed();
            assert!(!f.state.iter().any(VisitState::is_resident));
            run.snapshots.push(snapshot);
        }
        run.tokens.push(argmax(&logits));
        run.logits.push(logits);
        id = forced.map_or(run.tokens[step], |tokens| tokens[step]);
    }
    if chain {
        eprintln!(
            "{}: {} chain tokens, {} snapshots, {} ms in snapshots",
            fixture.name,
            fixture.limit,
            run.snapshots.len(),
            download.as_millis()
        );
    }
    run
}

pub(super) fn compare(fixture: &Fixture, every: usize, oracle: &Run, chain: &Run) {
    let name = fixture.name;
    // Rows first: a wrong readback explains wrong logits, the reverse does not hold.
    let known: Vec<&PrefixState> = oracle
        .snapshots
        .iter()
        .enumerate()
        .filter(|(step, _)| (step + 1).is_multiple_of(every) || step + 1 == fixture.limit)
        .map(|(_, state)| state)
        .collect();
    assert_eq!(known.len(), chain.snapshots.len(), "{name}");
    for (step, (known, read)) in known.iter().zip(&chain.snapshots).enumerate() {
        let mut one = Error::default();
        assert_eq!(known.position, read.position, "{name}");
        for (slot, (a, b)) in known.visits.iter().zip(&read.visits).enumerate() {
            assert_eq!(a.position(), b.position(), "{name}");
            let (aa, bb) = (a.shadow_planes(), b.shadow_planes());
            for (plane, (known, read)) in [
                "keys",
                "values",
                "conv k",
                "conv v",
                "conv attn",
                "conv mlp",
            ]
            .into_iter()
            .zip(aa.iter().zip(&bb))
            {
                assert_eq!(known.len(), read.len(), "{name}: {plane}: row count");
                let mut plane_error = Error::default();
                for (x, y) in known.iter().zip(read) {
                    assert_eq!(x.len(), y.len(), "{name}: {plane}: row width");
                    plane_error.rows(x, y);
                    one.rows(x, y);
                }
                // One line per plane that differs beyond rounding, to locate a wrong readback.
                if plane_error.max_abs() > 0.001 {
                    eprintln!(
                        "{name} every {every}: snapshot {step} visit {slot} {plane}: {} rows, max_abs={}",
                        known.len(),
                        plane_error.max_abs()
                    );
                }
            }
        }
        one.check(
            &format!("{name} every {every}: snapshot {step} K/V and convolution rows"),
            0.1,
            0.001,
        );
    }
    let mut logits = Error::default();
    for (step, (a, b)) in oracle.logits.iter().zip(&chain.logits).enumerate() {
        assert_eq!(argmax(a), argmax(b), "{name}: step {step}: greedy token");
        let mut step_error = Error::default();
        step_error.rows(a, b);
        eprintln!(
            "{name} every {every}: step {step} logits max_abs={}",
            step_error.max_abs()
        );
        logits.rows(a, b);
    }
    logits.check(&format!("{name} every {every}: logits"), 0.02, 0.001);
}

#[test]
#[ignore = "requires a leased CUDA device, the Inkling GGUF and FIFI_INKLING_FIXTURES"]
fn snapshots_after_chain_tokens_match_the_host_path() {
    let path = std::path::PathBuf::from(std::env::var_os("FIFI_INKLING_MODEL_DIR").unwrap());
    for fixture in fixtures() {
        let oracle = {
            // A final feature tap forces ordinary visit dispatch: the chain never admits.
            let mut f = CpuForward::open(&path, 768).unwrap();
            f.enable_cuda(0).unwrap();
            f.taps = vec![f.visits.len()];
            generate(&mut f, &fixture, None, 1)
        };
        // A prefill that follows chain tokens reads host rows too; the token after it may chain.
        let (known, known_state) = reference_after_prefill(&path, &fixture);
        let mut after_prefill = Vec::new();
        for every in [64, 7, 1] {
            let mut f = CpuForward::open(&path, 768).unwrap();
            f.enable_cuda(0).unwrap();
            let chain = generate(&mut f, &fixture, Some(&oracle.tokens), every);
            compare(&fixture, every, &oracle, &chain);
            f.prefill_grouped(&[7, 8]).unwrap();
            let saved = f.snapshot().unwrap();
            let mut rows = Error::default();
            for (a, b) in known_state.visits.iter().zip(&saved.visits) {
                assert_eq!(a.position(), b.position());
                for (x, y) in a.shadow_rows().into_iter().zip(b.shadow_rows()) {
                    rows.rows(x, y);
                }
            }
            rows.check(
                &format!(
                    "{}: rows after prefill that follows chain tokens",
                    fixture.name
                ),
                0.1,
                0.001,
            );
            after_prefill.push(f.token(9, true).unwrap().unwrap());
            if every == 64 {
                // Diagnostic: the ordinary path on the same materialized rows.
                f.restore(&saved);
                let mut host = Error::default();
                host.rows(&known, &after_prefill[0]);
                eprintln!(
                    "{}: token after prefill, chain against ordinary: max_abs={}",
                    fixture.name,
                    host.max_abs()
                );
                f.taps = vec![f.visits.len()];
                let ordinary = f.token(9, true).unwrap().unwrap();
                let mut again = Error::default();
                again.rows(&known, &ordinary);
                eprintln!(
                    "{}: token after prefill, ordinary path on the chain rows against ordinary: max_abs={}",
                    fixture.name,
                    again.max_abs()
                );
                let mut both = Error::default();
                both.rows(&after_prefill[0], &ordinary);
                eprintln!(
                    "{}: token after prefill, chain against ordinary on the same rows: max_abs={}",
                    fixture.name,
                    both.max_abs()
                );
            }
        }
        let mut error = Error::default();
        for logits in &after_prefill {
            error.rows(&known, logits);
        }
        error.check(
            &format!("{}: prefill after chain tokens", fixture.name),
            0.02,
            0.001,
        );
    }
}

/// The ordinary path over the same prompt and continuation, then the same two prefill rows.
fn reference_after_prefill(path: &std::path::Path, fixture: &Fixture) -> (Vec<f32>, PrefixState) {
    let mut f = CpuForward::open(path, 768).unwrap();
    f.enable_cuda(0).unwrap();
    f.taps = vec![f.visits.len()];
    generate(&mut f, fixture, None, 1);
    f.prefill_grouped(&[7, 8]).unwrap();
    let state = f.snapshot().unwrap();
    (f.token(9, true).unwrap().unwrap(), state)
}
