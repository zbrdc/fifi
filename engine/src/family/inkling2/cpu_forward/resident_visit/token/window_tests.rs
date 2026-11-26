//! The sliding-window boundary of the whole-token chain. With a context past 1,024 rows the
//! local K/V rings hold `min(context, sliding_window_size)` rows and wrap while the chain owns
//! them; the host path drops the oldest row at the same positions (`VisitState::push`). Every
//! token from the row that fills the ring to the row that overwrites its first slot is compared
//! with the host-orchestrated path: K/V rows, convolution histories (through a snapshot each
//! token) and logits. Requires `FIFI_INKLING_CUDA_RESIDENT_TOKEN=1`; skipped otherwise.
use super::materialize_tests::{Fixture, compare, generate};
use super::*;

const WINDOW: usize = 1024;
/// Tokens decoded after the prompt: eight fill the ring (positions 1016..1023), then it wraps.
const STEPS: usize = 36;

#[test]
#[ignore = "requires a leased CUDA device, the Inkling GGUF and FIFI_INKLING_CUDA_RESIDENT_TOKEN=1"]
fn chain_tokens_across_the_attention_window_match_the_host_path() {
    assert!(
        crate::family::inkling2::switches::Switch::ResidentToken.on(),
        "window test requires the resident-token chain"
    );
    let path = std::path::PathBuf::from(std::env::var_os("FIFI_INKLING_MODEL_DIR").unwrap());
    let vocab = 200_058;
    let mut seed = 7_u32;
    let prompt = (0..WINDOW - 7)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) % vocab
        })
        .collect::<Vec<_>>();
    let fixture = Fixture {
        name: "window_1024",
        prompt,
        limit: STEPS,
    };
    let context = WINDOW - 7 + STEPS;
    let oracle = {
        // A final feature tap forces ordinary visit dispatch: the chain never admits.
        let mut f = CpuForward::open(&path, context).unwrap();
        f.enable_cuda(0).unwrap();
        f.taps = vec![f.visits.len()];
        generate(&mut f, &fixture, None, 1)
    };
    let mut f = CpuForward::open(&path, context).unwrap();
    f.enable_cuda(0).unwrap();
    let chain = generate(&mut f, &fixture, Some(&oracle.tokens), 1);
    assert!(f.position > WINDOW, "position {}", f.position);
    // The ring of every visit holds exactly the window once it has wrapped, as on the host.
    let local = f
        .visits
        .iter()
        .find(|v| v.attention == Attention::Local)
        .unwrap();
    assert_eq!(
        f.state[local.state_slot].shadow_planes()[0].len(),
        WINDOW.min(f.state[local.state_slot].position())
    );
    compare(&fixture, 1, &oracle, &chain);
}
