//! Property tests of the device sampler. No full row is read back to the host.
use super::*;
use crate::kernels::cuda::{CudaContext, testkit};

fn settings(
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    seed: u64,
    pos: u64,
) -> SampleParams {
    SampleParams {
        temperature,
        top_k,
        top_p,
        min_p,
        seed,
        position: pos,
    }
}

/// Draws once and returns the token id.
fn draw(ctx: &CudaContext, stream: &DeviceStream, logits: &[f32], params: SampleParams) -> u32 {
    let n = logits.len();
    let l = testkit::upload_f32(ctx, logits);
    let probs = testkit::upload_filled(ctx, 0, n * 4);
    let token = testkit::upload_filled(ctx, 0, 4);
    let logprob = testkit::upload_filled(ctx, 0, 4);
    let status = testkit::upload_filled(ctx, 0, 4);
    ctx.sample_gemma(
        stream,
        SampleGemmaCall {
            token: token.range(0, 4).expect("token"),
            logprob: logprob.range(0, 4).expect("logprob"),
            status: status.range(0, 4).expect("status"),
            probs: probs.range(0, n * 4).expect("probs"),
            logits: l.range(0, n * 4).expect("logits"),
            n,
            params,
        },
    )
    .expect("sample");
    stream.synchronize().expect("drain");
    let bytes = testkit::read_bytes(ctx, &token, 4);
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// A zero temperature takes the argmax, with the lowest id on a tie.
#[test]
fn zero_temperature_takes_the_lowest_argmax() {
    if !testkit::device_present("sampler greedy") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut logits = vec![0.0f32; 64];
    logits[9] = 5.0;
    logits[41] = 5.0;
    assert_eq!(
        draw(&ctx, &stream, &logits, settings(0.0, 0, 1.0, 0.0, 7, 0)),
        9
    );
}

/// top_k 1 always takes the argmax when sampling.
#[test]
fn top_k_one_takes_the_argmax_for_every_position() {
    if !testkit::device_present("sampler top_k one") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut logits = vec![0.0f32; 64];
    logits[17] = 3.0;
    for pos in 0..8u64 {
        assert_eq!(
            draw(&ctx, &stream, &logits, settings(1.0, 1, 1.0, 0.0, 11, pos)),
            17
        );
    }
}

/// The same seed and position give the same token; distinct positions vary.
#[test]
fn draws_repeat_by_position_and_vary_across_positions() {
    if !testkit::device_present("sampler determinism") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let logits = vec![0.0f32; 64];
    let first = draw(&ctx, &stream, &logits, settings(1.0, 0, 1.0, 0.0, 3, 5));
    let again = draw(&ctx, &stream, &logits, settings(1.0, 0, 1.0, 0.0, 3, 5));
    assert_eq!(first, again, "same seed and position");
    let mut distinct = std::collections::BTreeSet::new();
    for pos in 0..64u64 {
        distinct.insert(draw(
            &ctx,
            &stream,
            &logits,
            settings(1.0, 0, 1.0, 0.0, 3, pos),
        ));
    }
    assert!(
        distinct.len() > 8,
        "only {} distinct tokens",
        distinct.len()
    );
}

/// A peaked distribution lands on the dominant token at every position.
#[test]
fn a_peaked_distribution_selects_the_dominant_token() {
    if !testkit::device_present("sampler peaked") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut logits = vec![-40.0f32; 64];
    logits[33] = 40.0;
    for pos in 0..16u64 {
        assert_eq!(
            draw(&ctx, &stream, &logits, settings(1.0, 0, 0.95, 0.0, 5, pos)),
            33
        );
    }
}

/// The wrapper refuses top_p outside (0, 1] before any native call.
#[test]
fn refuses_a_top_p_of_zero() {
    if !testkit::device_present("sampler refusal") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let n = 64;
    let l = testkit::upload_f32(&ctx, &vec![0.0f32; n]);
    let probs = testkit::upload_filled(&ctx, 0, n * 4);
    let token = testkit::upload_filled(&ctx, 0, 4);
    let logprob = testkit::upload_filled(&ctx, 0, 4);
    let status = testkit::upload_filled(&ctx, 0, 4);
    let err = ctx
        .sample_gemma(
            &stream,
            SampleGemmaCall {
                token: token.range(0, 4).expect("token"),
                logprob: logprob.range(0, 4).expect("logprob"),
                status: status.range(0, 4).expect("status"),
                probs: probs.range(0, n * 4).expect("probs"),
                logits: l.range(0, n * 4).expect("logits"),
                n,
                params: settings(1.0, 0, 0.0, 0.0, 1, 0),
            },
        )
        .expect_err("top_p refused");
    assert!(matches!(err, crate::EngError::BadRequest(_)), "{err:?}");
}

#[test]
fn a_zero_uniform_skips_excluded_chunks() {
    if !testkit::device_present("sampler zero uniform") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut logits = [-30.0; 32];
    logits[31] = 30.0;
    assert_eq!(
        draw(
            &ctx,
            &stream,
            &logits,
            settings(1.0, 1, 1.0, 0.0, 6_786_339, 1)
        ),
        31
    );
}

#[test]
fn top_k_one_breaks_equal_logits_by_lowest_index() {
    if !testkit::device_present("sampler tied top_k") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    assert_eq!(
        draw(&ctx, &stream, &[0.0; 32], settings(1.0, 1, 1.0, 0.0, 3, 5)),
        0
    );
}

#[test]
fn top_p_uses_the_mass_remaining_after_top_k() {
    if !testkit::device_present("sampler composed top_k top_p") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut logits = [-100.0; 32];
    for (index, weight) in [1.0_f32, 0.75, 0.5, 0.25].into_iter().enumerate() {
        logits[index] = weight.ln();
    }
    for position in 0..16 {
        assert_eq!(
            draw(
                &ctx,
                &stream,
                &logits,
                settings(1.0, 2, 0.5, 0.0, 3, position)
            ),
            0,
            "position {position}"
        );
    }
}
