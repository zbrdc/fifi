//! Reuse a captured real layer-2 post-norm input; this is not a model qualification.

use super::*;
use crate::expert_tier::{CudaStreamBanks, ExpertId, ExpertPayload, ExpertPayloadSource};
use crate::family::flint::cuda_ffn::device::streaming;
use std::collections::BTreeSet;

struct Traced<S> {
    source: S,
    fetched: BTreeSet<ExpertId>,
}

impl<S: ExpertPayloadSource> ExpertPayloadSource for Traced<S> {
    fn parts(&self) -> usize {
        self.source.parts()
    }

    fn payload(&mut self, id: ExpertId, part: usize) -> Result<ExpertPayload, EngError> {
        self.fetched.insert(id);
        self.source.payload(id, part)
    }

    fn release_experts(&mut self, ids: &[ExpertId]) -> Result<(), EngError> {
        self.source.release_experts(ids)
    }

    fn release_layer(&mut self, _: u32) -> Result<(), EngError> {
        panic!("fetch must not release a whole layer containing hot experts")
    }
}

fn real_post_norm() -> Vec<f32> {
    let path = std::env::var("FIFI_FLINT_FETCH_INPUT").expect("frozen real L2 p03 capture");
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "51fd72e70291c8e08e33f1aafbf78069579c1c9a0b87cab553a031a9bafb638e"
    );
    let capture: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(capture["model_sha256"], gguf::MODEL_SHA256);
    assert_eq!(
        capture["oracle_sha256"],
        "db8399a9a5a23c35c7921823e93c655feb6098185667d8cc7ede325a4e0d3e18"
    );
    assert_eq!(
        capture["prefixes_sha256"],
        "2b6f2c43d5a66a85c5feb53d541e2ceda086e2563a8e9ab00e4a64fbada1e814"
    );
    assert_eq!(capture["layer"], 2);
    assert_eq!(capture["rows"], 5);
    assert_eq!(capture["instrumented_matches_untraced"], true);
    assert_eq!(capture["diagnostic_only"], true);
    let input = capture["post_norm_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            let bits = u32::try_from(v.as_u64().unwrap()).unwrap();
            assert_eq!(bits & 0xffff, 0);
            let value = f32::from_bits(bits);
            assert!(value.is_finite());
            value
        })
        .collect::<Vec<_>>();
    assert_eq!(input.len(), 5 * WIDTH);
    input
}

#[test]
#[ignore = "requires real Flint GGUF, frozen native L2 p03 input and CUDA"]
fn real_layer_2_mixed_host_fetch_is_bit_exact_to_all_hot() {
    const M: usize = 5;
    let input = real_post_norm();
    let identity_path = std::env::var("FIFI_FLINT_MODEL_IDENTITY").unwrap();
    let identity = serde_json::from_reader(File::open(identity_path).unwrap()).unwrap();
    let file = File::open(std::env::var("FIFI_FLINT_GGUF").unwrap()).unwrap();
    gguf::verify_prepared_identity(&file, &identity).unwrap();
    let header = GgufHeader::parse(
        &mut BufReader::new(&file),
        file.metadata().unwrap().len(),
        64 << 20,
    )
    .unwrap();
    let ctx = CudaContext::new().expect("real Flint fetch requires CUDA");
    let stream = ctx.stream().unwrap();
    let input = ctx.upload(&f32_bytes(&input)).unwrap();
    let Ffn::Moe {
        router,
        experts,
        shared,
    } = Ffn::load(&ctx, &file, &header, 2).unwrap()
    else {
        panic!("real layer 2 must be MoE");
    };
    let (expected, ids) = {
        let views = experts
            .iter()
            .map(|e| e.view(&ctx).unwrap())
            .collect::<Vec<_>>();
        let weights = FfnWeights::Moe {
            router: router.range(0, router.bytes()).unwrap(),
            experts: &views,
            shared: shared.view(&ctx).unwrap(),
        };
        let block = weights.block(M).unwrap();
        let mut frame = Frame::new(&ctx, &stream).unwrap();
        let x = frame
            .input(
                input.range(0, input.bytes()).unwrap(),
                RowShape {
                    rows: M,
                    columns: WIDTH,
                },
            )
            .unwrap();
        let routed = route(
            &mut frame,
            router.range(0, router.bytes()).unwrap(),
            x,
            block,
        )
        .unwrap();
        let ids = routed.ids.clone();
        let output = moe_routed(frame, x, block, &weights, routed).unwrap();
        (
            values(&stream, &output)
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            ids,
        )
    };
    let hot_id = ids[0] as u32;
    let cold = ids
        .iter()
        .copied()
        .filter(|&id| id as u32 != hot_id)
        .map(|expert| ExpertId {
            layer: 2,
            expert: expert as u32,
        })
        .collect::<BTreeSet<_>>();
    assert!(!cold.is_empty());
    // Consume the all-hot bank; every cold GPU owner is dropped before fetching.
    let hot = experts
        .into_iter()
        .enumerate()
        .filter_map(|(id, expert)| (id as u32 == hot_id).then_some((id as u32, expert)))
        .collect::<Vec<_>>();
    assert_eq!(hot.len(), 1);
    let views = hot
        .iter()
        .map(|(id, e)| (*id, e.view(&ctx).unwrap()))
        .collect::<Vec<_>>();
    let mut banks = CudaStreamBanks::new(&ctx, streaming::plan(M).unwrap()).unwrap();
    let mut source = Traced {
        source: streaming::GgufSource::new(&file, &header).unwrap(),
        fetched: BTreeSet::new(),
    };
    let output = streaming::forward(
        &ctx,
        &stream,
        streaming::MoeCall {
            layer: 2,
            rows: M,
            input: input.range(0, input.bytes()).unwrap(),
            router: router.range(0, router.bytes()).unwrap(),
            shared: shared.view(&ctx).unwrap(),
            hot: &views,
            source: &mut source,
            banks: &mut banks,
        },
    )
    .unwrap();
    let actual = values(&stream, &output)
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect::<Vec<_>>();
    assert_eq!(
        actual, expected,
        "mixed fetch changed original-rank BF16 FFN output"
    );
    assert_eq!(
        source.fetched, cold,
        "only cold consumed IDs may be fetched"
    );
    gguf::verify_prepared_identity(&file, &identity).unwrap();
    println!(
        "FLINT_FETCH_REAL layer=2 rows={M} hot=1 cold={} bit_exact=true",
        cold.len()
    );
}
