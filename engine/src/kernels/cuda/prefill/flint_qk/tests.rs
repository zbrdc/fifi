use super::*;
use crate::kernels::cuda::{DeviceBuffer, testkit};
use sha2::{Digest, Sha256};

struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
    q: DeviceBuffer,
    k: DeviceBuffer,
    gamma: [DeviceBuffer; 2],
    out: [DeviceBuffer; 2],
    scratch: [DeviceBuffer; 2],
    flag: DeviceBuffer,
    rows: usize,
}

impl Rig {
    fn new(q: &[f32], k: &[f32], gamma: [&[f32]; 2]) -> Self {
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let out = std::array::from_fn(|i| {
            testkit::upload_filled(&ctx, 0xa5, [q.len(), k.len()][i] * 4 + 32)
        });
        let scratch = std::array::from_fn(|i| {
            testkit::upload_filled(&ctx, 0xa5, [q.len(), k.len()][i] * 4 + 32)
        });
        Self {
            stream,
            rows: q.len() / 4096,
            q: testkit::upload_f32(&ctx, q),
            k: testkit::upload_f32(&ctx, k),
            gamma: gamma.map(|values| testkit::upload_f32(&ctx, values)),
            out,
            scratch,
            flag: ctx.upload(&2u32.to_ne_bytes()).unwrap(),
            ctx,
        }
    }

    fn call(&self, position: Option<u32>) -> QkProducerCall<'_> {
        let [q, k] = [self.rows * 16384, self.rows * 2048];
        QkProducerCall {
            q: self.q.range(0, q).unwrap(),
            k: self.k.range(0, k).unwrap(),
            gamma: self.gamma.each_ref().map(|b| b.range(0, 512).unwrap()),
            out: [
                self.out[0].range(16, q).unwrap(),
                self.out[1].range(16, k).unwrap(),
            ],
            scratch: [
                self.scratch[0].range(16, q).unwrap(),
                self.scratch[1].range(16, k).unwrap(),
            ],
            flag: self.flag.range(0, 4).unwrap(),
            rows: self.rows,
            position,
        }
    }

    fn run(&self, position: Option<u32>) {
        self.ctx
            .flint_qk_producer(&self.stream, self.call(position))
            .unwrap();
        self.stream.synchronize().unwrap();
        self.ctx.ensure_usable().unwrap();
    }

    fn status(&self) -> u32 {
        u32::from_ne_bytes(
            testkit::read_bytes(&self.ctx, &self.flag, 4)
                .try_into()
                .unwrap(),
        )
    }

    fn output(&self, index: usize) -> Vec<u8> {
        let bytes = self.rows * [16384, 2048][index];
        let all = testkit::read_bytes(&self.ctx, &self.out[index], bytes + 32);
        assert_eq!(&all[..16], &[0xa5; 16]);
        assert_eq!(&all[bytes + 16..], &[0xa5; 16]);
        let scratch = testkit::read_bytes(&self.ctx, &self.scratch[index], bytes + 32);
        assert_eq!(&scratch[..16], &[0xa5; 16]);
        assert_eq!(&scratch[bytes + 16..], &[0xa5; 16]);
        all[16..bytes + 16].to_vec()
    }

    fn assert_unwritten(&self) {
        for pair in [&self.out, &self.scratch] {
            for (i, buffer) in pair.iter().enumerate() {
                let bytes = self.rows * [16384, 2048][i] + 32;
                assert!(
                    testkit::read_bytes(&self.ctx, buffer, bytes)
                        .iter()
                        .all(|&v| v == 0xa5)
                );
            }
        }
        assert_eq!(self.status(), 2);
    }
}

#[test]
fn safe_refusals_leave_outputs_and_flag_untouched() {
    if !testkit::device_present("D128 Q/K safe preflight") {
        return;
    }
    let rig = Rig::new(&vec![1.0; 4 * 4096], &vec![1.0; 4 * 512], [&[1.0; 128]; 2]);
    let foreign = CudaContext::new().unwrap();
    let foreign_q = testkit::upload_f32(&foreign, &vec![1.0; 4 * 4096]);
    let foreign_stream = foreign.stream().unwrap();
    for which in 0..15 {
        let mut call = rig.call(None);
        match which {
            0 => call.rows = 0,
            1 => call.rows = 4097,
            2 => call.rows = usize::MAX,
            3 => call.position = Some(u32::MAX),
            4 => call.out[0] = rig.out[0].range(17, 4 * 16384).unwrap(),
            5 => call.out[0] = rig.out[0].range(16, 4 * 16384 - 4).unwrap(),
            6 => call.k = rig.k.range(0, 4 * 2048 - 4).unwrap(),
            7 => call.gamma[1] = rig.gamma[1].range(0, 508).unwrap(),
            8 => call.flag = rig.flag.range(0, 3).unwrap(),
            9 => call.out[0] = call.q,
            10 => call.scratch[0] = call.out[0],
            11 => call.out[1] = rig.out[0].range(16, 4 * 2048).unwrap(),
            12 => call.flag = rig.gamma[0].range(0, 4).unwrap(),
            13 => call.q = foreign_q.range(0, 4 * 16384).unwrap(),
            14 => call.scratch[1] = call.k,
            _ => unreachable!(),
        }
        assert!(
            matches!(
                rig.ctx.flint_qk_producer(&rig.stream, call),
                Err(EngError::BadRequest(_))
            ),
            "case {which}"
        );
    }
    assert!(matches!(
        rig.ctx.flint_qk_producer(&foreign_stream, rig.call(None)),
        Err(EngError::BadRequest(_))
    ));
    rig.stream.synchronize().unwrap();
    rig.assert_unwritten();
    rig.ctx.ensure_usable().unwrap();
}

#[test]
fn analytic_norm_rope_and_read_only_gamma_alias() {
    if !testkit::device_present("D128 Q/K publication order") {
        return;
    }
    let q: Vec<_> = (0..2 * 4096)
        .map(|i| if i % 128 == 0 { 1.0 } else { 0.0 })
        .collect();
    let k: Vec<_> = (0..2 * 512)
        .map(|i| if i % 128 == 0 { 1.0 } else { 0.0 })
        .collect();
    let rig = Rig::new(&q, &k, [&[1.0; 128]; 2]);
    let mut call = rig.call(None);
    call.gamma[1] = call.gamma[0];
    rig.ctx.flint_qk_producer(&rig.stream, call).unwrap();
    rig.stream.synchronize().unwrap();
    for i in 0..2 {
        let values = testkit::floats(&rig.output(i));
        for head in values.as_chunks::<128>().0 {
            assert_eq!(head[0], 11.3125);
            assert!(head[1..].iter().all(|&x| x == 0.0));
        }
    }
    rig.run(Some(0));
    for i in 0..2 {
        let values = testkit::floats(&rig.output(i));
        let heads = [32, 4][i];
        for head in values[heads * 128..].as_chunks::<128>().0 {
            assert_eq!(head[0], 6.125);
            assert_eq!(head[64], 9.5);
        }
    }
    assert_eq!(rig.status(), 2);
    assert_eq!(
        testkit::floats(&testkit::read_bytes(&rig.ctx, &rig.q, q.len() * 4)),
        q
    );
    assert_eq!(
        testkit::floats(&testkit::read_bytes(&rig.ctx, &rig.k, k.len() * 4)),
        k
    );
}

#[test]
fn numeric_mask8_fences_norm_and_rope_before_shared_publication() {
    if !testkit::device_present("D128 Q/K numeric fence") {
        return;
    }
    let base = vec![1048576.0; 4096];
    let key = vec![1048576.0; 512];
    for which in 0..6 {
        let mut q = base.clone();
        let mut gamma = [1.0; 128];
        let mut gamma_k = [1.0; 128];
        let position = if which == 4 { Some(1) } else { None };
        match which {
            0 => q[0] = f32::NAN,
            1 => gamma[0] = f32::INFINITY,
            2 => gamma.fill(f32::MAX),
            3 => gamma.fill(f32::from_bits(0x7f7f8000)),
            4 => gamma.fill(f32::from_bits(0x7f7f0000)),
            5 => gamma_k[0] = f32::INFINITY,
            _ => unreachable!(),
        }
        let rig = Rig::new(&q, &key, [&gamma, &gamma_k]);
        rig.run(position);
        assert_eq!(
            rig.status(),
            10,
            "case{which}: numeric8, preserved2, no publication1"
        );
        for i in 0..2 {
            assert!(
                testkit::floats(&rig.output(i))
                    .iter()
                    .all(|v| v.is_finite())
            );
        }
    }
    let gamma = [f32::from_bits(0x7f7f7fff); 128];
    let rig = Rig::new(&base, &key, [&gamma, &[1.0; 128]]);
    rig.run(None);
    assert_eq!(rig.status(), 2);
    assert!(
        testkit::floats(&rig.output(0))
            .iter()
            .all(|v| v.to_bits() == 0x7f7f0000)
    );
}

#[derive(serde::Deserialize)]
struct GoldenCase {
    rows: usize,
    position: Option<u32>,
    artifacts: Vec<Artifact>,
}

#[derive(serde::Deserialize)]
struct Artifact {
    name: String,
    file: String,
    f32_values: usize,
    sha256: String,
}

fn qk_output_names_are_complete(names: &[&str]) -> bool {
    matches!(names, ["q", "k"] | ["k", "q"])
}

#[test]
fn frozen_artifacts_require_exactly_one_q_and_one_k() {
    assert!(qk_output_names_are_complete(&["q", "k"]));
    assert!(qk_output_names_are_complete(&["k", "q"]));
    for names in [
        &[][..],
        &["q"],
        &["k"],
        &["q", "q"],
        &["k", "k"],
        &["q", "v"],
        &["q", "k", "q"],
    ] {
        assert!(!qk_output_names_are_complete(names), "{names:?}");
    }
}

#[test]
#[ignore = "explicit frozen independent FP64 golden qualification; requires GPU and FIFI_FLINT_QK_GOLDENS"]
fn frozen_fp64_goldens_match_all_words_including_signed_zero() {
    assert!(testkit::device_present("D128 Q/K frozen FP64 oracle"));
    let directory = std::path::PathBuf::from(std::env::var("FIFI_FLINT_QK_GOLDENS").unwrap());
    let manifest_bytes = std::fs::read(directory.join("manifest.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&manifest_bytes)),
        "00414e9a1857762e3c3d64f3ad55ba750468e2585a25aa095767d1d1a48489a9"
    );
    let manifest: Vec<GoldenCase> = serde_json::from_slice(&manifest_bytes).unwrap();
    assert_eq!(manifest.len(), 11);
    let cases: std::collections::BTreeSet<_> =
        manifest.iter().map(|c| (c.rows, c.position)).collect();
    // shortcut: goldens stop at position 4095; add owner cases before claiming higher-position bitwise parity.
    let wanted: std::collections::BTreeSet<_> = [1, 2, 4, 8, 4096]
        .into_iter()
        .flat_map(|rows| [(rows, None), (rows, Some(0))])
        .chain([(4, Some(37))])
        .collect();
    assert_eq!(cases, wanted);
    let gamma_q: [f32; 128] =
        std::array::from_fn(|i| f32::from_bits(0x3f700001 + ((i as u32 % 17) << 12)));
    let gamma_k: [f32; 128] =
        std::array::from_fn(|i| f32::from_bits(0x3f800003 + ((i as u32 % 11) << 10)));
    for case in manifest {
        let q: Vec<_> = (0..case.rows * 4096)
            .map(|i| {
                if i % 67 == 0 {
                    -0.0
                } else {
                    ((i % 29) as i32 - 14) as f32 / 8.0
                }
            })
            .collect();
        let k: Vec<_> = (0..case.rows * 512)
            .map(|i| {
                if i % 71 == 0 {
                    -0.0
                } else {
                    ((i % 23) as i32 - 11) as f32 / 16.0
                }
            })
            .collect();
        let rig = Rig::new(&q, &k, [&gamma_q, &gamma_k]);
        rig.run(case.position);
        assert_eq!(rig.status(), 2);
        let names: Vec<_> = case.artifacts.iter().map(|a| a.name.as_str()).collect();
        assert!(qk_output_names_are_complete(&names));
        for artifact in case.artifacts {
            let index = match artifact.name.as_str() {
                "q" => 0,
                "k" => 1,
                _ => panic!("unexpected artifact"),
            };
            assert_eq!(artifact.f32_values, case.rows * [4096, 512][index]);
            let relative = std::path::Path::new(&artifact.file);
            assert!(relative.components().count() == 1 && !relative.is_absolute());
            let expected = std::fs::read(directory.join(relative)).unwrap();
            assert_eq!(expected.len(), artifact.f32_values * 4);
            assert_eq!(format!("{:x}", Sha256::digest(&expected)), artifact.sha256);
            drop(expected);
            assert_eq!(
                format!("{:x}", Sha256::digest(rig.output(index))),
                artifact.sha256,
                "rows={} pos={:?} {}",
                case.rows,
                case.position,
                artifact.name
            );
        }
    }
}
