use super::Exl3LinearCall;
use crate::EngError;
use crate::kernels::cuda::context::CapturePlan;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, testkit};
use crate::weights::exl3::Codebook;

struct Buffers {
    trellis: DeviceBuffer,
    scales: DeviceBuffer,
    x: DeviceBuffer,
    xh: DeviceBuffer,
    z: DeviceBuffer,
    out: DeviceBuffer,
    flag: DeviceBuffer,
    rows: usize,
}
impl Buffers {
    fn new(ctx: &CudaContext, rows: usize) -> Self {
        let mut scales = 1.0f32.to_le_bytes().repeat(128);
        scales.extend([0; 4]);
        Self {
            trellis: testkit::upload_filled(ctx, 0, 6144),
            scales: testkit::upload_bytes(ctx, &scales),
            x: testkit::upload_f32(ctx, &vec![1.0; rows * 128]),
            xh: testkit::upload_filled(ctx, 0xff, rows * 512),
            z: testkit::upload_filled(ctx, 0xff, rows * 7 * 512 + 4),
            out: testkit::upload_filled(ctx, 0xcd, rows * 512),
            flag: testkit::upload_bytes(ctx, &8u32.to_le_bytes()),
            rows,
        }
    }
    fn call(&self) -> Exl3LinearCall<'_> {
        self.call_split(2)
    }
    fn call_split(&self, split_k: u32) -> Exl3LinearCall<'_> {
        Exl3LinearCall {
            trellis: self.trellis.range(0, 6144).unwrap(),
            suh: self.scales.range(0, 512).unwrap(),
            svh: self.scales.range(0, 512).unwrap(),
            bias: None,
            x: self.x.range(0, self.rows * 512).unwrap(),
            xh: self.xh.range(0, self.rows * 512).unwrap(),
            z: self
                .z
                .range(0, self.rows * 512 * (split_k as usize - 1))
                .unwrap(),
            out: self.out.range(0, self.rows * 512).unwrap(),
            flag: self.flag.range(0, 4).unwrap(),
            rows: self.rows,
            split_k,
            k: 128,
            n: 128,
            bits_twice: 6,
            book: Codebook::Mul1,
        }
    }
}
#[test]
fn short_misaligned_and_overlapping_ranges_refuse_without_writes() {
    if !testkit::device_present("EXL3 invalid working spans") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    for rows in [1, 9, 144] {
        let project = if rows == 1 {
            CudaContext::exl3_linear
        } else {
            CudaContext::exl3_linear_midrows
        };
        let buffers = Buffers::new(&ctx, rows);
        if rows > 1 {
            for bad_rows in [0, 8, 145, usize::MAX] {
                let mut call = buffers.call();
                call.rows = bad_rows;
                assert!(matches!(
                    project(&ctx, &stream, call),
                    Err(EngError::BadRequest(_))
                ));
            }
        }
        for case in 0..11 {
            let mut call = buffers.call();
            let role = match case {
                0 => {
                    call.x = buffers.x.range(0, 511).unwrap();
                    "EXL3 x"
                }
                1 => {
                    call.suh = buffers.scales.range(1, 512).unwrap();
                    "EXL3 suh"
                }
                2 => {
                    call.out = call.x;
                    "EXL3 out"
                }
                3 => {
                    call.z = call.out;
                    "EXL3 z"
                }
                4 => {
                    call.flag = buffers.out.range(0, 4).unwrap();
                    "EXL3 flag"
                }
                5 => {
                    call.bits_twice = 9;
                    "EXL3 linear"
                }
                6 => {
                    call = buffers.call_split(4);
                    call.z = buffers.z.range(0, 512).unwrap();
                    "EXL3 z"
                }
                7 => {
                    call = buffers.call_split(8);
                    call.out = buffers.z.range(6 * 512, 512).unwrap();
                    "EXL3 out"
                }
                8 => {
                    call.split_k = 3;
                    "EXL3 linear"
                }
                9 => {
                    call = buffers.call_split(4);
                    call.flag = buffers.z.range(2 * 512 + 4, 4).unwrap();
                    "EXL3 flag"
                }
                _ => {
                    call = buffers.call_split(8);
                    call.z = buffers.z.range(1, 7 * 512).unwrap();
                    "EXL3 z"
                }
            };
            let error = project(&ctx, &stream, call).unwrap_err();
            let EngError::BadRequest(message) = error else {
                panic!("{error:?}");
            };
            assert!(message.contains(role), "case{case}: {message}");
            assert_eq!(
                testkit::read_bytes(&ctx, &buffers.out, rows * 512),
                vec![0xcd; rows * 512]
            );
            assert_eq!(
                testkit::read_bytes(&ctx, &buffers.flag, 4),
                8u32.to_le_bytes()
            );
        }
        project(&ctx, &stream, buffers.call()).unwrap();
        stream.synchronize().unwrap();
        let out = testkit::read_bytes(&ctx, &buffers.out, rows * 512);
        let expected =
            if !super::midrows_enabled(rows, std::env::var_os("FIFI_EXL3_LEGACY").as_deref()) {
                -442.0
            } else {
                // Raw1 is half-exact, but H128 produces a nonrepresentable MMA operand.
                let mut prefix = [0.0; 128];
                prefix[0] = 128.0f64.sqrt();
                let xh = testkit::read_bytes(&ctx, &buffers.xh, 512);
                let operands = super::linear_tests::rounded_mma_inputs(&xh, &prefix);
                // Zero trellis has constant W'=-442/128; the output-H DC term is sqrt128.
                operands.iter().sum::<f64>() * (-442.0 / 128.0) * prefix[0]
            };
        assert!(
            (f64::from(f32::from_le_bytes(out[..4].try_into().unwrap())) - expected).abs() < 1e-3
        );
    }
}
#[test]
fn foreign_buffer_and_stream_refuse_without_poisoning_the_context() {
    if !testkit::device_present("EXL3 foreign owners") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    for rows in [1, 9, 144] {
        let project = if rows == 1 {
            CudaContext::exl3_linear
        } else {
            CudaContext::exl3_linear_midrows
        };
        let buffers = Buffers::new(&ctx, rows);
        let other = CudaContext::new().unwrap();
        let foreign = testkit::upload_filled(&other, 0, rows * 512);
        let mut call = buffers.call();
        call.out = foreign.range(0, rows * 512).unwrap();
        assert!(matches!(
            project(&ctx, &stream, call),
            Err(EngError::BadRequest(_))
        ));
        let foreign_stream = other.stream().unwrap();
        assert!(matches!(
            project(&ctx, &foreign_stream, buffers.call()),
            Err(EngError::BadRequest(_))
        ));
        project(&ctx, &stream, buffers.call()).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(
            testkit::read_bytes(&ctx, &buffers.flag, 4),
            8u32.to_le_bytes()
        );
    }
}
#[test]
fn captured_projection_replays_the_same_three_kernels() {
    if !testkit::device_present("EXL3 capture replay") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    for rows in [1, 9, 144] {
        let project = if rows == 1 {
            CudaContext::exl3_linear
        } else {
            CudaContext::exl3_linear_midrows
        };
        let buffers = Buffers::new(&ctx, rows);
        for split_k in [2, 4, 8] {
            project(&ctx, &stream, buffers.call_split(split_k)).unwrap();
            stream.synchronize().unwrap();
            let expected = testkit::read_bytes(&ctx, &buffers.out, rows * 512);
            let pins = [
                &buffers.trellis,
                &buffers.scales,
                &buffers.x,
                &buffers.xh,
                &buffers.z,
                &buffers.out,
                &buffers.flag,
            ];
            let (graph, ()) = ctx
                .capture(
                    &stream,
                    CapturePlan {
                        pins: &pins,
                        holds: vec![],
                    },
                    || project(&ctx, &stream, buffers.call_split(split_k)),
                )
                .unwrap();
            assert_eq!(graph.nodes(), 3);
            for _ in 0..2 {
                buffers
                    .out
                    .write_at(&stream, 0, &vec![0xaa; rows * 512])
                    .unwrap();
                graph.launch(&stream).unwrap();
                stream.synchronize().unwrap();
                assert_eq!(
                    testkit::read_bytes(&ctx, &buffers.out, rows * 512),
                    expected
                );
                assert_eq!(
                    testkit::read_bytes(&ctx, &buffers.flag, 4),
                    8u32.to_le_bytes()
                );
            }
        }
    }
}
