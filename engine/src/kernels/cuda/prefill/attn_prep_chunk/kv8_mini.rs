//! The bounded KV8 primitive composition, before family MMQ/DeltaNet integration.

use super::{BASE, D, EPS, data, le, qwen, vendored};
use crate::EngError;
use crate::kernels::cuda::prefill::attn_prep_chunk::{
    AttnPrepChunkCall, AttnPrepChunkShape, PrepChunkMode,
};
use crate::kernels::cuda::resident::{
    attn_decode::{AttnSpan, AttnSplit},
    attn_decode_q8::{AttnDecodeQ8Call, AttnQ8Shape, attn_q8_part_bytes},
    attn_prep::{AttnPrepRowsCall, RopeParams, q_bytes},
    kv8_append::{Kv8CodeScale, Kv8Data, Kv8Shape},
    step_params::{OFFSET_KV_LEN, OFFSET_POS, OFFSET_STATUS, StepParams},
};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, testkit};

const M: usize = 16;
const CHUNK: usize = 0;
const DECODE: usize = 1;
const OUTPUT: usize = 7;
const LABELS: [&str; 8] = ["q", "k", "v", "kc", "ks", "vc", "vs", "attention"];

struct Mini {
    ctx: CudaContext,
    stream: DeviceStream,
    shape: AttnPrepChunkShape,
    proj: DeviceBuffer,
    qn: DeviceBuffer,
    kn: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    params: DeviceBuffer,
    status: DeviceBuffer,
    length: DeviceBuffer,
    part: DeviceBuffer,
    out: [DeviceBuffer; 2],
    planes: [[DeviceBuffer; 4]; 2],
}

fn planes(ctx: &CudaContext, shape: Kv8Shape) -> Result<[DeviceBuffer; 4], EngError> {
    Ok([
        ctx.upload(&vec![0; shape.code_bytes()])?,
        ctx.upload(&vec![0; shape.scale_bytes()])?,
        ctx.upload(&vec![0; shape.code_bytes()])?,
        ctx.upload(&vec![0; shape.scale_bytes()])?,
    ])
}

impl Mini {
    fn new(p0: i32, cells: usize) -> Result<Self, EngError> {
        let params = StepParams::new(0, p0)?;
        let shape = qwen(M, cells, i64::from(p0));
        let host = data(&shape, 0xc0de_1616 ^ p0 as u64, 1.7);
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let kv = Kv8Shape {
            rows: M,
            n_kv_head: 2,
            head_dim: D,
            cells,
            code_scale: Kv8CodeScale::StoredHalf,
        };
        let result = Self {
            proj: ctx.upload(&le(&host.proj))?,
            qn: ctx.upload(&le(&host.q_norm))?,
            kn: ctx.upload(&le(&host.k_norm))?,
            q: ctx.allocate(q_bytes(M, shape.n_head))?,
            k: ctx.allocate(kv.row_bytes())?,
            v: ctx.allocate(kv.row_bytes())?,
            params: ctx.upload(&params.to_bytes())?,
            status: ctx.upload(&0i32.to_le_bytes())?,
            length: ctx.upload(&params.kv_len.to_le_bytes())?,
            part: ctx.allocate(attn_q8_part_bytes(M, 2, D, AttnSplit::Q8.max_splits)?)?,
            out: [
                ctx.allocate(q_bytes(M, shape.n_head))?,
                ctx.allocate(q_bytes(M, shape.n_head))?,
            ],
            planes: [planes(&ctx, kv)?, planes(&ctx, kv)?],
            ctx,
            stream,
            shape,
        };
        Ok(result)
    }

    fn kv_shape(&self, rows: usize) -> Kv8Shape {
        Kv8Shape {
            rows,
            n_kv_head: 2,
            head_dim: D,
            cells: self.shape.rows,
            code_scale: Kv8CodeScale::StoredHalf,
        }
    }

    fn kv_data(&self, route: usize, row: usize, rows: usize) -> Result<Kv8Data<'_>, EngError> {
        let bytes = self.kv_shape(1).row_bytes();
        let p = &self.planes[route];
        Ok(Kv8Data {
            k: self.k.range(row * bytes, rows * bytes)?,
            v: self.v.range(row * bytes, rows * bytes)?,
            k_codes: p[0].range(0, p[0].bytes())?,
            k_scales: p[1].range(0, p[1].bytes())?,
            v_codes: p[2].range(0, p[2].bytes())?,
            v_scales: p[3].range(0, p[3].bytes())?,
        })
    }

    fn attend(&self, route: usize, row: usize, rows: usize) -> Result<(), EngError> {
        let bytes = q_bytes(1, self.shape.n_head);
        let p = &self.planes[route];
        self.ctx.attn_decode_q8(
            &self.stream,
            AttnDecodeQ8Call {
                shape: AttnQ8Shape {
                    m: rows,
                    n_head: 16,
                    n_kv_head: 2,
                    head_dim: D,
                    cells: self.shape.rows,
                    ld: self.shape.ld,
                },
                span: AttnSpan::FULL,
                split: AttnSplit::Q8,
                scale: 0.0625,
                q: self.q.range(row * bytes, rows * bytes)?,
                proj: Some(
                    self.proj
                        .range(row * self.shape.ld * 4, rows * self.shape.ld * 4)?,
                ),
                k_codes: p[0].range(0, p[0].bytes())?,
                k_scales: p[1].range(0, p[1].bytes())?,
                v_codes: p[2].range(0, p[2].bytes())?,
                v_scales: p[3].range(0, p[3].bytes())?,
                kv_len: if route == CHUNK {
                    self.length.range(0, 4)?
                } else {
                    self.params.range(OFFSET_KV_LEN, 4)?
                },
                part: self.part.range(0, self.part.bytes())?,
                out: self.out[route].range(row * bytes, rows * bytes)?,
            },
        )
    }

    fn chunk(&self) -> Result<(), EngError> {
        self.ctx.attn_prep_chunk(
            &self.stream,
            AttnPrepChunkCall {
                shape: self.shape,
                rope: RopeParams {
                    eps: EPS,
                    rope_base: BASE,
                },
                proj: self.proj.range(0, self.proj.bytes())?,
                q_norm: self.qn.range(0, self.qn.bytes())?,
                k_norm: self.kn.range(0, self.kn.bytes())?,
                q: self.q.range(0, self.q.bytes())?,
                mode: PrepChunkMode::RowsOut {
                    k_rows: self.k.range(0, self.k.bytes())?,
                    v_rows: self.v.range(0, self.v.bytes())?,
                },
                status: Some(self.status.range(0, 4)?),
            },
        )?;
        self.ctx.kv8_chunk(
            &self.stream,
            self.kv_shape(M),
            self.kv_data(CHUNK, 0, M)?,
            self.shape.p0 as usize,
        )?;
        self.attend(CHUNK, 0, M)
    }

    fn step(&self, row: usize) -> Result<(), EngError> {
        let pos = self.shape.p0 as i32 + row as i32;
        self.params
            .write_at(&self.stream, 0, &StepParams::new(0, pos)?.to_bytes())?;
        let mut shape = vendored(&self.shape);
        shape.m = 1;
        let kv_bytes = self.kv_shape(1).row_bytes();
        self.ctx.attn_prep_rows(
            &self.stream,
            AttnPrepRowsCall {
                shape,
                rope: RopeParams {
                    eps: EPS,
                    rope_base: BASE,
                },
                admit_pos: pos as usize,
                proj: self
                    .proj
                    .range(row * self.shape.ld * 4, self.shape.ld * 4)?,
                q_norm: self.qn.range(0, self.qn.bytes())?,
                k_norm: self.kn.range(0, self.kn.bytes())?,
                q: self.q.range(row * q_bytes(1, 16), q_bytes(1, 16))?,
                k_rows: self.k.range(row * kv_bytes, kv_bytes)?,
                v_rows: self.v.range(row * kv_bytes, kv_bytes)?,
                pos: self.params.range(OFFSET_POS, 4)?,
                status: self.params.range(OFFSET_STATUS, 4)?,
            },
        )?;
        self.ctx.kv8_append(
            &self.stream,
            self.kv_shape(1),
            self.kv_data(DECODE, row, 1)?,
            self.params.range(OFFSET_POS, 4)?,
            self.params.range(OFFSET_STATUS, 4)?,
        )?;
        self.attend(DECODE, row, 1)
    }

    fn read(&self, buffer: &DeviceBuffer) -> Result<Vec<u8>, EngError> {
        let mut bytes = vec![0; buffer.bytes()];
        buffer.read_at(&self.stream, 0, &mut bytes)?;
        Ok(bytes)
    }

    fn capture(&self, route: usize) -> Result<Vec<Vec<u8>>, EngError> {
        let mut result = vec![
            self.read(&self.q)?,
            self.read(&self.k)?,
            self.read(&self.v)?,
        ];
        for plane in &self.planes[route] {
            result.push(self.read(plane)?);
        }
        result.push(self.read(&self.out[route])?);
        Ok(result)
    }
}

struct HostCase {
    p0: i32,
    chunk: Vec<Vec<u8>>,
    decode: Vec<Vec<u8>>,
    status: Vec<u8>,
    decode_status: Vec<u8>,
    wrong_length: Option<Vec<u8>>,
}

fn run_case(p0: i32, cells: usize) -> Result<HostCase, EngError> {
    let fx = Mini::new(p0, cells)?;
    fx.chunk()?;
    let chunk = fx.capture(CHUNK)?;
    for row in 0..M {
        fx.step(row)?;
    }
    let decode = fx.capture(DECODE)?;
    let mut decode_status = vec![0; 4];
    fx.params
        .read_at(&fx.stream, OFFSET_STATUS, &mut decode_status)?;
    let wrong_length = if p0 == 61 {
        // This wrong count is still in range, so only a causal comparison can reject it.
        fx.length
            .write_at(&fx.stream, 0, &(p0 + M as i32).to_le_bytes())?;
        fx.attend(CHUNK, 0, M)?;
        Some(fx.read(&fx.out[CHUNK])?)
    } else {
        None
    };
    Ok(HostCase {
        p0,
        chunk,
        decode,
        status: fx.read(&fx.status)?,
        decode_status,
        wrong_length,
    })
}

fn finite(bytes: &[u8]) -> bool {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .all(|value| f32::from_le_bytes(*value).is_finite())
}

#[test]
fn kv8_mini_chunk_matches_sixteen_decode_rows() {
    if !testkit::device_present("kv8_mini_chunk_matches_sixteen_decode_rows") {
        return;
    }
    let phase = || -> Result<Vec<HostCase>, EngError> {
        [(0, 16), (61, 128), (80, 96)]
            .into_iter()
            .map(|(p0, cells)| run_case(p0, cells))
            .collect()
    };
    let cases = phase().expect("the device phase completed and dropped every CUDA owner");
    for case in cases {
        assert_eq!(case.status, [0; 4], "chunk status at {}", case.p0);
        assert_eq!(case.decode_status, [0; 4], "decode status at {}", case.p0);
        for (index, (chunk, decode)) in case.chunk.iter().zip(&case.decode).enumerate() {
            assert!(
                chunk == decode,
                "p0 {}, {}, first differing byte {:?}",
                case.p0,
                LABELS[index],
                chunk.iter().zip(decode).position(|(a, b)| a != b)
            );
        }
        for index in [0, 1, 2, OUTPUT] {
            assert!(
                finite(&case.chunk[index]) && finite(&case.decode[index]),
                "non-finite {} at {}",
                LABELS[index],
                case.p0
            );
        }
        if let Some(wrong) = case.wrong_length {
            assert!(
                finite(&wrong),
                "the wrong length must pass the native capacity guard"
            );
            assert!(
                wrong != case.chunk[OUTPUT],
                "the causal negative control was ineffective"
            );
        }
    }
}
