//! Private attention health; checked public projection and FFN boundaries stay separate.

use super::super::FrameFlag;
use super::{FlintLayerView, Frame, KvPlan, Result, WIDTH, bad, exl3, producer_status};
use crate::EngError;
use crate::family::project::cuda::{Exl3MatrixView, exl3::Exl3Io};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

const QUERY: usize = 4096;
const KV: usize = 512;
const WORDS: usize = 13;
const GROUP_BYTES: usize = WORDS * 4;
const RING_BYTES: usize = 49 * GROUP_BYTES;

#[cfg(test)]
std::thread_local! {
    static TEST_OBSERVATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn test_observations() -> usize {
    TEST_OBSERVATIONS.with(std::cell::Cell::get)
}

#[derive(Clone, Copy)]
#[repr(usize)]
pub(super) enum Slot {
    Norm,
    RawQ,
    PublishQ,
    RawK,
    PublishK,
    RawV,
    PublishV,
    Producer,
    Attention,
    RawO,
    PublishO,
    Residual,
    PostNorm,
}

// Pre/post/complete share a raw producer word; completion-only tests use mask2, not mask1.
#[cfg(test)]
const CHECKPOINTS: [Slot; 21] = [
    Slot::Norm,
    Slot::RawQ,
    Slot::RawQ,
    Slot::RawQ,
    Slot::PublishQ,
    Slot::RawK,
    Slot::RawK,
    Slot::RawK,
    Slot::PublishK,
    Slot::RawV,
    Slot::RawV,
    Slot::RawV,
    Slot::PublishV,
    Slot::Producer,
    Slot::Attention,
    Slot::RawO,
    Slot::RawO,
    Slot::RawO,
    Slot::PublishO,
    Slot::Residual,
    Slot::PostNorm,
];

const SLOTS: [Slot; WORDS] = [
    Slot::Norm,
    Slot::RawQ,
    Slot::PublishQ,
    Slot::RawK,
    Slot::PublishK,
    Slot::RawV,
    Slot::PublishV,
    Slot::Producer,
    Slot::Attention,
    Slot::RawO,
    Slot::PublishO,
    Slot::Residual,
    Slot::PostNorm,
];

fn validate(bytes: &[u8; GROUP_BYTES]) -> Result<()> {
    for (word, slot) in bytes.as_chunks::<4>().0.iter().zip(SLOTS) {
        let status = u32::from_ne_bytes(*word);
        match slot {
            Slot::Producer => producer_status(status)?,
            Slot::RawQ | Slot::RawK | Slot::RawV | Slot::RawO if status & 1 != 0 => {
                return Err(EngError::ModelOutput(
                    "EXL3 projection flagged invalid arithmetic".into(),
                ));
            }
            _ => Frame::validate_health(status)?,
        }
    }
    Ok(())
}

pub(super) struct Health(DeviceBuffer);

impl Health {
    fn enabled() -> bool {
        crate::family::flint::settings::late_health_selected(
            std::env::var_os("FIFI_FLINT_LATE_HEALTH").as_deref(),
        )
    }

    fn retained_bytes(rows: usize) -> Result<u64> {
        // Includes all norm/projection/producer/CPF/residual buffers and the input row.
        let floats = 16 * WIDTH + 9 * QUERY + 10 * KV;
        rows.checked_mul(floats)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| EngError::MemoryPlan("Flint late-health retention overflows".into()))
    }

    pub(super) fn bytes(plan: &KvPlan) -> Result<u64> {
        if !Self::enabled() {
            return Ok(0);
        }
        let rows = plan
            .context_rows()
            .min(crate::family::flint::settings::PREFILL_MAX_ROWS);
        Self::retained_bytes(rows)?
            .checked_add(RING_BYTES as u64)
            .ok_or_else(|| EngError::MemoryPlan("Flint late-health charge overflows".into()))
    }

    pub(super) fn new(ctx: &CudaContext) -> Result<Option<Self>> {
        if Self::enabled() {
            Ok(Some(Self(ctx.upload(&[0; RING_BYTES])?)))
        } else {
            Ok(None)
        }
    }

    pub(super) fn clear(&self, stream: &DeviceStream) -> Result<()> {
        self.0.write_at(stream, 0, &[0; RING_BYTES])
    }

    pub(super) fn pending_error(&self, stream: &DeviceStream) -> Result<Option<EngError>> {
        let mut bytes = [0; RING_BYTES];
        self.0.read_at(stream, 0, &mut bytes)?;
        for group in bytes.as_chunks::<GROUP_BYTES>().0 {
            if let Err(error) = validate(group) {
                return Ok(Some(error));
            }
        }
        Ok(None)
    }
}

pub(super) struct Group<'c, 's> {
    pub(super) ctx: &'c CudaContext,
    pub(super) stream: &'s DeviceStream,
    words: Option<DeviceRange<'c>>,
    frames: Vec<Frame<'c, 's>>,
    buffers: Vec<DeviceBuffer>,
    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    allocation_start: Option<usize>,
    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    allocation_rows: usize,
}

impl<'c, 's> Group<'c, 's> {
    pub(super) fn new(
        ctx: &'c CudaContext,
        stream: &'s DeviceStream,
        health: Option<&'c Health>,
        layer: usize,
    ) -> Result<Self> {
        let words = health
            .map(|health| health.0.range(layer * GROUP_BYTES, GROUP_BYTES))
            .transpose()?;
        Ok(Self {
            ctx,
            stream,
            words,
            frames: Vec::new(),
            buffers: Vec::new(),
            #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
            allocation_rows: 1,
            #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
            allocation_start: health.map(|_| {
                ctx.buffer_requests()
                    .allocations
                    .iter()
                    .map(|(&bytes, &count)| bytes * count)
                    .sum()
            }),
        })
    }

    pub(super) fn deferred(&self) -> bool {
        self.words.is_some()
    }

    pub(super) fn validate_layer(&self, layer: &FlintLayerView<'_>) -> Result<()> {
        if self.deferred()
            && [layer.q.shape, layer.k.shape, layer.v.shape, layer.o.shape]
                != [[WIDTH, QUERY], [WIDTH, KV], [WIDTH, KV], [QUERY, WIDTH]]
        {
            return Err(bad("late-health attention geometry exceeds its admission"));
        }
        Ok(())
    }

    pub(super) fn frame(&self, slot: Slot) -> Result<Frame<'c, 's>> {
        match self.words {
            Some(words) => Frame::deferred(
                self.ctx,
                self.stream,
                words
                    .buffer()
                    .range(words.offset() + slot as usize * 4, 4)?,
            ),
            None => Frame::new(self.ctx, self.stream),
        }
    }

    pub(super) fn finish(
        &mut self,
        mut frame: Frame<'c, 's>,
        output: usize,
    ) -> Result<DeviceBuffer> {
        let Some(words) = self.words else {
            return frame.finish(output);
        };
        let FrameFlag::Deferred(flag) = &frame.flag else {
            return Err(bad("attention group requires its deferred frame"));
        };
        if !std::ptr::eq(flag.buffer(), words.buffer())
            || flag
                .offset()
                .checked_sub(words.offset())
                .is_none_or(|offset| offset % 4 != 0 || offset >= GROUP_BYTES)
        {
            return Err(bad("frame health belongs to another attention group"));
        }
        let output = frame.buffers.swap_remove(output).0;
        self.frames.push(frame);
        Ok(output)
    }

    pub(super) fn keep(&mut self, buffer: DeviceBuffer) {
        self.buffers.push(buffer);
    }

    pub(super) fn check(&self) -> Result<()> {
        if let Some(words) = self.words {
            #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
            {
                let allocated: usize = self
                    .ctx
                    .buffer_requests()
                    .allocations
                    .iter()
                    .map(|(&bytes, &count)| bytes * count)
                    .sum();
                let allocated = allocated - self.allocation_start.unwrap();
                let limit = Health::retained_bytes(self.allocation_rows)?;
                assert!(
                    allocated as u64 <= limit,
                    "attention allocations {allocated} exceed retention {limit}"
                );
            }
            #[cfg(test)]
            if let Some((checkpoint, status)) = self.ctx.test_take_health_fault(CHECKPOINTS.len()) {
                let slot = CHECKPOINTS[checkpoint] as usize;
                words.buffer().write_at(
                    self.stream,
                    words.offset() + slot * 4,
                    &status.to_ne_bytes(),
                )?;
            }
            let mut bytes = [0; GROUP_BYTES];
            words
                .buffer()
                .read_at(self.stream, words.offset(), &mut bytes)?;
            #[cfg(test)]
            TEST_OBSERVATIONS.with(|count| count.set(count.get() + 1));
            validate(&bytes)?;
        }
        Ok(())
    }

    pub(super) fn project(
        &mut self,
        weight: Exl3MatrixView<'_>,
        x: DeviceRange<'_>,
        slot: Slot,
    ) -> Result<DeviceBuffer> {
        if !self.deferred() {
            return exl3::project(self.ctx, self.stream, weight, x);
        }
        self.ctx.ensure_usable()?;
        let [k, n] = weight.shape;
        let input = exl3::row_shape(x.bytes(), k)?;
        #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
        {
            self.allocation_rows = input.rows;
        }
        let output = super::RowShape {
            rows: input.rows,
            columns: n,
        };
        let mut frame = self.frame(slot)?;
        let xh = frame.allocate(input)?;
        let z = frame.allocate(output)?;
        let out = frame.allocate(output)?;
        let io = Exl3Io {
            x,
            xh: frame.range(xh)?,
            z: frame.range(z)?,
            out: frame.range(out)?,
            flag: frame.flag()?,
            rows: input.rows,
        };
        self.ctx.exl3_linear(self.stream, weight.linear_call(&io))?;
        self.finish(frame, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn late_health_domains_preserve_error_provenance() {
        validate(&[0; GROUP_BYTES]).unwrap();
        for (index, slot) in SLOTS.into_iter().enumerate() {
            for status in [1_u32, 2, 8, 1 << 31] {
                let mut bytes = [0; GROUP_BYTES];
                bytes[index * 4..index * 4 + 4].copy_from_slice(&status.to_ne_bytes());
                let expected = match slot {
                    Slot::Producer => producer_status(status),
                    Slot::RawQ | Slot::RawK | Slot::RawV | Slot::RawO if status & 1 != 0 => Err(
                        EngError::ModelOutput("EXL3 projection flagged invalid arithmetic".into()),
                    ),
                    _ => Frame::validate_health(status),
                };
                assert_eq!(validate(&bytes), expected, "slot {index}, status {status}");
            }
        }
    }

    #[test]
    fn late_health_earliest_domain_wins() {
        for (index, slot) in SLOTS.into_iter().enumerate() {
            assert_eq!(slot as usize, index);
        }
        let mut bytes = [0; GROUP_BYTES];
        bytes[Slot::Norm as usize * 4..][..4].copy_from_slice(&2_u32.to_ne_bytes());
        bytes[Slot::RawQ as usize * 4..][..4].copy_from_slice(&1_u32.to_ne_bytes());
        assert_eq!(validate(&bytes), Frame::validate_health(2));
        bytes.fill(0);
        bytes[Slot::RawK as usize * 4..][..4].copy_from_slice(&1_u32.to_ne_bytes());
        bytes[Slot::Producer as usize * 4..][..4].copy_from_slice(&8_u32.to_ne_bytes());
        assert_eq!(
            validate(&bytes),
            Err(EngError::ModelOutput(
                "EXL3 projection flagged invalid arithmetic".into()
            ))
        );
    }

    #[test]
    fn late_health_charge_covers_every_retained_attention_allocation() {
        let norm = 3 * WIDTH;
        let qkv = WIDTH + 4 * QUERY + 2 * (WIDTH + 4 * KV);
        let producer = 2 * (QUERY + KV);
        let cpf = 2 * QUERY;
        let output = QUERY + 4 * WIDTH;
        let residual = 2 * WIDTH;
        let postnorm = 3 * WIDTH;
        let floats = WIDTH + norm + qkv + producer + cpf + output + residual + postnorm;
        for rows in 1..=8 {
            assert_eq!(
                Health::retained_bytes(rows).unwrap(),
                (rows * floats * 4) as u64
            );
        }
        assert_eq!(RING_BYTES, 2548);
        assert!(Health::retained_bytes(usize::MAX).is_err());
    }

    #[test]
    #[ignore = "requires admitted GPU; late-health ON arm pinned by GPU runner; real projection producer"]
    fn late_health_real_projection_bit1_keeps_arithmetic_provenance() {
        use super::super::super::exl3_dispatch_tests::{upload_f32, weight};
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let health = Health::new(&ctx).unwrap().expect("late-health ON required");
        let weight = weight(&ctx, [256, 128], 0.25);
        let input = upload_f32(&ctx, &[f32::NAN; 256]);
        let mut group = Group::new(&ctx, &stream, Some(&health), 0).unwrap();
        let _published = super::super::projection(
            &mut group,
            weight.view().unwrap(),
            &input,
            None,
            "real_fault.q_proj",
            [Slot::RawQ, Slot::PublishQ],
        )
        .unwrap();
        let mut word = [0; 4];
        health
            .0
            .read_at(&stream, Slot::RawQ as usize * 4, &mut word)
            .unwrap();
        assert_ne!(u32::from_ne_bytes(word) & 1, 0);
        assert!(!ctx.test_health_fault_pending());
        let expected =
            || EngError::ModelOutput("EXL3 projection flagged invalid arithmetic".into());
        assert_eq!(group.check(), Err(expected()));
        assert_eq!(
            super::super::abort_error(
                super::super::f16_row(&[f32::NAN; 512]).unwrap_err(),
                health.pending_error(&stream),
            ),
            expected(),
        );
        ctx.ensure_usable().unwrap();
    }

    #[test]
    #[ignore = "requires admitted GPU; late-health ON arm pinned by GPU runner; real Q/K producer"]
    fn late_health_real_producer_fault_keeps_backend_provenance() {
        use super::super::super::exl3_dispatch_tests::upload_f32;
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let health = Health::new(&ctx).unwrap().expect("late-health ON required");
        let q = upload_f32(&ctx, &[1.0; QUERY]);
        let k = upload_f32(&ctx, &[1.0; KV]);
        let gamma_q = upload_f32(&ctx, &[f32::INFINITY; 128]);
        let gamma_k = upload_f32(&ctx, &[1.0; 128]);
        let mut group = Group::new(&ctx, &stream, Some(&health), 0).unwrap();
        let _outputs = super::super::prepare_qk(
            &mut group,
            [&q, &k],
            [
                gamma_q.range(0, 512).unwrap(),
                gamma_k.range(0, 512).unwrap(),
            ],
            None,
        )
        .unwrap();
        let mut word = [0; 4];
        health
            .0
            .read_at(&stream, Slot::Producer as usize * 4, &mut word)
            .unwrap();
        let status = u32::from_ne_bytes(word);
        assert_ne!(status & 8, 0);
        assert!(!ctx.test_health_fault_pending());
        assert_eq!(group.check(), producer_status(status));
        assert_eq!(
            super::super::abort_error(
                super::super::f16_row(&[f32::NAN; 512]).unwrap_err(),
                health.pending_error(&stream),
            ),
            producer_status(status).unwrap_err(),
        );
        health.clear(&stream).unwrap();
        assert_eq!(health.pending_error(&stream).unwrap(), None);
        ctx.ensure_usable().unwrap();
    }

    #[test]
    #[ignore = "requires admitted GPU; private frame ownership boundary"]
    fn late_health_group_rejects_checked_and_foreign_frames() {
        let ctx = CudaContext::new().expect("admitted CUDA device required");
        let stream = ctx.stream().unwrap();
        let health = Health(ctx.upload(&[0; RING_BYTES]).unwrap());
        let other = Health(ctx.upload(&[0; RING_BYTES]).unwrap());
        let mut group = Group::new(&ctx, &stream, Some(&health), 0).unwrap();
        let before = ctx.scalar_read_requests();
        let mut standalone = group.frame(Slot::Norm).unwrap();
        let output = standalone
            .allocate(super::super::RowShape {
                rows: 1,
                columns: 1,
            })
            .unwrap();
        assert!(matches!(
            standalone.finish(output),
            Err(EngError::BadRequest(_))
        ));
        let mut checked = Frame::new(&ctx, &stream).unwrap();
        let output = checked
            .allocate(super::super::RowShape {
                rows: 1,
                columns: 1,
            })
            .unwrap();
        assert!(matches!(
            group.finish(checked, output),
            Err(EngError::BadRequest(_))
        ));
        for (owner, layer) in [(&health, 1), (&other, 0)] {
            let foreign = Group::new(&ctx, &stream, Some(owner), layer).unwrap();
            let mut frame = foreign.frame(Slot::Norm).unwrap();
            let output = frame
                .allocate(super::super::RowShape {
                    rows: 1,
                    columns: 1,
                })
                .unwrap();
            assert!(matches!(
                group.finish(frame, output),
                Err(EngError::BadRequest(_))
            ));
        }
        let mut frame = group.frame(Slot::Norm).unwrap();
        let output = frame
            .allocate(super::super::RowShape {
                rows: 1,
                columns: 1,
            })
            .unwrap();
        let output = group.finish(frame, output).unwrap();
        group.check().unwrap();
        assert_eq!(ctx.scalar_read_requests(), before);
        drop(output);
    }
}
