//! The device half of the feeds: the PLE key and value projections on the card, and the sink that
//! writes a token start to the device.
//!
//! WHAT: [`DevicePle`] keeps the stored F16 or BF16 `key_proj` (`width x ple_embed`) and
//! `value_proj` (`hidden x ple_embed`) on the device as they are in the pack (65 MiB, no F32
//! copy) and runs them with `CudaContext::dense_gemv`. [`DevicePle::enqueue`] writes the embedding
//! and queues the two launches on the step stream; [`DevicePle::collect`] reads the keys and the
//! value back (44 KiB), behind the queued work. [`DeviceStart`] is the [`StartSink`] over the
//! device words block and two F32 buffers: it builds the `StepParams` block (token and position)
//! and writes the initial stream and the PLE embedding, all ordered on the step stream.
//!
//! WHY the projections run here: they depend on the token only, but the token is known when the
//! step starts, so the host would have the 12,800 x 2,560 product to finish while the device runs
//! layers 0 and 1 (about 1 ms). Measured on this machine under load 13-20 (8 threads, F16 weights),
//! the host product took 1.57 ms at best and 2.6 ms at p50 (p99 7.6 ms), 4-5 ms at p50 on 1-4
//! threads, and it reads 65 MiB (131 MiB as F32) from host memory per token, which competes
//! with the pinned expert copies and keeps 8 host threads busy. The kernel reads the 65 MiB on the
//! device in well under 0.1 ms (the kernel's own measured time; the route's own time, with the copies and the
//! read-back, is not measured yet). The rest of the PLE block (norms, gate, convolution ring, about
//! 100 us) stays on the host in `ple::step`, which takes the downloaded keys and value in place of
//! its two `linear` calls.
//!
//! SYNCHRONIZATION: `DeviceBuffer::write_at` and `read_at` are pageable copies that return when the
//! copy finished, and `collect` reads behind the queued work. So [`DevicePle::project`] (and the
//! [`DevicePleRoute`] built on it) is a host-device synchronization in the middle of the step: it
//! waits for everything queued on the stream before it. A captured graph of the token therefore
//! must be cut at the PLE layer, or capture only the segments before and after it. The start sink
//! writes are the same kind of copy.

use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceStream,
        resident::{
            dense_gemv::{DenseDtype, DenseGemvCall, DenseGemvShape},
            step_params::StepParams,
        },
    },
    tensor::Dtype,
};

use super::StartSink;
use crate::family::flash_next::cuda::ple_feed::{PleKeyValue, PleProjection};
use std::rc::Rc;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

/// The PLE projections resident on the device.
pub(in crate::family) struct DevicePle {
    key_weight: DeviceBuffer,
    value_weight: DeviceBuffer,
    embedding: DeviceBuffer,
    keys: DeviceBuffer,
    value: DeviceBuffer,
    dtype: DenseDtype,
    width: usize,
    hidden: usize,
    embed: usize,
    rows_max: usize,
}

impl DevicePle {
    /// Uploads the projections as stored: `key` is `width x embed` values of `stored` and `value`
    /// is `hidden x embed`, little-endian row-major (the pack layout `[out, in]`). `stored` is the
    /// dtype of the tensors in the checkpoint; anything but F16 (the EXL3 pack) and BF16 (the
    /// official checkpoint) is refused.
    pub(in crate::family) fn upload(
        context: &CudaContext,
        stored: Dtype,
        key: &[u8],
        value: &[u8],
        width: usize,
        hidden: usize,
        embed: usize,
        rows_max: usize,
    ) -> Result<Self, EngError> {
        let dtype = match stored {
            Dtype::F16 => DenseDtype::F16,
            Dtype::Bf16 => DenseDtype::Bf16,
            other => {
                return Err(EngError::Checkpoint(format!(
                    "the PLE projections are stored as {other:?}, the device route takes F16 or BF16"
                )));
            }
        };
        let overflow = || EngError::BadRequest("ple projection sizes overflow".to_owned());
        let key_bytes = width
            .checked_mul(embed)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(overflow)?;
        let value_bytes = hidden
            .checked_mul(embed)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(overflow)?;
        if key.len() != key_bytes || value.len() != value_bytes {
            return Err(EngError::Checkpoint(format!(
                "ple projections hold {} and {} bytes, expected {key_bytes} and {value_bytes}",
                key.len(),
                value.len()
            )));
        }
        let place = |bytes: &[u8]| -> Result<DeviceBuffer, EngError> {
            let buffer = context.allocate(bytes.len())?;
            context.upload_at(&buffer, 0, bytes)?;
            Ok(buffer)
        };
        Ok(Self {
            key_weight: place(key)?,
            value_weight: place(value)?,
            embedding: context.allocate(rows_max * embed * 4)?,
            keys: context.allocate(rows_max * width * 4)?,
            value: context.allocate(rows_max * hidden * 4)?,
            dtype,
            width,
            hidden,
            embed,
            rows_max,
        })
    }

    /// Device bytes held by this object.
    pub(in crate::family) fn device_bytes(&self) -> usize {
        (self.width + self.hidden) * self.embed * 2 + (self.embed + self.width + self.hidden) * 4
    }

    /// The most rows one [`Self::enqueue_chunk`] takes.
    pub(in crate::family) fn rows_max(&self) -> usize {
        self.rows_max
    }

    /// Writes `embedding` and queues both projections on `stream`. The results stay on the device
    /// until [`Self::collect`]. `&mut self`: the buffers are shared, so a second `enqueue` before
    /// the `collect` of the first would overwrite its results.
    pub(in crate::family) fn enqueue(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
        embedding: &[f32],
    ) -> Result<(), EngError> {
        if embedding.len() != self.embed {
            return Err(EngError::BadRequest(format!(
                "the PLE embedding has {} values, expected {}",
                embedding.len(),
                self.embed
            )));
        }
        self.embedding.write_at(stream, 0, &f32_bytes(embedding))?;
        for (rows, weight, y, y_len) in [
            (self.width, &self.key_weight, &self.keys, self.width),
            (self.hidden, &self.value_weight, &self.value, self.hidden),
        ] {
            context.dense_gemv(
                stream,
                DenseGemvCall {
                    shape: DenseGemvShape::packed(1, rows, self.embed),
                    dtype: self.dtype,
                    w: weight.range(0, rows * self.embed * 2)?,
                    x: self.embedding.range(0, self.embed * 4)?,
                    y: y.range(0, y_len * 4)?,
                },
            )?;
        }
        Ok(())
    }

    /// Reads the keys and the value of the last [`Self::enqueue`], behind the queued work.
    pub(in crate::family) fn collect(
        &self,
        stream: &DeviceStream,
    ) -> Result<PleKeyValue, EngError> {
        let mut keys = vec![0u8; self.width * 4];
        let mut value = vec![0u8; self.hidden * 4];
        self.keys.read_at(stream, 0, &mut keys)?;
        self.value.read_at(stream, 0, &mut value)?;
        Ok(PleKeyValue {
            keys: f32_values(&keys),
            value: f32_values(&value),
        })
    }

    /// [`Self::enqueue`] and [`Self::collect`].
    pub(in crate::family) fn project(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
        embedding: &[f32],
    ) -> Result<PleKeyValue, EngError> {
        self.enqueue(context, stream, embedding)?;
        self.collect(stream)
    }

    /// Writes the `m` embeddings of one chunk (already scaled) and queues both projections as
    /// one `m`-row launch each, so the weights stream once for the whole chunk instead of once
    /// per row. The results stay on the device until [`Self::collect_chunk`].
    pub(in crate::family) fn enqueue_chunk(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
        embeddings: &[f32],
        m: usize,
    ) -> Result<(), EngError> {
        if embeddings.len() != m * self.embed {
            return Err(EngError::BadRequest(format!(
                "the PLE chunk holds {} embedding values for {m} rows of {}",
                embeddings.len(),
                self.embed
            )));
        }
        if m > self.rows_max() {
            return Err(EngError::BadRequest(format!(
                "a PLE chunk of {m} rows exceeds the {} the device buffers hold",
                self.rows_max()
            )));
        }
        self.embedding.write_at(stream, 0, &f32_bytes(embeddings))?;
        for (rows, weight, y, y_len) in [
            (self.width, &self.key_weight, &self.keys, m * self.width),
            (
                self.hidden,
                &self.value_weight,
                &self.value,
                m * self.hidden,
            ),
        ] {
            context.dense_gemv(
                stream,
                DenseGemvCall {
                    shape: DenseGemvShape::packed(m, rows, self.embed),
                    dtype: self.dtype,
                    w: weight.range(0, rows * self.embed * 2)?,
                    x: self.embedding.range(0, m * self.embed * 4)?,
                    y: y.range(0, y_len * 4)?,
                },
            )?;
        }
        Ok(())
    }

    /// Reads the key and value rows of the last [`Self::enqueue_chunk`], behind the queued work.
    pub(in crate::family) fn collect_chunk(
        &self,
        stream: &DeviceStream,
        m: usize,
    ) -> Result<Vec<PleKeyValue>, EngError> {
        let mut key_bytes = vec![0u8; m * self.width * 4];
        let mut value_bytes = vec![0u8; m * self.hidden * 4];
        self.keys.read_at(stream, 0, &mut key_bytes)?;
        self.value.read_at(stream, 0, &mut value_bytes)?;
        let keys = f32_values(&key_bytes);
        let values = f32_values(&value_bytes);
        Ok((0..m)
            .map(|r| PleKeyValue {
                keys: keys[r * self.width..(r + 1) * self.width].to_vec(),
                value: values[r * self.hidden..(r + 1) * self.hidden].to_vec(),
            })
            .collect())
    }
}

/// The device route of the PLE step: [`PleProjection`] over [`DevicePle`], for `PleFeed`. It owns
/// shared handles to the step's context and stream, so the projection queues behind the layers
/// already enqueued on that stream and `project` returns when its results are read back.
pub(in crate::family) struct DevicePleRoute {
    ple: DevicePle,
    context: Rc<CudaContext>,
    stream: Rc<DeviceStream>,
}

impl DevicePleRoute {
    pub(in crate::family) fn new(
        ple: DevicePle,
        context: Rc<CudaContext>,
        stream: Rc<DeviceStream>,
    ) -> Self {
        Self {
            ple,
            context,
            stream,
        }
    }
}

impl PleProjection for DevicePleRoute {
    fn project(&mut self, embedding: &[f32]) -> Result<PleKeyValue, EngError> {
        self.ple.project(&self.context, &self.stream, embedding)
    }
}

/// The token start on the device: where the Flash state keeps the step parameters, the hyper
/// stream and the PLE embedding. The offsets come from the state layout.
pub(in crate::family) struct DeviceStart<'a> {
    pub(in crate::family) stream: &'a DeviceStream,
    pub(in crate::family) words: &'a DeviceBuffer,
    pub(in crate::family) params_offset: usize,
    pub(in crate::family) hyper: &'a DeviceBuffer,
    pub(in crate::family) hyper_offset: usize,
    pub(in crate::family) ple: &'a DeviceBuffer,
    pub(in crate::family) ple_offset: usize,
}

impl StartSink for DeviceStart<'_> {
    fn write_step_params(&mut self, token: i32, pos: i32) -> Result<(), EngError> {
        let params = StepParams::new(token, pos)?;
        self.words
            .write_at(self.stream, self.params_offset, &params.to_bytes())
    }

    fn write_stream(&mut self, stream: &[f32]) -> Result<(), EngError> {
        self.hyper
            .write_at(self.stream, self.hyper_offset, &f32_bytes(stream))
    }

    fn write_ple_embedding(&mut self, embedding: &[f32]) -> Result<(), EngError> {
        self.ple
            .write_at(self.stream, self.ple_offset, &f32_bytes(embedding))
    }
}

#[cfg(test)]
mod tests;
