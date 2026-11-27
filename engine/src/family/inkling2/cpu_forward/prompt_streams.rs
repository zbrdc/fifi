//! The depth streams of a prompt's rows, kept on the device between visits.
//!
//! Each row's state is its current row and then its sealed rows (each stored RMS-normalized, as
//! the host [`Stream`] stores them), `1 + capacity` rows apart. Every row of a prompt crosses the
//! same visits, so all rows hold the same number of sealed rows. The mix and the seal are the
//! decode token's kernels, a block per row, so a prompt row's states and mixes have the bits a
//! decode token's would of the same inputs; both agree with the host's within rounding.

use super::{MIX_EPS, Stream, bad};
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange, DeviceStream,
        resident::depth_attention::{
            AddCall, MAX_BLOCKS, MixStatesCall, MixStatesShape, SealCall, States, StreamShape,
        },
    },
};

const WORD: usize = size_of::<f32>();

pub(super) struct PromptStreams {
    states: DeviceBuffer,
    health: DeviceBuffer,
    rows: usize,
    width: usize,
    blocks: usize,
    capacity: usize,
}

/// What one mix reads besides the states, and where it writes.
pub(super) struct MixTarget<'a> {
    /// The depth query; without one the output is the raw current row.
    pub(super) query: Option<DeviceRange<'a>>,
    /// The norm weight and epsilon of `CpuForward::norm`, applied after the mix.
    pub(super) norm: Option<(DeviceRange<'a>, f32)>,
    /// The first sealed row the mix reads.
    pub(super) first_block: usize,
    /// Row `r` of the output starts `r * stride` floats into `out`.
    pub(super) out: DeviceRange<'a>,
    pub(super) stride: usize,
}

impl PromptStreams {
    /// Uploads `streams` (one per prompt row, all with the same sealed rows) with room for
    /// `capacity` sealed rows each.
    pub(super) fn upload(
        context: &CudaContext,
        stream: &DeviceStream,
        streams: &[&Stream],
        capacity: usize,
    ) -> Result<Self, EngError> {
        let first = streams
            .first()
            .ok_or_else(|| bad("a device prompt needs at least one row"))?;
        let (width, blocks) = (first.current().len(), first.block_count());
        if capacity > MAX_BLOCKS
            || blocks > capacity
            || width == 0
            || streams
                .iter()
                .any(|s| s.block_count() != blocks || s.current().len() != width)
        {
            return Err(bad("prompt rows do not share one device stream shape"));
        }
        let stride = (1 + capacity) * width * WORD;
        let states = context.allocate(streams.len() * stride)?;
        // The device zeroes the room for later seals; only each row's rows cross the bus, one
        // enqueued copy per row from a packed host image.
        states.zero_at(stream, 0, streams.len() * stride)?;
        let used = (1 + blocks) * width * WORD;
        let mut bytes = vec![0; streams.len() * used];
        for (row, out) in streams.iter().zip(bytes.chunks_exact_mut(used)) {
            row.write_device_state(out);
        }
        for (index, row) in bytes.chunks_exact(used).enumerate() {
            states.write_at_enqueue(stream, index * stride, row)?;
        }
        let health = context.allocate(WORD)?;
        health.write_at(stream, 0, &[0; WORD])?;
        Ok(Self {
            states,
            health,
            rows: streams.len(),
            width,
            blocks,
            capacity,
        })
    }

    /// A copy of the states on the device, with its own clean health word: a pass's prelude,
    /// or the restart of a pass from it.
    pub(super) fn duplicate(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
    ) -> Result<Self, EngError> {
        let states = context.allocate(self.states.bytes())?;
        context.copy_device(
            stream,
            self.states.range(0, self.states.bytes())?,
            states.range(0, states.bytes())?,
        )?;
        let health = context.allocate(WORD)?;
        health.write_at(stream, 0, &[0; WORD])?;
        Ok(Self {
            states,
            health,
            rows: self.rows,
            width: self.width,
            blocks: self.blocks,
            capacity: self.capacity,
        })
    }

    pub(super) fn rows(&self) -> usize {
        self.rows
    }

    /// The sealed rows each state holds.
    pub(super) fn blocks(&self) -> usize {
        self.blocks
    }

    /// Whether another block can be sealed.
    pub(super) fn has_room(&self) -> bool {
        self.blocks < self.capacity
    }

    /// The states through `used` rows of the last one: the exact range a launch takes.
    fn used(&self, used: usize) -> Result<DeviceRange<'_>, EngError> {
        let rows = (self.rows - 1) * (1 + self.capacity) + used;
        self.states.range(0, rows * self.width * WORD)
    }

    fn states(&self) -> States {
        States {
            rows: self.rows,
            stride: 1 + self.capacity,
        }
    }

    fn health(&self) -> Result<DeviceRange<'_>, EngError> {
        self.health.range(0, WORD)
    }

    /// `Stream::mix(first_block, query)` of every row, optionally normalized and scaled.
    pub(super) fn mix(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        target: MixTarget<'_>,
    ) -> Result<(), EngError> {
        let (gamma, norm_eps) = target.norm.unzip();
        context.depth_attention_mix_states(
            stream,
            MixStatesCall {
                state: self.used(self.blocks + 1)?,
                query: target.query,
                gamma,
                out: target.out,
                health: self.health()?,
            },
            MixStatesShape {
                first_block: target.first_block,
                blocks: self.blocks,
                width: self.width,
                out_stride: target.stride,
                mix_eps: MIX_EPS,
                norm_eps: norm_eps.unwrap_or(1.0),
            },
            self.states(),
        )
    }

    /// `Stream::seal(next)` of every row; `next` holds one row per state, packed.
    pub(super) fn seal(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
        next: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        if !self.has_room() {
            return Err(bad(
                "the device prompt streams have no room for another block",
            ));
        }
        context.depth_attention_seal_states(
            stream,
            SealCall {
                state: self.used(self.blocks + 2)?,
                next,
                health: self.health()?,
            },
            StreamShape {
                blocks: self.blocks,
                width: self.width,
                eps: MIX_EPS,
            },
            self.states(),
        )?;
        self.blocks += 1;
        Ok(())
    }

    /// `Stream::add_current(row)` of every row; `rows` holds one row per state, packed.
    pub(super) fn add(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        rows: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        context.depth_attention_add_states(
            stream,
            AddCall {
                state: self.used(1)?,
                row: rows,
                health: self.health()?,
            },
            self.width,
            self.states(),
        )
    }

    /// Fails when any launch since the last check saw a value that was not finite.
    pub(super) fn check(&self, stream: &DeviceStream) -> Result<(), EngError> {
        let mut word = [0; WORD];
        self.health.read_at(stream, 0, &mut word)?;
        if f32::from_ne_bytes(word).to_bits() != 0 {
            return Err(bad("non-finite Inkling device prompt stream"));
        }
        Ok(())
    }

    /// The host streams the device states hold.
    pub(super) fn download(&self, stream: &DeviceStream) -> Result<Vec<Stream>, EngError> {
        self.check(stream)?;
        let mut bytes = vec![0; self.states.bytes()];
        self.states.read_at(stream, 0, &mut bytes)?;
        let floats = |bytes: &[u8]| -> Vec<f32> {
            bytes
                .as_chunks::<WORD>()
                .0
                .iter()
                .map(|w| f32::from_ne_bytes(*w))
                .collect()
        };
        let row = self.width * WORD;
        Ok(bytes
            .chunks_exact((1 + self.capacity) * row)
            .map(|state| {
                let current = floats(&state[..row]);
                let sealed = (0..self.blocks)
                    .map(|b| floats(&state[(1 + b) * row..(2 + b) * row]))
                    .collect();
                Stream::from_parts(current, sealed)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::cuda::{device_present, lane_mix, lane_norm};

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0
        }

        fn row(&mut self, width: usize) -> Vec<f32> {
            (0..width).map(|_| self.next()).collect()
        }
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    fn upload_floats(context: &CudaContext, stream: &DeviceStream, values: &[f32]) -> DeviceBuffer {
        let buffer = context.allocate(values.len() * WORD).unwrap();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        buffer.write_at(stream, 0, &bytes).unwrap();
        buffer
    }

    fn read_floats(buffer: &DeviceBuffer, stream: &DeviceStream) -> Vec<f32> {
        let mut bytes = vec![0; buffer.bytes()];
        buffer.read_at(stream, 0, &mut bytes).unwrap();
        bytes
            .as_chunks::<WORD>()
            .0
            .iter()
            .map(|w| f32::from_ne_bytes(*w))
            .collect()
    }

    /// A host stream's state, current row first, as the device lays it out.
    fn state_floats(stream: &Stream) -> Vec<f32> {
        let mut bytes = vec![0; stream.device_state_bytes()];
        stream.write_device_state(&mut bytes);
        bytes
            .as_chunks::<WORD>()
            .0
            .iter()
            .map(|w| f32::from_ne_bytes(*w))
            .collect()
    }

    fn state_bits(stream: &Stream) -> Vec<u32> {
        bits(&state_floats(stream))
    }

    fn max_error(device: &[f32], host: &[f32]) -> f32 {
        device
            .iter()
            .zip(host)
            .map(|(d, h)| (d - h).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn device_prompt_streams_mix_seal_and_add_in_the_lane_order() {
        if !device_present("Inkling device prompt streams") {
            return;
        }
        let context = CudaContext::new().unwrap();
        let stream = context.stream().unwrap();
        let mut rng = Lcg(7);
        let (rows, eps) = (256, 1e-5);
        // Widths below, at and above one load of every lane (2304 is nine), and the widest mix
        // of the most rows (4096 values, 63 sealed rows and the current one).
        for (width, sealed) in [
            (2304, 0),
            (2304, 1),
            (2304, 6),
            (1000, 6),
            (1000, 40),
            (960, 6),
            (100, 3),
            (4096, MAX_BLOCKS),
        ] {
            let host: Vec<Stream> = (0..rows)
                .map(|_| {
                    let mut s = Stream::new(rng.row(width));
                    for _ in 0..sealed {
                        s.seal(rng.row(width));
                    }
                    s
                })
                .collect();
            let refs: Vec<&Stream> = host.iter().collect();
            let capacity = (sealed.max(6) + 2).min(MAX_BLOCKS);
            let mut device = PromptStreams::upload(&context, &stream, &refs, capacity).unwrap();
            let query = rng.row(width);
            let gamma = rng.row(width);
            let (q, g) = (
                upload_floats(&context, &stream, &query),
                upload_floats(&context, &stream, &gamma),
            );
            // Mixes from every first block into a strided output: within rounding of the host's
            // mix, and the fused norm the lane-order norm of that mix, bit for bit.
            let stride = width + 5;
            let out = context.allocate(rows * stride * WORD).unwrap();
            let firsts: Vec<usize> = if sealed > 8 {
                vec![0, 1, sealed / 2, sealed]
            } else {
                (0..=sealed).collect()
            };
            for first in firsts {
                let mut mixed: Vec<Vec<f32>> = Vec::new();
                for norm in [false, true] {
                    device
                        .mix(
                            &context,
                            &stream,
                            MixTarget {
                                query: Some(q.range(0, q.bytes()).unwrap()),
                                norm: norm.then(|| (g.range(0, g.bytes()).unwrap(), eps)),
                                first_block: first,
                                out: out.range(0, ((rows - 1) * stride + width) * WORD).unwrap(),
                                stride,
                            },
                        )
                        .unwrap();
                    let values = read_floats(&out, &stream);
                    for (r, h) in host.iter().enumerate() {
                        let actual = &values[r * stride..][..width];
                        if norm {
                            let expected = lane_norm(&mixed[r], Some(&gamma), eps);
                            assert_eq!(
                                bits(actual),
                                bits(&expected),
                                "sealed {sealed} first {first} norm row {r}"
                            );
                        } else {
                            let error = max_error(actual, &h.mix(first, &query));
                            assert!(
                                error <= 1e-4,
                                "sealed {sealed} first {first} row {r}: max error {error}"
                            );
                            let state = state_floats(h);
                            let (current, rows) = state.split_at(width);
                            let rows: Vec<&[f32]> = rows.chunks(width).collect();
                            let lane = lane_mix(current, &rows, first, &query, MIX_EPS);
                            assert_eq!(
                                bits(actual),
                                bits(&lane),
                                "sealed {sealed} first {first} lane order row {r}"
                            );
                            mixed.push(actual.to_vec());
                        }
                    }
                }
            }
            // Without a query: the raw current row, alone or normalized and scaled.
            for norm in [false, true] {
                device
                    .mix(
                        &context,
                        &stream,
                        MixTarget {
                            query: None,
                            norm: norm.then(|| (g.range(0, g.bytes()).unwrap(), eps)),
                            first_block: 0,
                            out: out.range(0, rows * width * WORD).unwrap(),
                            stride: width,
                        },
                    )
                    .unwrap();
                let values = read_floats(&out, &stream);
                for (r, h) in host.iter().enumerate() {
                    let expected = if norm {
                        lane_norm(h.current(), Some(&gamma), eps)
                    } else {
                        h.current().to_vec()
                    };
                    assert_eq!(
                        bits(&values[r * width..][..width]),
                        bits(&expected),
                        "current row {r} norm {norm}"
                    );
                }
            }
            // Seal, then add, while a block is free: the sealed row is the lane-order norm of the
            // old current row and the add one rounded sum per element.
            if sealed == capacity {
                continue;
            }
            let next: Vec<Vec<f32>> = (0..rows).map(|_| rng.row(width)).collect();
            let addend: Vec<Vec<f32>> = (0..rows).map(|_| rng.row(width)).collect();
            let next_buffer = upload_floats(&context, &stream, &next.concat());
            let add_buffer = upload_floats(&context, &stream, &addend.concat());
            device
                .seal(
                    &context,
                    &stream,
                    next_buffer.range(0, next_buffer.bytes()).unwrap(),
                )
                .unwrap();
            device
                .add(
                    &context,
                    &stream,
                    add_buffer.range(0, add_buffer.bytes()).unwrap(),
                )
                .unwrap();
            let downloaded = device.download(&stream).unwrap();
            for (r, ((h, n), a)) in host.iter().zip(next).zip(&addend).enumerate() {
                let mut expected = state_bits(h);
                let row = expected.split_off(width);
                let current: Vec<f32> = n.iter().zip(a).map(|(x, y)| x + y).collect();
                let mut expected_state = bits(&current);
                expected_state.extend(row);
                expected_state.extend(bits(&lane_norm(h.current(), None, MIX_EPS)));
                assert_eq!(
                    state_bits(&downloaded[r]),
                    expected_state,
                    "state of row {r}"
                );
            }
        }
    }
}
