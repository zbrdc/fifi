//! Prompt visits run on the device prompt streams. The depth mixes, norms, attention, residual,
//! MLP input, routing, experts and both output convolutions stay on the device; the host reads
//! back each visit's statuses and expert grouping, and the last raw rows its convolution
//! histories keep. The core passes' prelude and message bank stay on the device too (`banked`
//! runs the visits that read the bank).

use super::super::prompt_streams::{MixTarget, PromptStreams};
use super::{
    Attention, CpuForward, EngError, GgufTextWeights, LayerVisit, MessageBank, PromptRow, Role,
    VisitState, attention_norm_name, bad, resident_mlp::Route,
};
use crate::family::inkling2::{cuda_projection::gguf::attention::prefill, switches::Switch};
use crate::kernels::cuda::{
    CudaContext, DeviceBuffer, DeviceRange, DeviceStream,
    resident::depth_attention::{MAX_BLOCKS, MAX_STATES},
};

mod banked;

pub(super) use banked::banked_enabled;

const WORD: usize = size_of::<f32>();
/// Constant rows per visit: three attention queries (0..3) and norms (3..6), the MLP query (6)
/// and norm (7); banked visits add the three attention control queries (8..11) and the MLP
/// control query (11).
const CONSTANTS: usize = 12;

/// A prompt's device streams and the planes its visits reuse.
pub(super) struct DevicePrompt {
    streams: PromptStreams,
    /// The streams at the first core pass's start, which each later pass restarts from.
    prelude: Option<PromptStreams>,
    /// The previous pass's streams, which a banked visit's controllers read.
    bank: Option<Bank>,
    stream: DeviceStream,
    /// Each row's three attention inputs, packed.
    inputs: DeviceBuffer,
    /// Each row's MLP input.
    mlp: DeviceBuffer,
    constants: DeviceBuffer,
    /// The controller pairs of banked visits, allocated by the first.
    controls: Option<Controls>,
}

/// The message bank on the device: `MessageBank` of every row.
struct Bank {
    streams: PromptStreams,
    first_block: usize,
}

/// Each attention role's [input, message] pair per row (role-major), and two health words per
/// pair for the gates prepared from them.
struct Controls {
    pairs: DeviceBuffer,
    health: DeviceBuffer,
}

impl DevicePrompt {
    /// Uploads the rows' streams, or declines when the device path or one it uses is off, or
    /// the prompt does not fit. `peak` is the most sealed blocks the coming visits reach.
    pub(super) fn open(
        forward: &CpuForward,
        weights: &GgufTextWeights,
        rows: &[PromptRow<'_>],
        peak: usize,
    ) -> Result<Option<Self>, EngError> {
        let Some((context, _)) = weights.resident_attention() else {
            return Ok(None);
        };
        let Some(first) = rows.first() else {
            return Ok(None);
        };
        let blocks = first.stream.block_count();
        let capacity = peak.min(MAX_BLOCKS);
        // The visits project and attend through the batched prefill bridge, so its switches
        // decline the device prompt as they decline the host's grouped visits.
        if !Switch::PrefillDevicePrompt.on()
            || !prefill::enabled()
            || !super::prefill_projection_batch_enabled()
            || (!prefill::resident_query_enabled() && !prefill::device_side_enabled())
            || forward.draft_window_enabled()
            || !(2..=MAX_STATES).contains(&rows.len())
            || blocks > capacity
            || !forward.taps.is_empty()
            || rows
                .iter()
                .any(|row| row.bank.is_some() || row.prelude.is_some() || row.journal.is_some())
        {
            return Ok(None);
        }
        let width = forward.hidden();
        let opened = (|| {
            let stream = context.stream()?;
            let streams: Vec<_> = rows.iter().map(|row| &row.stream).collect();
            Ok::<_, EngError>(Self {
                streams: PromptStreams::upload(context, &stream, &streams, capacity)?,
                prelude: None,
                bank: None,
                inputs: context.allocate(rows.len() * 3 * width * WORD)?,
                mlp: context.allocate(rows.len() * width * WORD)?,
                constants: context.allocate(CONSTANTS * width * WORD)?,
                controls: None,
                stream,
            })
        })();
        match opened {
            Ok(prompt) => Ok(Some(prompt)),
            Err(EngError::MemoryPlan(_)) => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// `PromptRow::begin_pass` of every row: the first pass keeps a prelude of the streams,
    /// a later one banks the streams and restarts them from the prelude. A failed allocation
    /// declines, with the streams unchanged.
    pub(super) fn begin_pass(
        &mut self,
        context: &CudaContext,
        pass: usize,
    ) -> Result<bool, EngError> {
        let begun = if pass == 0 {
            self.streams
                .duplicate(context, &self.stream)
                .map(|prelude| self.prelude = Some(prelude))
        } else {
            let Some(prelude) = &self.prelude else {
                return Ok(false);
            };
            let first_block = prelude.blocks() + 1;
            prelude.duplicate(context, &self.stream).map(|restart| {
                let streams = std::mem::replace(&mut self.streams, restart);
                self.bank = Some(Bank {
                    streams,
                    first_block,
                });
            })
        };
        match begun {
            Ok(()) => Ok(true),
            Err(EngError::MemoryPlan(_)) => Ok(false),
            Err(error) => Err(error),
        }
    }

    /// Whether the visits read the message bank.
    pub(super) fn banked(&self) -> bool {
        self.bank.is_some()
    }

    /// The coda reads neither the prelude nor the bank.
    pub(super) fn end_passes(&mut self) {
        self.prelude = None;
        self.bank = None;
    }

    /// Fails when a launch on the device streams, prelude or bank saw a value that was not
    /// finite: the checks [`Self::materialize`] makes, without the download.
    pub(super) fn check(&self) -> Result<(), EngError> {
        self.streams.check(&self.stream)?;
        if let Some(prelude) = &self.prelude {
            prelude.check(&self.stream)?;
        }
        if let Some(bank) = &self.bank {
            bank.streams.check(&self.stream)?;
        }
        Ok(())
    }

    /// Writes the device streams, prelude and bank back into the rows.
    pub(super) fn materialize(&self, rows: &mut [PromptRow<'_>]) -> Result<(), EngError> {
        let download = |streams: &PromptStreams| {
            let streams = streams.download(&self.stream)?;
            if streams.len() != rows.len() {
                return Err(bad("device prompt streams lost a row"));
            }
            Ok(streams)
        };
        let streams = download(&self.streams)?;
        let mut preludes = self
            .prelude
            .as_ref()
            .map(download)
            .transpose()?
            .map(Vec::into_iter);
        let mut banks = self
            .bank
            .as_ref()
            .map(|bank| Ok::<_, EngError>((download(&bank.streams)?.into_iter(), bank.first_block)))
            .transpose()?;
        for (row, stream) in rows.iter_mut().zip(streams) {
            row.stream = stream;
            row.prelude = preludes.as_mut().and_then(Iterator::next);
            row.bank = banks.as_mut().and_then(|(banks, first_block)| {
                banks.next().map(|stream| MessageBank {
                    first_block: *first_block,
                    stream,
                })
            });
        }
        Ok(())
    }

    fn constant(&self, index: usize, width: usize) -> Result<DeviceRange<'_>, EngError> {
        constant(&self.constants, index, width)
    }
}

/// Rows of the next projection launch, of `left` rows and a stage of `capacity`: the
/// tensor-core GEMM takes all it can hold, the exact kernels a whole tile of their weight visit
/// (256, 128 or 64 rows) or what is left. Each row's outputs do not depend on the count.
fn projection_rows(left: usize, capacity: usize, tensor_core: bool) -> usize {
    let rows = left.min(capacity);
    if tensor_core {
        return rows;
    }
    [256, 128, 64]
        .into_iter()
        .find(|&tile| rows >= tile)
        .unwrap_or(rows)
}

#[test]
fn exact_projections_take_whole_visit_tiles() {
    let runs = |mut left: usize, capacity, tensor_core| {
        let mut runs = Vec::new();
        while left > 0 {
            let rows = projection_rows(left, capacity, tensor_core);
            runs.push(rows);
            left -= rows;
        }
        runs
    };
    assert_eq!(runs(512, 1024, false), [256, 256]);
    assert_eq!(runs(511, 1024, false), [256, 128, 64, 63]);
    assert_eq!(runs(512, 100, false), [64; 8]);
    assert_eq!(runs(30, 1024, false), [30]);
    assert_eq!(runs(512, 1024, true), [512]);
    assert_eq!(runs(512, 300, true), [300, 212]);
}

fn constant(
    constants: &DeviceBuffer,
    index: usize,
    width: usize,
) -> Result<DeviceRange<'_>, EngError> {
    constants.range(index * width * WORD, width * WORD)
}

impl CpuForward {
    /// Runs an unbanked prompt visit on the device prompt streams. Returns false, with nothing
    /// advanced, when the visit cannot run there.
    pub(super) fn device_prompt_visit(
        &self,
        visit: LayerVisit,
        state: &mut VisitState,
        prompt: &mut DevicePrompt,
        weights: &GgufTextWeights,
        (layer, prefix): (&str, &str),
    ) -> Result<bool, EngError> {
        let rows = prompt.streams.rows();
        let width = self.hidden();
        let physical = visit.weight_layer;
        let experts = self.spec.n_routed_experts as usize;
        let router = format!("{prefix}.gate.weight");
        let expert_names = [
            format!("{prefix}.experts.w13_weight"),
            format!("{prefix}.experts.w2_weight"),
        ];
        let names = expert_names.each_ref().map(String::as_str);
        let caps = [self.math.mlp_gate_softcap, self.math.mlp_up_softcap];
        let seal = physical.is_multiple_of(self.spec.n_layers_per_attn_res_block);
        if (seal && !prompt.streams.has_room())
            || weights.resident_prefill_router_shape(physical as u32, &router, rows)
                != Some([experts, width])
        {
            return Ok(false);
        }
        let Some((context, attention)) = weights.resident_attention() else {
            return Ok(false);
        };
        let Some(bridge) = self.prefill_bridge(layer, visit, prefill::resident_query_enabled())?
        else {
            return Ok(false);
        };
        let keep = bridge.keep();
        let widths = [width; 3];
        let Some(projector) = self.attention_requests(layer, visit.attention, |calls| {
            weights.prefill_projection(widths, calls)
        })?
        else {
            return Ok(false);
        };
        let projected = projector.capacity_for(widths, keep)?;
        if bridge.tile_capacity().min(projected) < 2 {
            return Ok(false);
        }
        let Some(mut conv) = bridge.admit_conv(self, state, rows)? else {
            return Ok(false);
        };
        let Some(mut mlp_conv) = self.admit_prompt_mlp_conv(state, layer, rows)? else {
            return Ok(false);
        };
        let dims = [rows, self.spec.num_experts_per_tok as usize, width, experts];
        if !weights.admit_resident_prefill_experts(names, physical as u32, dims, caps)? {
            return Ok(false);
        }
        self.upload_visit_constants(prompt, visit, layer, false)?;
        let bias = self.weights.vec(&format!("{prefix}.gate.bias"), experts)?;
        let eps = self.math.rms_norm_eps;
        // From here the visit advances state; a failure is an error, not a decline.
        let local = visit.attention == Attention::Local;
        let row_words = 3 * width;
        for role in 0..3 {
            // Local roles share the first role's input, as `attention_inputs_scalar` does.
            let source = if local { 0 } else { role };
            prompt.streams.mix(
                context,
                &prompt.stream,
                MixTarget {
                    query: (visit.weight_layer > 0)
                        .then(|| prompt.constant(source, width))
                        .transpose()?,
                    norm: Some((prompt.constant(3 + source, width)?, eps)),
                    first_block: 0,
                    out: prompt
                        .inputs
                        .range(role * width * WORD, ((rows - 1) * row_words + width) * WORD)?,
                    stride: row_words,
                },
            )?;
        }
        prompt.stream.synchronize()?;
        let mut first = 0;
        while first < rows {
            // The attention splits each projected run into its own tiles.
            let count = projection_rows(rows - first, projected, false);
            let x = prompt
                .inputs
                .range(first * row_words * WORD, count * row_words * WORD)?;
            let (host, side) = projector.tile_device(x, (count, widths), keep)?;
            if !bridge
                .consume(self, state, host, side, Some((&mut conv, false)))?
                .is_empty()
            {
                return Err(bad("unexpected device prompt attention readback"));
            }
            first += count;
        }
        let attended = conv.convolved_plane()?;
        if seal {
            prompt.streams.seal(context, &prompt.stream, attended)?;
        } else {
            prompt.streams.add(context, &prompt.stream, attended)?;
        }
        let activations = prompt.mlp.range(0, rows * width * WORD)?;
        prompt.streams.mix(
            context,
            &prompt.stream,
            MixTarget {
                query: Some(prompt.constant(6, width)?),
                norm: Some((prompt.constant(7, width)?, eps)),
                first_block: 0,
                out: activations,
                stride: width,
            },
        )?;
        // The attention plane is free once the residual has read it.
        prompt.stream.synchronize()?;
        bridge.publish_conv_tail(state, conv)?;
        let route = Route {
            physical: physical as u32,
            weights,
            router: &router,
            bias: &bias,
            experts: &expert_names,
            caps,
        };
        route.dispatch(
            self,
            rows,
            activations,
            Some(mlp_conv.expert_destination()?),
        )?;
        attention.finish_prefill_expert_conv(context, &mut mlp_conv)?;
        prompt
            .streams
            .add(context, &prompt.stream, mlp_conv.convolved_plane()?)?;
        prompt.stream.synchronize()?;
        prompt.streams.check(&prompt.stream)?;
        self.publish_prompt_mlp_tail(state, mlp_conv)?;
        static USED: std::sync::Once = std::sync::Once::new();
        USED.call_once(|| eprintln!("Inkling CUDA kept prompt depth streams on the device"));
        Ok(true)
    }

    /// The visit's attention queries and norms (one role on local visits), its MLP query and
    /// norm, and on a banked visit the control queries, in the order of [`CONSTANTS`].
    fn upload_visit_constants(
        &self,
        prompt: &DevicePrompt,
        visit: LayerVisit,
        layer: &str,
        banked: bool,
    ) -> Result<(), EngError> {
        let width = self.hidden();
        let local = visit.attention == Attention::Local;
        let roles = if local { 1 } else { 3 };
        let physical = visit.weight_layer;
        let mut words = vec![0.0; CONSTANTS * width];
        let mut put = |index: usize, values: &[f32]| {
            words[index * width..][..width].copy_from_slice(values);
        };
        if let Some(query) = physical.checked_sub(1) {
            for role in 0..roles {
                let name = format!("model.llm.attn_res.attn_queries.{query}");
                put(role, &self.weights.row(&name, role, width)?);
            }
        }
        for role in 0..roles {
            put(
                3 + role,
                &self
                    .weights
                    .vec(&attention_norm_name(layer, local, role), width)?,
            );
        }
        put(
            6,
            &self
                .weights
                .vec(&format!("model.llm.attn_res.mlp_queries.{physical}"), width)?,
        );
        put(
            7,
            &self
                .weights
                .vec(&format!("{layer}.mlp_norm.weight"), width)?,
        );
        if banked {
            let roles = [Role::Query, Role::Key, Role::Value, Role::Mlp];
            for (index, role) in (8..).zip(roles) {
                put(index, &self.control_query(physical, role)?);
            }
        }
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
        // Queued behind the previous visit's reads on the prompt stream; the controllers on
        // their own stream read the constants only after the prompt stream's next synchronize.
        prompt.constants.write_at_enqueue(&prompt.stream, 0, &bytes)
    }
}
