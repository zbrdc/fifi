//! Launch-only local decode attention: the enqueue-only form of the host-entry chain
//! (`chained_local_decode` -> `step_local` -> `finish_local_decode`) for rows that stay in
//! device memory. Nothing here uploads, downloads or synchronizes; non-finite values
//! surface in the request's status words, which the caller reads once per token.
//!
//! The banked global variant uses separate role rows and dynamic taps.

mod global;
mod plain;

use crate::{
    EngError,
    family::inkling2::{
        Attention, LayerVisit,
        cpu_weights::Projection,
        cuda_projection::dense::{
            ControllerWeights, ResidentLocal, ResidentLocalInputs, ResidentLocalTargets,
            ResidentProjection,
        },
    },
    kernels::cuda::{
        Claim, DeviceRange, DeviceStream, require_apart, require_fit,
        resident::relative_prep::PrepMath,
    },
};

use super::{
    GgufCuda,
    attention::{LocalShape, OutputProjection, ResidentAttention, ResidentParts, ResidentStatus},
    bad,
};

/// The widths and tap count the caller sizes its own buffers with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family::inkling2) struct LocalDecodeWidths {
    pub hidden: usize,
    /// Key/value heads times the head width; one K, V, raw K or raw V row.
    pub kv_channels: usize,
    /// Query heads times the head width; the attention output before its projection.
    pub query_channels: usize,
    /// Query heads times the relative dimensions; the relative coefficients row.
    pub relative_words: usize,
    /// `sconv_kernel_size`; every history holds `taps - 1` rows.
    pub taps: usize,
}

/// Constant weight rows the caller uploads once per layer, in this order.
#[derive(Clone, Copy)]
pub(in crate::family::inkling2) struct PrepWeights<'a> {
    /// `attn.q_l2_clip.scale`: `query_channels` words.
    pub query_scale: DeviceRange<'a>,
    /// `attn.rel_l2_clip.scale`: `relative_words` words.
    pub relative_scale: DeviceRange<'a>,
    /// `attn.k_sconv.weight`: `kv_channels * taps` words, tap-major per channel.
    pub key_kernel: DeviceRange<'a>,
    /// `attn.v_sconv.weight`: `kv_channels * taps` words.
    pub value_kernel: DeviceRange<'a>,
}

/// The static `attn_sconv` convolution of the projected output.
#[derive(Clone, Copy)]
pub(in crate::family::inkling2) struct ConvRows<'a> {
    /// `attn_sconv.weight`: `hidden * taps` words, tap-major per channel.
    pub kernel: DeviceRange<'a>,
    /// `(taps - 1) * hidden` words, newest row first, zero rows for a short history.
    pub history: DeviceRange<'a>,
}

/// One visit to enqueue. Every range is caller-owned device memory that stays valid until
/// the stream completes the visit.
#[derive(Clone, Copy)]
pub(in crate::family::inkling2) struct ResidentLocalDecode<'a> {
    pub visit: LayerVisit,
    /// The next position of the visit on the host; `pos` must hold it on the device.
    pub position: usize,
    /// The prep math of the host: `qk_l2_clip_threshold`, `rel_l2_clip_threshold` and
    /// `rms_norm_eps` (also the eps of the controller norm).
    pub math: PrepMath,
    /// The shared attention input row: `hidden` words.
    pub input: DeviceRange<'a>,
    /// The bank messages of the Q, K and V roles: `hidden` words each. `None` when the
    /// visit has no bank; the roles then share the input, as `attention_inputs_scalar` does.
    pub messages: Option<[DeviceRange<'a>; 3]>,
    /// The role norm gammas: `hidden` words each (`attn_norm.weight` thrice for local).
    pub gammas: [DeviceRange<'a>; 3],
    pub weights: PrepWeights<'a>,
    /// Raw K and V convolution histories: `(taps - 1) * kv_channels` words each, newest
    /// row first; zero rows stand for a shorter history. The chain shifts both.
    pub raw_history: [DeviceRange<'a>; 2],
    pub attn_conv: ConvRows<'a>,
    /// Four bytes holding `position` as a native `i32` (for example `StepParams.pos`).
    pub pos: DeviceRange<'a>,
    /// `hidden` words: the attention output after `wo_ud` and `attn_sconv`.
    pub out: DeviceRange<'a>,
    /// `4 * kv_channels` words: `[raw K, raw V, clipped K, normalized V]`.
    pub log: DeviceRange<'a>,
    /// `status_words()` words; see the module docs of [`LocalStatus`].
    pub status: DeviceRange<'a>,
}

/// The status words of one visit, all zero when every launch was finite: the prep and
/// attention status, the controller health, the gate health and the convolution health.
#[derive(Clone, Copy)]
pub(in crate::family::inkling2) struct LocalStatus {
    /// `1 + 2 * (query heads + kv heads)` `u32` words: the prep and attention status,
    /// then one per-head prep flag.
    pub prep_words: usize,
    /// Three controller health words, or one for the identity control of a visit without a bank.
    pub roles: usize,
    /// Six gate norm health words with a bank, none without.
    pub gates: usize,
}

impl LocalStatus {
    pub(in crate::family::inkling2) fn words(self) -> usize {
        self.prep_words + self.roles + self.gates + 1
    }
}

impl GgufCuda {
    pub(in crate::family::inkling2) fn prepare_local_decode(
        &self,
        visit: LayerVisit,
    ) -> Result<(), EngError> {
        self.prepare_token_decode(visit, true)
    }

    pub(in crate::family::inkling2) fn prepare_token_decode(
        &self,
        visit: LayerVisit,
        bank: bool,
    ) -> Result<(), EngError> {
        let attention = self
            .resident_attention
            .as_ref()
            .ok_or_else(|| bad("missing resident local attention"))?;
        let shape = attention
            .local_shape(visit.state_slot)
            .ok_or_else(|| bad("unknown resident local visit slot"))?;
        attention.prepare_resident_workspace(&self.context, (shape, self.hidden), bank)
    }

    pub(in crate::family::inkling2) fn publish_local_decode(
        &self,
        visit: LayerVisit,
        position: usize,
    ) {
        self.resident_attention
            .as_ref()
            .expect("resident local visit was admitted")
            .publish_resident_cursor(visit.state_slot, position);
    }

    pub(in crate::family::inkling2) fn resident_local_decode_widths(
        &self,
        visit: LayerVisit,
    ) -> Option<LocalDecodeWidths> {
        let attention = self.resident_attention.as_ref()?;
        let shape = attention.local_shape(visit.state_slot)?;
        let taps = attention.local_taps();
        Some(LocalDecodeWidths {
            hidden: self.hidden,
            kv_channels: shape.kv_channels(),
            query_channels: shape.query_channels(),
            relative_words: shape.relative_words(),
            taps,
        })
    }

    pub(in crate::family::inkling2) fn resident_local_decode_status_words(
        &self,
        visit: LayerVisit,
        bank: bool,
    ) -> Option<usize> {
        let attention = self.resident_attention.as_ref()?;
        let shape = attention.local_shape(visit.state_slot)?;
        Some(local_status(shape, bank).words())
    }

    /// True only when [`Self::enqueue_local_decode`] would run for this visit: a local
    /// visit with a known device cursor, every controller and projection resident and the
    /// output projection chained.
    pub(in crate::family::inkling2) fn resident_local_decode_supported(
        &self,
        visit: LayerVisit,
        position: usize,
    ) -> bool {
        self.resident_local_decode_parts(visit, position).is_ok()
    }

    /// `(attention, shape, projection)` for a visit that [`Self::resident_local_decode_supported`]
    /// accepts; the error names the first missing piece.
    pub(in crate::family::inkling2) fn resident_local_decode_parts(
        &self,
        visit: LayerVisit,
        position: usize,
    ) -> Result<(&ResidentAttention, LocalShape, OutputProjection<'_>), EngError> {
        if visit.attention != Attention::Local {
            return Err(bad("resident local decode needs an Attention::Local visit"));
        }
        self.resident_decode_parts(visit, position, true)
    }

    fn resident_decode_parts(
        &self,
        visit: LayerVisit,
        position: usize,
        banked: bool,
    ) -> Result<(&ResidentAttention, LocalShape, OutputProjection<'_>), EngError> {
        let attention = self
            .resident_attention
            .as_ref()
            .ok_or_else(|| bad("resident local decode needs resident attention"))?;
        let shape = attention
            .local_shape(visit.state_slot)
            .ok_or_else(|| bad("resident local decode needs a known visit slot"))?;
        if !attention.can_chain_local(visit.state_slot, position) {
            return Err(bad(
                "resident local decode needs the device cursor at the position",
            ));
        }
        if !self.step.local_inputs_fit(self.hidden, banked) {
            return Err(bad("resident local decode exceeds the projection arena"));
        }
        let layer = format!("model.llm.layers.{}", visit.weight_layer);
        let projection = self
            .relative_attention_projection(
                &format!("{layer}.attn.wo_ud.weight"),
                [self.hidden, shape.query_channels()],
            )?
            .ok_or_else(|| bad("resident local decode needs the resident output projection"))?;
        let parts = SupportParts {
            layer: &layer,
            shape,
            hidden: self.hidden,
        };
        self.resident_projections(parts)?;
        if banked {
            self.resident_controllers(parts)?;
        }
        Ok((attention, shape, projection))
    }

    fn resident_projections(&self, parts: SupportParts<'_>) -> Result<(), EngError> {
        let SupportParts {
            layer,
            shape,
            hidden,
        } = parts;
        if !self.group_attention {
            return Err(bad("resident local decode needs grouped attention weights"));
        }
        let requests = [
            ("wq_du.weight", hidden, shape.query_channels()),
            ("wk_dv.weight", hidden, shape.kv_channels()),
            ("wv_dv.weight", hidden, shape.kv_channels()),
            ("wr_du.weight", hidden, shape.relative_words()),
            ("attn_sink_proj.weight", hidden, shape.query_heads),
            ("gate_attn.weight", hidden, shape.query_channels()),
        ];
        for (name, cols, rows) in requests {
            let resident = self
                .dense
                .get(&format!("{layer}.attn.{name}"))
                .ok_or_else(|| bad("resident local decode lacks a resident projection"))?;
            if resident.0.shape != [rows, cols] || self.step.uses_q8_activations(resident.1) {
                return Err(bad("resident local decode projection shape differs"));
            }
        }
        Ok(())
    }

    fn resident_controllers(&self, parts: SupportParts<'_>) -> Result<(), EngError> {
        let SupportParts { layer, .. } = parts;
        if !self.group_projections {
            return Err(bad(
                "resident local decode needs grouped controller weights",
            ));
        }
        let rank = self.gate_rank;
        let hidden = self.hidden;
        for role in ["attn", "k", "v"] {
            for (part, shape) in [
                ("down", [rank, 2 * hidden]),
                ("gain", [hidden, rank]),
                ("forget", [hidden, rank]),
            ] {
                let name = format!("{layer}.controllers.{role}.{part}_weight");
                let resident = self
                    .dense
                    .get(&name)
                    .or_else(|| self.token_controllers.get(&name))
                    .ok_or_else(|| bad("resident local decode lacks a resident controller"))?;
                if resident.0.shape != shape {
                    return Err(bad("resident local decode controller shape differs"));
                }
            }
        }
        if self.step.controller_capacity(hidden).unwrap_or(0) < 2 {
            return Err(bad(
                "resident local decode needs the multirow controller arena",
            ));
        }
        Ok(())
    }

    /// `(name, words)` of every constant row the caller uploads once per layer: the
    /// shared gamma, `PrepWeights` in order, then the `attn_sconv` kernel.
    pub(in crate::family::inkling2) fn resident_local_decode_weight_rows(
        &self,
        visit: LayerVisit,
        layer: &str,
    ) -> Result<[(String, usize); 6], EngError> {
        let widths = self
            .resident_local_decode_widths(visit)
            .ok_or_else(|| bad("resident local decode needs resident attention"))?;
        let global = visit.attention == Attention::Global;
        let rows = [
            (
                if global {
                    "q_input_norm.weight"
                } else {
                    "attn_norm.weight"
                },
                widths.hidden,
            ),
            ("attn.q_l2_clip.scale", widths.query_channels),
            ("attn.rel_l2_clip.scale", widths.relative_words),
            (
                if global {
                    "attn.k_sconv.static_w"
                } else {
                    "attn.k_sconv.weight"
                },
                widths.kv_channels * widths.taps,
            ),
            (
                if global {
                    "attn.v_sconv.static_w"
                } else {
                    "attn.v_sconv.weight"
                },
                widths.kv_channels * widths.taps,
            ),
            ("attn_sconv.weight", widths.hidden * widths.taps),
        ];
        Ok(rows.map(|(name, words)| (format!("{layer}.{name}"), words)))
    }
}

#[derive(Clone, Copy)]
struct SupportParts<'a> {
    layer: &'a str,
    shape: LocalShape,
    hidden: usize,
}

pub(super) fn local_status(shape: LocalShape, bank: bool) -> LocalStatus {
    LocalStatus {
        prep_words: 1 + 2 * (shape.query_heads + shape.kv_heads),
        roles: if bank { 3 } else { 1 },
        gates: if bank { 6 } else { 0 },
    }
}

/// The controller weights (a bank only) and the six attention projections of a visit.
pub(super) struct ResidentWeights<'a> {
    pub(super) controllers: Option<[ControllerWeights<'a>; 3]>,
    pub(super) projections: [ResidentProjection<'a>; 6],
}

/// The controller and projection weights by tensor name, or the error naming the gap.
fn resident_local_weights<'a>(
    cuda: &'a GgufCuda,
    request: &ResidentLocalDecode<'_>,
    shape: LocalShape,
) -> Result<ResidentWeights<'a>, EngError> {
    let hidden = cuda.hidden;
    let [q, rel, kv] = [
        shape.query_channels(),
        shape.relative_words(),
        shape.kv_channels(),
    ];
    let prefix = format!("model.llm.layers.{}.attn.", request.visit.weight_layer);
    let names = [
        "wq_du.weight",
        "wk_dv.weight",
        "wv_dv.weight",
        "wr_du.weight",
        "attn_sink_proj.weight",
        "gate_attn.weight",
    ]
    .map(|suffix| format!("{prefix}{suffix}"));
    let [inputs, rows] = [[0, 1, 2, 0, 0, 0], [q, kv, kv, rel, shape.query_heads, q]];
    let requests = std::array::from_fn::<_, 8, _>(|i| {
        (i < 6).then(|| Projection {
            name: &names[i],
            input: inputs[i],
            rows: rows[i],
        })
    });
    let weights = cuda
        .attention_group([hidden; 3], requests)
        .ok_or_else(|| bad("resident local decode needs grouped attention weights"))??;
    let [
        Some(wq),
        Some(wk),
        Some(wv),
        Some(wr),
        Some(sinks),
        Some(gates),
        None,
        None,
    ] = weights
    else {
        return Err(bad(
            "resident local decode lacks a resident attention projection",
        ));
    };
    let controllers = request
        .messages
        .is_some()
        .then(|| resident_controllers(cuda, request))
        .transpose()?;
    Ok(ResidentWeights {
        controllers,
        projections: [wq, wk, wv, wr, sinks, gates],
    })
}

fn resident_controllers<'a>(
    cuda: &'a GgufCuda,
    request: &ResidentLocalDecode<'_>,
) -> Result<[ControllerWeights<'a>; 3], EngError> {
    let layer = format!("model.llm.layers.{}", request.visit.weight_layer);
    let names = ["attn", "k", "v"].map(|role| {
        ["down", "gain", "forget"].map(|part| format!("{layer}.controllers.{role}.{part}_weight"))
    });
    let names = names
        .each_ref()
        .map(|names| names.each_ref().map(String::as_str));
    let [rank, hidden] = [cuda.gate_rank, cuda.hidden];
    let mut weights = Vec::with_capacity(3);
    for names in names {
        let missing = || bad("resident local decode lacks a resident controller");
        weights.push(
            cuda.resolve_controller_weights(names, 2 * hidden, [rank, hidden], true)
                .ok_or_else(missing)??,
        );
    }
    Ok([weights[0], weights[1], weights[2]])
}

/// The caller's ranges by role: name, range and the exact word count it must hold.
pub(super) struct RequestRows<'a> {
    pub(super) reads: Vec<(&'static str, DeviceRange<'a>, usize)>,
    pub(super) writes: Vec<(&'static str, DeviceRange<'a>, usize)>,
}

const WORD: usize = size_of::<f32>();

pub(super) fn request_rows<'a>(
    request: &ResidentLocalDecode<'a>,
    (shape, hidden, taps): (LocalShape, usize, usize),
) -> RequestRows<'a> {
    let [q, rel, kv] = [
        shape.query_channels(),
        shape.relative_words(),
        shape.kv_channels(),
    ];
    let past = taps - 1;
    let mut reads = vec![("attention input", request.input, hidden)];
    reads.extend(
        request
            .messages
            .iter()
            .flatten()
            .map(|row| ("bank message", *row, hidden)),
    );
    reads.extend(
        request
            .gammas
            .iter()
            .map(|row| ("role gamma", *row, hidden)),
    );
    reads.extend([
        ("query scale", request.weights.query_scale, q),
        ("relative scale", request.weights.relative_scale, rel),
        (
            "key convolution kernel",
            request.weights.key_kernel,
            kv * taps,
        ),
        (
            "value convolution kernel",
            request.weights.value_kernel,
            kv * taps,
        ),
        (
            "attention convolution kernel",
            request.attn_conv.kernel,
            hidden * taps,
        ),
        ("position", request.pos, 1),
    ]);
    let status = local_status(shape, request.messages.is_some()).words();
    let writes = vec![
        ("attention output", request.out, hidden),
        ("log", request.log, 4 * kv),
        ("status", request.status, status),
        ("raw key history", request.raw_history[0], past * kv),
        ("raw value history", request.raw_history[1], past * kv),
        (
            "attention convolution history",
            request.attn_conv.history,
            past * hidden,
        ),
    ];
    RequestRows { reads, writes }
}

fn claim<'a>(name: &'static str, range: &'a DeviceRange<'a>, words: usize) -> Claim<'a> {
    Claim::of(name, range, words * WORD)
}

/// Every range holds exactly its words at float alignment, and the ranges the chain
/// writes share no byte with each other or with a range it only reads.
pub(super) fn validate_rows(rows: &RequestRows<'_>) -> Result<(), EngError> {
    let all = rows.reads.iter().chain(&rows.writes);
    for (name, range, words) in all {
        let bytes = words * WORD;
        require_fit(name, range, bytes, WORD)?;
        if range.bytes() != bytes {
            return Err(bad(format!(
                "the CUDA resident local {name} range holds {} bytes but needs exactly {bytes}",
                range.bytes()
            )));
        }
    }
    for (index, (name, range, words)) in rows.writes.iter().enumerate() {
        let write = claim(name, range, *words);
        for (other, range, words) in rows.writes.iter().skip(index + 1) {
            require_apart(&write, &claim(other, range, *words))?;
        }
        for (other, range, words) in &rows.reads {
            require_apart(&write, &claim(other, range, *words))?;
        }
    }
    Ok(())
}

impl GgufCuda {
    /// The enqueue-only form of the host local decode chain for one visit: controllers
    /// (with a bank), Q/K/V projections, local prep over the caller's K/V histories,
    /// attention over the resident ring, the output projection and the `attn_sconv`
    /// convolution into `out`. Every extent is checked before the first launch; nothing
    /// is uploaded, downloaded or synchronized. All work is queued on `stream`, in order.
    pub(in crate::family::inkling2) fn enqueue_local_decode(
        &self,
        stream: &DeviceStream,
        request: ResidentLocalDecode<'_>,
    ) -> Result<(), EngError> {
        let (attention, shape, projection) =
            self.resident_local_decode_parts(request.visit, request.position)?;
        let taps = attention.local_taps();
        let rows = request_rows(&request, (shape, self.hidden, taps));
        validate_rows(&rows)?;
        let weights = resident_local_weights(self, &request, shape)?;
        let dense = self.resident_dense(stream, request, shape, weights);
        attention.enqueue_resident_local(
            &self.context,
            stream,
            &request,
            ResidentParts {
                shape,
                projection,
                hidden: self.hidden,
            },
            dense,
        )
    }

    /// The closure the attention half calls with its scratch rows and status words.
    fn resident_dense<'a>(
        &'a self,
        stream: &'a DeviceStream,
        request: ResidentLocalDecode<'a>,
        shape: LocalShape,
        weights: ResidentWeights<'a>,
    ) -> impl FnOnce([DeviceRange<'_>; 4], ResidentStatus<'_>) -> Result<(), EngError> + 'a {
        move |[q, relative, sinks, gates], status| {
            let inputs = match request.messages {
                Some(messages) => ResidentLocalInputs::Controlled {
                    input: request.input,
                    messages,
                    gammas: request.gammas,
                },
                None => ResidentLocalInputs::Plain {
                    input: request.input,
                    gamma: request.gammas[0],
                    zeros: status
                        .zeros
                        .ok_or_else(|| bad("missing resident identity controller rows"))?,
                },
            };
            let kv = shape.kv_channels();
            let log = |index: usize| request.log.slice(index * kv * WORD, kv * WORD);
            let [wq, wk, wv, wr, wsinks, wgates] = weights.projections;
            self.step.enqueue_local_resident(
                &self.context,
                stream,
                ResidentLocal {
                    inputs,
                    controllers: weights.controllers,
                    eps: request.math.eps,
                    weights: [wq, wk, wv, wr, wsinks, wgates],
                },
                ResidentLocalTargets {
                    projections: [q, log(0)?, log(1)?, relative, sinks, gates],
                    controller_health: status.controller_health,
                    gate_health: status.gate_health,
                },
            )
        }
    }
}
