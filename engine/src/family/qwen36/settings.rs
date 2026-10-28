//! Settings of one Q36 engine entry in the `enginesrv` config (decision 0046; packet Q36-F3a).
//!
//! The entry carries `id`, `family`, `model_dir`, `tokenizer_dir`, and `model` (server config) plus
//! the `qwen36` object, which `Q36Options` reads. A key that is absent takes its default or its
//! environment fallback. This file only reads and checks values. It opens no file and starts no
//! thread. Q36 results never satisfy Phase 0B or Phase 1B.
//!
//! The dense Qwen3.8 family uses these settings on MLX and expects the same pinned tokenizer
//! bundle as Q36. Its session cap is explicit and includes its attention and DeltaNet state.

use super::host::HostShape;
use super::host_load::rotary_dim;
use crate::family::session::StoreLimits;
use crate::family::spec::{AttentionKind, FamilySpec, QWEN3_6_35B_A3B};
use crate::{EngError, tensor::Dtype, tokenizer::TokenizerPin};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Vocabulary size of the pinned `tokenizer.json` with its added tokens.
const TOKENIZER_VOCAB_SIZE: u32 = 248_077;
/// Positions of the default context limit of the CUDA arm (decision 0046 item 6). The CUDA arm
/// sizes its default F32 cache by the limit, so it keeps 32,768 until the G-KV8 default flip (it
/// builds the 8-bit cache when `kv_dtype` is `q8`).
pub(crate) const DEFAULT_CONTEXT_LIMIT: u32 = 32_768;
/// Positions of the default context limit of the MLX arm: 262,144, the headline of decision 0046
/// A7. The 8-bit cache holds them in 2.85 GB (10,880 bytes per position with BF16 scales). The
/// gate of record Q36-KV8-GATE measured the fit, the retrieval, and the tool-call format on the
/// 36 GB Mac Studio. The default form is the 8-bit cache, so the limit belongs to that form: an
/// explicit `kv_dtype` of `f32` needs an explicit `context_limit` and `session_cap_bytes` too.
pub(crate) const MLX_DEFAULT_CONTEXT_LIMIT: u32 = 262_144;
/// The default cache cap of the MLX arm: 4 GiB admits 262,144 positions of the 8-bit form
/// (2,852,126,720 bytes with BF16 scales, 3,019,898,880 with F32 scales) plus the DeltaNet state
/// (65,863,680 bytes). The F32 form at 32,768 positions (1,342,177,280 bytes) fits as well.
pub(crate) const DEFAULT_SESSION_CAP: u64 = 4 << 30;
/// The largest chunk that the prefill plan may pick when `prefill_budget_bytes` is set. One chunk
/// takes about 0.9 s at 290 tok/s, so a cancellation waits that long at most. The plan result sets
/// the worker piece (`ExecutorOps::prefill_piece`). Without a plan, or when the plan admits no
/// chunk, the piece is `NO_BUDGET_PIECE`.
pub(crate) const DEFAULT_PREFILL_CHUNK: u32 = 256;
/// The default chunk of a BF16 engine (decision 0046 A6 item 11, amendment of 2026-10-11): 256,
/// the same as an F32 engine. Fix 5 of packet Q36-BF16-b had pinned it to 1 (no chunked prefill)
/// while the chunk-size dependence of the BF16 prefill was open; the serving shapes (chunk 256 and
/// the tails after a 256 head) measured inside the range of mlx-lm, and the pin cost about 14x on
/// prompt speed. The constant stays separate so that a re-pin is a one-line change. An explicit
/// `prefill_chunk` in the config always wins.
pub(crate) const BF16_DEFAULT_PREFILL_CHUNK: u32 = 256;
/// The fallback worker piece for an executor that plans no chunk: no budget, or a plan that admits
/// none. The executor then runs one token per step.
/// The worker checks the cancel flag only between pieces. At about 62 tok/s, a piece of 16 waits
/// about 0.26 s, and a piece of 256 would wait about 4.1 s.
pub(crate) const NO_BUDGET_PIECE: u32 = 16;
/// The largest accepted prefill chunk. It matches the decoder's `MAX_CHUNK_ROWS`; 2048 is opt-in
/// (the default stays `DEFAULT_PREFILL_CHUNK`). On the M4 Max it shortened a 49K prefill by 4%.
pub(crate) const MAX_PREFILL_CHUNK: u32 = 2048;
/// The smallest multi-token candidate in Q36's prefill admission ladder.
pub(crate) const MIN_PREFILL_CHUNK_ROWS: u32 = 4;
/// The CUDA arm cap of M_sess (lane C, decision 0049 A6 item 2): the largest chunk the
/// prefill ladder admits, independent of the MLX prefill_chunk ceiling above.
pub(crate) const DEFAULT_CUDA_PREFILL_CHUNK: u32 = 2048;

/// SHA-256 of the pinned files of `<model_dir>`
/// (`engine/testdata/mac-phase0a.json`).
const TOKENIZER_JSON_SHA256: &str =
    "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4";
const TOKENIZER_CONFIG_SHA256: &str =
    "e98f1901ac6f0adff67b1d540bfa0c36ac1a0cf59eb72ed78146ef89aafa1182";
const CHAT_TEMPLATE_SHA256: &str =
    "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259";
const GENERATION_CONFIG_SHA256: &str =
    "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e";

/// The backend a build serves Q36 from (CUDA-SERVE-1 2.4): MLX on Apple Silicon, CUDA on
/// Linux with `native-cuda`; a build with neither keeps the MLX reading, as before.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Backend {
    Mlx,
    Cuda,
}

/// The prefix-store limits of a `prefix_reuse` engine (decision 0036, addendum 2): two
/// pinned roots and six LRU entries. An engine without the flag keeps the addendum 1
/// store.
const PREFIX_STORE_LIMITS: StoreLimits = StoreLimits {
    roots: 2,
    entries: 6,
};

/// The prefix-store limits of an engine with the given process-wide `prefix_reuse` flag
/// (the one mapping the MLX rules build on, `start::rules_of`): the worker plans
/// intermediate save points only when the store holds at least two LRU entries, and the
/// engine's ready note reports that same limit from the built executor. The DEFAULT for
/// every spec stays the addendum 2 six (the eviction-cliff review verified that an
/// entry's real cost grows with its depth - a snapshot holds a zero-copy share of the
/// attention buffers and owns a depth-sized copy after the next append, while its own
/// charge is the DeltaNet state only - so a budget-derived default waits for the replay
/// measurement); `entries_override` is the operator's exact opt-in count, honored for
/// every spec that reaches the store.
pub(crate) fn prefix_store_limits_of(
    prefix_reuse: bool,
    entries_override: Option<usize>,
) -> StoreLimits {
    if !prefix_reuse {
        return StoreLimits::default();
    }
    StoreLimits {
        roots: PREFIX_STORE_LIMITS.roots,
        entries: entries_override.unwrap_or(PREFIX_STORE_LIMITS.entries),
    }
}

/// The `qwen36` object of an engine entry. Every member is optional.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Q36Options {
    /// The MLX allocator cache limit in bytes. Fallback: `Q36_CACHE_LIMIT`. There is no default.
    pub(crate) cache_limit_bytes: Option<u64>,
    /// The first position that the engine refuses, from 2. The default is 262,144 on the MLX arm
    /// (the 8-bit cache, decision 0046 A7) and 32,768 on the CUDA arm. The executor rejects a
    /// limit whose cache does not fit `session_cap_bytes`.
    pub(crate) context_limit: Option<u32>,
    /// The bytes of session state that the engine admits. Q36 defaults to 4 GiB.
    /// The dense family requires an explicit cap covering its configured context.
    pub(crate) session_cap_bytes: Option<u64>,
    /// The prompt tokens per `prefill` call, from 1 to 1024. The default is 256 for an F32 engine and
    /// for a BF16 engine (`BF16_DEFAULT_PREFILL_CHUNK`). It is the largest
    /// chunk that the planner may choose; the worker piece is the planned chunk, or 16
    /// (`NO_BUDGET_PIECE`) when no chunk is planned: no budget, or a budget that admits none.
    /// CUDA: ignored with one start line until lane M's CPF-7a integration commit wires it
    /// as the cap of `M_sess` (decision 0049, A6 item 2; lane C's ruling of 2026-10-14).
    pub(crate) prefill_chunk: Option<u32>,
    /// The engine memory budget in bytes for the admission of the chunked prefill (decision 0044
    /// item 6), from 1. Absent means off: every prompt token is one step. The executor picks the
    /// largest chunk of at most `prefill_chunk` rows whose workspace fits the budget after the
    /// weights and the reserve.
    pub(crate) prefill_budget_bytes: Option<u64>,
    /// The bytes that the planner keeps free for the session state, the MLX cache, and the
    /// pressure margin. It has no default: it is required with `prefill_budget_bytes` and refused
    /// without it. A value of 0 is allowed and admits a larger chunk than the memory plan allows.
    pub(crate) prefill_session_reserve_bytes: Option<u64>,
    /// The dtype of the decode activations (packet Q36-BF16-a): `f32` or `bf16`. The default is
    /// `bf16`. It also picks the scale storage: `bf16` keeps the stored BF16 scales and biases,
    /// `f32` keeps the G0 widening, and the decoder constructor refuses any other pairing.
    pub(crate) activation_dtype: Option<String>,
    /// The form of the attention KV cache (packet KV8-INTEGRATE, decision 0046 A7): `f32`, the
    /// pair of F32 capacity buffers, or `q8`, the 8-bit affine codes with scales and biases of
    /// the activation dtype. The default is `q8` on the MLX arm (the user's decision of
    /// 2026-10-13 after the A7 rule-4 Mac gate, record Q36-KV8-GATE) and `f32` on the CUDA arm
    /// until the G-KV8 default flip; the CUDA arm builds the 8-bit cache (Int8G32) when the value is
    /// `q8` (KV8-B). `f32` stays the control on the MLX arm.
    pub(crate) kv_dtype: Option<String>,
    /// Opt-in host-backed CUDA Q8 KV budget, including pinned staging and owned metadata.
    pub(crate) kv_offload_host_bytes: Option<u64>,
    /// Graph capture of the CUDA arm (default false, gate G1 eager); the MLX arm ignores it.
    pub(crate) capture: Option<bool>,
    /// `mlx_compile` on the Q36 DeltaNet and device-routed MoE decode blocks (packet
    /// MX-COMPILE-1; the amendment of decision 0039 item 8 that decision 0046 A5 reopened): the
    /// engine traces both blocks once after the weights load and runs every one-token decode
    /// step through the compiled closures, which gains 9.6 to 13.0 percent of decode speed
    /// (measured at depths 30,000 to 16) and moves F32 logits by at most 5e-5 (compile is
    /// deterministic run to run but not bit-identical to the uncompiled path). The default is
    /// `false`; the shipped Mac config sets `true` (user decision 2026-10-14 after the Mac gates
    /// of MX-COMPILE-1a). The MLX arm reads it; the CUDA arm records it as ignored.
    pub(crate) compile: Option<bool>,
    /// Concurrent MLX sessions, from 1 to 8. The default is 1; larger windows require compile.
    pub(crate) max_seqs: Option<usize>,
    /// The exact LRU entry count of the prefix store of a `prefix_reuse` engine, as the
    /// operator's OPT-IN override (the eviction-cliff slice). The default for every spec
    /// stays the fixed six of decision 0036 addendum 2 - an entry's real cost grows with
    /// its depth (a snapshot holds a share of the attention buffers and owns a depth-sized
    /// copy after the next append; its own charge is the DeltaNet state only), so a
    /// budget-derived default waits for the replay measurement. Validated from 1 to 4096
    /// at parse; ignored with a start line when `prefix_reuse` is off, and by the CUDA arm
    /// (an MLX key).
    pub(crate) prefix_store_entries: Option<usize>,
    /// The tiered expert mode of the CUDA arm (OFF-2). Absent means off: the routed experts are
    /// all resident on the device, as before. The MLX arm refuses the member.
    pub(crate) tiered: Option<TieredOptions>,
}

/// The `tiered` member of the `qwen36` object. The CPU tier maps the model file read-only for the whole
/// serve, so the operator's contract is that the model file must not change while it is served: no
/// truncation (the next touch of a lost page faults with SIGBUS) and no in-place overwrite (the CPU tier
/// would read the new bytes without any fault).
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct TieredOptions {
    /// Expert bank slots per layer, from 8 (the picks per token) to 256. Required: the number
    /// depends on the card, the context and the desktop, so no default covers it.
    pub(crate) slots: Option<u32>,
    /// CPU worker threads, from 1 to 8. The default is 6.
    pub(crate) threads: Option<u32>,
    /// The longest wait for one layer's CPU experts, in milliseconds, from 1 to 600,000. The
    /// default is 10,000.
    pub(crate) timeout_ms: Option<u32>,
    /// Fill every layer's bank with its lowest-numbered experts before the first token. The
    /// default is false: the banks start empty and policy P fills them.
    pub(crate) preload: Option<bool>,
}

/// The checked tiered settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TieredSettings {
    pub(crate) slots: usize,
    pub(crate) threads: usize,
    pub(crate) timeout_ms: u32,
    pub(crate) preload: bool,
}

/// The environment fallbacks, read by the caller so that this file stays pure.
#[derive(Debug, Default)]
pub(crate) struct Q36Env {
    /// `Q36_CACHE_LIMIT`, a byte count.
    pub(crate) cache_limit: Option<String>,
}

/// The dtype of the Q36 decode activations (packet Q36-BF16-a): the hidden state between blocks,
/// the inputs of the quantized matmuls, and the converted layer norms. `bf16` is the serving
/// default (user decision of 2026-10-10; decision 0046 A6); `f32` is the qualification control.
/// Scale storage follows the same setting, so the two are one lever.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Activation {
    F32,
    Bf16,
}

/// The config text of `activation`.
pub(crate) fn activation_name(activation: Activation) -> &'static str {
    match activation {
        Activation::F32 => "f32",
        Activation::Bf16 => "bf16",
    }
}

impl Activation {
    /// The resident dtype of this activation: the value that `Consts::act` holds and that the
    /// kernels take as operands.
    pub(crate) fn dtype(self) -> Dtype {
        match self {
            Activation::F32 => Dtype::F32,
            Activation::Bf16 => Dtype::Bf16,
        }
    }

    /// The activation whose resident dtype is `dtype`, the inverse of `dtype` (packet
    /// KV8-INTEGRATE): the resident state reads the activation back from `Consts::act` when it
    /// charges the cache. Any other dtype is `None`.
    pub(crate) fn of_dtype(dtype: Dtype) -> Option<Self> {
        match dtype {
            Dtype::F32 => Some(Self::F32),
            Dtype::Bf16 => Some(Self::Bf16),
            _ => None,
        }
    }
}

/// The group size of the Q8 cache form, along the head dimension (decision 0046 A7): 64 lanes
/// share one scale and one bias. `head_dim` must be a multiple of this; the decoder constructor
/// refuses a Q8 engine whose head dimension is not (packet KV8-INTEGRATE item 2). The one
/// definition: the byte charge below, and the quantize, the append, and the attention of the Q8
/// form, all read it.
pub(crate) const Q8_GROUP: u32 = 64;

/// The bits of the Q8 cache form (decision 0046 A7): 8-bit affine codes, four per U32 word. The
/// code width also sits in the shapes of the cache buffers (`attn.rs` divides the head dimension
/// by 4) and in the byte charge of `kv_growth_per_position` (one byte per code), so a change of
/// this constant needs those places and the portable check of `quantize_rows` (packet KV8-OPS).
pub(crate) const Q8_BITS: u32 = 8;

/// The LOGICAL bytes that one more position adds to the cache of ONE attention layer (packet
/// KV8-INTEGRATE item 5), from the cache form and the activation dtype — the one byte authority:
/// `attn::growth_bytes` (which `state::kv_growth_bytes` calls) and `cache_growth_per_position`
/// reach it.
///
/// - `F32`: keys and values `[1, KVH, 1, D]` F32, `KVH * D * 2 * 4` bytes. The real model
///   (KVH 2, D 256) adds 4,096 bytes per layer and position and 40,960 over its 10 attention
///   layers.
/// - `Q8`: per side (keys, values) codes `[1, KVH, 1, D/4]` U32 of `KVH * D` bytes, plus scales
///   and biases `[1, KVH, 1, D/64]` of the ACTIVATION dtype, `2 * KVH * (D/64) * scale_bytes`
///   bytes per side. The real model adds 1,088 bytes per layer and position at BF16 scales
///   (10,880 over the 10 layers) and 1,152 at F32 scales (11,520).
///
/// A Q8 form whose `head_dim` is not a multiple of `Q8_GROUP` (64) is `BadRequest`. An overflow
/// is `MemoryPlan`.
pub(crate) fn kv_growth_per_position(
    kv_heads: u32,
    head_dim: u32,
    kv: KvDtype,
    act: Activation,
) -> Result<u64, EngError> {
    if kv == KvDtype::Q8 && !head_dim.is_multiple_of(Q8_GROUP) {
        return Err(bad(format!(
            "kv_dtype q8 needs a head_dim that is a multiple of {Q8_GROUP}, got {head_dim}"
        )));
    }
    let scale_bytes = act.dtype().byte_width();
    let heads = u64::from(kv_heads);
    let width = u64::from(head_dim);
    let bytes = match kv {
        // Keys and values `[1, KVH, 1, D]` F32.
        KvDtype::F32 => heads
            .checked_mul(width)
            .and_then(|row| row.checked_mul(2))
            .and_then(|bytes| bytes.checked_mul(4)),
        // Keys and values `[1, KVH, 1, D]` BF16: the F32 shapes at half storage.
        KvDtype::Bf16 => heads
            .checked_mul(width)
            .and_then(|row| row.checked_mul(2))
            .and_then(|bytes| bytes.checked_mul(2)),
        // Per side: codes `[1, KVH, 1, D/4]` U32 (`KVH * D` bytes), plus scales and biases
        // `[1, KVH, 1, D/64]` (`2 sides * 2 arrays * KVH * groups * scale_bytes`).
        KvDtype::Q8 => heads.checked_mul(width).and_then(|codes| {
            let codes = codes.checked_mul(2)?;
            let params = width
                .checked_div(u64::from(Q8_GROUP))
                .and_then(|groups| heads.checked_mul(groups))
                .and_then(|n| n.checked_mul(2))
                .and_then(|n| n.checked_mul(2))
                .and_then(|n| n.checked_mul(scale_bytes))?;
            codes.checked_add(params)
        }),
    };
    bytes.ok_or_else(|| EngError::MemoryPlan("attention growth bytes overflow u64".into()))
}

/// Parses the `activation_dtype` member (packet Q36-BF16-a): `f32` or `bf16`. Any other text is
/// refused in the style of the numeric members.
pub(crate) fn parse_activation(text: &str) -> Result<Activation, EngError> {
    match text {
        "f32" => Ok(Activation::F32),
        "bf16" => Ok(Activation::Bf16),
        other => Err(bad(format!(
            "activation_dtype must be f32 or bf16, got {other:?}"
        ))),
    }
}

/// The activation dtype of `text`, the raw member value: `None` is the default `bf16`.
pub(crate) fn activation_of(text: Option<&str>) -> Result<Activation, EngError> {
    match text {
        None => Ok(Activation::Bf16),
        Some(text) => parse_activation(text),
    }
}

/// The scale-pairing rule of packet Q36-BF16-a: the stored scales and biases (counted by
/// `ResidentWeights::scale_census`) must all have the dtype that matches `activation`, because the
/// setting is one lever. A `bf16` engine over F32 scales promotes every `qmm` back to F32, and an
/// `f32` engine over BF16 scales makes MLX cast the scales on every call (about 31.6 tok/s, the
/// trap that the G0 measurement closed). Returns the refusal text of a mismatched pair.
pub(crate) fn scale_pairing_error(
    weights: u64,
    f32_parts: u64,
    bf16_parts: u64,
    other_parts: u64,
    activation: Activation,
) -> Option<String> {
    let (name, scale_dtype, wrong) = match activation {
        Activation::F32 => ("f32", "F32", bf16_parts + other_parts),
        Activation::Bf16 => ("bf16", "BF16", f32_parts + other_parts),
    };
    (wrong > 0).then(|| {
        format!(
            "activation_dtype {name} needs {scale_dtype} scales and biases; the census holds \
             {weights} packed weights with {f32_parts} F32, {bf16_parts} BF16, and {other_parts} other parts"
        )
    })
}

/// The form of the attention KV cache of one engine (packet KV8-INTEGRATE; decision 0046 A7):
/// `F32` keeps the pair of F32 capacity buffers `[1, KVH, cap, D]` (4,096 bytes per layer and
/// position on the real model, 40,960 over its 10 attention layers); `Q8` keeps the 8-bit affine
/// codes `[1, KVH, cap, D/4]` beside scales and biases `[1, KVH, cap, D/64]` of the ACTIVATION
/// dtype (1,088 bytes per layer and position at BF16 scales, 1,152 at F32 scales; 10,880 and
/// 11,520 over the 10 attention layers). `F32` is the default and the control that the A7
/// quality gate compares against; the shipped default flips to `Q8` only after that gate
/// passes (A7 rule 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KvDtype {
    /// The F32 capacity buffers, the control.
    F32,
    /// The BF16 capacity buffers: the F32 path's shapes at half storage (KVH 2 x D 256 over
    /// the 10 full-attention layers = ~20 KiB per position vs 40 for F32), appended with a
    /// cast and read natively behind the adapter's existing `allow_bf16` - the same KV form
    /// mlx-lm's default arm runs. The decode-gap probe; fits every depth inside the KV budget
    /// (5.0 GB at 262,144).
    Bf16,
    /// The 8-bit affine codes, scales, and biases (group 64, bits 8).
    Q8,
}

/// The config text of `kv_dtype`.
pub(crate) fn kv_dtype_name(kv: KvDtype) -> &'static str {
    match kv {
        KvDtype::F32 => "f32",
        KvDtype::Bf16 => "bf16",
        KvDtype::Q8 => "q8",
    }
}

/// Parses the `kv_dtype` member (packet KV8-INTEGRATE): `f32` or `q8`. Any other text is refused
/// in the style of the numeric members.
pub(crate) fn parse_kv_dtype(text: &str) -> Result<KvDtype, EngError> {
    match text {
        "f32" => Ok(KvDtype::F32),
        "bf16" => Ok(KvDtype::Bf16),
        "q8" => Ok(KvDtype::Q8),
        other => Err(bad(format!(
            "kv_dtype must be f32, bf16, or q8, got {other:?}"
        ))),
    }
}

/// The cache form of `text`, the raw member value. `None` is the default of the arm: the 8-bit
/// cache on the MLX arm (decision 0046 A7, gate of record Q36-KV8-GATE) and the F32 cache on the
/// CUDA arm until the G-KV8 default flip. An explicit `q8` is valid on both arms.
pub(crate) fn kv_dtype_of(text: Option<&str>, backend: Backend) -> Result<KvDtype, EngError> {
    match (text, backend) {
        (Some(text), _) => parse_kv_dtype(text),
        (None, Backend::Mlx) => Ok(KvDtype::Q8),
        (None, Backend::Cuda) => Ok(KvDtype::F32),
    }
}

/// Whether the chunked prefill may run with the activation dtype `act` (packet Q36-BF16-b): the
/// chunk path is dtype aware — its blocks run the same boundary casts as the decode step and the
/// chunk loop guards every dtype — so both activation dtypes plan a chunk and only the budget and
/// the largest chunk can keep one out. This is the one authority for the rule:
/// `Q36Executor::plan_prefill` asks it with `Consts::act`, and the start-up line of
/// `start::plan_prefill` follows the plan and names no dtype rule.
pub(crate) fn chunked_prefill_allowed(act: Dtype) -> bool {
    matches!(act, Dtype::F32 | Dtype::Bf16)
}

/// The checked settings of one Q36 engine.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Q36Settings {
    /// The selected descriptor, shared by admission and executor loading.
    pub(crate) spec: FamilySpec,
    /// The checkpoint directory with the safetensors shards and `config.json`.
    pub(crate) artifact: PathBuf,
    /// The directory with `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, and
    /// `generation_config.json`.
    pub(crate) tokenizer_dir: PathBuf,
    /// The MLX allocator cache limit in bytes.
    pub(crate) cache_limit: u64,
    /// The first position that the engine refuses.
    pub(crate) context_limit: u32,
    /// The bytes of cache growth that the engine admits.
    pub(crate) session_cap: u64,
    /// The prompt tokens per `prefill` call. MLX: with a budget it is also the largest planned
    /// chunk. CUDA: always 0 (the key is ignored with one start line until CPF-7a's
    /// integration commit, decision 0049 A6 item 2); the chunk cap the step holds is
    /// `DEFAULT_CUDA_PREFILL_CHUNK`, set through `QwenResidentStep::set_m_sess_cap`.
    pub(crate) prefill_chunk: u32,
    /// The budget of the chunked prefill, or `None` when the chunked prefill is off.
    pub(crate) prefill_budget: Option<u64>,
    /// The bytes that the planner keeps free. It is used only when `prefill_budget` is set.
    pub(crate) prefill_reserve: u64,
    /// The dtype of the decode activations (packet Q36-BF16-a), which also picks the scale storage.
    /// The default is `bf16`.
    pub(crate) activation: Activation,
    /// The form of the attention KV cache (packet KV8-INTEGRATE). The default is `f32`, the
    /// control.
    pub(crate) kv: KvDtype,
    /// Host-backed CUDA Q8 KV is disabled when this budget is absent.
    pub(crate) kv_offload_host_bytes: Option<u64>,
    /// Graph capture of the CUDA arm; the MLX arm ignores it with one start line.
    pub(crate) capture: bool,
    /// `mlx_compile` on the DeltaNet and device-routed MoE decode blocks (packet MX-COMPILE-1).
    /// The default is `false`; the MLX arm reads it, admits the model at load, and warms one
    /// program per distinct compiled block, and the CUDA arm records the key as ignored.
    pub(crate) compile: bool,
    /// Maximum concurrent sessions. One preserves the original single-job worker.
    pub(crate) max_seqs: usize,
    /// The exact LRU entry count override of the prefix store, or `None` for the default
    /// addendum 2 cap of six (the eviction-cliff slice). Validated by the parse;
    /// `prefix_store_limits_of` applies it.
    pub(crate) prefix_store_entries: Option<usize>,
    /// The backend these settings were read for. The `prefix_reuse` cap rule applies to the MLX
    /// arm only.
    pub(crate) backend: Backend,
    /// The process-wide `prefix_reuse` flag (decision 0036), set by the factory after `new`:
    /// with the flag on, this engine's prefix store keeps the addendum 2 limits (two pinned
    /// roots, six LRU entries); with it off, the addendum 1 store (zero roots, one entry). No
    /// config member reads it: the flag belongs to the server configuration file, and `new`
    /// leaves it false.
    pub(crate) prefix_reuse: bool,
    /// The tiered expert mode, or `None` for the plain engine. Never on without the member.
    pub(crate) tiered: Option<TieredSettings>,
    /// The MLX allocator and session keys the CUDA arm ignores (the KV cache is sized by
    /// `context_limit` alone, the prefill runs one token per step): the start logs one line
    /// each. Empty on the MLX arm. `prefill_chunk` is among them until lane M's CPF-7a
    /// integration commit accepts it as the cap of `M_sess` (decision 0049, A6 item 2).
    pub(crate) ignored_mlx_keys: Vec<&'static str>,
}

fn bad(message: String) -> EngError {
    EngError::BadRequest(format!("engine config: {message}"))
}

/// `value` or `default`, which must lie in `low..=high`.
fn ranged<T: Copy + PartialOrd + std::fmt::Display>(
    key: &str,
    value: Option<T>,
    default: T,
    low: T,
    high: T,
) -> Result<T, EngError> {
    let value = value.unwrap_or(default);
    if value >= low && value <= high {
        Ok(value)
    } else {
        Err(bad(format!(
            "{key} must be between {low} and {high}, found {value}"
        )))
    }
}

fn cache_limit(options: &Q36Options, env: &Q36Env) -> Result<u64, EngError> {
    let limit = match options.cache_limit_bytes {
        Some(value) => value,
        None => env
            .cache_limit
            .as_deref()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                bad("cache_limit_bytes is missing and Q36_CACHE_LIMIT is not set".into())
            })?
            .parse::<u64>()
            .map_err(|_| bad("Q36_CACHE_LIMIT is not a byte count".into()))?,
    };
    if limit == 0 {
        return Err(bad("the cache limit must be above 0".into()));
    }
    Ok(limit)
}

/// `value` when it is set, which must be at least 1.
fn at_least_one(key: &str, value: Option<u64>) -> Result<Option<u64>, EngError> {
    value
        .map(|value| ranged(key, Some(value), 1, 1, u64::MAX))
        .transpose()
}

/// The reserve of the chunked prefill. A budget needs the member, because no default covers the
/// session state, the MLX cache, and the pressure margin of the memory plan. A reserve without a
/// budget is a mistake, because nothing would read it. With neither, the prefill runs one token
/// at a time, nothing reads the reserve, and it stays 0.
fn prefill_reserve(options: &Q36Options, budget: Option<u64>) -> Result<u64, EngError> {
    match (options.prefill_session_reserve_bytes, budget) {
        (Some(_), None) => Err(bad(
            "prefill_session_reserve_bytes needs prefill_budget_bytes".into(),
        )),
        (None, Some(_)) => Err(bad(
            "prefill_budget_bytes needs prefill_session_reserve_bytes".into(),
        )),
        (Some(reserve), Some(_)) => Ok(reserve),
        (None, None) => Ok(0),
    }
}

/// The DeltaNet state bytes of one model: per DeltaNet layer the recurrent state
/// `[value_heads, value_head_dim, key_head_dim]` and the convolution ring
/// `[conv_dim, conv_kernel - 1]`, F32. It is derived from the family spec, never a literal:
/// `QWEN3_6_35B_A3B` holds 65,863,680 bytes in its 30 DeltaNet layers.
fn delta_state_bytes(spec: &FamilySpec) -> Result<u64, EngError> {
    let Some(linear) = spec.linear else {
        return Err(bad(
            "the model has no linear-attention state to reserve".into()
        ));
    };
    let rec = u64::from(linear.value_heads)
        .checked_mul(u64::from(linear.value_head_dim))
        .and_then(|elems| elems.checked_mul(u64::from(linear.key_head_dim)))
        .and_then(|elems| elems.checked_mul(4))
        .ok_or_else(|| bad("the DeltaNet state bytes overflow u64".into()))?;
    let ring = u64::from(linear.conv_dim()?)
        .checked_mul(u64::from(linear.conv_kernel - 1))
        .and_then(|elems| elems.checked_mul(4))
        .ok_or_else(|| bad("the DeltaNet state bytes overflow u64".into()))?;
    let per_layer = rec
        .checked_add(ring)
        .ok_or_else(|| bad("the DeltaNet state bytes overflow u64".into()))?;
    let layers = spec
        .layers_attention
        .iter()
        .filter(|kind| **kind == AttentionKind::DeltaNet)
        .count();
    u64::try_from(layers)
        .ok()
        .and_then(|layers| per_layer.checked_mul(layers))
        .ok_or_else(|| bad("the DeltaNet state bytes overflow u64".into()))
}

/// The cache bytes that one position adds over every full-attention layer, in the cache form
/// `kv` (packet KV8-INTEGRATE): the F32 control is keys and values `[1, heads_kv, 1, head_dim]`
/// F32, and the Q8 form is the 8-bit codes beside scales and biases of the activation dtype
/// (`kv_growth_per_position`, the one authority). It is derived from the family spec, never a
/// literal: `QWEN3_6_35B_A3B` adds 40,960 bytes per position in its 10 attention layers in the
/// F32 form, 10,880 at Q8 with BF16 scales, and 11,520 at Q8 with F32 scales.
fn cache_growth_per_position(
    spec: &FamilySpec,
    kv: KvDtype,
    act: Activation,
) -> Result<u64, EngError> {
    let per_layer = kv_growth_per_position(spec.heads_kv, spec.head_dim, kv, act)?;
    let layers = spec
        .layers_attention
        .iter()
        .filter(|kind| **kind == AttentionKind::Full)
        .count();
    u64::try_from(layers)
        .ok()
        .and_then(|layers| per_layer.checked_mul(layers))
        .ok_or_else(|| bad("the cache growth per position overflows u64".into()))
}

/// The tiered member, checked: CUDA only, `slots` required, no capture (the tiered step waits for
/// the host inside every layer, which a captured graph cannot do), no chunked prefill (the CUDA
/// arm refuses the budget keys by name above, and the tiered prefill runs one token per step;
/// `chunk_ready` keeps a tiered session off the chunk path, plan rev 3 §C.4).
fn tiered_settings(
    options: &Q36Options,
    backend: Backend,
) -> Result<Option<TieredSettings>, EngError> {
    let Some(tiered) = options.tiered else {
        return Ok(None);
    };
    if backend == Backend::Mlx {
        return Err(bad("tiered is a CUDA member; the MLX arm refuses it".into()));
    }
    if options.capture == Some(true) {
        return Err(bad(
            "tiered and capture exclude each other: the tiered step is eager only".into(),
        ));
    }
    let slots = tiered
        .slots
        .ok_or_else(|| bad("tiered needs slots (bank slots per layer, 8 to 256)".into()))?;
    Ok(Some(TieredSettings {
        slots: usize::try_from(ranged("tiered.slots", Some(slots), 8, 8, 256)?)
            .map_err(|_| bad("tiered.slots exceeds usize".into()))?,
        threads: usize::try_from(ranged("tiered.threads", tiered.threads, 6, 1, 8)?)
            .map_err(|_| bad("tiered.threads exceeds usize".into()))?,
        timeout_ms: ranged("tiered.timeout_ms", tiered.timeout_ms, 10_000, 1, 600_000)?,
        preload: tiered.preload.unwrap_or(false),
    }))
}

/// Refuses one MLX chunked-prefill key on the CUDA arm, naming it: the arm prefills one token
/// per step until CPF lands, so accepting the key would promise a speed it does not have.
fn refuse_cuda_key(key: &str, value: Option<u64>) -> Result<(), EngError> {
    if value.is_some() {
        return Err(bad(format!(
            "{key} is an MLX chunked-prefill key; the CUDA arm prefills one token per step until CPF"
        )));
    }
    Ok(())
}

/// The MLX allocator, compile, and session keys the CUDA arm ignores (the KV cache is sized by
/// `context_limit` alone, the prefill runs one token per step, and the CUDA arm does not build
/// the MLX compile seam): the start logs one line each. `prefill_chunk` stays ignored until
/// lane M's CPF-7a integration commit accepts it as the cap of `M_sess` (decision 0049, A6
/// item 2; lane C's ruling of 2026-10-14: the key promises no speed the arm does not have).
fn ignored_mlx_keys_of(options: &Q36Options) -> Vec<&'static str> {
    [
        (options.cache_limit_bytes.is_some(), "cache_limit_bytes"),
        (options.session_cap_bytes.is_some(), "session_cap_bytes"),
        (options.prefill_chunk.is_some(), "prefill_chunk"),
        (options.compile.is_some(), "compile"),
        (
            options.prefix_store_entries.is_some(),
            "prefix_store_entries",
        ),
    ]
    .into_iter()
    .filter(|&(set, _)| set)
    .map(|(_, key)| key)
    .collect()
}

fn kv_offload_of(options: &Q36Options, backend: Backend) -> Result<Option<u64>, EngError> {
    let Some(bytes) = at_least_one("kv_offload_host_bytes", options.kv_offload_host_bytes)? else {
        return Ok(None);
    };
    if backend != Backend::Cuda {
        return Err(bad("kv_offload_host_bytes is CUDA-only".into()));
    }
    if kv_dtype_of(options.kv_dtype.as_deref(), backend)? != KvDtype::Q8 {
        return Err(bad("kv_offload_host_bytes requires kv_dtype q8".into()));
    }
    if options.capture.unwrap_or(false) || options.tiered.is_some() {
        return Err(bad(
            "kv_offload_host_bytes requires capture=false and no tiered experts".into(),
        ));
    }
    Ok(Some(bytes))
}

impl Q36Settings {
    /// Checks the entry and its `qwen36` object (absent means all members absent); the config
    /// value wins over the environment. The backend fixes which keys are read: `Cuda` refuses
    /// the chunked-prefill budget keys by name and makes the allocator and session keys
    /// optional (ignored, recorded for the start lines); `Mlx` reads them all.
    pub(crate) fn new(
        model_dir: &Path,
        tokenizer_dir: &Path,
        options: Option<&Q36Options>,
        env: &Q36Env,
        backend: Backend,
    ) -> Result<Self, EngError> {
        Self::new_for_family(
            model_dir,
            tokenizer_dir,
            options,
            env,
            backend,
            QWEN3_6_35B_A3B,
        )
    }

    /// Reads the shared options for the selected descriptor. Dense execution is MLX-only.
    pub(crate) fn new_for_family(
        model_dir: &Path,
        tokenizer_dir: &Path,
        options: Option<&Q36Options>,
        env: &Q36Env,
        backend: Backend,
        spec: FamilySpec,
    ) -> Result<Self, EngError> {
        if spec.feed_forward.is_some() && backend == Backend::Cuda {
            return Err(bad(format!(
                "{} requires the native-mlx backend on Apple Silicon; the CUDA Q36 loader does not support the dense family",
                spec.name
            )));
        }
        let default = Q36Options::default();
        let options = options.unwrap_or(&default);
        let kv_offload_host_bytes = kv_offload_of(options, backend)?;
        let (prefill_budget, prefill_reserve, activation, kv, cache_limit, ignored_mlx_keys) =
            match backend {
                Backend::Mlx => {
                    let budget =
                        at_least_one("prefill_budget_bytes", options.prefill_budget_bytes)?;
                    let reserve = prefill_reserve(options, budget)?;
                    let activation = activation_of(options.activation_dtype.as_deref())?;
                    let kv = kv_dtype_of(options.kv_dtype.as_deref(), Backend::Mlx)?;
                    (
                        budget,
                        reserve,
                        activation,
                        kv,
                        cache_limit(options, env)?,
                        Vec::new(),
                    )
                }
                Backend::Cuda => {
                    refuse_cuda_key("prefill_budget_bytes", options.prefill_budget_bytes)?;
                    refuse_cuda_key(
                        "prefill_session_reserve_bytes",
                        options.prefill_session_reserve_bytes,
                    )?;
                    let activation = activation_of(options.activation_dtype.as_deref())?;
                    // KV8-B B4: unlike main, the arm does NOT refuse `q8` here — the CUDA
                    // start maps the parsed form onto the step's `KvMode` (F32 to the fp16
                    // control cache, Q8 to Int8G32) and refuses the Tiled layout there.
                    let kv = kv_dtype_of(options.kv_dtype.as_deref(), Backend::Cuda)?;
                    (None, 0, activation, kv, 0, ignored_mlx_keys_of(options))
                }
            };
        // The MLX error order of the base: budget, reserve, activation_dtype, kv_dtype, cache_limit,
        // context_limit, session_cap, prefill_chunk (KUBDAN-MSG-71 item 4). `cache_limit` is
        // computed in the match above, BEFORE `context_limit` and the second match, so a config
        // with both faults names the cache-limit fault first; the CUDA arm refuses its keys in
        // the first match, reads no cache limit, and takes only `context_limit` below.
        let context_limit = ranged(
            "context_limit",
            options.context_limit,
            spec.feed_forward.map_or(
                match backend {
                    Backend::Mlx => MLX_DEFAULT_CONTEXT_LIMIT,
                    Backend::Cuda => DEFAULT_CONTEXT_LIMIT,
                },
                |dense| dense.context_limit,
            ),
            2,
            spec.feed_forward
                .map_or(u32::MAX, |dense| dense.context_limit),
        )?;
        if spec.feed_forward.is_some() {
            let cache = cache_growth_per_position(&spec, kv, activation)?
                .checked_mul(u64::from(context_limit))
                .ok_or_else(|| bad("the session cache bytes overflow u64".into()))?;
            let fixed = delta_state_bytes(&spec)?;
            let required = cache
                .checked_add(fixed)
                .ok_or_else(|| bad("the session cap requirement overflows u64".into()))?;
            if options.session_cap_bytes.is_none_or(|cap| cap < required) {
                return Err(bad(format!(
                    "{} needs explicit session_cap_bytes of at least {required} bytes for context limit {context_limit}: {cache} attention cache bytes plus {fixed} DeltaNet bytes; set or raise session_cap_bytes",
                    spec.name
                )));
            }
        }
        let (session_cap, prefill_chunk) = match backend {
            Backend::Mlx => (
                ranged(
                    "session_cap_bytes",
                    options.session_cap_bytes,
                    DEFAULT_SESSION_CAP,
                    1,
                    u64::MAX,
                )?,
                ranged(
                    "prefill_chunk",
                    options.prefill_chunk,
                    if activation == Activation::Bf16 {
                        BF16_DEFAULT_PREFILL_CHUNK
                    } else {
                        DEFAULT_PREFILL_CHUNK
                    },
                    1,
                    MAX_PREFILL_CHUNK,
                )?,
            ),
            // Inert: the CUDA arm reads none of them (the KV is sized by `context_limit`,
            // the worker piece is `NO_BUDGET_PIECE`, `prefill_chunk` is ignored with its
            // start line until CPF-7a's integration commit, decision 0049 A6 item 2).
            Backend::Cuda => (0, 0),
        };
        let tiered = tiered_settings(options, backend)?;
        let max_seqs = ranged("max_seqs", options.max_seqs, 1, 1, 8)?;
        // The override validates at boot through `ranged` (the same message shape as every
        // other member) so a typo cannot reach the store; the range keeps a fat finger from
        // asking for a million live slots.
        let prefix_store_entries = match options.prefix_store_entries {
            None => None,
            Some(value) => Some(ranged("prefix_store_entries", Some(value), value, 1, 4096)?),
        };
        if max_seqs > 1 {
            if backend != Backend::Mlx {
                return Err(bad("max_seqs above 1 requires the MLX backend".into()));
            }
            if spec.feed_forward.is_some() {
                return Err(bad(
                    "max_seqs above 1 requires MoE device routing; the dense family runs one sequence at a time"
                        .into(),
                ));
            }
            if !options.compile.unwrap_or(false) {
                return Err(bad("max_seqs above 1 requires compile=true".into()));
            }
            if prefill_budget.is_none() || prefill_chunk < MIN_PREFILL_CHUNK_ROWS {
                return Err(bad(format!(
                    "max_seqs above 1 requires prefill_budget_bytes and prefill_chunk at least {MIN_PREFILL_CHUNK_ROWS}"
                )));
            }
            if prefill_budget.is_some_and(|budget| prefill_reserve.max(session_cap) >= budget) {
                return Err(bad(
                    "max_seqs above 1 requires prefill_budget_bytes above both prefill_session_reserve_bytes and session_cap_bytes"
                        .into(),
                ));
            }
        }
        Ok(Self {
            spec,
            artifact: model_dir.to_path_buf(),
            tokenizer_dir: tokenizer_dir.to_path_buf(),
            cache_limit,
            context_limit,
            session_cap,
            prefill_chunk,
            prefill_budget,
            prefill_reserve,
            activation,
            kv,
            kv_offload_host_bytes,
            capture: options.capture.unwrap_or(false),
            // The CUDA arm has no MLX compile seam: the key stays false there and
            // `ignored_mlx_keys_of` records it for the start line instead.
            compile: match backend {
                Backend::Mlx => options.compile.unwrap_or(false),
                Backend::Cuda => false,
            },
            max_seqs,
            prefix_store_entries,
            backend,
            prefix_reuse: false,
            tiered,
            ignored_mlx_keys,
        })
    }

    /// The `prefix_reuse` cap rule (decision 0036, addendum 1 item 7; fix round Q36-PR-1 item 1).
    /// With the process-wide `prefix_reuse` flag on, one saved entry and the live session share
    /// this engine's `SessionBudget`: the entry charges its attention caches plus the fixed
    /// DeltaNet state, and the live session can hold the same caches again up to the context
    /// limit. The cap must therefore cover `2 * growth_per_position * context_limit +
    /// fixed_bytes` (2,750,218,240 bytes at the default context limit of 32,768 in the default
    /// F32 cache form; the growth follows `kv_dtype`, so a Q8 engine with BF16 scales needs
    /// 2 * 10,880 * 32,768 + 65,863,680), or a later job fails with `MemoryPlan` although the
    /// same job succeeds with `prefix_reuse` off. Engine start refuses a smaller cap and answers
    /// with the cap, the requirement, the context limit, and the hint `raise session_cap_bytes`.
    /// With the flag off the rule does not apply. It does not apply to the CUDA arm either: that
    /// arm ignores `session_cap_bytes` (the value is 0), and its entry lives in host RAM bounded
    /// by `context_limit` (decision 0036, addendum 2, CUDA arm). Roots and periodic entries
    /// (decision 0036, addendum 2, item 6) do not raise this requirement: they are best effort
    /// inside the cap, a denied save skips only that save, and the requirement keeps covering
    /// exactly the live session plus the one tip entry at the context limit.
    pub(crate) fn check_prefix_reuse_cap(&self, prefix_reuse: bool) -> Result<(), EngError> {
        if prefix_reuse && self.kv_offload_host_bytes.is_some() {
            return Err(bad(
                "kv_offload_host_bytes does not support prefix_reuse; set prefix_reuse=false"
                    .into(),
            ));
        }
        if !prefix_reuse || self.backend == Backend::Cuda {
            return Ok(());
        }
        let growth = cache_growth_per_position(&self.spec, self.kv, self.activation)?;
        let fixed = delta_state_bytes(&self.spec)?;
        let required = growth
            .checked_mul(2)
            .and_then(|twice| twice.checked_mul(u64::from(self.context_limit)))
            .and_then(|cache| cache.checked_add(fixed))
            .ok_or_else(|| bad("the prefix_reuse cap requirement overflows u64".into()))?;
        if self.session_cap < required {
            let (cap, limit) = (self.session_cap, self.context_limit);
            let message = format!(
                "prefix_reuse needs session_cap_bytes of at least {required} bytes for context limit {limit}: 2 * {growth} cache bytes per position plus {fixed} DeltaNet bytes; the cap is {cap} bytes; raise session_cap_bytes"
            );
            return Err(bad(message));
        }
        Ok(())
    }
}

/// The executor's checked shape, before any shard opens or any weight is imported.
pub(super) fn shape_of(config: &[u8], spec: &FamilySpec) -> Result<HostShape, EngError> {
    spec.check_config(config)?;
    HostShape::from_spec(spec, rotary_dim(config, spec.head_dim)?)
}

fn digest(text: &str) -> Result<[u8; 32], EngError> {
    let invalid = || EngError::Checkpoint(format!("tokenizer pin {text} is not 64 hex digits"));
    let mut out = [0u8; 32];
    if text.len() != 64 {
        return Err(invalid());
    }
    for (slot, pair) in out.iter_mut().zip(text.as_bytes().as_chunks::<2>().0) {
        *slot = std::str::from_utf8(pair.as_slice())
            .ok()
            .and_then(|digits| u8::from_str_radix(digits, 16).ok())
            .ok_or_else(invalid)?;
    }
    Ok(out)
}

/// The pin of the tokenizer files of the pinned checkpoint. `TokenizerBundle::load` rejects any
/// other `tokenizer.json`, `tokenizer_config.json`, template, or `generation_config.json`.
pub(crate) fn tokenizer_pin() -> Result<TokenizerPin, EngError> {
    Ok(TokenizerPin {
        tokenizer_json: digest(TOKENIZER_JSON_SHA256)?,
        tokenizer_config_json: digest(TOKENIZER_CONFIG_SHA256)?,
        chat_template_jinja: digest(CHAT_TEMPLATE_SHA256)?,
        generation_config_json: digest(GENERATION_CONFIG_SHA256)?,
        vocab_size: TOKENIZER_VOCAB_SIZE,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        BF16_DEFAULT_PREFILL_CHUNK, Backend, DEFAULT_CONTEXT_LIMIT, DEFAULT_PREFILL_CHUNK,
        DEFAULT_SESSION_CAP, MLX_DEFAULT_CONTEXT_LIMIT, NO_BUDGET_PIECE, Q36Env, Q36Options,
        Q36Settings, TieredSettings, activation_name, activation_of, cache_growth_per_position,
        chunked_prefill_allowed, delta_state_bytes, digest, kv_dtype_name, kv_dtype_of,
        parse_kv_dtype, scale_pairing_error, tokenizer_pin,
    };
    use crate::EngError;
    use crate::family::session::StoreLimits;
    use crate::family::spec::QWEN3_6_35B_A3B;
    use crate::tensor::Dtype;
    use std::path::{Path, PathBuf};

    fn cuda_offload(body: &str) -> Result<Q36Settings, EngError> {
        let options: Q36Options = serde_json::from_str(body).unwrap();
        settings_for(Some(&options), &Q36Env::default(), Backend::Cuda)
    }

    #[test]
    fn host_kv_offload_is_opt_in_and_does_not_enable_prefix_reuse() {
        let resident = settings_for(None, &Q36Env::default(), Backend::Cuda).unwrap();
        assert_eq!(resident.kv_offload_host_bytes, None);
        let paged = cuda_offload(r#"{"kv_dtype":"q8","kv_offload_host_bytes":8388608}"#).unwrap();
        assert_eq!(paged.kv_offload_host_bytes, Some(8_388_608));
        assert!(!paged.prefix_reuse);
        assert!(paged.check_prefix_reuse_cap(false).is_ok());
        assert!(error_text(paged.check_prefix_reuse_cap(true)).contains("prefix_reuse"));
        assert!(
            cuda_offload(r#"{"kv_dtype":"q8","kv_offload_host_bytes":18446744073709551615}"#)
                .is_ok()
        );
    }

    #[test]
    fn host_kv_offload_refuses_unsupported_modes_by_key() {
        for body in [
            r#"{"kv_offload_host_bytes":1}"#,
            r#"{"kv_dtype":"f32","kv_offload_host_bytes":1}"#,
            r#"{"kv_dtype":"q8","kv_offload_host_bytes":0}"#,
            r#"{"kv_dtype":"q8","kv_offload_host_bytes":1,"capture":true}"#,
            r#"{"kv_dtype":"q8","kv_offload_host_bytes":1,"tiered":{"slots":8}}"#,
        ] {
            assert!(message(cuda_offload(body)).contains("kv_offload_host_bytes"));
        }
        let options: Q36Options =
            serde_json::from_str(r#"{"kv_dtype":"q8","kv_offload_host_bytes":8388608}"#).unwrap();
        let refused = message(settings_for(
            Some(&options),
            &Q36Env::default(),
            Backend::Mlx,
        ));
        assert!(refused.contains("kv_offload_host_bytes") && refused.contains("CUDA"));
    }

    #[test]
    fn host_kv_budget_uses_strict_unsigned_json_and_rejects_typos() {
        for body in [
            r#"{"kv_offload_host_bytes":-1}"#,
            r#"{"kv_offload_host_bytes":1.5}"#,
            r#"{"kv_offload_host_bytes":"8388608"}"#,
            r#"{"kv_offload_host_bytes":true}"#,
            r#"{"kv_offload_host_bytes":18446744073709551616}"#,
            r#"{"kv_offload_host_byte":8388608}"#,
        ] {
            assert!(serde_json::from_str::<Q36Options>(body).is_err());
        }
    }

    fn with_limit() -> Q36Options {
        Q36Options {
            cache_limit_bytes: Some(1024),
            ..Q36Options::default()
        }
    }

    fn dense_settings(options: &Q36Options) -> Result<Q36Settings, EngError> {
        Q36Settings::new_for_family(
            Path::new("/m"),
            Path::new("/t"),
            Some(options),
            &Q36Env::default(),
            Backend::Mlx,
            crate::family::QWEN3_8_27B_DENSE,
        )
    }

    #[test]
    fn dense_requires_an_explicit_cap_covering_its_cache_and_delta_state() {
        let options = with_limit();
        for cap in [None, Some(9_283_698_687)] {
            let text = message(dense_settings(&Q36Options {
                session_cap_bytes: cap,
                ..options.clone()
            }));
            assert!(text.contains("at least 9283698688 bytes"), "{text}");
            assert!(text.contains("9126805504 attention cache bytes"), "{text}");
            assert!(text.contains("156893184 DeltaNet bytes"), "{text}");
        }
        let settings = dense_settings(&Q36Options {
            session_cap_bytes: Some(9_283_698_688),
            ..options
        })
        .unwrap();
        assert_eq!(settings.context_limit, 262_144);
        assert_eq!(settings.session_cap, 9_283_698_688);
        assert!(settings.check_prefix_reuse_cap(false).is_ok());
        let prefix_error = error_text(settings.check_prefix_reuse_cap(true));
        assert!(prefix_error.contains("at least 18410504192 bytes"));
        assert!(
            dense_settings(&Q36Options {
                session_cap_bytes: Some(18_410_504_192),
                ..with_limit()
            })
            .unwrap()
            .check_prefix_reuse_cap(true)
            .is_ok()
        );
    }

    #[test]
    fn dense_cap_tracks_context_cache_form_and_activation() {
        for (kv, activation, required) in [
            ("q8", "bf16", 156_962_816),
            ("q8", "f32", 156_966_912),
            ("f32", "bf16", 157_155_328),
        ] {
            let options = Q36Options {
                context_limit: Some(2),
                kv_dtype: Some(kv.into()),
                activation_dtype: Some(activation.into()),
                session_cap_bytes: Some(required),
                ..with_limit()
            };
            assert!(dense_settings(&options).is_ok(), "{kv}/{activation}");
            assert!(
                dense_settings(&Q36Options {
                    session_cap_bytes: Some(required - 1),
                    ..options
                })
                .is_err(),
                "{kv}/{activation}"
            );
        }
    }

    #[test]
    fn dense_is_refused_on_cuda_before_any_file_or_device_access() {
        let text = message(Q36Settings::new_for_family(
            Path::new("/nonexistent"),
            Path::new("/nonexistent"),
            None,
            &Q36Env::default(),
            Backend::Cuda,
            crate::family::QWEN3_8_27B_DENSE,
        ));
        assert!(text.contains("qwen3_8_27b_dense requires the native-mlx backend"));
    }

    #[test]
    fn dense_refuses_max_seqs_above_one_even_with_valid_mlx_settings() {
        let options = Q36Options {
            max_seqs: Some(2),
            session_cap_bytes: Some(9_283_698_688),
            ..with_limit()
        };
        let text = message(dense_settings(&options));
        assert!(
            text.contains("max_seqs above 1 requires MoE device routing"),
            "{text}"
        );
    }

    #[test]
    fn dense_context_cannot_exceed_the_descriptor_limit() {
        let text = message(dense_settings(&Q36Options {
            context_limit: Some(262_145),
            session_cap_bytes: Some(u64::MAX),
            ..with_limit()
        }));
        assert!(
            text.contains("context_limit must be between 2 and 262144"),
            "{text}"
        );
    }

    #[test]
    fn the_executor_shape_helper_uses_the_selected_dense_dimensions() {
        // Checkpoint facts from the existing dense config tests, not from the descriptor.
        let config = serde_json::json!({
            "model_type": "qwen3_5",
            "quantization": {"mode": "affine", "bits": 6, "group_size": 64},
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 5120, "intermediate_size": 17408,
                "num_hidden_layers": 64, "num_attention_heads": 24,
                "num_key_value_heads": 4, "head_dim": 256, "vocab_size": 248320,
                "max_position_embeddings": 262144, "full_attention_interval": 4,
                "attn_output_gate": true, "tie_word_embeddings": false,
                "attention_bias": false, "hidden_act": "silu", "rms_norm_eps": 1e-6,
                "linear_num_key_heads": 16, "linear_key_head_dim": 128,
                "linear_num_value_heads": 48, "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "rope_parameters": {
                    "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                    "mrope_interleaved": true, "mrope_section": [11, 11, 10]
                },
                "layer_types": (0..64).map(|i| if i % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }).collect::<Vec<_>>()
            }
        })
        .to_string();
        let shape = super::shape_of(config.as_bytes(), &crate::family::QWEN3_8_27B_DENSE)
            .expect("dense executor shape");
        assert_eq!(shape.hidden, 5120);
        assert_eq!(shape.kinds.len(), 64);
        for (index, kind) in shape.kinds.iter().enumerate() {
            assert_eq!(
                *kind,
                if index % 4 == 3 {
                    crate::family::AttentionKind::Full
                } else {
                    crate::family::AttentionKind::DeltaNet
                }
            );
        }
        assert_eq!(shape.attn.heads_q, 24);
        assert_eq!(shape.attn.heads_kv, 4);
        assert_eq!(shape.attn.head_dim, 256);
        assert_eq!(shape.attn.rotary_dim, 64);
        assert_eq!(shape.delta.value_heads, 48);
        assert_eq!(shape.dense_inter, Some(17_408));
        assert!(super::shape_of(config.as_bytes(), &QWEN3_6_35B_A3B).is_err());
        let q36 = super::shape_of(
            include_bytes!("../../../tests/fixtures/qwen3_6_35b_a3b_config.json"),
            &QWEN3_6_35B_A3B,
        )
        .expect("unchanged Q36 executor shape");
        assert_eq!(q36.hidden, 2048);
        assert_eq!(q36.kinds.len(), 40);
        assert_eq!(q36.attn.heads_q, 16);
        assert_eq!(q36.attn.heads_kv, 2);
        assert_eq!(q36.delta.value_heads, 32);
        assert_eq!(q36.dense_inter, None);
    }

    fn settings_for(
        options: Option<&Q36Options>,
        env: &Q36Env,
        backend: Backend,
    ) -> Result<Q36Settings, EngError> {
        Q36Settings::new(Path::new("/m"), Path::new("/t"), options, env, backend)
    }

    fn settings(options: Option<&Q36Options>, env: &Q36Env) -> Result<Q36Settings, EngError> {
        settings_for(options, env, Backend::Mlx)
    }

    fn message(result: Result<Q36Settings, EngError>) -> String {
        match result {
            Err(EngError::BadRequest(text)) => text,
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    /// The message of a refused call of any kind.
    fn error_text<T: std::fmt::Debug>(result: Result<T, EngError>) -> String {
        match result {
            Err(EngError::BadRequest(text)) => text,
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    /// The rule derives its byte counts from the family spec and the cache form, never from a
    /// literal (packet KV8-INTEGRATE item 5).
    #[test]
    fn the_prefix_reuse_bytes_are_derived_from_the_family_spec() {
        // 30 DeltaNet layers of a `[32, 128, 128]` state and an `[8192, 3]` ring, F32.
        assert_eq!(
            delta_state_bytes(&QWEN3_6_35B_A3B).unwrap(),
            65_863_680,
            "DeltaNet state bytes"
        );
        // 10 attention layers: the F32 control keeps `[1, 2, 1, 256]` keys and values, and the
        // Q8 form charges 1,088 per layer and position at BF16 scales, 1,152 at F32 scales.
        assert_eq!(
            cache_growth_per_position(
                &QWEN3_6_35B_A3B,
                super::KvDtype::F32,
                super::Activation::F32
            )
            .unwrap(),
            40_960,
            "cache bytes per position, F32 control"
        );
        assert_eq!(
            cache_growth_per_position(
                &QWEN3_6_35B_A3B,
                super::KvDtype::Q8,
                super::Activation::Bf16
            )
            .unwrap(),
            10_880,
            "cache bytes per position, Q8 with BF16 scales"
        );
        assert_eq!(
            cache_growth_per_position(&QWEN3_6_35B_A3B, super::KvDtype::Q8, super::Activation::F32)
                .unwrap(),
            11_520,
            "cache bytes per position, Q8 with F32 scales"
        );
    }

    /// The rule accepts exactly `2 * growth_per_position * context_limit + fixed_bytes` and
    /// refuses one byte below it.
    #[test]
    fn the_prefix_reuse_cap_rule_accepts_the_required_cap_and_refuses_one_byte_less() {
        let required = 2 * 40_960 * u64::from(DEFAULT_CONTEXT_LIMIT) + 65_863_680;
        assert_eq!(required, 2_750_218_240, "the required cap");
        let enough = Q36Options {
            cache_limit_bytes: Some(1024),
            session_cap_bytes: Some(required),
            ..f32_form_at_the_cuda_limit()
        };
        let passed = settings(Some(&enough), &Q36Env::default())
            .unwrap()
            .check_prefix_reuse_cap(true);
        assert_eq!(passed, Ok(()), "the required cap passes");
        let one_less = Q36Options {
            cache_limit_bytes: Some(1024),
            session_cap_bytes: Some(required - 1),
            ..f32_form_at_the_cuda_limit()
        };
        let refused = settings(Some(&one_less), &Q36Env::default())
            .unwrap()
            .check_prefix_reuse_cap(true);
        assert!(
            refused.is_err(),
            "one byte below the requirement: {refused:?}"
        );
    }

    /// The refusal names the cap, the requirement, the context limit, and the hint.
    #[test]
    fn the_prefix_reuse_cap_error_names_the_cap_the_requirement_the_limit_and_the_hint() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            session_cap_bytes: Some(2_750_218_239),
            ..f32_form_at_the_cuda_limit()
        };
        let refused = settings(Some(&options), &Q36Env::default())
            .unwrap()
            .check_prefix_reuse_cap(true);
        assert_eq!(
            error_text(refused),
            "engine config: prefix_reuse needs session_cap_bytes of at least 2750218240 bytes for \
             context limit 32768: 2 * 40960 cache bytes per position plus 65863680 DeltaNet bytes; \
             the cap is 2750218239 bytes; raise session_cap_bytes"
        );
    }

    /// The F32 control at 32,768 positions: the form and the limit that the first prefix-reuse
    /// numbers (2,750,218,240 bytes) were derived for, stated because neither is the MLX default.
    fn f32_form_at_the_cuda_limit() -> Q36Options {
        Q36Options {
            kv_dtype: Some("f32".to_string()),
            context_limit: Some(DEFAULT_CONTEXT_LIMIT),
            ..Q36Options::default()
        }
    }

    /// The default MLX engine holds the 8-bit cache at 262,144 positions, so the prefix-reuse rule
    /// asks for `2 * 10,880 * 262,144 + 65,863,680` = 5,770,117,120 bytes: more than the default
    /// cap of 4 GiB, which covers one live session (2,917,990,400 bytes) but not the entry beside
    /// it. The rule accepts the requirement and refuses one byte less, and the refusal names the
    /// cap, the requirement, the limit, and the hint.
    #[test]
    fn the_default_engine_needs_a_larger_cap_for_prefix_reuse_and_the_error_says_so() {
        let required = 2 * 10_880 * u64::from(MLX_DEFAULT_CONTEXT_LIMIT) + 65_863_680;
        assert_eq!(
            required, 5_770_117_120,
            "the required cap of the default form"
        );
        assert!(
            DEFAULT_SESSION_CAP < required,
            "the default cap is below it"
        );
        assert!(
            DEFAULT_SESSION_CAP >= 10_880 * u64::from(MLX_DEFAULT_CONTEXT_LIMIT) + 65_863_680,
            "the default cap covers one live session of the default form"
        );
        let at_default_cap = settings(Some(&with_limit()), &Q36Env::default()).unwrap();
        assert_eq!(
            error_text(at_default_cap.check_prefix_reuse_cap(true)),
            "engine config: prefix_reuse needs session_cap_bytes of at least 5770117120 bytes for \
             context limit 262144: 2 * 10880 cache bytes per position plus 65863680 DeltaNet \
             bytes; the cap is 4294967296 bytes; raise session_cap_bytes"
        );
        let enough = Q36Options {
            session_cap_bytes: Some(required),
            ..with_limit()
        };
        assert_eq!(
            settings(Some(&enough), &Q36Env::default())
                .unwrap()
                .check_prefix_reuse_cap(true),
            Ok(()),
            "the requirement passes"
        );
        let one_less = Q36Options {
            session_cap_bytes: Some(required - 1),
            ..with_limit()
        };
        assert!(
            settings(Some(&one_less), &Q36Env::default())
                .unwrap()
                .check_prefix_reuse_cap(true)
                .is_err(),
            "one byte below the requirement is refused"
        );
    }

    /// The CUDA arm has no session cap (it is 0 and ignored), so the rule must not refuse it: a
    /// CUDA start with the flag on passes at the default limit of 32,768 and at 40,960, the limit
    /// that the 33,650-token pi transcript of the 3090 gate needs. A cap of 1 byte on the MLX arm
    /// is refused, which shows that the backend alone decides.
    #[test]
    fn the_prefix_reuse_cap_rule_does_not_apply_to_the_cuda_arm() {
        for limit in [DEFAULT_CONTEXT_LIMIT, 40_960] {
            let options = Q36Options {
                context_limit: Some(limit),
                ..Q36Options::default()
            };
            let cuda = settings_for(Some(&options), &Q36Env::default(), Backend::Cuda).unwrap();
            assert_eq!(cuda.session_cap, 0, "the CUDA arm has no session cap");
            assert_eq!(
                cuda.check_prefix_reuse_cap(true),
                Ok(()),
                "CUDA, context limit {limit}"
            );
        }
        let mlx_options = Q36Options {
            cache_limit_bytes: Some(1024),
            session_cap_bytes: Some(1),
            ..Q36Options::default()
        };
        let mlx = settings(Some(&mlx_options), &Q36Env::default()).unwrap();
        assert!(
            mlx.check_prefix_reuse_cap(true).is_err(),
            "the MLX arm still refuses a cap of 1 byte"
        );
    }

    /// Without the flag the rule does not apply, whatever the cap is.
    #[test]
    fn the_prefix_reuse_cap_rule_is_off_when_the_flag_is_off() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            session_cap_bytes: Some(1),
            ..Q36Options::default()
        };
        let passed = settings(Some(&options), &Q36Env::default())
            .unwrap()
            .check_prefix_reuse_cap(false);
        assert_eq!(passed, Ok(()), "the flag is off");
    }

    #[test]
    fn an_object_with_a_cache_limit_takes_the_defaults() {
        let got = settings(Some(&with_limit()), &Q36Env::default()).unwrap();
        assert_eq!(
            got,
            Q36Settings {
                spec: QWEN3_6_35B_A3B,
                artifact: PathBuf::from("/m"),
                tokenizer_dir: PathBuf::from("/t"),
                cache_limit: 1024,
                context_limit: 262_144,
                session_cap: 4 << 30,
                prefill_chunk: 256,
                prefill_budget: None,
                prefill_reserve: 0,
                activation: super::Activation::Bf16,
                kv: super::KvDtype::Q8,
                kv_offload_host_bytes: None,
                capture: false,
                compile: false,
                max_seqs: 1,
                prefix_store_entries: None,
                backend: Backend::Mlx,
                prefix_reuse: false,
                tiered: None,
                ignored_mlx_keys: Vec::new(),
            }
        );
        assert_eq!(MLX_DEFAULT_CONTEXT_LIMIT, 262_144);
        assert_eq!(
            DEFAULT_CONTEXT_LIMIT, 32_768,
            "the CUDA arm keeps its limit"
        );
        assert_eq!(DEFAULT_SESSION_CAP, 4_294_967_296);
        assert_eq!(DEFAULT_PREFILL_CHUNK, 256);
        assert_eq!(BF16_DEFAULT_PREFILL_CHUNK, 256);
        assert_eq!(NO_BUDGET_PIECE, 16);
        // The reserve of `config/qwen36-mac-studio-36gb.json` is a plan sum (see its README): the
        // 8-bit cache at 262,144 positions, the DeltaNet state, the cache limit of the file, and
        // the pressure margin.
        assert_eq!(262_144u64 * 10_880, 2_852_126_720);
        assert_eq!(
            2_852_126_720u64 + 65_863_680 + 4_294_967_296 + (1 << 30),
            8_286_699_520
        );
    }

    /// The `prefix_reuse` member of the settings (packet SERVE-2-c item 1): `new` leaves it
    /// false on both backends, because no config member or environment key carries the flag;
    /// the factory sets it after `new` from the process-wide flag of the configuration file.
    #[test]
    fn the_settings_default_prefix_reuse_to_false() {
        for backend in [Backend::Mlx, Backend::Cuda] {
            let options = match backend {
                Backend::Mlx => with_limit(),
                Backend::Cuda => Q36Options::default(),
            };
            let got = settings_for(Some(&options), &Q36Env::default(), backend)
                .map(|got| got.prefix_reuse)
                .unwrap_or_else(|_| panic!("{backend:?} settings must build"));
            assert!(!got, "{backend:?}: the factory sets the flag after `new`");
        }
    }

    /// BF16 serving policy (decision 0046 A6 item 11, amendment of 2026-10-11): without a
    /// `prefill_chunk` the default chunk is 256 for a BF16 engine and for an F32 engine; an
    /// explicit `prefill_chunk` wins for both. The shipped config leaves the member absent.
    #[test]
    fn the_default_chunk_depends_on_the_activation_and_an_explicit_chunk_wins() {
        let bf16 = settings(Some(&with_limit()), &Q36Env::default()).unwrap();
        assert_eq!(
            bf16.activation,
            super::Activation::Bf16,
            "the default activation"
        );
        assert_eq!(bf16.prefill_chunk, 256, "BF16 default: the chunk of F32");
        let f32 = Q36Options {
            activation_dtype: Some("f32".to_string()),
            ..with_limit()
        };
        let got = settings(Some(&f32), &Q36Env::default()).unwrap();
        assert_eq!(got.prefill_chunk, 256, "F32 default");
        for activation in ["bf16", "f32"] {
            let options = Q36Options {
                activation_dtype: Some(activation.to_string()),
                prefill_chunk: Some(64),
                ..with_limit()
            };
            let got = settings(Some(&options), &Q36Env::default()).unwrap();
            assert_eq!(
                got.prefill_chunk, 64,
                "an explicit chunk wins for {activation}"
            );
        }
        let maximum = Q36Options {
            prefill_chunk: Some(1024),
            ..with_limit()
        };
        assert_eq!(
            settings(Some(&maximum), &Q36Env::default())
                .unwrap()
                .prefill_chunk,
            1024,
            "the maximum accepted chunk"
        );
    }

    #[test]
    fn every_member_of_the_object_overrides_its_default() {
        let options = Q36Options {
            cache_limit_bytes: Some(5),
            context_limit: Some(65_536),
            session_cap_bytes: Some(3_000_000_000),
            prefill_chunk: Some(64),
            prefill_budget_bytes: Some(30_000_000_000),
            prefill_session_reserve_bytes: Some(7),
            prefix_store_entries: Some(9),
            activation_dtype: Some("f32".to_string()),
            kv_dtype: Some("q8".to_string()),
            kv_offload_host_bytes: None,
            capture: Some(true),
            compile: Some(true),
            max_seqs: Some(4),
            tiered: None,
        };
        let got = settings(Some(&options), &Q36Env::default()).unwrap();
        assert_eq!(got.cache_limit, 5);
        assert_eq!(got.context_limit, 65_536);
        assert_eq!(got.session_cap, 3_000_000_000);
        assert_eq!(got.prefill_chunk, 64);
        assert_eq!(got.prefill_budget, Some(30_000_000_000));
        assert_eq!(got.prefill_reserve, 7);
        assert_eq!(got.activation, super::Activation::F32);
        assert_eq!(got.kv, super::KvDtype::Q8, "the kv_dtype member overrides");
        assert!(got.capture, "every member of the object overrides");
        assert!(got.compile, "compile: every member of the object overrides");
        assert_eq!(got.max_seqs, 4);
        assert_eq!(got.prefix_store_entries, Some(9));
        assert_eq!(got.artifact, PathBuf::from("/m"));
        assert_eq!(got.tokenizer_dir, PathBuf::from("/t"));
    }

    /// The compile key of packet MX-COMPILE-1a: the default is false (the shipped Mac file sets
    /// true, pinned by `the_shipped_mac_studio_config_passes_every_check_before_the_backend`),
    /// the MLX arm reads the member, and the CUDA arm
    /// keeps false and records the key for its start line.
    #[test]
    fn the_compile_key_defaults_to_false_the_mlx_arm_reads_it_and_the_cuda_arm_ignores_it() {
        let off = settings_for(Some(&with_limit()), &Q36Env::default(), Backend::Mlx).unwrap();
        assert!(!off.compile, "compile is off without the member");
        let on = Q36Options {
            compile: Some(true),
            ..with_limit()
        };
        assert!(
            settings_for(Some(&on), &Q36Env::default(), Backend::Mlx)
                .unwrap()
                .compile,
            "the MLX arm reads the member"
        );
        // A CUDA entry with only the compile key: no other MLX key is set, so the ignored list
        // names the compile key alone.
        let cuda_only: Q36Options = serde_json::from_str("{\"compile\": true}").unwrap();
        let got = settings_for(Some(&cuda_only), &Q36Env::default(), Backend::Cuda).unwrap();
        assert!(!got.compile, "the CUDA arm keeps the compile seam off");
        assert_eq!(
            got.ignored_mlx_keys,
            vec!["compile"],
            "the CUDA arm records the key for its start line"
        );
        let unknown = serde_json::from_str::<Q36Options>("{\"compile\": 1}");
        assert!(unknown.is_err(), "the member is a boolean, not a count");
    }

    #[test]
    fn concurrent_sessions_are_opt_in_and_require_compiled_mlx() {
        let env = Q36Env::default();
        let mut options = with_limit();
        for backend in [Backend::Mlx, Backend::Cuda] {
            assert_eq!(
                settings_for(Some(&options), &env, backend)
                    .unwrap()
                    .max_seqs,
                1
            );
        }
        options.max_seqs = Some(4);
        assert!(message(settings(Some(&options), &env)).contains("compile=true"));
        options.compile = Some(true);
        assert!(message(settings_for(Some(&options), &env, Backend::Cuda)).contains("MLX"));
        assert!(message(settings(Some(&options), &env)).contains("prefill_budget_bytes"));
        options.prefill_budget_bytes = Some(8 << 30);
        options.prefill_session_reserve_bytes = Some(0);
        assert_eq!(settings(Some(&options), &env).unwrap().max_seqs, 4);
        options.prefill_chunk = Some(1);
        assert!(message(settings(Some(&options), &env)).contains("prefill_chunk"));
        options.prefill_chunk = Some(2);
        for value in [0, 9] {
            options.max_seqs = Some(value);
            assert!(message(settings(Some(&options), &env)).contains("max_seqs"));
        }
        for value in [
            serde_json::json!(-1),
            serde_json::json!(2.5),
            serde_json::json!("4"),
        ] {
            assert!(
                serde_json::from_value::<Q36Options>(serde_json::json!({"max_seqs": value}))
                    .is_err()
            );
        }
    }

    #[test]
    fn concurrent_sessions_refuse_unplannable_chunk_bounds_before_io() {
        let env = Q36Env::default();
        for chunk in [1, 2, 3, 4] {
            let mut options = Q36Options {
                compile: Some(true),
                max_seqs: Some(4),
                prefill_chunk: Some(chunk),
                prefill_budget_bytes: Some(8 << 30),
                prefill_session_reserve_bytes: Some(0),
                ..with_limit()
            };
            let result = settings(Some(&options), &env);
            if chunk < 4 {
                assert!(message(result).contains("prefill_chunk"), "chunk {chunk}");
            } else {
                assert!(result.is_ok(), "chunk {chunk}");
            }
            options.max_seqs = Some(1);
            assert!(settings(Some(&options), &env).is_ok(), "solo chunk {chunk}");
        }
    }

    #[test]
    fn concurrent_sessions_refuse_exhausted_prefill_reserves_before_io() {
        let env = Q36Env::default();
        for (cap, reserve, budget, admitted) in [
            (8, 0, 9, true),
            (8, 0, 8, false),
            (8, 0, 7, false),
            (1, 8, 9, true),
            (1, 8, 8, false),
            (1, 8, 7, false),
            (u64::MAX, 0, u64::MAX, false),
            (1, u64::MAX, u64::MAX, false),
        ] {
            let mut options = Q36Options {
                compile: Some(true),
                max_seqs: Some(4),
                prefill_chunk: Some(4),
                session_cap_bytes: Some(cap),
                prefill_budget_bytes: Some(budget),
                prefill_session_reserve_bytes: Some(reserve),
                ..with_limit()
            };
            let result = settings(Some(&options), &env);
            if admitted {
                assert!(
                    result.is_ok(),
                    "cap {cap}, reserve {reserve}, budget {budget}"
                );
            } else {
                assert!(message(result).contains("prefill_budget_bytes"));
            }
            options.max_seqs = Some(1);
            assert!(settings(Some(&options), &env).is_ok(), "solo reserve floor");
        }
    }

    #[test]
    fn the_environment_fills_the_cache_limit_and_the_object_wins() {
        let env = Q36Env {
            cache_limit: Some("77".to_string()),
        };
        assert_eq!(settings(None, &env).unwrap().cache_limit, 77);
        assert_eq!(
            settings(Some(&with_limit()), &env).unwrap().cache_limit,
            1024,
            "the object wins"
        );
    }

    #[test]
    fn a_missing_or_bad_cache_limit_names_the_key_and_the_variable() {
        assert_eq!(
            message(settings(None, &Q36Env::default())),
            "engine config: cache_limit_bytes is missing and Q36_CACHE_LIMIT is not set"
        );
        let empty = Q36Env {
            cache_limit: Some(String::new()),
        };
        assert_eq!(
            message(settings(None, &empty)),
            "engine config: cache_limit_bytes is missing and Q36_CACHE_LIMIT is not set"
        );
        let env = Q36Env {
            cache_limit: Some("many".to_string()),
        };
        assert_eq!(
            message(settings(None, &env)),
            "engine config: Q36_CACHE_LIMIT is not a byte count"
        );
    }

    #[test]
    fn a_cache_limit_of_zero_is_refused() {
        let options = Q36Options {
            cache_limit_bytes: Some(0),
            ..Q36Options::default()
        };
        assert_eq!(
            message(settings(Some(&options), &Q36Env::default())),
            "engine config: the cache limit must be above 0"
        );
    }

    #[test]
    fn the_numeric_members_are_range_checked_and_name_the_key() {
        let cases = [
            (
                Q36Options {
                    context_limit: Some(1),
                    ..with_limit()
                },
                "context_limit must be between 2 and 4294967295, found 1",
            ),
            (
                Q36Options {
                    prefill_chunk: Some(0),
                    ..with_limit()
                },
                "prefill_chunk must be between 1 and 2048, found 0",
            ),
            (
                Q36Options {
                    prefill_chunk: Some(2049),
                    ..with_limit()
                },
                "prefill_chunk must be between 1 and 2048, found 2049",
            ),
            (
                Q36Options {
                    session_cap_bytes: Some(0),
                    ..with_limit()
                },
                "session_cap_bytes must be between 1 and 18446744073709551615, found 0",
            ),
        ];
        for (options, want) in cases {
            let got = message(settings(Some(&options), &Q36Env::default()));
            assert_eq!(got, format!("engine config: {want}"));
        }
    }

    #[test]
    fn the_mlx_first_error_order_is_budget_reserve_cache_context_session_chunk() {
        // The base order (KUBDAN-MSG-71 item 4): budget, reserve, cache_limit, context_limit,
        // session_cap, prefill_chunk. `cache_limit` is computed before `context_limit`, so a
        // config with BOTH faults names the cache-limit fault first — a missing limit (no
        // key, no env) or a zero one against `context_limit: 1` — and
        // `{context_limit: 1, prefill_chunk: 0}` names `context_limit` before the chunk.
        let both = Q36Options {
            context_limit: Some(1),
            prefill_chunk: Some(0),
            ..with_limit()
        };
        assert_eq!(
            message(settings(Some(&both), &Q36Env::default())),
            "engine config: context_limit must be between 2 and 4294967295, found 1"
        );
        let all = Q36Options {
            context_limit: Some(1),
            session_cap_bytes: Some(0),
            prefill_chunk: Some(0),
            ..with_limit()
        };
        assert_eq!(
            message(settings(Some(&all), &Q36Env::default())),
            "engine config: context_limit must be between 2 and 4294967295, found 1"
        );
        // The two pairs with a FAULTING cache limit pin the flip the cases above cannot
        // see (`with_limit()` never faults the cache limit): the cache-limit fault wins.
        let no_limit = Q36Options {
            context_limit: Some(1),
            ..Q36Options::default()
        };
        assert_eq!(
            message(settings(Some(&no_limit), &Q36Env::default())),
            "engine config: cache_limit_bytes is missing and Q36_CACHE_LIMIT is not set"
        );
        let zero_limit = Q36Options {
            cache_limit_bytes: Some(0),
            context_limit: Some(1),
            ..Q36Options::default()
        };
        assert_eq!(
            message(settings(Some(&zero_limit), &Q36Env::default())),
            "engine config: the cache limit must be above 0"
        );
    }

    #[test]
    fn the_object_reads_from_json_and_rejects_an_unknown_member() {
        let text = r#"{"cache_limit_bytes": 7, "context_limit": 4096, "session_cap_bytes": 9, "prefill_chunk": 3}"#;
        let options: Q36Options = serde_json::from_str(text).unwrap();
        assert_eq!(
            options,
            Q36Options {
                cache_limit_bytes: Some(7),
                context_limit: Some(4096),
                session_cap_bytes: Some(9),
                prefill_chunk: Some(3),
                ..Q36Options::default()
            }
        );
        let error = serde_json::from_str::<Q36Options>(r#"{"manifest": "/m"}"#).unwrap_err();
        assert!(
            error.to_string().contains("unknown field `manifest`"),
            "{error}"
        );
    }

    #[test]
    fn a_prefill_budget_must_be_positive_and_a_reserve_needs_a_budget() {
        let zero = Q36Options {
            prefill_budget_bytes: Some(0),
            ..with_limit()
        };
        assert_eq!(
            message(settings(Some(&zero), &Q36Env::default())),
            "engine config: prefill_budget_bytes must be between 1 and 18446744073709551615, found 0"
        );
        let alone = Q36Options {
            prefill_session_reserve_bytes: Some(5),
            ..with_limit()
        };
        assert_eq!(
            message(settings(Some(&alone), &Q36Env::default())),
            "engine config: prefill_session_reserve_bytes needs prefill_budget_bytes"
        );
        let unreserved = Q36Options {
            prefill_budget_bytes: Some(5),
            ..with_limit()
        };
        assert_eq!(
            message(settings(Some(&unreserved), &Q36Env::default())),
            "engine config: prefill_budget_bytes needs prefill_session_reserve_bytes"
        );
        let both = Q36Options {
            prefill_budget_bytes: Some(1),
            prefill_session_reserve_bytes: Some(0),
            ..with_limit()
        };
        let got = settings(Some(&both), &Q36Env::default()).unwrap();
        assert_eq!((got.prefill_budget, got.prefill_reserve), (Some(1), 0));
    }

    #[test]
    fn the_prefill_reserve_is_the_member_and_never_a_default() {
        let off = settings(Some(&with_limit()), &Q36Env::default()).unwrap();
        assert_eq!((off.prefill_budget, off.prefill_reserve), (None, 0));
        let options = Q36Options {
            cache_limit_bytes: Some(24),
            session_cap_bytes: Some(1000),
            prefill_budget_bytes: Some(5000),
            prefill_session_reserve_bytes: Some(77),
            ..Q36Options::default()
        };
        let got = settings(Some(&options), &Q36Env::default()).unwrap();
        assert_eq!(got.prefill_reserve, 77, "not the cap plus the cache limit");
        let max = Q36Options {
            prefill_session_reserve_bytes: Some(u64::MAX),
            ..options
        };
        let got = settings(Some(&max), &Q36Env::default()).unwrap();
        assert_eq!(got.prefill_reserve, u64::MAX);
    }

    #[test]
    fn the_prefill_members_read_from_json() {
        let text =
            r#"{"prefill_budget_bytes": 32212254720, "prefill_session_reserve_bytes": 6776750080}"#;
        let options: Q36Options = serde_json::from_str(text).unwrap();
        assert_eq!(
            options,
            Q36Options {
                prefill_budget_bytes: Some(32_212_254_720),
                prefill_session_reserve_bytes: Some(6_776_750_080),
                ..Q36Options::default()
            }
        );
    }

    #[test]
    fn activation_dtype_parses_f32_and_bf16_and_the_default_is_bf16() {
        let bf16 = Q36Options {
            activation_dtype: Some("bf16".to_string()),
            ..with_limit()
        };
        let f32 = Q36Options {
            activation_dtype: Some("f32".to_string()),
            ..with_limit()
        };
        assert_eq!(
            settings(Some(&bf16), &Q36Env::default())
                .unwrap()
                .activation,
            super::Activation::Bf16
        );
        assert_eq!(
            settings(Some(&f32), &Q36Env::default()).unwrap().activation,
            super::Activation::F32
        );
        assert_eq!(
            settings(Some(&with_limit()), &Q36Env::default())
                .unwrap()
                .activation,
            super::Activation::Bf16,
            "the serving default of the user decision of 2026-10-10"
        );
        assert_eq!(activation_of(None).unwrap(), super::Activation::Bf16);
        assert_eq!(activation_name(super::Activation::Bf16), "bf16");
        assert_eq!(activation_name(super::Activation::F32), "f32");
    }

    /// Packet KV8-INTEGRATE item 5, hand-derived: the per-layer, per-position charge of both
    /// cache forms. The real model is KVH 2, D 256 — codes `2 * 256 * 2 = 1,024` bytes, scales
    /// and biases `2 sides * 2 arrays * 2 * 4 groups * scale_bytes`, so 64 bytes at BF16 scales
    /// (2) and 128 at F32 scales (4): 1,088 and 1,152 per layer and position, 10,880 and 11,520
    /// over the 10 attention layers. The F32 form stays the control at 4,096 and 40,960, and a
    /// head dimension below the group (TINY_GQA's 32) refuses the Q8 form.
    #[test]
    fn the_growth_bytes_derive_both_cache_forms_by_hand() {
        use super::{Activation, KvDtype, kv_growth_per_position};
        // The F32 control: the tiny GQA layer (2 heads of 32) and the real layer (2 of 256).
        assert_eq!(
            kv_growth_per_position(2, 32, KvDtype::F32, Activation::F32).unwrap(),
            512
        );
        assert_eq!(
            kv_growth_per_position(2, 256, KvDtype::F32, Activation::F32).unwrap(),
            4_096
        );
        // The Q8 form, both activation dtypes: the tiny KV8 fixture (2 heads of 64) and the
        // real layer (2 of 256). The fixture: codes 2 * 64 * 2 = 256 bytes, plus scales and
        // biases 2 sides * 2 arrays * 2 * 1 group * scale_bytes — 32 at F32, 16 at BF16.
        assert_eq!(
            kv_growth_per_position(2, 64, KvDtype::Q8, Activation::F32).unwrap(),
            288
        );
        assert_eq!(
            kv_growth_per_position(2, 64, KvDtype::Q8, Activation::Bf16).unwrap(),
            272
        );
        assert_eq!(
            kv_growth_per_position(2, 256, KvDtype::Q8, Activation::Bf16).unwrap(),
            1_088
        );
        assert_eq!(
            kv_growth_per_position(2, 256, KvDtype::Q8, Activation::F32).unwrap(),
            1_152
        );
        // Over the 10 attention layers of the real model, both forms.
        assert_eq!(
            10 * kv_growth_per_position(2, 256, KvDtype::F32, Activation::F32).unwrap(),
            40_960
        );
        assert_eq!(
            10 * kv_growth_per_position(2, 256, KvDtype::Q8, Activation::Bf16).unwrap(),
            10_880
        );
        assert_eq!(
            10 * kv_growth_per_position(2, 256, KvDtype::Q8, Activation::F32).unwrap(),
            11_520
        );
        // Decision 0046 A7 sizing: 2.85 GB at 262,144 positions instead of 10.7 GB.
        assert_eq!(10_880u64 * 262_144, 2_852_126_720);
        assert_eq!(40_960u64 * 262_144, 10_737_418_240);
        // A head dimension below the group refuses the Q8 form (TINY_GQA's 32), naming the
        // setting.
        let refused = kv_growth_per_position(2, 32, KvDtype::Q8, Activation::F32);
        assert!(
            matches!(&refused, Err(EngError::BadRequest(m)) if m.contains("kv_dtype q8") && m.contains("multiple of 64")),
            "the head-dimension rule: {refused:?}"
        );
        // An overflow of either form is MemoryPlan: the F32 form at the extremes, the Q8 form
        // with a head dimension of 2^31 lanes (a multiple of 64 whose codes alone overflow).
        assert!(kv_growth_per_position(u32::MAX, u32::MAX, KvDtype::F32, Activation::F32).is_err());
        assert!(kv_growth_per_position(u32::MAX, 1 << 31, KvDtype::Q8, Activation::Bf16).is_err());
    }

    /// The `kv_dtype` pairs with `activation_dtype` freely: the Q8 scales follow the activation
    /// dtype, so all four combinations are legal settings (packet KV8-INTEGRATE items 1 and 3).
    #[test]
    fn every_kv_dtype_pairs_with_every_activation_dtype() {
        for kv in [super::KvDtype::F32, super::KvDtype::Q8] {
            for (text, act) in [
                ("f32", super::Activation::F32),
                ("bf16", super::Activation::Bf16),
            ] {
                let options = Q36Options {
                    activation_dtype: Some(text.to_string()),
                    kv_dtype: Some(kv_dtype_name(kv).to_string()),
                    ..with_limit()
                };
                let got = settings(Some(&options), &Q36Env::default()).unwrap();
                assert_eq!(
                    (got.activation, got.kv),
                    (act, kv),
                    "the pair {text}/{kv:?}"
                );
            }
        }
    }

    #[test]
    fn a_bad_activation_dtype_is_refused_with_the_member_style() {
        for text in ["f16", "BF16", ""] {
            let options = Q36Options {
                activation_dtype: Some(text.to_string()),
                ..with_limit()
            };
            assert_eq!(
                message(settings(Some(&options), &Q36Env::default())),
                format!("engine config: activation_dtype must be f32 or bf16, got {text:?}"),
                "the text {text:?}"
            );
        }
        let json = r#"{"activation_dtype": "bf16"}"#;
        let options: Q36Options = serde_json::from_str(json).unwrap();
        assert_eq!(options.activation_dtype.as_deref(), Some("bf16"));
    }

    /// Packet KV8-INTEGRATE item 1: `kv_dtype` parses `f32` and `q8`, reads from JSON, and refuses
    /// any other text in the style of its neighbors. The default is `q8` on the MLX arm (the
    /// user's decision of 2026-10-13 after the gate of record Q36-KV8-GATE; the F32 control stays
    /// one member away) and `f32` on the CUDA arm (the default flips to q8 only after G-KV8).
    #[test]
    fn kv_dtype_parses_f32_and_q8_and_the_default_is_q8_on_mlx_and_f32_on_cuda() {
        for (text, want) in [("f32", super::KvDtype::F32), ("q8", super::KvDtype::Q8)] {
            let options = Q36Options {
                kv_dtype: Some(text.to_string()),
                ..with_limit()
            };
            assert_eq!(
                settings(Some(&options), &Q36Env::default()).unwrap().kv,
                want,
                "the text {text}"
            );
        }
        assert_eq!(
            settings(Some(&with_limit()), &Q36Env::default())
                .unwrap()
                .kv,
            super::KvDtype::Q8,
            "the MLX default is the 8-bit cache"
        );
        assert_eq!(kv_dtype_of(None, Backend::Mlx).unwrap(), super::KvDtype::Q8);
        assert_eq!(
            kv_dtype_of(None, Backend::Cuda).unwrap(),
            super::KvDtype::F32,
            "the CUDA default is the F32 cache"
        );
        assert_eq!(
            kv_dtype_of(Some("f32"), Backend::Mlx).unwrap(),
            super::KvDtype::F32,
            "an explicit f32 selects the control on the MLX arm"
        );
        assert_eq!(
            kv_dtype_of(Some("q8"), Backend::Cuda).unwrap(),
            super::KvDtype::Q8,
            "parsing does not decide availability; the CUDA start maps q8 onto Int8G32 (KV8-B B4)"
        );
        assert_eq!(kv_dtype_name(super::KvDtype::F32), "f32");
        assert_eq!(kv_dtype_name(super::KvDtype::Q8), "q8");
        assert_eq!(parse_kv_dtype("f32").unwrap(), super::KvDtype::F32);
        let json = r#"{"kv_dtype": "q8"}"#;
        let options: Q36Options = serde_json::from_str(json).unwrap();
        assert_eq!(options.kv_dtype.as_deref(), Some("q8"));
    }

    #[test]
    fn a_bad_kv_dtype_is_refused_with_the_member_style() {
        for text in ["f16", "Q8", "int8", ""] {
            let options = Q36Options {
                kv_dtype: Some(text.to_string()),
                ..with_limit()
            };
            assert_eq!(
                message(settings(Some(&options), &Q36Env::default())),
                format!("engine config: kv_dtype must be f32, bf16, or q8, got {text:?}"),
                "the text {text:?}"
            );
        }
    }

    /// KV8-B B4-F: the CUDA arm refuses an unknown `kv_dtype` text at its own arm too.
    #[test]
    fn the_cuda_arm_refuses_an_unknown_kv_dtype_text_at_its_own_member() {
        let options = Q36Options {
            kv_dtype: Some("int8".to_string()),
            ..with_limit()
        };
        assert_eq!(
            message(settings_for(
                Some(&options),
                &Q36Env::default(),
                Backend::Cuda
            )),
            "engine config: kv_dtype must be f32, bf16, or q8, got \"int8\"",
        );
    }

    #[test]
    fn the_scale_pairing_rule_accepts_only_the_matching_pair() {
        // The real model: 431 packed weights, each with a scale and a bias tensor, 862 parts.
        assert_eq!(862, 2 * 431, "every packed weight holds two parts");
        for (f32_parts, bf16_parts, activation) in [
            (862u64, 0u64, super::Activation::F32),
            (0, 862, super::Activation::Bf16),
            (43, 0, super::Activation::F32),
            (0, 43, super::Activation::Bf16),
        ] {
            assert_eq!(
                scale_pairing_error(431, f32_parts, bf16_parts, 0, activation),
                None,
                "the matching pair passes"
            );
        }
        let bf16_over_f32 = scale_pairing_error(431, 862, 0, 0, super::Activation::Bf16).unwrap();
        assert!(
            bf16_over_f32.contains("activation_dtype bf16 needs BF16 scales and biases"),
            "{bf16_over_f32}"
        );
        let f32_over_bf16 = scale_pairing_error(431, 0, 862, 0, super::Activation::F32).unwrap();
        assert!(
            f32_over_bf16.contains("activation_dtype f32 needs F32 scales and biases"),
            "{f32_over_bf16}"
        );
        for activation in [super::Activation::F32, super::Activation::Bf16] {
            assert!(
                scale_pairing_error(431, 861, 1, 0, activation).is_some(),
                "one stray part refuses the pair"
            );
            assert!(
                scale_pairing_error(431, 862, 0, 1, activation).is_some(),
                "one other-dtype part refuses the pair"
            );
        }
    }

    /// The chunk rule of packet Q36-BF16-b: the chunk path is dtype aware, so both activation
    /// dtypes plan a chunk. `Q36Executor::plan_prefill` asks the rule with `Consts::act`, which
    /// the setting maps through `Activation::dtype`, and the start-up line of `start::plan_prefill`
    /// follows the plan and names no dtype rule.
    #[test]
    fn the_chunk_rule_plans_a_chunk_for_both_activation_dtypes() {
        assert!(
            chunked_prefill_allowed(Dtype::F32),
            "the F32 control keeps the chunked prefill"
        );
        assert!(
            chunked_prefill_allowed(Dtype::Bf16),
            "the chunk path is dtype aware (packet Q36-BF16-b)"
        );
        assert!(
            !chunked_prefill_allowed(Dtype::F16),
            "a dtype the decoder cannot hold plans no chunk (fix1 of packet Q36-BF16-b): the rule
            stays a rule, not a constant true"
        );
        assert_eq!(
            super::Activation::F32.dtype(),
            Dtype::F32,
            "the F32 control maps to the F32 the rule asks for"
        );
        assert_eq!(
            super::Activation::Bf16.dtype(),
            Dtype::Bf16,
            "the BF16 setting maps to the BF16 the rule accepts"
        );
    }

    #[test]
    fn the_pin_holds_the_digest_of_each_tokenizer_file() {
        let pin = tokenizer_pin().unwrap();
        assert_eq!(pin.vocab_size, 248_077);
        assert_eq!(pin.tokenizer_json[..4], [0x87, 0xa7, 0x83, 0x0d]);
        assert_eq!(pin.tokenizer_json[28..], [0x3d, 0xb7, 0x2d, 0xe4]);
        assert_eq!(pin.tokenizer_config_json[..4], [0xe9, 0x8f, 0x19, 0x01]);
        assert_eq!(pin.chat_template_jinja[..4], [0xe8, 0x4f, 0x32, 0xa2]);
        assert_eq!(pin.generation_config_json[..4], [0xe7, 0x0c, 0x13, 0x6c]);
    }

    #[test]
    fn a_digest_that_is_not_64_hex_digits_is_refused() {
        assert!(digest(&"a".repeat(64)).is_ok());
        for text in ["", &"a".repeat(63), &"a".repeat(65), &"g".repeat(64)] {
            assert!(
                matches!(digest(text), Err(EngError::Checkpoint(_))),
                "{text:?}"
            );
        }
    }

    #[test]
    fn the_cuda_arm_parses_the_8_bit_form_and_the_start_maps_it() {
        // KV8-B B4 (merged with main): the CUDA arm no longer refuses `q8` at the settings —
        // the start maps the parsed form onto the step's `KvMode` (F32 to the fp16 control
        // cache, Q8 to Int8G32) and refuses the Tiled layout there.
        let q8: Q36Options = serde_json::from_str("{\"kv_dtype\": \"q8\"}").unwrap();
        assert_eq!(
            settings_for(Some(&q8), &Q36Env::default(), Backend::Cuda)
                .expect("the CUDA arm parses the 8-bit form")
                .kv,
            super::KvDtype::Q8
        );
        let f32_form: Q36Options = serde_json::from_str("{\"kv_dtype\": \"f32\"}").unwrap();
        let got = settings_for(Some(&f32_form), &Q36Env::default(), Backend::Cuda).unwrap();
        assert_eq!(got.kv, super::KvDtype::F32);
        // The MLX arm takes the same member.
        let mlx = Q36Options {
            kv_dtype: Some("q8".to_string()),
            ..with_limit()
        };
        assert_eq!(
            settings_for(Some(&mlx), &Q36Env::default(), Backend::Mlx)
                .unwrap()
                .kv,
            super::KvDtype::Q8
        );
    }

    /// The default flip of the MLX arm does not reach the CUDA arm: it keeps the F32 cache and
    /// 32,768 positions, which its F32 cache is sized by.
    #[test]
    fn the_cuda_arm_keeps_the_f32_form_and_32768_positions_by_default() {
        let got = settings_for(None, &Q36Env::default(), Backend::Cuda).unwrap();
        assert_eq!(got.kv, super::KvDtype::F32);
        assert_eq!(got.context_limit, 32_768);
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    fn a_cuda_entry_needs_no_cache_limit_and_refuses_the_prefill_budget_by_name() {
        // A CUDA entry is configurable with an empty object: no cache limit, no environment.
        let got = settings_for(None, &Q36Env::default(), Backend::Cuda).unwrap();
        assert_eq!(
            (got.context_limit, got.cache_limit, got.capture),
            (DEFAULT_CONTEXT_LIMIT, 0, false)
        );
        // Default options set no MLX key, so NOTHING is recorded for the start lines
        // (KUBDAN-MSG-71 item 5: the empty case pins the filter of `ignored_mlx_keys_of`).
        assert!(
            got.ignored_mlx_keys.is_empty(),
            "default cuda options record no ignored key: {:?}",
            got.ignored_mlx_keys
        );
        assert_eq!(
            got.prefill_budget, None,
            "the CUDA arm plans no budget: the worker piece is NO_BUDGET_PIECE"
        );
        for key in ["prefill_budget_bytes", "prefill_session_reserve_bytes"] {
            let text: String = format!("{{\"{key}\": 5}}");
            let options: Q36Options = serde_json::from_str(&text).unwrap();
            assert_eq!(
                message(settings_for(
                    Some(&options),
                    &Q36Env::default(),
                    Backend::Cuda
                )),
                format!(
                    "engine config: {key} is an MLX chunked-prefill key; the CUDA arm \
                     prefills one token per step until CPF"
                )
            );
        }
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    fn the_cuda_arm_ignores_prefill_chunk_until_cpf_7a_wires_it() {
        use super::DEFAULT_CUDA_PREFILL_CHUNK;
        // Lane C's ruling of 2026-10-14 (option (a) on the review's B-2/B-1): the CUDA arm
        // keeps ignoring the key, with its start line, until lane M's CPF-7a integration
        // commit accepts it as the cap of `M_sess` (decision 0049, A6 item 2). Any value is
        // taken — 0 included, the CLI twin's case — and the field stays the inert 0; the
        // step's cap is `DEFAULT_CUDA_PREFILL_CHUNK` until the config is wired.
        let value = Q36Options {
            prefill_chunk: Some(0),
            ..Q36Options::default()
        };
        let got = settings_for(Some(&value), &Q36Env::default(), Backend::Cuda).unwrap();
        assert_eq!(
            got.ignored_mlx_keys,
            ["prefill_chunk"],
            "the chunk key is ignored with its start line"
        );
        assert_eq!(got.prefill_chunk, 0, "the CUDA field stays inert");
        assert_eq!(
            got.prefill_budget, None,
            "the CUDA arm plans no budget: the worker piece is NO_BUDGET_PIECE"
        );
        assert_eq!(
            DEFAULT_CUDA_PREFILL_CHUNK, 2048,
            "the step's cap stays the no-cap default"
        );
    }

    fn tiered(body: &str) -> Q36Options {
        serde_json::from_str(&format!("{{{body}}}")).expect("a valid qwen36 object")
    }

    #[test]
    fn the_tiered_member_is_off_by_default_and_takes_its_defaults_when_given() {
        let off = settings_for(None, &Q36Env::default(), Backend::Cuda).unwrap();
        assert_eq!(off.tiered, None, "tiered is never on without the member");
        let minimal = tiered(r#""tiered": {"slots": 40}"#);
        let got = settings_for(Some(&minimal), &Q36Env::default(), Backend::Cuda).unwrap();
        assert_eq!(
            got.tiered,
            Some(TieredSettings {
                slots: 40,
                threads: 6,
                timeout_ms: 10_000,
                preload: false
            })
        );
        let full =
            tiered(r#""tiered": {"slots": 61, "threads": 4, "timeout_ms": 5000, "preload": true}"#);
        let got = settings_for(Some(&full), &Q36Env::default(), Backend::Cuda).unwrap();
        assert_eq!(
            got.tiered,
            Some(TieredSettings {
                slots: 61,
                threads: 4,
                timeout_ms: 5000,
                preload: true
            })
        );
    }

    #[test]
    fn the_tiered_member_is_checked_by_name() {
        let refused = |body: &str, backend: Backend, needle: &str| {
            let options = tiered(body);
            let text = message(settings_for(Some(&options), &Q36Env::default(), backend));
            assert!(text.contains(needle), "{body}: {text}");
        };
        let cuda = Backend::Cuda;
        refused(r#""tiered": {}"#, cuda, "tiered needs slots");
        refused(r#""tiered": {"slots": 7}"#, cuda, "tiered.slots");
        refused(r#""tiered": {"slots": 257}"#, cuda, "tiered.slots");
        refused(
            r#""tiered": {"slots": 40, "threads": 0}"#,
            cuda,
            "tiered.threads",
        );
        refused(
            r#""tiered": {"slots": 40, "threads": 9}"#,
            cuda,
            "tiered.threads",
        );
        refused(
            r#""tiered": {"slots": 40, "timeout_ms": 0}"#,
            cuda,
            "tiered.timeout_ms",
        );
        refused(
            r#""tiered": {"slots": 40, "timeout_ms": 600001}"#,
            cuda,
            "tiered.timeout_ms",
        );
        refused(
            r#""cache_limit_bytes": 1024, "tiered": {"slots": 40}"#,
            Backend::Mlx,
            "tiered is a CUDA member",
        );
        refused(
            r#""capture": true, "tiered": {"slots": 40}"#,
            cuda,
            "exclude each other",
        );
        let unknown = serde_json::from_str::<Q36Options>(r#"{"tiered": {"slots": 40, "bank": 1}}"#);
        assert!(
            unknown.is_err(),
            "an unknown tiered member is refused by the parser"
        );
    }

    /// The eviction-cliff slice, reshaped by the review: the default LRU entry count of a
    /// `prefix_reuse` store stays the fixed addendum 2 six for EVERY spec (an entry's real
    /// cost grows with its depth - a snapshot owns a depth-sized copy of the attention
    /// buffers after the next append while its own charge is the DeltaNet state only - so
    /// no budget-derived default ships before the replay measurement); the operator's
    /// exact override is honored for every spec that reaches the store.
    #[test]
    fn the_prefix_store_entry_count_stays_six_unless_overridden() {
        let limits = |flag: bool, set: Option<usize>| super::prefix_store_limits_of(flag, set);
        assert_eq!(
            limits(true, None),
            StoreLimits {
                roots: 2,
                entries: 6
            },
            "the default stays the addendum 2 cap for every spec"
        );
        assert_eq!(
            limits(true, Some(32)),
            StoreLimits {
                roots: 2,
                entries: 32
            },
            "the exact override wins when set"
        );
        assert_eq!(
            limits(true, Some(6)),
            StoreLimits {
                roots: 2,
                entries: 6
            },
            "setting 6 equals the default"
        );
        assert_eq!(
            limits(false, Some(32)),
            StoreLimits::default(),
            "the flag off keeps the addendum 1 store, override and all"
        );
    }

    /// The override validates at boot: 0 and 4097 are refused by name, 6 is accepted, and
    /// the absent member stays `None` for the default cap.
    #[test]
    fn prefix_store_entries_validates_at_parse() {
        let refused = |value: usize| {
            error_text(settings_for(
                Some(&Q36Options {
                    cache_limit_bytes: Some(1024),
                    prefix_store_entries: Some(value),
                    ..Q36Options::default()
                }),
                &Q36Env::default(),
                Backend::Mlx,
            ))
        };
        assert!(
            refused(0).contains("prefix_store_entries"),
            "zero: {}",
            refused(0)
        );
        assert!(
            refused(4097).contains("prefix_store_entries"),
            "above the range: {}",
            refused(4097)
        );
        let six = settings(
            Some(&Q36Options {
                cache_limit_bytes: Some(1024),
                prefix_store_entries: Some(6),
                ..Q36Options::default()
            }),
            &Q36Env::default(),
        )
        .unwrap();
        assert_eq!(six.prefix_store_entries, Some(6), "the accepted override");
        let absent = settings(
            Some(&Q36Options {
                cache_limit_bytes: Some(1024),
                ..Q36Options::default()
            }),
            &Q36Env::default(),
        )
        .unwrap();
        assert_eq!(
            absent.prefix_store_entries, None,
            "absent stays the default cap"
        );
    }
}
