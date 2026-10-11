//! Safe wrapper of the step parameter kernels (RD-4d, decision 0047): the `StepParams` block,
//! `step_begin`, `step_advance` and the greedy `argmax` with token chaining.
//!
//! Design guidance only, no code copied. The ideas are the per-launch data in a small device
//! table, so one captured graph serves every position (lesson C3), and the device-written next
//! token (lesson C1; greedy only, no device sampler, decision 0047 item 8).
//!
//! The order of one resident step, which the caller (RD-9d) must keep: `step_begin` first, the
//! layers, then `argmax` as the LAST kernel that writes `status`, then `step_advance`, then one
//! 8-byte copy of the result pair. Before every launch the caller checks [`ensure_rows`] for the
//! whole look-ahead window: a device word out of range is a silent out-of-bounds access in a
//! consumer kernel that nothing can repair.
//!
//! Every call checks all arguments first and answers `BadRequest`; the ranges are converted with
//! `raw_range` before `launch_on`, so on a latched context a call with a bad range or argument
//! still answers `BadRequest`, and a call with good arguments answers the latch error
//! (`launch_on` checks the terminal flag itself).
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for RD-9a (decision 0047)")
)]

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{Claim, require_apart, require_fit};
use crate::kernels::cuda::native::resident::step_params as raw;
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};

/// Status mask 1: a router fault (OR-ed in by the router kernel, RD-5).
pub(crate) const STATUS_ROUTER_FAULT: i32 = 1;
/// Status mask 2: at least one logit is not finite (OR-ed in by `argmax`).
pub(crate) const STATUS_NON_FINITE_LOGITS: i32 = 2;
/// Status mask 4: the step state is invalid (set by `step_begin`, or by `embed` for a token id
/// outside the table).
pub(crate) const STATUS_INVALID_STEP_STATE: i32 = 4;
/// Bytes of the [`StepParams`] block.
pub(crate) const STEP_PARAMS_BYTES: usize = raw::STEP_PARAMS_BYTES;
/// Byte offset of `token_id` in the block.
pub(crate) const OFFSET_TOKEN_ID: usize = 0;
/// Byte offset of `pos` in the block.
pub(crate) const OFFSET_POS: usize = 4;
/// Byte offset of `kv_len` in the block.
pub(crate) const OFFSET_KV_LEN: usize = 8;
/// Byte offset of `kv_slot` in the block.
pub(crate) const OFFSET_KV_SLOT: usize = 12;
/// Byte offset of `status` in the block.
pub(crate) const OFFSET_STATUS: usize = 16;
/// Bytes of the argmax scratch.
pub(crate) const ARGMAX_PART_BYTES: usize = raw::ARGMAX_PART_BYTES;
/// Bytes of the result pair.
pub(crate) const RESULT_PAIR_BYTES: usize = raw::RESULT_PAIR_BYTES;
/// Largest `kv_len`: a step word never reaches 2^30.
pub(crate) const WORD_MAX: i32 = (1 << 30) - 1;
/// Largest `pos`, so that `kv_len = pos + 1` always fits.
pub(crate) const POS_MAX: i32 = WORD_MAX - 1;
/// Largest `m` of one `step_advance`.
pub(crate) const MAX_ADVANCE: usize = raw::MAX_ADVANCE;
/// Largest number of logits of one `argmax`.
pub(crate) const MAX_LOGITS: usize = raw::MAX_LOGITS;

/// The per-step parameter block that the resident kernels read from device memory.
///
/// 32 bytes, 4-byte aligned for the kernels (256-byte aligned in the arena). A valid state is
/// `0 <= pos <= POS_MAX`, `kv_len == pos + 1` and `kv_slot == pos`. `token_id` and `status` stay
/// the device source of truth; the [`ResultPair`] is a copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct StepParams {
    /// The token the next step embeds; written by the host, then by `argmax` (token chaining).
    pub(crate) token_id: i32,
    /// Position of token 0 of the step.
    pub(crate) pos: i32,
    /// Key count after the append of the step: `pos + 1`.
    pub(crate) kv_len: i32,
    /// Reserved; equals `pos`.
    pub(crate) kv_slot: i32,
    /// Flags by mask: 1 router fault, 2 non-finite logits, 4 invalid step state.
    pub(crate) status: i32,
    /// Padding to 32 bytes; nobody writes it.
    pub(crate) pad: [i32; 3],
}

impl StepParams {
    /// The start block of a step at `pos` with `token_id`: `kv_len = pos + 1`, `kv_slot = pos`
    /// and `status = 0`. `BadRequest` for a negative token or a `pos` outside `0..=POS_MAX`.
    pub(crate) fn new(token_id: i32, pos: i32) -> Result<Self, EngError> {
        if token_id < 0 {
            return Err(EngError::BadRequest(format!(
                "the step token id must not be negative, got {token_id}"
            )));
        }
        if !(0..=POS_MAX).contains(&pos) {
            return Err(EngError::BadRequest(format!(
                "the step position must be in 0..={POS_MAX}, got {pos}"
            )));
        }
        Ok(Self {
            token_id,
            pos,
            kv_len: pos + 1,
            kv_slot: pos,
            status: 0,
            pad: [0; 3],
        })
    }

    /// The block as the device holds it: five little-endian words, then 12 zero bytes of padding.
    pub(crate) fn to_bytes(self) -> [u8; STEP_PARAMS_BYTES] {
        let words = [
            self.token_id,
            self.pos,
            self.kv_len,
            self.kv_slot,
            self.status,
            self.pad[0],
            self.pad[1],
            self.pad[2],
        ];
        let mut bytes = [0_u8; STEP_PARAMS_BYTES];
        let (chunks, _) = bytes.as_chunks_mut::<4>();
        for (chunk, word) in chunks.iter_mut().zip(words) {
            *chunk = word.to_le_bytes();
        }
        bytes
    }

    /// The inverse of [`StepParams::to_bytes`].
    pub(crate) fn from_bytes(bytes: [u8; STEP_PARAMS_BYTES]) -> Self {
        let mut words = [0_i32; 8];
        let (chunks, _) = bytes.as_chunks::<4>();
        for (word, chunk) in words.iter_mut().zip(chunks) {
            *word = i32::from_le_bytes(*chunk);
        }
        Self {
            token_id: words[0],
            pos: words[1],
            kv_len: words[2],
            kv_slot: words[3],
            status: words[4],
            pad: [words[5], words[6], words[7]],
        }
    }

    /// The state a refused `step_advance` leaves: `pos`, `kv_len` and `kv_slot` are -1 and
    /// the status is 0. The consumer kernels reuse it for their poisoned-state tests.
    #[cfg(test)]
    pub(crate) fn poisoned() -> Self {
        Self {
            token_id: 0,
            pos: -1,
            kv_len: -1,
            kv_slot: -1,
            status: 0,
            pad: [0; 3],
        }
    }
}

/// The host check before a launch: `BadRequest` unless `pos <= POS_MAX`, `advance >= 1` and
/// `pos + advance <= min(rows, POS_MAX)`.
///
/// `advance` is the total advance of every step enqueued before the host next reads a result
/// pair: with token chaining the host may enqueue several replays ahead, and each one runs on the
/// words the previous one left. The device cannot refuse a step; a word out of range is an
/// out-of-bounds access in a consumer kernel.
pub(crate) fn ensure_rows(pos: usize, advance: usize, rows: usize) -> Result<(), EngError> {
    let top = usize::try_from(POS_MAX).map_or(0, |top| top.min(rows));
    let end = pos.checked_add(advance);
    if advance >= 1 && pos <= top && end.is_some_and(|end| end <= top) {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "the step needs positions {pos} + {advance}, but only {top} rows are available"
    )))
}

/// The status word of a step, by mask: 1 router fault, 2 non-finite logits, 4 invalid step state.
///
/// Mask 4 has two sources: a defect (the host check [`ensure_rows`] missed a case), or the
/// embedding gather, which sets it for a token id outside the vocabulary (RD-4c). The word cannot
/// tell the causes apart. Either way the resident state of the session is invalid until the
/// executor resets it. The wrapper never latches on a status word; it only returns the error value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StepStatus(pub(crate) i32);

impl StepStatus {
    /// True when mask 1 is set.
    pub(crate) fn router_fault(self) -> bool {
        self.0 & STATUS_ROUTER_FAULT != 0
    }

    /// True when mask 2 is set.
    pub(crate) fn non_finite_logits(self) -> bool {
        self.0 & STATUS_NON_FINITE_LOGITS != 0
    }

    /// True when mask 4 is set.
    pub(crate) fn invalid_step_state(self) -> bool {
        self.0 & STATUS_INVALID_STEP_STATE != 0
    }

    /// The bits that no mask names.
    pub(crate) fn unknown_bits(self) -> i32 {
        self.0 & !(STATUS_ROUTER_FAULT | STATUS_NON_FINITE_LOGITS | STATUS_INVALID_STEP_STATE)
    }

    /// The error the status maps to, or `None`. Precedence when several bits are set: mask 4,
    /// then mask 1, then mask 2. Unknown bits alone map to no error.
    pub(crate) fn error(self) -> Option<EngError> {
        let (operation, message) = if self.invalid_step_state() {
            (
                "step_params",
                "invalid step params or an out-of-range token id",
            )
        } else if self.router_fault() {
            ("route_experts", "the router reported a fault")
        } else if self.non_finite_logits() {
            ("sample", "logits hold a non-finite value")
        } else {
            return None;
        };
        Some(EngError::Backend {
            operation,
            message: message.to_owned(),
        })
    }
}

/// The two words one 8-byte device-to-host copy gives the host after `argmax`.
///
/// Three conditions hold (lane C, C2). (a) `result` is 8-byte aligned in its buffer. (b) `argmax`
/// is the LAST kernel that writes `status` in the step; a late writer makes the pair stale, and
/// [`ResultPair::matches_params`] detects it in tests. (c) The [`StepParams`] fields stay the
/// device source of truth; the pair is a copy, nothing on the device reads it, and no call writes
/// `StepParams` from a pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResultPair {
    /// The chosen token (-1 for a row with no comparable value).
    pub(crate) token: i32,
    /// The status after the OR of `argmax`.
    pub(crate) status: StepStatus,
}

impl ResultPair {
    /// Decodes the 8 bytes: the token, then the status, little endian.
    pub(crate) fn from_bytes(bytes: [u8; RESULT_PAIR_BYTES]) -> Self {
        Self {
            token: i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            status: StepStatus(i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])),
        }
    }

    /// True when the pair equals the `token_id` and `status` words of `params`. A test aid: it
    /// needs an extra copy of the block, so it is for tests and the eager path, not the graph.
    pub(crate) fn matches_params(&self, params: &StepParams) -> bool {
        self.token == params.token_id && self.status.0 == params.status
    }

    /// The token, unless the status maps to an error. The host maps the status to the error
    /// before it reads the token, so the filler token of a flagged row never reaches the sampler.
    pub(crate) fn into_token(self) -> Result<i32, EngError> {
        self.status.error().map_or(Ok(self.token), Err)
    }
}

/// `n` as the C length: `BadRequest` unless `1 <= n <= MAX_LOGITS`.
pub(crate) fn check_n(n: usize) -> Result<c_int, EngError> {
    match c_int::try_from(n) {
        Ok(value) if (1..=MAX_LOGITS).contains(&n) => Ok(value),
        _ => Err(EngError::BadRequest(format!(
            "the CUDA argmax takes 1 to {MAX_LOGITS} values, got {n}"
        ))),
    }
}

/// What one `argmax` reads and writes. `result` may be absent.
pub(crate) struct ArgmaxCall<'b> {
    /// Number of logits, `1..=MAX_LOGITS`.
    pub(crate) n: usize,
    /// The logits row: `4 * n` bytes of f32, 4-byte aligned. Read only.
    pub(crate) logits: DeviceRange<'b>,
    /// Scratch of [`ARGMAX_PART_BYTES`], 4-byte aligned; needs no initialization.
    pub(crate) part: DeviceRange<'b>,
    /// The [`StepParams`] block: `token_id` and `status` are written.
    pub(crate) params: DeviceRange<'b>,
    /// The result pair, [`RESULT_PAIR_BYTES`] at an 8-byte aligned offset, or `None`.
    pub(crate) result: Option<DeviceRange<'b>>,
}

impl CudaContext {
    /// Enqueues `step_begin` on `stream`: the status of `params` becomes 0, or 4 when the step
    /// state is invalid. Must be the first kernel of the step. `params` holds at least
    /// [`STEP_PARAMS_BYTES`] bytes at a 4-byte aligned offset.
    pub(crate) fn step_begin(
        &self,
        stream: &DeviceStream,
        params: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        let params_raw = self.raw_range(&params)?;
        require_fit("params", &params, STEP_PARAMS_BYTES, 4)?;
        self.launch_on(stream, move |raw_stream| {
            raw::step_begin_launch(raw_stream, params_raw)
        })
    }

    /// Enqueues `step_advance` by `m` (`1..=MAX_ADVANCE`) on `stream`: `pos`, `kv_len` and
    /// `kv_slot` move together when the state is valid and `pos + m <= POS_MAX`; in every other
    /// case all three become -1. `token_id` and `status` are never written. Must be the last
    /// kernel of the step; the caller has checked [`ensure_rows`] for the whole look-ahead window.
    pub(crate) fn step_advance(
        &self,
        stream: &DeviceStream,
        params: DeviceRange<'_>,
        m: u32,
    ) -> Result<(), EngError> {
        let params_raw = self.raw_range(&params)?;
        let m_raw = usize::try_from(m)
            .ok()
            .filter(|m| (1..=MAX_ADVANCE).contains(m))
            .and_then(|m| c_int::try_from(m).ok())
            .ok_or_else(|| {
                EngError::BadRequest(format!(
                    "the CUDA step advance takes 1 to {MAX_ADVANCE} positions, got {m}"
                ))
            })?;
        require_fit("params", &params, STEP_PARAMS_BYTES, 4)?;
        self.launch_on(stream, move |raw_stream| {
            raw::step_advance_launch(raw_stream, params_raw, m_raw)
        })
    }

    /// Enqueues the greedy argmax over one logits row on `stream` (two kernels). The token is
    /// the largest value, the lowest index on a tie; it is written to `params.token_id`. Status
    /// mask 2 is OR-ed into `params.status` when any logit is not finite, and the result pair
    /// (when given) gets the token and the status after the OR. Must be the last kernel that
    /// writes `status`; the host maps the status to its error before it reads the token
    /// ([`ResultPair::into_token`]).
    pub(crate) fn argmax(
        &self,
        stream: &DeviceStream,
        call: ArgmaxCall<'_>,
    ) -> Result<(), EngError> {
        let raw_call = self.plan_argmax(&call)?;
        self.launch_on(stream, move |raw_stream| {
            raw::argmax_launch(raw_stream, &raw_call)
        })
    }

    /// Every check of an argmax, then the native view of its ranges.
    fn plan_argmax<'b>(&self, call: &ArgmaxCall<'b>) -> Result<raw::ArgmaxRaw<'b>, EngError> {
        let params = self.raw_range(&call.params)?;
        let logits = self.raw_range(&call.logits)?;
        let part = self.raw_range(&call.part)?;
        let result = call
            .result
            .as_ref()
            .map(|range| self.raw_range(range))
            .transpose()?;
        let n = check_n(call.n)?;
        let logits_bytes = call.n.checked_mul(4).ok_or_else(|| {
            EngError::BadRequest("the CUDA argmax logits size overflows".to_owned())
        })?;
        require_fit("params", &call.params, STEP_PARAMS_BYTES, 4)?;
        require_fit("logits", &call.logits, logits_bytes, 4)?;
        require_fit("part", &call.part, ARGMAX_PART_BYTES, 4)?;
        if let Some(range) = &call.result {
            require_fit("result", range, RESULT_PAIR_BYTES, 8)?;
        }
        let mut claims = vec![
            Claim::of("logits", &call.logits, logits_bytes),
            Claim::of("part", &call.part, ARGMAX_PART_BYTES),
            Claim::of("params", &call.params, STEP_PARAMS_BYTES),
        ];
        if let Some(range) = &call.result {
            claims.push(Claim::of("result", range, RESULT_PAIR_BYTES));
        }
        for (i, a) in claims.iter().enumerate() {
            for b in &claims[i + 1..] {
                require_apart(a, b)?;
            }
        }
        Ok(raw::ArgmaxRaw {
            logits,
            part,
            params,
            result,
            n,
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
