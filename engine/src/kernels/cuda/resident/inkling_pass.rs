//! Row-group glue of the device-resident Inkling prefill pass, launch-only: the depth stream
//! of every prompt row (`Stream::mix`, `Stream::seal`, `Stream::add_current` of the host) and
//! the K/V preparation of a visit's rows (`prepare_attention_row`). Contracts:
//! `native/resident/inkling_pass.h`.
//!
//! A stream state is `sealed[blocks][rows][width]` (normalized rows, oldest first) followed by
//! `current[rows][width]`. `health` holds one word per row and is left alone, or set to quiet
//! NaN when a value of that row was not finite; the pass reads every health word once, at its
//! fence.

#[cfg(test)]
mod tests;

use std::mem::{align_of, size_of};

use crate::EngError;
use crate::kernels::cuda::{
    CudaContext, DeviceRange, DeviceStream,
    context::{Claim, require_apart, require_fit},
};

use super::super::native::resident::inkling_pass::{self as native};
pub(crate) use super::super::native::resident::inkling_pass::{HEAD, MAX_BLOCKS, TAPS};
use crate::kernels::cuda::native::resident::RawRange;

const MAX_WIDTH: usize = 8192;

pub(crate) struct PassMixCall<'a> {
    /// `blocks * rows` sealed rows of `width`, oldest block first.
    pub(crate) sealed: DeviceRange<'a>,
    pub(crate) current: DeviceRange<'a>,
    pub(crate) query: DeviceRange<'a>,
    /// Row `r` starts at word `r * out_stride` of `out`.
    pub(crate) out: DeviceRange<'a>,
    pub(crate) health: DeviceRange<'a>,
}

pub(crate) struct PassSealCall<'a> {
    /// `blocks + 1` sealed rows per prompt row; row `blocks` receives the new sealed row.
    pub(crate) sealed: DeviceRange<'a>,
    pub(crate) current: DeviceRange<'a>,
    pub(crate) next: DeviceRange<'a>,
    pub(crate) health: DeviceRange<'a>,
}

pub(crate) struct PassAddCall<'a> {
    pub(crate) current: DeviceRange<'a>,
    pub(crate) row: DeviceRange<'a>,
    pub(crate) health: DeviceRange<'a>,
}

pub(crate) struct PassPrepCall<'a> {
    /// Row `r`: its raw K at word 0 and its raw V at word `kv_heads * HEAD`, `raw_stride` words.
    pub(crate) raw: DeviceRange<'a>,
    /// The attention sidecar: row `r` starts at word `r * side_stride`.
    pub(crate) side: DeviceRange<'a>,
    /// The taps - 1 raw rows before the group, oldest first, one plane each for K and V.
    pub(crate) history: [DeviceRange<'a>; 2],
    /// `kv_heads * HEAD * TAPS` weights each.
    pub(crate) kernel: [DeviceRange<'a>; 2],
    /// Per row `kv_heads * TAPS` dynamic coefficients, or any range when `dynamic` is false.
    pub(crate) dynamic: DeviceRange<'a>,
    /// u32 per row; a non-finite prepared value sets bit 0.
    pub(crate) status: DeviceRange<'a>,
}

/// `[rows, blocks, first_block, width, out_stride]`.
pub(crate) type PassMixShape = [usize; 5];
/// `[rows, blocks, width]`.
pub(crate) type PassSealShape = [usize; 3];
/// `[rows, width]`.
pub(crate) type PassAddShape = [usize; 2];
/// `[raw_stride, side_stride, key_at, value_at, dynamic_stride]` in words.
pub(crate) type PassPrepLayout = [usize; 5];

impl CudaContext {
    /// `Stream::mix(first_block, query)` of every prompt row: mix row `j` is sealed row
    /// `blocks - 1 - j` for `j < blocks - first_block`, then the current row scaled by its RMS
    /// norm with `eps`; the weights are `exp(score - peak)` and the output the weighted rows
    /// in that order, divided by their mass.
    pub(crate) fn inkling_pass_mix(
        &self,
        stream: &DeviceStream,
        call: PassMixCall<'_>,
        [rows, blocks, first, width, out_stride]: PassMixShape,
        eps: f32,
    ) -> Result<(), EngError> {
        // Zero sealed blocks is the first request: the mix is the normalized current row alone.
        if rows == 0 || first > blocks || blocks >= MAX_BLOCKS {
            return Err(EngError::BadRequest(format!(
                "the Inkling pass mix takes 0 to {} sealed blocks and first_block at \
                 most blocks, got blocks {blocks}, first_block {first}, rows {rows}",
                MAX_BLOCKS - 1
            )));
        }
        if !eps.is_finite() || eps <= 0.0 {
            return Err(EngError::BadRequest(
                "the Inkling pass mix needs a finite positive eps".into(),
            ));
        }
        let row = width_row(width)?;
        let out = ((rows - 1) * out_stride + width) * size_of::<f32>();
        require_fit(
            "Inkling pass mix sealed",
            &call.sealed,
            blocks * rows * row,
            WORD,
        )?;
        require_fit("Inkling pass mix current", &call.current, rows * row, WORD)?;
        require_fit("Inkling pass mix query", &call.query, row, WORD)?;
        require_fit("Inkling pass mix output", &call.out, out, WORD)?;
        let health = rows * size_of::<f32>();
        require_fit("Inkling pass mix health", &call.health, health, WORD)?;
        let spans = [
            Claim::of("Inkling pass mix sealed", &call.sealed, blocks * rows * row),
            Claim::of("Inkling pass mix current", &call.current, rows * row),
            Claim::of("Inkling pass mix query", &call.query, row),
            Claim::of("Inkling pass mix output", &call.out, out),
            Claim::of("Inkling pass mix health", &call.health, health),
        ];
        let apart = |i: usize, j: usize| require_apart(&spans[i], &spans[j]);
        apart(1, 0)?;
        apart(2, 0)?;
        apart(2, 1)?;
        apart(3, 0)?;
        apart(3, 1)?;
        apart(3, 2)?;
        apart(4, 0)?;
        apart(4, 1)?;
        apart(4, 2)?;
        apart(4, 3)?;
        self.ensure_usable()?;
        let ranges = [
            self.raw_range(&call.sealed)?,
            self.raw_range(&call.current)?,
            self.raw_range(&call.query)?,
            self.raw_range(&call.out)?,
            self.raw_range(&call.health)?,
        ];
        self.launch_on(stream, |stream| {
            native::mix(
                stream,
                ranges,
                [rows, blocks, first, width, out_stride],
                eps,
            )
        })
    }

    /// `Stream::seal(next)` of every prompt row: sealed row `blocks` takes the normalized current
    /// row, then the current row takes `next`.
    pub(crate) fn inkling_pass_seal(
        &self,
        stream: &DeviceStream,
        call: PassSealCall<'_>,
        [rows, blocks, width]: PassSealShape,
        eps: f32,
    ) -> Result<(), EngError> {
        // Sealed row `blocks` receives the new row, so a stream may hold `MAX_BLOCKS - 1` already.
        if rows == 0 || blocks >= MAX_BLOCKS {
            return Err(EngError::BadRequest(format!(
                "the Inkling pass seal takes 0 to {} sealed blocks, got {blocks}",
                MAX_BLOCKS - 1
            )));
        }
        if !eps.is_finite() || eps <= 0.0 {
            return Err(EngError::BadRequest(
                "the Inkling pass seal needs a finite positive eps".into(),
            ));
        }
        let row = width_row(width)?;
        let current = rows * row;
        let health = rows * size_of::<f32>();
        require_fit(
            "Inkling pass seal sealed",
            &call.sealed,
            (blocks + 1) * current,
            WORD,
        )?;
        require_fit("Inkling pass seal current", &call.current, current, WORD)?;
        require_fit("Inkling pass seal next", &call.next, current, WORD)?;
        require_fit("Inkling pass seal health", &call.health, health, WORD)?;
        let spans = [
            Claim::of(
                "Inkling pass seal sealed",
                &call.sealed,
                (blocks + 1) * current,
            ),
            Claim::of("Inkling pass seal current", &call.current, current),
            Claim::of("Inkling pass seal next", &call.next, current),
            Claim::of("Inkling pass seal health", &call.health, health),
        ];
        let apart = |i: usize, j: usize| require_apart(&spans[i], &spans[j]);
        apart(1, 0)?;
        apart(2, 0)?;
        apart(2, 1)?;
        apart(3, 0)?;
        apart(3, 1)?;
        apart(3, 2)?;
        self.ensure_usable()?;
        let ranges = [
            self.raw_range(&call.sealed)?,
            self.raw_range(&call.current)?,
            self.raw_range(&call.next)?,
            self.raw_range(&call.health)?,
        ];
        self.launch_on(stream, |stream| {
            native::seal(stream, ranges, [rows, blocks, width], eps)
        })
    }

    /// `Stream::add_current(row)` of every prompt row.
    pub(crate) fn inkling_pass_add(
        &self,
        stream: &DeviceStream,
        call: PassAddCall<'_>,
        [rows, width]: PassAddShape,
    ) -> Result<(), EngError> {
        if rows == 0 {
            return Err(EngError::BadRequest(
                "the Inkling pass add needs at least one row".into(),
            ));
        }
        let row = width_row(width)?;
        let bytes = rows * row;
        let health = rows * size_of::<f32>();
        require_fit("Inkling pass add current", &call.current, bytes, WORD)?;
        require_fit("Inkling pass add row", &call.row, bytes, WORD)?;
        require_fit("Inkling pass add health", &call.health, health, WORD)?;
        let current = Claim::of("Inkling pass add current", &call.current, bytes);
        let row = Claim::of("Inkling pass add row", &call.row, bytes);
        let health = Claim::of("Inkling pass add health", &call.health, health);
        require_apart(&current, &row)?;
        require_apart(&current, &health)?;
        require_apart(&row, &health)?;
        self.ensure_usable()?;
        let ranges = [
            self.raw_range(&call.current)?,
            self.raw_range(&call.row)?,
            self.raw_range(&call.health)?,
        ];
        self.launch_on(stream, |stream| native::add(stream, ranges, [rows, width]))
    }

    /// `prepare_attention_row` of the host for the K and V planes of every row: the short causal
    /// convolution in the host's lag order, then the per-head L2 clip of K and the per-head RMS
    /// norm of V. The result goes to the sidecar rows the attention tile reads.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn inkling_pass_prep(
        &self,
        stream: &DeviceStream,
        call: PassPrepCall<'_>,
        rows: usize,
        kv_heads: usize,
        [raw_stride, side_stride, key_at, value_at, dynamic_stride]: PassPrepLayout,
        (qk_limit, eps): (f32, f32),
        dynamic: bool,
    ) -> Result<(), EngError> {
        let channels = kv_heads
            .checked_mul(HEAD)
            .filter(|&channels| (1..=MAX_WIDTH).contains(&channels) && kv_heads >= 1)
            .ok_or_else(|| {
                EngError::BadRequest("the Inkling pass prep needs 1 to 64 K/V heads".into())
            })?;
        if rows == 0
            || raw_stride < 2 * channels
            || key_at + channels > side_stride
            || value_at + channels > side_stride
            || (dynamic && dynamic_stride < kv_heads * TAPS)
            || [qk_limit, eps].iter().any(|&v| !v.is_finite() || v <= 0.0)
        {
            return Err(EngError::BadRequest(
                "the Inkling pass prep layout or limits are invalid".into(),
            ));
        }
        let word = size_of::<f32>();
        let raw = ((rows - 1) * raw_stride + 2 * channels) * word;
        let side = ((rows - 1) * side_stride + key_at.max(value_at) + channels) * word;
        let history = (TAPS - 1) * channels * word;
        let kernel = channels * TAPS * word;
        let dynamic_bytes = if dynamic {
            ((rows - 1) * dynamic_stride + 2 * kv_heads * TAPS) * word
        } else {
            word
        };
        let status = rows * size_of::<u32>();
        require_fit("Inkling pass prep raw", &call.raw, raw, word)?;
        require_fit("Inkling pass prep side", &call.side, side, word)?;
        require_fit(
            "Inkling pass prep history K",
            &call.history[0],
            history,
            word,
        )?;
        require_fit(
            "Inkling pass prep history V",
            &call.history[1],
            history,
            word,
        )?;
        require_fit("Inkling pass prep kernel K", &call.kernel[0], kernel, word)?;
        require_fit("Inkling pass prep kernel V", &call.kernel[1], kernel, word)?;
        require_fit(
            "Inkling pass prep dynamic",
            &call.dynamic,
            dynamic_bytes,
            word,
        )?;
        require_fit(
            "Inkling pass prep status",
            &call.status,
            status,
            align_of::<u32>(),
        )?;
        let spans = [
            Claim::of("Inkling pass prep raw", &call.raw, raw),
            Claim::of("Inkling pass prep side", &call.side, side),
            Claim::of("Inkling pass prep history K", &call.history[0], history),
            Claim::of("Inkling pass prep history V", &call.history[1], history),
            Claim::of("Inkling pass prep kernel K", &call.kernel[0], kernel),
            Claim::of("Inkling pass prep kernel V", &call.kernel[1], kernel),
            Claim::of("Inkling pass prep dynamic", &call.dynamic, dynamic_bytes),
            Claim::of("Inkling pass prep status", &call.status, status),
        ];
        for i in 0..spans.len() {
            for j in 0..i {
                require_apart(&spans[i], &spans[j])?;
            }
        }
        self.ensure_usable()?;
        let ranges = [
            self.raw_range(&call.raw)?,
            self.raw_range(&call.side)?,
            self.raw_range(&call.history[0])?,
            self.raw_range(&call.history[1])?,
            self.raw_range(&call.kernel[0])?,
            self.raw_range(&call.kernel[1])?,
            self.raw_range(&call.dynamic)?,
            self.raw_range(&call.status)?,
        ];
        self.launch_on(stream, |stream| {
            native::prep(
                stream,
                ranges,
                [
                    rows,
                    raw_stride,
                    side_stride,
                    key_at,
                    value_at,
                    dynamic_stride,
                    kv_heads,
                ],
                [qk_limit, eps],
                dynamic,
            )
        })
    }
}

/// One range of a pass launch: its name, the range, the bytes the launch uses, whether the
/// launch writes those bytes, and whether it must be u32- rather than f32-aligned.
struct Part<'a> {
    name: &'static str,
    range: DeviceRange<'a>,
    bytes: usize,
    writer: bool,
}

/// Every part holds its extent in one allocation, and every writer is apart from every other
/// part it does not replace (`replaced` pairs a writer with the range it overwrites).
fn prove(parts: &[Part<'_>], replaced: &[(usize, usize)]) -> Result<(), EngError> {
    for (i, part) in parts.iter().enumerate() {
        require_fit(part.name, &part.range, part.bytes, WORD)?;
        if part.writer {
            let claim = Claim::of(part.name, &part.range, part.bytes);
            for (j, other) in parts.iter().enumerate() {
                if i != j && part.range.offset() != other.range.offset() {
                    require_apart(&claim, &Claim::of(other.name, &other.range, other.bytes))?;
                }
            }
        }
    }
    let _ = replaced;
    Ok(())
}

impl CudaContext {
    /// `control_input_with_scales` of every prompt row: `gate[r] = [norm(in[r]) | norm(msg[r])]`
    /// with `eps`; `gate` row `r` starts at word `r * gate_stride`.
    pub(crate) fn inkling_pass_gate(
        &self,
        stream: &DeviceStream,
        [input, message, gate, health]: [DeviceRange<'_>; 4],
        [rows, width, gate_stride]: [usize; 3],
        eps: f32,
    ) -> Result<(), EngError> {
        let row = width_row(width)?;
        if rows == 0 || !eps.is_finite() || eps <= 0.0 {
            return Err(EngError::BadRequest(
                "the Inkling pass gate needs rows and a finite positive eps".into(),
            ));
        }
        let extent = (rows - 1) * gate_stride + 2 * width;
        let parts = [
            Part {
                name: "Inkling pass gate input",
                range: input,
                bytes: rows * row,
                writer: false,
            },
            Part {
                name: "Inkling pass gate message",
                range: message,
                bytes: rows * row,
                writer: false,
            },
            Part {
                name: "Inkling pass gate output",
                range: gate,
                bytes: extent * WORD,
                writer: true,
            },
            Part {
                name: "Inkling pass gate health",
                range: health,
                bytes: rows * WORD,
                writer: true,
            },
        ];
        prove(&parts, &[])?;
        self.ensure_usable()?;
        let ranges = [
            self.raw_range(&parts[0].range)?,
            self.raw_range(&parts[1].range)?,
            self.raw_range(&parts[2].range)?,
            self.raw_range(&parts[3].range)?,
        ];
        let shape = [rows, width, width, width, gate_stride];
        self.launch_on(stream, |stream| native::gate(stream, ranges, shape, eps))
    }

    /// `apply_control` then `norm` of every prompt row (`control`), or `norm` alone: the row
    /// is normalized with `eps` and scaled by `gamma`. `out` may be `input`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn inkling_pass_apply(
        &self,
        stream: &DeviceStream,
        [input, message, gain, forget, gamma, out, health]: [DeviceRange<'_>; 7],
        [rows, width]: [usize; 2],
        eps: f32,
        control: bool,
    ) -> Result<(), EngError> {
        let row = width_row(width)?;
        if rows == 0 || !eps.is_finite() || eps <= 0.0 {
            return Err(EngError::BadRequest(
                "the Inkling pass apply needs rows and a finite positive eps".into(),
            ));
        }
        let rows_bytes = rows * row;
        fn optional<'a>(
            range: DeviceRange<'a>,
            name: &'static str,
            bytes: usize,
        ) -> Result<Part<'a>, EngError> {
            Ok(Part {
                name,
                range,
                bytes,
                writer: false,
            })
        }
        let optional_bytes = if control { rows_bytes } else { WORD };
        let parts = [
            Part {
                name: "Inkling pass apply input",
                range: input,
                bytes: rows_bytes,
                writer: false,
            },
            optional(message, "Inkling pass apply message", optional_bytes)?,
            optional(gain, "Inkling pass apply gain", optional_bytes)?,
            optional(forget, "Inkling pass apply forget", optional_bytes)?,
            Part {
                name: "Inkling pass apply gamma",
                range: gamma,
                bytes: row,
                writer: false,
            },
            Part {
                name: "Inkling pass apply output",
                range: out,
                bytes: rows_bytes,
                writer: true,
            },
            Part {
                name: "Inkling pass apply health",
                range: health,
                bytes: rows * WORD,
                writer: true,
            },
        ];
        prove(&parts, &[])?;
        self.ensure_usable()?;
        let mut ranges = Vec::with_capacity(7);
        for part in &parts {
            ranges.push(self.raw_range(&part.range)?);
        }
        let ranges = <[RawRange<'_>; 7]>::try_from(ranges).unwrap_or_else(|_| unreachable!());
        self.launch_on(stream, |stream| {
            native::apply(stream, ranges, [rows, width, width], eps, control)
        })
    }

    /// The last three rows of `old` (its `past` valid rows, oldest first) followed by the rows
    /// of `source`, oldest first, into the three rows of `out`.
    pub(crate) fn inkling_pass_tails(
        &self,
        stream: &DeviceStream,
        [source, old, out]: [DeviceRange<'_>; 3],
        [rows, stride, width, past]: [usize; 4],
    ) -> Result<(), EngError> {
        let row = width_row(width)?;
        if rows == 0 || past >= TAPS || stride < width {
            return Err(EngError::BadRequest(
                "the Inkling pass tails need rows and fewer past rows than taps".into(),
            ));
        }
        let parts = [
            Part {
                name: "Inkling pass tails source",
                range: source,
                bytes: ((rows - 1) * stride + width) * WORD,
                writer: false,
            },
            Part {
                name: "Inkling pass tails old",
                range: old,
                bytes: (TAPS - 1) * row,
                writer: false,
            },
            Part {
                name: "Inkling pass tails output",
                range: out,
                bytes: (TAPS - 1) * row,
                writer: true,
            },
        ];
        prove(&parts, &[])?;
        self.ensure_usable()?;
        let ranges = [
            self.raw_range(&parts[0].range)?,
            self.raw_range(&parts[1].range)?,
            self.raw_range(&parts[2].range)?,
        ];
        self.launch_on(stream, |stream| {
            native::tails(stream, ranges, [rows, stride, width, past])
        })
    }

    /// `side[r * stride + pos_at] = u32(first + r)`: the position words of one visit's
    /// attention input sidecar.
    pub(crate) fn inkling_pass_positions(
        &self,
        stream: &DeviceStream,
        side: DeviceRange<'_>,
        [rows, stride, pos_at, first]: [usize; 4],
    ) -> Result<(), EngError> {
        if rows == 0 || pos_at >= stride || !side.offset().is_multiple_of(WORD) {
            return Err(EngError::BadRequest(
                "the Inkling pass positions need rows and a position column inside the row".into(),
            ));
        }
        require_fit(
            "Inkling pass positions sidecar",
            &side,
            ((rows - 1) * stride + pos_at + 1) * WORD,
            WORD,
        )?;
        self.ensure_usable()?;
        let range = self.raw_range(&side)?;
        self.launch_on(stream, |stream| {
            native::positions(stream, range, [rows, stride, pos_at, first])
        })
    }

    /// Sets the `count` u32 status words of `words` to zero.
    pub(crate) fn inkling_pass_zero(
        &self,
        stream: &DeviceStream,
        words: DeviceRange<'_>,
        count: usize,
    ) -> Result<(), EngError> {
        if count == 0 || count > 1 << 20 {
            return Err(EngError::BadRequest(
                "the Inkling pass zero takes 1 to 2^20 words".into(),
            ));
        }
        require_fit(
            "Inkling pass zero words",
            &words,
            count * size_of::<u32>(),
            WORD,
        )?;
        self.ensure_usable()?;
        let range = self.raw_range(&words)?;
        self.launch_on(stream, |stream| native::zero(stream, range, count))
    }
}

fn width_row(width: usize) -> Result<usize, EngError> {
    if !(1..=MAX_WIDTH).contains(&width) {
        return Err(EngError::BadRequest(format!(
            "the Inkling pass kernels take 1 to {MAX_WIDTH} columns, got {width}"
        )));
    }
    Ok(width * size_of::<f32>())
}

const WORD: usize = size_of::<f32>();
