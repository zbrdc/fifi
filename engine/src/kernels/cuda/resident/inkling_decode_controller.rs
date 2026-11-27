//! The fused controller chain of one Inkling 2 decode visit (`FIFI_INKLING_CUDA_DECODE_GLUE`),
//! launch-only: the gate norms and the down reduction of one to four roles in one launch, both
//! controller projections and the controller norm in another
//! (`native/resident/inkling_decode_controller.h`). Every operation is the one of the launches the
//! chain replaces, so a visit's bits do not depend on the switch. The decode glue selects the
//! chain; with it off the separate launches run.

#[cfg(test)]
mod tests;

use std::mem::{align_of, size_of};

use super::super::native::resident::inkling_decode_controller::{self as native};
pub(crate) use super::super::native::resident::inkling_decode_controller::{
    CONTROLLER_MAX_ROLES, PAIR_RANK,
};
use crate::EngError;
use crate::kernels::cuda::{
    CudaContext, DeviceRange, DeviceStream,
    context::{Claim, require_apart, require_fit},
};

const MAX_WIDTH: usize = 4096;
const MAX_RANK: usize = 1024;
const WORD: usize = size_of::<f32>();

/// One role of the gate launch: `input` and `message` (`width` floats each) are normalized
/// into `gate` (`2 * width`: the normed input, then the normed message), `down` (`rank` rows of
/// `2 * width` F32) reduces `gate` into `reduced` (`rank`), and each health word takes a quiet
/// NaN when its half was not finite (the two may be one word).
#[derive(Clone, Copy)]
pub(crate) struct GateRole<'b> {
    pub(crate) input: DeviceRange<'b>,
    pub(crate) message: DeviceRange<'b>,
    pub(crate) gate: DeviceRange<'b>,
    pub(crate) down: DeviceRange<'b>,
    pub(crate) reduced: DeviceRange<'b>,
    pub(crate) health: [DeviceRange<'b>; 2],
}

/// One role of the pair launch: `gain_w` and `forget_w` (`width` rows of [`PAIR_RANK`] F32)
/// project `reduced` into `gain` and `forget`, and the last block writes `out = rmsnorm((input
/// - tanh(forget) * input) + tanh(gain) * message) * gamma` and `health`, as
/// `inkling_controller_norm_rows` does. Every row but the weights holds `width` floats.
#[derive(Clone, Copy)]
pub(crate) struct ApplyRole<'b> {
    pub(crate) input: DeviceRange<'b>,
    pub(crate) message: DeviceRange<'b>,
    pub(crate) gamma: DeviceRange<'b>,
    pub(crate) reduced: DeviceRange<'b>,
    pub(crate) gain_w: DeviceRange<'b>,
    pub(crate) forget_w: DeviceRange<'b>,
    pub(crate) gain: DeviceRange<'b>,
    pub(crate) forget: DeviceRange<'b>,
    pub(crate) out: DeviceRange<'b>,
    pub(crate) health: DeviceRange<'b>,
}

fn check_common(
    roles: usize,
    width: usize,
    tickets: &DeviceRange<'_>,
    eps: f32,
) -> Result<(), EngError> {
    if !(1..=CONTROLLER_MAX_ROLES).contains(&roles)
        || !(16..=MAX_WIDTH).contains(&width)
        || !width.is_multiple_of(16)
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(EngError::BadRequest(format!(
            "the CUDA Inkling decode controller takes 1 to {CONTROLLER_MAX_ROLES} roles of 16 to \
             {MAX_WIDTH} columns in steps of 16 and a finite positive eps, got roles {roles}, \
             width {width}, eps {eps}"
        )));
    }
    let bytes = roles * size_of::<u32>();
    require_fit(
        "Inkling controller tickets",
        tickets,
        bytes,
        align_of::<u32>(),
    )?;
    if tickets.bytes() != bytes {
        return Err(EngError::BadRequest(format!(
            "the CUDA Inkling controller tickets range holds {} bytes but the call needs exactly {bytes}",
            tickets.bytes()
        )));
    }
    Ok(())
}

/// A claim on the bytes a launch uses of `range`.
fn claim<'b>(
    name: &'static str,
    range: &DeviceRange<'b>,
    bytes: usize,
    align: usize,
) -> Result<Claim<'b>, EngError> {
    require_fit(name, range, bytes, align)?;
    Ok(Claim::of(name, range, bytes))
}

/// Every written claim shares no byte with the tickets, with a read claim, with a claim of
/// another role and with another written claim, except the pairs `allow_equal` names.
fn prove(
    writes: &[Claim<'_>],
    reads: &[Claim<'_>],
    tickets: &Claim<'_>,
    others: &[Claim<'_>],
    allow_equal: impl Fn(usize, usize) -> bool,
) -> Result<(), EngError> {
    for (i, write) in writes.iter().enumerate() {
        require_apart(write, tickets)?;
        for read in reads.iter().chain(others) {
            require_apart(write, read)?;
        }
        for (j, other) in writes.iter().enumerate().skip(i + 1) {
            if !allow_equal(i, j) {
                require_apart(write, other)?;
            }
        }
    }
    Ok(())
}

impl CudaContext {
    /// `gate = [rms(input) | rms(message)]` with `gate_eps` and `reduced = down * gate` for
    /// every role; the role's ticket is cleared for the pair launch that follows.
    pub(crate) fn decode_controller_gate_down(
        &self,
        stream: &DeviceStream,
        roles: &[GateRole<'_>],
        tickets: DeviceRange<'_>,
        [width, rank]: [usize; 2],
        gate_eps: f32,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        check_common(roles.len(), width, &tickets, gate_eps)?;
        if !(8..=MAX_RANK).contains(&rank) || !rank.is_multiple_of(8) {
            return Err(EngError::BadRequest(format!(
                "the CUDA Inkling decode controller takes 8 to {MAX_RANK} down rows in steps of 8, got {rank}"
            )));
        }
        let tickets_claim = Claim::of("Inkling controller tickets", &tickets, roles.len() * 4);
        let [row, gate, reduced] = [width * WORD, 2 * width * WORD, rank * WORD];
        let mut others: Vec<Claim<'_>> = Vec::new();
        let mut planned = Vec::with_capacity(roles.len());
        for role in roles {
            let input = claim("Inkling controller input", &role.input, row, 4)?;
            let message = claim("Inkling controller message", &role.message, row, 4)?;
            let gate_row = claim("Inkling controller gate", &role.gate, gate, 16)?;
            let down = claim("Inkling controller down", &role.down, gate * rank, 16)?;
            let reduced_row = claim("Inkling controller reduced", &role.reduced, reduced, 4)?;
            let health = [
                claim("Inkling controller gate health", &role.health[0], WORD, 4)?,
                claim("Inkling controller gate health", &role.health[1], WORD, 4)?,
            ];
            // The two health words may be one word: each writer stores the same quiet NaN.
            prove(
                &[gate_row, reduced_row, health[0], health[1]],
                &[input, message, down],
                &tickets_claim,
                &others,
                |i, j| i == 2 && j == 3,
            )?;
            others.extend([input, message, gate_row, reduced_row, health[0], health[1]]);
            planned.push(native::GateRole {
                input: self.raw_range(&role.input)?,
                message: self.raw_range(&role.message)?,
                gate: self.raw_range(&role.gate)?,
                down: self.raw_range(&role.down)?,
                reduced: self.raw_range(&role.reduced)?,
                health: [
                    self.raw_range(&role.health[0])?,
                    self.raw_range(&role.health[1])?,
                ],
            });
        }
        let tickets = self.raw_range(&tickets)?;
        self.launch_on(stream, |raw| {
            native::gate_down(raw, &planned, tickets, [width, rank], gate_eps)
        })
    }

    /// `gain = gain_w * reduced`, `forget = forget_w * reduced`, then the controller norm into
    /// `out` and `health`, for every role; the last block of a role runs its norm.
    pub(crate) fn decode_controller_pair_norm(
        &self,
        stream: &DeviceStream,
        roles: &[ApplyRole<'_>],
        tickets: DeviceRange<'_>,
        width: usize,
        eps: f32,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        check_common(roles.len(), width, &tickets, eps)?;
        let tickets_claim = Claim::of("Inkling controller tickets", &tickets, roles.len() * 4);
        let row = width * WORD;
        let [reduced, projection] = [PAIR_RANK * WORD, row * PAIR_RANK];
        let mut others: Vec<Claim<'_>> = Vec::new();
        let mut planned = Vec::with_capacity(roles.len());
        for role in roles {
            let input = claim("Inkling controller input", &role.input, row, 4)?;
            let message = claim("Inkling controller message", &role.message, row, 4)?;
            let gamma = claim("Inkling controller gamma", &role.gamma, row, 4)?;
            let reduced_row = claim("Inkling controller reduced", &role.reduced, reduced, 4)?;
            let gain_w = claim(
                "Inkling controller gain weights",
                &role.gain_w,
                projection,
                16,
            )?;
            let forget_w = claim(
                "Inkling controller forget weights",
                &role.forget_w,
                projection,
                16,
            )?;
            let gain = claim("Inkling controller gain", &role.gain, row, 4)?;
            let forget = claim("Inkling controller forget", &role.forget, row, 4)?;
            let out = claim("Inkling controller output", &role.out, row, 4)?;
            let health = claim("Inkling controller health", &role.health, WORD, 4)?;
            let reads = [input, message, gamma, reduced_row, gain_w, forget_w];
            prove(
                &[gain, forget, out, health],
                &reads,
                &tickets_claim,
                &others,
                |_, _| false,
            )?;
            others.extend([
                input,
                message,
                gamma,
                reduced_row,
                gain,
                forget,
                out,
                health,
            ]);
            planned.push(native::ApplyRole {
                input: self.raw_range(&role.input)?,
                message: self.raw_range(&role.message)?,
                gamma: self.raw_range(&role.gamma)?,
                reduced: self.raw_range(&role.reduced)?,
                gain_w: self.raw_range(&role.gain_w)?,
                forget_w: self.raw_range(&role.forget_w)?,
                gain: self.raw_range(&role.gain)?,
                forget: self.raw_range(&role.forget)?,
                out: self.raw_range(&role.out)?,
                health: self.raw_range(&role.health)?,
            });
        }
        let tickets = self.raw_range(&tickets)?;
        self.launch_on(stream, |raw| {
            native::pair_norm(raw, &planned, tickets, width, eps)
        })
    }
}
