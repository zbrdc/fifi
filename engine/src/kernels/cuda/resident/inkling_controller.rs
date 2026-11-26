//! Apply the Q/K/V cross-pass gates and weighted RMSNorm on the projection stream.

#[cfg(test)]
mod tests;

use std::mem::{align_of, size_of};

use crate::EngError;
use crate::kernels::cuda::{
    CudaContext, DeviceRange, DeviceStream,
    context::{Claim, require_apart, require_fit},
};

pub(crate) use super::super::native::resident::inkling_controller::{Arithmetic, MAX_ROW_ROLES};
use super::super::native::resident::inkling_controller::{
    launch, launch_rows, prepare, rebuild_gates,
};

impl CudaContext {
    /// Rebuild gates from the host's exact RMS scales. The workspace is packed
    /// as gate outputs, [input, message, gamma] triples, then scale pairs.
    pub(crate) fn inkling_rebuild_gates(
        &self,
        stream: &DeviceStream,
        workspace: DeviceRange<'_>,
        shape: [usize; 2],
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let workspace = self.raw_range(&workspace)?;
        self.launch_on(stream, |raw| rebuild_gates(raw, workspace, shape))
    }

    /// Enqueue gate-half RMS normalization and broadcast the shared gamma into
    /// the controller's raw triples. The five ranges are `[pairs, gamma, triples,
    /// gate_pairs, half_health]`; health is written separately from controller apply.
    pub(crate) fn inkling_prepare_controller(
        &self,
        stream: &DeviceStream,
        ranges: [DeviceRange<'_>; 5],
        shape_eps: ([usize; 2], f32),
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let [a, b, c, d, e] = ranges.each_ref().map(|r| self.raw_range(r));
        let raw = [a?, b?, c?, d?, e?];
        self.launch_on(stream, |stream| prepare(stream, raw, shape_eps))
    }

    /// The workspace packs three [input, message, gamma] triples, three
    /// [gain, forget] pairs and three normalized output rows, in that order.
    /// Health is three floats: zero on success or NaN on invalid arithmetic.
    pub(crate) fn inkling_controller_norm(
        &self,
        stream: &DeviceStream,
        workspace: DeviceRange<'_>,
        health: DeviceRange<'_>,
        (width, eps): (usize, f32),
    ) -> Result<(), EngError> {
        self.inkling_controller_norm_roles(stream, workspace, health, ([width, 3], eps))
    }

    /// Up to 16 independent controller roles over the caller's rows: per role
    /// `[input, message, gamma, gain, forget, out]`, `width` floats each, so nothing is packed
    /// into a workspace. Every output row shares no byte with any other row; `health` (one
    /// float per role) with none of them.
    pub(crate) fn inkling_controller_norm_rows(
        &self,
        stream: &DeviceStream,
        rows: &[[DeviceRange<'_>; 6]],
        health: DeviceRange<'_>,
        (width, eps): (usize, f32),
    ) -> Result<(), EngError> {
        let roles = rows.len();
        if !(1..=4096).contains(&width)
            || !(1..=MAX_ROW_ROLES).contains(&roles)
            || !eps.is_finite()
            || eps <= 0.0
        {
            return Err(EngError::BadRequest(format!(
                "the CUDA Inkling controller norm takes 1 to {MAX_ROW_ROLES} roles of 1 to 4096 columns and a finite positive eps, got roles {roles}, width {width}, eps {eps}"
            )));
        }
        let word = size_of::<f32>();
        let row = width * word;
        let names = [
            "Inkling controller input",
            "Inkling controller message",
            "Inkling controller gamma",
            "Inkling controller gain",
            "Inkling controller forget",
            "Inkling controller output",
        ];
        let mut claims = Vec::with_capacity(6 * roles);
        for role in rows {
            for (name, range) in names.iter().zip(role) {
                require_fit(name, range, row, align_of::<f32>())?;
                if range.bytes() != row {
                    return Err(EngError::BadRequest(format!(
                        "the CUDA {name} range holds {} bytes but the call needs exactly {row}",
                        range.bytes()
                    )));
                }
                claims.push(Claim::of(name, range, row));
            }
        }
        let status_words = roles * word;
        require_fit(
            "Inkling controller health",
            &health,
            status_words,
            align_of::<f32>(),
        )?;
        if health.bytes() != status_words {
            return Err(EngError::BadRequest(format!(
                "the CUDA Inkling controller health range holds {} bytes but the call needs exactly {status_words}",
                health.bytes()
            )));
        }
        let status = Claim::of("Inkling controller health", &health, status_words);
        for (i, claim) in claims.iter().enumerate() {
            for (j, other) in claims.iter().enumerate() {
                if i != j && (i % 6 == 5 || j % 6 == 5) {
                    require_apart(claim, other)?;
                }
            }
            require_apart(&status, claim)?;
        }
        self.ensure_usable()?;
        let raw = rows
            .iter()
            .flatten()
            .map(|range| self.raw_range(range))
            .collect::<Result<Vec<_>, _>>()?;
        let health = self.raw_range(&health)?;
        self.launch_on(stream, |stream| {
            launch_rows(stream, &raw, roles, health, (width, eps))
        })
    }

    /// Packs 1..=1536 independent roles: triples, pairs, then outputs.
    /// Always uses generic arithmetic. Health has one float per role;
    /// workspace and health must not overlap.
    pub(crate) fn inkling_controller_norm_roles(
        &self,
        stream: &DeviceStream,
        workspace: DeviceRange<'_>,
        health: DeviceRange<'_>,
        shape_eps: ([usize; 2], f32),
    ) -> Result<(), EngError> {
        self.inkling_controller_norm_with_arithmetic(
            stream,
            workspace,
            health,
            shape_eps,
            Arithmetic::Generic,
        )
    }

    /// Same packed layout and health contract; grouped prefill may opt into parity.
    pub(crate) fn inkling_controller_norm_with_arithmetic(
        &self,
        stream: &DeviceStream,
        workspace: DeviceRange<'_>,
        health: DeviceRange<'_>,
        (shape, eps): ([usize; 2], f32),
        arithmetic: Arithmetic,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let workspace = self.raw_range(&workspace)?;
        let health = self.raw_range(&health)?;
        self.launch_on(stream, |raw| {
            launch(raw, workspace, health, shape, eps, arithmetic)
        })
    }
}
