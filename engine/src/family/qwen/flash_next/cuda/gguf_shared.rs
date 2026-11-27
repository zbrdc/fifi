//! Staged single-row GGUF shared-expert execution. The routed MoE and serving remain separate.
#![cfg_attr(
    not(test),
    allow(dead_code, reason = "staged for GGUF decode integration")
)]

use super::{
    step::{FlashStep, refuse},
    step_plan::{Site, Slot},
};
use crate::{
    EngError,
    family::{exl3_pack::layer_name, project::cuda::CudaWeight},
    kernels::cuda::prefill::swiglu_rows::{SwigluRowsCall, SwigluRowsShape},
};

impl FlashStep<'_> {
    /// Writes the ungated shared expert output to `SharedOut`; the MoE combine applies its gate.
    pub(super) fn gguf_shared_expert(&self, layer: usize) -> Result<(), EngError> {
        if self.rows != 1 || self.shared_prefill() {
            return Err(refuse("GGUF shared expert supports decode only".into()));
        }
        if !self.layers.contains(&layer) {
            return Err(refuse(
                "GGUF shared expert layer is outside the owner window".into(),
            ));
        }
        let hidden = self.geometry.hidden;
        let shared = self.geometry.shared_inter;
        let gate = layer_name(layer, "mlp.shared_expert.gate_proj");
        let up = layer_name(layer, "mlp.shared_expert.up_proj");
        let down = layer_name(layer, "mlp.shared_expert.down_proj");
        for (name, k, n) in [
            (&gate, hidden, shared),
            (&up, hidden, shared),
            (&down, shared, hidden),
        ] {
            match self.weights.get(name) {
                Some(CudaWeight::Native { weight, .. })
                    if weight.k() == k && weight.n_rows() == n && weight.experts().is_none() => {}
                _ => {
                    return Err(refuse(format!(
                        "{name} is not a compatible native GGUF projection"
                    )));
                }
            }
        }
        let twice = shared
            .checked_mul(2)
            .ok_or_else(|| refuse("shared expert width overflows".into()))?;
        let x = self.slot(Slot::Mixed(Site::Mlp))?;
        let gu = self.slot(Slot::SharedGu)?;
        let act = self.slot(Slot::SharedAct)?;
        let out = self.slot(Slot::SharedOut)?;
        #[cfg(test)]
        if std::env::var_os("FIFI_FLASH_GGUF_TEST_SHARED_PRECISE").is_some() {
            let gate_out = gu.buffer().range(gu.offset(), shared * 4)?;
            let up_out = gu.buffer().range(gu.offset() + shared * 4, shared * 4)?;
            self.gguf_shared_precise(&gate, x, gate_out)?;
            self.gguf_shared_precise(&up, x, up_out)?;
        } else {
            self.project_group(&[(&gate, 0, shared), (&up, shared, shared)], x, &gu, 1)?;
        }
        #[cfg(not(test))]
        self.project_group(&[(&gate, 0, shared), (&up, shared, shared)], x, &gu, 1)?;
        self.ctx.swiglu_rows(
            &self.stream,
            SwigluRowsCall {
                shape: SwigluRowsShape {
                    rows: 1,
                    n_ff: shared,
                    up_off: shared,
                    ld_in: twice,
                    ld_out: shared,
                },
                gu,
                act,
            },
        )?;
        #[cfg(test)]
        if std::env::var_os("FIFI_FLASH_GGUF_TEST_SHARED_PRECISE").is_some() {
            return self.gguf_shared_precise(&down, act, out);
        }
        self.project(&down, act, out, 1)
    }

    /// Precision ablation only: Q6_K F32, or Q8_0 dequantizing GEMM with FP16
    /// intermediates. This is not a production dispatch or a parity guarantee.
    #[cfg(test)]
    fn gguf_shared_precise(
        &self,
        name: &str,
        input: crate::kernels::cuda::DeviceRange<'_>,
        output: crate::kernels::cuda::DeviceRange<'_>,
    ) -> Result<(), EngError> {
        use super::step::project_q6_f32;
        use crate::{
            family::step::rows::NativeRow,
            kernels::cuda::{prefill::gemm_dq::GemmDqTypedCall, resident::matvec_q::QuantType},
        };
        let Some(CudaWeight::Native { weight, .. }) = self.weights.get(name) else {
            return Err(refuse(format!("{name}: expected native shared projection")));
        };
        match weight.quant() {
            QuantType::Q6K => project_q6_f32(
                &self.ctx,
                &self.stream,
                NativeRow {
                    weight,
                    input,
                    output,
                },
            ),
            QuantType::Q80 => self.ctx.gemm_dq_codebook(
                &self.stream,
                QuantType::Q80,
                GemmDqTypedCall {
                    w: weight.buffer().range(0, weight.buffer().bytes())?,
                    x: input,
                    y: output,
                    k: weight.k(),
                    rows: weight.n_rows(),
                    ncols: 1,
                    x_col_stride: 0,
                    y_col_stride: 0,
                    x_f32: true,
                },
            ),
            ty => Err(refuse(format!(
                "{name}: unsupported shared precision probe {ty:?}"
            ))),
        }
    }
}
