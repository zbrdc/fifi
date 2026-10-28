//! Paged decode preflights both kernels before either can enqueue; the quantizer is unchanged.
use super::*;
use crate::kernels::cuda::{
    native::resident::paged_position,
    resident::{
        paged_position::{PagedPositionCall, raw_call},
        step_params::OFFSET_STATUS,
    },
};

impl CudaContext {
    pub(crate) fn kv8_paged_append(
        &self,
        stream: &DeviceStream,
        shape: Kv8Shape,
        data: Kv8Data<'_>,
        position: PagedPositionCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_uncaptured("paged KV decode append")?;
        self.check_launch_stream(stream)?;
        let ints = check_shape("paged KV decode append", &shape, 1)?;
        check_fit(&data, &shape)?;
        check_apart(&data, &shape)?;
        let raw = raw_data(self, &data, ints)?;
        let control = raw_call(self, &position, shape.cells)?;
        let params = Claim::of("paged params", &position.params, 32);
        let local = Claim::of("paged local", &position.local, 4);
        for input in [
            Claim::of("k", &data.k, shape.row_bytes()),
            Claim::of("v", &data.v, shape.row_bytes()),
            Claim::of("k_codes", &data.k_codes, shape.code_bytes()),
            Claim::of("k_scales", &data.k_scales, shape.scale_bytes()),
            Claim::of("v_codes", &data.v_codes, shape.code_bytes()),
            Claim::of("v_scales", &data.v_scales, shape.scale_bytes()),
        ] {
            require_apart(&params, &input)?;
            require_apart(&local, &input)?;
        }
        let status = position
            .params
            .buffer()
            .range(position.params.offset() + OFFSET_STATUS, 4)?;
        let words = Kv8WordsRaw {
            pos: self.raw_range(&position.local)?,
            status: self.raw_range(&status)?,
        };
        self.launch_on(stream, |raw_stream| {
            paged_position::launch(raw_stream, &control)
        })?;
        // Separate dispatches record the completed first enqueue even if the second fails.
        self.launch_on(stream, move |raw_stream| {
            kv8_append_launch(raw_stream, raw, words)
        })
    }
}
