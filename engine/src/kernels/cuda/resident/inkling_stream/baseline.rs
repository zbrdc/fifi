use super::*;
impl CudaContext {
    pub(crate) fn inkling_stream_norm(
        &self,
        stream: &DeviceStream,
        call: NormCall<'_>,
        (width, eps): (usize, f32),
    ) -> Result<(), EngError> {
        check_shape("norm", 0, 0, width, eps)?;
        let row = width * size_of::<f32>();
        let health = size_of::<f32>();
        check_row("Inkling stream norm source", &call.src, row)?;
        check_row("Inkling stream norm output", &call.dst, row)?;
        check_row("Inkling stream health", &call.health, health)?;
        writes_apart(
            [
                Claim::of("Inkling stream norm output", &call.dst, row),
                Claim::of("Inkling stream health", &call.health, health),
            ],
            [Claim::of("Inkling stream norm source", &call.src, row)],
        )?;
        self.ensure_usable()?;
        let (src, dst, health) = (
            self.raw_range(&call.src)?,
            self.raw_range(&call.dst)?,
            self.raw_range(&call.health)?,
        );
        self.launch_on(stream, |raw| {
            super::super::super::native::resident::inkling_stream_baseline::norm(
                raw, src, dst, health, width, eps,
            )
        })
    }
}
