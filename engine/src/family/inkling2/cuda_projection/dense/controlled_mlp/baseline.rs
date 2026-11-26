// The original resident path; the decode-glue switch leaves this path unchanged.
use super::*;

impl DenseStep {
    pub(super) fn launch_controlled_baseline(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        (rows, work, eps): (&RoutedRows<'_>, &Workspace<'_>, f32),
        enqueue: impl FnOnce(&DeviceStream, [DeviceRange<'_>; 4]) -> Result<(), EngError>,
    ) -> Result<(), EngError> {
        let row = rows.input.bytes();
        // The gate halves: the normalized input, then the normalized message.
        for (src, dst) in [
            (rows.input, work.gate.slice(0, row)?),
            (rows.message, work.gate.slice(row, row)?),
        ] {
            context.inkling_stream_norm(
                stream,
                NormCall {
                    src,
                    dst,
                    health: rows.gate_health,
                },
                (work.width, MIX_EPS),
            )?;
        }
        for (src, at) in [(rows.input, 0), (rows.message, row), (rows.gamma, 2 * row)] {
            context.copy_device(stream, src, work.norm.slice(at, row)?)?;
        }
        context.copy_device(stream, rows.bias, work.bias)?;
        context.dense_gemv_group(stream, &work.calls)?;
        context.inkling_controller_norm_roles(
            stream,
            work.norm,
            rows.health,
            ([work.width, 1], eps),
        )?;
        enqueue(stream, [work.x, work.bias, rows.status, rows.out])
    }
}
