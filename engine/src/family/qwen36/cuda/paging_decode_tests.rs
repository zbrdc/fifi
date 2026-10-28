//! GPU-authoritative decode routing; the original native quantizer is the byte oracle.
use super::{
    common_tests::dims,
    native_device_rig::device_present,
    paging_pipeline_tests::{floats, inputs, plane, read, upload},
    paging_plan::PagingPlan,
    paging_store::{PageId, PageToken, PagedKv},
    state_layout::Plane,
};
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceBuffer,
        resident::{
            kv8_append::{Kv8CodeScale, Kv8Data, Kv8Shape},
            step_params::{OFFSET_POS, OFFSET_STATUS, STEP_PARAMS_BYTES, StepParams},
        },
    },
};

const C: usize = 256;
const D: usize = 256;
const KV: usize = 2;

pub(super) fn data<'a>(
    k: &'a DeviceBuffer,
    v: &'a DeviceBuffer,
    out: &'a DeviceBuffer,
    pool: &super::state_layout::KvPool,
) -> Result<Kv8Data<'a>, EngError> {
    Ok(Kv8Data {
        k: k.range(0, k.bytes())?,
        v: v.range(0, v.bytes())?,
        k_codes: plane(out, pool, Plane::K)?,
        k_scales: plane(out, pool, Plane::KScales)?,
        v_codes: plane(out, pool, Plane::V)?,
        v_scales: plane(out, pool, Plane::VScales)?,
    })
}

fn check_boundary() -> Result<(), EngError> {
    let dimensions = dims();
    let plan = PagingPlan::new(&dimensions, 321)?;
    assert_eq!(plan.group_cells, C);
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let (k, v) = inputs(1);
    let k = upload(&ctx, &stream, &floats(&k))?;
    let v = upload(&ctx, &stream, &floats(&v))?;
    let shape = Kv8Shape {
        rows: 1,
        n_kv_head: KV,
        head_dim: D,
        cells: C,
        code_scale: Kv8CodeScale::StoredHalf,
    };
    let reference = upload(&ctx, &stream, &vec![0; plan.group.total])?;
    ctx.kv8_chunk(&stream, shape, data(&k, &v, &reference, &plan.group)?, 0)?;
    let expected = read(&ctx, &stream, reference.range(0, reference.bytes())?)?;
    for p in plan.group.planes() {
        let (offset, bytes) = plan.group.plane_extent(0, *p)?;
        assert!(expected[offset..offset + bytes].iter().any(|b| *b != 0));
    }

    // The legacy append must refuse global 256 in a 256-cell page, not wrap it.
    let words = StepParams::new(7, 256)?.to_bytes();
    let params = upload(&ctx, &stream, &words)?;
    let control = upload(&ctx, &stream, &vec![0; plan.group.total])?;
    ctx.kv8_append(
        &stream,
        shape,
        data(&k, &v, &control, &plan.group)?,
        params.range(OFFSET_POS, 4)?,
        params.range(OFFSET_STATUS, 4)?,
    )?;
    assert!(
        read(&ctx, &stream, control.range(0, control.bytes())?)?
            .iter()
            .all(|b| *b == 0)
    );
    let observed: [u8; STEP_PARAMS_BYTES] =
        read(&ctx, &stream, params.range(0, STEP_PARAMS_BYTES)?)?
            .try_into()
            .unwrap();
    let mut refused = StepParams::new(7, 256)?;
    refused.status = 4;
    assert_eq!(observed, refused.to_bytes());
    eprintln!("PAGED_DECODE_GLOBAL_CONTROL_PASS global=256 local=0 no_wrap=true");

    let params = upload(&ctx, &stream, &words)?;
    let mut store = PagedKv::allocate(&ctx, &dimensions, 321, 8 << 20)?;
    let page = PageId {
        full_layer: 9,
        group: 1,
    };
    store.load(&ctx, &stream, 0, page)?;
    eprintln!("PAGED_DECODE_NATIVE_READY global=256 slot=0 group=1");
    store.append_decode(
        &ctx,
        &stream,
        0,
        PageToken {
            page,
            position: 256,
            params: params.range(0, STEP_PARAMS_BYTES)?,
            k: k.range(0, k.bytes())?,
            v: v.range(0, v.bytes())?,
        },
    )?;
    assert_eq!(
        read(&ctx, &stream, params.range(0, STEP_PARAMS_BYTES)?)?,
        words
    );
    let actual = read(&ctx, &stream, store.resident(0)?)?;
    assert_eq!(actual.len(), expected.len() + 4);
    assert_eq!(&actual[..plan.group.total], expected);
    assert_eq!(&actual[plan.group.total..], 0_i32.to_le_bytes());
    eprintln!("PAGED_DECODE_BOUNDARY_PASS global_unchanged=true four_planes_equal=true");
    Ok(())
}

#[test]
fn decode_uses_global_gpu_position_without_wrapping_or_changing_kv8() {
    if !device_present("paged decode global-position boundary") {
        return;
    }
    check_boundary().expect("native paged decode boundary");
}
