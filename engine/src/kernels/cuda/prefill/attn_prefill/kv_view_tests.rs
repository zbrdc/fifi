//! Device allocations back the range/owner proofs; no attention kernel is launched.

use super::storage_tests::quantized;
use super::{AttnKvLayout, AttnKvStorage, AttnKvView, AttnScalarLayout, AttnStrides};
use crate::EngError;
use crate::kernels::cuda::context::{Claim, DeviceBuffer};
use crate::kernels::cuda::{CudaContext, testkit};
use crate::tensor::Dtype;

#[test]
fn kv_view_admits_all_tagged_read_planes_without_backend_dispatch() {
    if !testkit::device_present("kv_view_admits_all_tagged_read_planes") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let payload = testkit::upload_filled(&ctx, 0, 2 * 1024 * 1024);
    let parameters = testkit::upload_filled(&ctx, 0, 1024);
    let shape = [512, 4, 128];
    let storage = [AttnKvStorage::F16, AttnKvStorage::Bf16, AttnKvStorage::F32]
        .into_iter()
        .chain(quantized(32));
    for storage in storage {
        let cell = storage.payload_cell_bytes(shape[2]).expect("payload cell");
        let affine = matches!(
            storage,
            AttnKvStorage::Q8Affine { .. } | AttnKvStorage::Q4Affine { .. }
        );
        let layout = AttnKvLayout {
            storage,
            payload_strides: AttnStrides {
                time_stride_bytes: shape[1] * cell,
                head_stride_bytes: cell,
            },
            scales: affine.then_some(AttnScalarLayout {
                dtype: Dtype::F32,
                strides: AttnStrides {
                    time_stride_bytes: 0,
                    head_stride_bytes: 0,
                },
                group_stride_bytes: 4,
            }),
            bias: None,
        };
        let extents = layout
            .extents(shape[0], shape[1], shape[2])
            .expect("extents");
        let view = AttnKvView {
            layout,
            payload: payload
                .range(16, extents.payload_bytes)
                .expect("payload range"),
            scales: extents
                .scale_bytes
                .map(|bytes| parameters.range(8, bytes).expect("scale range")),
            bias: None,
        };
        assert_eq!(
            view.check(&ctx, shape, &[]).expect("checked read planes"),
            extents
        );
    }
}

fn affine_view<'b>(payload: &'b DeviceBuffer, parameters: &'b DeviceBuffer) -> AttnKvView<'b> {
    AttnKvView {
        layout: AttnKvLayout {
            storage: AttnKvStorage::Q8Affine {
                group_size: 32,
                signed: true,
            },
            payload_strides: AttnStrides {
                time_stride_bytes: 128,
                head_stride_bytes: 64,
            },
            scales: Some(AttnScalarLayout {
                dtype: Dtype::F32,
                strides: AttnStrides {
                    time_stride_bytes: 32,
                    head_stride_bytes: 16,
                },
                group_stride_bytes: 4,
            }),
            bias: Some(AttnScalarLayout {
                dtype: Dtype::Bf16,
                strides: AttnStrides {
                    time_stride_bytes: 16,
                    head_stride_bytes: 8,
                },
                group_stride_bytes: 2,
            }),
        },
        payload: payload.range(16, 4096).expect("payload"),
        scales: Some(parameters.range(32, 4096).expect("scales")),
        bias: Some(parameters.range(64, 4096).expect("bias")),
    }
}

fn refusal(view: AttnKvView<'_>, ctx: &CudaContext) -> String {
    match view.check(ctx, [2, 2, 64], &[]) {
        Err(EngError::BadRequest(message)) => message,
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[test]
fn kv_view_rejects_missing_extra_short_and_misaligned_planes() {
    if !testkit::device_present("kv_view_rejects_bad_planes") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let payload = testkit::upload_filled(&ctx, 0, 8192);
    let parameters = testkit::upload_filled(&ctx, 0, 8192);
    let view = affine_view(&payload, &parameters);
    view.check(&ctx, [2, 2, 64], &[]).expect("baseline");
    for bad in [
        AttnKvView {
            payload: payload.range(16, 255).unwrap(),
            ..view
        },
        AttnKvView {
            scales: None,
            ..view
        },
        AttnKvView {
            scales: Some(parameters.range(32, 55).unwrap()),
            ..view
        },
        AttnKvView {
            scales: Some(parameters.range(33, 4096).unwrap()),
            ..view
        },
        AttnKvView { bias: None, ..view },
        AttnKvView {
            bias: Some(parameters.range(64, 27).unwrap()),
            ..view
        },
        AttnKvView {
            bias: Some(parameters.range(65, 4096).unwrap()),
            ..view
        },
    ] {
        assert!(!refusal(bad, &ctx).is_empty());
    }
    let mut optional = view;
    optional.layout.bias = None;
    assert!(refusal(optional, &ctx).contains("presence"));
    optional.bias = None;
    optional
        .check(&ctx, [2, 2, 64], &[])
        .expect("bias absent means zero");
    let mut float = optional;
    float.layout.storage = AttnKvStorage::F16;
    float.layout.scales = None;
    assert!(refusal(float, &ctx).contains("presence"));
    float.scales = None;
    float
        .check(&ctx, [2, 2, 64], &[])
        .expect("float without parameter planes");
    float.payload = payload.range(17, 4096).unwrap();
    assert!(!refusal(float, &ctx).is_empty());
    float.layout.storage = AttnKvStorage::F32;
    float.payload = payload.range(18, 4096).unwrap();
    assert!(!refusal(float, &ctx).is_empty());
    let byte_aligned = AttnKvView {
        payload: payload.range(17, 4096).unwrap(),
        ..view
    };
    byte_aligned
        .check(&ctx, [2, 2, 64], &[])
        .expect("Q8 codes need only byte alignment");
}

#[test]
fn kv_view_checks_active_strides_but_not_empty_or_unused_axes() {
    if !testkit::device_present("kv_view_checks_active_strides") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let payload = testkit::upload_filled(&ctx, 0, 8192);
    let parameters = testkit::upload_filled(&ctx, 0, 8192);
    let view = affine_view(&payload, &parameters);
    for axis in 0..8 {
        let mut bad = view;
        match axis {
            0 => bad.layout.storage = AttnKvStorage::F16,
            1 => bad.layout.payload_strides.time_stride_bytes += 1,
            2 => {
                bad.layout
                    .scales
                    .as_mut()
                    .unwrap()
                    .strides
                    .time_stride_bytes += 1
            }
            3 => {
                bad.layout
                    .scales
                    .as_mut()
                    .unwrap()
                    .strides
                    .head_stride_bytes += 1
            }
            4 => bad.layout.scales.as_mut().unwrap().group_stride_bytes += 1,
            5 => bad.layout.bias.as_mut().unwrap().strides.time_stride_bytes += 1,
            6 => bad.layout.bias.as_mut().unwrap().strides.head_stride_bytes += 1,
            _ => bad.layout.bias.as_mut().unwrap().group_stride_bytes += 1,
        }
        if axis == 0 {
            bad.layout.scales = None;
            bad.scales = None;
            bad.layout.bias = None;
            bad.bias = None;
            bad.layout.payload_strides.head_stride_bytes = 129;
        }
        if axis == 1 {
            bad.layout.storage = AttnKvStorage::Bf16;
            bad.layout.scales = None;
            bad.scales = None;
            bad.layout.bias = None;
            bad.bias = None;
        }
        assert!(refusal(bad, &ctx).contains("stride"));
    }
    let mut singleton = view;
    singleton.layout.payload_strides = AttnStrides {
        time_stride_bytes: usize::MAX,
        head_stride_bytes: usize::MAX,
    };
    for layout in [&mut singleton.layout.scales, &mut singleton.layout.bias]
        .into_iter()
        .flatten()
    {
        layout.strides = singleton.layout.payload_strides;
        layout.group_stride_bytes = usize::MAX;
    }
    singleton
        .check(&ctx, [1, 1, 32], &[])
        .expect("unused strides do not address memory");
    singleton.payload = payload.range(17, 4096).unwrap();
    singleton.scales = Some(parameters.range(33, 4096).unwrap());
    singleton.bias = Some(parameters.range(65, 4096).unwrap());
    let write = Claim::of("out", &singleton.payload, 256);
    singleton
        .check(&ctx, [0, 2, 64], &[write])
        .expect("empty rows have no read claims");
    let write = Claim::of("out", &singleton.payload, 256);
    singleton
        .check(&ctx, [2, 0, 64], &[write])
        .expect("empty heads have no read claims");
    singleton.layout.scales.as_mut().unwrap().dtype = Dtype::I16;
    assert!(matches!(
        singleton.check(&ctx, [0, 2, 64], &[]),
        Err(EngError::BadRequest(_))
    ));
}

#[test]
fn kv_view_checks_every_plane_owner_and_exact_read_claims() {
    if !testkit::device_present("kv_view_checks_owners_and_aliases") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let other = CudaContext::new().expect("other context");
    let payload = testkit::upload_filled(&ctx, 0, 8192);
    let parameters = testkit::upload_filled(&ctx, 0, 8192);
    let foreign = testkit::upload_filled(&other, 0, 8192);
    let view = affine_view(&payload, &parameters);
    view.check(&ctx, [2, 2, 64], &[])
        .expect("read-only parameter overlap is legal");
    for bad in [
        AttnKvView {
            payload: foreign.range(16, 4096).unwrap(),
            ..view
        },
        AttnKvView {
            scales: Some(foreign.range(32, 4096).unwrap()),
            ..view
        },
        AttnKvView {
            bias: Some(foreign.range(64, 4096).unwrap()),
            ..view
        },
    ] {
        assert!(refusal(bad, &ctx).contains("another adapter"));
    }
    for range in [
        payload.range(32, 16).unwrap(),
        parameters.range(40, 4).unwrap(),
        parameters.range(88, 2).unwrap(),
    ] {
        let writes = [Claim::of("out", &range, range.bytes())];
        let error = view
            .check(&ctx, [2, 2, 64], &writes)
            .expect_err("writable overlap");
        assert!(matches!(error, EngError::BadRequest(_)));
        assert!(format!("{error:?}").contains("overlap"));
    }
    for range in [
        payload.range(272, 16).unwrap(),
        parameters.range(92, 4).unwrap(),
    ] {
        view.check(&ctx, [2, 2, 64], &[Claim::of("out", &range, range.bytes())])
            .expect("adjacent writes use spans, not range capacity");
    }
}
