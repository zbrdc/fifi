//! Page boundaries, inconsistent device words, and rejection before either native enqueue.
use super::{
    common_tests::dims,
    native_device_rig::device_present,
    paging_decode_tests::data,
    paging_pipeline_tests::{floats, inputs, read, upload},
    paging_plan::PagingPlan,
    paging_store::{PageId, PageToken, PagedKv},
};
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext,
        resident::{
            kv8_append::{Kv8CodeScale, Kv8Shape},
            step_params::{STEP_PARAMS_BYTES, StepParams},
        },
    },
};

fn check_case(position: usize, params: StepParams, valid: bool) -> Result<(), EngError> {
    let dimensions = dims();
    let plan = PagingPlan::new(&dimensions, 321)?;
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let (k, v) = inputs(1);
    let k = upload(&ctx, &stream, &floats(&k))?;
    let v = upload(&ctx, &stream, &floats(&v))?;
    let words = upload(&ctx, &stream, &params.to_bytes())?;
    let mut store = PagedKv::allocate(&ctx, &dimensions, 321, 8 << 20)?;
    let page = PageId {
        full_layer: 9,
        group: position / plan.group_cells,
    };
    store.load(&ctx, &stream, 0, page)?;
    store.append_decode(
        &ctx,
        &stream,
        0,
        PageToken {
            page,
            position,
            params: words.range(0, STEP_PARAMS_BYTES)?,
            k: k.range(0, k.bytes())?,
            v: v.range(0, v.bytes())?,
        },
    )?;
    let mut expected_words = params;
    if !valid {
        expected_words.status |= 4;
    }
    assert_eq!(
        read(&ctx, &stream, words.range(0, STEP_PARAMS_BYTES)?)?,
        expected_words.to_bytes()
    );
    let actual = read(&ctx, &stream, store.resident(0)?)?;
    assert_eq!(actual.len(), plan.group.total + 4);
    let local = if valid {
        (position % plan.group_cells) as i32
    } else {
        -1
    };
    assert_eq!(&actual[plan.group.total..], local.to_le_bytes());

    let reference = upload(&ctx, &stream, &vec![0; plan.group.total])?;
    if valid {
        ctx.kv8_chunk(
            &stream,
            Kv8Shape {
                rows: 1,
                n_kv_head: 2,
                head_dim: 256,
                cells: plan.group_cells,
                code_scale: Kv8CodeScale::StoredHalf,
            },
            data(&k, &v, &reference, &plan.group)?,
            position % plan.group_cells,
        )?;
    }
    let expected = read(&ctx, &stream, reference.range(0, reference.bytes())?)?;
    assert_eq!(&actual[..plan.group.total], expected);
    Ok(())
}

#[test]
fn decode_routes_nine_global_positions_without_changing_codes_scales_or_words() {
    if !device_present("paged decode nine global positions") {
        return;
    }
    for position in [0, 63, 64, 65, 255, 256, 257, 319, 320] {
        let mut params = StepParams::new(7, position as i32).unwrap();
        params.pad = [11, 17, 23];
        check_case(position, params, true).expect("native paged decode position");
    }
    eprintln!("PAGED_DECODE_POSITIONS_PASS cases=9");
}

#[test]
fn inconsistent_gpu_words_preserve_global_state_and_cannot_write_any_kv_plane() {
    if !device_present("paged decode inconsistent GPU words") {
        return;
    }
    let mut cases = vec![
        StepParams::new(7, 255).unwrap(),
        StepParams::new(7, 257).unwrap(),
        StepParams::new(7, 321).unwrap(),
    ];
    let base = StepParams::new(7, 256).unwrap();
    let mut wrong_len = base;
    wrong_len.kv_len = 256;
    cases.push(wrong_len);
    let mut wrong_slot = base;
    wrong_slot.kv_slot = 257;
    cases.push(wrong_slot);
    let mut negative = base;
    negative.pos = -1;
    negative.kv_len = 0;
    negative.kv_slot = -1;
    cases.push(negative);
    for status in [1, 2, 4] {
        let mut prior_fault = base;
        prior_fault.status = status;
        cases.push(prior_fault);
    }
    assert_eq!(cases.len(), 9);
    for mut params in cases {
        params.pad = [11, 17, 23];
        check_case(256, params, false).expect("native paged decode guard");
    }
    eprintln!("PAGED_DECODE_GPU_GUARDS_PASS cases=9 mask4=true no_kv_writes=true");
}

fn check_preflight() -> Result<(), EngError> {
    let dimensions = dims();
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let (k, v) = inputs(1);
    let mut k_bytes = floats(&k);
    let original = StepParams::new(7, 255)?.to_bytes();
    k_bytes[..STEP_PARAMS_BYTES].copy_from_slice(&original);
    let k = upload(&ctx, &stream, &k_bytes)?;
    let v = upload(&ctx, &stream, &floats(&v))?;
    let words = upload(&ctx, &stream, &original)?;
    let mut store = PagedKv::allocate(&ctx, &dimensions, 321, 8 << 20)?;
    let page = PageId {
        full_layer: 9,
        group: 1,
    };
    for case in 0..5 {
        store.load(&ctx, &stream, 0, page)?;
        let queued = stream.enqueued();
        let call = PageToken {
            page,
            position: if case == 3 { 321 } else { 256 },
            params: if case == 4 {
                k.range(0, STEP_PARAMS_BYTES)?
            } else {
                words.range(0, if case == 2 { 4 } else { STEP_PARAMS_BYTES })?
            },
            k: k.range(0, if case == 0 { 4 } else { k.bytes() })?,
            v: v.range(0, if case == 1 { 4 } else { v.bytes() })?,
        };
        assert!(matches!(
            store.append_decode(&ctx, &stream, 0, call),
            Err(EngError::BadRequest(_))
        ));
        assert_eq!(
            stream.enqueued(),
            queued,
            "refused call must not queue control or append"
        );
        store.writeback(&ctx, &stream, 0)?;
        assert_eq!(
            stream.enqueued(),
            queued,
            "refused call must preserve a clean slot"
        );
        assert_eq!(
            read(&ctx, &stream, words.range(0, STEP_PARAMS_BYTES)?)?,
            original
        );
        assert_eq!(
            read(&ctx, &stream, k.range(0, STEP_PARAMS_BYTES)?)?,
            original
        );
    }
    Ok(())
}

#[test]
fn bad_decode_inputs_refuse_before_control_enqueue_and_preserve_clean_writeback() {
    if !device_present("paged decode preflight before either enqueue") {
        return;
    }
    check_preflight().expect("native paged decode preflight");
    eprintln!("PAGED_DECODE_PREFLIGHT_PASS cases=5 no_enqueue=true clean=true");
}
