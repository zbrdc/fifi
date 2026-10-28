use super::*;
use crate::kernels::cuda::{CudaContext, testkit};

#[test]
fn row_bf16_native_refusals_precede_launch_and_preserve_buffers() {
    if !testkit::device_present("row BF16 native preflight") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let x = testkit::upload_f32(&ctx, &[1.0; 9]);
    let y = testkit::upload_filled(&ctx, 0xa5, 36);
    let flag = ctx.upload(&8u32.to_ne_bytes()).unwrap();
    let xr = ctx.raw_range(&x.range(0, 32).unwrap()).unwrap();
    let yr = ctx.raw_range(&y.range(0, 32).unwrap()).unwrap();
    let fr = ctx.raw_range(&flag.range(0, 4).unwrap()).unwrap();
    type InvalidCase = (&'static str, fn(&mut Bf16PublishArgsRepr));
    let cases: &[InvalidCase] = &[
        ("zero rows", |a| a.rows = 0),
        ("zero cols", |a| a.columns = 0),
        ("rows limit", |a| a.rows = 65_537),
        ("column limit", |a| a.columns = (1 << 22) + 1),
        ("product limit", |a| {
            a.rows = 65_536;
            a.columns = 65_537;
        }),
        ("short x", |a| a.x_bytes = 31),
        ("short y", |a| a.y_bytes = 31),
        ("short flag", |a| a.flag_bytes = 3),
        ("unaligned x", |a| a.x = a.x.wrapping_byte_add(1)),
        ("alias xy", |a| a.y = a.x.cast_mut()),
        ("alias flag", |a| a.flag = a.x.cast::<u32>().cast_mut()),
        ("null x", |a| a.x = std::ptr::null()),
        ("null y", |a| a.y = std::ptr::null_mut()),
        ("null flag", |a| a.flag = std::ptr::null_mut()),
        ("address overflow", |a| a.y = (usize::MAX - 3) as *mut f32),
    ];
    for &(name, corrupt) in cases {
        let mut args = Bf16PublishArgsRepr {
            x: xr.address().cast::<f32>().cast_const(),
            x_bytes: 32,
            y: yr.address().cast(),
            y_bytes: 32,
            flag: fr.address().cast(),
            flag_bytes: 4,
            rows: 2,
            columns: 4,
        };
        corrupt(&mut args);
        let (status, stage) = ctx
            .launch_on(&stream, |raw| {
                let mut stage = -77;
                // SAFETY: a live host descriptor carries deliberately invalid spans.
                // The private C entry must reject them before any device dereference.
                let status = unsafe { bf16_publish_raw(&args, raw.0.as_ptr(), &mut stage) };
                Ok((status, stage))
            })
            .unwrap();
        assert_eq!(status, invalid_argument().status, "{name}");
        assert_eq!(stage, STAGE_NONE, "{name}");
        ctx.ensure_usable().unwrap();
    }
    let (status, stage) = ctx
        .launch_on(&stream, |raw| {
            let mut stage = -77;
            // SAFETY: NULL descriptor is explicitly refused before native use.
            let status = unsafe { bf16_publish_raw(std::ptr::null(), raw.0.as_ptr(), &mut stage) };
            Ok((status, stage))
        })
        .unwrap();
    assert_eq!(status, invalid_argument().status);
    assert_eq!(stage, STAGE_NONE);
    stream.synchronize().unwrap();
    assert_eq!(
        testkit::read_bytes(&ctx, &x, 36),
        [1.0f32; 9]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>()
    );
    assert_eq!(testkit::read_bytes(&ctx, &y, 36), vec![0xa5; 36]);
    assert_eq!(testkit::read_bytes(&ctx, &flag, 4), 8u32.to_ne_bytes());
}
