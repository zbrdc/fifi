use super::*;
use crate::kernels::cuda::{CudaContext, DeviceStream, testkit};

fn expect_native_refusal(
    ctx: &CudaContext,
    stream: &DeviceStream,
    args: Option<&RowPow2ArgsRepr>,
    null_stream: bool,
) {
    let (status, stage) = ctx
        .launch_on(stream, |raw| {
            let mut stage = -77;
            let pointer = args.map_or(std::ptr::null(), std::ptr::from_ref);
            let stream = if null_stream {
                std::ptr::null_mut()
            } else {
                raw.0.as_ptr()
            };
            // SAFETY: the host argument is live or NULL. Deliberately malformed spans
            // must be refused by the private C entry before any device dereference.
            let status = unsafe { row_pow2_launch_raw(pointer, stream, &mut stage) };
            // Expected negative status stays data, not a terminal adapter fault.
            Ok((status, stage))
        })
        .unwrap();
    assert_eq!(status, invalid_argument().status);
    assert_eq!(stage, STAGE_NONE, "refusal must happen before launch");
    ctx.ensure_usable().unwrap();
}

#[test]
fn row_pow2_native_refuses_malformed_arguments_before_launch() {
    if !testkit::device_present("row pow2 native refusals") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let x = testkit::upload_f32(&ctx, &[1.0; 8]);
    let y = testkit::upload_filled(&ctx, 0xa5, 36);
    let scales = testkit::upload_f32(&ctx, &[1.0; 2]);
    let flag = ctx.upload(&8u32.to_ne_bytes()).unwrap();
    let xr = ctx.raw_range(&x.range(0, 32).unwrap()).unwrap();
    let yr = ctx.raw_range(&y.range(0, 32).unwrap()).unwrap();
    let sr = ctx.raw_range(&scales.range(0, 8).unwrap()).unwrap();
    let fr = ctx.raw_range(&flag.range(0, 4).unwrap()).unwrap();
    type InvalidCase = (&'static str, fn(&mut RowPow2ArgsRepr));
    let cases: &[InvalidCase] = &[
        ("mode", |a| a.mode = 2),
        ("reserved", |a| a.reserved = 1),
        ("rows", |a| a.rows = 65_537),
        ("columns", |a| a.columns = (1 << 22) + 1),
        ("product", |a| {
            a.rows = 65_536;
            a.columns = 65_537;
        }),
        ("short x", |a| a.x_bytes = 31),
        ("short y", |a| a.y_bytes = 31),
        ("short scales", |a| a.scales_bytes = 7),
        ("short flag", |a| a.flag_bytes = 3),
        ("unaligned x", |a| a.x = a.x.wrapping_byte_add(1)),
        ("alias xy", |a| a.y = a.x.cast_mut()),
        ("alias scales", |a| a.scales = a.x.cast_mut()),
        ("alias flag", |a| a.flag = a.scales.cast()),
        ("null x", |a| a.x = std::ptr::null()),
        ("null y", |a| a.y = std::ptr::null_mut()),
        ("null scales", |a| a.scales = std::ptr::null_mut()),
        ("null flag", |a| a.flag = std::ptr::null_mut()),
        ("address end overflow", |a| {
            a.x = (usize::MAX - 3) as *const f32
        }),
    ];
    for &(name, corrupt) in cases {
        let mut args = RowPow2ArgsRepr {
            x: xr.address().cast::<f32>().cast_const(),
            x_bytes: 32,
            y: yr.address().cast(),
            y_bytes: 32,
            scales: sr.address().cast(),
            scales_bytes: 8,
            flag: fr.address().cast(),
            flag_bytes: 4,
            rows: 2,
            columns: 4,
            mode: 0,
            reserved: 0,
        };
        corrupt(&mut args);
        expect_native_refusal(&ctx, &stream, Some(&args), false);
        assert_eq!(testkit::read_bytes(&ctx, &y, 36), vec![0xa5; 36], "{name}");
    }
    expect_native_refusal(&ctx, &stream, None, false);
    let args = RowPow2ArgsRepr {
        x: xr.address().cast::<f32>().cast_const(),
        x_bytes: 32,
        y: yr.address().cast(),
        y_bytes: 32,
        scales: sr.address().cast(),
        scales_bytes: 8,
        flag: fr.address().cast(),
        flag_bytes: 4,
        rows: 2,
        columns: 4,
        mode: 0,
        reserved: 0,
    };
    expect_native_refusal(&ctx, &stream, Some(&args), true);
    stream.synchronize().unwrap();
    assert_eq!(
        testkit::read_bytes(&ctx, &x, 32),
        [1.0f32; 8]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        testkit::read_bytes(&ctx, &scales, 8),
        [1.0f32; 2]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>()
    );
    assert_eq!(testkit::read_bytes(&ctx, &flag, 4), 8u32.to_ne_bytes());
}
