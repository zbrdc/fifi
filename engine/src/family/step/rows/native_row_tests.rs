use super::*;

#[test]
#[ignore = "requires an admitted CUDA device"]
fn exact_native_row_uses_f32_and_preserves_arena_boundaries() {
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let (k, rows) = (32, 2);
    let ty = QuantType::Q80;
    let mut blocks = vec![0; ty.weight_bytes(rows, k).unwrap()];
    for (index, row) in blocks
        .chunks_exact_mut(ty.weight_bytes(1, k).unwrap())
        .enumerate()
    {
        row[..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // f16 scale 1
        row[2] = (index + 1) as u8;
    }
    let matrix =
        NativeWeight::from_buffer(ctx.upload(&blocks).unwrap(), ty, k, rows, None).unwrap();
    let stack =
        NativeWeight::from_buffer(ctx.upload(&blocks).unwrap(), ty, k, rows, Some(2)).unwrap();
    let mut values = vec![0.0f32; k + 4];
    values[4] = 0.125;
    values[5] = 2.0; // Zero weight, but this would set the activation scale of a q8_1 row.
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let input = ctx.upload(&bytes).unwrap();
    let output = ctx.upload(&[0xa5; 16]).unwrap();
    let call = NativeRow {
        weight: &matrix,
        input: input.range(16, k * 4).unwrap(),
        output: output.range(4, rows * 4).unwrap(),
    };
    let read = || {
        let mut buf = [0; 16];
        output.read_at(&stream, 0, &mut buf).unwrap();
        buf
    };
    assert!(matches!(
        NativeRow {
            weight: &stack,
            ..call
        }
        .exact(&ctx, &stream),
        Err(EngError::BadRequest(_))
    ));
    assert_eq!(read(), [0xa5; 16]);
    call.exact(&ctx, &stream).unwrap();
    let got = read();
    assert_eq!(&got[..4], &[0xa5; 4]);
    assert_eq!(&got[4..8], &0.125f32.to_ne_bytes());
    assert_eq!(&got[8..12], &0.25f32.to_ne_bytes());
    assert_eq!(&got[12..], &[0xa5; 4]);
}
