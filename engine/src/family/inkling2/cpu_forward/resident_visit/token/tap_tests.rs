use super::*;

/// Kernel-only real-weight check: form the effective taps on CUDA and evaluate
/// the resulting static convolution independently against the original dynamic
/// equation. The role rows, coefficients and kernels use the actual checkpoint.
#[test]
#[ignore = "leased CUDA and real Inkling Mini GGUF"]
fn real_effective_tap_kernel_preserves_dynamic_convolution() {
    let path = std::env::var_os("FIFI_INKLING_MODEL_DIR").unwrap();
    let f = CpuForward::open(std::path::Path::new(&path), 8).unwrap();
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let prefix = "model.llm.layers.11.attn";
    let width = 1024;
    let inputs: Vec<_> = (1..=4)
        .map(|id| f.embed(id).unwrap().into_current())
        .collect();
    let mut max = 0f64;
    let mut diff = 0.;
    let mut norm = 0.;
    let mut exact = 0;
    let mut count = 0;
    for (plane, projection) in [("k", "wk_dv"), ("v", "wv_dv")] {
        let w = f
            .weights
            .vec(&format!("{prefix}.{plane}_sconv.static_w"), 4 * width)
            .unwrap();
        let coeff = f
            .weights
            .project(
                &format!("{prefix}.{plane}_sconv.weight_proj"),
                &inputs[0],
                32,
            )
            .unwrap();
        let raw: Vec<_> = inputs
            .iter()
            .map(|x| {
                f.weights
                    .project(&format!("{prefix}.{projection}.weight"), x, width)
                    .unwrap()
            })
            .collect();
        let upload = |row: &[f32]| {
            let mut buf = Vec::new();
            append(&mut buf, row);
            ctx.upload(&buf).unwrap()
        };
        let (sw, c) = (upload(&w), upload(&coeff));
        let out = ctx.allocate(4 * width * WORD).unwrap();
        ctx.effective_taps(
            &stream,
            [
                sw.range(0, sw.bytes()).unwrap(),
                c.range(0, c.bytes()).unwrap(),
                out.range(0, out.bytes()).unwrap(),
            ],
            width,
        )
        .unwrap();
        let mut bytes = vec![0; out.bytes()];
        out.read_at(&stream, 0, &mut bytes).unwrap();
        let effective: Vec<_> = bytes
            .as_chunks::<WORD>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect();
        for past in 0..4 {
            for channel in 0..width {
                let mut a = 0.;
                let mut b = 0.;
                for lag in 0..=past {
                    a += (w[lag * width + channel] + coeff[channel / 128 * 4 + lag])
                        * raw[lag][channel];
                    b += effective[channel * 4 + 3 - lag] * raw[lag][channel];
                }
                a = (a + raw[0][channel]) * 9f32.sqrt().recip();
                b = (b + raw[0][channel]) * 5f32.sqrt().recip();
                assert!(a.is_finite() && b.is_finite());
                let d = f64::from(a) - f64::from(b);
                max = max.max(d.abs());
                diff += d * d;
                norm += f64::from(a).powi(2);
                exact += usize::from(a.to_bits() == b.to_bits());
                count += 1;
            }
        }
    }
    let rel = (diff / norm).sqrt();
    eprintln!(
        "dynamic conv: exact={exact}/{count} max_abs={max} rel_l2={rel} limits=0.00001,0.000002"
    );
    assert!(max <= 1e-5 && rel <= 2e-6);
}
