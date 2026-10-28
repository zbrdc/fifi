use inference_engine::{
    EngError,
    family::nemotron35::forward::mamba::{MambaConfig, mamba_block_forward},
    tensor::{Dtype, Tensor},
};

fn tensor(shape: &[u32], values: &[f32]) -> Tensor {
    Tensor::try_from_host(
        Dtype::F32,
        shape.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap()
}

fn values(t: &Tensor) -> Vec<f32> {
    t.host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn close(actual: &Tensor, expected: &[f32]) {
    let actual = values(actual);
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && (a - e).abs() < 4e-6, "{i}: {a} != {e}");
    }
}

struct Fixture {
    config: MambaConfig,
    norm: Tensor,
    mixer_norm: Tensor,
    hidden: Tensor,
    input: Tensor,
    output: Tensor,
    conv: Tensor,
    bias: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    d: Tensor,
}

impl Fixture {
    fn new() -> Self {
        // hidden=[2,0,0,0], mean(square)=1, eps=3: RMSNorm=[1,0,0,0].
        // in_proj rows: gate(4), x(4), B(3), C(3), dt(2). No x_proj exists.
        let projected = [
            1., -1., 2., 0.5, 1., 2., -1., 0.5, 1., -1., 0.5, 1., 0.5, -1., 0., 0.,
        ];
        let input: Vec<_> = projected.iter().flat_map(|v| [*v, 0., 0., 0.]).collect();
        Self {
            config: MambaConfig {
                num_heads: 2,
                head_dim: 2,
                state_size: 3,
                n_groups: 1,
                conv_kernel: 2,
                eps: 3.,
                time_step_min: 1.,
            },
            norm: tensor(&[4], &[1.; 4]),
            mixer_norm: tensor(&[4], &[1., 2., 0.5, -1.]),
            hidden: tensor(&[1, 4], &[2., 0., 0., 0.]),
            input: tensor(&[16, 4], &input),
            output: tensor(
                &[4, 4],
                &[
                    1., 0., 0., 0., 0., 2., 0., 0., 0., 0., 3., 0., 0., 0., 0., 4.,
                ],
            ),
            conv: tensor(&[10, 1, 2], &[0.5, 1.].repeat(10)),
            bias: tensor(&[10], &[0.25; 10]),
            dt_bias: tensor(&[2], &[0., 3_f32.ln()]),
            a_log: tensor(&[2], &[0., 2_f32.ln()]),
            d: tensor(&[2], &[0.5, -0.25]),
        }
    }
    fn run(&self, conv: &Tensor, ssm: &Tensor) -> Result<(Tensor, Tensor, Tensor), EngError> {
        mamba_block_forward(
            &self.norm,
            &self.mixer_norm,
            &self.hidden,
            &self.input,
            &self.output,
            &self.conv,
            &self.bias,
            &self.dt_bias,
            &self.a_log,
            &self.d,
            conv,
            ssm,
            self.config,
        )
    }
}

#[test]
fn mamba_block_one_step_matches_literal_recurrence_gate_and_residual() {
    let f = Fixture::new();
    let conv = tensor(&[10, 2], &[0.; 20]);
    let ssm = tensor(&[2, 2, 3], &[0.; 12]);
    let (output, next_conv, next_ssm) = f.run(&conv, &ssm).unwrap();
    // Roll [0,0] to [0,xBC]. Conv=[0.5,1] and bias=.25 precede SiLU.
    // Activated x=[.971624826,2.035463704,-.240615976,.509384024],
    // B=[.971624826,-.240615976,.509384024], C=[.971624826,.509384024,-.240615976].
    // dt=max(softplus([0,ln(3)]),1)=[1,ln(4)]; NO .1 upper clamp.
    // A=-exp([0,ln(2)])=[-1,-2]; decay=[exp(-1),1/16].
    // s'[h,d,n]=decay[h]*s[h,d,n]+dt[h]*B[n]*x[h,d].
    // s'[0,0,0]=1*.971624826^2=.944054803; y=sum(s'*C)+D*x.
    // y=[1.164903289,2.440364119,-.172981935,.366202759].
    // BEFORE RMSNorm: y*SiLU(gate)=[.851612543,-.656314995,-.304723965,.113973162].
    // Divide by sqrt(mean(square)+3), multiply mixer_norm, project, add hidden.
    close(
        &next_conv,
        &[
            0., 1., 0., 2., 0., -1., 0., 0.5, 0., 1., 0., -1., 0., 0.5, 0., 1., 0., 0.5, 0., -1.,
        ],
    );
    close(
        &next_ssm,
        &[
            0.9440548,
            -0.23378846,
            0.49493016,
            1.9777071,
            -0.48976508,
            1.0368327,
            -0.32409963,
            0.08026096,
            -0.16991246,
            0.6861189,
            -0.16991246,
            0.3597047,
        ],
    );
    close(&output, &[2.467703, -1.4417847, -0.2510302, -0.25037485]);
    assert_eq!(output.shape(), &[1, 4]);
    assert_eq!(next_conv.shape(), conv.shape());
    assert_eq!(next_ssm.shape(), ssm.shape());
    close(&conv, &[0.; 20]);
    close(&ssm, &[0.; 12]);
}

#[test]
fn mamba_block_three_steps_consume_returned_states() {
    let f = Fixture::new();
    let mut conv = tensor(&[10, 2], &[0.; 20]);
    let mut ssm = tensor(&[2, 2, 3], &[0.; 12]);
    // On steps 2 and 3 conv=SiLU(1.5*xBC+.25). For example x0=1.490917403.
    // Step 2 s00=exp(-1)*.944054803+1.490917403^2=2.570133057.
    // Step 3 s00=exp(-1)*2.570133057+1.490917403^2=3.168333817.
    let states = [
        [
            0.9440548,
            -0.23378846,
            0.49493016,
            1.9777071,
            -0.48976508,
            1.0368327,
            -0.32409963,
            0.08026096,
            -0.16991246,
            0.6861189,
            -0.16991246,
            0.3597047,
        ],
        [
            2.570133,
            -0.50104034,
            1.2720226,
            5.3921723,
            -1.0511235,
            2.6686833,
            -0.59561604,
            0.11244406,
            -0.2927423,
            1.5538712,
            -0.2927423,
            0.7633819,
        ],
        [
            3.1683338,
            -0.59935683,
            1.5578989,
            6.6482844,
            -1.2576357,
            3.2690077,
            -0.61258584,
            0.1144555,
            -0.30041915,
            1.6081057,
            -0.30041915,
            0.78861177,
        ],
    ];
    let outputs = [
        [2.467703, -1.4417847, -0.2510302, -0.25037485],
        [3.0996378, -3.3950477, -0.67470264, -0.82896304],
        [3.1928911, -3.6834168, -0.6343875, -0.784579],
    ];
    for t in 0..3 {
        let (output, next_conv, next_ssm) = f.run(&conv, &ssm).unwrap();
        close(&output, &outputs[t]);
        close(&next_ssm, &states[t]);
        assert_ne!(values(&ssm), values(&next_ssm));
        conv = next_conv;
        ssm = next_ssm;
    }
    close(
        &conv,
        &[
            1., 1., 2., 2., -1., -1., 0.5, 0.5, 1., 1., -1., -1., 0.5, 0.5, 1., 1., 0.5, 0.5, -1.,
            -1.,
        ],
    );
}

#[test]
fn mamba_block_repeats_contiguous_bc_groups_and_normalizes_each_gate_group() {
    let mut f = Fixture::new();
    f.config.num_heads = 4;
    f.config.head_dim = 1;
    f.config.n_groups = 2;
    f.config.time_step_min = 0.;
    let mut input = vec![0.; 24 * 4];
    for (i, gate) in [1., -1., 2., 0.5].into_iter().enumerate() {
        input[i * 4] = gate;
    }
    f.input = tensor(&[24, 4], &input);
    f.conv = tensor(&[16, 1, 2], &[0.; 32]);
    f.bias = tensor(
        &[16],
        &[
            1., 2., -1., 0.5, 1., 0., 0., 2., 0., 0., 1., 0., 0., 3., 0., 0.,
        ],
    );
    f.dt_bias = tensor(&[4], &[0.; 4]);
    f.a_log = tensor(&[4], &[0.; 4]);
    f.d = tensor(&[4], &[0.; 4]);
    let (output, _, ssm) = f
        .run(&tensor(&[16, 2], &[0.; 32]), &tensor(&[4, 1, 3], &[0.; 12]))
        .unwrap();
    // B0=SiLU(1), B1=SiLU(2); C0=SiLU(1), C1=SiLU(3).
    // Heads 0,1 share group 0; heads 2,3 share group 1. dt=ln(2).
    // s00=ln(2)*SiLU(1)^2=.370450185; s20=ln(2)*SiLU(2)*SiLU(-1)=-.328389315.
    // Gate first, then RMSNorm on [head0,head1] and [head2,head3] separately.
    close(
        &ssm,
        &[
            0.3704502,
            0.,
            0.,
            0.8926547,
            0.,
            0.,
            -0.32838932,
            0.,
            0.,
            0.38002512,
            0.,
            0.,
        ],
    );
    close(&output, &[2.1136463, -0.40297118, -1.1790134, -0.6428148]);
}

#[test]
fn mamba_block_softplus_handles_extreme_logits_without_an_upper_clamp() {
    let mut f = Fixture::new();
    f.dt_bias = tensor(&[2], &[1000., -1000.]);
    f.config.time_step_min = 0.001;
    let (output, _, ssm) = f
        .run(&tensor(&[10, 2], &[0.; 20]), &tensor(&[2, 2, 3], &[0.; 12]))
        .unwrap();
    // softplus(1000)=1000; softplus(-1000) rounds to 0, then clamps to .001.
    // s00=1000*.971624826^2=944.054803; s100=.001*.971624826*(-.240615976).
    assert!((values(&ssm)[0] - 944.0548).abs() < 0.001);
    assert!((values(&ssm)[6] + 0.00023378846).abs() < 1e-9);
    assert!(values(&output).iter().all(|x| x.is_finite()));
}

#[test]
fn mamba_block_rejects_tensor_shape_and_dtype_mismatches_as_checkpoint() {
    let mutations: [fn(&mut Fixture); 10] = [
        |f| f.norm = tensor(&[1, 4], &[1.; 4]),
        |f| f.mixer_norm = tensor(&[1, 4], &[1.; 4]),
        |f| f.hidden = tensor(&[4], &[1.; 4]),
        |f| f.input = tensor(&[4, 16], &[0.; 64]),
        |f| f.output = tensor(&[16], &[0.; 16]),
        |f| f.conv = tensor(&[10, 2], &[0.; 20]),
        |f| f.bias = tensor(&[1, 10], &[0.; 10]),
        |f| f.dt_bias = tensor(&[1, 2], &[0.; 2]),
        |f| f.a_log = tensor(&[1, 2], &[0.; 2]),
        |f| f.d = Tensor::try_from_host(Dtype::F16, vec![2], vec![0; 4]).unwrap(),
    ];
    let conv = tensor(&[10, 2], &[0.; 20]);
    let ssm = tensor(&[2, 2, 3], &[0.; 12]);
    for mutate in mutations {
        let mut f = Fixture::new();
        mutate(&mut f);
        assert!(matches!(f.run(&conv, &ssm), Err(EngError::Checkpoint(_))));
    }
    let f = Fixture::new();
    assert!(matches!(
        f.run(&tensor(&[2, 10], &[0.; 20]), &ssm),
        Err(EngError::Checkpoint(_))
    ));
    assert!(matches!(
        f.run(&conv, &tensor(&[2, 3, 2], &[0.; 12])),
        Err(EngError::Checkpoint(_))
    ));
    assert!(matches!(
        f.run(
            &conv,
            &Tensor::try_from_host(Dtype::F16, vec![2, 2, 3], vec![0; 24]).unwrap()
        ),
        Err(EngError::Checkpoint(_))
    ));
}

#[test]
fn mamba_block_rejects_invalid_or_overflowing_geometry() {
    let mutations: [fn(&mut MambaConfig); 8] = [
        |c| c.num_heads = 0,
        |c| c.head_dim = 0,
        |c| c.state_size = 0,
        |c| c.conv_kernel = 0,
        |c| c.n_groups = 0,
        |c| c.n_groups = 3,
        |c| c.head_dim = u32::MAX,
        |c| c.state_size = u32::MAX,
    ];
    for mutate in mutations {
        let mut f = Fixture::new();
        mutate(&mut f.config);
        assert!(matches!(
            f.run(&tensor(&[10, 2], &[0.; 20]), &tensor(&[2, 2, 3], &[0.; 12])),
            Err(EngError::Checkpoint(_))
        ));
    }
}

#[test]
fn mamba_block_rejects_nonfinite_data_and_invalid_numerics_without_mutating_state() {
    let conv = tensor(&[10, 2], &[0.; 20]);
    let ssm = tensor(&[2, 2, 3], &[0.; 12]);
    let mutations: [fn(&mut Fixture); 6] = [
        |f| f.config.eps = 0.,
        |f| f.config.eps = f32::NAN,
        |f| f.config.time_step_min = -1.,
        |f| f.config.time_step_min = f32::INFINITY,
        |f| f.a_log = tensor(&[2], &[f32::NAN, 0.]),
        |f| f.hidden = tensor(&[1, 4], &[f32::MAX; 4]),
    ];
    for mutate in mutations {
        let mut f = Fixture::new();
        mutate(&mut f);
        assert!(matches!(f.run(&conv, &ssm), Err(EngError::BadRequest(_))));
    }
    let mut f = Fixture::new();
    f.output = tensor(
        &[4, 4],
        &[f32::MAX, -f32::MAX, -f32::MAX, -f32::MAX].repeat(4),
    );
    assert!(matches!(f.run(&conv, &ssm), Err(EngError::BadRequest(_))));
    close(&conv, &[0.; 20]);
    close(&ssm, &[0.; 12]);
}

#[test]
fn mamba_block_rejects_overflowing_dt_projection_before_clamp() {
    let mut f = Fixture::new();
    f.hidden = tensor(&[1, 4], &[2., 2., 0., 0.]);
    f.norm = tensor(&[4], &[f32::MAX, f32::MAX, 1., 1.]);
    let mut input = vec![0.; 64];
    // Normalized first two values are finite, but 2*x0 + (-2*x1) = inf-inf.
    // A lower clamp must not mask the resulting NaN dt with its finite minimum.
    input[14 * 4] = 2.;
    input[14 * 4 + 1] = -2.;
    f.input = tensor(&[16, 4], &input);
    assert!(matches!(
        f.run(&tensor(&[10, 2], &[0.; 20]), &tensor(&[2, 2, 3], &[0.; 12])),
        Err(EngError::BadRequest(_))
    ));
}
