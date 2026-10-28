//! Portable F32 decode step from transformers' NemotronHMamba2Mixer.torch_forward.
//! Reference: models/nemotron_h/modeling_nemotron_h.py, SHA-256
//! 859ff78fae71fa3f8857e4ea0620c43aebea624e310e8422c124c507131398d2.

use crate::{
    EngError,
    family::{nemotron35::projection::HostProjector, project::Projector},
    tensor::{Dtype, Tensor},
};

/// Mixer geometry and the torch reference's forward-time numerical parameters.
/// The checkpoint uses (128, 64, 128, 8, 4), epsilon 1e-5, minimum dt 0.001.
#[derive(Clone, Copy, Debug)]
pub struct MambaConfig {
    pub num_heads: u32,
    pub head_dim: u32,
    pub state_size: u32,
    pub n_groups: u32,
    pub conv_kernel: u32,
    pub eps: f32,
    pub time_step_min: f32,
}

/// Run one unpadded token `[1, hidden_size]`, including pre-norm and residual.
/// Projections use `[out, in]`, without biases. `a_log` is the stored log parameter.
/// `conv_weight` is `[conv_dim, 1, kernel]`; `conv_bias` and `D` are mandatory.
/// Borrow states `[conv_dim, kernel]` and `[heads, head_dim, state_size]` in F32,
/// returning `(output, next_conv_state, next_ssm_state)` without modifying inputs.
/// Start with zero states. Prefill callers loop over tokens and retain returned states;
/// the reference's full-sequence branch uses a separate chunked scan (lines 472-553).
#[allow(clippy::too_many_arguments)]
pub fn mamba_block_forward(
    norm_weight: &Tensor,
    mixer_norm_weight: &Tensor,
    hidden: &Tensor,
    in_proj_weight: &Tensor,
    out_proj_weight: &Tensor,
    conv_weight: &Tensor,
    conv_bias: &Tensor,
    dt_bias: &Tensor,
    a_log: &Tensor,
    d: &Tensor,
    conv_state: &Tensor,
    ssm_state: &Tensor,
    config: MambaConfig,
) -> Result<(Tensor, Tensor, Tensor), EngError> {
    let mut projector = HostProjector;
    mamba_block_forward_with_projector(
        norm_weight,
        mixer_norm_weight,
        hidden,
        in_proj_weight,
        out_proj_weight,
        conv_weight,
        conv_bias,
        dt_bias,
        a_log,
        d,
        conv_state,
        ssm_state,
        config,
        &mut projector,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn mamba_block_forward_with_projector<P: Projector<Weight = Tensor>>(
    norm_weight: &Tensor,
    mixer_norm_weight: &Tensor,
    hidden: &Tensor,
    in_proj_weight: &Tensor,
    out_proj_weight: &Tensor,
    conv_weight: &Tensor,
    conv_bias: &Tensor,
    dt_bias: &Tensor,
    a_log: &Tensor,
    d: &Tensor,
    conv_state: &Tensor,
    ssm_state: &Tensor,
    config: MambaConfig,
    projector: &mut P,
    in_proj_name: Option<&str>,
    out_proj_name: Option<&str>,
) -> Result<(Tensor, Tensor, Tensor), EngError> {
    let &[1, width] = hidden.shape() else {
        return Err(extent("hidden must have shape [1, hidden_size]"));
    };
    let c = config;
    if width == 0
        || c.num_heads == 0
        || c.head_dim == 0
        || c.state_size == 0
        || c.n_groups == 0
        || c.conv_kernel == 0
        || !c.num_heads.is_multiple_of(c.n_groups)
    {
        return Err(extent(
            "dimensions must be positive and heads divisible by groups",
        ));
    }
    if !c.eps.is_finite() || c.eps <= 0.0 || !c.time_step_min.is_finite() || c.time_step_min < 0.0 {
        return Err(invalid(
            "epsilon must be positive and minimum dt nonnegative and finite",
        ));
    }
    // Reference lines 128,144-162,179: intermediate=heads*dim; conv=x+B+C.
    // Real header agrees: in_proj=[18560,4096], rows gate8192+x8192+B1024+C1024+dt128;
    // out_proj=[4096,8192]. There is no separate x_proj or dt projection matrix.
    let intermediate = product(c.num_heads, c.head_dim)?;
    let grouped_state = product(c.n_groups, c.state_size)?;
    let conv_dim = sum(intermediate, product(2, grouped_state)?)?;
    let projection_dim = sum(sum(intermediate, conv_dim)?, c.num_heads)?;
    checked_len(&[projection_dim, width])?;
    checked_len(&[c.num_heads, c.head_dim, c.state_size])?;
    let hidden = read_f32(hidden, &[1, width])?;
    let norm_weight = read_f32(norm_weight, &[width])?;
    let mixer_norm_weight = read_f32(mixer_norm_weight, &[intermediate])?;
    // Name-hinted weights were validated at load (dense_weight); the cached
    // dummy shape [1] is just a marker. Skip their geometry check.
    if in_proj_name.is_none() {
        checked_bytes(in_proj_weight, &[projection_dim, width])?;
    }
    if out_proj_name.is_none() {
        checked_bytes(out_proj_weight, &[width, intermediate])?;
    }
    let conv_weight = read_f32(conv_weight, &[conv_dim, 1, c.conv_kernel])?;
    let conv_bias = read_f32(conv_bias, &[conv_dim])?;
    let dt_bias = read_f32(dt_bias, &[c.num_heads])?;
    let a_log = read_f32(a_log, &[c.num_heads])?;
    let d = read_f32(d, &[c.num_heads])?;
    let mut conv_state = read_f32(conv_state, &[conv_dim, c.conv_kernel])?;
    let mut ssm_state = read_f32(ssm_state, &[c.num_heads, c.head_dim, c.state_size])?;

    // Sub-op timing (decode-only diagnostic, controlled by NEM35_SUBOP_TIMING).
    let subop = std::env::var_os("NEM35_SUBOP_TIMING").is_some_and(|v| v == "1");
    let mut t = std::time::Instant::now();
    let lap = |name: &str, timer: &mut std::time::Instant| {
        if subop {
            eprintln!("mamba_subop {name} {:?}", timer.elapsed());
            *timer = std::time::Instant::now();
        }
    };

    // Block lines 930-934 and NemotronHRMSNorm lines 585-590: norm before mixer.
    let normalized = rms_norm(&hidden, &norm_weight, width as usize, c.eps)?;
    lap("rms_norm", &mut t);
    let projected = match in_proj_name.and_then(|name| projector.linear_by_name(name, &normalized))
    {
        Some(result) => result?,
        None => projector.linear(in_proj_weight, &normalized)?,
    };
    lap("in_proj", &mut t);
    let projected = check_projection_output(projected, projection_dim as usize, "input")?;
    if projected.iter().any(|v| !v.is_finite()) {
        return Err(invalid("input projection is not finite"));
    }
    // Mixer lines 387-390: d_mlp=0; split gate, xBC, dt in that order.
    let (gate, rest) = projected.split_at(intermediate as usize);
    let (xbc, dt) = rest.split_at(conv_dim as usize);
    let mut convolved = Vec::with_capacity(conv_dim as usize);
    // cache_utils.py lines 813-819: shift oldest out, insert raw xBC at newest end.
    // Mixer lines 395-416: depthwise convolution, add bias, then SiLU on x,B,C.
    for (channel, state) in conv_state
        .chunks_exact_mut(c.conv_kernel as usize)
        .enumerate()
    {
        state.rotate_left(1);
        state[c.conv_kernel as usize - 1] = xbc[channel];
        let weights =
            &conv_weight[channel * c.conv_kernel as usize..(channel + 1) * c.conv_kernel as usize];
        let value = state.iter().zip(weights).map(|(s, w)| s * w).sum::<f32>() + conv_bias[channel];
        convolved.push(silu(value));
    }
    lap("conv_silu", &mut t);
    let (x, bc) = convolved.split_at(intermediate as usize);
    let (b, scan_c) = bc.split_at(grouped_state as usize);
    let scan = scan_step(x, b, scan_c, dt, &dt_bias, &a_log, &d, &mut ssm_state, c);
    lap("ssm_scan", &mut t);
    // Mixer lines 174-176,555-556; Zamba2RMSNormGated lines 58-69
    // (models/zamba2/modeling_zamba2.py): SiLU gate BEFORE grouped RMSNorm.
    let gated: Vec<_> = scan.iter().zip(gate).map(|(y, g)| y * silu(*g)).collect();
    let normalized = rms_norm(
        &gated,
        &mixer_norm_weight,
        (intermediate / c.n_groups) as usize,
        c.eps,
    )?;
    lap("gate_norm", &mut t);
    let output = match out_proj_name.and_then(|name| projector.linear_by_name(name, &normalized)) {
        Some(result) => result?,
        None => projector.linear(out_proj_weight, &normalized)?,
    };
    lap("out_proj", &mut t);
    let mut output = check_projection_output(output, width as usize, "output")?;
    // Mixer lines 555-557 and block line 947: out_proj, then original residual.
    for (y, residual) in output.iter_mut().zip(hidden) {
        *y += residual;
    }
    lap("residual", &mut t);
    Ok((
        make_tensor(&[1, width], output)?,
        make_tensor(&[conv_dim, c.conv_kernel], conv_state)?,
        make_tensor(&[c.num_heads, c.head_dim, c.state_size], ssm_state)?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn scan_step(
    x: &[f32],
    b: &[f32],
    scan_c: &[f32],
    dt: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
    d: &[f32],
    state: &mut [f32],
    c: MambaConfig,
) -> Vec<f32> {
    let mut output = vec![0.0; x.len()];
    for h in 0..c.num_heads as usize {
        // Lines 424-430: per-head bias, softplus, LOWER clamp only, A=-exp(A_log).
        // Deliberately follows torch_forward, not fused kernels (lines 231,275-276,
        // 307,366: dt_limit). time_step_max is initialization-only (lines 988-995),
        // NOT a forward upper clamp (line 427 explicitly comments it out).
        let step = softplus(dt[h] + dt_bias[h]).max(c.time_step_min);
        let a = -a_log[h].exp();
        let decay = (step * a).exp();
        // Lines 435-455: repeat each B/C group over consecutive heads.
        let group = h / (c.num_heads / c.n_groups) as usize;
        for dim in 0..c.head_dim as usize {
            let i = h * c.head_dim as usize + dim;
            let mut y = 0.0;
            for n in 0..c.state_size as usize {
                let s = &mut state[i * c.state_size as usize + n];
                let group_offset = group * c.state_size as usize + n;
                // Lines 439-449: state'=state*exp(dt*A)+(dt*B)*x, all F32.
                *s = *s * decay + (step * b[group_offset]) * x[i];
                // Lines 458-468: dot(state',C), then the per-head D*x skip.
                y += *s * scan_c[group_offset];
            }
            output[i] = y + d[h] * x[i];
        }
    }
    output
}

fn silu(x: f32) -> f32 {
    x * if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        x.exp() / (1.0 + x.exp())
    }
}

fn softplus(x: f32) -> f32 {
    // torch.nn.functional.softplus uses threshold 20, avoiding positive overflow.
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

fn rms_norm(
    input: &[f32],
    weight: &[f32],
    group_size: usize,
    eps: f32,
) -> Result<Vec<f32>, EngError> {
    let mut output = Vec::with_capacity(input.len());
    for (group, weights) in input
        .chunks_exact(group_size)
        .zip(weight.chunks_exact(group_size))
    {
        let variance = group.iter().map(|v| v * v).sum::<f32>() / group_size as f32 + eps;
        if !variance.is_finite() {
            return Err(invalid("RMS variance is not finite"));
        }
        let inverse_rms = variance.sqrt().recip();
        output.extend(
            group
                .iter()
                .zip(weights)
                .map(|(v, w)| w * (v * inverse_rms)),
        );
    }
    Ok(output)
}

fn check_projection_output(
    values: Vec<f32>,
    expected: usize,
    projection: &str,
) -> Result<Vec<f32>, EngError> {
    if values.len() != expected {
        return Err(EngError::Backend {
            operation: "nemotron35_mamba",
            message: format!(
                "{projection} projection returned {} values, expected {expected}",
                values.len()
            ),
        });
    }
    Ok(values)
}

fn extent(message: &str) -> EngError {
    EngError::Checkpoint(format!("NemotronH mamba: {message}"))
}

fn invalid(message: &str) -> EngError {
    EngError::BadRequest(format!("NemotronH mamba: {message}"))
}

fn product(a: u32, b: u32) -> Result<u32, EngError> {
    a.checked_mul(b)
        .ok_or_else(|| extent("dimension product overflows u32"))
}

fn sum(a: u32, b: u32) -> Result<u32, EngError> {
    a.checked_add(b)
        .ok_or_else(|| extent("dimension sum overflows u32"))
}

fn checked_len(shape: &[u32]) -> Result<usize, EngError> {
    shape
        .iter()
        .try_fold(4usize, |bytes, &dim| bytes.checked_mul(dim as usize))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| extent("tensor byte length exceeds host limits"))
}

/// Validates dtype, shape, extent and finiteness without allocating.
fn checked_bytes<'a>(tensor: &'a Tensor, shape: &[u32]) -> Result<&'a [u8], EngError> {
    let expected = checked_len(shape)?;
    if tensor.dtype() != Dtype::F32 || tensor.shape() != shape {
        return Err(extent(
            "tensor dtype or shape does not match block geometry",
        ));
    }
    let bytes = tensor
        .host_bytes()
        .ok_or_else(|| extent("tensor is not host-backed"))?;
    if bytes.len() != expected {
        return Err(extent("tensor byte length does not match block geometry"));
    }
    if bytes
        .as_chunks::<4>()
        .0
        .iter()
        .any(|b| !f32::from_le_bytes(*b).is_finite())
    {
        return Err(invalid("input tensors must contain finite values"));
    }
    Ok(bytes)
}

fn read_f32(tensor: &Tensor, shape: &[u32]) -> Result<Vec<f32>, EngError> {
    let bytes = checked_bytes(tensor, shape)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

fn make_tensor(shape: &[u32], values: Vec<f32>) -> Result<Tensor, EngError> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("mamba output or state is not finite"));
    }
    Tensor::try_from_host(
        Dtype::F32,
        shape.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

#[cfg(test)]
mod projector_tests {
    use super::{MambaConfig, mamba_block_forward_with_projector};
    use crate::{
        EngError,
        family::project::Projector,
        tensor::{Dtype, Tensor},
    };

    struct Inputs {
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
        conv_state: Tensor,
        ssm_state: Tensor,
    }

    impl Inputs {
        fn new() -> Self {
            Self {
                config: MambaConfig {
                    num_heads: 1,
                    head_dim: 2,
                    state_size: 1,
                    n_groups: 1,
                    conv_kernel: 1,
                    eps: 1.0,
                    time_step_min: 0.0,
                },
                norm: tensor(&[2], &[1.0, 1.0]),
                mixer_norm: tensor(&[2], &[1.0, 1.0]),
                hidden: tensor(&[1, 2], &[1.0, 0.0]),
                input: tensor(&[7, 2], &[0.0; 14]),
                output: tensor(&[2, 2], &[0.0; 4]),
                conv: tensor(&[4, 1, 1], &[0.0; 4]),
                bias: tensor(&[4], &[0.0; 4]),
                dt_bias: tensor(&[1], &[0.0]),
                a_log: tensor(&[1], &[0.0]),
                d: tensor(&[1], &[0.0]),
                conv_state: tensor(&[4, 1], &[0.0; 4]),
                ssm_state: tensor(&[1, 2, 1], &[0.0; 2]),
            }
        }

        fn run<P: Projector<Weight = Tensor>>(
            &self,
            projector: &mut P,
        ) -> Result<(Tensor, Tensor, Tensor), EngError> {
            mamba_block_forward_with_projector(
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
                &self.conv_state,
                &self.ssm_state,
                self.config,
                projector,
                None,
                None,
            )
        }
    }

    #[derive(Clone, Copy)]
    enum Behavior {
        ReturnZeros,
        FailAt(usize),
        ReturnShortAt(usize),
    }

    #[derive(Debug)]
    struct Call {
        shape: Vec<u32>,
        weight: Vec<u8>,
        input: Vec<f32>,
    }

    struct RecordingProjector {
        behavior: Behavior,
        calls: Vec<Call>,
    }

    impl RecordingProjector {
        fn new(behavior: Behavior) -> Self {
            Self {
                behavior,
                calls: Vec::new(),
            }
        }
    }

    impl Projector for RecordingProjector {
        type Weight = Tensor;

        fn linear(&mut self, weight: &Tensor, input: &[f32]) -> Result<Vec<f32>, EngError> {
            let call = self.calls.len() + 1;
            self.calls.push(Call {
                shape: weight.shape().to_vec(),
                weight: weight.host_bytes().unwrap().to_vec(),
                input: input.to_vec(),
            });
            if matches!(self.behavior, Behavior::FailAt(n) if n == call) {
                return Err(EngError::BadRequest(format!("projector failure {call}")));
            }
            let output_len = weight.shape()[0] as usize;
            let output_len = if matches!(self.behavior, Behavior::ReturnShortAt(n) if n == call) {
                output_len - 1
            } else {
                output_len
            };
            Ok(vec![0.0; output_len])
        }

        fn expert_linear(&mut self, _: &Tensor, _: u32, _: &[f32]) -> Result<Vec<f32>, EngError> {
            Err(EngError::BadRequest("not used in this test".into()))
        }

        fn rows(&mut self, _: &Tensor, _: &[u32]) -> Result<Vec<f32>, EngError> {
            Err(EngError::BadRequest("not used in this test".into()))
        }
    }

    fn tensor(shape: &[u32], values: &[f32]) -> Tensor {
        Tensor::try_from_host(
            Dtype::F32,
            shape.to_vec(),
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        )
        .unwrap()
    }

    fn values(tensor: &Tensor) -> Vec<f32> {
        tensor
            .host_bytes()
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect()
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    fn cuda_projected_mamba_matches_host_output_and_states_without_input_mutation() {
        use crate::family::nemotron35::{
            cuda_projection::CudaProjector, projection::HostProjector,
        };

        let mut inputs = Inputs::new();
        inputs.hidden = tensor(&[1, 2], &[1.0, 2.0]);
        // Single selected term per row makes both projections exact without a tolerance.
        inputs.input = tensor(
            &[7, 2],
            &[
                1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            ],
        );
        inputs.output = tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        inputs.conv = tensor(&[4, 1, 1], &[1.0; 4]);
        inputs.bias = tensor(&[4], &[0.125, 0.25, 0.375, 0.5]);
        inputs.d = tensor(&[1], &[1.0]);
        inputs.conv_state = tensor(&[4, 1], &[0.25, 0.5, 0.75, 1.0]);
        inputs.ssm_state = tensor(&[1, 2, 1], &[0.25, -0.5]);
        let tensors = [
            &inputs.norm,
            &inputs.mixer_norm,
            &inputs.hidden,
            &inputs.input,
            &inputs.output,
            &inputs.conv,
            &inputs.bias,
            &inputs.dt_bias,
            &inputs.a_log,
            &inputs.d,
            &inputs.conv_state,
            &inputs.ssm_state,
        ];
        let before: Vec<_> = tensors
            .iter()
            .map(|t| t.host_bytes().unwrap().to_vec())
            .collect();
        let expected = inputs.run(&mut HostProjector).unwrap();
        use crate::{
            family::nemotron35::admission::{
                self, ACTIVATION_WORKSPACE_BYTES,
                cuda::{CudaPlanInput, admit_cuda_plan},
            },
            memory::MemoryPolicy,
        };
        let mut source_bytes = 0_u64;
        for tensor in &tensors {
            source_bytes = source_bytes
                .checked_add(u64::try_from(tensor.host_bytes().unwrap().len()).unwrap())
                .unwrap();
        }
        let matrices = [&inputs.input, &inputs.output];
        let largest = matrices
            .iter()
            .map(|tensor| u64::try_from(tensor.host_bytes().unwrap().len()).unwrap())
            .max()
            .unwrap();
        let max_shape = matrices.iter().fold([0_u64; 2], |maximum, tensor| {
            [
                maximum[0].max(u64::from(tensor.shape()[0])),
                maximum[1].max(u64::from(tensor.shape()[1])),
            ]
        });
        let host_base_bytes = source_bytes
            .checked_mul(3)
            .and_then(|bytes| bytes.checked_add(ACTIVATION_WORKSPACE_BYTES))
            .unwrap();
        let admission = admit_cuda_plan(
            CudaPlanInput {
                host_base_bytes,
                largest_tensor_bytes: largest,
                packed_pair_bytes: 0,
                max_shape,
                device_weight_cache_bytes: 0,
            },
            MemoryPolicy::default_share(),
            admission::host_available_bytes().expect("measured host memory"),
        )
        .unwrap();
        let mut projector = CudaProjector::new(admission).unwrap();
        let actual = inputs.run(&mut projector).unwrap();
        for (actual, expected) in [
            (&actual.0, &expected.0),
            (&actual.1, &expected.1),
            (&actual.2, &expected.2),
        ] {
            assert_eq!(actual.shape(), expected.shape());
            assert_eq!(actual.host_bytes(), expected.host_bytes());
            assert!(values(actual).iter().any(|&v| v != 0.0));
        }
        assert_ne!(actual.0.host_bytes(), inputs.hidden.host_bytes());
        assert_ne!(actual.1.host_bytes(), inputs.conv_state.host_bytes());
        assert_ne!(actual.2.host_bytes(), inputs.ssm_state.host_bytes());
        for (tensor, bytes) in tensors.iter().zip(before) {
            assert_eq!(tensor.host_bytes().unwrap(), bytes);
        }
    }

    #[test]
    fn shared_forward_calls_supplied_projector_for_both_matrices_in_order() {
        let inputs = Inputs::new();
        let mut projector = RecordingProjector::new(Behavior::ReturnZeros);
        let (output, conv, ssm) = inputs.run(&mut projector).unwrap();

        assert_eq!(projector.calls.len(), 2);
        assert_eq!(projector.calls[0].shape, [7, 2]);
        assert_eq!(
            projector.calls[0].weight,
            inputs.input.host_bytes().unwrap()
        );
        assert_eq!(projector.calls[0].input.len(), 2);
        assert_eq!(projector.calls[1].shape, [2, 2]);
        assert_eq!(
            projector.calls[1].weight,
            inputs.output.host_bytes().unwrap()
        );
        assert_eq!(projector.calls[1].input.len(), 2);
        assert_eq!(values(&output), [1.0, 0.0]);
        assert_eq!(values(&conv), [0.0; 4]);
        assert_eq!(values(&ssm), [0.0; 2]);
    }

    #[test]
    fn projector_errors_propagate_from_first_and_second_call_without_input_mutation() {
        for failed_call in [1, 2] {
            let inputs = Inputs::new();
            let original_hidden = inputs.hidden.host_bytes().unwrap().to_vec();
            let original_conv = inputs.conv_state.host_bytes().unwrap().to_vec();
            let original_ssm = inputs.ssm_state.host_bytes().unwrap().to_vec();
            let mut projector = RecordingProjector::new(Behavior::FailAt(failed_call));

            let error = inputs.run(&mut projector).unwrap_err();
            assert_eq!(
                error,
                EngError::BadRequest(format!("projector failure {failed_call}"))
            );
            assert_eq!(projector.calls.len(), failed_call);
            assert_eq!(inputs.hidden.host_bytes().unwrap(), original_hidden);
            assert_eq!(inputs.conv_state.host_bytes().unwrap(), original_conv);
            assert_eq!(inputs.ssm_state.host_bytes().unwrap(), original_ssm);
        }
    }

    #[test]
    fn projection_weights_remain_finite_validated_before_dispatch() {
        for input_projection in [true, false] {
            let mut inputs = Inputs::new();
            if input_projection {
                inputs.input = tensor(&[7, 2], &[f32::NAN; 14]);
            } else {
                inputs.output = tensor(&[2, 2], &[f32::NAN; 4]);
            }
            let mut projector = RecordingProjector::new(Behavior::ReturnZeros);

            assert!(matches!(
                inputs.run(&mut projector),
                Err(EngError::BadRequest(_))
            ));
            assert!(projector.calls.is_empty());
        }
    }

    #[test]
    fn malformed_projection_lengths_return_backend_errors_instead_of_slicing() {
        for malformed_call in [1, 2] {
            let inputs = Inputs::new();
            let mut projector = RecordingProjector::new(Behavior::ReturnShortAt(malformed_call));

            assert!(matches!(
                inputs.run(&mut projector),
                Err(EngError::Backend {
                    operation: "nemotron35_mamba",
                    ..
                })
            ));
            assert_eq!(projector.calls.len(), malformed_call);
        }
    }
}
