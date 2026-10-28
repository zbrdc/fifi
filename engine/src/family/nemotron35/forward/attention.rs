//! Semantics follow transformers/models/nemotron_h/modeling_nemotron_h.py
//! (SHA-256 859ff78fae71fa3f8857e4ea0620c43aebea624e310e8422c124c507131398d2).

use crate::{
    EngError,
    family::{
        nemotron35::projection::{HostProjector, project_rows},
        project::Projector,
    },
    tensor::{Dtype, Tensor},
};

/// Run a full causal sequence `[T, hidden_size]` through one attention block.
/// All weights are F32; projections use `[out, in]` storage, without biases.
/// The pinned geometry is 32 query heads, 2 KV heads, dimension 128, epsilon 1e-5.
/// Smaller geometries use the same operation for portable reference checks.
#[allow(clippy::too_many_arguments)]
pub fn attention_block_forward(
    norm_weight: &Tensor,
    hidden: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
    o_weight: &Tensor,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<Tensor, EngError> {
    let mut projector = HostProjector;
    attention_block_forward_with_projector(
        norm_weight,
        hidden,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        num_heads,
        num_kv_heads,
        head_dim,
        eps,
        &mut projector,
        None,
        None,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_block_forward_with_projector<P: Projector<Weight = Tensor>>(
    norm_weight: &Tensor,
    hidden: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
    o_weight: &Tensor,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    eps: f32,
    projector: &mut P,
    q_name: Option<&str>,
    k_name: Option<&str>,
    v_name: Option<&str>,
    o_name: Option<&str>,
) -> Result<Tensor, EngError> {
    let &[tokens, width] = hidden.shape() else {
        return Err(invalid("hidden must have shape [T, hidden_size]"));
    };
    if tokens == 0 || width == 0 {
        return Err(invalid("hidden dimensions must exceed zero"));
    }
    if num_heads == 0
        || num_kv_heads == 0
        || !num_heads.is_multiple_of(num_kv_heads)
        || head_dim == 0
    {
        return Err(invalid(
            "head counts must be positive with query heads divisible by KV heads",
        ));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(invalid("epsilon must be finite and positive"));
    }
    let q_width = num_heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid("query width overflow"))?;
    let kv_width = num_kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid("KV width overflow"))?;
    checked_len(tokens, q_width)?;
    checked_len(tokens, kv_width)?;
    let hidden_values = read_f32(hidden, &[tokens, width])?;
    // Name-hinted weights were validated at load; skip their geometry check.
    if q_name.is_none() {
        validated_f32_bytes(q_weight, &[q_width, width])?;
    }
    if k_name.is_none() {
        validated_f32_bytes(k_weight, &[kv_width, width])?;
    }
    if v_name.is_none() {
        validated_f32_bytes(v_weight, &[kv_width, width])?;
    }
    if o_name.is_none() {
        validated_f32_bytes(o_weight, &[width, q_width])?;
    }

    // NemotronHBlock.forward lines 927-931: preserve the input, then pre-normalize.
    let normalized = super::rms_norm(hidden, norm_weight, eps)?;
    let normalized = read_f32(&normalized, &[tokens, width])?;
    // NemotronHAttention lines 847-849, 861-866: bias-free Q/K/V linear projections.
    let q = project_rows(
        projector,
        q_weight,
        &normalized,
        width as usize,
        q_width as usize,
        "nemotron35_attention",
        "query",
        q_name,
    )?;
    let k = project_rows(
        projector,
        k_weight,
        &normalized,
        width as usize,
        kv_width as usize,
        "nemotron35_attention",
        "key",
        k_name,
    )?;
    let v = project_rows(
        projector,
        v_weight,
        &normalized,
        width as usize,
        kv_width as usize,
        "nemotron35_attention",
        "value",
        v_name,
    )?;
    // NemotronHAttention.forward lines 861-893: no positional encoding is applied.
    let context = causal_attention(
        &q,
        &k,
        &v,
        num_heads as usize,
        num_kv_heads as usize,
        head_dim as usize,
    )?;
    // NemotronHAttention.forward lines 891-892: merge heads and apply o_proj.
    let mut output = project_rows(
        projector,
        o_weight,
        &context,
        q_width as usize,
        width as usize,
        "nemotron35_attention",
        "output",
        o_name,
    )?;
    // NemotronHBlock.forward lines 933-944: add the original, unscaled residual.
    // rescale_prenorm_residual only changes initialization (lines 1013-1031), not forward.
    for (value, residual) in output.iter_mut().zip(hidden_values) {
        *value += residual;
        if !value.is_finite() {
            return Err(invalid("attention output is not finite"));
        }
    }
    Tensor::try_from_host(
        Dtype::F32,
        vec![tokens, width],
        output.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

/// One token through an attention block with incremental key/value caches
/// (`keys`/`values` are `[capacity, kv_width]`; row `position` receives this
/// token's projections). The per-row math is identical to
/// `attention_block_forward_with_projector` — projections are row-independent
/// (project_rows calls `linear` per row), and the causal loop reuses the same
/// dot, softmax, and accumulation orders for this one position — so results
/// are bit-identical to replaying the full prefix.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the batch block function"
)]
pub(crate) fn attention_block_step_with_projector<P: Projector<Weight = Tensor>>(
    norm_weight: &Tensor,
    hidden_row: &[f32],
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
    o_weight: &Tensor,
    keys: &mut [f32],
    values: &mut [f32],
    position: usize,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    eps: f32,
    projector: &mut P,
    q_name: Option<&str>,
    k_name: Option<&str>,
    v_name: Option<&str>,
    o_name: Option<&str>,
) -> Result<Vec<f32>, EngError> {
    let width = norm_weight.shape()[0] as usize;
    let head_dim = head_dim as usize;
    let num_heads = num_heads as usize;
    let num_kv_heads = num_kv_heads as usize;
    if num_heads == 0
        || num_kv_heads == 0
        || !num_heads.is_multiple_of(num_kv_heads)
        || head_dim == 0
    {
        return Err(invalid(
            "head counts must be positive with query heads divisible by KV heads",
        ));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(invalid("epsilon must be finite and positive"));
    }
    let q_width = num_heads * head_dim;
    let kv_width = num_kv_heads * head_dim;
    if hidden_row.len() != width {
        return Err(invalid("hidden row width does not match the norm"));
    }
    let kv_offset = position * kv_width;
    if kv_offset + kv_width > keys.len() || keys.len() != values.len() {
        return Err(invalid("key/value cache does not hold this position"));
    }
    // Same pre-normalize as the batch path, for this row only.
    let row = tensor_f32(&[1, width as u32], hidden_row)?;
    let normalized = super::rms_norm(&row, norm_weight, eps)?;
    let normalized = read_f32(&normalized, &[1, width as u32])?;
    let mut project = |weight: &Tensor, out: usize, label: &str, name: Option<&str>| {
        project_rows(
            projector,
            weight,
            &normalized,
            width,
            out,
            "nemotron35_attention",
            label,
            name,
        )
    };
    let q = project(q_weight, q_width, "query", q_name)?;
    let k = project(k_weight, kv_width, "key", k_name)?;
    let v = project(v_weight, kv_width, "value", v_name)?;
    keys[kv_offset..kv_offset + kv_width].copy_from_slice(&k);
    values[kv_offset..kv_offset + kv_width].copy_from_slice(&v);
    // causal_attention's per-(token, head) body, for t = position over the
    // cached prefix: sequential f32 dot sums, fold-max softmax, in-order
    // source accumulation, GQA contiguous repeat, head_dim**-0.5 scaling.
    let group_size = num_heads / num_kv_heads;
    let scaling = (head_dim as f32).sqrt().recip();
    let mut context = vec![0.0f32; q_width];
    let mut scores = Vec::with_capacity(position + 1);
    for head in 0..num_heads {
        let query_offset = head * head_dim;
        let query = &q[query_offset..query_offset + head_dim];
        let kv_head = head / group_size;
        scores.clear();
        for source in 0..=position {
            let key_offset = source * kv_width + kv_head * head_dim;
            let key = &keys[key_offset..key_offset + head_dim];
            let score = query.iter().zip(key).map(|(q, k)| q * k).sum::<f32>() * scaling;
            if !score.is_finite() {
                return Err(invalid("attention score is not finite"));
            }
            scores.push(score);
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for score in &mut scores {
            *score = (*score - max).exp();
        }
        let denominator = scores.iter().sum::<f32>();
        for (source, score) in scores.iter().enumerate() {
            let value_offset = source * kv_width + kv_head * head_dim;
            let probability = score / denominator;
            for d in 0..head_dim {
                context[query_offset + d] += probability * values[value_offset + d];
            }
        }
    }
    let mut output = project_rows(
        projector,
        o_weight,
        &context,
        q_width,
        width,
        "nemotron35_attention",
        "output",
        o_name,
    )?;
    for (value, residual) in output.iter_mut().zip(hidden_row) {
        *value += residual;
        if !value.is_finite() {
            return Err(invalid("attention output is not finite"));
        }
    }
    Ok(output)
}

fn tensor_f32(shape: &[u32], data: &[f32]) -> Result<Tensor, EngError> {
    Tensor::try_from_host(
        Dtype::F32,
        shape.to_vec(),
        data.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

fn invalid(message: &str) -> EngError {
    EngError::BadRequest(format!("NemotronH attention: {message}"))
}

fn checked_len(rows: u32, cols: u32) -> Result<usize, EngError> {
    let len = (rows as usize).checked_mul(cols as usize);
    len.filter(|len| {
        len.checked_mul(4)
            .is_some_and(|bytes| bytes <= isize::MAX as usize)
    })
    .ok_or_else(|| invalid("tensor byte length exceeds host limits"))
}

fn read_f32(tensor: &Tensor, shape: &[u32]) -> Result<Vec<f32>, EngError> {
    let bytes = validated_f32_bytes(tensor, shape)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

fn validated_f32_bytes<'a>(tensor: &'a Tensor, shape: &[u32]) -> Result<&'a [u8], EngError> {
    if tensor.dtype() != Dtype::F32 || tensor.shape() != shape {
        return Err(invalid(
            "tensor dtype or shape does not match the block geometry",
        ));
    }
    let bytes = tensor
        .host_bytes()
        .ok_or_else(|| invalid("tensor is not host-backed"))?;
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

#[allow(clippy::too_many_arguments)]
fn causal_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, EngError> {
    let q_width = num_heads * head_dim;
    let kv_width = num_kv_heads * head_dim;
    let tokens = q.len() / q_width;
    let mut context = vec![0.0; q.len()];
    let mut scores = Vec::with_capacity(tokens);
    // eager_attention_forward lines 813-816: repeat each KV head contiguously.
    let group_size = num_heads / num_kv_heads;
    // NemotronHAttention.__init__ line 843: scaling = head_dim**-0.5.
    let scaling = (head_dim as f32).sqrt().recip();
    for t in 0..tokens {
        for head in 0..num_heads {
            let query_offset = t * q_width + head * head_dim;
            let query = &q[query_offset..query_offset + head_dim];
            let kv_head = head / group_size;
            scores.clear();
            // NemotronHAttention line 845 and eager_attention_forward lines 819-824:
            // causal masking, scaled QK^T, and F32 softmax. Only positions <= t exist here.
            for source in 0..=t {
                let key_offset = source * kv_width + kv_head * head_dim;
                let key = &k[key_offset..key_offset + head_dim];
                let score = query.iter().zip(key).map(|(q, k)| q * k).sum::<f32>() * scaling;
                if !score.is_finite() {
                    return Err(invalid("attention score is not finite"));
                }
                scores.push(score);
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for score in &mut scores {
                *score = (*score - max).exp();
            }
            let denominator = scores.iter().sum::<f32>();
            for (source, score) in scores.iter().enumerate() {
                let value_offset = source * kv_width + kv_head * head_dim;
                let probability = score / denominator;
                // eager_attention_forward lines 826-827: weighted V, then head concatenation.
                for d in 0..head_dim {
                    context[query_offset + d] += probability * v[value_offset + d];
                }
            }
        }
    }
    Ok(context)
}

#[cfg(test)]
mod projector_tests {
    use super::attention_block_forward_with_projector;
    use crate::{
        EngError,
        family::project::Projector,
        tensor::{Dtype, Tensor},
    };

    #[derive(Clone, Copy)]
    enum Behavior {
        Multiply,
        FailAt(usize),
        ReturnShortAt(usize),
    }

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
                return Err(EngError::BadRequest(format!(
                    "test projector failure {call}"
                )));
            }
            let &[rows, columns] = weight.shape() else {
                return Err(EngError::BadRequest("test weight rank".into()));
            };
            let rows = rows as usize;
            let columns = columns as usize;
            let values = weight
                .host_bytes()
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect::<Vec<_>>();
            let output = values
                .chunks_exact(columns)
                .map(|row| row.iter().zip(input).map(|(w, x)| w * x).sum())
                .collect::<Vec<_>>();
            if matches!(self.behavior, Behavior::ReturnShortAt(n) if n == call) {
                return Ok(output[..rows - 1].to_vec());
            }
            Ok(output)
        }

        fn expert_linear(&mut self, _: &Tensor, _: u32, _: &[f32]) -> Result<Vec<f32>, EngError> {
            Err(EngError::BadRequest("unused expert projection".into()))
        }

        fn rows(&mut self, _: &Tensor, _: &[u32]) -> Result<Vec<f32>, EngError> {
            Err(EngError::BadRequest("unused row projection".into()))
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

    fn identity() -> Tensor {
        tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0])
    }

    #[test]
    fn shared_attention_projects_each_nonzero_token_in_qkvo_order() {
        let norm = tensor(&[2], &[1.0, 1.0]);
        let hidden = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let q = identity();
        let k = identity();
        let v = identity();
        let o = identity();
        let inputs =
            [&norm, &hidden, &q, &k, &v, &o].map(|weight| weight.host_bytes().unwrap().to_vec());
        let mut projector = RecordingProjector::new(Behavior::Multiply);

        attention_block_forward_with_projector(
            &norm,
            &hidden,
            &q,
            &k,
            &v,
            &o,
            1,
            1,
            2,
            1.0,
            &mut projector,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(projector.calls.len(), 8);
        for (projection, weight) in [&q, &k, &v, &o].into_iter().enumerate() {
            for (token, call) in projector.calls[projection * 2..projection * 2 + 2]
                .iter()
                .enumerate()
            {
                assert_eq!(call.shape, [2, 2]);
                assert_eq!(call.weight, weight.host_bytes().unwrap());
                assert_eq!(call.input.len(), 2);
                if projection == 0 {
                    let (x, y): (f32, f32) = if token == 0 { (1.0, 2.0) } else { (3.0, 4.0) };
                    let scale = ((x * x + y * y) / 2.0 + 1.0).sqrt().recip();
                    assert_eq!(call.input, [x * scale, y * scale]);
                }
            }
        }
        assert_ne!(projector.calls[0].input, projector.calls[1].input);
        for (before, weight) in inputs.iter().zip([&norm, &hidden, &q, &k, &v, &o]) {
            assert_eq!(weight.host_bytes().unwrap(), before);
        }
    }

    #[test]
    fn shared_attention_propagates_projector_errors_without_mutating_inputs() {
        let norm = tensor(&[2], &[1.0, 1.0]);
        let hidden = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let q = identity();
        let k = identity();
        let v = identity();
        let o = identity();
        let original =
            [&norm, &hidden, &q, &k, &v, &o].map(|weight| weight.host_bytes().unwrap().to_vec());

        for failed_call in [1, 4, 8] {
            let mut projector = RecordingProjector::new(Behavior::FailAt(failed_call));
            let error = attention_block_forward_with_projector(
                &norm,
                &hidden,
                &q,
                &k,
                &v,
                &o,
                1,
                1,
                2,
                1.0,
                &mut projector,
                None,
                None,
                None,
                None,
            )
            .unwrap_err();
            assert_eq!(
                error,
                EngError::BadRequest(format!("test projector failure {failed_call}"))
            );
            assert_eq!(projector.calls.len(), failed_call);
            for (before, weight) in original.iter().zip([&norm, &hidden, &q, &k, &v, &o]) {
                assert_eq!(weight.host_bytes().unwrap(), before);
            }
        }
    }

    #[test]
    fn shared_attention_rejects_malformed_projection_lengths() {
        let norm = tensor(&[2], &[1.0, 1.0]);
        let hidden = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let q = identity();
        let k = identity();
        let v = identity();
        let o = identity();
        for malformed_call in [1, 8] {
            let mut projector = RecordingProjector::new(Behavior::ReturnShortAt(malformed_call));
            assert!(matches!(
                attention_block_forward_with_projector(
                    &norm,
                    &hidden,
                    &q,
                    &k,
                    &v,
                    &o,
                    1,
                    1,
                    2,
                    1.0,
                    &mut projector,
                    None,
                    None,
                    None,
                    None,
                ),
                Err(EngError::Backend { .. })
            ));
            assert_eq!(projector.calls.len(), malformed_call);
        }
    }

    #[test]
    fn incremental_kv_steps_are_bit_identical_to_the_batch_replay() {
        use super::attention_block_step_with_projector;
        use crate::family::nemotron35::projection::HostProjector;

        let hidden = 16;
        let heads = 4;
        let kv_heads = 2;
        let head_dim = 4;
        let q_width = heads * head_dim;
        let kv_width = kv_heads * head_dim;
        let tokens = 12;
        let capacity = 16;
        // Deterministic, non-trivial values: a small LCG keeps every row
        // distinct without pulling in a random dependency.
        let mut state = 0x2545_F491_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as i32 as f32 / (1 << 20) as f32
        };
        let rows: Vec<Vec<f32>> = (0..tokens)
            .map(|_| (0..hidden).map(|_| next()).collect())
            .collect();
        let mut weight = |out: usize, _name: &str| {
            let values: Vec<f32> = (0..out * hidden).map(|_| next()).collect();
            Tensor::try_from_host(
                Dtype::F32,
                vec![out as u32, hidden as u32],
                values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            )
            .unwrap()
        };
        // Dense [out, hidden] tensors pass validation; projections use the
        // shared per-row linear path.
        let norm = Tensor::try_from_host(
            Dtype::F32,
            vec![hidden as u32],
            vec![1.0f32; hidden]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        )
        .unwrap();
        let q = weight(q_width, "q");
        let k = weight(kv_width, "k");
        let v = weight(kv_width, "v");
        let o = weight(hidden, "o");

        // Batch replay: the whole sequence through one call.
        let batch_input = Tensor::try_from_host(
            Dtype::F32,
            vec![tokens as u32, hidden as u32],
            rows.iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>()
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect(),
        )
        .unwrap();
        let mut batch_projector = HostProjector;
        let batched = super::attention_block_forward_with_projector(
            &norm,
            &batch_input,
            &q,
            &k,
            &v,
            &o,
            heads as u32,
            kv_heads as u32,
            head_dim as u32,
            1e-5,
            &mut batch_projector,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let batched = batched.host_bytes().unwrap();

        // Incremental: one row per token through the step function.
        let mut keys = vec![0.0f32; capacity * kv_width];
        let mut values = vec![0.0f32; capacity * kv_width];
        let mut step_projector = HostProjector;
        for (position, row) in rows.iter().enumerate() {
            let out = attention_block_step_with_projector(
                &norm,
                row,
                &q,
                &k,
                &v,
                &o,
                &mut keys,
                &mut values,
                position,
                heads as u32,
                kv_heads as u32,
                head_dim as u32,
                1e-5,
                &mut step_projector,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            let expected = &batched[position * hidden * 4..(position + 1) * hidden * 4];
            let actual: Vec<u8> = out.iter().flat_map(|x| x.to_le_bytes()).collect();
            assert_eq!(actual.as_slice(), expected, "token {position}");
        }
    }

    #[test]
    fn attention_step_rejects_short_caches_and_bad_geometry() {
        use super::attention_block_step_with_projector;
        use crate::family::nemotron35::projection::HostProjector;

        let norm = Tensor::try_from_host(
            Dtype::F32,
            vec![8],
            [1.0f32; 8].iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap();
        let zeros = |out: u32, hidden: u32| {
            Tensor::try_from_host(
                Dtype::F32,
                vec![out, hidden],
                vec![0.0f32; (out * hidden) as usize]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            )
            .unwrap()
        };
        let row = vec![0.1f32; 8];
        let q = zeros(16, 8);
        let k = zeros(8, 8);
        let v = zeros(8, 8);
        let o = zeros(8, 16);
        let mut projector = HostProjector;
        let mut full_keys = vec![0.0f32; 4 * 8];
        let mut full_values = vec![0.0f32; 4 * 8];
        // Cache too small for the position.
        let mut short_keys = vec![0.0f32; 8];
        let mut short_values = vec![0.0f32; 8];
        assert!(
            attention_block_step_with_projector(
                &norm,
                &row,
                &q,
                &k,
                &v,
                &o,
                &mut short_keys,
                &mut short_values,
                1,
                4,
                2,
                4,
                1e-5,
                &mut projector,
                None,
                None,
                None,
                None,
            )
            .is_err()
        );
        // Mismatched key/value lengths.
        let mut wide_values = vec![0.0f32; 4 * 8 + 8];
        assert!(
            attention_block_step_with_projector(
                &norm,
                &row,
                &q,
                &k,
                &v,
                &o,
                &mut full_keys,
                &mut wide_values,
                1,
                4,
                2,
                4,
                1e-5,
                &mut projector,
                None,
                None,
                None,
                None,
            )
            .is_err()
        );
        // Indivisible head counts and a non-finite epsilon.
        assert!(
            attention_block_step_with_projector(
                &norm,
                &row,
                &q,
                &k,
                &v,
                &o,
                &mut full_keys,
                &mut full_values,
                0,
                3,
                2,
                4,
                1e-5,
                &mut projector,
                None,
                None,
                None,
                None,
            )
            .is_err()
        );
        assert!(
            attention_block_step_with_projector(
                &norm,
                &row,
                &q,
                &k,
                &v,
                &o,
                &mut full_keys,
                &mut full_values,
                0,
                4,
                2,
                4,
                f32::NAN,
                &mut projector,
                None,
                None,
                None,
                None,
            )
            .is_err()
        );
    }
}
