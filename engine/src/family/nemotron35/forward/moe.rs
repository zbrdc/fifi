//! Portable F32 NemotronH MoE, from transformers/models/nemotron_h/modeling_nemotron_h.py.
//! Reference SHA-256: 859ff78fae71fa3f8857e4ea0620c43aebea624e310e8422c124c507131398d2.

use crate::{
    EngError,
    expert_tier::{Overlap, run_overlapped},
    family::{
        nemotron35::{
            exec::{ExpertIndex, ExpertProjection},
            projection::{HostProjector, project_rows},
        },
        project::Projector,
    },
    tensor::{Dtype, Tensor},
};

/// Forward-time routing and normalization parameters; expert counts come from router rows.
#[derive(Clone, Copy, Debug)]
pub struct MoeConfig {
    /// Reference line 710 uses `n_group`, not Mamba's `n_groups` (8). The model uses 1.
    pub n_group: u32,
    pub topk_group: u32,
    pub experts_per_token: u32,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    pub eps: f32,
}

/// Bias-free non-gated MLP matrices, stored `[intermediate, input]`, `[input, intermediate]`.
#[derive(Clone, Copy)]
pub struct MlpWeights<'a> {
    pub up: &'a Tensor,
    pub down: &'a Tensor,
}

/// A caller-loaded expert; no unselected expert weights are required.
pub struct ExpertWeights<'a> {
    pub index: u32,
    pub weights: MlpWeights<'a>,
}

#[derive(Clone, Copy, Debug)]
pub struct RoutedExpert {
    pub index: u32,
    pub weight: f32,
}

/// Route already pre-normalized `[T, hidden]` rows, returning one selection per token.
/// Weights include scaling. Correction bias changes selection only (reference lines 708, 724, 755).
/// Results use descending corrected score, breaking ties by index; PyTorch topk does not specify tie order.
pub fn route(
    normalized_hidden: &Tensor,
    router_weight: &Tensor,
    correction_bias: &Tensor,
    config: MoeConfig,
) -> Result<Vec<Vec<RoutedExpert>>, EngError> {
    let mut projector = HostProjector;
    route_with_projector(
        normalized_hidden,
        router_weight,
        correction_bias,
        config,
        &mut projector,
    )
}

pub(crate) fn route_with_projector<P: Projector<Weight = Tensor>>(
    normalized_hidden: &Tensor,
    router_weight: &Tensor,
    correction_bias: &Tensor,
    config: MoeConfig,
    projector: &mut P,
) -> Result<Vec<Vec<RoutedExpert>>, EngError> {
    let (tokens, width) = matrix_shape(normalized_hidden)?;
    let (experts, router_width) = matrix_shape(router_weight)?;
    let c = config;
    if router_width != width
        || c.n_group == 0
        || !experts.is_multiple_of(c.n_group)
        || experts / c.n_group < 2
        || c.topk_group == 0
        || c.topk_group > c.n_group
        || c.experts_per_token == 0
        || c.experts_per_token > (experts / c.n_group) * c.topk_group
        || !c.routed_scaling_factor.is_finite()
        || c.routed_scaling_factor <= 0.0
    {
        return Err(invalid("invalid routing geometry or scaling"));
    }
    // Router lines 757-760: F32 linear, without bias; sigmoid is applied later (707).
    let hidden = read_f32(normalized_hidden, &[tokens, width])?;
    validated_f32_bytes(router_weight, &[experts, width])?;
    let bias = read_f32(correction_bias, &[experts])?;
    let logits = project_rows(
        projector,
        router_weight,
        &hidden,
        width as usize,
        experts as usize,
        "nemotron35_moe",
        "router",
        None,
    )?;
    logits
        .chunks_exact(experts as usize)
        .map(|row| select(row, &bias, c))
        .collect()
}

fn select(logits: &[f32], bias: &[f32], c: MoeConfig) -> Result<Vec<RoutedExpert>, EngError> {
    // Lines 707-713: sigmoid plus correction for choice; group score is its top TWO sum, not max.
    let scores: Vec<_> = logits.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
    let corrected: Vec<_> = scores.iter().zip(bias).map(|(s, b)| s + b).collect();
    finite(&corrected)?;
    let group_size = logits.len() / c.n_group as usize;
    let mut groups: Vec<_> = corrected
        .chunks_exact(group_size)
        .enumerate()
        .map(|(group, row)| {
            let order = descending_indices(row);
            (group, row[order[0]] + row[order[1]])
        })
        .collect();
    if groups.iter().any(|(_, score)| !score.is_finite()) {
        return Err(invalid("group score is not finite"));
    }
    groups.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    // Lines 714-724: mask unselected groups, top-k corrected scores, gather ORIGINAL sigmoid scores.
    let mut indices: Vec<_> = descending_indices(&corrected)
        .into_iter()
        .filter(|index| {
            groups[..c.topk_group as usize]
                .iter()
                .any(|(g, _)| *g == index / group_size)
        })
        .take(c.experts_per_token as usize)
        .map(|index| RoutedExpert {
            index: index as u32,
            weight: scores[index],
        })
        .collect();
    // Lines 725-728: optionally renormalize with 1e-20, then scale, never a softmax.
    let denominator = if c.norm_topk_prob {
        indices.iter().map(|e| e.weight).sum::<f32>() + 1e-20
    } else {
        1.0
    };
    for expert in &mut indices {
        expert.weight = expert.weight / denominator * c.routed_scaling_factor;
    }
    Ok(indices)
}

fn descending_indices(scores: &[f32]) -> Vec<usize> {
    let mut indices: Vec<_> = (0..scores.len()).collect();
    indices.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    indices
}

/// Apply one ungated relu2 expert to `[T, input]` (reference lines 625-632, 657-661).
pub fn expert_forward(hidden: &Tensor, weights: MlpWeights<'_>) -> Result<Tensor, EngError> {
    let mut projector = HostProjector;
    expert_forward_with_projector(hidden, weights, &mut projector)
}

pub(crate) fn expert_forward_with_projector<P: Projector<Weight = Tensor>>(
    hidden: &Tensor,
    weights: MlpWeights<'_>,
    projector: &mut P,
) -> Result<Tensor, EngError> {
    let (tokens, width) = matrix_shape(hidden)?;
    let hidden = read_f32(hidden, &[tokens, width])?;
    let output = mlp(projector, &hidden, width, weights, None, None)?;
    tensor(&[tokens, width], output)
}

/// Apply caller-loaded selected experts to latent `[T, latent]` rows.
/// Selection weights are used as returned by `route`; this function does not normalize them again.
pub fn apply_routed_experts(
    hidden: &Tensor,
    selection: &[Vec<RoutedExpert>],
    experts: &[ExpertWeights<'_>],
) -> Result<Tensor, EngError> {
    let mut projector = HostProjector;
    apply_routed_experts_with_projector(hidden, selection, experts, &mut projector)
}

pub(crate) fn apply_routed_experts_with_projector<P: Projector<Weight = Tensor>>(
    hidden: &Tensor,
    selection: &[Vec<RoutedExpert>],
    experts: &[ExpertWeights<'_>],
    projector: &mut P,
) -> Result<Tensor, EngError> {
    let (tokens, width) = matrix_shape(hidden)?;
    let hidden = read_f32(hidden, &[tokens, width])?;
    if selection.len() != tokens as usize || selection.iter().any(Vec::is_empty) {
        return Err(invalid("selection must contain one nonempty row per token"));
    }
    for row in selection {
        for (position, e) in row.iter().enumerate() {
            if !e.weight.is_finite()
                || e.weight < 0.0
                || row[..position].iter().any(|prior| prior.index == e.index)
                || !experts.iter().any(|loaded| loaded.index == e.index)
            {
                return Err(invalid("invalid selection or missing selected expert"));
            }
        }
    }
    let mut loaded: Vec<_> = experts.iter().collect();
    loaded.sort_by_key(|expert| expert.index);
    if loaded.windows(2).any(|pair| pair[0].index == pair[1].index) {
        return Err(invalid("duplicate loaded expert"));
    }
    let mut output = vec![0.0; hidden.len()];
    // Lines 643-667: process hit experts in index order, weight their outputs, then accumulate.
    for expert in loaded {
        let hits: Vec<_> = selection
            .iter()
            .enumerate()
            .filter_map(|(token, row)| {
                row.iter()
                    .find(|e| e.index == expert.index)
                    .map(|e| (token, e.weight))
            })
            .collect();
        if hits.is_empty() {
            continue;
        }
        let input: Vec<_> = hits
            .iter()
            .flat_map(|(t, _)| {
                hidden[*t * width as usize..(*t + 1) * width as usize]
                    .iter()
                    .copied()
            })
            .collect();
        let projected = mlp(projector, &input, width, expert.weights, None, None)?;
        for ((token, weight), row) in hits.iter().zip(projected.chunks_exact(width as usize)) {
            for (dest, value) in output[*token * width as usize..(*token + 1) * width as usize]
                .iter_mut()
                .zip(row)
            {
                *dest += value * weight;
            }
        }
    }
    tensor(&[tokens, width], output)
}

/// Run pre-norm, routing, latent experts, unweighted shared MLP, and the original residual.
/// All tensors are host F32, with bias-free `[out, in]` projections.
/// Call `route` first to load selected weights; this convenience function recomputes that selection.
/// Latent projections are `[latent, hidden]` and `[hidden, latent]` (reference lines 699-701).
#[allow(clippy::too_many_arguments)]
pub fn moe_block_forward(
    norm_weight: &Tensor,
    hidden: &Tensor,
    router_weight: &Tensor,
    correction_bias: &Tensor,
    latent_up: &Tensor,
    latent_down: &Tensor,
    experts: &[ExpertWeights<'_>],
    shared: MlpWeights<'_>,
    config: MoeConfig,
) -> Result<Tensor, EngError> {
    let mut projector = HostProjector;
    moe_block_forward_with_projector(
        norm_weight,
        hidden,
        router_weight,
        correction_bias,
        latent_up,
        latent_down,
        experts,
        shared,
        config,
        &mut projector,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_block_forward_with_projector<P: Projector<Weight = Tensor>>(
    norm_weight: &Tensor,
    hidden: &Tensor,
    router_weight: &Tensor,
    correction_bias: &Tensor,
    latent_up: &Tensor,
    latent_down: &Tensor,
    experts: &[ExpertWeights<'_>],
    shared: MlpWeights<'_>,
    config: MoeConfig,
    projector: &mut P,
) -> Result<Tensor, EngError> {
    let normalized = super::rms_norm(hidden, norm_weight, config.eps)?;
    // Lines 732-741: route in hidden space BEFORE reducing to latent space.
    let selected = route_with_projector(
        &normalized,
        router_weight,
        correction_bias,
        config,
        projector,
    )?;
    moe_apply_selected_with_projector(
        hidden,
        &normalized,
        &selected,
        latent_up,
        latent_down,
        experts,
        shared,
        projector,
    )
}

/// Apply a previously routed block without repeating normalization or expert selection.
/// `normalized_hidden` and `selection` must come from this block's RMS norm and `route`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_apply_selected(
    hidden: &Tensor,
    normalized_hidden: &Tensor,
    selection: &[Vec<RoutedExpert>],
    latent_up: &Tensor,
    latent_down: &Tensor,
    experts: &[ExpertWeights<'_>],
    shared: MlpWeights<'_>,
) -> Result<Tensor, EngError> {
    let mut projector = HostProjector;
    moe_apply_selected_with_projector(
        hidden,
        normalized_hidden,
        selection,
        latent_up,
        latent_down,
        experts,
        shared,
        &mut projector,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_apply_selected_with_projector<P: Projector<Weight = Tensor>>(
    hidden: &Tensor,
    normalized_hidden: &Tensor,
    selection: &[Vec<RoutedExpert>],
    latent_up: &Tensor,
    latent_down: &Tensor,
    experts: &[ExpertWeights<'_>],
    shared: MlpWeights<'_>,
    projector: &mut P,
) -> Result<Tensor, EngError> {
    let (tokens, width) = matrix_shape(hidden)?;
    let hidden_values = read_f32(hidden, &[tokens, width])?;
    let normalized = read_f32(normalized_hidden, &[tokens, width])?;
    let (latent, _) = matrix_shape(latent_up)?;
    validated_f32_bytes(latent_up, &[latent, width])?;
    validated_f32_bytes(latent_down, &[width, latent])?;
    let latent_hidden = tensor(
        &[tokens, latent],
        project_rows(
            projector,
            latent_up,
            &normalized,
            width as usize,
            latent as usize,
            "nemotron35_moe",
            "latent up",
            None,
        )?,
    )?;
    let routed =
        apply_routed_experts_with_projector(&latent_hidden, selection, experts, projector)?;
    let routed = read_f32(&routed, &[tokens, latent])?;
    let mut output = project_rows(
        projector,
        latent_down,
        &routed,
        latent as usize,
        width as usize,
        "nemotron35_moe",
        "latent down",
        None,
    )?;
    // Lines 743-744 and MLP 602-607: shared path sees normalized HIDDEN, with no router weight.
    let shared_output = mlp(projector, &normalized, width, shared, None, None)?;
    // Block lines 933-944: unscaled original residual, not the normalized MoE-local residual.
    for ((value, shared), residual) in output.iter_mut().zip(shared_output).zip(hidden_values) {
        *value += shared;
        *value += residual;
    }
    tensor(&[tokens, width], output)
}

fn mlp<P: Projector<Weight = Tensor>>(
    projector: &mut P,
    input: &[f32],
    width: u32,
    weights: MlpWeights<'_>,
    up_name: Option<&str>,
    down_name: Option<&str>,
) -> Result<Vec<f32>, EngError> {
    let (intermediate, _) = matrix_shape(weights.up)?;
    validated_f32_bytes(weights.up, &[intermediate, width])?;
    validated_f32_bytes(weights.down, &[width, intermediate])?;
    let mut activation = project_rows(
        projector,
        weights.up,
        input,
        width as usize,
        intermediate as usize,
        "nemotron35_moe",
        "expert up",
        up_name,
    )?;
    // Reference lines 634, 658: no gate; relu2 is ReLU(x)^2 (activations.py:211-214,340).
    for value in &mut activation {
        *value = value.max(0.0).powi(2);
    }
    project_rows(
        projector,
        weights.down,
        &activation,
        intermediate as usize,
        width as usize,
        "nemotron35_moe",
        "expert down",
        down_name,
    )
}

fn invalid(message: &str) -> EngError {
    EngError::BadRequest(format!("NemotronH MoE: {message}"))
}

fn matrix_shape(t: &Tensor) -> Result<(u32, u32), EngError> {
    match t.shape() {
        &[rows, cols] if rows > 0 && cols > 0 => Ok((rows, cols)),
        _ => Err(invalid("expected nonempty matrix")),
    }
}

fn finite(values: &[f32]) -> Result<(), EngError> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(invalid("non-finite tensor value"))
    }
}

fn read_f32(t: &Tensor, shape: &[u32]) -> Result<Vec<f32>, EngError> {
    let bytes = validated_f32_bytes(t, shape)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

thread_local! {
    /// Data pointers already finiteness-validated: the first use of a weight
    /// scans every f32; subsequent uses of the same tensor skip the scan.
    /// This preserves the trust boundary (every NEW tensor is validated)
    /// while eliminating the ~200 MiB/layer per-token re-scan of cached
    /// weights that was 90% of the cpu_routed weight_load cost.
    static VALIDATED_PTRS: std::cell::RefCell<std::collections::HashSet<usize>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

fn validated_f32_bytes<'a>(t: &'a Tensor, shape: &[u32]) -> Result<&'a [u8], EngError> {
    if t.dtype() != Dtype::F32 || t.shape() != shape {
        return Err(invalid("unexpected dtype or shape"));
    }
    let bytes = t
        .host_bytes()
        .ok_or_else(|| invalid("expected host-backed tensor"))?;
    let pointer = bytes.as_ptr() as usize;
    let already_validated = VALIDATED_PTRS.with(|set| set.borrow().contains(&pointer));
    if !already_validated {
        if bytes
            .as_chunks::<4>()
            .0
            .iter()
            .any(|b| !f32::from_le_bytes(*b).is_finite())
        {
            return Err(invalid("non-finite tensor value"));
        }
        VALIDATED_PTRS.with(|set| set.borrow_mut().insert(pointer));
    }
    Ok(bytes)
}

fn tensor(shape: &[u32], values: Vec<f32>) -> Result<Tensor, EngError> {
    finite(&values)?;
    Tensor::try_from_host(
        Dtype::F32,
        shape.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

/// Projects one packed expert matrix against one activation row.
///
/// The projector is passed in so the caller keeps one mutable borrow.
pub(crate) type PackedProjection<'a, P> = dyn FnMut(&mut P, ExpertIndex, ExpertProjection, &[f32], &mut [f32]) -> Result<(), EngError>
    + 'a;

/// Phase-batched packed projection: one call covers every (expert, token)
/// pair of one phase. `requests.len()` pairs; `inputs` holds the pairs'
/// input rows concatenated; `outputs` receives the pairs' output rows
/// concatenated. Results must equal the per-pair `PackedProjection`
/// element for element.
pub(crate) type BatchedPackedProjection<'a, P> = dyn FnMut(&mut P, &[(ExpertIndex, ExpertProjection)], &[f32], &mut [f32]) -> Result<(), EngError>
    + 'a;

/// Accumulates the selected experts through packed projections.
pub(crate) fn apply_routed_experts_packed<P: Projector<Weight = Tensor>>(
    hidden: &Tensor,
    selection: &[Vec<RoutedExpert>],
    intermediate: usize,
    projector: &mut P,
    project: &mut PackedProjection<'_, P>,
) -> Result<Tensor, EngError> {
    let (tokens, width) = matrix_shape(hidden)?;
    let hidden_values = read_f32(hidden, &[tokens, width])?;
    if selection.len() != tokens as usize || selection.iter().any(Vec::is_empty) {
        return Err(invalid("selection must contain one nonempty row per token"));
    }
    if intermediate == 0 || width == 0 {
        return Err(invalid("packed expert geometry must be positive"));
    }
    let mut order: Vec<u32> = Vec::new();
    for row in selection {
        for (position, e) in row.iter().enumerate() {
            let repeated = row[..position].iter().any(|prior| prior.index == e.index);
            if !e.weight.is_finite() || e.weight < 0.0 || repeated {
                return Err(invalid("invalid selection"));
            }
            if !order.contains(&e.index) {
                order.push(e.index);
            }
        }
    }
    // Lines 643-667: process hit experts in index order, weight their outputs, then accumulate.
    order.sort_unstable();
    let mut output = vec![0.0; hidden_values.len()];
    let mut up = vec![0.0; intermediate];
    let mut down = vec![0.0; width as usize];
    for index in order {
        for (token, row) in selection.iter().enumerate() {
            let Some(hit) = row.iter().find(|e| e.index == index) else {
                continue;
            };
            let start = token * width as usize;
            let input = &hidden_values[start..start + width as usize];
            project(
                projector,
                ExpertIndex::Routed(index),
                ExpertProjection::Up,
                input,
                &mut up,
            )?;
            // The decoded path rejects a non-finite projection before ReLU;
            // f32::max would silently turn NaN into zero here.
            if up.iter().any(|value| !value.is_finite()) {
                return Err(invalid("expert up projection is not finite"));
            }
            // Reference lines 634, 658: relu2 is ReLU(x)^2.
            for value in &mut up {
                *value = value.max(0.0).powi(2);
            }
            project(
                projector,
                ExpertIndex::Routed(index),
                ExpertProjection::Down,
                &up,
                &mut down,
            )?;
            for (dest, value) in output[start..start + width as usize].iter_mut().zip(&down) {
                *dest += value * hit.weight;
            }
        }
    }
    tensor(&[tokens, width], output)
}

/// Phase-batched routed experts: all up projections in ONE device batch,
/// host relu2, all down projections in ONE batch, then weight and accumulate
/// in exactly the sequential loop's pair order, so the result is bit for bit
/// `apply_routed_experts_packed` when the batch matches single calls.
pub(crate) fn apply_routed_experts_packed_phased<P: Projector<Weight = Tensor>>(
    hidden: &Tensor,
    selection: &[Vec<RoutedExpert>],
    intermediate: usize,
    projector: &mut P,
    project_batch: &mut BatchedPackedProjection<'_, P>,
) -> Result<Tensor, EngError> {
    let (tokens, width) = matrix_shape(hidden)?;
    let hidden_values = read_f32(hidden, &[tokens, width])?;
    if selection.len() != tokens as usize || selection.iter().any(Vec::is_empty) {
        return Err(invalid("selection must contain one nonempty row per token"));
    }
    if intermediate == 0 || width == 0 {
        return Err(invalid("packed expert geometry must be positive"));
    }
    let mut order: Vec<u32> = Vec::new();
    for row in selection {
        for (position, e) in row.iter().enumerate() {
            let repeated = row[..position].iter().any(|prior| prior.index == e.index);
            if !e.weight.is_finite() || e.weight < 0.0 || repeated {
                return Err(invalid("invalid selection"));
            }
            if !order.contains(&e.index) {
                order.push(e.index);
            }
        }
    }
    order.sort_unstable();
    // Pair list in the sequential loop's exact order: expert (sorted), then
    // token order; each pair carries its selection weight.
    let mut pairs: Vec<(u32, usize, f32)> = Vec::new();
    for &index in &order {
        for (token, row) in selection.iter().enumerate() {
            if let Some(hit) = row.iter().find(|e| e.index == index) {
                pairs.push((index, token, hit.weight));
            }
        }
    }
    // Phase 1: every up projection in one batch.
    let up_requests: Vec<(ExpertIndex, ExpertProjection)> = pairs
        .iter()
        .map(|&(index, _, _)| (ExpertIndex::Routed(index), ExpertProjection::Up))
        .collect();
    let mut up_inputs = Vec::with_capacity(pairs.len() * width as usize);
    for &(_, token, _) in &pairs {
        let start = token * width as usize;
        up_inputs.extend_from_slice(&hidden_values[start..start + width as usize]);
    }
    let mut up_outputs = vec![0.0; pairs.len() * intermediate];
    project_batch(projector, &up_requests, &up_inputs, &mut up_outputs)?;
    // relu2 per pair, with the sequential loop's finiteness rejection.
    for pair in 0..pairs.len() {
        let row = &mut up_outputs[pair * intermediate..(pair + 1) * intermediate];
        if row.iter().any(|value| !value.is_finite()) {
            return Err(invalid("expert up projection is not finite"));
        }
        for value in row.iter_mut() {
            *value = value.max(0.0).powi(2);
        }
    }
    // Phase 2: every down projection in one batch.
    let down_requests: Vec<(ExpertIndex, ExpertProjection)> = pairs
        .iter()
        .map(|&(index, _, _)| (ExpertIndex::Routed(index), ExpertProjection::Down))
        .collect();
    let mut down_outputs = vec![0.0; pairs.len() * width as usize];
    project_batch(projector, &down_requests, &up_outputs, &mut down_outputs)?;
    // Weight and accumulate in the sequential loop's pair order.
    let mut output = vec![0.0; hidden_values.len()];
    for (pair, &(_, token, weight)) in pairs.iter().enumerate() {
        let start = token * width as usize;
        let down = &down_outputs[pair * width as usize..(pair + 1) * width as usize];
        for (dest, value) in output[start..start + width as usize].iter_mut().zip(down) {
            *dest += value * weight;
        }
    }
    tensor(&[tokens, width], output)
}

#[allow(clippy::too_many_arguments)]
/// Same block math as the projector variant, with packed expert projections.
pub(crate) fn moe_apply_selected_packed<P: Projector<Weight = Tensor>>(
    hidden: &Tensor,
    normalized_hidden: &Tensor,
    selection: &[Vec<RoutedExpert>],
    latent_up: &Tensor,
    latent_down: &Tensor,
    shared: MlpWeights,
    intermediate: usize,
    projector: &mut P,
    project: &mut PackedProjection<'_, P>,
    project_batch: Option<&mut BatchedPackedProjection<'_, P>>,
    precomputed_shared: Option<&[f32]>,
    latent_up_name: Option<&str>,
    latent_down_name: Option<&str>,
    shared_up_name: Option<&str>,
    shared_down_name: Option<&str>,
) -> Result<Tensor, EngError> {
    let (tokens, width) = matrix_shape(hidden)?;
    let hidden_values = read_f32(hidden, &[tokens, width])?;
    let normalized = read_f32(normalized_hidden, &[tokens, width])?;
    // Name-hinted latent: shape from the resident entry; skip validation.
    let (latent, _) = match latent_up_name.and_then(|n| projector.resident_shape(n)) {
        Some([rows, _]) => (rows as u32, rows),
        None => {
            let (l, w) = matrix_shape(latent_up)?;
            if latent_up_name.is_none() {
                validated_f32_bytes(latent_up, &[l, w])?;
            }
            (l, w as usize)
        }
    };
    if latent_down_name.is_none() {
        validated_f32_bytes(latent_down, &[width, latent])?;
    }
    let latent_hidden = tensor(
        &[tokens, latent],
        project_rows(
            projector,
            latent_up,
            &normalized,
            width as usize,
            latent as usize,
            "nemotron35_moe",
            "latent up",
            None,
        )?,
    )?;
    let routed = match project_batch {
        Some(batch) => apply_routed_experts_packed_phased(
            &latent_hidden,
            selection,
            intermediate,
            projector,
            batch,
        )?,
        None => apply_routed_experts_packed(
            &latent_hidden,
            selection,
            intermediate,
            projector,
            project,
        )?,
    };
    let routed = read_f32(&routed, &[tokens, latent])?;
    let mut output = project_rows(
        projector,
        latent_down,
        &routed,
        latent as usize,
        width as usize,
        "nemotron35_moe",
        "latent down",
        latent_down_name,
    )?;
    let shared_output: Vec<f32> = match precomputed_shared {
        Some(values) if values.len() == (tokens as usize) * (width as usize) => values.to_vec(),
        _ => mlp(
            projector,
            &normalized,
            width,
            shared,
            shared_up_name,
            shared_down_name,
        )?,
    };
    for ((value, shared), residual) in output.iter_mut().zip(shared_output).zip(hidden_values) {
        *value += shared;
        *value += residual;
    }
    tensor(&[tokens, width], output)
}

/// One decode row of the MoE block with the routed experts supplied as a closure.
///
/// `routed` takes the latent row (the `fc1_latent_proj` output) and the row's selection and returns
/// the weighted expert sum in latent space, which is what `apply_routed_experts_packed` computes;
/// the CPU kernel fuses both expert projections, so it cannot go through the per-projection
/// callbacks of `moe_apply_selected_packed`. The closure runs on its own thread while the caller
/// computes the shared expert through the projector, so the device work hides the CPU call. The
/// arithmetic order after the join is the order of `moe_apply_selected_packed`: latent down, then
/// the shared output, then the residual.
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_apply_selected_routed<P, R>(
    hidden: &Tensor,
    normalized_hidden: &Tensor,
    selection: &[Vec<RoutedExpert>],
    latent_up: &Tensor,
    latent_down: &Tensor,
    shared: MlpWeights,
    projector: &mut P,
    routed: R,
    precomputed_shared: Option<Vec<f32>>,
) -> Result<(Tensor, Overlap), EngError>
where
    P: Projector<Weight = Tensor>,
    R: FnOnce(&[f32], &[RoutedExpert]) -> Result<Vec<f32>, EngError> + Send,
{
    let (tokens, width) = matrix_shape(hidden)?;
    if tokens != 1 || selection.len() != 1 || selection[0].is_empty() {
        return Err(invalid("routed-closure MoE decodes exactly one row"));
    }
    let hidden_values = read_f32(hidden, &[tokens, width])?;
    let normalized = read_f32(normalized_hidden, &[tokens, width])?;
    let (latent, _) = matrix_shape(latent_up)?;
    // validated_f32_bytes validates on first use and skips on subsequent
    // calls (thread-local pointer set): no per-token 200 MiB re-scan.
    validated_f32_bytes(latent_up, &[latent, width])?;
    validated_f32_bytes(latent_down, &[width, latent])?;
    for (position, e) in selection[0].iter().enumerate() {
        let repeated = selection[0][..position]
            .iter()
            .any(|prior| prior.index == e.index);
        if !e.weight.is_finite() || e.weight < 0.0 || repeated {
            return Err(invalid("invalid selection"));
        }
    }
    let latent_row = project_rows(
        projector,
        latent_up,
        &normalized,
        width as usize,
        latent as usize,
        "nemotron35_moe",
        "latent up",
        None,
    )?;
    let row = &selection[0];
    let (routed_values, shared_output, timing) = run_overlapped(
        || routed(&latent_row, row),
        || {
            if let Some(pre) = precomputed_shared {
                Ok(pre)
            } else {
                mlp(projector, &normalized, width, shared, None, None)
            }
        },
    )?;
    let routed_values = routed_values?;
    if routed_values.len() != latent as usize {
        return Err(invalid("routed experts returned the wrong row width"));
    }
    let shared_output = shared_output?;
    let mut output = project_rows(
        projector,
        latent_down,
        &routed_values,
        latent as usize,
        width as usize,
        "nemotron35_moe",
        "latent down",
        None,
    )?;
    for ((value, shared), residual) in output.iter_mut().zip(shared_output).zip(hidden_values) {
        *value += shared;
        *value += residual;
    }
    Ok((tensor(&[tokens, width], output)?, timing))
}

/// The chunked twin of [`moe_apply_selected_routed`]: `rows` tokens through one
/// MoE layer, routed contributions supplied by a batched callback. The CPU
/// pool uses ascending expert order; the opt-in device callback uses selection
/// order and approximate F16-operand projections. Latent projections retain
/// the sequential per-row `project_rows` calls.
#[allow(clippy::too_many_arguments, reason = "mirrors the single-row twin")]
pub(crate) fn moe_apply_selected_routed_batch<P, R>(
    hidden: &Tensor,
    normalized_hidden: &Tensor,
    selection: &[Vec<RoutedExpert>],
    latent_up: &Tensor,
    latent_down: &Tensor,
    shared: MlpWeights<'_>,
    projector: &mut P,
    routed_batch: R,
) -> Result<Tensor, EngError>
where
    P: Projector<Weight = Tensor>,
    // FnOnce only: the glue runs inline, and the device arm's closure
    // borrows the single-threaded CUDA context.
    R: FnOnce(&mut P, &[f32], &[&[RoutedExpert]]) -> Result<Vec<f32>, EngError>,
{
    let (tokens, width) = matrix_shape(hidden)?;
    let tokens = tokens as usize;
    if tokens == 0 || selection.len() != tokens || tokens > u32::MAX as usize {
        return Err(invalid("batched routed MoE needs one selection per row"));
    }
    let row_tokens = u32::try_from(tokens).map_err(|_| invalid("row count exceeds u32"))?;
    let hidden_values = read_f32(hidden, &[row_tokens, width])?;
    let normalized = read_f32(normalized_hidden, &[row_tokens, width])?;
    let (latent, _) = matrix_shape(latent_up)?;
    validated_f32_bytes(latent_up, &[latent, width])?;
    validated_f32_bytes(latent_down, &[width, latent])?;
    for row in selection {
        if row.is_empty() {
            return Err(invalid("invalid selection"));
        }
        for (position, e) in row.iter().enumerate() {
            let repeated = row[..position].iter().any(|prior| prior.index == e.index);
            if !e.weight.is_finite() || e.weight < 0.0 || repeated {
                return Err(invalid("invalid selection"));
            }
        }
    }
    let latent_rows = project_rows(
        projector,
        latent_up,
        &normalized,
        width as usize,
        latent as usize,
        "nemotron35_moe",
        "latent up",
        None,
    )?;
    let rows: Vec<&[RoutedExpert]> = selection.iter().map(|row| row.as_slice()).collect();
    let routed_values = routed_batch(projector, &latent_rows, &rows)?;
    if routed_values.len() != tokens * latent as usize {
        return Err(invalid("routed experts returned the wrong batch width"));
    }
    // Shared expert per row through the same per-row mlp the sequential
    // path runs; the residual add and the latent-down projection are per
    // row as well.
    let mut output = Vec::with_capacity(hidden_values.len());
    for (index, row_hidden) in normalized.chunks_exact(width as usize).enumerate() {
        let shared_row = mlp(projector, row_hidden, width, shared, None, None)?;
        let routed_row = &routed_values[index * latent as usize..(index + 1) * latent as usize];
        let mut down = project_rows(
            projector,
            latent_down,
            routed_row,
            latent as usize,
            width as usize,
            "nemotron35_moe",
            "latent down",
            None,
        )?;
        for ((value, shared_value), residual) in down
            .iter_mut()
            .zip(shared_row)
            .zip(&hidden_values[index * width as usize..(index + 1) * width as usize])
        {
            *value += shared_value;
            *value += residual;
        }
        output.extend_from_slice(&down);
    }
    tensor(&[row_tokens, width], output)
}

#[cfg(test)]
mod projector_tests {
    use super::{ExpertWeights, MlpWeights, MoeConfig, moe_block_forward_with_projector};
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

    struct Inputs {
        norm: Tensor,
        hidden: Tensor,
        router: Tensor,
        bias: Tensor,
        latent_up: Tensor,
        latent_down: Tensor,
        expert_up: [Tensor; 2],
        expert_down: [Tensor; 2],
        shared_up: Tensor,
        shared_down: Tensor,
        config: MoeConfig,
    }

    impl Inputs {
        fn new() -> Self {
            let expert_up = [tensor(&[1, 1], &[2.0]), tensor(&[1, 1], &[-1.0])];
            let expert_down = [tensor(&[1, 1], &[3.0]), tensor(&[1, 1], &[4.0])];
            Self {
                norm: tensor(&[2], &[1.0, 1.0]),
                hidden: tensor(&[2, 2], &[1.0, 2.0, 3.0, 1.0]),
                router: tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]),
                bias: tensor(&[2], &[0.0, 0.0]),
                latent_up: tensor(&[1, 2], &[1.0, 1.0]),
                latent_down: tensor(&[2, 1], &[1.0, -1.0]),
                expert_up,
                expert_down,
                shared_up: tensor(&[1, 2], &[1.0, 0.5]),
                shared_down: tensor(&[2, 1], &[2.0, -1.0]),
                config: MoeConfig {
                    n_group: 1,
                    topk_group: 1,
                    experts_per_token: 1,
                    norm_topk_prob: true,
                    routed_scaling_factor: 1.0,
                    eps: 1.0,
                },
            }
        }

        fn snapshots(&self) -> Vec<Vec<u8>> {
            [
                &self.norm,
                &self.hidden,
                &self.router,
                &self.bias,
                &self.latent_up,
                &self.latent_down,
                &self.expert_up[0],
                &self.expert_down[0],
                &self.expert_up[1],
                &self.expert_down[1],
                &self.shared_up,
                &self.shared_down,
            ]
            .map(|tensor| tensor.host_bytes().unwrap().to_vec())
            .to_vec()
        }

        fn weights(&self) -> Vec<&Tensor> {
            vec![
                &self.norm,
                &self.hidden,
                &self.router,
                &self.bias,
                &self.latent_up,
                &self.latent_down,
                &self.expert_up[0],
                &self.expert_down[0],
                &self.expert_up[1],
                &self.expert_down[1],
                &self.shared_up,
                &self.shared_down,
            ]
        }

        fn run(&self, projector: &mut RecordingProjector) -> Result<Tensor, EngError> {
            let experts = [
                ExpertWeights {
                    index: 0,
                    weights: MlpWeights {
                        up: &self.expert_up[0],
                        down: &self.expert_down[0],
                    },
                },
                ExpertWeights {
                    index: 1,
                    weights: MlpWeights {
                        up: &self.expert_up[1],
                        down: &self.expert_down[1],
                    },
                },
            ];
            moe_block_forward_with_projector(
                &self.norm,
                &self.hidden,
                &self.router,
                &self.bias,
                &self.latent_up,
                &self.latent_down,
                &experts,
                MlpWeights {
                    up: &self.shared_up,
                    down: &self.shared_down,
                },
                self.config,
                projector,
            )
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

    #[test]
    fn shared_moe_projects_rows_in_router_and_expert_order() {
        let inputs = Inputs::new();
        let snapshots = inputs.snapshots();
        let mut projector = RecordingProjector::new(Behavior::Multiply);

        inputs.run(&mut projector).unwrap();

        assert_eq!(projector.calls.len(), 14);
        let expected_weights = [
            &inputs.router,
            &inputs.router,
            &inputs.latent_up,
            &inputs.latent_up,
            &inputs.expert_up[0],
            &inputs.expert_down[0],
            &inputs.expert_up[1],
            &inputs.expert_down[1],
            &inputs.latent_down,
            &inputs.latent_down,
            &inputs.shared_up,
            &inputs.shared_up,
            &inputs.shared_down,
            &inputs.shared_down,
        ];
        for (call, weight) in projector.calls.iter().zip(expected_weights) {
            assert_eq!(call.shape, weight.shape());
            assert_eq!(call.weight, weight.host_bytes().unwrap());
        }
        assert_ne!(projector.calls[0].input, projector.calls[1].input);
        assert_eq!(projector.calls[2].input, projector.calls[0].input);
        assert_eq!(projector.calls[3].input, projector.calls[1].input);
        // The router selects expert 0 for token 1 and expert 1 for token 0.
        assert_eq!(
            projector.calls[4].input,
            [projector.calls[3].input.iter().sum::<f32>()]
        );
        assert_eq!(
            projector.calls[6].input,
            [projector.calls[2].input.iter().sum::<f32>()]
        );
        for (before, weight) in snapshots.iter().zip(inputs.weights()) {
            assert_eq!(weight.host_bytes().unwrap(), before);
        }
    }

    #[test]
    fn shared_moe_propagates_errors_without_mutating_inputs() {
        let inputs = Inputs::new();
        let snapshots = inputs.snapshots();

        for failed_call in [1, 5, 14] {
            let mut projector = RecordingProjector::new(Behavior::FailAt(failed_call));
            let error = inputs.run(&mut projector).unwrap_err();
            assert_eq!(
                error,
                EngError::BadRequest(format!("test projector failure {failed_call}"))
            );
            assert_eq!(projector.calls.len(), failed_call);
            for (before, weight) in snapshots.iter().zip(inputs.weights()) {
                assert_eq!(weight.host_bytes().unwrap(), before);
            }
        }
    }

    #[test]
    fn shared_moe_rejects_malformed_projection_lengths() {
        let inputs = Inputs::new();
        for malformed_call in [1, 5, 14] {
            let mut projector = RecordingProjector::new(Behavior::ReturnShortAt(malformed_call));
            assert!(matches!(
                inputs.run(&mut projector),
                Err(EngError::Backend { .. })
            ));
            assert_eq!(projector.calls.len(), malformed_call);
        }
    }
}
