//! Parity of the device-resident pass against the flag-off path, in two processes: one writes
//! the reference (flag off), the other compares (flag on).
//!
//! ```text
//! FIFI_INKLING_PASS_PARITY_OUT=ref.json cargo test ... resident_pass_parity -- --ignored
//! FIFI_INKLING_CUDA_RESIDENT_PREFILL_PASS=1 FIFI_INKLING_PASS_PARITY_REF=ref.json cargo test ...
//! ```
use super::*;
use std::{env, fs, io::Write, path::Path};

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use super::super::{MessageBank, TapRow, rms_norm_with_scale};

/// Decode steps whose logits the reference keeps, besides the last step.
const LOGIT_STEPS: usize = 3;

fn fingerprint(
    model: &mut CpuForward,
    tokens: &[u32],
    prefix_rows: usize,
    one_row_as_token: bool,
) -> serde_json::Value {
    let start = std::time::Instant::now();
    if one_row_as_token {
        // Flag off refuses a one-row prefill (one token takes the token path), so that path is
        // the reference of a one-row block. The logits of this call are not a step's.
        model.token(tokens[0], true).unwrap();
    } else {
        model.prefill_grouped(&tokens[..prefix_rows]).unwrap();
    }
    let prefill_us = start.elapsed().as_micros() as u64;
    let visits: Vec<String> = (0..model.state.len())
        .map(|slot| format!("{:016x}", model.state[slot].parity_fingerprint()))
        .collect();
    // Per-part fingerprints of every visit, and the rows of the visits a dump names (default: the
    // first visit and the first banked one), so a differing run shows which part differs first.
    let (mut parts, mut detail) = (Vec::new(), serde_json::Map::new());
    for slot in 0..model.state.len() {
        let (slot_parts, slot_detail) = model.state[slot].parity_parts();
        parts.push(serde_json::json!(slot_parts));
        if [0, 18].contains(&slot) {
            detail.insert(slot.to_string(), slot_detail);
        }
    }
    let mut parts = serde_json::json!({"parts": parts, "detail": detail});
    if let Ok(slots) = env::var("FIFI_INKLING_PASS_DUMP_SLOTS") {
        // Every row of the named visits (a per-row comparison; large).
        let rows: Vec<serde_json::Value> = slots
            .split(',')
            .map(|slot| model.state[slot.trim().parse::<usize>().unwrap()].parity_rows())
            .collect();
        parts["rows"] = serde_json::json!(rows);
    }
    // Every decode step is teacher-forced; the logits of the first steps and of the last step are
    // kept (the full vocabulary), so the reference stays small for any block size.
    let steps = tokens.len() - prefix_rows;
    let mut logits = Vec::new();
    let mut argmax: Vec<u32> = Vec::with_capacity(steps);
    for &token in &tokens[prefix_rows..] {
        let full = model.token(token, true).unwrap().unwrap();
        argmax.push(
            full.iter()
                .enumerate()
                .max_by(|(a, x), (b, y)| x.total_cmp(y).then(b.cmp(a)))
                .map_or(0, |(a, _)| a as u32),
        );
        logits.extend(full);
    }
    let vocab = logits.len() / steps;
    // `FIFI_INKLING_PASS_PARITY_DUMP_LOGITS=all` also leaves every step's logits as raw f32
    // beside the value file, for the per-step analysis of an arm's deviation.
    if let Some(path) = env::var_os("FIFI_INKLING_PASS_PARITY_DUMP")
        && env::var("FIFI_INKLING_PASS_PARITY_DUMP_LOGITS").as_deref() == Ok("all")
    {
        let path = Path::new(&path);
        if let Ok(mut sink) = std::fs::File::create(path.with_extension("logits.f32")) {
            for step in 0..steps {
                sink.write_all(&bytes(&logits[step * vocab..(step + 1) * vocab]))
                    .unwrap();
            }
        }
    }
    let kept: Vec<u32> = (0..steps)
        .filter(|&step| step < LOGIT_STEPS || step + 1 == steps)
        .flat_map(|step| {
            logits[step * vocab..(step + 1) * vocab]
                .iter()
                .map(|x| x.to_bits())
        })
        .collect();
    serde_json::json!({
        "rows": prefix_rows,
        "prefill_us": prefill_us,
        "visits": visits,
        "logits": kept,
        "argmax": argmax,
        "steps": steps,
        "parts": parts,
    })
}

#[test]
#[ignore = "requires a leased CUDA device, the Inkling GGUF and fixed prompt token IDs"]
fn resident_pass_parity() {
    let checkpoint = env::var_os("FIFI_INKLING_MODEL_DIR").expect("set the Inkling GGUF path");
    let tokens: Vec<u32> = env::var("FIFI_INKLING_TOKEN_IDS")
        .expect("set comma-separated token IDs")
        .split(',')
        .map(|value| value.parse().expect("valid token ID"))
        .collect();
    let rows: usize = env::var("FIFI_INKLING_PASS_ROWS")
        .map(|value| value.parse().expect("rows"))
        .unwrap_or(64);
    assert!((1..=512).contains(&rows) && rows < tokens.len());
    let reference = env::var_os("FIFI_INKLING_PASS_PARITY_REF");
    let output = env::var_os("FIFI_INKLING_PASS_PARITY_OUT");
    assert_ne!(
        reference.is_none(),
        output.is_none(),
        "set exactly one of REF or OUT"
    );
    let mut model = CpuForward::open(Path::new(&checkpoint), tokens.len()).unwrap();
    model.enable_cuda(0).unwrap();
    let value = fingerprint(&mut model, &tokens, rows, rows == 1 && output.is_some());
    // The host-side dumps belong to the run that writes the reference; a comparing arm always
    // goes on to the comparison.
    if let Some(out) = env::var_os("FIFI_INKLING_PASS_HOST_VISIT0_OUT").filter(|_| output.is_some())
    {
        // The host's attention input, raw K/V and prepared rows of the first visit for every
        // prompt row (`FIFI_INKLING_PASS_DUMP_VISIT=0:path` carries the pass's side).
        let visit = model.visits[0];
        let layer = format!("model.llm.layers.{}", visit.weight_layer);
        let mut state = VisitState::default();
        let (mut input, mut raw_k, mut raw_v) = (Vec::new(), Vec::new(), Vec::new());
        for &id in &tokens[..rows] {
            let stream = model.embed(id).unwrap();
            let inputs = model
                .attention_inputs(visit, &layer, &stream, None)
                .unwrap();
            input.extend(inputs[0].iter().map(|x| x.to_bits()));
            let projected = model
                .project_attention(&layer, visit.attention, &inputs)
                .unwrap();
            raw_k.extend(projected[1].iter().map(|x| x.to_bits()));
            raw_v.extend(projected[2].iter().map(|x| x.to_bits()));
            model
                .attention(&layer, visit, &mut state, projected)
                .unwrap();
        }
        // The host-only replays below cost minutes at 512 rows; `FIFI_INKLING_PASS_HOST_REPLAY=0`
        // keeps just the first visit's rows.
        if std::env::var("FIFI_INKLING_PASS_HOST_REPLAY").as_deref() == Ok("0") {
            fs::write(
                out,
                serde_json::to_vec(&serde_json::json!({
                    "input": input, "raw_k": raw_k, "raw_v": raw_v, "state": state.parity_rows(),
                    "h": model.hidden(),
                }))
                .unwrap(),
            )
            .unwrap();
            if let Some(out) = &output {
                fs::write(out, serde_json::to_vec(&value).unwrap()).unwrap();
            }
            return;
        }
        // A second model without CUDA replays the first two layers of every prompt row on the
        // host and keeps the stream after each, with the second layer's attention input.
        let host = CpuForward::open(Path::new(&checkpoint), rows).unwrap();
        let mut host_state: Vec<VisitState> = (0..host.state.len())
            .map(|_| VisitState::default())
            .collect();
        let (mut after, mut inputs) = (Vec::new(), Vec::new());
        // The second layer's raw K/V and prepared rows, from a fresh visit state.
        let mut raw = Vec::new();
        let mut second = VisitState::default();
        for &id in &tokens[..rows] {
            let mut stream = host.embed(id).unwrap();
            let mut taps = TapRow::new(&[], host.hidden());
            for (n, visit) in host.visits[..2].iter().enumerate() {
                let layer = format!("model.llm.layers.{}", visit.weight_layer);
                if n == 1 {
                    let inputs1 = host
                        .attention_inputs(*visit, &layer, &stream, None)
                        .unwrap();
                    inputs.extend(inputs1[0].iter().map(|x| x.to_bits()));
                    let projected = host
                        .project_attention(&layer, visit.attention, &inputs1)
                        .unwrap();
                    raw.extend(projected[1].iter().map(|x| x.to_bits()));
                    raw.extend(projected[2].iter().map(|x| x.to_bits()));
                    host.attention(&layer, *visit, &mut second, projected)
                        .unwrap();
                }
                host.run_visits(
                    &host.visits[n..n + 1],
                    &mut host_state,
                    &mut stream,
                    None,
                    &mut taps,
                    false,
                )
                .unwrap();
                after.extend(stream.current().iter().map(|x| x.to_bits()));
            }
        }
        // The first banked visit (the second pass over the core): its live stream, three
        // attention inputs and raw K/V on the host, with the bank `forward` builds for that pass.
        let core = &host.spec.core;
        let width = core.core_end - core.core_start;
        let banked = core.core_start + width;
        let bank_visit = host.visits[banked];
        let bank_layer = format!("model.llm.layers.{}", bank_visit.weight_layer);
        let mut bank_state: Vec<VisitState> = (0..host.state.len())
            .map(|_| VisitState::default())
            .collect();
        let (mut before18, mut inputs18, mut raw18) = (Vec::new(), Vec::new(), Vec::new());
        let (mut control18, mut messages18, mut gains18, mut forgets18) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let rank = host.spec.core.gate_rank as usize;
        let mut bank_rows = Vec::new();
        for &id in &tokens[..rows] {
            let mut stream = host.embed(id).unwrap();
            let mut taps = TapRow::new(&[], host.hidden());
            host.run_visits(
                &host.visits[..core.core_start],
                &mut bank_state,
                &mut stream,
                None,
                &mut taps,
                false,
            )
            .unwrap();
            let prelude = stream.clone();
            host.run_visits(
                &host.visits[core.core_start..banked],
                &mut bank_state,
                &mut stream,
                None,
                &mut taps,
                false,
            )
            .unwrap();
            let bank = MessageBank {
                first_block: prelude.block_count() + 1,
                stream: std::mem::replace(&mut stream, prelude.clone()),
            };
            before18.extend(stream.current().iter().map(|x| x.to_bits()));
            let mut bank_bytes = Vec::new();
            bank.stream.append_device_state(&mut bank_bytes);
            bank_rows.extend(
                bank_bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b)),
            );
            let inputs = host
                .attention_inputs(bank_visit, &bank_layer, &stream, Some(&bank))
                .unwrap();
            inputs18.extend(inputs.iter().flat_map(|r| r.iter().map(|x| x.to_bits())));
            // The pieces of the controlled input, per role: the residual mix before control, the
            // bank message, the controller's gain and forget rows.
            for (role, name) in [Role::Query, Role::Key, Role::Value]
                .into_iter()
                .enumerate()
            {
                let mut mixed = host.attention_input(bank_visit, &stream, role).unwrap();
                let message = bank.stream.mix(
                    bank.first_block,
                    &host.control_query(bank_visit.weight_layer, name).unwrap(),
                );
                rms_norm_with_scale(&mut mixed, MIX_EPS);
                let mut right = message.clone();
                rms_norm_with_scale(&mut right, MIX_EPS);
                let mut gate = mixed;
                gate.extend_from_slice(&right);
                let [gain, forget] = host
                    .weights
                    .controller(
                        control_names(&bank_layer, name)
                            .each_ref()
                            .map(String::as_str),
                        &gate,
                        [rank, host.hidden()],
                    )
                    .unwrap();
                for (to, row) in [
                    &mut control18,
                    &mut messages18,
                    &mut gains18,
                    &mut forgets18,
                ]
                .into_iter()
                .zip([&gate[..host.hidden()], &message, &gain, &forget])
                {
                    to.extend(row.iter().map(|x| x.to_bits()));
                }
            }
            let projected = host
                .project_attention(&bank_layer, bank_visit.attention, &inputs)
                .unwrap();
            raw18.extend(projected[1].iter().map(|x| x.to_bits()));
            raw18.extend(projected[2].iter().map(|x| x.to_bits()));
        }
        fs::write(
            out,
            serde_json::to_vec(&serde_json::json!({
                "input": input, "raw_k": raw_k, "raw_v": raw_v, "state": state.parity_rows(),
                "h": host.hidden(), "input1": inputs,
                "raw1": raw, "second": second.parity_rows(),
                "after": after,
                "before18": before18, "inputs18": inputs18, "raw18": raw18,
                "normed18": control18, "messages18": messages18, "gains18": gains18, "forgets18": forgets18,
                // The bank stream as the host keeps it, per token: the current row, then the
                // normalized sealed rows, oldest first.
                "bank18": bank_rows,
            }))
            .unwrap(),
        )
        .unwrap();
    }
    if let Some(out) = output {
        fs::write(out, serde_json::to_vec(&value).unwrap()).unwrap();
        eprintln!("Inkling resident pass reference written ({rows} rows)");
        return;
    }
    let expected: serde_json::Value =
        serde_json::from_slice(&fs::read(reference.unwrap()).unwrap()).unwrap();
    let visits = value["visits"].as_array().unwrap();
    let expected_visits = expected["visits"].as_array().unwrap();
    let unequal_visits = visits
        .iter()
        .zip(expected_visits)
        .filter(|(a, b)| a != b)
        .count();
    let logits: Vec<u32> = serde_json::from_value(value["logits"].clone()).unwrap();
    let expected_logits: Vec<u32> = serde_json::from_value(expected["logits"].clone()).unwrap();
    let max_abs = logits
        .iter()
        .zip(&expected_logits)
        .map(|(a, b)| (f32::from_bits(*a) - f32::from_bits(*b)).abs())
        .fold(0.0, f32::max);
    let argmax: Vec<u32> = serde_json::from_value(value["argmax"].clone()).unwrap();
    let expected_argmax: Vec<u32> = serde_json::from_value(expected["argmax"].clone()).unwrap();
    let argmax_mismatches = argmax
        .iter()
        .zip(&expected_argmax)
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "Inkling resident pass parity {}",
        serde_json::json!({"rows": rows, "unequal_visits": unequal_visits, "visits": visits.len(),
            "logits_max_abs": max_abs, "argmax_mismatches": argmax_mismatches, "steps": argmax.len(),
            "prefill_us_pass": value["prefill_us"], "prefill_us_host": expected["prefill_us"]})
    );
    // `FIFI_INKLING_PASS_PARITY_DUMP=<path>`: this arm's fingerprints and logits, for per-step
    // comparisons outside the test.
    if let Some(out) = env::var_os("FIFI_INKLING_PASS_PARITY_DUMP") {
        let mut copy = value.clone();
        copy.as_object_mut().map(|object| object.remove("parts"));
        fs::write(out, serde_json::to_vec(&copy).unwrap()).unwrap();
    }
    assert_eq!(visits.len(), expected_visits.len());
    if unequal_visits != 0
        && let Some(out) = env::var_os("FIFI_INKLING_PASS_PARTS_OUT")
    {
        {
            fs::write(
                out,
                serde_json::to_vec(
                    &serde_json::json!({"pass": value["parts"], "host": expected["parts"]}),
                )
                .unwrap(),
            )
            .unwrap();
        }
    }
    assert_eq!(
        unequal_visits, 0,
        "visit state differs from the flag-off path"
    );
    assert!(
        max_abs < 1e-2,
        "logits differ from the flag-off path by {max_abs}"
    );
    assert_eq!(
        argmax, expected_argmax,
        "a greedy decode step picks another token"
    );
}

/// Every constant the pass plans for every visit resolves in the checkpoint (host only: no CUDA).
#[test]
#[ignore = "requires the Inkling GGUF"]
fn pass_constant_names() {
    let checkpoint = env::var_os("FIFI_INKLING_MODEL_DIR").expect("set the Inkling GGUF path");
    let model = CpuForward::open(Path::new(&checkpoint), 512 + 64).unwrap();
    let mut consts = Consts::default();
    let mut geometries: [Option<Geometry>; 2] = [None, None];
    for visit in &model.visits {
        let heads = model.pass_heads(visit.attention);
        let kind = usize::from(visit.attention == Attention::Global);
        let geometry = *geometries[kind].get_or_insert(Geometry {
            query: heads[0],
            kv: heads[1],
            channels: heads[0] * HEAD_DIM,
            kv_channels: heads[1] * HEAD_DIM,
            relative: heads[0] * model.math.d_rel,
        });
        CpuForward::visit_consts(&model, *visit, geometry, &mut consts).unwrap_or_else(|error| {
            panic!(
                "visit {} (layer {}): {:?}",
                visit.state_slot,
                visit.weight_layer,
                error.describe()
            )
        });
    }
    eprintln!(
        "Inkling resident pass constants resolve for {} visits: {} words",
        model.visits.len(),
        consts.bytes.len()
    );
}

#[test]
fn feature_switches_need_exactly_one() {
    // The serving A/B turns the pass off with `=0`; anything but `1` must leave it off.
    let on = |value| {
        crate::family::switch_table::read("FIFI_INKLING_CUDA_RESIDENT_PREFILL_PASS", value, false)
    };
    assert!(on(Some("1".into())));
    for value in ["0", "", "true", "yes"] {
        assert!(!on(Some(value.into())), "{value:?}");
    }
    assert!(!on(None));
}
