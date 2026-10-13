//! The primitive histogram of ONE Q36 decode token: fifi against mlx-lm (packet GPU-GAP-COUNT).
//!
//! The decode GPU time of the engine sits 2.7 ms per token above mlx-lm at prompt depth 16
//! (Q36-GAP section 3c), and the cause is not decomposed. Every extra MLX primitive of the token
//! graph costs encode and dispatch time on the GPU, so this module counts the primitives of the
//! LAZY graph of one decode step — before it is evaluated — and the companion scripts of the
//! packet (`mlx_graph_count.py`, `compare_histograms.py`, scratch only) do the same for mlx-lm
//! and name the top three contributors to the difference.
//!
//! One ignored Mac-only real-checkpoint test, `decode_graph_primitives`, plus the portable unit
//! tests of the pure parts below it. Inputs by environment (the rule set of packet GPU-GAP-COUNT,
//! the same variables the decode-phase timing harness reads):
//!
//! | Variable | Meaning | Default |
//! | --- | --- | --- |
//! | `Q36_CKPT_DIR` | the pinned checkpoint directory | required |
//! | `IDS_FILE` | the fifi render dump with the 919 prompt ids | required |
//! | `REF_JSON` | the reference with the 3 `suffix_ids` and the token counts | required |
//! | `Q36_ACTIVATION` | the decode activation dtype | required: `bf16` or `f32`, no default |
//! | `Q36_OUT_DIR` | the output directory; must not exist, created at the start of the run | required |
//!
//! The run prefills the first 16 ids of the tool prompt (`IDS_FILE` plus the `suffix_ids` of
//! `REF_JSON`, the 922-id tool prompt of the decode-phase timing harness) through the chunked
//! prefill in `Prompt` mode, sets the test-only graph probe of the decoder, and runs ONE decode step in
//! `StepMode::Generate`: the probe dumps `MlxKernels::qualification_graph_text` — the text of
//! `mlx_print_graph` over the lazy outputs of the step — before the step's single `evaluate`.
//! The dump evaluates, synchronizes, and releases nothing, so the step's synchronization budget
//! is the budget of a plain `Generate` step. Both activation dtypes run as two invocations with
//! a fresh `Q36_OUT_DIR` each; every invocation writes:
//!
//! - `graph_fifi.txt`, the raw `mlx_print_graph` text;
//! - `primitive_histogram_fifi.json`, the histogram (name, count, sorted by count), the total
//!   node count, and the qualification call counters of the one decode step. The engine keeps no
//!   mlx-c call counter, so the report gives the next-best existing counters: the
//!   `qualification_timing` import, readback, and evaluate call counts and the
//!   `qualification_sync_count` delta of the step.
//!
//! The `mlx_print_graph` format the parser below assumes (from the MLX source at the commit the
//! vendored `mlx-c` pins; the C++ itself is not vendored, and the parser stays tolerant to
//! spacing): two header lines `Inputs: <arrays>` and `Outputs: <arrays>`, then ONE line per
//! primitive node, `<primitive> <inputs> -> <outputs>`, where every array prints as
//! `name [shape, dtype]` with a bracketed shape — so a node line always separates its primitive
//! name from the first array with the ` [` of that array's shape. Primitive names are the MLX
//! primitive class names (`Matmul`, `RMSNorm`, `GatherQMM`); a few carry spaces (`Scatter Sum`),
//! which the first-array cut handles.
//!
//! The exact Mac command (placeholders only; no personal path lives in this repository):
//!
//! ```text
//! Q36_CKPT_DIR=<CKPT_DIR> \
//! IDS_FILE=<IDS_FILE> \
//! REF_JSON=<REF_JSON> \
//! Q36_OUT_DIR=<OUT_DIR> \
//! Q36_ACTIVATION=bf16 \
//! cargo test --release --features native-mlx,native-fault-qualification --lib \
//!   family::qwen36::resident::graph_count::decode_graph_primitives -- --ignored --exact --nocapture
//! ```
//!
//! The f32 control is the same command with `Q36_ACTIVATION=f32` and a fresh `Q36_OUT_DIR`.

use super::super::settings::{Activation, parse_activation};

/// The activation the run REQUIRES from the raw `Q36_ACTIVATION` text (packet GPU-GAP-COUNT, the
/// rule of the decode-phase timing harness): the variable must name `bf16` or `f32`, the words
/// of `settings::parse_activation`, so a plain invocation cannot silently run the f32 control.
fn required_activation(raw: Option<&str>) -> Result<Activation, String> {
    match raw {
        None => Err("Q36_ACTIVATION is unset; set it to bf16 or f32".to_string()),
        Some(text) => parse_activation(text)
            .map_err(|_| format!("Q36_ACTIVATION must be bf16 or f32, got {text:?}")),
    }
}

/// The whole tool prompt: the `ids` field of the ids file plus the `suffix_ids` field of the
/// reference, checked against `prompt_tokens` and `total_tokens` (the loader rule of the
/// decode-phase timing harness). Pure, so its checks carry Linux unit tests.
fn tool_ids(
    ids_value: &serde_json::Value,
    ref_value: &serde_json::Value,
) -> Result<Vec<u32>, String> {
    let ids = id_list(ids_value, "ids").ok_or("the ids file has no ids array of numbers")?;
    let want = ref_value["prompt_tokens"]
        .as_u64()
        .and_then(|tokens| usize::try_from(tokens).ok())
        .ok_or("the reference has no prompt_tokens count")?;
    if ids.len() != want {
        return Err(format!(
            "the ids field holds {} ids, the reference names prompt_tokens {want}",
            ids.len()
        ));
    }
    let suffix = id_list(ref_value, "suffix_ids")
        .ok_or("the reference has no suffix_ids array of numbers")?;
    if suffix.len() != 3 {
        return Err(format!(
            "the tool suffix is 3 ids, the reference names {}",
            suffix.len()
        ));
    }
    let mut all = ids;
    all.extend(suffix);
    let total = ref_value["total_tokens"]
        .as_u64()
        .and_then(|tokens| usize::try_from(tokens).ok())
        .ok_or("the reference has no total_tokens count")?;
    if all.len() != total {
        return Err(format!(
            "the prompt ids plus the suffix reach {} tokens, the reference names total_tokens {total}",
            all.len()
        ));
    }
    Ok(all)
}

/// The numbers of the array field `name`, or `None` when the field is not an array of small
/// non-negative integers.
fn id_list(value: &serde_json::Value, name: &str) -> Option<Vec<u32>> {
    let items = value[name].as_array()?;
    let ids = items
        .iter()
        .map(|item| {
            let number = item.as_u64()?;
            u32::try_from(number).ok()
        })
        .collect::<Option<Vec<_>>>()?;
    if ids.is_empty() { None } else { Some(ids) }
}

/// The primitive histogram of the text of `mlx_print_graph` (packet GPU-GAP-COUNT): one entry per
/// primitive name with its node count, sorted by count (descending) and then by name, so two
/// dumps compare row by row and the heaviest primitive comes first. The count is the number of
/// node lines, one per line that names a primitive; the two header lines of the format count
/// nothing. The total node count is the sum of the counts.
pub(super) fn primitive_histogram(text: &str) -> Vec<(String, usize)> {
    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for name in text.lines().filter_map(primitive_name) {
        *counts.entry(name).or_default() += 1;
    }
    let mut histogram: Vec<(String, usize)> = counts.into_iter().collect();
    histogram.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    histogram
}

/// The primitive name of one line of the `mlx_print_graph` text, or `None` when the line names no
/// primitive (empty, or one of the two header lines). The node grammar is
/// `<primitive> <inputs> -> <outputs>`; the name is the text from the line start to the first
/// array reference, because every array prints as `name [shape, dtype]` and a shape always opens
/// a bracket, so the first ` <name> [` in the node part starts the input list. A primitive with
/// no inputs keeps the whole part before ` -> ` as its name.
fn primitive_name(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with("Inputs:") || trimmed.starts_with("Outputs:") {
        return None;
    }
    let node = trimmed.split(" -> ").next().unwrap_or(trimmed);
    let bytes = node.as_bytes();
    let mut cut = bytes.len();
    let mut from = 0;
    while let Some(offset) = bytes[from..].iter().position(|&byte| byte == b' ') {
        let space = from + offset;
        let token = &bytes[space + 1..];
        let token_end = token
            .iter()
            .position(|&byte| byte == b' ')
            .unwrap_or(token.len());
        if token_end > 0 && bytes[space + 1 + token_end..].starts_with(b" [") {
            cut = space;
            break;
        }
        from = space + 1;
    }
    let name = node[..cut].trim_end();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

/// The prompt depth the run prefills before the ONE decode step: 16 tokens, the depth of the
/// Q36-GAP decode rows this module decomposes.
#[cfg(all(
    feature = "native-mlx",
    feature = "native-fault-qualification",
    target_os = "macos",
    target_arch = "aarch64",
))]
const DEPTH: usize = 16;

/// Dumps the primitive histogram of one decode token on the real checkpoint (see the module
/// doc for the environment and the outputs). The probe reads the lazy outputs BEFORE the step's
/// `evaluate`, so the dump describes the pending graph of the token, never a computed one.
#[cfg(all(
    feature = "native-mlx",
    feature = "native-fault-qualification",
    target_os = "macos",
    target_arch = "aarch64",
))]
#[test]
#[ignore = "needs the real checkpoint on the Mac; run with --ignored"]
fn decode_graph_primitives() {
    use std::{cell::RefCell, rc::Rc};

    use super::decoder::{DEFAULT_CONTEXT_LIMIT, ResidentDecoder, StepMode};
    use super::parity::{CONFIG_SHA256, sha256};
    use super::real::{env_path, open};
    use crate::{
        family::{qwen36::host_load::HostWeights, spec::QWEN3_6_35B_A3B},
        kernels::MlxKernels,
    };

    let _serial = crate::native_test_lock();
    let ckpt_dir = env_path("Q36_CKPT_DIR");
    // The output directory must be new; it is created before the model loads, so a refused run
    // leaves an empty directory and no partial files.
    let out = std::env::var("Q36_OUT_DIR").unwrap_or_else(|_| panic!("set Q36_OUT_DIR"));
    assert!(
        std::fs::symlink_metadata(&out).is_err(),
        "Q36_OUT_DIR must not exist yet: {out}"
    );
    let out_dir = std::path::Path::new(&out).to_path_buf();
    std::fs::create_dir(&out_dir)
        .unwrap_or_else(|error| panic!("create Q36_OUT_DIR {out}: {error}"));
    let ids = {
        let (ids_path, ref_path) = (env_path("IDS_FILE"), env_path("REF_JSON"));
        let ids_value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&ids_path).unwrap()).unwrap();
        let ref_value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&ref_path).unwrap()).unwrap();
        tool_ids(&ids_value, &ref_value).unwrap_or_else(|error| panic!("{error}"))
    };
    assert!(
        ids.len() > DEPTH,
        "the tool prompt holds the {DEPTH}-token prefill and the one decode token"
    );
    let activation = match std::env::var("Q36_ACTIVATION") {
        Ok(raw) => required_activation(Some(&raw)).unwrap_or_else(|error| panic!("{error}")),
        Err(std::env::VarError::NotPresent) => {
            panic!("{}", required_activation(None).unwrap_err())
        }
        Err(error) => panic!("Q36_ACTIVATION: {error}"),
    };

    // The load of the real-checkpoint gates: the pinned config, the product constructor of the
    // `Q36_ACTIVATION` lever, and a zero state.
    let real = open(&ckpt_dir);
    assert_eq!(
        sha256(&real.config),
        CONFIG_SHA256,
        "config.json differs from the pinned checkpoint"
    );
    let host = HostWeights::load(&real.ckpt, &QWEN3_6_35B_A3B, &real.config).unwrap();
    let kv = super::real::kv_dtype_from_env();
    let mut decoder = ResidentDecoder::new_with_kv(
        MlxKernels::try_new().unwrap(),
        &real.ckpt,
        host,
        DEFAULT_CONTEXT_LIMIT,
        activation,
        kv,
    )
    .unwrap();
    let mut state = decoder.zero_state().unwrap();
    // Prefill depth 16 through the chunked prefill, as the decode rows of Q36-GAP ran it.
    decoder
        .prefill(&mut state, &ids[..DEPTH], StepMode::Prompt)
        .unwrap();
    assert_eq!(state.pos() as usize, DEPTH, "the prefill reached the depth");

    // The probe dumps the lazy outputs of the ONE decode step before its single evaluate.
    let sink: Rc<RefCell<Option<(usize, String)>>> = Rc::new(RefCell::new(None));
    let probe_sink = sink.clone();
    decoder.set_graph_probe(Box::new(move |kernels, outputs| {
        let text = kernels.qualification_graph_text(outputs)?;
        *probe_sink.borrow_mut() = Some((outputs.len(), text));
        Ok(())
    }));
    let timing_before = decoder.kernels.qualification_timing();
    let syncs_before = decoder.kernels.qualification_sync_count();
    let stepped = decoder
        .step(&mut state, ids[DEPTH], StepMode::Generate)
        .unwrap();
    let timing = decoder.kernels.qualification_timing();
    let syncs = decoder
        .kernels
        .qualification_sync_count()
        .saturating_sub(syncs_before);
    assert!(
        stepped.token.is_some(),
        "the decode step read one token back"
    );

    let (outputs_dumped, text) = Rc::try_unwrap(sink)
        .ok()
        .and_then(|cell| cell.into_inner())
        .unwrap_or_else(|| panic!("the graph probe dumped nothing; the step ran without it"));
    let histogram = primitive_histogram(&text);
    let total: usize = histogram.iter().map(|&(_, count)| count).sum();
    let rows: Vec<serde_json::Value> = histogram
        .iter()
        .map(|(name, count)| serde_json::json!({ "name": name, "count": count }))
        .collect();
    let report = serde_json::json!({
        "tool": "fifi decode_graph_primitives (mlx_print_graph)",
        "activation_dtype": super::super::settings::activation_name(activation),
        "kv_dtype": super::super::settings::kv_dtype_name(kv),
        "depth": DEPTH,
        "decode_token": ids[DEPTH],
        "outputs_dumped": outputs_dumped,
        "total_nodes": total,
        "histogram": rows,
        "calls_per_token": {
            "import": timing.import_calls.saturating_sub(timing_before.import_calls),
            "readback": timing.readback_calls.saturating_sub(timing_before.readback_calls),
            "evaluate": timing.evaluate_calls.saturating_sub(timing_before.evaluate_calls),
            "synchronize": syncs,
        },
        "calls_note": "the engine keeps no mlx-c call counter; these are the existing \
                       qualification counters of the ONE decode step: the import, readback, and \
                       evaluate call counts of qualification_timing and the synchronize count",
    });
    std::fs::write(out_dir.join("graph_fifi.txt"), &text).unwrap();
    std::fs::write(
        out_dir.join("primitive_histogram_fifi.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    println!("total nodes of the decode graph: {total}");
    println!("wrote {}", out_dir.join("graph_fifi.txt").display());
    println!(
        "wrote {}",
        out_dir.join("primitive_histogram_fifi.json").display()
    );
}

mod tests {
    use super::*;

    /// A hand-written sample of the `mlx_print_graph` format at the MLX commit the vendored
    /// `mlx-c` pins: the `Inputs:`/`Outputs:` header lines, then one
    /// `<primitive> <inputs> -> <outputs>` line per node, every array `name [shape, dtype]`.
    /// `Scatter Sum` exercises a primitive name with a space; `RandomBits` a node with no
    /// inputs (two spaces before the arrow).
    const SAMPLE: &str = "Inputs: A [[1, 2048], float32], B [[2048], float32]\n\
                          Outputs: E [[1, 8], float32]\n\
                          FastRMSNorm A [[1, 2048], float32], B [[2048], float32] -> C [[1, 2048], float32]\n\
                          Matmul C [[1, 2048], float32] -> D [[1, 512], float32]\n\
                          Matmul C [[1, 2048], float32] -> E [[1, 8], float32]\n\
                          Scatter Sum A [[4], uint32] -> F [[4], float32]\n\
                          RandomBits  -> G [[8], float32]\n";

    /// The histogram counts one node per line that names a primitive, skips the two header
    /// lines, keeps a spaced name whole, and sorts by count first and name second, so the total
    /// is the node count of the graph (5 here, against 7 lines of text).
    #[test]
    fn the_histogram_counts_one_node_per_line_that_names_a_primitive() {
        assert_eq!(
            primitive_histogram(SAMPLE),
            vec![
                ("Matmul".to_string(), 2),
                ("FastRMSNorm".to_string(), 1),
                ("RandomBits".to_string(), 1),
                ("Scatter Sum".to_string(), 1),
            ],
            "sorted by count, then name; the header lines count nothing"
        );
        let total: usize = primitive_histogram(SAMPLE)
            .iter()
            .map(|&(_, count)| count)
            .sum();
        assert_eq!(total, 5, "one node per node line");
    }

    /// Text without a node line counts no nodes: the empty dump and a dump of the two header
    /// lines alone both give an empty histogram, so a header-only parse bug cannot hide behind a
    /// non-empty text.
    #[test]
    fn header_only_text_counts_no_nodes() {
        assert!(primitive_histogram("").is_empty(), "no text, no nodes");
        assert!(
            primitive_histogram("Inputs: \nOutputs: A [[1], float32]\n").is_empty(),
            "the header lines name no primitive"
        );
    }

    /// An unseen primitive still counts: the parser names a node line by its grammar, not by a
    /// known-name list, so a new MLX primitive (here the custom Metal kernel) appears in the
    /// histogram instead of being dropped.
    #[test]
    fn an_unknown_primitive_name_still_counts() {
        let text = "CustomKernel A [[1, 128], float32] -> B [[1, 128], float32]\n";
        assert_eq!(
            primitive_histogram(text),
            vec![("CustomKernel".to_string(), 1)],
            "the parser is tolerant to names it has never seen"
        );
    }

    /// `required_activation` refuses an unset variable with the variable's name and its two
    /// values, accepts exactly the settings words, and names the bad text it refused (the rule
    /// of packet GPU-GAP-COUNT: a plain invocation cannot silently run the f32 control).
    #[test]
    fn required_activation_requires_one_of_the_two_words() {
        assert_eq!(
            required_activation(None),
            Err("Q36_ACTIVATION is unset; set it to bf16 or f32".to_string())
        );
        assert_eq!(required_activation(Some("bf16")), Ok(Activation::Bf16));
        assert_eq!(required_activation(Some("f32")), Ok(Activation::F32));
        assert_eq!(
            required_activation(Some("f64")),
            Err("Q36_ACTIVATION must be bf16 or f32, got \"f64\"".to_string())
        );
    }

    /// `tool_ids` joins the ids file with the reference suffix and checks every count against
    /// the reference (`prompt_tokens`, the 3-id suffix, `total_tokens`): the good sample passes,
    /// and each broken sample fails with the count that disagrees.
    #[test]
    fn tool_ids_checks_the_tool_prompt_fields() {
        let ids_value = serde_json::json!({ "ids": [760, 6511, 314] });
        let good = serde_json::json!({
            "prompt_tokens": 3,
            "suffix_ids": [9338, 369, 198],
            "total_tokens": 6,
        });
        assert_eq!(
            tool_ids(&ids_value, &good),
            Ok(vec![760, 6511, 314, 9338, 369, 198])
        );
        let short_prompt = serde_json::json!({
            "prompt_tokens": 4,
            "suffix_ids": [9338, 369, 198],
            "total_tokens": 7,
        });
        assert_eq!(
            tool_ids(&ids_value, &short_prompt),
            Err("the ids field holds 3 ids, the reference names prompt_tokens 4".to_string())
        );
        let two_suffix = serde_json::json!({
            "prompt_tokens": 3,
            "suffix_ids": [9338, 369],
            "total_tokens": 5,
        });
        assert_eq!(
            tool_ids(&ids_value, &two_suffix),
            Err("the tool suffix is 3 ids, the reference names 2".to_string())
        );
        let wrong_total = serde_json::json!({
            "prompt_tokens": 3,
            "suffix_ids": [9338, 369, 198],
            "total_tokens": 9,
        });
        assert_eq!(
            tool_ids(&ids_value, &wrong_total),
            Err(
                "the prompt ids plus the suffix reach 6 tokens, the reference names total_tokens 9"
                    .to_string()
            )
        );
    }
}
