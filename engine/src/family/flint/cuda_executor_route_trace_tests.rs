//! Opt-in real-tokenizer, real-executor route capture; not a latency benchmark.
use super::super::cuda_ffn::device::route_trace;
use super::super::exl3_resident::FlintFfnView;
use super::*;
use crate::tokenizer::{
    DecodeFinish, IncrementalDecoder, PromptMessage, Role, Thinking, TokenizerBundle,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::PathBuf};

const BASE: &str = "b033424b35ea43f98a07fd4137beb317cc73db42";
const INPUT_512: &str = "a2db8b850ac1b9b82a878703bead157e20ad9247ef1de357a6444bb5084725ad";
const INPUT_4000: &str = "f02982d738eb922f4fe4d286185bc61dacd36c714ca7119497db6acc1c4f456b";
const INPUT_11444: &str = "d9138724f83da85e472b09314863b1a425d07dec4e681746945110c51f2216e7";
const INPUT_16384: &str = "21ca4243c9a64a352d2daee9469fddbf4c7310a2c2cafdb3b1ff19c6ff6c2339";
const H2_IDS_U32LE: &str = "53360c5d887413fa32d68b11e290dfaaf337c2d786f90ffa3d5371b28db382af";
fn completion_tokens(prompt_tokens: usize) -> usize {
    if matches!(prompt_tokens, 11444 | 16384) {
        256
    } else {
        64
    }
}
fn bad(message: impl std::fmt::Display) -> EngError {
    EngError::BadRequest(format!("Flint route capture: {message}"))
}
fn required_path(name: &str) -> Result<PathBuf> {
    let path = PathBuf::from(std::env::var_os(name).ok_or_else(|| bad(format!("set {name}")))?);
    if !path.is_absolute() {
        return Err(bad(format!("{name} must be absolute")));
    }
    Ok(path)
}
fn file_hash(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(bad)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 << 10];
    loop {
        let n = file.read(&mut buffer).map_err(bad)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}
fn utc() -> Result<String> {
    let output = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%S.%NZ"])
        .output()
        .map_err(bad)?;
    if !output.status.success() {
        return Err(bad("UTC clock command failed"));
    }
    Ok(String::from_utf8(output.stdout)
        .map_err(bad)?
        .trim()
        .to_owned())
}
fn request(path: &Path) -> Result<(Value, usize, String)> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(bad)?
        .take(65537)
        .read_to_end(&mut bytes)
        .map_err(bad)?;
    if bytes.len() > 65536 {
        return Err(bad("retained input exceeds bound"));
    }
    let hash = format!("{:x}", Sha256::digest(&bytes));
    let input: Value = serde_json::from_slice(&bytes).map_err(bad)?;
    match hash.as_str() {
        INPUT_512 => Ok((input, 512, hash)),
        INPUT_11444 => {
            let mut body = input;
            body["model"] = json!("flint-pilot");
            Ok((body, 11444, hash))
        }
        INPUT_4000 | INPUT_16384 => {
            let expected = if hash == INPUT_16384 { 16384 } else { 4000 };
            let max_tokens = completion_tokens(expected);
            let text = input["text"]
                .as_str()
                .ok_or_else(|| bad("retained input has no text"))?;
            Ok((
                json!({"model":"flint-pilot","messages":[{"role":"user","content":text}],
                "temperature":0,"max_tokens":max_tokens,"stream":true}),
                expected,
                hash,
            ))
        }
        _ => Err(bad(
            "input is not one of the four retained corpus byte hashes",
        )),
    }
}
fn prompt(bundle: &TokenizerBundle, body: &Value, expected: usize) -> Result<Vec<u32>> {
    let messages = body["messages"]
        .as_array()
        .ok_or_else(|| bad("missing messages"))?;
    if messages.len() != 1
        || messages[0]["role"] != "user"
        || body["temperature"] != 0
        || body["max_tokens"] != completion_tokens(expected)
        || body["stream"] != true
    {
        return Err(bad("retained request controls differ"));
    }
    let content = messages[0]["content"]
        .as_str()
        .ok_or_else(|| bad("missing user content"))?;
    let message = PromptMessage {
        role: Role::User,
        content: Some(content.to_owned()),
        reasoning_content: None,
        name: None,
        tool_call_id: None,
        tool_calls: vec![],
    };
    let ids = bundle.encode_prompt(&[message], &[], Thinking::Default)?;
    if ids.len() != expected {
        return Err(bad(format!(
            "actual prompt count {}, expected {expected}",
            ids.len()
        )));
    }
    if expected == 11444 && ids_hash(&ids) != H2_IDS_U32LE {
        return Err(bad("actual H2 prompt IDs differ from the sealed input IDs"));
    }
    Ok(ids)
}
fn ids_hash(ids: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for id in ids {
        hasher.update(id.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}
fn generate(
    executor: &mut FlintExecutor,
    session: &mut FlintSession,
    bundle: &TokenizerBundle,
    tokens: &[u32],
    max_tokens: usize,
) -> Result<(Vec<u32>, String)> {
    executor.check_prompt(tokens.len())?;
    executor.reset(session)?;
    {
        let _phase = route_trace::phase("prefill")?;
        executor.prefill(session, tokens)?;
    }
    if session.state.fed() != tokens.len() - 1 {
        return Err(bad("prefill did not retain exactly one pending tail"));
    }
    let mut generation = GenerationState::new(IncrementalDecoder::new(bundle.tokenizer()))
        .with_max_tokens(u32::try_from(max_tokens).map_err(bad)?);
    let mut ids = Vec::with_capacity(max_tokens);
    let mut text = String::new();
    let _phase = route_trace::phase("decode")?;
    for index in 0..max_tokens {
        let batch = executor.decode(session, &mut generation)?;
        if batch.tokens.len() != 1
            || batch.finish != (index + 1 == max_tokens).then_some(DecodeFinish::Length)
        {
            return Err(bad("early EOS, missing output or incorrect length finish"));
        }
        ids.extend_from_slice(&batch.tokens);
        text.push_str(&batch.text);
        executor.after_emit()?;
    }
    if session.state.fed() != tokens.len() + max_tokens - 1 {
        return Err(bad("decode fed-row accounting differs"));
    }
    executor.boundary(session);
    Ok((ids, text))
}
fn source_hashes() -> Value {
    let files: [(&str, &[u8]); 5] = [
        ("device.rs", include_bytes!("cuda_ffn/device.rs")),
        ("runtime.rs", include_bytes!("cuda_ffn/device/runtime.rs")),
        (
            "route_trace.rs",
            include_bytes!("cuda_ffn/device/route_trace.rs"),
        ),
        ("cuda_executor.rs", include_bytes!("cuda_executor.rs")),
        (
            "cuda_executor_route_trace_tests.rs",
            include_bytes!("cuda_executor_route_trace_tests.rs"),
        ),
    ];
    let map: serde_json::Map<String, Value> = files
        .into_iter()
        .map(|(name, bytes)| {
            (
                name.to_owned(),
                Value::String(format!("{:x}", Sha256::digest(bytes))),
            )
        })
        .collect();
    Value::Object(map)
}
#[test]
fn retained_request_shapes_and_wrong_byte_hash_are_explicit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("wrong.json");
    std::fs::write(&path, b"{\"text\":\"not the frozen corpus\"}").unwrap();
    assert!(request(&path).is_err());
}
#[test]
#[ignore = "CPU only: pinned publisher tokenizer and FIFI_FLINT_ROUTE_TRACE_INPUT required"]
fn retained_16k_request_controls_and_prompt_count() -> Result<()> {
    let (body, expected, hash) = request(&required_path("FIFI_FLINT_ROUTE_TRACE_INPUT")?)?;
    assert_eq!(hash, INPUT_16384);
    assert_eq!(expected, 16384);
    assert_eq!(body["max_tokens"], 256);
    let bundle = crate::tokenizer::flint_tests::load_bundle();
    assert_eq!(prompt(&bundle, &body, expected)?.len(), 16384);
    Ok(())
}
#[test]
#[ignore = "CPU only: sealed H2 input and publisher tokenizer required; no rendering of new golden IDs"]
fn retained_h2_request_controls_and_prompt_ids() -> Result<()> {
    let (body, expected, hash) = request(&required_path("FIFI_FLINT_ROUTE_TRACE_INPUT")?)?;
    assert_eq!(
        hash,
        "d9138724f83da85e472b09314863b1a425d07dec4e681746945110c51f2216e7"
    );
    assert_eq!(expected, 11444);
    assert_eq!(body["max_tokens"], 256);
    let bundle = crate::tokenizer::flint_tests::load_bundle();
    let ids = prompt(&bundle, &body, expected)?;
    assert_eq!(
        ids_hash(&ids),
        "53360c5d887413fa32d68b11e290dfaaf337c2d786f90ffa3d5371b28db382af"
    );
    Ok(())
}
#[test]
#[ignore = "coordinator lease only: real Flint pack/tokenizer and FIFI_FLINT_ROUTE_TRACE_INPUT/OUT; no timing claim"]
fn trace_real_exl3_routes() -> Result<()> {
    let input = required_path("FIFI_FLINT_ROUTE_TRACE_INPUT")?;
    let output = required_path("FIFI_FLINT_ROUTE_TRACE_OUT")?;
    if output.exists() {
        return Err(bad("output must not already exist"));
    }
    let root = required_path("FLINT_EXL3_PACK")?
        .canonicalize()
        .map_err(bad)?;
    if crate::tokenizer::flint_tests::fixture_dir()
        .canonicalize()
        .map_err(bad)?
        != root
    {
        return Err(bad("tokenizer and model pack directories differ"));
    }
    let bundle = crate::tokenizer::flint_tests::load_bundle();
    let (body, expected, input_hash) = request(&input)?;
    let tokens = prompt(&bundle, &body, expected)?;
    let prompt_hash = ids_hash(&tokens);
    let max_tokens = completion_tokens(expected);
    let (context_limit, byte_cap) = if matches!(expected, 11444 | 16384) {
        (33280, 64 << 20)
    } else {
        (4096, 32 << 20)
    };
    let options_json =
        json!({"context_limit":context_limit,"kv_profile":"f16","prefill_chunk_size":8});
    let options: FlintOptions = serde_json::from_value(options_json.clone()).map_err(bad)?;
    let mut assets = serde_json::Map::new();
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ] {
        assets.insert(name.to_owned(), file_hash(&root.join(name))?.into());
    }
    let elf_hash = file_hash(&std::env::current_exe().map_err(bad)?)?;
    let (mut executor, mut session) = FlintExecutor::open(
        &root,
        &options,
        bundle.eos_token_ids().to_vec(),
        bundle.tokenizer().vocab_size() as usize,
    )?;
    let mut layers = Vec::new();
    for index in 0..executor.model.layer_count() {
        if matches!(executor.model.layer(index)?.ffn, FlintFfnView::Moe { .. }) {
            layers.push(index);
        }
    }
    let control = generate(&mut executor, &mut session, &bundle, &tokens, max_tokens)?;
    let mut header = json!({"request_id":format!("retained-{expected}"),"source_base":BASE,
        "compiled_source_sha256":source_hashes(),"elf_sha256":elf_hash,"started_utc":utc()?,
        "model_pack_root":root,"resident_weight_bytes":executor.model.device_weight_bytes(),"model_config_sha256":format!("{:x}",Sha256::digest(executor.model.model_config())),
        "tokenizer_asset_sha256":assets,"input_path":input,"input_sha256":input_hash,
        "prompt_tokens":tokens.len(),"prompt_ids_u32le_sha256":prompt_hash,
        "executor_options":options_json,"prefix_reuse":false,"temperature":0,"max_tokens":max_tokens,
        "text_hash_domain":"executor incremental UTF-8, not HTTP-visible text","byte_cap":byte_cap});
    if expected == 11444 {
        header["retained_case"] = "H2".into();
        header["corpus_source_commit"] = "e9faed0c5ed3d088942624df38b49cfb3ea781fb".into();
        header["request_model_alias"] = body["model"].clone();
    }
    let capture = route_trace::Capture::begin(&output, header, &layers, tokens.len())?;
    let traced = generate(&mut executor, &mut session, &bundle, &tokens, max_tokens)?;
    if control != traced {
        return Err(bad("traced/untraced full IDs or text differ"));
    }
    let stopped = capture.stop()?;
    drop(session);
    drop(executor);
    // The existing query rejects a latched cleanup failure before entering native CUDA.
    let cleanup = crate::kernels::cuda::device_count();
    stopped.complete(json!({"finished_utc":utc()?,"generated_ids":traced.0,"control_equal":true,
        "full_text_sha256":format!("{:x}",Sha256::digest(traced.1.as_bytes())),
        "completion_tokens":max_tokens,"finish_reason":"length","cleanup_check":"post-drop guarded device_count"}),cleanup)
}
