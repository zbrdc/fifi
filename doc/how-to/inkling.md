# Run Inkling 2 Mini on Apple Silicon

The experimental native text worker reads the original mixed NVFP4/MXFP8 safetensors release. Download the text checkpoint and its tokenizer to a local SSD.

```sh
hf download thinkingmachines/Inkling-2-Mini \
  --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 \
  --include config.json model.safetensors.index.json model-00001-of-00001.safetensors \
    tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json \
  --local-dir ~/models/inkling2-mini
```

From the engine directory, build the native MLX server and start it with a short context. Set `ENGINESRV_API_KEY` to your local server key before you start it.

```sh
cargo build --release --locked --features native-mlx --bin enginesrv
target/release/enginesrv serve --model ~/models/inkling2-mini \
  --context 128 --listen 127.0.0.1:18795
```

Use the model name from `GET /v1/models`. The worker supports one active sequence and processes prompt tokens in order. It keeps 48 independent sets of KV and convolution history for the 24 physical layers. The CPU evaluates dense projections and attention. Metal evaluates the eight selected experts at each visit. The planner reserves space for dense weights, sequence state and staging, then uses the remaining budget for expert weights. On a cache miss, fifi reads the selected expert ranges from the checkpoint. The operating system can serve those reads from its file cache.

With a Metal recommendation of at most 16 GiB, Inkling defaults to a 5 GiB working set for safetensors and 6 GiB for GGUF. Longer contexts can raise that default to cover required allocations within Metal's recommendation. Larger working sets use Metal's recommendation. Set `FIFI_MLX_WORKING_SET_MIB` to override the ceiling, or `FIFI_MLX_AUTO_BUDGET=1` to use Metal's full recommendation. `GET /v1/recent-requests` reports the measured prompt and generation timings.

For chat requests, disabling thinking sets the publisher's reasoning effort to 0.0. Inkling can still emit reasoning, so a short output limit can be reached before the first answer token. The server returns reasoning and answer text in their respective fields. [Publisher guidance](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/skills/inkling/SKILL.md#thinking-effort-set-it-everywhere)

The GGUF inspector and both experimental text backends read the mainline `inkling2_loop` layout and the older `inkling2` layout. Unsloth uses `inkling2_loop` at the repository root and `inkling2` in `Old_Support/`. Both experimental text adapters need the pinned publisher tokenizer. Use a directory containing one text GGUF or pass the selected model file, including files with a `.gguf_file` suffix.

The GGUF adapters advance prompt tokens serially and can group expert projections within each visit. The safetensors worker can group multiple prompt tokens per visit.

```sh
target/release/enginesrv serve --model /path/to/Inkling-2-Mini-IQ2_XXS.gguf \
  --tokenizer-dir ~/models/inkling2-mini --context 192 --listen 127.0.0.1:18795
```

Native MLX builds on Apple Silicon use up to four attention projection workers for both formats. GGUF also uses mapped dense weights, direct packed Metal kernels where supported, and grouped expert projections. Where all projections support direct packed kernels, it retains one copy of each expert. Set `FIFI_INKLING_ATTN_CPU_WORKERS=1` to use one worker, or set the corresponding `FIFI_INKLING_GGUF_*` setting to `0` for a comparison. `FIFI_INKLING2_MINI_MLX=0` disables the native family. `FIFI_INKLING2_MINI_GGUF_MLX=0` disables GGUF serving.

| Artifact | Inspect | Load and generate | HTTP |
| --- | --- | --- | --- |
| Publisher NVFP4/MXFP8 safetensors | Yes | Real checkpoint, Metal experts | Raw, chat, and multi-turn WebGUI |
| Mainline `inkling2_loop` GGUF | Yes | Real IQ2_XXS, Metal experts | Explicit file-path raw completions and streamed chat |
| Repository-root `inkling2` GGUF | Yes | Real IQ2_XXS, Metal experts | Explicit file-path raw completions and streamed chat |

`enginesrv check` and `serve --print-config` select the native Inkling family for both GGUF layouts and for the publisher safetensors under the 6 GiB ceiling. They inspect the vision projector (`clip` metadata) and the DSpark drafter (`dflash` metadata) only. Neither runs on this path.

## Check the serving contract

Use a dedicated server on loopback with the original checkpoint and the verified 192-token context. Start it with a JSON config of this form, substituting the absolute model path.

```json
{
  "listen": "127.0.0.1:18795",
  "api_key_env": "ENGINESRV_API_KEY",
  "prefix_reuse": false,
  "sample_trace": true,
  "engines": [{
    "id": "inkling",
    "family": "inkling2_mini",
    "model": "inkling",
    "model_dir": "/path/to/inkling2-mini",
    "tokenizer_dir": "/path/to/inkling2-mini",
    "context_limit": 192
  }]
}
```

Start it with `enginesrv --config /path/to/config.json`. To exercise the WebGUI, set `web_dir` at the top level to the built WebGUI directory. Open the loopback address in a browser, select Inkling in the model picker, and send “Name the capital of France. Answer in one word.” For a follow-up, send “Which country is that city in? Answer in one word.” The measured browser run printed “Paris” and “France” using a 192-token context and a 144-token output limit. Run the suite from the repository root once the model is ready. Python 3.9 or later is sufficient.

```sh
python3 -B engine/tests/tools/check_inkling_serving.py \
  --base-url http://127.0.0.1:18795 --model inkling \
  --report /path/to/scratch/inkling-serving-evidence.json
```

To check raw text generation on the running server, send the same greedy three-token request used by the suite.

```sh
curl --fail-with-body --silent --show-error http://127.0.0.1:18795/v1/completions \
  -H "Authorization: Bearer ${ENGINESRV_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling","prompt":"The capital of France is","max_tokens":3,"temperature":0}'
```

The verified response has a `choices[0].text` value of `" Paris.\n\n"`, a `length` finish reason, five prompt tokens, and three completion tokens.

The report preserves JSON responses and stream events, including reasoning. The test checks raw and chat output caps, streaming parity, the sampled EOS ID, and its inclusion in completion usage. It also checks explicit multi-turn history, repeatability across requests, and reset after an observed client disconnect. The raw reference check re-encodes the first three matching tokens of the llama.cpp IQ2_XXS and native outputs using the pinned tokenizer. The server exposes sampled token IDs on the chat route.

## Reproduce a first and repeat benchmark

Run the driver from the repository root with the release server binary and a new output directory.

```sh
FIFI_INKLING_PREFILL_ROWS=1 FIFI_INKLING_ATTN_CPU_WORKERS=1 \
  python3 -B engine/tests/tools/bench_inkling.py \
  --server engine/target/release/enginesrv --model ~/models/inkling2-mini \
  --output-dir /path/to/scratch/inkling-5g --ceiling-mib 5120 --max-tokens 12
```

The driver starts one server on an OS-assigned loopback port, sends two identical greedy requests, and checks that both reach the requested output limit. It writes the server config, startup and phase log, and a JSON receipt. The receipt records wall time, request timings, process-attributed disk bytes, and system swap-page deltas on macOS. Existing Inkling environment settings carry through to the server. Use `--ceiling-mib auto` to remove a configured ceiling, `--tokenizer-dir` for a separate pinned bundle, and `--commit`, `--build`, and `--checkpoint-id` to record artifact provenance. Runs do not control the OS page cache.

Use these prompt values with `--prompt` and keep the context at 128 for the reported safetensors comparisons. The driver checks the prompt length and both output counts. Run each arm in a fresh process and use a new output directory.

| Workload | Prompt as a Python string expression | Prompt tokens | `--max-tokens` |
| --- | --- | ---: | ---: |
| Short | `"The capital of France is"` | 5 | 12 |
| Longer prefill | `"Rivers flow into the sea. " * 8 + "Name a river in France."` | 62 | 8 |
| Sustained decode | `"Write a detailed history of Paris, beginning with its earliest settlements."` | 14 | 64 |

For the sustained 5 GiB storage comparison, set `FIFI_INKLING_MMAP_DENSE` and `FIFI_INKLING_FILE_CACHE_ONLY` to `0` in the private/LRU arm and `1` in the mapped/file-cache arm. For the longer-prefill comparison, keep both settings at `1` and change `FIFI_INKLING_PREFILL_ROWS` from `1` to `4`. Set `FIFI_INKLING_COMPONENT_TIMING=1` for the phase-level CPU, Metal, and logical dense I/O counters.

With GGUF, `FIFI_INKLING_ROUTE_TRACE=1` writes ordered expert selections,
plane sizes and cache counters for each visit. Use a separate trace run
when comparing throughput.

GGUF checkpoints using single-copy packed Metal experts default to a
segmented LRU cache. Hits can
promote planes into a protected queue within 90% of the same Metal
budget. Sustained misses without a protected hit let a changed working
set reclaim the oldest protected entries. Set
`FIFI_INKLING_GGUF_SEGMENTED_LRU=0` to compare with plain LRU at the
same checkpoint and working-set budget.

For GGUF, the same timing output reports `gguf_file_read_bytes`,
`gguf_unique_file_read_bytes`, and `gguf_metal_staged_bytes`. The first two
report total file demand and one copy per distinct expert tensor plane in each
phase. The last counts the packed or expanded weights staged for Metal,
including weights served from the retained RAM tier. Use the benchmark's
process-attributed disk counter to check physical SSD reads.

### Packed GGUF attention projections

`FIFI_INKLING_GGUF_PACKED_DOT=1` selects direct Q6_K and Q8_0 row dots
for mapped GGUF weights. The packed rows preserve the FP32 accumulation
order. The buffered reference and mapped path produced bitwise-equal
logits on both complete IQ2_XXS GGUF layouts.

`FIFI_INKLING_GGUF_Q8_ACTIVATIONS=1` quantizes inputs to mapped Q8_0
projections in 32-value groups using the shared CPU quantizer. This
experiment requires AArch64 dot-product support. Both settings are
default-off.

On a 16 GiB M4 Mac mini, the server answered a five-token prompt with four output tokens. It used the mainline IQ2_XXS GGUF, a 192-token context, and a 6 GiB cap. With packed rows in both arms, the warm repeat hit its first token in 3.61 s (FP32 activations) and 2.09 s (Q8 activations). Decode rose from 0.98 to 1.37 tok/s. The two runs returned identical text and recorded zero logical expert bytes, physical SSD bytes, and swap-outs on the repeats. Both arms passed the idle-host audit.

A separate comparison at five matched token positions measured mean
KLD of 0.00107 nats against FP32 activations on the same GGUF. Four of the
five positions retained the same top token. Quality and throughput over
longer prompts were not measured.
