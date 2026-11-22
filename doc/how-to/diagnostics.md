# Diagnostics: doctor, check, and the support bundle

Use these commands to diagnose why fifi will not run, before or after a failure.
None loads model weights, starts a server, or touches the network.

```text
enginesrv doctor [--config <path>] [--json] [--bundle <file>]
enginesrv check <model_dir-or-gguf> [--tokenizer-dir <dir>] [--family <name>]
                 [--context <tokens>] [--kv <f32|bf16|q8>] [--json]
enginesrv config effective <path> [--json]
```

- `doctor` — what this build can serve on this machine: compiled backends,
  CUDA device and free VRAM, host RAM and threads. With `--config`, also
  validates that file.
- `check` — a model directory or one `.gguf` file: compatibility verdict,
  the checks behind it, and a memory estimate where supported. An explicit
  file excludes other weight files in the directory and defaults tokenizer
  side files to its parent. For Qwen3.6 CUDA, the weight check names the
  selected GGUF and the estimate uses its byte count. Exit 0 when servable.
- `config effective` — the resolved configuration with the origin of every
  value (config file, environment, family default, or compiled default).

## The support bundle

```bash
enginesrv doctor --bundle fifi-bundle.json --config engine/config/qwen36-cuda-3090-kv8-prefix.json
```

This writes one JSON document you can attach to an issue. It works with no GPU,
no model, and no running server — those facts are recorded as unavailable,
never guessed. An existing file is never overwritten; the write is atomic
(a `.part` sibling, then one rename).

What is inside:

- `fifi` — version and platform; `os` — the distribution's `PRETTY_NAME`.
- `hardware` — the same probe `doctor` runs, every value labeled
  `measured` or `unavailable`.
- `engines` — for each configured engine: the preflight report `check`
  prints, and the effective settings rows `config effective` prints,
  each with its origin. A broken config or an unreadable model directory
  degrades to an `unavailable` note for that engine; the rest of the
  bundle is still written.
- `manifest` — which sections were collected, unavailable, or not
  requested.
- `summary` — the same text the CLI prints.

What is deliberately not inside: API keys or any environment values
beyond the allowlisted `Q36_CACHE_LIMIT` the settings rows read, prompts
or completions (no request data is an input), logs (no log source is
read — attach yours separately), and model or tokenizer contents. Home
directory paths become `~` everywhere, including paths quoted inside
refusal messages. Loading stages exist only in a running server, so the
bundle records that section as unavailable rather than inventing it.

Inspect the file before sharing it:

```bash
jq .summary fifi-bundle.json
jq '.engines[0].preflight.checks' fifi-bundle.json
```

A trimmed sample (schema `fifi-diagnostic-bundle/1`, synthetic values):

```json
{
  "bundle_schema": "fifi-diagnostic-bundle/1",
  "generated": {"measured": true, "unix_seconds": 1762200000},
  "fifi": {"version": "0.1.0", "platform": {"os": "linux", "arch": "x86_64"}},
  "hardware": {
    "backends": {"cuda_compiled": true, "mlx_compiled": false, "none_compiled": false},
    "cuda": {
      "device_count": {"measured": 1},
      "first_device_free_bytes": {"measured": 21474836480},
      "first_device_total_bytes": {"measured": 24220051456}
    },
    "host": {"mem_total_bytes": {"measured": 67426263040}, "cpu_threads": 24}
  },
  "engines": [
    {
      "id": "q36",
      "family": "qwen3_6_35b_a3b",
      "model_dir": "~/models/qwen36",
      "preflight": {
        "verdict": "qualified",
        "checks": [
          {"name": "gguf header", "status": "pass", "detail": "model.gguf: 32 B"},
          {"name": "backend", "status": "pass", "detail": "cuda"}
        ],
        "next_step": "start the engine"
      },
      "effective_settings": [
        {"key": "kv_dtype", "value": "q8", "origin": "config"},
        {"key": "context_limit", "value": "32768", "origin": "family default"}
      ]
    }
  ],
  "manifest": [
    {"name": "hardware", "status": "collected"},
    {"name": "engines", "status": "collected", "count": 1},
    {"name": "startup_stages", "status": "unavailable"},
    {"name": "logs", "status": "not_collected"}
  ]
}
```
