# Serve a model without a config file

`enginesrv serve` starts one model from a path or a name. It builds the same configuration a file would hold and takes the same startup checks, so nothing experimental is switched on this way.

```sh
export ENGINESRV_API_KEY=my-secret-key
enginesrv serve --model ~/models/qwen3.6/Qwen3.6-35B-A3B-Q4_K_M.gguf
```

The server listens on `127.0.0.1:18791` and reads the API key from `ENGINESRV_API_KEY`. It refuses to start without the key.

## What `--model` accepts

| Argument | What the server loads |
| --- | --- |
| A directory | The checkpoint in it. The family comes from `config.json`, or from the GGUF file's `general.architecture` when `config.json` is absent. |
| A `.gguf` file | That file. The server reads the tokenizer and the chat template from the file's header. |
| A plain name, such as `qwen3.6` | The local model of that name. Search order: the models directory (`~/models`, or `FIFI_MODELS_DIR`), the Ollama store (`~/.ollama/models`, or `OLLAMA_MODELS`, where a tag uses `name:tag`), and the Hugging Face hub cache. The server refuses multiple matches in one store and prints their paths. |
| `org/repo` or `org/repo:QUANT` | A GGUF repository of the Hugging Face hub. The server downloads one quantization into the hub cache and serves it. Without `:QUANT`, it picks the largest quantization that fits the card at the default context, otherwise the smallest. It prints the choice and the reason. |
| Nothing | The most recent local model that this build can serve. The command prints which model it picked and which newer ones it skipped. |

A bare GGUF file selects its family from its architecture: `qwen35moe` is Qwen3.6, `gemma4` is Bolt E4B and `raon1` is Raon-1-mini. SenseNova-Loop needs `--family sensenova`. See [Models](../reference/models.md) for what each family serves on.

## Options

| Option | Meaning |
| --- | --- |
| `--tokenizer-dir <dir>` | Read the tokenizer files from this directory instead of the model path. |
| `--family <name>` | Name the family instead of detecting it. |
| `--context <n>` | Context limit in tokens, for the families that take one. |
| `--kv f32\|q8\|bf16` | KV cache format, for the families that take one. |
| `--listen <addr>` | Listen address. It must be on this machine. |
| `--print-config` | Print the configuration file that this start is equal to, and exit. |

To change anything else, print the configuration with `--print-config`, edit it, and start with `--config`. See the [configuration reference](../reference/configuration.md).

## Downloads

A download by `org/repo` uses the Hugging Face hub cache, so other tools that read that cache find the files too.

| Variable | Meaning |
| --- | --- |
| `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` | Access token for gated or private repositories. |
| `FIFI_HF_ENDPOINT` or `HF_ENDPOINT` | Hub address, for a mirror. The first one wins when both are set. |
| `FIFI_HF_HUB`, `HF_HUB_CACHE` or `HF_HOME` | Location of the hub cache, in that order of precedence (`HF_HOME` holds it in its `hub` directory). |
