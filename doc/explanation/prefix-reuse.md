# How conversation reuse works

Chat clients send the whole conversation with every request.
Without reuse, the engine processes every token of that history again on each turn, so replies take longer to start as the conversation grows.

With `"prefix_reuse": true` in the configuration file, the engine saves its state at the end of each finished turn.
When the next request starts with the same tokens, the engine restores that state and processes only the new part.
On a long conversation this turns tens of seconds of prompt processing into a few seconds.

## Matching the prompt

Reuse needs the new prompt to start with exactly the tokens the engine saw.
Three settings help with that:

- **Warm turns** (`ENGINESRV_WARM_TURNS`, on by default). The server remembers the token sequence of the last conversation, so the next turn only tokenizes its new messages. A request from another conversation replaces that memory.
- **Block saves** (`ENGINESRV_PREFIX_BLOCK_MATCH`, on by default). Besides the end of each turn, the engine saves its state every 4,096 tokens. When a client rewrites the end of the conversation, the engine restores the last block before the change instead of starting over. In one measurement, a 174,829-token prompt whose last 23 tokens changed reached its first token in 11.8 s with block saves and 40.7 s without.
- **Reasoning replay** (`ENGINESRV_THINK_REPLAY`, on by default for Qwen3.6). Clients resend the model's earlier reasoning. fifi passes it to the chat template, which keeps the reasoning of turns in the current tool-calling loop. A system message inside that loop is rendered as a tool response (`ENGINESRV_NOTICE_TOOLWRAP`), so the template does not treat it as a new user turn and drop the loop's reasoning.

## Where saved prefixes live

On Apple Silicon, saved prefixes share `session_cap_bytes` with the live session. The store keeps the 6 most recently used prefixes (`prefix_store_entries`).

On NVIDIA CUDA, the newest saved prefix stays on the GPU until a later request overwrites it, so restoring it needs no copy from host memory.
Up to two saved prefixes are also kept in pinned host memory; see [How fifi plans memory](memory.md#on-nvidia-cuda).

## Which engines support it

| Model and backend | Prefix reuse |
| --- | --- |
| Qwen3.6, Apple Silicon | yes; off in the shipped Mac file |
| Qwen3.6, CUDA | yes; on in all shipped CUDA files |
| Flint, Apple Silicon | experimental: also set `FIFI_FLINT_PREFIX_REUSE=1` |
| Flint, CUDA | experimental: also set `FIFI_FLINT_PREFIX_REUSE=1`; `kv_profile` `f16` only |

## Seeing it work

The `usage` of each reply reports how many prompt tokens came from a saved prefix: `reused_tokens` and `prompt_tokens_details.cached_tokens` on the OpenAI API, `cache_read_input_tokens` on the Anthropic API.
