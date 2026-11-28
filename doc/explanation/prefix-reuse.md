# How conversation reuse works

Chat clients send the whole conversation with every request.
Without reuse, the engine processes every token of that history again on each turn, so replies take longer to start as the conversation grows.

With `"prefix_reuse": true` in the configuration file, the engine saves its state after each finished turn. An omitted key resolves per family at startup: on for qualified reuse paths (Flash-Next CUDA, Qwen3.6 CUDA, and Bolt on CUDA), off for others. Marking boundaries alone would only split prefill and change outputs. On Bolt, the engine also checks host memory at startup and before every save. See [Bolt](memory.md#bolt).
When the next request starts with the same tokens, the engine restores that state and processes only the new part.
On a long conversation, this cuts prompt processing from tens of seconds to a few seconds.

## Matching the prompt

Reuse needs the new prompt to start with the exact tokens the engine saw.
Three settings help with that:

- **Warm turns** (`ENGINESRV_WARM_TURNS`, on by default). The server remembers the last conversation's token sequence, so the next turn tokenizes only new messages. A request from another conversation replaces that memory. The seeded slot stays for an identical retry (`ENGINESRV_SEED_RESENDS`, on by default). Any value other than `1`, `true`, or `on` disables it. With that switch on, `ENGINESRV_RESEND_EXTRA_BYTES` above zero retains one displaced conversation's result. The budget alone retains nothing.
- **Block saves** (`ENGINESRV_PREFIX_BLOCK_MATCH`, on by default). Beside the turn end, the engine saves one block checkpoint at a 4,096-token boundary just before the prompt end. On Qwen3.6 CUDA, each conversation keeps its own block checkpoint, and later cuts replace earlier ones until the store evicts the least recently used. Other engines keep one block checkpoint per numeric path. Block saves require a store that holds at least two saved prefixes. When a client edits the end of a conversation, the engine restores the block checkpoint before the edit. On a 174,829-token prompt whose last 23 tokens changed, first-token time dropped from 40.7 s to 11.8 s.
- **Reasoning replay** (`ENGINESRV_THINK_REPLAY`, on by default for Qwen3.6). Clients resend earlier model reasoning. fifi passes it to the chat template, keeping reasoning in the current tool-calling loop. `ENGINESRV_NOTICE_TOOLWRAP` renders a system message inside that loop as a tool response so the template preserves earlier reasoning.

## Where saved prefixes live

On Apple Silicon, saved prefixes share `session_cap_bytes` with the live session. The store keeps the 6 most recently used prefixes (`prefix_store_entries`).

On NVIDIA CUDA, the newest saved prefix stays on the GPU until a later request overwrites it, so restoring it needs no host copy.
The engine keeps up to eight saved prefixes in pinned host memory. See [How fifi plans memory](memory.md#on-nvidia-cuda).
With the disk cache ([configuration reference](../reference/configuration.md#prefix-cache-on-disk-cuda)), Qwen3.6 CUDA also writes prefixes to a directory when pinned memory fills. Background threads write evicting entries, and graceful stops flush memory-only tips. On a pinned-memory miss, the engine checks disk before reprocessing the prompt, trading full prefill for one verified disk read. Because files hold exact tokens and model state, the disk cache is a privacy choice, supports disablement, and serves GGUF loads only.

## Which engines support it

| Model and backend | Prefix reuse |
| --- | --- |
| Qwen3.6, Apple Silicon | yes. Off in the shipped Mac file |
| Qwen3.6, CUDA | yes. On in all shipped CUDA files |
| Bolt (Gemma 4 family), CUDA | yes. On by default. Saved prefixes are host copies funded from free host memory |
| Flint, Apple Silicon | experimental: also set `FIFI_FLINT_PREFIX_REUSE=1` |
| Flint, CUDA | experimental: also set `FIFI_FLINT_PREFIX_REUSE=1`. `kv_profile` `f16` only |

## Seeing it work

The `usage` of each reply reports how many prompt tokens came from a saved prefix: `reused_tokens` and `prompt_tokens_details.cached_tokens` on the OpenAI API, `cache_read_input_tokens` on the Anthropic API.

The request-log line and `/v1/recent-requests` also carry the engine's own
prefix report: `prefix_entries` and `prefix_bytes`, the stored snapshots the
executor holds for the request's engine. On Qwen3.6 with CUDA,
`prefix_bytes` is the committed payload of those entries. The RAM the prefix
store actually holds is larger: pinned slots are allocated at capacity, so a
half-full entry still occupies its full slot. Read `prefix_bytes` as the
payload figure, not as the store's memory footprint.
