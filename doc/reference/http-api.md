# HTTP API

`enginesrv` serves the routes below on the `listen` address of its [configuration file](configuration.md).
The contract is in [`api/openapi.yaml`](../../api/openapi.yaml). It describes every route below, including this server's extensions to the OpenAI and Anthropic APIs.

## Routes

| Method and path | Key needed | Purpose |
| --- | --- | --- |
| `GET /health` | no | Answers `{"status":"ok"}` while the process runs. |
| `GET /ready` | no | `200 {"status":"ready"}` when at least one engine is ready or busy. Otherwise `503` with code `not_ready` and each engine's state. |
| `GET /v1/models` | yes | Model names, each with the engines that serve it and their states. |
| `GET /v1/system` | yes | Build and process facts: version, compiled backends, host memory, CPU threads, engines, states, and context limits. No device probe. See [Monitor routes](#monitor-routes). |
| `POST /v1/chat/completions` | yes | OpenAI-style chat completion, with or without streaming. |
| `POST /v1/completions` | yes | Raw text completion of one prompt string, with or without streaming. The server does not apply the chat template. |
| `POST /v1/messages` | yes | Anthropic-style Messages request, with or without streaming. |
| `POST /v1/messages/count_tokens` | yes | Token count of a Messages request's prompt. No generation runs. |
| `POST /v1/token/encode` | yes | Token ids of text, or of chat messages after the chat template. No generation runs. |
| `POST /v1/token/decode` | yes | Text of a list of token ids. No generation runs. |
| `POST /v1/apply-template` | yes | Prompt text that the chat template makes from chat messages. No generation runs. |
| `GET /v1/sampler-trace` | yes | One reply's armed sampler-trace packet, as `?id=<reply id>`. See [Sampler trace](#sampler-trace). |
| `GET /v1/telemetry` | yes | GPU rows from `nvidia-smi` for the monitor panel. See [Monitor routes](#monitor-routes). |
| `GET /v1/recent-requests` | yes | Last 64 request-log rows, newest first. See [Monitor routes](#monitor-routes). |
| `GET /v1/logs` | yes | Server log lines. See [`api/openapi.yaml`](../../api/openapi.yaml). |
| `GET /v1/route-trace` | yes | Router staging ring of the last 16 decode steps. Answers `404` with the enabling hint unless `Q36_ROUTE_STAGE=1`. See [Monitor routes](#monitor-routes). |
| `GET /v1/engines` | yes | List configured engines. |
| `POST /v1/engines` | yes | Start a new engine dynamically. |
| `DELETE /v1/engines/{id}` | yes | Stop and remove an engine added through `/v1/engines`. |
| `GET /v1/hf/search` | yes | Search Hugging Face repositories. |
| `GET /v1/hf/cache` | yes | List models in the local Hugging Face hub cache. |
| `GET /v1/hf/tree` | yes | List files in a Hugging Face repository. |
| `GET /v1/hf/downloads` | yes | List active and completed Hugging Face downloads. |
| `POST /v1/hf/downloads` | yes | Start a Hugging Face download. |
| `DELETE /v1/hf/downloads/{id}` | yes | Cancel a Hugging Face download. |

## Authentication

Send the API key as `Authorization: Bearer <key>`.
The `/v1/messages` routes also accept `x-api-key: <key>`. When that header is present, it must match, and the server skips the bearer header check.
A missing or wrong key returns `401`.

## Sampler trace

`POST /v1/chat/completions` accepts the request-body property `x-sample-trace`
(boolean, default `false`). It is a body property, not an HTTP header. The
server ignores a header of that name. Setting it `true` arms capture for that
reply: the server records each sampled token's kept candidates and their
probabilities, and serves the packet at `GET /v1/sampler-trace?id=<reply id>`
with `Cache-Control: no-store`. Only the chat route arms capture.
`/v1/completions` and `/v1/messages` do not.

Capture is opt-in twice: the server's `sample_trace` setting must permit it,
and the request must arm it. Without the setting, the server refuses an armed
request with `400`. The ring keeps the last four replies' traces under a global
position budget and drops the oldest. An unknown or evicted id reports `404`,
so a dropped trace is explicit. The server captures nothing retroactively.
Arming happens before generation starts.

The packet carries the reply's generated text: the sampled pieces
include reasoning and tool-call text, unredacted, and the candidates
carry token ids with their probabilities. One process key
guards every API route, so any key holder can read any retained trace,
and reply ids are sequential and guessable. Treat the route accordingly
when prompts carry sensitive content.

## Chat completions

Request fields:

| Field | Default | Notes |
| --- | --- | --- |
| `model` | | A name from `/v1/models`. The first engine serving it takes the request. |
| `messages` | required | |
| `stream` | `false` | Server-sent events, ending with `data: [DONE]`. |
| `max_tokens` | 2,048 | Legacy field, deprecated in favor of `max_completion_tokens`. Lowered when the prompt plus the cap would pass the engine's context limit. |
| `max_completion_tokens` | 2,048 | Current OpenAI field. Same bounds and clamp as `max_tokens`, except `null` counts as absent while a `null` `max_tokens` fails. When you send both, `max_completion_tokens` wins. The request log records the winning field in `max_tokens_cap_field`. |
| `temperature` | from the model's `generation_config.json`, else 0.7 | 0 to 2. `0` selects greedy decoding. |
| `top_p` | from `generation_config.json`, else 1.0 | 0 to 1. |
| `top_k` | from `generation_config.json`, else 0 | `0` turns top-k off. |
| `stop` | none | Stop strings, matched on the decoded text. |
| `tools` | none | OpenAI function tools. |
| `tool_choice` | `auto` | `auto` or `none`. `required` and a named function return `400`. |
| `response_format` | `text` | `json_object` and `json_schema` return `400`. |
| `x-thinking` | `default` | Server extension: `on`, `off` or `default` (the chat template decides). |
| `chat_template_kwargs` | none | Server accepts only `enable_thinking`: `true` acts as `x-thinking: on`, `false` as `off`. |
| `x-engine` | none | Server extension: the `id` of the engine to use. Overrides `model`. An unknown id returns `404`. |
| `x-session` | none | Binds to the serving engine on first use and sticks to it. Unbound with no other routing field returns `404`. |

The request needs `model` or `x-engine`.

The reply message has `content`, `tool_calls` when the model called a tool, and `reasoning_content` with the thinking text when there was any.
`finish_reason` is `stop`, `length` or `tool_calls`.

`usage` has `prompt_tokens`, `completion_tokens` and `total_tokens`, plus `reused_tokens` and `prompt_tokens_details.cached_tokens`: the prompt tokens served from a saved prefix.
`ENGINESRV_CACHE_USAGE=0` removes the last two.

## Completions

`POST /v1/completions` continues one prompt string.
The server tokenizes the prompt as written. It does not apply the chat template and adds no beginning-of-sequence token.
Text in the prompt that spells a special token, such as `<|im_end|>`, tokenizes as that special token. The server does not escape it, unlike the content of chat messages.
The reply text is the generated text as it is, with no think block, tool call or whitespace cut out.

| Field | Default | Notes |
| --- | --- | --- |
| `model` | | As for chat completions. |
| `prompt` | required | One non-empty string. A list of strings or of token ids returns `400`. |
| `stream` | `false` | Server-sent events, ending with `data: [DONE]`. |
| `max_tokens` | 2,048 | As for chat completions. This legacy route does not read `max_completion_tokens`. The OpenAI reference defines that field only for chat completions. |
| `temperature`, `top_p`, `top_k` | as for chat completions | Same bounds and defaults. |
| `stop` | none | One string or an array of strings, matched on the decoded text. The stop text is not part of the reply. |
| `x-engine`, `x-session` | none | As for chat completions. |

`n` other than 1, `echo: true`, and a `logprobs`, `best_of` or `suffix` that is not `null` return `400`.
The server accepts and ignores `seed`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `stream_options` and `user`, as on chat completions.
A stream always ends with the usage chunk, whatever `stream_options` says.
A body over 16 MiB returns `400`.

The reply has `object` `text_completion` and one choice with `index`, `text`, `logprobs` (always `null`) and `finish_reason` (`stop` or `length`).
`usage` has the fields of chat completions.
A stream sends one chunk for each piece of text, then a chunk with empty `text`, the `finish_reason` and the `usage`, then `data: [DONE]`.
Chunks carry no role.
A failure after the first chunk sends an `event: error` with the usage counted so far and no `[DONE]`.
The server marks no prefix boundary for a completion, so it does not save or restore a prefix for it.

## Token and template routes

The three routes below use the tokenizer and chat template of the selected engine.
They queue no job, so they answer while an engine is loading or after it failed.
Each takes `model`, `x-engine` and `x-session` as chat completions do: `x-engine` overrides `model`, an unknown engine or model returns `404`, and a request with none of them returns `400`.
A bad field returns `400` with the field's name in the message, and so does a body over 16 MiB.
The server ignores sampling fields on these routes: `temperature`, `top_p`, `top_k`, `max_tokens`, `stop` and `stream`.
Each served request writes one line to the request log, as `POST /v1/messages/count_tokens` does. A failed request writes none.
The line has no finish reason or timings. Its `prompt_tokens` is the number of IDs returned by encode, received by decode, or zero for apply-template.

`POST /v1/token/encode` takes either `text` or `messages`, not both.

| Field | Notes |
| --- | --- |
| `text` | A string, tokenized as written. The server adds no beginning-of-sequence token, and text of a special token such as `<|im_start|>` becomes that token. |
| `messages` | Chat messages, rendered with the chat template and the generation prompt, then tokenized. The ids match what a chat completion with the same fields sends to the engine. |
| `tools`, `tool_choice`, `x-thinking`, `chat_template_kwargs` | With `messages` only. They have the meaning and checks of chat completions, so `response_format` of `json_object` or `json_schema` and refused `tool_choice` values return `400`. |

The reply is `{"tokens": [...], "length": n}`: the ids, and their count.

`POST /v1/token/decode` takes `tokens`, an array of token ids.
Each id is an integer from 0 to the vocabulary size minus 1. An id outside that range returns `400`.
An array longer than the selected engine's context limit, in tokens, returns `400`.
The reply is `{"text": "..."}`. The decoder always omits special tokens. There is no switch to keep them.

`POST /v1/apply-template` takes `messages` and the same optional fields as encode with `messages`.
The reply is `{"prompt": "..."}`: the text that a chat completion would tokenize, ending with the generation prompt.
A chat template that renders no text returns `400` with the message `rendered prompt is empty` on this route and on encode with `messages`.

## Messages

Request fields follow the Anthropic API. `model`, `max_tokens` and `messages` are required. `system`, `stop_sequences`, `temperature`, `top_p`, `top_k`, `stream`, `tools`, `tool_choice` and `metadata` are optional.
Sampling bounds and defaults are the same as for chat completions.

| Field | Notes |
| --- | --- |
| `thinking` | `{"type": "enabled"}` or `"adaptive"` turns thinking on, `"disabled"` turns it off. Without it, the chat template decides. The server checks `budget_tokens` but does not limit thinking. |
| `tool_choice` | `auto` or `none`. `any` and `tool` return `400`. |
| `metadata` | Accepted and ignored. |

Content blocks in the reply are `text`, `thinking` (with an empty `signature`) and `tool_use`.
`usage.cache_read_input_tokens` counts prompt tokens served from a saved prefix, and `input_tokens` counts the rest. `cache_creation_input_tokens` is always 0.

## Monitor routes

The four GET routes below serve the monitor panel and diagnostics.
They answer with the process API key, like every other `/v1` route.

`GET /v1/system` reports build and process facts: crate version, native backends compiled in, host memory, and CPU threads. It also lists engines with their states, resolved context limits, and retained effective settings.
The serving process does not probe devices, because a failed device query could latch the backend.
For device facts, run the short-lived `enginesrv doctor` command instead.

`GET /v1/telemetry` reports GPU rows read from `nvidia-smi`, run as a subprocess: index, name, GPU utilization, memory used and total in bytes, temperature, and power draw.
The server holds a snapshot for 2 seconds.
With no GPU tool, or after a failed run, the route answers `200` with empty `gpus` and an `error` string. The absence is data for the panel, not a request failure.
A field that `nvidia-smi` prints but the server cannot parse stays `null`, never zero.

`GET /v1/recent-requests` answers the last 64 request-log rows, newest first, as `{"requests": [...]}`.
Each row keeps the allowlisted keys of the request log (request id, endpoint, engine, token counts, timings). Every row includes every key, with `null` for missing values.
`count_tokens` lines never land in the ring.
There is no filtering and no aggregation: the panel computes its own figures.

`GET /v1/route-trace` answers the routing stage's ring of the last 16 decode steps, process wide.
The stage is opt-in at process level only: start the engine with `Q36_ROUTE_STAGE=1`.
With the flag off the route answers `404` with the enabling hint, never an empty packet, so a reader cannot mistake off for no activity.
Steps are oldest first. Each layer entry carries the expert `ids` (u16, where 65535 marks a router fault), `weights`, and shared `gate`.
The reply carries `Cache-Control: no-store`.

## Errors

Errors have the form `{"error": {"message": "...", "code": "..."}}`.
The Messages routes use the Anthropic error format instead.
OpenAI-style routes (`/v1/chat/completions`, `/v1/completions`, and streams) include `usage` beside `error` when a failure occurs after the request starts running. That `usage` records the accepted prompt tokens and the tokens generated so far, without subtracting any held-back stop tail. A prefill refusal reports the accepted prompt with 0 completion tokens. `reused_tokens` is 0 on most failures, because only the `Finished` event reports a prefix hit. In that case, `event: error` carries the stream's only usage report with no `[DONE]`. Validation errors and Messages-route errors carry no `usage`.

| Status | Code | Meaning |
| --- | --- | --- |
| 400 | `bad_request` | The request has invalid syntax or asks for an unsupported option. |
| 400 | `context_length_exceeded` | The prompt does not fit the engine's context limit. |
| 401 | `unauthorized` | Missing or wrong API key. |
| 404 | `not_found` | Unknown route, model or engine. |
| 409 | `memory_plan`, `context_limit` | The engine cannot fit the request in its memory plan or context. |
| 422 | `checkpoint_failure` | The engine's checkpoint failed to load or read. |
| 500 | `backend_error` | The engine failed while running the request, or is in state `error` or `native-failed`. |
| 503 | `overloaded` | The engine's queue is full, or the engine is not available. |
| 503 | `not_ready` | Returned only by `/ready`. |

## Engine states

`/v1/models` and the `not_ready` message report each engine as `loading`, `ready`, `busy`, `error` (it failed to load), `stopped`, or `native-failed` (the GPU backend failed and the engine takes no more requests).

If every engine fails to load, the server exits with a non-zero code instead of staying up. It prints each engine's cause on standard error.
So `/ready` answers `503` while an engine is still loading. A failure after a successful start shows as `native-failed`, not `error`.
