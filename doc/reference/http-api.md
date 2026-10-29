# HTTP API

`enginesrv` serves the routes below on the `listen` address of its [configuration file](configuration.md).
The full contract is [`api/openapi.yaml`](../../api/openapi.yaml).

## Routes

| Method and path | Key needed | Purpose |
| --- | --- | --- |
| `GET /health` | no | Answers `{"status":"ok"}` while the process runs. |
| `GET /ready` | no | `200 {"status":"ready"}` when at least one engine is ready or busy; otherwise `503` with code `not_ready` and the state of each engine. |
| `GET /v1/models` | yes | Model names, each with the engines that serve it and their states. |
| `POST /v1/chat/completions` | yes | OpenAI-style chat completion, with or without streaming. |
| `POST /v1/messages` | yes | Anthropic-style Messages request, with or without streaming. |
| `POST /v1/messages/count_tokens` | yes | Token count of a Messages request's prompt. No generation runs. |

## Authentication

Send the API key as `Authorization: Bearer <key>`.
The `/v1/messages` routes also accept `x-api-key: <key>`; when that header is present it must match, and the bearer header is not checked.
A missing or wrong key returns `401`.

## Chat completions

Request fields:

| Field | Default | Notes |
| --- | --- | --- |
| `model` | | A name from `/v1/models`. The first engine serving it takes the request. |
| `messages` | required | |
| `stream` | `false` | Server-sent events, ending with `data: [DONE]`. |
| `max_tokens` | 2,048 | Lowered when the prompt plus `max_tokens` would pass the engine's context limit. |
| `temperature` | from the model's `generation_config.json`, else 0.7 | 0 to 2. `0` selects greedy decoding. |
| `top_p` | from `generation_config.json`, else 1.0 | 0 to 1. |
| `top_k` | from `generation_config.json`, else 0 | `0` turns top-k off. |
| `stop` | none | Stop strings, matched on the decoded text. |
| `tools` | none | OpenAI function tools. |
| `tool_choice` | `auto` | `auto` or `none`. `required` and a named function return `400`. |
| `response_format` | `text` | `json_object` and `json_schema` return `400`. |
| `x-thinking` | `default` | Server extension: `on`, `off` or `default` (the chat template decides). |
| `chat_template_kwargs` | none | Only `enable_thinking` is accepted: `true` acts as `x-thinking: on`, `false` as `off`. |
| `x-engine` | none | Server extension: the `id` of the engine to use. Overrides `model`. An unknown id returns `404`. |
| `x-session` | none | Reserved; returns `404`. |

The request needs `model` or `x-engine`.

The reply message has `content`, `tool_calls` when the model called a tool, and `reasoning_content` with the thinking text when there was any.
`finish_reason` is `stop`, `length` or `tool_calls`.

`usage` has `prompt_tokens`, `completion_tokens` and `total_tokens`, plus `reused_tokens` and `prompt_tokens_details.cached_tokens`: the prompt tokens served from a saved prefix.
`ENGINESRV_CACHE_USAGE=0` removes the last two.

## Messages

Request fields follow the Anthropic API: `model`, `max_tokens` and `messages` are required; `system`, `stop_sequences`, `temperature`, `top_p`, `top_k`, `stream`, `tools`, `tool_choice` and `metadata` are optional.
Sampling bounds and defaults are the same as for chat completions.

| Field | Notes |
| --- | --- |
| `thinking` | `{"type": "enabled"}` or `"adaptive"` turns thinking on, `"disabled"` turns it off. Without it, the chat template decides. `budget_tokens` is checked but does not limit thinking. |
| `tool_choice` | `auto` or `none`. `any` and `tool` return `400`. |
| `metadata` | Accepted and ignored. |

Content blocks in the reply are `text`, `thinking` (with an empty `signature`) and `tool_use`.
`usage.cache_read_input_tokens` counts prompt tokens served from a saved prefix, and `input_tokens` counts the rest. `cache_creation_input_tokens` is always 0.

## Errors

Errors have the form `{"error": {"message": "...", "code": "..."}}`.
The Messages routes use the Anthropic error format instead.

| Status | Code | Meaning |
| --- | --- | --- |
| 400 | `bad_request` | The request is malformed or asks for something the server does not support. |
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
