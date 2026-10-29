# Connect a client

`enginesrv` speaks the OpenAI chat completions API and the Anthropic Messages API, so most SDKs and tools work once you change their base URL and key.
The examples use the address and model name of the shipped Qwen3.6 configs: `http://127.0.0.1:18791` and `fifi-q36`.

## Find the model name

```sh
curl http://127.0.0.1:18791/v1/models -H "Authorization: Bearer $ENGINESRV_API_KEY"
```

Each entry's `id` is a name you can send as `model`. It is the engine's `model` key, or the family name when the key is not set.

## OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18791/v1", api_key="my-secret-key")

reply = client.chat.completions.create(
    model="fifi-q36",
    messages=[{"role": "user", "content": "Write a haiku about cats."}],
)
print(reply.choices[0].message.content)
```

Add `stream=True` to receive the reply as it is generated.

Tools that use the OpenAI SDK usually read `OPENAI_BASE_URL` and `OPENAI_API_KEY`:

```sh
export OPENAI_BASE_URL=http://127.0.0.1:18791/v1
export OPENAI_API_KEY=my-secret-key
```

## Anthropic SDK

```python
import anthropic

client = anthropic.Anthropic(base_url="http://127.0.0.1:18791", api_key="my-secret-key")

message = client.messages.create(
    model="fifi-q36",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a haiku about cats."}],
)
print(message.content[0].text)
```

The SDK sends the key in the `x-api-key` header, which the `/v1/messages` routes accept.
Tools built on it usually read `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY`.

## Turn thinking on or off

Qwen3.6 can think before it answers. The chat template decides by default.

- OpenAI API: send `"chat_template_kwargs": {"enable_thinking": false}`, or the server's own field `"x-thinking": "off"` (`"on"` and `"default"` also work). With the Python SDK, pass them in `extra_body`.
- Anthropic API: send `"thinking": {"type": "disabled"}`, or `"enabled"`.

The thinking text comes back separately: in `reasoning_content` on the OpenAI API, and in `thinking` blocks on the Anthropic API.

## Use tools

Send tools in the usual format of each API. The model decides when to call them.
`tool_choice` accepts `auto` and `none` on both APIs. The server cannot force a call, so `required`, a named function, `any` and `tool` return `400`.

## Pick an engine

When several engines serve the same model name, the server uses the first one in the config.
To pick one by its `id`, add `"x-engine": "q36"` to the request body (in `extra_body` with the OpenAI SDK).

## What is not supported

- `response_format` of `json_object` or `json_schema` returns `400`; the server has no constrained decoding.
- `x-session` returns `404`; the server keeps no named sessions.

See the [HTTP API reference](../reference/http-api.md) for every field.
