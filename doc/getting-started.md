# Getting started

In this tutorial you build fifi, start its server, and send it your first requests.
You do not need a GPU or a model download: you will use the built-in `fake` engine, which answers every request with a fixed sentence.
Once this works, the same steps run a real model.

## What you need

- Linux or macOS with a terminal.
- `git`, `curl` and a C compiler. On macOS, run `xcode-select --install`; on Debian or Ubuntu, `sudo apt install build-essential`.
- A few GB of free disk space for the build.

## 1. Install Rust

fifi is written in Rust. Install the Rust toolchain with `rustup`:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Accept the defaults, then open a new terminal so that `cargo` is on your `PATH`.
The repository pins the exact Rust version it needs; `cargo` downloads it on first use.

## 2. Get the source and build

```sh
git clone https://github.com/zbrdc/fifi.git
cd fifi/engine
cargo build --release --bin enginesrv
```

This builds without a GPU backend, which is all the `fake` engine needs.
When it finishes, the server is at `target/release/enginesrv`.

## 3. Write a config file

The server reads one JSON file that lists its engines.
Create `fifi.json` in the `engine` directory:

```json
{
  "listen": "127.0.0.1:18791",
  "api_key_env": "FIFI_API_KEY",
  "engines": [
    {
      "id": "demo",
      "family": "fake",
      "model_dir": "src/server/testdata/tokenizer",
      "tokenizer_dir": "src/server/testdata/tokenizer"
    }
  ]
}
```

- `listen` is the address of the server. It must be on this machine (`127.0.0.1` or `::1`).
- `api_key_env` names the environment variable that holds the API key clients must send.
- `engines` lists the models to serve. `fake` needs only a tokenizer; the repository ships a small one for tests.

## 4. Start the server

Choose an API key, put it in the variable you named, and start the server:

```sh
export FIFI_API_KEY=my-secret-key
target/release/enginesrv --config fifi.json
```

The server prints:

```text
engine demo: sampling defaults temperature 0.7, top_p 1, top_k 0 from src/server/testdata/tokenizer/generation_config.json
warm turns: ON (default; ENGINESRV_WARM_TURNS=0 disables)
enginesrv listening on 127.0.0.1:18791
```

Leave it running and open a second terminal for the next steps.
The server also prints one JSON line for each request it handles.

## 5. Check that it is ready

```sh
curl http://127.0.0.1:18791/ready
```

```json
{"status":"ready"}
```

`/ready` answers once at least one engine has loaded. It is one of the two routes that need no API key; the other is `/health`.

## 6. Send a chat request

```sh
curl http://127.0.0.1:18791/v1/chat/completions \
  -H "Authorization: Bearer my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "fake", "messages": [{"role": "user", "content": "Hello!"}]}'
```

The reply follows the OpenAI chat completions format:

```json
{
  "id": "chatcmpl-0000000000000001",
  "object": "chat.completion",
  "created": 1793230956,
  "model": "fake",
  "engine": "demo",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello from the fake engine."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 25, "completion_tokens": 28, "total_tokens": 53, "reused_tokens": 0, "prompt_tokens_details": {"cached_tokens": 0}}
}
```

`model` is the name clients use to pick an engine; it defaults to the family name.
Without the `Authorization` header, the server answers `401`.

## 7. Stream the reply

Add `"stream": true` to receive the reply as it is generated:

```sh
curl -N http://127.0.0.1:18791/v1/chat/completions \
  -H "Authorization: Bearer my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "fake", "stream": true, "messages": [{"role": "user", "content": "Hello!"}]}'
```

The server sends one `data:` line per piece of text and ends with `data: [DONE]`:

```text
data: {"id":"chatcmpl-0000000000000002",...,"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}

data: {"id":"chatcmpl-0000000000000002",...,"choices":[{"index":0,"delta":{"content":"H"}}]}

...

data: [DONE]
```

## 8. Use the Anthropic-style API

The same engine also answers the Messages API. It accepts the key in `x-api-key`:

```sh
curl http://127.0.0.1:18791/v1/messages \
  -H "x-api-key: my-secret-key" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model": "fake", "max_tokens": 64, "messages": [{"role": "user", "content": "Hello!"}]}'
```

```json
{"id":"msg_0000000000000003","type":"message","role":"assistant","model":"fake","content":[{"type":"text","text":"Hello from the fake engine."}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":25,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":28}}
```

## 9. Stop the server

Press `Ctrl+C` in the terminal where the server runs.

## What you did

You built `enginesrv`, described an engine in a config file, protected the server with an API key, and talked to it through both the OpenAI-style and the Anthropic-style APIs.
Every real model works the same way; only the engine entry and the build change.

## Next steps

- Run a real model: [Qwen3.6](how-to/qwen36.md), [Flint](how-to/flint.md) or [Nemotron 3.5 Super](how-to/nemotron.md).
- Point an SDK or another tool at the server: [Connect a client](how-to/clients.md).
- See every config key: [Configuration reference](reference/configuration.md).
