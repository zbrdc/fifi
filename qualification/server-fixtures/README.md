# Server client fixtures

These fixtures run `enginesrv` on its `fake` engine and call it with `curl` and, when it is installed, the pinned OpenAI SDK 2.6.1.
They need no network. The server listens on 127.0.0.1 only.
They give portable client evidence for the routes of the first server. They do not test a real model.

```sh
cd engine && cargo build --offline --bin enginesrv
export TMPDIR=<task scratch directory>
qualification/server-fixtures/run.sh <target>/debug/enginesrv
```

- `expected/` holds the recorded curl answers. The ids and the creation times are normalized.
- `run.sh --write-expected` records them again. Use it only after a deliberate change.
- `sdk_check.py` needs `openai==2.6.1`. Install it as `qualification/openai-sdk-fixture/README.md` describes.
- Without the SDK, `run.sh` prints `SKIPPED openai-sdk` and exits with code 3. `--allow-sdk-skip` changes the exit code to 0. The summary line still says `SKIPPED`.
- `run.sh` stops the server on every exit path, including SIGINT and SIGTERM. Each curl request has `--connect-timeout 5` and `--max-time 30`. Python runs with `-I`.
- `selftest.sh [PATH-TO-ENGINESRV]` sends SIGTERM to `run.sh` during a run. It fails when an `enginesrv` process of that run survives. `RUN_SH=<script>` selects another script for a mutation check.
