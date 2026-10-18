# OpenAI SDK fixture check

This directory holds the offline check for task P1-HTTP-READINESS-01.
It parses recorded payloads through the pinned `openai` SDK model classes.

## Reproduce

```sh
uv venv "$SCRATCH/venv" --python 3.13
uv pip install --python "$SCRATCH/venv/bin/python" --require-hashes -r requirements.txt
"$SCRATCH/venv/bin/python" check_fixtures.py
"$SCRATCH/venv/bin/openapi-spec-validator" ../../api/openapi.yaml
```

- `requirements.in` names the pinned top-level packages. `jsonschema` and `pyyaml` were already pinned as transitive packages in `requirements.txt`.
- `requirements.txt` pins every package by version and hash. `uv pip compile --generate-hashes` produced it.
- `fixtures/contract_*.json` use only the fields that `api/openapi.yaml` defines today.
  `check_fixtures.py` runs `jsonschema` on each one and its `api/openapi.yaml` schema, and records the errors under `contract_schema_errors`.
- `fixtures/proposed_*.json` add the fields that the SDK requires.
- `expected_results.json` is the recorded output. The script exits 1 on any difference.
- The script exits 1 when a `contract_*.json` fixture fails its schema, also with `--write-expected`.
- The check needs no network access. The client path uses an `httpx` mock transport.
- Run `check_fixtures.py --write-expected` only after a deliberate fixture or version change.

The fixtures cover `openai` 2.6.1 only. They do not cover other clients.
