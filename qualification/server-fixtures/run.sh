#!/bin/bash
# Client fixtures for enginesrv on the fake engine: curl cases that follow docs/api.md and,
# when the pinned OpenAI SDK is installed, the same flows through the SDK.
# Usage: run.sh [--write-expected] [--allow-sdk-skip] [PATH-TO-ENGINESRV]
# The script needs no network: the server listens on 127.0.0.1 only.
# Exit codes: 0 all passed, 1 a check failed, 2 setup failed, 3 the SDK check was skipped.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WRITE=0
ALLOW_SKIP=0
BIN=""
for arg in "$@"; do
  case "$arg" in
    --write-expected) WRITE=1 ;;
    --allow-sdk-skip) ALLOW_SKIP=1 ;;
    *) BIN="$arg" ;;
  esac
done
BIN="${BIN:-${ENGINESRV:-}}"
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
  echo "setup: pass the enginesrv binary as an argument or set ENGINESRV" >&2
  exit 2
fi
if [ -z "${TMPDIR:-}" ]; then
  echo "setup: set TMPDIR to the task scratch directory" >&2
  exit 2
fi
WORK="$(mktemp -d "$TMPDIR/server-fixtures.XXXXXX")" || {
  echo "setup: mktemp failed under $TMPDIR" >&2
  exit 2
}
KEY="sk-fixture-key-31337"
export ENGINESRV_FIXTURE_KEY="$KEY"
cat > "$WORK/config.json" <<CONFIG
{
  "listen": "127.0.0.1:0",
  "api_key_env": "ENGINESRV_FIXTURE_KEY",
  "engines": [{
    "id": "e1", "model": "demo", "family": "fake",
    "model_dir": "unused",
    "tokenizer_dir": "$ROOT/engine/src/server/testdata/tokenizer",
    "fake": {"reply": "Hello from the fake engine."}
  }, {
    "id": "e2", "model": "toolbot", "family": "fake",
    "model_dir": "unused",
    "tokenizer_dir": "$ROOT/engine/src/server/testdata/tokenizer",
    "fake": {"reply": "<think>I need the sum.</think>Adding now.<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>"}
  }]
}
CONFIG
"$BIN" --config "$WORK/config.json" > "$WORK/stdout.txt" 2> "$WORK/stderr.txt" &
SERVER=$!
# Stop and reap the server on every exit path, including an interrupt or a SIGTERM.
stop_server() {
  if [ -n "$SERVER" ]; then
    kill "$SERVER" 2> /dev/null
    wait "$SERVER" 2> /dev/null
    SERVER=""
  fi
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ADDRESS=""
for _ in $(seq 1 100); do
  ADDRESS="$(sed -n 's/^enginesrv listening on //p' "$WORK/stdout.txt" | head -1)"
  [ -n "$ADDRESS" ] && break
  sleep 0.1
done
if [ -z "$ADDRESS" ]; then
  echo "setup: enginesrv printed no address" >&2
  cat "$WORK/stderr.txt" >&2
  exit 2
fi
URL="http://$ADDRESS"
PASSED=0
FAILED=0
OUT="$WORK/actual"
mkdir -p "$OUT" "$HERE/expected"

normalize() {
  sed -E 's/"id":"chatcmpl-[0-9a-f]+"/"id":"ID"/g; s/"created":[0-9]+/"created":0/g; s/"call_[0-9a-f]+_([0-9]+)"/"call_ID_\1"/g'
}

# check NAME METHOD PATH [BODY] [AUTH]: one curl case. AUTH is "none" to send no key.
check() {
  local name="$1" method="$2" path="$3" body="${4:-}" auth="${5:-key}"
  local args=(-sS -N --connect-timeout 5 --max-time 30 -X "$method" -o "$WORK/body.txt" -w '%{http_code} %{content_type}' "$URL$path")
  [ "$auth" = "key" ] && args+=(-H "Authorization: Bearer $KEY")
  [ -n "$body" ] && args+=(-H 'Content-Type: application/json' -d "$body")
  local meta
  meta="$(curl "${args[@]}")" || { echo "FAIL $name: curl failed"; FAILED=$((FAILED + 1)); return; }
  { echo "$meta"; normalize < "$WORK/body.txt"; } > "$OUT/$name.txt"
  if [ "$WRITE" = 1 ]; then
    cp "$OUT/$name.txt" "$HERE/expected/$name.txt"
    echo "WROTE $name"
  elif diff -u "$HERE/expected/$name.txt" "$OUT/$name.txt" > "$WORK/diff.txt"; then
    echo "PASS $name"
    PASSED=$((PASSED + 1))
  else
    echo "FAIL $name"
    cat "$WORK/diff.txt"
    FAILED=$((FAILED + 1))
  fi
}

USER_HI='{"role":"user","content":"hi"}'
ADD_TOOL='{"type":"function","function":{"name":"add","description":"Adds two integers","parameters":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}}}'
check health GET /health "" none
check ready GET /ready "" none
check models GET /v1/models
check models_without_key GET /v1/models "" none
check chat POST /v1/chat/completions "{\"model\":\"demo\",\"messages\":[$USER_HI]}"
check chat_stream POST /v1/chat/completions "{\"model\":\"demo\",\"stream\":true,\"messages\":[$USER_HI]}"
check chat_max_tokens POST /v1/chat/completions "{\"model\":\"demo\",\"max_tokens\":3,\"messages\":[$USER_HI]}"
check chat_stop POST /v1/chat/completions "{\"model\":\"demo\",\"stop\":[\"fake\"],\"messages\":[$USER_HI]}"
check chat_x_engine POST /v1/chat/completions "{\"x-engine\":\"e1\",\"messages\":[$USER_HI]}"
check chat_tool_round_trip POST /v1/chat/completions '{"model":"demo","tools":[{"type":"function","function":{"name":"add","description":"Adds numbers","parameters":{"type":"object"},"strict":true}}],"tool_choice":"auto","messages":[{"role":"user","content":"add 1 and 2"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"add","arguments":"{\"a\":1,\"b\":2}"}}]},{"role":"tool","tool_call_id":"call_1","content":"3"}]}'
check chat_toolbot POST /v1/chat/completions "{\"model\":\"toolbot\",\"tools\":[$ADD_TOOL],\"messages\":[$USER_HI]}"
check chat_toolbot_stream POST /v1/chat/completions "{\"model\":\"toolbot\",\"stream\":true,\"tools\":[$ADD_TOOL],\"messages\":[$USER_HI]}"
check chat_toolbot_without_tools POST /v1/chat/completions "{\"model\":\"toolbot\",\"messages\":[$USER_HI]}"
check chat_without_key POST /v1/chat/completions "{\"model\":\"demo\",\"messages\":[$USER_HI]}" none
check chat_max_tokens_zero POST /v1/chat/completions "{\"model\":\"demo\",\"max_tokens\":0,\"messages\":[$USER_HI]}"
check chat_unknown_model POST /v1/chat/completions "{\"model\":\"other\",\"messages\":[$USER_HI]}"
check chat_session POST /v1/chat/completions "{\"model\":\"demo\",\"x-session\":\"s1\",\"messages\":[$USER_HI]}"
check chat_no_route POST /v1/chat/completions "{\"messages\":[$USER_HI]}"
check chat_json_object POST /v1/chat/completions "{\"model\":\"demo\",\"response_format\":{\"type\":\"json_object\"},\"messages\":[$USER_HI]}"
check chat_tool_choice_required POST /v1/chat/completions "{\"model\":\"demo\",\"tool_choice\":\"required\",\"messages\":[$USER_HI]}"
check chat_stream_unknown_model POST /v1/chat/completions "{\"model\":\"other\",\"stream\":true,\"messages\":[$USER_HI]}"
check unknown_route GET /v1/engines

SDK="not run"
if python3 -I -c 'import openai, sys; sys.exit(0 if openai.__version__ == "2.6.1" else 1)' 2> /dev/null; then
  if python3 -I "$HERE/sdk_check.py" "$ADDRESS" "$KEY"; then SDK="PASSED"; else SDK="FAILED"; FAILED=$((FAILED + 1)); fi
else
  SDK="SKIPPED"
  echo "SKIPPED openai-sdk: openai==2.6.1 is not installed for python3." >&2
  echo "  Create the pinned environment from qualification/openai-sdk-fixture/requirements.txt and run again." >&2
fi

kill -TERM "$SERVER"
for _ in $(seq 1 100); do
  kill -0 "$SERVER" 2> /dev/null || break
  sleep 0.1
done
if kill -0 "$SERVER" 2> /dev/null; then
  echo "FAIL shutdown: enginesrv did not exit after SIGTERM"
  kill -KILL "$SERVER"
  FAILED=$((FAILED + 1))
elif wait "$SERVER"; then
  echo "PASS shutdown: exit code 0"
  PASSED=$((PASSED + 1))
  SERVER=""
else
  echo "FAIL shutdown: exit code $?"
  FAILED=$((FAILED + 1))
  SERVER=""
fi
if grep -q "$KEY" "$WORK/stdout.txt" "$WORK/stderr.txt"; then
  echo "FAIL log: the key appears in the server output"
  FAILED=$((FAILED + 1))
else
  echo "PASS log: the key is not in the server output"
  PASSED=$((PASSED + 1))
fi

echo "RESULT curl-and-process: $PASSED passed, $FAILED failed; openai-sdk: $SDK"
if [ "$FAILED" -ne 0 ]; then exit 1; fi
if [ "$SDK" = "SKIPPED" ] && [ "$ALLOW_SKIP" = 0 ]; then exit 3; fi
exit 0
