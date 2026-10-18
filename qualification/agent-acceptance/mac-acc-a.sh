#!/usr/bin/env bash
# Mac side of the AGENT-1 acceptance runs. It also runs on Linux as a rehearsal (REHEARSE=1).
# Usage: [ENGINE=flash|q36] mac-acc-a.sh serve <scratch dir> <worktree>
#   serve  build and start enginesrv, wait for /ready, create <scratch>/READY, wait for
#          <scratch>/STOP (checked during the build, the load, and the run), stop the server, write
#          the evidence. The client runs on the launcher host through an ssh tunnel, never here:
#          the model's code runs only in the container sandbox of the launcher host.
# stdin, first line: the bearer key. It is never written to disk or to a log.
# Environment (all optional):
#   PORT 18791, READY_TIMEOUT 1800 s, STOP_TIMEOUT 900 s, SAMPLE_SECONDS 10,
#   ENGINE flash (default) or q36. flash serves Qwen3.8-Flash-Next as model id `flash`. q36 serves
#   Qwen3.6-35B-A3B (family qwen3_6_35b_a3b) as model id `fifi-q36` from the pinned checkpoint
#   Q36_DIR below. Before the server starts, the script hashes config.json,
#   model.safetensors.index.json, tokenizer.json, and chat_template.jinja of that checkpoint
#   against q36-pins.sha256 (beside this script). A mismatch or a missing file stops with status 88.
#   FLASH_NEXT_ARTIFACT, FLASH_NEXT_ARTIFACT_MANIFEST, FLASH_NEXT_CACHE_LIMIT (the Mac defaults below;
#   they fill the flash_next object of config.json),
#   Q36_CACHE_LIMIT (q36 only; it cannot change the served config: a value that differs from the
#   shipped file stops with status 64. The qwen36 object comes from engine/config/qwen36-mac-studio-36gb.json
#   in the worktree),
#   Q36_DIR_OVERRIDE and Q36_PINS_OVERRIDE (REHEARSE=1 only; a real run always uses the pinned constants),
#   REHEARSE=1 with SERVER_CMD="<command that serves on $PORT>" and SERVER_MATCH="<text in its ps line>".
# The script starts one server and stops only that server, by the PID it recorded. It never uses
# SIGKILL. It never touches port 8788 (laya3). A serving run does not qualify Phase 0B or Phase 1B.
# Bash 3.2 compatible (macOS).
set -u
MODE=${1:-}; MS=${2:-}; MW=${3:-}
if [ -z "$MODE" ] || [ -z "$MS" ] || [ -z "$MW" ]; then
  echo "usage: mac-acc-a.sh serve <scratch dir> <worktree>" >&2; exit 64
fi
case "$MODE" in serve) ;; *) echo "unknown mode $MODE; only serve exists (the client runs on the launcher host)" >&2; exit 64;; esac
# check_port_engine: refuses port 8788 (laya3) and an unknown ENGINE. It runs twice: on the
# arguments and the environment, and again after env.sh was sourced, because env.sh can set both.
check_port_engine() {
  if [ "$PORT" = "8788" ]; then echo "port 8788 belongs to laya3" >&2; exit 64; fi
  case "$ENGINE" in flash|q36) ;; *) echo "ENGINE must be flash or q36" >&2; exit 64;; esac
}
PORT=${PORT:-18791}
READY_TIMEOUT=${READY_TIMEOUT:-1800}
STOP_TIMEOUT=${STOP_TIMEOUT:-900}
SAMPLE_SECONDS=${SAMPLE_SECONDS:-10}
REHEARSE=${REHEARSE:-0}
ENGINE=${ENGINE:-flash}
check_port_engine
AH="$MW/qualification/agent-acceptance"
EV="$MS/evidence"
IFS= read -r ENGINESRV_API_KEY || true
if [ -z "$ENGINESRV_API_KEY" ]; then echo "the first line of stdin must be the bearer key" >&2; exit 64; fi
export ENGINESRV_API_KEY
mkdir -p "$EV"

hashfile() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(stamp)] $*"; }
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
swap_used() { sysctl -n vm.swapusage 2>/dev/null | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p'; }
snapshot() {
  # $1 = before|after
  {
    echo "time $(stamp)"
    echo "swap_used_mb $(swap_used)"
    if command -v vm_stat >/dev/null 2>&1; then vm_stat; fi
    if command -v lsof >/dev/null 2>&1; then
      echo "listen_$PORT: $(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | tail -n +2 | tr '\n' ';')"
      echo "listen_8788: $(lsof -nP -iTCP:8788 -sTCP:LISTEN 2>/dev/null | tail -n +2 | tr '\n' ';')"
    fi
    echo "splash: $(pgrep -fl splash 2>/dev/null | tr '\n' ';')"
  } > "$EV/system-$1.txt"
}

SERVER_PID=""; SAMPLER_PID=""; STOPPED=0
stop_server() {
  # Stops the server this script started, by its recorded PID. Never SIGKILL.
  if [ -z "$SERVER_PID" ] || [ "$STOPPED" = 1 ]; then return 0; fi
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    if ps -p "$SERVER_PID" -o command= 2>/dev/null | grep -q -- "$SERVER_MATCH"; then
      say "SIGTERM to server PID $SERVER_PID"
      kill -TERM "$SERVER_PID"
      waited=0
      while kill -0 "$SERVER_PID" 2>/dev/null && [ "$waited" -lt "$STOP_TIMEOUT" ]; do sleep 2; waited=$((waited + 2)); done
      if kill -0 "$SERVER_PID" 2>/dev/null; then
        say "STOP-FAILED: server PID $SERVER_PID still runs after ${STOP_TIMEOUT}s; it is left running"
        echo "$SERVER_PID" > "$EV/STOP-FAILED"
        return 95
      fi
      say "server stopped after ${waited}s"
    else
      say "PID $SERVER_PID is not the server this script started; it is not touched"
    fi
  fi
  STOPPED=1
  return 0
}
cleanup() {
  if [ -n "$SAMPLER_PID" ]; then kill "$SAMPLER_PID" 2>/dev/null; fi
  stop_server
}
trap cleanup EXIT

# --- identity and inputs -------------------------------------------------------------------
if [ "$REHEARSE" = 1 ]; then
  : "${SERVER_CMD:?SERVER_CMD is required with REHEARSE=1}"
  SERVER_MATCH=${SERVER_MATCH:-fake_server}
  ART=${FLASH_NEXT_ARTIFACT:-/nonexistent}; MAN=${FLASH_NEXT_ARTIFACT_MANIFEST:-/nonexistent}
  CACHE=${FLASH_NEXT_CACHE_LIMIT:-1073741824}
else
  # shellcheck disable=SC1091
  REHEARSE_BEFORE_ENV=$REHEARSE
  source "$MS/env.sh"
  # env.sh is a user file. It must not turn a real run into a rehearsal (that enables the Q36
  # overrides), reach port 8788, or select an unknown engine.
  REHEARSE=$REHEARSE_BEFORE_ENV
  check_port_engine
  ART=${FLASH_NEXT_ARTIFACT:-/Users/dan/fifi/scrch/e9be673/artifact}
  MAN=${FLASH_NEXT_ARTIFACT_MANIFEST:-$ART/download-manifest.json}
  CACHE=${FLASH_NEXT_CACHE_LIMIT:-1073741824}
  export FLASH_NEXT_ARTIFACT="$ART" FLASH_NEXT_ARTIFACT_MANIFEST="$MAN" FLASH_NEXT_CACHE_LIMIT="$CACHE"
  # The server reads the paths from config.json. These variables serve the Mac tests and the identity record.
  export INFERENCE_FLASH_NEXT_TOKENIZER_DIR="$ART"
  export CARGO_TARGET_DIR="$MS/target"
  SERVER_MATCH="enginesrv"
fi
if port_busy "$PORT"; then say "port $PORT is in use; this script will not stop another process"; exit 70; fi
# The Q36 checkpoint and its pins are constants. They are assigned here, after env.sh was sourced,
# so that nothing env.sh sets can change them (the selftest checks this). Only a rehearsal may name
# another directory and another pin file (Q36_DIR_OVERRIDE, Q36_PINS_OVERRIDE).
# The pinned Q36 checkpoint is the checkpoint of the Q36-F3c smoke run on the Mac.
Q36_PINNED_DIR=/Users/dan/edge0/models/edge0-35b
Q36_MODEL_ID=fifi-q36
Q36_FILES="config.json model.safetensors.index.json tokenizer.json chat_template.jinja"
Q36_DIR=$Q36_PINNED_DIR; Q36_PINS=$AH/q36-pins.sha256
if [ "$REHEARSE" = 1 ]; then
  Q36_DIR=${Q36_DIR_OVERRIDE:-$Q36_DIR}; Q36_PINS=${Q36_PINS_OVERRIDE:-$Q36_PINS}
fi
# The q36 engine serves the qwen36 object of the shipped config, copied as is from the source tree
# this run builds ($MW), never from a host copy. Nothing from env.sh or the environment can change
# it. Q36_CACHE_LIMIT is refused when it is set to anything but the value of the shipped file.
Q36_SHIPPED=$MW/engine/config/qwen36-mac-studio-36gb.json
Q36_OBJECT=""; Q36_CACHE=""; Q36_SHIPPED_SHA=""
if [ "$ENGINE" = q36 ]; then
  Q36_PARSED=$(python3 -I - "$Q36_SHIPPED" <<'PY'
import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
(engine,) = config["engines"]
obj = engine["qwen36"]
for key in ("cache_limit_bytes", "prefill_budget_bytes", "prefill_session_reserve_bytes"):
    assert type(obj[key]) is int and obj[key] > 0, key
assert "prefill_chunk" not in obj, "prefill_chunk must stay absent"
print(obj["cache_limit_bytes"])
print(json.dumps(obj, separators=(",", ":")))
PY
  ) || { echo "cannot read the qwen36 object of $Q36_SHIPPED" >&2; exit 65; }
  Q36_CACHE=$(printf '%s\n' "$Q36_PARSED" | sed -n 1p); Q36_OBJECT=$(printf '%s\n' "$Q36_PARSED" | sed -n 2p)
  Q36_SHIPPED_SHA=$(hashfile "$Q36_SHIPPED" | cut -d' ' -f1)
  if [ -n "${Q36_CACHE_LIMIT:-}" ] && [ "$Q36_CACHE_LIMIT" != "$Q36_CACHE" ]; then
    echo "Q36_CACHE_LIMIT=$Q36_CACHE_LIMIT differs from the shipped config ($Q36_CACHE); the shipped config is served as is" >&2; exit 64
  fi
fi

# q36_check_pins: hashes the four checkpoint files and compares each with the pin file. It prints
# one `artifact_file <name> <hash>` line per file (the hash is `missing` for an absent file) and
# returns 1 on any mismatch. Nothing starts before it returns 0.
q36_check_pins() {
  local name want got bad=0
  for name in $Q36_FILES; do
    want=$(awk -v n="$name" '$2 == n { print $1 }' "$Q36_PINS" 2>/dev/null)
    if [ -f "$Q36_DIR/$name" ]; then got=$(hashfile "$Q36_DIR/$name" | cut -d' ' -f1); else got=missing; fi
    echo "artifact_file $name $got"
    if [ -z "$want" ] || [ "$got" != "$want" ]; then
      echo "Q36 PIN MISMATCH: $name has ${got}, the pin is ${want:-absent}" >&2
      bad=1
    fi
  done
  return $bad
}
if [ "$ENGINE" = q36 ]; then
  Q36_IDENTITY=$(q36_check_pins) || {
    say "the Q36 checkpoint $Q36_DIR does not match q36-pins.sha256; nothing is started"
    printf '%s\n' "$Q36_IDENTITY" > "$EV/q36-pin-check.txt"; exit 88
  }
fi
{
  echo "time $(stamp)"; echo "mode $MODE"; echo "engine $ENGINE"; echo "host $(hostname)"; echo "uname $(uname -a)"
  echo "port $PORT"; echo "rehearse $REHEARSE"
  echo "head $(git -C "$MW" rev-parse HEAD 2>/dev/null)"
  echo "status_porcelain $(git -C "$MW" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$ENGINE" = q36 ]; then
    echo "model_id $Q36_MODEL_ID"; echo "model_dir $Q36_DIR"
    printf '%s\n' "$Q36_IDENTITY"
    echo "cache_limit_bytes $Q36_CACHE"
    echo "shipped_config $Q36_SHIPPED"; echo "shipped_config_sha256 $Q36_SHIPPED_SHA"; echo "qwen36_object $Q36_OBJECT"
  elif [ "$REHEARSE" != 1 ]; then
    echo "manifest_sha256 $(hashfile "$MAN" | cut -d' ' -f1)"
    for f in tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json config.json model.safetensors.index.json; do
      echo "artifact_file $f $(hashfile "$ART/$f" | cut -d' ' -f1)"
    done
    echo "cache_limit_bytes $CACHE"
  fi
} > "$EV/identity.txt"
snapshot before

# write_config: the enginesrv config follows engine/src/server/config.rs (EngineConfig,
# deny_unknown_fields), engine/src/family/flash_next/settings.rs (FlashOptions) and
# engine/src/family/qwen36/settings.rs (Q36Options). `family` is the family name. `model` is set so
# that GET /v1/models data[].id is the id of the client and of pi: `flash`, or `fifi-q36`. The
# prefill chunk keeps its default (q36: the shipped qwen36 object has none). The file holds no key: only the variable name.
write_config() {
  if [ "$ENGINE" = q36 ]; then
    cat > "$MS/config.json" <<CONFIG
{"listen":"127.0.0.1:$PORT","api_key_env":"ENGINESRV_API_KEY","engines":[{"id":"q36","family":"qwen3_6_35b_a3b","model":"$Q36_MODEL_ID","model_dir":"$Q36_DIR","tokenizer_dir":"$Q36_DIR","qwen36":$Q36_OBJECT}]}
CONFIG
  else
    cat > "$MS/config.json" <<CONFIG
{"listen":"127.0.0.1:$PORT","api_key_env":"ENGINESRV_API_KEY","engines":[{"id":"flash","family":"qwen3_8_flash_next","model":"flash","model_dir":"$ART","tokenizer_dir":"$ART","flash_next":{"manifest":"$MAN","cache_limit_bytes":$CACHE}}]}
CONFIG
  fi
  cp "$MS/config.json" "$EV/config.json"
}

# A rehearsal writes and keeps the same config (the fake server ignores it).
write_config

# --- build and start ---------------------------------------------------------------------------
if [ "$REHEARSE" != 1 ]; then
  say "building enginesrv"
  ( cd "$MW/engine" && exec cargo build --offline --release --features native-mlx --bin enginesrv ) > "$MS/build.log" 2>&1 &
  BUILD_PID=$!
  while kill -0 "$BUILD_PID" 2>/dev/null; do
    if [ -f "$MS/STOP" ]; then say "STOP during the build"; kill -TERM "$BUILD_PID" 2>/dev/null; wait "$BUILD_PID" 2>/dev/null; exit 87; fi
    sleep 2
  done
  wait "$BUILD_PID" || { say "build failed; see build.log"; cp "$MS/build.log" "$EV/"; exit 80; }
  BIN="$CARGO_TARGET_DIR/release/enginesrv"
  echo "binary_sha256 $(hashfile "$BIN" | cut -d' ' -f1)" >> "$EV/identity.txt"
  SERVER_CMD="$BIN --config $MS/config.json"
fi
# Standard error holds the request log (one JSON line per chat request: server.md "Request log")
# and startup errors. It goes to server.log, which the checkers read. Standard output goes apart.
: > "$MS/server.log"; : > "$MS/server.out"
say "starting the server"
nohup bash -c "exec $SERVER_CMD" > "$MS/server.out" 2> "$MS/server.log" < /dev/null &
SERVER_PID=$!
echo "$SERVER_PID" > "$MS/server.pid"
sleep 1
if ! kill -0 "$SERVER_PID" 2>/dev/null; then say "the server exited at once"; cp "$MS/server.log" "$MS/server.out" "$EV/"; exit 81; fi
( while kill -0 "$SERVER_PID" 2>/dev/null; do echo "$(date +%s) $(ps -o rss= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')"; sleep "$SAMPLE_SECONDS"; done ) > "$EV/rss.tsv" &
SAMPLER_PID=$!

waited=0; ready=0
while [ "$waited" -lt "$READY_TIMEOUT" ]; do
  if [ -f "$MS/STOP" ]; then say "STOP during the load"; exit 87; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then say "the server died while loading"; cp "$MS/server.log" "$MS/server.out" "$EV/"; exit 82; fi
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://127.0.0.1:$PORT/ready" 2>/dev/null || true)
  if [ "$code" = "200" ]; then ready=1; break; fi
  sleep 5; waited=$((waited + 5))
done
if [ "$ready" != 1 ]; then say "no /ready after ${READY_TIMEOUT}s"; cp "$MS/server.log" "$MS/server.out" "$EV/"; exit 83; fi
echo "ready_seconds $waited" >> "$EV/identity.txt"
say "ready after about ${waited}s"

# --- the run -------------------------------------------------------------------------------------
STATUS=0
touch "$MS/READY"; say "READY; waiting for $MS/STOP"
while [ ! -f "$MS/STOP" ] && kill -0 "$SERVER_PID" 2>/dev/null; do sleep 2; done
if [ ! -f "$MS/STOP" ]; then say "the server ended before STOP"; STATUS=86; fi
say "run status $STATUS"

# --- stop and collect -----------------------------------------------------------------------------
stop_server; STOP_STATUS=$?
kill "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""
snapshot after
cp "$MS/server.log" "$EV/server.log"; cp "$MS/server.out" "$EV/server.out"
peak=$(awk 'NF == 2 && $2 + 0 > m { m = $2 + 0 } END { print m + 0 }' "$EV/rss.tsv")
before=$(sed -n 's/^swap_used_mb //p' "$EV/system-before.txt" | head -1)
after=$(sed -n 's/^swap_used_mb //p' "$EV/system-after.txt" | head -1)
grown=$(awk -v b="${before:-0}" -v a="${after:-0}" 'BEGIN { print (a + 0 > b + 0) ? "yes" : "no" }')
{
  echo "mode $MODE"; echo "run_status $STATUS"; echo "stop_status $STOP_STATUS"
  echo "rss_peak_kb $peak"; echo "swap_used_mb_before ${before:-n/a}"; echo "swap_used_mb_after ${after:-n/a}"
  echo "swap_grew $grown"
  echo "scope structural result; not a speed claim; not Phase 0B or Phase 1B"
} > "$EV/summary.txt"
( cd "$EV" && find . -type f ! -path ./SHA256SUMS | sort | while IFS= read -r f; do hashfile "$f"; done > SHA256SUMS )
say "evidence in $EV"
if [ "$STOP_STATUS" != 0 ]; then exit "$STOP_STATUS"; fi
exit "$STATUS"
