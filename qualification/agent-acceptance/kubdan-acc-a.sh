#!/usr/bin/env bash
# kubdan side of the ACC-A CUDA run (CUDA-SERVE gate G2: enginesrv built with --features
# native-cuda serves Qwen3.6-35B-A3B Q4_K_M from a GGUF on the RTX 3090). It also runs on any
# host as a rehearsal (REHEARSE=1) with a fake nvidia-smi, a fake cargo, and SERVER_CMD.
# Usage: [CAPTURE=0|1] [CONTEXT_LIMIT=16384] kubdan-acc-a.sh serve <scratch dir> <worktree>
#   serve  preflight the GPU, hash the model, build enginesrv, start it, wait for /ready, create
#          <scratch>/READY, wait for <scratch>/STOP (checked during the build, the load, and the
#          run), stop the server, write the evidence. The client runs on the launcher host
#          through an ssh tunnel, never here: the model's code runs only in the container sandbox
#          of the launcher host.
# stdin, first line: the bearer key. It is never written to disk or to a log, never passed in an
# argument (a fourth argument is refused), and never taken from the environment
# (an ENGINESRV_API_KEY that is already set is refused).
# Environment (all optional):
#   PORT 18792 (8788 belongs to laya3 and 18791 to the Mac; both are refused), READY_TIMEOUT 1800 s,
#   STOP_TIMEOUT 900 s, SAMPLE_SECONDS 10 (rss.tsv), VRAM_SECONDS 5 (vram.tsv), CAPTURE 0 (default)
#   or 1 (graph capture; a separate run each), CONTEXT_LIMIT 16384 (default).
# CUDA toolkit: it comes from the environment of kubdan. The script resolves it as CUDA_PATH when
# set, else /usr/local/cuda, else nvcc on PATH (the build image of the RD lanes ships CUDA 12.2),
# exports CUDA_PATH for the build, and fails with status 73 when nvcc is missing.
# The build uses ONLY the local NVMe of kubdan: the worktree under /media/dan/nvme/fifi/wrktr/runs
# and the scratch (target dir, TMPDIR, TMP, TEMP) under /media/dan/nvme/fifi/scrch/runs, never the
# NFS share (pinu's small root disk). Anything else is refused with status 75.
# Preflight (each stops with a distinct status and starts nothing): the GPU is free (nvidia-smi
# answers with a GPU, memory.used below 2,048 MiB, no compute process) status 71, no other
# enginesrv of this user listens on $PORT status 72 (any other listener: 70), the free disk of
# those roots is at least 30 GiB status 74.
# The model dir is the pinned constant Q36_CUDA_PINNED_DIR below. The GGUF (name, size, sha256, as
# the gate pins it) and the four tokenizer files are hashed against q36-cuda-pins.sha256 (beside
# this script) before the build; a mismatch or a missing file stops with status 88. Only a
# rehearsal (REHEARSE=1) may name another model dir, pin file, or GGUF size
# (Q36_CUDA_DIR_OVERRIDE, Q36_CUDA_PINS_OVERRIDE, Q36_CUDA_SIZE_OVERRIDE) and other NVMe roots
# (KUBDAN_SCRCH_ROOT, KUBDAN_WRKTR_ROOT, KUBDAN_MIN_FREE_MB); a real run always uses the constants.
# REHEARSE=1 needs SERVER_CMD (a command that serves on $PORT; the build step still runs, with a
# fake cargo) and SERVER_MATCH (default fake_server).
# ENGINESRV_BIN (optional): a binary built BEFORE the run, so that the build does not use up a
# time-limited GPU hold. A real run takes it only from PREBUILT_ROOT (/media/dan/nvme/fifi/scrch/
# prebuilt) and only with a file `<binary>.commit` that holds the commit of the worktree under
# test (status 76 otherwise); the run then skips the build and records the binary hash and
# commit. A rehearsal may name another root (KUBDAN_PREBUILT_ROOT).
# ENGINESRV_IMAGE (optional): serve from a container of that image, so that the CUDA runtime is the
# one the binary was built with. A real run accepts only the pinned build image
# (inference-cuda-build:cuda12.2-ubuntu22.04, status 75 otherwise). The server runs as
# `docker run --pull never --rm --name <unique> --network host --gpus all --cpus 8 --memory 16g
# --memory-swap 16g -e ENGINESRV_API_KEY -v /media/dan/nvme:/media/dan/nvme <image> timeout -s KILL
# <SERVER_LIFETIME, default 1740> <binary> --config <config>`: the key travels in the environment of
# the docker client by name only, never in an argument. The inner timeout bounds the container.
# The script stops the docker client by its recorded PID (SIGTERM, which the client forwards).
# The script starts one server and stops only that server, by the PID it recorded. It never uses
# SIGKILL. It never touches port 8788. A pass is a structural result; no speed claim.
set -u
MODE=${1:-}; MS=${2:-}; MW=${3:-}
if [ -z "$MODE" ] || [ -z "$MS" ] || [ -z "$MW" ] || [ $# -gt 3 ]; then
  echo "usage: kubdan-acc-a.sh serve <scratch dir> <worktree>   (the bearer key goes on stdin, never on a command line)" >&2; exit 64
fi
case "$MODE" in serve) ;; *) echo "unknown mode $MODE; only serve exists (the client runs on the launcher host)" >&2; exit 64;; esac
# check_port_engine: refuses port 8788 (laya3), port 18791 (the Mac), and an unknown ENGINE.
check_port_engine() {
  if [ "$PORT" = "8788" ]; then echo "port 8788 belongs to laya3" >&2; exit 64; fi
  if [ "$PORT" = "18791" ]; then echo "port 18791 is the Mac port; the kubdan engine never uses it" >&2; exit 64; fi
  case "$ENGINE" in q36cuda) ;; *) echo "ENGINE must be q36cuda" >&2; exit 64;; esac
}
PORT=${PORT:-18792}
READY_TIMEOUT=${READY_TIMEOUT:-1800}
STOP_TIMEOUT=${STOP_TIMEOUT:-900}
SAMPLE_SECONDS=${SAMPLE_SECONDS:-10}
VRAM_SECONDS=${VRAM_SECONDS:-5}
REHEARSE=${REHEARSE:-0}
ENGINE=${ENGINE:-q36cuda}
CAPTURE=${CAPTURE:-0}
CONTEXT_LIMIT=${CONTEXT_LIMIT:-16384}
check_port_engine
case "$CAPTURE" in 0|1) ;; *) echo "CAPTURE must be 0 or 1" >&2; exit 64;; esac
case "$CONTEXT_LIMIT" in ''|*[!0-9]*) echo "CONTEXT_LIMIT must be a positive integer" >&2; exit 64;; esac
[ "$CONTEXT_LIMIT" -ge 2 ] || { echo "CONTEXT_LIMIT must be at least 2" >&2; exit 64; }
ENGINESRV_IMAGE=${ENGINESRV_IMAGE:-}
SERVER_LIFETIME=${SERVER_LIFETIME:-1740}
case "$SERVER_LIFETIME" in ''|*[!0-9]*) echo "SERVER_LIFETIME must be a positive integer" >&2; exit 64;; esac
PINNED_IMAGE=inference-cuda-build:cuda12.2-ubuntu22.04
if [ -n "$ENGINESRV_IMAGE" ] && [ "$REHEARSE" != 1 ] && [ "$ENGINESRV_IMAGE" != "$PINNED_IMAGE" ]; then
  echo "ENGINESRV_IMAGE $ENGINESRV_IMAGE is not the pinned build image $PINNED_IMAGE" >&2; exit 75
fi
AH="$MW/qualification/agent-acceptance"
EV="$MS/evidence"
if [ -n "${ENGINESRV_API_KEY:-}" ]; then echo "ENGINESRV_API_KEY is set in the environment; the key comes on stdin only" >&2; exit 64; fi
IFS= read -r ENGINESRV_API_KEY || true
if [ -z "$ENGINESRV_API_KEY" ]; then echo "the first line of stdin must be the bearer key" >&2; exit 64; fi
export ENGINESRV_API_KEY
mkdir -p "$EV"

hashfile() { sha256sum "$@"; }
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(stamp)] $*"; }
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
snapshot() {
  # $1 = before|after. Linux host: memory, the GPU, and the listeners of the port.
  {
    echo "time $(stamp)"
    echo "free_m $(free -m 2>/dev/null | awk 'NR == 2 { print $2, $3, $4 }')"
    echo "enginesrv_pids $(pgrep -x enginesrv 2>/dev/null | tr '\n' ';')"
    nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,clocks.sm,driver_version --format=csv,noheader 2>/dev/null
    ss -ltn "sport = :$PORT" 2>/dev/null
  } > "$EV/system-$1.txt"
}

SERVER_PID=""; SAMPLER_PID=""; VRAM_PID=""; STOPPED=0
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
  if [ -n "$VRAM_PID" ]; then kill "$VRAM_PID" 2>/dev/null; fi
  stop_server
}
trap cleanup EXIT

# --- preflight: nothing starts unless every check passes --------------------------------------
# The CUDA toolkit of kubdan: CUDA_PATH when set, else /usr/local/cuda, else nvcc on PATH.
if [ -n "${CUDA_PATH:-}" ]; then
  NVCC="$CUDA_PATH/bin/nvcc"
elif [ -x /usr/local/cuda/bin/nvcc ]; then
  CUDA_PATH=/usr/local/cuda; NVCC="$CUDA_PATH/bin/nvcc"
else
  NVCC=$(command -v nvcc 2>/dev/null || true)
  if [ -n "$NVCC" ]; then CUDA_PATH=$(cd "$(dirname "$NVCC")/.." && pwd); fi
fi
if [ -z "$NVCC" ] || [ ! -x "$NVCC" ]; then
  echo "nvcc is missing (CUDA_PATH=${CUDA_PATH:-unset}); set CUDA_PATH to the CUDA 12 toolkit root of kubdan" >&2; exit 73
fi
export CUDA_PATH
NVCC_VERSION=$("$NVCC" --version 2>/dev/null | tail -1)

# The GPU is shared with the user's gaming and with other lanes' baton-gated holds: it must be
# free. nvidia-smi must answer with a GPU, memory.used must be below GPU_FREE_LIMIT_MIB, and no
# compute process may run.
GPU_FREE_LIMIT_MIB=2048
GPU_LINE=$(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
case "$GPU_LINE" in
  *[![:space:]]*) ;;
  *) echo "nvidia-smi reports no GPU; the run does not start" >&2; exit 71;;
esac
GPU_NAME=$(printf '%s\n' "$GPU_LINE" | awk -F', *' '{ gsub(/^ +| +$/, "", $1); print $1 }')
GPU_USED=$(printf '%s\n' "$GPU_LINE" | awk -F', *' '{ gsub(/ /, "", $2); print $2 }')
if ! printf '%s\n' "$GPU_USED" | grep -qE '^[0-9]+$'; then
  echo "nvidia-smi gave no usable memory.used (line: $GPU_LINE)" >&2; exit 71
fi
if [ "$GPU_USED" -ge "$GPU_FREE_LIMIT_MIB" ]; then
  echo "the GPU is not free: memory.used is $GPU_USED MiB (limit $GPU_FREE_LIMIT_MIB)" >&2; exit 71
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
  echo "the GPU is not free: a compute process is present" >&2; exit 71
fi

# The port: another enginesrv of this user (72) or any other listener (70) stops the run.
if port_busy "$PORT"; then
  if ss -ltnp "sport = :$PORT" 2>/dev/null | grep -q 'users:(("enginesrv'; then
    echo "another enginesrv of this user listens on port $PORT" >&2; exit 72
  fi
  say "port $PORT is in use; this script will not stop another process"; exit 70
fi

# Local NVMe roots (constants; only a rehearsal may name others). The build never runs on the NFS
# share, which is pinu's small root disk.
WRKTR_ROOT=/media/dan/nvme/fifi/wrktr/runs
SCRCH_ROOT=/media/dan/nvme/fifi/scrch/runs
PREBUILT_ROOT=/media/dan/nvme/fifi/scrch/prebuilt
MIN_FREE_MB=30720
if [ "$REHEARSE" = 1 ]; then
  WRKTR_ROOT=${KUBDAN_WRKTR_ROOT:-$WRKTR_ROOT}; SCRCH_ROOT=${KUBDAN_SCRCH_ROOT:-$SCRCH_ROOT}
  PREBUILT_ROOT=${KUBDAN_PREBUILT_ROOT:-$PREBUILT_ROOT}
  MIN_FREE_MB=${KUBDAN_MIN_FREE_MB:-$MIN_FREE_MB}
fi
case "$MW" in "$WRKTR_ROOT"/*) ;; *) echo "the worktree $MW is not under the local NVMe root $WRKTR_ROOT; the build never runs on the NFS share" >&2; exit 75;; esac
case "$MS" in "$SCRCH_ROOT"/*) ;; *) echo "the scratch $MS is not under the local NVMe root $SCRCH_ROOT" >&2; exit 75;; esac
ENGINESRV_BIN=${ENGINESRV_BIN:-}
if [ -n "$ENGINESRV_IMAGE" ] && [ -z "$ENGINESRV_BIN" ]; then
  echo "ENGINESRV_IMAGE needs a prebuilt ENGINESRV_BIN: the container runs that binary" >&2; exit 75
fi
if [ -n "$ENGINESRV_BIN" ]; then
  case "$ENGINESRV_BIN" in
    "$PREBUILT_ROOT"/*..*|*/../*) echo "ENGINESRV_BIN $ENGINESRV_BIN holds a parent reference" >&2; exit 75;;
    "$PREBUILT_ROOT"/*) ;;
    *) echo "ENGINESRV_BIN $ENGINESRV_BIN is not under the prebuilt root $PREBUILT_ROOT" >&2; exit 75;;
  esac
  [ -f "$ENGINESRV_BIN" ] && [ -x "$ENGINESRV_BIN" ] || { echo "ENGINESRV_BIN $ENGINESRV_BIN is not an executable file" >&2; exit 75; }
  WT_COMMIT=$(git -C "$MW" rev-parse HEAD 2>/dev/null || true)
  BIN_COMMIT=$(head -n 1 "$ENGINESRV_BIN.commit" 2>/dev/null || true)
  if [ -z "$WT_COMMIT" ] || [ "$WT_COMMIT" != "$BIN_COMMIT" ]; then
    echo "the prebuilt binary was built from commit '${BIN_COMMIT:-none}', the worktree is at '${WT_COMMIT:-unknown}'" >&2; exit 76
  fi
fi
free_mb() { df -Pk "$1" 2>/dev/null | awk 'NR == 2 { print int($4 / 1024) }'; }
for root in "$SCRCH_ROOT" "$WRKTR_ROOT"; do
  avail=$(free_mb "$root")
  if [ -z "$avail" ] || [ "$avail" -lt "$MIN_FREE_MB" ]; then
    echo "the free disk of $root is ${avail:-unknown} MiB, below the $MIN_FREE_MB MiB the build needs" >&2; exit 74
  fi
done

# --- model pins -------------------------------------------------------------------------------
# The pinned CUDA model dir and its pins are constants, assigned after the argument checks, so
# that nothing in the environment can change them (the selftest checks this). Only a rehearsal may
# name another directory, pin file, or GGUF size. The pin values are the gate's frozen identity:
# the GGUF name, size, and sha256 of qualification/cuda-rd/a5-gated-steps.json, and the four
# tokenizer file hashes that engine/src/family/qwen36/settings.rs pins.
Q36_CUDA_PINNED_DIR=/media/dan/nvme/models/qwen3.6-35b-a3b
Q36_MODEL_ID=fifi-q36
Q36_CUDA_GGUF=Qwen3.6-35B-A3B-Q4_K_M.gguf
Q36_CUDA_GGUF_SIZE=20419565568
Q36_CUDA_FILES="$Q36_CUDA_GGUF tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json"
Q36_CUDA_DIR=$Q36_CUDA_PINNED_DIR; Q36_CUDA_PINS=$AH/q36-cuda-pins.sha256
if [ "$REHEARSE" = 1 ]; then
  Q36_CUDA_DIR=${Q36_CUDA_DIR_OVERRIDE:-$Q36_CUDA_DIR}; Q36_CUDA_PINS=${Q36_CUDA_PINS_OVERRIDE:-$Q36_CUDA_PINS}
  Q36_CUDA_GGUF_SIZE=${Q36_CUDA_SIZE_OVERRIDE:-$Q36_CUDA_GGUF_SIZE}
fi
# q36_cuda_check_pins: hashes the GGUF and the four tokenizer files and compares each with the pin
# file; it also checks the GGUF size. It prints one `artifact_file <name> <hash>` line per file
# (the hash is `missing` for an absent file) and returns 1 on any mismatch. Nothing starts before
# it returns 0.
q36_cuda_check_pins() {
  local name want got size bad=0
  for name in $Q36_CUDA_FILES; do
    want=$(awk -v n="$name" '$2 == n { print $1 }' "$Q36_CUDA_PINS" 2>/dev/null)
    if [ -f "$Q36_CUDA_DIR/$name" ]; then got=$(hashfile "$Q36_CUDA_DIR/$name" | cut -d' ' -f1); else got=missing; fi
    echo "artifact_file $name $got"
    if [ -z "$want" ] || [ "$got" != "$want" ]; then
      echo "Q36 CUDA PIN MISMATCH: $name has ${got}, the pin is ${want:-absent}" >&2
      bad=1
    fi
  done
  size=missing
  if [ -f "$Q36_CUDA_DIR/$Q36_CUDA_GGUF" ]; then size=$(wc -c < "$Q36_CUDA_DIR/$Q36_CUDA_GGUF" | tr -d ' '); fi
  echo "artifact_size $Q36_CUDA_GGUF $size"
  if [ "$size" != "$Q36_CUDA_GGUF_SIZE" ]; then
    echo "Q36 CUDA PIN MISMATCH: $Q36_CUDA_GGUF has size ${size}, the pin is $Q36_CUDA_GGUF_SIZE" >&2
    bad=1
  fi
  return $bad
}
Q36_CUDA_IDENTITY=$(q36_cuda_check_pins) || {
  say "the CUDA model dir $Q36_CUDA_DIR does not match q36-cuda-pins.sha256; nothing is started"
  printf '%s\n' "$Q36_CUDA_IDENTITY" > "$EV/q36-cuda-pin-check.txt"; exit 88
}
{
  echo "time $(stamp)"; echo "mode $MODE"; echo "engine $ENGINE"; echo "host $(hostname)"; echo "uname $(uname -a)"
  echo "port $PORT"; echo "rehearse $REHEARSE"; echo "capture $CAPTURE"; echo "context_limit $CONTEXT_LIMIT"
  echo "ready_timeout $READY_TIMEOUT"
  echo "head $(git -C "$MW" rev-parse HEAD 2>/dev/null)"
  echo "status_porcelain $(git -C "$MW" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  echo "model_id $Q36_MODEL_ID"; echo "model_dir $Q36_CUDA_DIR"
  printf '%s\n' "$Q36_CUDA_IDENTITY"
  echo "gpu_name $GPU_NAME"; echo "gpu_free_limit_mib $GPU_FREE_LIMIT_MIB"
  echo "cuda_path $CUDA_PATH"; echo "nvcc $NVCC_VERSION"
  echo "wrktr_root $WRKTR_ROOT"; echo "scrch_root $SCRCH_ROOT"
} > "$EV/identity.txt"
snapshot before

# write_config: the CUDA arm of the Q36 family (CUDA-SERVE plan section 2): one engine entry, the
# model id fifi-q36, `capture` from CAPTURE and `context_limit` from CONTEXT_LIMIT in the qwen36
# object, and no MLX-only key (cache_limit, session_cap, and the prefill_* terms do not apply to
# the CUDA arm; prefix_reuse is not set). The file holds no key: only the variable name
# (api_key_env), so the evidence copy needs no redaction.
write_config() {
  # `capture` is a JSON boolean in the engine config (Q36Options.capture: Option<bool>).
  local CAPTURE_JSON=false
  if [ "$CAPTURE" = 1 ]; then CAPTURE_JSON=true; fi
  cat > "$MS/config.json" <<CONFIG
{"listen":"127.0.0.1:$PORT","api_key_env":"ENGINESRV_API_KEY","engines":[{"id":"q36cuda","family":"qwen3_6_35b_a3b","model":"$Q36_MODEL_ID","model_dir":"$Q36_CUDA_DIR","tokenizer_dir":"$Q36_CUDA_DIR","qwen36":{"capture":$CAPTURE_JSON,"context_limit":$CONTEXT_LIMIT}}]}
CONFIG
  cp "$MS/config.json" "$EV/config.json"
}

# A rehearsal writes and keeps the same config (the fake server ignores it).
write_config

# --- build and start ---------------------------------------------------------------------------
if [ "$REHEARSE" = 1 ]; then
  : "${SERVER_CMD:?SERVER_CMD is required with REHEARSE=1}"
  SERVER_MATCH=${SERVER_MATCH:-fake_server}
else
  SERVER_MATCH="enginesrv"
fi
export CARGO_TARGET_DIR="$MS/target" TMPDIR="$MS/tmp" TMP="$MS/tmp" TEMP="$MS/tmp"
mkdir -p "$CARGO_TARGET_DIR" "$TMPDIR"
if [ -n "$ENGINESRV_BIN" ]; then
  say "using the prebuilt enginesrv $ENGINESRV_BIN (commit $BIN_COMMIT); no build"
  BIN="$ENGINESRV_BIN"
  { echo "binary_prebuilt yes"; echo "binary_commit $BIN_COMMIT"; } >> "$EV/identity.txt"
else
  say "building enginesrv (native-cuda)"
  ( cd "$MW/engine" && exec cargo build --release --features native-cuda --bin enginesrv ) > "$MS/build.log" 2>&1 &
  BUILD_PID=$!
  while kill -0 "$BUILD_PID" 2>/dev/null; do
    if [ -f "$MS/STOP" ]; then say "STOP during the build"; kill -TERM "$BUILD_PID" 2>/dev/null; wait "$BUILD_PID" 2>/dev/null; exit 87; fi
    sleep 2
  done
  wait "$BUILD_PID" || { say "build failed; see build.log"; cp "$MS/build.log" "$EV/"; exit 80; }
  BIN="$CARGO_TARGET_DIR/release/enginesrv"
  echo "binary_prebuilt no" >> "$EV/identity.txt"
fi
echo "binary_sha256 $(hashfile "$BIN" | cut -d' ' -f1)" >> "$EV/identity.txt"
if [ "$REHEARSE" != 1 ]; then SERVER_CMD="$BIN --config $MS/config.json"; fi
if [ -n "$ENGINESRV_IMAGE" ]; then
  CONTAINER="acca-$(basename "$MS")"
  SERVER_CMD="docker run --pull never --rm --name $CONTAINER --network host --gpus all --cpus 8 --memory 16g --memory-swap 16g -e ENGINESRV_API_KEY -v /media/dan/nvme:/media/dan/nvme $ENGINESRV_IMAGE timeout -s KILL $SERVER_LIFETIME $BIN --config $MS/config.json"
  { echo "server_image $ENGINESRV_IMAGE"; echo "server_image_id $(docker image inspect --format '{{.Id}}' "$ENGINESRV_IMAGE" 2>/dev/null || echo unknown)"; echo "server_container $CONTAINER"; echo "server_lifetime $SERVER_LIFETIME"; } >> "$EV/identity.txt"
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
( while kill -0 "$SERVER_PID" 2>/dev/null; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,clocks.sm --format=csv,noheader 2>/dev/null | head -1)"; sleep "$VRAM_SECONDS"; done ) > "$EV/vram.tsv" &
VRAM_PID=$!

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
kill "$VRAM_PID" 2>/dev/null; VRAM_PID=""
snapshot after
cp "$MS/server.log" "$EV/server.log"; cp "$MS/server.out" "$EV/server.out"
peak=$(awk 'NF == 2 && $2 + 0 > m { m = $2 + 0 } END { print m + 0 }' "$EV/rss.tsv")
vram_peak=$(awk '{ gsub(/[^0-9]/, "", $2); if ($2 + 0 > m) m = $2 + 0 } END { print m + 0 }' "$EV/vram.tsv")
{
  echo "mode $MODE"; echo "engine $ENGINE"; echo "capture $CAPTURE"; echo "run_status $STATUS"; echo "stop_status $STOP_STATUS"
  echo "rss_peak_kb $peak"; echo "vram_peak_mib $vram_peak"
  echo "scope structural result; not a speed claim; not a decode-speed gate"
} > "$EV/summary.txt"
( cd "$EV" && find . -type f ! -path ./SHA256SUMS | sort | while IFS= read -r f; do hashfile "$f"; done > SHA256SUMS )
say "evidence in $EV"
if [ "$STOP_STATUS" != 0 ]; then exit "$STOP_STATUS"; fi
exit "$STATUS"
