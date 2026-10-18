#!/usr/bin/env bash
# ACC-A launcher (Linux host). It holds the serve lock, starts enginesrv on the serve host
# (mac-acc-a.sh on the Mac, kubdan-acc-a.sh on kubdan for the q36cuda engine), opens an ssh
# tunnel, runs the scripted tool loop HERE, and stops only the server it
# started. The model's code never runs on a host: the tool loop writes the model's file into a
# scratch directory on this host and runs it only in `docker run` (sandbox.py: digest-pinned image
# in SANDBOX_IMAGE, no network, read-only root, non-root user, memory, CPU, and process limits,
# timeout, script mounted read-only). Without a present SANDBOX_IMAGE the launcher stops with
# status 67 (69 for a daemon that is not rootless) before it takes the lock or starts anything. It pulls no image: a pull needs the
# user's approval. A pass is a structural result: no speed claim, not Phase 0B or Phase 1B.
# Usage: run-acc-a.sh <new evidence dir> [--mode acc-a|smoke] [--branch main] [--engine flash|q36|q36cuda]
#                     [--rehearse]
#   --engine    flash (default) serves Qwen3.8-Flash-Next as model id `flash` on the Mac. q36 serves
#               Qwen3.6-35B-A3B as model id `fifi-q36` on the Mac (mac-acc-a.sh checks four file
#               hashes first; a mismatch gives status 88). q36cuda serves the same model id on
#               kubdan's RTX 3090 from the pinned GGUF checkpoint (kubdan-acc-a.sh; pin mismatch
#               88). The q36cuda run needs the coordinator's baton (GPU_BATON names an existing
#               file; without it status 66, before the lock, the build, and any model call) and a
#               working `ssh -o BatchMode=yes kubdan` (status 64); it holds kubdan-gpu.lock. The
#               CUDA prefill is token by token, so the client timeout is ACC_REQUEST_TIMEOUT per
#               request (default 1,800 s), recorded in result.json and launcher.txt.
#   --rehearse  run everything on this host against fake_server.py: no lock, no ssh, no model.
#   smoke mode runs no model code and needs no sandbox.
# Exit status: that of the run (0 pass, 2 pipeline defect, 3 quality miss, 4 budget), or 64 to 95
# for setup failures (66 the GPU baton is missing, 67 sandbox image missing, 88 the Q36 checkpoint
# does not match its pins, 86 the server ended during the run, 95 the server
# did not stop). The kubdan serve side stops with 71 to 75 before it starts anything (71 the GPU is
# not free, 72 another enginesrv on the port, 73 nvcc missing, 74 disk below 30 GiB, 75 a build
# path outside the local NVMe). An abort (SIGTERM, SIGINT) stops the server session, the tunnel,
# and the client.
set -u
PORT_BEFORE=${PORT:-}
# shellcheck source=driver-lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/driver-lib.sh"
OUT=${1:-}; [ -n "$OUT" ] || die "usage: run-acc-a.sh <new evidence dir> [--mode ...]" 64
shift
MODE=acc-a; BRANCH=main; REHEARSE=0; ENGINE=flash
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE=$2; shift 2;;
    --branch) BRANCH=$2; shift 2;;
    --engine) ENGINE=${2:-}; shift 2;;
    --rehearse) REHEARSE=1; shift;;
    *) die "unknown option $1" 64;;
  esac
done
case "$MODE" in acc-a|smoke) ;; *) die "mode must be acc-a or smoke" 64;; esac
case "$ENGINE" in flash|q36|q36cuda) ;; *) die "engine must be flash, q36, or q36cuda" 64;; esac
MODEL_NAME=$(engine_model_id)
# The kubdan engine serves on its own port: 18792 unless PORT names one. 8788 and the Mac port
# 18791 are refused by check_port.
if [ "$ENGINE" = q36cuda ] && [ -z "$PORT_BEFORE" ]; then PORT=18792; fi
check_port
[ ! -e "$OUT" ] || die "$OUT exists; give a new directory" 64
# The kubdan GPU is shared with the user's gaming and with other lanes' baton-gated holds: the run
# needs the coordinator's baton (GPU_BATON, an existing file the coordinator created after git-db
# granted the 3090) and, outside a rehearsal, a working BatchMode ssh to kubdan. Both refusals
# come before the lock, the build, and any model call. The launcher never creates the baton; it
# copies it into the evidence below.
if [ "$ENGINE" = q36cuda ]; then
  require_baton
  [ "$REHEARSE" = 1 ] || check_kubdan_ssh
fi
if [ "$MODE" = acc-a ]; then require_sandbox; fi
# The port must be free on this host in every mode: the server (rehearsal) or the ssh tunnel (real
# run) takes it, and the bearer key goes to whatever answers there.
if local_port_busy "$PORT"; then die "port $PORT is in use on this host" 70; fi
mkdir -p "$OUT"
if [ "$ENGINE" = q36cuda ]; then cp "$GPU_BATON" "$OUT/gpu-baton.txt"; fi
if [ "$MODE" = acc-a ]; then record_sandbox "$OUT/sandbox-daemon.txt"; fi
KEY=$(new_key)
# A test may fix the key (rehearsal only), so that selftest.sh can search the disk for it.
if [ "$REHEARSE" = 1 ] && [ -n "${ACC_TEST_KEY:-}" ]; then KEY=$ACC_TEST_KEY; fi
trap 'exit 143' TERM INT HUP
trap launcher_cleanup EXIT
say "ACC launcher: mode $MODE, engine $ENGINE (model $MODEL_NAME), branch $BRANCH, rehearse $REHEARSE, evidence $OUT"
# The CUDA prefill runs token by token (a 6,000-token tool prompt takes minutes), so the client
# timeout of the q36cuda engine is a configurable, recorded figure: ACC_REQUEST_TIMEOUT seconds
# per request (default 1,800). It goes to the client and into result.json and launcher.txt.
CLIENT_TIMEOUT_ARGS=""
if [ "$ENGINE" = q36cuda ]; then
  ACC_REQUEST_TIMEOUT=${ACC_REQUEST_TIMEOUT:-1800}
  case "$ACC_REQUEST_TIMEOUT" in ''|*[!0-9.]*) die "ACC_REQUEST_TIMEOUT must be a number of seconds" 64;; esac
  CLIENT_TIMEOUT_ARGS="--timeout $ACC_REQUEST_TIMEOUT"
fi

session_start "$BRANCH" "agent1-$MODE"
wait_ready || {
  # A pin mismatch or a kubdan preflight refusal ends the serve session before READY. Report its
  # own status, not the generic 83.
  session_stop
  seal "$OUT"
  if [ "$SERVE_STATUS" = 88 ]; then echo "the Q36 checkpoint does not match its pins; see $OUT/serve.out" >&2; exit 88; fi
  case "$SERVE_STATUS" in
    71|72|73|74|75) echo "the kubdan serve side refused the run (status $SERVE_STATUS); see $OUT/serve.out" >&2; exit "$SERVE_STATUS";;
  esac
  exit 83
}
tunnel_open || { echo "/ready is not 200 through the tunnel" >&2; seal "$OUT"; exit 84; }

BASE_URL="http://127.0.0.1:$PORT/v1"
export ENGINESRV_API_KEY="$KEY"
case "$MODE" in
  acc-a)
    python3 "$AH_DIR/acc_a.py" --base-url "$BASE_URL" --engine "$ENGINE" --model "$MODEL_NAME" $CLIENT_TIMEOUT_ARGS --out "$OUT/acc-a" --workdir "$OUT/acc-a-work" \
      < /dev/null > "$OUT/acc-a.out" 2>&1 9>&- &
    CHILD_PID=$! ;;
  smoke)
    python3 "$AH_DIR/smoke.py" --base-url "$BASE_URL" --out "$OUT/smoke.json" \
      < /dev/null > "$OUT/smoke.out" 2>&1 9>&- &
    CHILD_PID=$! ;;
esac
wait "$CHILD_PID"; CLIENT_STATUS=$?
CHILD_PID=""
unset ENGINESRV_API_KEY
say "client status $CLIENT_STATUS"

session_stop
if [ "$REHEARSE" = 1 ]; then
  mkdir -p "$OUT/evidence"; cp -R "$MS/evidence/." "$OUT/evidence/"
else
  evidence_fetch "$OUT" || true
  if [ "$ENGINE" = q36cuda ]; then
    ssh_kubdan "ss -ltn 'sport = :8788'" > "$OUT/kubdan-after-8788.txt" 2>&1
  else
    ssh_mac "lsof -nP -iTCP:8788 -sTCP:LISTEN" > "$OUT/mac-after-8788.txt" 2>&1
  fi
fi
STATUS=$CLIENT_STATUS
if [ "$MODE" = acc-a ] && [ -f "$OUT/acc-a/result.json" ]; then
  python3 "$AH_DIR/check_server_log.py" --log "$OUT/evidence/server.log" --result "$OUT/acc-a/result.json"
  LOG_STATUS=$?
  if [ "$STATUS" = 0 ] && [ "$LOG_STATUS" != 0 ]; then STATUS=2; fi
fi
if [ "$SERVE_STATUS" = 95 ] || { [ "$STATUS" = 0 ] && [ "$SERVE_STATUS" != 0 ]; }; then STATUS=$SERVE_STATUS; fi
if key_in_dir "$OUT"; then
  echo "the key appears in the evidence; removing $OUT" >&2; rm -rf "$OUT"; exit 90
fi
{ echo "mode $MODE"; echo "engine $ENGINE"; echo "model $MODEL_NAME"; echo "branch $BRANCH"; echo "rehearse $REHEARSE"; echo "status $STATUS"
  echo "client_status $CLIENT_STATUS"; echo "serve_status $SERVE_STATUS"
  if [ "$ENGINE" = q36cuda ]; then
    echo "capture ${CAPTURE:-0}"; echo "context_limit ${CONTEXT_LIMIT:-16384}"; echo "request_timeout ${ACC_REQUEST_TIMEOUT:-1800}"
  fi
  echo "launcher_head $(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null)"; } > "$OUT/launcher.txt"
seal "$OUT"
say "run status $STATUS; evidence sealed in $OUT/SHA256SUMS"
exit "$STATUS"
