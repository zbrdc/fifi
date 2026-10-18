# Shared functions of run-acc-a.sh and run-acc-b.sh (Linux host). Sourced, not run.
# Model-written code never runs on a host. The launchers need SANDBOX_IMAGE (name@sha256:<digest>,
# already present; the scripts never pull) and run the code only through sandbox.py.
# The Mac is reached as dan@192.168.0.16 (override with MAC). The Mac lock is the host-wide file
# that every Mac qualification run holds (override with MAC_LOCK).
# The kubdan RTX 3090 (engine q36cuda) is reached as the ssh alias kubdan (override with KUBDAN).
# Its lock is kubdan-gpu.lock beside the Mac lock (override with KUBDAN_LOCK), and a run of the
# q36cuda engine also needs the coordinator's baton (GPU_BATON): see require_baton in run-acc-a.sh.
COORD=${COORD:-/home/dan/fifi/scrch/f2674c9/p0b-exec-w2-coord}
MAC=${MAC:-dan@192.168.0.16}
MAC_LOCK=${MAC_LOCK:-$COORD/mac.lock}
MAC_ENV=${MAC_ENV:-/Users/dan/fifi/scrch/841aaca/p0b-w2-s1b1/env.sh}
KUBDAN=${KUBDAN:-kubdan}
KUBDAN_LOCK=${KUBDAN_LOCK:-$COORD/kubdan-gpu.lock}
# The kubdan build paths (constants of kubdan-acc-a.sh): the checkout and the scratch (target dir,
# TMPDIR) live on the local NVMe of kubdan, never on the NFS share.
KUBDAN_RUNS_SCRCH=/media/dan/nvme/fifi/scrch/runs
KUBDAN_RUNS_WRKTR=/media/dan/nvme/fifi/wrktr/runs
PORT=${PORT:-18791}
AH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$AH_DIR/../.." && pwd)

die() { echo "error: $*" >&2; exit "${2:-1}"; }
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
new_key() { head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n'; }
local_port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
ssh_mac() { ssh -o BatchMode=yes -o ServerAliveInterval=60 "$MAC" "$@"; }
ssh_kubdan() { ssh -o BatchMode=yes -o ServerAliveInterval=60 "$KUBDAN" "$@"; }
# session_host / session_ssh: the serve host of the current engine (the Mac, or kubdan for q36cuda).
session_host() { if [ "${ENGINE:-flash}" = q36cuda ]; then printf '%s' "$KUBDAN"; else printf '%s' "$MAC"; fi; }
session_ssh() { ssh -o BatchMode=yes -o ServerAliveInterval=60 "$(session_host)" "$@"; }

# require_sandbox: stops with status 67 (image) or 69 (daemon) and a message unless the ROOTLESS
# docker daemon answers and SANDBOX_IMAGE names a digest-pinned image that is present on it. It
# pulls nothing. Call it before any server, lock, or model call.
require_sandbox() {
  python3 -I "$AH_DIR/sandbox.py" >&2 || exit $?
}

# record_sandbox <file>: writes the pinned daemon, its runc and containerd versions, and the image id.
record_sandbox() {
  python3 -I "$AH_DIR/sandbox.py" --record "$1" > /dev/null || exit $?
}

# key_in_dir <dir>: succeeds when the bearer key (variable KEY) is in a file below <dir>. The
# key goes to grep from a process substitution, never in argv and never through a here-string:
# a here-string can spill the key into a temporary file on bash before 5.1 (design section 8).
key_in_dir() {
  grep -rqF -f <(printf '%s\n' "$KEY") "$1"
}

# engine_model_id: the model id that GET /v1/models lists for ENGINE (flash by default). The
# client, the fake server of a rehearsal, and the config of the serve scripts must agree on it.
ENGINE=${ENGINE:-flash}
engine_model_id() {
  case "$ENGINE" in
    flash) echo flash;;
    q36|q36cuda) echo fifi-q36;;
    *) die "engine must be flash, q36, or q36cuda" 64;;
  esac
}

# require_baton: the kubdan GPU is shared with the user's gaming and with other lanes'
# baton-gated holds, so a q36cuda run needs the coordinator's baton: GPU_BATON must name an
# existing file that the coordinator created after git-db granted the 3090. The launcher never
# creates it; it copies it into the evidence. Call it before the lock, the build, and any model
# call; a missing baton stops with status 66.
require_baton() {
  if [ -z "${GPU_BATON:-}" ] || [ ! -f "$GPU_BATON" ]; then
    die "GPU_BATON must name the coordinator's baton file for the 3090; this run does not start without it" 66
  fi
}

# check_kubdan_ssh: the launcher drives kubdan over ssh with no interactive prompt.
check_kubdan_ssh() {
  ssh -o BatchMode=yes "$KUBDAN" true >/dev/null 2>&1 || die "ssh -o BatchMode=yes $KUBDAN does not work; fix the alias or the key" 64
}

check_port() {
  if [ "$PORT" = "8788" ]; then die "port 8788 belongs to laya3; it is never used" 64; fi
  # 18791 is the Mac ACC port; the kubdan engine has its own.
  if [ "${ENGINE:-}" = q36cuda ] && [ "$PORT" = "18791" ]; then die "port 18791 is the Mac port; the kubdan engine never uses it" 64; fi
}

# mac_prepare <branch> <run name>: bundles the branch, copies it to the Mac, and clones it.
# Sets MS (scratch) and MW (worktree) on the Mac. Needs the Mac lock to be held by the caller.
mac_prepare() {
  local branch=$1 name=$2 bundle
  # A unique name per run: the rm -rf below never touches the directory of an earlier run, whose
  # server may still run after a STOP-FAILED. Old run directories stay on the Mac for the coordinator.
  name="$name-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  MS=/Users/dan/fifi/scrch/runs/$name
  MW=/Users/dan/fifi/wrktr/runs/$name
  bundle=$(mktemp -d "${TMPDIR:-/tmp}/ah-bundle.XXXXXX")/src.bundle
  git -C "$REPO_DIR" bundle create "$bundle" "$branch" 2>&1 | tail -1
  ssh_mac "rm -rf $MS && mkdir -p $MS && cp $MAC_ENV $MS/env.sh" || die "cannot prepare $MS on the Mac" 81
  scp -q -o BatchMode=yes "$bundle" "$MAC:$MS/src.bundle" || die "cannot copy the bundle" 81
  ssh_mac "rm -rf $MW && git clone -q -b $branch $MS/src.bundle $MW && git -C $MW log --oneline -1" \
    || die "cannot clone the bundle on the Mac" 81
  rm -rf "$(dirname "$bundle")"
}

# kubdan_prepare <branch> <run name>: bundles the branch, copies it to kubdan by scp, and clones
# it on the local NVMe. Sets MS (scratch) and MW (worktree) on kubdan. Needs the kubdan lock to be
# held by the caller. The paths are the constants of kubdan-acc-a.sh: the build never runs on the
# NFS share.
kubdan_prepare() {
  local branch=$1 name=$2 bundle
  name="$name-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  MS=$KUBDAN_RUNS_SCRCH/$name
  MW=$KUBDAN_RUNS_WRKTR/$name
  bundle=$(mktemp -d "${TMPDIR:-/tmp}/ah-bundle.XXXXXX")/src.bundle
  git -C "$REPO_DIR" bundle create "$bundle" "$branch" 2>&1 | tail -1
  ssh_kubdan "rm -rf $MS $MW && mkdir -p $MS $KUBDAN_RUNS_WRKTR" || die "cannot prepare $MS on kubdan" 81
  scp -q -o BatchMode=yes "$bundle" "$KUBDAN:$MS/src.bundle" || die "cannot copy the bundle to kubdan" 81
  ssh_kubdan "git clone -q -b $branch $MS/src.bundle $MW && git -C $MW log --oneline -1" \
    || die "cannot clone the bundle on kubdan" 81
  rm -rf "$(dirname "$bundle")"
}

# kubdan_serve_command: the remote command that starts the kubdan serve side (MS and MW are set).
# KUBDAN_SERVE_WRAP (optional, for example a hold wrapper such as `<gpu_run.sh> 1800`) goes right
# before `bash`, after the environment assignments, so the wrapper runs the serve script with them.
# ENGINESRV_BIN (optional) names a prebuilt binary on kubdan: see kubdan-acc-a.sh.
kubdan_serve_command() {
  local bin=""
  [ -n "${ENGINESRV_BIN:-}" ] && bin="ENGINESRV_BIN=$ENGINESRV_BIN "
  [ -n "${ENGINESRV_IMAGE:-}" ] && bin="${bin}ENGINESRV_IMAGE=$ENGINESRV_IMAGE "
  [ -n "${SERVER_LIFETIME:-}" ] && bin="${bin}SERVER_LIFETIME=$SERVER_LIFETIME "
  printf '%s' "PORT=$PORT ENGINE=$ENGINE CAPTURE=${CAPTURE:-0} CONTEXT_LIMIT=${CONTEXT_LIMIT:-16384} READY_TIMEOUT=${READY_TIMEOUT:-1800} STOP_TIMEOUT=${STOP_TIMEOUT:-900} ${bin}${KUBDAN_SERVE_WRAP:+$KUBDAN_SERVE_WRAP }bash $MW/qualification/agent-acceptance/kubdan-acc-a.sh serve $MS $MW"
}

# evidence_fetch <local dir>: copies the serve host's evidence directory and checks its SHA256SUMS.
evidence_fetch() {
  local out=$1
  mkdir -p "$out/evidence"
  session_ssh "cd $MS/evidence && tar cf - ." | tar xf - -C "$out/evidence" || die "cannot copy the evidence" 82
  (cd "$out/evidence" && sha256sum -c SHA256SUMS --quiet) || die "the evidence hashes do not match" 82
}

# seal <local dir>: writes SHA256SUMS over every file of the directory, except the top-level
# SHA256SUMS itself. The inner evidence/SHA256SUMS is covered.
seal() {
  (cd "$1" && find . -type f ! -path ./SHA256SUMS | sort | while IFS= read -r f; do sha256sum "$f"; done > SHA256SUMS)
}

# --- the Mac serve session --------------------------------------------------------------------
# A launcher that calls session_start must install `trap launcher_cleanup EXIT` and
# `trap 'exit 143' TERM INT HUP`, so that an abort leaves no server, no ssh session, no tunnel,
# and no held lock. Background children close fd 9 (9>&-), so that they do not keep the lock.
SERVE_PID=""; TUNNEL_PID=""; CHILD_PID=""; SERVE_STATUS=0
SESSION_STOP_WAIT=${SESSION_STOP_WAIT:-1000}   # a little above the Mac STOP_TIMEOUT of 900 s

# session_start <branch> <run name>: needs OUT, KEY, PORT, REHEARSE. Sets MS and MW, SERVE_PID.
# flash and q36 serve on the Mac (mac-acc-a.sh); q36cuda serves on kubdan (kubdan-acc-a.sh) and
# holds kubdan-gpu.lock instead of the Mac lock. A rehearsal runs the serve script on this host.
session_start() {
  local branch=$1 name=$2
  if [ "$REHEARSE" = 1 ] && [ "$ENGINE" = q36cuda ]; then
    MS=$OUT/kubdan-scratch; MW=$REPO_DIR; mkdir -p "$MS"
    printf '%s\n' "$KEY" | REHEARSE=1 ENGINE=$ENGINE PORT=$PORT CAPTURE=${CAPTURE:-0} CONTEXT_LIMIT=${CONTEXT_LIMIT:-16384} \
      SAMPLE_SECONDS=${SAMPLE_SECONDS:-1} VRAM_SECONDS=${VRAM_SECONDS:-1} SERVER_MATCH=fake_server \
      SERVER_CMD="python3 $AH_DIR/fake_server.py --port $PORT --ready-delay ${FAKE_READY_DELAY:-2} --mode ${FAKE_MODE:-good} --model-id ${MODEL_ID:-$(engine_model_id)}" \
      bash "$AH_DIR/kubdan-acc-a.sh" serve "$MS" "$MW" > "$OUT/serve.out" 2>&1 9>&- &
    SERVE_PID=$!
  elif [ "$REHEARSE" = 1 ]; then
    MS=$OUT/mac-scratch; MW=$REPO_DIR; mkdir -p "$MS"
    printf '%s\n' "$KEY" | REHEARSE=1 ENGINE=$ENGINE PORT=$PORT SAMPLE_SECONDS=${SAMPLE_SECONDS:-1} SERVER_MATCH=fake_server \
      SERVER_CMD="python3 $AH_DIR/fake_server.py --port $PORT --ready-delay ${FAKE_READY_DELAY:-2} --mode ${FAKE_MODE:-good} --model-id ${MODEL_ID:-$(engine_model_id)}" \
      bash "$AH_DIR/mac-acc-a.sh" serve "$MS" "$MW" > "$OUT/serve.out" 2>&1 9>&- 8>&- &
    SERVE_PID=$!
  elif [ "$ENGINE" = q36cuda ]; then
    exec 9>"$KUBDAN_LOCK"
    say "waiting for the kubdan GPU lock $KUBDAN_LOCK"
    flock 9
    say "lock acquired"
    ssh_kubdan "nvidia-smi --query-gpu=name,memory.used,driver_version --format=csv,noheader; ss -ltn 'sport = :$PORT'; ss -ltn 'sport = :8788'" > "$OUT/kubdan-before.txt" 2>&1
    kubdan_prepare "$branch" "$name"
    printf '%s\n' "$KEY" | ssh_kubdan "$(kubdan_serve_command)" > "$OUT/serve.out" 2>&1 9>&- &
    SERVE_PID=$!
  else
    exec 9>"$MAC_LOCK"
    say "waiting for the Mac lock $MAC_LOCK"
    flock 9
    say "lock acquired"
    ssh_mac "pgrep -fl splash; lsof -nP -iTCP:$PORT -sTCP:LISTEN; lsof -nP -iTCP:8788 -sTCP:LISTEN" > "$OUT/mac-before.txt" 2>&1
    mac_prepare "$branch" "$name"
    printf '%s\n' "$KEY" | ssh_mac "PORT=$PORT ENGINE=$ENGINE bash $MW/qualification/agent-acceptance/mac-acc-a.sh serve $MS $MW" > "$OUT/serve.out" 2>&1 9>&- 8>&- &
    SERVE_PID=$!
  fi
}

mac_has() { # <file name in the scratch of the serve host>
  if [ "$REHEARSE" = 1 ]; then [ -f "$MS/$1" ]; else session_ssh "test -f $MS/$1"; fi
}

# wait_ready: waits for READY of the serve session. Returns 83 when the session ended or timed out.
wait_ready() {
  local waited=0 limit=${READY_WAIT:-4500}   # the Mac builds enginesrv (scratch is wiped), then loads
  until mac_has READY; do
    kill -0 "$SERVE_PID" 2>/dev/null || { echo "the server session ended before READY; see $OUT/serve.out" >&2; return 83; }
    [ "$waited" -lt "$limit" ] || { echo "no READY after ${limit}s" >&2; return 83; }
    sleep 5; waited=$((waited + 5))
  done
  say "server ready after about ${waited}s"
}

tunnel_open() { # the server binds the loopback of the serve host only
  if [ "$REHEARSE" != 1 ]; then
    # The local end binds 127.0.0.1 only, whatever GatewayPorts says. The caller checked that the
    # local port was free, and the tunnel must still run before the key goes to that port.
    ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -N -L "127.0.0.1:$PORT:127.0.0.1:$PORT" "$(session_host)" 9>&- &
    TUNNEL_PID=$!
    for _ in $(seq 1 30); do
      [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://127.0.0.1:$PORT/ready")" = 200 ] && break
      sleep 2
    done
    kill -0 "$TUNNEL_PID" 2>/dev/null || { echo "the ssh tunnel is not running" >&2; return 1; }
  fi
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 10 "http://127.0.0.1:$PORT/ready")" = 200 ]
}

tunnel_close() {
  if [ -n "$TUNNEL_PID" ]; then kill "$TUNNEL_PID" 2>/dev/null; wait "$TUNNEL_PID" 2>/dev/null; fi
  TUNNEL_PID=""
}

# child_stop: ends the background client or pi job (CHILD_PID) and its direct children.
child_stop() {
  if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
    local kid
    for kid in $(pgrep -P "$CHILD_PID" 2>/dev/null); do kill -TERM "$kid" 2>/dev/null; done
    kill -TERM "$CHILD_PID" 2>/dev/null
    wait "$CHILD_PID" 2>/dev/null
  fi
  CHILD_PID=""
}

# session_stop: closes the tunnel and ends the serve session this launcher started. It asks the
# Mac script to stop its own server (the STOP file) and waits for the session to end. It may run
# twice. Sets SERVE_STATUS.
session_stop() {
  tunnel_close
  [ -n "$SERVE_PID" ] || return 0
  if kill -0 "$SERVE_PID" 2>/dev/null; then
    if [ "$REHEARSE" = 1 ]; then touch "$MS/STOP"; else session_ssh "touch $MS/STOP" 9>&- || true; fi
    local waited=0
    while kill -0 "$SERVE_PID" 2>/dev/null && [ "$waited" -lt "$SESSION_STOP_WAIT" ]; do sleep 1; waited=$((waited + 1)); done
    if kill -0 "$SERVE_PID" 2>/dev/null; then
      echo "the server session did not end after ${SESSION_STOP_WAIT}s; closing its ssh connection" >&2
      kill -TERM "$SERVE_PID" 2>/dev/null
    fi
  fi
  wait "$SERVE_PID" 2>/dev/null; SERVE_STATUS=$?
  SERVE_PID=""
}

# launcher_cleanup: the EXIT trap of every launcher that starts a serve session.
launcher_cleanup() {
  child_stop
  session_stop
}
