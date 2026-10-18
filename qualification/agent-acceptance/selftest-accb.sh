#!/usr/bin/env bash
# Self-test of the ACC-B launcher (packet "selftest-accb.sh"). pinu only: it needs the rootless
# Docker daemon, the pinned image, the pi 1.0.0 release, and node v22.22.3, and it dies when one
# of them is absent. No Mac, no model, no network egress: every server is fake_server.py on pinu
# loopback and every forward is accb-relay.py. Containers are named accb-t-<check> or
# accb-<run> only; a WATCH watchdog ends a stuck one; every check compares the list of other
# containers with the P10 baseline; destructive checks re-run the P8 resource guard first
# (4 GiB free memory, 20 GiB free disk) and stop below it. SBX commands are built from
# `run-acc-b.sh --print-spec`, so they use the final flag list of design 4.3 (checklist SBX).
# Usage: selftest-accb.sh [agent-acceptance dir under test] [scratch dir] [check ...]
#   Defaults: the script's directory, a new $TMPDIR directory, all checks.
# Exit status: 0 when every check passes, 1 otherwise. It tests the launcher, not the engine.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AHD=${1:-$HERE}
TOP=${2:-$(mktemp -d "${TMPDIR:-/tmp}/accb-selftest.XXXXXX")}
if [ $# -gt 2 ]; then CHECKS=("${@:3}"); else CHECKS=(); fi
FAILS=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; FAILS=$((FAILS + 1)); }
expect_status() { if [ "$1" = "$2" ]; then ok "$3 (status $2)"; else bad "$3: want status $1, got $2"; fi; }
die() { echo "ERROR: $*" >&2; exit 2; }
mkdir -p "$TOP"

DK="docker --host unix:///run/user/1000/docker.sock"
IMG=python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
TEST_KEY=accb-selftest-key-4451
export SANDBOX_IMAGE=$IMG
export ACCB_SCRATCH="$TOP/accb-scratch"
export SESSION_STOP_WAIT=${SESSION_STOP_WAIT:-120}
export STOP_TIMEOUT=${STOP_TIMEOUT:-60}

# --- preflight: this packet runs on pinu ------------------------------------------------------------
$DK info --format '{{.SecurityOptions}}' >/dev/null 2>&1 || die "the rootless daemon does not answer"
$DK info --format '{{.SecurityOptions}}' 2>/dev/null | grep -q rootless || die "the daemon is not rootless"
$DK image inspect --format '{{.Id}}' "$IMG" >/dev/null 2>&1 || die "the pinned image is absent; a pull needs approval"
[ -d "$HOME/.pi/agent/install/releases/1.0.0" ] || die "the pi 1.0.0 release is absent"
[ -x "$HOME/.local/share/pi-node/node-v22.22.3-linux-x64/bin/node" ] || die "node v22.22.3 is absent"
command -v curl >/dev/null || die "curl is absent"
command -v ssh >/dev/null || die "ssh is absent"
command -v tar >/dev/null || die "tar is absent"

# --- the local test port (fix1) -----------------------------------------------------------------------
# Every pinu-local listener of this self-test (the fake servers, the relay, the m1 black hole)
# and of the launcher rehearsals takes its TCP port from ACCB_TEST_PORT. A free port is chosen
# at start: probe 18900-18990 first (outside the ephemeral range, so an outgoing connection of
# another workload cannot take it between the probe and the run); bind to port 0 only when the
# whole range is busy; never 18791 (another session's tunnel may hold it on pinu) and never
# 8788 (laya3). An inherited ACCB_TEST_PORT is honoured when it is a usable port number
# (mutate-accb.sh sets one).
case "${ACCB_TEST_PORT:-}" in
  ''|18791|8788|*[!0-9]*)
    ACCB_TEST_PORT=$(python3 -I <<'PICKPY'
import socket

def free(port):
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()

port = next((p for p in range(18900, 18991) if free(p)), 0)
if port == 0:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.close()
    if port in (18791, 8788) or not free(port):
        port = 0
print(port)
PICKPY
)
    ;;
esac
[ "$ACCB_TEST_PORT" != 0 ] || die "no free local port in 18900-18990"
export ACCB_TEST_PORT

# --- the spec: placeholders plus the real reflink copies, and the final flag list -------------------
SPEC=$TOP/spec
if ! bash "$AHD/run-acc-b.sh" --print-spec "$SPEC" > "$TOP/spec.out" 2> "$TOP/spec.err"; then
  die "--print-spec failed: $(tail -2 "$TOP/spec.err")"
fi
mapfile -t SPEC_ARGS < <(sed -n 's/^ARG //p' "$TOP/spec.out")
[ "${#SPEC_ARGS[@]}" -gt 30 ] || die "the spec output has no argument list"
SPEC_SSH_CMD=$(sed -n 's/^SSH_CMD //p' "$TOP/spec.out" | head -1)
sed -n '/^SSH_CONFIG_BEGIN$/,/^SSH_CONFIG_END$/p' "$TOP/spec.out" | sed '1d;$d' > "$TOP/spec-ssh-config"
# The real container files for the entry, bridge, and agent checks. The mount list is unchanged.
cp "$AHD/accb-entry.sh" "$SPEC/entry.sh"
cp "$AHD/accb-bridge.js" "$SPEC/bridge.js"
cat > "$SPEC/agent/models.json" <<'JSON'
{
  "providers": {
    "fifi": {
      "baseUrl": "http://127.0.0.1:18791/v1",
      "api": "openai-completions",
      "apiKey": "$ENGINESRV_API_KEY",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens",
        "sendSessionAffinityHeaders": false
      },
      "models": [
        {"id": "flash", "name": "Fifi Flash-Next (local, AGENT-1a)", "reasoning": false,
         "input": ["text"], "contextWindow": 6400, "maxTokens": 256,
         "samplingParams": {"temperature": 0.2},
         "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}
      ]
    }
  }
}
JSON
cat > "$SPEC/agent/settings.json" <<'JSON'
{"defaultProvider": "fifi", "defaultModel": "flash", "defaultThinkingLevel": "off",
 "httpIdleTimeoutMs": 0, "retry": {"enabled": false}}
JSON

# --- helpers ---------------------------------------------------------------------------------------
CLEANUP_CONT=""
WATCH_PIDS=()
p10_list() { $DK ps --format '{{.Names}} {{.ID}} {{.State}}' 2>/dev/null | grep -v '^accb-' || true; }
P10=$(p10_list); echo "$P10" > "$TOP/p10-baseline.txt"
others_unchanged() { # <label>: the other containers equal the P10 baseline
  local now; now=$(p10_list)
  if [ "$now" = "$P10" ]; then ok "$1: the other containers are unchanged"; else bad "$1: the other containers changed from the P10 baseline: $now"; fi
}
p8_guard() { # <label>: re-runs P8 before a destructive check
  local free_b disk_b
  free_b=$(free -b | awk '/^Mem:/ {print $7}'); disk_b=$(df -B1 / | awk 'NR==2 {print $4}')
  if [ "$free_b" -gt 4294967296 ] && [ "$disk_b" -gt 21474836480 ]; then
    ok "P8 before $1: ${free_b} bytes free memory, ${disk_b} bytes free disk"; return 0
  fi
  bad "P8 below the thresholds before $1 (memory $free_b, disk $disk_b); the check is skipped"
  return 1
}
watch() { # <name> <seconds>: the checklist WATCH watchdog
  ( local i
    for ((i=0; i<$2; i++)); do
      $DK ps -aq --filter "name=$1" 2>/dev/null | grep -q . || exit 0
      sleep 1
    done
    $DK kill "$1" 2>/dev/null ) &
  WATCH_PIDS+=("$!")
}
cleanup_now() {
  [ -n "$CLEANUP_CONT" ] && $DK rm -f "$CLEANUP_CONT" >/dev/null 2>&1
  local p n
  for p in "${WATCH_PIDS[@]:-}"; do kill "$p" 2>/dev/null; done
  for n in $($DK ps -aq --filter name=accb- 2>/dev/null); do $DK rm -f "$n" >/dev/null 2>&1; done
}
trap cleanup_now EXIT

sbx_args() { # <name> <entrypoint>: the spec flag list with name and entrypoint replaced,
  #            the trailing command dropped — append the command as extra arguments
  local i
  for ((i=0; i<${#SPEC_ARGS[@]}; i++)); do
    if [ "${SPEC_ARGS[i]}" = "--name" ]; then printf '%s\n' "--name" "$1"; i=$((i+1))
    elif [ "${SPEC_ARGS[i]}" = "--entrypoint" ]; then printf '%s\n' "--entrypoint" "$2"; i=$((i+1))
    elif [ "$i" -eq $((${#SPEC_ARGS[@]}-1)) ]; then :
    else printf '%s\n' "${SPEC_ARGS[i]}"; fi
  done
}
sbx() { # <name> <entrypoint> [command...]: runs an SBX container; prints stdout, status in $?
  local name=$1 ep=$2; shift 2
  local args=(); mapfile -t args < <(sbx_args "$name" "$ep")
  CLEANUP_CONT="$name"
  "${args[@]}" "$@" < /dev/null          # the list already starts with docker --host ... run -i
}
entry_args() { # <name> [pi_timeout]: the full flag list for the real entry, PI_TIMEOUT replaced
  local i name=$1 timeout=${2:-5400}
  for ((i=0; i<${#SPEC_ARGS[@]}; i++)); do
    if [ "${SPEC_ARGS[i]}" = "--name" ]; then printf '%s\n' "--name" "$name"; i=$((i+1))
    elif [ "${SPEC_ARGS[i]}" = "PI_TIMEOUT=5400" ]; then printf '%s\n' "PI_TIMEOUT=$timeout"
    else printf '%s\n' "${SPEC_ARGS[i]}"; fi
  done
}
PY_RUN="import sys
dev=[l.split(':',1)[0].strip() for l in open('/proc/net/dev').read().splitlines()[2:] if l.strip()]
route=open('/proc/net/route').read().splitlines()
if dev != ['lo'] or len(route) != 1:
    print('NETPRE failed: interfaces', dev, 'route lines', len(route)); sys.exit(99)
"
mk_fixture() { # <dir> <kind>: the rehearsal fixtures, one read-only hello.py per behaviour.
  # An earlier check may have staged the same kind and left hello.py mode 0444, so the old copy
  # is removed before the heredoc rewrites it (the directory stays writable).
  # The fake model (fake_server.py --mode pi-tools, a frozen file) always calls the write tool on
  # "hello.py" and then runs `python3 hello.py`. The staged hello.py is mode 0444, so pi's write
  # tool cannot overwrite it (EACCES) and the scripted bash command runs the staged script: it
  # stands in for the model commands of packet section 9 (sleep 600, `yes | head -c 200000000`,
  # the F1 forged lines). (A sitecustomize.py in the task directory does not work: CPython
  # imports sitecustomize before the script directory joins sys.path.)
  mkdir -p "$1"
  rm -f "$1/hello.py"
  case "$2" in
    plain) : ;;
    sleep) printf 'import time\ntime.sleep(600)\n' > "$1/hello.py";;
    flood) cat > "$1/hello.py" <<'FIX'
# The model floods the shared container stdout (packet d1: `yes | head -c 200000000`).
# One 64 KiB line per chunk: the launcher reader is line-based (T22), counts ${#line} + 1 bytes,
# and the stamped file must stay within the d1 bound.
try:
    handle = open("/proc/1/fd/1", "wb")
    chunk = b"x" * 65535 + b"\n"
    for _ in range(4000):          # 262 MB on the container stdout
        handle.write(chunk)
    handle.flush()
except OSError:
    pass
FIX
      ;;
    floodnn) cat > "$1/hello.py" <<'FIX'
# Round 5, blocking item 1: the same flood with NO newline at all. A `read -r line` stamper
# would buffer all 256 MB in launcher memory; the bounded byte reader must stop storing at
# the 48 MiB cap, keep draining, and end with the cap-exit status (docker kill, 137).
try:
    handle = open("/proc/1/fd/1", "wb")
    for _ in range(4000):          # 256 MB with no newline anywhere
        handle.write(b"x" * 65536)
    handle.flush()
except OSError:
    pass
FIX
      ;;
    errflood) cat > "$1/hello.py" <<'FIX'
# Round 5, blocking item 2: the model floods the container stderr (the `yes >&2` shape). The
# docker client stderr goes to pi/accb-peak.txt and must be capped at 16 MiB while the client
# is never blocked or SIGPIPEd and the run itself is unaffected.
try:
    handle = open("/proc/1/fd/2", "wb")
    for _ in range(4000):          # 256 MB on the container stderr
        handle.write(b"e" * 65536)
    handle.flush()
except OSError:
    pass
print(2 + 3)
FIX
      ;;
    keysleep) cat > "$1/hello.py" <<'FIX'
import os, time
# T16 (design 6, residual): the model reads the run key from its own environment and puts it
# into the shared stdout, then hangs. The transcript then holds the key, so every exit path
# (the deadline, SIGTERM) must scan the evidence for it.
try:
    out = open("/proc/1/fd/1", "w")
    out.write(os.environ.get("ENGINESRV_API_KEY", "no-key") + "\n")
    out.close()
except OSError:
    pass
print(2 + 3)
time.sleep(600)
FIX
      ;;
    lateforge) cat > "$1/hello.py" <<'FIX'
import subprocess
# Design 4.4 step 5 (the PID sweep over /proc): a leftover process must not outlive pi. This fixture
# leaves a watcher behind that waits for pi to end, then bursts forged ACCB-EXPORT lines onto
# the shared stdout. With the sweep the watcher is dead before the entry prints the real
# export, so the LAST marker line is the real one; without it, a forged line becomes the last
# marker and the extractor refuses the archive. The /opt/accb guard keeps the rerun in the
# checker sandbox silent, and the 12 MiB file stretches the entry tail over the burst. The
# big file is written only where /work is writable (the run container); the read-only rerun
# sandbox of the checker skips it and hello.py still prints exactly 5 there.
try:
    with open("/work/big.bin", "wb") as handle:
        handle.write(b"z" * (12 * 1024 * 1024))
except OSError:
    pass
script = """
[ -e /opt/accb/entry.sh ] || exit 0
while true; do
  found=0
  for p in /proc/[0-9]*/cmdline; do
    if tr "\\000" " " < "$p" 2>/dev/null | grep -q "pi.*cli[.]js"; then found=1; fi
  done
  [ "$found" = 0 ] && break
  sleep 0.05
done
i=0
while [ "$i" -lt 20000 ]; do
  printf "ACCB-EXPORT LATE-FORGE\\n" > /proc/1/fd/1
  i=$((i + 1))
done
"""
subprocess.Popen(["/bin/sh", "-c", script])
print(2 + 3)
FIX
      ;;
    forge) cat > "$1/hello.py" <<'FIX'
import os
# F1 (design 4.5): a model command writes forged lines into the shared container stdout.
# All three go to PID 1's stdout so the check finds them raw in pi.stamped.tsv. The guard keeps
# the rerun of the exported hello.py clean: in the sandbox of check_pi_run.py, /proc/1/fd/1 is
# the rerun's own captured stdout, and the script must print exactly 5 there.
if os.path.exists("/opt/accb/entry.sh"):
    try:
        out = open("/proc/1/fd/1", "w")
        out.write("ACCB-EXIT 42\n")
        out.write("ACCB-EXPORT FORGED\n")
        out.write('{"type":"forged"}\n')
        out.close()
    except OSError:
        pass
print(2 + 3)
FIX
      ;;
    keyleak) printf 'the key is %s\n' "$TEST_KEY" > "$1/note.txt";;
  esac
  [ "$2" = plain ] || [ "$2" = keyleak ] || chmod 0444 "$1/hello.py"
}
run_launcher() { # <label> <out> <fixture kind> [env assignments...]: a full rehearsal run
  local label=$1 out=$2 kind=$3; shift 3
  mk_fixture "$TOP/fix-$kind" "$kind"
  env "$@" ACC_TEST_KEY="$TEST_KEY" bash "$AHD/run-acc-b.sh" "$out" --rehearse \
    --fixture "$TOP/fix-$kind" > "$TOP/$label.log" 2>&1
}
bg_run() { # <label> <fixture kind> [env assignments...]: a rehearsal launcher in the background; sets RUN_PID
  local label=$1 kind=$2; shift 2
  mk_fixture "$TOP/fix-$kind" "$kind"
  env "$@" ACC_TEST_KEY="$TEST_KEY" bash "$AHD/run-acc-b.sh" "$TOP/out-$label" --rehearse \
    --fixture "$TOP/fix-$kind" > "$TOP/$label.log" 2>&1 &
  RUN_PID=$!
}
wait_container() { # until the run's container (accb-<time>-<hex>) runs; sets CONT; fails when the
  # launcher ended. The accb-t- test containers are skipped: the launcher's own S12 probe
  # (accb-t-s12-…) starts before the run, and a check that inspects the run must not wake on it.
  local n=0; CONT=""
  while [ "$n" -lt 240 ]; do
    CONT=$($DK ps --format '{{.Names}}' --filter name=accb- 2>/dev/null | grep -v '^accb-t-' | head -1)
    [ -n "$CONT" ] && return 0
    kill -0 "$RUN_PID" 2>/dev/null || return 1
    sleep 0.5; n=$((n + 1))
  done
  return 1
}
end_run() { # SIGTERM to a background launcher; sets RUN_STATUS
  kill -TERM "$RUN_PID" 2>/dev/null
  local n=0
  while kill -0 "$RUN_PID" 2>/dev/null && [ "$n" -lt 120 ]; do sleep 1; n=$((n + 1)); done
  kill -0 "$RUN_PID" 2>/dev/null && kill -KILL "$RUN_PID" 2>/dev/null
  wait "$RUN_PID" 2>/dev/null; RUN_STATUS=$?
  RUN_PID=""
}
relay_up() { # <socket dir>: a relay on the spec IPC directory, mode 666 (design 4.2)
  rm -f "$SPEC/ipc/fifi.sock"
  python3 -I "$AHD/accb-relay.py" --socket "$SPEC/ipc/fifi.sock" --target "127.0.0.1:$ACCB_TEST_PORT" &
  RELAY_PID=$!
  local n=0
  until [ -S "$SPEC/ipc/fifi.sock" ] || [ "$n" -ge 50 ]; do sleep 0.1; n=$((n + 1)); done
  chmod 666 "$SPEC/ipc/fifi.sock"
}
relay_down() { kill "$RELAY_PID" 2>/dev/null; wait "$RELAY_PID" 2>/dev/null; RELAY_PID=""; }
black_hole() { # a listener that accepts and never answers: pi hangs on it (M1)
  python3 -I - "$ACCB_TEST_PORT" <<'BH' &
import socket
import sys
server = socket.socket(); server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", int(sys.argv[1]))); server.listen(64)
conns = []
while True:
    try:
        conn, _ = server.accept()
    except OSError:
        break
    conns.append(conn)
BH
  HOLE_PID=$!
}
hole_down() { kill "$HOLE_PID" 2>/dev/null; wait "$HOLE_PID" 2>/dev/null; HOLE_PID=""; }

# --- stage 1a: host facts ---------------------------------------------------------------------------
chk_p1() {
  if $DK info --format '{{.SecurityOptions}}' 2>/dev/null | grep -q rootless; then ok "p1: the daemon reports rootless"; else bad "p1: no rootless in the security options"; fi
}
chk_p3() {
  local id digests
  id=$($DK image inspect --format '{{.Id}}' "$IMG" 2>/dev/null) || { bad "p3: no image"; return; }
  digests=$($DK image inspect --format '{{.RepoDigests}}' "$IMG" 2>/dev/null)
  case "$id" in sha256:a6e34c598f24*) ok "p3: the image id starts with the pinned prefix ($id)";; *) bad "p3: image id $id";; esac
  echo "$digests" | grep -q "sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1" \
    && ok "p3: RepoDigests holds the pinned digest" || bad "p3: RepoDigests $digests"
  echo "$id" > "$TOP/p3-image-id.txt"; $DK history --no-trunc "$IMG" > "$TOP/p3-history.txt" 2>&1
}
chk_p6() {
  local leftovers
  leftovers=$($DK ps -aq --filter name=accb- 2>/dev/null)
  [ -z "$leftovers" ] && ok "p6: no accb- container exists" || bad "p6: leftover containers: $leftovers"
  mkdir -p /run/user/1000/acc-b
  local in_dir; in_dir=$(ls -A /run/user/1000/acc-b 2>/dev/null | grep -vx lock || true)
  [ -z "$in_dir" ] && ok "p6: the acc-b directory holds only lock" || bad "p6: the acc-b directory holds $in_dir"
  flock -n /run/user/1000/acc-b/lock -c true 2>/dev/null && ok "p6: the lock is free" || bad "p6: the lock is held with no run"
  # The launcher must really take it (C21): while a rehearsal runs, flock -n fails.
  bg_run p6 sleep
  if wait_container; then
    if flock -n /run/user/1000/acc-b/lock -c true 2>/dev/null; then bad "p6: the lock stayed free during a run (C21)"; else ok "p6: the launcher holds the lock during a run"; fi
    end_run
    expect_status 143 "$RUN_STATUS" "p6: the run ends with the signal status"
  else
    bad "p6: the rehearsal never started a container"
    [ -n "$RUN_PID" ] && end_run
  fi
}
chk_p8() {
  local free_b disk_b
  free_b=$(free -b | awk '/^Mem:/ {print $7}'); disk_b=$(df -B1 / | awk 'NR==2 {print $4}')
  if [ "$free_b" -gt 4294967296 ]; then ok "p8: ${free_b} bytes of memory free (> 4 GiB)"; else bad "p8: only ${free_b} bytes free"; fi
  if [ "$disk_b" -gt 21474836480 ]; then ok "p8: ${disk_b} bytes of disk free (> 20 GiB)"; else bad "p8: only ${disk_b} bytes free"; fi
}
chk_p10() {
  if [ -n "$P10" ]; then ok "p10: the baseline of other containers is recorded: $(echo "$P10" | tr '\n' ';')"; else ok "p10: no other container runs"; fi
}

# --- stage 1a: security flags -----------------------------------------------------------------------
chk_s1() {
  local out; out=$(sbx accb-t-s1 /usr/local/bin/python -I -c "
import re
status = open('/proc/self/status').read()
caps = [line.split(':')[1].strip() for line in status.splitlines() if line.startswith(('CapInh','CapPrm','CapEff','CapBnd','CapAmb'))]
print('caps', caps)")
  [ "$out" = "caps ['0000000000000000', '0000000000000000', '0000000000000000', '0000000000000000', '0000000000000000']" ] \
    && ok "s1: all capability sets are empty" || bad "s1: $out"
  $DK rm -f accb-t-s1 >/dev/null 2>&1; CLEANUP_CONT=""
}
chk_s2() {
  local out; out=$(sbx accb-t-s2 /usr/local/bin/python -I -c "
status = open('/proc/self/status').read()
nnp = [l.split(':')[1].strip() for l in status.splitlines() if l.startswith('NoNewPrivs')]
import os
print('nnp', nnp, 'uid', os.getuid())")
  [ "$out" = "nnp ['1'] uid 65534" ] && ok "s2: NoNewPrivs is 1 and the uid is 65534" || bad "s2: $out"
  $DK rm -f accb-t-s2 >/dev/null 2>&1; CLEANUP_CONT=""
}
chk_s3() {
  local out; out=$(sbx accb-t-s3 /usr/local/bin/python -I -c "
seccomp = [l.split(':')[1].strip() for l in open('/proc/self/status').read().splitlines() if l.startswith('Seccomp:')]
print('seccomp', seccomp)")
  [ "$out" = "seccomp ['2']" ] && ok "s3: Seccomp is 2" || bad "s3: $out"
  $DK rm -f accb-t-s3 >/dev/null 2>&1; CLEANUP_CONT=""
}
chk_s4() {
  local out; out=$(sbx accb-t-s4 /usr/local/bin/python -I -c "print(open('/proc/self/uid_map').read().strip())")
  [ "$out" = "0 0 4294967295" ] && bad "s4: the uid_map is the rootful full mapping ($out)" \
    || ok "s4: the uid_map is a rootless mapping ($out)"
  $DK rm -f accb-t-s4 >/dev/null 2>&1; CLEANUP_CONT=""
}
chk_s5() {
  local out; out=$(sbx accb-t-s5 /bin/bash -c 'su -c id >/dev/null 2>&1; echo su=$?; mount -t tmpfs x /mnt >/dev/null 2>&1; echo mount=$?')
  local su_rc mount_rc
  su_rc=$(printf '%s\n' "$out" | sed -n 's/^su=//p'); mount_rc=$(printf '%s\n' "$out" | sed -n 's/^mount=//p')
  if [ -n "$su_rc" ] && [ "$su_rc" != 0 ] && [ -n "$mount_rc" ] && [ "$mount_rc" != 0 ]; then
    ok "s5: su fails ($su_rc) and mount fails ($mount_rc) as 65534"
  else
    bad "s5: su=$su_rc mount=$mount_rc"
  fi
  $DK rm -f accb-t-s5 >/dev/null 2>&1; CLEANUP_CONT=""
}
chk_s6() {
  local out; out=$(sbx accb-t-s6 /usr/local/bin/python -I -c "
import ctypes, errno
libc = ctypes.CDLL(None, use_errno=True)
rc = libc.unshare(0x10000000)   # CLONE_NEWUSER
print('rc', rc, 'errno', ctypes.get_errno(), ctypes.get_errno() == errno.EPERM)")
  [ "$out" = "rc -1 errno 1 True" ] && ok "s6: unshare(CLONE_NEWUSER) fails with EPERM" || bad "s6: $out"
  $DK rm -f accb-t-s6 >/dev/null 2>&1; CLEANUP_CONT=""
}
chk_s7() {
  local out; out=$(sbx accb-t-s7 /usr/local/bin/python -I -c "
import socket, errno
results = []
for family, args in ((socket.AF_PACKET, (socket.AF_PACKET, socket.SOCK_RAW)), (socket.AF_INET, (socket.AF_INET, socket.SOCK_RAW, 1))):
    try:
        socket.socket(*args).close(); results.append('open')
    except OSError as error:
        results.append(error.errno == errno.EPERM)
print('raw', results)")
  [ "$out" = "raw [True, True]" ] && ok "s7: raw packet and raw inet sockets fail with EPERM" || bad "s7: $out"
  $DK rm -f accb-t-s7 >/dev/null 2>&1; CLEANUP_CONT=""
}
chk_s8() {
  local out; out=$(sbx accb-t-s8 /usr/local/bin/python -I -c "
import errno, os
results = []
for path in ('/proc/sysrq-trigger', '/sys/fs/cgroup/cgroup.procs'):
    try:
        handle = open(path, 'wb'); handle.write(b'x'); results.append('wrote ' + path)
    except OSError as error:
        results.append(error.errno)
try:
    data = open('/proc/kcore', 'rb').read(16)
    results.append('read kcore' if data else 'kcore masked')
except OSError as error:
    results.append('kcore ' + str(error.errno))
try:
    os.mknod('/tmp/n', 0o600 | 0o020000, os.makedev(1, 3)); results.append('made node')
except OSError as error:
    results.append('mknod ' + str(error.errno == errno.EPERM))
print('s8', results)")
  echo "$out" | grep -q "wrote \|read kcore\|made node" && bad "s8: a protected write succeeded: $out" \
    || ok "s8: sysrq, cgroup.procs, kcore, and mknod all fail ($out)"
  $DK rm -f accb-t-s8 >/dev/null 2>&1; CLEANUP_CONT=""
}
chk_s9() {
  # The harness never allocates a tty, so /dev holds the default set without `console`
  # (runc creates /dev/console only with a tty; recorded difference to the checklist list).
  local out; out=$(sbx accb-t-s9 /bin/bash -c 'ls /dev' | sort | tr '\n' ' ')
  local want="core fd full mqueue null ptmx pts random shm stderr stdin stdout tty urandom zero "
  if [ "$out" = "$want" ]; then ok "s9: /dev holds only the default minimal set (no console without a tty)"; else bad "s9: /dev holds: $out (want $want)"; fi
  $DK rm -f accb-t-s9 >/dev/null 2>&1; CLEANUP_CONT=""
}
chk_s10() {
  sbx accb-t-s10 /bin/bash -c 'sleep 8' >/dev/null 2>&1 &
  local n=0 sample
  until [ -n "$(pgrep -f 'accb-t-s10' | head -1)" ] || [ "$n" -ge 20 ]; do sleep 0.3; n=$((n + 1)); done
  sample=$(ps -eo args 2>/dev/null | grep -F 'docker' | grep -v grep || true)
  if printf '%s\n' "$sample" | grep -q '/var/run/docker.sock'; then bad "s10: a docker client names /var/run/docker.sock: $sample"; else ok "s10: no docker client names /var/run/docker.sock"; fi
  wait
  $DK rm -f accb-t-s10 >/dev/null 2>&1; CLEANUP_CONT=""
}
chk_s12() {
  local out probe
  probe=accb-t-s12-chk
  out=$(sbx "$probe" /bin/true)
  $DK inspect "$probe" > "$TOP/s12-inspect.json" 2>/dev/null
  local result
  result=$(python3 -I - "$TOP/s12-inspect.json" "$SPEC" <<'S12PY'
import json, sys
data = json.load(open(sys.argv[1]))
run = data[0] if isinstance(data, list) else data
hc, cfg = run["HostConfig"], run["Config"]
spec = sys.argv[2]
problems = []
def need(ok, what):
    if not ok:
        problems.append(what)
need(hc.get("Privileged") is False, "Privileged false")
need(hc.get("NetworkMode") == "none", "NetworkMode none")
need(hc.get("ReadonlyRootfs") is True, "ReadonlyRootfs true")
need(hc.get("CapDrop") == ["ALL"], "CapDrop ALL")
need(not hc.get("CapAdd"), "CapAdd empty")
need((hc.get("SecurityOpt") or []) == ["no-new-privileges"],
     "SecurityOpt exactly no-new-privileges, got %r" % (hc.get("SecurityOpt"),))
need(hc.get("PidMode") == "", "PidMode default, got %r" % (hc.get("PidMode"),))
need(hc.get("IpcMode") == "private", "IpcMode private, got %r" % (hc.get("IpcMode"),))
need(hc.get("UsernsMode") == "", "UsernsMode default, got %r" % (hc.get("UsernsMode"),))
need(hc.get("PidsLimit") == 128, "PidsLimit 128")
need(hc.get("Memory") == 1073741824, "Memory 1073741824")
need(hc.get("MemorySwap") == 1073741824, "MemorySwap 1073741824")
need(hc.get("NanoCpus") == 1000000000, "NanoCpus 1000000000")
need(hc.get("ShmSize") == 16777216, "ShmSize 16777216")
need(sorted(hc.get("Ulimits") or [], key=lambda u: u.get("Name", "")) ==
     [{"Name": "core", "Hard": 0, "Soft": 0}, {"Name": "nofile", "Hard": 512, "Soft": 512}],
     "Ulimits core=0 and nofile=512:512, got %r" % (hc.get("Ulimits"),))
need(hc.get("Tmpfs") ==
     {"/home/acc": "rw,noexec,nosuid,nodev,size=32m,uid=65534,gid=65534,mode=0755",
      "/tmp": "rw,noexec,nosuid,nodev,size=16m,mode=1777",
      "/work": "rw,noexec,nosuid,nodev,size=16m,uid=65534,gid=65534,mode=0755"},
     "tmpfs option strings of design 4.3, got %r" % (hc.get("Tmpfs"),))
need(cfg.get("User") == "65534:65534", "User 65534:65534")
need((hc.get("LogConfig") or {}).get("Type") == "none", "LogConfig.Type none")
need(not (hc.get("PortBindings") or {}), "PortBindings empty")
need(not (hc.get("Devices") or []), "Devices empty")
env = cfg.get("Env") or []
need(not [e for e in env if e.startswith("ENGINESRV_API_KEY")], "Config.Env holds no key")
mounts = run.get("Mounts") or []
need(len(mounts) == 7, "exactly seven bind mounts, got %d" % len(mounts))
for m in mounts:
    need(m.get("RW") is False, "mount %s is not read-only" % m.get("Destination"))
    need("docker.sock" not in (m.get("Source") or ""), "no docker.sock mount")
    need(not (m.get("Source") or "").startswith(("/home/dan/.pi", "/home/dan/.local")),
         "no live pi or node path: %s" % m.get("Source"))
need({m.get("Destination") for m in mounts} ==
     {"/fixture", "/opt/pi", "/opt/node/bin/node", "/in/agent",
      "/opt/accb/entry.sh", "/opt/accb/bridge.js", "/ipc"},
     "the seven destinations of design 4.3")
want_sources = {"/fixture": spec + "/fixture", "/opt/pi": spec + "/opt/pi",
                "/opt/node/bin/node": spec + "/opt/node", "/in/agent": spec + "/agent",
                "/opt/accb/entry.sh": spec + "/entry.sh",
                "/opt/accb/bridge.js": spec + "/bridge.js", "/ipc": spec + "/ipc"}
for m in mounts:
    need(m.get("Source") == want_sources.get(m.get("Destination")),
         "mount source of %s is %r, want %r" % (m.get("Destination"), m.get("Source"),
                                                want_sources.get(m.get("Destination"))))
need(set((hc.get("Tmpfs") or {})) == {"/tmp", "/work", "/home/acc"},
     "tmpfs exactly /tmp /work /home/acc")
print("ok" if not problems else "; ".join(problems))
S12PY
)
  if [ "$result" = ok ]; then ok "s12: docker inspect equals design 4.3 field by field"; else bad "s12: $result"; fi
  $DK rm -f "$probe" >/dev/null 2>&1; CLEANUP_CONT=""
}

# --- stage 1a: mounts -------------------------------------------------------------------------------
MOUNTINFO_PY="mounts = []
for line in open('/proc/self/mountinfo'):
    fields = line.split()
    mounts.append((fields[4], fields[5].split(',')))
for dest, opts in mounts:
    print(dest, 'ro' if 'ro' in opts else 'rw')"
chk_h1() {
  local out; out=$(sbx accb-t-h1 /usr/local/bin/python -I -c "$MOUNTINFO_PY")
  local result
  result=$(python3 -I - <<PY
lines = """$out""".splitlines()
allowed = {'/', '/proc', '/dev', '/dev/pts', '/dev/mqueue', '/dev/shm', '/sys', '/sys/fs/cgroup',
           '/etc/hosts', '/etc/hostname', '/etc/resolv.conf', '/tmp', '/work', '/home/acc',
           '/fixture', '/opt/pi', '/opt/node/bin/node', '/in/agent', '/opt/accb/entry.sh',
           '/opt/accb/bridge.js', '/ipc'}
# runc binds the device nodes into /dev as individual mounts and masks /proc and /sys paths;
# the design names these implicit mounts allowed (4.3 note, H2).
allowed |= {'/dev/null', '/dev/zero', '/dev/full', '/dev/random', '/dev/urandom', '/dev/tty'}
dests = [l.split(' ', 1)[0] for l in lines if l.strip()]
extra = [d for d in dests if d not in allowed and not (d.startswith('/proc/') or d.startswith('/sys/'))]
home = [d for d in dests if d.startswith('/home/dan') or d.startswith('/run/user')]
print('ok' if not extra and not home else 'extra %s under %s' % (extra, home))
PY
)
  if [ "$result" = ok ]; then ok "h1: the mount destinations are exactly the allowed set"; else bad "h1: $result"; fi
  $DK rm -f accb-t-h1 >/dev/null 2>&1; CLEANUP_CONT=""
}
chk_h2() {
  local out; out=$(sbx accb-t-h2 /usr/local/bin/python -I -c "$MOUNTINFO_PY")
  local result
  result=$(python3 -I - <<PY
writable = [l[:-3] for l in """$out""".splitlines() if l.strip().endswith(' rw')]
exempt = {'/proc', '/dev', '/dev/pts', '/dev/mqueue', '/dev/shm', '/tmp', '/work', '/home/acc', '/sys/fs/cgroup'}
bad = [w for w in writable if w not in exempt
       and not (w.startswith('/proc/') or w.startswith('/sys/') or w.startswith('/dev/'))]
root_ro = any(l.startswith('/ ro') for l in """$out""".splitlines())
print('ok' if not bad and root_ro else 'writable %s root_ro %s' % (bad, root_ro))
PY
)
  if [ "$result" = ok ]; then ok "h2: every non-exempt destination is read-only and / is ro"; else bad "h2: $result"; fi
  $DK rm -f accb-t-h2 >/dev/null 2>&1; CLEANUP_CONT=""
}
chk_h3() {
  local out; out=$(sbx accb-t-h3 /bin/bash -c 'ls /home; ls -A /ipc' | tr '\n' ' ')
  [ "$out" = "acc " ] && ok "h3: /home holds only acc and /ipc is empty" || bad "h3: got '$out'"
  $DK rm -f accb-t-h3 >/dev/null 2>&1; CLEANUP_CONT=""
}

# --- stage 1a: network ------------------------------------------------------------------------------
chk_n1() {
  local out; out=$(sbx accb-t-n1 /usr/local/bin/python -I -c "$PY_RUN
lines = [l.split() for l in open('/proc/net/if_inet6').read().splitlines() if l.strip()]
print('dev ok, if_inet6', all(l[-1] == 'lo' for l in lines) and len(lines) >= 1)")
  [ "$out" = "dev ok, if_inet6 True" ] \
    && ok "n1: lo is the only interface and ::1 the only IPv6 address" || bad "n1: $out"
  $DK rm -f accb-t-n1 >/dev/null 2>&1; CLEANUP_CONT=""
}
net_check() { # <name> <python body>: runs NETPRE + the body, expects a nonzero exit or a False flag
  local name=$1 body=$2 out status
  out=$(sbx "accb-t-$name" /usr/local/bin/python -I -c "$PY_RUN
$body")
  status=$?
  if [ "$status" = 99 ]; then bad "$name: NETPRE failed: $out"; else printf '%s' "$out"; fi
  $DK rm -f "accb-t-$name" >/dev/null 2>&1; CLEANUP_CONT=""
  return $status
}
chk_n2() {
  local out; out=$(net_check n2 "
import socket
try:
    socket.socket().settimeout(3); socket.create_connection(('1.1.1.1', 443), timeout=3); print('CONNECTED')
except OSError as error:
    print('unreachable', error.errno is not None)")
  [ "$out" = "unreachable True" ] && ok "n2: 1.1.1.1:443 is unreachable" || bad "n2: $out"
}
chk_n3() {
  local out; out=$(net_check n3 "
import socket
try:
    socket.getaddrinfo('example.com', 443); print('RESOLVED')
except socket.gaierror:
    print('gaierror')")
  [ "$out" = "gaierror" ] && ok "n3: DNS raises gaierror" || bad "n3: $out"
}
chk_n4() {
  local out; out=$(net_check n4 "
import socket
results = []
for port in (22, 18792):
    try:
        socket.create_connection(('192.168.0.16', port), timeout=3); results.append('CONNECTED %d' % port)
    except OSError:
        results.append('failed %d' % port)
print(' ', results)")
  echo "$out" | grep -q CONNECTED && bad "n4: a Mac port answered: $out" || ok "n4: 192.168.0.16:22 and :18792 both fail ($out)"
}
chk_n5() {
  local out; out=$(net_check n5 "
import socket
results = []
for host, port in (('127.0.0.1', 22), ('127.0.0.1', 4546), ('172.17.0.1', 22), ('192.168.0.20', 22)):
    try:
        socket.create_connection((host, port), timeout=3); results.append('CONNECTED %s' % host)
    except OSError:
        results.append('failed %s' % host)
print(' ', results)")
  echo "$out" | grep -q CONNECTED && bad "n5: a host port answered: $out" || ok "n5: loopback and LAN host ports all fail ($out)"
}
chk_n6() {
  local out; out=$(net_check n6 "
import subprocess
done = subprocess.run(['ls', '/var/run/docker.sock', '/run/docker.sock', '/run/user'], capture_output=True, text=True)
print(' ', done.stderr.count('No such file or directory'))")
  [ "$out" = "  3" ] && ok "n6: no docker socket and no /run/user is visible" || bad "n6: $out"
}
chk_n7() {
  local out; out=$(net_check n7 "
import socket
results = []
for address in (('2606:4700:4700::1111', 443), ('::1', 22)):
    try:
        socket.socket(socket.AF_INET6); socket.create_connection(address, timeout=3); results.append('CONNECTED')
    except OSError:
        results.append('failed')
print(' ', results)")
  echo "$out" | grep -q CONNECTED && bad "n7: IPv6 reached out: $out" || ok "n7: IPv6 to the internet and to ::1 fails ($out)"
}
chk_n8() {
  local out; out=$(net_check n8 "
lines = [l.split() for l in open('/proc/net/unix').read().splitlines()[1:]]
bad = [l[-1] for l in lines if l and (l[-1].startswith(('/run', '/var/run', '/home')) or l[-1].startswith('@'))]
print(' bad', bad)")
  [ "$out" = " bad []" ] && ok "n8: /proc/net/unix shows no host socket" || bad "n8: $out"
}

# --- stage 1a: writes and files ----------------------------------------------------------------------
chk_w1() {
  local out; out=$(sbx accb-t-w1 /bin/bash -c 'for p in /x /opt/pi/x /fixture/x /ipc/x /in/agent/x /opt/accb/x; do touch "$p" 2>&1 | head -1; done')
  local count; count=$(printf '%s\n' "$out" | grep -c "Read-only file system")
  [ "$count" = 6 ] && ok "w1: /x and the five read-only mounts all fail with Read-only file system" || bad "w1: only $count of 6 refused: $out"
  $DK rm -f accb-t-w1 >/dev/null 2>&1; CLEANUP_CONT=""
}
chk_w3() {
  local out; out=$(sbx accb-t-w3 /bin/bash -c 'cp /bin/true /tmp/t && /tmp/t; echo tmp=$?; cp /bin/true /work/t && /work/t; echo work=$?; printf "print(7*6)\n" > /tmp/s.py; /usr/local/bin/python -I /tmp/s.py')
  local tmp_rc work_rc calc
  tmp_rc=$(printf '%s\n' "$out" | sed -n 's/^tmp=//p'); work_rc=$(printf '%s\n' "$out" | sed -n 's/^work=//p'); calc=$(printf '%s\n' "$out" | tail -1)
  if [ "$tmp_rc" = 126 ] && [ "$work_rc" = 126 ] && [ "$calc" = 42 ]; then
    ok "w3: /tmp and /work are noexec (126) while python still runs files from them"
  else
    bad "w3: tmp=$tmp_rc work=$work_rc python=$calc: $out"
  fi
  $DK rm -f accb-t-w3 >/dev/null 2>&1; CLEANUP_CONT=""
}
chk_w4() {
  p8_guard w4 || return 0
  local out; out=$(sbx accb-t-w4 /bin/bash -c 'for d in /work /tmp /home/acc /dev/shm; do dd if=/dev/zero of=$d/f bs=1M count=100 2>&1 | tail -1; echo "$d $(stat -c %s $d/f 2>/dev/null)"; rm -f $d/f; done')
  local result
  result=$(python3 -I - <<PY
lines = [l for l in """$out""".splitlines() if l]
want = {'/work': 16777216, '/tmp': 16777216, '/home/acc': 33554432, '/dev/shm': 16777216}
ok_all = all(any(l.startswith(d + ' ') and l.split()[1].isdigit() and int(l.split()[1]) <= cap
                 for l in lines) for d, cap in want.items())
print('ok' if ok_all else 'over the cap')
PY
)
  if [ "$result" = ok ]; then ok "w4: the tmpfs mounts fill at their caps (16/16/32/16 MiB)"; else bad "w4: $out"; fi
  $DK rm -f accb-t-w4 >/dev/null 2>&1; CLEANUP_CONT=""
}
chk_w5() {
  local out; out=$(sbx accb-t-w5 /bin/bash -c 'df -B1 /tmp /work /home/acc /dev/shm' | awk 'NR>1 {print $6, $2}')
  local result
  result=$(python3 -I - <<PY
rows = [tuple(l.split()) for l in """$out""".splitlines() if l]
want = {'/tmp': 16777216, '/work': 16777216, '/home/acc': 33554432, '/dev/shm': 16777216}
good = all(dest in want and abs(int(size) - want[dest]) < want[dest] // 50 for dest, size in rows) and len(rows) == 4
print('ok' if good else rows)
PY
)
  if [ "$result" = ok ]; then ok "w5: df shows 16M/16M/32M/16M"; else bad "w5: $result"; fi
  $DK rm -f accb-t-w5 >/dev/null 2>&1; CLEANUP_CONT=""
}

# --- stage 1a: limits -------------------------------------------------------------------------------
chk_l1() {
  p8_guard l1 || return 0
  local out
  out=$(sbx accb-t-l1 /usr/local/bin/python -I -c "
import os, sys
limits = open('/sys/fs/cgroup/pids.max').read().strip()
print('pids.max', limits, flush=True)
if limits != '128':
    sys.exit(99)
import time
children = []
try:
    while True:
        pid = os.fork()
        if pid == 0:
            time.sleep(60); os._exit(0)
        children.append(pid)
except BlockingIOError:
    print('eagain after', len(children), flush=True)
events = dict(l.split() for l in open('/sys/fs/cgroup/pids.events').read().splitlines())
print('pids.events', events, flush=True)
sys.stdout.flush()
os._exit(0)")
  local rc=$?
  CLEANUP_CONT=""; watch accb-t-l1 30
  if [ "$rc" = 99 ]; then bad "l1a: pids.max is not 128 (abort 99)"; else ok "l1a: pids.max is 128"; fi
  echo "$out" | grep -q "eagain after" && ok "l1b: the fork loop hits EAGAIN under 129 children ($(echo "$out" | sed -n 's/eagain after //p'))" || bad "l1b: $out"
  echo "$out" | grep -q "pids.events.*max 1\|max [1-9]" && ok "l1b: pids.events shows max > 0" || bad "l1b: pids.events shows no max: $out"
  $DK rm -f accb-t-l1 >/dev/null 2>&1
  # L1c: the bash fork bomb stays under the limit and leaves nothing.
  sbx accb-t-l1 /bin/bash -c ":(){ :|:& };:" >/dev/null 2>&1 &
  sleep 5
  local current
  current=$($DK exec accb-t-l1 cat /sys/fs/cgroup/pids.current 2>/dev/null || echo gone)
  $DK kill accb-t-l1 >/dev/null 2>&1
  [ "$current" = "gone" ] || [ "$current" -le 128 ] 2>/dev/null \
    && ok "l1c: the fork bomb stays within pids.current 128 ($current)" || bad "l1c: pids.current $current"
  $DK rm -f accb-t-l1 >/dev/null 2>&1; CLEANUP_CONT=""
  [ -z "$($DK ps -aq --filter name=accb-)" ] && ok "l1: no container is left" || bad "l1: a container is left"
  others_unchanged l1
}
chk_l2() {
  p8_guard l2 || return 0
  local out rc limits swap
  out=$(sbx accb-t-l2 /usr/local/bin/python -I -c "
import sys
limits = open('/sys/fs/cgroup/memory.max').read().strip()
swap = open('/sys/fs/cgroup/memory.swap.max').read().strip()
print('memory.max', limits, 'memory.swap.max', swap)
if limits != '1073741824' or swap != '0':
    sys.exit(99)
blob = bytearray(2 * 1024 * 1024 * 1024)
for i in range(0, len(blob), 4096):
    blob[i] = 1
print('allocated')")
  rc=$?
  CLEANUP_CONT=""
  if [ "$rc" = 99 ]; then bad "l2a: the memory limits are not 1073741824/0 (abort 99)"; return; fi
  ok "l2a: memory.max 1073741824 and memory.swap.max 0"
  [ "$rc" = 137 ] && ok "l2b: the 2 GiB allocation ends with status 137" || bad "l2b: status $rc ($out)"
  local oom
  oom=$($DK inspect --format '{{.State.OOMKilled}}' accb-t-l2 2>/dev/null)
  [ "$oom" = "true" ] && ok "l2b: OOMKilled is true" || bad "l2b: OOMKilled is $oom"
  $DK rm -f accb-t-l2 >/dev/null 2>&1
  others_unchanged l2
}
chk_l4() {
  p8_guard l4 || return 0
  local out
  out=$(sbx accb-t-l4 /bin/bash -c '
    end=$((SECONDS + 10))
    busy() { while [ $SECONDS -lt $end ]; do :; done; }
    busy & busy & wait
    cat /sys/fs/cgroup/cpu.stat')
  local usage throttled
  usage=$(printf '%s\n' "$out" | sed -n 's/^usage_usec //p'); throttled=$(printf '%s\n' "$out" | sed -n 's/^nr_throttled //p')
  if [ -n "$throttled" ] && [ "$throttled" -gt 0 ] && [ -n "$usage" ] && [ "$usage" -le 24000000 ]; then
    ok "l4: nr_throttled $throttled > 0 and usage_usec $usage is within 1.2x the wall time"
  else
    bad "l4: usage_usec ${usage:-missing}, nr_throttled ${throttled:-missing}"
  fi
  $DK rm -f accb-t-l4 >/dev/null 2>&1; CLEANUP_CONT=""
}
chk_m1() {
  p8_guard m1 || return 0
  black_hole
  relay_up "$SPEC/ipc"
  local t0 t1 args=()
  mapfile -t args < <(entry_args accb-t-m1 5)
  t0=$(date +%s%N)
  printf '%s\n' "$TEST_KEY" | timeout -s KILL 25 "${args[@]}" > "$TOP/m1.out" 2> "$TOP/m1.err"
  local rc=$?
  t1=$(date +%s%N)
  CLEANUP_CONT=""
  local ms=$(( (t1 - t0) / 1000000 ))
  if [ "$rc" = 137 ] && [ "$ms" -ge 4500 ] && [ "$ms" -le 12000 ]; then
    ok "m1: PI_TIMEOUT=5 with a hanging task ends 137 after ${ms}ms (5 to 8 s plus start-up)"
  else
    bad "m1: status $rc after ${ms}ms; stderr: $(head -3 "$TOP/m1.err")"
  fi
  $DK rm -f accb-t-m1 >/dev/null 2>&1
  [ -z "$($DK ps -aq --filter name=accb-)" ] && ok "m1: nothing is left" || bad "m1: a container is left"
  relay_down; hole_down
}
chk_d2() {
  local out probe
  probe=accb-t-d2
  out=$(sbx "$probe" /bin/bash -c 'echo hello-d2')
  [ "$out" = "hello-d2" ] || bad "d2: the probe container printed '$out'"
  local driver logs
  driver=$($DK inspect --format '{{.HostConfig.LogConfig.Type}}' "$probe" 2>/dev/null)
  logs=$($DK logs "$probe" 2>&1 || true)
  [ "$driver" = none ] && ok "d2: LogConfig.Type is none" || bad "d2: LogConfig.Type is $driver"
  if printf '%s\n' "$logs" | grep -qi "does not support reading\|no logs\|configured logging"; then
    ok "d2: docker logs reports that no logs are available"
  elif [ -z "$(printf '%s\n' "$logs" | tr -d '[:space:]')" ]; then
    ok "d2: docker logs prints nothing"
  else
    bad "d2: docker logs printed: $logs"
  fi
  $DK rm -f "$probe" >/dev/null 2>&1; CLEANUP_CONT=""
}

# --- stage 1b/2: launcher and sandbox ----------------------------------------------------------------
chk_flags() {
  local want=()
  mapfile -t want <<WANTLIST
docker
--host
unix:///run/user/1000/docker.sock
run
-i
--name
accb-t-spec
--pull
never
--network
none
--read-only
--user
65534:65534
--cap-drop
ALL
--security-opt
no-new-privileges
--memory
1024m
--memory-swap
1024m
--cpus
1
--pids-limit
128
--shm-size
16m
--ulimit
nofile=512:512
--ulimit
core=0
--log-driver
none
--tmpfs
/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777
--tmpfs
/work:rw,noexec,nosuid,nodev,size=16m,uid=65534,gid=65534,mode=0755
--tmpfs
/home/acc:rw,noexec,nosuid,nodev,size=32m,uid=65534,gid=65534,mode=0755
--mount
type=bind,src=$SPEC/fixture,dst=/fixture,ro
--mount
type=bind,src=$SPEC/opt/pi,dst=/opt/pi,ro
--mount
type=bind,src=$SPEC/opt/node,dst=/opt/node/bin/node,ro
--mount
type=bind,src=$SPEC/agent,dst=/in/agent,ro
--mount
type=bind,src=$SPEC/entry.sh,dst=/opt/accb/entry.sh,ro
--mount
type=bind,src=$SPEC/bridge.js,dst=/opt/accb/bridge.js,ro
--mount
type=bind,src=$SPEC/ipc,dst=/ipc,ro
-e
HOME=/home/acc
-e
PI_CODING_AGENT_DIR=/home/acc/agent
-e
PI_OFFLINE=1
-e
PI_SKIP_VERSION_CHECK=1
-e
PI_TELEMETRY=0
-e
PYTHONDONTWRITEBYTECODE=1
-e
PI_TIMEOUT=5400
-e
EXPORT_MAX=16777216
--entrypoint
/bin/bash
python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
/opt/accb/entry.sh
WANTLIST
  if [ "${#SPEC_ARGS[@]}" = "${#want[@]}" ] && [ "${SPEC_ARGS[*]}" = "${want[*]}" ]; then
    ok "flags: --print-spec prints the exact 4.3 argument list (${#want[@]} args)"
  else
    bad "flags: the spec list differs from design 4.3 ($(printf '%s\n' "${SPEC_ARGS[@]}" | md5sum | cut -c1-8) vs $(printf '%s\n' "${want[@]}" | md5sum | cut -c1-8))"
    diff <(printf '%s\n' "${SPEC_ARGS[@]}") <(printf '%s\n' "${want[@]}") | head -12
  fi
}
chk_sshg() {
  # The printed ssh_config must equal the 4.2 line list (C20), and ssh -G must show Appendix A (P9, C2).
  local want="Host mac
  HostName 192.168.0.16
  User dan
  IdentitiesOnly yes
  IdentityFile ~/.ssh/id_ed25519
  IdentityAgent none
  StrictHostKeyChecking yes
  UserKnownHostsFile $SPEC/known_hosts
  GlobalKnownHostsFile /dev/null
  BatchMode yes
  ControlMaster no
  ControlPath none
  ForwardAgent no
  ForwardX11 no
  PermitLocalCommand no
  ProxyCommand none
  ProxyJump none
  ClearAllForwardings no
  StreamLocalBindMask 0111
  StreamLocalBindUnlink no
  ExitOnForwardFailure yes
  ConnectTimeout 15
  ServerAliveInterval 30
  ServerAliveCountMax 3"
  if [ "$(cat "$TOP/spec-ssh-config")" = "$want" ]; then ok "sshg: the printed ssh_config equals design 4.2"; else bad "sshg: the printed ssh_config differs"; diff "$TOP/spec-ssh-config" <(printf '%s\n' "$want") | head; fi
  local forward
  forward=$(printf '%s\n' "$SPEC_SSH_CMD" | grep -o -- '-L [^ ]*' | sed 's/^-L //')
  [ -n "$forward" ] || { bad "sshg: no -L in the printed ssh command"; return; }
  # The printed forward must end at the design destination, written HERE as a literal: reading
  # the expected value back from the launcher's own spec compares the launcher with itself and
  # a wrong destination port survives every check (review finding 3, mutation c2).
  local want_dest=127.0.0.1:18791
  [ "$forward" = "$SPEC/ipc/fifi.sock:$want_dest" ] \
    || { bad "sshg: the forward is $forward, want $SPEC/ipc/fifi.sock:$want_dest (the design 4.2 literal, not the launcher's own spec)"; return; }
  local out
  out=$(ssh -G -F "$TOP/spec-ssh-config" -a -x -N -L "$forward" mac 2>&1) || { bad "sshg: ssh -G failed"; return; }
  local fails=""
  sg() { printf '%s\n' "$out" | grep -Eq -- "$1" || fails="$fails|$2"; }
  printf '%s\n' "$out" | grep -c -- '^localforward ' | grep -qx 1 || fails="$fails|not one localforward"
  local sock dest
  sock=${forward%%:*}; dest=${forward#*:}
  printf '%s\n' "$out" | grep -qFx -- "localforward $sock [${dest%%:*}]:${dest##*:}" || fails="$fails|localforward is not the printed forward"
  printf '%s\n' "$out" | grep -qE -- '^(remoteforward|dynamicforward) ' && fails="$fails|extra forward"
  for pair in "streamlocalbindmask 0111" "streamlocalbindunlink no" "controlmaster false" \
              "forwardagent no" "identitiesonly yes" "stricthostkeychecking true" \
              "userknownhostsfile $SPEC/known_hosts" "identityfile ~/.ssh/id_ed25519" \
              "exitonforwardfailure yes" "batchmode yes" "globalknownhostsfile /dev/null" \
              "user dan" "hostname 192.168.0.16" "clearallforwardings no" "permitlocalcommand no" \
              "connecttimeout 15" "serveraliveinterval 30" "serveralivecountmax 3" "identityagent none"; do
    printf '%s\n' "$out" | grep -qx -- "$pair" || fails="$fails|$pair"
  done
  for key in controlpath proxycommand proxyjump; do
    line=$(printf '%s\n' "$out" | grep -E -- "^$key " || true)
    [ -z "$line" ] || [ "$line" = "$key none" ] || fails="$fails|$key $line"
  done
  if [ -z "$fails" ]; then ok "sshg: ssh -G equals Appendix A (forward $forward)"; else bad "sshg:$fails"; fi
}
chk_m2() {
  rm -rf "$TOP/out-m2"
  run_launcher m2 "$TOP/out-m2" sleep ACCB_DEADLINE=10
  local status=$?
  expect_status 91 "$status" "m2: the launcher deadline exits 91"
  [ -z "$($DK ps -aq --filter name=accb-)" ] && ok "m2: the container is gone" || bad "m2: a container is left: $($DK ps -aq --filter name=accb-)"
  grep -q "deadline_hit 1" "$TOP/out-m2/pi/docker-exit.txt" 2>/dev/null && ok "m2: docker-exit.txt records the deadline" || bad "m2: no deadline record"
}
chk_d1() {
  p8_guard d1 || return 0
  rm -rf "$TOP/out-d1"
  bg_run d1 flood
  local n=0 rss kid_rss max_rss=0
  while kill -0 "$RUN_PID" 2>/dev/null && [ "$n" -lt 120 ]; do
    rss=$(ps -o rss= -p "$RUN_PID" 2>/dev/null | tr -d ' '); rss=${rss:-0}
    for kid in $(pgrep -P "$RUN_PID" 2>/dev/null); do
      kid_rss=$(ps -o rss= -p "$kid" 2>/dev/null | tr -d ' '); rss=$((rss + ${kid_rss:-0}))
    done
    [ "$rss" -gt "$max_rss" ] && max_rss=$rss
    sleep 1; n=$((n + 1))
  done
  wait "$RUN_PID" 2>/dev/null; local status=$?; RUN_PID=""
  if [ "$max_rss" -lt 204800 ]; then ok "d1: the launcher stays under 200 MiB RSS (peak ${max_rss} KiB)"; else bad "d1: launcher RSS peaked at ${max_rss} KiB"; fi
  local size
  size=$(stat -c %s "$TOP/out-d1/pi/pi.stamped.tsv" 2>/dev/null || echo missing)
  if [ "$size" != missing ] && [ "$size" -le 51000000 ]; then ok "d1: the stamped file stops at the 48 MiB cap ($size bytes)"; else bad "d1: stamped file is $size bytes"; fi
  grep -q "cap_hit yes" "$TOP/out-d1/pi/docker-exit.txt" 2>/dev/null && ok "d1: the cap hit is recorded" || bad "d1: no cap record"
  [ -z "$($DK ps -aq --filter name=accb-)" ] && ok "d1: the container is gone" || bad "d1: a container is left"
}
chk_d1b() {
  # Round 5, blocking item 1: the same flood with NO newline at all. The reader must stop
  # storing at the cap (bounded memory and file) and the run must end with the cap-exit status.
  p8_guard d1b || return 0
  rm -rf "$TOP/out-d1b"
  bg_run d1b floodnn
  local n=0 rss kid_rss max_rss=0
  while kill -0 "$RUN_PID" 2>/dev/null && [ "$n" -lt 240 ]; do
    rss=$(ps -o rss= -p "$RUN_PID" 2>/dev/null | tr -d ' '); rss=${rss:-0}
    for kid in $(pgrep -P "$RUN_PID" 2>/dev/null); do
      kid_rss=$(ps -o rss= -p "$kid" 2>/dev/null | tr -d ' '); rss=$((rss + ${kid_rss:-0}))
    done
    [ "$rss" -gt "$max_rss" ] && max_rss=$rss
    sleep 1; n=$((n + 1))
  done
  wait "$RUN_PID" 2>/dev/null; RUN_PID=""
  if [ "$max_rss" -lt 204800 ]; then ok "d1b: bounded memory with no newline at all (peak ${max_rss} KiB)"; else bad "d1b: launcher RSS peaked at ${max_rss} KiB"; fi
  local size
  size=$(stat -c %s "$TOP/out-d1b/pi/pi.stamped.tsv" 2>/dev/null || echo missing)
  if [ "$size" != missing ] && [ "$size" -le 51000000 ]; then ok "d1b: the stamped file stays bounded ($size bytes)"; else bad "d1b: stamped file is $size bytes"; fi
  grep -q "cap_hit yes" "$TOP/out-d1b/pi/docker-exit.txt" 2>/dev/null && ok "d1b: the cap hit is recorded" || bad "d1b: no cap record"
  local client
  client=$(sed -n 's/^docker_client //p' "$TOP/out-d1b/pi/docker-exit.txt" 2>/dev/null)
  [ "$client" = 137 ] && ok "d1b: the cap-exit status is 137 (docker kill)" || bad "d1b: docker_client is $client"
  [ -z "$($DK ps -aq --filter name=accb-)" ] && ok "d1b: the container is gone" || bad "d1b: a container is left"
}
chk_d4() {
  # Round 5, blocking item 2: the docker client stderr (pi/accb-peak.txt) is capped at 16 MiB
  # while the stdout cap is not involved and the run completes.
  p8_guard d4 || return 0
  rm -rf "$TOP/out-d4"
  bg_run d4 errflood
  local n=0 rss kid_rss max_rss=0
  while kill -0 "$RUN_PID" 2>/dev/null && [ "$n" -lt 240 ]; do
    rss=$(ps -o rss= -p "$RUN_PID" 2>/dev/null | tr -d ' '); rss=${rss:-0}
    for kid in $(pgrep -P "$RUN_PID" 2>/dev/null); do
      kid_rss=$(ps -o rss= -p "$kid" 2>/dev/null | tr -d ' '); rss=$((rss + ${kid_rss:-0}))
    done
    [ "$rss" -gt "$max_rss" ] && max_rss=$rss
    sleep 1; n=$((n + 1))
  done
  wait "$RUN_PID" 2>/dev/null; local status=$?; RUN_PID=""
  if [ "$max_rss" -lt 204800 ]; then ok "d4: bounded memory under a stderr flood (peak ${max_rss} KiB)"; else bad "d4: launcher RSS peaked at ${max_rss} KiB"; fi
  local size
  size=$(stat -c %s "$TOP/out-d4/pi/accb-peak.txt" 2>/dev/null || echo missing)
  if [ "$size" != missing ] && [ "$size" -gt 0 ] && [ "$size" -le 16777300 ]; then ok "d4: accb-peak.txt is capped at 16 MiB ($size bytes)"; else bad "d4: accb-peak.txt is $size bytes"; fi
  grep -q "accb-peak stderr cut at" "$TOP/out-d4/pi/accb-peak.txt" 2>/dev/null && ok "d4: the truncation is recorded" || bad "d4: no truncation record"
  grep -q "cap_hit no" "$TOP/out-d4/pi/docker-exit.txt" 2>/dev/null && ok "d4: the stdout cap is not involved" || bad "d4: the stdout cap fired"
  expect_status 3 "$status" "d4: the run completes (a quality miss: the flooded hello.py reruns to other output)"
  [ -z "$($DK ps -aq --filter name=accb-)" ] && ok "d4: the container is gone" || bad "d4: a container is left"
}
chk_d3() {
  ( cd "$AHD" && python3 -I - "$TOP" <<'D3PY'
import os, pathlib, sys, time
sys.path.insert(0, ".")
import sandbox
assert sandbox.OUTPUT_CAP == 1048576, sandbox.OUTPUT_CAP
box = sandbox.Sandbox(os.environ["SANDBOX_IMAGE"], scratch=pathlib.Path(sys.argv[1]) / "d3-stage")
box.preflight()
cmd = box.command("/x", "n")
assert cmd[cmd.index("--log-driver") + 1] == "none"
assert cmd[cmd.index("--shm-size") + 1] == "16m"
script = pathlib.Path(sys.argv[1]) / "flood.py"
script.write_text("import sys\nfor _ in range(200000):\n    sys.stdout.write('x' * 1024)\n    sys.stdout.flush()\n")
started = time.monotonic()
code, text = box.run_python(script)
assert code is None, code
assert "more than 1048576 bytes" in text, text[-200:]
assert len(text) < 1048576 + 200, len(text)
print("d3 ok: cut at 1 MiB after %.1fs" % (time.monotonic() - started))
D3PY
) && ok "d3: sandbox.py caps the output at 1 MiB with --log-driver none and --shm-size 16m" || bad "d3: see above"
}
chk_e1() {
  local out
  out=$(sbx accb-t-e1 /bin/bash -c '/opt/node/bin/node --version')
  [ "$out" = "v22.22.3" ] && ok "e1: node --version in the container is v22.22.3" || bad "e1: node printed '$out'"
  $DK rm -f accb-t-e1 >/dev/null 2>&1; CLEANUP_CONT=""
}
chk_e2() {
  local out version display writable fds
  out=$(sbx accb-t-e2 /bin/bash -c '/opt/node/bin/node /opt/pi/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js --version &
    pid=$!
    sleep 0.4
    echo "fds $(ls /proc/$pid/fd 2>/dev/null | wc -l)"
    wait $pid
    cat > /tmp/e2.py <<PYEOF
import os
writable = []
for line in open("/proc/self/mountinfo"):
    fields = line.split()
    if "rw" in fields[5].split(","):
        writable.append(fields[4])
exempt = {"/proc", "/dev", "/dev/pts", "/dev/mqueue", "/dev/shm", "/tmp", "/work", "/home/acc", "/sys/fs/cgroup"}
def ok_mount(dest):
    return dest in exempt or dest.startswith(("/proc/", "/sys/", "/dev/"))
print("display", os.environ.get("DISPLAY", "unset"))
print("extra_writable", [w for w in writable if not ok_mount(w)])
print("memory_peak", open("/sys/fs/cgroup/memory.peak").read().strip() if os.path.exists("/sys/fs/cgroup/memory.peak") else "none")
print("pids_peak", open("/sys/fs/cgroup/pids.peak").read().strip() if os.path.exists("/sys/fs/cgroup/pids.peak") else "none")
PYEOF
    /usr/local/bin/python -I /tmp/e2.py')
  version=$(printf '%s\n' "$out" | sed -n '1p')
  display=$(printf '%s\n' "$out" | sed -n 's/^display //p')
  writable=$(printf '%s\n' "$out" | sed -n 's/^extra_writable //p')
  fds=$(printf '%s\n' "$out" | sed -n 's/^fds //p')
  [ "$version" = "1.0.0" ] && ok "e2: pi --version in the container is 1.0.0" || bad "e2: pi printed '$version'"
  [ "$display" = "unset" ] && ok "e2: DISPLAY is unset" || bad "e2: DISPLAY is $display"
  [ "$writable" = "[]" ] && ok "e2: the writable mounts are only /tmp, /work, /home/acc, /dev/shm, and /dev" \
    || bad "e2: unexpected writable mounts: $writable"
  echo "recorded for U5: open-file count ${fds:-none}, $(printf '%s\n' "$out" | sed -n 's/^memory_peak //p') memory.peak, $(printf '%s\n' "$out" | sed -n 's/^pids_peak //p') pids.peak"
  $DK rm -f accb-t-e2 >/dev/null 2>&1; CLEANUP_CONT=""
}
chk_e3() {
  local out
  out=$(sbx accb-t-e3 /bin/bash -c '
    export PI_CODING_AGENT_DIR=/home/acc/agent ENGINESRV_API_KEY=list-models-probe
    cp -a /in/agent "$HOME/agent"
    /opt/node/bin/node /opt/pi/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js --offline -nc -ne -ns -np --list-models 2>&1 | head -5')
  echo "$out" | grep -q fifi && ok "e3: after the entry copy, --list-models lists provider fifi" || bad "e3: $(printf '%s\n' "$out" | head -3)"
  $DK rm -f accb-t-e3 >/dev/null 2>&1; CLEANUP_CONT=""
}
chk_e4() {
  # (a) a full rehearsal passes and records socket_mode 666 (C3): without the chmod the bridge
  #     cannot connect, so the run itself fails.
  rm -rf "$TOP/out-e4"
  run_launcher e4 "$TOP/out-e4" plain
  expect_status 0 $? "e4: the rehearsal run passes end to end"
  grep -q "socket_mode 666" "$TOP/out-e4/sandbox/ssh.txt" 2>/dev/null && ok "e4: the socket mode 666 is recorded" || bad "e4: no socket_mode 666 record"
  # (b) without the relay socket the entry exits 65.
  rm -f "$SPEC/ipc/fifi.sock"
  local args=(); mapfile -t args < <(entry_args accb-t-e4b)
  printf '%s\n' "$TEST_KEY" | timeout -s KILL 40 "${args[@]}" > "$TOP/e4b.out" 2>&1
  expect_status 65 $? "e4: the entry exits 65 without the relay socket"
  $DK rm -f accb-t-e4b >/dev/null 2>&1; CLEANUP_CONT=""
  # (c) through the bridge: /ready is 200, a wrong key gets 401.
  ENGINESRV_API_KEY=$TEST_KEY python3 -I "$AHD/fake_server.py" --port "$ACCB_TEST_PORT" --mode pi-tools --model-id flash > "$TOP/e4-server.log" 2>&1 &
  local server=$!
  relay_up "$SPEC/ipc"
  local out
  out=$(sbx accb-t-e4c /bin/bash -c '/opt/node/bin/node /opt/accb/bridge.js & sleep 1; cat > /tmp/q.py <<QEOF
import urllib.request, urllib.error
def code(path, key):
    req = urllib.request.Request("http://127.0.0.1:18791" + path)
    if key:
        req.add_header("Authorization", "Bearer " + key)
    try:
        return urllib.request.urlopen(req, timeout=5).status
    except urllib.error.HTTPError as error:
        return error.code
print("ready", code("/ready", None))
print("wrong", code("/v1/models", "wrong-key"))
QEOF
/usr/local/bin/python -I /tmp/q.py')
  local ready wrong
  ready=$(printf '%s\n' "$out" | sed -n 's/^ready //p'); wrong=$(printf '%s\n' "$out" | sed -n 's/^wrong //p')
  [ "$ready" = 200 ] && ok "e4: GET /ready through the bridge is 200" || bad "e4: /ready gave $ready ($out)"
  [ "$wrong" = 401 ] && ok "e4: a wrong key gets 401" || bad "e4: a wrong key gave $wrong"
  $DK rm -f accb-t-e4c >/dev/null 2>&1; CLEANUP_CONT=""
  relay_down; kill "$server" 2>/dev/null; wait "$server" 2>/dev/null
}
chk_b1() {
  ENGINESRV_API_KEY=$TEST_KEY python3 -I "$AHD/fake_server.py" --port "$ACCB_TEST_PORT" --mode pi-tools --model-id flash > "$TOP/b1-server.log" 2>&1 &
  local server=$!
  relay_up "$SPEC/ipc"
  local out
  out=$(sbx accb-t-b1 /bin/bash -c '/opt/node/bin/node /opt/accb/bridge.js & sleep 1; cat > /tmp/b1.py <<QEOF
import socket, time
socks = []
for _ in range(20):
    try:
        socks.append(socket.create_connection(("127.0.0.1", 18791), timeout=5))
    except OSError:
        pass
time.sleep(1.0)
open_now = 0
for sock in socks:
    sock.setblocking(False)
    try:
        data = sock.recv(1)
        if data != b"":
            open_now += 1
    except BlockingIOError:
        open_now += 1
    except OSError:
        pass
print("open", open_now, "connected", len(socks))
try:
    probe = socket.create_connection(("127.0.0.1", 18791), timeout=5)
    probe.sendall(b"GET /ready HTTP/1.0\r\n\r\n")
    status = probe.recv(64).split()[1].decode()
except OSError:
    status = "error"
print("ready", status)
QEOF
/usr/local/bin/python -I /tmp/b1.py')
  local open_count connected ready
  open_count=$(printf '%s\n' "$out" | sed -n 's/^open //;s/ connected.*//p'); connected=$(printf '%s\n' "$out" | sed -n 's/^open [0-9]* connected //p')
  ready=$(printf '%s\n' "$out" | sed -n 's/^ready //p')
  if [ "${connected:-0}" = 20 ] && [ "${open_count:-99}" -le 8 ]; then
    ok "b1: of 20 connections at most 8 stay open ($open_count) and the server stays ready ($ready)"
  else
    bad "b1: connected ${connected:-none}, open ${open_count:-none}, ready $ready"
  fi
  $DK rm -f accb-t-b1 >/dev/null 2>&1; CLEANUP_CONT=""
  relay_down; kill "$server" 2>/dev/null; wait "$server" 2>/dev/null
}
chk_k0() {
  env ACC_TEST_KEY= bash "$AHD/run-acc-b.sh" "$TOP/out-k0" --rehearse > "$TOP/k0.log" 2>&1
  expect_status 64 $? "k0: an empty key is refused with 64"
  grep -q "the bearer key is empty" "$TOP/k0.log" && ok "k0: the K0 message names the empty key" || bad "k0: no K0 message: $(head -2 "$TOP/k0.log")"
  [ -e "$TOP/out-k0" ] && bad "k0: an evidence directory was made" || ok "k0: nothing was staged"
}
chk_k1() {
  rm -rf "$TOP/out-k1"
  run_launcher k1 "$TOP/out-k1" plain
  expect_status 0 $? "k1: the rehearsal passes"
  if grep -rqF -f <(printf '%s\n' "$TEST_KEY") "$TOP/out-k1" 2>/dev/null; then
    bad "k1: the key appears in the evidence"
  else
    ok "k1: the key appears nowhere in the evidence"
  fi
}
chk_k2() {
  bg_run k2 sleep
  if ! wait_container; then bad "k2: the run never started a container"; end_run; return; fi
  $DK inspect "$CONT" > "$TOP/k2-inspect.txt" 2>/dev/null
  if grep -qF -f <(printf '%s\n' "$TEST_KEY") "$TOP/k2-inspect.txt"; then bad "k2: the key is in docker inspect (C6)"; else ok "k2: docker inspect holds no key"; fi
  end_run
  expect_status 143 "$RUN_STATUS" "k2: the background run ends with the signal status"
  others_unchanged k2
}
chk_k3() {
  bg_run k3 sleep
  if ! wait_container; then bad "k3: the run never started a container"; end_run; return; fi
  ps -eo args > "$TOP/k3-ps.txt" 2>/dev/null
  if grep -qF -f <(printf '%s\n' "$TEST_KEY") "$TOP/k3-ps.txt"; then bad "k3: the key is in the argv of a process"; else ok "k3: ps -eo args holds no key"; fi
  local relay_pid docker_pid cmdline clean=yes
  relay_pid=$(pgrep -f "accb-relay.py --socket /run/user/1000/acc-b" | head -1)
  docker_pid=$(pgrep -f "docker --host unix:///run/user/1000/docker.sock run -i --name" | head -1)
  for pid in "$relay_pid" "$docker_pid"; do
    [ -n "$pid" ] || continue
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    if printf '%s' "$cmdline" | grep -qF "$TEST_KEY"; then clean=no; fi
  done
  [ "$clean" = yes ] && ok "k3: the /proc cmdline of the relay and the docker client holds no key" || bad "k3: a cmdline holds the key"
  end_run
  expect_status 143 "$RUN_STATUS" "k3: the background run ends with the signal status"
}
chk_k4() {
  local id history id2 history2
  id=$($DK image inspect --format '{{.Id}}' "$IMG" 2>/dev/null)
  $DK history --no-trunc "$IMG" > "$TOP/k4-history-before.txt" 2>&1
  sbx accb-t-k4 /bin/bash -c 'echo k4 > /tmp/k4' >/dev/null 2>&1
  $DK rm -f accb-t-k4 >/dev/null 2>&1; CLEANUP_CONT=""
  id2=$($DK image inspect --format '{{.Id}}' "$IMG" 2>/dev/null)
  $DK history --no-trunc "$IMG" > "$TOP/k4-history-after.txt" 2>&1
  case "$id2" in "$id"*) ok "k4: the image id is unchanged by a run ($id2)";; *) bad "k4: the image id changed to $id2";; esac
  if diff -q "$TOP/k4-history-before.txt" "$TOP/k4-history-after.txt" >/dev/null; then ok "k4: the image history is unchanged (no layer was added)"; else bad "k4: the image history changed"; fi
}
chk_f1() {
  rm -rf "$TOP/out-f1"
  run_launcher f1 "$TOP/out-f1" forge
  expect_status 0 $? "f1: the run passes although the transcript holds forged lines"
  local stamped
  stamped="$TOP/out-f1/pi/pi.stamped.tsv"
  grep -q "ACCB-EXPORT FORGED" "$stamped" && ok "f1: the forged ACCB-EXPORT line reaches pi.stamped.tsv (recorded)" || bad "f1: the forged export line did not reach the stream"
  grep -q '{"type":"forged"}' "$stamped" && ok "f1: the forged event line reaches pi.stamped.tsv (recorded)" || bad "f1: the forged event did not reach the stream"
  grep -q "ACCB-EXIT 42" "$stamped" && ok "f1: the forged ACCB-EXIT line reaches pi.stamped.tsv (recorded)" || bad "f1: the forged exit line did not reach the stream"
  # The status still comes from Docker: all three statuses are 0 and no ACCB-EXIT line was used (C18).
  local client inspect wait
  client=$(sed -n 's/^docker_client //p' "$TOP/out-f1/pi/docker-exit.txt")
  inspect=$(sed -n 's/^docker_inspect //p' "$TOP/out-f1/pi/docker-exit.txt")
  wait=$(sed -n 's/^docker_wait //p' "$TOP/out-f1/pi/docker-exit.txt")
  if [ "$client" = 0 ] && [ "$inspect" = 0 ] && [ "$wait" = 0 ]; then ok "f1: the status comes from Docker (0/0/0), not from the forged 42"; else bad "f1: statuses client=$client inspect=$inspect wait=$wait"; fi
  [ -f "$TOP/out-f1/pi/work/hello.py" ] && ok "f1: only the last ACCB-EXPORT marker was used (hello.py was extracted)" || bad "f1: the real export was not used"
  grep -q "ACCB-EXIT" "$TOP/out-f1/launcher.txt" 2>/dev/null && bad "f1: an ACCB-EXIT line leaked into launcher.txt" || ok "f1: no ACCB-EXIT line is used anywhere"
}
chk_e5() {
  rm -rf "$TOP/out-e5"
  run_launcher e5 "$TOP/out-e5" keyleak
  expect_status 90 $? "e5: a leaked key removes the evidence and exits 90"
  [ -e "$TOP/out-e5" ] && bad "e5: the evidence still exists" || ok "e5: the evidence was removed"
  grep -q "the key appears in the evidence" "$TOP/e5.log" && ok "e5: the removal is reported" || bad "e5: no removal report: $(tail -2 "$TOP/e5.log")"
}
chk_e6() {
  ( cd "$AHD" && python3 -I -m unittest discover -s . -p 'test_accb_export.py' > "$TOP/e6-unittest.log" 2>&1 ) \
    && ok "e6: the extractor unittest passes (the E6 cases)" || bad "e6: the extractor unittest failed ($(tail -2 "$TOP/e6-unittest.log"))"
  # The CLI path the launcher uses, with a real tar and a forged earlier marker.
  local src=$TOP/e6src out=$TOP/e6out
  rm -rf "$src" "$out"; mkdir -p "$src/d" "$out"
  printf 'print(2 + 3)\n' > "$src/hello.py"
  printf 'x' > "$src/d/inner"
  weird=$'bad\nname.txt'
  : > "$src/$weird"                 # a control character in the name: skipped and repr-escaped
  local data forged
  data=$(tar -C "$src" -cf - . | base64 -w0)
  forged=$(printf 'ACCB-EXPORT FORGED\nACCB-EXPORT %s\n' "$data")
  python3 -I "$AHD/accb_export.py" --stream <(printf '%s\n' "$forged") --dest "$out/work" --manifest "$out/manifest.txt" \
    && ok "e6: the extractor CLI runs on a real tar with a forged earlier marker" || bad "e6: the CLI failed"
  [ "$(cat "$out/work/hello.py" 2>/dev/null)" = "print(2 + 3)" ] && ok "e6: the real archive was extracted" || bad "e6: hello.py was not extracted"
  grep -q "kept" "$out/manifest.txt" && ok "e6: the manifest records kept entries" || bad "e6: no kept line"
  if python3 -I - "$out/manifest.txt" <<'MEOF'
import sys
lines = open(sys.argv[1]).read().splitlines()
assert any("bad\\nname" in line for line in lines), "no escaped form"
assert not [line for line in lines if any(ord(c) < 0x20 and c != "\t" for c in line)], "a raw control character"
MEOF
  then
    ok "e6: the manifest escapes names (repr, no raw newline)"
  else
    bad "e6: the manifest does not escape: $(grep -c . "$out/manifest.txt") lines"
  fi
  awk 'length($0) > 260 { bad_line = 1 } END { exit bad_line ? 1 : 0 }' "$out/manifest.txt" \
    && ok "e6: manifest lines are bounded (the 200-character cut)" || bad "e6: a manifest line is over the cut"
}
chk_e7() {
  local SH_DIR=$TOP/shims ARGV_LOG=$TOP/e7-argv.log
  mkdir -p "$SH_DIR"; : > "$ARGV_LOG"
  local tool real
  for tool in python3 docker; do
    real=$(command -v "$tool")
    printf '#!/bin/sh\nprintf "%%s\\n" "%s $*" >> "%s"\nexec "%s" "$@"\n' "$tool" "$ARGV_LOG" "$real" > "$SH_DIR/$tool"
    chmod +x "$SH_DIR/$tool"
  done
  rm -rf "$TOP/out-e7"
  ( export PATH="$SH_DIR:$PATH"
    run_launcher e7 "$TOP/out-e7" plain )
  local status=$?
  expect_status 0 "$status" "e7: the rehearsal passes with logged argv"
  local checks_tsv=$TOP/out-e7/check/acc-b-checks.tsv
  if [ -f "$checks_tsv" ] && ! grep -q "False" "$checks_tsv"; then ok "e7: every checker row is ok"; else bad "e7: checker rows: $(cat "$checks_tsv" 2>/dev/null | grep -c True)/$(grep -c . "$checks_tsv" 2>/dev/null)"; fi
  # Transcript counts equal the server log.
  local result
  result=$(python3 -I - "$TOP/out-e7" <<'E7PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
assistants = 0
for line in (out / "pi" / "pi.stamped.tsv").read_text().splitlines():
    payload = line.split("\t", 1)[1]
    if payload.startswith("ACCB-"):   # entry-protocol line: extractor input or data, not an event
        continue
    event = json.loads(payload)
    if event.get("type") == "message_end" and (event.get("message") or {}).get("role") == "assistant":
        assistants += 1
entries = [json.loads(l) for l in (out / "evidence" / "server.log").read_text().splitlines() if l.strip()]
entries = [e for e in entries if isinstance(e, dict) and "prompt_tokens" in e]
print("ok" if assistants == len(entries) else "assistants %d, server entries %d" % (assistants, len(entries)))
E7PY
)
  [ "$result" = ok ] && ok "e7: the assistant turns equal the server-log requests" || bad "e7: $result"
  # hello.py was rerun in a second sandbox container with --network none and --log-driver none,
  # and the checker log names it.
  local sandbox_line
  sandbox_line=$(grep -m1 "docker.*acc-sandbox-" "$ARGV_LOG" | tr ' ' '\n')
  if printf '%s\n' "$sandbox_line" | grep -q -- "--network" && printf '%s\n' "$sandbox_line" | grep -qx none \
     && printf '%s\n' "$sandbox_line" | grep -q -- "--log-driver" && printf '%s\n' "$sandbox_line" | grep -qx none; then
    ok "e7: hello.py was rerun in a sandbox container with --network none and --log-driver none"
  else
    bad "e7: no sandbox rerun found in the argv log"
  fi
  grep -q "container acc-sandbox-" "$TOP/e7.log" && ok "e7: the checker log names the sandbox container" || bad "e7: the checker log names no container"
  # No model-written file ran on the host.
  if grep -qE "python3 [^ ]*$TOP/out-e7/pi/work/" "$ARGV_LOG"; then
    bad "e7: a model-written file ran on the host: $(grep -E "python3 [^ ]*$TOP/out-e7/pi/work/" "$ARGV_LOG" | head -1)"
  else
    ok "e7: no model-written file ran on the host ($(wc -l < "$ARGV_LOG") logged calls)"
  fi
  if grep -rqF -f <(printf '%s\n' "$TEST_KEY") "$TOP/out-e7" 2>/dev/null; then bad "e7: the key is in the evidence"; else ok "e7: the key is nowhere in the evidence"; fi
}
chk_e9() {
  # The mount path guard: a colon, a comma, and a newline are refused with 64 before docker run.
  local bad1=$TOP/"fix:ture" bad2=$TOP/"fix,ture" bad3=$TOP/fix$'\n'ture
  mkdir -p "$bad1" "$bad2" "$bad3"
  ACC_TEST_KEY=$TEST_KEY bash "$AHD/run-acc-b.sh" "$TOP/out-e9a" --rehearse --fixture "$bad1" > "$TOP/e9a.log" 2>&1
  expect_status 64 $? "e9: a fixture path with a colon is refused with 64"
  ACC_TEST_KEY=$TEST_KEY bash "$AHD/run-acc-b.sh" "$TOP/out-e9b" --rehearse --fixture "$bad2" > "$TOP/e9b.log" 2>&1
  expect_status 64 $? "e9: a fixture path with a comma is refused with 64"
  ACC_TEST_KEY=$TEST_KEY bash "$AHD/run-acc-b.sh" "$TOP/out-e9c" --rehearse --fixture "$bad3" > "$TOP/e9c.log" 2>&1
  expect_status 64 $? "e9: a fixture path with a newline is refused with 64"
  ACCB_SCRATCH=$TOP/scratch$'\n'dir ACC_TEST_KEY=$TEST_KEY bash "$AHD/run-acc-b.sh" "$TOP/out-e9d" --rehearse \
    > "$TOP/e9d.log" 2>&1
  expect_status 64 $? "e9: an ACCB_SCRATCH with a newline is refused with 64"
  if [ -n "$($DK ps -aq --filter name=accb-)" ]; then bad "e9: a container was started anyway"; else ok "e9: no container was started for a bad path"; fi
  if [ -e "$TOP/out-e9a" ] || [ -e "$TOP/out-e9b" ] || [ -e "$TOP/out-e9c" ] || [ -e "$TOP/out-e9d" ]; then bad "e9: an evidence directory was made"; else ok "e9: no evidence directory was made"; fi
  # The lock descriptors are closed in the children: the relay, the docker client, and the
  # serve session child (fix5 item 7) hold neither the acc-b lock (fd 8) nor the Mac lock (fd 9).
  bg_run e9 sleep
  if ! wait_container; then bad "e9: the run never started a container"; end_run; return; fi
  local relay_pid docker_pid serve_pid result
  relay_pid=$(pgrep -f "accb-relay.py --socket /run/user/1000/acc-b" | head -1)
  docker_pid=$(pgrep -f "docker --host unix:///run/user/1000/docker.sock run -i --name" | head -1)
  serve_pid=$(pgrep -f "mac-acc-a.sh serve" | head -1)
  result=yes
  for pid in "$relay_pid" "$docker_pid" "$serve_pid"; do
    [ -n "$pid" ] || { result="no pid"; continue; }
    if ls -l "/proc/$pid/fd" 2>/dev/null | grep -qE "acc-b/lock|mac\.lock"; then result="lock fd held by pid $pid"; fi
  done
  [ "$result" = yes ] && ok "e9: neither the relay, the docker client, nor the serve session holds a lock fd" || bad "e9: $result"
  end_run
  expect_status 143 "$RUN_STATUS" "e9: the background run ends with the signal status"
}
chk_st4() {
  # Round 5, blocking item 4: the docker status must be an integer before it reaches the
  # checker, and a status defect is a pipeline defect that is never discarded.
  # The checker is invoked exactly as the launcher invokes it (no -I: it imports its siblings).
  python3 "$AHD/check_pi_run.py" --transcript /dev/null --exit-code none --workdir "$TOP" \
    --server-log /dev/null --out "$TOP/st4-out" > "$TOP/st4.log" 2>&1
  expect_status 1 $? "st4: the checker refuses a non-integer --exit-code with status 1"
  grep -q "usage error" "$TOP/st4.log" && ok "st4: the refusal is a usage error" || bad "st4: no usage error: $(head -2 "$TOP/st4.log")"
  grep -q "Traceback" "$TOP/st4.log" && bad "st4: the refusal printed a traceback" || ok "st4: no traceback"
  rm -rf "$TOP/out-st4"
  run_launcher st4 "$TOP/out-st4" plain
  expect_status 0 $? "st4: a clean run passes the status checks"
  local client inspect wait check_status
  client=$(sed -n 's/^docker_client //p' "$TOP/out-st4/pi/docker-exit.txt")
  inspect=$(sed -n 's/^docker_inspect //p' "$TOP/out-st4/pi/docker-exit.txt")
  wait=$(sed -n 's/^docker_wait //p' "$TOP/out-st4/pi/docker-exit.txt")
  if [ "$client" = 0 ] && [ "$inspect" = 0 ] && [ "$wait" = 0 ]; then ok "st4: the three docker statuses agree (0/0/0)"; else bad "st4: client=$client inspect=$inspect wait=$wait"; fi
  grep -q "^check_status 0$" "$TOP/out-st4/launcher.txt" && ok "st4: launcher.txt records check_status 0" || bad "st4: no check_status record"
}
chk_pin() {
  # Round 5, item 8: the launcher pins the rerun image itself, so a caller that sets no
  # SANDBOX_IMAGE still gets the pinned one for the hello.py rerun.
  mk_fixture "$TOP/fix-plain" plain
  rm -rf "$TOP/out-pin"
  env -u SANDBOX_IMAGE ACC_TEST_KEY="$TEST_KEY" bash "$AHD/run-acc-b.sh" "$TOP/out-pin" --rehearse \
    --fixture "$TOP/fix-plain" > "$TOP/pin.log" 2>&1
  expect_status 0 $? "pin: the rehearsal passes with SANDBOX_IMAGE unset"
  grep -q "hello.py rerun in the sandbox container" "$TOP/pin.log" \
    && ok "pin: the checker reran hello.py in the pinned sandbox" || bad "pin: no sandbox rerun: $(tail -2 "$TOP/pin.log")"
}
chk_reap() {
  # Round 5, item 12: at the start the launcher reaps stale containers of its own earlier runs
  # (the run-name pattern and its S12 probe pattern) and nothing else.
  $DK rm -f accb-20200101T000000Z-dead accb-t-s12-dead accb-t-decoy >/dev/null 2>&1
  $DK run -d --name accb-20200101T000000Z-dead --pull never --network none "$IMG" sleep 300 >/dev/null 2>&1
  $DK run -d --name accb-t-s12-dead --pull never --network none "$IMG" sleep 300 >/dev/null 2>&1
  $DK run -d --name accb-t-decoy --pull never --network none "$IMG" sleep 300 >/dev/null 2>&1
  CLEANUP_CONT=accb-t-decoy
  rm -rf "$TOP/out-reap"
  run_launcher reap "$TOP/out-reap" plain
  expect_status 0 $? "reap: the rehearsal passes with stale containers present"
  [ -z "$($DK ps -aq --filter name=accb-20200101T000000Z-dead)" ] && ok "reap: the stale run container is gone" || bad "reap: the stale run container is left"
  [ -z "$($DK ps -aq --filter name=accb-t-s12-dead)" ] && ok "reap: the stale S12 probe container is gone" || bad "reap: the stale probe container is left"
  [ -n "$($DK ps -aq --filter name=accb-t-decoy)" ] && ok "reap: a container that is not this launcher's is untouched" || bad "reap: the decoy was removed"
  $DK rm -f accb-t-decoy >/dev/null 2>&1; CLEANUP_CONT=""
  others_unchanged reap
}
chk_e10() {
  # Design 4.4 step 5 (the PID sweep over /proc): the leftover watcher of the lateforge fixture must be
  # dead before the entry prints the real export, so the LAST ACCB-EXPORT marker is the real
  # one and hello.py is extracted; no forged line reaches the stream at all.
  rm -rf "$TOP/out-e10"
  run_launcher e10 "$TOP/out-e10" lateforge
  expect_status 0 $? "e10: the run passes with a leftover forger in the fixture"
  [ -f "$TOP/out-e10/pi/work/hello.py" ] && ok "e10: the real export was the last marker (hello.py extracted)" || bad "e10: the real export was not used"
  if grep -q "LATE-FORGE" "$TOP/out-e10/pi/pi.stamped.tsv" 2>/dev/null; then
    bad "e10: a forged marker line reached the stream although the watcher should be dead"
  else
    ok "e10: no forged marker line reached the stream (the watcher died with pi)"
  fi
  [ -z "$($DK ps -aq --filter name=accb-)" ] && ok "e10: the container is gone" || bad "e10: a container is left"
}
chk_k5() {
  # Round 5, blocking item 3: the key scan runs on the deadline path too. The key is in the
  # transcript, the exit stays 91 (the stronger status), and the leak is recorded.
  rm -rf "$TOP/out-k5"
  bg_run k5 keysleep ACCB_DEADLINE=90
  local out=$TOP/out-k5 n=0 keyed=no
  while kill -0 "$RUN_PID" 2>/dev/null && [ "$n" -lt 150 ]; do
    if [ -f "$out/pi/pi.stamped.tsv" ] && grep -qF -f <(printf '%s\n' "$TEST_KEY") "$out/pi/pi.stamped.tsv" 2>/dev/null; then keyed=yes; break; fi
    sleep 1; n=$((n + 1))
  done
  [ "$keyed" = yes ] && ok "k5: the leaked key reaches the transcript before the deadline" || bad "k5: the key never reached the transcript"
  wait "$RUN_PID" 2>/dev/null; local status=$?; RUN_PID=""
  expect_status 91 "$status" "k5: the deadline still exits 91 with a leaked key"
  [ -f "$out/sandbox/teardown.txt" ] && grep -q "^key_leak yes$" "$out/sandbox/teardown.txt" \
    && ok "k5: the leak is recorded in the evidence (sandbox/teardown.txt)" || bad "k5: no key_leak record"
  [ -d "$out" ] && ok "k5: the evidence stays for the stronger status" || bad "k5: the evidence was removed"
  [ -z "$($DK ps -aq --filter name=accb-)" ] && ok "k5: the container is gone" || bad "k5: a container is left"
}
chk_k6() {
  # Round 5, blocking item 3: the key scan runs on the SIGTERM path too, and the seal covers
  # the final stamped file (the stamping reader is waited for before sealing).
  rm -rf "$TOP/out-k6"
  bg_run k6 keysleep
  local out=$TOP/out-k6 n=0 keyed=no
  while [ "$n" -lt 150 ]; do
    kill -0 "$RUN_PID" 2>/dev/null || break
    if [ -f "$out/pi/pi.stamped.tsv" ] && grep -qF -f <(printf '%s\n' "$TEST_KEY") "$out/pi/pi.stamped.tsv" 2>/dev/null; then keyed=yes; break; fi
    sleep 1; n=$((n + 1))
  done
  [ "$keyed" = yes ] && ok "k6: the leaked key reaches the transcript" || bad "k6: the key never reached the transcript"
  end_run
  expect_status 143 "$RUN_STATUS" "k6: SIGTERM keeps status 143 with a leaked key"
  [ -f "$out/sandbox/teardown.txt" ] && grep -q "^key_leak yes$" "$out/sandbox/teardown.txt" \
    && ok "k6: the leak is recorded in the evidence" || bad "k6: no key_leak record"
  ( cd "$out" && sha256sum -c SHA256SUMS --quiet ) 2>/dev/null \
    && ok "k6: the sealed hashes match the final files" || bad "k6: SHA256SUMS does not match the final files"
  [ -z "$($DK ps -aq --filter name=accb-)" ] && ok "k6: the container is gone" || bad "k6: a container is left"
}
chk_x1() {
  bg_run x1 sleep
  if ! wait_container; then bad "x1: the run never started a container"; end_run; return; fi
  kill -TERM "$RUN_PID"
  local n=0
  while kill -0 "$RUN_PID" 2>/dev/null && [ "$n" -lt 120 ]; do sleep 1; n=$((n + 1)); done
  wait "$RUN_PID" 2>/dev/null; local status=$?; RUN_PID=""
  expect_status 143 "$status" "x1: SIGTERM ends the launcher with 143"
  sleep 2
  [ -z "$($DK ps -aq --filter name=accb-)" ] && ok "x1: the container is gone" || bad "x1: the container is left (C16)"
  pgrep -f "accb-relay.py --socket /run/user/1000/acc-b" >/dev/null && bad "x1: the relay is left" || ok "x1: the relay is gone"
  [ "$(ls -A /run/user/1000/acc-b 2>/dev/null | grep -vx lock || true)" = "" ] && ok "x1: the socket directory is removed" || bad "x1: /run/user/1000/acc-b holds $(ls -A /run/user/1000/acc-b | tr '\n' ' ')"
  others_unchanged x1
}
chk_r2modes() {
  bg_run r2 sleep
  if ! wait_container; then bad "r2modes: the run never started a container"; end_run; return; fi
  local sock dir parent
  sock=$(find /run/user/1000/acc-b -name fifi.sock 2>/dev/null | head -1)
  if [ -n "$sock" ]; then
    dir=$(dirname "$sock"); parent=$(dirname "$dir")
    [ "$(stat -c %a "$sock")" = 666 ] && ok "r2modes: the socket is 666 during the run" || bad "r2modes: the socket is $(stat -c %a "$sock")"
    [ "$(stat -c %a "$dir")" = 755 ] && ok "r2modes: the IPC directory is 755" || bad "r2modes: the IPC directory is $(stat -c %a "$dir")"
    [ "$(stat -c %a "$parent")" = 700 ] && ok "r2modes: /run/user/1000/acc-b is 700" || bad "r2modes: the parent is $(stat -c %a "$parent")"
  else
    bad "r2modes: no socket found during the run"
  fi
  end_run
  expect_status 143 "$RUN_STATUS" "r2modes: the run ends with the signal status"
}

# --- run the checks ----------------------------------------------------------------------------------
ALL_CHECKS=(p1 p3 p6 p8 p10 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s12 h1 h2 h3 n1 n2 n3 n4 n5 n6 n7 n8 w1 w3 w4 w5 l1 l2 l4 m1 d2 flags sshg m2 d1 d1b d4 d3 e1 e2 e3 e4 b1 k0 k1 k2 k3 k4 f1 e5 e6 e7 e9 st4 pin reap e10 k5 k6 x1 r2modes)
if [ "${#CHECKS[@]}" = 0 ]; then CHECKS=("${ALL_CHECKS[@]}"); fi
for check in "${CHECKS[@]}"; do
  if declare -f "chk_$check" >/dev/null; then
    echo "--- $check"
    "chk_$check"
  else
    bad "unknown check $check"
  fi
done
echo
echo "selftest scratch: $TOP"
if [ "$FAILS" = 0 ]; then echo "accb selftest: all checks passed"; exit 0; fi
echo "accb selftest: $FAILS checks failed"; exit 1
