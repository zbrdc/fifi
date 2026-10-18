#!/usr/bin/env bash
# Mutation check of selftest-accb.sh: one mutation per control of design 6, plus the five
# mutations the design names for its own self-tests (the C14 and C15 rows, the empty-key guard,
# the C23 row, and the manifest escaping). Each mutation copies agent-acceptance to a scratch
# directory, patches the named file with the named change, verifies that the patch changed the
# file, runs the named selftest-accb.sh check against the copy, and requires a failure. A
# surviving mutation is a gap in the tests. Never commit a mutation.
# Usage: mutate-accb.sh [scratch dir]    Exit status: 0 when every mutation is killed.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOP=${1:-$(mktemp -d "${TMPDIR:-/tmp}/accb-mutate.XXXXXX")}
mkdir -p "$TOP"
# fix1: a free local port for every rehearsal listener of the self-tests run below, chosen at
# start (probe 18900-18990 first — outside the ephemeral range — bind to port 0 only when the
# range is busy; never 18791, never 8788). selftest-accb.sh honours the inherited
# ACCB_TEST_PORT, so every mutation run uses this port.
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
[ "$ACCB_TEST_PORT" != 0 ] || { echo "ERROR: no free local port in 18900-18990" >&2; exit 2; }
export ACCB_TEST_PORT
SURVIVED=0
# mutate <name> <file> <sed program> <killer check>: copy, patch, check, require a failure.
mutate() {
  local name=$1 file=$2 prog=$3 check=$4
  local dir=$TOP/$name/agent-acceptance
  rm -rf "$TOP/$name"; mkdir -p "$TOP/$name"
  cp -r "$HERE"/. "$dir"/ || { echo "ERROR $name: cannot copy"; SURVIVED=$((SURVIVED + 1)); return; }
  if [ "$dir/run-acc-b.sh" -ef "$HERE/run-acc-b.sh" ]; then echo "ERROR $name: the copy aliases the original"; exit 2; fi
  sed -e "$prog" "$HERE/$file" > "$dir/$file" || { echo "ERROR $name: sed failed"; SURVIVED=$((SURVIVED + 1)); return; }
  if cmp -s "$HERE/$file" "$dir/$file"; then echo "ERROR $name: the patch changed nothing"; SURVIVED=$((SURVIVED + 1)); return; fi
  bash "$HERE/selftest-accb.sh" "$dir" "$TOP/$name/run" "$check" > "$TOP/$name/out.log" 2>&1
  local status=$?
  if [ "$status" = 0 ]; then
    echo "SURVIVED $name ($check)"
    SURVIVED=$((SURVIVED + 1))
  elif ! grep -q '^accb selftest:' "$TOP/$name/out.log"; then
    echo "ERROR $name: the self-test did not run to its summary (status $status)"
    SURVIVED=$((SURVIVED + 1))
  else
    echo "killed   $name ($(grep -c '^FAIL' "$TOP/$name/out.log") failed checks, killer $check)"
  fi
}

# --- the controls of design 6, one mutation each ------------------------------------------------------
mutate c1-network-none run-acc-b.sh '/^    --network none \\/d' n1
mutate c2-forward-destination run-acc-b.sh 's|^FWD_DEST=127.0.0.1:18791$|FWD_DEST=127.0.0.1:18792|' sshg
mutate c3-socket-mode run-acc-b.sh '/^chmod 666 "\$IPC\/fifi.sock"/d; /^\[ "\$(stat -c %a "\$IPC\/fifi.sock")" = 666 \]/d' e4
mutate c4-read-only run-acc-b.sh '/^    --read-only \\/d' w1
mutate c5-extra-mount run-acc-b.sh 's|^    --user 65534:65534 \\|    --user 65534:65534 \\\n    -v /home/dan:/home/dan \\|' h1
mutate c6-key-in-env run-acc-b.sh 's|^mapfile -t DOCKER_ARGS < <(docker_run_args "\$STAGED_FIX" "\$STAGE" "\$IPC")|&\nDOCKER_ARGS=("\${DOCKER_ARGS[@]:0:5}" -e "ENGINESRV_API_KEY=\$KEY" "\${DOCKER_ARGS[@]:5}")|' k2
mutate c7-pids-limit run-acc-b.sh 's|--pids-limit 128|--pids-limit 64|' l1
mutate c8-memory run-acc-b.sh 's|--memory 1024m|--memory 256m|' l2
mutate c9-no-timeout accb-entry.sh 's|timeout -s KILL "\$PI_TIMEOUT" /opt/node/bin/node|/opt/node/bin/node|' m1
mutate c10-log-driver run-acc-b.sh 's|--log-driver none|--log-driver json-file|' d2
mutate c11-no-nnp run-acc-b.sh 's|--cap-drop ALL --security-opt no-new-privileges|--cap-drop ALL|' s2
mutate c12-image-pin run-acc-b.sh 's|a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1|b6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1|g' flags
mutate c13-bridge-cap accb-bridge.js 's|MAX_CONNECTIONS = 8|MAX_CONNECTIONS = 100|' b1
mutate c14-accept-links accb_export.py 's|if not (member.isreg() or member.isdir()):|if False:|' e6
mutate c15-no-key-scan run-acc-b.sh 's|if key_in_dir "\$EV" .*; then|if false; then|' e5
mutate c16-no-trap-rm run-acc-b.sh '/^    \$DK rm -f "\$RUN" 8>&- 9>&- 2>\/dev\/null$/d' x1
mutate c17-host-python check_pi_run.py 's|code, text = sandbox.run_python(target)|cp = __import__("subprocess").run(["python3", str(target)], capture_output=True, text=True)\n        code, text = (cp.returncode, cp.stdout)|' e7
mutate c18-exit-line run-acc-b.sh 's|DOCKER_STATUS=\$(cat "\$CLIENT_STATUS".*|DOCKER_STATUS=$(grep "^ACCB-EXIT " "$EV/pi/pi.stamped.tsv" 2>/dev/null \| tail -1 \| cut -d" " -f2); DOCKER_STATUS=${DOCKER_STATUS:-42}|' f1
mutate c19-no-host run-acc-b.sh 's|--host "\$DKHOST" ||; s|^DK="docker --host \$DKHOST"|DK="docker"|' flags
mutate c20-no-controlmaster run-acc-b.sh '/^  ControlMaster no$/d' sshg
mutate c21-no-flock run-acc-b.sh '/^flock 8$/d' p6
mutate c22-live-pi-mount run-acc-b.sh 's|--mount type=bind,src="\$2/opt/pi",dst=/opt/pi,ro|--mount type=bind,src=/home/dan/.pi/agent/install/releases/1.0.0,dst=/opt/pi,ro|' flags
mutate c23-no-path-guard run-acc-b.sh '/^  case "\$FIXTURE" in /d; s|^    case "\$src" in |    : # refusal removed by the mutation: |' e9
# --- the five self-test mutations the design names beyond the table -----------------------------------
mutate k0-no-empty-key-guard run-acc-b.sh '/^\[ -n "\$KEY" \] || die "the bearer key is empty/d' k0
mutate escaping-no-repr accb_export.py 's|repr_cut(member.name)|member.name|g' e6

# --- the round-5 review fixes (fix5 items 1 to 3, 5, 7 to 13) -----------------------------------------
mutate stamper-no-cap run-acc-b.sh 's|cap = int(sys.argv\[1\])|cap = 10 ** 12|' d1
mutate stderr-no-cap run-acc-b.sh 's|scap = int(sys.argv\[2\])|scap = 10 ** 12|' d4
mutate trap-no-keyscan run-acc-b.sh 's|if { key_in_dir "\$EV"|if { false|' k5
mutate extract-dot-part accb_export.py 's|if part == ".":|if part == "." and False:|' e6
mutate extract-entry-limit accb_export.py 's|if entries > MAX_ENTRIES:|if entries > MAX_ENTRIES + 1000000:|' e6
mutate extract-nfc-dup accb_export.py 's|if key in seen:|if False:|' e6
mutate no-image-pin run-acc-b.sh '/^export SANDBOX_IMAGE=/d' pin
mutate no-reap run-acc-b.sh '/^  for c in \$reaped; do \$DK rm -f "\$c" 8>&- 9>&- 2>\/dev\/null; done$/d' reap
mutate entry-no-kill1 accb-entry.sh '/^if \[ "\$\$" = 1 \]; then$/,/^fi$/d' e10
mutate entry-export-blocking accb-entry.sh '/^printf .ACCB-EXPORT /s| > /proc/self/fd/1$||' e10
mutate c4b-tmpfs-noexec run-acc-b.sh 's|/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777|/tmp:rw,nosuid,nodev,size=16m,mode=1777|' s12
mutate c4c-status-none run-acc-b.sh 's|status = proc.wait()|status = "none"|' st4
mutate c8b-memory-swap run-acc-b.sh 's|--memory-swap 1024m|--memory-swap 2048m|' s12
mutate c8c-shm-size run-acc-b.sh 's|--shm-size 16m|--shm-size 32m|' s12
mutate c8d-no-ulimit run-acc-b.sh 's|--ulimit nofile=512:512 --ulimit core=0|--ulimit core=0|' s12
mutate c11b-seccomp run-acc-b.sh 's|--cap-drop ALL --security-opt no-new-privileges|--cap-drop ALL --security-opt no-new-privileges --security-opt seccomp=unconfined|' s12
mutate s12-pids-wrong run-acc-b.sh 's|need(hc.get("PidsLimit") == 128|need(hc.get("PidsLimit") == 64|' k1
mutate path-newline run-acc-b.sh "/case \"\\\$FIXTURE\" in /s|'\\\\n'|'\\\\t'|; /^case \"\\\$ACCB_SCRATCH\" in /s|'\\\\n'|'\\\\t'|" e9

echo
if [ "$SURVIVED" = 0 ]; then echo "every mutation was killed"; exit 0; fi
echo "$SURVIVED mutation(s) survived"; exit 1
