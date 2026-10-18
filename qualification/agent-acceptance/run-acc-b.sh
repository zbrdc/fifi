#!/usr/bin/env bash
# ACC-B launcher (Linux host): the pi coding agent against enginesrv on the Mac, with pi itself
# inside a sandbox container (design 4.2, 4.3, 4.4; option (a), user-approved 2026-10-10).
# pi has the write and bash tools, so a model controls a shell. That shell runs only in the
# container of design 4.3: --network none, a read-only root, tmpfs /tmp /work /home/acc,
# uid 65534, memory, cpu, and pid limits, --log-driver none, no pull. The only route out is the
# unix socket of an ssh -L forward that ends at the Mac loopback port 18791; the container reaches
# it through accb-bridge.js. The launcher opens no TCP port on this host. The status comes from
# Docker only (design 4.5): the transcript is data, and a pass needs the server-log cross-check of
# check_pi_run.py. --rehearse replaces the Mac with fake_server.py and the ssh forward with
# accb-relay.py (design 9). --print-spec <dir> stages the run (placeholders plus the real reflink
# copies), prints the exact 4.3 flag list, the ssh_config, and the ssh command line, and exits 0:
# it touches no daemon and takes no lock (the self-tests build their SBX commands from it).
# Usage: run-acc-b.sh <new evidence dir> [--branch main] [--rehearse] [--fixture <dir>]
#   or: run-acc-b.sh --print-spec <dir>
# Exit status: 0 pass, 2 pipeline defect, 3 quality miss (that of check_pi_run.py), or the serve
# session's own status when the checker passed (86 the server ended early); 64 usage,
# 66 the GPU baton is missing (an ACC-A CUDA-engine refusal of driver-lib.sh; run-acc-b never
# exits it, but selftest.sh compares this list with every literal die/exit of run-acc-b.sh and
# driver-lib.sh),
# 67 sandbox image missing, 69 daemon not rootless (both from sandbox.py), 70 local port busy
# (rehearsal only), 81/82 prepare/tunnel, 83 not ready, 84 forward not ready,
# 89 pin/hash/flag mismatch, 90 key found in the evidence, 91 launcher deadline,
# 92 ssh staging failure, 93 pre-start resource guard, 143 a SIGTERM/SIGINT/SIGHUP end. The
# README table of run-acc-b statuses lists the same numbers, and selftest.sh compares the two.
set -u
# shellcheck source=driver-lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/driver-lib.sh"

# --- constants (design 4.2, 4.3, 10) -----------------------------------------------------------------
ENGINE=flash                  # ACC-B serves Flash-Next only
DKHOST=unix:///run/user/1000/docker.sock
DK="docker --host $DKHOST"    # the pinned rootless daemon (C19); never /var/run/docker.sock
PINNED_IMAGE=python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
PINNED_ID_PREFIX=sha256:a6e34c598f24
NODE_SHA256=e6ec2c188d83d813f81f2de8aea084d74dce603ac1abedd0a30ad941b10087b2
LOCKFILE_SHA256=6cd08f50db03de4551ad6b3902be9d69e61ae0960242104881d7c7ef576f169c
METADATA_SHA256=af16649e4e9988b79ea478d6cb8def4682ce6b36ac3e4e8b812ea7bdd32514ea
PI_RELEASE=~/.pi/agent/install/releases/1.0.0
NODE_BIN=~/.local/share/pi-node/node-v22.22.3-linux-x64/bin/node
ACC_B_DIR=/run/user/1000/acc-b          # 0700; $IPC below is 0755, the socket 0666 (4.2)
SSH_HOSTNAME=192.168.0.16
# The forward destination of design 4.2 (the Mac loopback port) as ONE source of truth: the
# --print-spec command line, the ssh -G record, the assert_ssh_g expectation, and the forward
# itself all read these three lines, so they cannot drift apart; selftest-accb.sh checks the
# printed forward against its own literal destination instead.
FWD_DEST=127.0.0.1:18791
FWD_DEST_HOST=${FWD_DEST%%:*}
FWD_DEST_PORT=${FWD_DEST##*:}
SSH_DEADLINE=6600
STDOUT_CAP=50331648                   # 48 MiB on the container stdout (design 5 F4, C10, D1)
STDERR_CAP=16777216                   # 16 MiB on the docker client stderr (C10, T10, T22). The
                                     # design states no number for stderr; 16 MiB is the fix5
                                     # packet's value, equal to EXPORT_MAX.
ACCB_DEADLINE=${ACCB_DEADLINE:-6600}   # the launcher deadline; overridable so M2 can set 10 s
ACCB_SCRATCH=${ACCB_SCRATCH:-${TMPDIR:-/tmp}}   # holds acc-b/<run>/stage, outside $EV (4.4)

# --- arguments (step 1) ----------------------------------------------------------------------------
USAGE='usage: run-acc-b.sh <new evidence dir> [--branch main] [--rehearse] [--fixture <dir>] [--print-spec <dir>]'
OUT=""; BRANCH=main; REHEARSE=0; FIXTURE=""; SPEC=""
while [ $# -gt 0 ]; do
  case "$1" in
    --branch)   [ $# -ge 2 ] || die "--branch needs a value" 64; BRANCH=$2; shift 2;;
    --rehearse) REHEARSE=1; shift;;
    --fixture)  [ $# -ge 2 ] || die "--fixture needs a value" 64; FIXTURE=$2; shift 2;;
    --print-spec) [ $# -ge 2 ] || die "--print-spec needs a value" 64; SPEC=$2; shift 2;;
    -*) die "unknown option $1" 64;;
    *) [ -z "$OUT" ] || die "two evidence directories were given" 64; OUT=$1; shift;;
  esac
done
check_port
# The rehearsal's pinu-local listeners (fake_server.py and accb-relay.py) take their TCP port
# from ACCB_TEST_PORT (default 18791, the design value); a real run opens no TCP port on pinu at
# all (4.2), and its forward destination stays the design's 18791 on the Mac side.
if [ "$REHEARSE" = 1 ]; then PORT=${ACCB_TEST_PORT:-18791}; check_port; fi
[ ! -e "$OUT" ] || [ -z "$OUT" ] || die "$OUT exists; give a new directory" 64
# The path guard (T33, C23, fix5 item 10): a colon or a comma would break a docker mount
# option, and a newline would split one argument into two in the mapfile of docker_run_args.
# ACCB_SCRATCH feeds the same argument list through $STAGE.
if [ -n "$FIXTURE" ]; then
  [ -d "$FIXTURE" ] || die "the fixture directory $FIXTURE does not exist" 64
  case "$FIXTURE" in *:*|*,*|*$'\n'*) die "the fixture path $FIXTURE holds a colon, a comma, or a newline; a docker mount option or the argument list would break (T33)" 64;; esac
fi
case "$ACCB_SCRATCH" in *:*|*,*|*$'\n'*) die "ACCB_SCRATCH ($ACCB_SCRATCH) holds a colon, a comma, or a newline" 64;; esac

# --- helpers ---------------------------------------------------------------------------------------
hash_note() { printf '%s\n' "$*" >> "$HASHES"; }   # $HASHES is set before staging
reflink_copy() { # <src> <dst>: design 4.3 note / 10; records whether the copy is a reflink
  local src=$1 dst=$2
  mkdir -p "$(dirname "$dst")"
  if cp -a --reflink=always "$src" "$dst" 2>/dev/null; then
    REFLINK_NOTE="reflink"
  else
    cp -a "$src" "$dst" || die "cannot copy $src into the run directory" 89
    REFLINK_NOTE="full copy"
  fi
}
expect_sha() { # <file> <want sha256> <label>
  local got
  got=$(sha256sum "$1" 2>/dev/null | cut -d' ' -f1)
  [ "$got" = "$2" ] || die "the $3 hash of $1 is ${got:-missing}, want $2" 89
  hash_note "$3 $(basename "$1") $got"
}
pi_tree_hash() { # <dir>: the tree hash of design 10 — for each sorted file or symlink, one line of
  # path, mode, and sha256 (files) or readlink text (links); the hash of that list. One python
  # process instead of two forks per file: the release holds about 15,000 files.
  python3 -I - "$1" <<'TREEPY'
import hashlib, os, subprocess, sys
root = sys.argv[1]
names = subprocess.run(["find", ".", "(", "-type", "f", "-o", "-type", "l", ")", "-print0"],
                      cwd=root, capture_output=True, check=True).stdout[:-1].split(b"\0")
total = hashlib.sha256()
for raw in sorted(names):                      # sorted bytes = LC_ALL=C sort -z
    path = os.fsdecode(raw)
    full = os.path.join(root, path)
    st = os.lstat(full)
    if os.path.islink(full):
        value = os.readlink(full)
    else:
        digest = hashlib.sha256()
        with open(full, "rb") as handle:
            for block in iter(lambda: handle.read(65536), b""):
                digest.update(block)
        value = digest.hexdigest()
    total.update(f"{path} {st.st_mode & 0o7777:o} {value}\n".encode())
print(total.hexdigest())
TREEPY
}
# The bounded run reader (fix5 blocking item 1 and 2, design C10/T10/T22): one python process
# feeds the key to the docker client on its stdin (the key is never in argv or a file), stamps
# the container stdout line by line into pi.stamped.tsv, caps the docker client stderr into
# pi/accb-peak.txt, and records the client exit status. It reads fixed-size blocks, never a
# whole line: `read -r line` would buffer an unbounded newline-free stream in launcher memory.
# The 48 MiB cap counts the stored payload bytes (${#line} + 1 semantics); a line that cannot
# fit (including a partial line with no newline at all, and the ~22 MiB real export line, which
# fits) triggers the cap: docker kill is spawned, never waited for, and the reader keeps
# draining so the client never wedges on a full pipe (T22). Stderr is cut at STDERR_CAP with a
# truncation note and also keeps draining.
ACCB_STAMPER='import subprocess, sys, threading, time

cap = int(sys.argv[1])
scap = int(sys.argv[2])
dkhost = sys.argv[3]
run = sys.argv[4]
cap_note = sys.argv[5]
status_file = sys.argv[6]
stamped_file = sys.argv[7]
peak_file = sys.argv[8]
docker_cmd = sys.argv[sys.argv.index("--") + 1:]
key = sys.stdin.buffer.readline()          # the bearer key, only ever on a pipe
proc = subprocess.Popen(docker_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
try:
    proc.stdin.write(key)
    proc.stdin.close()
except BrokenPipeError:
    pass
state = {"total": 0, "capped": False}

def cap_hit():
    # T22: the kill is spawned, never waited for; the reader keeps draining the stream, or the
    # docker client blocks on a full pipe and the kill never lands.
    try:
        with open(cap_note, "w") as handle:
            handle.write("cap_hit yes\n")
    except OSError:
        pass
    subprocess.Popen(["docker", "--host", dkhost, "kill", run],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("the container wrote more than %d bytes; docker kill %s" % (cap, run), file=sys.stderr)
    return True

def pump_stdout():
    buf = bytearray()
    with open(stamped_file, "wb", buffering=0) as out:   # unbuffered: a line is on disk at once
        while True:
            data = proc.stdout.read1(65536)   # read1: return what is there, never wait for 64 KiB
            if not data:
                break
            if state["capped"]:
                continue                   # keep draining, store nothing
            buf += data
            while True:
                cut = buf.find(b"\n")
                if cut < 0:
                    break
                line = bytes(buf[:cut])
                del buf[:cut + 1]
                if state["total"] + len(line) + 1 <= cap:
                    out.write(("%.9f\t" % time.time()).encode() + line + b"\n")
                    state["total"] += len(line) + 1
                else:
                    state["capped"] = cap_hit()
                    buf.clear()
                    break
            if not state["capped"] and state["total"] + len(buf) + 1 > cap:
                # a partial line with no newline at all can never fit: cap hit, drop it, drain on
                state["capped"] = cap_hit()
                buf.clear()

def pump_stderr():
    written = 0
    with open(peak_file, "wb", buffering=0) as out:
        while True:
            data = proc.stderr.read1(65536)
            if not data:
                break
            room = scap - written
            if room < len(data):
                out.write(data[:room])
                out.write(b"\n[accb-peak stderr cut at %d bytes]\n" % scap)
                while proc.stderr.read1(65536):
                    pass                   # keep draining so the client never blocks
                return
            out.write(data)
            written += len(data)

t_out = threading.Thread(target=pump_stdout)
t_err = threading.Thread(target=pump_stderr)
t_out.start()
t_err.start()
status = proc.wait()
t_out.join()
t_err.join()
with open(status_file, "w") as handle:
    handle.write("%s\n" % status)
'
docker_run_args() { # <fixture> <stage> <ipc>: the design 4.3 argument list, one per line
  printf '%s\n' docker --host "$DKHOST" run -i --name "$RUN_ARG" --pull never \
    --network none \
    --read-only \
    --user 65534:65534 \
    --cap-drop ALL --security-opt no-new-privileges \
    --memory 1024m --memory-swap 1024m \
    --cpus 1 --pids-limit 128 \
    --shm-size 16m \
    --ulimit nofile=512:512 --ulimit core=0 \
    --log-driver none \
    --tmpfs /tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777 \
    --tmpfs /work:rw,noexec,nosuid,nodev,size=16m,uid=65534,gid=65534,mode=0755 \
    --tmpfs /home/acc:rw,noexec,nosuid,nodev,size=32m,uid=65534,gid=65534,mode=0755 \
    --mount type=bind,src="$1",dst=/fixture,ro \
    --mount type=bind,src="$2/opt/pi",dst=/opt/pi,ro \
    --mount type=bind,src="$2/opt/node",dst=/opt/node/bin/node,ro \
    --mount type=bind,src="$2/agent",dst=/in/agent,ro \
    --mount type=bind,src="$2/entry.sh",dst=/opt/accb/entry.sh,ro \
    --mount type=bind,src="$2/bridge.js",dst=/opt/accb/bridge.js,ro \
    --mount type=bind,src="$3",dst=/ipc,ro \
    -e HOME=/home/acc -e PI_CODING_AGENT_DIR=/home/acc/agent \
    -e PI_OFFLINE=1 -e PI_SKIP_VERSION_CHECK=1 -e PI_TELEMETRY=0 \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PI_TIMEOUT=5400 -e EXPORT_MAX=16777216 \
    --entrypoint /bin/bash \
    "$PINNED_IMAGE" \
    /opt/accb/entry.sh
}
write_ssh_config() { # <stage>: the launcher-owned ssh_config of design 4.2 (C20)
  cat > "$1/ssh_config" <<SSHCFG
Host mac
  HostName $SSH_HOSTNAME
  User dan
  IdentitiesOnly yes
  IdentityFile ~/.ssh/id_ed25519
  IdentityAgent none
  StrictHostKeyChecking yes
  UserKnownHostsFile $1/known_hosts
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
  ServerAliveCountMax 3
SSHCFG
}
assert_ssh_g() { # <config> <socket-path> <known_hosts>: stops with 92 unless ssh -G shows 4.2 (P9)
  local config=$1 sock=$2 known=$3 out
  out=$(ssh -G -F "$config" -a -x -N -L "$sock:$FWD_DEST" mac 2>&1) || { echo "$out" >&2; die "ssh -G failed on $config" 92; }
  local fails=""
  sg() { printf '%s\n' "$out" | grep -E -q -- "$1" || fails="$fails
$2"; }
  sgeq() { local line; line=$(printf '%s\n' "$out" | grep -E -- "^$1\$" || true); [ "$line" = "$2" ] || fails="$fails
$1 is '$line', want '$2'"; }
  [ "$(printf '%s\n' "$out" | grep -c -- '^localforward ')" = 1 ] || fails="$fails
not exactly one localforward line"
  sgeq "localforward $sock [$FWD_DEST_HOST]:$FWD_DEST_PORT"
  [ -z "$(printf '%s\n' "$out" | grep -E -- '^(remoteforward|dynamicforward) ')" ] || fails="$fails
an extra forward exists"
  sgeq "streamlocalbindmask 0111"
  sgeq "streamlocalbindunlink no"
  sgeq "controlmaster false"
  sgeq "forwardagent no"
  sgeq "identitiesonly yes"
  sgeq "stricthostkeychecking true"
  sgeq "userknownhostsfile $known"
  line=$(printf '%s\n' "$out" | grep -E -- '^globalknown(hosts|s)file ' || true)
  [ "$line" = "globalknownhostsfile /dev/null" ] || [ "$line" = "globalknownsfile /dev/null" ] \
    || fails="$fails\nglobal known hosts is '$line', want /dev/null"
  sgeq "identityfile ~/.ssh/id_ed25519"
  sgeq "exitonforwardfailure yes"
  sgeq "batchmode yes"
  for key in controlpath proxycommand proxyjump; do
    line=$(printf '%s\n' "$out" | grep -E -- "^$key " || true)
    [ -z "$line" ] || [ "$line" = "$key none" ] || fails="$fails
$key is '$line', want absent or none"
  done
  if [ -n "$fails" ]; then printf 'ssh -G mismatch:%s\n' "$fails" >&2; die "the ssh -G output differs from design 4.2 (P9)" 92; fi
}
stage_build() { # <stage> <ipc> <spec:0|1> — builds the run directory and verifies the pins (P4/P5)
  local stage=$1 ipc=$2 spec=$3 fix
  mkdir -p "$stage/agent" "$stage/fixture" "$ipc"
  if [ "$spec" = 1 ]; then
    : > "$stage/entry.sh"; : > "$stage/bridge.js"     # placeholders: the mount list is what matters
  else
    cp "$AH_DIR/accb-entry.sh" "$stage/entry.sh" || die "cannot stage accb-entry.sh" 89
    cp "$AH_DIR/accb-bridge.js" "$stage/bridge.js" || die "cannot stage accb-bridge.js" 89
    # The scratch agent directory. PI_CODING_AGENT_DIR replaces ~/.pi/agent for this run
    # (pi docs: environment-variables.md "PI_CODING_AGENT_DIR"). contextWindow is 6400 on purpose:
    # pi sends max_tokens = min(maxTokens, contextWindow - estimated prompt - 4096) with a minimum
    # of 1; a window of 2051 would make pi send max_tokens 1.
    cat > "$stage/agent/models.json" <<JSON
{
  "providers": {
    "fifi": {
      "baseUrl": "http://127.0.0.1:18791/v1",
      "api": "openai-completions",
      "apiKey": "\$ENGINESRV_API_KEY",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens",
        "sendSessionAffinityHeaders": false
      },
      "models": [
        {
          "id": "flash",
          "name": "Fifi Flash-Next (local, AGENT-1a)",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 6400,
          "maxTokens": 256,
          "samplingParams": {"temperature": 0.2},
          "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
        }
      ]
    }
  }
}
JSON
    cat > "$stage/agent/settings.json" <<JSON
{
  "defaultProvider": "fifi",
  "defaultModel": "flash",
  "defaultThinkingLevel": "off",
  "httpIdleTimeoutMs": 0,
  "retry": {"enabled": false}
}
JSON
  fi
  # The reflink copies of the pi release and the node binary (U2, C22). The live files in ~/.pi
  # and ~/.local/share are never mounted (T30).
  reflink_copy "$PI_RELEASE" "$stage/opt/pi"
  local pi_copy_note="$REFLINK_NOTE"
  reflink_copy "$NODE_BIN" "$stage/opt/node"
  hash_note "pi_copy $pi_copy_note"
  hash_note "node_copy $REFLINK_NOTE"
  # P5: uid 65534 in the container must be able to read and traverse the copies.
  [ "$(find "$stage/opt/pi" ! -perm -o+r | wc -l)" = 0 ] || die "a file under $stage/opt/pi lacks other-read" 89
  [ "$(find "$stage/opt/pi" -type d ! -perm -o+x | wc -l)" = 0 ] || die "a directory under $stage/opt/pi lacks other-execute" 89
  [ "$(find "$stage/opt/node" ! -perm -o+r | wc -l)" = 0 ] || die "$stage/opt/node lacks other-read" 89
  # P4a/P4b: the pins of the pi release and the node binary, and the tree hash of the copy (10).
  expect_sha "$stage/opt/node" "$NODE_SHA256" "node_bin"
  expect_sha "$stage/opt/pi/package-lock.json" "$LOCKFILE_SHA256" "package_lock"
  expect_sha "$stage/opt/pi/metadata.json" "$METADATA_SHA256" "metadata"
  TREE_HASH=$(pi_tree_hash "$stage/opt/pi")
  hash_note "pi_tree_hash $TREE_HASH"
  [ -n "$FIXTURE" ] && fix="$FIXTURE" || fix="$stage/fixture"
  for src in "$fix" "$stage/opt/pi" "$stage/opt/node" "$stage/agent" "$stage/entry.sh" "$stage/bridge.js" "$ipc"; do
    case "$src" in *:*|*,*) die "the mount source $src holds a colon or a comma (T33)" 64;; esac
  done
  STAGED_FIX="$fix"
}
s12_compare() { # <inspect file> <fixture> <stage> <ipc>: the field-by-field compare of S12; dies with 89
  python3 -I - "$1" "$2" "$3" "$4" <<'S12PY'
import json, sys
data = json.load(open(sys.argv[1]))
run = data[0] if isinstance(data, list) else data
hc, cfg = run["HostConfig"], run["Config"]
fixture, stage, ipc = sys.argv[2], sys.argv[3], sys.argv[4]
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
need(cfg.get("User") == "65534:65534", "User 65534:65534")
need((hc.get("LogConfig") or {}).get("Type") == "none", "LogConfig.Type none")
need(not (hc.get("PortBindings") or {}), "PortBindings empty")
need(not (hc.get("Devices") or []), "Devices empty")
env = cfg.get("Env") or []
need(not [e for e in env if e.startswith("ENGINESRV_API_KEY")], "Config.Env holds no key")
need(hc.get("Tmpfs") ==
     {"/home/acc": "rw,noexec,nosuid,nodev,size=32m,uid=65534,gid=65534,mode=0755",
      "/tmp": "rw,noexec,nosuid,nodev,size=16m,mode=1777",
      "/work": "rw,noexec,nosuid,nodev,size=16m,uid=65534,gid=65534,mode=0755"},
     "tmpfs option strings of design 4.3, got %r" % (hc.get("Tmpfs"),))
mounts = run.get("Mounts") or []
need(len(mounts) == 7, "exactly seven bind mounts, got %d" % len(mounts))
want_sources = {"/fixture": fixture, "/opt/pi": stage + "/opt/pi",
                "/opt/node/bin/node": stage + "/opt/node", "/in/agent": stage + "/agent",
                "/opt/accb/entry.sh": stage + "/entry.sh",
                "/opt/accb/bridge.js": stage + "/bridge.js", "/ipc": ipc}
for m in mounts:
    need(m.get("RW") is False, "mount %s is not read-only" % m.get("Destination"))
    need(m.get("Source") == want_sources.get(m.get("Destination")),
         "mount source of %s is %r, want %r" % (m.get("Destination"), m.get("Source"),
                                                want_sources.get(m.get("Destination"))))
    need("docker.sock" not in (m.get("Source") or ""), "no docker.sock mount")
    need(not (m.get("Source") or "").startswith(("/home/dan/.pi", "/home/dan/.local")),
         "no live pi or node path: %s" % m.get("Source"))
need({m.get("Destination") for m in mounts} == set(want_sources),
     "the seven destinations of design 4.3")
need(set((hc.get("Tmpfs") or {})) == {"/tmp", "/work", "/home/acc"},
     "tmpfs exactly /tmp /work /home/acc")
if problems:
    print("S12 MISMATCH: " + "; ".join(problems))
    sys.exit(1)
print("s12 fields ok: privileged no, network none, read-only, caps dropped, nnp only, no pid/ipc/"
      "userns mode, limits and ulimits, 7 ro mounts with the staged sources, tmpfs strings of "
      "4.3, no key, no docker.sock")
S12PY
}

# --- --print-spec: no daemon, no lock (step 18) ------------------------------------------------------
if [ -n "$SPEC" ]; then
  [ -z "$OUT" ] && [ -z "$FIXTURE" ] || die "--print-spec takes no evidence dir and no fixture" 64
  [ ! -e "$SPEC" ] || die "$SPEC exists; give a new directory" 64
  mkdir -p "$SPEC"
  HASHES="$SPEC/hashes.txt"; : > "$HASHES"
  RUN_ARG=accb-t-spec
  stage_build "$SPEC" "$SPEC/ipc" 1
  docker_run_args "$SPEC/fixture" "$SPEC" "$SPEC/ipc" | sed 's/^/ARG /'
  write_ssh_config "$SPEC"
  echo SSH_CONFIG_BEGIN
  cat "$SPEC/ssh_config"
  echo SSH_CONFIG_END
  echo "SSH_CMD timeout -s TERM -k 10 $SSH_DEADLINE ssh -n -F $SPEC/ssh_config -a -x -N -L $SPEC/ipc/fifi.sock:$FWD_DEST mac"
  echo "SSH_READY curl -s -o /dev/null -w '%{http_code}' -m 10 --unix-socket $SPEC/ipc/fifi.sock http://127.0.0.1/ready"
  exit 0
fi

# --- the run (steps 2 to 17) ------------------------------------------------------------------------
[ -n "$OUT" ] || die "$USAGE" 64
# The sandbox gate validates the caller's image when one is set (the self-test fake), and the
# pinned image otherwise. The rerun of the checker then always uses the pinned image (fix5
# item 8): sandbox.py and check_pi_run.py see it even when the caller set none.
export SANDBOX_IMAGE="${SANDBOX_IMAGE:-$PINNED_IMAGE}"
require_sandbox                       # 67 image, 69 daemon; pulls nothing
export SANDBOX_IMAGE="$PINNED_IMAGE"
# The key (step 3). A rehearsal may fix it with ACC_TEST_KEY; a set-but-empty value is honoured,
# so the K0 guard is testable. The launcher refuses an empty key at once, before any docker or
# search runs: an empty pattern line would match every line (design 8, item 3).
KEY=$(new_key)
if [ "$REHEARSE" = 1 ] && [ "${ACC_TEST_KEY+x}" = x ]; then KEY=$ACC_TEST_KEY; fi
[ -n "$KEY" ] || die "the bearer key is empty (K0); an empty pattern would match every line" 64
# The pinned image (step 2): the id must start with the pinned prefix and RepoDigests must hold
# the pinned digest (P3). The fake docker of selftest.sh stops earlier, with 67 or 69.
IMG_ID=$($DK image inspect --format '{{.Id}}' "$PINNED_IMAGE" 2>/dev/null) \
  || die "the pinned image $PINNED_IMAGE is not present on the rootless daemon" 67
case "$IMG_ID" in
  "$PINNED_ID_PREFIX"*) ;;
  *) die "the image id is $IMG_ID, want the $PINNED_ID_PREFIX prefix" 89;;
esac
$DK image inspect --format '{{.RepoDigests}}' "$PINNED_IMAGE" 2>/dev/null \
  | grep -q "sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1" \
  || die "the pinned digest is not in RepoDigests of $PINNED_IMAGE" 89
EV=$OUT
mkdir -p "$EV/pi" "$EV/sandbox"
HASHES="$EV/sandbox/hashes.txt"; : > "$HASHES"
hash_note "image_id $IMG_ID"
record_sandbox "$EV/sandbox/versions.txt"   # docker version + info: rootless, runc, containerd (P1b)
{ ssh -V; echo "mem_free_b $(free -b | awk '/^Mem:/ {print $7}')"; echo "disk_free_b $(df -B1 / | awk 'NR==2 {print $4}')"; } >> "$EV/sandbox/versions.txt" 2>&1
$DK ps --format '{{.Names}} {{.ID}} {{.State}}' 8>&- 9>&- 2>/dev/null | grep -v '^accb-' | sed 's/^/others_/' >> "$EV/sandbox/versions.txt"   # the P10 baseline
# The resource guard (step 4, C21, P8): more than 4 GiB free memory and 20 GiB free disk, or stop 93.
FREE_B=$(free -b | awk '/^Mem:/ {print $7}'); DISK_B=$(df -B1 / | awk 'NR==2 {print $4}')
[ "$FREE_B" -gt 4294967296 ] || die "only $FREE_B bytes of memory are free (want > 4 GiB)" 93
[ "$DISK_B" -gt 21474836480 ] || die "only $DISK_B bytes of disk are free (want > 20 GiB)" 93
# The port (4.2): a real run is fixed to 18791 (an inherited PORT that differs is a caller
# error) and opens no TCP port on this host; a rehearsal needs its ACCB_TEST_PORT free here
# (70, rehearsal only).
if [ "$REHEARSE" = 1 ]; then
  local_port_busy "$PORT" && die "port $PORT is in use on this host; it is not touched" 70
else
  [ "$PORT" = 18791 ] || die "PORT is fixed to 18791 for ACC-B (got $PORT)" 64
fi
# The lock (step 5, C21, P6, T29): fd 8, blocking flock; every background child closes 8>&- 9>&-.
mkdir -p "$ACC_B_DIR" && chmod 0700 "$ACC_B_DIR"
[ "$(stat -c %a "$ACC_B_DIR")" = 700 ] || die "$ACC_B_DIR is not mode 700" 89
exec 8>>"$ACC_B_DIR/lock"
flock 8
RUN=accb-$(date -u +%Y%m%dT%H%M%SZ)-$(head -c 2 /dev/urandom | od -An -tx1 | tr -d ' \n')
IPC="$ACC_B_DIR/$RUN"
RUNROOT="$ACCB_SCRATCH/acc-b/$RUN"
STAGE="$RUNROOT/stage"
CLIENT_STATUS="$RUNROOT/docker-client-status"
CAP_NOTE="$RUNROOT/cap-hit"
FORWARD_PID=""; PROBE=""; DEADLINE_HIT=0; DOCKER_STATUS=""; TREE_HASH=""; DPIPE=""; RAN_CONTAINER=0
# Stale containers of earlier runs of THIS launcher only, never anything else (fix5 item 12):
# the design 4.3 flag list is exact and carries no label, so the reap matches the launcher's own
# naming patterns (the run container accb-<UTC time>-<4 hex> and the S12 probe
# accb-t-s12-<4 hex>) and no other accb- or accb-t- name.
reaped=$($DK ps -a --filter name=accb- --format '{{.Names}}' 8>&- 9>&- 2>/dev/null \
  | grep -E '^accb-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{4}$|^accb-t-s12-[0-9a-f]{4}$' || true)
if [ -n "$reaped" ]; then
  say "reaping stale containers of earlier runs of this launcher: $(echo "$reaped" | tr '\n' ' ')"
  for c in $reaped; do $DK rm -f "$c" 8>&- 9>&- 2>/dev/null; done
  echo "reaped_stale $(echo "$reaped" | tr '\n' ' ')" >> "$EV/sandbox/versions.txt"
fi

# --- the EXIT trap (step 17, C16, design 14): the same order as the teardown -------------------------
accb_teardown() {
  local status=$? n line leftover mismatch=0
  trap 'exit 143' TERM INT HUP
  # (1) the container: record its status, then remove it. The pi child ends with the container.
  if [ -n "$RUN" ]; then
    if [ -d "$EV" ]; then
      $DK inspect --format 'status {{.State.Status}} exit {{.State.ExitCode}}' "$RUN" 8>&- 9>&- 2>/dev/null \
        >> "$EV/pi/docker-exit.txt" || true
    fi
    $DK rm -f "$RUN" 8>&- 9>&- 2>/dev/null
  fi
  # The S12 probe (step 10) is this launcher's own container; remove it by name when the run
  # ended before its own rm (a TERM during the probe window would otherwise leave it behind).
  if [ -n "$PROBE" ]; then
    $DK rm -f "$PROBE" 8>&- 9>&- 2>/dev/null
    PROBE=""
  fi
  # (2) the forward (ssh or relay) by its recorded PID: SIGTERM, wait. No SIGKILL, no signal by
  # pattern.
  if [ -n "$FORWARD_PID" ] && kill -0 "$FORWARD_PID" 2>/dev/null; then
    kill -TERM "$FORWARD_PID" 2>/dev/null
    n=0
    while kill -0 "$FORWARD_PID" 2>/dev/null && [ "$n" -lt 30 ]; do sleep 1; n=$((n + 1)); done
    wait "$FORWARD_PID" 2>/dev/null
  fi
  FORWARD_PID=""
  # (3) the socket, the IPC directory, and the acc-b directory when empty. The lock descriptor
  # is closed first, so the rmdir of the lock directory can succeed on the last run and stays
  # silent when another run is waiting on the lock (fix5 item 12).
  rm -f "$IPC/fifi.sock" 2>/dev/null
  rmdir "$IPC" 2>/dev/null
  exec 8>&- 2>/dev/null || true
  rmdir "$ACC_B_DIR" 2>/dev/null || true
  # (4) the serve session: the STOP file, the Mac lock released.
  session_stop
  # The stamping reader must end before the seal, so SHA256SUMS matches the final stamped file
  # (fix5 item 12): the container and the forward are gone by now, so the reader is at EOF; the
  # bound covers a wedged docker client.
  if [ -n "$DPIPE" ] && kill -0 "$DPIPE" 2>/dev/null; then
    n=0
    while kill -0 "$DPIPE" 2>/dev/null && [ "$n" -lt 60 ]; do sleep 1; n=$((n + 1)); done
  fi
  DPIPE=""
  # (5) re-hash pi and node (X4, C22): a mismatch fails the run with 89.
  if [ -d "$EV" ] && [ -f "$STAGE/opt/node" ]; then
    {
      echo "after node_bin $(sha256sum "$STAGE/opt/node" 2>/dev/null | cut -d' ' -f1)"
      echo "after package_lock $(sha256sum "$STAGE/opt/pi/package-lock.json" 2>/dev/null | cut -d' ' -f1)"
      echo "after metadata $(sha256sum "$STAGE/opt/pi/metadata.json" 2>/dev/null | cut -d' ' -f1)"
      echo "after pi_tree_hash $(pi_tree_hash "$STAGE/opt/pi")"
    } >> "$EV/sandbox/hashes.txt"
    [ "$(sha256sum "$STAGE/opt/node" 2>/dev/null | cut -d' ' -f1)" = "$NODE_SHA256" ] || mismatch=1
    [ "$(sha256sum "$STAGE/opt/pi/package-lock.json" 2>/dev/null | cut -d' ' -f1)" = "$LOCKFILE_SHA256" ] || mismatch=1
    [ "$(sha256sum "$STAGE/opt/pi/metadata.json" 2>/dev/null | cut -d' ' -f1)" = "$METADATA_SHA256" ] || mismatch=1
    [ "$(pi_tree_hash "$STAGE/opt/pi")" = "$TREE_HASH" ] || mismatch=1
  fi
  # The run scratch (the pi and node copies, the client status, the probe inspect) is removed at
  # teardown (fix5 item 12); the evidence keeps only the hashes.
  rm -rf "$RUNROOT" 2>/dev/null
  RUNROOT=""
  # The key scan (C15, C6) runs on every exit path after the container ran, not only on the
  # normal path (fix5 item 3): a leak is exit 90 with the evidence removed as on the normal
  # path; an exit that is already abnormal keeps its stronger status and the leak is recorded.
  KEY_LEAK=no
  if [ "$RAN_CONTAINER" = 1 ] && [ -d "$EV" ] && [ -n "${KEY:-}" ]; then
    if { key_in_dir "$EV" || grep -rqF -f <(printf '%s\n' "$KEY") "$EV/sandbox/inspect.json" 2>/dev/null; }; then
      KEY_LEAK=yes
    fi
  fi
  # (6) the local X1 to X4 checks; a leftover is listed, never deleted by guess.
  if [ -d "$EV" ]; then
    {
      leftover=$($DK ps -aq --filter name=accb- 2>/dev/null | tr -d ' \n')
      echo "x1_containers_left ${leftover:-none}"
      echo "x2_forward_pids $(pgrep -f "$IPC/fifi.sock" 2>/dev/null | tr -d ' \n' || echo none)"
      echo "x3_accb_dir $(ls -A "$ACC_B_DIR" 2>/dev/null | tr -d ' \n' || echo gone)"
      echo "x4_rehash $([ "$mismatch" = 0 ] && echo ok || echo MISMATCH)"
      echo "key_leak $KEY_LEAK"
      echo "others_after $($DK ps --format '{{.Names}} {{.ID}} {{.State}}' 2>/dev/null | grep -v '^accb-' || true)"
    } > "$EV/sandbox/teardown.txt"
    seal "$EV"     # teardown.txt joins the sealed evidence
  fi
  if [ "$KEY_LEAK" = yes ]; then
    if [ "$status" = 0 ] || [ "$status" = 2 ] || [ "$status" = 3 ]; then
      echo "the key appears in the evidence; removing $EV" >&2
      rm -rf "$EV"
      status=90
    else
      echo "the key appears in the evidence; the stronger exit status $status is kept and the leak is recorded in sandbox/teardown.txt" >&2
    fi
  fi
  [ "$mismatch" = 0 ] || status=89
  exit "$status"
}
trap 'exit 143' TERM INT HUP
trap accb_teardown EXIT
# The trap is installed before $IPC is created (fix5 item 12): an abort between the lock and
# the socket directory is cleaned by the trap, not left behind.
mkdir -m 0755 "$IPC"
[ "$(stat -c %a "$IPC")" = 755 ] || die "$IPC is not mode 755" 89
say "ACC-B launcher: branch $BRANCH, rehearse $REHEARSE, run $RUN, evidence $EV"

# --- staging (step 6) ------------------------------------------------------------------------------
stage_build "$STAGE" "$IPC" 0
mkdir -p "$EV/sandbox/stage"
cp "$STAGE/entry.sh" "$STAGE/bridge.js" "$EV/sandbox/stage/"
cp "$STAGE/agent/models.json" "$STAGE/agent/settings.json" "$EV/sandbox/stage/"
RUN_ARG="$RUN"
mapfile -t DOCKER_ARGS < <(docker_run_args "$STAGED_FIX" "$STAGE" "$IPC")
printf '%s\n' "${DOCKER_ARGS[@]}" > "$EV/sandbox/docker-run.txt"

# --- ssh staging, real mode only (step 7, C20) ------------------------------------------------------
if [ "$REHEARSE" != 1 ]; then
  write_ssh_config "$STAGE"
  ssh-keygen -F "$SSH_HOSTNAME" -f ~/.ssh/known_hosts > "$STAGE/known_hosts" 2>/dev/null
  [ -s "$STAGE/known_hosts" ] || die "ssh-keygen -F $SSH_HOSTNAME printed nothing; no known host key" 92
  hash_note "known_hosts $(sha256sum "$STAGE/known_hosts" | cut -d' ' -f1)"
  hash_note "ssh_config $(sha256sum "$STAGE/ssh_config" | cut -d' ' -f1)"
  assert_ssh_g "$STAGE/ssh_config" "$IPC/fifi.sock" "$STAGE/known_hosts"
  ssh -G -F "$STAGE/ssh_config" -a -x -N -L "$IPC/fifi.sock:$FWD_DEST" mac >> "$EV/sandbox/versions.txt" 2>&1
fi

# --- the server session (step 8), through driver-lib.sh --------------------------------------------
# The rehearsal fake server is started with --mode pi-tools --model-id flash (step 8): pi-tools
# scripts the `write` and `bash` tool calls of the task, the only tools the entry grants.
FAKE_MODE=pi-tools
MODEL_ID=flash
if [ "$REHEARSE" = 1 ]; then
  # In a rehearsal the serve session must not hold the acc-b lock fd (4.2, Appendix B item 5):
  # session_start runs with fd 8 closed for the duration of the call (the launcher keeps the lock
  # on its own fd 8, and the saved copy is not inherited by children). The session stays a child
  # of this shell, so session_stop can `wait` it and SERVE_STATUS is its real status instead of
  # the 127 of `wait` on a process that is not a child of this shell.
  session_start "$BRANCH" "agent1-acc-b" 8>&- || die "the serve session did not start" 81
else
  session_start "$BRANCH" "agent1-acc-b"
fi
wait_ready || { seal "$EV"; exit 83; }

# --- the forward (step 9, C2, C3, R2) -------------------------------------------------------------
if [ "$REHEARSE" != 1 ]; then
  FORWARD_CMD=(timeout -s TERM -k 10 "$SSH_DEADLINE" ssh -n -F "$STAGE/ssh_config" -a -x -N -L "$IPC/fifi.sock:$FWD_DEST" mac)
  "${FORWARD_CMD[@]}" 8>&- 9>&- &
else
  python3 -I "$AH_DIR/accb-relay.py" --socket "$IPC/fifi.sock" --target "127.0.0.1:$PORT" 8>&- 9>&- &
fi
FORWARD_PID=$!
n=0
until [ -S "$IPC/fifi.sock" ]; do
  kill -0 "$FORWARD_PID" 2>/dev/null || { echo "the forward is not running" >&2; seal "$EV"; exit 84; }
  [ "$n" -lt 100 ] || { echo "the forward socket never appeared" >&2; seal "$EV"; exit 84; }
  sleep 0.2; n=$((n + 1))
done
chmod 666 "$IPC/fifi.sock"    # 4.2: never rely on a umask (C3)
{
  echo "forward_pid $FORWARD_PID"
  echo "forward_ps $(ps -o args= -p "$FORWARD_PID" 2>/dev/null)"
  echo "socket_mode $(stat -c %a "$IPC/fifi.sock")"
  echo "ipc_mode $(stat -c %a "$IPC")"
  echo "accb_dir_mode $(stat -c %a "$ACC_B_DIR")"
} > "$EV/sandbox/ssh.txt"
[ "$(stat -c %a "$IPC/fifi.sock")" = 666 ] || { echo "the socket mode is not 666" >&2; seal "$EV"; exit 84; }
[ "$(stat -c %a "$IPC")" = 755 ] || { echo "the IPC directory mode is not 755" >&2; seal "$EV"; exit 84; }
[ "$(stat -c %a "$ACC_B_DIR")" = 700 ] || { echo "the acc-b directory mode is not 700" >&2; seal "$EV"; exit 84; }
[ "$(curl -s -o /dev/null -w '%{http_code}' -m 10 --unix-socket "$IPC/fifi.sock" http://127.0.0.1/ready)" = 200 ] \
  || { echo "/ready is not 200 through the socket" >&2; seal "$EV"; exit 84; }

# --- the S12 probe (step 10): the flag list, field by field, before the key is sent ---------------
PROBE="accb-t-s12-$(head -c 2 /dev/urandom | od -An -tx1 | tr -d ' \n')"
run_probe() { # <name> <entrypoint>: the 4.3 list with the name and entrypoint replaced, command dropped
  local i saved=$RUN_ARG A
  RUN_ARG="$1"
  mapfile -t A < <(docker_run_args "$STAGED_FIX" "$STAGE" "$IPC")
  RUN_ARG="$saved"
  for ((i=0; i<${#A[@]}; i++)); do
    if [ "${A[i]}" = "--entrypoint" ]; then printf '%s\n' "--entrypoint" "$2"; i=$((i+1))
    elif [ "$i" -eq $((${#A[@]}-1)) ]; then :;      # drop the trailing /opt/accb/entry.sh
    else printf '%s\n' "${A[i]}"; fi
  done
}
mapfile -t P_ARGS < <(run_probe "$PROBE" /bin/true)
"${P_ARGS[@]}" 8>&- 9>&- < /dev/null > "$EV/sandbox/preflight.txt" 2>&1 || die "the S12 probe container did not run" 89
$DK inspect "$PROBE" > "$RUNROOT/probe-inspect.json" 2>/dev/null || die "cannot inspect the S12 probe" 89
s12_compare "$RUNROOT/probe-inspect.json" "$STAGED_FIX" "$STAGE" "$IPC" >> "$EV/sandbox/preflight.txt" 2>&1 \
  || die "the probe container differs from design 4.3 (S12)" 89
$DK rm -f "$PROBE" 8>&- 9>&- 2>/dev/null
PROBE=""

# --- the run (step 11) ------------------------------------------------------------------------------
say "starting the container (deadline ${ACCB_DEADLINE}s)"
RAN_CONTAINER=1
printf '%s\n' "$KEY" | python3 -I -c "$ACCB_STAMPER" \
  "$STDOUT_CAP" "$STDERR_CAP" "$DKHOST" "$RUN" "$CAP_NOTE" "$CLIENT_STATUS" \
  "$EV/pi/pi.stamped.tsv" "$EV/pi/accb-peak.txt" -- "${DOCKER_ARGS[@]}" 8>&- 9>&- &
DPIPE=$!
DEADLINE_AT=$(( $(date +%s) + ACCB_DEADLINE ))
while kill -0 "$DPIPE" 2>/dev/null; do
  if [ "$(date +%s)" -ge "$DEADLINE_AT" ]; then
    DEADLINE_HIT=1
    $DK kill "$RUN" 8>&- 9>&- 2>/dev/null
    echo "the launcher deadline of ${ACCB_DEADLINE}s hit; docker kill $RUN" >&2
    n=0
    while kill -0 "$DPIPE" 2>/dev/null && [ "$n" -lt 60 ]; do sleep 1; n=$((n + 1)); done
    break
  fi
  sleep 1
done
wait "$DPIPE" 2>/dev/null; DPIPE=""
CAP_HIT=no; [ -f "$CAP_NOTE" ] && CAP_HIT=yes

# --- the status, from Docker only (step 12, C18) ----------------------------------------------------
DOCKER_STATUS=$(cat "$CLIENT_STATUS" 2>/dev/null || echo none)
INSPECT_STATUS=$($DK inspect --format '{{.State.ExitCode}}' "$RUN" 8>&- 9>&- 2>/dev/null || echo none)
WAIT_STATUS=$($DK wait "$RUN" 8>&- 9>&- 2>/dev/null | tr -d ' \n' || echo none)
$DK inspect "$RUN" > "$EV/sandbox/inspect.json" 8>&- 9>&- 2>/dev/null
# The same field-by-field S12 compare runs on the real run container, not only on the probe
# (fix5 item 9).
s12_compare "$EV/sandbox/inspect.json" "$STAGED_FIX" "$STAGE" "$IPC" >> "$EV/sandbox/preflight.txt" 2>&1 \
  || die "the run container differs from design 4.3 (S12)" 89
{
  echo "docker_client $DOCKER_STATUS"
  echo "docker_inspect $INSPECT_STATUS"
  echo "docker_wait $WAIT_STATUS"
  echo "cap_hit $CAP_HIT"
  echo "deadline_hit $DEADLINE_HIT"
} > "$EV/pi/docker-exit.txt"
echo "$DOCKER_STATUS" > "$EV/pi/pi.exit"
if [ "$DEADLINE_HIT" = 1 ]; then
  seal "$EV"
  exit 91               # C9: the launcher deadline (M2)
fi
STATUS_AGREE=ok
for s in "$INSPECT_STATUS" "$WAIT_STATUS"; do
  [ "$s" = "$DOCKER_STATUS" ] || STATUS_AGREE="disagree ($s)"
done
# The status that is passed to the checker must be an integer (fix5 item 4): the stamper writes
# it, and its fallback is the word `none`.
case "$DOCKER_STATUS" in
  ''|*[!0-9]*) STATUS_AGREE="not an integer ($DOCKER_STATUS)";;
esac
[ "$STATUS_AGREE" = ok ] || { echo "the docker statuses are not usable: client $DOCKER_STATUS, inspect $INSPECT_STATUS, wait $WAIT_STATUS ($STATUS_AGREE); a pipeline defect" >&2; CHECK_STATUS=2; }
$DK rm -f "$RUN" 8>&- 9>&- 2>/dev/null
say "container ended with docker status $DOCKER_STATUS"

# --- the export (step 13, C14) ---------------------------------------------------------------------
cut -f2- "$EV/pi/pi.stamped.tsv" > "$EV/pi/pi.jsonl"
python3 -I "$AH_DIR/accb_export.py" --stream "$EV/pi/pi.jsonl" --dest "$EV/pi/work" \
  --manifest "$EV/pi/export-manifest.txt" || die "the export extractor failed" 2

# --- stop the serve session and copy the evidence (step 14) -----------------------------------------
session_stop
if [ "$REHEARSE" = 1 ]; then
  mkdir -p "$EV/evidence"; cp -R "$MS/evidence/." "$EV/evidence/" || die "cannot copy the rehearsal evidence" 82
else
  mac_fetch "$EV" || true
  ssh_mac "lsof -nP -iTCP:8788 -sTCP:LISTEN" > "$EV/mac-after-8788.txt" 2>&1
fi
CHECK_STATUS=""
if [ "$STATUS_AGREE" = ok ]; then
  python3 "$AH_DIR/check_pi_run.py" --transcript "$EV/pi/pi.stamped.tsv" --exit-code "$DOCKER_STATUS" \
    --workdir "$EV/pi/work" --server-log "$EV/evidence/server.log" --out "$EV/check"
  CHECK_STATUS=$?
else
  # Fix5 item 4: a status disagreement or a non-integer status is a pipeline defect that is
  # never discarded; the checker is skipped because its --exit-code would be unusable.
  CHECK_STATUS=2
fi
if [ "$CHECK_STATUS" = 0 ] && [ "$SERVE_STATUS" != 0 ]; then CHECK_STATUS=$SERVE_STATUS; fi

# --- the key scan (step 15, C15, C6) ---------------------------------------------------------------
if key_in_dir "$EV" || grep -rqF -f <(printf '%s\n' "$KEY") "$EV/sandbox/inspect.json" 2>/dev/null; then
  echo "the key appears in the evidence; removing $EV" >&2
  rm -rf "$EV"
  exit 90
fi

# --- launcher.txt and the seal (step 16) ------------------------------------------------------------
{ echo "branch $BRANCH"; echo "rehearse $REHEARSE"; echo "run $RUN"; echo "docker_status $DOCKER_STATUS";
  echo "check_status ${CHECK_STATUS:-none}"; echo "serve_status $SERVE_STATUS";
  echo "launcher_head $(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null)"; } > "$EV/launcher.txt"
seal "$EV"
say "docker status $DOCKER_STATUS; check status ${CHECK_STATUS}; evidence sealed in $EV/SHA256SUMS"
exit "${CHECK_STATUS:-2}"
