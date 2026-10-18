#!/usr/bin/env bash
# Self-test of the launch scripts. It rehearses them on this host against fake_server.py: no Mac,
# no ssh, no lock, no model, no container. The sandbox is fake_docker.py, which runs nothing. The
# pi rehearsal runs only when `pi` is installed and SELFTEST_SKIP_PI is unset; the abort checks
# use a fake `pi` and always run.
# Usage: selftest.sh [scratch dir]   (default: a new directory under $TMPDIR)
# Environment: STOP_TIMEOUT (seconds the Mac script waits for its server, default 60),
#   SELFTEST_PORT_A, SELFTEST_PORT_B, SELFTEST_SKIP_PI.
# Exit status: 0 when every check passes, 1 otherwise. It tests the scripts, not the engine.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOP=${1:-$(mktemp -d)}
FAILS=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; FAILS=$((FAILS + 1)); }
expect_status() { # <want> <got> <label>
  if [ "$1" = "$2" ]; then ok "$3 (status $2)"; else bad "$3: want status $1, got $2"; fi
}
mkdir -p "$TOP"
# The scripts find the client files beside themselves, and the repository two levels above. In a
# repository checkout that layout exists. Here it is rebuilt in the scratch directory with a link.
REPO=$TOP/repo
mkdir -p "$REPO/qualification"
ln -sfn "$HERE" "$REPO/qualification/agent-acceptance"
# The q36 engine reads the shipped config from the source tree it runs (engine/config).
mkdir -p "$REPO/engine"; ln -sfn "$HERE/../../engine/config" "$REPO/engine/config"
AH=$REPO/qualification/agent-acceptance

for f in mac-acc-a.sh kubdan-acc-a.sh driver-lib.sh run-acc-a.sh run-acc-b.sh accb-entry.sh selftest.sh selftest-accb.sh selftest-cuda.sh selftest-q36-config.sh mutate-accb.sh mutate-q36-config.sh mutate-q36-cuda.sh; do
  if bash -n "$AH/$f"; then ok "syntax $f"; else bad "syntax $f"; fi
done

# No broad kill form anywhere in this directory (process-signal rule of 2026-10-11): signals go
# to recorded or discovered pids only, never to every process at once, and never by name
# pattern. Comment lines are allowed to mention the rule; the pattern below is assembled from
# pieces so that this check does not match its own source line.
KILLPAT='kill [^|;]*-1\b|pk''ill|ki''llall'
broad_kills=$(grep -rInE --exclude-dir=__pycache__ "$KILLPAT" "$AH" | grep -vE '^[^:]+:[0-9]+: *#' || true)
if [ -n "$broad_kills" ]; then
  bad "a broad kill form survives outside comments:"$'\n'"$broad_kills"
else
  ok "no broad kill form outside comments (only signals to explicit pids)"
fi

# The run-acc-b exit statuses cannot drift from the README again (review finding 5: the README
# once listed 80 and 85, which the script never exits). The README table, the script's own
# exit-status comment, and the literal die/exit statuses of run-acc-b.sh and driver-lib.sh must
# agree.
if python3 -I - "$AH" <<'STATPY'
import re, sys
ah = sys.argv[1]
readme = open(ah + "/README.md").read()
m = re.search(r"Exit status of `run-acc-b\.sh`.*?\n\n((?:\|[^\n]*\n)+)", readme, re.S)
assert m, "README.md has no run-acc-b status table"
rows = [r for r in m.group(1).splitlines() if re.match(r"^\| *\d+ *\|", r)]
assert rows, "the run-acc-b status table has no numbered rows"
readme_set = {re.match(r"^\| *(\d+) *\|", r).group(1) for r in rows}
script = open(ah + "/run-acc-b.sh").read()
m2 = re.search(r"^# Exit status:.*?(?=\n[^#])", script, re.M | re.S)
assert m2, "run-acc-b.sh has no exit-status comment"
comment_set = set(re.findall(r"\d+", m2.group(0)))
assert readme_set == comment_set, (
    "README %s != script comment %s" % (sorted(readme_set, key=int), sorted(comment_set, key=int)))
lit = set()
for name in ("run-acc-b.sh", "driver-lib.sh"):
    for line in open(ah + "/" + name):
        code = line.split("#", 1)[0]
        lit |= set(re.findall(r'\bdie\s+"[^"]*"\s+(\d+)\b', code))
        lit |= set(re.findall(r"\bexit (\d+)\b", code))
missing = lit - readme_set
assert not missing, "literal exit statuses missing from the README table: %s" % sorted(missing, key=int)
print("statuses ok:", " ".join(sorted(readme_set, key=int)))
STATPY
then ok "the run-acc-b status table matches the script comment and every literal die/exit"; else bad "the run-acc-b status table drifted (see the python message above)"; fi

PORT_A=${SELFTEST_PORT_A:-18797}
PORT_B=${SELFTEST_PORT_B:-18799}
TEST_KEY=selftest-key-8841
# The fake docker runs nothing. The image name is digest-pinned and "present" for the fake only.
export DOCKER=$AH/fake_docker.py
export SANDBOX_IMAGE=fake@sha256:0000000000000000000000000000000000000000000000000000000000000000
export STOP_TIMEOUT=${STOP_TIMEOUT:-60}
# A launcher that cannot end its server session must fail the test, not wait 1000 s.
export SESSION_STOP_WAIT=${SESSION_STOP_WAIT:-30}
# A fake ssh and flock: a launcher that reaches the Mac path in these tests leaves a marker.
FB=$TOP/fakebin; mkdir -p "$FB"
for tool in ssh scp flock; do
  printf '#!/bin/sh\ntouch "%s/%s-called"\nexit 1\n' "$TOP" "$tool" > "$FB/$tool"; chmod +x "$FB/$tool"
done

run_a() { # <label> <out> [env assignments...]
  local label=$1 out=$2; shift 2
  env "$@" ACC_TEST_KEY=$TEST_KEY PORT=$PORT_A bash "$AH/run-acc-a.sh" "$out" --rehearse > "$TOP/$label.log" 2>&1
  return $?
}
leftovers() { # <label>: no server, session, or fake pi may remain
  if pgrep -f "fake_server.py --port $PORT_A|fake_server.py --port $PORT_B|mac-acc-a.sh serve $TOP" > /dev/null; then bad "$1: a server or session is still running"; else ok "$1: no server or session remains"; fi
}

run_a good "$TOP/a-good"; expect_status 0 $? "ACC-A rehearsal passes"
if (cd "$TOP/a-good" && sha256sum -c SHA256SUMS --quiet); then ok "SHA256SUMS verifies"; else bad "SHA256SUMS"; fi
if grep -q ' \./evidence/SHA256SUMS$' "$TOP/a-good/SHA256SUMS"; then ok "the top SHA256SUMS covers evidence/SHA256SUMS"; else bad "the top SHA256SUMS has no line for evidence/SHA256SUMS"; fi
if grep -q ' \./SHA256SUMS$' "$TOP/a-good/SHA256SUMS"; then bad "the top SHA256SUMS lists itself"; else ok "the top SHA256SUMS does not list itself"; fi
if (cd "$TOP/a-good/evidence" && sha256sum -c SHA256SUMS --quiet); then ok "inner SHA256SUMS verifies"; else bad "inner SHA256SUMS"; fi
for f in acc-a/timings.tsv acc-a/transcript.jsonl acc-a/result.json evidence/summary.txt evidence/server.log evidence/rss.tsv evidence/identity.txt; do
  if [ -s "$TOP/a-good/$f" ]; then ok "evidence $f"; else bad "evidence $f is missing or empty"; fi
done
if grep -q 'component runc ' "$TOP/a-good/sandbox-daemon.txt" && grep -q 'component containerd ' "$TOP/a-good/sandbox-daemon.txt" && grep -q 'docker_host unix:///run/user/' "$TOP/a-good/sandbox-daemon.txt"; then ok "the evidence records the rootless daemon, runc, and containerd"; else bad "sandbox-daemon.txt is missing or incomplete"; fi
if grep -q '"ok": true' "$TOP/a-good/acc-a/result.json"; then ok "the server log post-check passed"; else bad "the server log post-check is missing"; fi
if grep -q '^swap_grew ' "$TOP/a-good/evidence/summary.txt"; then ok "summary records swap growth"; else bad "swap line"; fi
leftovers "ACC-A"
if [ -e "$TOP/ssh-called" ] || [ -e "$TOP/flock-called" ]; then bad "a rehearsal reached ssh or flock"; else ok "a rehearsal never reaches ssh or flock"; fi

run_a again "$TOP/a-good"; expect_status 64 $? "an existing evidence directory is refused"
env PORT=8788 bash "$AH/run-acc-a.sh" "$TOP/a-8788" --rehearse > "$TOP/a-8788.log" 2>&1; expect_status 64 $? "run-acc-a refuses port 8788"
env PORT=8788 bash "$AH/run-acc-b.sh" "$TOP/b-8788" --rehearse > "$TOP/b-8788.log" 2>&1; expect_status 64 $? "run-acc-b refuses port 8788"
printf 'k\n' | env REHEARSE=1 PORT=8788 SERVER_CMD=true bash "$AH/mac-acc-a.sh" serve "$TOP/m8788" "$REPO" > "$TOP/m-8788.log" 2>&1; expect_status 64 $? "mac-acc-a refuses port 8788"
printf '\n' | env REHEARSE=1 PORT=$PORT_A SERVER_CMD=true bash "$AH/mac-acc-a.sh" serve "$TOP/mnokey" "$REPO" > "$TOP/m-nokey.log" 2>&1; expect_status 64 $? "mac-acc-a needs a key on stdin"
for old in acc-a smoke; do
  printf 'k\n' | env REHEARSE=1 PORT=$PORT_A SERVER_CMD=true bash "$AH/mac-acc-a.sh" $old "$TOP/mold-$old" "$REPO" > "$TOP/m-old-$old.log" 2>&1
  expect_status 64 $? "mac-acc-a no longer has the mode $old (no client runs on the Mac)"
done

run_a quality "$TOP/a-quality" FAKE_MODE=quality-text; expect_status 3 $? "a model quality miss gives status 3"
run_a pipeline "$TOP/a-pipeline" FAKE_MODE=leak; expect_status 2 $? "a pipeline defect gives status 2"
env ACC_TEST_KEY=$TEST_KEY PORT=$PORT_A bash "$AH/run-acc-a.sh" "$TOP/a-smoke2" --rehearse --mode smoke > "$TOP/a-smoke2.log" 2>&1; expect_status 0 $? "smoke mode passes"

# The engine selector. flash is the default. q36 rehearses against a scratch checkpoint whose four
# file hashes make the rehearsal pin file; the pinned Mac path and hashes are checked as constants.
if grep -q '"family":"qwen3_8_flash_next"' "$TOP/a-good/evidence/config.json" \
   && grep -q '"model":"flash"' "$TOP/a-good/evidence/config.json" && ! grep -q qwen36 "$TOP/a-good/evidence/config.json" \
   && grep -q '^engine flash$' "$TOP/a-good/evidence/identity.txt" && grep -q '^engine flash$' "$TOP/a-good/launcher.txt" \
   && grep -q '"engine": "flash"' "$TOP/a-good/acc-a/result.json"; then ok "the default engine is flash: config, identity, launcher, and result agree"; else bad "default engine records"; fi
PIN_FILE=$AH/q36-pins.sha256
for want in a822a9e48b0aafbe3144ec37d4fb067e178ed96615ce6e4420b3149893cc5767:config.json \
            e359cd95565275b923fed8df019dd8e869d9af5dc4a621b4114ab7a8e6c534d1:model.safetensors.index.json \
            87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4:tokenizer.json \
            e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259:chat_template.jinja; do
  if grep -qx "${want%%:*}  ${want#*:}" "$PIN_FILE"; then ok "the Q36 pin for ${want#*:} is the approved hash"; else bad "Q36 pin ${want#*:}"; fi
done
if [ "$(wc -l < "$PIN_FILE")" = 4 ]; then ok "the Q36 pin file has exactly four lines"; else bad "Q36 pin file length"; fi
if grep -q '^Q36_PINNED_DIR=/Users/dan/edge0/models/edge0-35b$' "$AH/mac-acc-a.sh"; then ok "the pinned Q36 checkpoint path is the edge0-35b directory"; else bad "pinned Q36 path"; fi
Q36CK=$TOP/q36-ck; mkdir -p "$Q36CK"
for f in config.json model.safetensors.index.json tokenizer.json chat_template.jinja; do echo "scratch $f" > "$Q36CK/$f"; done
(cd "$Q36CK" && sha256sum config.json model.safetensors.index.json tokenizer.json chat_template.jinja) > "$TOP/q36-pins.test"
run_q36() { # <label> <out> [env assignments...]
  local label=$1 out=$2; shift 2
  env "$@" ACC_TEST_KEY=$TEST_KEY PORT=$PORT_A Q36_DIR_OVERRIDE="$Q36CK" Q36_PINS_OVERRIDE="$TOP/q36-pins.test" \
    bash "$AH/run-acc-a.sh" "$out" --rehearse --engine q36 > "$TOP/$label.log" 2>&1
  return $?
}
run_q36 q36 "$TOP/a-q36"; expect_status 0 $? "ACC-A rehearsal with --engine q36 passes"
if python3 -I - "$TOP/a-q36/evidence/config.json" "$Q36CK" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
want = {"id": "q36", "family": "qwen3_6_35b_a3b", "model": "fifi-q36", "model_dir": sys.argv[2],
        "tokenizer_dir": sys.argv[2],
        "qwen36": {"cache_limit_bytes": 4294967296, "context_limit": 262144,
                   "session_cap_bytes": 4294967296, "prefill_budget_bytes": 32212254720,
                   "prefill_session_reserve_bytes": 8286699520,
                   "activation_dtype": "bf16", "kv_dtype": "q8", "compile": True}}
assert config["engines"] == [want], config["engines"]
assert config["api_key_env"] == "ENGINESRV_API_KEY" and config["listen"].startswith("127.0.0.1:"), config
PY
then ok "the Q36 config holds the family, model id fifi-q36, both directories, and the shipped qwen36 object"; else bad "Q36 config"; fi
if python3 -I - "$TOP/a-q36/acc-a/timings.tsv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
assert rows, "no request rows"
assert all(r["total_seconds"] for r in rows), rows
streams = [r for r in rows if r["stream"] == "True"]
assert streams and all(r["first_delta_seconds"] for r in streams), streams
PY
then ok "the Q36 timings hold total seconds per request and the first delta of each streamed request"; else bad "Q36 timings"; fi
# The focused config checks and their mutations (mac-acc-a.sh read as a user would run it).
if bash "$AH/selftest-q36-config.sh" "$AH" "$TOP/q36cfg" > "$TOP/q36cfg.log" 2>&1; then ok "the Q36 shipped-config self-test passes"; else bad "Q36 shipped-config self-test (see $TOP/q36cfg.log)"; fi
if bash "$AH/mutate-q36-config.sh" "$TOP/q36mut" > "$TOP/q36mut.log" 2>&1; then ok "every Q36 shipped-config mutation is killed"; else bad "Q36 mutation survived (see $TOP/q36mut.log)"; fi
if grep -q '"model": "fifi-q36"' "$TOP/a-q36/acc-a/result.json" \
   && grep -q '"engine": "q36"' "$TOP/a-q36/acc-a/result.json" && grep -q '^model fifi-q36$' "$TOP/a-q36/launcher.txt" \
   && grep -q '"verdict": "pass"' "$TOP/a-q36/acc-a/result.json"; then ok "the Q36 run records the model id and engine in the result"; else bad "Q36 result records"; fi
if grep -q '"id": "fifi-q36"' "$TOP/a-q36/acc-a/models.json"; then ok "the server lists the model id fifi-q36"; else bad "models.json"; fi
if grep -q '^engine q36$' "$TOP/a-q36/evidence/identity.txt" && grep -q '^model_id fifi-q36$' "$TOP/a-q36/evidence/identity.txt" \
   && grep -q '^cache_limit_bytes 4294967296$' "$TOP/a-q36/evidence/identity.txt" \
   && [ "$(grep -c '^artifact_file ' "$TOP/a-q36/evidence/identity.txt")" = 4 ] \
   && grep -qx "artifact_file tokenizer.json $(sha256sum "$Q36CK/tokenizer.json" | cut -d' ' -f1)" "$TOP/a-q36/evidence/identity.txt"; then ok "the Q36 identity records the engine and the four artifact hashes"; else bad "Q36 identity"; fi
if (cd "$TOP/a-q36" && sha256sum -c SHA256SUMS --quiet); then ok "the Q36 SHA256SUMS verifies"; else bad "Q36 SHA256SUMS"; fi
leftovers "ACC-A q36"
echo "tampered" >> "$Q36CK/chat_template.jinja"
run_q36 q36bad "$TOP/a-q36bad"; expect_status 88 $? "a changed Q36 artifact stops with status 88"
if grep -q 'Q36 PIN MISMATCH: chat_template.jinja' "$TOP/q36bad.log" || grep -q 'Q36 PIN MISMATCH: chat_template.jinja' "$TOP/a-q36bad/serve.out" 2>/dev/null; then ok "the mismatch names the changed file"; else bad "mismatch message"; fi
if [ -e "$TOP/a-q36bad/acc-a" ] || pgrep -f "fake_server.py --port $PORT_A" > /dev/null; then bad "a pin mismatch started the client or a server"; else ok "a pin mismatch starts no server and no client"; fi
leftovers "ACC-A q36 mismatch"
env ACC_TEST_KEY=$TEST_KEY PORT=$PORT_A bash "$AH/run-acc-a.sh" "$TOP/a-q36pinned" --rehearse --engine q36 > "$TOP/q36pinned.log" 2>&1; expect_status 88 $? "without an override a rehearsal checks the pinned Mac path, which is absent here"
echo "scratch chat_template.jinja" > "$Q36CK/chat_template.jinja"  # restore the content that matches the test pins
# Outside a rehearsal the overrides do nothing: the script checks the pinned Mac path, absent here,
# and stops with 88 before the build. (If it honored them it would go on and fail with 80.)
mkdir -p "$TOP/mreal"; : > "$TOP/mreal/env.sh"
printf 'k\n' | env ENGINE=q36 PORT=$PORT_A Q36_DIR_OVERRIDE="$Q36CK" Q36_PINS_OVERRIDE="$TOP/q36-pins.test" bash "$AH/mac-acc-a.sh" serve "$TOP/mreal" "$REPO" > "$TOP/m-real-q36.log" 2>&1
expect_status 88 $? "outside a rehearsal the Q36 overrides are ignored"
# env.sh is the user's file and a real run sources it. Whatever it sets, the pinned path, the pin
# file, and the file list stay constants. Here env.sh points them at the scratch checkpoint, whose
# files match the test pins: if that worked, the run would go on and fail with 80.
mkdir -p "$TOP/mreal2"
printf 'Q36_PINNED_DIR=%s\nQ36_DIR=%s\nQ36_PINS=%s\nQ36_FILES="config.json"\n' "$Q36CK" "$Q36CK" "$TOP/q36-pins.test" > "$TOP/mreal2/env.sh"
printf 'k\n' | env ENGINE=q36 PORT=$PORT_A bash "$AH/mac-acc-a.sh" serve "$TOP/mreal2" "$REPO" > "$TOP/m-real2-q36.log" 2>&1
expect_status 88 $? "a Q36 constant set in env.sh has no effect on a real run"
# The run must have checked the pinned path against the real pins, and all four files. (Status 88
# alone would also come from a redirected check that fails for another reason.)
if grep -q 'Q36 PIN MISMATCH: config.json has missing, the pin is a822a9e48b0aafbe3144ec37d4fb067e178ed96615ce6e4420b3149893cc5767' "$TOP/m-real2-q36.log" \
   && grep -q 'Q36 PIN MISMATCH: chat_template.jinja has missing' "$TOP/m-real2-q36.log" \
   && grep -q 'the Q36 checkpoint /Users/dan/edge0/models/edge0-35b does not match' "$TOP/m-real2-q36.log" \
   && ! grep -q "$Q36CK" "$TOP/m-real2-q36.log"; then ok "env.sh did not change the pinned path, the pin file, or the file list"; else bad "env.sh changed a Q36 constant"; fi
# env.sh is sourced after the argument checks, so it gets the same refusals: port 8788 (the same
# exit code and message as the argv case; the check must not depend on laya3 listening there), an
# unknown engine, and no switch to rehearsal mode.
printf 'k\n' | env REHEARSE=1 PORT=8788 SERVER_CMD=true bash "$AH/mac-acc-a.sh" serve "$TOP/m8788b" "$REPO" > "$TOP/m-8788-argv.log" 2>&1
ARGV_8788_STATUS=$?
mkdir -p "$TOP/mreal3"; printf 'PORT=8788\n' > "$TOP/mreal3/env.sh"
printf 'k\n' | env PORT=$PORT_A bash "$AH/mac-acc-a.sh" serve "$TOP/mreal3" "$REPO" > "$TOP/m-env-8788.log" 2>&1
expect_status "$ARGV_8788_STATUS" $? "an env.sh that sets PORT=8788 is refused with the exit code of the argv case"
if [ "$(cat "$TOP/m-env-8788.log")" = "port 8788 belongs to laya3" ] && [ "$(cat "$TOP/m-8788-argv.log")" = "port 8788 belongs to laya3" ]; then ok "the env.sh refusal has the message of the argv case"; else bad "env.sh 8788 message"; fi
mkdir -p "$TOP/mreal4"; printf 'ENGINE=gpt\n' > "$TOP/mreal4/env.sh"
printf 'k\n' | env PORT=$PORT_A bash "$AH/mac-acc-a.sh" serve "$TOP/mreal4" "$REPO" > "$TOP/m-env-engine.log" 2>&1
expect_status 64 $? "an env.sh that sets an unknown ENGINE is refused"
mkdir -p "$TOP/mreal5"; printf 'REHEARSE=1\n' > "$TOP/mreal5/env.sh"
printf 'k\n' | env ENGINE=q36 PORT=$PORT_A SERVER_CMD=true Q36_DIR_OVERRIDE="$Q36CK" Q36_PINS_OVERRIDE="$TOP/q36-pins.test" bash "$AH/mac-acc-a.sh" serve "$TOP/mreal5" "$REPO" > "$TOP/m-env-rehearse.log" 2>&1
expect_status 88 $? "an env.sh that sets REHEARSE=1 does not enable the Q36 overrides"
env ACC_TEST_KEY=$TEST_KEY PORT=$PORT_A bash "$AH/run-acc-a.sh" "$TOP/a-engbad" --rehearse --engine gpt > "$TOP/engbad.log" 2>&1; expect_status 64 $? "an unknown engine is refused"
if [ -e "$TOP/a-engbad" ]; then bad "an unknown engine made files"; else ok "an unknown engine makes no files"; fi
# The client checks the id: a server that lists another id than the engine's gives a pipeline defect.
run_q36 q36id "$TOP/a-q36id" MODEL_ID=other; expect_status 2 $? "a server that lists another model id than the engine gives a pipeline defect"

# The bearer key must not appear in the argv of any process. Tools that could take it get shims that
# log their argv and run the real tool; `ps -eo args` is sampled during the run as well.
SH_DIR=$TOP/shims; mkdir -p "$SH_DIR"; ARGV_LOG=$TOP/argv.log; : > "$ARGV_LOG"
for tool in grep python3 curl sha256sum sed cut find date; do
  real=$(command -v $tool)
  printf '#!/bin/sh\nprintf "%%s\\n" "%s $*" >> "%s"\nexec "%s" "$@"\n' "$tool" "$ARGV_LOG" "$real" > "$SH_DIR/$tool"; chmod +x "$SH_DIR/$tool"
done
env PATH=$SH_DIR:$PATH ACC_TEST_KEY=$TEST_KEY PORT=$PORT_A bash "$AH/run-acc-a.sh" "$TOP/a-argv" --rehearse > "$TOP/a-argv.log" 2>&1 &
ARGV_RUN=$!
: > "$TOP/ps.log"
while kill -0 "$ARGV_RUN" 2>/dev/null; do ps -eo args >> "$TOP/ps.log"; done
wait "$ARGV_RUN"; expect_status 0 $? "ACC-A rehearsal with logged argv passes"
if [ -s "$ARGV_LOG" ] && grep -q 'grep -rqF -f' "$ARGV_LOG"; then ok "the argv log saw the key scan (grep -rqF -f, the pattern never in argv)"; else bad "the argv log is empty or has no key scan"; fi
if grep -qF -f - "$ARGV_LOG" "$TOP/ps.log" <<< "$TEST_KEY"; then bad "the bearer key appears in the argv of a process"; else ok "the bearer key is in no argv ($(wc -l < "$ARGV_LOG") logged calls, $(wc -l < "$TOP/ps.log") ps lines)"; fi

# The sandbox gate: no image, no run. Nothing starts, no directory is made, no Mac tool is reached.
env -u SANDBOX_IMAGE PORT=$PORT_A bash "$AH/run-acc-a.sh" "$TOP/a-nosb" --rehearse > "$TOP/a-nosb.log" 2>&1; expect_status 67 $? "ACC-A without SANDBOX_IMAGE stops"
if grep -q "needs the user's approval to pull" "$TOP/a-nosb.log"; then ok "the stop message names the approval"; else bad "stop message"; fi
if [ -e "$TOP/a-nosb" ]; then bad "ACC-A made an evidence directory without a sandbox"; else ok "ACC-A without a sandbox starts nothing"; fi
env PATH=$FB:$PATH SANDBOX_IMAGE=absent@sha256:0000000000000000000000000000000000000000000000000000000000000000 PORT=$PORT_A bash "$AH/run-acc-a.sh" "$TOP/a-absent" > "$TOP/a-absent.log" 2>&1; expect_status 67 $? "ACC-A with an absent image stops before the lock"
env FAKE_DOCKER_ROOTLESS=0 PORT=$PORT_A bash "$AH/run-acc-a.sh" "$TOP/a-rootful" --rehearse > "$TOP/a-rootful.log" 2>&1; expect_status 69 $? "ACC-A with a daemon that is not rootless stops"
if [ -e "$TOP/a-rootful" ]; then bad "ACC-A made files before it refused the daemon"; else ok "ACC-A refuses a daemon that is not rootless before it makes files"; fi
env -u SANDBOX_IMAGE PORT=$PORT_B bash "$AH/run-acc-b.sh" "$TOP/b-nosb" --rehearse > "$TOP/b-nosb.log" 2>&1; expect_status 67 $? "ACC-B without SANDBOX_IMAGE refuses"
if [ -e "$TOP/b-nosb" ] || pgrep -f "fake_server.py --port $PORT_B" > /dev/null; then bad "ACC-B started something without a sandbox"; else ok "ACC-B without a sandbox starts nothing"; fi
# The mount path guard (T33, C23): a fixture path with a colon or a comma is refused with 64
# before anything starts (the ACC-B pi rehearsal itself moved to selftest-accb.sh, which needs
# the real daemon and the pinned image).
mkdir -p "$TOP/fix:ture" "$TOP/fix,ture"
for bad_path in "$TOP/fix:ture" "$TOP/fix,ture"; do
  label=$(basename "$bad_path" | tr ':,' '--')
  ACC_TEST_KEY=$TEST_KEY PORT=$PORT_B bash "$AH/run-acc-b.sh" "$TOP/b-path-$label" --rehearse --fixture "$bad_path" > "$TOP/b-path-$label.log" 2>&1
  expect_status 64 $? "ACC-B refuses the fixture path ${bad_path##*/}"
done
# A newline in a path argument is refused as well: docker_run_args is split with mapfile on
# newlines, so a newline would break one argument into two.
nl_dir=$TOP/fix$'\n'ture
mkdir -p "$nl_dir"
ACC_TEST_KEY=$TEST_KEY PORT=$PORT_B bash "$AH/run-acc-b.sh" "$TOP/b-path-nl" --rehearse --fixture "$nl_dir" > "$TOP/b-path-nl.log" 2>&1
expect_status 64 $? "ACC-B refuses a fixture path with a newline"
ACCB_SCRATCH=$TOP/scratch$'\n'dir ACC_TEST_KEY=$TEST_KEY PORT=$PORT_B bash "$AH/run-acc-b.sh" "$TOP/b-scratch-nl" --rehearse > "$TOP/b-scratch-nl.log" 2>&1
expect_status 64 $? "ACC-B refuses an ACCB_SCRATCH with a newline"
if [ -e "$TOP/ssh-called" ] || [ -e "$TOP/flock-called" ] || [ -n "$(ls -d "$TOP"/b-path-*/ 2>/dev/null)" ]; then bad "a bad mount path started something or made files"; else ok "a bad mount path starts nothing and makes no files"; fi
# The empty-key guard (K0): KEY= stops with the K0 error and never runs grep with an empty pattern.
env ACC_TEST_KEY= PORT=$PORT_B bash "$AH/run-acc-b.sh" "$TOP/b-nokey" --rehearse > "$TOP/b-nokey.log" 2>&1; expect_status 64 $? "ACC-B with an empty key refuses (K0)"
if grep -q "the bearer key is empty" "$TOP/b-nokey.log"; then ok "the K0 message names the empty key"; else bad "no K0 message: $(head -2 "$TOP/b-nokey.log")"; fi
if [ -e "$TOP/b-nokey" ]; then bad "ACC-B made files with an empty key"; else ok "an empty key starts nothing"; fi

# The local port must be free in every mode: a listener there gives status 70 before ssh or flock.
python3 -I -c "import socket, time; s = socket.socket(); s.bind(('127.0.0.1', $PORT_A)); s.listen(); time.sleep(60)" &
LISTENER=$!
sleep 1
PATH=$FB:$PATH PORT=$PORT_A bash "$AH/run-acc-a.sh" "$TOP/a-busy" > "$TOP/a-busy.log" 2>&1; expect_status 70 $? "ACC-A with a busy local port stops (real mode)"
if [ -e "$TOP/ssh-called" ] || [ -e "$TOP/flock-called" ] || [ -e "$TOP/a-busy" ]; then bad "a busy port did not stop the launcher before ssh, flock, or any file"; else ok "a busy port stops the launcher before ssh, flock, or any file"; fi
kill "$LISTENER" 2>/dev/null; wait "$LISTENER" 2>/dev/null

# An abort leaves nothing behind: no server, no session, no pi. The launcher gets SIGTERM.
abort_run() { # <label> <script> <out> <ready test> [env assignments...]
  local label=$1 script=$2 out=$3 ready=$4; shift 4
  env "$@" ACC_TEST_KEY=$TEST_KEY PATH=$FB:$PATH bash "$AH/$script" "$out" --rehearse > "$TOP/$label.log" 2>&1 &
  local pid=$! n=0
  until eval "$ready" || [ $n -ge 120 ]; do sleep 1; n=$((n + 1)); done
  if ! eval "$ready"; then bad "$label: the run did not reach the abort point"; fi
  kill -TERM "$pid" 2>/dev/null
  n=0; while kill -0 "$pid" 2>/dev/null && [ $n -lt 120 ]; do sleep 1; n=$((n + 1)); done
  if kill -0 "$pid" 2>/dev/null; then bad "$label: the launcher did not end after SIGTERM"; kill -KILL "$pid" 2>/dev/null; fi
  wait "$pid" 2>/dev/null; expect_status 143 $? "$label ends with the signal status"
  leftovers "$label"
}
abort_run "abort ACC-A during the load" run-acc-a.sh "$TOP/a-abort" 'pgrep -f "fake_server.py --port $PORT_A" > /dev/null' PORT=$PORT_A FAKE_READY_DELAY=600

# The key must never reach the disk. The scripts remove the evidence if it does; this checks the scratch too.
# The ACC-B pi rehearsal and its mutations run in selftest-accb.sh on pinu (they need the real
# rootless daemon and the pinned image).
# The CUDA launcher mode and the kubdan serve script: selftest-cuda.sh is sourced here, so it shares
# the counters, the scratch $TOP, and the final key scan below. Its fake servers bind ports above
# 18800 (SELFTEST_PORT_C, SELFTEST_PORT_D).
source "$HERE/selftest-cuda.sh" "$HERE" "$TOP"
if grep -rqF -f - "$TOP" 2>/dev/null <<< "$TEST_KEY"; then bad "the key is on the disk under $TOP"; else ok "the key is not on the disk"; fi
echo "selftest scratch: $TOP"
if [ "$FAILS" = 0 ]; then echo "selftest: all checks passed"; exit 0; fi
echo "selftest: $FAILS checks failed"; exit 1
