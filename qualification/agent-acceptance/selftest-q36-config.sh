#!/usr/bin/env bash
# Focused self-test of the Q36 config that mac-acc-a.sh serves. It runs mac-acc-a.sh directly in
# rehearsal and real mode against a scratch source tree. The server command is `true`, so the run
# stops with status 81 after it wrote evidence/config.json and evidence/identity.txt. No Mac, no
# server, no model. mutate-q36-config.sh runs this script against mutated copies of mac-acc-a.sh.
# Usage: selftest-q36-config.sh [agent-acceptance dir under test] [scratch dir]
# Exit status: 0 when every check passes, 1 otherwise.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AHD=${1:-$HERE}
TOP=${2:-$(mktemp -d)}
FAILS=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; FAILS=$((FAILS + 1)); }
expect_status() { if [ "$1" = "$2" ]; then ok "$3 (status $2)"; else bad "$3: want status $1, got $2"; fi; }
SHIPPED=$HERE/../../engine/config/qwen36-mac-studio-36gb.json
PORT=${SELFTEST_PORT_A:-18797}
mkdir -p "$TOP"

# mk_tree <name> [shipped file]: a source tree with the script under test and a shipped config.
mk_tree() {
  mkdir -p "$TOP/$1/qualification" "$TOP/$1/engine/config"
  ln -sfn "$AHD" "$TOP/$1/qualification/agent-acceptance"
  cp "${2:-$SHIPPED}" "$TOP/$1/engine/config/qwen36-mac-studio-36gb.json"
}
CK=$TOP/ck; mkdir -p "$CK"
for f in config.json model.safetensors.index.json tokenizer.json chat_template.jinja; do echo "scratch $f" > "$CK/$f"; done
(cd "$CK" && sha256sum config.json model.safetensors.index.json tokenizer.json chat_template.jinja) > "$TOP/pins"
# serve <name> <scratch> [env...]: a rehearsal serve on tree <name>.
serve() {
  local name=$1 ms=$2; shift 2
  printf 'k\n' | env "$@" REHEARSE=1 ENGINE=q36 PORT=$PORT SERVER_CMD=true Q36_DIR_OVERRIDE="$CK" Q36_PINS_OVERRIDE="$TOP/pins" \
    bash "$AHD/mac-acc-a.sh" serve "$ms" "$TOP/$name" > "$ms.log" 2>&1
}

mk_tree t1
serve t1 "$TOP/m1"; expect_status 81 $? "a rehearsal reaches the server start with the shipped config"
python3 -I - "$TOP/m1/evidence/config.json" "$SHIPPED" "$CK" <<'PY' && ok "the served qwen36 object is the shipped object with its eight keys and no prefill_chunk" || bad "served qwen36 object"
import json, sys
served = json.load(open(sys.argv[1]))
shipped = json.load(open(sys.argv[2]))["engines"][0]["qwen36"]
(engine,) = served["engines"]
assert engine["qwen36"] == shipped, engine["qwen36"]
assert engine["qwen36"] == {"cache_limit_bytes": 4294967296, "context_limit": 262144,
                            "session_cap_bytes": 4294967296, "prefill_budget_bytes": 32212254720,
                            "prefill_session_reserve_bytes": 8286699520,
                            "activation_dtype": "bf16", "kv_dtype": "q8",
                            "compile": True}, engine["qwen36"]
assert "prefill_chunk" not in engine["qwen36"]
assert (engine["id"], engine["family"], engine["model"]) == ("q36", "qwen3_6_35b_a3b", "fifi-q36")
assert engine["model_dir"] == engine["tokenizer_dir"] == sys.argv[3]
assert served["api_key_env"] == "ENGINESRV_API_KEY" and served["listen"] == "127.0.0.1:%s" % __import__("os").environ.get("SELFTEST_PORT_A", "18797")
PY
want=$(sha256sum "$SHIPPED" | cut -d' ' -f1)
if grep -qx "shipped_config_sha256 $want" "$TOP/m1/evidence/identity.txt" && grep -qx 'cache_limit_bytes 4294967296' "$TOP/m1/evidence/identity.txt"; then ok "identity.txt records the sha256 of the shipped file"; else bad "shipped sha256 record"; fi

# The object comes from the tree this run builds: another file in another tree gives another object.
python3 -I - "$SHIPPED" "$TOP/other.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["engines"][0]["qwen36"]["prefill_budget_bytes"] = 123456789
json.dump(c, open(sys.argv[2], "w"))
PY
mk_tree t2 "$TOP/other.json"
serve t2 "$TOP/m2"; expect_status 81 $? "a rehearsal on a tree with another shipped file"
if grep -q '"prefill_budget_bytes":123456789' "$TOP/m2/evidence/config.json" && grep -qx "shipped_config_sha256 $(sha256sum "$TOP/other.json" | cut -d' ' -f1)" "$TOP/m2/evidence/identity.txt"; then ok "the object and the hash come from the file in the source tree"; else bad "object from the source tree"; fi

# A shipped file with prefill_chunk is refused: the engine default must apply.
python3 -I - "$SHIPPED" "$TOP/chunk.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["engines"][0]["qwen36"]["prefill_chunk"] = 64
json.dump(c, open(sys.argv[2], "w"))
PY
mk_tree t3 "$TOP/chunk.json"
serve t3 "$TOP/m3"; expect_status 65 $? "a shipped file with prefill_chunk is refused"

# A shipped file without one of the three keys is refused.
python3 -I - "$SHIPPED" "$TOP/nobudget.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); del c["engines"][0]["qwen36"]["prefill_budget_bytes"]
json.dump(c, open(sys.argv[2], "w"))
PY
mk_tree t4 "$TOP/nobudget.json"
serve t4 "$TOP/m8"; expect_status 65 $? "a shipped file without prefill_budget_bytes is refused"

# Q36_CACHE_LIMIT cannot change the served object: another value is refused, the same value is not.
serve t1 "$TOP/m4" Q36_CACHE_LIMIT=1073741824; expect_status 64 $? "a rehearsal with another Q36_CACHE_LIMIT is refused"
if grep -q 'Q36_CACHE_LIMIT=1073741824 differs' "$TOP/m4.log" && [ ! -e "$TOP/m4/evidence/config.json" ]; then ok "the refusal names the variable and writes no config"; else bad "refusal message"; fi
serve t1 "$TOP/m5" Q36_CACHE_LIMIT=4294967296; expect_status 81 $? "the same Q36_CACHE_LIMIT as the file is accepted"
mkdir -p "$TOP/m6"; printf 'Q36_CACHE_LIMIT=1073741824\n' > "$TOP/m6/env.sh"
printf 'k\n' | env ENGINE=q36 PORT=$PORT bash "$AHD/mac-acc-a.sh" serve "$TOP/m6" "$TOP/t1" > "$TOP/m6.log" 2>&1
expect_status 64 $? "an env.sh that sets another Q36_CACHE_LIMIT is refused"
mkdir -p "$TOP/m7"; printf 'Q36_CACHE_LIMIT=4294967296\n' > "$TOP/m7/env.sh"
printf 'k\n' | env ENGINE=q36 PORT=$PORT bash "$AHD/mac-acc-a.sh" serve "$TOP/m7" "$TOP/t1" > "$TOP/m7.log" 2>&1
expect_status 88 $? "an env.sh with the file value passes the config step (the pin check stops it here)"

echo
if [ "$FAILS" = 0 ]; then echo "q36 config selftest: all checks passed"; exit 0; fi
echo "q36 config selftest: $FAILS check(s) failed"; exit 1
