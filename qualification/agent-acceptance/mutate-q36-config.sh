#!/usr/bin/env bash
# Mutation check of selftest-q36-config.sh. Each mutation breaks one rule of the Q36 config step in
# a scratch copy of this directory. The self-test must fail on every copy. A mutation that the
# self-test does not kill is a gap in the test.
# Usage: mutate-q36-config.sh [scratch dir]    Exit status: 0 when every mutation is killed.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOP=${1:-$(mktemp -d)}
mkdir -p "$TOP"
SURVIVED=0
# mutate <name> <sed program>: copy, patch, check that the patch changed the file, run the self-test.
mutate() {
  local name=$1 prog=$2 dir=$TOP/$1/agent-acceptance
  mkdir -p "$TOP/$name"; rm -rf "$dir"; mkdir -p "$dir"; cp -r "$HERE"/. "$dir"/; rm -f "$dir/selftest.sh"  # selftest.sh holds the test key literal
  if [ -L "$dir" ] || [ "$dir/mac-acc-a.sh" -ef "$HERE/mac-acc-a.sh" ]; then echo "ERROR $name: the copy aliases the original"; exit 2; fi
  sed -e "$prog" "$HERE/mac-acc-a.sh" > "$dir/mac-acc-a.sh"
  if cmp -s "$HERE/mac-acc-a.sh" "$dir/mac-acc-a.sh"; then echo "ERROR $name: the patch changed nothing"; SURVIVED=$((SURVIVED + 1)); return; fi
  bash "$HERE/selftest-q36-config.sh" "$dir" "$TOP/$name/run" > "$TOP/$name/out.log" 2>&1
  local status=$?
  if [ "$status" = 0 ]; then
    echo "SURVIVED $name"; SURVIVED=$((SURVIVED + 1))
  elif ! grep -q '^q36 config selftest: .* failed' "$TOP/$name/out.log"; then
    echo "ERROR $name: the self-test did not run to its summary"; SURVIVED=$((SURVIVED + 1))
  else
    echo "killed   $name ($(grep -c '^FAIL' "$TOP/$name/out.log") failed checks)"
  fi
}
mutate only-cache-limit 's|"qwen36":\$Q36_OBJECT|"qwen36":{"cache_limit_bytes":$Q36_CACHE}|'
mutate host-copy 's|^Q36_SHIPPED=.*|Q36_SHIPPED=$(cd "$(dirname "$0")/../.." \&\& pwd -P)/engine/config/qwen36-mac-studio-36gb.json|'
mutate env-override 's|\[ -n "\${Q36_CACHE_LIMIT:-}" \] \&\& \[ "\$Q36_CACHE_LIMIT" != "\$Q36_CACHE" \]|false|'
mutate env-override-wins 's|^  Q36_OBJECT=\$(printf.*|&; [ -n "${Q36_CACHE_LIMIT:-}" ] \&\& Q36_OBJECT=$(printf "%s" "$Q36_OBJECT" \| sed "s/\\"cache_limit_bytes\\":[0-9]*/\\"cache_limit_bytes\\":$Q36_CACHE_LIMIT/") \&\& Q36_CACHE=$Q36_CACHE_LIMIT|; s|^  if \[ -n "\${Q36_CACHE_LIMIT:-}" \].*|  if false; then|'
mutate accept-prefill-chunk '/assert "prefill_chunk" not in obj/d'
mutate no-sha-record '/echo "shipped_config_sha256/d'
mutate no-budget-check 's|"prefill_budget_bytes", ||'
echo
if [ "$SURVIVED" = 0 ]; then echo "every mutation was killed"; exit 0; fi
echo "$SURVIVED mutation(s) survived"; exit 1
