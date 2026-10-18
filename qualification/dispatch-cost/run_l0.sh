#!/bin/sh
# L0 baseline runner. Usage: run_l0.sh OUT_DIR RUNS [REPS] [WARMUP]
# Run from the repository root on the Mac with the build environment active.
# Exit status: 0 PASS, 1 OPEN, 2 invalid input or usage error, 3 NONSTANDARD, 4 a setup or bookkeeping failure outside the gate, 5 hash failure after a PASS.
# Only the gate gives 0 to 3.
set -eu
( set -o pipefail ) 2>/dev/null && set -o pipefail
# Any stop before the final status lines that is not a usage error exits 4, whatever the failed command returned.
stage=pre; usage_error=0
trap 'rc=$?; if [ "$stage" = pre ] && [ "$rc" -ne 0 ] && [ "$usage_error" -eq 0 ]; then exit 4; fi' EXIT
if [ "$#" -lt 2 ]; then
  echo "usage: run_l0.sh OUT_DIR RUNS [REPS] [WARMUP]" >&2; usage_error=1; exit 2
fi
out=$1; runs=$2; reps=${3:-100}; warmup=${4:-10}
mkdir -p "$out"
[ -z "$(ls -A "$out")" ] || { echo "output directory is not empty" >&2; usage_error=1; exit 2; }
cargo build --release --locked --offline --manifest-path qualification/dispatch-cost/Cargo.toml > "$out/build.log" 2>&1 \
  || { echo "build failed" >&2; exit 4; }
bin=${CARGO_TARGET_DIR:-qualification/dispatch-cost/target}/release/dispatch-cost-qualification
snap() {
  echo "== $1 $(date -u +%FT%TZ)"; uptime
  ioreg -r -d 1 -w0 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' || echo "gpu utilization unavailable"
  # sed reads all input, so ps gets no SIGPIPE under pipefail.
  ps -axo pid,pcpu,etime,command -r | sed -n 1,8p | cut -c1-140
  pgrep -fl 8788 || echo "no port 8788 process"
  lsof -nP -iTCP:8788 -sTCP:LISTEN 2>/dev/null || true
  # Capture first, so a failed pmset fails the snapshot under every shell.
  therm=$(pmset -g therm 2>&1) || return 1
  echo "$therm" | sed -n 1,4p
}
# Check each manifest value here. A failure inside a command substitution does not stop the shell.
m_commit=$(git rev-parse HEAD) || exit 4
m_status=$(git status --porcelain) || exit 4
if [ -z "$m_status" ]; then m_dirty=0; else m_dirty=$(printf '%s\n' "$m_status" | wc -l | tr -d ' '); fi
m_cpu=$(sysctl -n machdep.cpu.brand_string) || exit 4
m_os=$(sw_vers -productVersion) || exit 4
m_mem=$(sysctl -n hw.memsize) || exit 4
m_rustc=$(rustc --version) || exit 4
m_lock=$(grep -A2 'name = "mlx-sys"' engine/Cargo.lock) || exit 4
m_hash=$(shasum -a 256 "$bin") || exit 4
m_hash=${m_hash%% *}
for v in "$m_commit" "$m_cpu" "$m_os" "$m_mem" "$m_rustc" "$m_lock" "$m_hash"; do
  [ -n "$v" ] || { echo "empty manifest value" >&2; exit 4; }
done
m_tree=$(git rev-parse HEAD:engine/vendor/mlx-sys 2>/dev/null) || m_tree=unavailable
{
  echo "commit $m_commit"; echo "dirty $m_dirty"
  echo "machine $m_cpu $m_os mem $m_mem"
  echo "rustc $m_rustc"
  echo "$m_lock" | tr '\n' ' '; echo
  echo "vendor-mlx-sys-tree $m_tree"
  echo "settle_seconds ${SETTLE_SECONDS:-90}"
  echo "command $bin --layers 1,8 --reps $reps --warmup $warmup --output RUN.json (x$runs)"
  echo "binary_sha256 $m_hash"
  snap start
} > "$out/manifest.txt" 2>&1
# Let the 1 minute load average settle after the build.
sleep "${SETTLE_SECONDS:-90}"
i=1
while [ "$i" -le "$runs" ]; do
  snap "before-run-$i" >> "$out/host.log" 2>&1
  "$bin" --layers 1,8 --reps "$reps" --warmup "$warmup" --output "$out/rust-$i.json" > /dev/null 2> "$out/rust-$i.err" \
    || { echo "harness run $i failed" >&2; exit 4; }
  snap "after-run-$i" >> "$out/host.log" 2>&1
  i=$((i+1))
done
snap end >> "$out/manifest.txt" 2>&1
gate_rc=0
python3 qualification/dispatch-cost/l0_gate.py "$out"/rust-*.json --output "$out/gate.json" > "$out/gate.txt" 2>&1 || gate_rc=$?
cat "$out/gate.txt"
echo "gate_exit_status $gate_rc" >> "$out/gate.txt"
# Hash only the files that exist. The gate writes no gate.json for invalid input.
: > "$out/sha256.txt"
hash_rc=0
for f in "$out"/build.log "$out"/rust-*.json "$out"/*.err "$out"/gate.json "$out"/gate.txt "$out"/host.log "$out"/manifest.txt; do
  if [ -f "$f" ]; then shasum -a 256 "$f" >> "$out/sha256.txt" || { hash_rc=1; echo "hash failed: $f" >&2; }; fi
done
# sha256.txt holds paths as given, with the "$out" prefix. Run `shasum -c` from the directory of the run.
stage=post
[ "$gate_rc" -eq 0 ] && [ "$hash_rc" -ne 0 ] && exit 5
exit "$gate_rc"
