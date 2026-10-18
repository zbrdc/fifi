#!/bin/sh
# Zero-copy import evidence: build and run the native-import harness on the Mac.
# Usage: run-import-evidence.sh WORKTREE OUTDIR
set -eu

ACTIVATE=/Users/dan/fifi/scrch/9071b86/p0a04-native-gate/environment/activate.sh
MLX_SOURCE=/Users/dan/fifi/scrch/9071b86/p0a04-native-gate/environment/sources/mlx
MLX_COMMIT=64ea011cb65f14d9ce2737e60db9a4ae91ed7441
TASK_ROOT=/Users/dan/fifi/scrch/52640f0/p0a05-reserve-baseline
MANIFEST=qualification/native-import/Cargo.toml

fail() {
    echo "run-import-evidence: $1" >&2
    exit 1
}

[ "$#" -eq 2 ] || fail "usage: $0 WORKTREE OUTDIR"
WORKTREE=$1
OUTDIR=$2
TARGET_DIR=$TASK_ROOT/target
TMP_DIR=$TASK_ROOT/tmp
BIN=$TARGET_DIR/release/native-import-qualification

[ -f "$ACTIVATE" ] || fail "missing pinned activation $ACTIVATE"
[ -d "$MLX_SOURCE" ] || fail "missing pinned MLX source $MLX_SOURCE"
[ -f "$WORKTREE/$MANIFEST" ] || fail "missing $MANIFEST under $WORKTREE"

. "$ACTIVATE"

[ "$(git -C "$MLX_SOURCE" rev-parse HEAD)" = "$MLX_COMMIT" ] || fail "MLX source is not at $MLX_COMMIT"
[ -z "$(git -C "$MLX_SOURCE" status --porcelain)" ] || fail "pinned MLX source tree is dirty"
[ -z "$(git -C "$WORKTREE" status --porcelain --untracked-files=no)" ] || fail "worktree is dirty"

mkdir -p "$OUTDIR" "$TARGET_DIR" "$TMP_DIR"
export CARGO_TARGET_DIR=$TARGET_DIR
export TMPDIR=$TMP_DIR
export TMP=$TMP_DIR
export TEMP=$TMP_DIR
export INFERENCE_MLX_SOURCE=$MLX_SOURCE

git -C "$WORKTREE" rev-parse HEAD > "$OUTDIR/source-commit.txt"
{
    echo "worktree $WORKTREE"
    echo "source_commit $(cat "$OUTDIR/source-commit.txt")"
    echo "mlx_source $MLX_SOURCE"
    echo "mlx_commit $(git -C "$MLX_SOURCE" rev-parse HEAD)"
    echo "runner $(id -un)@$(hostname)"
    echo "started $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} > "$OUTDIR/env.log"

snapshot_mem() {
    {
        sysctl vm.swapusage
        vm_stat
        memory_pressure
    } > "$1" 2>&1
}

snapshot_mem "$OUTDIR/mem-host-before.log"

if ! /usr/bin/time -l cargo build --offline --locked --release \
    --manifest-path "$WORKTREE/$MANIFEST" > "$OUTDIR/build.log" 2>&1; then
    fail "release build failed; see $OUTDIR/build.log"
fi

[ -x "$BIN" ] || fail "build did not produce $BIN"

check_case() {
    log=$1
    bytes=$2
    grep -q "\"stage\":\"identity\",\"bytes\":$bytes," "$log" ||
        fail "missing identity line for $bytes bytes in $log"
    for stage in imported copy_check gpu_complete host_copy content_check arrays_freed released settled; do
        grep -q "\"stage\":\"$stage\"" "$log" ||
            fail "missing $stage stage in $log"
    done
}

run_case() {
    label=$1
    bytes=$2
    log=$OUTDIR/import-$bytes-$label.log
    snapshot_mem "$OUTDIR/mem-before-$bytes-$label.log"
    if ! /usr/bin/time -l "$BIN" "$bytes" > "$log" 2>&1; then
        fail "case $label ($bytes bytes) failed; see $log"
    fi
    snapshot_mem "$OUTDIR/mem-after-$bytes-$label.log"
    check_case "$log" "$bytes"
}

run_case preflight 67108864
run_case main-1 1073741824
run_case main-2 1073741824
run_case main-3 1073741824

snapshot_mem "$OUTDIR/mem-host-after.log"
shasum -a 256 "$BIN" > "$OUTDIR/binary.sha256"

cd "$OUTDIR"
shasum -a 256 env.log source-commit.txt build.log binary.sha256 \
    mem-host-before.log mem-host-after.log \
    import-*.log mem-before-*.log mem-after-*.log > import-evidence.sha256

echo "evidence written to $OUTDIR"
echo "fill import-evidence-record.md from the logs and import-evidence.sha256"
