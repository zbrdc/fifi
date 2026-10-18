#!/usr/bin/env bash
# Runs INSIDE a gpu_run.sh hold (host process, NOT in a container; the container rules of RD-9a do not apply to it).
# Usage: hold.sh WANT_EXE_SHA256 BINARY WATCH_FILE ARGS...   (ARGS are the arguments of `cuda-rd-run run`, including --out FILE)
# Order: idle check -> executable sha256 equals the one the release check examined -> run with an inner deadline while the
# card is sampled every 0.5 s into WATCH_FILE (kept; the evaluator judges it against the timed regions) -> post record.
# Exits: the binary's own code (0 ok, 2 STOP, 3 REFUSED), 4 not idle, 6 wrong executable.
set -u
start=$(date +%s)
F="$(cd "$(dirname "$0")" && pwd)"
want="$1"; bin="$2"; watch="$3"; shift 3
rc=0; python3 -I "$F/idle_check.py" pre || rc=$?
case "$rc" in
  0) ;;
  2) echo "STOP: the idle check could not run (nvidia-smi failed)"; exit 2 ;;
  *) exit 4 ;;
esac
got="$(sha256sum "$bin" | cut -d' ' -f1)"
echo "HOLD exe_sha256 $got"
[ "$got" = "$want" ] || { echo "STOP: the executable is not the one the release check examined"; exit 6; }
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
[ -n "$gpu" ] && [ -n "$driver" ] || { echo "STOP: nvidia-smi gave no GPU name or driver version"; exit 2; }
"$bin" "$@" --gpu-name "$gpu" --driver "$driver" --deadline-secs 270 &
pid=$!
python3 -I "$F/idle_check.py" watch "$watch" "$pid" &
wpid=$!
wait "$pid"; rc=$?
wait "$wpid"
python3 -I "$F/idle_check.py" post
echo "HOLD seconds $(( $(date +%s) - start ))"
exit "$rc"
