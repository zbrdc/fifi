#!/bin/bash
# Self-check for run.sh: SIGTERM during a run must leave no enginesrv process behind.
# Usage: [RUN_SH=PATH-TO-A-RUN-SCRIPT] selftest.sh [PATH-TO-ENGINESRV]
# The check uses its own TMPDIR below $TMPDIR, so it sees only the server of this run.
# Exit codes: 0 the server stopped, 1 the server survived or the run was not interrupted, 2 setup failed.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${RUN_SH:-$HERE/run.sh}"
BIN="${1:-${ENGINESRV:-}}"
if [ -z "$BIN" ] || [ ! -x "$BIN" ] || [ -z "${TMPDIR:-}" ]; then
  echo "setup: pass the enginesrv binary and set TMPDIR to the task scratch directory" >&2
  exit 2
fi
SUB="$(mktemp -d "$TMPDIR/selftest.XXXXXX")" || { echo "setup: mktemp failed" >&2; exit 2; }
trap 'pkill -KILL -f "$SUB/" 2> /dev/null; rm -rf "$SUB"' EXIT
TMPDIR="$SUB" "$RUN" --allow-sdk-skip "$BIN" > "$SUB/run.out" 2>&1 &
RUNPID=$!
# Wait until the server listens, then send SIGTERM to the script while the cases run.
for _ in $(seq 1 500); do
  grep -qs '^enginesrv listening on ' "$SUB"/server-fixtures.*/stdout.txt && break
  sleep 0.01
done
kill -TERM "$RUNPID" 2> /dev/null
wait "$RUNPID"
STATUS=$?
sleep 0.5
if [ "$STATUS" -eq 0 ] || [ "$STATUS" -eq 1 ]; then
  echo "FAIL selftest: run.sh finished before the signal (exit $STATUS)"
  exit 1
fi
if pgrep -f "$SUB/" > /dev/null; then
  echo "FAIL selftest: enginesrv still runs after SIGTERM (run.sh exit $STATUS)"
  exit 1
fi
echo "PASS selftest: run.sh exit $STATUS and no enginesrv remains"
