#!/bin/sh
# HumanEval baseline: generate and grade on the Mac.
# Usage: run-baseline.sh MODEL_DIR PROBLEMS OUTDIR
set -eu

if [ "$#" -ne 3 ]; then
    echo "usage: $0 MODEL_DIR PROBLEMS OUTDIR" >&2
    exit 2
fi

if [ -z "${TMPDIR:-}" ]; then
    echo "set TMPDIR to the task scratch directory" >&2
    exit 2
fi
export TMPDIR

MODEL_DIR=$1
PROBLEMS=$2
OUTDIR=$3
PY=/Users/dan/edge0/.venv/bin/python

HERE=$(cd "$(dirname "$0")" && pwd)

lines=$(wc -l < "$PROBLEMS" | tr -d ' ')
if [ "$lines" -ne 164 ]; then
    echo "problems file has $lines lines, expected 164: $PROBLEMS" >&2
    exit 1
fi

mkdir -p "$OUTDIR"

"$PY" "$HERE/he_run_mlx.py" "$MODEL_DIR" "$OUTDIR/he_responses.jsonl" "$PROBLEMS" > "$OUTDIR/he_run.log" 2>&1
"$PY" "$HERE/he_grade.py" "$OUTDIR/he_responses.jsonl" baseline "$PROBLEMS" > "$OUTDIR/he_grade.log" 2>&1
cat "$OUTDIR/he_grade.log"
