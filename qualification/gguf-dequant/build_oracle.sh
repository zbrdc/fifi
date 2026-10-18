#!/bin/sh
# Build the oracle from the pinned llama.cpp files and write fixtures.
# Usage: build_oracle.sh <oracle-src-dir> <work-dir> <seed> [<real-dir>]
# <oracle-src-dir> holds ggml/src/ggml-quants.c and the headers at the pinned commit.
# <real-dir> (optional) holds real-<name>.blocks files from extract_real.py. The
# script converts each one with the nofma oracle into real-<name>.f32 there.
set -eu
SRC=$1; WORK=$2; SEED=$3; REAL=${4:-}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WORK/fixtures-nofma" "$WORK/fixtures-fma"
: > "$WORK/build-info.txt"
{ clang --version; echo "target: $(clang -dumpmachine)"; } >> "$WORK/build-info.txt"
for mode in nofma fma; do
  if [ $mode = nofma ]; then FLAG=-ffp-contract=off; else FLAG=-ffp-contract=on; fi
  CMD="clang -O2 $FLAG -I SRC/ggml/src -I SRC/ggml/include oracle.c ggml-quants.c -o oracle-$mode -lm"
  echo "build $mode: $CMD" >> "$WORK/build-info.txt"
  clang -O2 $FLAG -I"$SRC/ggml/src" -I"$SRC/ggml/include" \
    "$HERE/oracle.c" "$SRC/ggml/src/ggml-quants.c" -o "$WORK/oracle-$mode" -lm
  "$WORK/oracle-$mode" "$WORK/fixtures-$mode" "$SEED" > "$WORK/oracle-$mode.log"
done
if [ -n "$REAL" ]; then
  for f in "$REAL"/real-*.blocks; do
    n=$(basename "$f" .blocks); n=${n#real-}
    "$WORK/oracle-nofma" --convert "$n" "$f" "$REAL/real-$n.f32"
    "$WORK/oracle-fma" --convert "$n" "$f" "$WORK/real-$n.fma.f32"
  done
fi
