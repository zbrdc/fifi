#!/usr/bin/env bash
# RD-10b run script. One subcommand per step. Every path is spelled out; no shell state is assumed between calls.
# Usage: run_stage1.sh <reset|clean-tree|deps|gguf|build|build-tests|check-build|selftest|public-items|idle|call NAME|check-raw NAME|skip NAME...|wrapper|report|cleanup>
# Environment (in front of each command): RD10_WT (worktree), RD10_DATE (YYYY-MM-DD), RD10_S (scratch, default below),
# RD10_HEAD (commit under test), RD10_CAPTURE (graph|eager; eager only when RD-11a is absent).
# GPU holds are HOST processes started by gpu_run.sh (flock + gnutimeout -s KILL); they do NOT run in a container.
# Only the `wrapper`-free CPU steps use docker (run-cpu.sh); the image is inference-cuda-build:cuda12.2-ubuntu22.04.
set -euo pipefail
S="${RD10_S:-/media/dan/nvme/fifi/scrch/c62fd02/tasks/rd-10}"
F="$(cd "$(dirname "$0")" && pwd)"
D=/media/dan/nvme/fifi/scrch/c62fd02/research/fifi/packets/RD-9a-files/docker
BATON=/media/dan/nvme/fifi/scrch/c62fd02/gpu-baton/gpu_run.sh
BDIR=/media/dan/nvme/fifi/scrch/c62fd02/gpu-baton
IMAGE=inference-cuda-build:cuda12.2-ubuntu22.04
GG=/media/dan/nvme/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q4_K_M.gguf
WT="${RD10_WT:?set RD10_WT}"
QUAL="$WT/qualification/cuda-rd"
BIN="$S/target/release/cuda-rd-run"      # cuda-check.sh forces CARGO_TARGET_DIR=$RD9A_SCRATCH/target inside the container
export TMPDIR="$S/tmp" TMP="$S/tmp" TEMP="$S/tmp"
mkdir -p "$S/raw" "$S/logs" "$S/tmp"
cmd="${1:?subcommand}"; shift || true

batch_ok() {   # between batches: no foreign baton, no user hold
  if compgen -G "$BDIR/*.wanted" >/dev/null || compgen -G "$BDIR/user.granted" >/dev/null; then
    echo "STOP: a baton (*.wanted) or user.granted is present: wait or report"; ls "$BDIR"; return 2; fi
}
cleanup_containers() {   # after any kill: own leftovers only (names rd9a-*), never someone else's
  docker ps --filter ancestor="$IMAGE" --format '{{.ID}} {{.Names}}' | while read -r id name; do
    case "$name" in rd9a-*) docker rm -f "$id" ;; *) echo "not mine, left alone: $name" ;; esac; done
}
case "$cmd" in
  deps)    # RD10_CAPTURE=eager (the labelled EAGER PRE-RUN): the 17 names other than RD-11a; graph mode: all 18
    python3 -I "$F/check_deps.py" "$(git -C "$WT" rev-parse --git-common-dir)/.." "${RD10_HEAD:?}" "$S/deps.txt" ${RD10_CAPTURE:+$([ "$RD10_CAPTURE" = eager ] && echo --eager)} ;;
  gguf)    # CPU, about a minute: size and sha256 of the 20 GB file against the frozen values
    python3 -I "$F/check_gguf.py" "$GG" | tee "$S/gguf.json" ;;
  build)   # CPU only, release, CUDA 12.2 image. The qualification crate has no [features]: the dependency enables native-cuda.
    bash "$F/run_stage1.sh" clean-tree
    RD9A_ENGINE="$WT/engine" RD9A_SCRATCH="$S" bash "$D/run-cpu.sh" qual-build \
      env CUDA_RD_COMMIT="$(git -C "$WT" rev-parse HEAD)" cargo build --release --offline --locked --manifest-path "$QUAL/Cargo.toml" ;;   # 16c F1: the image has no git; run-cpu.sh forwards only its listed vars, so the commit rides in the command string
  build-tests)   # CPU: compile the engine lib tests (RD-2e's are lib modules) outside any hold; the log names the executable
    RD9A_ENGINE="$WT/engine" RD9A_SCRATCH="$S" bash "$D/run-cpu.sh" wrapper-build \
      cargo test --release --offline --locked --features native-cuda --no-run --message-format=json
    python3 -I "$F/find_test_exes.py" "$S/logs/wrapper-build.log" | tee "$S/wrapper-exes.txt" ;;
  check-build)
    test -x "$BIN" || { echo "STOP: $BIN missing or not executable"; exit 2; }
    python3 -I "$F/check_release_build.py" "$BIN" --out "$S/release.json"; sha256sum "$BIN"
    commit="$(python3 -I -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_info"]["commit"])' "$S/release.json")"
    [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo "STOP: the binary records commit '$commit' (not a full 40-hex commit): build with run_stage1.sh build, which passes CUDA_RD_COMMIT (the image has no git)"; exit 2; }
    [ "$commit" = "${RD10_HEAD:?set RD10_HEAD}" ] || { echo "STOP: the binary records commit $commit, RD10_HEAD is $RD10_HEAD"; exit 2; } ;;   # 16c F1: rawcheck would STOP on 'unknown' only AFTER the paid run
  reset)   # Step 1: a rerun starts clean (lane L's M4): the budget, the last hold length, the void count and every raw file
    rm -f "$S/budget-seconds" "$S/last-hold-seconds" "$S/void-count"; rm -f "$S"/raw/*
    # the old call logs are MOVED (kept as evidence) so that a new report lists only its own holds (lane L's N3)
    prev="$S/logs/prev-$(date +%Y%m%d-%H%M%S)"
    if compgen -G "$S/logs/call-*.log" >/dev/null; then mkdir -p "$prev"; mv "$S"/logs/call-*.log "$prev"/; fi
    # a leftover report of the earlier run is untracked in the worktree and would make `clean-tree` STOP the next `report` (lane L's N10): moved too
    # only UNTRACKED reports: a committed report of an earlier row (the eager pre-run before the captured run) stays where it is
    for old in "$WT/qualification/cuda-rd"/stage1-*.md; do
      [ -e "$old" ] && ! git -C "$WT" ls-files --error-unmatch "$old" >/dev/null 2>&1 && { mkdir -p "$prev"; mv "$old" "$prev"/; }
    done
    echo "state reset" ;;
  clean-tree)   # lane L's M6 and N4: refuse uncommitted changes AND untracked files (an untracked .cargo/config.toml or rust-toolchain.toml
                # changes the build without a tracked diff); ignored files (.gitignore) are fine
    test -z "$(git -C "$WT" status --porcelain --untracked-files=all)" || { echo "STOP: the worktree has uncommitted changes or untracked files"; exit 2; } ;;
  selftest) python3 -I "$F/run_selftests.py" "$S/selftest" "$S/selftest-result.json" ;;
  public-items)   # authoritative: deterministic SOURCE check (no build, no RUSTC_BOOTSTRAP); secondary: HTML listing for the handoff
    python3 -I "$F/public_items.py" "$WT/engine/src" ${RD10_BASE_REV:+--base-rev "$RD10_BASE_REV" --repo "$WT"} --write "$S/public-items.txt"
    RD9A_ENGINE="$WT/engine" RD9A_SCRATCH="$S" bash "$D/run-cpu.sh" doc cargo doc --no-deps --offline --locked --features native-cuda
    find "$S/target/doc/inference_engine/family/qwen36" -name '*.html' | sort | sed "s#^$S/target/doc/##" > "$S/public-items-html.txt"
    cat "$S/public-items-html.txt" ;;
  idle)    batch_ok
           set +e; "$BATON" 60 python3 -I "$F/idle_check.py" pre | tee -a "$S/logs/idle.log"; rc=${PIPESTATUS[0]}; set -e
           [ "$rc" -eq 0 ] || { echo "STOP: the idle check failed (rc=$rc)"; exit 2; } ;;
  call)    # one hold of at most 300 s: idle check, exe check, run with --deadline-secs 270, watch, post record
    name="${1:?call name}"; cap="${RD10_CAPTURE:-graph}"
    want="$(python3 -I -c 'import json,sys;print(json.load(open(sys.argv[1]))["exe_sha256"])' "$S/release.json")"
    sha="$(python3 -I -c 'import json,sys;print(json.load(open(sys.argv[1]))["sha256"])' "$S/gguf.json")"
    case "$name" in
      A1) a=(--mode forced --prompts P1,P3 --capture "$cap") ;;
      A2) a=(--mode forced --prompts P2,P4,P5 --capture "$cap") ;;
      E1) a=(--mode forced --prompts P1,P3 --capture eager) ;;
      E2) a=(--mode forced --prompts P2,P4,P5 --capture eager) ;;
      B1|B2|B3) a=(--mode free --prompts P2,P3,P4 --capture "$cap") ;;
      B4|B5|B6) a=(--mode free --prompts P5 --capture "$cap") ;;
      C1|C2|C3|C4|C5) a=(--mode free --prompts P1 --capture "$cap") ;;
      *) echo "STOP: unknown call $name"; exit 2 ;;
    esac
    rm -f "$S/raw/$name.json" "$S/raw/$name.killed" "$S/raw/$name.blocked" "$S/raw/$name.skipped" "$S/raw/$name.watch.jsonl"
    for attempt in 1 2 3; do
      batch_ok
      # hold-time budget of 40 minutes in total, checked before EVERY hold (attempts too): skip the rest rather than overrun it
      used=$(cat "$S/budget-seconds" 2>/dev/null || echo 0); last=$(cat "$S/last-hold-seconds" 2>/dev/null || echo 300)
      if [ $(( used + last * 12 / 10 )) -gt 2400 ]; then echo "BUDGET: $used s used, next hold about $last s: the 40-minute budget would be exceeded; skipping $name"
        echo "budget" > "$S/raw/$name.skipped"; rm -f "$S/raw/$name.json" "$S/raw/$name.watch.jsonl"; exit 1; fi
      set +e
      "$BATON" 300 bash "$F/hold.sh" "$want" "$BIN" "$S/raw/$name.watch.jsonl" run --gguf "$GG" --gguf-sha256 "$sha" \
        --oracle "$QUAL/oracle" --runs 1 "${a[@]}" --out "$S/raw/$name.json" 2>&1 | tee -a "$S/logs/call-$name.log"
      rc="${PIPESTATUS[0]}"
      set -e
      secs=$( (grep -o 'HOLD seconds [0-9]*' "$S/logs/call-$name.log" || echo 'HOLD seconds 300') | tail -1 | cut -d' ' -f3)   # an early exit (not idle, wrong exe) prints no seconds: count 300
      echo $(( $(cat "$S/budget-seconds" 2>/dev/null || echo 0) + secs )) > "$S/budget-seconds"; echo "$secs" > "$S/last-hold-seconds"
      case "$rc" in
        0) ;;
        3) echo "REFUSED: the gate refused the run (memory); no retry. Go to RUN report"; exit 3 ;;
        137) echo "killed at the hold limit" | tee -a "$S/logs/call-$name.log"; echo killed > "$S/raw/$name.killed"; cleanup_containers; exit 1 ;;
        *) echo "STOP: call $name exit $rc"; exit 2 ;;
      esac
      set +e; python3 -I "$F/assemble_report.py" --check-raw "$S/raw/$name.json"; crc=$?; set -e
      case "$crc" in
        0) echo 0 > "$S/void-count"; exit 0 ;;                           # a good run resets the count of consecutive voids
        1) exit 1 ;;                                                     # incomplete: too slow (a FAIL path, see RD-10b)
        5) n=$(( $(cat "$S/void-count" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$S/void-count"   # voided: environmental
           cp "$S/raw/$name.watch.jsonl" "$S/logs/clock-$name-void$n-$(date +%s).jsonl"
           rm -f "$S/raw/$name.json" "$S/raw/$name.watch.jsonl"
           if [ "$n" -ge 3 ]; then echo "BLOCKED: 3 consecutive voided runs (clock or thermal problem); clock logs in $S/logs/clock-*"
             echo blocked > "$S/raw/$name.blocked"; exit 7; fi
           echo "run voided ($n consecutive); repeating the call" ;;
        *) exit "$crc" ;;
      esac
    done ;;
  skip)   # after a too-slow call: mark the calls that will not run, so the report still gives a FAIL verdict
    for n in "$@"; do rm -f "$S/raw/$n.json"; echo "stopped early" > "$S/raw/$n.skipped"; done ;;
  check-raw) python3 -I "$F/assemble_report.py" --check-raw "$S/raw/${1:?name}.json" ;;
  wrapper)   # the RD-2e test executables built by `build-tests`, run in one hold, release profile, filter projection_repeat_
    batch_ok
    { echo "engine_commit $(git -C "$WT" rev-parse HEAD)"; echo "profile release"
      "$BATON" 300 bash -c '"$1" projection_repeat_ --nocapture --test-threads=1' _ "$(head -1 "$S/wrapper-exes.txt")"; } \
      2>&1 | tee "$S/wrapper.log" ;;
  report)  # RD10_CAPTURE=eager writes the pre-run file name and the data must say eager too
    # the date is exactly YYYY-MM-DD (a value such as "eager-prerun-2026-10-10" made the captured name collide with the eager one: lane L's N4)
    [[ "${RD10_DATE:?}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || { echo "STOP: RD10_DATE must be YYYY-MM-DD, got '$RD10_DATE'"; exit 2; }
    bash "$F/run_stage1.sh" clean-tree
    if [ "${RD10_CAPTURE:-graph}" = eager ]; then pre=(--prerun); out="stage1-eager-prerun-${RD10_DATE}.md"
    else pre=(); out="stage1-${RD10_DATE}.md"; fi
    # never overwrite a report that already exists, committed or not (lane L's N3): a second run on the same date needs another date
    [ ! -e "$QUAL/$out" ] || { echo "STOP: $QUAL/$out already exists; it is not overwritten"; exit 2; }
    python3 -I "$F/assemble_report.py" --raw "$S/raw" --release-check "$S/release.json" --wrapper-log "$S/wrapper.log" \
      --selftest-result "$S/selftest-result.json" --logs "$S/logs" --head "$(git -C "$WT" rev-parse "${RD10_HEAD:?}")" --date "$RD10_DATE" "${pre[@]}" --out "$QUAL/$out" ;;
  cleanup) cleanup_containers ;;
  *) echo "STOP: unknown subcommand $cmd"; exit 2 ;;
esac
