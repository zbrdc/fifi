# HumanEval baseline harness

This directory restores the HumanEval baseline harness for the inference-engine repository. `he_run_mlx.py` generates completions with any mlx-lm-loadable model. `he_grade.py` grades the responses against the official tests. The legacy reference implementer's harness supplies the scripts. `he_run_mlx.py` carries one change: the problems path is a required third argv with no default. The script exits with an error and a nonzero status when the path is missing or unreadable. `he_grade.py` requires the problems path and checks its inputs (see Input checks). The instruction template `INSTR`, `extract_block()`, `final_text()`, and `passes()` stay byte-identical to the originals.

## Pinned protocol

- 164 problems from the official OpenAI HumanEval dataset
- Greedy decoding: `make_sampler(temp=0.0)`
- 2,000-token output budget per problem
- Official per-problem tests, `check(entry_point)`
- Isolated subprocess per problem (`python -I`), 10 s timeout
- Metric: pass@1

Provenance and hashes of the recovered originals and the modified copies are in `MANIFEST.json`. The recovered originals live in `/media/dan/nvme/fifi/scrch/4ee8756/recovered-20260920/legacy-reference-implementer/`.

The engine records this baseline. It does not gate it (`docs/design/tests.md:145`).

## Baseline settings

User decision, item 9, 2026-10-01 (question 28, C10):

- HumanEval baselines run with thinking off.
- Every baseline that a gate compares uses this one fixed setting.
- The bf16 reference host is not named yet. It is not the target machine.
- A host larger than the 36 GB Mac Studio is a reference host only.

## Usage

Usage: `he_run_mlx.py MODEL_DIR OUT PROBLEMS`. The script requires all three arguments. The script exits with an error and a nonzero status when `PROBLEMS` is missing or unreadable.

## Problems file

The problems file is a JSON Lines file from the official OpenAI HumanEval dataset with exactly 164 problems. Each line must have the fields `task_id`, `prompt`, `test`, and `entry_point`. The harness performs no download. The coordinating agent supplies the file path.

## Input checks

Usage: `he_grade.py RESPONSES LABEL PROBLEMS`. All three arguments are required.

`he_grade.py` exits with an error message and a nonzero status in these cases:

- The problems file does not hold exactly 164 unique task IDs, each with the string fields `task_id`, `prompt`, `test`, and `entry_point`.
- A file cannot be read or decoded, or a line is not valid JSON or not a JSON object.
- The response file is empty.
- A response repeats a task ID, names a task ID absent from the problems file, or lacks a string `task_id` or `response`.

The grader reports pass@1 over 164. A missing response counts as a failure. The grader prints the missing count.

## Running on the Mac

The Mac runner uses the edge0 virtual environment at `/Users/dan/edge0/.venv/bin/python` with `mlx 0.30.6` and `mlx-lm 0.31.0` installed.

The coordinating agent runs:

```sh
TMPDIR=TASK_SCRATCH ./run-baseline.sh MODEL_DIR PROBLEMS OUTDIR
```

which is equivalent to:

```sh
PY=/Users/dan/edge0/.venv/bin/python
$PY he_run_mlx.py MODEL_DIR OUTDIR/he_responses.jsonl PROBLEMS
$PY he_grade.py OUTDIR/he_responses.jsonl baseline PROBLEMS
```

The script exits with an error when `TMPDIR` is unset or empty. Grading writes its temporary directories there.

The runner writes logs to `OUTDIR`. The scripts use no network and download nothing.

## Historical cross-reference

The recovered `he_edge0_result.txt` reports edge0-35b pass@1 140/164 = 85.4% under the Edge0 stack with adapters. That result is not the plain-base mlx-lm baseline and must not be copied as one. A fresh baseline requires a fresh run of this harness.
