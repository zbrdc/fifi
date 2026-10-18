# Rule C grading tools

Rule C is the cross-device agreement rule of CUDA-SERVE-PLAN ("SERVE-1 G1 on Xavier"): the fifi CUDA backend is graded
against a frozen CPU oracle, with every tolerance derived only from independent sources — cpu6 (CPU `-t 6`), cudaOld
(llama.cpp CUDA acecd5603) and cudaNew (llama.cpp CUDA 23b0202a1), all teacher-forced on the cpu12 (`-t12`) greedy
stream; fifi itself is excluded from the derivation. The rule has two parts, both leave-one-out checked on the sources:
distribution bounds (median, p95, p99 of the absolute top-1 logit deviation vs cpu12 must each stay within 1.25x the
largest value among the independent sources; on the Xavier acceptance inputs 0.165 / 0.651 / 1.131) and a margin-based
token gate (on steps whose cpu12 top-1/top-2 margin exceeds 1.25x the largest margin at which an independent source
flipped — 0.511 there — fifi must pick the cpu12 top-1 token). These tools compute the rule; the S3c retolerance
rederivation behind it is `a5_retol.py` section 5.

## Files

- `a5_retol.py` rederives the per-step tolerance S_t from the union of the independent deviations and regrades fifi and
  each held-out source (S3c retol).
- `grade.py` grades forced `cuda-rd-raw/2` files against the frozen oracle and prints the bounds verdict.
  It imports `a5_eval` from `../cuda-rd`, which is RD-10's frozen verdict apparatus and is not modified here.
- `determ.py` checks self-determinism of two raw files.
- `rawcheck_rulec.py` is the shared input gate all three tools run before any number is printed: it refuses (message
  naming the file, non-zero exit) a raw whose schema is not `cuda-rd-raw/2`, whose `incomplete`/`refused` flag is set,
  with duplicate prompt ids, a prompt without runs, a run whose step count is not the oracle's (grade.py and
  a5_retol.py know the oracle; determ.py compares its two files instead), a mode other than the one the tool grades
  (`forced` for grade.py and a5_retol.py), or a non-finite value in a field that tool reads (argmax_logit, top logits,
  lse). Prompt COVERAGE is content-based and checked per invocation by the tools through `check_coverage`: the union
  of prompt ids over all graded files must be the oracle's gated set exactly (P1-P5), each id exactly once — a file's
  name only selects it and never relaxes a check. `check_steps` gates the reference run.json trees (step count and
  finite `top` logits) that `a5_retol.py` reads outside the manifest.
- `selftest_rulec.py` is the self-test below.

## Reading the output: where the rule C verdict is

- `grade.py` prints the rule C bounds verdict on its last line, `VERDICT PASS` / `VERDICT FAIL` (median <= 0.165,
  p95 <= 0.651, p99 <= 1.131). The token mismatches in its `TOTAL` line are printed for information: the token gate is
  evaluated where the flip margins live, in `a5_retol.py` section 5.
- `a5_retol.py` section 5 carries the rule C verdict: the `- fifi row 1: ... -> PASS/FAIL` bounds line, the three
  `- <source>: bounds ... -> PASS/FAIL` leave-one-out lines, the `- fifi top-1 token mismatches on the N steps ...: M`
  token-gate line and the three `- LOO <source>: threshold ...: flips of the held-out source above it: K` lines.
  For a rule C PASS, the token line must read `: 0` and all three LOO lines must read `: 0` alongside the bounds
  `-> PASS` lines. The script exits non-zero when any of them fails.
- Sections 2 and 3 of `a5_retol.py` are the SUPERSEDED per-step L/T rule, printed for information: their FAIL on the
  accepted Xavier data is expected. So are `a5_eval`'s A5 lines in `grade.py` (the frozen A5 gate does not pass on
  these raws; the PASS exists only under rule C).

Exit codes of `grade.py` and `a5_retol.py`: 0 verdict PASS, 1 verdict FAIL (and `VERDICT PARTIAL` under
`grade.py --allow-partial`, which grades a subset of the oracle prompts and can never be read as a PASS), 2 refused
input or input problem (bad raw, missing or tampered oracle or reference tree, unreadable file — message on stderr,
no numbers), 3 crash or internal error. `determ.py`: 0 DETERMINISTIC, 1 NOT IDENTICAL, 2 refused or unreadable,
3 crash or internal error.

## Two percentile definitions

The tools deliberately keep the two percentile definitions they were born with, documented where each is defined:

- `grade.py` `percentile` (nearest rank, `rank = max(1, round(q*n + 0.4999999))` of a sorted list) produces the p95
  and p99 in `grade.py`'s `TOTAL` line; the ruled rule-C numbers (n = 632: 0.1069 / 0.5317 / 1.0295) come from it.
- `a5_retol.py` `q` (index `int(q*n)` of the sorted list) produces every percentile of sections 4 and 5; section 1's
  S_t p95 column uses the same index formula inline (`sts[int(.95 * len(sts))]`).

They agree at the recorded sizes (n = 622 per source, n = 632 for fifi) but differ in general: at n = 620, q = 0.95,
`0.95*620 == 589.0` exactly in floating point, so `q` picks the 590th value while the nearest rank picks the 589th
(this is the `- fifi: n=620 ...` line of section 4). The self-test pins both definitions on a list where they differ.

## Commands

`ORACLE` is the oracle tree with `cpu/clean`, `cpu/threads6`, `cuda/follow` and `prompts` (checked by `a5_eval` against
the hashes frozen in `qualification/cuda-rd/a5-gated-steps.json`); `LCPP-NEW` holds llama.cpp-new follow runs
(`<P>/run.json`); `RAW` holds `cuda-rd-raw/2` files.

```sh
python3 -I -B qualification/rule-c/a5_retol.py \
  --oracle-dir ORACLE --lcpp-new-dir LCPP-NEW \
  --raw-dir RAW --run-names A1 A2 --out retol.md
python3 -I -B qualification/rule-c/grade.py ORACLE RAW/A1.json RAW/A2.json
python3 -I -B qualification/rule-c/determ.py RAW/D1.json RAW/D2.json
```

`a5_retol.py` writes the markdown report to `--out` (removed again when the run refuses), hashes the oracle it reads
through `a5_eval.Reference` and finite/shape-checks the reference trees. The `--gated-steps` default is the frozen
`../cuda-rd/a5-gated-steps.json`, resolved from the script and pinned to its frozen hash; an explicit other file is
trusted as given (the self-test's injection hook). `grade.py` builds the same `a5_eval.Reference` (same default);
`--allow-partial` grades a subset of the oracle prompts and prints `VERDICT PARTIAL` (exit 1) instead of refusing.
`determ.py` prints `DETERMINISTIC` only when every step of every prompt is equal in both files.

## Self-test

```sh
python3 -I -B qualification/rule-c/selftest_rulec.py
```

Runs stdlib tests on small synthetic raw/oracle fixtures written to a temp dir; exit 0 only if all pass. The first
line names the directory under test: set `RULEC_TOOLS_DIR` to point the self-test at another copy of the tools
(mutant proving) — a stale value would make it test a different tree, so read the line.
