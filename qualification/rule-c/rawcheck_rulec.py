"""Shared input gate for the rule C tools (a5_retol.py, grade.py, determ.py).

rawcheck_rulec.check_raw refuses a cuda-rd-raw/2 document, with a message naming the file and a non-zero exit
in the caller, BEFORE any number is printed, when
  - the schema is not "cuda-rd-raw/2";
  - the incomplete or refused flag is set;
  - the mode (when the caller passes one) is not the required one;
  - prompt ids are duplicated, a prompt has no runs, or a run's step count is not the oracle's (positional step
    rows: a count other than the oracle stream length is the only way to leave a gap in the step index range
    0..N-1) -- checked only when the caller passes the oracle's count (grade.py and a5_retol.py know it; determ.py
    compares its two files instead);
  - a value the calling tool reads (argmax_logit, top logits, lse) is not a finite number.
`fields` names what to finite-check, so a tool never validates a field it does not use. Prompt COVERAGE (the
union of ids over all files of one invocation) is content-based and checked by the tools through
check_coverage; a file's name only selects it, it never relaxes a check.
Also check_steps(): the shape/finiteness gate for the reference run.json trees a5_retol.py reads.
Percentiles are NOT part of this module: grade.py keeps its nearest-rank percentile and a5_retol.py keeps its
own q(); see the docstrings there and the README.  stdlib only.
"""
import math

SCHEMA = "cuda-rd-raw/2"


class Refused(Exception):
    """Bad raw input: the message names the file and every reason found."""


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _raise(name, reasons):
    if reasons:
        raise Refused(f"{name}: " + "; ".join(dict.fromkeys(reasons)))


def _check_schema_flags_mode(raw, reasons, mode):
    if raw.get("schema") != SCHEMA:
        reasons.append(f"schema {raw.get('schema')!r} is not {SCHEMA!r}")
    if raw.get("incomplete"):
        reasons.append("incomplete flag set")
    if raw.get("refused"):
        reasons.append(f"refused flag set ({raw.get('refused')})")
    if mode is not None and raw.get("mode") != mode:
        reasons.append(f"mode {raw.get('mode')!r} is not {mode!r}")


def _check_prompt_ids(prompts, reasons):
    """Reports non-object prompts, missing ids and duplicates; returns [(pid, runs)] for the well-formed ones."""
    ids = []
    wellformed = []
    for prompt in prompts:
        if not isinstance(prompt, dict):
            reasons.append("a prompt that is not an object")
            continue
        pid = prompt.get("id")
        if not pid:
            reasons.append("a prompt without an id")
            continue
        ids.append(pid)
        runs = prompt.get("runs")
        if not isinstance(runs, list):
            reasons.append(f"{pid}: no runs list")
        elif not runs:
            reasons.append(f"{pid}: no runs")
        else:
            wellformed.append((pid, runs))
    dupes = sorted({pid for pid in ids if ids.count(pid) > 1})
    if dupes:
        reasons.append(f"duplicate prompt ids {dupes}")
    return wellformed


def _check_run(pid, i, run, want, reasons):
    """Shape and oracle step count of one run; returns its steps list, or None when malformed."""
    if not isinstance(run, dict):
        reasons.append(f"{pid} run {i}: no steps list")
        return None
    steps = run.get("steps")
    if not isinstance(steps, list):
        reasons.append(f"{pid} run {i}: no steps list")
        return None
    if want is not None and len(steps) != want:
        reasons.append(f"{pid} run {i}: {len(steps)} steps, the oracle stream has {want} "
                       f"(step index gap or truncation)")
    return steps


def _nonfinite(step, field):
    if field == "top32":
        top = step.get("top32")
        return any(not (isinstance(pair, (list, tuple)) and len(pair) == 2 and _finite(pair[1]))
                   for pair in (top if isinstance(top, list) else []))
    return not _finite(step.get(field))


def _check_finite(pid, i, steps, fields, reasons):
    bad = {}
    for j, step in enumerate(steps):
        for field in fields:
            if _nonfinite(step, field):
                first, count = bad.get(field, (j, 0))
                bad[field] = (first, count + 1)
    for field, (first, count) in bad.items():
        where = f"{pid} run {i} step {first}"
        reasons.append(f"{where}: {field} holds a non-finite value "
                       f"({count} of {len(steps)} steps)" if count > 1 else
                       f"{where}: {field} is not a finite number ({steps[first].get(field)!r})")


def check_raw(raw, name, fields=(), n_steps=None, mode=None):
    """Refuse `raw` (a parsed raw document read from file `name`) when it breaks one of the rules above.

    `fields` names the float members to finite-check per step ("argmax_logit", "lse", "top32" - the top logits).
    `n_steps` is the oracle's step count, as an int or a per-prompt-id dict; None (no oracle known) skips the
    count check.  `mode` refuses every other mode when given (grade.py and a5_retol.py grade forced runs).
    Raises Refused naming `name`; returns silently when the document is acceptable.
    """
    reasons = []
    _check_schema_flags_mode(raw, reasons, mode)
    prompts = raw.get("prompts")
    if not isinstance(prompts, list):
        reasons.append("no prompts list")
        _raise(name, reasons)
    for pid, runs in _check_prompt_ids(prompts, reasons):
        want = n_steps.get(pid) if isinstance(n_steps, dict) else n_steps
        for i, run in enumerate(runs):
            steps = _check_run(pid, i, run, want, reasons)
            if steps is not None:
                _check_finite(pid, i, steps, fields, reasons)
    _raise(name, reasons)


def check_coverage(per_file_ids, oracle_prompts, allow_partial=False):
    """Content-based prompt coverage over ALL files of one invocation (MAJOR-1): every prompt id exactly once,
    every id in the oracle's gated set, and - unless `allow_partial` - the union equal to it. `per_file_ids` is
    [(file name, [ids])]. Returns True when the coverage is partial (only possible with allow_partial)."""
    seen = {}
    for fname, ids in per_file_ids:
        for pid in ids:
            if pid in seen:
                raise Refused(f"{fname}: prompt {pid} also comes from {seen[pid]}")
            seen[pid] = fname
    unknown = sorted(set(seen) - set(oracle_prompts))
    if unknown:
        raise Refused(f"prompt {unknown[0]} is not in the oracle's gated set")
    missing = sorted(set(oracle_prompts) - set(seen))
    if missing and not allow_partial:
        raise Refused(f"the oracle's prompt set is not covered, missing {missing}")
    return bool(missing)


def _bad_top_pair(step):
    """The first malformed 'top' pair of a reference run.json step (not (id, finite logit)), or None."""
    top = step.get("top") if isinstance(step, dict) else None
    for pair in (top if isinstance(top, list) else []):
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2 and _finite(pair[1])):
            return pair
    return None


def check_steps(steps, name, want):
    """Shape and finiteness gate for one reference run.json steps list (the parts a5_retol.py reads: the step
    count and every 'top' logit). Same refusal style as check_raw."""
    reasons = []
    if not isinstance(steps, list):
        reasons.append("no steps list")
    else:
        if want is not None and len(steps) != want:
            reasons.append(f"{len(steps)} steps, the oracle stream has {want} (step index gap or truncation)")
        for j, step in enumerate(steps):
            pair = _bad_top_pair(step)
            if pair is not None:
                reasons.append(f"step {j}: top holds a non-finite value ({pair!r})")
    _raise(name, reasons)
