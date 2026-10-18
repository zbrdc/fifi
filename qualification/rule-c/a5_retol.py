#!/usr/bin/env python3
"""Rule C retolerance analysis: re-derive S_t from the UNION of independent deviations and regrade (S3c retol).

This is rule C of CUDA-SERVE-PLAN ("SERVE-1 G1 on Xavier"), the cross-device agreement rule lane C ruled on 2026-10-14.
References (all teacher-forced on the CPU -t12 greedy stream): cpu12 = the reference; independent deviation sources:
cpu6 (CPU -t 6), cudaOld (llama.cpp CUDA acecd5603), cudaNew (llama.cpp CUDA 23b0202a1). fifi is EXCLUDED from the derivation.
Inputs: the oracle dir (cpu/clean, cpu/threads6, cuda/follow run.json trees), the llama.cpp-new follow dir, cuda-rd-raw/2
raw files, and the frozen a5-gated-steps.json (default ../cuda-rd/a5-gated-steps.json, resolved from this file). The
oracle files are hash-checked through a5_eval.Reference (the same frozen hash table grade.py uses) before any source
is read, and the raw files pass the shared raw gate (rawcheck_rulec.check_raw) before any number is printed.
Rule: S_t = max(floor, max over independent sources of |logit_src(tok) - logit_cpu12(tok)|), tok = cpu12 top-1 token;
logit set L: all sources agree on top-1 and cpu12 margin > 2 S_t -> |fifi logit - cpu12 logit| <= 2 S_t;
token set T: cpu12 margin > S_t -> fifi top-1 == cpu12 top-1.  stdlib only.

WHERE THE RULE C VERDICT IS (per the SERVE-1 G1 on Xavier ruling): the section 5 lines carry it - the
"- fifi row 1: ... -> PASS/FAIL" bounds line, the three leave-one-out "- <src>: bounds ... -> PASS/FAIL" lines,
the "- fifi top-1 token mismatches on the N steps ...: M" token-gate line and the three "- LOO <src>: threshold ...
above it: K" lines. Sections 2 and 3 are the SUPERSEDED per-step L/T rule, printed for information: their FAIL on
the accepted Xavier data is expected. Exit 0 verdict PASS, 1 verdict FAIL, 2 refused input.

usage: a5_retol.py --oracle-dir DIR --lcpp-new-dir DIR --raw-dir DIR [--run-names A1 A2]
                   [--gated-steps JSON] --out OUT.md
"""
import argparse
import collections
import json
import statistics
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUAL = HERE.parent / "cuda-rd"
for _p in (str(HERE), str(QUAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import a5_eval as ev  # noqa: E402
import rawcheck_rulec as rc  # noqa: E402

PROMPTS = ['P1', 'P2', 'P3', 'P4', 'P5']
FLOOR, FACTOR = 0.05, 2.0
M = 1.25
ALL = ('cpu6', 'cudaOld', 'cudaNew')
GATED_DEFAULT = HERE.parent / 'cuda-rd' / 'a5-gated-steps.json'


def steps(path):
    return json.load(open(path))['steps']


def make_sources(oracle, new):
    o, n = Path(oracle), Path(new)
    return {
        'cpu12': lambda p: steps(o / 'cpu/clean' / p / 'run.json'),
        'cpu6': lambda p: steps(o / 'cpu/threads6' / p / 'run.json'),
        'cudaOld': lambda p: steps(o / 'cuda/follow' / p / 'run.json'),
        'cudaNew': lambda p: steps(n / p / 'run.json'),
    }


def logit_of(step, tok):
    for t, v in step['top']:
        if t == tok:
            return v
    return None


def load_fifi(docs):
    fifi = {}
    for doc in docs:
        for pr in doc['prompts']:
            fifi[pr['id']] = pr['runs'][0]['steps']
    return fifi


def build(p, src):
    runs = {k: f(p) for k, f in src.items()}
    rows = []
    for t, rs in enumerate(runs['cpu12']):
        tok = rs['top'][0][0]; l0 = rs['top'][0][1]; margin = l0 - rs['top'][1][1]
        dev = {}; agree = True; missing = False
        for k in ('cpu6', 'cudaOld', 'cudaNew'):
            lv = logit_of(runs[k][t], tok)
            if lv is None: missing = True; dev[k] = None
            else: dev[k] = abs(lv - l0)
            if runs[k][t]['top'][0][0] != tok: agree = False
        rows.append({'t': t, 'tok': tok, 'l0': l0, 'margin': margin, 'dev': dev, 'agree': agree and not missing,
                     'ok_def': not missing})
    return rows


def spread(row, sources):
    vals = [row['dev'][k] for k in sources]
    if any(v is None for v in vals): return None
    return max(FLOOR, max(vals))


def classes(rows, sources):
    T, L = [], []
    for r in rows:
        s = spread(r, sources)
        if s is None: continue
        if r['margin'] > s: T.append((r, s))
        if r['agree'] and r['margin'] > FACTOR * s: L.append((r, s))
    return T, L


def grade_fifi(f, rows, sources):
    T, L = classes(rows, sources); tfail = []; lfail = []; ratios = []
    for r, s in T:
        if f[r['t']]['argmax'] != r['tok']: tfail.append(r['t'])
    for r, s in L:
        st = f[r['t']]
        if st['argmax'] != r['tok']: continue  # counted as a token failure already (T contains L)
        d = abs(st['argmax_logit'] - r['l0']); ratios.append(d / (FACTOR * s))
        if d > FACTOR * s: lfail.append(r['t'])
    return len(T), len(L), tfail, lfail, ratios


def grade_source(p, rows, held, derive_from, src):
    """leave-one-out: derive S_t from derive_from, grade held-out source `held` (logit deviation vs cpu12 and token agreement)."""
    T, L = classes(rows, derive_from); run = src[held](p); tfail = lfail = 0
    for r, s in T:
        if run[r['t']]['top'][0][0] != r['tok']: tfail += 1
    for r, s in L:
        if run[r['t']]['top'][0][0] != r['tok']: continue
        if r['dev'][held] is not None and r['dev'][held] > FACTOR * s: lfail += 1
    return len(T), len(L), tfail, lfail


def ratios_for(data, sources_for_S, grade_fn):
    r = []
    for p in data:
        for row in data[p]:
            s = spread(row, sources_for_S)
            if s is None or not (row['agree'] and row['margin'] > FACTOR * s): continue
            d = grade_fn(p, row)
            if d is not None: r.append(d / s)
    return r


def dev_list(src, data):
    return [r['dev'][src] for p in data for r in data[p] if r['dev'][src] is not None and r['agree']]


def fifi_list(fifi, data):
    return [abs(fifi[p][r['t']]['argmax_logit'] - r['l0']) for p in data for r in data[p]
            if r['ok_def'] and fifi[p][r['t']]['argmax'] == r['tok']]


def stats(a):
    return (statistics.median(a), q(a, .95), q(a, .99), max(a))


def q(a, x):
    """a5_retol's own percentile (lane L's): unsorted input, index int(x*n) of the sorted list. NOT the nearest-rank
    percentile grade.py uses: they agree at the recorded sizes (n=622, n=632) but differ in general -- e.g. at n=620,
    q=0.95, where 0.95*620 == 589.0 exactly and the two pick adjacent values (this is the section 4 fifi line)."""
    a = sorted(a); return a[min(len(a) - 1, int(x * len(a)))]


def flip_margins(src, data, sources):
    run = {p: sources[src](p) for p in data}
    return [r['margin'] for p in data for r in data[p] if run[p][r['t']]['top'][0][0] != r['tok']]


def flip_threshold(margins_by_source, m=M):
    """Rule C token gate: threshold = m x the largest flip margin among the independent sources (0 when none ever flips)."""
    return m * max((max(v) if v else 0) for v in margins_by_source.values())


def token_gate(fifi, data, mstar):
    """fifi top-1 token mismatches on the steps whose cpu12 margin exceeds mstar: (n gated steps, n mismatches)."""
    nT = sum(1 for p in data for r in data[p] if r['margin'] > mstar)
    tf = sum(1 for p in data for r in data[p] if r['margin'] > mstar and fifi[p][r['t']]['argmax'] != r['tok'])
    return nT, tf


def sec1_union(data, gated_path):
    out = []
    w = out.append
    gated = json.load(open(gated_path))['prompts']
    w('## 1. Union spread S_t and the new step sets (sources: cpu6, cudaOld, cudaNew vs cpu12; floor 0.05; factor 2)')
    w('| prompt | steps | frozen rule gated (old) | new logit set L | new token set T | S_t median | S_t p95 | S_t max |')
    w('|---|---|---|---|---|---|---|---|')
    for p in PROMPTS:
        rows = data[p]; T, L = classes(rows, ALL)
        sts = sorted(spread(r, ALL) for r in rows if spread(r, ALL) is not None)
        old = gated[p]['n_gated']
        w(f'| {p} | {len(rows)} | {old} | {len(L)} | {len(T)} | {statistics.median(sts):.2f} | {sts[int(.95 * len(sts))]:.2f} | {sts[-1]:.2f} |')
    w('')
    return out


def sec2_regrade(fifi, data):
    out = []
    w = out.append
    w('## 2. fifi row 1 (eager pre-run, A1/A2 raw) regraded under the union rule')
    w('| prompt | token set T | token fails | logit set L | logit fails | max dev/tol over L | verdict |')
    w('|---|---|---|---|---|---|---|')
    tot = collections.Counter(); allpass = True
    for p in PROMPTS:
        nT, nL, tf, lf, ratios = grade_fifi(fifi[p], data[p], ALL)
        ok = not tf and not lf; allpass &= ok
        tot['T'] += nT; tot['L'] += nL; tot['tf'] += len(tf); tot['lf'] += len(lf)
        w(f'| {p} | {nT} | {len(tf)} {tf if tf else ""} | {nL} | {len(lf)} {lf if lf else ""} | {max(ratios):.2f} | {"PASS" if ok else "FAIL"} |')
    w(f'| all | {tot["T"]} | {tot["tf"]} | {tot["L"]} | {tot["lf"]} | | {"PASS" if allpass else "FAIL"} |')
    w('')
    return out


def sec3_leave_one_out(data, src, fifi):
    out = []
    w = out.append
    w('## 3. Leave-one-out (derive S_t from the other two independent sources, grade the held-out source by the same rule)')
    w('| held-out source | derived from | prompt | T | token fails | L | logit fails |')
    w('|---|---|---|---|---|---|---|')
    for held in ALL:
        rest = tuple(k for k in ALL if k != held)
        ft = fl = nt = nl = 0
        for p in PROMPTS:
            nT, nL, tf, lf = grade_source(p, data[p], held, rest, src)
            w(f'| {held} | {"+".join(rest)} | {p} | {nT} | {tf} | {nL} | {lf} |')
            ft += tf; fl += lf; nt += nT; nl += nL
        w(f'| **{held} total** | {"+".join(rest)} | all | {nt} | {ft} | {nl} | {fl} |')
    w('')
    w('Also fifi graded against the union of ONLY the llama.cpp sources, and the pairwise-max variant, for sensitivity:')
    for name, srcs in (('union without cpu6 (cudaOld+cudaNew)', ('cudaOld', 'cudaNew')), ('cpu6 only', ('cpu6',)),
                       ('cudaNew only', ('cudaNew',)), ('cudaOld only', ('cudaOld',))):
        tf = lf = nT = nL = 0
        for p in PROMPTS:
            a, b, c, d, _ = grade_fifi(fifi[p], data[p], srcs); nT += a; nL += b; tf += len(c); lf += len(d)
        w(f'- {name}: T={nT} L={nL} token fails={tf} logit fails={lf}')
    return out


def sec4_calibration(fifi, data, src):
    out = []
    w2 = out.append
    w2('## 4. Calibration of the factor k (logit set fixed at k=2 step selection; dev/S_t over those steps)')
    w2('| graded | S_t from | steps | median dev/S | p95 | p99 | max dev/S |')
    w2('|---|---|---|---|---|---|---|')
    for held in ALL:
        rest = tuple(k for k in ALL if k != held)
        rr = ratios_for(data, rest, lambda p, row, held=held: row['dev'][held])
        w2(f'| {held} (held out) | {"+".join(rest)} | {len(rr)} | {statistics.median(rr):.2f} | {q(rr, .95):.2f} | {q(rr, .99):.2f} | {max(rr):.2f} |')
    fr = ratios_for(data, ALL, lambda p, row: abs(fifi[p][row['t']]['argmax_logit'] - row['l0'])
                    if fifi[p][row['t']]['argmax'] == row['tok'] else None)
    w2(f'| fifi | cpu6+cudaOld+cudaNew | {len(fr)} | {statistics.median(fr):.2f} | {q(fr, .95):.2f} | {q(fr, .99):.2f} | {max(fr):.2f} |')
    w2('')
    w2('Absolute top-1 logit deviation vs cpu12 over ALL steps where defined (no step selection), per source: median / p95 / p99 / max')
    for name in ('cpu6', 'cudaOld', 'cudaNew'):
        a = dev_list(name, data)
        w2(f'- {name}: n={len(a)} {statistics.median(a):.2f} / {q(a, .95):.2f} / {q(a, .99):.2f} / {max(a):.2f}')
    a = [abs(fifi[p][r["t"]]["argmax_logit"] - r["l0"]) for p in PROMPTS for r in data[p]
         if r["agree"] and fifi[p][r["t"]]["argmax"] == r["tok"]]
    w2(f'- fifi: n={len(a)} {statistics.median(a):.2f} / {q(a, .95):.2f} / {q(a, .99):.2f} / {max(a):.2f}')
    return out


def sec5_rule_c(fifi, data, src):
    out = []
    c = out.append
    c('## 5. Candidate rule C: distribution bounds + margin-based token agreement (derived from independent sources only)')
    c(f'Bound for each statistic (median, p95, p99) = {M} x the largest value of that statistic among the independent '
      f'sources; the max is reported, not gated (one rare event, P4 step 85, dominates it).')
    fs = stats(fifi_list(fifi, data))
    indep = {s: stats(dev_list(s, data)) for s in ALL}
    for s, v in indep.items(): c(f'- {s}: median {v[0]:.3f} p95 {v[1]:.3f} p99 {v[2]:.3f} max {v[3]:.2f}')
    bound = [M * max(indep[s][i] for s in ALL) for i in range(3)]
    c(f'- derived bounds (3 sources): median <= {bound[0]:.3f}, p95 <= {bound[1]:.3f}, p99 <= {bound[2]:.3f}')
    fs_ok = all(fs[i] <= bound[i] for i in range(3))
    c(f'- fifi row 1: median {fs[0]:.3f} p95 {fs[1]:.3f} p99 {fs[2]:.3f} max {fs[3]:.2f} -> '
      + ('PASS' if fs_ok else 'FAIL'))
    c('Leave-one-out (bounds from the other two, grade the held-out):')
    loo_ok = True
    for held in ALL:
        rest = [s for s in ALL if s != held]
        b = [M * max(indep[s][i] for s in rest) for i in range(3)]
        h = indep[held]
        ok_h = all(h[i] <= b[i] for i in range(3)); loo_ok = loo_ok and ok_h
        c(f'- {held}: bounds {b[0]:.3f}/{b[1]:.3f}/{b[2]:.3f} vs {h[0]:.3f}/{h[1]:.3f}/{h[2]:.3f} -> '
          + ('PASS' if ok_h else 'FAIL'))
    c('Token agreement: margin of the cpu12 reference at every step where an independent source picked a different top-1:')
    fm = {s: sorted(flip_margins(s, data, src)) for s in ALL}
    for s in ALL: c(f'- {s}: {len(fm[s])} flips, margins ' + ', '.join(f'{x:.3f}' for x in fm[s]))
    mstar = flip_threshold(fm)
    c(f'- threshold = {M} x the largest flip margin among independent sources = {mstar:.3f}')
    nT, tf = token_gate(fifi, data, mstar)
    token_ok = tf == 0
    c(f'- fifi top-1 token mismatches on the {nT} steps with margin > {mstar:.3f}: {tf}')
    loo_thr_ok = True
    for held in ALL:
        rest = [s for s in ALL if s != held]
        ms = flip_threshold({s: fm[s] for s in rest})
        bad = sum(1 for x in fm[held] if x > ms)
        loo_thr_ok = loo_thr_ok and bad == 0
        c(f'- LOO {held}: threshold from {"+".join(rest)} = {ms:.3f}; flips of the held-out source above it: {bad}')
    return out, fs_ok and loo_ok and token_ok and loo_thr_ok


def main(argv=None):
    ap = argparse.ArgumentParser(description='rule C retolerance analysis (CUDA-SERVE-PLAN "SERVE-1 G1 on Xavier"; S3c retol)')
    ap.add_argument('--oracle-dir', required=True,
                    help='oracle dir with cpu/clean, cpu/threads6 and cuda/follow <P>/run.json trees (cpu12/cpu6/cudaOld)')
    ap.add_argument('--lcpp-new-dir', required=True, help='llama.cpp-new follow dir with <P>/run.json (cudaNew)')
    ap.add_argument('--raw-dir', required=True, help='dir holding the cuda-rd-raw/2 files <run>.json (the fifi runs)')
    ap.add_argument('--run-names', nargs=2, default=['A1', 'A2'], metavar=('RUN1', 'RUN2'),
                    help='the two raw file names inside --raw-dir, without .json (default: A1 A2)')
    ap.add_argument('--gated-steps', default=GATED_DEFAULT, type=Path,
                    help='frozen a5-gated-steps.json for the "old" gated column and the oracle hash check '
                         '(default: ../cuda-rd/a5-gated-steps.json; an explicit other file is trusted as given)')
    ap.add_argument('--out', required=True, type=Path, help='markdown report to write')
    args = ap.parse_args(argv)
    try:  # the oracle hash check, the source-tree check and the raw gate run before any number is printed
        src = make_sources(args.oracle_dir, args.lcpp_new_dir)
        if args.gated_steps.resolve() == GATED_DEFAULT.resolve():
            ref = ev.Reference(oracle=Path(args.oracle_dir))
        else:
            ref = ev.Reference(oracle=Path(args.oracle_dir), gated_path=args.gated_steps,
                               expect=ev.sha_file(args.gated_steps))
        docs = []
        for n in args.run_names:
            doc = json.load(open(Path(args.raw_dir) / f'{n}.json'))
            rc.check_raw(doc, f'{n}.json', fields=('argmax_logit',), n_steps={p: len(ref.streams[p]) for p in PROMPTS},
                         mode='forced')
            docs.append((n, doc))
        rc.check_coverage([(f'{n}.json', [pr['id'] for pr in d['prompts']]) for n, d in docs], sorted(ref.streams))
        for p in sorted(ref.streams):  # the reference trees are not all manifest-hashed: gate what is read
            for sname, get in src.items():
                rc.check_steps(get(p), f'{sname} {p}', len(ref.streams[p]))
    except rc.Refused as refused:
        print(f'refused: {refused}', file=sys.stderr)
        args.out.unlink(missing_ok=True)  # no stale report from an earlier run may survive a refusal
        return 2
    fifi = load_fifi([d for _, d in docs])
    data = {p: build(p, src) for p in PROMPTS}
    out_path = args.out
    out = sec1_union(data, args.gated_steps) + sec2_regrade(fifi, data) + sec3_leave_one_out(data, src, fifi)
    print('\n'.join(out))
    open(out_path, 'w').write('\n'.join(out) + '\n')
    cal = sec4_calibration(fifi, data, src)
    print('\n'.join(cal))
    open(out_path, 'a').write('\n' + '\n'.join(cal) + '\n')
    cand, verdict = sec5_rule_c(fifi, data, src)
    print('\n'.join(cand))
    open(out_path, 'a').write('\n' + '\n'.join(cand) + '\n')
    return 0 if verdict else 1


def cli(argv=None):
    try:
        return main(argv)
    except (ev.Stop, OSError, json.JSONDecodeError) as stopped:
        print(f'stopped: {stopped}', file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 3


if __name__ == '__main__':
    sys.exit(cli())
