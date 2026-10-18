"""RD-10 report text: renders the tables of the stage 1 record from already evaluated results (RD-10a-12a). Pure formatting, CPU only,
stdlib only, `python3 -I`. Design guidance only, no code copied."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a5_eval as ev  # noqa: E402

LABEL = "EAGER PRE-RUN, not a stage gate"


def hold_records(logs):
    """Lines the hold wrapper wrote (HOLD ...): pre/post clocks, exe sha, disturbance verdict. Appended by script, never by hand."""
    out = []
    for f in sorted(Path(logs).glob("call-*.log")) if logs else []:
        out += [f"{f.name}: {ln}" for ln in f.read_text().splitlines() if ln.startswith("HOLD ")]
    return out


def render(res, ident, hashes, why, date, prerun, ref, partial_note, holds=(), clocks=None):
    L = []
    w = L.append
    lab = f" ({LABEL})" if prerun else ""
    w(f"# CUDA-RD stage 1 {LABEL if prerun else 'qualification run'} {date}\n")
    w("Identities (from the raw files; all 12 files agree):\n")
    for k in sorted(ident):
        w(f"- {k}: {ident[k]}")
    w(f"- a5-gated-steps.json sha256 {ref.gated_sha256} (the tracked copy, checked equal to the frozen 18d329a6... before evaluation)")
    w(f"- reference inputs verified against the hashes frozen in the gated file: {len(ref.verified)} committed, "
      f"{len(ref.not_committed)} frozen but not committed (references of other builds)")
    for k, v in sorted(hashes.items()):
        w(f"- sha256 {k}: {v}")
    if prerun:
        w("- capture outcome of every run (RD-9d `CaptureOutcome`): Eager (capture_requested = eager and captured = false on every run). "
          "This record reports the first token, the A5 parity and the eager tok/s as INFORMATION; it can never satisfy stage 1, and the "
          "captured gate run follows as a separate later row once the slim RD-11a has merged.")
    w(f"- exit status of this evaluator: {1 if why else 0}")
    w("- timing rules (REPLY-72 Q8, Q9; lane C): the graph is instantiated and the prompt pass has synchronized before the timed "
      "region; the card is sampled every 0.5 s DURING the timed region only; any sample below 1395 MHz, or a busy (>= 50 percent) "
      "median below 1600 MHz (judged only with at least 10 busy samples), would have voided the run (a voided run is repeated, never counted); the first timed step is "
      "reported beside the median step and flagged when over 2x, never dropped\n")
    if res is None:
        w("## Result\n\n" + "; ".join(why) + "\n" + partial_note)
        return "\n".join(L) + "\n"
    sp = res["speed"]
    w(f"## Table 1: exactness and speed{lab}\n")
    w("| prompt | prompt tokens | gen | gated | A5 fails | excluded (share, information only, REPLY-41/REPLY-69) | max logit dev | max abs lse diff vs CPU | first divergence vs llama.cpp CPU (free run, info) | margin there | tok/s (P1 median of 5, others of 3) | top-5 set agree |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in ev.PROMPTS:
        a, f = res["a5"][p], res["free"][p][0]
        ne, nt, sh = ref.excluded_share(p)
        tok = sp["median"] if p == "P1" else res["speed_others"][p]
        w(f"| {p} | {ref.block(p)['n_prompt_tokens']} | 128 | {a['n_gated']} | {len(a['fails'])} | {ne}/{nt} ({sh:.2f} percent) | "
          f"{a['max_logit_dev']:.4f} | {a['lse_abs_diff_max']:.4f} | {'none' if f['first_divergence'] is None else f['first_divergence']} | "
          f"{'-' if f['margin_at_divergence'] is None else round(f['margin_at_divergence'], 4)} | {tok:.2f} | {a['top5_set_agree']}/128 |")
    w("\nTimed region: the 128 decode steps (device step, token read-back, copy of the 993 KB row into a preallocated host "
      "buffer); host top-32, hash and log-sum-exp run after the clock stops. Decode steps only, prompt tokens excluded.")
    w("ms/token breakdown (P1, median run): " + " + ".join(f"{k} {v:.2f}" for k, v in sp["breakdown_ms"].items()) +
      f" = {sum(sp['breakdown_ms'].values()):.2f} against 1000/tok/s = {sp['ms_per_token']:.2f} (error {sp['breakdown_err'] * 100:.1f} percent, limit 3). "
      f"Host analysis outside the clock: {sp['analysis_ms']:.2f} ms/token. Host-blocking syncs per token: 1 and host writes per token: 1 (the start-block write) on every P1 run (both self-reported by the step counters, lane C Q-C6; a tiered run adds its in-layer waits and is checked as a lower bound). "
      f"Run spread (max-min)/median: {sp['spread'] * 100:.1f} percent. Largest phase: {max(sp['breakdown_ms'], key=sp['breakdown_ms'].get)}.\n")
    fl = sp["first_step_flagged"]
    w(f"First timed step (P1 median run): {sp['first_step_ms']:.2f} ms beside the median step {sp['median_step_ms']:.2f} ms"
      f"{' FLAG: over 2x' if sp['first_step_ms'] > ev.FIRST_STEP_FLAG * sp['median_step_ms'] else ''}; runs whose first step is over 2x the median: "
      f"{[f'C{i + 1}' for i in fl] if fl else 'none'} (included in the timing, never dropped).\n")
    w("Clocks during the timed region (total and busy sample counts, minimum and median of the busy samples per run; floor 1395 MHz, busy median at least 1600 MHz):\n")
    w("| call | prompt | samples | busy samples | min MHz | busy median MHz | median rule | voided |\n|---|---|---|---|---|---|---|---|")
    for (n, pid), c in sorted((clocks or {}).items()):
        w(f"| {n} | {pid} | {c['n']} | {c['n_busy']} | {c['min_mhz']:.0f} | {'-' if c['median_busy_mhz'] is None else format(c['median_busy_mhz'], '.0f')} | {c['median_rule']} | {'YES' if c['voided'] else 'no'} |")
    w("")
    w(f"## Table 2: determinism and capture{lab}\n")
    w("| check | runs | result |\n|---|---|---|")
    for p in ev.PROMPTS:
        w(f"| repeated runs identical (token ids and row hashes), {p} | {res['det'][p]['runs']} | {'pass' if res['det'][p]['ok'] else 'FAIL'} |")
    w(f"| forced and free row hashes equal up to the first divergence | 5 prompts | {'pass' if all(r['ok'] for r in res['tf_free'].values()) else 'FAIL'} |")
    w(f"| captured step equals uncaptured launch (row hash, all 128 rows){' (trivial in a pre-run)' if prerun else ''} | 5 prompts | {'pass' if all(res['cap_eq'].values()) else 'FAIL'} |")
    w(f"| every graph-mode run was really captured and replayed | all gate runs | {'n/a (pre-run)' if prerun else ('FAIL' if any('uncaptured' in x for x in why) else 'pass')} |\n")
    w(f"## Table 3: gate{lab}\n")
    w("| stage | gate tok/s | measured (median of 5, P1) | exactness (A5, every gated step) | verdict |\n|---|---|---|---|---|")
    ex = "pass" if all(r["ok"] for r in res["a5"].values()) else "FAIL"
    w(f"| 1 | {sp['gate']:.0f} | {sp['median']:.2f} | {ex} | {(LABEL + ': tables ') if prerun else ''}{'pass' if prerun and not why else ('PASS' if not why else 'FAIL')}{' (information only, never a stage 1 result)' if prerun else ''} |\n")
    if why:
        w("Misses: " + "; ".join(why) + "\n")
    w("## Hold records (appended by script from the call logs)\n")
    w("\n".join(f"- {h}" for h in holds) or "- none")
    return "\n".join(L) + "\n"
