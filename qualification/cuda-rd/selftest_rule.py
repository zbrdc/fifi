#!/usr/bin/env python3
"""CPU self-test of a5_eval.py: the A5 rule (stand-ins pass, mutations fail), determinism, forced-vs-free consistency, speed.
Run: python3 -I selftest_rule.py. One line per check; exit 1 on any unexpected result."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a5_eval as ev  # noqa: E402
import fixtures as fx  # noqa: E402

results = []


def check(name, cond):
    results.append(cond)
    print(("ok   " if cond else "FAIL ") + name)


def raises(exc, fn, *a):
    try:
        fn(*a)
    except exc:
        return True
    return False


def main():
    ref = ev.Reference()
    check("gated file refused when its sha256 is wrong", raises(ev.Stop, ev.Reference, ev.HERE / "a5-gated-steps.json", ev.HERE / "oracle", "0" * 64))
    check("gated file missing -> typed STOP, no traceback", raises(ev.Stop, ev.Reference, ev.HERE / "nope.json"))
    check("committed oracle inputs verified against the hashes frozen in the gated file", len(ref.verified) == 15)
    # A5 rule on stand-ins
    for label, d in (("ref cpu-t12", "cpu/clean"), ("t6", "cpu/threads6"), ("cuda", "cuda/follow")):
        for p in ev.PROMPTS:
            run = fx.make_run(ref, d, p, "forced")
            r = ev.eval_a5(run, ref.block(p)["prompt_ids_sha256"], ref, p)
            check(f"A5 {label} {p}: {r['n_gated']} gated, max dev {r['max_logit_dev']:.3f}", r["ok"])
    check("gated counts equal REPLY-41 (122,122,116,125,122)", [ref.block(p)["n_gated"] for p in ev.PROMPTS] == [122, 122, 116, 125, 122])
    sha1 = ref.block("P1")["prompt_ids_sha256"]
    g = ref.block("P1")["gated"][10]
    base = fx.make_run(ref, "cpu/clean", "P1", "forced")
    ex = ref.block("P1")["excluded"][0]["step"]

    def mut(fn, p="P1", run=None):
        r = fx.clone(run or fx.make_run(ref, "cpu/clean", p, "forced"))
        fn(r)
        return ev.eval_a5(r, ref.block(p)["prompt_ids_sha256"], ref, p)

    def logit(r, d):
        r["steps"][g["step"]]["forced_logit"] += d
        r["steps"][g["step"]]["argmax_logit"] += d
        r["steps"][g["step"]]["top32"][0][1] += d
    check("clause top-1 differs at a gated step -> FAIL", not mut(lambda r: r["steps"][g["step"]].__setitem__("argmax", g["reference_token"] + 1))["ok"])
    check("clause logit 1.01 x tolerance -> FAIL", not mut(lambda r: logit(r, g["tolerance"] * 1.01))["ok"])
    check("clause logit 0.99 x tolerance -> PASS", mut(lambda r: logit(r, g["tolerance"] * 0.99))["ok"])
    check("clause excluded step broken -> PASS (not graded)", mut(lambda r: r["steps"][ex].__setitem__("argmax", 7))["ok"])
    check("clause forced_logit != argmax_logit at a gated step -> FAIL", not mut(lambda r: r["steps"][g["step"]].__setitem__("forced_logit", r["steps"][g["step"]]["forced_logit"] + 1e-3))["ok"])
    check("clause top32[0] disagrees with argmax_logit at a gated step -> FAIL", not mut(lambda r: r["steps"][g["step"]]["top32"][0].__setitem__(1, 0.0))["ok"])
    check("clause NaN logit at a gated step -> FAIL", not mut(lambda r: r["steps"][g["step"]].__setitem__("forced_logit", float("nan")))["ok"])
    check("clause 127 steps -> FAIL", not mut(lambda r: r["steps"].pop())["ok"])
    check("clause all 116 gated steps of P3 wrong -> exactly 116 fails", len(mut(lambda r: [r["steps"][x["step"]].__setitem__("argmax", x["reference_token"] + 1) for x in ref.block("P3")["gated"]], "P3")["fails"]) == 116)
    check("clause wrong prompt-id hash -> refused", raises(ev.Stop, ev.eval_a5, base, "0" * 64, ref, "P1"))
    # determinism, tf-vs-free, speed
    f1 = [fx.make_run(ref, "cuda/free", "P2", "free", 50, k) for k in range(3)]
    check("determinism: 3 identical runs pass", ev.eval_determinism(f1, 3)["ok"])
    f2 = fx.clone(f1); f2[1]["steps"][77]["hash"] = "0" * 16
    check("determinism: one row hash differs -> FAIL", not ev.eval_determinism(f2, 3)["ok"])
    f2 = fx.clone(f1); f2[2]["steps"][5]["argmax"] += 1
    check("determinism: one token differs -> FAIL", not ev.eval_determinism(f2, 3)["ok"])
    check("determinism: a single run, or fewer runs than required -> FAIL", not ev.eval_determinism(f1[:1], 3)["ok"] and not ev.eval_determinism(f1[:2], 3)["ok"])
    fo = fx.make_run(ref, "cuda/follow", "P2", "forced")
    check("tf-vs-free: cuda/free and cuda/follow rows agree up to the first divergence", ev.eval_tf_vs_free(f1[0], fo, ev.eval_free(f1, ref, "P2")[0]["first_divergence"])["ok"])
    f2 = fx.clone(f1[0]); f2["steps"][3]["hash"] = "0" * 16
    check("tf-vs-free: a differing early row -> FAIL", not ev.eval_tf_vs_free(f2, fo, 40)["ok"])
    fdiv = ev.eval_free([fx.make_run(ref, "cuda/free", "P3", "free")], ref, "P3")[0]
    check(f"free-run: cuda/free P3 first divergence {fdiv['first_divergence']} (77 in compare/S1.md)", fdiv["first_divergence"] == 77)
    p1 = [fx.make_run(ref, "cuda/free", "P1", "free", ms, k) for k, ms in enumerate((50, 52, 51, 53, 49))]
    sp = ev.eval_speed(p1)
    check(f"speed: median {sp['median']:.2f} tok/s at 51 ms passes", sp["pass"] and sp["breakdown_ok"] and sp["syncs_ok"])
    check("speed: 56 ms/token (17.86 tok/s) fails", not ev.eval_speed([fx.make_run(ref, "cuda/free", "P1", "free", 56, k) for k in range(5)])["pass"])
    check("speed: 55.0 ms/token (18.18 tok/s) passes", ev.eval_speed([fx.make_run(ref, "cuda/free", "P1", "free", 55, k) for k in range(5)])["pass"])
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
