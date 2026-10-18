"""Checks that the oracle constants in host.rs equal a fresh run of q36_oracle.py.

Usage (from any directory):
    python3 -I qualification/q36/oracle/check_consts.py engine/src/family/qwen36/host.rs
The script puts its own directory on the import path, so `-I` works regardless of the current directory.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q36_oracle as o

text = open(sys.argv[1]).read()


def flat(name):
    m = re.search(r'const %s: [^=]*= (.*?);\n' % name, text, re.S)
    assert m, name
    return [float(x) for x in re.findall(r'-?\d+\.\d+(?:e-?\d+)?|-?\d+', m.group(1))]


out, state = o.delta_scenario()
mo, _, _ = o.moe_scenario()
lg, la, _ = o.model_scenario([1, 4, 2])
want = {
    'DELTA_OUT': [v for r in out for v in r],
    'DELTA_STATE': state,
    'ATTN_OUT': [v for r in o.attn_scenario() for v in r],
    'ATTN_OUT_ROT4': [v for r in o.attn_scenario(4) for v in r],
    'MOE_OUT': [v for r in mo for v in r],
    'MODEL_LOGITS': [v for r in lg for v in r],
    'MODEL_LAYER_0': [v for t in range(3) for v in la[t][0]],
    'MODEL_LAYER_1': [v for t in range(3) for v in la[t][1]],
}
bad = 0
for name, expected in want.items():
    got = flat(name)
    ok = len(got) == len(expected) and all(abs(a - b) <= 1e-9 for a, b in zip(got, expected))
    print(('ok   ' if ok else 'DIFF ') + name, len(got))
    bad += not ok
sys.exit(1 if bad else 0)
