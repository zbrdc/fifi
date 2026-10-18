import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q36_oracle as o

def arr(name, rows, doc):
    flat = isinstance(rows[0], float)
    t = "[f64; %d]" % len(rows) if flat else "[[f64; %d]; %d]" % (len(rows[0]), len(rows))
    def fmtrow(r): return "[" + ", ".join(repr(round(v, 9)) for v in r) + "]"
    body = fmtrow(rows) if flat else "[\n    " + ",\n    ".join(fmtrow(r) for r in rows) + ",\n]"
    return "/// %s\nconst %s: %s = %s;\n" % (doc, name, t, body)

out, state = o.delta_scenario()
print(arr("DELTA_OUT", out, "Oracle: DeltaNet output at steps 0, 1, 2 (`Delta` of `q36_oracle.py`)."))
print(arr("DELTA_STATE", state, "Oracle: recurrent state `[4 heads][2 value][2 key]` after step 2."))
print(arr("ATTN_OUT", o.attn_scenario(), "Oracle: attention output at positions 0, 1, 2."))
print(arr("ATTN_OUT_ROT4", o.attn_scenario(4), "Oracle: attention output at positions 0, 1, 2 with rotary_dim 4 (all of head_dim), so theta matters."))
mo, sel, margin = o.moe_scenario()
print(arr("MOE_OUT", mo, "Oracle: MoE output for the inputs 120, 121, 122; the experts chosen are %s." % sel))
lg, la, mg = o.model_scenario([1, 4, 2])
print(arr("MODEL_LOGITS", lg, "Oracle: logits for tokens 1, 4, 2."))
for l in range(2):
    print(arr("MODEL_LAYER_%d" % l, [la[t][l] for t in range(3)], "Oracle: residual after layer %d at steps 0, 1, 2." % l))
