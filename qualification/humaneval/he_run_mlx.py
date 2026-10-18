"""HumanEval runner: any mlx-lm-loadable model, with per-problem timing."""
from __future__ import annotations

import json
import os
import sys
import time

from mlx_lm import load, generate  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

INSTR = ("Complete the following Python function. Output ONLY a single "
         "```python code block containing the COMPLETE function "
         "(signature and body). No explanation.\n\n```python\n{prompt}```")
MAXTOK = 2000


def main(model_path, out_path, problems_path):
    model, tokenizer = load(os.path.expanduser(model_path))
    sampler = make_sampler(temp=0.0)
    problems = [json.loads(l) for l in open(problems_path)]
    with open(out_path, "w") as f:
        for i, p in enumerate(problems):
            t0 = time.perf_counter()
            text = tokenizer.apply_chat_template(
                [{"role": "user",
                  "content": INSTR.format(prompt=p["prompt"])}],
                tokenize=False, add_generation_prompt=True)
            out = generate(model, tokenizer, prompt=text, max_tokens=MAXTOK,
                           sampler=sampler, verbose=False)
            dt = time.perf_counter() - t0
            f.write(json.dumps({"task_id": p["task_id"], "response": out,
                                "seconds": round(dt, 1)}) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/164", flush=True)
    print("ALL DONE", flush=True)


def check_problems_path(path):
    try:
        with open(path, "rb") as f:
            f.read(1)
    except OSError as e:
        sys.exit(f"he_run_mlx: error: cannot read problems file {path}: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: he_run_mlx.py MODEL_DIR OUT PROBLEMS")
    check_problems_path(sys.argv[3])
    main(sys.argv[1], sys.argv[2], sys.argv[3])
