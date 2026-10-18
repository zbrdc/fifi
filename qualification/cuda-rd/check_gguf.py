#!/usr/bin/env python3
"""RD-10b Step 3 (CPU, about a minute): size and SHA-256 of the GGUF against the values frozen in a5-gated-steps.json.
Usage: check_gguf.py GGUF   Prints JSON {path,size,sha256}; exit 0 equal, 2 STOP (typed)."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a5_eval as ev  # noqa: E402

try:
    frozen = ev.read_json(ev.HERE / "a5-gated-steps.json")["gguf"]
    p = Path(sys.argv[1])
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    res = {"path": str(p), "size": p.stat().st_size, "sha256": h.hexdigest()}
except (OSError, IndexError, ev.Stop) as e:
    print(f"STOP: {e}")
    sys.exit(2)
print(json.dumps(res))
if res["size"] != frozen["size"] or res["sha256"] != frozen["sha256"]:
    print(f"STOP: GGUF differs from the frozen identity {frozen}")
    sys.exit(2)
