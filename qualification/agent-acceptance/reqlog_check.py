"""Checks of the server request log (`docs/design/server.md` "Request log").

The server writes one JSON line on standard error for each chat request that has a prompt, with
the keys in KEYS. Lines that are not JSON, such as startup errors, are skipped.
"""

import json
import pathlib

KEYS = ("request_id", "model", "stream", "status", "outcome", "error_code", "finish_reason",
        "prompt_tokens", "completion_tokens", "reused_tokens", "queue_wait_ms",
        "time_to_first_token_ms", "decode_tokens_per_second")


def read_entries(path, key="prompt_tokens"):
    """Returns the JSON objects of the log file that hold `key`."""
    entries = []
    for line in pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and key in entry:
            entries.append(entry)
    return entries


def all_served_ok(entries):
    """True when every request got HTTP 200 and ended with outcome `ok`."""
    return all(e.get("status") == 200 and e.get("outcome") == "ok" for e in entries)


def not_ok_detail(entries):
    return "; ".join(f"line {n}: status {e.get('status')!r} outcome {e.get('outcome')!r}"
                     for n, e in enumerate(entries, 1)
                     if not (e.get("status") == 200 and e.get("outcome") == "ok"))
