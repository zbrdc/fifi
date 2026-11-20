"""Full-request timing for a future paired inference measurement.

The caller owns admission, locking, service and socket validation. This module
does not issue network requests by itself, or identify a prefill/decode phase.
"""

import hashlib
import re
import time

from scripts.bench_p2p_prefill import sample_of


def measure_request(
    send,
    *,
    index,
    payload,
    expected_prompt_tokens,
    payload_sha256,
    clock=time.monotonic_ns,
):
    """Call a complete request and return its validated sample and interval.

    `send` must return a fully read JSON response. A raised error produces no
    completed interval; durations cover submission through response read.
    """
    if type(index) is not int or not 0 <= index < 10_000:
        raise ValueError("invalid request index")
    if type(expected_prompt_tokens) is not int or expected_prompt_tokens < 1:
        raise ValueError("expected prompt token count is required")
    if not isinstance(payload_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", payload_sha256
    ):
        raise ValueError("a pinned payload digest is required")
    if not isinstance(payload, bytes) or not payload or len(payload) > 16 * 1024 * 1024:
        raise ValueError("a bounded immutable request payload is required")
    if hashlib.sha256(payload).hexdigest() != payload_sha256:
        raise ValueError("request payload does not match its pinned digest")
    began = clock()
    reply = send(payload)
    ended = clock()
    if (
        type(began) is not int
        or type(ended) is not int
        or began <= 0
        or ended <= began
        or ended > 2**63 - 1
    ):
        raise ValueError("invalid monotonic request interval")
    try:
        sample = sample_of(reply, (ended - began) / 1_000_000_000)
        choice = reply["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ValueError("incomplete pair response")
        response_hash = hashlib.sha256(
            choice["message"]["content"].encode("utf-8")
        ).hexdigest()
    except Exception:
        # The benchmark parser can include arbitrary returned text in errors.
        raise ValueError("invalid pair response") from None
    if sample["prompt_tokens"] != expected_prompt_tokens:
        raise ValueError("unexpected prompt token count")
    if sample["reuse_class"] != "cold" or sample["reused_tokens"] != 0:
        raise ValueError("cold prompt reuse must be explicitly reported")
    # The fixed synthetic answer was verified by sample_of; don't persist it.
    del sample["answer"]
    request_id = reply.get("id")
    if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
        raise ValueError("response has no bounded request ID")
    interval = {
        "index": index,
        "status": "completed",
        "finish_reason": "stop",
        "prompt_tokens": sample["prompt_tokens"],
        "completion_tokens": sample["completion_tokens"],
        "reused_tokens": 0,
        "payload_sha256": payload_sha256,
        "response_sha256": response_hash,
        "request_id": request_id,
        "start_ns": began,
        "end_ns": ended,
    }
    return sample, interval
