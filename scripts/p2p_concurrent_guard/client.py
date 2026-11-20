"""Bounded local HTTP request primitive; no command-line or live host adapter.

The parent must enforce an absolute deadline, contain the client process and
verify server-side idle after completion or cancellation. A socket timeout or
an empty client scope does not prove that server-side work has drained.
"""

import http.client
import json
import time

from .core import Interval

HOST = "127.0.0.1"
PORT = 18793
PATH = "/v1/chat/completions"
MODEL = "fifi-q36-2080ti"
MAX_BODY = 1_048_576
MAX_OUTPUT_TOKENS = 128
PROMPT = "Explain a simple approach for testing a function that sorts a list."


def request_once(key: str, *, connection_factory=http.client.HTTPConnection, clock=time.monotonic_ns):
    """One fixed non-streaming request; never retain response content or auth."""
    if not isinstance(key, str) or not key or "\r" in key or "\n" in key:
        raise ValueError("invalid authorization")
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": False,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0,
        },
        separators=(",", ":"),
    ).encode()
    connection = connection_factory(HOST, PORT, timeout=5)
    started = clock()
    try:
        connection.request(
            "POST", PATH, body=body,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        # http.client does not follow redirects. Reject them explicitly.
        if response.status != 200:
            raise ValueError("request did not return 200")
        data = response.read(MAX_BODY + 1)
        ended = clock()
        if len(data) > MAX_BODY:
            raise ValueError("response exceeds limit")
        parsed = json.loads(data)
        if not isinstance(parsed, dict):
            raise ValueError("invalid response")
        usage = parsed.get("usage")
        if not isinstance(usage, dict):
            raise ValueError("missing usage")
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        choices, request_id = parsed.get("choices"), parsed.get("id")
        if (
            type(prompt) is not int or prompt <= 0
            or type(completion) is not int or not 0 < completion <= MAX_OUTPUT_TOKENS
            or not isinstance(choices, list) or len(choices) != 1
            or not isinstance(choices[0], dict) or not choices[0].get("finish_reason")
            or not isinstance(request_id, str) or not request_id or len(request_id) > 256
        ):
            raise ValueError("invalid completed response")
        interval = Interval(started, ended)
        if not interval.valid():
            raise ValueError("invalid timing")
        return {"interval": interval, "prompt_tokens": prompt, "completion_tokens": completion}
    finally:
        connection.close()
