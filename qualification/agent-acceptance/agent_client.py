"""A minimal OpenAI-compatible chat client for the AGENT-1 acceptance runs.

It uses only the Python standard library. It sends one request and returns the parsed reply with
timings. It makes no claim about SDK compatibility: the pinned SDK fixtures do that.
This is a test client. The engine never runs it.
"""

import http.client
import json
import time
import urllib.parse

# Port 8788 belongs to the laya3 service. This client never connects to it.
FORBIDDEN_PORT = 8788
# The bearer key goes only to this host. The runs use a loopback address or an ssh tunnel to it.
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


class ClientError(Exception):
    """A usage error. No request was sent."""


def split_base_url(base_url):
    """Returns (host, port, path prefix). Refuses port 8788, non-loopback hosts, and https."""
    parts = urllib.parse.urlsplit(base_url)
    if parts.scheme != "http" or not parts.hostname:
        raise ClientError("the base URL must be http://host:port/prefix")
    if parts.hostname not in LOOPBACK_HOSTS:
        raise ClientError("the base URL host must be a loopback address (127.0.0.1, localhost, ::1); the key goes nowhere else")
    port = parts.port or 80
    if port == FORBIDDEN_PORT:
        raise ClientError("port 8788 belongs to laya3; this client never connects to it")
    return parts.hostname, port, parts.path.rstrip("/")


# A streamed rate needs an interval of at least this long. A complete tool call reaches the client
# as a burst, a few milliseconds from the first delta to the end of the reply, and the quotient of
# such a burst is not a decode speed. 0.1 s is above the jitter of one local burst and below any
# real incremental decode of 2 or more tokens at the speeds this engine targets.
MIN_RATE_INTERVAL_SECONDS = 0.1


def decode_rate(completion_tokens, first_text_delta_seconds, total_seconds, text_delta_count):
    """Streamed decode tokens per second over (total_seconds - first_text_delta_seconds), or None.

    A text delta holds content or reasoning. A delta that holds only tool calls does not count,
    because the server sends a complete tool call in one burst. The first token arrives with the
    first text delta, so the interval holds completion_tokens - 1 tokens. The interval ends at the
    end of the whole reply, not at the last delta. A rate needs at least 2 text deltas, at least 2
    completion tokens, and an interval of at least MIN_RATE_INTERVAL_SECONDS. Otherwise the result
    is None, which timings.tsv writes as an empty field, never as a number.
    """
    if first_text_delta_seconds is None or total_seconds is None:
        return None
    if not isinstance(text_delta_count, int) or text_delta_count < 2:
        return None
    interval = total_seconds - first_text_delta_seconds
    if not isinstance(completion_tokens, int) or completion_tokens < 2:
        return None
    if interval < MIN_RATE_INTERVAL_SECONDS:
        return None
    return (completion_tokens - 1) / interval


class StreamReader:
    """Folds server-sent events into one message and a list of framing problems."""

    def __init__(self, clock, started):
        self.clock = clock
        self.started = started
        self.problems = []
        self.chunks = []  # one dict per data event, in order
        self.keepalives = 0
        self.done = False
        self.error_event = None
        self.raw = []
        self._data = []
        self._event = None

    def feed_line(self, line):
        """Handles one line without its line break."""
        self.raw.append(line)
        if line == "":
            self._finish_event()
        elif line.startswith(":"):
            self.keepalives += 1
        elif line.startswith("event:"):
            self._event = line[6:].strip()
        elif line.startswith("data:"):
            self._data.append(line[5:].lstrip(" "))

    def _finish_event(self):
        if not self._data and self._event is None:
            return
        data, event = "\n".join(self._data), self._event
        self._data, self._event = [], None
        if event == "error":
            self.error_event = data
            return
        if data == "[DONE]":
            self.done = True
            return
        if self.done:
            self.problems.append("data after [DONE]")
        try:
            chunk = json.loads(data)
        except ValueError:
            self.problems.append("a data event is not JSON")
            return
        self.chunks.append({"seconds": self.clock() - self.started, "chunk": chunk})

    def message(self):
        """Returns the folded reply: content, reasoning, tool calls, finish reason, usage."""
        content, reasoning, calls = [], [], {}
        finish, usage = None, None
        role_first = False
        final_delta_empty = False
        delta_seconds = []
        text_delta_seconds = []
        final_index = len(self.chunks) - 1
        call_chunks = []
        usage_chunk = None
        finish_chunk = None
        for number, entry in enumerate(self.chunks):
            chunk = entry["chunk"]
            if number == 0:
                choices = chunk.get("choices") or [{}]
                role_first = (choices[0].get("delta") or {}).get("role") == "assistant"
            if number == final_index:
                final_choices = chunk.get("choices") or [{}]
                final_delta_empty = not (final_choices[0].get("delta") or {})
            if chunk.get("usage") is not None:
                usage, usage_chunk = chunk["usage"], number
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if choice.get("finish_reason"):
                    finish, finish_chunk = choice["finish_reason"], number
                has_text = False
                has_plain_text = False
                if delta.get("content"):
                    content.append(delta["content"])
                    has_text = has_plain_text = True
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                    has_text = has_plain_text = True
                if delta.get("tool_calls"):
                    call_chunks.append(number)
                    has_text = True
                    for part in delta["tool_calls"]:
                        call = calls.setdefault(
                            part.get("index", 0), {"id": None, "name": None, "arguments": ""}
                        )
                        call["id"] = part.get("id", call["id"])
                        function = part.get("function") or {}
                        call["name"] = function.get("name", call["name"])
                        call["arguments"] += function.get("arguments", "")
                if has_text:
                    delta_seconds.append(entry["seconds"])
                if has_plain_text:
                    text_delta_seconds.append(entry["seconds"])
        return {
            "content": "".join(content),
            "reasoning_content": "".join(reasoning),
            "tool_calls": [calls[index] for index in sorted(calls)],
            "finish_reason": finish,
            "usage": usage,
            "role_first": role_first,
            "final_delta_empty": final_delta_empty,
            "chunks": len(self.chunks),
            "final_chunk": final_index,
            "tool_call_chunks": call_chunks,
            "finish_chunk": finish_chunk,
            "usage_chunk": usage_chunk,
            "first_chunk_seconds": self.chunks[0]["seconds"] if self.chunks else None,
            "first_delta_seconds": delta_seconds[0] if delta_seconds else None,
            "last_delta_seconds": delta_seconds[-1] if delta_seconds else None,
            "text_delta_count": len(text_delta_seconds),
            "first_text_delta_seconds": text_delta_seconds[0] if text_delta_seconds else None,
        }


def get_json(base_url, path, key, timeout):
    """GET a JSON document. Returns (status, parsed body or None)."""
    host, port, prefix = split_base_url(base_url)
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        headers = {"Authorization": "Bearer " + key} if key else {}
        connection.request("GET", prefix + path, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        try:
            return response.status, json.loads(raw)
        except ValueError:
            return response.status, None
    finally:
        connection.close()


def chat(base_url, key, body, timeout, clock=time.monotonic):
    """POST one chat request and return a dict with the reply and the timings.

    The dict never holds the key. For a stream it holds the folded message, the framing problems,
    and the raw event lines. For a plain reply it holds the parsed message.
    """
    host, port, prefix = split_base_url(base_url)
    stream = bool(body.get("stream"))
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    result = {"stream": stream, "problems": []}
    try:
        started = clock()
        connection.request(
            "POST",
            prefix + "/chat/completions",
            body=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + key,
            },
        )
        response = connection.getresponse()
        result["status"] = response.status
        result["content_type"] = response.getheader("Content-Type") or ""
        result["headers_seconds"] = clock() - started
        if response.status == 200 and stream:
            reader = StreamReader(clock, started)
            while True:
                line = response.readline()
                if not line:
                    break
                try:
                    reader.feed_line(line.decode("utf-8").rstrip("\r\n"))
                except UnicodeDecodeError:
                    reader.problems.append("a line is not UTF-8")
            if reader._data:
                reader._finish_event()
            result["total_seconds"] = clock() - started
            result["message"] = reader.message()
            result["problems"] = list(reader.problems)
            result["keepalives"] = reader.keepalives
            result["done"] = reader.done
            result["error_event"] = reader.error_event
            result["raw"] = reader.raw
            if not result["content_type"].startswith("text/event-stream"):
                result["problems"].append("content type is not text/event-stream")
        else:
            raw = response.read()
            result["total_seconds"] = clock() - started
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
                result["problems"].append("the body is not JSON")
            result["json"] = parsed
            if response.status == 200 and isinstance(parsed, dict):
                choices = parsed.get("choices") or [{}]
                message = dict(choices[0].get("message") or {})
                result["message"] = {
                    "content": message.get("content") or "",
                    "reasoning_content": message.get("reasoning_content") or "",
                    "tool_calls": [
                        {
                            "id": call.get("id"),
                            "name": (call.get("function") or {}).get("name"),
                            "arguments": (call.get("function") or {}).get("arguments", ""),
                        }
                        for call in message.get("tool_calls") or []
                    ],
                    "finish_reason": choices[0].get("finish_reason"),
                    "usage": parsed.get("usage"),
                }
        return result
    finally:
        connection.close()
