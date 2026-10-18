"""A local fake of the chat server, for the self-test and the launch rehearsal of AGENT-1.

It follows `api/openapi.yaml` closely enough to drive `acc_a.py`: `/health`, `/ready`,
`/v1/models`, and `/v1/chat/completions` (plain and stream, bearer key, tool calls). It runs no
model. The mode selects a scripted fault. It is a test double of the server, not a Rust result.
"""

import argparse
import http.server
import json
import os
import sys
import threading
import time

MODES = ("pi-tools", "good", "quality-text", "wrong-sum", "leak", "badjson", "http500", "no-done",
         "big-prompt", "escape", "finish-before-calls", "final-length", "calls-in-final-chunk",
         "content-in-final-chunk", "usage-early", "no-role", "wrong-content-type", "cut-stream",
         "bad-models", "dot-path", "bad-utf8")


def words(messages):
    """A prompt token count: one per word plus 100 for the template."""
    return 100 + sum(len(str(m.get("content") or "").split()) for m in messages)


def plan_reply(messages, mode):
    """Returns ("tool", name, arguments_text, content) or ("text", content)."""
    tool_messages = sum(1 for m in messages if m.get("role") == "tool")
    if mode == "pi-tools":
        if tool_messages == 0:
            return ("tool", "write", json.dumps({"path": "hello.py", "content": "print(2 + 3)\n"}),
                    None)
        if tool_messages == 1:
            return ("tool", "bash", json.dumps({"command": "python3 hello.py"}), None)
        return ("text", "Done. hello.py prints 5.")
    if mode == "quality-text":
        return ("text", "I would write hello.py with print(2 + 3).")
    if mode == "leak":
        return ("text", "<tool_call>\n<function=write_file>\n</function>\n</tool_call>")
    if tool_messages == 0:
        if mode == "badjson":
            return ("tool", "write_file", '{"path": "hello.py", "content": ', None)
        path = {"escape": "../evil.py", "dot-path": "."}.get(mode, "hello.py")
        total = "2 + 4" if mode == "wrong-sum" else "2 + 3"
        return ("tool", "write_file",
                json.dumps({"path": path, "content": f"print({total})\n"}), None)
    if tool_messages == 1:
        return ("tool", "run_python", json.dumps({"path": "hello.py"}), None)
    return ("text", "Done. hello.py prints the sum.")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "fake-enginesrv"

    def log_message(self, *_):
        pass

    def _json(self, status, value):
        raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _log(self, body, status, outcome, error_code, finish, prompt, completion):
        """One request-log line in the shape of engine/src/server/reqlog.rs."""
        if not self.server.log:
            return
        line = json.dumps({
            "request_id": "chatcmpl-fake" if status == 200 else None,
            "model": self.server.model_id, "stream": bool(body.get("stream")),
            "status": status, "outcome": outcome, "error_code": error_code,
            "finish_reason": finish, "prompt_tokens": prompt, "completion_tokens": completion,
            "reused_tokens": 0, "queue_wait_ms": None, "time_to_first_token_ms": 1.0,
            "decode_tokens_per_second": None}) + "\n"
        if self.server.log == "-":
            sys.stderr.write(line)
            sys.stderr.flush()
        else:
            with open(self.server.log, "a", encoding="utf-8") as handle:
                handle.write(line)

    def _authorized(self):
        return self.headers.get("Authorization") == "Bearer " + self.server.key

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        if self.path == "/ready":
            if time.monotonic() < self.server.ready_at:
                return self._json(503, {"error": {"message": "loading", "code": "not_ready"}})
            return self._json(200, {"status": "ready"})
        if not self._authorized():
            return self._json(401, {"error": {"message": "bad key", "code": "unauthorized"}})
        if self.path == "/v1/models" and self.server.mode == "bad-models":
            self.wfile.write(b"NOT HTTP\r\n\r\n")  # a status line that http.client cannot parse
            self.close_connection = True
            return
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [
                {"id": self.server.model_id, "object": "model", "owned_by": "fake"}]})
        return self._json(404, {"error": {"message": "no route", "code": "not_found"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path != "/v1/chat/completions":
            return self._json(404, {"error": {"message": "no route", "code": "not_found"}})
        if not self._authorized():
            return self._json(401, {"error": {"message": "bad key", "code": "unauthorized"}})
        mode = self.server.mode
        if mode == "http500":
            self._log(body, 500, "error", "backend_error", None, 0, 0)
            return self._json(500, {"error": {"message": "Model execution failed.",
                                               "code": "backend_error"}})
        if self.server.dump:
            with open(self.server.dump, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(body) + "\n")
        time.sleep(self.server.prefill_delay)
        reply = plan_reply(body.get("messages") or [], mode)
        prompt = 1500 if mode == "big-prompt" else words(body.get("messages") or [])
        if reply[0] == "tool":
            _, name, arguments, content = reply
            finish, completion = "tool_calls", len(arguments) // 4 + 1
        else:
            _, content = reply
            name = arguments = None
            finish = "length" if mode == "final-length" else "stop"
            completion = len(content.split()) + 1
        usage = {"prompt_tokens": prompt, "completion_tokens": completion,
                 "total_tokens": prompt + completion, "reused_tokens": 0}
        call = {"id": "call_1", "type": "function",
                "function": {"name": name, "arguments": arguments}}
        self._log(body, 200, "ok", None, finish, prompt, completion)
        if not body.get("stream"):
            message = {"role": "assistant", "content": content}
            if name:
                message["tool_calls"] = [call]
            return self._json(200, {"id": "chatcmpl-fake", "object": "chat.completion",
                                    "created": 1, "model": self.server.model_id,
                                    "choices": [{"index": 0, "message": message,
                                                 "finish_reason": finish}], "usage": usage})
        self.send_response(200)
        self.send_header("Content-Type",
                         "text/plain" if mode == "wrong-content-type" else "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(text):
            raw = text.encode()
            self.wfile.write(f"{len(raw):x}\r\n".encode() + raw + b"\r\n")
            self.wfile.flush()

        def chunk(delta, finish_reason=None, usage_value=None):
            value = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 1,
                     "model": self.server.model_id,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
            if finish_reason is None:
                del value["choices"][0]["finish_reason"]
            if usage_value is not None:
                value["usage"] = usage_value
            send("data: " + json.dumps(value) + "\n\n")

        chunk({} if mode == "no-role" else {"role": "assistant"})
        send(": keep-alive\n\n")
        if mode == "bad-utf8":
            self.wfile.write(b"4\r\n\xff\xfe\n\n\r\n")  # a chunk that is not UTF-8
            self.wfile.flush()
        if mode == "cut-stream":
            # A chunk that announces 255 bytes and sends 7, then the connection ends.
            self.wfile.write(b"ff\r\ndata: {")
            self.wfile.flush()
            self.close_connection = True
            return
        final_usage = None if mode == "usage-early" else usage
        early_usage = usage if mode == "usage-early" else None
        time.sleep(self.server.chunk_delay)
        if name:
            # The shape of the merged server: a start entry and an arguments entry, each in its
            # own chunk, then a final chunk with an empty delta, the finish reason, and the usage.
            start = {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                     "function": {"name": name, "arguments": ""}}]}
            fragment = {"tool_calls": [{"index": 0, "function": {"arguments": arguments}}]}
            if mode == "finish-before-calls":
                chunk({}, finish, usage)
                chunk(start)
                chunk(fragment)
            elif mode == "calls-in-final-chunk":
                chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                       "function": {"name": name, "arguments": arguments}}]},
                      finish, usage)
            else:
                chunk(start)
                chunk(fragment, usage_value=early_usage)
                chunk({"content": "x"} if mode == "content-in-final-chunk" else {},
                      finish, final_usage)
        else:
            pieces = content.split(" ")
            for piece in pieces[:-1]:
                chunk({"content": piece + " "})
                time.sleep(self.server.chunk_delay)
            chunk({"content": pieces[-1]})
            chunk({}, finish, final_usage)
        if mode != "no-done":
            send("data: [DONE]\n\n")
        send("")  # the zero-length chunk ends the body


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, key, mode, ready_delay=0.0, prefill_delay=0.0,
                 chunk_delay=0.0, log=None, dump=None, model_id="fake-flash"):
        super().__init__(address, Handler)
        self.key, self.mode, self.log, self.dump = key, mode, log, dump
        self.model_id = model_id
        self.ready_at = time.monotonic() + ready_delay
        self.prefill_delay, self.chunk_delay = prefill_delay, chunk_delay


def start(key, mode="good", port=0, **options):
    """Starts a server on 127.0.0.1 in a thread. Returns (server, base_url)."""
    server = Server(("127.0.0.1", port), key, mode, **options)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=18791)
    parser.add_argument("--key-env", default="ENGINESRV_API_KEY")
    parser.add_argument("--mode", choices=MODES, default="good")
    parser.add_argument("--ready-delay", type=float, default=0.0)
    parser.add_argument("--log", default="-",
                        help="file for one JSON line per request; - is stderr, as the engine does")
    parser.add_argument("--model-id", default="fake-flash")
    parser.add_argument("--dump", help="append each request body as one JSON line (no headers)")
    args = parser.parse_args(argv)
    if args.port == 8788:
        print("port 8788 belongs to laya3", file=sys.stderr)
        return 1
    key = os.environ.get(args.key_env, "")
    if not key:
        print(f"{args.key_env} is not set", file=sys.stderr)
        return 1
    server = Server(("127.0.0.1", args.port), key, args.mode, args.ready_delay, log=args.log, dump=args.dump, model_id=args.model_id)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
