"""ACC-A: the scripted tool-calling loop against one chat server (AGENT-1a acceptance).

The task: create hello.py that prints the sum of 2 and 3, then run it. The client sends the same
first request twice, once plain and once as a stream, then continues the plain conversation with
tool messages until the model stops. It checks the pipeline and records the model quality.

The model's code never runs on a host: the tool loop runs hello.py only in the container sandbox
of sandbox.py (SANDBOX_IMAGE, digest-pinned, no network). A missing image stops the run before
any model call.

Exit status: 0 pass, 1 usage error, 2 pipeline defect, 3 model quality miss on every attempt,
4 prompt budget exceeded, 67 sandbox image missing. A pass is a structural result. It makes no speed claim and it does not
qualify Phase 0B or Phase 1B.
"""

import argparse
import hashlib
import http.client
import json
import os
import pathlib
import shutil
import socket
import sys

import agent_client
import reqlog_check
import sandbox as sandbox_module

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a text file in the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "relative file path"},
                    "content": {"type": "string", "description": "file text"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run a Python file in the working directory and return its output.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "relative file path"}},
                "required": ["path"],
            },
        },
    },
]
SYSTEM = "You are a coding agent. Use the tools to do the task. Keep replies short."
TASK = "Create hello.py that prints the sum of 2 and 3, then run it."
MARKUP = ("<tool_call>", "</tool_call>", "<function=", "<parameter=", "<think>", "</think>")

ENGINES = ("flash", "q36", "q36cuda")
EXIT_PASS, EXIT_USAGE, EXIT_PIPELINE, EXIT_QUALITY, EXIT_BUDGET = 0, 1, 2, 3, 4
EXIT_SANDBOX = sandbox_module.EXIT_SANDBOX


class Budget(Exception):
    """A response reported more prompt tokens than the budget allows."""


def sha256_file(path):
    """Returns the SHA-256 of a file as hex."""
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def check_args(tool_name, arguments_text):
    """Returns (parses, problems). `parses` is False when the text is not a JSON object."""
    try:
        arguments = json.loads(arguments_text)
    except ValueError:
        return False, ["arguments are not JSON"]
    if not isinstance(arguments, dict):
        return False, ["arguments are not a JSON object"]
    spec = next((t["function"] for t in TOOLS if t["function"]["name"] == tool_name), None)
    if spec is None:
        return True, [f"unknown tool {tool_name}"]
    schema = spec["parameters"]
    problems = []
    for key in schema["required"]:
        if key not in arguments:
            problems.append(f"missing argument {key}")
    for key, value in arguments.items():
        if key not in schema["properties"]:
            problems.append(f"unexpected argument {key}")
        elif not isinstance(value, str):
            problems.append(f"argument {key} is not a string")
    return True, problems


def call_shape(calls):
    """The tool names and the sorted argument keys of a list of calls, for comparison."""
    shape = []
    for call in calls:
        try:
            keys = sorted(json.loads(call["arguments"]))
        except (ValueError, TypeError):
            keys = None
        shape.append((call["name"], keys))
    return shape


def safe_path(workdir, text):
    """Returns the absolute path of a relative file below workdir, or None."""
    if not isinstance(text, str) or not text or os.path.isabs(text):
        return None
    target = (workdir / text).resolve()
    root = workdir.resolve()
    if target != root and root not in target.parents:
        return None
    return target


def run_tool(workdir, name, arguments_text, sandbox):
    """Runs one tool call. Returns the text that goes back to the model.

    write_file only writes a file. run_python runs the file in the container sandbox.
    """
    try:
        arguments = json.loads(arguments_text)
    except ValueError:
        return "error: arguments are not JSON"
    if not isinstance(arguments, dict):
        return "error: arguments are not an object"
    try:
        return _run_tool_checked(workdir, name, arguments, sandbox)
    except (OSError, ValueError) as error:  # ValueError includes UnicodeError
        # The path or the content came from the model. A bad value is a failed tool call, not a crash.
        return f"error: the tool call failed: {type(error).__name__}"


def _run_tool_checked(workdir, name, arguments, sandbox):
    target = safe_path(workdir, arguments.get("path"))
    if target is None:
        return "error: the path must be a relative path below the working directory"
    if name == "write_file":
        content = arguments.get("content")
        if not isinstance(content, str):
            return "error: content must be a string"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        os.chmod(target, 0o644)
        return f"ok: wrote {len(content.encode('utf-8'))} bytes to {arguments['path']}"
    if name == "run_python":
        if not target.is_file():
            return "error: no such file"
        code, text = sandbox.run_python(target)
        if code is None:
            return text
        return f"exit {code}\n{text[:2000]}"
    return f"error: unknown tool {name}"


def program_output(workdir, sandbox):
    """Runs hello.py in the container sandbox. Returns its stripped stdout, or None."""
    target = workdir / "hello.py"
    if not target.is_file():
        return None
    code, text = sandbox.run_python(target)
    return text.strip() if code == 0 else None


class Attempt:
    """The records and the checks of one attempt."""

    def __init__(self, number, out_dir, args, key):
        self.number = number
        self.out_dir = out_dir
        self.args = args
        self.key = key
        self.requests = []
        self.checks = []

    def check(self, name, ok, kind, detail=""):
        """Records a check. `kind` is "pipeline" or "quality"."""
        self.checks.append({"name": name, "ok": bool(ok), "kind": kind, "detail": detail})
        return bool(ok)

    def failed(self, kind):
        return [c for c in self.checks if not c["ok"] and c["kind"] == kind]

    def send(self, label, body, stream):
        """Sends one request, records it, and enforces the prompt budget."""
        body = dict(body, stream=stream)
        try:
            reply = agent_client.chat(self.args.base_url, self.key, body, self.args.timeout)
        except (OSError, socket.timeout, http.client.HTTPException) as error:
            self.check("request_completes", False, "pipeline", f"{label}: {error!r}")
            return None
        number = len(self.requests) + 1
        message = reply.get("message") or {}
        usage = message.get("usage") or {}
        first = message.get("first_delta_seconds")
        record = {
            "attempt": self.number,
            "n": number,
            "label": label,
            "stream": stream,
            "status": reply.get("status"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "headers_seconds": reply.get("headers_seconds"),
            "first_delta_seconds": first if stream else None,
            "total_seconds": reply.get("total_seconds"),
            "decode_tok_s": agent_client.decode_rate(
                usage.get("completion_tokens"), message.get("first_text_delta_seconds"),
                reply.get("total_seconds"), message.get("text_delta_count"),
            )
            if stream
            else None,
            "finish_reason": message.get("finish_reason"),
            "keepalives": reply.get("keepalives"),
        }
        self.requests.append(record)
        if stream and reply.get("raw") is not None:
            raw_path = self.out_dir / f"sse-{self.number}-{number}.txt"
            raw_path.write_text("\n".join(reply["raw"]) + "\n", encoding="utf-8")
        with open(self.out_dir / "transcript.jsonl", "a", encoding="utf-8") as handle:
            summary = {k: v for k, v in reply.items() if k != "raw"}
            handle.write(
                json.dumps({"record": record, "request": body, "reply": summary}) + "\n"
            )
        prompt = usage.get("prompt_tokens")
        if isinstance(prompt, int) and prompt > self.args.prompt_limit:
            raise Budget(f"{label} reported {prompt} prompt tokens")
        return reply

    def reply_checks(self, label, reply, stream):
        """Pipeline checks that hold for every reply. Returns True when the reply is usable."""
        if not self.check(f"{label}_http_200", reply.get("status") == 200, "pipeline",
                          f"status {reply.get('status')}"):
            return False
        if not self.check(f"{label}_framing", not reply.get("problems"), "pipeline",
                          "; ".join(reply.get("problems") or [])):
            return False
        message = reply.get("message") or {}
        usage = message.get("usage")
        self.check(
            f"{label}_usage",
            isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int)
            and isinstance(usage.get("completion_tokens"), int),
            "pipeline",
        )
        self.check(
            f"{label}_no_markup_leak",
            not any(tag in (message.get("content") or "") for tag in MARKUP),
            "pipeline",
            "tool call or think markup in content",
        )
        for call in message.get("tool_calls") or []:
            parses, _ = check_args(call.get("name"), call.get("arguments") or "")
            self.check(f"{label}_arguments_json", parses, "pipeline", repr(call.get("arguments")))
        if stream:
            self.check(f"{label}_done", reply.get("done") is True, "pipeline", "no [DONE]")
            self.check(f"{label}_no_error_event", reply.get("error_event") is None, "pipeline")
            self.check(f"{label}_role_first", message.get("role_first") is True, "pipeline")
            final = message.get("final_chunk")
            self.check(
                f"{label}_tool_calls_in_own_chunks",
                all(index < final for index in message.get("tool_call_chunks") or []),
                "pipeline",
                f"tool call chunks {message.get('tool_call_chunks')}, final chunk {final}",
            )
            self.check(f"{label}_final_delta_empty", message.get("final_delta_empty") is True,
                       "pipeline", "the final chunk holds only the finish reason and the usage")
            self.check(
                f"{label}_finish_in_final_chunk",
                message.get("finish_chunk") == message.get("final_chunk"),
                "pipeline",
                f"finish chunk {message.get('finish_chunk')}, final chunk {message.get('final_chunk')}",
            )
            self.check(
                f"{label}_usage_in_final_chunk",
                message.get("usage_chunk") == message.get("final_chunk"),
                "pipeline",
            )
        return not self.failed("pipeline")

    def verdict(self):
        if self.failed("pipeline"):
            return "pipeline_defect"
        if self.failed("quality"):
            return "quality_miss"
        return "pass"


def run_attempt(number, args, key, out_dir, workdir, sandbox):
    """Runs the scripted loop once. Returns the Attempt."""
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    os.chmod(workdir, 0o755)
    attempt = Attempt(number, out_dir, args, key)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    base = {
        "model": args.model,
        "tools": TOOLS,
        "temperature": 0.2,
        "max_tokens": 256,
        "x-thinking": "off",
    }
    first = attempt.send("r1", dict(base, messages=messages), stream=False)
    if first is None or not attempt.reply_checks("r1", first, stream=False):
        return attempt
    message = first["message"]
    calls = message["tool_calls"]
    if not attempt.check("r1_finish_tool_calls", message["finish_reason"] == "tool_calls" and calls,
                         "quality", f"finish {message['finish_reason']}"):
        return attempt
    attempt.check("r1_tool_names_listed",
                  all(c["name"] in {t["function"]["name"] for t in TOOLS} for c in calls), "quality")
    for call in calls:
        _, problems = check_args(call["name"], call["arguments"])
        attempt.check("r1_arguments_match_schema", not problems, "quality", "; ".join(problems))
    # The same request as a stream.
    streamed = attempt.send("r1s", dict(base, messages=messages), stream=True)
    if streamed is None or not attempt.reply_checks("r1s", streamed, stream=True):
        return attempt
    s_message = streamed["message"]
    attempt.check("r1s_finish_tool_calls", s_message["finish_reason"] == "tool_calls", "quality")
    attempt.check(
        "r1s_matches_r1_name_and_keys",
        call_shape(s_message["tool_calls"]) == call_shape(calls),
        "quality",
        f"stream {call_shape(s_message['tool_calls'])}, plain {call_shape(calls)}",
    )
    # Continue the plain conversation.
    reply, requests_used = first, 1
    while True:
        message = reply["message"]
        if message["finish_reason"] != "tool_calls":
            attempt.check("final_content_not_empty", bool(message["content"].strip()), "quality")
            attempt.check("final_finish_stop", message["finish_reason"] == "stop", "quality",
                          f"finish {message['finish_reason']}")
            break
        if requests_used >= args.max_requests:
            attempt.check("loop_ends_within_limit", False, "quality", f"{requests_used} requests")
            break
        messages.append({
            "role": "assistant",
            "content": message["content"] or None,
            "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in message["tool_calls"]
            ],
        })
        for call in message["tool_calls"]:
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": run_tool(workdir, call["name"], call["arguments"], sandbox)})
        reply = attempt.send(f"r{requests_used + 1}", dict(base, messages=messages), stream=False)
        requests_used += 1
        if reply is None or not attempt.reply_checks(f"r{requests_used}", reply, stream=False):
            return attempt
    printed = program_output(workdir, sandbox)
    attempt.check("hello_py_prints_5", printed == "5", "quality", f"output {printed!r}")
    return attempt


def write_outputs(out_dir, attempts, verdict, args):
    rows = [r for a in attempts for r in a.requests]
    columns = ["attempt", "n", "label", "stream", "status", "prompt_tokens", "completion_tokens",
               "headers_seconds", "first_delta_seconds", "total_seconds", "decode_tok_s",
               "finish_reason", "keepalives"]
    with open(out_dir / "timings.tsv", "w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join("" if row[c] is None else str(row[c]) for c in columns) + "\n")
    with open(out_dir / "checks.tsv", "w", encoding="utf-8") as handle:
        handle.write("attempt\tcheck\tok\tkind\tdetail\n")
        for attempt in attempts:
            for c in attempt.checks:
                handle.write(f"{attempt.number}\t{c['name']}\t{c['ok']}\t{c['kind']}\t"
                             f"{c['detail'].replace(chr(9), ' ').replace(chr(10), ' ')}\n")
    summary = {
        "verdict": verdict,
        "attempts": [{"attempt": a.number, "verdict": a.verdict(), "requests": len(a.requests)}
                     for a in attempts],
        "base_url": args.base_url,
        "model": args.model,
        "engine": args.engine,
        "requests_sent": len(rows),
        "scope": "structural result; not a speed claim; not Phase 0B or Phase 1B",
    }
    if args.engine == "q36cuda":
        # The CUDA prefill is token by token, so the per-request timeout is a recorded figure.
        summary["timeout_seconds"] = args.timeout
    (out_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def count_log_lines(path, key_name):
    """Counts the JSON lines of the server log that hold `key_name`."""
    return len(reqlog_check.read_entries(path, key_name))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", required=True, help="http://127.0.0.1:18791/v1")
    parser.add_argument("--key-env", default="ENGINESRV_API_KEY",
                        help="name of the environment variable that holds the bearer key")
    parser.add_argument("--model", default="auto",
                        help="model id (it must be listed by /v1/models), or auto to use the first")
    parser.add_argument("--engine", choices=ENGINES, default="flash",
                        help="engine label that result.json records (flash, q36, or q36cuda)")
    parser.add_argument("--out", required=True, help="evidence directory (created)")
    parser.add_argument("--workdir", required=True, help="scratch directory for the tool files")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-requests", type=int, default=4)
    parser.add_argument("--prompt-limit", type=int, default=1400)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--server-log", help="server stderr log with one JSON line per request")
    parser.add_argument("--log-key", default="finish_reason",
                        help="JSON key that marks a request line in the server log")
    parser.add_argument("--sandbox-image", default=os.environ.get("SANDBOX_IMAGE", ""),
                        help="digest-pinned image for the model's code (default: $SANDBOX_IMAGE)")
    parser.add_argument("--docker-host", default=None,
                        help="the rootless docker socket (default: SANDBOX_DOCKER_HOST or "
                             "unix:///run/user/<uid>/docker.sock)")
    parser.add_argument("--docker", default=os.environ.get("DOCKER", "docker"),
                        help="the docker client (default: $DOCKER or docker)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    key = os.environ.get(args.key_env, "")
    if not key:
        print(f"the environment variable {args.key_env} is not set", file=sys.stderr)
        return EXIT_USAGE
    try:
        agent_client.split_base_url(args.base_url)
    except agent_client.ClientError as error:
        print(str(error), file=sys.stderr)
        return EXIT_USAGE
    workdir = pathlib.Path(args.workdir)
    sandbox = sandbox_module.Sandbox(
        args.sandbox_image, docker=args.docker, host=args.docker_host,
        scratch=workdir.with_name(workdir.name + "-stage"))
    try:
        sandbox.preflight()  # before any request: no model call may produce code without it
    except sandbox_module.SandboxError as error:
        print(str(error), file=sys.stderr)
        return error.exit_code
    if workdir.exists() and (not workdir.is_dir() or any(workdir.iterdir())):
        print(f"--workdir {workdir} exists and is not empty; the run deletes it, so give a new one",
              file=sys.stderr)
        return EXIT_USAGE
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        status, models = agent_client.get_json(args.base_url, "/models", key, 60)
    except (OSError, http.client.HTTPException) as error:
        print(f"GET /models failed: {error!r}", file=sys.stderr)
        write_outputs(out_dir, [], "pipeline_defect", args)
        return EXIT_PIPELINE
    (out_dir / "models.json").write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    if args.model == "auto":
        data = (models or {}).get("data") or []
        if status != 200 or not data:
            print("GET /models returned no model", file=sys.stderr)
            write_outputs(out_dir, [], "pipeline_defect", args)
            return EXIT_PIPELINE
        args.model = data[0]["id"]
    else:
        listed = [m.get("id") for m in ((models or {}).get("data") or []) if isinstance(m, dict)]
        if status != 200 or args.model not in listed:
            print(f"GET /models does not list the model {args.model}: {listed}", file=sys.stderr)
            write_outputs(out_dir, [], "pipeline_defect", args)
            return EXIT_PIPELINE
    attempts, verdict = [], "quality_miss"
    try:
        for number in range(1, args.attempts + 1):
            attempt = run_attempt(number, args, key, out_dir, workdir, sandbox)
            attempts.append(attempt)
            if attempt.verdict() == "pass":
                verdict = "pass"
                break
            if attempt.verdict() == "pipeline_defect":
                verdict = "pipeline_defect"
                break
    except Budget as error:
        write_outputs(out_dir, attempts, "budget_exceeded", args)
        print(f"prompt budget exceeded: {error}", file=sys.stderr)
        return EXIT_BUDGET
    except sandbox_module.SandboxError as error:
        write_outputs(out_dir, attempts, "sandbox_error", args)
        print(f"sandbox error: {error}", file=sys.stderr)
        return error.exit_code
    finally:
        sandbox.release()
    if args.server_log:
        sent = sum(len(a.requests) for a in attempts)
        logged = count_log_lines(args.server_log, args.log_key)
        entries = reqlog_check.read_entries(args.server_log, args.log_key)
        ok = logged == sent
        served = reqlog_check.all_served_ok(entries)
        if attempts:
            attempts[-1].check("server_log_one_line_per_request", ok, "pipeline",
                               f"{sent} requests sent, {logged} log lines")
            attempts[-1].check("server_log_all_200_ok", served, "pipeline",
                               reqlog_check.not_ok_detail(entries))
            if not (ok and served) and verdict == "pass":
                verdict = "pipeline_defect"
    write_outputs(out_dir, attempts, verdict, args)
    print(f"ACC-A verdict: {verdict}")
    return {"pass": EXIT_PASS, "pipeline_defect": EXIT_PIPELINE,
            "quality_miss": EXIT_QUALITY}[verdict]


if __name__ == "__main__":
    sys.exit(main())
