"""ACC-B checker: reads the JSON event stream of one `pi --mode json -p` run and the server log.

Pass criteria (plan agent-1-plan.md, section 2, step B): pi exits 0; the transcript holds at least
one assistant tool call, one tool result, and a final assistant message with stop reason `stop`;
hello.py prints 5; the server log shows at least 2 requests, one per assistant message (so pi did
not retry), HTTP 200 with outcome `ok` on every request, and no request above the prompt limit.
The input is plain JSON lines, or lines that a launcher stamped as `<epoch seconds><TAB><json>`.
hello.py runs only in the container sandbox of sandbox.py (SANDBOX_IMAGE); without it the checker
stops before it reads anything.

Exit status: 0 pass, 1 usage error, 2 pipeline defect, 3 model quality miss, 67 sandbox image
missing.
A pass is a structural result. It makes no speed claim and does not qualify Phase 0B or 1B.
"""

import argparse
import json
import os
import pathlib
import sys

import accb_export
import reqlog_check
import sandbox as sandbox_module

MARKUP = ("<tool_call>", "</tool_call>", "<function=", "<parameter=", "<think>", "</think>")


class Parser(argparse.ArgumentParser):
    """A usage error is status 1, not argparse's 2 (2 means a pipeline defect here)."""

    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"usage error: {message}", file=sys.stderr)
        sys.exit(1)


def read_events(path):
    """Returns [(stamp or None, event)] from a plain or stamped JSON lines file.

    The transcript is untrusted (design 4.5): hostile JSON is recorded as an unparsable line,
    never a traceback. A deeply nested line raises RecursionError from the JSON decoder; it is
    caught with ValueError so the run reports a pipeline defect (E8).

    A line that starts with `ACCB-` is an entry-protocol line, not a pi event: the
    `ACCB-EXPORT <base64>` marker is extractor input (packet section 5.13), and any other
    printed `ACCB-` line (a forged `ACCB-EXIT`, say) is data (section 9: there is no
    ACCB-EXIT line). Such lines are skipped, never counted as unparsable events.

    The file is read with errors="replace" like the extractor, so invalid bytes are model
    data and parse as no event (a pipeline defect), never a UnicodeDecodeError traceback.
    Lines split on `\n` only: str.splitlines also splits on U+2028, U+2029, and U+0085, which
    would cut one event in half when a separator appears inside a JSON string. Valid JSON that
    is not an object (`[1]`, `"s"`, `5`) is not a pi event either: it is recorded as an
    unparsable line, never a later AttributeError.
    """
    events = []
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    for line in text.split("\n"):
        if not line.strip():
            continue
        stamp = None
        head, tab, rest = line.partition("\t")
        if tab:
            try:
                stamp, line = float(head), rest
            except ValueError:
                pass
        if line.startswith("ACCB-"):
            continue
        try:
            parsed = json.loads(line)
        except (ValueError, RecursionError):
            events.append((stamp, {"type": "unparsable", "line": accb_export.repr_cut(line)}))
            continue
        if not isinstance(parsed, dict):
            events.append((stamp, {"type": "unparsable", "line": accb_export.repr_cut(line)}))
            continue
        events.append((stamp, parsed))
    return events


def assistant_text(message):
    return "".join(c.get("text", "") for c in message.get("content", []) if c.get("type") == "text")


def program_output(workdir, sandbox):
    """Runs hello.py in the container sandbox. Returns its stripped stdout, or None."""
    target = pathlib.Path(workdir) / "hello.py"
    if not target.is_file():
        return None
    code, text = sandbox.run_python(target)
    return text.strip() if code == 0 else None


def per_message_seconds(events):
    """Seconds from each assistant message_start to its first update and to its end."""
    rows, current = [], None
    for stamp, event in events:
        kind = event.get("type")
        message = event.get("message") or {}
        if kind == "message_start" and message.get("role") == "assistant":
            current = {"start": stamp, "first_update": None}
        elif kind == "message_update" and current and current["first_update"] is None:
            current["first_update"] = stamp
        elif kind == "message_end" and message.get("role") == "assistant" and current:
            usage = message.get("usage") or {}
            if None not in (current["start"], current["first_update"], stamp):
                rows.append({"to_first_update": current["first_update"] - current["start"],
                             "to_end": stamp - current["start"],
                             "output_tokens": usage.get("output")})
            current = None
    return rows


def evaluate(events, exit_code, workdir, server_log, prompt_limit, min_requests, sandbox):
    """Returns a list of (name, ok, kind) checks. kind is pipeline or quality."""
    checks = []

    def add(name, ok, kind):
        checks.append((name, bool(ok), kind))

    ends = [e["message"] for _, e in events if e.get("type") == "message_end"]
    assistants = [m for m in ends if m.get("role") == "assistant"]
    add("pi_exit_0", exit_code == 0, "pipeline")
    add("transcript_parses", all(e.get("type") != "unparsable" for _, e in events), "pipeline")
    add("agent_settled", any(e.get("type") == "agent_settled" for _, e in events), "pipeline")
    add("no_retry_flag", not any(e.get("willRetry") for _, e in events
                                 if e.get("type") == "agent_end"), "pipeline")
    add("no_error_stop", not any(m.get("stopReason") in ("error", "aborted") for m in assistants),
        "pipeline")
    add("no_markup_in_text", not any(tag in assistant_text(m) for m in assistants for tag in MARKUP),
        "pipeline")
    calls = [c for m in assistants for c in m.get("content", []) if c.get("type") == "toolCall"]
    add("tool_arguments_are_objects", all(isinstance(c.get("arguments"), dict) for c in calls),
        "pipeline")
    results = [m for m in ends if m.get("role") == "toolResult"]
    add("has_tool_call", len(calls) >= 1, "quality")
    add("has_tool_result", len(results) >= 1, "quality")
    final = assistants[-1] if assistants else {}
    add("final_stop_with_text",
        final.get("stopReason") == "stop" and assistant_text(final).strip() != "", "quality")
    add("hello_py_prints_5", program_output(workdir, sandbox) == "5", "quality")
    if server_log:
        entries = reqlog_check.read_entries(server_log)
        add("server_log_requests_at_least_min", len(entries) >= min_requests, "pipeline")
        add("one_request_per_assistant_message", len(entries) == len(assistants), "pipeline")
        add("all_requests_200_ok", reqlog_check.all_served_ok(entries), "pipeline")
        add("prompt_tokens_within_limit",
            all(isinstance(e.get("prompt_tokens"), int) and e["prompt_tokens"] <= prompt_limit
                for e in entries), "pipeline")
    return checks


def main(argv=None):
    parser = Parser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--server-log", required=True,
                        help="the server request log: a pass needs the cross-check (design 4.5)")
    parser.add_argument("--prompt-limit", type=int, default=1800)
    parser.add_argument("--min-requests", type=int, default=2)
    parser.add_argument("--out", required=True, help="result directory (created)")
    parser.add_argument("--sandbox-image", default=os.environ.get("SANDBOX_IMAGE", ""),
                        help="digest-pinned image for hello.py (default: $SANDBOX_IMAGE)")
    parser.add_argument("--docker", default=os.environ.get("DOCKER", "docker"))
    parser.add_argument("--docker-host", default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    work = pathlib.Path(args.workdir)
    sandbox = sandbox_module.Sandbox(args.sandbox_image, docker=args.docker,
                                     host=args.docker_host, scratch=work.with_name(work.name + "-stage"))
    try:
        sandbox.preflight()
    except sandbox_module.SandboxError as error:
        print(str(error), file=sys.stderr)
        return error.exit_code
    try:
        events = read_events(args.transcript)
    except OSError as error:
        print(f"cannot read the transcript: {error}", file=sys.stderr)
        return 1
    try:
        checks = evaluate(events, args.exit_code, args.workdir, args.server_log,
                          args.prompt_limit, args.min_requests, sandbox)
    except sandbox_module.SandboxError as error:
        print(f"sandbox error: {error}", file=sys.stderr)
        return error.exit_code
    finally:
        sandbox.release()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "acc-b-checks.tsv", "w", encoding="utf-8") as handle:
        handle.write("check\tok\tkind\n")
        for name, ok, kind in checks:
            handle.write(f"{name}\t{ok}\t{kind}\n")
    verdict = "pass"
    if any(not ok and kind == "quality" for _, ok, kind in checks):
        verdict = "quality_miss"
    if any(not ok and kind == "pipeline" for _, ok, kind in checks):
        verdict = "pipeline_defect"
    rows = per_message_seconds(events)
    (out / "acc-b-result.json").write_text(json.dumps(
        {"verdict": verdict, "assistant_messages": len(rows) or None, "timings": rows,
         "hello_py_container": sandbox.last_container,
         "scope": "structural result; not a speed claim; not Phase 0B or Phase 1B"}, indent=2)
        + "\n", encoding="utf-8")
    print(f"ACC-B verdict: {verdict}")
    if sandbox.last_container:
        print(f"hello.py rerun in the sandbox container {sandbox.last_container}")
    return {"pass": 0, "pipeline_defect": 2, "quality_miss": 3}[verdict]


if __name__ == "__main__":
    sys.exit(main())
