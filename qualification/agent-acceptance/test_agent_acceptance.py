"""Self-test of the ACC-A client against the local fake server. Run, from this directory:
python3 -I -m unittest discover -s . -p 'test_*.py'
(`-I` hides the current directory, so a plain `-m unittest test_x` needs `-I` left out.)
It needs no network beyond 127.0.0.1, no model, and no container: the model's code runs in the
sandbox of sandbox.py, which these tests point at fake_docker.py. They test the client, not the
engine.
"""

import contextlib
import io
import json
import os
import pathlib
import tempfile
import unittest
import unittest.mock

import acc_a
import agent_client
import fake_server
import sandbox
import smoke

KEY = "k-test-secret-7781"
HERE = pathlib.Path(__file__).parent
FAKE_DOCKER = str(HERE / "fake_docker.py")
IMAGE = "fake@sha256:" + "0" * 64


def fake_sandbox(tmp):
    return sandbox.Sandbox(IMAGE, docker=FAKE_DOCKER, scratch=pathlib.Path(tmp) / "stage")


@contextlib.contextmanager
def served(mode="good", **options):
    server, base_url = fake_server.start(KEY, mode, **options)
    try:
        yield base_url
    finally:
        server.shutdown()
        server.server_close()


def run_acc_a(base_url, tmp, extra=(), image=IMAGE):
    """Runs acc_a.main with the key in the environment. Returns (status, stderr text, out dir)."""
    out = pathlib.Path(tmp) / "out"
    argv = ["--base-url", base_url, "--out", str(out), "--workdir", str(pathlib.Path(tmp) / "work"),
            "--timeout", "20", "--sandbox-image", image, "--docker", FAKE_DOCKER, *extra]
    errors = io.StringIO()
    old = os.environ.get("ENGINESRV_API_KEY")
    os.environ["ENGINESRV_API_KEY"] = KEY
    try:
        with contextlib.redirect_stderr(errors), contextlib.redirect_stdout(io.StringIO()):
            status = acc_a.main(argv)
    finally:
        if old is None:
            del os.environ["ENGINESRV_API_KEY"]
        else:
            os.environ["ENGINESRV_API_KEY"] = old
    return status, errors.getvalue(), out


def checks(out):
    """Returns {check name: ok} from checks.tsv (the last attempt wins)."""
    rows = {}
    for line in (out / "checks.tsv").read_text().splitlines()[1:]:
        _, name, ok, *_ = line.split("\t")
        rows[name] = ok == "True"
    return rows


def folded_stream(times, chunks):
    """Folds `chunks` read at the clock `times` (seconds since the request start) into a message."""
    clock = iter(times)
    reader = agent_client.StreamReader(lambda: next(clock), 0.0)
    for chunk in chunks:
        for line in ("data: " + json.dumps(chunk), ""):
            reader.feed_line(line)
    return reader.message()


ROLE = {"choices": [{"index": 0, "delta": {"role": "assistant"}}]}
FINAL = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}


def text_chunk(text):
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


def call_chunk(name, arguments):
    function = {"name": name, "arguments": arguments}
    return {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "c1", "type": "function", "function": function}]}}]}


def rate_of(message, completion_tokens, total_seconds):
    return agent_client.decode_rate(completion_tokens, message["first_text_delta_seconds"],
                                    total_seconds, message["text_delta_count"])


class DecodeRate(unittest.TestCase):
    def test_the_rate_is_tokens_after_the_first_over_the_time_after_the_first_text_delta(self):
        # 11 tokens: 10 after the first, over 4.0 - 2.0 = 2 seconds, with 3 text deltas.
        self.assertEqual(agent_client.decode_rate(11, 2.0, 4.0, 3), 5.0)

    def test_an_incremental_text_stream_gets_a_rate(self):
        message = folded_stream([0.5, 1.0, 1.5, 2.0, 2.5],
                                [ROLE, text_chunk("a"), text_chunk("b"), text_chunk("c"), FINAL])
        self.assertEqual((message["text_delta_count"], message["first_text_delta_seconds"]), (3, 1.0))
        self.assertEqual(rate_of(message, 11, 3.0), 5.0)  # 10 tokens over 3.0 - 1.0 s

    def test_a_burst_tool_call_stream_gets_no_rate_even_when_the_reply_ends_late(self):
        # The two tool-call chunks arrive together at 158.0 s and the reply ends at 158.5 s: the
        # interval is 0.5 s, above the minimum, but the deltas hold no text.
        message = folded_stream([158.0, 158.0, 158.0, 158.5],
                                [ROLE, call_chunk("write_file", ""), call_chunk("write_file", "{}"), FINAL])
        self.assertEqual((message["text_delta_count"], message["first_text_delta_seconds"]), (0, None))
        self.assertIsNone(rate_of(message, 45, 158.5))

    def test_a_tool_call_chunk_does_not_count_as_a_text_delta(self):
        message = folded_stream([1.0, 2.0, 3.0, 4.0],
                                [ROLE, call_chunk("run_python", "{}"), text_chunk("a"), FINAL])
        self.assertEqual((message["text_delta_count"], message["first_text_delta_seconds"]), (1, 3.0))
        self.assertIsNone(rate_of(message, 20, 9.0))  # one text delta is not enough

    def test_a_short_interval_gets_no_rate(self):
        # Three text deltas, but the reply ends 3/32 s (0.094 s, below the 0.1 s minimum) later.
        message = folded_stream([2.0, 2.0, 2.0, 2.0], [ROLE, text_chunk("a"), text_chunk("b"), text_chunk("c")])
        self.assertIsNone(rate_of(message, 11, 2.09375))
        self.assertEqual(rate_of(message, 11, 2.125), 80.0)  # 1/8 s is above the minimum: 10 / 0.125

    def test_fewer_than_two_text_deltas_have_no_rate(self):
        self.assertIsNone(agent_client.decode_rate(45, 2.0, 4.0, 1))
        self.assertIsNone(agent_client.decode_rate(45, 2.0, 4.0, 0))
        self.assertIsNone(agent_client.decode_rate(45, 2.0, 4.0, None))

    def test_a_single_token_or_a_missing_or_reversed_time_has_no_rate(self):
        self.assertIsNone(agent_client.decode_rate(1, 2.0, 4.0, 5))
        self.assertIsNone(agent_client.decode_rate(10, 3.0, 3.0, 5))
        self.assertIsNone(agent_client.decode_rate(10, 4.0, 3.0, 5))
        self.assertIsNone(agent_client.decode_rate(10, None, 3.0, 5))
        self.assertIsNone(agent_client.decode_rate(10, 3.0, None, 5))

    def test_the_stream_reader_counts_the_deltas_that_hold_text_or_reasoning(self):
        reasoning = {"choices": [{"index": 0, "delta": {"reasoning_content": "r"}}]}
        message = folded_stream([1.0, 2.0, 3.0, 4.0, 5.0],
                                [ROLE, text_chunk("a"), reasoning, call_chunk("x", "{}"), FINAL])
        self.assertEqual(message["text_delta_count"], 2)
        self.assertEqual(message["first_text_delta_seconds"], 2.0)

    def test_the_timings_row_of_a_burst_tool_call_stream_has_an_empty_rate(self):
        # The fake server sends a tool call as a burst, as the engine does.
        with served() as url, tempfile.TemporaryDirectory() as tmp:
            status, errors, out = run_acc_a(url, tmp)
            self.assertEqual(status, acc_a.EXIT_PASS, errors)
            rows = [r.split("\t") for r in (out / "timings.tsv").read_text().splitlines()]
            header = rows[0]
            for row in rows[1:]:
                self.assertEqual(row[header.index("decode_tok_s")], "", row)

    def test_a_streamed_request_with_incremental_text_writes_the_rate_in_timings(self):
        # acc_a must pass the first TEXT delta, the end of the reply, and the text delta count.
        # A tool-call delta comes first (1.0 s), the first text delta later (1.5 s): the rate
        # counts from the text delta, so 10 tokens over 3.0 - 1.5 s.
        lines = [ROLE, call_chunk("x", "{}"), text_chunk("a"), text_chunk("b"), text_chunk("c"), FINAL]
        message = folded_stream([0.5, 1.0, 1.5, 2.0, 2.5, 2.75], lines)
        message["usage"] = {"prompt_tokens": 7, "completion_tokens": 11}
        reply = {"status": 200, "message": message, "headers_seconds": 0.1, "total_seconds": 3.0,
                 "keepalives": 0, "raw": None}
        args = type("Args", (), {"base_url": "http://127.0.0.1:1/v1", "timeout": 1.0,
                                 "prompt_limit": 1400})()
        with tempfile.TemporaryDirectory() as tmp, unittest.mock.patch.object(
                agent_client, "chat", return_value=reply):
            attempt = acc_a.Attempt(1, pathlib.Path(tmp), args, KEY)
            attempt.send("r1s", {"model": "m"}, True)
            self.assertAlmostEqual(attempt.requests[0]["decode_tok_s"], 10 / 1.5)
            attempt.send("r1", {"model": "m"}, False)
            self.assertIsNone(attempt.requests[1]["decode_tok_s"])
            # One text delta and a long reply: the text delta count gates the rate.
            single = folded_stream([0.5, 1.0, 2.0], [ROLE, text_chunk("a"), FINAL])
            single["usage"] = {"prompt_tokens": 7, "completion_tokens": 11}
            reply["message"] = single
            attempt.send("r1s", {"model": "m"}, True)
            self.assertIsNone(attempt.requests[2]["decode_tok_s"])


class StreamFolding(unittest.TestCase):
    def fold(self, lines):
        reader = agent_client.StreamReader(lambda: 5.0, 1.0)
        for line in lines:
            reader.feed_line(line)
        return reader

    def test_comments_are_counted_and_tool_fragments_are_joined_by_index(self):
        first = {"choices": [{"index": 0, "delta": {"role": "assistant"}}]}
        part_a = {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "c1", "type": "function",
             "function": {"name": "write_file", "arguments": '{"path": '}}]}}]}
        part_b = {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '"a.py"}'}},
            {"index": 1, "id": "c2", "type": "function",
             "function": {"name": "run_python", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}}
        lines = []
        for chunk in (first, part_a, part_b):
            lines += ["data: " + json.dumps(chunk), ""]
        lines += [": keep-alive", "", "data: [DONE]", ""]
        reader = self.fold(lines)
        message = reader.message()
        self.assertEqual(reader.keepalives, 1)
        self.assertTrue(reader.done)
        self.assertEqual(reader.problems, [])
        self.assertTrue(message["role_first"])
        self.assertEqual(message["tool_calls"], [
            {"id": "c1", "name": "write_file", "arguments": '{"path": "a.py"}'},
            {"id": "c2", "name": "run_python", "arguments": "{}"}])
        self.assertEqual(message["tool_call_chunks"], [1, 2])
        self.assertEqual(message["final_chunk"], 2)
        self.assertEqual(message["finish_reason"], "tool_calls")
        self.assertEqual(message["usage_chunk"], 2)

    def test_data_that_is_not_json_and_data_after_done_are_problems(self):
        reader = self.fold(["data: {not json", "", "data: [DONE]", "", 'data: {"choices": []}', ""])
        self.assertEqual(reader.problems, ["a data event is not JSON", "data after [DONE]"])

    def test_an_error_event_is_kept(self):
        reader = self.fold(["event: error", 'data: {"error": {"code": "backend_error"}}', ""])
        self.assertIn("backend_error", reader.error_event)
        self.assertEqual(reader.chunks, [])


class Tools(unittest.TestCase):
    def test_arguments_are_checked_against_the_schema(self):
        self.assertEqual(acc_a.check_args("write_file", '{"path": "a", "content": "b"}'), (True, []))
        self.assertEqual(acc_a.check_args("write_file", '{"path": "a"}'),
                         (True, ["missing argument content"]))
        self.assertEqual(acc_a.check_args("run_python", '{"path": 3}'),
                         (True, ["argument path is not a string"]))
        self.assertEqual(acc_a.check_args("run_python", '{"path": "a", "x": "y"}'),
                         (True, ["unexpected argument x"]))
        self.assertEqual(acc_a.check_args("run_python", "{"), (False, ["arguments are not JSON"]))
        self.assertEqual(acc_a.check_args("run_python", "[]"),
                         (False, ["arguments are not a JSON object"]))

    def test_a_path_outside_the_working_directory_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp) / "work"
            work.mkdir()
            for path in ("../evil.py", "/tmp/evil-abs.py", "sub/../../evil.py"):
                text = acc_a.run_tool(work, "write_file",
                                      json.dumps({"path": path, "content": "x"}), fake_sandbox(tmp))
                self.assertTrue(text.startswith("error:"), text)
            self.assertFalse((pathlib.Path(tmp) / "evil.py").exists())
            self.assertFalse(pathlib.Path("/tmp/evil-abs.py").exists())

    def test_a_written_file_runs_and_its_output_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp)
            box = fake_sandbox(tmp)
            acc_a.run_tool(work, "write_file", json.dumps({"path": "hello.py",
                                                           "content": "print(2 + 3)\n"}), box)
            self.assertEqual(acc_a.run_tool(work, "run_python", '{"path": "hello.py"}', box),
                             "exit 0\n5\n")
            self.assertEqual(acc_a.program_output(work, box), "5")


class Loop(unittest.TestCase):
    def test_a_good_server_passes_with_four_requests_and_full_evidence(self):
        with served() as url, tempfile.TemporaryDirectory() as tmp:
            status, errors, out = run_acc_a(url, tmp)
            self.assertEqual(status, acc_a.EXIT_PASS, errors)
            rows = (out / "timings.tsv").read_text().splitlines()
            self.assertEqual([r.split("\t")[2] for r in rows[1:]], ["r1", "r1s", "r2", "r3"])
            self.assertEqual(json.loads((out / "result.json").read_text())["verdict"], "pass")
            self.assertTrue((out / "transcript.jsonl").is_file())
            self.assertTrue((out / "sse-1-2.txt").is_file())
            self.assertTrue(all(checks(out).values()), checks(out))
            stream_row = rows[2].split("\t")
            self.assertEqual(stream_row[3], "True")
            self.assertEqual(stream_row[5], str(100 + len(acc_a.SYSTEM.split())
                                                + len(acc_a.TASK.split())))

    def test_the_key_never_reaches_the_evidence(self):
        with served() as url, tempfile.TemporaryDirectory() as tmp:
            status, errors, out = run_acc_a(url, tmp)
            self.assertEqual(status, 0, errors)
            for path in out.iterdir():
                self.assertNotIn(KEY, path.read_text(), path.name)

    def test_the_server_log_must_hold_one_line_per_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "server.log"
            with served(log=str(log)) as url:
                status, errors, out = run_acc_a(url, tmp, ["--server-log", str(log)])
                self.assertEqual(status, 0, errors)
                self.assertTrue(checks(out)["server_log_one_line_per_request"])
            self.assertEqual(acc_a.count_log_lines(log, "finish_reason"), 4)
            self.assertEqual(acc_a.count_log_lines(log, "absent"), 0)

    def test_a_server_log_with_a_missing_line_is_a_pipeline_defect(self):
        with tempfile.TemporaryDirectory() as tmp:
            short = pathlib.Path(tmp) / "short.log"
            short.write_text(json.dumps({"finish_reason": "stop"}) + "\n")
            with served() as url:
                status, _, out = run_acc_a(url, tmp, ["--server-log", str(short)])
            self.assertEqual(status, acc_a.EXIT_PIPELINE)
            self.assertFalse(checks(out)["server_log_one_line_per_request"])

    def test_leaked_markup_is_a_pipeline_defect(self):
        with served("leak") as url, tempfile.TemporaryDirectory() as tmp:
            status, _, out = run_acc_a(url, tmp)
            self.assertEqual(status, acc_a.EXIT_PIPELINE)
            self.assertFalse(checks(out)["r1_no_markup_leak"])

    def test_arguments_that_are_not_json_are_a_pipeline_defect(self):
        with served("badjson") as url, tempfile.TemporaryDirectory() as tmp:
            status, _, out = run_acc_a(url, tmp)
            self.assertEqual(status, acc_a.EXIT_PIPELINE)
            self.assertFalse(checks(out)["r1_arguments_json"])

    def test_an_http_error_is_a_pipeline_defect(self):
        with served("http500") as url, tempfile.TemporaryDirectory() as tmp:
            status, _, out = run_acc_a(url, tmp)
            self.assertEqual(status, acc_a.EXIT_PIPELINE)
            self.assertFalse(checks(out)["r1_http_200"])

    def test_a_stream_without_done_is_a_pipeline_defect(self):
        with served("no-done") as url, tempfile.TemporaryDirectory() as tmp:
            status, _, out = run_acc_a(url, tmp)
            self.assertEqual(status, acc_a.EXIT_PIPELINE)
            self.assertFalse(checks(out)["r1s_done"])

    def test_a_chunk_after_the_finish_chunk_is_a_pipeline_defect(self):
        with served("finish-before-calls") as url, tempfile.TemporaryDirectory() as tmp:
            status, _, out = run_acc_a(url, tmp)
            self.assertEqual(status, acc_a.EXIT_PIPELINE)
            self.assertFalse(checks(out)["r1s_finish_in_final_chunk"])

    def test_a_text_answer_is_a_quality_miss_after_three_attempts(self):
        with served("quality-text") as url, tempfile.TemporaryDirectory() as tmp:
            status, _, out = run_acc_a(url, tmp)
            self.assertEqual(status, acc_a.EXIT_QUALITY)
            summary = json.loads((out / "result.json").read_text())
            self.assertEqual([a["verdict"] for a in summary["attempts"]], ["quality_miss"] * 3)

    def test_a_wrong_program_is_a_quality_miss(self):
        with served("wrong-sum") as url, tempfile.TemporaryDirectory() as tmp:
            status, _, out = run_acc_a(url, tmp, ["--attempts", "1"])
            self.assertEqual(status, acc_a.EXIT_QUALITY)
            self.assertFalse(checks(out)["hello_py_prints_5"])

    def test_a_prompt_over_the_budget_stops_the_run(self):
        with served("big-prompt") as url, tempfile.TemporaryDirectory() as tmp:
            status, errors, out = run_acc_a(url, tmp)
            self.assertEqual(status, acc_a.EXIT_BUDGET)
            self.assertIn("1500", errors)

    def test_a_path_escape_by_the_model_writes_no_file_outside_the_working_directory(self):
        with served("escape") as url, tempfile.TemporaryDirectory() as tmp:
            run_acc_a(url, tmp, ["--attempts", "1"])
            self.assertFalse((pathlib.Path(tmp) / "evil.py").exists())

    def test_port_8788_is_refused_before_any_connection(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, errors, _ = run_acc_a("http://127.0.0.1:8788/v1", tmp)
            self.assertEqual(status, acc_a.EXIT_USAGE)
            self.assertIn("8788", errors)

    def test_a_missing_key_variable_is_a_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                status = acc_a.main(["--base-url", "http://127.0.0.1:1/v1", "--key-env",
                                     "NO_SUCH_VARIABLE_FOR_ACC_A", "--out", tmp + "/o",
                                     "--workdir", tmp + "/w"])
            self.assertEqual(status, acc_a.EXIT_USAGE)
            self.assertIn("NO_SUCH_VARIABLE_FOR_ACC_A", errors.getvalue())


class Sandboxing(unittest.TestCase):
    def test_a_missing_sandbox_image_stops_the_run_before_any_model_call(self):
        for image in ("", "python:3.12-slim", "absent@sha256:" + "0" * 64):
            with self.subTest(image=image):
                with tempfile.TemporaryDirectory() as tmp:
                    dump = pathlib.Path(tmp) / "bodies.jsonl"
                    with served(dump=str(dump)) as url:
                        status, errors, out = run_acc_a(url, tmp, image=image)
                    self.assertEqual(status, acc_a.EXIT_SANDBOX)
                    self.assertIn("sandbox image missing; needs the user's approval to pull", errors)
                    self.assertFalse(dump.exists())  # the server never saw a chat request
                    self.assertFalse(out.exists())  # nothing was fetched or written

    def test_a_daemon_that_is_not_rootless_stops_the_run_before_any_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump = pathlib.Path(tmp) / "bodies.jsonl"
            os.environ["FAKE_DOCKER_ROOTLESS"] = "0"
            try:
                with served(dump=str(dump)) as url:
                    status, errors, _ = run_acc_a(url, tmp)
            finally:
                del os.environ["FAKE_DOCKER_ROOTLESS"]
            self.assertEqual(status, 69)
            self.assertIn("not rootless", errors)
            self.assertFalse(dump.exists())

    def test_the_models_code_runs_only_through_the_isolated_container_command(self):
        with served() as url, tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "docker.log"
            os.environ["FAKE_DOCKER_LOG"] = str(log)
            try:
                status, errors, out = run_acc_a(url, tmp)
            finally:
                del os.environ["FAKE_DOCKER_LOG"]
            self.assertEqual(status, acc_a.EXIT_PASS, errors)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            runs = [c for c in calls if c[0] == "run"]
            # the model's run_python call, and the final check of hello.py
            self.assertEqual(len(runs), 2)
            for run in runs:
                self.assertEqual(run[run.index("--log-driver") + 1], "none")
                for flag, value in (("--network", "none"), ("--user", "65534:65534"),
                                    ("--memory", "256m"), ("--cpus", "1"), ("--pids-limit", "64")):
                    self.assertEqual(run[run.index(flag) + 1], value)
                self.assertIn("--read-only", run)
                self.assertIn(IMAGE, run)
            self.assertNotIn(KEY, log.read_text())  # the key is not passed to the container

    def test_the_final_check_of_hello_py_uses_the_sandbox_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp)
            (work / "hello.py").write_text("print(2 + 3)\n")
            broken = sandbox.Sandbox(IMAGE, docker=str(HERE / "no-such-docker"), scratch=work / "s")
            with self.assertRaises(sandbox.SandboxError):
                acc_a.program_output(work, broken)  # no host fallback: the host never runs it


class ModelControlledArguments(unittest.TestCase):
    """Arguments of a tool call come from the model. A bad value is a failed tool call."""

    def call(self, tmp, name, arguments):
        work = pathlib.Path(tmp) / "work"
        work.mkdir(exist_ok=True)
        return acc_a.run_tool(work, name, json.dumps(arguments), fake_sandbox(tmp))

    def test_bad_paths_and_content_give_an_error_result_and_never_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.call(tmp, "write_file", {"path": "hello.py", "content": "print(2 + 3)\n"})
            cases = {
                "the working directory itself": {"path": ".", "content": "x"},
                "a NUL in the path": {"path": "a\x00b", "content": "x"},
                "a path under an existing file": {"path": "hello.py/sub.py", "content": "x"},
                "a path of 300 characters": {"path": "a" * 300, "content": "x"},
                "a lone surrogate in the content": {"path": "b.py", "content": "\ud800"},
            }
            for label, arguments in cases.items():
                with self.subTest(label):
                    text = self.call(tmp, "write_file", arguments)
                    self.assertTrue(text.startswith("error:"), text)
            self.assertTrue(self.call(tmp, "run_python", {"path": "a\x00b"}).startswith("error:"))
            self.assertEqual(self.call(tmp, "run_python", {"path": "hello.py"}), "exit 0\n5\n")

    def test_a_run_that_writes_to_a_directory_path_still_ends_with_a_result(self):
        with served("dot-path") as url, tempfile.TemporaryDirectory() as tmp:
            status, _, out = run_acc_a(url, tmp, ["--attempts", "1"])
            self.assertEqual(status, acc_a.EXIT_QUALITY)
            self.assertEqual(json.loads((out / "result.json").read_text())["verdict"],
                             "quality_miss")


class SandboxFailures(unittest.TestCase):
    def test_a_docker_failure_is_a_sandbox_error_with_a_result_not_a_quality_miss(self):
        with served() as url, tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_DOCKER_EXIT"] = "127"
            try:
                status, errors, out = run_acc_a(url, tmp)
            finally:
                del os.environ["FAKE_DOCKER_EXIT"]
            self.assertEqual(status, acc_a.EXIT_SANDBOX)
            self.assertIn("docker failed", errors)
            self.assertEqual(json.loads((out / "result.json").read_text())["verdict"],
                             "sandbox_error")

    def test_a_stage_directory_with_files_stops_the_run_and_is_not_deleted(self):
        with served() as url, tempfile.TemporaryDirectory() as tmp:
            stage = pathlib.Path(tmp) / "work-stage"
            stage.mkdir()
            (stage / "keep.txt").write_text("keep")
            status, errors, _ = run_acc_a(url, tmp)
            self.assertEqual(status, acc_a.EXIT_SANDBOX)
            self.assertEqual((stage / "keep.txt").read_text(), "keep")
            self.assertIn("not empty", errors)


class StreamRule(unittest.TestCase):
    """The adapted stream rule: role first, tool calls in their own chunks, then an empty final
    chunk with the finish reason and the usage, and the text/event-stream content type."""

    def failed(self, mode):
        with served(mode) as url, tempfile.TemporaryDirectory() as tmp:
            status, _, out = run_acc_a(url, tmp, ["--attempts", "1"])
            self.assertEqual(status, acc_a.EXIT_PIPELINE)
            return {name for name, ok in checks(out).items() if not ok}

    def test_a_tool_call_in_the_final_chunk_fails_the_own_chunk_check(self):
        self.assertIn("r1s_tool_calls_in_own_chunks", self.failed("calls-in-final-chunk"))

    def test_content_in_the_final_chunk_fails_only_the_empty_final_delta_check(self):
        failed = self.failed("content-in-final-chunk")
        self.assertIn("r1s_final_delta_empty", failed)
        self.assertNotIn("r1s_tool_calls_in_own_chunks", failed)

    def test_usage_in_an_earlier_chunk_fails_the_usage_in_final_chunk_check(self):
        failed = self.failed("usage-early")
        self.assertIn("r1s_usage_in_final_chunk", failed)
        self.assertNotIn("r1s_usage", failed)

    def test_a_stream_line_that_is_not_utf8_fails_the_framing_check(self):
        self.assertIn("r1s_framing", self.failed("bad-utf8"))

    def test_a_first_chunk_without_the_role_fails_the_role_check(self):
        self.assertIn("r1s_role_first", self.failed("no-role"))

    def test_a_stream_that_is_not_text_event_stream_fails_the_framing_check(self):
        self.assertIn("r1s_framing", self.failed("wrong-content-type"))


class RequestFields(unittest.TestCase):
    def test_every_request_carries_the_fields_that_the_engine_run_depends_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump = pathlib.Path(tmp) / "bodies.jsonl"
            with served(dump=str(dump)) as url:
                status, errors, _ = run_acc_a(url, tmp)
            self.assertEqual(status, acc_a.EXIT_PASS, errors)
            bodies = [json.loads(line) for line in dump.read_text().splitlines()]
            self.assertEqual([b["stream"] for b in bodies], [False, True, False, False])
            for body in bodies:
                self.assertEqual(body["x-thinking"], "off")
                self.assertEqual(body["max_tokens"], 256)
                self.assertEqual(body["temperature"], 0.2)
                self.assertEqual(body["model"], "fake-flash")
                self.assertEqual([t["function"]["name"] for t in body["tools"]],
                                 ["write_file", "run_python"])

    def test_the_smoke_request_turns_thinking_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump = pathlib.Path(tmp) / "bodies.jsonl"
            with served(dump=str(dump)) as url:
                status, _ = run_smoke(url, tmp)
            self.assertEqual(status, 0)
            body = json.loads(dump.read_text().splitlines()[0])
            self.assertEqual((body["x-thinking"], body["max_tokens"]), ("off", 8))


class FinalReply(unittest.TestCase):
    def test_a_final_reply_cut_by_the_length_limit_is_a_quality_miss(self):
        with served("final-length") as url, tempfile.TemporaryDirectory() as tmp:
            status, _, out = run_acc_a(url, tmp, ["--attempts", "1"])
            self.assertEqual(status, acc_a.EXIT_QUALITY)
            rows = checks(out)
            self.assertFalse(rows["final_finish_stop"])
            self.assertTrue(rows["final_content_not_empty"])


class BrokenConnections(unittest.TestCase):
    def test_a_stream_cut_inside_a_chunk_is_a_pipeline_defect_with_a_result(self):
        with served("cut-stream") as url, tempfile.TemporaryDirectory() as tmp:
            status, _, out = run_acc_a(url, tmp, ["--attempts", "1"])
            self.assertEqual(status, acc_a.EXIT_PIPELINE)
            self.assertFalse(checks(out)["request_completes"])
            self.assertEqual(json.loads((out / "result.json").read_text())["verdict"],
                             "pipeline_defect")

    def test_an_unparsable_status_line_on_models_is_a_pipeline_defect_with_a_result(self):
        with served("bad-models") as url, tempfile.TemporaryDirectory() as tmp:
            status, errors, out = run_acc_a(url, tmp)
            self.assertEqual(status, acc_a.EXIT_PIPELINE, errors)
            self.assertEqual(json.loads((out / "result.json").read_text())["verdict"],
                             "pipeline_defect")

    def test_an_explicit_model_that_models_does_not_list_is_a_pipeline_defect_without_requests(self):
        with served() as url, tempfile.TemporaryDirectory() as tmp:
            status, errors, out = run_acc_a(url, tmp, ["--model", "fifi-q36", "--engine", "q36"])
            self.assertEqual(status, acc_a.EXIT_PIPELINE, errors)
            self.assertIn("does not list the model fifi-q36", errors)
            result = json.loads((out / "result.json").read_text())
            self.assertEqual((result["verdict"], result["requests_sent"]), ("pipeline_defect", 0))

    def test_an_explicit_listed_model_is_used_and_the_engine_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump = pathlib.Path(tmp) / "bodies.jsonl"
            with served(dump=str(dump)) as url:
                status, errors, out = run_acc_a(url, tmp, ["--model", "fake-flash", "--engine", "q36"])
            self.assertEqual(status, acc_a.EXIT_PASS, errors)
            result = json.loads((out / "result.json").read_text())
            self.assertEqual((result["model"], result["engine"]), ("fake-flash", "q36"))
            bodies = [json.loads(line) for line in dump.read_text().splitlines()]
            self.assertEqual({b["model"] for b in bodies}, {"fake-flash"})

    def test_the_default_engine_label_is_flash(self):
        with served() as url, tempfile.TemporaryDirectory() as tmp:
            status, errors, out = run_acc_a(url, tmp)
            self.assertEqual(status, acc_a.EXIT_PASS, errors)
            self.assertEqual(json.loads((out / "result.json").read_text())["engine"], "flash")

    def test_the_smoke_check_reports_an_unparsable_reply_instead_of_a_traceback(self):
        with served("bad-models") as url, tempfile.TemporaryDirectory() as tmp:
            status, text = run_smoke(url, tmp)
            self.assertEqual(status, 2)
            self.assertIn("smoke: request failed", text)


class Guards(unittest.TestCase):
    def test_the_key_goes_only_to_a_loopback_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, errors, _ = run_acc_a("http://192.168.0.16:18791/v1", tmp)
            self.assertEqual(status, acc_a.EXIT_USAGE)
            self.assertIn("loopback", errors)

    def test_a_working_directory_with_files_is_not_deleted(self):
        with served() as url, tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp) / "work"
            work.mkdir()
            (work / "precious.txt").write_text("keep")
            status, errors, _ = run_acc_a(url, tmp)
            self.assertEqual(status, acc_a.EXIT_USAGE)
            self.assertEqual((work / "precious.txt").read_text(), "keep")
            self.assertIn("not empty", errors)

    def test_a_request_log_line_with_status_500_is_a_pipeline_defect(self):
        line = (HERE / "testdata" / "reqlog-ok.jsonl").read_text().splitlines()[0]
        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "server.log"
            bad = dict(json.loads(line), status=500, outcome="error", error_code="backend_error")
            log.write_text("".join(l + "\n" for l in [line, json.dumps(bad), line, line]))
            with served() as url:
                status, _, out = run_acc_a(url, tmp, ["--server-log", str(log)])
            self.assertEqual(status, acc_a.EXIT_PIPELINE)
            rows = checks(out)
            self.assertTrue(rows["server_log_one_line_per_request"])
            self.assertFalse(rows["server_log_all_200_ok"])


def run_smoke(url, tmp):
    old = os.environ.get("ENGINESRV_API_KEY")
    os.environ["ENGINESRV_API_KEY"] = KEY
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            status = smoke.main(["--base-url", url, "--out", str(pathlib.Path(tmp) / "s.json"),
                                 "--timeout", "20"])
    finally:
        if old is None:
            del os.environ["ENGINESRV_API_KEY"]
        else:
            os.environ["ENGINESRV_API_KEY"] = old
    return status, out.getvalue()


class Smoke(unittest.TestCase):
    def run_smoke(self, url, tmp):
        return run_smoke(url, tmp)

    def test_a_good_server_passes_the_smoke_check_and_the_summary_is_saved(self):
        with served() as url, tempfile.TemporaryDirectory() as tmp:
            status, text = self.run_smoke(url, tmp)
            self.assertEqual(status, 0, text)
            summary = json.loads((pathlib.Path(tmp) / "s.json").read_text())
            self.assertEqual((summary["status"], summary["model"]), (200, "fake-flash"))
            self.assertNotIn(KEY, text)

    def test_a_server_error_fails_the_smoke_check(self):
        with served("http500") as url, tempfile.TemporaryDirectory() as tmp:
            status, _ = self.run_smoke(url, tmp)
            self.assertEqual(status, 2)


class FakeServer(unittest.TestCase):
    def test_a_wrong_key_gets_401_and_ready_waits_for_the_delay(self):
        with served(ready_delay=0.3) as url:
            self.assertEqual(agent_client.get_json(url, "/models", "wrong", 5)[0], 401)
            host = url.rsplit("/v1", 1)[0]
            self.assertEqual(agent_client.get_json(host, "/ready", "", 5)[0], 503)
            import time
            time.sleep(0.4)
            self.assertEqual(agent_client.get_json(host, "/ready", "", 5), (200, {"status": "ready"}))


if __name__ == "__main__":
    unittest.main()
