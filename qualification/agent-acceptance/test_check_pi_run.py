"""Self-test of the ACC-B checker. Run, from this directory:
python3 -I -m unittest discover -s . -p 'test_*.py'
hello.py runs in the sandbox of sandbox.py, which these tests point at fake_docker.py.
The fixture testdata/pi-run-ok.jsonl is a real `pi --mode json` run (pi 1.0.0) against the local
fake server in mode pi-tools; the prompt is the ACC-B task.
"""

import contextlib
import io
import json
import os
import pathlib
import tempfile
import unittest

import check_pi_run

HERE = pathlib.Path(__file__).parent
FIXTURE = HERE / "testdata" / "pi-run-ok.jsonl"
# Request log lines in the real shape (engine/src/server/reqlog.rs): see testdata/reqlog-ok.jsonl.
LOG = [json.loads(line) for line in (HERE / "testdata" / "reqlog-ok.jsonl").read_text().splitlines()]
IMAGE = "fake@sha256:" + "0" * 64
SANDBOX = ["--sandbox-image", IMAGE, "--docker", str(HERE / "fake_docker.py")]


def run_checker(tmp, events=None, exit_code=0, log=LOG, hello="print(2 + 3)\n", extra=(), raw=()):
    tmp = pathlib.Path(tmp)
    transcript = tmp / "pi.jsonl"
    if events is None:
        transcript.write_text(FIXTURE.read_text())
    else:
        transcript.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    if raw:                       # printed stream lines, as the launcher stamps them
        with open(transcript, "a", encoding="utf-8") as handle:
            handle.write("".join(line + "\n" for line in raw))
    work = tmp / "work"
    work.mkdir(exist_ok=True)
    if hello is not None:
        (work / "hello.py").write_text(hello)
    server_log = tmp / "server.log"
    server_log.write_text("".join(json.dumps(e) + "\n" for e in log))
    with contextlib.redirect_stdout(io.StringIO()):
        status = check_pi_run.main(["--transcript", str(transcript), "--exit-code", str(exit_code),
                                    "--workdir", str(work), "--server-log", str(server_log),
                                    "--out", str(tmp / "out"), *SANDBOX, *extra])
    rows = {}
    for line in (tmp / "out" / "acc-b-checks.tsv").read_text().splitlines()[1:]:
        name, ok, _ = line.split("\t")
        rows[name] = ok == "True"
    return status, rows


def fixture_events():
    return [json.loads(line) for line in FIXTURE.read_text().splitlines()]


class Checker(unittest.TestCase):
    def test_the_recorded_pi_run_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, rows = run_checker(tmp)
            self.assertEqual(status, 0, rows)
            self.assertTrue(all(rows.values()), rows)

    def test_a_nonzero_pi_exit_is_a_pipeline_defect(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, rows = run_checker(tmp, exit_code=1)
            self.assertEqual(status, 2)
            self.assertFalse(rows["pi_exit_0"])

    def test_a_retry_shows_as_more_requests_than_assistant_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, rows = run_checker(tmp, log=LOG + [LOG[-1]])
            self.assertEqual(status, 2)
            self.assertFalse(rows["one_request_per_assistant_message"])

    def test_a_request_with_status_500_is_a_pipeline_defect(self):
        with tempfile.TemporaryDirectory() as tmp:
            failed = [dict(LOG[0], status=500, outcome="error", error_code="backend_error"),
                      *LOG[1:]]
            status, rows = run_checker(tmp, log=failed)
            self.assertEqual(status, 2)
            self.assertFalse(rows["all_requests_200_ok"])
            self.assertTrue(rows["one_request_per_assistant_message"])

    def test_a_cancelled_request_is_a_pipeline_defect(self):
        with tempfile.TemporaryDirectory() as tmp:
            cancelled = [*LOG[:2], dict(LOG[2], outcome="cancelled")]
            status, rows = run_checker(tmp, log=cancelled)
            self.assertEqual(status, 2)
            self.assertFalse(rows["all_requests_200_ok"])

    def test_a_docker_failure_while_running_hello_py_is_a_sandbox_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_DOCKER_EXIT"] = "126"
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    status, _ = run_checker(tmp)
            except FileNotFoundError:
                status = 67  # no checks file: the checker stopped before it wrote one
            finally:
                del os.environ["FAKE_DOCKER_EXIT"]
            self.assertEqual(status, 67)

    def test_a_missing_server_log_is_a_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as caught:
                with contextlib.redirect_stderr(io.StringIO()):
                    check_pi_run.main(["--transcript", str(pathlib.Path(tmp) / "pi.jsonl"),
                                       "--exit-code", "0", "--workdir", tmp,
                                       "--out", str(pathlib.Path(tmp) / "out"), *SANDBOX])
            self.assertEqual(caught.exception.code, 1)

    def test_hostile_nested_json_is_a_pipeline_defect_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = pathlib.Path(tmp) / "pi.jsonl"
            transcript.write_text("[" * 100000 + "]" * 100000 + "\n")
            work = pathlib.Path(tmp) / "work"
            work.mkdir()
            server_log = pathlib.Path(tmp) / "server.log"
            server_log.write_text("".join(json.dumps(e) + "\n" for e in LOG))
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                status = check_pi_run.main(["--transcript", str(transcript), "--exit-code", "0",
                                            "--workdir", str(work), "--server-log", str(server_log),
                                            "--out", str(pathlib.Path(tmp) / "out"), *SANDBOX])
            self.assertEqual(status, 2)
            self.assertNotIn("Traceback", errors.getvalue())

    def test_a_missing_sandbox_image_stops_before_anything_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                status = check_pi_run.main(["--transcript", str(pathlib.Path(tmp) / "none"),
                                            "--exit-code", "0", "--workdir", tmp,
                                            "--server-log", str(pathlib.Path(tmp) / "none.log"),
                                            "--out", str(pathlib.Path(tmp) / "out"),
                                            "--sandbox-image", "", "--docker", "docker"])
            self.assertEqual(status, 67)
            self.assertIn("needs the user's approval to pull", errors.getvalue())
            self.assertFalse((pathlib.Path(tmp) / "out").exists())

    def test_a_prompt_over_the_limit_is_a_pipeline_defect(self):
        with tempfile.TemporaryDirectory() as tmp:
            big = [dict(LOG[0], prompt_tokens=1801), *LOG[1:]]
            status, rows = run_checker(tmp, log=big)
            self.assertEqual(status, 2)
            self.assertFalse(rows["prompt_tokens_within_limit"])

    def test_an_error_stop_is_a_pipeline_defect(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = fixture_events()
            for event in events:
                message = event.get("message") or {}
                if event.get("type") == "message_end" and message.get("role") == "assistant":
                    message["stopReason"] = "error"
            status, rows = run_checker(tmp, events)
            self.assertEqual(status, 2)
            self.assertFalse(rows["no_error_stop"])

    def test_leaked_markup_in_the_final_text_is_a_pipeline_defect(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = fixture_events()
            last = [e for e in events if e.get("type") == "message_end"
                    and e["message"].get("role") == "assistant"][-1]
            last["message"]["content"][0]["text"] = "<tool_call>\n<function=bash>"
            status, rows = run_checker(tmp, events)
            self.assertEqual(status, 2)
            self.assertFalse(rows["no_markup_in_text"])

    def test_a_wrong_program_is_a_quality_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, rows = run_checker(tmp, hello="print(2 + 4)\n")
            self.assertEqual(status, 3)
            self.assertFalse(rows["hello_py_prints_5"])

    def test_a_run_without_tool_calls_is_a_quality_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = [e for e in fixture_events()
                      if e.get("type") not in ("tool_execution_start", "tool_execution_end")]
            events = [e for e in events if (e.get("message") or {}).get("role") != "toolResult"]
            for event in events:
                message = event.get("message") or {}
                if message.get("role") == "assistant":
                    message["content"] = [c for c in message["content"] if c.get("type") != "toolCall"]
            status, rows = run_checker(tmp, events, hello=None)
            self.assertEqual(status, 3)
            self.assertFalse(rows["has_tool_call"])
            self.assertFalse(rows["has_tool_result"])

    def test_a_final_message_cut_by_the_length_limit_is_a_quality_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = fixture_events()
            last = [e for e in events if e.get("type") == "message_end"
                    and e["message"].get("role") == "assistant"][-1]
            last["message"]["stopReason"] = "length"
            status, rows = run_checker(tmp, events)
            self.assertEqual(status, 3)
            self.assertFalse(rows["final_stop_with_text"])

    def test_the_export_marker_line_is_extractor_input_not_an_event(self):
        # The ACCB-EXPORT marker line is extractor input (packet section 5.13, design 4.6 rule 1),
        # not a pi event: the checker must skip it and pass the run on the real events.
        with tempfile.TemporaryDirectory() as tmp:
            status, rows = run_checker(tmp, raw=["ACCB-EXPORT aGVsbG8K"])
            self.assertEqual(status, 0, rows)
            self.assertTrue(all(rows.values()), rows)

    def test_forged_accb_lines_are_data_not_events(self):
        # F1 (design 4.5): a model command can print ACCB-EXIT and ACCB-EXPORT lines and a
        # forged event into the stream. Printed lines are data (section 9: there is no
        # ACCB-EXIT line): they must not fail a run whose independent evidence is good.
        with tempfile.TemporaryDirectory() as tmp:
            status, rows = run_checker(
                tmp, raw=["ACCB-EXIT 42", "ACCB-EXPORT FORGED", '{"type":"forged"}'])
            self.assertEqual(status, 0, rows)
            self.assertTrue(all(rows.values()), rows)

    def test_a_non_integer_exit_code_is_a_usage_error(self):
        # Round 5, blocking item 4: the docker status can be the string `none`; the checker
        # must refuse it with a usage error (status 1), never a traceback, so the launcher
        # validates the status before passing it.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as caught:
                with contextlib.redirect_stderr(io.StringIO()) as errors:
                    check_pi_run.main(["--transcript", str(pathlib.Path(tmp) / "pi.jsonl"),
                                       "--exit-code", "none", "--workdir", tmp,
                                       "--server-log", str(pathlib.Path(tmp) / "server.log"),
                                       "--out", str(pathlib.Path(tmp) / "out"), *SANDBOX])
            self.assertEqual(caught.exception.code, 1)
            self.assertNotIn("Traceback", errors.getvalue())

    def test_invalid_utf8_bytes_are_a_pipeline_defect_not_a_crash(self):
        # Round 5, blocking item 6: invalid bytes in the transcript are model data; reading
        # them must not raise (exit 1 with a traceback). They parse as no event, so the run is
        # a pipeline defect (status 2) like any other unparsable line.
        with tempfile.TemporaryDirectory() as tmp:
            transcript = pathlib.Path(tmp) / "pi.jsonl"
            transcript.write_bytes(b'{"type":"session"}\n{"type":"x","v":"\xff"}\n')
            work = pathlib.Path(tmp) / "work"
            work.mkdir()
            server_log = pathlib.Path(tmp) / "server.log"
            server_log.write_text("".join(json.dumps(e) + "\n" for e in LOG))
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                status = check_pi_run.main(["--transcript", str(transcript), "--exit-code", "0",
                                            "--workdir", str(work), "--server-log", str(server_log),
                                            "--out", str(pathlib.Path(tmp) / "out"), *SANDBOX])
            self.assertEqual(status, 2)
            self.assertNotIn("Traceback", errors.getvalue())

    def test_non_object_json_lines_are_not_events(self):
        # Round 5, blocking item 6: `[1]`, `"s"`, and `5` are valid JSON but not pi events.
        # They are recorded as unparsable (a pipeline defect), and never raise AttributeError.
        with tempfile.TemporaryDirectory() as tmp:
            status, rows = run_checker(tmp, raw=["[1]", '"s"', "5"])
            self.assertEqual(status, 2)
            self.assertFalse(rows["transcript_parses"])

    def test_a_u2028_inside_a_json_string_does_not_split_the_event(self):
        # Round 5, blocking item 6: lines split on \n only. A U+2028 (or U+2029, U+0085)
        # inside a JSON string value must not split one event into two unparsable halves.
        with tempfile.TemporaryDirectory() as tmp:
            event = json.dumps({"type": "note", "text": "before\u2028after\u2029end\u0085tail"},
                                ensure_ascii=False)
            status, rows = run_checker(tmp, raw=[event])
            self.assertEqual(status, 0, rows)
            self.assertTrue(all(rows.values()), rows)

    def test_stamped_lines_give_per_message_seconds(self):
        stamped = []
        clock = 100.0
        for event in fixture_events():
            clock += 0.5
            stamped.append((clock, event))
        rows = check_pi_run.per_message_seconds(stamped)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["to_end"] > r["to_first_update"] > 0 for r in rows), rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "stamped.tsv"
            path.write_text("".join(f"{s}\t{json.dumps(e)}\n" for s, e in stamped))
            read = check_pi_run.read_events(path)
            self.assertEqual(read[0][0], 100.5)
            self.assertEqual(read[0][1]["type"], "session")


if __name__ == "__main__":
    unittest.main()
