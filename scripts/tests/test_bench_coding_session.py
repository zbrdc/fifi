"""Host-only checks for the synthetic multi-turn coding benchmark."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import bench_coding_session as coding  # noqa: E402


def event(value):
    return ("data: " + json.dumps(value) + "\n\n").encode()


def count(body):
    if "text" in body:
        return len(body["text"])
    return sum(len(m["content"]) for m in body["messages"]) + 17


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def reply(self, data, content_type="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/system":
            self.server.system_calls += 1
            limit = self.server.context_limit
            if self.server.change_context and self.server.system_calls > 2:
                limit -= 1
            payload = {"engines": [{"model": "test-model", "state": "ready", "context_limit": limit}]}
        elif self.path == "/v1/recent-requests":
            payload = {"requests": [self.server.last_record]}
        else:
            self.send_error(404)
            return
        self.reply(json.dumps(payload).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/token/encode":
            length = count(body)
            self.reply(json.dumps({"length": length, "tokens": [0] * length}).encode())
            return
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        self.server.turns.append(body["messages"])
        turn = len(self.server.turns)
        self.server.auth = self.headers.get("Authorization")
        replacement = self.server.first_edit if turn == 1 else self.server.second_edit
        answer = json.dumps({"replacement": replacement})
        reused = self.server.first_reuse if turn == 1 else self.server.second_reuse
        prompt = count(body)
        finish = self.server.finish if turn == 2 else "stop"
        usage = {"prompt_tokens": prompt + self.server.usage_delta, "completion_tokens": 12}
        if reused is not None:
            usage["reused_tokens"] = reused
        request_id = f"request-{turn}"
        self.server.last_record = {"request_id": request_id, "model": "test-model", "status": 200,
                                   "outcome": "ok", "stream": True, "finish_reason": finish,
                                   "prompt_tokens": prompt, "completion_tokens": 12,
                                   "reused_tokens": reused, "queue_wait_ms": 10, "prefill_ms": 2000,
                                   "decode_tokens_per_second": 22.0}
        payload = b"".join([
            event({"id": request_id, "choices": [{"delta": {"content": answer[:5]}}]}),
            event({"id": request_id, "choices": [{"delta": {"content": answer[5:]}}]}),
            event({"id": request_id, "choices": [{"delta": {}, "finish_reason": finish}], "usage": usage}),
        ])
        if not self.server.incomplete:
            payload += b"data: [DONE]\n\n"
        self.reply(payload, "text/event-stream")


class CodingSessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "coding.json"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.context_limit = 10000
        self.server.system_calls = 0
        self.server.change_context = False
        self.server.first_edit = "return value.strip().lower()"
        self.server.second_edit = "return value.strip().upper()"
        self.server.first_reuse = 0
        self.server.second_reuse = 2500
        self.server.usage_delta = 0
        self.server.finish = "stop"
        self.server.incomplete = False
        self.server.turns = []
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def cli(self, *extra, env=None):
        return subprocess.run([sys.executable, str(ROOT / "scripts/bench_coding_session.py"),
                               "--url", f"http://127.0.0.1:{self.server.server_port}",
                               "--model", "test-model", "--prompt-tokens", "3000",
                               "--output", str(self.output), *extra],
                              capture_output=True, text=True, timeout=12, env=env)

    def test_two_turn_flow_records_real_usage_and_uncached_rate(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(self.output.read_text())
        self.assertTrue(data["passed"])
        self.assertEqual(data["reuse_mode"], "required")
        self.assertEqual([len(turn) for turn in self.server.turns], [1, 3])
        self.assertEqual(self.server.turns[1][0], self.server.turns[0][0])
        self.assertEqual(self.server.turns[1][1]["content"], data["turns"][0]["answer"])
        self.assertEqual(self.server.turns[1][2]["content"], coding.FOLLOWUP)
        self.assertEqual(data["turns"][0]["reused_tokens"], 0)
        second = data["turns"][1]
        self.assertEqual(second["reused_tokens"], 2500)
        self.assertTrue(second["reuse_passed"])
        self.assertTrue(second["reuse_check_passed"])
        self.assertEqual(second["server_uncached_tokens_per_second"],
                         (second["prompt_tokens"] - 2500) / 2)
        self.assertEqual(second["client_uncached_tokens_per_second_estimate"],
                         (second["prompt_tokens"] - 2500) / second["first_visible_seconds"])
        self.assertEqual(second["server_decode_tokens_per_second"], 22)
        self.assertNotIn("server_prompt_tokens_per_second", second)
        self.assertFalse(self.output.with_name("coding.json.partial").exists())

    def test_warm_reuse_is_required_by_default(self):
        self.server.second_reuse = 0
        result = self.cli()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(json.loads(self.output.read_text())["turns"][1]["reuse_passed"])
        self.output.unlink()
        self.server.turns.clear()
        result = self.cli("--min-reused-tokens", "0")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(self.output.exists())
        self.server.turns.clear()
        self.server.second_reuse = 2500
        result = self.cli("--min-reused-tokens", "2600")
        self.assertEqual(result.returncode, 2, result.stderr)

    def test_disabled_mode_checks_zero_reuse_without_claiming_warm_reuse(self):
        self.server.second_reuse = 0
        result = self.cli("--reuse-mode", "disabled")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(self.output.read_text())
        self.assertEqual(data["reuse_mode"], "disabled")
        self.assertIsNone(data["min_reused_tokens"])
        self.assertIsNone(data["turns"][1]["reuse_passed"])
        self.assertTrue(data["turns"][1]["reuse_check_passed"])
        self.output.unlink()
        self.server.turns.clear()
        self.server.second_reuse = 2500
        result = self.cli("--reuse-mode", "disabled")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(json.loads(self.output.read_text())["turns"][1]["reuse_check_passed"])

    def test_wrong_or_truncated_second_edit_fails(self):
        for replacement, finish in (("return value.strip().lower()", "stop"),
                                    ("return value.strip().upper()", "length")):
            with self.subTest(finish=finish):
                self.server.second_edit = replacement
                self.server.finish = finish
                result = self.cli()
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(json.loads(self.output.read_text())["passed"])
                self.output.unlink()
                self.server.turns.clear()

    def test_bad_first_edit_cannot_run_code_or_trigger_followup(self):
        marker = Path(self.directory.name) / "ran"
        self.server.first_edit = f"__import__('pathlib').Path({str(marker)!r}).touch()\n    return value"
        result = self.cli()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(len(self.server.turns), 1)
        self.assertFalse(marker.exists())
        self.assertFalse(json.loads(self.output.read_text())["turns"][0]["edit_passed"])

    def test_missing_reuse_and_mismatched_usage_leave_partial_results(self):
        for first_reuse, second_reuse, delta in ((1, 2500, 0), (0, None, 0), (0, 2500, 1)):
            with self.subTest(first_reuse=first_reuse, second_reuse=second_reuse, delta=delta):
                self.server.first_reuse, self.server.second_reuse, self.server.usage_delta = (
                    first_reuse, second_reuse, delta)
                result = self.cli()
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(json.loads(self.output.with_name("coding.json.partial").read_text())["state"], "failed")
                self.output.with_name("coding.json.partial").unlink()
                self.server.turns.clear()

    def test_followup_checks_context_again(self):
        self.server.change_context = True
        result = self.cli()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(len(self.server.turns), 1)
        self.assertIn("context changed", result.stderr)

    def test_reservations_are_checked_before_generation(self):
        self.server.context_limit = 3200
        result = self.cli()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(len(self.server.turns), 0)

    def test_incomplete_stream_is_not_a_pass(self):
        self.server.incomplete = True
        result = self.cli()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(self.output.exists())

    def test_existing_outputs_and_api_key_handling(self):
        self.output.write_text("existing")
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.output.read_text(), "existing")
        self.output.unlink()
        env = {**os.environ, "TEST_FIFI_KEY": "synthetic-secret"}
        result = self.cli("--api-key-env", "TEST_FIFI_KEY", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.server.auth, "Bearer synthetic-secret")
        self.assertNotIn("synthetic-secret", self.output.read_text())

    def test_large_fake_prompts_leave_room_for_both_turns(self):
        for target in (131072, 257000):
            with self.subTest(target=target):
                self.server.context_limit = 262144
                result = self.cli("--prompt-tokens", str(target), "--max-tokens", "128")
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(self.output.read_text())
                self.assertTrue(target <= data["turns"][0]["prompt_tokens"] <= target + 128)
                self.assertLessEqual(data["turns"][1]["prompt_tokens"] + 128, 262144)
                self.output.unlink()
                self.server.turns.clear()

    def test_ast_check_rejects_extra_statements_and_wrong_protocol(self):
        self.assertTrue(coding.valid_replacement('{"replacement":"return value.strip().lower()"}', "lower"))
        for answer in ('return value.strip().lower()',
                       '{"replacement":"return value.strip().lower()\\n    pass"}',
                       '{"replacement":"return value.strip().upper()"}',
                       '{"replacement":"return value.strip().lower()", "extra":1}'):
            self.assertFalse(coding.valid_replacement(answer, "lower"))


if __name__ == "__main__":
    unittest.main()
