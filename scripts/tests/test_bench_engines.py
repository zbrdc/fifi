"""CPU-only checks for the enginesrv measurement driver."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "bench_engines.py"
SPEC = importlib.util.spec_from_file_location("bench_engines_under_test", SCRIPT)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def line(value):
    return ("data: " + json.dumps(value) + "\n").encode()


def stream_lines():
    return [
        line(
            {
                "id": "reply-1",
                "choices": [{"delta": {"role": "assistant", "content": ""}}],
            }
        ),
        line(
            {
                "id": "reply-1",
                "choices": [{"delta": {"content": "several tokens together"}}],
            }
        ),
        line({"id": "reply-1", "choices": [{"delta": {"content": "more tokens"}}]}),
        line(
            {
                "id": "reply-1",
                "choices": [{"delta": {}, "finish_reason": "length"}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 96,
                    "reused_tokens": 30,
                },
            }
        ),
        b"data: [DONE]\n",
    ]


def row(**changes):
    result = {
        "request_id": "reply-1",
        "model": "fifi-q36",
        "status": 200,
        "outcome": "ok",
        "stream": True,
        "finish_reason": "length",
        "prompt_tokens": 100,
        "completion_tokens": 96,
        "decode_tokens_per_second": 37.25,
    }
    result.update(changes)
    return result


class StreamTests(unittest.TestCase):
    def test_stream_chunks_do_not_determine_server_token_rate(self):
        stream = BENCH.parse_stream(stream_lines(), 10.0, lambda: 11.25)
        self.assertEqual(stream["first_delta_seconds"], 1.25)
        self.assertEqual(stream["usage"]["completion_tokens"], 96)
        self.assertEqual(BENCH.matching_server_rate([row()], "fifi-q36", stream), 37.25)
        self.assertEqual(BENCH.reused_tokens(stream["usage"]), 30)

    def test_content_observer_receives_only_valid_visible_text(self):
        seen = []
        stream = BENCH.parse_stream(stream_lines(), 10.0, lambda: 11.25, on_content=seen.append)
        self.assertEqual(seen, ["several tokens together", "more tokens"])
        self.assertEqual(stream["usage"]["completion_tokens"], 96)
        bad = [line({"choices": [{"delta": {"content": False}}]})] + stream_lines()
        with self.assertRaisesRegex(ValueError, "content is not text"):
            BENCH.parse_stream(bad, 10.0, on_content=seen.append)

    def test_rejects_incomplete_and_inconsistent_streams(self):
        cases = [
            stream_lines()[:-1],
            stream_lines()[:-2] + [b"data: [DONE]\n"],
            [line({"id": "a", "choices": []}), line({"id": "b", "choices": []})],
            [b"event: error\n", line({"error": {"message": "failure"}})],
        ]
        for lines in cases:
            with (
                self.subTest(lines=lines),
                self.assertRaises((ValueError, RuntimeError)),
            ):
                BENCH.parse_stream(lines, 10.0, lambda: 11.0)

    def test_missing_mismatched_or_ambiguous_telemetry_is_not_a_rate(self):
        stream = BENCH.parse_stream(stream_lines(), 10.0, lambda: 11.0)
        for rows in (
            [],
            [row(model="other")],
            [row(completion_tokens=2)],
            [row(prompt_tokens=101)],
            [row(outcome="error")],
            [row(stream=False)],
            [row(finish_reason="stop")],
            [row(decode_tokens_per_second=float("nan"))],
            [row(decode_tokens_per_second=True)],
            [row(), row()],
        ):
            with self.subTest(rows=rows):
                self.assertIsNone(BENCH.matching_server_rate(rows, "fifi-q36", stream))

    def test_unavailable_telemetry_does_not_estimate_from_chunks(self):
        stream = BENCH.parse_stream(stream_lines(), 10.0, lambda: 11.0)
        with mock.patch.object(
            BENCH.urllib.request, "urlopen", side_effect=OSError("unavailable")
        ):
            self.assertIsNone(
                BENCH.recent_server_rate(
                    "http://example.invalid", {}, "fifi-q36", stream
                )
            )

    def test_reuse_must_be_a_valid_prompt_subset(self):
        self.assertEqual(
            BENCH.reused_tokens(
                {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 3}}
            ),
            3,
        )
        self.assertIsNone(BENCH.reused_tokens({"prompt_tokens": 10}))
        with self.assertRaises(ValueError):
            BENCH.reused_tokens({"prompt_tokens": 10, "reused_tokens": 11})

    def test_conflicting_or_malformed_reuse_cannot_qualify_a_cold_prompt(self):
        for flat, details in [
            (0, {"cached_tokens": 2047}),
            (-1, {"cached_tokens": 0}),
            (False, {"cached_tokens": 0}),
            (0, {"cached_tokens": 1.5}),
            (0, {"cached_tokens": 2049}),
            (0, []),
        ]:
            with (
                self.subTest(flat=flat, details=details),
                self.assertRaises(ValueError),
            ):
                BENCH.reused_tokens(
                    {
                        "prompt_tokens": 2048,
                        "reused_tokens": flat,
                        "prompt_tokens_details": details,
                    }
                )

    def test_first_request_rate_counts_new_tokens_and_keeps_missing_reuse_unknown(self):
        for reused, count, rate in [
            (0, 100, 50),
            (20, 80, 40),
            (100, 0, 0),
            (None, None, None),
        ]:
            usage = {
                "prompt_tokens": 100,
                "completion_tokens": 1,
                "reused_tokens": reused,
            }
            replies = [
                io.BytesIO(json.dumps({"usage": usage}).encode()),
                io.BytesIO(b"".join(stream_lines())),
            ]
            with (
                self.subTest(reused=reused),
                mock.patch.object(BENCH.urllib.request, "urlopen", side_effect=replies),
                mock.patch.object(BENCH.time, "monotonic", side_effect=[10, 12, 13]),
                mock.patch.object(BENCH, "recent_server_rate", return_value=None),
            ):
                result = BENCH.bench("http://localhost", "test-key", "fifi-q36", 100)
                self.assertEqual(result["estimated_prefill_tps"], 50)
                self.assertEqual(result["first_request_new_tokens"], count)
                self.assertEqual(result["estimated_new_prefill_tps"], rate)

    def test_valid_and_absent_reuse_fields(self):
        for flat, nested, expected in [
            (0, 0, 0),
            (3, 3, 3),
            (None, 10, 10),
            (4, None, 4),
            (None, None, None),
        ]:
            with self.subTest(flat=flat, nested=nested):
                self.assertEqual(
                    BENCH.reused_tokens(
                        {
                            "prompt_tokens": 10,
                            "reused_tokens": flat,
                            "prompt_tokens_details": {"cached_tokens": nested},
                        }
                    ),
                    expected,
                )


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, body, content_type="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if (
            self.path != "/v1/recent-requests"
            or self.headers.get("Authorization") != "Bearer test-key"
            or getattr(self.server, "telemetry_unavailable", False)
        ):
            self.send_error(404)
            return
        self.reply(json.dumps({"requests": [row()]}).encode())

    def do_POST(self):
        if (
            self.path != "/v1/chat/completions"
            or self.headers.get("Authorization") != "Bearer test-key"
        ):
            self.send_error(404)
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if request.get("stream"):
            self.reply(b"".join(stream_lines()), "text/event-stream")
        else:
            body = {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 1,
                    "reused_tokens": 20,
                }
            }
            self.reply(json.dumps(body).encode())


class IntegrationTests(unittest.TestCase):
    def run_cli(self, telemetry_unavailable=False):
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        server.telemetry_unavailable = telemetry_unavailable
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                f"http://127.0.0.1:{server.server_port}",
                "test-key",
                "fifi-q36",
                "25",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )

    def test_cli_correlates_server_rate_and_reports_reuse(self):
        result = self.run_cli()
        self.assertIn("server decode t/s", result.stdout)
        self.assertIn("| 20 | 80 |", result.stdout)
        self.assertIn("| 30 | 96 | length | 37.25 |", result.stdout)

    def test_cli_reports_na_when_server_telemetry_is_unavailable(self):
        result = self.run_cli(telemetry_unavailable=True)
        self.assertIn("| 30 | 96 | length | n/a |", result.stdout)
        self.assertNotIn("37.25", result.stdout)


if __name__ == "__main__":
    unittest.main()
