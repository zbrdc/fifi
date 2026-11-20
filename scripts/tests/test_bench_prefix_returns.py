"""Run the prefix-return measurement through HTTP/SSE against bounded prefix stores."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_prefix_returns as driver
import record_prefix_checkpoint_bytes as checkpoints


def tokens(source):
    text = source.get("text")
    if text is None:
        text = "".join(
            m["role"] + "\n" + m["content"] + "\n" for m in source["messages"]
        )
    return text.encode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def reply(self, body, content_type="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/system":
            value = {
                "engines": [
                    {
                        "model": "fixture",
                        "state": "ready",
                        "context_limit": self.server.limit,
                    }
                ]
            }
        elif self.path == "/v1/recent-requests":
            value = {"requests": [self.server.row]}
        else:
            self.send_error(404)
            return
        self.reply(json.dumps(value).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ids = tokens(body)
        if self.path == "/v1/token/encode":
            self.reply(json.dumps({"tokens": list(ids), "length": len(ids)}).encode())
            return
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        server = self.server
        server.requests += 1
        cached = tokens({"messages": body["messages"][:2]})
        reused = len(cached) if cached in server.cache else 0
        if reused:
            server.cache.remove(cached)
        server.cache.append(cached)
        server.cache = server.cache[-server.slots :]
        values = re.findall(
            r"ARCHIVE MARKER \w+: (\d{6})\.", body["messages"][0]["content"]
        )
        answer = "|".join(values)
        if server.requests == server.bad_answer_at:
            answer = "wrong archive"
        usage = {
            "prompt_tokens": len(ids),
            "completion_tokens": 12,
            "reused_tokens": reused,
            "prompt_tokens_details": {"cached_tokens": reused},
        }
        rid = "fixture-1" if server.duplicate_ids else f"fixture-{server.requests}"
        server.row = dict(
            usage,
            request_id=rid,
            model="fixture",
            status=200,
            outcome="ok",
            stream=True,
            finish_reason="stop",
            prefill_ms=4,
            time_to_first_token_ms=7,
            queue_wait_ms=1,
            decode_tokens_per_second=50 if server.report_decode else None,
            endpoint="/v1/chat/completions",
            prefix_entries=len(server.cache),
            prefix_bytes=sum(len(entry) * 8 for entry in server.cache),
            synthetic_sizes=[len(entry) * 8 for entry in server.cache],
        )
        server.log_rows.append(server.row)
        chunks = [
            {"id": rid, "choices": [{"delta": {"content": answer}}]},
            {
                "id": rid,
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": usage,
            },
        ]
        self.reply(
            (
                "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                + "data: [DONE]\n\n"
            ).encode(),
            "text/event-stream",
        )


class PrefixReturnsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.limit = 262144
        self.server.bad_answer_at = 0
        self.server.report_decode = True
        self.server.duplicate_ids = False
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def args(self, slots, expect, sizes):
        self.server.slots = slots
        self.server.cache = []
        self.server.requests = 0
        self.server.log_rows = []
        return argparse.Namespace(
            url=f"http://127.0.0.1:{self.server.server_port}",
            model="fixture",
            contexts=sizes,
            tolerance=128,
            max_tokens=32,
            timeout=15,
            api_key_env=None,
            seed="matched-arms",
            expect_return=expect,
            output=Path(self.tmp.name) / f"{slots}.json",
        )

    def test_matched_arms_return_the_same_token_ids_and_distinct_known_answers(self):
        arms = []
        for slots, expected in [(1, "evicted"), (2, "retained")]:
            args = self.args(slots, expected, [2048, 16_384, 262_144])
            self.assertTrue(driver.run(args))
            depths = json.loads(args.output.read_text())["depths"]
            arms.append([[t["prompt_ids_sha256"] for t in d["turns"]] for d in depths])
            for row in depths:
                cold, adjacent, other, back = row["turns"]
                self.assertTrue(all(t["recall_passed"] for t in row["turns"]))
                self.assertLessEqual(
                    max(t["prompt_tokens"] + args.max_tokens for t in row["turns"]),
                    row["context_tokens"],
                )
                self.assertEqual(cold["reused_tokens"], 0)
                self.assertGreater(
                    adjacent["reused_tokens"], row["common_prefix_tokens"]
                )
                self.assertNotEqual(cold["expected_answer"], other["expected_answer"])
                self.assertEqual(
                    {cold["prompt_ids_sha256"], adjacent["prompt_ids_sha256"]},
                    {back["prompt_ids_sha256"]},
                )
                self.assertNotEqual(
                    back["prompt_ids_sha256"], other["prompt_ids_sha256"]
                )
                self.assertEqual(
                    back["reused_tokens"],
                    0 if slots == 1 else adjacent["reused_tokens"],
                )
                self.assertEqual(
                    back["new_prompt_tokens"],
                    back["prompt_tokens"] - back["reused_tokens"],
                )
                self.assertEqual(
                    back["server_new_prompt_tokens_per_second"],
                    back["new_prompt_tokens"] * 250,
                )
                self.assertEqual(back["server_time_to_first_token_ms"], 7)
                self.assertEqual(back["server_decode_tokens_per_second"], 50)
        self.assertEqual(*arms)

    def test_checkpoint_record_matches_completed_chat_rows(self):
        args = self.args(2, "retained", [2048, 262_144])
        self.server.limit = 65_536
        self.assertFalse(driver.run(args))
        benchmark = json.loads(args.output.read_text())
        rows = self.server.log_rows
        log = ["server ready\n", *(json.dumps(row) for row in reversed(rows))]
        result = checkpoints.record_checkpoint_bytes(benchmark, log)
        self.assertEqual(result["capacity_refused_contexts"], [262_144])
        samples = result["samples"]
        self.assertEqual([row["prefix_entries"] for row in samples], [1, 1, 2, 2])
        self.assertEqual([row["prefix_bytes"] for row in samples], [
            sum(checkpoint_bytes for checkpoint_bytes in row["synthetic_sizes"])
            for row in rows
        ])
        self.assertGreater(samples[-1]["reused_tokens"], 0)
        self.assertNotIn("prompt", samples[-1])
        self.assertNotIn("synthetic_sizes", samples[-1])

        invalid_logs = [
            log[:-1],
            [*log, log[-1]],
            [*log[:-1], json.dumps(dict(rows[0], reused_tokens=100))],
            [*log[:-1], json.dumps(dict(rows[0], prefix_bytes=True))],
        ]
        for invalid in invalid_logs:
            with self.subTest(invalid=invalid[-1]):
                with self.assertRaises(ValueError):
                    checkpoints.record_checkpoint_bytes(benchmark, invalid)

    def test_capacity_refusals_are_recorded_without_sending_inference(self):
        self.server.limit = 65_536
        args = self.args(1, "evicted", [2048, 131072, 262144])
        self.assertFalse(driver.run(args))
        rows = json.loads(args.output.read_text())["depths"]
        self.assertEqual(
            [r["state"] for r in rows],
            ["passed", "capacity_refused", "capacity_refused"],
        )
        self.assertEqual(self.server.requests, 4)

    def test_return_must_recover_the_expected_archive_and_expected_reuse(self):
        for slots, bad_answer in [(1, 0), (2, 4)]:
            self.server.bad_answer_at = bad_answer
            args = self.args(slots, "retained", [2048, 16384])
            self.assertFalse(driver.run(args))
            (row,) = json.loads(args.output.read_text())["depths"]
            self.assertEqual(row["state"], "failed")
            self.assertEqual(row["turns"][-1]["recall_passed"], not bad_answer)
            self.assertEqual(row["turns"][-1]["reuse_passed"], bool(bad_answer))
            self.assertEqual(self.server.requests, 4)

    def test_missing_decode_telemetry_cannot_qualify_timings(self):
        self.server.report_decode = False
        args = self.args(2, "retained", [2048])
        self.assertFalse(driver.run(args))
        self.assertEqual(
            json.loads(args.output.read_text())["depths"][0]["state"],
            "timing_incomplete",
        )

    def test_repeated_request_id_preserves_partial_evidence_and_refuses_overwrite(self):
        self.server.duplicate_ids = True
        args = self.args(1, "evicted", [2048])
        with self.assertRaisesRegex(ValueError, "repeated a request ID"):
            driver.run(args)
        partial = args.output.with_name(args.output.name + ".partial")
        self.assertEqual(json.loads(partial.read_text())["state"], "failed")
        self.assertFalse(args.output.exists())
        before = partial.read_bytes()
        with self.assertRaises(FileExistsError):
            driver.run(args)
        self.assertEqual(partial.read_bytes(), before)
