from pathlib import Path
import sys
from threading import Event
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_mixed_load as mixed


class MixedLoadTests(unittest.TestCase):
    def pair(self, first, admission):
        fixture = {"cold": "cold", "count": "count"}

        def simulate(client, model, name, notify):
            if name == first:
                notify("admitted")
                notify("visible")
                return dict(start=0, admitted=0.1, times=[2, 4], end=5)
            return dict(start=admission - 0.01, admitted=admission, times=[6, 7], end=8)

        with patch.object(mixed, "timed_stream", simulate):
            return mixed.run_pair(None, "model", fixture, first)

    def test_second_admitted_during_decode_or_prefill(self):
        for first, admission in [("count", 3), ("cold", 1)]:
            with self.subTest(first=first):
                self.assertEqual(set(self.pair(first, admission)), {"count", "cold"})

    def test_late_admission_rejects_false_overlap(self):
        for first, admission in [("count", 4), ("cold", 2)]:
            with self.subTest(first=first), self.assertRaisesRegex(ValueError, "admitted"):
                self.pair(first, admission)

    def test_false_warm_usage(self):
        with self.assertRaises(ValueError):
            mixed.checked_reuse({"prompt_tokens": 300, "reused_tokens": 0,
                                 "prompt_tokens_details": {"cached_tokens": 299}})

    def test_control_requires_an_ordered_count_and_length_stop(self):
        sample = {"pieces": ["\n".join(map(str, range(1, 40)))],
                  "stream": {"usage": {"completion_tokens": 128}, "finish_reason": "length"}}
        self.assertTrue(mixed.valid_count(sample))
        sample["pieces"].append("\n41")
        self.assertFalse(mixed.valid_count(sample))

    def test_backlog_starts_cold_while_both_warm_clients_are_outstanding(self):
        cold_started = Event()
        first_open = Event()
        calls = []

        def simulate(client, model, kind, notify):
            if kind == "cold":
                self.assertTrue(first_open.is_set())
                self.assertGreaterEqual(calls.count("count"), 2)
                cold_started.set()
            else:
                calls.append(kind)
                first_open.set()
                notify("admitted")
                notify("visible")
                self.assertTrue(cold_started.wait(5))
            return {"stream": {"request_id": kind}}

        with patch.object(mixed, "timed_stream", simulate), patch.object(
                mixed.common, "recent_server_metrics", return_value={}):
            a, b, cold = mixed.run_backlog(None, "model", "cold", "count", 2)
        self.assertTrue(1 <= len(a) <= 2 and 1 <= len(b) <= 2)
        self.assertEqual(cold["stream"]["request_id"], "cold")

    def test_fresh_rate_excludes_restored_prompt_tokens(self):
        r = {"stream": {"request_id": "count", "finish_reason": "length", "usage": {
                 "prompt_tokens": 100, "reused_tokens": 90, "completion_tokens": 2}},
             "times": [2, 3], "pieces": ["1", "2"], "start": 0, "admitted": 1, "end": 4,
             "metrics": {"server_queue_wait_ms": 0, "server_prefill_ms": 50,
                         "server_decode_tokens_per_second": 1}}
        row = mixed.summary(None, "model", r, mixed.Prompt([], 100, 2, "test"), 0)
        self.assertEqual(row["fresh_tokens"], 10)
        self.assertEqual(row["fresh_prefill_tokens_per_second"], 200)
        self.assertIsNone(row["p95_visible_gap_ms"])

    def test_backlog_counts_overlapping_requests_once_and_preserves_gaps(self):
        cold = {"admitted": 4, "end": 14}
        warm = [{"admitted": a, "end": b} for a, b in [(1, 6), (2, 8), (7, 9), (10, 16), (20, 30)]]
        measured = mixed.backlog_overlap(cold, warm)
        self.assertEqual(measured["warm_admitted_during_cold"], 2)
        self.assertEqual(measured["warm_completed_during_cold"], 3)
        self.assertAlmostEqual(measured["cold_interval_with_warm_outstanding_fraction"], 0.9)

    def test_percentile_floor_and_nearest_rank(self):
        self.assertIsNone(mixed.p95(list(range(19))))
        self.assertEqual(mixed.p95(list(range(1, 21))), 19)
