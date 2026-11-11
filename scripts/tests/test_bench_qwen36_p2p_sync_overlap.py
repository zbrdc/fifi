import importlib.util
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "benchmarks/qwen36-p2p-sync-overlap.py"
SPEC = importlib.util.spec_from_file_location("p2p_sync_overlap", PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class IntervalTests(unittest.TestCase):
    def test_union_deduplicates_overlaps_and_touching_intervals(self):
        self.assertEqual(
            analysis.union([(8, 12), (1, 5), (4, 9), (3, 3), (20, 22)]),
            [(1, 12), (20, 22)],
        )
        with self.assertRaises(ValueError):
            analysis.union([(2, 1)])

    def test_intersection_handles_gaps_and_nested_ranges(self):
        self.assertEqual(
            analysis.intersection([(0, 5), (8, 12)], [(2, 10), (12, 13)]),
            [(2, 5), (8, 10)],
        )
        self.assertEqual(analysis.intersection([], [(0, 1)]), [])

    def test_summarize_counts_overlapping_gpu_work_once(self):
        scale = 1_000_000_000

        def intervals(pairs):
            return [(a * scale, b * scale) for a, b in pairs]

        result = analysis.summarize(
            intervals([(0, 10)]),
            {0: intervals([(0, 4)]), 1: intervals([(4, 9)])},
            {0: intervals([(4, 8)]), 1: intervals([(0, 3)])},
            {0: intervals([(0, 8)]), 1: intervals([(0, 9)])},
        )
        self.assertEqual(result["sync_union_seconds"], 10)
        self.assertEqual(result["sync_with_q4_either_gpu_seconds"], 9)
        self.assertEqual(result["sync_with_recorded_activity_either_gpu_seconds"], 9)
        self.assertEqual(result["sync_without_recorded_activity_either_gpu_seconds"], 1)
        self.assertEqual(result["sync_with_3060_wait_and_3080_q4_seconds"], 4)
        self.assertEqual(result["sync_with_3080_wait_and_3060_q4_seconds"], 3)

    def test_missing_gpu_events_and_wrong_trace_are_rejected(self):
        with self.assertRaises(ValueError):
            analysis.summarize([(0, 1)], {0: [(0, 1)]}, {}, {})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "wrong.sqlite"
            path.write_bytes(b"not the captured trace")
            with self.assertRaisesRegex(ValueError, "trace hash"):
                analysis.analyze(path)


if __name__ == "__main__":
    unittest.main()
