"""Self-test of summarize.py on synthetic CSVs. Run: python3 -I test_summarize.py"""
import csv
import importlib.util
import pathlib
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("summarize", HERE / "summarize.py")
summarize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarize)

COLUMNS = ["shape", "mode", "events", "call", "expert", "total_ns"] + list(summarize.PHASES) + [
    "other_ns", "memset_win_ns", "kernel_win_ns", "floor_ns"]
SHAPES = list(summarize.COUNTS)


def write_run(directory, calls, malloc=100, free=50, shape_totals=None, totals=None,
              kernel_win=-1, floor=200):
    """Writes the four CSVs. A call's total is shape_totals[shape], else totals[call].

    kernel_win is written only in the events runs (the host runs hold -1, as bench.cu does).
    """
    for mode in summarize.MODES:
        for events in (0, 1):
            path = directory / f"{mode}-ev{events}.csv"
            with open(path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(COLUMNS)
                for shape in SHAPES:
                    for call in range(calls):
                        total = shape_totals[shape] if shape_totals else totals[call]
                        phases = [malloc] * 3 + [10, 10, 10, 10] + [free] * 3
                        window = kernel_win if events else -1
                        writer.writerow([shape, mode, events, call, 0, total] + phases + [0, -1, window, floor])


def median_total(directory, shape):
    result = summarize.summarize(directory)
    return result["modes"]["cold"]["shapes"][shape]["total_ns"]


# 150 calls of 1000 ns and 50 of 5000 ns: median 1000, mean 2000, p90 5000 (distinct on purpose).
SKEWED = [1000] * 150 + [5000] * 50
SKEWED_MEDIAN, SKEWED_MEAN, SKEWED_P90 = 1000, 2000, 5000
KERNEL_WIN, FLOOR = 3000, 2200


def token_ms(per_call_ns):
    """Per-token milliseconds when every one of the 1271 calls costs per_call_ns."""
    return 1271 * per_call_ns / 1e6


def skewed_summary():
    """Returns (result dict, markdown text) of a run with skewed totals."""
    with tempfile.TemporaryDirectory() as tmp:
        write_run(pathlib.Path(tmp), 200, totals=SKEWED, kernel_win=KERNEL_WIN, floor=FLOOR)
        result = summarize.summarize(tmp)
    return result, summarize.markdown(result)


def token_table(markdown, mode):
    """Returns (header cells, {row label: value cells}) of the per-token table of one mode."""
    lines = markdown.split("\n")
    start = lines.index(f"## {mode}: implied cost per token, milliseconds")
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    header, body = rows[0], rows[2:]
    return header, {row[0]: row[1:] for row in body}


def find_row(table, prefix):
    matches = [label for label in table if label.startswith(prefix)]
    assert len(matches) == 1, (prefix, list(table))
    return table[matches[0]]


class SummarizeTest(unittest.TestCase):
    def test_p90_is_nearest_rank(self):
        values = list(range(1, 201))
        self.assertEqual(summarize.p90(values), 180)
        self.assertEqual(summarize.median(values), 100.5)

    def test_per_token_is_count_times_median(self):
        # Counts written out here, independent of summarize.COUNTS, in the order of SHAPES.
        counts = [40, 100, 40, 40, 30, 60, 640, 320, 1]
        shape_totals = {shape: 1000 * (i + 1) for i, shape in enumerate(SHAPES)}
        expected_ns = sum(c * 1000 * (i + 1) for i, c in enumerate(counts))
        with tempfile.TemporaryDirectory() as tmp:
            write_run(pathlib.Path(tmp), 200, shape_totals=shape_totals)
            result = summarize.summarize(tmp)
        cold = result["modes"]["cold"]["per_token_ms"]
        self.assertEqual(sum(counts), 1271)
        self.assertAlmostEqual(cold["total"]["median"], expected_ns / 1e6)
        self.assertAlmostEqual(cold["floor"]["median"], 1271 * 200 / 1e6)

    def test_fewer_than_200_calls_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(pathlib.Path(tmp), 199, shape_totals={s: 1000 for s in SHAPES})
            with self.assertRaises(SystemExit):
                summarize.summarize(tmp)

    def test_alloc_free_and_rest_are_per_call_sums(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(pathlib.Path(tmp), 200, malloc=100, free=50, shape_totals={s: 1000 for s in SHAPES})
            result = summarize.summarize(tmp)
        shape = result["modes"]["warm"]["shapes"]["attn_q"]
        self.assertEqual(shape["alloc_free_ns"]["median"], 450)
        self.assertEqual(shape["non_alloc_ns"]["median"], 550)

    def test_p90_follows_the_slow_tail(self):
        # 20 slow calls of 200 stay above the 90th percentile; 21 slow calls reach it.
        with tempfile.TemporaryDirectory() as tmp:
            write_run(pathlib.Path(tmp), 200, totals=[1000] * 180 + [5000] * 20)
            total = median_total(tmp, "exp_down")
        self.assertEqual((total["median"], total["p90"]), (1000, 1000))
        with tempfile.TemporaryDirectory() as tmp:
            write_run(pathlib.Path(tmp), 200, totals=[1000] * 179 + [5000] * 21)
            total = median_total(tmp, "exp_down")
        self.assertEqual((total["median"], total["p90"]), (1000, 5000))

    def test_per_token_mean_is_count_times_mean(self):
        result, _ = skewed_summary()
        cold = result["modes"]["cold"]["per_token_ms"]
        self.assertAlmostEqual(cold["total"]["mean"], token_ms(SKEWED_MEAN))
        self.assertAlmostEqual(cold["total"]["median"], token_ms(SKEWED_MEDIAN))
        self.assertAlmostEqual(cold["non_alloc"]["mean"], token_ms(SKEWED_MEAN - 450))
        self.assertAlmostEqual(cold["alloc_free"]["mean"], token_ms(450))

    def test_in_call_gpu_window_comes_from_the_events_run(self):
        # The host runs hold -1 in kernel_win_ns; only the events runs carry the window.
        result, _ = skewed_summary()
        for mode in summarize.MODES:
            window = result["modes"][mode]["per_token_ms"]["in_call_gpu_window"]
            for which in ("median", "mean", "p90"):
                self.assertAlmostEqual(window[which], token_ms(KERNEL_WIN))
            floor = result["modes"][mode]["per_token_ms"]["floor"]
            self.assertAlmostEqual(floor["median"], token_ms(FLOOR))

    def test_markdown_per_token_columns(self):
        _, markdown = skewed_summary()
        header, table = token_table(markdown, "cold")
        self.assertEqual(header, ["quantity"] + list(summarize.TOKEN_COLUMNS.values()))
        self.assertEqual(header[1], "median-based (sum of per-call medians)")
        row = find_row(table, "all calls as measured")
        expected = [token_ms(SKEWED_MEDIAN), token_ms(SKEWED_MEAN), token_ms(SKEWED_P90)]
        self.assertEqual(row, [f"{value:.2f}" for value in expected])

    def test_markdown_p90_column_says_it_is_not_a_per_token_p90(self):
        _, markdown = skewed_summary()
        header, _ = token_table(markdown, "warm")
        self.assertEqual(header[3], "sum of per-call p90 (loose upper bound, not a per-token p90)")

    def test_markdown_mean_column_is_labelled_expected_cost(self):
        _, markdown = skewed_summary()
        header, table = token_table(markdown, "warm")
        self.assertEqual(header[2], "mean-based (sum of per-call means, the expected cost)")
        self.assertEqual(find_row(table, "all calls as measured")[1], f"{token_ms(SKEWED_MEAN):.2f}")

    def test_markdown_in_call_window_row(self):
        _, markdown = skewed_summary()
        for mode in summarize.MODES:
            _, table = token_table(markdown, mode)
            row = find_row(table, "GPU pack+matmul window inside the call")
            self.assertEqual(row, [f"{token_ms(KERNEL_WIN):.2f}"] * 3)

    def test_markdown_floor_row_is_labelled_cold_biased(self):
        _, markdown = skewed_summary()
        _, table = token_table(markdown, "cold")
        label = ("GPU pack+matmul on persistent buffers launched right after the L2 flush "
                 "(cold: biased high by L2/TLB misses)")
        self.assertEqual(table[label], [f"{token_ms(FLOOR):.2f}"] * 3)
        self.assertNotIn("pack+matmul only, buffers persistent", markdown)

    def test_markdown_non_alloc_row_states_persistent_workspace_cost(self):
        _, markdown = skewed_summary()
        _, table = token_table(markdown, "cold")
        label = ("without malloc/free = cost of a persistent-workspace call "
                 "(H2D, memset, launch, sync, D2H still paid)")
        self.assertEqual(table[label][0], f"{token_ms(SKEWED_MEDIAN - 450):.2f}")

    def test_markdown_rows_keep_their_order(self):
        _, markdown = skewed_summary()
        _, table = token_table(markdown, "cold")
        self.assertEqual(list(table), list(summarize.TOKEN_ROWS.values()))
        prefixes = ["all calls", "of which", "without malloc/free", "GPU pack+matmul window", "GPU pack+matmul on"]
        self.assertEqual([label[:len(p)] for label, p in zip(table, prefixes)], prefixes)

    def test_markdown_floor_column_has_footnote(self):
        _, markdown = skewed_summary()
        self.assertEqual(markdown.count(summarize.FLOOR_NOTE), 2)
        self.assertIn("biased high by L2/TLB misses", summarize.FLOOR_NOTE)


if __name__ == "__main__":
    unittest.main()
