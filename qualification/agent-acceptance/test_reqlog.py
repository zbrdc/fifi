"""Self-test of the request-log checks. Run, from this directory:
python3 -I -m unittest discover -s . -p 'test_*.py'
The fixture testdata/reqlog-ok.jsonl has the keys of engine/src/server/reqlog.rs `render`.
"""

import json
import pathlib
import re
import tempfile
import unittest

import check_server_log
import reqlog_check

HERE = pathlib.Path(__file__).parent
FIXTURE = HERE / "testdata" / "reqlog-ok.jsonl"
RENDER = HERE.parent.parent / "engine" / "src" / "server" / "reqlog.rs"


def lines():
    return [json.loads(line) for line in FIXTURE.read_text().splitlines()]


class Shape(unittest.TestCase):
    def test_the_fixture_has_exactly_the_documented_keys(self):
        for entry in lines():
            self.assertEqual(tuple(entry), reqlog_check.KEYS)

    def test_the_keys_match_the_server_source_when_it_is_in_the_checkout(self):
        if not RENDER.is_file():
            self.skipTest("engine/src/server/reqlog.rs is not in this checkout")
        text = RENDER.read_text()
        body = text[text.index("fn render"):]
        body = body[:body.index(".to_string()")]
        self.assertEqual(tuple(re.findall(r'^\s*"(\w+)":', body, re.MULTILINE)), reqlog_check.KEYS)


class Verdicts(unittest.TestCase):
    def test_status_200_with_outcome_ok_is_served(self):
        self.assertTrue(reqlog_check.all_served_ok(lines()))

    def test_other_statuses_and_outcomes_are_not(self):
        for change in ({"status": 409}, {"status": None}, {"outcome": "error"},
                       {"outcome": "cancelled"}):
            with self.subTest(change=change):
                self.assertFalse(reqlog_check.all_served_ok([dict(lines()[0], **change)]))


class PostCheck(unittest.TestCase):
    def run_check(self, tmp, entries, result):
        log = pathlib.Path(tmp) / "server.log"
        log.write_text("startup: loading\n" + "".join(json.dumps(e) + "\n" for e in entries))
        path = pathlib.Path(tmp) / "result.json"
        path.write_text(json.dumps(result))
        status = check_server_log.main(["--log", str(log), "--result", str(path)])
        return status, json.loads(path.read_text())

    def test_a_matching_clean_log_keeps_the_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, result = self.run_check(tmp, lines(), {"verdict": "pass", "requests_sent": 3})
            self.assertEqual((status, result["verdict"]), (0, "pass"))

    def test_a_missing_line_turns_a_pass_into_a_pipeline_defect(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, result = self.run_check(tmp, lines()[:2], {"verdict": "pass", "requests_sent": 3})
            self.assertEqual((status, result["verdict"]), (2, "pipeline_defect"))

    def test_a_context_limit_line_turns_a_pass_into_a_pipeline_defect(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = dict(lines()[2], status=409, outcome="error", error_code="context_limit")
            status, result = self.run_check(tmp, [*lines()[:2], bad],
                                            {"verdict": "pass", "requests_sent": 3})
            self.assertEqual((status, result["verdict"]), (2, "pipeline_defect"))

    def test_a_quality_miss_stays_a_quality_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, result = self.run_check(tmp, lines()[:1],
                                            {"verdict": "quality_miss", "requests_sent": 3})
            self.assertEqual((status, result["verdict"]), (2, "quality_miss"))


if __name__ == "__main__":
    unittest.main()
