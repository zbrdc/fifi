"""CLI tests for the advisory Git growth report."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "size_report.py"
TEST_SCRATCH = Path(os.environ["TMPDIR"])


class SizeReportTests(unittest.TestCase):
    def setUp(self):
        if not TEST_SCRATCH.is_dir():
            self.fail("TMPDIR must name the task scratch directory")
        self.temporary = tempfile.TemporaryDirectory(
            prefix="size-report-", dir=str(TEST_SCRATCH))
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name) / "repo"
        self.repo.mkdir()
        self.git("init", "--quiet")

    def git(self, *args, check=True):
        completed = subprocess.run(
            ["git", *args], cwd=str(self.repo), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True)
        if check and completed.returncode != 0:
            self.fail("git command failed: %s\n%s" % (args, completed.stderr))
        return completed

    def write(self, relative_path, content):
        path = self.repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8", newline="")
        return path

    def commit(self, message):
        self.git("add", "-A")
        self.git("-c", "user.name=Growth Report Test", "-c",
                 "user.email=growth-report@example.invalid", "commit",
                 "--quiet", "-m", message)
        return self.git("rev-parse", "HEAD").stdout.strip()

    def info_attributes(self, content):
        path = self.repo / ".git" / "info" / "attributes"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def assert_text_payload_counts(self, report):
        self.assertEqual(len(report["files"]), 1)
        item = report["files"][0]
        self.assertEqual(item["path"], "payload.dat")
        self.assertEqual(item["kind"], "text")
        self.assertEqual(item["added_nonblank"], 2)
        self.assertEqual(item["removed_nonblank"], 1)
        self.assertEqual(report["totals"]["added_nonblank"], 2)
        self.assertEqual(report["totals"]["removed_nonblank"], 1)
        self.assertEqual(report["totals"]["binary_files"], 0)

    def report(self, *args, cwd=None):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), *args], cwd=str(cwd or self.repo),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            self.fail("size report failed: %s\n%s" % (args, completed.stderr))
        report = json.loads(completed.stdout)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(set(report), {
            "schema_version", "basis", "totals", "categories", "files",
            "unclassified_paths", "classification_limits",
        })
        self.assertIn("added_nonblank", report["totals"])
        self.assertIn("removed_nonblank", report["totals"])
        for item in report["files"]:
            self.assertIn("added_nonblank", item)
            self.assertIn("removed_nonblank", item)
        return report

    def test_staged_mode_reads_index_instead_of_working_tree(self):
        path = self.write("engine/src/value.rs", "before\n")
        self.commit("baseline")

        path.write_text("staged first\nstaged second\n", encoding="utf-8")
        self.git("add", "engine/src/value.rs")
        path.write_text(
            "unstaged first\nunstaged second\nunstaged third\n", encoding="utf-8")

        report = self.report("--staged", "--json")
        self.assertEqual(report["basis"]["type"], "staged")
        self.assertEqual(report["files"][0]["added_nonblank"], 2)
        self.assertEqual(report["files"][0]["removed_nonblank"], 1)
        self.assertEqual(report["files"][0]["category"], "production")
        self.assertEqual(report["files"][0]["kind"], "text")
        self.assertNotIn("unstaged", json.dumps(report))

    def test_staged_mode_uses_index_attributes_not_working_tree_attributes(self):
        self.write(".gitattributes", "*.dat -diff\n")
        self.write("payload.dat", "before\n")
        self.commit("binary attribute baseline")

        self.write(".gitattributes", "*.dat diff\n")
        self.write("payload.dat", "staged first\nstaged second\n")
        self.git("add", "-A")
        self.write(".gitattributes", "*.dat -diff\n")

        report = self.report("--staged", "--json")
        item = next(record for record in report["files"]
                    if record["path"] == "payload.dat")
        self.assertEqual(item["kind"], "text")
        self.assertEqual(item["added_nonblank"], 2)
        self.assertEqual(item["removed_nonblank"], 1)

    def test_staged_binary_attribute_comes_from_index_not_working_tree(self):
        self.write(".gitattributes", "*.dat diff\n")
        self.write("payload.dat", "before\n")
        self.commit("text attribute baseline")

        self.write(".gitattributes", "*.dat -diff\n")
        self.write("payload.dat", "staged first\nstaged second\n")
        self.git("add", "-A")
        self.write(".gitattributes", "*.dat diff\n")

        report = self.report("--staged", "--json")
        item = next(record for record in report["files"]
                    if record["path"] == "payload.dat")
        self.assertEqual(item["kind"], "binary")
        self.assertIsNone(item["added_nonblank"])
        self.assertIsNone(item["removed_nonblank"])

    def test_staged_comparison_ignores_repository_local_attribute_override(self):
        self.write(".gitattributes", "*.dat diff\n")
        self.write("payload.dat", "before\n")
        self.commit("text attribute baseline")
        self.write("payload.dat", "staged first\nstaged second\n")
        self.git("add", "payload.dat")
        info_attributes = self.info_attributes("*.dat -diff\n")
        info_before = info_attributes.read_bytes()

        report = self.report("--staged", "--json")

        self.assert_text_payload_counts(report)
        self.assertEqual(info_attributes.read_bytes(), info_before)

    def test_commit_comparison_ignores_repository_local_attribute_override(self):
        self.write(".gitattributes", "*.dat diff\n")
        self.write("payload.dat", "before\n")
        base = self.commit("text attribute base")
        self.write("payload.dat", "head first\nhead second\n")
        head = self.commit("text attribute head")
        info_attributes = self.info_attributes("*.dat -diff\n")
        info_before = info_attributes.read_bytes()

        report = self.report("--base", base, "--head", head, "--json")

        self.assert_text_payload_counts(report)
        self.assertEqual(info_attributes.read_bytes(), info_before)

    def test_commit_comparison_is_independent_of_live_index_attributes(self):
        self.write(".gitattributes", "*.dat diff\n")
        self.write("payload.dat", "before\n")
        base = self.commit("text attribute base")
        self.write("payload.dat", "head first\nhead second\n")
        head = self.commit("text attribute head")

        aligned = self.report("--base", base, "--head", head, "--json")
        self.assert_text_payload_counts(aligned)

        self.write(".gitattributes", "*.dat binary\n")
        self.git("add", ".gitattributes")
        self.write(".gitattributes", "*.dat diff\n")
        self.assertEqual(self.git("show", ":.gitattributes").stdout,
                         "*.dat binary\n")
        self.assertEqual((self.repo / ".gitattributes").read_text(encoding="utf-8"),
                         "*.dat diff\n")

        conflicting = self.report("--base", base, "--head", head, "--json")
        self.assert_text_payload_counts(conflicting)
        self.assertEqual(conflicting, aligned)

    def test_commit_patch_uses_selected_snapshot_diff_attribute(self):
        self.write(".gitattributes", "*.dat diff\n")
        self.write("payload.dat", b"before\x00payload\n")
        base = self.commit("explicit text diff base")
        self.write("payload.dat", b"head first\x00payload\nhead second\n")
        head = self.commit("explicit text diff head")

        report = self.report("--base", base, "--head", head, "--json")

        self.assert_text_payload_counts(report)

    def test_commit_mode_uses_head_attributes_not_working_tree_attributes(self):
        self.write(".gitattributes", "*.dat -diff\n")
        self.write("payload.dat", "before\n")
        base = self.commit("binary attribute baseline")

        self.write(".gitattributes", "*.dat diff\n")
        self.write("payload.dat", "head first\nhead second\n")
        head = self.commit("text attribute head")
        self.write(".gitattributes", "*.dat -diff\n")

        report = self.report("--base", base, "--head", head, "--json")
        item = next(record for record in report["files"]
                    if record["path"] == "payload.dat")
        self.assertEqual(item["kind"], "text")
        self.assertEqual(item["added_nonblank"], 2)
        self.assertEqual(item["removed_nonblank"], 1)

    def test_unborn_staged_report_preserves_index_entries_refs_and_files(self):
        self.write("new.txt", "staged content\n")
        self.git("add", "new.txt")
        working_file = self.write("new.txt", "unstaged content\n")
        entries_before = self.git("ls-files", "--stage", "-z").stdout
        refs_before = self.git("for-each-ref", "--format=%(refname) %(objectname)").stdout
        working_before = working_file.read_bytes()
        config_path = self.repo / ".git" / "config"
        config_before = config_path.read_bytes()
        head_before = self.git("rev-parse", "--verify", "--quiet", "HEAD", check=False)
        self.assertNotEqual(head_before.returncode, 0)

        report = self.report("--staged", "--json")

        self.assertEqual(report["totals"]["added_nonblank"], 1)
        self.assertEqual(self.git("ls-files", "--stage", "-z").stdout,
                         entries_before)
        self.assertEqual(self.git("for-each-ref", "--format=%(refname) %(objectname)").stdout,
                         refs_before)
        self.assertNotEqual(
            self.git("rev-parse", "--verify", "--quiet", "HEAD", check=False).returncode,
            0)
        self.assertEqual(working_file.read_bytes(), working_before)
        self.assertEqual(config_path.read_bytes(), config_before)

    def test_staged_mode_reports_additions_before_the_first_commit(self):
        self.write("new.txt", "first\n\n  \nlast")
        self.git("add", "new.txt")

        report = self.report("--staged", "--json")
        self.assertEqual(report["basis"]["base_revision"], None)
        self.assertEqual(report["totals"]["added_nonblank"], 2)
        self.assertEqual(report["totals"]["removed_nonblank"], 0)
        self.assertEqual(report["files"][0]["status"], "added")

    def test_commit_mode_compares_the_requested_revisions(self):
        self.write("engine/src/item.rs", "old value\n")
        base = self.commit("base")
        self.write("engine/src/item.rs", "new value\n")
        head = self.commit("head")

        report = self.report("--base", base, "--head", head, "--json")
        self.assertEqual(report["basis"]["base_revision"], base)
        self.assertEqual(report["basis"]["head_revision"], head)
        self.assertEqual(report["totals"]["added_nonblank"], 1)
        self.assertEqual(report["totals"]["removed_nonblank"], 1)

    def test_add_delete_rename_blank_lines_and_unusual_names(self):
        self.write("old.txt", "alpha\n\nold without newline")
        self.write("gone.txt", "removed\n")
        self.write("rename.txt", "same content\n")
        self.commit("baseline")

        self.write("old.txt", "alpha\n\nnew without newline")
        (self.repo / "gone.txt").unlink()
        self.write("added.txt", "first\n\n   \nomega")
        self.git("mv", "rename.txt", "renamed.txt")
        unusual_name = "space name\tline\nλ.txt"
        self.write(unusual_name, "unicode line")
        self.git("add", "-A")

        report = self.report("--staged", "--json")
        self.assertEqual(report["totals"]["added_nonblank"], 4)
        self.assertEqual(report["totals"]["removed_nonblank"], 2)
        files = {item["path"]: item for item in report["files"]}
        self.assertEqual(files["gone.txt"]["removed_nonblank"], 1)
        self.assertEqual(files["added.txt"]["added_nonblank"], 2)
        self.assertEqual(files["renamed.txt"]["status"], "renamed")
        self.assertEqual(files["renamed.txt"]["previous_path"], "rename.txt")
        self.assertEqual(files[unusual_name]["added_nonblank"], 1)
        self.assertEqual(files["old.txt"]["added_nonblank"], 1)
        self.assertEqual(files["old.txt"]["removed_nonblank"], 1)

    def test_binary_changes_are_explicit_and_not_counted_as_text(self):
        self.write("payload.bin", b"old\x00payload\n")
        self.commit("binary baseline")
        self.write("payload.bin", b"new\x00payload\n")
        self.git("add", "payload.bin")

        report = self.report("--staged", "--json")
        item = report["files"][0]
        self.assertEqual(item["kind"], "binary")
        self.assertIsNone(item["added_nonblank"])
        self.assertIsNone(item["removed_nonblank"])
        self.assertEqual(report["totals"]["binary_files"], 1)
        self.assertEqual(report["totals"]["added_nonblank"], 0)

    def test_file_symlink_type_transitions_in_staged_and_commit_modes(self):
        path = self.repo / "item.txt"
        self.write("item.txt", "old first\nold second\n")
        file_base = self.commit("regular file baseline")

        path.unlink()
        os.symlink("link-target", str(path))
        self.git("add", "item.txt")
        staged = self.report("--staged", "--json")
        self.assertEqual(len(staged["files"]), 1)
        self.assertEqual(staged["files"][0]["status"], "type_changed")
        self.assertEqual(staged["files"][0]["kind"], "symlink")
        self.assertEqual(staged["files"][0]["added_nonblank"], 1)
        self.assertEqual(staged["files"][0]["removed_nonblank"], 2)

        symlink_head = self.commit("replace file with symlink")
        committed = self.report(
            "--base", file_base, "--head", symlink_head, "--json")
        self.assertEqual(len(committed["files"]), 1)
        self.assertEqual(committed["files"][0]["status"], "type_changed")
        self.assertEqual(committed["files"][0]["kind"], "symlink")
        self.assertEqual(committed["files"][0]["added_nonblank"], 1)
        self.assertEqual(committed["files"][0]["removed_nonblank"], 2)

        path.unlink()
        self.write("item.txt", "new first\nnew second\n")
        self.git("add", "item.txt")
        staged = self.report("--staged", "--json")
        self.assertEqual(len(staged["files"]), 1)
        self.assertEqual(staged["files"][0]["status"], "type_changed")
        self.assertEqual(staged["files"][0]["kind"], "symlink")
        self.assertEqual(staged["files"][0]["added_nonblank"], 2)
        self.assertEqual(staged["files"][0]["removed_nonblank"], 1)

        file_head = self.commit("replace symlink with file")
        committed = self.report(
            "--base", symlink_head, "--head", file_head, "--json")
        self.assertEqual(len(committed["files"]), 1)
        self.assertEqual(committed["files"][0]["status"], "type_changed")
        self.assertEqual(committed["files"][0]["kind"], "symlink")
        self.assertEqual(committed["files"][0]["added_nonblank"], 2)
        self.assertEqual(committed["files"][0]["removed_nonblank"], 1)

    def test_symlink_target_is_reported_without_following_it(self):
        link = self.repo / "link"
        os.symlink("missing-target", str(link))
        self.git("add", "link")

        report = self.report("--staged", "--json")
        item = report["files"][0]
        self.assertEqual(item["kind"], "symlink")
        self.assertEqual(item["added_nonblank"], 1)
        self.assertEqual(item["removed_nonblank"], 0)
        self.assertFalse((self.repo / "missing-target").exists())

    def test_categories_are_separate_and_unknown_paths_are_listed(self):
        paths = {
            "engine/src/model.rs": "production",
            "engine/tests/model.rs": "tests",
            "engine/src/model_test.rs": "tests",
            "qualification/demo/run.py": "qualification",
            "scripts/helper.py": "tooling",
            "docs/design/demo.md": "docs",
            "src/generated/model.rs": "generated",
            "engine/vendor/copy.rs": "vendor",
            "testdata/prompt.json": "fixtures",
            "engine/src/weights/gguf_tables.rs": "lookup_tables",
            "misc/unmapped.data": "unclassified",
        }
        for path in paths:
            self.write(path, "one line\n")
        self.git("add", "-A")

        report = self.report("--staged", "--json")
        by_path = {item["path"]: item for item in report["files"]}
        self.assertEqual({path: by_path[path]["category"] for path in paths}, paths)
        self.assertEqual(report["unclassified_paths"], ["misc/unmapped.data"])
        self.assertEqual(set(report["categories"]), set(paths.values()))
        self.assertTrue(any("inline" in note.lower()
                            for note in report["classification_limits"]))

    def test_repository_relative_paths_ignore_diff_relative_from_subdirectory(self):
        self.write("engine/src/value.rs", "old source\n")
        self.write("docs/guide.md", "old guide\n")
        base = self.commit("base")
        self.write("engine/src/value.rs", "new source\n")
        self.write("docs/guide.md", "new guide\n")
        head = self.commit("head")
        self.git("config", "diff.relative", "true")
        subdirectory = self.repo / "engine" / "src"
        subdirectory.mkdir(parents=True, exist_ok=True)

        report = self.report("--base", base, "--head", head, "--json",
                             cwd=subdirectory)
        self.assertEqual([item["path"] for item in report["files"]],
                         ["docs/guide.md", "engine/src/value.rs"])
        self.assertEqual(report["totals"]["changed_files"], 2)

    def test_submodule_output_is_stable_when_diff_submodule_is_log(self):
        source = Path(self.temporary.name) / "submodule-source"
        source.mkdir()
        subprocess.run(["git", "init", "--quiet"], cwd=str(source), check=True)
        (source / "README.md").write_text("first\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=str(source), check=True)
        subprocess.run(["git", "-c", "user.name=Growth Report Test", "-c",
                        "user.email=growth-report@example.invalid", "commit",
                        "--quiet", "-m", "submodule base"], cwd=str(source),
                       check=True)
        self.git("-c", "protocol.file.allow=always", "submodule", "add",
                 "--quiet", str(source), "deps/example")
        base = self.commit("submodule baseline")

        (source / "README.md").write_text("first\nsecond\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=str(source), check=True)
        subprocess.run(["git", "-c", "user.name=Growth Report Test", "-c",
                        "user.email=growth-report@example.invalid", "commit",
                        "--quiet", "-m", "submodule update"], cwd=str(source),
                       check=True)
        submodule_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(source), text=True).strip()
        submodule = self.repo / "deps" / "example"
        subprocess.run(["git", "-c", "protocol.file.allow=always", "-C",
                        str(submodule), "fetch", "--quiet", "origin"], check=True)
        subprocess.run(["git", "-C", str(submodule), "checkout", "--quiet",
                        "--detach", submodule_revision], check=True)
        self.git("add", "deps/example")
        head = self.commit("advance submodule")
        self.git("config", "diff.submodule", "log")

        report = self.report("--base", base, "--head", head, "--json")
        self.assertEqual(len(report["files"]), 1)
        self.assertEqual(report["files"][0]["path"], "deps/example")
        self.assertEqual(report["files"][0]["kind"], "submodule")
        self.assertEqual(report["totals"]["submodule_files"], 1)
        self.assertEqual(report["totals"]["added_nonblank"], 0)
        self.assertEqual(report["totals"]["removed_nonblank"], 0)

    def test_generated_multi_extension_filename_is_classified(self):
        self.write("engine/src/value.generated.d.ts", "old declaration\n")
        base = self.commit("generated base")
        self.write("engine/src/value.generated.d.ts", "new declaration\n")
        head = self.commit("generated head")

        report = self.report("--base", base, "--head", head, "--json")
        self.assertEqual(report["files"][0]["category"], "generated")

    def test_invalid_revisions_and_usage_errors_exit_nonzero(self):
        self.write("tracked.txt", "content\n")
        self.commit("baseline")

        invalid = subprocess.run(
            [sys.executable, str(SCRIPT), "--base", "not-a-revision",
             "--head", "HEAD", "--json"], cwd=str(self.repo),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertNotEqual(invalid.returncode, 0)
        self.assertEqual(invalid.stdout, "")

        usage = subprocess.run(
            [sys.executable, str(SCRIPT), "--base", "HEAD", "--json"],
            cwd=str(self.repo), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True)
        self.assertNotEqual(usage.returncode, 0)
        self.assertEqual(usage.stdout, "")


if __name__ == "__main__":
    unittest.main()
