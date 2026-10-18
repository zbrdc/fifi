"""Integration tests for the real opt-in Git hook entry points."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_FILES = (
    ".githooks/pre-commit",
    ".githooks/commit-msg",
    "scripts/check_staged.py",
    "scripts/check_doc_links.py",
    "scripts/size_report.py",
)


class WorkflowIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path(os.environ.get("TMPDIR", "")).resolve()
        if not scratch.is_dir():
            self.fail("TMPDIR must name the task scratch directory")
        self.temporary = tempfile.TemporaryDirectory(
            prefix="workflow-integration-", dir=str(scratch)
        )
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()

        self.environment = dict(os.environ)
        for name in (
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_COMMON_DIR",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_PREFIX",
            "GIT_ATTR_SOURCE",
            "GIT_CONFIG",
            "GIT_CONFIG_GLOBAL",
            "GIT_CONFIG_SYSTEM",
            "GIT_CONFIG_COUNT",
            "GIT_CONFIG_PARAMETERS",
        ):
            self.environment.pop(name, None)
        self.environment["GIT_CONFIG_NOSYSTEM"] = "1"
        empty_global_config = self.root / "empty-global.gitconfig"
        empty_global_config.touch()
        self.environment["GIT_CONFIG_GLOBAL"] = str(empty_global_config)
        self.environment["GIT_TERMINAL_PROMPT"] = "0"

        python_directory = str(Path(sys.executable).resolve().parent)
        self.environment["PATH"] = os.pathsep.join(
            (python_directory, self.environment.get("PATH", os.defpath))
        )
        if shutil.which("python3", path=self.environment["PATH"]) is None:
            self.fail("the task Python environment must provide python3")

        for relative in REPOSITORY_FILES:
            source = ROOT / relative
            target = self.repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        disabled_hooks = self.repo / ".test-hooks-disabled"
        disabled_hooks.mkdir()
        self.git("init", "--quiet")
        self.git("config", "user.name", "Workflow integration test")
        self.git("config", "user.email", "workflow-integration@example.invalid")
        self.git("config", "core.hooksPath", str(disabled_hooks))
        self.write("app.py", "value = 1\n")
        self.git("add", "-A")
        self.git("commit", "--quiet", "-m", "test(workflow): create isolated baseline")

    def git(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", *arguments],
            cwd=self.repo,
            env=self.environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            self.fail(f"git {arguments!r} failed: {result.stderr}")
        return result

    def write(self, relative: str, content: str) -> Path:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def run_hook(self, name: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.repo / ".githooks" / name), *arguments],
            cwd=self.repo,
            env=self.environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def run_pre_commit(self) -> subprocess.CompletedProcess[str]:
        return self.run_hook("pre-commit")

    def staged_report(self, result: subprocess.CompletedProcess[str]) -> dict[str, object]:
        marker = "Advisory growth report (no blocking threshold):\n"
        self.assertIn(marker, result.stdout)
        report_text = result.stdout.split(marker, 1)[1]
        report, _ = json.JSONDecoder().raw_decode(report_text)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["basis"]["type"], "staged")
        self.assertEqual(report["basis"]["head_revision"], "index")
        return report

    def test_staged_invalid_python_fails_after_worktree_repair(self) -> None:
        self.write("app.py", "def broken(:\n    pass\n")
        self.git("add", "app.py")
        self.write("app.py", "value = 2\n")

        result = self.run_pre_commit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid Python syntax in staged file app.py", result.stderr)
        self.assertNotIn("Advisory growth report", result.stdout)
        self.assertEqual((self.repo / "app.py").read_text(encoding="utf-8"), "value = 2\n")

    def test_staged_valid_python_passes_with_invalid_worktree_and_real_report(self) -> None:
        self.write("app.py", "value = 2\n")
        self.git("add", "app.py")
        self.write("app.py", "def broken(:\n")

        result = self.run_pre_commit()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Staged Python syntax check passed (1 regular file(s)).", result.stdout)
        report = self.staged_report(result)
        self.assertEqual(report["totals"]["added_nonblank"], 1)
        self.assertEqual(report["totals"]["removed_nonblank"], 1)
        self.assertEqual(report["files"][0]["path"], "app.py")
        self.assertEqual(report["files"][0]["kind"], "text")
        self.assertEqual((self.repo / "app.py").read_text(encoding="utf-8"), "def broken(:\n")

    def test_real_growth_report_is_advisory_for_large_staged_change(self) -> None:
        line_count = 10_000
        self.write(
            "large-change.txt",
            "".join(f"synthetic line {number}\n" for number in range(line_count)),
        )
        self.git("add", "large-change.txt")

        result = self.run_pre_commit()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Advisory growth report (no blocking threshold):", result.stdout)
        report = self.staged_report(result)
        self.assertEqual(report["totals"]["added_nonblank"], line_count)
        self.assertEqual(report["totals"]["changed_files"], 1)
        self.assertEqual(report["files"][0]["path"], "large-change.txt")
        self.assertEqual(report["files"][0]["added_nonblank"], line_count)

    def test_pre_commit_rejects_document_link_to_ignored_path(self) -> None:
        self.write(".gitignore", "/docs/plans/\n")
        self.write("README.md", "See [the plan](docs/plans/phase-0a.md) for the gates.\n")
        self.git("add", "-A")

        result = self.run_pre_commit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("check_doc_links", result.stderr)
        self.assertIn("docs/plans/phase-0a.md", result.stderr)

    def test_pre_commit_accepts_document_link_to_tracked_file(self) -> None:
        self.write("docs/notes.md", "# Notes\n")
        self.write("README.md", "See [the notes](docs/notes.md) for context.\n")
        self.git("add", "-A")

        result = self.run_pre_commit()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Document link check passed", result.stdout)
        self.assertIn("Advisory growth report", result.stdout)

    def test_commit_message_hook_accepts_scoped_subject_and_rejects_unscoped(self) -> None:
        valid_message = self.write(
            "valid-message.txt", "fix(storage): preserve staged report output\n\nBody.\n"
        )
        valid_result = self.run_hook("commit-msg", str(valid_message))
        self.assertEqual(valid_result.returncode, 0, valid_result.stderr)
        self.assertIn("Conventional Commit subject passed.", valid_result.stdout)

        invalid_message = self.write("invalid-message.txt", "feat: missing scope\n")
        invalid_result = self.run_hook("commit-msg", str(invalid_message))
        self.assertNotEqual(invalid_result.returncode, 0)
        self.assertIn("Conventional Commit", invalid_result.stderr)


if __name__ == "__main__":
    unittest.main()
