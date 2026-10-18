"""Offline behavior tests for the opt-in staged Git hooks.

The isolated repositories use a narrow size_report.py test double. The combined
acceptance check must use the real growth reporter from its owning lane.
"""

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
HOOK_PATHS = (".githooks/pre-commit", ".githooks/commit-msg", ".githooks/pre-push")
COUNTER_REPORT = {
    "schema_version": 1,
    "basis": {
        "type": "staged",
        "base_revision": "0" * 40,
        "head_revision": "index",
    },
    "totals": {
        "added_nonblank": 1,
        "removed_nonblank": 0,
        "changed_files": 1,
        "binary_files": 0,
        "symlink_files": 0,
        "submodule_files": 0,
    },
    "categories": {},
    "files": [],
    "unclassified_paths": [],
    "classification_limits": [],
}


class StagedGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Staged gate test")
        self.git("config", "user.email", "staged-gates@example.invalid")

        for relative in HOOK_PATHS + ("scripts/check_staged.py", "scripts/check_doc_links.py"):
            source = ROOT / relative
            if source.is_file():
                target = self.repo / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

        self.write("scripts/size_report.py", self.counter_source(COUNTER_REPORT))
        self.write("app.py", "value = 1\n")
        self.write("rename.py", self.rename_source())
        self.write("removed.py", "value = 'retained before deletion'\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "test(staged-gates): create isolated baseline")

    def git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if check and result.returncode != 0:
            self.fail(f"git {args!r} failed: {result.stderr}")
        return result

    def write(self, relative: str, content: str) -> Path:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    @staticmethod
    def rename_source() -> str:
        return "value = 1\n" + "# retained line\n" * 12

    @staticmethod
    def counter_source(report: object, exit_status: int = 0) -> str:
        rendered = json.dumps(report)
        return (
            "#!/usr/bin/env python3\n"
            "# Narrow test double for python3 scripts/size_report.py --staged --json.\n"
            "import json, sys\n"
            "assert sys.argv[1:] == ['--staged', '--json']\n"
            f"print({rendered!r})\n"
            f"raise SystemExit({exit_status})\n"
        )

    def install_counter_double(self, report: object, exit_status: int = 0) -> None:
        self.write("scripts/size_report.py", self.counter_source(report, exit_status))

    def run_hook(
        self,
        name: str,
        *args: str,
        env: dict[str, str] | None = None,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        hook = self.repo / ".githooks" / name
        return subprocess.run(
            [str(hook), *args],
            cwd=self.repo,
            env=env,
            input=stdin,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def run_pre_commit(self) -> subprocess.CompletedProcess[str]:
        return self.run_hook("pre-commit")

    def run_pre_push(self, *lines: str) -> subprocess.CompletedProcess[str]:
        return self.run_hook("pre-push", stdin="".join(line + chr(10) for line in lines))

    def stage(self, *paths: str) -> None:
        self.git("add", "--", *paths)

    def check_message(self, text: str) -> subprocess.CompletedProcess[str]:
        message = self.write("message.txt", text)
        return self.run_hook("commit-msg", str(message))

    def test_pre_push_allows_fast_forward_to_main(self) -> None:
        old = self.git("rev-parse", "HEAD").stdout.strip()
        self.write("app.py", "value = 2\n")
        self.stage("app.py")
        self.git("commit", "-q", "-m", "test(staged-gates): advance main")
        new = self.git("rev-parse", "HEAD").stdout.strip()

        result = self.run_pre_push(f"refs/heads/main {new} refs/heads/main {old}")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Pre-push main protection passed.", result.stdout)

    def test_pre_push_blocks_non_fast_forward_to_main(self) -> None:
        old = self.git("rev-parse", "HEAD").stdout.strip()
        self.write("app.py", "value = 2\n")
        self.stage("app.py")
        self.git("commit", "--amend", "-q", "-m", "test(staged-gates): rewrite main")
        new = self.git("rev-parse", "HEAD").stdout.strip()
        self.assertNotEqual(old, new)

        result = self.run_pre_push(f"refs/heads/main {new} refs/heads/main {old}")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("non-fast-forward push to main is not allowed", result.stderr)

    def test_pre_push_blocks_deleting_main(self) -> None:
        zeros = "0" * 40

        result = self.run_pre_push(f"refs/heads/main {zeros} refs/heads/main {'a' * 40}")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("deleting main is not allowed", result.stderr)

    def test_pre_push_ignores_other_branches(self) -> None:
        result = self.run_pre_push(f"refs/heads/topic {'b' * 40} refs/heads/topic {'c' * 40}")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Pre-push main protection passed.", result.stdout)

    def test_staged_credential_pattern_is_blocked(self) -> None:
        self.write("token.txt", "key = ghp_" + "a" * 36 + chr(10))
        self.stage("token.txt")

        result = self.run_pre_commit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("credential pattern", result.stderr)

    def test_staged_credential_prefix_lookalike_passes(self) -> None:
        self.write("notes.md", "the prefix ghp_ alone is not a token" + chr(10))
        self.stage("notes.md")

        result = self.run_pre_commit()

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_staged_invalid_python_fails_even_when_worktree_is_repaired(self) -> None:
        self.write("app.py", "def broken(:\n    pass\n")
        self.stage("app.py")
        self.write("app.py", "value = 2\n")

        result = self.run_pre_commit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid Python syntax", result.stderr)
        self.assertIn("app.py", result.stderr)

    def test_staged_valid_python_passes_even_when_worktree_is_invalid(self) -> None:
        self.write("app.py", "value = 2\n")
        self.stage("app.py")
        self.write("app.py", "def broken(:\n")

        result = self.run_pre_commit()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Advisory growth", result.stdout)

    def test_staged_whitespace_fails_even_when_worktree_is_repaired(self) -> None:
        self.write("app.py", "value = 2  \n")
        self.stage("app.py")
        self.write("app.py", "value = 2\n")

        result = self.run_pre_commit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("git diff --cached --check", result.stderr)

    def test_added_invalid_python_is_checked(self) -> None:
        self.write("added.py", "if True print('invalid')\n")
        self.stage("added.py")

        result = self.run_pre_commit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("added.py", result.stderr)

    def test_deleted_python_is_not_read_from_the_worktree(self) -> None:
        self.git("rm", "-q", "removed.py")

        result = self.run_pre_commit()

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_renamed_python_uses_the_new_index_path(self) -> None:
        self.git("mv", "rename.py", "renamed.py")
        self.write("renamed.py", "value = (\n" + "# retained line\n" * 12)
        self.stage("renamed.py")
        status = self.git("diff", "--cached", "--name-status")
        self.assertIn("R", status.stdout)

        result = self.run_pre_commit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("renamed.py", result.stderr)

    def test_newline_in_python_filename_is_not_split_as_multiple_paths(self) -> None:
        odd_name = "odd\nname.py"
        self.write(odd_name, "if True print('invalid')\n")
        self.stage(odd_name)

        result = self.run_pre_commit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid Python syntax", result.stderr)
        self.assertIn("odd\\nname.py", result.stderr)

    def test_python_symlink_is_not_followed_or_parsed_as_source(self) -> None:
        external = self.root / "outside.py"
        external.write_text("def broken(:\n", encoding="utf-8")
        link = self.repo / "linked.py"
        link.symlink_to("../outside.py")
        self.stage("linked.py")

        result = self.run_pre_commit()

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_staged_program_is_parsed_but_never_executed(self) -> None:
        marker = self.root / "staged-program-ran"
        self.write(
            "effect.py",
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('unexpected')\n",
        )
        self.stage("effect.py")

        result = self.run_pre_commit()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(marker.exists())

    def test_large_successful_growth_report_remains_advisory(self) -> None:
        report = dict(COUNTER_REPORT)
        report["totals"] = {"added_nonblank_lines": 10**12, "removed_nonblank_lines": 0}
        self.install_counter_double(report)

        result = self.run_pre_commit()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("10", result.stdout)
        self.assertIn("no blocking threshold", result.stdout)

    def test_growth_report_accepts_the_staged_basis_object(self) -> None:
        self.install_counter_double(COUNTER_REPORT)

        result = self.run_pre_commit()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Advisory growth report", result.stdout)

    def test_growth_report_rejects_a_non_staged_basis(self) -> None:
        report = dict(COUNTER_REPORT)
        report["basis"] = {
            "type": "commits",
            "base_revision": "0" * 40,
            "head_revision": "1" * 40,
        }
        self.install_counter_double(report)

        result = self.run_pre_commit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("basis must describe the staged index", result.stderr)

    def test_growth_command_failure_is_explicit(self) -> None:
        self.install_counter_double(COUNTER_REPORT, exit_status=7)

        result = self.run_pre_commit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("size_report.py", result.stderr)
        self.assertIn("exit status 7", result.stderr)

    def test_missing_growth_tool_is_explicit(self) -> None:
        (self.repo / "scripts/size_report.py").unlink()

        result = self.run_pre_commit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required growth reporter", result.stderr)

    def test_invalid_growth_schema_is_explicit(self) -> None:
        self.install_counter_double({"schema_version": 2})

        result = self.run_pre_commit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("growth report schema", result.stderr)

    def test_pre_commit_reports_missing_python3(self) -> None:
        no_python = self.root / "no-python-bin"
        no_python.mkdir()
        git_path = shutil.which("git")
        self.assertIsNotNone(git_path)
        (no_python / "git").symlink_to(git_path)
        env = dict(os.environ)
        env["PATH"] = str(no_python)

        result = self.run_hook("pre-commit", env=env)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required tool python3 was not found", result.stderr)

    def test_commit_subjects_allow_scoped_types_and_breaking_marker(self) -> None:
        valid_subjects = [
            "feat(engine): add a staged check",
            "fix(loader): reject invalid input",
            "perf(cache): improve lookup",
            "test(weights): add a fixture",
            "docs(workflow): explain opt-in hooks",
            "style(format): keep output stable",
            "chore(tooling): refresh scripts",
            "refactor(core)!: change the contract",
        ]
        for subject in valid_subjects:
            with self.subTest(subject=subject):
                message = self.write("message.txt", subject + "\n\nBody text.\n")
                result = self.run_hook("commit-msg", str(message))
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_commit_subjects_require_an_allowed_type_scope_and_subject(self) -> None:
        invalid_subjects = [
            "other(engine): unsupported type",
            "feat: missing scope",
            "feat(): empty scope",
            "feat(   ): blank scope",
            "docs(workflow):   ",
            "feat(engine)!:",
        ]
        for subject in invalid_subjects:
            with self.subTest(subject=subject):
                message = self.write("message.txt", subject + "\n")
                result = self.run_hook("commit-msg", str(message))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Conventional Commit", result.stderr)

    def test_commit_msg_accepts_a_plain_message(self) -> None:
        result = self.check_message("feat(engine): add a check\n\nBody text.\n")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Conventional Commit subject passed.", result.stdout)

    def test_commit_msg_rejects_every_attribution_pattern(self) -> None:
        cases = [
            "Co-Authored-By: Bot <bot@example.invalid>",
            "claude-session: 0123456789abcdef",
            "Worker: mimo-v2.6",
            "Generated-By: pi",
            "Body text generated with pi",
            "Body text made via pi",
            "Assistant: Claude",
            "Company: Anthropic",
            "Email: noreply@anthropic.com",
        ]
        for line in cases:
            with self.subTest(line=line):
                result = self.check_message("feat(engine): add a check\n\n" + line + "\n")

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("line 3", result.stderr)
                self.assertIn("no attribution lines", result.stderr)
                self.assertIn("user rule", result.stderr)

    def test_commit_msg_rejects_an_attribution_trailer_after_a_blank_line(self) -> None:
        result = self.check_message(
            "feat(engine): add a check\n"
            "\n"
            "Body text.\n"
            "\n"
            "Co-authored-by: Bot <bot@example.invalid>\n"
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("line 5", result.stderr)
        self.assertIn("no attribution lines", result.stderr)
        self.assertIn("user rule", result.stderr)

    def test_commit_msg_accepts_worker_and_via_in_prose(self) -> None:
        result = self.check_message(
            "feat(engine): note the worker thread\n"
            "\n"
            "the worker thread drains the queue, and the change lands\n"
            "merged via the gateway after review.\n"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Conventional Commit subject passed.", result.stdout)

    def test_commit_msg_ignores_hash_comment_lines(self) -> None:
        result = self.check_message(
            "feat(engine): add a check\n"
            "# Please enter the commit message for your changes. Lines starting\n"
            "# Co-Authored-By: Bot <bot@example.invalid>\n"
            "# Claude-Session: 0123456789abcdef\n"
            "# Generated with pi\n"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Conventional Commit subject passed.", result.stdout)

    def test_pre_commit_reports_missing_git(self) -> None:
        no_git = self.root / "no-git-bin"
        no_git.mkdir()
        python_path = Path(sys.executable).resolve()
        (no_git / "python3").symlink_to(python_path)
        env = dict(os.environ)
        env["PATH"] = str(no_git)

        result = self.run_hook("pre-commit", env=env)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required tool git was not found", result.stderr)

    def test_commit_msg_hook_reports_missing_python3(self) -> None:
        no_python = self.root / "no-python-bin"
        no_python.mkdir()
        env = dict(os.environ)
        env["PATH"] = str(no_python)
        message = self.write("message.txt", "feat(engine): a subject\n")

        result = self.run_hook("commit-msg", str(message), env=env)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required tool python3 was not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
