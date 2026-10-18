"""Behavior tests for scripts/check_doc_links.py.

Every test builds its own isolated repository under the task scratch directory.
The repository carries ledger-style ignore rules and one document under test.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "scripts/check_doc_links.py"
GITIGNORE = """\
/WORK_LOG.md
/docs/
/specs/
/research/
/AGENTS.md
/CLAUDE.md
/CONVENTIONS.md
/PLAN.md
/scripts/size-report.md
.env
"""


class DocumentLinkCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path(os.environ.get("TMPDIR", "")).resolve()
        if not scratch.is_dir():
            self.fail("TMPDIR must name the task scratch directory")
        self.temporary = tempfile.TemporaryDirectory(
            prefix="doc-link-check-", dir=str(scratch)
        )
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()

        self.environment = dict(os.environ)
        self.environment["GIT_CONFIG_NOSYSTEM"] = "1"
        empty_global_config = self.root / "empty-global.gitconfig"
        empty_global_config.touch()
        self.environment["GIT_CONFIG_GLOBAL"] = str(empty_global_config)

        target = self.repo / SCRIPT
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / SCRIPT, target)
        self.write(".gitignore", GITIGNORE)
        self.git("init", "-q")
        self.git("config", "user.name", "Document link test")
        self.git("config", "user.email", "doc-links@example.invalid")
        self.git("add", "-A")

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

    def run_check(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, SCRIPT],
            cwd=self.repo,
            env=self.environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def stage(self, relative: str, content: str) -> None:
        self.write(relative, content)
        self.git("add", relative)

    def stage_forced(self, relative: str, content: str) -> None:
        """Stage a document that sits at an ignored path; the checker scans the index as-is."""
        self.write(relative, content)
        self.git("add", "-f", relative)

    def test_link_to_ignored_file_is_reported(self) -> None:
        self.stage("README.md", "See [the plan](docs/plans/phase-0a.md) for the gates.\n")

        result = self.run_check()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("README.md:1", result.stderr)
        self.assertIn("docs/plans/phase-0a.md", result.stderr)
        self.assertIn("ignored path", result.stderr)

    def test_link_with_title_to_ignored_file_is_reported(self) -> None:
        self.stage(
            "README.md",
            'See [the plan](docs/plans/phase-0a.md "the gates") for the gates.\n',
        )

        result = self.run_check()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("README.md:1", result.stderr)
        self.assertIn("docs/plans/phase-0a.md", result.stderr)

    def test_link_to_tracked_document_passes(self) -> None:
        self.stage("engine/testdata/notes.md", "# Notes\n")
        self.stage("README.md", "Read [the notes](engine/testdata/notes.md).\n")

        result = self.run_check()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Document link check passed", result.stdout)

    def test_plain_text_mention_of_ignored_path_passes(self) -> None:
        self.stage("README.md", "The plan file `docs/plans/phase-0a.md` is a plain citation.\n")

        result = self.run_check()

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_absolute_url_link_passes(self) -> None:
        self.stage(
            "README.md",
            "See [the archived plan](https://example.invalid/docs/plans/phase-0a.md).\n",
        )

        result = self.run_check()

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fragment_only_link_passes(self) -> None:
        self.stage("README.md", "Jump [to the top](#top) of this document.\n")

        result = self.run_check()

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_link_to_ignored_directory_without_trailing_slash_is_reported(self) -> None:
        self.stage("README.md", "See [the plans](docs/plans) for the gates.\n")

        result = self.run_check()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("README.md:1", result.stderr)
        self.assertIn("docs/plans", result.stderr)
        self.assertIn("ignored path", result.stderr)

    def test_link_with_single_quoted_title_to_ignored_file_is_reported(self) -> None:
        self.stage(
            "README.md",
            "See [the plan](docs/plans/phase-0a.md 'the gates') for the gates.\n",
        )

        result = self.run_check()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("README.md:1", result.stderr)
        self.assertIn("docs/plans/phase-0a.md", result.stderr)

    def test_link_with_query_suffix_to_ignored_file_is_reported(self) -> None:
        self.stage("README.md", "See [the log](WORK_LOG.md?x=1) for the gates.\n")

        result = self.run_check()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("README.md:1", result.stderr)
        self.assertIn("WORK_LOG.md", result.stderr)

    def test_yaml_plain_path_value_to_ignored_file_is_reported(self) -> None:
        self.stage("report.yaml", "evidence: docs/plans/p.md\n")

        result = self.run_check()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("report.yaml:1", result.stderr)
        self.assertIn("docs/plans/p.md", result.stderr)

    def test_yaml_plain_value_outside_ignored_paths_passes(self) -> None:
        self.stage("qualification/report.yaml", "name: zero\nphase: 0a\n")

        result = self.run_check()

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_html_href_link_to_ignored_file_is_reported(self) -> None:
        self.stage(
            "README.md",
            'See <a href="docs/plans/phase-0a.md">the plan</a> for the gates.\n',
        )

        result = self.run_check()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("README.md:1", result.stderr)
        self.assertIn("docs/plans/phase-0a.md", result.stderr)

    def test_angle_autolink_to_ignored_file_is_reported(self) -> None:
        self.stage("README.md", "See <docs/plans/phase-0a.md> for the gates.\n")

        result = self.run_check()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("README.md:1", result.stderr)
        self.assertIn("docs/plans/phase-0a.md", result.stderr)

    def test_link_inside_fenced_code_block_is_reported(self) -> None:
        # The checker has never exempted fenced code blocks; pin that behavior.
        self.stage(
            "README.md",
            "Example:\n\n```\n[the plan](docs/plans/phase-0a.md)\n```\n",
        )

        result = self.run_check()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("README.md:4", result.stderr)
        self.assertIn("docs/plans/phase-0a.md", result.stderr)

    def test_reference_style_link_to_ignored_file_is_reported(self) -> None:
        self.stage(
            "README.md",
            "See [the plan] for the gates.\n\n[the plan]: docs/plans/phase-0a.md\n",
        )

        result = self.run_check()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("README.md:3", result.stderr)
        self.assertIn("docs/plans/phase-0a.md", result.stderr)

    def test_relative_link_from_subdirectory_is_reported(self) -> None:
        self.stage_forced(
            "docs/design/tests.md",
            "The [record](../reviews/2026-09-20/portable-loader.md) holds the result.\n",
        )

        result = self.run_check()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("docs/design/tests.md:1", result.stderr)
        self.assertIn("docs/reviews/2026-09-20/portable-loader.md", result.stderr)

    def test_yaml_document_link_to_ignored_file_is_reported(self) -> None:
        self.stage_forced(
            "specs/release-plan.yaml",
            "phases:\n  - name: zero\n    evidence: [record](../docs/reviews/2026-09-20/r.md)\n",
        )

        result = self.run_check()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("specs/release-plan.yaml:3", result.stderr)
        self.assertIn("docs/reviews/2026-09-20/r.md", result.stderr)

    def test_document_symlink_is_not_followed(self) -> None:
        self.stage_forced("AGENTS.md", "# Rules\n")
        (self.repo / "LINK.md").symlink_to("AGENTS.md")
        self.git("add", "LINK.md")

        result = self.run_check()

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_link_outside_the_repository_passes(self) -> None:
        self.stage_forced("docs/deep/nested/notes.md", "See [the note](../../../../outside.md).\n")

        result = self.run_check()

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_link_to_each_ignored_product_document_is_reported(self) -> None:
        targets = (
            "docs/design/kernels.md",
            "specs/release-plan.yaml",
            "research/prefetch/README.md",
            "PLAN.md",
            "AGENTS.md",
            "CLAUDE.md",
            "CONVENTIONS.md",
        )
        for target in targets:
            with self.subTest(target=target):
                self.stage("README.md", f"See [the document]({target}) for the details.\n")

                result = self.run_check()

                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("README.md:1", result.stderr)
                self.assertIn(target, result.stderr)
                self.assertIn("ignored path", result.stderr)

    def test_plain_text_citation_of_ignored_product_documents_passes(self) -> None:
        self.stage(
            "README.md",
            "The rules come from `docs/design/server.md` and decision 0047; "
            "the conventions of CONVENTIONS.md and the plan of PLAN.md stay plain words.\n",
        )

        result = self.run_check()

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_relative_link_from_qualification_readme_is_reported(self) -> None:
        self.stage(
            "qualification/os-reserve/README.md",
            "The budget rule is [scheduler.md](../../docs/design/scheduler.md).\n",
        )

        result = self.run_check()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("qualification/os-reserve/README.md:1", result.stderr)
        self.assertIn("docs/design/scheduler.md", result.stderr)


if __name__ == "__main__":
    unittest.main()
