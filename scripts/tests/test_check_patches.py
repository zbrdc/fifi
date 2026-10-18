"""Behavior tests for engine/vendor/xavier-qwen/check_patches.py on a synthetic vendor tree.

The fixtures under scripts/tests/fixtures/check_patches/ are fake vendor trees ("good", "import-only",
"extra-file"); each has vendor/ and a shim/ directory beside it. Every test works on a copy in a
temporary directory. The script is loaded by path; no real vendor file is read. Without GNU patch on PATH
(the Mac portable suite may lack it) the module prints `SKIPPED test_check_patches: GNU patch not found`
and skips; with patch present a wrong check fails the run.
Environment: CHECK_PATCHES_SCRIPT overrides the script path; FIXTURE_KEEP is a new directory that
also receives every tree the tests ran on.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURES = HERE / "fixtures" / "check_patches"
SCRIPT = pathlib.Path(os.environ.get(
    "CHECK_PATCHES_SCRIPT", ROOT / "engine" / "vendor" / "xavier-qwen" / "check_patches.py")).resolve()
KEEP = os.environ.get("FIXTURE_KEEP")


def setUpModule():
    if shutil.which("patch") is None:
        print("SKIPPED test_check_patches: GNU patch not found")
        raise unittest.SkipTest("GNU patch not found")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def build(dest, name="good"):
    """Copies fixtures/check_patches/<name> to dest; returns (vendor dir, provenance dict)."""
    shutil.copytree(FIXTURES / name, dest, symlinks=True)
    vendor = dest / "vendor"
    return vendor, json.loads((vendor / "provenance.json").read_text())


def save(vendor, provenance):
    (vendor / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def run(vendor, script=None, *args):
    command = [sys.executable, "-I", str(script or SCRIPT)] + ([str(vendor)] if vendor else []) + list(args)
    done = subprocess.run(command, capture_output=True, text=True)
    return done.returncode, done.stdout


def patch_entry(provenance, name):
    return next(entry for entry in provenance["patches"] if entry["id"] == name)


def add_undeclared_hunk(vendor):
    """Appends to FX-1.patch a hunk that edits src/inc/common.h, which FX-1's entry does not list."""
    current = (vendor / "src/inc/common.h").read_text()
    hunk = "".join(difflib.unified_diff(current.replace("COMMON 1", "COMMON 0").splitlines(True),
                                        current.splitlines(True), "a/src/inc/common.h", "b/src/inc/common.h"))
    with open(vendor / "patches/FX-1.patch", "a") as stream:
        stream.write(hunk)


def drop_license(vendor, provenance):
    (vendor / "LICENSE").unlink()
    mutate_json(vendor, provenance, lambda d: d["files"].pop("LICENSE"))


def rewrite_text(vendor, old, new):
    text = (vendor / "provenance.json").read_text()
    (vendor / "provenance.json").write_text(text.replace(old, new, 1))


def edit_file(vendor, provenance, path, text, rehash):
    (vendor / path).write_text(text)
    if rehash:
        key = "patched_sha256" if "patched_sha256" in provenance["files"][path] else "vendored_sha256"
        provenance["files"][path][key] = sha(text)
        save(vendor, provenance)


def mutate_json(vendor, provenance, change):
    change(provenance)
    save(vendor, provenance)


# name -> (mutation(vendor, provenance), text the FAIL line must contain)
COMMON = "src/inc/common.h"
MUTATIONS = {
    "extra file": (lambda v, p: (v / "src/fake/stray.cuh").write_text("x\n"), "extra src/fake/stray.cuh"),
    "missing file": (lambda v, p: (v / COMMON).unlink(), COMMON + ": listed file is missing"),
    "unpatched file edited": (lambda v, p: edit_file(v, p, COMMON, "#define COMMON 2\n", False),
                              COMMON + ": file on disk does not match vendored_sha256"),
    "LICENSE edited": (lambda v, p: edit_file(v, p, "LICENSE", "other\n", False), "LICENSE: file on disk"),
    "LICENSE absent from list and tree": (drop_license, "LICENSE is not listed in files"),
    "patched file edited": (lambda v, p: edit_file(v, p, "src/fake/util.cuh", "int y;\n", False),
                            "util.cuh: file on disk does not match patched_sha256"),
    "chain broken at the start": (lambda v, p: mutate_json(v, p, lambda d: patch_entry(d, "FX-1")["files"][
        "src/fake/util.cuh"].update(before_sha256="0" * 64)), "first before_sha256"),
    "chain link broken": (lambda v, p: mutate_json(v, p, lambda d: patch_entry(d, "FX-3")["files"][
        "src/fake/util.cuh"].update(before_sha256="0" * 64)), "patch chain is broken"),
    "patched_sha256 is not the last after": (lambda v, p: mutate_json(v, p, lambda d: patch_entry(d, "FX-3")[
        "files"]["src/fake/util.cuh"].update(after_sha256="0" * 64)), "patched_sha256 is not the last after_sha256"),
    "stored diff edits an undeclared file": (lambda v, p: add_undeclared_hunk(v), "does not declare"),
    "include resolves nowhere": (lambda v, p: edit_file(v, p, "src/main.cu", '#include "nowhere.h"\n', True),
                                 '#include "nowhere.h" is neither'),
    "include resolves outside the root": (lambda v, p: edit_file(
        v, p, "src/fake/core.cuh", '#include "../../../shim/fake_ggml.h"\n', True), "neither a vendored file"),
    "shim file missing": (lambda v, p: (v / "../shim/fake_ggml.h").unlink(), "shim fake_ggml.h"),
    "shim file edited": (lambda v, p: (v / "../shim/fake_ggml.h").write_text("// other\n"), "shim fake_ggml.h"),
    "null shim hash with the file present": (lambda v, p: mutate_json(v, p, lambda d: d["shims"][0].update(
        sha256=None)), "shim fake_ggml.h: file at"),
    "shim shadows a vendored file": (lambda v, p: mutate_json(v, p, lambda d: d["shims"].append(
        {"include": "common.h", "stands_in_for": "x/common.h", "path": "../shim/fake_ggml.h",
         "sha256": d["shims"][0]["sha256"]})), 'shim "common.h" shadows'),
    "two shims with one include name": (lambda v, p: mutate_json(v, p, lambda d: d["shims"].append(
        dict(d["shims"][0], stands_in_for="fake/include/other.h"))), "two shims have the same include name"),
    "shim stands in for a vendored file": (lambda v, p: mutate_json(v, p, lambda d: d["shims"][0].update(
        stands_in_for="fake/" + COMMON)), "is vendored, not shimmed"),
    "shim included by no file": (lambda v, p: mutate_json(v, p, lambda d: d["shims"].append(
        dict(d["shims"][0], include="unused.h", stands_in_for="fake/include/unused.h"))),
        "unused.h is included by no listed file"),
    "stored diff not referenced": (lambda v, p: (v / "patches/stray.patch").write_text("x\n"), "extra patches/stray.patch"),
    "stored diff damaged": (lambda v, p: (v / "patches/FX-1.patch").write_text(
        (v / "patches/FX-1.patch").read_text().replace("x + 1", "x + 7")), "FX-1: reverse apply failed"),
    "second copy of the script": (lambda v, p: shutil.copyfile(SCRIPT, v / "check_patches.py"),
                                  "extra check_patches.py"),
    "symbolic link": (lambda v, p: (v / "src/link.h").symlink_to("inc/common.h"), "symbolic link"),
    "patched_sha256 without a patch": (lambda v, p: mutate_json(v, p, lambda d: d["files"]["LICENSE"].update(
        patched_sha256="1" * 64)), "LICENSE: patched_sha256 but no patch lists the file"),
    "patch lists an unlisted file": (lambda v, p: mutate_json(v, p, lambda d: patch_entry(d, "FX-1")["files"].update(
        {"src/zzz.h": {"before_sha256": None, "after_sha256": "2" * 64}})), "which \"files\" does not list"),
    "listed path leaves the tree": (lambda v, p: mutate_json(v, p, lambda d: d["files"].update(
        {"../escape.h": {"vendored_sha256": "3" * 64}})), "leaves the vendor directory"),
    "listed path is absolute": (lambda v, p: mutate_json(v, p, lambda d: d["files"].update(
        {"/etc/escape.h": {"vendored_sha256": "3" * 64}})), "leaves the vendor directory"),
    "two listed paths differ only in case": (lambda v, p: mutate_json(v, p, lambda d: d["files"].update(
        {"license": {"vendored_sha256": "4" * 64}})), "differ only in case"),
    "repeated JSON key": (lambda v, p: rewrite_text(v, '"project": "fixture",', '"project": "fixture", "project": "b",'),
                          "repeated JSON key"),
    "malformed JSON": (lambda v, p: (v / "provenance.json").write_text("{"), "provenance.json is malformed"),
    "a files entry is not an object": (lambda v, p: mutate_json(v, p, lambda d: d["files"].update(
        {"src/main.cu": "x"})), "provenance.json is malformed"),
    "provenance.json missing": (lambda v, p: (v / "provenance.json").unlink(), "no provenance.json"),
}


GONE = "src/fake/gone.cuh"


def with_trim(vendor, provenance, change=None):
    """Adds a trim block (one removed file, one entry) to the good tree, then applies change(trim dict)."""
    trim = {"removed_files": {GONE: {"upstream_path": "fake/" + GONE, "upstream_sha256": sha("gone\n")}},
            "entries": [{"id": "FX-4", "description": "fixture", "files": [GONE]}]}
    provenance["import_file_count"] = sum(1 for e in provenance["files"].values() if "upstream_sha256" in e) + 1
    if change:
        change(trim, vendor, provenance)
    provenance["trim"] = trim
    save(vendor, provenance)


# name -> (mutation(vendor, provenance), text the FAIL line must contain); every row starts from a good tree with a trim block
TRIM_MUTATIONS = {
    "trim: corrupt upstream_sha256": (lambda t, v, p: t["removed_files"][GONE].update(upstream_sha256="XYZ"),
                                      "upstream_sha256 is not 64 lowercase hex"),
    "trim: upper-case upstream_sha256": (lambda t, v, p: t["removed_files"][GONE].update(upstream_sha256="A" * 64),
                                         "upstream_sha256 is not 64 lowercase hex"),
    "trim: upstream_path missing": (lambda t, v, p: t["removed_files"][GONE].pop("upstream_path"), "upstream_path is missing"),
    "trim: removed file still in the tree": (lambda t, v, p: (v / GONE).write_text("gone\n"), "is removed but still in the tree"),
    "trim: removed file also listed in files": (lambda t, v, p: p["files"].update(
        {GONE: {"upstream_path": "fake/other", "upstream_sha256": "5" * 64, "vendored_sha256": "5" * 64}}),
        "is removed and also listed in files"),
    "trim: removed file is the upstream of a listed file": (lambda t, v, p: p["files"]["src/fake/pre.cuh"].update(
        upstream_path="fake/" + GONE), "also the upstream of a listed file"),
    "trim: entry names a path that was dropped from removed_files": (lambda t, v, p: t["entries"][0]["files"].append(
        "src/fake/dropped.cuh"), "is not in removed_files"),
    "trim: removed file covered by no entry": (lambda t, v, p: t["removed_files"].update(
        {"src/fake/gone2.cuh": {"upstream_path": "fake/src/fake/gone2.cuh", "upstream_sha256": "6" * 64}}),
        "no entry lists src/fake/gone2.cuh"),
    "trim: no entries": (lambda t, v, p: t.pop("entries"), "entries is missing or empty"),
    "trim: removed path named by two entries": (lambda t, v, p: t["entries"].append({"id": "FX-5", "files": [GONE]}),
                                                "named by two entries"),
    "trim: removed_files is not an object": (lambda t, v, p: t.update(removed_files=[GONE]), "removed_files is not an object"),
    "trim: an entry lists no file": (lambda t, v, p: t["entries"].append({"id": "FX-5", "files": []}), "entry FX-5 lists no file"),
    "trim: upstream_path does not end with the removed path": (lambda t, v, p: t["removed_files"][GONE].update(
        upstream_path="fake/src/fake/other.cuh"), "does not end with the removed path"),
    "trim: import_file_count does not match": (lambda t, v, p: p.update(import_file_count=p["import_file_count"] + 1),
                                               "import_file_count is"),
    "trim: removed path leaves the tree": (lambda t, v, p: t["removed_files"].update(
        {"../escape.h": t["removed_files"][GONE]}) or t["entries"][0]["files"].append("../escape.h"), "leaves the vendor directory"),
}


class CheckPatches(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="cpfv0-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def keep(self, name, dest):
        if KEEP:
            shutil.copytree(dest, pathlib.Path(KEEP) / name.replace(" ", "-"), symlinks=True)

    def test_good_tree(self):
        vendor, _ = build(self.tmp / "t")
        code, out = run(vendor)
        self.keep("good", self.tmp / "t")
        self.assertEqual(code, 0, out)
        for line in ("ok hashes", "ok tree: 11 files", "ok includes: 4 quoted includes, 1 through shims",
                     "ok shims: 1 listed, 0 not written yet", "ok replay: FX-1, FX-2, FX-3"):
            self.assertIn(line, out)

    def test_good_tree_with_a_trim_block(self):
        vendor, provenance = build(self.tmp / "t")
        with_trim(vendor, provenance)
        code, out = run(vendor)
        self.assertEqual(code, 0, out)
        self.assertIn("ok trim: 1 removed files", out)

    def test_a_deleted_trim_block_fails_when_the_import_count_is_larger(self):
        vendor, provenance = build(self.tmp / "t")
        provenance["import_file_count"] = sum(1 for e in provenance["files"].values() if "upstream_sha256" in e) + 1
        save(vendor, provenance)
        code, out = run(vendor)
        self.assertEqual(code, 1, out)
        self.assertIn("import_file_count is", out)

    def test_a_foreign_trim_block_is_ignored(self):
        vendor, provenance = build(self.tmp / "t")
        provenance["trim"] = {"approval": "x", "kept": {"a": ["f"]}, "removed": {"a": ["g"]}}
        save(vendor, provenance)
        code, out = run(vendor)
        self.assertEqual(code, 0, out)
        self.assertNotIn("ok trim", out)

    def test_every_trim_mutation_fails_for_its_reason(self):
        for name, (change, reason) in TRIM_MUTATIONS.items():
            with self.subTest(name):
                vendor, provenance = build(self.tmp / name.replace(" ", "-").replace(":", ""))
                with_trim(vendor, provenance, change)
                code, out = run(vendor)
                self.assertEqual(code, 1, out)
                self.assertIn("FAIL", out)
                self.assertIn(reason, out)

    def test_committed_extra_file_fixture_fails(self):
        vendor, _ = build(self.tmp / "t", "extra-file")
        code, out = run(vendor)
        self.assertEqual(code, 1, out)
        self.assertIn("extra src/fake/stray.cuh", out)

    def test_import_only_tree_with_a_planned_shim(self):
        vendor, _ = build(self.tmp / "t", "import-only")
        code, out = run(vendor)
        self.keep("import-only", self.tmp / "t")
        self.assertEqual(code, 0, out)
        self.assertIn("ok shims: 1 listed, 1 not written yet", out)
        self.assertIn("ok replay: no stored diff", out)

    def test_script_in_its_own_vendor_directory_runs_without_an_argument(self):
        vendor, _ = build(self.tmp / "t")
        shutil.copyfile(SCRIPT, vendor / "check_patches.py")
        code, out = run(None, vendor / "check_patches.py")
        self.assertEqual(code, 0, out)
        self.assertIn("ok tree: 12 files", out)

    def test_two_arguments_print_the_usage_and_exit_2(self):
        code, out = run(None, None, "a", "b")
        self.assertEqual(code, 2, out)
        self.assertIn("Usage:", out)

    def test_documented_scan_limits_pass(self):
        vendor, provenance = build(self.tmp / "t")
        text = ('/* c */ #include "nowhere.h"\n#include MACRO\n#include <nowhere.h>\n'
                '#include \\\n"nowhere.h"\n')
        edit_file(vendor, provenance, "src/main.cu", text, True)
        code, out = run(vendor)
        self.assertEqual(code, 0, out)

    def test_every_mutation_fails_for_its_reason(self):
        for name, (mutate, reason) in MUTATIONS.items():
            with self.subTest(name):
                dest = self.tmp / name.replace(" ", "-")
                vendor, provenance = build(dest)
                mutate(vendor, provenance)
                code, out = run(vendor)
                self.keep(name, dest)
                self.assertEqual(code, 1, out)
                self.assertIn("FAIL", out)
                self.assertIn(reason, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
