"""Self-test of the ACC-B export extractor (design 4.6; checklist E6). Run, from this directory:
python3 -I -m unittest test_accb_export
The extractor reads the last ACCB-EXPORT marker of a container stream, decodes strict base64,
and unpacks the tar into an empty directory. Every rule of design 4.6 has a case here: hostile
members (dot-dot, absolute, links, devices, FIFOs, setuid, control characters, long names,
too many parts, over-limit sizes, case and NFC duplicates, more than 1,000 entries), a real
`tar -C <dir> -cf - .` archive, a truncated archive, bad base64, a forged earlier marker line,
and the repr escaping of design 4.5 in the manifest.
"""

import base64
import io
import os
import pathlib
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
import unittest

import accb_export


def info(name, size=0, mode=0o644, kind=tarfile.REGTYPE, linkname=None, devmajor=0, devminor=0):
    member = tarfile.TarInfo(name)
    member.size = size
    member.mode = mode
    member.type = kind
    member.linkname = linkname or ""
    member.devmajor, member.devminor = devmajor, devminor
    return member


def archive_bytes(entries):
    """entries: [(TarInfo, bytes or None)] -> the bytes of a tar archive."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for member, data in entries:
            archive.addfile(member, io.BytesIO(data) if data is not None else None)
    return buffer.getvalue()


def stream_of(data, forged=None):
    """A stream whose last ACCB-EXPORT line holds data, after an optional forged marker."""
    lines = []
    if forged is not None:
        lines.append("ACCB-EXPORT " + forged)
    lines.append("ACCB-EXPORT " + base64.b64encode(data).decode())
    return "\n".join(lines) + "\n"


def rows(lines):
    """The manifest lines as rows: action, reason, size, sha256, name (repr)."""
    parsed = []
    for line in lines:
        action, reason, size, digest, name = line.split("\t", 4)
        parsed.append((action, reason, size, digest, name))
    return parsed


def kept_names(lines):
    import ast
    return [ast.literal_eval(name) for action, _, _, _, name in rows(lines) if action == "kept"]


def run_extract(tmp, stream):
    dest = pathlib.Path(tmp) / "work"
    lines = accb_export.extract_export(stream, dest)
    return dest, lines


class Payload(unittest.TestCase):
    def test_only_the_last_marker_line_is_used(self):
        real = archive_bytes([(info("keep", 2), b"ok")])
        forged = archive_bytes([(info("drop", 2), b"no")])
        with tempfile.TemporaryDirectory() as tmp:
            dest, lines = run_extract(tmp, stream_of(real, forged=base64.b64encode(forged).decode()))
            self.assertEqual(kept_names(lines), ["keep"])
            self.assertEqual((dest / "keep").read_bytes(), b"ok")

    def test_a_stream_without_a_marker_records_no_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest, lines = run_extract(tmp, '{"type":"message_end"}\n')
            self.assertEqual(rows(lines), [("skipped", "no export", "-", "-", "''")])
            self.assertEqual(list(dest.iterdir()), [])

    def test_a_bad_base64_line_records_bad_base64(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest, lines = run_extract(tmp, "ACCB-EXPORT not*base64!\n")
            self.assertEqual(rows(lines), [("skipped", "bad base64", "-", "-", "''")])
            self.assertEqual(list(dest.iterdir()), [])

    def test_an_empty_payload_records_an_unreadable_archive(self):
        # Round 5, blocking item 5: an empty payload must never crash the extractor (rule 3);
        # it records a manifest line naming the failure and refuses cleanly.
        with tempfile.TemporaryDirectory() as tmp:
            dest, lines = run_extract(tmp, stream_of(b""))
            self.assertEqual(rows(lines), [("skipped", "unreadable archive", "-", "-", "''")])
            self.assertEqual(list(dest.iterdir()), [])

    def test_a_non_tar_payload_records_an_unreadable_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest, lines = run_extract(tmp, stream_of(b"this is not a tar archive at all"))
            self.assertEqual(rows(lines), [("skipped", "unreadable archive", "-", "-", "''")])
            self.assertEqual(list(dest.iterdir()), [])


class RealArchive(unittest.TestCase):
    def test_a_real_tar_of_a_directory_is_fully_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = pathlib.Path(tmp) / "src"
            (source / "d").mkdir(parents=True)
            (source / "x").write_bytes(b"hello\n")
            (source / "d" / "y").write_bytes(b"deep\n")
            data = subprocess.run(["tar", "-C", str(source), "-cf", "-", "."],
                                  capture_output=True, check=True).stdout
            dest, lines = run_extract(tmp, stream_of(data))
            self.assertEqual((dest / "x").read_bytes(), b"hello\n")
            self.assertEqual((dest / "d" / "y").read_bytes(), b"deep\n")
            self.assertTrue((dest / "d").is_dir())
            self.assertEqual(stat.S_IMODE((dest / "x").stat().st_mode), 0o644)
            self.assertEqual(stat.S_IMODE((dest / "d").stat().st_mode), 0o755)
            self.assertEqual(sorted(kept_names(lines)), sorted(["./x", "./d", "./d/y"]))
            self.assertNotIn(".", kept_names(lines))  # the root entry is skipped silently

    def test_modes_owner_and_time_are_normalized(self):
        entries = [(info("f", 3, mode=0o600), b"abc"), (info("d", 0, mode=0o700, kind=tarfile.DIRTYPE), None)]
        with tempfile.TemporaryDirectory() as tmp:
            before = time.time()
            dest, lines = run_extract(tmp, stream_of(archive_bytes(entries)))
            self.assertEqual(stat.S_IMODE((dest / "f").stat().st_mode), 0o644)
            self.assertEqual(stat.S_IMODE((dest / "d").stat().st_mode), 0o755)
            self.assertEqual((dest / "f").stat().st_uid, os.getuid())
            self.assertGreaterEqual((dest / "f").stat().st_mtime, int(before))

    def test_a_setuid_member_keeps_its_data_but_not_its_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest, lines = run_extract(
                tmp, stream_of(archive_bytes([(info("s", 2, mode=0o4755), b"hi")])))
            self.assertEqual((dest / "s").read_bytes(), b"hi")
            self.assertEqual(stat.S_IMODE((dest / "s").stat().st_mode), 0o644)
            self.assertEqual(kept_names(lines), ["s"])


class HostileMembers(unittest.TestCase):
    def hostile(self, entries, kept=()):
        with tempfile.TemporaryDirectory() as tmp:
            dest, lines = run_extract(tmp, stream_of(archive_bytes(entries)))
            self.assertEqual(kept_names(lines), list(kept))
            files = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
            clean = [accb_export.normalized(n)[0] for n in kept]
            self.assertEqual(files, sorted(n for n in clean if (dest / n).is_file()))
            for name in clean:
                self.assertTrue((dest / name).exists(), name)
            reasons = {name: reason for _, reason, _, _, name in rows(lines) if _ == "skipped"}
            self.assertTrue(all(reasons.values()))
            return lines

    def test_a_dot_dot_name_is_skipped(self):
        self.hostile([(info("../x"), None)])

    def test_an_absolute_name_is_skipped(self):
        self.hostile([(info("/etc/passwd"), None)])

    def test_an_empty_part_and_a_backslash_are_skipped(self):
        self.hostile([(info("a//b"), None), (info("a\\b"), None)])

    def test_a_dot_part_is_skipped(self):
        # Round 5, blocking item 5: a `.` part that remains after the one allowed leading `./`
        # strip is rejected: `a/./b`, `d/.`, and `././x`. (A single leading `./` is stripped by
        # rule 6, so the names of a real `tar -C <dir> -cf - .` archive stay legal.)
        self.hostile([(info("a/./b"), None), (info("d/."), None), (info("././x"), None)])

    def test_the_reviewer_probe_is_rejected_after_the_real_entry(self):
        # The reviewer probe: a tar with hello.py then ././hello.py. The second name strips one
        # leading `./` and still holds a `.` part, so only the first entry is kept.
        with tempfile.TemporaryDirectory() as tmp:
            dest, lines = run_extract(
                tmp, stream_of(archive_bytes([(info("hello.py", 6), b"print1"),
                                              (info("././hello.py", 6), b"print2")])))
            self.assertEqual(kept_names(lines), ["hello.py"])
            self.assertEqual((dest / "hello.py").read_bytes(), b"print1")
            reasons = [reason for _, reason, _, _, _ in rows(lines) if reason]
            self.assertIn("dot part", " ".join(reasons))

    def test_a_symlink_member_is_skipped(self):
        self.hostile([(info("sym", 0, kind=tarfile.SYMTYPE, linkname="/etc"), None)])

    def test_a_hard_link_member_is_skipped(self):
        self.hostile([(info("hard", 0, kind=tarfile.LNKTYPE, linkname="f"), None)])

    def test_a_device_member_is_skipped(self):
        self.hostile([(info("dev", 0, kind=tarfile.CHRTYPE, devmajor=1, devminor=3), None)])

    def test_a_fifo_member_is_skipped(self):
        self.hostile([(info("fifo", 0, kind=tarfile.FIFOTYPE), None)])

    def test_a_name_with_a_control_character_is_skipped(self):
        self.hostile([(info("a\nb"), None), (info("a\x7fb"), None)])

    def test_a_name_over_200_characters_is_skipped(self):
        self.hostile([(info("n" * 201), None)])

    def test_a_name_with_more_than_eight_parts_is_skipped(self):
        self.hostile([(info("a/b/c/d/e/f/g/h", 0, kind=tarfile.DIRTYPE), None)],
                     kept=["a/b/c/d/e/f/g/h"])
        self.hostile([(info("a/b/c/d/e/f/g/h/i"), None)])

    def test_a_case_duplicate_is_skipped_and_the_first_kept(self):
        self.hostile([(info("File", 1), b"1"), (info("file", 1), b"2")], kept=["File"])

    def test_an_nfc_duplicate_is_skipped_and_the_first_kept(self):
        self.hostile([(info("caf\u00e9", 1), b"1"), (info("cafe\u0301", 1), b"2")], kept=["caf\u00e9"])

    def test_a_file_over_4MiB_is_skipped(self):
        self.hostile([(info("big", 4 * 1024 * 1024 + 1), b"x" * (4 * 1024 * 1024 + 1)),
                      (info("small", 1), b"y")], kept=["small"])

    def test_a_member_declaring_over_4MiB_in_a_sparse_header_is_skipped(self):
        if not shutil.which("tar"):
            self.skipTest("no tar for the sparse archive")
        with tempfile.TemporaryDirectory() as tmp:
            sparse = pathlib.Path(tmp) / "sparse"
            with open(sparse, "wb") as handle:
                handle.truncate(6 * 1024 * 1024)
                handle.seek(6 * 1024 * 1024 - 1)
                handle.write(b"x")
            data = subprocess.run(["tar", "--sparse", "-C", tmp, "-cf", "-", "sparse"],
                                  capture_output=True, check=True).stdout
            dest, lines = run_extract(tmp, stream_of(data))
            self.assertEqual(kept_names(lines), [])
            self.assertFalse((dest / "sparse").exists())
            self.assertIn("4 MiB", " ".join(reason for _, reason, _, _, _ in rows(lines)))

    def test_the_total_of_16MiB_stops_later_members(self):
        # Five 3 MiB files reach 15 MiB; the sixth would cross 16 MiB in total and is skipped
        # before any write. (A single larger file is stopped by the 4 MiB per-file rule instead.)
        entries = [(info("a%d" % i, 3 * 1024 * 1024), bytes([48 + i]) * (3 * 1024 * 1024))
                   for i in range(5)]
        entries.append((info("over", 2 * 1024 * 1024), b"z" * (2 * 1024 * 1024)))
        with tempfile.TemporaryDirectory() as tmp:
            dest, lines = run_extract(tmp, stream_of(archive_bytes(entries)))
            self.assertEqual(kept_names(lines), ["a%d" % i for i in range(5)])
            self.assertFalse((dest / "over").exists())
            self.assertIn("total over 16 MiB", " ".join(r for _, r, _, _, _ in rows(lines)))

    def test_more_than_1000_entries_stops_at_the_1000th(self):
        entries = [(info("f%04d" % i), b"x") for i in range(1001)]
        with tempfile.TemporaryDirectory() as tmp:
            dest, lines = run_extract(tmp, stream_of(archive_bytes(entries)))
            self.assertEqual(len(kept_names(lines)), 1000)
            self.assertFalse((dest / "f1000").exists())
            # Round 5, blocking item 5: the truncation itself is recorded as a manifest line.
            self.assertIn("entry list truncated at 1000",
                          " ".join(reason for _, reason, _, _, _ in rows(lines)))


class Truncated(unittest.TestCase):
    def test_a_truncated_archive_records_partial_and_keeps_complete_entries(self):
        whole = archive_bytes([(info("first", 512), b"a" * 512), (info("second", 512), b"b" * 512)])
        cut = whole[:1636]  # 100 bytes into the data of the second member (tar pads the rest)
        with tempfile.TemporaryDirectory() as tmp:
            dest, lines = run_extract(tmp, stream_of(cut))
            self.assertEqual((dest / "first").read_bytes(), b"a" * 512)
            self.assertIn("partial export", " ".join(reason for _, reason, _, _, _ in rows(lines)))

    def test_a_truncated_header_still_keeps_the_earlier_entries(self):
        whole = archive_bytes([(info("one", 0), None), (info("two", 0), None)])
        cut = whole[:512 + 100]  # the header of the second member is incomplete
        with tempfile.TemporaryDirectory() as tmp:
            dest, lines = run_extract(tmp, stream_of(cut))
            self.assertTrue((dest / "one").exists())
            self.assertIn("partial export", " ".join(reason for _, reason, _, _, _ in rows(lines)))


class Manifest(unittest.TestCase):
    def test_manifest_lines_hold_action_reason_size_sha256_and_repr(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest, lines = run_extract(tmp, stream_of(archive_bytes([(info("f", 3), b"abc")])))
            action, reason, size, digest, name = rows(lines)[0]
            self.assertEqual((action, reason, size), ("kept", "ok", "3"))
            import hashlib
            self.assertEqual(digest, hashlib.sha256(b"abc").hexdigest())
            self.assertEqual(name, "'f'")

    def test_the_name_is_repr_escaped_and_cut_at_200_characters(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest, lines = run_extract(
                tmp, stream_of(archive_bytes([(info("bad\nname", 0, kind=tarfile.FIFOTYPE), None),
                                              (info("n" * 300), None)])))
            names = [name for _, _, _, _, name in rows(lines)]
            self.assertIn("'bad\\nname'", names)  # repr: a newline never reaches the manifest raw
            for line in lines:
                self.assertLessEqual(len(line), 250)
                self.assertNotIn("\n", line)


class LineSeparators(unittest.TestCase):
    def test_a_u2028_in_a_later_line_cannot_forge_a_marker(self):
        # Round 5, blocking item 6: lines must split on \n only. With str.splitlines, a U+2028
        # inside a JSON string splits that line and the fragment after it starts with the
        # marker, so a hostile stream could make a forged payload the LAST marker line.
        with tempfile.TemporaryDirectory() as tmp:
            source = pathlib.Path(tmp) / "src"
            source.mkdir()
            (source / "hello.py").write_text("print(2 + 3)\n")
            data = subprocess.run(["tar", "-C", str(source), "-cf", "-", "."],
                                  capture_output=True, check=True).stdout
            late = '{"note":"after \u2028ACCB-EXPORT not*base64!"}'
            stream = stream_of(data) + late + "\n"
            dest, lines = run_extract(tmp, stream)
            self.assertEqual(kept_names(lines), ["./hello.py"])
            self.assertEqual((dest / "hello.py").read_text(), "print(2 + 3)\n")
            self.assertNotIn("bad base64", " ".join(reason for _, reason, _, _, _ in rows(lines)))


class Cli(unittest.TestCase):
    def test_the_cli_extracts_a_stream_file_and_writes_the_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = pathlib.Path(tmp) / "src"
            source.mkdir()
            (source / "hello.py").write_text("print(2 + 3)\n")
            data = subprocess.run(["tar", "-C", str(source), "-cf", "-", "."],
                                  capture_output=True, check=True).stdout
            stream = pathlib.Path(tmp) / "pi.jsonl"
            stream.write_text('{"type":"message_end"}\n' + stream_of(data))
            dest = pathlib.Path(tmp) / "out" / "work"
            manifest = pathlib.Path(tmp) / "out" / "export-manifest.txt"
            status = accb_export.main(["--stream", str(stream), "--dest", str(dest),
                                       "--manifest", str(manifest)])
            self.assertEqual(status, 0)
            self.assertEqual((dest / "hello.py").read_text(), "print(2 + 3)\n")
            self.assertTrue(manifest.is_file())
            self.assertEqual(stat.S_IMODE(dest.stat().st_mode), 0o700)

    def test_the_cli_refuses_a_destination_that_is_not_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = pathlib.Path(tmp) / "work"
            dest.mkdir()
            (dest / "x").write_text("x")
            status = accb_export.main(["--stream", "/dev/null", "--dest", str(dest),
                                       "--manifest", str(pathlib.Path(tmp) / "m")])
            self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
