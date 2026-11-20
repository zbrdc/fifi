import fcntl
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

import p2p_test_support  # noqa: F401 -- make repository packages visible under python -I

from scripts.p2p_concurrent_guard.host_attestation import (
    _seal_pinned,
    process_identity,
    process_start_time,
    sealed_companion,
    sealed_gate,
)


class AttestationTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, "memfd_create"), "Linux memfd required")
    def test_pinned_companion_is_immutable_even_if_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "companion.py"
            source.write_bytes(b"print('old')\n")
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            fd = _seal_pinned(source, expected)
            try:
                source.write_bytes(b"print('new')\n")
                self.assertEqual(os.read(fd, 100), b"print('old')\n")
                with self.assertRaises(OSError):
                    os.write(fd, b"x")
                with self.assertRaises(OSError):
                    os.ftruncate(fd, 0)
                flags = fcntl.fcntl(fd, fcntl.F_GET_SEALS)
                self.assertTrue(flags & fcntl.F_SEAL_WRITE)
            finally:
                os.close(fd)

    def test_companion_requires_exact_hash_and_does_not_follow_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source"
            path.write_bytes(b"not the official companion")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                sealed_companion(path)
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                _seal_pinned(path, "0" * 64)
            link = path.parent / "link"
            link.symlink_to(path)
            with self.assertRaises(OSError):
                _seal_pinned(link, hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(TypeError):
                sealed_companion(path, "0" * 64)

    def test_proc_stat_parser_accepts_parentheses_but_rejects_wrong_pid(self):
        line = "42 (test ) process) " + " ".join(["S"] + ["0"] * 18 + ["123"])
        self.assertEqual(process_start_time(line, 42), 123)
        with self.assertRaises(ValueError):
            process_start_time(line, 43)
        with self.assertRaises(ValueError):
            process_start_time("42 (x) S", 42)

    def test_process_identity_detects_restart_config_and_binary_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / "42"
            proc.mkdir()
            exe = root / "program"
            exe.write_bytes(b"binary-1")
            (proc / "exe").symlink_to(exe)
            for name in ("cmdline", "environ", "cgroup"):
                (proc / name).write_bytes(name.encode())
            stat_file = proc / "stat"

            def set_start(start):
                stat_file.write_text(
                    "42 (program) " + " ".join(["S"] + ["0"] * 18 + [str(start)])
                )

            set_start(10)
            config = root / "config"
            config.write_bytes(b"v1")
            first = process_identity(42, [config], proc_root=root)
            self.assertEqual(first, process_identity(42, [config], proc_root=root))
            config.write_bytes(b"v2")
            self.assertNotEqual(first, process_identity(42, [config], proc_root=root))
            config.write_bytes(b"v1")
            set_start(11)
            self.assertNotEqual(first, process_identity(42, [config], proc_root=root))
            set_start(10)
            exe.write_bytes(b"binary-2")
            self.assertNotEqual(first, process_identity(42, [config], proc_root=root))
            with self.assertRaises(ValueError):
                process_identity(42, [], proc_root=root)
            with self.assertRaises(ValueError):
                process_identity(42, [config, config], proc_root=root)

    @unittest.skipUnless(
        os.environ.get("FIFI_REAL_COMPANION_PIN_TEST") == "1",
        "explicit source-only test",
    )
    def test_exact_companion_digest_pins_and_seals(self):
        path = (
            "/home/ben/fifi-scratch/07dda740c23e/2080-concurrent-companion/companion.py"
        )
        fd = sealed_companion(path)
        try:
            with self.assertRaises(OSError):
                os.write(fd, b"no mutation")
        finally:
            os.close(fd)

    @unittest.skipUnless(hasattr(os, "memfd_create"), "Linux memfd required")
    def test_tracked_gate_bytes_match_pin_and_are_sealed(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "p2p_concurrent_guard/gated_companion.py"
        )
        fd = sealed_gate(path)
        try:
            with self.assertRaises(OSError):
                os.write(fd, b"no mutation")
        finally:
            os.close(fd)


if __name__ == "__main__":
    unittest.main()
