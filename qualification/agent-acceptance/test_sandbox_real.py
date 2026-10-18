"""Optional check of sandbox.py against the real docker and the pinned image. It runs only when
SANDBOX_REAL_IMAGE names a digest-pinned image that is present locally; it never pulls. The
programs below are written by this test, not by a model. Run, from this directory:
SANDBOX_REAL_IMAGE=python@sha256:<digest> python3 -I -m unittest discover -s . -p 'test_sandbox_real.py'
"""

import os
import pathlib
import subprocess
import tempfile
import time
import unittest

import sandbox

IMAGE = os.environ.get("SANDBOX_REAL_IMAGE", "")


@unittest.skipUnless(IMAGE, "SANDBOX_REAL_IMAGE is not set")
class RealContainer(unittest.TestCase):
    def run_program(self, text, seconds=30):
        with tempfile.TemporaryDirectory() as tmp:
            script = pathlib.Path(tmp) / "hello.py"
            script.write_text(text)
            box = sandbox.Sandbox(IMAGE, scratch=pathlib.Path(tmp) / "stage", seconds=seconds)
            box.preflight()
            return box.run_python(script)

    def test_the_default_host_is_the_rootless_daemon_and_the_record_names_its_versions(self):
        box = sandbox.Sandbox(IMAGE)
        box.preflight()
        self.assertEqual(box.host, f"unix:///run/user/{os.getuid()}/docker.sock")
        text = box.record()
        self.assertIn("name=rootless", text)
        self.assertRegex(text, r"component runc \S+")
        self.assertRegex(text, r"component containerd \S+")

    def test_the_root_daemon_is_refused_even_though_the_user_may_use_it(self):
        root_socket = "unix:///var/run/docker.sock"
        if not pathlib.Path("/var/run/docker.sock").exists():
            self.skipTest("no root daemon socket on this host")
        with self.assertRaises(sandbox.DaemonError):
            sandbox.Sandbox(IMAGE, host=root_socket).preflight()

    def test_a_program_prints_through_the_real_container(self):
        self.assertEqual(self.run_program("print(2 + 3)\n"), (0, "5\n"))

    def test_the_container_has_no_network(self):
        code, text = self.run_program(
            "import socket\ns = socket.socket()\ns.settimeout(3)\ns.connect(('1.1.1.1', 53))\n")
        self.assertNotEqual(code, 0)
        self.assertIn("Error", text)

    def test_the_root_and_the_script_directory_are_read_only_but_tmp_is_not(self):
        code, text = self.run_program(
            "import os\n"
            "open('/tmp/x', 'w').write('1')\n"
            "for p in ('/etc/x', '/work/x'):\n"
            "    try:\n        open(p, 'w')\n        print('wrote', p)\n"
            "    except OSError as e:\n        print('refused', p)\n"
            "print(os.getuid())\n")
        self.assertEqual((code, text), (0, "refused /etc/x\nrefused /work/x\n65534\n"))

    def test_the_key_is_not_in_the_container_environment(self):
        os.environ["ENGINESRV_API_KEY"] = "must-not-appear"
        try:
            code, text = self.run_program("import os\nprint('ENGINESRV_API_KEY' in os.environ, 'must-not-appear' in ''.join(os.environ.values()))\n")
        finally:
            del os.environ["ENGINESRV_API_KEY"]
        self.assertEqual((code, text), (0, "False False\n"))

    def test_a_print_loop_is_cut_at_the_cap_and_its_container_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = pathlib.Path(tmp) / "hello.py"
            script.write_text("while True:\n    print('x' * 100)\n")
            box = sandbox.Sandbox(IMAGE, scratch=pathlib.Path(tmp) / "stage", output_cap=8192)
            box.preflight()
            started = time.monotonic()
            code, text = box.run_python(script)
            elapsed = time.monotonic() - started
        self.assertIsNone(code)
        self.assertIn("more than 8192 bytes", text)
        self.assertLess(len(text), 8192 + 200)
        self.assertLess(elapsed, 20)  # far below the 30 s timeout: the cap ended it
        left = subprocess.run(["docker", "ps", "-aq", "--filter", "name=acc-sandbox-"],
                              capture_output=True, text=True).stdout.strip()
        self.assertEqual(left, "")

    def test_an_endless_loop_is_killed_at_the_timeout(self):
        code, text = self.run_program("while True:\n    pass\n", seconds=2)
        self.assertIsNone(code)
        self.assertIn("longer than 2 seconds", text)


if __name__ == "__main__":
    unittest.main()
