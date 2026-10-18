"""Self-test of sandbox.py against fake_docker.py. No container runs. Run, from this directory:
python3 -I -m unittest discover -s . -p 'test_*.py'
"""

import json
import os
import pathlib
import stat
import tempfile
import time
import unittest

import sandbox

HERE = pathlib.Path(__file__).parent
FAKE_DOCKER = str(HERE / "fake_docker.py")
IMAGE = "fake@sha256:" + "0" * 64


def make(tmp, image=IMAGE, docker=FAKE_DOCKER, seconds=30):
    return sandbox.Sandbox(image, docker=docker, scratch=pathlib.Path(tmp) / "stage", seconds=seconds)


class Command(unittest.TestCase):
    def test_the_command_holds_every_isolation_option_and_the_pinned_image(self):
        argv = make("/x").command("/scratch/stage-1", "name-1")
        pairs = {"--network": "none", "--user": "65534:65534", "--memory": "256m",
                 "--memory-swap": "256m", "--cpus": "1", "--pids-limit": "64",
                 "--cap-drop": "ALL", "--security-opt": "no-new-privileges",
                 "-v": "/scratch/stage-1:/work:ro"}
        for flag, value in pairs.items():
            self.assertEqual(argv[argv.index(flag) + 1], value, flag)
        self.assertIn("--read-only", argv)
        self.assertTrue(argv[argv.index("--tmpfs") + 1].startswith("/tmp:"))
        self.assertEqual(argv[argv.index(IMAGE):], [IMAGE, "timeout", "-s", "KILL", "30",
                                                    "python", "-I", "/work/script.py"])
        self.assertEqual(argv[argv.index("--pull") + 1], "never")
        self.assertEqual(argv[argv.index("--log-driver") + 1], "none")
        self.assertNotIn("--privileged", argv)
        self.assertNotIn("-p", argv)

    def test_a_directory_name_that_could_add_a_mount_option_is_refused(self):
        for path in ("/a:b", "/a,b"):
            with self.assertRaises(sandbox.SandboxError):
                make("/x").command(path, "n")


class Preflight(unittest.TestCase):
    def test_a_present_pinned_image_passes(self):
        make("/x").preflight()

    def test_an_unset_unpinned_or_absent_image_stops_with_the_approval_message(self):
        for image in ("", "python:3.12-slim", "fake@sha256:abc", "absent@sha256:" + "1" * 64):
            with self.subTest(image=image):
                with self.assertRaises(sandbox.SandboxError) as caught:
                    make("/x", image=image).preflight()
                self.assertIn("sandbox image missing; needs the user's approval to pull",
                              str(caught.exception))

    def test_a_missing_docker_client_stops_the_run(self):
        with self.assertRaises(sandbox.SandboxError):
            make("/x", docker="/nonexistent/docker").preflight()

    def test_preflight_never_pulls(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "calls.log"
            os.environ["FAKE_DOCKER_LOG"] = str(log)
            try:
                with self.assertRaises(sandbox.SandboxError):
                    make(tmp, image="absent@sha256:" + "1" * 64).preflight()
            finally:
                del os.environ["FAKE_DOCKER_LOG"]
            verbs = [json.loads(line)[0] for line in log.read_text().splitlines()]
            self.assertEqual(verbs, ["info", "image"])


class Daemon(unittest.TestCase):
    """The sandbox uses the rootless daemon only (pinu also runs a root dockerd)."""

    def with_env(self, name, value, call):
        os.environ[name] = value
        try:
            return call()
        finally:
            del os.environ[name]

    def test_the_default_host_is_the_rootless_socket_of_this_user(self):
        self.assertEqual(sandbox.default_host(), f"unix:///run/user/{os.getuid()}/docker.sock")
        self.assertEqual(self.with_env("SANDBOX_DOCKER_HOST", "unix:///x.sock", sandbox.default_host),
                         "unix:///x.sock")

    def test_a_daemon_that_is_not_rootless_stops_the_run_with_status_69(self):
        for variable in ("FAKE_DOCKER_ROOTLESS", "FAKE_DOCKER_NODAEMON"):
            with self.subTest(variable):
                value = "0" if variable == "FAKE_DOCKER_ROOTLESS" else "1"
                with self.assertRaises(sandbox.DaemonError) as caught:
                    self.with_env(variable, value, make("/x").preflight)
                self.assertEqual(caught.exception.exit_code, 69)

    def test_a_root_daemon_is_refused_before_the_image_is_looked_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "calls.log"
            os.environ["FAKE_DOCKER_LOG"], os.environ["FAKE_DOCKER_ROOTLESS"] = str(log), "0"
            try:
                with self.assertRaises(sandbox.DaemonError):
                    make(tmp).preflight()
            finally:
                del os.environ["FAKE_DOCKER_LOG"], os.environ["FAKE_DOCKER_ROOTLESS"]
            self.assertEqual([json.loads(l)[0] for l in log.read_text().splitlines()], ["info"])

    def test_every_docker_call_names_the_pinned_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            hosts = pathlib.Path(tmp) / "hosts.log"
            script = pathlib.Path(tmp) / "hello.py"
            script.write_text("print(1)\n")
            os.environ["FAKE_DOCKER_HOSTS"] = str(hosts)
            os.environ["DOCKER_HOST"] = "unix:///wrong/from/the/environment.sock"
            try:
                box = sandbox.Sandbox(IMAGE, docker=FAKE_DOCKER, host="unix:///pinned.sock",
                                      scratch=pathlib.Path(tmp) / "stage")
                box.preflight()
                box.run_python(script)
            finally:
                del os.environ["FAKE_DOCKER_HOSTS"], os.environ["DOCKER_HOST"]
            seen = hosts.read_text().split()
            self.assertGreaterEqual(len(seen), 4)  # info, image inspect, run, rm
            self.assertEqual(set(seen), {"unix:///pinned.sock"})

    def test_the_record_names_the_runc_and_containerd_versions(self):
        text = make("/x").record()
        self.assertIn("component runc fake-1", text)
        self.assertIn("component containerd fake-2", text)
        self.assertIn(f"image {IMAGE}", text)
        self.assertIn("docker_host unix:///run/user/", text)
        self.assertIn("name=rootless", text)


class Run(unittest.TestCase):
    def test_a_program_runs_through_docker_and_its_stage_directory_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = pathlib.Path(tmp) / "hello.py"
            script.write_text("print(2 + 3)\n")
            box = make(tmp)
            self.assertEqual(box.run_python(script), (0, "5\n"))
            self.assertEqual(list((pathlib.Path(tmp) / "stage").iterdir()), [])

    def test_the_staged_script_is_world_readable_so_that_a_non_root_user_can_run_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = pathlib.Path(tmp) / "hello.py"
            script.write_text("print(1)\n")
            os.chmod(script, 0o600)
            seen = {}

            class Spy(sandbox.Sandbox):
                def command(self, host_dir, name):
                    staged = pathlib.Path(host_dir)
                    seen["dir"] = stat.S_IMODE(staged.stat().st_mode)
                    seen["file"] = stat.S_IMODE((staged / sandbox.SCRIPT_NAME).stat().st_mode)
                    return super().command(host_dir, name)

            Spy(IMAGE, docker=FAKE_DOCKER, scratch=pathlib.Path(tmp) / "stage").run_python(script)
            self.assertEqual(seen, {"dir": 0o755, "file": 0o644})

    def test_a_killed_run_is_reported_as_a_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = pathlib.Path(tmp) / "hello.py"
            script.write_text("print(1)\n")
            os.environ["FAKE_DOCKER_EXIT"] = "137"
            try:
                code, text = make(tmp).run_python(script)
            finally:
                del os.environ["FAKE_DOCKER_EXIT"]
            self.assertIsNone(code)
            self.assertIn("longer than 30 seconds", text)

    def test_docker_failure_codes_are_a_sandbox_error_not_a_program_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = pathlib.Path(tmp) / "hello.py"
            script.write_text("print(1)\n")
            for code in ("125", "126", "127"):
                with self.subTest(code=code):
                    os.environ["FAKE_DOCKER_EXIT"] = code
                    try:
                        with self.assertRaises(sandbox.SandboxError):
                            make(tmp).run_python(script)
                    finally:
                        del os.environ["FAKE_DOCKER_EXIT"]

    def test_a_program_that_floods_its_output_is_cut_at_the_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = pathlib.Path(tmp) / "hello.py"
            script.write_text("print(1)\n")
            os.environ["FAKE_DOCKER_FLOOD"] = "1"
            try:
                box = sandbox.Sandbox(IMAGE, docker=FAKE_DOCKER, scratch=pathlib.Path(tmp) / "stage",
                                      output_cap=4096)
                started = time.monotonic()
                code, text = box.run_python(script)
            finally:
                del os.environ["FAKE_DOCKER_FLOOD"]
            self.assertIsNone(code)
            self.assertIn("more than 4096 bytes", text)
            self.assertLess(len(text), 4096 + 200)
            self.assertLess(time.monotonic() - started, 20)

    def test_the_docker_client_never_sees_a_key_in_its_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = pathlib.Path(tmp) / "hello.py"
            script.write_text("print(2 + 3)\n")
            os.environ["ENGINESRV_API_KEY"] = "must-not-reach-docker"
            try:
                # the fake docker refuses (125) when it sees a variable that looks like a secret
                self.assertEqual(make(tmp).run_python(script), (0, "5\n"))
            finally:
                del os.environ["ENGINESRV_API_KEY"]

    def test_a_missing_docker_client_is_a_sandbox_error_at_run_time_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = pathlib.Path(tmp) / "hello.py"
            script.write_text("print(1)\n")
            with self.assertRaises(sandbox.SandboxError):
                make(tmp, docker="/nonexistent/docker").run_python(script)

    def test_only_a_scratch_directory_that_the_sandbox_made_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = pathlib.Path(tmp) / "hello.py"
            script.write_text("print(1)\n")
            own = pathlib.Path(tmp) / "own"
            box = sandbox.Sandbox(IMAGE, docker=FAKE_DOCKER, scratch=own)
            box.run_python(script)
            box.release()
            self.assertFalse(own.exists())
            existing = pathlib.Path(tmp) / "existing"
            existing.mkdir()
            box = sandbox.Sandbox(IMAGE, docker=FAKE_DOCKER, scratch=existing)
            box.run_python(script)
            box.release()
            self.assertTrue(existing.is_dir())

    def test_preflight_refuses_a_scratch_directory_with_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = pathlib.Path(tmp) / "stage"
            stage.mkdir()
            (stage / "x").write_text("x")
            with self.assertRaises(sandbox.SandboxError):
                make(tmp).preflight()

    def test_the_docker_client_environment_is_pinned_to_the_host_and_has_no_context(self):
        env = sandbox.clean_environment({"DOCKER_CONTEXT": "default", "DOCKER_HOST": "unix:///wrong",
                                         "PATH": "/bin"}, host="unix:///pinned.sock")
        self.assertEqual(env, {"DOCKER_HOST": "unix:///pinned.sock", "PATH": "/bin"})

    def test_the_environment_for_docker_has_no_key(self):
        env = sandbox.clean_environment({"PATH": "/bin", "ENGINESRV_API_KEY": "k",
                                         "SOME_TOKEN": "t", "HOME": "/h"})
        self.assertEqual(env, {"PATH": "/bin", "HOME": "/h"})


if __name__ == "__main__":
    unittest.main()
