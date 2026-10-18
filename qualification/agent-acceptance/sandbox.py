"""Container sandbox for code that a model wrote.

Model-written code never runs on a host. It runs only in `docker run` with an image that the
caller pins by digest (SANDBOX_IMAGE, `name@sha256:<64 hex>`), no network, a read-only root, a
small tmpfs, a non-root user, memory, CPU, and process limits, a timeout, and the script mounted
read-only. The harness never pulls an image: a missing image stops the run, because a pull needs
the user's approval. The key is not passed to the container.

The harness talks to the ROOTLESS Docker daemon only. pinu also runs a root dockerd, and the user is
in group docker, so the socket is pinned explicitly (`docker -H unix:///run/user/<uid>/docker.sock`,
override with SANDBOX_DOCKER_HOST) and `docker info` must report the security option `rootless`
before any run. A root daemon, or no daemon at the socket, stops the run with status 69.

`python3 -I sandbox.py` checks the configuration and exits with 0, or with 67 (image) or 69
(daemon) and a message. `python3 -I sandbox.py --record FILE` also writes the daemon, runc, and
containerd versions and the image id to FILE.
"""

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

EXIT_SANDBOX = 67
EXIT_DAEMON = 69
IMAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:-]*@sha256:[0-9a-f]{64}$")
MISSING = "sandbox image missing; needs the user's approval to pull"
SCRIPT_NAME = "script.py"
# A run that is killed by `timeout -s KILL` ends with 137, one that is stopped by SIGTERM with 124.
TIMEOUT_CODES = (124, 137)
# docker itself uses 125 (daemon error), 126 (command not invokable), and 127 (command not found).
DOCKER_FAILURE_CODES = (125, 126, 127)
# The output of the program is capped: a print loop must not fill the memory or the disk of the host.
OUTPUT_CAP = 1024 * 1024


class SandboxError(Exception):
    """The sandbox is not configured or its image is absent. Nothing ran."""

    exit_code = EXIT_SANDBOX


class DaemonError(SandboxError):
    """The docker daemon is not the rootless daemon, or it does not answer. Nothing ran."""

    exit_code = EXIT_DAEMON


def default_host():
    """The socket of the rootless daemon of this user, unless SANDBOX_DOCKER_HOST overrides it."""
    return os.environ.get("SANDBOX_DOCKER_HOST") or f"unix:///run/user/{os.getuid()}/docker.sock"


def clean_environment(environ=None, host=None):
    """The environment for the docker client: no variable that looks like a key or a token, and no
    DOCKER_CONTEXT. DOCKER_HOST names the pinned daemon when `host` is given."""
    environ = os.environ if environ is None else environ
    clean = {name: value for name, value in environ.items()
             if not re.search(r"KEY|TOKEN|SECRET|PASSWORD", name, re.IGNORECASE)
             and name != "DOCKER_CONTEXT"}
    if host:
        clean["DOCKER_HOST"] = host
    return clean


class Sandbox:
    def __init__(self, image, docker=None, scratch=None, seconds=30, output_cap=OUTPUT_CAP,
                 host=None):
        self.image = image or ""
        self.docker = docker or os.environ.get("DOCKER") or "docker"
        self.host = host or default_host()
        self.scratch = pathlib.Path(scratch) if scratch else None
        self.seconds = seconds
        self.output_cap = output_cap
        self.last_container = None   # the name of the container of the last run_python (for the log)
        self._created_scratch = False

    def release(self):
        """Removes the scratch directory, but only when this object created it."""
        if self.scratch is not None and self._created_scratch:
            shutil.rmtree(self.scratch, ignore_errors=True)
        self._created_scratch = False

    def _client(self, *args):
        return [self.docker, "-H", self.host, *args]

    def _env(self):
        return clean_environment(host=self.host)

    def _query(self, *args):
        try:
            return subprocess.run(self._client(*args), capture_output=True, text=True, timeout=60,
                                  env=self._env())
        except (OSError, subprocess.TimeoutExpired) as error:
            raise DaemonError(f"cannot run {self.docker}: {error!r}") from error

    def assert_rootless(self):
        """Raises DaemonError unless the daemon at self.host answers and reports `rootless`."""
        done = self._query("info", "--format", "{{json .SecurityOptions}}")
        if done.returncode != 0:
            raise DaemonError(f"no docker daemon answers at {self.host}: {done.stderr.strip()[:300]}")
        if "name=rootless" not in done.stdout:
            raise DaemonError(f"the docker daemon at {self.host} is not rootless "
                              f"(security options {done.stdout.strip()[:200]}); the sandbox "
                              "uses the rootless daemon only")

    def record(self):
        """Text for the evidence: the pinned host and the versions of daemon, runc, containerd."""
        info = self._query("version", "--format", "{{json .Server.Components}}")
        security = self._query("info", "--format", "{{json .SecurityOptions}}")
        image = self._query("image", "inspect", "--format", "{{.Id}}", self.image)
        lines = [f"docker_host {self.host}", f"image {self.image}",
                 f"image_id {image.stdout.strip()}",
                 f"security_options {security.stdout.strip()}"]
        try:
            for component in json.loads(info.stdout):
                lines.append(f"component {component.get('Name')} {component.get('Version')}")
        except ValueError:
            lines.append(f"components unreadable: {info.stdout.strip()[:200]}")
        return "\n".join(lines) + "\n"

    def preflight(self):
        """Raises SandboxError unless the rootless daemon answers and the pinned image is present
        on it. Never pulls."""
        if not self.image:
            raise SandboxError(f"{MISSING}: SANDBOX_IMAGE is not set. Set it to a digest-pinned "
                               "image that is already present, name@sha256:<64 hex>.")
        if not IMAGE_PATTERN.match(self.image):
            raise SandboxError(f"{MISSING}: SANDBOX_IMAGE must have the form name@sha256:<64 hex>, "
                               f"got {self.image!r}.")
        self.assert_rootless()
        done = self._query("image", "inspect", self.image)
        if done.returncode != 0:
            raise SandboxError(f"{MISSING}: {self.image} is not present locally.")
        if self.scratch is not None:
            if self.scratch.exists() and (not self.scratch.is_dir() or any(self.scratch.iterdir())):
                raise SandboxError(f"the stage directory {self.scratch} exists and is not empty; "
                                   "give a new --workdir")

    def command(self, host_dir, name):
        """The `docker run` argument list for the script in host_dir. No shell is involved."""
        host_dir = str(host_dir)
        if ":" in host_dir or "," in host_dir:
            raise SandboxError(f"the script directory {host_dir!r} has a colon or a comma")
        return [
            self.docker, "-H", self.host, "run", "--rm", "--pull", "never", "--name", name,
            "--log-driver", "none",
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
            "--user", "65534:65534",
            "--memory", "256m", "--memory-swap", "256m",
            "--cpus", "1",
            "--pids-limit", "64",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--shm-size", "16m",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp",
            "-v", f"{host_dir}:/work:ro", "-w", "/work",
            self.image,
            "timeout", "-s", "KILL", str(self.seconds),
            "python", "-I", f"/work/{SCRIPT_NAME}",
        ]

    def run_python(self, script):
        """Runs one Python file in the container. Returns (exit code or None, output text).

        The file is copied into a fresh private directory, which is mounted read-only. The exit
        code is None when the run timed out or wrote more than output_cap bytes. Output is read
        through a capped reader, and the container is removed when the cap or the time is hit.
        Raises SandboxError when docker itself failed (status 125 to 127).
        """
        script = pathlib.Path(script)
        if self.scratch is None:
            raise SandboxError("the sandbox needs a scratch directory")
        if not self.scratch.exists():
            self.scratch.mkdir(parents=True)
            self._created_scratch = True
        stage = pathlib.Path(tempfile.mkdtemp(prefix="stage-", dir=self.scratch))
        name = "acc-sandbox-" + uuid.uuid4().hex[:12]
        self.last_container = name
        try:
            os.chmod(stage, 0o755)
            target = stage / SCRIPT_NAME
            shutil.copyfile(script, target)
            os.chmod(target, 0o644)
            return self._run(stage, name)
        finally:
            if stage.parent == self.scratch:
                shutil.rmtree(stage, ignore_errors=True)

    def _run(self, stage, name):
        env = self._env()
        try:
            proc = subprocess.Popen(self.command(stage, name), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, env=env)
        except OSError as error:
            raise SandboxError(f"cannot run {self.docker}: {error!r}") from error
        chunks, state = [], {"size": 0, "over": False}

        def reader():
            while True:
                data = proc.stdout.read1(8192)
                if not data:
                    return
                room = self.output_cap - state["size"]
                chunks.append(data[:max(room, 0)])
                state["size"] += len(data)
                if state["size"] > self.output_cap:
                    # Keep draining without storing: a client that stops reading can block
                    # `docker rm -f` while the container writes. The main loop removes the container.
                    state["over"] = True

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        deadline = time.monotonic() + self.seconds + 20
        timed_out = False
        try:
            while thread.is_alive() and not state["over"]:  # the reader ends at EOF
                if time.monotonic() > deadline:
                    timed_out = True
                    break
                thread.join(0.05)
            if state["over"] or timed_out:
                self._remove(name)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            thread.join(5)
        finally:
            self._remove(name)
            proc.stdout.close()
        text = b"".join(chunks).decode("utf-8", errors="replace")
        if state["over"]:
            return None, text + f"\nerror: the program wrote more than {self.output_cap} bytes of output"
        if timed_out or proc.returncode in TIMEOUT_CODES:
            return None, f"error: the program ran longer than {self.seconds} seconds"
        if proc.returncode in DOCKER_FAILURE_CODES:
            raise SandboxError(f"docker failed with status {proc.returncode} (the program cannot "
                               f"use the exit codes 125 to 127): {text[:500]}")
        return proc.returncode, text

    def _remove(self, name):
        subprocess.run(self._client("rm", "-f", name), capture_output=True, timeout=60,
                       env=self._env(), check=False)


def from_environment(scratch=None):
    return Sandbox(os.environ.get("SANDBOX_IMAGE", ""), scratch=scratch)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    sandbox = from_environment()
    try:
        sandbox.preflight()
        if argv[:1] == ["--record"] and len(argv) == 2:
            pathlib.Path(argv[1]).write_text(sandbox.record(), encoding="utf-8")
    except SandboxError as error:
        print(str(error), file=sys.stderr)
        return error.exit_code
    print(f"sandbox image present on the rootless daemon {sandbox.host}: {sandbox.image}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
