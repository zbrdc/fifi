#!/usr/bin/env python3
"""Check tiny-model HTTP generation through the installed Python MLX reference."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def final_status(generation_ok, owned_server_stopped, cleanup_ok=True):
    ok = generation_ok and owned_server_stopped and cleanup_ok
    return "passed" if ok else "failed"


def stop_process(process, errors):
    """Stop the owned server. Record each failure in errors and continue."""
    def attempt(name, action):
        try:
            action()
        except subprocess.TimeoutExpired:
            pass
        except Exception as error:
            errors.append(name + ": " + type(error).__name__ + ": " + str(error))
    attempt("terminate", process.terminate)
    attempt("wait after terminate", lambda: process.wait(timeout=10))
    try:
        stopped = process.poll() is not None
    except Exception:
        stopped = False
    if not stopped:
        attempt("kill", process.kill)
        attempt("wait after kill", lambda: process.wait(timeout=10))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "failed",
        "scope": "Python MLX HTTP reference only. No Rust engine qualification.",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "runner_sha256": digest(Path(__file__)),
        "python": platform.python_version(),
    }
    process = None
    generation_ok = False
    interrupted = False
    try:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Run this check on the Apple Silicon Mac")
        if digest(args.manifest) != "84655d8a2307218df32d4c5eb6d41cc6c2958ee819510820f1bc4483dedff9be":
            raise RuntimeError("Unexpected pinned artifact manifest")
        manifest = json.loads(args.manifest.read_text())
        if manifest["repo"] != "mlx-community/SmolLM2-135M-Instruct-8bit":
            raise RuntimeError("Unexpected checkpoint")
        if manifest["revision"] != "0f0d9b8218915bc34d401e1a340b8c049d300d5e":
            raise RuntimeError("Unexpected checkpoint revision")
        files = manifest["files"]
        if not any(f["name"] == "model.safetensors" for f in files):
            raise RuntimeError("Manifest has no weight file")
        for item in files:
            name = item["name"]
            if Path(name).name != name or name in (".", ".."):
                raise RuntimeError("Manifest contains an invalid file name")
            path = args.model / name
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise RuntimeError("Checkpoint identity mismatch: " + name)
        result.update(repo=manifest["repo"], revision=manifest["revision"],
                      manifest_sha256=digest(args.manifest), model_files=files)
        result["packages"] = {name: importlib.metadata.version(name)
                              for name in ("mlx", "mlx-lm", "transformers", "safetensors")}
        env = os.environ.copy()
        for name in ("TMPDIR", "TMP", "TEMP", "HF_HOME", "XDG_CACHE_HOME"):
            path = args.output / name.lower()
            path.mkdir()
            env[name] = str(path)
        env["HF_HUB_CACHE"] = str(Path(env["HF_HOME"]) / "hub")
        Path(env["HF_HUB_CACHE"]).mkdir()
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
                   PYTHONDONTWRITEBYTECODE="1")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        # The port can race after release. A bind failure must fail this run.
        code = ("import mlx.core as mx; "
                "assert mx.metal.is_available(); "
                "mx.set_default_device(mx.gpu); "
                "mx.set_memory_limit(1073741824); "
                "mx.set_cache_limit(268435456); "
                "from mlx_lm.server import main; main()")
        command = [sys.executable, "-B", "-c", code, "--model", str(args.model),
                   "--host", "127.0.0.1", "--port", str(port), "--max-tokens", "32",
                   "--decode-concurrency", "1", "--prompt-concurrency", "1",
                   "--prompt-cache-size", "1"]
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        base = "http://127.0.0.1:" + str(port)
        with (args.output / "server.log").open("wb") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Reference server exited before readiness")
                try:
                    with opener.open(base + "/v1/models", timeout=2) as response:
                        models = json.load(response)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Reference server readiness timed out")
                    time.sleep(0.2)
            request = {
                "model": str(args.model),
                "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
                "temperature": 0, "max_tokens": 32, "stream": False,
            }
            started = time.monotonic()
            req = urllib.request.Request(base + "/v1/chat/completions",
                data=json.dumps(request).encode(), headers={"Content-Type": "application/json"})
            with opener.open(req, timeout=90) as response:
                reply = json.load(response)
            content = reply["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("Reference server produced no text")
            if process.poll() is not None:
                raise RuntimeError("Reference server exited during generation")
            generation_ok = True
            result.update(request=request, response=reply,
                          models=models, elapsed_seconds=time.monotonic()-started,
                          device="Metal GPU", memory_limit_bytes=1073741824,
                          cache_limit_bytes=268435456)
    except Exception as error:
        generation_ok = False
        result["error"] = type(error).__name__ + ": " + str(error)
    except KeyboardInterrupt:
        generation_ok = False
        interrupted = True
        result["error"] = "KeyboardInterrupt"
    finally:
        try:
            cleanup_errors = []
            if process is not None:
                stop_process(process, cleanup_errors)
                try:
                    result["owned_server_stopped"] = process.poll() is not None
                except Exception as error:
                    cleanup_errors.append("poll: " + type(error).__name__ + ": " + str(error))
                    result["owned_server_stopped"] = False
                if not result["owned_server_stopped"]:
                    result.setdefault("error", "Owned reference server did not stop")
            log = args.output / "server.log"
            try:
                if log.exists():
                    result["server_log_sha256"] = digest(log)
            except Exception as error:
                cleanup_errors.append("log digest: " + type(error).__name__ + ": " + str(error))
            if cleanup_errors:
                result["cleanup_errors"] = cleanup_errors
                result.setdefault("error", "Cleanup failed: " + cleanup_errors[0])
            if process is not None:
                result["status"] = final_status(generation_ok,
                                                result["owned_server_stopped"],
                                                not cleanup_errors)
        except KeyboardInterrupt:
            result["status"] = "failed"
            result.setdefault("error", "KeyboardInterrupt during cleanup")
            raise
        finally:
            (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result.get(k) for k in
                      ("status", "scope", "error", "owned_server_stopped")}, indent=2))
    if interrupted:
        return 130
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
