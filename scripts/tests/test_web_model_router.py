"""Exercise local model ordering and routing with loopback fake backends."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPT = Path(__file__).resolve().parents[1] / "web-model-router.mjs"


class Backend(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def reply(self):
        if self.path == "/v1/models" and self.server.unavailable:
            self.send_error(503)
            return
        if self.path == "/v1/models":
            value = {"object": "list", "data": [{"id": self.server.model}]}
        elif self.path.startswith("/v1/sampler-trace?"):
            value = {
                "backend": self.server.model,
                "auth_ok": self.headers.get("authorization")
                == "Bearer " + self.server.key,
                "forwarded_model_header": self.headers.get("x-fifi-trace-model"),
            }
        else:
            value = {
                "backend": self.server.model,
                "auth_ok": self.headers.get("authorization")
                == "Bearer " + self.server.key,
            }
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply()

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", "0")))
        if self.server.hold_stream:
            first, last = b"data: first\n\n", b"data: last\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            if self.server.chunked_stream:
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Content-Length", str(len(first) + len(last)))
            self.end_headers()
            self.write_stream_piece(first)
            self.wfile.flush()
            self.server.stream_started.set()
            deadline = time.monotonic() + 5
            while (
                not self.server.stream_release.is_set() and time.monotonic() < deadline
            ):
                readable, _, _ = select.select([self.connection], [], [], 0.05)
                if readable:
                    try:
                        closed = not self.connection.recv(1, socket.MSG_PEEK)
                    except OSError:
                        closed = True
                    if closed:
                        self.server.stream_disconnected.set()
                        return
            if self.server.stream_release.is_set():
                self.write_stream_piece(last)
                if self.server.chunked_stream:
                    self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            return
        self.reply()

    def write_stream_piece(self, data):
        if self.server.chunked_stream:
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        else:
            self.wfile.write(data)


class TrackingHTTPServer(ThreadingHTTPServer):
    """Track live backend sockets, including idle HTTP/1.1 connections."""

    def __init__(self, *args, **kwargs):
        self._connections = set()
        self._connections_changed = threading.Condition()
        super().__init__(*args, **kwargs)

    def get_request(self):
        connection, address = super().get_request()
        with self._connections_changed:
            self._connections.add(connection)
            self._connections_changed.notify_all()
        return connection, address

    def close_request(self, request):
        try:
            super().close_request(request)
        finally:
            with self._connections_changed:
                self._connections.discard(request)
                self._connections_changed.notify_all()

    def wait_for_no_connections(self, timeout=2):
        with self._connections_changed:
            return self._connections_changed.wait_for(
                lambda: not self._connections, timeout=timeout
            )


class RouterFixture(unittest.TestCase):
    close_backend_connections = False

    def setUp(self):
        node = shutil.which("node")
        if not node:
            self.fail("Node.js is required to test the model router")
        temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        primary = root / "primary"
        secondary = root / "secondary"
        primary.mkdir()
        secondary.mkdir()
        (primary / "api-key").write_text("primary-key\n")
        (secondary / "api-key").write_text("secondary-key\n")
        self.backends = []
        for model, key in (
            ("fifi-q36", "primary-key"),
            ("fifi-q36-2080ti", "secondary-key"),
        ):
            server = TrackingHTTPServer(("127.0.0.1", 0), Backend)
            server.model = model
            server.key = key
            server.unavailable = False
            server.hold_stream = False
            server.chunked_stream = False
            server.stream_started = threading.Event()
            server.stream_release = threading.Event()
            server.stream_disconnected = threading.Event()
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            self.backends.append(server)
        source = SCRIPT.read_text()
        for original, server in (
            ("http://127.0.0.1:18791", self.backends[0]),
            ("http://127.0.0.1:18793", self.backends[1]),
        ):
            self.assertEqual(source.count(original), 1)
            source = source.replace(original, f"http://127.0.0.1:{server.server_port}")
        copy = root / "router.mjs"
        copy.write_text(source)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = dict(
            os.environ,
            FIFI_PRIMARY_RUN=str(primary),
            FIFI_SECONDARY_RUN=str(secondary),
            FIFI_MODEL_ROUTER_PORT=str(port),
        )
        env.pop("FIFI_ROUTER_CLOSE_BACKEND_CONNECTIONS", None)
        if self.close_backend_connections:
            env["FIFI_ROUTER_CLOSE_BACKEND_CONNECTIONS"] = "1"
        self.proc = subprocess.Popen(
            [node, str(copy)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(self.stop_router)
        self.base = f"http://127.0.0.1:{port}"
        for _ in range(60):
            if self.proc.poll() is not None:
                self.fail("model router exited before accepting requests")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    pass
                break
            except OSError:
                time.sleep(0.05)
        else:
            self.fail("model router did not start")

    def stop_router(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=5)

    def get(self, path, headers=None):
        request = urllib.request.Request(self.base + path, headers=headers or {})
        with urllib.request.urlopen(request, timeout=4) as response:
            return json.load(response)

    def post(self, model):
        body = json.dumps({"model": model}).encode()
        request = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=4) as response:
            return json.load(response)


class ModelRouterTests(RouterFixture):
    close_backend_connections = True

    def test_primary_is_first_and_model_specific_routes_still_work(self):
        data = self.get("/v1/models")["data"]
        self.assertEqual(
            [model["id"] for model in data], ["fifi-q36", "fifi-q36-2080ti"]
        )
        self.assertEqual(
            self.get("/v1/system"), {"backend": "fifi-q36", "auth_ok": True}
        )
        self.assertEqual(
            self.post("fifi-q36"), {"backend": "fifi-q36", "auth_ok": True}
        )
        self.assertEqual(
            self.post("fifi-q36-2080ti"),
            {"backend": "fifi-q36-2080ti", "auth_ok": True},
        )
        self.assertEqual(self.post("unknown"), {"backend": "fifi-q36", "auth_ok": True})

    def test_primary_placeholder_stays_first_if_its_backend_is_unavailable(self):
        self.backends[0].unavailable = True
        self.assertEqual(
            [model["id"] for model in self.get("/v1/models")["data"]],
            ["fifi-q36", "fifi-q36-2080ti"],
        )

    def test_trace_lookup_uses_the_reply_model_even_when_ids_collide(self):
        path = "/v1/sampler-trace?id=shared-id"
        for model in ("fifi-q36", "fifi-q36-2080ti"):
            with self.subTest(model=model):
                result = self.get(path, {"X-Fifi-Trace-Model": model})
                self.assertEqual(result["backend"], model)
                self.assertTrue(result["auth_ok"])
                self.assertIsNone(result["forwarded_model_header"])
        self.assertEqual(self.get(path)["backend"], "fifi-q36")
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.get(path, {"X-Fifi-Trace-Model": "unknown-model"})
        self.assertEqual(error.exception.code, 400)

    def test_model_list_fails_when_both_backends_are_unavailable(self):
        for backend in self.backends:
            backend.unavailable = True
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.get("/v1/models")
        self.assertEqual(error.exception.code, 502)

    def test_completed_model_list_releases_backend_connections(self):
        self.assertEqual(len(self.get("/v1/models")["data"]), 2)
        for backend in self.backends:
            with self.subTest(model=backend.model):
                self.assertTrue(
                    backend.wait_for_no_connections(),
                    f"{backend.model} retained a completed model-list connection",
                )

    def test_completed_forwarded_request_releases_backend_connection(self):
        backend = self.backends[1]
        self.assertEqual(self.post(backend.model)["backend"], backend.model)
        self.assertTrue(
            backend.wait_for_no_connections(),
            "the 2080 backend retained a completed forwarded connection",
        )

    def open_held_stream(self, chunked):
        backend = self.backends[1]
        backend.hold_stream = True
        backend.chunked_stream = chunked
        self.addCleanup(backend.stream_release.set)
        body = json.dumps({"model": backend.model, "stream": True}).encode()
        request = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        response = urllib.request.urlopen(request, timeout=5)
        self.addCleanup(response.close)
        self.assertEqual(response.read(len(b"data: first\n\n")), b"data: first\n\n")
        self.assertTrue(backend.stream_started.wait(timeout=2))
        return backend, response

    def assert_stream_lifetime(self, chunked):
        backend, response = self.open_held_stream(chunked)
        with response:
            self.assertFalse(
                backend.wait_for_no_connections(timeout=0.2),
                "the active stream was disconnected before its response finished",
            )
            backend.stream_release.set()
            self.assertEqual(response.read(), b"data: last\n\n")
        self.assertTrue(
            backend.wait_for_no_connections(),
            "the completed stream retained its backend connection",
        )

    def test_stream_stays_connected_until_its_response_finishes(self):
        self.assert_stream_lifetime(chunked=False)

    def test_chunked_stream_stays_connected_until_its_response_finishes(self):
        self.assert_stream_lifetime(chunked=True)

    def test_client_disconnect_cancels_an_active_upstream_stream(self):
        backend, response = self.open_held_stream(chunked=True)
        response.close()
        self.assertTrue(
            backend.stream_disconnected.wait(timeout=2),
            "the upstream connection remained open after the client disconnected",
        )
        self.assertTrue(backend.wait_for_no_connections())


class DefaultConnectionPolicyTests(RouterFixture):
    def test_default_preserves_model_list_connection_reuse(self):
        self.assertEqual(len(self.get("/v1/models")["data"]), 2)
        self.assertFalse(
            self.backends[1].wait_for_no_connections(timeout=0.2),
            "the default policy unexpectedly closed the model-list connection",
        )


if __name__ == "__main__":
    unittest.main()
