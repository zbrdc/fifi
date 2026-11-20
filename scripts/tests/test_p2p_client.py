"""Contract checks for the bounded client, with no sockets or GPU work."""

import json
import unittest

import p2p_test_support  # noqa: F401
from scripts.p2p_concurrent_guard import client


class Response:
    status = 200

    def __init__(self, payload):
        self.payload = payload
        self.read_limit = None

    def read(self, limit):
        self.read_limit = limit
        return self.payload[:limit]


class Connection:
    def __init__(self, response, fail=False):
        self.response = response
        self.fail = fail
        self.closed = False
        self.request_args = None

    def request(self, *args, **kwargs):
        self.request_args = (args, kwargs)
        if self.fail:
            raise TimeoutError("request timeout")

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class BoundedClientTests(unittest.TestCase):
    def reply(self, **updates):
        value = {"id": "test", "choices": [{"finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 12, "completion_tokens": 3}}
        value.update(updates)
        return json.dumps(value).encode()

    def call(self, connection, times=(10, 20)):
        clock = iter(times)
        def factory(host, port, **kwargs):
            self.assertEqual((host, port, kwargs), ("127.0.0.1", 18793, {"timeout": 5}))
            return connection
        return client.request_once("test-key", connection_factory=factory,
                                   clock=lambda: next(clock))

    def test_success_returns_only_timing_and_counts_and_closes(self):
        response = Response(self.reply())
        connection = Connection(response)
        result = self.call(connection)
        self.assertEqual(set(result), {"interval", "prompt_tokens", "completion_tokens"})
        self.assertEqual(result["prompt_tokens"], 12)
        self.assertEqual(result["completion_tokens"], 3)
        self.assertTrue(result["interval"].valid())
        self.assertEqual(response.read_limit, client.MAX_BODY + 1)
        self.assertTrue(connection.closed)
        args, kwargs = connection.request_args
        self.assertEqual(args, ("POST", "/v1/chat/completions"))
        payload = json.loads(kwargs["body"])
        self.assertEqual(payload["max_tokens"], 128)
        self.assertFalse(payload["stream"])

    def test_transport_failure_closes_connection(self):
        connection = Connection(Response(b""), fail=True)
        with self.assertRaises(TimeoutError):
            self.call(connection)
        self.assertTrue(connection.closed)

    def test_redirect_is_refused_without_reading_body(self):
        response = Response(b"secret response")
        response.status = 302
        connection = Connection(response)
        with self.assertRaisesRegex(ValueError, "did not return 200"):
            self.call(connection)
        self.assertIsNone(response.read_limit)
        self.assertTrue(connection.closed)

    def test_oversized_and_malformed_replies_are_refused_and_closed(self):
        for payload in [b"x" * (client.MAX_BODY + 1), b"bad json", b"[]",
                        self.reply(usage={"prompt_tokens": True, "completion_tokens": 3}),
                        self.reply(usage={"prompt_tokens": 12, "completion_tokens": 129}),
                        self.reply(choices=[]), self.reply(id="")]:
            with self.subTest(size=len(payload)):
                connection = Connection(Response(payload))
                with self.assertRaises(ValueError):
                    self.call(connection)
                self.assertTrue(connection.closed)

    def test_nonpositive_interval_is_refused_and_closed(self):
        for times in [(20, 10), (10, 10)]:
            connection = Connection(Response(self.reply()))
            with self.assertRaisesRegex(ValueError, "timing"):
                self.call(connection, times)
            self.assertTrue(connection.closed)

    def test_bad_authorization_is_rejected_before_connecting(self):
        def no_connection(*args, **kwargs):
            self.fail("authorization must be checked before connection")
        for key in [None, "", "line\rbreak", "line\nbreak"]:
            with self.assertRaisesRegex(ValueError, "authorization"):
                client.request_once(key, connection_factory=no_connection)
