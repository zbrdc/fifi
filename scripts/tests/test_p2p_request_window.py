import hashlib
import unittest
from unittest.mock import Mock

import p2p_test_support  # noqa: F401 -- make repository packages visible under python -I

from scripts.p2p_concurrent_guard.request_window import measure_request

REQUEST = b"fixed synthetic request"
PAYLOAD = hashlib.sha256(REQUEST).hexdigest()


def reply(_payload=None):
    return {
        "id": "request-1",
        "usage": {"prompt_tokens": 40, "completion_tokens": 2, "reused_tokens": 0},
        "choices": [{"message": {"content": "4"}, "finish_reason": "stop"}],
    }


class RequestWindowTests(unittest.TestCase):
    def test_entire_send_and_read_is_timed_without_changing_rate(self):
        events = []

        def clock():
            events.append("clock")
            return 10_000_000_000 if len(events) == 1 else 12_000_000_000

        def send(payload):
            self.assertEqual(payload, REQUEST)
            events.append("request-and-full-read")
            return reply()

        sample, window = measure_request(
            send,
            index=0,
            payload=REQUEST,
            payload_sha256=PAYLOAD,
            expected_prompt_tokens=40,
            clock=clock,
        )
        self.assertEqual(events, ["clock", "request-and-full-read", "clock"])
        self.assertEqual(sample["estimated_prompt_tokens_per_second"], 20.0)
        self.assertEqual(sample["reuse_class"], "cold")
        self.assertEqual(window["status"], "completed")
        self.assertEqual(window["reused_tokens"], 0)
        self.assertEqual(window["payload_sha256"], PAYLOAD)
        self.assertEqual(
            (window["start_ns"], window["end_ns"]),
            (10_000_000_000, 12_000_000_000),
        )

    def test_failed_request_cannot_produce_completed_window(self):
        clock = Mock(return_value=10)
        with self.assertRaises(TimeoutError):
            measure_request(
                Mock(side_effect=TimeoutError()),
                index=0,
                payload=REQUEST,
                payload_sha256=PAYLOAD,
                expected_prompt_tokens=40,
                clock=clock,
            )
        self.assertEqual(clock.call_count, 1)

    def test_bad_times_or_result_fail_closed(self):
        for times in [
            (2, 2),
            (2, 1),
            (-1, 2),
            (0, 2),
            (1, 2**63),
            (float("nan"), 3),
            (2, float("inf")),
            (True, 3),
        ]:
            with self.subTest(times=times), self.assertRaises(ValueError):
                measure_request(
                    reply,
                    index=0,
                    payload=REQUEST,
                    payload_sha256=PAYLOAD,
                    expected_prompt_tokens=40,
                    clock=Mock(side_effect=times),
                )
        bad = reply()
        del bad["id"]
        with self.assertRaises(ValueError):
            measure_request(
                lambda _payload: bad,
                index=0,
                payload=REQUEST,
                payload_sha256=PAYLOAD,
                expected_prompt_tokens=40,
                clock=Mock(side_effect=(1, 2)),
            )
        bad = reply()
        bad["choices"][0]["message"]["content"] = "private sentinel"
        with self.assertRaises(ValueError) as failure:
            measure_request(
                lambda _payload: bad,
                index=0,
                payload=REQUEST,
                payload_sha256=PAYLOAD,
                expected_prompt_tokens=40,
                clock=Mock(side_effect=(1, 2)),
            )
        self.assertNotIn("private sentinel", str(failure.exception))

    def test_unknown_warm_or_wrong_size_is_not_a_cold_sample(self):
        for reused in (None, 1):
            bad = reply()
            if reused is None:
                del bad["usage"]["reused_tokens"]
            else:
                bad["usage"]["reused_tokens"] = reused
            with self.subTest(reused=reused), self.assertRaises(ValueError):
                measure_request(
                    lambda _payload: bad,
                    index=0,
                    payload=REQUEST,
                    payload_sha256=PAYLOAD,
                    expected_prompt_tokens=40,
                    clock=Mock(side_effect=(1, 2)),
                )
        with self.assertRaises(ValueError):
            measure_request(
                reply,
                index=0,
                payload=REQUEST,
                payload_sha256=PAYLOAD,
                expected_prompt_tokens=41,
                clock=Mock(side_effect=(1, 2)),
            )

    def test_truncated_reply_and_unpinned_payload_fail_closed(self):
        bad = reply()
        bad["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(ValueError):
            measure_request(
                lambda _payload: bad,
                index=0,
                payload=REQUEST,
                payload_sha256=PAYLOAD,
                expected_prompt_tokens=40,
                clock=Mock(side_effect=(1, 2)),
            )
        with self.assertRaises(ValueError):
            measure_request(
                reply,
                index=0,
                payload=REQUEST,
                payload_sha256="unknown",
                expected_prompt_tokens=40,
                clock=Mock(side_effect=(1, 2)),
            )


if __name__ == "__main__":
    unittest.main()
