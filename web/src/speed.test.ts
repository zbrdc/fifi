/** Tests for the live speed panel's pure summary (node:test; no DOM). */

import { test } from "node:test";
import { deepStrictEqual, strictEqual } from "node:assert";
import { speedSummary, type SpeedRow } from "./speed.js";

function row(overrides: Partial<SpeedRow>): SpeedRow {
  return {
    outcome: "ok",
    decode_tokens_per_second: null,
    time_to_first_token_ms: null,
    ...overrides,
  };
}

test("an empty ring has no medians and no finished count", () => {
  deepStrictEqual(speedSummary([]), { decodePerSecond: null, firstTokenMs: null, okCount: 0 });
});

test("medians ignore ok rows without the value", () => {
  const summary = speedSummary([
    row({ decode_tokens_per_second: 10, time_to_first_token_ms: 100 }),
    row({ decode_tokens_per_second: null, time_to_first_token_ms: 300 }),
    row({ decode_tokens_per_second: 20, time_to_first_token_ms: null }),
  ]);
  strictEqual(summary.decodePerSecond, 10, "one decode value: its own median");
  strictEqual(summary.firstTokenMs, 100, "one ttft value: its own median");
});

test("cancelled and error rows never feed a median (Opus MF1)", () => {
  const summary = speedSummary([
    row({ outcome: "cancelled", time_to_first_token_ms: 999, decode_tokens_per_second: 999 }),
    row({ outcome: "error", time_to_first_token_ms: 500, decode_tokens_per_second: 500 }),
    row({ time_to_first_token_ms: 100, decode_tokens_per_second: 7 }),
  ]);
  strictEqual(summary.firstTokenMs, 100, "the cancelled 999 ms must not become the median");
  strictEqual(summary.decodePerSecond, 7);
  strictEqual(summary.okCount, 1);
});

test("odd counts take the middle, even counts the lower middle", () => {
  const odd = speedSummary([
    row({ decode_tokens_per_second: 30 }),
    row({ decode_tokens_per_second: 10 }),
    row({ decode_tokens_per_second: 20 }),
  ]);
  strictEqual(odd.decodePerSecond, 20);
  const even = speedSummary([
    row({ decode_tokens_per_second: 30 }),
    row({ decode_tokens_per_second: 10 }),
    row({ decode_tokens_per_second: 25 }),
    row({ decode_tokens_per_second: 20 }),
  ]);
  strictEqual(even.decodePerSecond, 20, "sorted [10,20,25,30]: the lower middle");
});

test("every ok row counts, even without a value", () => {
  const summary = speedSummary([
    row({}),
    row({ decode_tokens_per_second: 5 }),
  ]);
  strictEqual(summary.okCount, 2);
  strictEqual(summary.decodePerSecond, 5);
});
