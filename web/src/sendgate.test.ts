/** Tests for the pure in-flight send gate (node:test; no DOM at runtime). */

import { test } from "node:test";
import { ok, strictEqual } from "node:assert";
import { SendGate } from "./sendgate.js";

test("a second tryBegin while a send is in flight is refused", () => {
  const gate = new SendGate();
  ok(gate.tryBegin());
  strictEqual(gate.tryBegin(), false);
});

test("end re-opens the gate for the next send", () => {
  const gate = new SendGate();
  ok(gate.tryBegin());
  gate.end();
  ok(gate.tryBegin());
});
