/** Tests for the pure dead-chat id set (node:test; no DOM at runtime). */

import { test } from "node:test";
import { strictEqual } from "node:assert";
import { DeadChats } from "./deadchats.js";

test("a failed delete leaves the chat saving", () => {
  const dead = new DeadChats();
  dead.mark("chat-1"); // marked optimistically before the delete
  dead.unmark("chat-1"); // the delete failed; the chat still exists
  strictEqual(dead.has("chat-1"), false);
});

test("a successful delete marks the chat dead", () => {
  const dead = new DeadChats();
  dead.mark("chat-1");
  strictEqual(dead.has("chat-1"), true);
});
