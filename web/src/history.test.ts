/** Tests for the pure request builder (node:test; no DOM at runtime). */

import { test } from "node:test";
import { deepStrictEqual, strictEqual } from "node:assert";
import { buildChatBody, buildMessages } from "./history.js";
import { defaultSettings, type Settings, type UiMessage } from "./store.js";

function msg(
  role: "user" | "assistant",
  content: string,
  state: UiMessage["state"] = "complete"
): UiMessage {
  return { id: role + "-" + content, role, content, state, createdAt: 0 };
}

/** Documentation guard of the removed pre-fix builder only: oldBuild is test-local, so this covers no production code. */
function oldBuild(systemMessage: string, history: UiMessage[], userText: string) {
  const messages: { role: string; content: string }[] = [];
  if (systemMessage.trim() !== "") {
    messages.push({ role: "system", content: systemMessage });
  }
  for (const m of history) {
    if (m.role === "user") messages.push({ role: "user", content: m.content });
    else if (m.content !== "") messages.push({ role: "assistant", content: m.content });
  }
  messages.push({ role: "user", content: userText });
  return messages;
}

test("the old two-step builder duplicated the newest user turn", () => {
  const history = [
    msg("user", "user1"),
    msg("assistant", "assistant1"),
    msg("user", "user2"),
  ];
  const userTurns = oldBuild("", history, "user2")
    .filter((m) => m.role === "user")
    .map((m) => m.content);
  deepStrictEqual(userTurns, ["user1", "user2", "user2"]);
});

test("after user1, assistant1, user2 each turn is sent exactly once", () => {
  const history = [
    msg("user", "user1"),
    msg("assistant", "assistant1"),
    msg("user", "user2"),
  ];
  deepStrictEqual(buildMessages("", history), [
    { role: "user", content: "user1" },
    { role: "assistant", content: "assistant1" },
    { role: "user", content: "user2" },
  ]);
});

test("a non-blank system message is sent first; a blank one is not sent", () => {
  const history = [msg("user", "hello")];
  deepStrictEqual(buildMessages("You are terse.", history), [
    { role: "system", content: "You are terse." },
    { role: "user", content: "hello" },
  ]);
  deepStrictEqual(buildMessages("   ", history), [{ role: "user", content: "hello" }]);
});

test("an assistant turn with empty content is skipped", () => {
  const history = [msg("user", "hi"), msg("assistant", ""), msg("user", "again")];
  deepStrictEqual(buildMessages("", history), [
    { role: "user", content: "hi" },
    { role: "user", content: "again" },
  ]);
});

test("a whitespace-only assistant turn is skipped", () => {
  const history = [msg("user", "hi"), msg("assistant", "   "), msg("user", "again")];
  deepStrictEqual(buildMessages("", history), [
    { role: "user", content: "hi" },
    { role: "user", content: "again" },
  ]);
});

test("a stopped partial assistant turn is kept", () => {
  const history = [msg("user", "hi"), msg("assistant", "partial answ", "stopped")];
  deepStrictEqual(buildMessages("", history), [
    { role: "user", content: "hi" },
    { role: "assistant", content: "partial answ" },
  ]);
});

test("reasoning text is never sent back", () => {
  const assistant = msg("assistant", "the answer");
  assistant.reasoning = "private chain of thought";
  deepStrictEqual(buildMessages("", [msg("user", "q"), assistant]), [
    { role: "user", content: "q" },
    { role: "assistant", content: "the answer" },
  ]);
});

test("the request body uses the contract keys", () => {
  const settings: Settings = {
    ...defaultSettings,
    model: "fifi-8b",
    drawer: { ...defaultSettings.drawer, topK: 0, thinking: "off" },
  };
  const body = buildChatBody(settings, buildMessages("", [msg("user", "hi")]));
  const sent = JSON.parse(JSON.stringify(body)) as Record<string, unknown>;
  deepStrictEqual(Object.keys(sent).sort(), [
    "max_tokens",
    "messages",
    "model",
    "temperature",
    "top_p",
    "x-thinking",
  ]);
  strictEqual(sent["x-thinking"], "off");

  const unset: Settings = { ...defaultSettings, model: "", drawer: { ...defaultSettings.drawer } };
  const bare = JSON.parse(JSON.stringify(buildChatBody(unset, []))) as Record<string, unknown>;
  deepStrictEqual(Object.keys(bare).sort(), [
    "max_tokens",
    "messages",
    "temperature",
    "top_p",
    "x-thinking",
  ]);
});

test("a system message and non-empty assistant turns are all kept in order", () => {
  const history = [
    msg("user", "user1"),
    msg("assistant", "assistant1"),
    msg("user", "user2"),
  ];
  deepStrictEqual(buildMessages("You are terse.", history), [
    { role: "system", content: "You are terse." },
    { role: "user", content: "user1" },
    { role: "assistant", content: "assistant1" },
    { role: "user", content: "user2" },
  ]);
});

test("the body carries the drawer and model values exactly", () => {
  const settings: Settings = {
    ...defaultSettings,
    model: "fifi-8b",
    drawer: {
      ...defaultSettings.drawer,
      temperature: 0.35,
      topP: 0.75,
      topK: 40,
      maxTokens: 777,
    },
  };
  const body = buildChatBody(settings, []);
  strictEqual(body.model, "fifi-8b");
  strictEqual(body.temperature, 0.35);
  strictEqual(body.top_p, 0.75);
  strictEqual(body.top_k, 40);
  strictEqual(body.max_tokens, 777);
});

test("the serialized body carries x-thinking and never x_thinking", () => {
  const settings: Settings = {
    ...defaultSettings,
    drawer: { ...defaultSettings.drawer, thinking: "off" },
  };
  const sent = JSON.parse(
    JSON.stringify(buildChatBody(settings, []))
  ) as Record<string, unknown>;
  strictEqual(sent["x-thinking"], "off");
  strictEqual(Object.hasOwn(sent, "x_thinking"), false);
});
