/** Tests for the pure settings coercion (node:test; no DOM at runtime). */

import { test } from "node:test";
import { deepStrictEqual, strictEqual } from "node:assert";
import { coerceSettings, defaultSettings } from "./store.js";

/** A fully valid stored drawer; each case replaces one field. */
const baseDrawer = {
  temperature: 0.5,
  topP: 0.5,
  topK: 8,
  maxTokens: 64,
  thinking: "on",
  systemMessage: "s",
} as const;

test("valid stored settings pass through unchanged", () => {
  const raw = {
    apiKey: "sk-1",
    theme: "dark",
    model: "fifi-8b",
    drawer: {
      temperature: 0.2,
      topP: 0.9,
      topK: 40,
      maxTokens: 512,
      thinking: "on",
      systemMessage: "be brief",
    },
  };
  deepStrictEqual(coerceSettings(raw), {
    apiKey: "sk-1",
    theme: "dark",
    model: "fifi-8b",
    drawer: {
      temperature: 0.2,
      topP: 0.9,
      topK: 40,
      maxTokens: 512,
      thinking: "on",
      systemMessage: "be brief",
    },
  });
});

test("a wrong-typed field falls back to its default", () => {
  const out = coerceSettings({
    apiKey: 42,
    theme: "blue",
    model: null,
    drawer: {
      temperature: "abc",
      topP: null,
      topK: "40",
      maxTokens: "2048",
      thinking: "maybe",
      systemMessage: 7,
    },
  });
  deepStrictEqual(out, defaultSettings);
});

test("missing or non-object input yields the defaults", () => {
  deepStrictEqual(coerceSettings(undefined), defaultSettings);
  deepStrictEqual(coerceSettings("junk"), defaultSettings);
  deepStrictEqual(coerceSettings({ drawer: "junk" }), defaultSettings);
});

test("null stored values yield the defaults instead of throwing", () => {
  deepStrictEqual(coerceSettings(null), defaultSettings);
  deepStrictEqual(coerceSettings({ drawer: null }), defaultSettings);
});

test("an array is never treated as a settings object", () => {
  const root = Object.assign(["junk"], {
    apiKey: "sk-1",
    drawer: { ...baseDrawer },
  });
  deepStrictEqual(coerceSettings(root), defaultSettings);
  const drawer = Object.assign([], {
    temperature: 0.5,
    topP: 0.5,
    topK: 8,
    maxTokens: 64,
  });
  deepStrictEqual(coerceSettings({ drawer }), defaultSettings);
});

test("out-of-contract drawer values fall back to the field default", () => {
  const cases: Array<{ key: "temperature" | "topP" | "topK" | "maxTokens"; value: unknown }> = [
    { key: "temperature", value: -1 },
    { key: "temperature", value: 2.5 },
    { key: "temperature", value: NaN },
    { key: "temperature", value: Infinity },
    { key: "temperature", value: -Infinity },
    { key: "temperature", value: null },
    { key: "temperature", value: [0.7] },
    { key: "temperature", value: "0.7" },
    { key: "topP", value: 1.5 },
    { key: "topP", value: -0.1 },
    { key: "topP", value: NaN },
    { key: "topP", value: Infinity },
    { key: "topP", value: null },
    { key: "topP", value: [1] },
    { key: "topP", value: "1" },
    { key: "topK", value: 1.5 },
    { key: "topK", value: -1 },
    { key: "topK", value: NaN },
    { key: "topK", value: Infinity },
    { key: "topK", value: null },
    { key: "topK", value: [8] },
    { key: "topK", value: "8" },
    { key: "maxTokens", value: 0 },
    { key: "maxTokens", value: 1.5 },
    { key: "maxTokens", value: -5 },
    { key: "maxTokens", value: NaN },
    { key: "maxTokens", value: Infinity },
    { key: "maxTokens", value: null },
    { key: "maxTokens", value: [64] },
    { key: "maxTokens", value: "64" },
  ];
  for (const c of cases) {
    const out = coerceSettings({ drawer: { ...baseDrawer, [c.key]: c.value } });
    deepStrictEqual(
      out.drawer,
      { ...baseDrawer, [c.key]: defaultSettings.drawer[c.key] },
      c.key + " <- " + String(c.value) + " must fall back to " + defaultSettings.drawer[c.key]
    );
  }
});

test("the exact contract boundary values are accepted", () => {
  const lower = coerceSettings({
    drawer: { ...baseDrawer, temperature: 0, topP: 0, topK: 0, maxTokens: 1 },
  });
  deepStrictEqual(lower.drawer, {
    ...baseDrawer,
    temperature: 0,
    topP: 0,
    topK: 0,
    maxTokens: 1,
  });
  const upper = coerceSettings({ drawer: { ...baseDrawer, temperature: 2, topP: 1 } });
  strictEqual(upper.drawer.temperature, 2);
  strictEqual(upper.drawer.topP, 1);
});
