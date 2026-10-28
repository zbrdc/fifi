/** Tests for the pure notice queue policy (node:test; no DOM at runtime). */

import { test } from "node:test";
import { deepStrictEqual } from "node:assert";
import { noticeList } from "./notices.js";

test("a repeated notice refreshes one banner instead of stacking", () => {
  deepStrictEqual(noticeList(["Could not save settings"], "Could not save settings"), [
    "Could not save settings",
  ]);
});

test("distinct notices queue newest last", () => {
  deepStrictEqual(noticeList(["a", "b"], "c"), ["a", "b", "c"]);
});

test("the queue drops the oldest beyond three banners", () => {
  deepStrictEqual(noticeList(["a", "b", "c"], "d"), ["b", "c", "d"]);
});

test("a refreshed notice moves last and still caps the queue", () => {
  deepStrictEqual(noticeList(["a", "b", "c"], "a"), ["b", "c", "a"]);
});
