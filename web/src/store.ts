/**
 * Persistence. Settings live in localStorage. Chats live in IndexedDB.
 * Views never touch localStorage or IndexedDB directly.
 */

import type { FinishReason, ThinkingMode, Usage } from "./api.js";

export interface DrawerSettings {
  temperature: number;
  topP: number;
  topK: number;
  maxTokens: number;
  thinking: ThinkingMode;
  systemMessage: string;
}

export type ThemeChoice = "system" | "light" | "dark";

export interface Settings {
  apiKey: string;
  theme: ThemeChoice;
  model: string;
  drawer: DrawerSettings;
}

export const defaultSettings: Settings = {
  apiKey: "",
  theme: "system",
  model: "",
  drawer: {
    temperature: 0.7,
    topP: 1,
    topK: 0,
    maxTokens: 2048,
    thinking: "default",
    systemMessage: "",
  },
};

const SETTINGS_KEY = "fifi-settings";

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw === null) return { ...defaultSettings };
    return coerceSettings(JSON.parse(raw));
  } catch {
    return { ...defaultSettings };
  }
}

/** Validate a persisted settings value field by field, falling back to
 *  the default wherever the stored value has the wrong type or sits
 *  outside the contract bounds of api/openapi.yaml ChatRequest. */
export function coerceSettings(raw: unknown): Settings {
  const root = objectOf(raw);
  const drawer = objectOf(root["drawer"]);
  return {
    apiKey: stringOf(root["apiKey"], defaultSettings.apiKey),
    theme: themeOf(root["theme"]),
    model: stringOf(root["model"], defaultSettings.model),
    drawer: {
      // temperature number in [0, 2]; top_p number in [0, 1];
      // top_k integer >= 0; max_tokens integer >= 1.
      temperature: boundedNumberOf(
        drawer["temperature"],
        defaultSettings.drawer.temperature,
        0,
        2
      ),
      topP: boundedNumberOf(drawer["topP"], defaultSettings.drawer.topP, 0, 1),
      topK: boundedIntegerOf(
        drawer["topK"],
        defaultSettings.drawer.topK,
        0,
        Number.POSITIVE_INFINITY
      ),
      maxTokens: boundedIntegerOf(
        drawer["maxTokens"],
        defaultSettings.drawer.maxTokens,
        1,
        Number.POSITIVE_INFINITY
      ),
      thinking: thinkingOf(drawer["thinking"]),
      systemMessage: stringOf(drawer["systemMessage"], defaultSettings.drawer.systemMessage),
    },
  };
}

function objectOf(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function stringOf(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

/** A finite number inside [min, max]; anything else falls back. */
function boundedNumberOf(value: unknown, fallback: number, min: number, max: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  return value < min || value > max ? fallback : value;
}

/** An integer inside [min, max]; non-integers fall back, never clamp. */
function boundedIntegerOf(value: unknown, fallback: number, min: number, max: number): number {
  if (typeof value !== "number" || !Number.isInteger(value)) return fallback;
  return value < min || value > max ? fallback : value;
}

function themeOf(value: unknown): ThemeChoice {
  return value === "light" || value === "dark" ? value : "system";
}

function thinkingOf(value: unknown): ThinkingMode {
  return value === "on" || value === "off" ? value : "default";
}

type SettingsListener = (s: Settings) => void;
const listeners = new Set<SettingsListener>();

/** Listener for settings persistence failures (for example a quota error). */
type StorageErrorListener = (message: string) => void;
const errorListeners = new Set<StorageErrorListener>();

export function saveSettings(next: Settings): void {
  let failure: unknown = null;
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(next));
  } catch (err) {
    failure = err;
  }
  for (const cb of listeners) cb(next);
  if (failure !== null) {
    const reason = failure instanceof Error ? failure.message : String(failure);
    for (const cb of errorListeners) cb("Could not save settings: " + reason);
  }
}

export function onSettingsChange(cb: SettingsListener): void {
  listeners.add(cb);
}

/** Subscribe to saveSettings persistence failures. */
export function onStorageError(cb: StorageErrorListener): void {
  errorListeners.add(cb);
}

/** One message as the interface holds it. */
export interface UiToolCall {
  name: string;
  arguments: string;
}

export type UiMessageState = "complete" | "streaming" | "stopped" | "error";

export interface UiMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  toolCalls?: UiToolCall[];
  state: UiMessageState;
  errorMessage?: string;
  finishReason?: FinishReason;
  usage?: Usage;
  durationMs?: number;
  createdAt: number;
}

export interface ChatRecord {
  id: string;
  title: string;
  model: string;
  createdAt: number;
  updatedAt: number;
  messages: UiMessage[];
}

const DB_NAME = "fifi-web";
const DB_VERSION = 1;
const STORE = "chats";

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (dbPromise === null) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE)) {
          const store = db.createObjectStore(STORE, { keyPath: "id" });
          store.createIndex("updatedAt", "updatedAt");
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error ?? new Error("IndexedDB open failed"));
    });
  }
  return dbPromise;
}

function request<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDb().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const tx = db.transaction(STORE, mode);
        const req = run(tx.objectStore(STORE));
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error ?? new Error("IndexedDB request failed"));
      })
  );
}

export function listChats(): Promise<ChatRecord[]> {
  return request<ChatRecord[]>("readonly", (s) => s.getAll()).then((rows) =>
    rows.sort((a, b) => b.updatedAt - a.updatedAt)
  );
}

export function getChat(id: string): Promise<ChatRecord | undefined> {
  return request<ChatRecord | undefined>("readonly", (s) => s.get(id));
}

export function putChat(chat: ChatRecord): Promise<ChatRecord> {
  return request<IDBValidKey>("readwrite", (s) => s.put(chat)).then(() => chat);
}

export function deleteChat(id: string): Promise<void> {
  return request<undefined>("readwrite", (s) => s.delete(id)).then(() => undefined);
}

export function newChatRecord(model: string): ChatRecord {
  const now = Date.now();
  return {
    id: crypto.randomUUID(),
    title: "New chat",
    model,
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

export function newUiMessage(role: "user" | "assistant", content: string): UiMessage {
  return { id: crypto.randomUUID(), role, content, state: "complete", createdAt: Date.now() };
}

/** Short title from the first user message. */
export function titleFromText(text: string): string {
  const clean = text.trim().replace(/\s+/g, " ");
  return clean.length <= 42 ? clean : clean.slice(0, 41) + "\u2026";
}
