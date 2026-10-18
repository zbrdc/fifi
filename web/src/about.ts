/**
 * About view: connection facts, server facts, and local settings.
 * The API key lives in this browser only.
 */

import type { Model } from "./api";
import { loadSettings, saveSettings, type ThemeChoice } from "./store";

export class AboutView {
  private readonly section: HTMLElement;
  private keyInput: HTMLInputElement | null = null;
  private keyReason: HTMLElement | null = null;
  private serverFacts: HTMLElement | null = null;

  constructor(section: HTMLElement) {
    this.section = section;
    this.render();
  }

  focusApiKey(reason: string): void {
    this.keyReason !== null ? (this.keyReason.textContent = reason) : undefined;
    this.keyInput?.focus();
  }

  updateModels(models: Model[] | null): void {
    if (this.serverFacts === null) return;
    this.serverFacts.textContent = "";
    if (models === null) {
      this.serverFacts.textContent = "No model list yet.";
      return;
    }
    if (models.length === 0) {
      this.serverFacts.textContent = "The server lists no models.";
      return;
    }
    const ul = document.createElement("ul");
    ul.style.margin = "0";
    ul.style.paddingLeft = "18px";
    for (const model of models) {
      const li = document.createElement("li");
      li.textContent = model.id + " \u00b7 " + (model.engines?.length ?? 0) + " engine(s)";
      ul.append(li);
    }
    this.serverFacts.append(ul);
  }

  private render(): void {
    this.section.textContent = "";
    const scroll = document.createElement("div");
    scroll.className = "panel-scroll";
    const column = document.createElement("div");
    column.className = "panel-column";
    const head = document.createElement("div");
    head.className = "panel-head";
    const h2 = document.createElement("h2");
    h2.textContent = "About";
    head.append(h2);
    column.append(head);

    const intro = document.createElement("p");
    intro.className = "sub";
    intro.textContent =
      "fifi serves one person at one machine. The model runs here. Nothing leaves it.";
    column.append(intro);

    const { card: connCard, body: connBody } = this.card("Connection");
    const grid = document.createElement("dl");
    grid.className = "fact-grid";
    const origin = window.location.origin;
    const routes: Array<[string, string]> = [
      ["Origin", origin],
      ["Health", origin + "/health"],
      ["Ready", origin + "/ready"],
      ["Models", origin + "/v1/models"],
      ["Chat", origin + "/v1/chat/completions"],
    ];
    for (const [name, url] of routes) {
      const dt = document.createElement("dt");
      dt.textContent = name;
      const dd = document.createElement("dd");
      const code = document.createElement("span");
      code.className = "mono";
      code.textContent = url;
      dd.append(code);
      grid.append(dt, dd);
    }
    connBody.append(grid);
    column.append(connCard);

    const { card: modelCard, body: modelBody } = this.card("Server");
    this.serverFacts = modelBody;
    modelBody.textContent = "No model list yet.";
    column.append(modelCard);

    const { card: settingsCard, body: settingsBody } = this.card("Settings");
    this.buildSettings(settingsBody);
    column.append(settingsCard);

    scroll.append(column);
    this.section.append(scroll);
  }

  private card(title: string): { card: HTMLElement; body: HTMLElement } {
    const card = document.createElement("div");
    card.className = "card";
    const h3 = document.createElement("h3");
    h3.textContent = title;
    const body = document.createElement("div");
    card.append(h3, body);
    return { card, body };
  }

  private buildSettings(body: HTMLElement): void {
    const reason = document.createElement("p");
    reason.className = "notice";
    reason.hidden = true;
    this.keyReason = reason;

    const keyLabel = document.createElement("label");
    keyLabel.className = "field-label";
    keyLabel.textContent = "API key";
    const keyRow = document.createElement("div");
    keyRow.className = "key-row";
    const input = document.createElement("input");
    input.type = "password";
    input.className = "text-field";
    input.id = "about-api-key";
    keyLabel.htmlFor = input.id;
    input.placeholder = "sk-\u2026";
    input.value = loadSettings().apiKey;
    this.keyInput = input;
    const show = document.createElement("button");
    show.className = "btn";
    show.type = "button";
    show.textContent = "Show";
    show.setAttribute("aria-pressed", "false");
    show.addEventListener("click", () => {
      const reveal = input.type === "password";
      input.type = reveal ? "text" : "password";
      show.textContent = reveal ? "Hide" : "Show";
      show.setAttribute("aria-pressed", reveal ? "true" : "false");
    });
    const save = document.createElement("button");
    save.className = "btn btn-primary";
    save.type = "button";
    save.textContent = "Save";
    save.addEventListener("click", () => {
      const settings = loadSettings();
      saveSettings({ ...settings, apiKey: input.value.trim() });
      reason.hidden = false;
      reason.textContent = "Key saved in this browser.";
    });
    keyRow.append(input, show, save);
    const keyNote = document.createElement("p");
    keyNote.className = "sub";
    keyNote.textContent = "The key lives in this browser only. It is sent as a bearer token to this origin.";

    const themeLabel = document.createElement("label");
    themeLabel.className = "field-label";
    themeLabel.textContent = "Appearance";
    const segmented = document.createElement("div");
    segmented.className = "segmented";
    segmented.setAttribute("role", "group");
    segmented.setAttribute("aria-label", "Appearance");
    const buttons = new Map<ThemeChoice, HTMLButtonElement>();
    for (const choice of ["system", "light", "dark"] as const) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = choice === "system" ? "System" : choice === "light" ? "Light" : "Dark";
      b.addEventListener("click", () => {
        const settings = loadSettings();
        saveSettings({ ...settings, theme: choice });
        applyTheme(choice);
        syncPressed();
      });
      buttons.set(choice, b);
      segmented.append(b);
    }
    const syncPressed = () => {
      const current = loadSettings().theme;
      for (const [choice, b] of buttons) {
        b.setAttribute("aria-pressed", choice === current ? "true" : "false");
      }
    };
    syncPressed();
    body.append(reason, keyLabel, keyRow, keyNote, themeLabel, segmented);
  }
}

/** Apply a theme choice to the document root. */
export function applyTheme(choice: ThemeChoice): void {
  if (choice === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = choice;
}
