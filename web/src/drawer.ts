/**
 * Sampling and session settings drawer. Slides over a dimmed background,
 * traps focus, and returns focus on close. Values persist through store.ts.
 */

import { defaultSettings, loadSettings, saveSettings, type DrawerSettings } from "./store";

export class Drawer {
  private readonly backdrop: HTMLElement;
  private readonly panel: HTMLElement;
  private lastFocus: HTMLElement | null = null;

  constructor(root: HTMLElement) {
    this.backdrop = document.createElement("div");
    this.backdrop.className = "drawer-backdrop";
    this.panel = document.createElement("div");
    this.panel.className = "drawer";
    this.panel.setAttribute("role", "dialog");
    this.panel.setAttribute("aria-modal", "true");
    this.panel.setAttribute("aria-label", "Sampling settings");
    root.append(this.backdrop, this.panel);
    this.build();
    this.backdrop.addEventListener("click", () => this.close());
  }

  open(): void {
    this.lastFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    this.syncFields();
    this.backdrop.classList.add("open");
    this.panel.classList.add("open");
    const first = this.panel.querySelector<HTMLElement>("input, select, textarea, button");
    first?.focus();
  }

  close(): void {
    this.backdrop.classList.remove("open");
    this.panel.classList.remove("open");
    this.lastFocus?.focus();
    this.lastFocus = null;
  }

  private get isOpen(): boolean {
    return this.panel.classList.contains("open");
  }

  handleKeydown(e: KeyboardEvent): void {
    if (!this.isOpen) return;
    if (e.key === "Escape") {
      e.preventDefault();
      this.close();
      return;
    }
    if (e.key !== "Tab") return;
    type Focusable = HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement | HTMLButtonElement;
    const focusables = Array.from(
      this.panel.querySelectorAll<Focusable>("input, select, textarea, button")
    ).filter((el) => !el.disabled);
    if (focusables.length === 0) return;
    const first = focusables[0]!;
    const last = focusables[focusables.length - 1]!;
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }

  private update(patch: Partial<DrawerSettings>): void {
    const settings = loadSettings();
    saveSettings({ ...settings, drawer: { ...settings.drawer, ...patch } });
  }

  private range(
    label: string,
    key: "temperature" | "topP",
    min: number,
    max: number,
    step: number
  ): HTMLElement {
    const row = document.createElement("div");
    const labelEl = document.createElement("label");
    labelEl.className = "field-label";
    labelEl.textContent = label;
    const wrap = document.createElement("div");
    wrap.className = "range-row";
    const input = document.createElement("input");
    input.type = "range";
    input.id = "drawer-" + key;
    labelEl.htmlFor = input.id;
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    const value = document.createElement("span");
    value.className = "range-value";
    const sync = () => {
      input.value = String(loadSettings().drawer[key]);
      value.textContent = input.value;
    };
    this.syncOnOpen.set(key, sync);
    input.addEventListener("input", () => {
      value.textContent = input.value;
      this.update({ [key]: Number(input.value) } as Partial<DrawerSettings>);
    });
    sync();
    wrap.append(input, value);
    row.append(labelEl, wrap);
    return row;
  }

  private number(
    label: string,
    key: "topK" | "maxTokens",
    min: number,
    placeholder: string
  ): HTMLElement {
    const row = document.createElement("div");
    const labelEl = document.createElement("label");
    labelEl.className = "field-label";
    labelEl.textContent = label;
    const input = document.createElement("input");
    input.type = "number";
    input.id = "drawer-" + key;
    labelEl.htmlFor = input.id;
    input.className = "number-field";
    input.min = String(min);
    input.placeholder = placeholder;
    input.addEventListener("change", () => {
      const parsed = Number(input.value);
      if (Number.isFinite(parsed) && parsed >= min) {
        this.update({ [key]: Math.floor(parsed) } as Partial<DrawerSettings>);
      } else {
        input.value = String(loadSettings().drawer[key]);
      }
    });
    this.syncOnOpen.set(key, () => (input.value = String(loadSettings().drawer[key])));
    row.append(labelEl, input);
    return row;
  }

  private readonly syncOnOpen = new Map<string, () => void>();

  private build(): void {
    const head = document.createElement("div");
    head.className = "drawer-head";
    const h2 = document.createElement("h2");
    h2.textContent = "Sampling";
    const close = document.createElement("button");
    close.className = "btn-icon";
    close.type = "button";
    close.setAttribute("aria-label", "Close settings");
    close.textContent = "\u2715";
    close.addEventListener("click", () => this.close());
    head.append(h2, close);

    const thinkingRow = document.createElement("div");
    const thinkingLabel = document.createElement("label");
    thinkingLabel.className = "field-label";
    thinkingLabel.textContent = "Thinking";
    const thinkingSelect = document.createElement("select");
    thinkingSelect.className = "select-field";
    thinkingSelect.id = "drawer-thinking";
    thinkingLabel.htmlFor = thinkingSelect.id;
    for (const mode of ["default", "on", "off"] as const) {
      const option = document.createElement("option");
      option.value = mode;
      option.textContent = mode === "default" ? "default (model choice)" : mode;
      thinkingSelect.append(option);
    }
    thinkingSelect.addEventListener("change", () => {
      this.update({ thinking: thinkingSelect.value as DrawerSettings["thinking"] });
    });
    this.syncOnOpen.set("thinking", () => (thinkingSelect.value = loadSettings().drawer.thinking));
    thinkingRow.append(thinkingLabel, thinkingSelect);

    const systemRow = document.createElement("div");
    const systemLabel = document.createElement("label");
    systemLabel.className = "field-label";
    systemLabel.textContent = "System message";
    const systemArea = document.createElement("textarea");
    systemArea.className = "text-field";
    systemArea.id = "drawer-system";
    systemLabel.htmlFor = systemArea.id;
    systemArea.placeholder = "Sent once, before the conversation";
    systemArea.addEventListener("change", () => {
      this.update({ systemMessage: systemArea.value });
    });
    this.syncOnOpen.set("systemMessage", () => (systemArea.value = loadSettings().drawer.systemMessage));
    systemRow.append(systemLabel, systemArea);

    const actions = document.createElement("div");
    actions.className = "drawer-actions";
    const reset = document.createElement("button");
    reset.className = "btn";
    reset.type = "button";
    reset.textContent = "Reset";
    reset.addEventListener("click", () => {
      const settings = loadSettings();
      saveSettings({ ...settings, drawer: { ...defaultSettings.drawer } });
      this.syncFields();
    });
    actions.append(reset);

    this.panel.append(
      head,
      this.range("Temperature", "temperature", 0, 2, 0.1),
      this.range("Top p", "topP", 0, 1, 0.01),
      this.number("Top k (0 is unset)", "topK", 0, "0"),
      this.number("Max tokens", "maxTokens", 1, "2048"),
      thinkingRow,
      systemRow,
      actions
    );
  }

  private syncFields(): void {
    for (const sync of this.syncOnOpen.values()) sync();
  }
}
