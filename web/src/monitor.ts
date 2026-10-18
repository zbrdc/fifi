/**
 * Monitor view: engine state and models from the shipped routes.
 * Unshipped routes hold no section here (design brief, S0).
 */

import { ApiError, getHealth, getReady, listModels, type Model } from "./api";
import { formatBytes, formatClock, formatDate, formatTokens } from "./format";

export interface MonitorData {
  healthOk: boolean;
  readyOk: boolean;
  notReadyCode: string | null;
  models: Model[] | null;
  modelsError: string | null;
  checkedAt: number;
}

export class MonitorView {
  private readonly section: HTMLElement;
  private data: MonitorData = {
    healthOk: false,
    readyOk: false,
    notReadyCode: null,
    models: null,
    modelsError: null,
    checkedAt: 0,
  };

  constructor(section: HTMLElement) {
    this.section = section;
    this.render();
  }

  /** One probe of the public routes plus the models route. */
  async refresh(): Promise<MonitorData> {
    const checkedAt = Date.now();
    let healthOk = false;
    try {
      await getHealth();
      healthOk = true;
    } catch {
      healthOk = false;
    }
    let readyOk = false;
    let notReadyCode: string | null = null;
    try {
      await getReady();
      readyOk = true;
    } catch (err) {
      // The contract gives /ready exactly two answers: 200 ready and 503
      // not_ready. Anything else, including an unreachable server, is a
      // non-answer and must not display a failure code.
      notReadyCode = err instanceof ApiError && err.code === "not_ready" ? "not_ready" : null;
    }
    let models: Model[] | null = null;
    let modelsError: string | null = null;
    if (healthOk) {
      try {
        models = (await listModels()).data;
      } catch (err) {
        modelsError = err instanceof ApiError ? err.code : "network";
      }
    }
    this.data = { healthOk, readyOk, notReadyCode, models, modelsError, checkedAt };
    this.render();
    return this.data;
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

  private fact(body: HTMLElement, name: string, value: string): void {
    const dt = document.createElement("dt");
    dt.textContent = name;
    const dd = document.createElement("dd");
    dd.textContent = value;
    const grid = body.querySelector("dl") ?? document.createElement("dl");
    if (!grid.isConnected) {
      grid.className = "fact-grid";
      body.append(grid);
    }
    grid.append(dt, dd);
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
    h2.textContent = "Monitor";
    const refresh = document.createElement("button");
    refresh.className = "btn";
    refresh.type = "button";
    refresh.textContent = "Refresh";
    refresh.addEventListener("click", () => void this.refresh());
    head.append(h2, refresh);
    column.append(head);

    const { card: healthCard, body: healthBody } = this.card("Health");
    this.fact(healthBody, "Process", this.data.healthOk ? "runs" : "no answer");
    this.fact(healthBody, "Checked", this.data.checkedAt === 0 ? "\u2014" : formatClock(this.data.checkedAt));
    column.append(healthCard);

    const { card: readyCard, body: readyBody } = this.card("Readiness");
    this.fact(
      readyBody,
      "State",
      this.data.readyOk ? "ready" : this.data.notReadyCode ?? "no answer"
    );
    column.append(readyCard);

    const { card: modelsCard, body: modelsBody } = this.card("Models");
    if (this.data.modelsError !== null) {
      const hint = document.createElement("p");
      hint.className = "panel-hint";
      hint.textContent =
        this.data.modelsError === "unauthorized"
          ? "Set the API key in About to see the engines."
          : "The models route failed: " + this.data.modelsError;
      modelsBody.append(hint);
    } else if (this.data.models === null) {
      const hint = document.createElement("p");
      hint.className = "panel-hint";
      hint.textContent = "No answer from the server yet.";
      modelsBody.append(hint);
    } else if (this.data.models.length === 0) {
      const hint = document.createElement("p");
      hint.className = "panel-hint";
      hint.textContent = "The server lists no models.";
      modelsBody.append(hint);
    } else {
      for (const model of this.data.models) this.renderModel(modelsBody, model);
    }
    column.append(modelsCard);

    const privacy = document.createElement("p");
    privacy.className = "sub";
    privacy.textContent = "This view shows only what the shipped routes report.";
    column.append(privacy);

    scroll.append(column);
    this.section.append(scroll);
  }

  private renderModel(body: HTMLElement, model: Model): void {
    const block = document.createElement("div");
    block.className = "model-block";
    const name = document.createElement("div");
    name.className = "mono";
    name.textContent = model.id;
    const sub = document.createElement("div");
    sub.className = "cache-facts";
    sub.textContent =
      "served since " + formatDate(model.created * 1000) +
      (model.owned_by !== "" ? " \u00b7 " + model.owned_by : "");
    block.append(name, sub);

    const engines = model.engines ?? [];
    if (engines.length === 0) {
      const none = document.createElement("div");
      none.className = "cache-facts";
      none.textContent = "no engines hold this model";
      block.append(none);
    }
    for (const engine of engines) {
      const row = document.createElement("div");
      row.className = "engine-row";
      const id = document.createElement("span");
      id.className = "engine-id";
      id.textContent = engine.id;
      const chip = document.createElement("span");
      chip.className = "state-chip s-" + engine.state;
      chip.textContent = engine.state;
      row.append(id, chip);
      block.append(row);
      if (engine.cache !== undefined) {
        const facts = document.createElement("div");
        facts.className = "cache-facts";
        facts.textContent =
          "KV " + formatBytes(engine.cache.kv_bytes) +
          " \u00b7 prefix hits " + formatTokens(engine.cache.prefix_hits) +
          " \u00b7 misses " + formatTokens(engine.cache.prefix_misses) +
          (engine.cache.snapshot_bytes !== undefined
            ? " \u00b7 snapshots " + formatBytes(engine.cache.snapshot_bytes)
            : "");
        block.append(facts);
      }
    }
    body.append(block);
  }
}
