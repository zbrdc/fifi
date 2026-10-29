/**
 * Boot, view routing, and event wiring. Views own their content; this
 * module owns the shell: routing, the status pill, the model picker, the
 * chat list.
 */

import { ApiError, getHealth, getReady, listModels, type Model } from "./api";
import { AboutView, applyTheme } from "./about";
import { ChatView } from "./chat";
import { Drawer } from "./drawer";
import { MonitorView } from "./monitor";
import { noticeList } from "./notices";
import { pollPlan } from "./panel";
import {
  deleteChat,
  listChats,
  loadSettings,
  onSettingsChange,
  onStorageError,
  saveSettings,
  type ChatRecord,
} from "./store";

type ViewName = "chat" | "monitor" | "about";
type PillState = "offline" | "starting" | "ready" | "busy" | "error";

const pillLabels: Record<PillState, string> = {
  offline: "Offline",
  starting: "Starting",
  ready: "Ready",
  busy: "Busy",
  error: "Error",
};

function el<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (found === null) throw new Error("Missing element #" + id);
  return found as T;
}


function announce(text: string): void {
  const region = document.getElementById("live-region");
  if (region !== null) region.textContent = text;
}

function main(): void {
  applyTheme(loadSettings().theme);

  const sidebar = el<HTMLElement>("sidebar");
  const sidebarToggle = el<HTMLButtonElement>("sidebar-toggle");
  const sidebarBackdrop = el<HTMLElement>("sidebar-backdrop");
  const chatList = el<HTMLUListElement>("chat-list");
  const newChatButton = el<HTMLButtonElement>("new-chat");
  const viewTitle = el<HTMLElement>("view-title");
  const modelPicker = el<HTMLSelectElement>("model-picker");
  const pill = el<HTMLElement>("status-pill");
  const pillLabel = el<HTMLElement>("status-label");

  const about = new AboutView(el<HTMLElement>("view-about"));
  const monitor = new MonitorView(el<HTMLElement>("view-monitor"));
  const drawer = new Drawer(el<HTMLElement>("drawer-root"));

  /* Dismissible notices for storage failures */

  const noticeHost = document.createElement("div");
  noticeHost.className = "notice-host";
  document.body.append(noticeHost);

  // One banner per distinct text, newest last, at most three at a time.
  let noticeTexts: string[] = [];
  const noticeBanners = new Map<string, HTMLElement>();

  function dismissNotice(text: string): void {
    noticeTexts = noticeTexts.filter((t) => t !== text);
    renderNotices();
  }

  function buildNoticeBanner(text: string): HTMLElement {
    const banner = document.createElement("div");
    banner.className = "notice-banner";
    const msg = document.createElement("span");
    msg.className = "notice-text";
    msg.textContent = text;
    const dismiss = document.createElement("button");
    dismiss.type = "button";
    dismiss.className = "notice-dismiss";
    dismiss.textContent = "Dismiss";
    dismiss.addEventListener("click", () => dismissNotice(text));
    banner.append(msg, dismiss);
    return banner;
  }

  function renderNotices(): void {
    for (const [text, banner] of noticeBanners) {
      if (!noticeTexts.includes(text)) {
        banner.remove();
        noticeBanners.delete(text);
      }
    }
    noticeHost.textContent = "";
    for (const text of noticeTexts) {
      let banner = noticeBanners.get(text);
      if (banner === undefined) {
        banner = buildNoticeBanner(text);
        noticeBanners.set(text, banner);
      }
      noticeHost.append(banner);
    }
  }

  function showNotice(text: string): void {
    noticeTexts = noticeList(noticeTexts, text);
    renderNotices();
  }

  onStorageError((message) => showNotice(message));

  let streaming = false;
  const chat = new ChatView(
    el<HTMLElement>("chat-column"),
    el<HTMLElement>("chat-scroll"),
    el<HTMLTextAreaElement>("composer-input"),
    el<HTMLButtonElement>("send-button"),
    el<HTMLElement>("send-icon"),
    el<HTMLElement>("stop-icon"),
    {
      onBusyChange: (busy) => {
        streaming = busy;
        if (busy) setPill("busy");
        else setPill(serverReady ? "ready" : serverReachable ? "starting" : "offline");
      },
      onUnauthorized: () => {
        showView("about");
        about.focusApiKey("The server rejected the API key. Enter the correct key, then send again.");
      },
      onChatsChanged: () => void renderChatList(),
      onNotice: (text) => showNotice(text),
      onAnnounce: (text) => announce(text),
    }
  );

  /* Status pill */

  let serverReachable = false;
  let serverReady = false;
  let currentPill: PillState = "offline";

  function setPill(state: PillState): void {
    if (state === currentPill) return;
    currentPill = state;
    pill.className = "pill pill-" + state;
    pillLabel.textContent = pillLabels[state];
    announce(pillLabels[state]);
  }

  /* Models and picker */

  let models: Model[] | null = null;
  let modelsUnauthorized = false;
  /** The view the user is on; the poll refreshes the Monitor only when
   * it is on screen (the live speed panel). */
  let currentView: ViewName = "chat";

  function renderPicker(): void {
    const settings = loadSettings();
    modelPicker.textContent = "";
    if (models === null || models.length === 0) {
      const option = document.createElement("option");
      option.textContent = modelsUnauthorized ? "API key needed" : "No models";
      option.value = "";
      modelPicker.append(option);
      modelPicker.disabled = true;
      return;
    }
    modelPicker.disabled = false;
    for (const model of models) {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = model.id;
      if (model.id === settings.model) option.selected = true;
      modelPicker.append(option);
    }
    if (settings.model === "" || !models.some((m) => m.id === settings.model)) {
      const first = models[0];
      if (first !== undefined) {
        modelPicker.value = first.id;
        saveSettings({ ...loadSettings(), model: first.id });
      }
    }
  }

  modelPicker.addEventListener("change", () => {
    saveSettings({ ...loadSettings(), model: modelPicker.value });
  });

  async function refreshModels(): Promise<void> {
    if (loadSettings().apiKey === "") {
      models = [];
      modelsUnauthorized = false;
      renderPicker();
      about.updateModels([]);
      return;
    }
    try {
      // Bound the poll: a hanging /v1/models must not wedge pollOutstanding.
      const list = await listModels(AbortSignal.timeout(4000));
      models = list.data;
      modelsUnauthorized = false;
    } catch (err) {
      if (err instanceof ApiError && err.code === "unauthorized") {
        modelsUnauthorized = true;
        renderPicker();
        return;
      }
      return; // Keep the last known list on a transient failure.
    }
    renderPicker();
    about.updateModels(models);
  }

  /* Health and readiness poll */

  let pollOutstanding = false;

  async function poll(): Promise<void> {
    if (pollOutstanding) return; // Skip a tick while a poll is in flight.
    pollOutstanding = true;
    try {
      // The live speed panel: whenever the Monitor view is on
      // screen — server up, starting, or down — the poll calls the full
      // guarded refresh, so health, readiness, models, and the ring move
      // under one truth and one timestamp; a dead engine can never keep
      // last-good "runs"/"ready" next to a current time. The decision
      // lives in the DOM-free panel module (pollPlan) and is table-tested
      // there, offline row included. The extra fetches per tick are cheap;
      // correctness beats the saving.
      if (
        pollPlan({
          monitorVisible: currentView === "monitor",
          lastKnownServerReady: serverReady,
        }).refreshMonitor
      ) {
        await monitor.refresh();
      }
      try {
        await getHealth(AbortSignal.timeout(4000));
        serverReachable = true;
      } catch {
        serverReachable = false;
      }
      if (!serverReachable) {
        serverReady = false;
        chat.setOffline(true);
        if (!streaming) setPill("offline");
        return;
      }
      chat.setOffline(false);
      try {
        await getReady(AbortSignal.timeout(4000));
        serverReady = true;
      } catch {
        serverReady = false;
      }
      if (!streaming) setPill(serverReady ? "ready" : "starting");
      if (serverReady) await refreshModels();
    } finally {
      pollOutstanding = false;
    }
  }

  /* Chat list */

  async function renderChatList(): Promise<void> {
    let chats: ChatRecord[];
    try {
      chats = await listChats();
    } catch {
      chatList.textContent = "";
      const item = document.createElement("li");
      item.className = "chat-empty";
      item.textContent = "Chat storage is unavailable.";
      chatList.append(item);
      return;
    }
    chatList.textContent = "";
    const currentId = chat.currentChatId();
    if (chats.length === 0) {
      const item = document.createElement("li");
      item.className = "chat-empty";
      item.textContent = "No chats yet.";
      chatList.append(item);
      return;
    }
    for (const record of chats) {
      const item = document.createElement("li");
      item.className = "chat-item";
      if (record.id === currentId) item.setAttribute("aria-current", "true");
      const open = document.createElement("button");
      open.type = "button";
      open.className = "chat-open";
      open.textContent = record.title;
      open.addEventListener("click", () => {
        chat.openChat(record.id).catch(() => {
          // openChat reports read failures through its own notice hook.
        });
        closeSidebar();
      });
      const del = document.createElement("button");
      del.type = "button";
      del.className = "chat-delete";
      del.setAttribute("aria-label", "Delete chat: " + record.title);
      del.textContent = "\u2715";
      del.addEventListener("click", async () => {
        chat.abort();
        chat.forgetChat(record.id);
        try {
          await deleteChat(record.id);
        } catch (err) {
          // The chat still exists; lift the dead mark so replies keep saving.
          chat.unforgetChat(record.id);
          showNotice(
            "Could not delete this chat: " +
              (err instanceof Error ? err.message : String(err))
          );
          return;
        }
        if (chat.currentChatId() === record.id) chat.newChat();
        await renderChatList();
      });
      item.append(open, del);
      chatList.append(item);
    }
  }

  newChatButton.addEventListener("click", () => {
    chat.newChat();
    void renderChatList();
    closeSidebar();
  });

  /* Routing */

  const views: Record<ViewName, HTMLElement> = {
    chat: el<HTMLElement>("view-chat"),
    monitor: el<HTMLElement>("view-monitor"),
    about: el<HTMLElement>("view-about"),
  };

  function showView(name: ViewName): void {
    currentView = name;
    for (const [key, section] of Object.entries(views) as Array<[ViewName, HTMLElement]>) {
      section.hidden = key !== name;
    }
    for (const link of document.querySelectorAll<HTMLAnchorElement>(".nav a")) {
      if (link.dataset.view === name) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    }
    viewTitle.textContent = name === "chat" ? "Chat" : name === "monitor" ? "Monitor" : "About";
    const drawerButton = document.getElementById("drawer-button");
    if (drawerButton !== null) drawerButton.hidden = name !== "chat";
    if (name === "monitor") void monitor.refresh();
    if (name === "about") about.updateModels(models);
    if (name === "chat") el<HTMLTextAreaElement>("composer-input").focus();
  }

  function routeFromHash(): ViewName {
    const hash = window.location.hash.replace(/^#\/?"?/, "");
    if (hash === "monitor" || hash === "about") return hash;
    return "chat";
  }

  window.addEventListener("hashchange", () => showView(routeFromHash()));

  /* Drawer */

  el<HTMLButtonElement>("drawer-button").addEventListener("click", () => drawer.open());
  document.addEventListener("keydown", (e) => drawer.handleKeydown(e));

  /* Sidebar on narrow screens */

  function closeSidebar(): void {
    document.body.classList.remove("sidebar-open");
    sidebarBackdrop.hidden = true;
    sidebarToggle.setAttribute("aria-expanded", "false");
  }

  sidebarToggle.addEventListener("click", () => {
    const open = !document.body.classList.contains("sidebar-open");
    document.body.classList.toggle("sidebar-open", open);
    sidebarBackdrop.hidden = !open;
    sidebarToggle.setAttribute("aria-expanded", String(open));
  });
  sidebarBackdrop.addEventListener("click", closeSidebar);

  /* Settings changes propagate to the picker and theme. */

  onSettingsChange((settings) => {
    applyTheme(settings.theme);
  });

  /* Boot */

  showView(routeFromHash());
  void renderChatList();
  void poll();
  window.setInterval(() => void poll(), 5000);

  void sidebar;
}

main();
