/**
 * Chat view: messages, composer, streaming render. Streaming text is
 * appended as text nodes through requestAnimationFrame. The conversation
 * is never re-rendered per token.
 */

import {
  ApiError,
  chatStream,
  describeError,
  type ChatRequestBody,
  type ToolCallDelta,
  type Usage,
} from "./api";
import { buildChatBody, buildMessages } from "./history";
import {
  deleteChat,
  getChat,
  loadSettings,
  newChatRecord,
  newUiMessage,
  putChat,
  type ChatRecord,
  type UiMessage,
} from "./store";
import { formatDuration, formatSpeed, formatTokens } from "./format";
import { DeadChats } from "./deadchats";
import { SendGate } from "./sendgate";

export interface ChatHooks {
  onBusyChange(busy: boolean): void;
  onUnauthorized(): void;
  onChatsChanged(): void;
  onNotice(text: string): void;
  /** One polite announcement per stream lifecycle event, never per token. */
  onAnnounce(text: string): void;
}

interface StreamDom {
  root: HTMLElement;
  waiting: HTMLElement;
  thinking: HTMLDetailsElement | null;
  thinkingText: Text | null;
  text: Text;
  tools: HTMLElement | null;
  meta: HTMLElement;
}

export class ChatView {
  private readonly column: HTMLElement;
  private readonly scroll: HTMLElement;
  private readonly input: HTMLTextAreaElement;
  private readonly sendButton: HTMLButtonElement;
  private readonly sendIcon: HTMLElement;
  private readonly stopIcon: HTMLElement;
  private readonly hooks: ChatHooks;

  private chat: ChatRecord | null = null;
  private controller: AbortController | null = null;
  private readonly sendGate = new SendGate();
  private offline = false;
  private readonly deadChats = new DeadChats();

  // Stream buffers flushed on an animation frame.
  private pendingContent = "";
  private pendingReasoning = "";
  private flushHandle = 0;
  private streamDom: StreamDom | null = null;
  private startedAt = 0;
  private streamChat: ChatRecord | null = null;
  private streamMessage: UiMessage | null = null;
  private toolCalls = new Map<number, { name: string; args: string }>();

  constructor(
    column: HTMLElement,
    scroll: HTMLElement,
    input: HTMLTextAreaElement,
    sendButton: HTMLButtonElement,
    sendIcon: HTMLElement,
    stopIcon: HTMLElement,
    hooks: ChatHooks
  ) {
    this.column = column;
    this.scroll = scroll;
    this.input = input;
    this.sendButton = sendButton;
    this.sendIcon = sendIcon;
    this.stopIcon = stopIcon;
    this.hooks = hooks;

    this.input.addEventListener("input", () => this.autosize());
    this.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        void this.send();
      }
    });
    this.sendButton.addEventListener("click", () => {
      if (this.controller !== null) this.controller.abort();
      else void this.send();
    });
    this.renderEmpty();
  }

  setOffline(offline: boolean): void {
    this.offline = offline;
    this.updateSendEnabled();
    if (offline) this.input.placeholder = "The server is unreachable. Sending is disabled.";
    else this.input.placeholder = "Message the model";
  }

  private updateSendEnabled(): void {
    const hasText = this.input.value.trim() !== "";
    const streaming = this.controller !== null;
    this.sendButton.disabled = !streaming && (!hasText || this.offline);
    this.sendIcon.hidden = streaming;
    this.stopIcon.hidden = !streaming;
    this.sendButton.setAttribute("aria-label", streaming ? "Stop" : "Send");
  }

  private autosize(): void {
    this.input.style.height = "auto";
    this.input.style.height = Math.min(
      this.input.scrollHeight,
      5 * 24 + 18
    ) + "px";
    this.updateSendEnabled();
  }

  /** Abort the in-flight stream, if any. The partial reply is kept. */
  abort(): void {
    if (this.controller !== null) this.controller.abort();
  }

  /** Mark a chat id as deleted so a finishing stream cannot write it back. */
  forgetChat(id: string): void {
    this.deadChats.mark(id);
  }

  /** Undo the dead mark: the delete failed, so the chat keeps saving. */
  unforgetChat(id: string): void {
    this.deadChats.unmark(id);
  }

  async openChat(id: string): Promise<void> {
    this.abort();
    let record: ChatRecord | undefined;
    try {
      record = await getChat(id);
    } catch (err) {
      // Leave the current view in place; the read failed, not the chat.
      this.hooks.onNotice("Could not open this chat: " + reasonOf(err));
      return;
    }
    this.chat = record ?? null;
    this.renderAll();
  }

  newChat(): void {
    this.abort();
    this.chat = null;
    this.renderEmpty();
    this.input.focus();
  }

  currentChatId(): string | null {
    return this.chat?.id ?? null;
  }

  private renderEmpty(): void {
    this.column.textContent = "";
    const hero = document.createElement("div");
    hero.className = "chat-empty-hero";
    const h2 = document.createElement("h2");
    h2.textContent = "Chat with fifi";
    const p = document.createElement("p");
    p.textContent = "The model runs on this machine. Messages stay in this browser.";
    hero.append(h2, p);
    this.column.append(hero);
  }

  private renderAll(): void {
    this.column.textContent = "";
    const messages = this.chat?.messages ?? [];
    if (messages.length === 0) {
      this.renderEmpty();
      return;
    }
    for (const m of messages) this.column.append(this.renderMessage(m));
    this.scrollToEnd();
  }

  private renderMessage(m: UiMessage): HTMLElement {
    const root = document.createElement("div");
    if (m.role === "user") {
      root.className = "msg msg-user";
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = m.content;
      root.append(bubble);
      return root;
    }
    root.className = "msg msg-assistant";
    if (m.reasoning !== undefined && m.reasoning !== "") {
      root.append(this.buildThinking(m.reasoning, true));
    }
    const text = document.createElement("div");
    text.className = "msg-text";
    text.textContent = m.content;
    root.append(text);
    if (m.toolCalls !== undefined && m.toolCalls.length > 0) {
      root.append(this.buildTools(m.toolCalls));
    }
    if (m.state === "error" && m.errorMessage !== undefined) {
      const err = document.createElement("div");
      err.className = "msg-error";
      err.textContent = m.errorMessage;
      root.append(err);
    }
    root.append(this.buildMeta(m));
    return root;
  }

  private buildThinking(reasoning: string, closed: boolean): HTMLDetailsElement {
    const details = document.createElement("details");
    details.className = "msg-thinking";
    if (!closed) details.open = true;
    const summary = document.createElement("summary");
    summary.textContent = "Thinking";
    const body = document.createElement("div");
    body.className = "thinking-text";
    body.textContent = reasoning;
    details.append(summary, body);
    return details;
  }

  private buildTools(calls: { name: string; arguments: string }[]): HTMLElement {
    const wrap = document.createElement("div");
    wrap.className = "msg-tools";
    for (const call of calls) {
      const block = document.createElement("div");
      block.className = "tool-call";
      const name = document.createElement("div");
      name.className = "tool-call-name";
      name.textContent = call.name;
      const args = document.createElement("pre");
      args.className = "tool-call-args";
      args.textContent = call.arguments;
      block.append(name, args);
      wrap.append(block);
    }
    return wrap;
  }

  private buildMeta(m: UiMessage): HTMLElement {
    const meta = document.createElement("div");
    meta.className = "msg-meta";
    const parts: string[] = [];
    if (m.state === "stopped") parts.push("stopped");
    if (m.finishReason === "length") parts.push("Cut off: length");
    else if (m.finishReason === "tool_calls") parts.push("Tool calls");
    if (m.usage !== undefined) {
      parts.push(
        formatTokens(m.usage.prompt_tokens) + " prompt",
        formatTokens(m.usage.completion_tokens) + " out"
      );
      if (m.usage.reused_tokens !== undefined && m.usage.reused_tokens > 0) {
        parts.push(formatTokens(m.usage.reused_tokens) + " reused");
      }
    }
    if (m.durationMs !== undefined) {
      parts.push(formatDuration(m.durationMs));
      if (m.usage !== undefined) {
        const speed = formatSpeed(m.usage.completion_tokens, m.durationMs);
        if (speed !== "") parts.push(speed);
      }
    }
    meta.textContent = parts.join(" \u00b7 ");
    return meta;
  }

  private scrollToEnd(): void {
    this.scroll.scrollTop = this.scroll.scrollHeight;
  }

  /** Build the request body from the chat as it stands and the drawer settings. */
  private buildBody(chat: ChatRecord): ChatRequestBody {
    const settings = loadSettings();
    return buildChatBody(
      settings,
      buildMessages(settings.drawer.systemMessage, chat.messages)
    );
  }

  private async send(): Promise<void> {
    const text = this.input.value.trim();
    if (text === "" || this.offline) return;
    // Close the gate and hold the controller before the first await, so a
    // second send, a chat switch, or a delete cannot start a second stream.
    if (!this.sendGate.tryBegin()) return;
    const controller = new AbortController();
    this.controller = controller;
    this.updateSendEnabled();
    try {
      if (this.chat === null) {
        this.chat = newChatRecord(loadSettings().model);
      }
      const chat = this.chat;

      const userMsg = newUiMessage("user", text);
      chat.messages.push(userMsg);
      chat.title = chat.messages.some((m) => m.role === "user")
        ? titleOf(chat)
        : chat.title;
      chat.updatedAt = Date.now();
      try {
        await putChat(chat);
      } catch (err) {
        // The chat lives in memory; the request still goes out.
        this.hooks.onNotice("Could not save this chat: " + reasonOf(err));
      }
      this.hooks.onChatsChanged();
      // A switch, a delete, or a stop click aborted the pending send.
      if (controller.signal.aborted) {
        this.endPendingSend(controller);
        return;
      }

      this.input.value = "";
      this.autosize();
      if (chat.messages.length <= 2) this.renderAll();
      else this.column.append(this.renderMessage(userMsg));
      this.scrollToEnd();

      const assistant = newUiMessage("assistant", "");
      assistant.state = "streaming";
      // The record joins the chat at stream start, so a reload, a chat switch,
      // and the next request all see it. buildMessages skips its empty content.
      chat.messages.push(assistant);
      this.startStream(chat, assistant, controller);
    } catch (err) {
      // The stream never started; drop the busy state instead of wedging it.
      this.endPendingSend(controller);
      this.hooks.onNotice("Could not send the message: " + reasonOf(err));
    }
  }

  /** Reopen the composer after a send that never reached the stream. */
  private endPendingSend(controller: AbortController): void {
    if (this.controller === controller) this.controller = null;
    this.sendGate.end();
    this.updateSendEnabled();
  }

  private startStream(chat: ChatRecord, assistant: UiMessage, controller: AbortController): void {
    const body = this.buildBody(chat);
    this.controller = controller;
    this.streamDom = this.createStreamDom();
    this.streamChat = chat;
    this.streamMessage = assistant;
    this.toolCalls.clear();
    this.startedAt = performance.now();
    this.updateSendEnabled();
    this.hooks.onBusyChange(true);
    this.hooks.onAnnounce("Answering");

    const finish = (state: "complete" | "stopped" | "error", errorMessage?: string, usage?: Usage) => {
      const dom = this.streamDom;
      if (dom !== null) {
        dom.waiting.remove();
        dom.text.parentElement?.classList.remove("caret");
      }
      // Keep the placeholder; the error line carries the reason.
      assistant.state = state;
      assistant.errorMessage = errorMessage;
      if (usage !== undefined) assistant.usage = usage;
      assistant.durationMs = performance.now() - this.startedAt;
      if (!this.deadChats.has(chat.id)) {
        chat.updatedAt = Date.now();
        putChat(chat)
          .then(() => this.hooks.onChatsChanged())
          .catch((err: unknown) =>
            this.hooks.onNotice("Could not save this chat: " + reasonOf(err))
          );
      }
      this.controller = null;
      this.sendGate.end();
      this.streamDom = null;
      this.streamChat = null;
      this.streamMessage = null;
      this.updateSendEnabled();
      this.hooks.onBusyChange(false);
      this.hooks.onAnnounce(
        state === "complete" ? "Reply finished" : state === "stopped" ? "Stopped" : "Reply failed"
      );
    };

    chatStream(
      body,
      {
        onRole: () => {
          const dom = this.streamDom;
          if (dom !== null) dom.waiting.textContent =
            "Admitted. Prefilling \u2014 this can take a while.";
        },
        onDelta: (delta) => {
          const msg = this.streamMessage;
          if (msg === null) return;
          if (delta.reasoning !== undefined) {
            msg.reasoning = (msg.reasoning ?? "") + delta.reasoning;
            this.pendingReasoning += delta.reasoning;
          }
          if (delta.content !== undefined) {
            msg.content += delta.content;
            this.pendingContent += delta.content;
          }
          if (delta.toolCall !== undefined) {
            this.mergeToolCall(delta.toolCall);
          }
          this.scheduleFlush();
        },
        onFinish: (reason, usage) => {
          const msg = this.streamMessage;
          if (msg === null) return;
          if (usage !== undefined) msg.usage = usage;
          msg.finishReason = reason;
        },
      },
      controller.signal
    )
      .then((end) => {
        this.flushNow();
        // A stream that ends without [DONE] was cancelled by disconnect.
        finish(end === "done" ? "complete" : "stopped");
      })
      .catch((err: unknown) => {
        this.flushNow();
        if (err instanceof Error && err.name === "AbortError") {
          finish("stopped");
          return;
        }
        if (err instanceof ApiError) {
          if (err.code === "unauthorized") this.hooks.onUnauthorized();
          finish("error", describeError(err), this.streamMessage?.usage);
          return;
        }
        finish("error", "The stream ended unexpectedly.");
      });
  }

  private mergeToolCall(frag: ToolCallDelta): void {
    const name = frag.function?.name;
    const args = frag.function?.arguments ?? "";
    const existing = this.toolCalls.get(frag.index);
    if (existing === undefined) {
      this.toolCalls.set(frag.index, { name: name ?? "", args });
    } else {
      if (name !== undefined) existing.name = name;
      existing.args += args;
    }
    const msg = this.streamMessage;
    if (msg !== null) {
      msg.toolCalls = [...this.toolCalls.values()].map((c) => ({
        name: c.name,
        arguments: c.args,
      }));
    }
    // Tool text renders on flush.
    this.renderToolsNow();
  }

  private createStreamDom(): StreamDom {
    const root = document.createElement("div");
    root.className = "msg msg-assistant";
    const waiting = document.createElement("div");
    waiting.className = "waiting";
    waiting.textContent = "Waiting for admission\u2026";
    const textEl = document.createElement("div");
    textEl.className = "msg-text caret";
    const text = document.createTextNode("");
    textEl.append(text);
    const meta = document.createElement("div");
    meta.className = "msg-meta";
    root.append(waiting, textEl, meta);
    this.column.append(root);
    this.scrollToEnd();
    return { root, waiting, thinking: null, thinkingText: null, text, tools: null, meta };
  }

  private scheduleFlush(): void {
    if (this.flushHandle !== 0) return;
    this.flushHandle = requestAnimationFrame(() => {
      this.flushHandle = 0;
      this.flushNow();
    });
  }

  private flushNow(): void {
    if (this.flushHandle !== 0) {
      cancelAnimationFrame(this.flushHandle);
      this.flushHandle = 0;
    }
    const dom = this.streamDom;
    const msg = this.streamMessage;
    if (dom === null || msg === null) {
      this.pendingContent = "";
      this.pendingReasoning = "";
      return;
    }
    if (this.pendingReasoning !== "") {
      if (dom.thinking === null) {
        const details = this.buildThinking("", false);
        dom.waiting.remove();
        dom.root.prepend(details);
        const body = details.querySelector<HTMLElement>(".thinking-text");
        dom.thinking = details;
        dom.thinkingText = body !== null ? document.createTextNode("") : null;
        body?.prepend(dom.thinkingText ?? "");
      }
      dom.thinkingText?.insertData(dom.thinkingText.length, this.pendingReasoning);
      this.pendingReasoning = "";
    }
    if (this.pendingContent !== "") {
      dom.waiting.remove();
      dom.text.insertData(dom.text.length, this.pendingContent);
      this.pendingContent = "";
      // Keep the caret and recent text in view without smooth scrolling.
      const anchor = dom.text.parentElement ?? dom.root;
      anchor.scrollIntoView({ block: "end" });
    }
  }

  private renderToolsNow(): void {
    const dom = this.streamDom;
    if (dom === null) return;
    if (dom.tools === null) {
      dom.tools = document.createElement("div");
      dom.tools.className = "msg-tools";
      dom.root.insertBefore(dom.tools, dom.meta);
    }
    dom.tools.textContent = "";
    for (const call of this.toolCalls.values()) {
      const block = document.createElement("div");
      block.className = "tool-call";
      const name = document.createElement("div");
      name.className = "tool-call-name";
      name.textContent = call.name;
      const args = document.createElement("pre");
      args.className = "tool-call-args";
      args.textContent = call.args;
      block.append(name, args);
      dom.tools.append(block);
    }
  }
}

function reasonOf(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function titleOf(chat: ChatRecord): string {
  const first = chat.messages.find((m) => m.role === "user");
  if (first === undefined) return chat.title;
  const clean = first.content.trim().replace(/\s+/g, " ");
  return clean.length <= 42 ? clean : clean.slice(0, 41) + "\u2026";
}
