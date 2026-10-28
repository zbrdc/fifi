/**
 * Server-sent-events parser for chat streams.
 * Follows the contract: comment lines are ignored, a blank line ends an
 * event, and "data: [DONE]" ends a successful stream.
 */

export interface SseEvent {
  event: string;
  data: string;
}

export class SseParser {
  private buffer = "";
  private eventName = "";
  private dataLines: string[] = [];

  constructor(private readonly dispatch: (ev: SseEvent) => void) {}

  push(chunk: string): void {
    this.buffer += chunk;
    for (;;) {
      const nl = this.buffer.indexOf("\n");
      if (nl < 0) break;
      const line = this.buffer.slice(0, nl).replace(/\r$/, "");
      this.buffer = this.buffer.slice(nl + 1);
      this.handleLine(line);
    }
  }

  /** Flush a trailing partial line at stream close. */
  close(): void {
    if (this.buffer !== "") {
      const line = this.buffer.replace(/\r$/, "");
      this.buffer = "";
      this.handleLine(line);
    }
    if (this.dataLines.length > 0 || this.eventName !== "") this.flush();
  }

  private handleLine(line: string): void {
    if (line === "") {
      this.flush();
      return;
    }
    if (line.startsWith(":")) return; // keep-alive comment
    const colon = line.indexOf(":");
    const field = colon < 0 ? line : line.slice(0, colon);
    let value = colon < 0 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") this.eventName = value;
    else if (field === "data") this.dataLines.push(value);
    // id and retry are not used by this server.
  }

  private flush(): void {
    const event = this.eventName || "message";
    const data = this.dataLines.join("\n");
    this.eventName = "";
    this.dataLines = [];
    if (data !== "" || this.eventNameWasSet(event)) this.dispatch({ event, data });
  }

  private eventNameWasSet(event: string): boolean {
    // An event with an explicit name but no data still matters (none exists
    // in this contract, but the guard keeps the parser honest).
    return event !== "message" && event !== "";
  }
}
