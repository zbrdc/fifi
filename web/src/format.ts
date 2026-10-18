/** Number and time formatting. Tabular-friendly, locale aware. */

export function formatTokens(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 10_000) return Math.round(n / 1_000) + "k";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "k";
  return String(n);
}

export function formatBytes(n: number): string {
  if (n >= 1024 ** 3) return (n / 1024 ** 3).toFixed(1) + " GiB";
  if (n >= 1024 ** 2) return (n / 1024 ** 2).toFixed(1) + " MiB";
  if (n >= 1024) return Math.round(n / 1024) + " KiB";
  return n + " B";
}

export function formatDuration(ms: number): string {
  if (ms >= 60_000) {
    const m = Math.floor(ms / 60_000);
    const s = Math.round((ms % 60_000) / 1_000);
    return m + "m " + s + "s";
  }
  if (ms >= 10_000) return Math.round(ms / 1_000) + " s";
  if (ms >= 1_000) return (ms / 1_000).toFixed(1) + " s";
  return Math.round(ms) + " ms";
}

export function formatSpeed(tokens: number, ms: number): string {
  if (ms <= 0 || tokens <= 0) return "";
  return (tokens / (ms / 1_000)).toFixed(1) + " tok/s";
}

export function formatClock(ts: number): string {
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function formatDate(ts: number): string {
  return new Date(ts).toLocaleDateString([], { year: "numeric", month: "short", day: "numeric" });
}
