/**
 * The live speed panel's pure summary: medians over FINISHED-OK rows only
 * (the panel shows what a healthy request delivered; cancelled and failed
 * rows never feed a median). DOM-free so the node:test suite runs it.
 */

export interface SpeedRow {
  outcome: "ok" | "error" | "cancelled";
  decode_tokens_per_second: number | null;
  time_to_first_token_ms: number | null;
}

export interface SpeedSummary {
  decodePerSecond: number | null;
  firstTokenMs: number | null;
  okCount: number;
}

/** The lower of the two middle values on even counts: dumb and stable. */
function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor((sorted.length - 1) / 2)] ?? null;
}

export function speedSummary(rows: SpeedRow[]): SpeedSummary {
  const ok = rows.filter((row) => row.outcome === "ok");
  return {
    decodePerSecond: median(
      ok
        .map((row) => row.decode_tokens_per_second)
        .filter((value): value is number => value !== null)
    ),
    firstTokenMs: median(
      ok
        .map((row) => row.time_to_first_token_ms)
        .filter((value): value is number => value !== null)
    ),
    okCount: ok.length,
  };
}
