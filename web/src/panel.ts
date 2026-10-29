/**
 * The speed panel's DOM-free state: the recent-rows fetch (driven by the
 * node:test suite with injected fakes — no global mocks, no real timers),
 * the probe merge, and the poll's per-tick decision. MonitorView renders
 * whatever these return.
 */

import { ApiError, listRecentRequests, type Model, type RecentRequest } from "./api.js";

/** The full panel state the view renders (re-exported as MonitorData). */
export interface PanelData {
  healthOk: boolean;
  readyOk: boolean;
  notReadyCode: string | null;
  models: Model[] | null;
  modelsError: string | null;
  recent: RecentRequest[] | null;
  recentError: string | null;
  checkedAt: number;
}

export interface PanelState {
  recent: RecentRequest[] | null;
  recentError: string | null;
  checkedAt: number;
}

/** Injected seams so the tests need no global fetch mock and no wall
 * clock: the ring fetch, the clock, and the abort timer. */
export interface RecentDeps {
  fetch?: (signal: AbortSignal) => Promise<{ requests: RecentRequest[] }>;
  now?: () => number;
  timeout?: (ms: number) => AbortSignal;
}

/** One refresh of the recent rows. A refused, offline, or hung route
 * leaves `recent` null and names the cause — never the last good rows,
 * which is what made the panel look alive while the engine was down. */
export async function refreshRecentRows(timeoutMs = 4000, deps: RecentDeps = {}): Promise<PanelState> {
  const fetchRing = deps.fetch ?? ((signal) => listRecentRequests(signal));
  const now = deps.now ?? Date.now;
  const timeout = deps.timeout ?? AbortSignal.timeout;
  let recent: RecentRequest[] | null = null;
  let recentError: string | null = null;
  try {
    recent = (await fetchRing(timeout(timeoutMs))).requests;
  } catch (err) {
    recentError = err instanceof ApiError ? err.code : "network";
  }
  return { recent, recentError, checkedAt: now() };
}

/** What one full sweep learned: the health/ready/models probe plus the
 * recent-rows fetch (null rows when the server was unreachable — the
 * sweep never fetches the ring from a dead process). */
export interface ProbeOutcome {
  healthOk: boolean;
  readyOk: boolean;
  notReadyCode: string | null;
  models: Model[] | null;
  modelsError: string | null;
  recent: RecentRequest[] | null;
  recentError: string | null;
}

/** The merge (Opus MF1): a probe FULLY REPLACES the panel data and stamps
 * one fresh time. Nothing survives from the previous state — that is the
 * invariant; the poll may only call the full sweep, so a dead engine can
 * never show a last-good "runs"/"ready" next to a current timestamp. */
export function applyProbe(probe: ProbeOutcome): PanelData {
  return { ...probe, checkedAt: Date.now() };
}

/** The poll's per-tick decision (Sol MF2): which refresh runs. The last
 * known server state is an input ONLY so the decision owns it — the
 * correct answer consults it for nothing, because the full sweep probes
 * health itself. Gating the monitor refresh on readiness was a
 * bug: the view froze when the engine was down or still starting. */
export function pollPlan(opts: {
  monitorVisible: boolean;
  lastKnownServerReady: boolean;
}): { refreshMonitor: boolean } {
  void opts.lastKnownServerReady;
  return { refreshMonitor: opts.monitorVisible };
}
