// Pure data shaping from harness results (the dashboard's only data source) into chart rows.
import type { CellSummary, ExperimentResult, MetricCI, TrialResult } from "../types/result";

export interface Point {
  x: number | string;
  mean: number;
  lo: number;
  hi: number;
  n: number;
}

export interface Series {
  name: string;
  points: Point[];
}

export function param(cell: CellSummary, key: string): string {
  const v = cell.params[key];
  return v === undefined ? "" : String(v);
}

function point(x: number | string, m: MetricCI): Point {
  return { x, mean: m.mean, lo: m.ci_low, hi: m.ci_high, n: m.n };
}

/** One series per value of `seriesKey`, x from `xKey` (numeric sweeps sorted ascending). */
export function sweep(result: ExperimentResult, xKey: string, seriesKey: string | null, metric: string): Series[] {
  const bySeries = new Map<string, Point[]>();
  for (const cell of result.cells) {
    const m = cell.metrics[metric];
    if (!m) continue;
    const name = seriesKey ? param(cell, seriesKey) : result.name;
    const raw = cell.params[xKey];
    const x = typeof raw === "number" ? raw : String(raw);
    bySeries.set(name, [...(bySeries.get(name) ?? []), point(x, m)]);
  }
  return [...bySeries.entries()].map(([name, points]) => ({
    name,
    points: points.sort((a, b) => (typeof a.x === "number" && typeof b.x === "number" ? a.x - b.x : 0)),
  }));
}

/** Rows keyed by x for a multi-series line chart: { x, [name]: mean, [name+"_ci"]: [lo, hi] }. */
export function toRows(series: Series[]): Record<string, unknown>[] {
  const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p.x)))];
  xs.sort((a, b) => (typeof a === "number" && typeof b === "number" ? a - b : 0));
  return xs.map((x) => {
    const row: Record<string, unknown> = { x };
    for (const s of series) {
      const p = s.points.find((q) => q.x === x);
      if (p) {
        row[s.name] = p.mean;
        row[`${s.name}_ci`] = [p.lo, p.hi];
      }
    }
    return row;
  });
}

/** One bar per cell, labelled by `label(cell)`, in cell order. */
export function bars(result: ExperimentResult, metric: string, label: (c: CellSummary) => string): Point[] {
  return result.cells.filter((c) => c.metrics[metric]).map((c) => point(label(c), c.metrics[metric]));
}

export function mean(values: number[]): number {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : NaN;
}

interface GatewayCollected {
  route_decisions?: Record<string, number>;
  remote_spend_usd?: number;
}

function gateway(t: TrialResult): GatewayCollected | undefined {
  const collected = (t.target as { collected?: { gateway?: GatewayCollected } }).collected;
  return collected?.gateway;
}

/** Share of requests the gateway sent to the remote, and remote $ per 1k requests, for one trial. */
export function gatewayEconomics(t: TrialResult): { remoteShare: number; usdPer1k: number } | null {
  const g = gateway(t);
  if (!g?.route_decisions) return null;
  const total = Object.values(g.route_decisions).reduce((a, b) => a + b, 0);
  if (!total) return null;
  const remote = Object.entries(g.route_decisions)
    .filter(([k]) => k.startsWith("remote:"))
    .reduce((a, [, v]) => a + v, 0);
  return { remoteShare: remote / total, usdPer1k: (1000 * (g.remote_spend_usd ?? 0)) / total };
}

/** Per-cell means of the gateway economics over ok trials (the harness does not aggregate these). */
export function gatewayByCell(result: ExperimentResult): Map<string, { remoteShare: number; usdPer1k: number }> {
  const out = new Map<string, { remoteShare: number; usdPer1k: number }>();
  for (const cell of result.cells) {
    const econ = result.trials
      .filter((t) => t.cell_id === cell.cell_id && t.status === "ok")
      .map(gatewayEconomics)
      .filter((e): e is { remoteShare: number; usdPer1k: number } => e !== null);
    if (econ.length) {
      out.set(cell.cell_id, {
        remoteShare: mean(econ.map((e) => e.remoteShare)),
        usdPer1k: mean(econ.map((e) => e.usdPer1k)),
      });
    }
  }
  return out;
}

export function fmt(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return "–";
  const abs = Math.abs(value);
  if (abs >= 1000) return value.toLocaleString("en-US", { maximumFractionDigits: 0 });
  return value.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}

export function fmtCI(p: Point, digits = 2, scale = 1): string {
  return `${fmt(p.mean * scale, digits)} [${fmt(p.lo * scale, digits)}, ${fmt(p.hi * scale, digits)}]`;
}
